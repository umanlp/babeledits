# %%
import argparse
import copy
import itertools
import json
import pickle
import random
import time
from collections import defaultdict

import babelnet as bn
import pandas as pd
from babelnet import BabelSynsetID, Language
from babelnet.data.relation import BabelPointer
import re 

## Params
parser = argparse.ArgumentParser(description="Process some data.")
parser.add_argument("--lang", default="it", help="Language")
parser.add_argument("--output_folder", default="datasets/v2", help="Output folder")
parser.add_argument(
    "--rel_path",
    default="datasets/v2/agg_relations_with_prompts.tsv",
    help="Path to relation file",
)
parser.add_argument(
    "--top_k", type=int, default=100, help="Top-k relations to consider"
)
parser.add_argument(
    "--max_edge", type=int, default=3, help="For each relation, how many edges to consider"
)

parser.add_argument("--synset_path", default="synsets/v2", help="synset path")
args = parser.parse_args()

lang = args.lang
output_folder = args.output_folder
rel_path = args.rel_path
top_k = args.top_k
synset_path = args.synset_path
max_edge = args.max_edge

# %%
# Load the pickle file
file_path = f"{synset_path}/{lang}/{lang}_syns.pkl"
with open(file_path, "rb") as f:
    data = pickle.load(f)

# %%


def convert_to_babel_relations(relations):
    babel_relations = []
    for r in relations:
        try:
            babel_relations.append(BabelPointer.from_name(r))
        except:
            if r == "GLOSS_DISAMBIGUATED":
                babel_relations.append(BabelPointer.GLOSS_DISAMBIGUATED)
            elif r == "REGION_MEMBER":
                babel_relations.append(BabelPointer.REGION_MEMBER)
            elif r == "TOPIC_MEMBER":
                babel_relations.append(BabelPointer.TOPIC_MEMBER)
            else:
                raise ValueError("Could not convert relation!")
    return babel_relations

def clean(sense):
    # Replace underscores with spaces
    sense = sense.replace('_', ' ')
    
    # Remove round brackets and everything in between
    sense = re.sub(r'\(.*?\)', '', sense)
    
    # Remove double quotes if they wrap the entire string
    if sense.startswith('"') and sense.endswith('"'):
        sense = sense[1:-1]
    
    return sense

# %%


rel_df = pd.read_csv(rel_path, sep="\t").iloc[:top_k].set_index("relation_name")
relations = rel_df.index.tolist()
relations = convert_to_babel_relations(relations)
languages = [
    Language.from_iso(l) for l in [lang, "en"]
]  # converting languages to BabelLanguage
edits = []

t_start = time.time()

# Get synset -> senses map, for each not-null synset which is a subject (i.e., derived from a wikipedia title)
# Second step serves to only get the synset->sense map only for synsets that have senses in both the source language and target language
synset_to_senses = {
    synset: {
        "sense_src": synset.main_sense(Language.from_iso(lang)),
        "sense_en": synset.main_sense(Language.EN),
    }
    for title, synset in data
    if synset is not None
}
synset_to_senses = {
    synset: {
        "sense_src": clean(senses["sense_src"].full_lemma),
        "sense_en": clean(senses["sense_en"].full_lemma),
    }
    for synset, senses in synset_to_senses.items()
    if all(senses.values())
}

# Get synset -> outgoing relation maps, only for the relations we selected a priori
# Second step serves to get synset -> relation -> edge, since each synset could have multiple instances of the same relation
synset_to_relations = {
    str(synset.id): [
        (e.pointer.name, e, e.target)
        for e in synset.outgoing_edges()
        if e.pointer in set(relations)
    ]
    for synset in synset_to_senses
}
synset_to_relations = {
    synset: {
        str(r): random.sample([{"edge_id": str(v[1]), "target_id": str(v[2])} for v in edge_data], max_edge)
        for r, edge_data in itertools.groupby(edge_list, key=lambda e: e[0])
    }
    for synset, edge_list in synset_to_relations.items()
}

# Extracting all the ids of target synsets (i.e., opposed synsets to subject synsets)
target_synset_ids = list(
    set(
        [
            BabelSynsetID(edge["target_id"])
            for relations in synset_to_relations.values()
            for edges in relations.values()
            for edge in edges
        ]
    )
)
print(f"Fetching {len(target_synset_ids)} target synsets")
target_synsets = bn.get_synsets(*(target_synset_ids))

# Similar to above
target_senses = {
    str(synset.id): {
        "sense_en": synset.main_sense(Language.EN),
        "sense_src": synset.main_sense(Language.from_iso(lang)),
    }
    for synset in target_synsets
}
target_senses = {
    syn_id: {
        "sense_en": clean(senses["sense_en"].full_lemma),
        "sense_src": clean(senses["sense_src"].full_lemma),
    }
    for syn_id, senses in target_senses.items()
    if all(senses.values())
}

# Let's iterate over all the subject synsets and store the data from the target synsets (if we have their senses)
for synset in synset_to_relations:
    relation_to_edges = synset_to_relations[synset]
    for relation in relation_to_edges:
        for edge in relation_to_edges[relation]:
            if (
                edge["target_id"] in target_senses
            ):  # add data only if we have senses for the target synset
                edge["target_sense_src"] = clean(target_senses[edge["target_id"]]["sense_src"])
                edge["target_sense_en"] = clean(target_senses[edge["target_id"]]["sense_en"])
        relation_to_edges[relation] = [
            edge for edge in relation_to_edges[relation] if "target_sense_en" in edge
        ]

# Make it so that the key is a string (useful later)
synset_to_senses = {str(synset.id): sense for synset, sense in synset_to_senses.items()}

print(f"Time taken: {(time.time()-t_start)/60} minutes")


# %%

rel_to_synsets = defaultdict(list)

# We create a data structure which is relation -> target synset, useful for the edit creation step
for d in list(synset_to_relations.values()):
    for k, v in d.items():
        [x.pop("edge_id") for x in v if "edge_id" in x]
        rel_to_synsets[k] += v

for rel in rel_to_synsets:
    unique_synsets = {x["target_id"]: x for x in rel_to_synsets[rel]}
    rel_to_synsets[rel] = list(unique_synsets.values())

# %%

# Iterating over the main structure of synset -> relations -> edges
for synset in synset_to_relations:
    relation_to_edges = synset_to_relations[synset]
    for relation in relation_to_edges:
        # Creating the pool of target synsets that we can sample from, excluding the ones that the subject synset is already linked to
        target_synsets = [edge["target_id"] for edge in relation_to_edges[relation]]
        syn_pool = [
            x for x in rel_to_synsets[relation] if x["target_id"] not in target_synsets
        ]
        if len(syn_pool) > 0:  # if there's something to pool from
            for edge in relation_to_edges[relation]:
                sampled_syn = copy.deepcopy(
                    random.sample(syn_pool, 1)[0]
                )  # needed to avoid nested structures
                if "edit" in sampled_syn:
                    sampled_syn.pop("edit")
                prompt = rel_df.loc[relation, "question"]
                prompt = prompt.replace(
                    "<subject>", synset_to_senses[synset]["sense_en"]
                )
                edge["edit"] = sampled_syn
                edge["edit"]["prompt_en"] = prompt
        relation_to_edges[relation] = [
            edge for edge in relation_to_edges[relation] if "edit" in edge
        ]

output = {
    synset_id: {"subject_senses": senses, "relations": synset_to_relations[synset_id]}
    for synset_id, senses in synset_to_senses.items()
}

with open(f"{output_folder}/{lang}.json", "w") as f:
    json.dump(output, f, indent=4, ensure_ascii=False)
f.close()

print(f"Done! Output saved to {output_folder}/{lang}.json")