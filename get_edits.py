# %%
import argparse
import copy
import itertools
import json
import pickle
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import babelnet as bn
import pandas as pd
from babelnet import BabelSynsetID, Language
from babelnet.data.relation import BabelPointer
from tqdm import tqdm

from get_synsets import check_langs

## Params
parser = argparse.ArgumentParser(description="Process some data.")
parser.add_argument(
    "--langs",
    nargs="+",
    default=[
        "af",
        "ar",
        "az",
        "bg",
        "bn",
        "de",
        "el",
        "en",
        "es",
        "et",
        "eu",
        "fa",
        "fi",
        "fr",
        "gu",
        "he",
        "hi",
        "ht",
        "hr",
        "hu",
        "id",
        "it",
        "ja",
        "jv",
        "ka",
        "kk",
        "ko",
        "lt",
        "ml",
        "mr",
        "ms",
        "my",
        "nl",
        "pa",
        "pl",
        "pt",
        "qu",
        "ro",
        "ru",
        "sw",
        "ta",
        "te",
        "th",
        "tl",
        "tr",
        "uk",
        "ur",
        "vi",
        "yo",
        "zh",
    ],
    help="List of languages",
)
parser.add_argument(
    "--output_folder", default="datasets/v4", help="Output folder"
)
parser.add_argument(
    "--rel_path",
    default="datasets/v4/agg_relations_with_prompts.tsv",
    help="Path to relation file",
)
parser.add_argument(
    "--top_k", type=int, default=100, help="Top-k relations to consider"
)
parser.add_argument(
    "--max_edge",
    type=int,
    default=3,
    help="For each relation, how many edges to consider",
)
parser.add_argument("--synset_path", default="synsets/v4", help="synset path")

args = parser.parse_args()

langs = args.langs
output_folder = args.output_folder
rel_path = args.rel_path
top_k = args.top_k
synset_path = args.synset_path
max_edge = args.max_edge


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
    sense = sense.replace("_", " ")

    # Remove round brackets and everything in between
    sense = re.sub(r"\(.*?\)", "", sense)

    # Remove double quotes if they wrap the entire string
    if sense.startswith('"') and sense.endswith('"'):
        sense = sense[1:-1]

    return sense.strip()


# %%


rel_df = pd.read_csv(rel_path, sep="\t").iloc[:top_k].set_index("relation_name")
relations = rel_df.index.tolist()
relations = convert_to_babel_relations(relations)


t_start = time.time()

file_path = f"{synset_path}/all_langs_syns.pkl"
print(f"Loading synsets from {file_path}")
with open(file_path, "rb") as f:
    data = pickle.load(f)

print(f"Loaded {len(data)} synsets from {file_path}")


# Get synset -> senses map, for each not-null synset which is a subject (i.e., derived from a wikipedia title)
# Second step serves to only get the synset->sense map only for synsets that have senses in both the source language and target language
synset_to_senses = {
    synset: {
        f"sense_{lang}": synset.main_sense(Language.from_iso(lang)) for lang in langs
    }
    for _, synset in tqdm(data, desc="Getting senses")
    if synset is not None
}

#TODO might not be necessary with the new setup of multiparallel filtered synsets
synset_to_senses = {
    synset: {
        f"sense_{lang}": clean(senses[f"sense_{lang}"].full_lemma) for lang in langs
    }
    for synset, senses in tqdm(synset_to_senses.items())
    if all(senses.values())
}

# Get synset -> outgoing relation maps, only for the relations we selected a priori
# Second step serves to get synset -> relation -> edge, since each synset could have multiple instances of the same relation
# Third step puts a max cap on the number of edges per relation
synset_to_relations = {
    str(synset.id): [
        (e.pointer.name, e, e.target)
        for e in synset.outgoing_edges()
        if e.pointer in set(relations)
    ]
    for synset in tqdm(synset_to_senses, desc="Getting relations")
}
synset_to_relations = {
    synset: {
        str(r): [{"edge_id": str(v[1]), "target_id": str(v[2])} for v in edge_data]
        for r, edge_data in itertools.groupby(edge_list, key=lambda e: e[0])
    }
    for synset, edge_list in tqdm(synset_to_relations.items(), desc="Grouping relations")
}

synset_to_relations = {
    synset: {
        relation: random.sample(edges, min(len(edges), max_edge))
        for relation, edges in edge_list.items()
    }
    for synset, edge_list in synset_to_relations.items()
}

# Extracting all the ids of target synsets (i.e., opposed synsets to subject synsets)
# target_synset_ids = list(
# set(
# [
# BabelSynsetID(edge["target_id"])
# for relations in synset_to_relations.values()
# for edges in relations.values()
# for edge in edges
# ]
# )
# )
# print(f"Fetching {len(target_synset_ids)} target synsets")
# target_synsets = bn.get_synsets(*(target_synset_ids))

babel_langs = set([Language.from_iso(lang) for lang in langs])
# target_synsets = [
#     synset
#     for synset in target_synsets
#     if synset is not None and check_langs(synset, babel_langs)
# ]

# # Similar to above
# target_senses = {
#     str(synset.id): {
#         f"sense_{lang}": synset.main_sense(Language.from_iso(lang)) for lang in langs
#     }
#     for synset in target_synsets
# }
# target_senses = {
#     syn_id: {
#         f"sense_{lang}": clean(senses[f"sense_{lang}"].full_lemma) for lang in langs
#     }
#     for syn_id, senses in target_senses.items()
#     if all(senses.values())
# }

print("Getting edges where target synset is in all selected languages")
# Let's iterate over all the subject synsets and store the data from the target synsets (if we have their senses)
for synset in tqdm(synset_to_relations, desc="Getting target synsets"):
    relation_to_edges = synset_to_relations[synset]
    shuffled_relations = list(relation_to_edges.keys())
    random.shuffle(shuffled_relations)
    found = False
    for relation in shuffled_relations:
        for edge in relation_to_edges[relation]:
            tgt_syn = BabelSynsetID(edge["target_id"]).to_synset()
            if check_langs(tgt_syn, babel_langs):
                target_senses = {
                    f"target_sense_{lang}": clean(
                        tgt_syn.main_sense(Language.from_iso(lang)).full_lemma
                    )
                    for lang in args.langs
                }
                edge.update(target_senses)
                found = True #TODO
                break
        relation_to_edges[relation] = [
            edge for edge in relation_to_edges[relation] if "target_sense_en" in edge
        ]
        if found: #TODO
            break
    synset_to_relations[synset] = {
        relation: edge_list
        for relation, edge_list in relation_to_edges.items()
        if len(edge_list) > 0
    }

# Remove synsets with no suitable edges
synset_to_relations = {
    synset: relations
    for synset, relations in synset_to_relations.items()
    if len(relations) > 0
}
# Make it so that the key is a string (useful later)
synset_to_senses = {str(synset.id): sense for synset, sense in synset_to_senses.items()}

print("Mapping relations to synsets!")
rel_to_synsets = defaultdict(list)

# We create a data structure which is relation -> target synset, useful for the edit creation step
for d in list(synset_to_relations.values()):
    for k, v in d.items():
        [x.pop("edge_id") for x in v if "edge_id" in x]
        rel_to_synsets[k] += v

for rel in rel_to_synsets:
    unique_synsets = {x["target_id"]: x for x in rel_to_synsets[rel]}
    rel_to_synsets[rel] = list(unique_synsets.values())

# Iterating over the main structure of synset -> relations -> edges
for synset in tqdm(synset_to_relations, desc="Creating edits"):
    relation_to_edges = synset_to_relations[synset]
    shuffled_relations = list(relation_to_edges.keys())
    random.shuffle(shuffled_relations)
    for relation in shuffled_relations:
        # Creating the pool of target synsets that we can sample from, excluding the ones that the subject synset is already linked to
        target_synsets = [edge["target_id"] for edge in relation_to_edges[relation]]
        syn_pool = [
            x for x in rel_to_synsets[relation] if x["target_id"] not in target_synsets
        ]
        if len(syn_pool) > 0:  # if there's something to pool from
            edge = random.choice(relation_to_edges[relation])
            sampled_syn = copy.deepcopy(random.choice(syn_pool))
            if "edit" in sampled_syn:
                sampled_syn.pop("edit")
            prompt = rel_df.loc[relation, "question"]
            prompt = prompt.replace("<subject>", synset_to_senses[synset]["sense_en"])
            edge["edit"] = sampled_syn
            edge["edit"]["prompt_en"] = prompt
        relation_to_edges[relation] = [
            edge for edge in relation_to_edges[relation] if "edit" in edge
        ]

# Cleaning up
synset_to_relations = {
    synset: {
        relation: edge_list for relation, edge_list in relations.items() if len(edge_list) > 0
    }
    for synset, relations in synset_to_relations.items()
}

synset_to_relations = {
    synset : relations
    for synset, relations in synset_to_relations.items()
    if len(relations) > 0
}

def sample_one_relation(relations_to_edges):
    relation = random.choice(list(relations_to_edges.keys()))
    return {relation : random.choice(relations_to_edges[relation]) }

output = {
    synset_id: {
        "subject_senses": senses,
        "relations": sample_one_relation(synset_to_relations[synset_id]),
    }
    for synset_id, senses in synset_to_senses.items()
    if synset_id in synset_to_relations
}

Path(output_folder).mkdir(parents=True, exist_ok=True)
with open(f"{output_folder}/all_langs_break.json", "w") as f: #TODO
    json.dump(output, f, indent=4, ensure_ascii=False)
f.close()

print(f"Time taken: {(time.time()-t_start)/60} minutes")
print(f"Done! Output saved to {output_folder}/all_langs.json") 
