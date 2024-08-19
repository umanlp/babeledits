# %%
import argparse
import copy
import itertools
import json
import pickle
import random
import time
from collections import defaultdict
from pathlib import Path

import babelnet as bn
import pandas as pd
from babelnet import BabelSynsetID, Language
from babelnet.data.relation import BabelPointer
from tqdm import tqdm

from get_synsets import check_langs
from utils import rename_key, clean

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
parser.add_argument("--output_folder", default="datasets/v4", help="Output folder")
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
parser.add_argument("--rephrase", action="store_true", help="rephrase the questions")
parser.add_argument(
    "--locality", action="store_true", help="whether to also get a locality"
)
parser.add_argument("--synset_path", default="synsets/v4", help="synset path")

args = parser.parse_args()

langs = sorted(args.langs)
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
        except:  # cases that for some reason are not handled by from_name
            if r == "GLOSS_DISAMBIGUATED":
                babel_relations.append(BabelPointer.GLOSS_DISAMBIGUATED)
            elif r == "REGION_MEMBER":
                babel_relations.append(BabelPointer.REGION_MEMBER)
            elif r == "TOPIC_MEMBER":
                babel_relations.append(BabelPointer.TOPIC_MEMBER)
            elif r == "DERIVATIONALLY_RELATED":
                babel_relations.append(BabelPointer.DERIVATIONALLY_RELATED)
            else:
                raise ValueError("Could not convert relation!")
    return babel_relations


PERSON_SYN_ID = "bn:00044576n"
IS_A_RELATION = BabelPointer.INSTANCE_OF


def extract_main_sense(synset, lang):
    if lang == "en" and any(
        [
            str(e.target) == PERSON_SYN_ID
            for e in synset.outgoing_edges(*[IS_A_RELATION])
        ]
    ):
        main_lemma = synset.main_sense(bn.Language.from_iso(lang)).full_lemma
        if "_" in main_lemma:
            main_lemma = main_lemma.replace("_", " ")
            return main_lemma
        all_senses = synset.senses(bn.Language.from_iso(lang))
        sense_found = False
        for s in all_senses:
            if main_lemma != s.full_lemma and main_lemma in s.full_lemma:
                sense_found = True
                break
        if sense_found:
            return s.full_lemma
        else:
            return main_lemma
    else:
        return synset.main_sense(bn.Language.from_iso(lang)).full_lemma


# %%


rel_df = pd.read_csv(rel_path, sep="\t").iloc[:top_k].set_index("relation_name")
relations = rel_df.index.tolist()
relations = convert_to_babel_relations(relations)


t_start = time.time()

file_path = f"{synset_path}/syns.pkl"
print(f"Loading synsets from {file_path}")
with open(file_path, "rb") as f:
    data = pickle.load(f)


print(f"Loaded {len(data)} synsets from {file_path}")


# Get synset -> senses map, for each n ot-null synset which is a subject (i.e., derived from a wikipedia title)
# Second step serves to only get the synset->sense map only for synsets that have senses in both the source language and target language
synset_to_senses = {
    synset: {lang: clean(extract_main_sense(synset, lang)) for lang in langs}
    for _, synset in tqdm(data, desc="Getting senses")
    if synset is not None
}

# Get synset -> outgoing relation maps, only for the relations we selected a priori
# Removes synset with no relations in the selected set
# Third step serves to get synset -> relation -> edge, since each synset could have multiple instances of the same relation
# Fourth step puts a max cap on the number of edges per relation
synset_to_relations = {
    str(synset.id): [
        (e.pointer.name, e, e.target)
        for e in synset.outgoing_edges()
        if e.pointer in set(relations)
    ]
    for synset in tqdm(synset_to_senses, desc="Getting relations")
}
synset_to_relations = {
    syn: rel for syn, rel in synset_to_relations.items() if len(rel) > 0
}
print(
    f"Got {len(synset_to_relations)} synsets with relations in the selected set out of {len(data)}"
)
synset_to_relations = {
    synset: {
        str(r): [
            {"edge_id": str(v[1]), "ground_truth_id": str(v[2])} for v in edge_data
        ]
        for r, edge_data in itertools.groupby(edge_list, key=lambda e: e[0])
    }
    for synset, edge_list in tqdm(
        synset_to_relations.items(), desc="Grouping relations"
    )
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
# targets = {
#     str(synset.id): {
#         f"sense_{lang}": synset.main_sense(Language.from_iso(lang)) for lang in langs
#     }
#     for synset in target_synsets
# }
# targets = {
#     syn_id: {
#         f"sense_{lang}": clean(senses[f"sense_{lang}"].full_lemma) for lang in langs
#     }
#     for syn_id, senses in targets.items()
#     if all(senses.values())
# }

print("Getting edges where target synset is in all selected languages")
# Let's iterate over all the subject synsets and store the data from the target synsets (if we have their senses)

targets_to_find = 2 if args.locality else 1
for synset in tqdm(synset_to_relations, desc="Getting target synsets"):
    relation_to_edges = synset_to_relations[synset]
    shuffled_relations = list(relation_to_edges.keys())
    random.shuffle(shuffled_relations)

    # Dictionary to track how many edges we have found for each relation
    relation_counter = {relation: 0 for relation in shuffled_relations}
    for relation in shuffled_relations:
        for edge in relation_to_edges[relation]:
            tgt_syn = BabelSynsetID(edge["ground_truth_id"]).to_synset()
            if check_langs(tgt_syn, babel_langs):
                targets = {
                    "ground_truths": {
                        lang: clean(extract_main_sense(tgt_syn, lang))
                        for lang in args.langs
                    }
                }
                edge.update(targets)
                relation_counter[relation] += 1
                break
        if sum(list(relation_counter.values())) >= targets_to_find:
            break
        # Continue to the next relation

    # Keep only those relations with enough suitable edges
    synset_to_relations[synset] = {
        relation: [e for e in edge_list if "ground_truths" in e]
        for relation, edge_list in relation_to_edges.items()
        if len([e for e in edge_list if "ground_truths" in e]) > 0
    }

# Remove synsets with fewer than targets_to_find distinct relations with suitable edges
synset_to_relations = {
    synset: relations
    for synset, relations in synset_to_relations.items()
    if len([relation for relation, edges in relations.items() if len(edges) > 0])
    >= targets_to_find
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
    unique_synsets = {x["ground_truth_id"]: x for x in rel_to_synsets[rel]}
    rel_to_synsets[rel] = list(unique_synsets.values())

# Iterating over the main structure of synset -> relations -> edges
for synset in tqdm(synset_to_relations, desc="Creating edits"):
    relation_to_edges = synset_to_relations[synset]
    shuffled_relations = list(relation_to_edges.keys())
    random.shuffle(shuffled_relations)
    for relation in shuffled_relations:
        # Creating the pool of target synsets that we can sample from, excluding the ones that the subject synset is already linked to
        target_synsets = [
            edge["ground_truth_id"] for edge in relation_to_edges[relation]
        ]
        syn_pool = [
            x
            for x in rel_to_synsets[relation]
            if x["ground_truth_id"] not in target_synsets
        ]
        if len(syn_pool) > 0:  # if there's something to pool from
            edge = random.choice(relation_to_edges[relation])
            sampled_syn = copy.deepcopy(random.choice(syn_pool))
            sampled_syn["target_id"] = sampled_syn.pop("ground_truth_id")
            sampled_syn["targets"] = sampled_syn.pop("ground_truths")
            if "edit" in sampled_syn:  # avoid infinit recursion
                sampled_syn.pop("edit")
            prompt = rel_df.loc[relation, "question"].replace(
                "<subject>", synset_to_senses[synset]["en"]
            )
            prompt_data = {"prompts": {"en": prompt}}
            if args.rephrase:
                rephrase_prompt = rel_df.loc[relation, "rephrase"].replace(
                    "<subject>", synset_to_senses[synset]["en"]
                )
                prompt_data["generality"] = {}
                prompt_data["generality"].update({"prompts_gen": {"en": rephrase_prompt}})
            edge["edit"] = sampled_syn
            edge["edit"].update(prompt_data)


# Keep only synsets that (i) have at least one/two relations and (ii) at least one edge with an edit
synset_to_relations = {
    synset: relations
    for synset, relations in synset_to_relations.items()
    if len(relations) >= targets_to_find
    and any(["edit" in edge for relation in relations for edge in relations[relation]])
}


def sample_one_relation(relations_to_edges):
    relation_pool = {
        rel: rel_data
        for rel, rel_data in relations_to_edges.items()
        if any("edit" in edge for edge in rel_data)
    }
    relation = random.choice(list(relation_pool.keys()))
    relation_data = random.choice([e for e in relation_pool[relation] if "edit" in e])
    return {relation: relation_data}


output = {
    synset_id: {
        "subjects": senses,
        "relations": sample_one_relation(synset_to_relations[synset_id]),
    }
    for synset_id, senses in synset_to_senses.items()
    if synset_id in synset_to_relations
}

print(f"Input size {len(data)}, Output size {len(output)}")
if args.locality:
    print("Getting locality sets")
    for synset_id in output:
        selected_relation = list(output[synset_id]["relations"].keys())[
            0
        ]  # assumes there's only one edit per entity
        relation_pool = {
            rel: rel_data
            for rel, rel_data in synset_to_relations[synset_id].items()
            if rel != selected_relation
        }
        rel_name = random.choice(list(relation_pool.keys()))
        sampled_rel = copy.deepcopy(random.choice(relation_pool[rel_name]))
        sampled_rel.pop("edit", None)
        sampled_rel = rename_key(sampled_rel, "ground_truth_id", "ground_truth_id_loc")
        sampled_rel = rename_key(sampled_rel, "ground_truths", "ground_truths_loc")
        sampled_rel.update(
            {
                "prompts_loc": {
                    "en": rel_df.loc[rel_name, "question"].replace(
                        "<subject>", output[synset_id]["subjects"]["en"]
                    )
                }
            }
        )
        
        output[synset_id]["relations"][selected_relation]["edit"]["locality"] = {}
        output[synset_id]["relations"][selected_relation]["edit"]["locality"].update(
            {rel_name: sampled_rel}
        )

print(f"Input size {len(data)}, Output size {len(output)}")
Path(output_folder).mkdir(parents=True, exist_ok=True)
with open(f"{output_folder}/dataset.json", "w") as f:  #
    json.dump(output, f, indent=4, ensure_ascii=False)
f.close()

print(f"Time taken: {(time.time()-t_start)/60} minutes")
print(f"Done! Output saved to {output_folder}/dataset.json")
