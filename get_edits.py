# %%

## Params
from datetime import timedelta, date
import babelnet as bn
from babelnet import BabelSynsetID, Language
year = 2022
# XTREME-R langs
langs = ["af","ar","az","bg","bn","de","el","en","es","et","eu","fa","fi","fr","gu","he","hi","ht","hu","id","it","ja","jv","ka","kk","ko","lt","ml","mr","ms","my","nl","pa","pl","pt","qu","ro","ru","sw","ta","te","th","tl","tr","uk","ur","vi","wo","yo","zh"]
start_date = date(year, 1, 1)
end_date = date(year, 12, 31)
top_k = 10000

# %%
import os
import pickle

lang = "it"
# Specify the path to the pickle file
file_path = f'synsets/{lang}/{lang}_syns.pkl'

# Load the pickle file
with open(file_path, 'rb') as f:
    data = pickle.load(f)
print(data[0])
print(data[0][1].outgoing_edges())

# %%
from babelnet.data.relation import BabelPointer

def get_data_from_synset(synset, languages, relations):
    senses = [(language, synset.main_sense(language)) for language in languages]
    senses = [(str(lang).lower(), sense.full_lemma) for lang, sense in senses if sense is not None]
    # shared_relations = set([e.pointer.name for e in synset.outgoing_edges()]).intersection(relations)
    shared_relations = {}
    for e in synset.outgoing_edges(*relations):
        if e.pointer.name not in shared_relations:
            shared_relations[e.pointer.name] = []
        shared_relations[e.pointer.name].append(e.target)

    # output = [get_data_from_relation(edge, lang) for edge in synset.outgoing_edges() if edge.pointer.name in shared_relations]

    # for rel, target_data in output:
    #     if len(target_data) > 0:
    #         shared_relations[rel].append(target_data)
    # shared_relations = {k: v for k, v in shared_relations.items() if len(v) > 0}
    return {"subject_id":str(synset.id), "relations": shared_relations, 
                      "subject_senses": senses}

def get_data_from_relation(edge, lang):
    tgt_syn = bn.get_synset(BabelSynsetID(edge.target))
    if Language.from_iso(lang) in tgt_syn.languages:
        # print(f"Adding target {tgt_syn.main_sense(Language.from_iso(lang)).full_lemma}")
        target_data = {"object_id" : edge.target}
        target_data.update({"object_sense_en": tgt_syn.main_sense(Language.EN).full_lemma})
        target_data.update({f"object_sense_{lang}": tgt_syn.main_sense(Language.from_iso(lang)).full_lemma})
    
        return edge.pointer.name, target_data
    else:
        return edge.pointer.name, {}

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
# %%
import pandas as pd
from babelnet import Language, BabelSynsetID
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import itertools

rel_df = pd.read_csv("datasets/v1/relations_with_examples_expressions.tsv", sep="\t")
relations = rel_df["relation_name"].tolist()
relations = convert_to_babel_relations(relations)
languages = [Language.from_iso(l) for l in [lang, "en"]]	
edits = []

t_start = time.time()
synset_to_senses = {synset:{"src_sense": synset.main_sense(Language.from_iso(lang)), "en_sense": synset.main_sense(Language.EN)} for title, synset in data[:10] if synset is not None}
synset_to_senses = {synset:{"src_sense":senses["src_sense"].full_lemma, "en_sense":senses["en_sense"].full_lemma} for synset, senses in synset_to_senses.items() if all(senses.values())}

synset_to_relations = {str(synset.id):[(e.pointer.name, e, e.target) for e in synset.outgoing_edges() if e.pointer in set(relations)] for synset in synset_to_senses}

target_synset_ids =  list(set([BabelSynsetID(edge["target_id"]) for relations in synset_to_relations.values() for edges in relations.values() for edge in edges]))
print(f"Fetching {len(target_synset_ids)} target synsets")
target_synsets = bn.get_synsets(*(target_synset_ids))
target_senses = {str(synset.id):{"en_sense":synset.main_sense(Language.EN), "src_sense":synset.main_sense(Language.from_iso(lang))} for synset in target_synsets}
target_senses = {syn_id:{"en_sense":senses["en_sense"].full_lemma, "src_sense":senses["src_sense"].full_lemma} for syn_id, senses in target_senses.items() if all(senses.values())}

for synset in synset_to_relations:
    relation_to_edges = synset_to_relations[synset]
    for relation in relation_to_edges:
        edges = relation_to_edges[relation]
        for edge in edges:
            if edge["target_id"] in target_senses:
                edge["target_sense_src"] = target_senses[edge["target_id"]]["src_sense"]
                edge["target_sense_en"] = target_senses[edge["target_id"]]["en_sense"]
            else:
                edges.remove(edge)

# for synset in synset_to_relations:
#     relations = synset_to_relations[synset] 
#     for relation in relations:
#         relations[relation] = [e for e in relations[relation] if e["target_sense_en"] != ""]

print(f"Time taken: {(time.time()-t_start)/60} minutes")

output = {str(synset.id): {"subject_senses": senses, "relations": synset_to_relations[str(synset.id)]} for synset, senses in synset_to_senses.items()}
import json
with open("datasets/v1/mini-dataset2.json", "w") as f:
    json.dump(output, f, indent=4)
f.close()
# %%
from collections import defaultdict
rel_to_synsets = defaultdict(list)

for d in list(synset_to_relations.values()):
    for k, v in d.items():
        [x.pop("edge_id") for x in v if "edge_id" in x]
        rel_to_synsets[k] += v

for rel in rel_to_synsets:
    unique_synsets = {x["target_id"]:x for x in rel_to_synsets[rel]}
    rel_to_synsets[rel] = list(unique_synsets.values())


# %%
import random
for synset in synset_to_relations:
    relation_to_edges = synset_to_relations[synset]
    for relation in relation_to_edges:
        edges = relation_to_edges[relation]
        for edge in edges:
            sampled_syn = random.sample(rel_to_synsets[relation], 1)[0]
        edges["edit"] = sampled_syn


# %%
synsets = [BabelSynsetID(x) for x in all_relations]
print(len(synsets))
t_start = time.time()
r = bn.get_synsets(*(synsets))
print(f"Time taken: {(time.time()-t_start)/60} minutes")
# %%
for relations in synset_to_relations.values():
    for rel in relations.values():
        for edge in edges:
            if edge["target_id"] in target_senses:
                edge["target_sense_en"] = target_senses[edge["target_id"]]["en_sense"]
                edge["target_sense_src"] = target_senses[edge["target_id"]]["src_sense"]
            else:
                edge["target_sense_en"] = ""
                edge["target_sense_src"] = ""
print(synset_to_relations)
# %%
