# %%

## Params
from datetime import timedelta, date

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
import babelnet as bn
from babelnet import Language, BabelSynsetID
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

rel_df = pd.read_csv("synsets/relations_with_examples_expressions.tsv", sep="\t")
relations = rel_df["relation_name"].tolist()
relations = convert_to_babel_relations(relations)
languages = [Language.from_iso(l) for l in [lang, "en"]]	
edits = []

t_start = time.time()
# with ThreadPoolExecutor(max_workers=10) as executor:
#     extracted = [executor.submit(get_data_from_synset, synset, lang, relations) for title, synset in data[:100] if synset is not None]
#     for future in as_completed(extracted):
#         edits.append(future.result())

edits = [get_data_from_synset(synset, languages, relations) for title, synset in data[:100] if synset is not None]
print(f"Time taken: {(time.time()-t_start)/60} minutes")


import json
with open("datasets/v1/mini-dataset.json", "w") as f:
    json.dump(edits, f, indent=4)
f.close()
# %%


from itertools import chain
all_relations = [chain(*element["relations"].values()) for element in edits]
all_relations = list(chain(*all_relations))
print(len(all_relations) , len(set(all_relations)))
synsets = [BabelSynsetID(x) for x in all_relations]
import babelnet as bn
t_start = time.time()
bn.get_synsets(*(synsets))
print(f"Time taken: {(time.time()-t_start)/60} minutes")
# %%
