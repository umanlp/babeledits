# %%
# Params 
output_folder = "datasets/v2"
langs = ["af","ar","az","bg","bn","de","el","en","es","et","eu","fa","fi","fr","gu","he","hi","ht","hu","id","it","ja","jv","ka","kk","ko","lt","ml","mr","ms","my","nl","pa","pl","pt","qu","ro","ru","sw","ta","te","th","tl","tr","uk","ur","vi","wo","yo","zh"]

# %%
from collections import defaultdict
from babelnet import Language
from babelnet.resources import BabelSynsetID
import babelnet as bn
import pickle
import pandas as pd
import random
import pickle
from collections import Counter
import pandas as pd

rel_counter = Counter()
for lang in langs:
    file_path = f'synsets2/{lang}/{lang}_syns.pkl'
    print(f"> Loading data for {lang} from {file_path}")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    f.close()
    print(f"> Data for {lang} was loaded")
    relations = [e.pointer.name for _, synset in data for e in synset.outgoing_edges()]
    rel_counter.update(relations)

# %%
import pandas as pd

rel_df = pd.DataFrame(rel_counter.items(), columns=['relation_name', 'count'])
rel_df.sort_values(by='count', ascending=False, inplace=True)
# remove all relations whose name ends with ym or YM, or that have some symbol derived from wordnet
rel_df = rel_df[~rel_df.relation_name.str.contains("%|#|~|@|%|\+")]
rel_df = rel_df[~rel_df.relation_name.str.endswith("ym") & ~rel_df.relation_name.str.endswith("YM")]
# rel_df.to_csv(save_path, index=False, sep="\t")
print(rel_df)

# %%
# Params
lang = "en"
file_path = f'synsets2/{lang}/{lang}_syns.pkl'
max_rel = 200

relations = rel_df["relation_name"].tolist()[:max_rel]

print(f"> Loading data for {lang} from {file_path}")
with open(file_path, 'rb') as f:
    data = pickle.load(f)

bad_relations = []
subj_and_obj = defaultdict(dict)
for relation in relations:
    print(relation, end=",")
    found = False
    while not found:

        count = 0
        max_count = 2000
        random.shuffle(data)
        for _, synset in data:
            count += 1
            if count > max_count:
                # print("Max count reached")
                bad_relations.append(relation)
                found = True
                break
            if synset is not None:
                for edge in synset.outgoing_edges():
                    if edge.pointer.name == relation:
                        subject_sense = synset.main_sense(Language.EN)
                        target_sense = bn.get_synset(BabelSynsetID(edge.target)).main_sense(Language.EN)
                        if all([subject_sense, target_sense]):
                            subject = subject_sense.full_lemma.replace("_", " ")
                            object = target_sense.full_lemma.replace("_", " ")
                            # print(f"{syn},{tgt_syn}")
                            subj_and_obj[relation]["subject"] = subject
                            subj_and_obj[relation]["object"] = object
                            found = True
                            break
                if found:
                    break

print(subj_and_obj)

# %%
# for each value in the column relation_name of rel_df, get the corresponding subject and object from subj_and_obj and add them to the dataframe
rel_df["subject"] = rel_df["relation_name"].apply(lambda x: subj_and_obj[x]["subject"] if x in subj_and_obj else None)
rel_df["object"] = rel_df["relation_name"].apply(lambda x: subj_and_obj[x]["object"] if x in subj_and_obj else None)
rel_df.to_csv(f"{output_folder}/agg_relations_with_examples.tsv", sep="\t", index=False)