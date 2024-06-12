# %%
# Params 
save_path = "datasets/v2/agg_relations_with_counts.tsv"
langs = ["af","ar","az","bg","bn","de","el","en","es","et","eu","fa","fi","fr","gu","he","hi","ht","hu","id","it","ja","jv","ka","kk","ko","lt","ml","mr","ms","my","nl","pa","pl","pt","qu","ro","ru","sw","ta","te","th","tl","tr","uk","ur","vi","wo","yo","zh"]

# %%

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
rel_df.to_csv(save_path, index=False, sep="\t")
print(rel_df)
