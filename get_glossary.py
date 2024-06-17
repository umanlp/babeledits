import sienna
import argparse
import pandas as pd
from pathlib import Path
# %%

# Read command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", default="datasets/v2", help="Path to the dataset directory")
parser.add_argument("--output_dir", default="translation/glossaries/v1", help="Path to the output directory")
parser.add_argument('--langs', nargs='+', default=["af","ar","az","bg","bn","de","el","en","es","et","eu","fa","fi","fr","gu","he","hi","ht","hu","id","it","ja","jv","ka","kk","ko","lt","ml","mr","ms","my","nl","pa","pl","pt","qu","ro","ru","sw","ta","te","th","tl","tr","uk","ur","vi","wo","yo","zh"], help='List of languages')
args = parser.parse_args([])

dataset_dir = args.dataset_dir
output_dir = args.output_dir
langs = args.langs

subj_transl = {lang: None for lang in langs}
for lang in langs:
    f = sienna.load(f"{dataset_dir}/{lang}.json")
    subj_transl[lang] = [(syn_id, data["subject_senses"]["sense_src"], data["subject_senses"]["sense_en"]) for syn_id, data in f.items()]
    subj_transl[lang] = pd.DataFrame(subj_transl[lang], columns=["synset_id", f"{lang}", "en"])

# %%
glossary = None
for df in subj_transl.values():
    if glossary is None:
        glossary = df
    else:
        glossary = pd.merge(glossary, df, on=["synset_id", "en"], how="outer")

# Reordering columns to have synset_id followed by languages
glossary = glossary[["synset_id", "en"] + langs]

def replace_underscores(text):
    return text.replace('_', ' ') if isinstance(text, str) else text

# Applying the function to the appropriate columns
for col in ["en"] + langs:
    glossary[col] = glossary[col].apply(replace_underscores)

Path(output_dir).mkdir(parents=True, exist_ok=True)
glossary.to_csv(f"{output_dir}/glossary.csv", index=False, na_rep="", encoding='utf-8')
glossary.drop(columns="synset_id").to_csv(f"{output_dir}/glossary_no_id.csv", index=False, na_rep="", encoding='utf-8')