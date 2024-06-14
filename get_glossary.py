import sienna
import argparse
import pandas as pd

# %%

# Read command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", default="datasets/v2", help="Path to the dataset directory")
parser.add_argument("--output_dir", default="translation/glossaries", help="Path to the output directory")
args = parser.parse_args([])

dataset_dir = args.dataset_dir
output_dir = args.output_dir

langs = ["it", "fr"]
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

# Reordering columns to have synset_id, en, fr, it
glossary = glossary[["synset_id", "en"] + langs]

def replace_underscores(text):
    return text.replace('_', ' ') if isinstance(text, str) else text

# Applying the function to the appropriate columns
for col in ["en"] + langs:
    glossary[col] = glossary[col].apply(replace_underscores)

print(glossary)
glossary.to_csv(f"{output_dir}/glossary.csv", index=False, na_rep="", encoding='utf-8')
glossary.drop(columns="synset_id").to_csv(f"{output_dir}/glossary_no_id.csv", index=False, na_rep="", encoding='utf-8')