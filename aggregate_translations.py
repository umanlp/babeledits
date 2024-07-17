# %%
import json
from pathlib import Path
import pandas as pd
import sienna
from utils import add_translation
import os
import argparse


parser = argparse.ArgumentParser(description='Aggregate Relations')
parser.add_argument('--translation_path', type=str, default='datasets/v4/tsv/tgt', help='Path to the translation files')
parser.add_argument('--dataset_path', type=str, default='datasets/v4/all_langs.json', help='Path to the dataset file')
parser.add_argument('--output_dir', type=str, default='datasets/v4/translated', help='Path to the output directory')

args = parser.parse_args()

translation_path = args.translation_path
dataset_path = args.dataset_path
output_dir = args.output_dir

def load_translations(translation_path):
    translation_files = [x for x in os.listdir(translation_path) if x.endswith(".tsv")]
    langs = [x.split(".")[0][-2:] for x in translation_files]

    lang_to_transl = {}
    for f, lang in zip(translation_files, langs):
        df = pd.read_csv(f"{translation_path}/{f}", sep="\t", names=["req_id", "src", f"tgt_{lang}", f"tgt_gloss_{lang}"], header=0)
        df = df.sort_values("req_id", ascending=True)
        lang_to_transl[lang] = df

    output_df = pd.concat(list(lang_to_transl.values()), axis=1)
    output_df = output_df.T.drop_duplicates(keep='first').T
    return langs,output_df

data = sienna.load(dataset_path)
langs, output_df = load_translations(translation_path, dataset_path)
# %%

print(f"Adding translations to the dataset in {output_dir}...")
add_translation(data, iter(output_df.iterrows()), "prompt_en", langs)

output_dir = Path(output_dir)
output_dir.mkdir(exist_ok=True)
with open(output_dir / Path(dataset_path).name, "w") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
f.close()
print("DONE!")
