# %%
import json
from pathlib import Path
import pandas as pd
import sienna
from utils import add_translations, extract
import os
import argparse


parser = argparse.ArgumentParser(description="Aggregate Relations")
parser.add_argument(
    "--translation_path",
    type=str,
    default="datasets/v4/tsv/tgt",
    help="Path to the translation files",
)
parser.add_argument(
    "--dataset_path",
    type=str,
    default="datasets/v4/all_langs.json",
    help="Path to the dataset file",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="datasets/v4/translated",
    help="Path to the output directory",
)
parser.add_argument(
    "--delete_same_prompt",
    action="store_true",
    help="Flag to delete translations with the same prompt",
)

args = parser.parse_args()

translation_path = args.translation_path
dataset_path = args.dataset_path
output_dir = args.output_dir


def load_translations(translation_path):
    translation_files = [x for x in os.listdir(translation_path) if x.endswith(".tsv")]
    langs = [x.split(".")[0][-2:] for x in translation_files]

    lang_to_transl = {}
    for f, lang in zip(translation_files, langs):
        df = pd.read_csv(
            f"{translation_path}/{f}",
            sep="\t",
            names=["req_id", "prompt_type", "src", f"tgt_{lang}", f"tgt_gloss_{lang}"],
            header=0,
        )
        df = df.sort_values("req_id", ascending=True)
        lang_to_transl[lang] = df

    output_df = pd.concat(list(lang_to_transl.values()), axis=1)
    output_df = output_df.T.drop_duplicates(keep="first").T
    return langs, output_df


data = sienna.load(dataset_path)
tgt_langs, output_df = load_translations(translation_path)
# %%

print(f"Adding translations to the dataset in {output_dir}...")
langs = tgt_langs + ["en"]
langs.sort()
add_translations(data, output_df, langs)

if (
    args.delete_same_prompt
):  # only for bilingual datasets where it could be that prompt = prompt_gloss
    print("Removing translations with the same prompt...")
    idxs_to_remove = []

    for lang in tgt_langs:
        prompts = extract(data, "prompts", lang)
        prompts_gloss = extract(data, "prompts_gloss", lang)
        for i, (prompt, prompt_gloss) in enumerate(zip(prompts, prompts_gloss)):
            if prompt == prompt_gloss:
                idxs_to_remove.append(i)
    keys_to_remove = [list(data.keys())[i] for i in idxs_to_remove]
    print(f"Removing {len(idxs_to_remove)} translations with the same prompt...")
    for k in keys_to_remove:
        data.pop(k)


output_dir = Path(output_dir)
output_dir.mkdir(exist_ok=True)
with open(output_dir / Path(dataset_path).name, "w") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
f.close()
print("DONE!")
