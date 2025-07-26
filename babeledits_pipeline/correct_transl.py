# %%
from re import A
import pandas as pd
from utils import extract_target, extract_subject

langs = "ar de es fr hr it ja nl sw zh".split(" ")
for lang in langs:
    path = f"datasets/v6_marked/tsv_marked/tgt/prompts_marked_{lang}.tsv"
    df = pd.read_csv(path, sep="\t")

    no_subj_mask = ~df[f"tgt_raw_{lang}"].str.contains("<s>")
    no_obj_mask = ~df[f"tgt_raw_{lang}"].str.contains("<o>")

    # print(df.loc[no_obj_mask, 'object'])
    if no_subj_mask.any():
        print(f"Fixing subjects for {lang} ({no_subj_mask.sum()})")
        df.loc[no_subj_mask, "subject"] = df.loc[no_subj_mask, f"tgt_raw_{lang}"].apply(
            lambda x: extract_subject(x)
        )
    if no_obj_mask.any():
        print(f"Fixing objects for {lang} ({no_obj_mask.sum()})")
        df.loc[no_obj_mask, "object"] = df.loc[no_obj_mask, f"tgt_raw_{lang}"].apply(
            lambda x: extract_target(x)
        )

    df.to_csv(path, sep="\t", index=False)

# print(print(df.loc[no_obj_mask, 'object']))
# %%

import pandas as pd
from utils import extract_target, extract_subject, clean_prompt

langs = "ar de es fr hr it ja nl sw zh".split(" ")
# langs = "ar it".split(" ")

for lang in langs:
    path = f"datasets/v6_marked/tsv_marked/tgt/prompts_marked_{lang}.tsv"
    df = pd.read_csv(path, sep="\t")
    df["subject"] = [extract_subject(x) for x in df[f"tgt_{lang}"]]
    df["object"] = [extract_target(x) for x in df[f"tgt_{lang}"]]
    df[f"tgt_raw_{lang}"] = df[f"tgt_{lang}"]
    df[f"tgt_{lang}"] = [clean_prompt(x) for x in df[f"tgt_{lang}"]]
    df = df[
        [
            "req_id",
            "prompt_type",
            "src",
            f"tgt_raw_{lang}",
            f"tgt_{lang}",
            "subject",
            "object",
        ]
    ]

    df.sort_values("req_id", inplace=True)
    df.to_csv(path, sep="\t", index=False)

# %%
from pathlib import Path
from utils import download_blob, clean_prompt, extract_target, extract_subject
from dataclasses import dataclass

index_path = "datasets/v6_marked/tsv_marked/tgt/index_marked.csv"
dataset_path = "datasets/v6_marked/dataset.json"
tgt_bucket_name = "babeledits-transl-tgt"
tsv_tgt_path = Path(dataset_path).parent / "tsv_marked" / "tgt"
index_df = pd.read_csv(
    index_path, names=["orig_file", "lang", "output_file"], usecols=[0, 1, 2]
)


@dataclass
class Args:
    locality: bool = True
    rephrase: bool = True


args = Args()

for index, row in index_df.iterrows():
    lang = row["lang"]
    prompt_tgt_path = tsv_tgt_path / f"prompts_marked_{lang}.tsv"
    tgt_blob_name = (
        row["output_file"].replace("gs://", "").replace(tgt_bucket_name + "/", "")
    )
    print(
        f"Downloading translations from {tgt_bucket_name} at location {tgt_blob_name} to {prompt_tgt_path}..."
    )
    download_blob(tgt_bucket_name, tgt_blob_name, prompt_tgt_path)

    df = pd.read_csv(
        prompt_tgt_path,
        sep="\t",
        names=["req_id", "src", f"tgt_{lang}"],
        header=0,
    )
    if args.rephrase and args.locality:
        pattern = ["prompt", "prompt_gen", "prompt_loc"]
        df.sort_values("req_id", inplace=True)
        df["prompt_type"] = pattern * (len(df) // len(pattern))
    elif args.rephrase:
        pattern = ["prompt", "prompt_gen"]
        df.sort_values("req_id", inplace=True)
        df["prompt_type"] = pattern * (len(df) // len(pattern))
    elif args.locality:
        pattern = ["prompt", "prompt_loc"]
        df.sort_values("req_id", inplace=True)
        df["prompt_type"] = pattern * (len(df) // len(pattern))
    else:
        df["prompt_type"] = "prompt"
    df["subject"] = [extract_subject(x) for x in df[f"tgt_{lang}"]]
    df["object"] = [extract_target(x) for x in df[f"tgt_{lang}"]]
    df[f"tgt_raw_{lang}"] = df[f"tgt_{lang}"]
    df[f"tgt_{lang}"] = [clean_prompt(x) for x in df[f"tgt_{lang}"]]
    df = df[
        [
            "req_id",
            "prompt_type",
            "src",
            f"tgt_raw_{lang}",
            f"tgt_{lang}",
            "subject",
            "object",
        ]
    ]

    df.sort_values("req_id", inplace=True)

    if df.isnull().values.any():
        print(f"⚠️ Data for {lang} has some problems with NaN values. Please check.")
    df.to_csv(prompt_tgt_path, sep="\t", index=False)

# %%

langs = "ar de es fr hr it ja nl sw zh".split(" ")
import pandas as pd

for lang in langs:
    path = f"datasets/v6_marked/tsv_marked/tgt/prompts_marked_{lang}.tsv"
    df = pd.read_csv(path, sep="\t")
    print(lang)
    print(df.isnull().sum().sum())
    print("__________________")
