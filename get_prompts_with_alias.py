import pandas as pd
import sienna
from utils import extract
import argparse
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from langcodes import Language
import random

parser = argparse.ArgumentParser(description="Process some parameters.")
parser.add_argument(
    "--dataset_path", type=str, required=True, help="Path to the dataset"
)
parser.add_argument(
    "--langs", nargs="+", required=True, help="Comma-separated list of languages"
)
parser.add_argument(
    "--device", type=str, default="cuda:0", help="Device to use for computation"
)

args = parser.parse_args()

dataset_path = args.dataset_path
langs = args.langs
device = args.device

model_name = "facebook/nllb-200-distilled-600M"
dataset = sienna.load(dataset_path)
subjects = extract(dataset, "subjects")
subject_alias = extract(dataset, "subjects_aliases")


# iterate over rows of df

errors = []
prompt_alias = []
counter = 0
dataset_dir = Path(dataset_path).parent
for lang in langs:
    if lang == "en":
        sel_lang = "en"
        while sel_lang == "en":
            sel_lang = random.choice(langs)
        prompt_path = f"{dataset_dir}/tsv/tgt/prompts_{sel_lang}.tsv"
    else:
        prompt_path = f"{dataset_dir}/tsv/tgt/prompts_{lang}.tsv"
    df = pd.read_csv(prompt_path, sep="\t")
    df.sort_values(by="req_id", inplace=True)

    df = df[df["prompt_type"] == "prompt"].reset_index(drop=True)
    for idx, row in df.iterrows():
        subj_alias = random.sample(subject_alias[idx][lang], 1)[0]
        prompt = row[f"tgt_gloss_{lang}"] if lang != "en" else row["src"]
        subj = subjects[idx][lang]
        if subj not in prompt:
            errors.append((idx, subj, prompt))
        else:
            prompt_alias.append(
                {
                    "syn_id": list(dataset.keys())[idx],
                    "lang": lang,
                    "prompt": prompt,
                    "subject_alias": subj_alias,
                    "prompt_alias": prompt.replace(subj, subj_alias),
                }
            )

alias_df = pd.DataFrame(prompt_alias)
print(alias_df)


nllb_df = pd.read_table("nllb_lang_codes.md", sep="|")
nllb_df.columns = (
    nllb_df.columns.str.strip()
)  # Remove leading/trailing whitespace from column names
nllb_df = nllb_df.apply(lambda x: x.str.strip())
nllb_df["lang3_code"] = nllb_df["FLORES-200 code"].apply(lambda x: x.split("_")[0])

lang_3_codes = [Language.get(language).to_alpha3() for language in langs]
corrections = {
    "fas": "pes",
    "ara": "arb",
    "que": "quy",
    "aze": "azj",
    "msa": "zsm",
    "swa": "swh",
    "uzb": "uzn",
    "nor": "nob",
    "fil": "tgl",
}
lang_3_codes = [corrections.get(code, code) for code in lang_3_codes]
lang2_to_3_codes = {lang2: lang3 for lang2, lang3 in zip(langs, lang_3_codes)}

for lang in lang_3_codes:
    nllb_lang = str(nllb_df[nllb_df["lang3_code"] == lang]["FLORES-200 code"].values[0])
    print(nllb_lang)


model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=True, device_map=device)

for lang in langs:
    nllb_lang = nllb_df[nllb_df["lang3_code"] == lang2_to_3_codes[lang]][
        "FLORES-200 code"
    ].values[0]
    prompts = alias_df[alias_df["lang"] == lang]["prompt_alias"].tolist()
    print("Correcting prompts for language", lang)
    print("Number of prompts:", len(prompts))
    print(prompts)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=True,
        src_lang=nllb_lang,
        device_map=device,
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(nllb_lang),
        max_length=30,
    )
    translations = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    alias_df.loc[alias_df["lang"] == lang, "prompts_corrected"] = translations

alias_df.to_csv(f"{dataset_dir}/prompts_with_alias.csv", index=False)
