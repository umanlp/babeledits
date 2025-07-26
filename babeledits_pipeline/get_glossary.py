# %%
import sienna
import argparse
import pandas as pd
from pathlib import Path

# Read command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_dir", default="datasets/v5", help="Path to the dataset directory"
)
parser.add_argument(
    "--output_dir", default="glossaries/v5", help="Path to the output directory"
)
parser.add_argument(
    "--langs",
    nargs="+",
    default=[
        "af",
        "ar",
        "az",
        "bg",
        "bn",
        "de",
        "el",
        "en",
        "es",
        "et",
        "eu",
        "fa",
        "fi",
        "fr",
        "gu",
        "he",
        "hi",
        "ht",
        "hr",
        "hu",
        "id",
        "it",
        "ja",
        "jv",
        "ka",
        "kk",
        "ko",
        "lt",
        "ml",
        "mr",
        "ms",
        "my",
        "nl",
        "pa",
        "pl",
        "pt",
        "qu",
        "ro",
        "ru",
        "sw",
        "ta",
        "te",
        "th",
        "tl",
        "tr",
        "uk",
        "ur",
        "vi",
        "yo",
        "zh",
    ],
    help="List of languages",
)
args, _ = parser.parse_known_args()

dataset_dir = args.dataset_dir
output_dir = args.output_dir
langs = sorted(args.langs)

print(f"Reading dataset from {dataset_dir}...")
data = sienna.load(f"{dataset_dir}/dataset.json")
glossary = [list(x["subjects"].values()) for x in data.values()]
glossary_df = pd.DataFrame(glossary, columns=langs)
glossary_df["synset_id"] = data.keys()
glossary_df = glossary_df[["synset_id"] + langs]
print(f"Glossary contains {len(glossary_df)} synsets")
print(glossary_df.head())

print(f"Writing glossary to {output_dir}...")
Path(output_dir).mkdir(parents=True, exist_ok=True)
glossary_df.to_csv(
    f"{output_dir}/glossary.csv", index=False, na_rep="", encoding="utf-8"
)
glossary_df.drop(columns="synset_id").to_csv(
    f"{output_dir}/glossary_no_id.csv", index=False, na_rep="", encoding="utf-8"
)
