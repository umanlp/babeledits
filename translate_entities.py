from pathlib import Path
import pandas as pd
from google.cloud import translate
from utils import extract
import sienna
import json
from typing import List
import html
import argparse
from utils import download_blob, extract, folder_exists, delete_folder, translate_text
from upload_glossary import upload_to_gcs


# Params
parser = argparse.ArgumentParser(description="Translate entities.")
parser.add_argument("--src_lang", type=str, default="en", help="Source language code")
parser.add_argument(
    "--tgt_langs", default=["it", "de", "fr"], nargs="+", help="Target language code(s)"
)
parser.add_argument(
    "--project_id", type=str, default="babeledits-trial", help="Project ID"
)
parser.add_argument(
    "--dataset_path",
    type=str,
    default="datasets/v6/translated/test.json",
    help="Dataset path",
)
# parser.add_argument(
#     "--save-path",
#     type=str,
#     default="datasets/v6/translated/translated_with_entities",
#     help="Save path",
# )
parser.add_argument(
    "--src_bucket_name",
    default="babeledits-transl-src",
    help="Name of the bucket which contains files to be translated",
)
parser.add_argument(
    "--tgt_bucket_name",
    default="babeledits-transl-tgt",
    help="Name of the bucket which will contain output translations",
)
parser.add_argument(
    "--src_blob_path",
    default="transl_entities/v6",
    help="Name of the path where the source files are stored",
)
parser.add_argument(
    "--tgt_blob_path",
    default="transl_entities/v6",
    help="Name of the path where the translations will stored",
)
parser.add_argument(
    "-d",
    "--delete",
    action="store_true",
    help="Delete the target folder in GCS without asking for user confirmation",
)

args = parser.parse_args()

src_lang = args.src_lang
tgt_langs = args.tgt_langs
project_id = args.project_id
dataset_path = args.dataset_path
# save_path = args.save_path
delete = args.delete
src_bucket_name = args.src_bucket_name
src_blob_name = Path(args.src_blob_path) / "entities_en.tsv"
tgt_bucket_name = args.tgt_bucket_name
tgt_blob_path = args.tgt_blob_path
if not args.tgt_blob_path.endswith("/"):
    tgt_blob_path += "/"

# Convert sentences to tsv, upload to GCS
data = sienna.load(dataset_path)
subjects = extract(data, "en", "subjects")
objects = extract(data, "en", "targets")
ground_truths = extract(data, "en", "ground_truths")
ground_truths_port = extract(data, "en", "ground_truths_port", strict=False)
ground_truths_port = [e for e in ground_truths_port if e]
ground_truths_loc = extract(data, "en", "ground_truths_loc", strict=False)
entities = subjects + objects + ground_truths_loc + ground_truths + ground_truths_port

df = pd.DataFrame(entities, columns=["entities"])
tsv_src_path = Path(dataset_path).parent / "tsv_entities" / "src"
tsv_src_path.mkdir(parents=True, exist_ok=True)
prompt_src_path = tsv_src_path / "entities_en.tsv"
df.to_csv(prompt_src_path, sep="\t")

print(
    f"Uploading prompts loaded from {prompt_src_path} to {src_bucket_name} at location {src_blob_name}..."
)
upload_to_gcs(str(src_bucket_name), str(prompt_src_path), str(src_blob_name))

input_uri = f"gs://{src_bucket_name}/{src_blob_name}"
output_uri = f"gs://{tgt_bucket_name}/{tgt_blob_path}"
print(f"Translating {len(entities)} prompts from {src_lang} to {tgt_langs}")
if folder_exists(output_uri):
    if args.delete:
        delete_folder(output_uri)
    else:
        user_input = input(
            f"The URI {output_uri} exists. Do you want to delete it? (yes/no): "
        )
        if user_input.lower() == "yes":
            delete_folder(output_uri)
        else:
            print("Exiting...")
            exit()
print(f"Input URI: {input_uri}", f"Output URI: {output_uri}", sep="\n")
response, file_names = translate_text(
    project_id, input_uri, output_uri, src_lang, tgt_langs
)
print(response)
print(f"Files produced {file_names}")

tsv_tgt_path = Path(dataset_path).parent / "tsv_entities" / "tgt"
tsv_tgt_path.mkdir(parents=True, exist_ok=True)
index_blob_name = [x for x in file_names if x.endswith("index.csv")][0]
index_path = tsv_tgt_path / "index.csv"
print(
    f"Downloading index as well from {tgt_bucket_name} at location {index_blob_name} to {index_path}..."
)
download_blob(tgt_bucket_name, index_blob_name, index_path)
index_df = pd.read_csv(
    index_path, names=["orig_file", "lang", "output_file"], usecols=[0, 1, 2]
)

for index, row in index_df.iterrows():
    lang = row["lang"]
    entities_tgt_path = tsv_tgt_path / f"entities_{lang}.tsv"
    tgt_blob_name = (
        row["output_file"].replace("gs://", "").replace(tgt_bucket_name + "/", "")
    )
    print(
        f"Downloading translations from {tgt_bucket_name} at location {tgt_blob_name} to {entities_tgt_path}..."
    )
    download_blob(tgt_bucket_name, tgt_blob_name, entities_tgt_path)
    df = pd.read_csv(
        entities_tgt_path,
        sep="\t",
        names=["req_id", "src", f"tgt_{lang}"],
        header=0,
    )
    df = df.sort_values("req_id", ascending=True)
    df.to_csv(entities_tgt_path, sep="\t", index=False)
    print(f"Saved entity translations for {lang} to {entities_tgt_path}")
