# %%
from typing import List
from google.cloud import translate
from upload_glossary import upload_to_gcs
import sienna
import pandas as pd
from pathlib import Path
import json
import argparse
from utils import add_translation, download_blob, extract, folder_exists, delete_folder
from google.cloud import storage

# from google.cloud import blob
import urllib.parse


def translate_text_with_glossary(
    project_id: str = "YOUR_PROJECT_ID",
    glossary_id: str = "YOUR_GLOSSARY_ID",
    input_uri: str = "YOUR_INPUT_URI",
    output_uri: str = "YOUR_OUTPUT_URI",
    source_language_code: str = "en",
    target_language_code: str = "it",
) -> translate.TranslateTextResponse:
    """Translates a given text using a glossary.

    Args:
        text: The text to translate.
        project_id: The ID of the GCP project that owns the glossary.
        glossary_id: The ID of the glossary to use.

    Returns:
        The translated text."""
    client = translate.TranslationServiceClient()
    location = "us-central1"
    parent = f"projects/{project_id}/locations/{location}"

    gcs_source = {"input_uri": input_uri}

    input_configs_element = {
        "gcs_source": gcs_source,
        "mime_type": "text/plain",  # Can be "text/plain" or "text/html".
    }
    gcs_destination = {"output_uri_prefix": output_uri}
    output_config = {"gcs_destination": gcs_destination}

    glossary = client.glossary_path(
        project_id,
        "us-central1",
        glossary_id,  # The location of the glossary
    )

    glossary_config = translate.TranslateTextGlossaryConfig(glossary=glossary)
    glossaries = {tgt: glossary_config for tgt in tgt_langs}  # target lang as key

    # Supported language codes: https://cloud.google.com/translate/docs/languages
    operation = client.batch_translate_text(
        request={
            "target_language_codes": target_language_code,
            "source_language_code": source_language_code,
            "parent": parent,
            "input_configs": [input_configs_element],
            "output_config": output_config,
            "glossaries": glossaries,
        }
    )

    print("Waiting for operation to complete...")
    response = operation.result()

    print(f"Total Characters: {response.total_characters}")
    print(f"Translated Characters: {response.translated_characters}")

    # Retrieve the translated file names from the output_uri
    storage_client = storage.Client()
    bucket_name = output_uri.split("/")[2]
    prefix = "/".join(output_uri.split("/")[3:])

    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    file_names = []
    for blob in blobs:
        print(f"Translated file: {blob.name}")
        file_names.append(blob.name)

    return response, file_names


parser = argparse.ArgumentParser(description="Translate text using a glossary")
parser.add_argument(
    "--dataset_path", default="datasets/v4/all_langs.json", help="Path to the dataset"
)
parser.add_argument(
    "--project_id", default="babeledits-trial", help="ID of the GCP project"
)
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
    default="translations/v4",
    help="Name of the path where the source files are stored",
)
parser.add_argument(
    "--tgt_blob_path",
    default="translations/v4/",
    help="Name of the path where the translations will stored",
)
parser.add_argument("--glossary_id", default="multi_v4", help="ID of the glossary")
parser.add_argument(
    "--search_key", default="prompt_en", help="Key to search in the dataset"
)
parser.add_argument("--src_lang", default="en", help="Source language code")
parser.add_argument(
    "--tgt_langs", default=["it", "de", "fr"], nargs="+", help="Target language code(s)"
)
parser.add_argument(
    "--output_dir", default="datasets/v4/translated", help="Output directory"
)
parser.add_argument(
    "-d",
    "--delete",
    action="store_true",
    help="Delete the target folder in GCS without asking for user confirmation",
)

args, _ = parser.parse_known_args()

dataset_path = args.dataset_path
project_id = args.project_id
glossary_id = args.glossary_id
search_key = args.search_key
src_lang = args.src_lang
tgt_langs = args.tgt_langs
output_dir = args.output_dir
src_bucket_name = args.src_bucket_name
src_blob_name = Path(args.src_blob_path) / "prompts_en.tsv"

tgt_bucket_name = args.tgt_bucket_name
tgt_blob_path = args.tgt_blob_path
if not args.tgt_blob_path.endswith("/"):
    tgt_blob_path += "/"

data = sienna.load(dataset_path)
print(f"Reading dataset from {dataset_path}...")
prompts = extract(data, search_key)

# Convert prompts to tsv, upload to GCS
df = pd.DataFrame(prompts, columns=["prompt"])
tsv_src_path = Path(dataset_path).parent / "tsv" / "src"
tsv_src_path.mkdir(parents=True, exist_ok=True)
prompt_src_path = tsv_src_path / "prompts_en.tsv"
df.to_csv(prompt_src_path, sep="\t")

print(
    f"Uploading prompts loaded from {prompt_src_path} to {src_bucket_name} at location {src_blob_name}..."
)
upload_to_gcs(str(src_bucket_name), str(prompt_src_path), str(src_blob_name))

input_uri = f"gs://{src_bucket_name}/{src_blob_name}"
output_uri = f"gs://{tgt_bucket_name}/{tgt_blob_path}"
print(
    f"Translating {len(prompts)} prompts from {src_lang} to {tgt_langs} using glossary {glossary_id}"
)
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
response, file_names = translate_text_with_glossary(
    project_id, glossary_id, input_uri, output_uri, src_lang, tgt_langs
)
print(response)
print(f"Files produced {file_names}")

# %%
tsv_tgt_path = Path(dataset_path).parent / "tsv" / "tgt"
tsv_tgt_path.mkdir(parents=True, exist_ok=True)
index_blob_name = [x for x in file_names if x.endswith("index.csv")][0]
index_path = tsv_tgt_path / "index.csv"
print(
    f"Downloading index as well from {tgt_bucket_name} at location {index_blob_name} to {index_path}..."
)
download_blob(tgt_bucket_name, index_blob_name, index_path)
index_df = pd.read_csv(index_path, names=["orig_file", "lang", "output_file"], usecols=[0,1,2])

for index, row in index_df.iterrows():
    lang = row["lang"]
    prompt_tgt_path = tsv_tgt_path / f"prompts_{lang}.tsv"
    tgt_blob_name = row["output_file"].replace("gs://", "").replace(tgt_bucket_name+"/", "")
    print(
        f"Downloading translations from {tgt_bucket_name} at location {tgt_blob_name} to {prompt_tgt_path}..."
    )
    download_blob(tgt_bucket_name, tgt_blob_name, prompt_tgt_path)
