from pathlib import Path
import pandas as pd
from google.cloud import translate
from utils import extract
import sienna
import json
from typing import List
from google.cloud import translate
import html
import argparse
from google.cloud import storage
from utils import download_blob, extract, folder_exists, delete_folder
from upload_glossary import upload_to_gcs


def translate_text(
    project_id: str = "YOUR_PROJECT_ID",
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

    # Supported language codes: https://cloud.google.com/translate/docs/languages
    operation = client.batch_translate_text(
        request={
            "target_language_codes": target_language_code,
            "source_language_code": source_language_code,
            "parent": parent,
            "input_configs": [input_configs_element],
            "output_config": output_config,
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


# Params
parser = argparse.ArgumentParser(description="Translate entities.")
parser.add_argument("--src-lang", type=str, default="en", help="Source language code")
parser.add_argument(
    "--tgt_langs", default=["it", "de", "fr"], nargs="+", help="Target language code(s)"
)
parser.add_argument("--version", type=str, default="v5", help="Version")
parser.add_argument(
    "--project-id", type=str, default="babeledits-trial", help="Project ID"
)
parser.add_argument(
    "--dataset-path",
    type=str,
    default="datasets/v6/translated/test.json",
    help="Dataset path",
)
parser.add_argument(
    "--save-path",
    type=str,
    default="datasets/v6/translated/translated_with_entities",
    help="Save path",
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
version = args.version
project_id = args.project_id
dataset_path = args.dataset_path
save_path = args.save_path
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
locality_objects = extract(data, "en", "ground_truths_loc")
entities = subjects + objects + locality_objects

df = pd.DataFrame(entities, columns=["entities"])
tsv_src_path = Path(dataset_path).parent / "tsv" / "src"
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

tsv_tgt_path = Path(dataset_path).parent / "tsv" / "tgt"
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

entities_dfs = []
for index, row in index_df.iterrows():
    lang = row["lang"]
    prompt_tgt_path = tsv_tgt_path / f"prompts_{lang}.tsv"
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
    df = df.sort_values("req_id", ascending=True)
    entities_dfs.append(df)

merged_df = pd.concat([df.filter(regex=r"^tgt_") for df in entities_dfs], axis=1)
merged_df["req_id"] = df["req_id"]
merged_df["src"] = df["src"]
merged_df = merged_df[
    ["req_id", "src"] + merged_df.filter(regex=r"^tgt_").columns.tolist()
]
merged_df = merged_df.drop_duplicates(
    subset=["src"] + merged_df.filter(regex=r"^tgt_").columns.tolist()
).dropna()

data_keys = ["subjects", "subjects_mt", "relations"]
edit_keys = [
    "target_id",
    "targets",
    "targets_mt",
    "prompts",
    "prompts_gloss",
    "generality",
    "locality",
]
locality_keys = [
    "ground_truth_id_loc",
    "ground_truths_loc",
    "ground_truths_loc_mt",
    "prompts_loc",
    "prompts_loc_gloss",
]


removed_datapoints = []
for idx, syn_id in enumerate(data):
    data[syn_id]["subjects_mt"] = {}
    en_subject = data[syn_id]["subjects"]["en"]
    subject_mt_transl = merged_df.loc[merged_df["src"] == en_subject]
    data[syn_id]["subjects_mt"].update(
        {
            tgt_lang: subject_mt_transl[f"tgt_{tgt_lang}"].item()
            for tgt_lang in sorted(args.tgt_langs)
        }
    )
    relation = list(data[syn_id]["relations"].keys())[0]
    en_target = data[syn_id]["relations"][relation]["edit"]["targets"]["en"]
    object_mt_transl = merged_df.loc[merged_df["src"] == en_target]
    data[syn_id]["relations"][relation]["edit"].update(
        {
            "targets_mt": {
                tgt_lang: object_mt_transl[f"tgt_{tgt_lang}"].item()
                for tgt_lang in sorted(args.tgt_langs)
            }
        }
    )

    loc_rel = list(data[syn_id]["relations"][relation]["edit"]["locality"].keys())[0]
    loc_gt = data[syn_id]["relations"][relation]["edit"]["locality"][loc_rel][
        "ground_truths_loc"
    ]["en"]
    loc_mt_transl = merged_df.loc[merged_df["src"] == loc_gt]
    data[syn_id]["relations"][relation]["edit"]["locality"][loc_rel].update(
        {
            "ground_truths_loc_mt": {
                tgt_lang: loc_mt_transl[f"tgt_{tgt_lang}"].item()
                for tgt_lang in sorted(args.tgt_langs)
            }
            for tgt_lang in sorted(args.tgt_langs)
        }
    )
    # # Re order
    data[syn_id] = {k: data[syn_id][k] for k in data_keys}
    data[syn_id]["relations"][relation]["edit"] = {
        k: data[syn_id]["relations"][relation]["edit"][k] for k in edit_keys
    }
    data[syn_id]["relations"][relation]["edit"]["locality"] = {
        k: data[syn_id]["relations"][relation]["edit"]["locality"][loc_rel][k]
        for k in locality_keys
    }


Path(save_path).mkdir(parents=True, exist_ok=True)
with open(f"{save_path}/{Path(dataset_path).name}", "w") as f:  #
    json.dump(data, f, indent=4, ensure_ascii=False)
f.close()
