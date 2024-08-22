from pathlib import Path
from utils import extract
import sienna
import json
from typing import List
from google.cloud import translate
import html
import argparse


def translate_entities(
    sentences: List[str] = "YOUR_TEXT_TO_TRANSLATE",
    project_id: str = "YOUR_PROJECT_ID",
    source_language_code: str = "en",
    target_language_code: str = "it",
) -> translate.TranslateTextResponse:
    """Translates a given text using a glossary.

    Args:
        text: The text to translate.
        project_id: The ID of the GCP project that owns the glossary.
        source_language_code: The language of the text to translate.
        target_language_code: The language to translate the text into.
    Returns:
        The translated text."""
    client = translate.TranslationServiceClient()
    location = "us-central1"
    parent = f"projects/{project_id}/locations/{location}"

    # Supported language codes: https://cloud.google.com/translate/docs/languages
    response = client.translate_text(
        request={
            "contents": sentences,
            "target_language_code": target_language_code,
            "source_language_code": source_language_code,
            "parent": parent,
        }
    )

    print("Translated text: \n")

    translations = []
    for translation in response.translations:
        cleaned_translation = html.unescape(translation.translated_text)
        print(f"\t {cleaned_translation}")
        translations.append(cleaned_translation)

    return response, translations


# Params
parser = argparse.ArgumentParser(description="Translate entities.")
parser.add_argument("--src-lang", type=str, default="en", help="Source language code")
parser.add_argument("--tgt-lang", type=str, default="it", help="Target language code")
parser.add_argument("--version", type=str, default="v5", help="Version")
parser.add_argument(
    "--project-id", type=str, default="babeledits-trial", help="Project ID"
)
parser.add_argument(
    "--dataset-path",
    type=str,
    default="datasets/v5/hard/it/translated",
    help="Dataset path",
)
parser.add_argument(
    "--save-path",
    type=str,
    default="datasets/v5/hard/it/translated_with_entities",
    help="Save path",
)

args = parser.parse_args()

src_lang = args.src_lang
tgt_lang = args.tgt_lang
version = args.version
project_id = args.project_id
dataset_path = args.dataset_path
save_path = args.save_path

# Convert sentences to tsv, upload to GCS
data = sienna.load(dataset_path + "/dataset.json")
subjects = extract(data, "en", "subjects")
objects = extract(data, "en", "targets")
entities = subjects + objects

all_translations = []

batch_size = 1024
num_batches = (len(entities) + batch_size - 1) // batch_size

for i in range(num_batches):
    start_idx = i * batch_size

    end_idx = min((i + 1) * batch_size, len(entities))
    batch_entities = entities[start_idx:end_idx]

    response, translations = translate_entities(batch_entities, project_id, src_lang, tgt_lang)
    all_translations.extend(translations)

print(f"Translations\n{translations}")

trans_subjects, trans_objects = (
    all_translations[: len(subjects)],
    all_translations[len(subjects) :],
)

data_keys = ["subjects", "subjects_mt", "relations"]
edit_keys = [
    "target_id",
    "targets",
    "targets_mt",
    "prompts",
    "prompts_gloss",
    "targets_mt",
]

def check_presence(datapoint, lang, trans_subject, subject):
    relation = list(datapoint["relations"].keys())[0]
    prompt = datapoint["relations"][relation]["edit"]["prompts"][lang]
    if trans_subject in prompt or subject in prompt:
        return True
    else:
        return False

removed_datapoints = []
for idx, syn_id in enumerate(data):
    if check_presence(data[syn_id], tgt_lang, trans_subjects[idx], subjects[idx]):
        data[syn_id]["subjects_mt"] = {}
        data[syn_id]["subjects_mt"].update({tgt_lang: trans_subjects[idx]})
        relation = list(data[syn_id]["relations"].keys())[0]
        data[syn_id]["relations"][relation]["edit"].update(
            {"targets_mt": {tgt_lang: trans_objects[idx]}}
        )
        # Re order
        data[syn_id] = {k: data[syn_id][k] for k in data_keys}
        data[syn_id]["relations"][relation]["edit"] = {
            k: data[syn_id]["relations"][relation]["edit"][k] for k in edit_keys
        }
    else:
        removed_datapoints.append(syn_id)

print("Original size of the data: ", len(data))
data = {k: data[k] for k in data if k not in removed_datapoints}
print(f"Removed {len(removed_datapoints)} datapoints")
print("Size of the data after filtering: ", len(data))

Path(save_path).mkdir(parents=True, exist_ok=True)
with open(f"{save_path}/dataset.json", "w") as f:  #
    json.dump(data, f, indent=4, ensure_ascii=False)
f.close()
