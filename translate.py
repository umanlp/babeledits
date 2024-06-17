# %%
from typing import List
from google.cloud import translate
import sienna
import html
from pathlib import Path
import json
import argparse


def extract(data, search_key):
    output = []
    if isinstance(data, dict):
        for key, value in data.items():
            if key == search_key:
                output.append(value)
            else:
                output.extend(extract(value, search_key))
    elif isinstance(data, list):
        for item in data:
            output.extend(extract(item, search_key))
    return output

def add_translation(data, transl_it, search_key, translation_key):
    if isinstance(data, dict):
        keys = list(data.keys())  
        for key in keys:
            value = data[key]
            if key == search_key:
                try:
                    next_transl = next(transl_it)
                    data[translation_key] = next_transl
                except StopIteration:
                    return
            else:
                add_translation(value, transl_it, search_key, translation_key)
    elif isinstance(data, list):
        for item in data:
            add_translation(item, transl_it, search_key, translation_key)


def translate_text_with_glossary(
    sentences: List[str] = "YOUR_TEXT_TO_TRANSLATE",
    project_id: str = "YOUR_PROJECT_ID",
    glossary_id: str = "YOUR_GLOSSARY_ID",
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

    glossary = client.glossary_path(
        project_id, "us-central1", glossary_id  # The location of the glossary
    )

    glossary_config = translate.TranslateTextGlossaryConfig(glossary=glossary)

    # Supported language codes: https://cloud.google.com/translate/docs/languages
    response = client.translate_text(
        request={
            "contents": sentences,
            "target_language_code": target_language_code,
            "source_language_code": source_language_code,
            "parent": parent,
            "glossary_config": glossary_config,
        }
    )

    print("Translated text: \n")

    translations = []
    for translation in response.glossary_translations:
        cleaned_translation = html.unescape(translation.translated_text)
        print(f"\t {cleaned_translation}")
        translations.append(cleaned_translation)

    return response, translations

parser = argparse.ArgumentParser(description="Translate text using a glossary")
parser.add_argument("--dataset_path", default="datasets/v2/it.json", help="Path to the dataset")
parser.add_argument("--project_id", default="babeledits-trial", help="ID of the GCP project")
parser.add_argument("--glossary_id", default="it_en_v1", help="ID of the glossary")
parser.add_argument("--search_key", default="prompt_en", help="Key to search in the dataset")
parser.add_argument("--src_lang", default="en", help="Source language code")
parser.add_argument("--tgt_lang", default="it", help="Target language code")
parser.add_argument("--output_dir", default="datasets/v2/translated", help="Output directory")

args = parser.parse_args([])

dataset_path = args.dataset_path
project_id = args.project_id
glossary_id = args.glossary_id
search_key = args.search_key
src_lang = args.src_lang
tgt_lang = args.tgt_lang
output_dir = args.output_dir

data = sienna.load(dataset_path)
prompts = extract(data, search_key)
response, translations = translate_text_with_glossary(prompts, project_id, glossary_id, src_lang, tgt_lang)
# %%

translation_key = "prompt_src"
add_translation(data, iter(translations), search_key, translation_key)

output_dir = Path(output_dir)
output_dir.mkdir(exist_ok=True)
with open(output_dir / Path(dataset_path).name, "w") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
f.close()