# %%
from google.cloud import translate
from upload_glossary import upload_to_gcs
import sienna
import pandas as pd
from pathlib import Path
import argparse
from utils import (
    download_blob,
    folder_exists,
    delete_folder,
    translate_text,
)
import json


# %%

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate text using a glossary")
    parser.add_argument(
        "--dataset_path",
        default="data_ppl/ME-PPL_50.json",
        help="Path to the dataset",
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
        default="translations_ppl/50",
        help="Name of the path where the source files are stored",
    )
    parser.add_argument(
        "--tgt_blob_path",
        default="translation_ppl/50/",
        help="Name of the path where the translations will stored",
    )
    parser.add_argument("--src_lang", default="en", help="Source language code")
    parser.add_argument(
        "--tgt_langs",
        default=["it", "ja", "fr", "de"],
        nargs="+",
        help="Target language code(s)",
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
    src_lang = args.src_lang
    tgt_langs = args.tgt_langs
    src_bucket_name = args.src_bucket_name
    src_blob_name = Path(args.src_blob_path) / f"{Path(args.dataset_path).name}.tsv"

    tgt_bucket_name = args.tgt_bucket_name
    tgt_blob_path = args.tgt_blob_path
    if not args.tgt_blob_path.endswith("/"):
        tgt_blob_path += "/"

    data = sienna.load(dataset_path)
    all_prompts = [x["Text"] for x in data]

    # Convert prompts to tsv, upload to GCS
    df = pd.DataFrame(all_prompts, columns=["prompt"])
    tsv_src_path = Path(dataset_path).parent / "tsv" / "src"
    tsv_src_path.mkdir(parents=True, exist_ok=True)
    prompt_src_path = tsv_src_path / f"{Path(args.dataset_path).name}.tsv"
    df.to_csv(prompt_src_path, sep="\t")

# %%

    print(
        f"Uploading prompts loaded from {prompt_src_path} to {src_bucket_name} at location {src_blob_name}..."
    )
    upload_to_gcs(str(src_bucket_name), str(prompt_src_path), str(src_blob_name))

    input_uri = f"gs://{src_bucket_name}/{src_blob_name}"
    output_uri = f"gs://{tgt_bucket_name}/{tgt_blob_path}"
    print(f"Translating {len(all_prompts)} prompts from {src_lang} to {tgt_langs}")
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

# %%

    tsv_tgt_path = Path(dataset_path).parent / "tsv" / "tgt"
    tsv_tgt_path.mkdir(parents=True, exist_ok=True)
    index_blob_name = [x for x in file_names if x.endswith("index.csv")][0]
    index_path = tsv_tgt_path / "index_marked.csv"
    print(
        f"Downloading index as well from {tgt_bucket_name} at location {index_blob_name} to {index_path}..."
    )
    download_blob(tgt_bucket_name, index_blob_name, index_path)
    index_df = pd.read_csv(
        index_path, names=["orig_file", "lang", "output_file"], usecols=[0, 1, 2]
    )

    lang_to_prompts = {}
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
            header=0)
        df.sort_values("req_id", inplace=True)
        df.to_csv(prompt_tgt_path, sep="\t")
        lang_to_prompts[lang] = df[f"tgt_{lang}"].tolist()

# %%
sorted_langs = sorted(list(lang_to_prompts.keys()) + ["en"]) 
output_path = dataset_path.replace(".json", "_translated.json")
for idx, datapoint in enumerate(data):
    datapoint['Text'] = {lang:lang_to_prompts[lang][idx].strip() if lang != "en" else datapoint['Text'].strip() for lang in sorted_langs}

with open(output_path, "w") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
