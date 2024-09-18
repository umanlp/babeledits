# %%
import argparse
import json
import os
from pathlib import Path

import pandas as pd
import sienna

from utils import insert_after, remove_space

parser = argparse.ArgumentParser(description="Aggregate Translations")
parser.add_argument(
    "--dataset_path",
    type=str,
    default="datasets/v6/dataset.json",
    help="Path to the dataset file",
)
parser.add_argument(
    "--translation_path",
    type=str,
    default="datasets/v6/tsv/tgt",
    help="Path to the translation files",
)
parser.add_argument(
    "--entity_path",
    type=str,
    default="datasets/v6/tsv_entities/tgt",
    help="Path to the entity files",
)
parser.add_argument(
    "--marked_translation_path",
    type=str,
    default="datasets/v6/tsv_marked/tgt",
    help="Path to the marked translation files",
)
parser.add_argument(
    "--delete_same_prompt",
    action="store_true",
    help="Flag to delete translations with the same prompt",
)

args = parser.parse_args()

def load_translations(translation_path):
    translation_files = [x for x in os.listdir(translation_path) if x.endswith(".tsv")]
    langs = [x.split(".")[0][-2:] for x in translation_files]

    lang_to_transl = {}

    def check(prompt, prompt_gloss):
        return prompt_gloss[:-2] == prompt[:-1]

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


dataset_path = args.dataset_path
entity_path = args.entity_path
pure_mt_path = args.translation_path 
marked_mt_path = args.marked_translation_path
data = sienna.load(dataset_path)

langs, mt_translations = load_translations(pure_mt_path)
langs.sort()

translation_files = [x for x in os.listdir(marked_mt_path) if x.endswith(".tsv")]
langs = [x.split(".")[0][-2:] for x in translation_files]

lang_to_transl = {}

for f, lang in zip(translation_files, langs):
    df = pd.read_csv(
        f"{marked_mt_path}/{f}",
        sep="\t",
        names=[
            "req_id",
            "prompt_type",
            "src",
            f"tgt_{lang}",
            f"subject_{lang}",
            f"object_{lang}",
        ],
        header=0,
    )
    df = df.sort_values("req_id", ascending=True)
    lang_to_transl[lang] = df

marked_translations = pd.concat(list(lang_to_transl.values()), axis=1)
marked_translations = marked_translations.T.drop_duplicates(keep="first").T

index_df = pd.read_csv(
    f"{entity_path}/index.csv",
    names=["orig_file", "lang", "output_file"],
    usecols=[0, 1, 2],
)

entities_dfs = []
for index, row in index_df.iterrows():
    lang = row["lang"]
    entities_tgt_path = Path(entity_path) / f"entities_{lang}.tsv"
    df = pd.read_csv(
        entities_tgt_path,
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


all_langs = sorted(langs + ["en"])
idx = 0
NUM_PROMPTS_PER_DATAPOINT = 3
for syn_idx, syn_id in enumerate(data):
    # Insert subjects_mt and subjects_mt_marked

    en_subj = data[syn_id]["subjects"]["en"]
    data[syn_id] = insert_after(
        data[syn_id],
        "subjects",
        "subjects_mt",
        {
            lang: merged_df.loc[
                merged_df["src"] == data[syn_id]["subjects"]["en"], f"tgt_{lang}"
            ].tolist()[0]
            if lang != "en"
            else en_subj
            for lang in all_langs
        },
    )
    data[syn_id] = insert_after(
        data[syn_id],
        "subjects_mt",
        "subjects_mt_marked",
        {
            lang: marked_translations.loc[idx, f"subject_{lang}"]
            if lang != "en"
            else en_subj
            for lang in all_langs
        },
    )

    for relation in data[syn_id]["relations"]:
        # Insert targets_mt and targets_mt_marked

        en_target = data[syn_id]["relations"][relation]["edit"]["targets"]["en"]
        data[syn_id]["relations"][relation]["edit"] = insert_after(
            data[syn_id]["relations"][relation]["edit"],
            "targets",
            "targets_mt",
            {
                lang: merged_df.loc[
                    merged_df["src"]
                    == data[syn_id]["relations"][relation]["edit"]["targets"]["en"],
                    f"tgt_{lang}",
                ].tolist()[0]
                if lang != "en"
                else en_target
                for lang in all_langs
            },
        )

        data[syn_id]["relations"][relation]["edit"] = insert_after(
            data[syn_id]["relations"][relation]["edit"],
            "targets_mt",
            "targets_mt_marked",
            {
                lang: marked_translations.loc[idx, f"object_{lang}"]
                if lang != "en"
                else en_target
                for lang in all_langs
            },
        )

        # Reliability: insert prompt_mt, prompt_marked, prompt_gloss

        en_prompt = data[syn_id]["relations"][relation]["edit"]["prompts"]["en"]
        gloss_prompts = {
            lang: remove_space(mt_translations.loc[idx, f"tgt_gloss_{lang}"])
            if lang != "en"
            else en_prompt
            for lang in all_langs
        }
        mt_prompts = {
            lang: mt_translations.loc[idx, f"tgt_{lang}"] if lang != "en" else en_prompt
            for lang in all_langs
        }
        marked_prompts = {
            lang: marked_translations.loc[idx, f"tgt_{lang}"]
            if lang != "en"
            else en_prompt
            for lang in all_langs
        }

        data[syn_id]["relations"][relation]["edit"] = insert_after(
            data[syn_id]["relations"][relation]["edit"],
            "prompts",
            "prompts_mt",
            mt_prompts,
        )

        data[syn_id]["relations"][relation]["edit"] = insert_after(
            data[syn_id]["relations"][relation]["edit"],
            "prompts_mt",
            "prompts_mt_marked",
            marked_prompts,
        )

        data[syn_id]["relations"][relation]["edit"] = insert_after(
            data[syn_id]["relations"][relation]["edit"],
            "prompts_mt_marked",
            "prompts_gloss",
            gloss_prompts,
        )
        data[syn_id]["relations"][relation]["edit"].pop("prompts")

        # Generality

        en_prompt = data[syn_id]["relations"][relation]["edit"]["generality"][
            "prompts_gen"
        ]["en"]
        gloss_prompts = {
            lang: remove_space(mt_translations.loc[idx + 1, f"tgt_gloss_{lang}"])
            if lang != "en"
            else en_prompt
            for lang in all_langs
        }
        mt_prompts = {
            lang: mt_translations.loc[idx + 1, f"tgt_{lang}"]
            if lang != "en"
            else en_prompt
            for lang in all_langs
        }
        marked_prompts = {
            lang: marked_translations.loc[idx + 1, f"tgt_{lang}"]
            if lang != "en"
            else en_prompt
            for lang in all_langs
        }

        data[syn_id]["relations"][relation]["edit"]["generality"] = insert_after(
            data[syn_id]["relations"][relation]["edit"]["generality"],
            "prompts_gen",
            "prompts_gen_mt",
            mt_prompts,
        )

        data[syn_id]["relations"][relation]["edit"]["generality"] = insert_after(
            data[syn_id]["relations"][relation]["edit"]["generality"],
            "prompts_gen_mt",
            "prompts_gen_mt_marked",
            marked_prompts,
        )

        data[syn_id]["relations"][relation]["edit"]["generality"] = insert_after(
            data[syn_id]["relations"][relation]["edit"]["generality"],
            "prompts_gen_mt_marked",
            "prompts_gen_gloss",
            gloss_prompts,
        )
        data[syn_id]["relations"][relation]["edit"]["generality"].pop("prompts_gen")

        # Locality
        for locality_relation in data[syn_id]["relations"][relation]["edit"][
            "locality"
        ]:
            # Inserting locality objects

            en_gt_loc = data[syn_id]["relations"][relation]["edit"]["locality"][
                locality_relation
            ]["ground_truths_loc"]["en"]

            data[syn_id]["relations"][relation]["edit"]["locality"][
                locality_relation
            ] = insert_after(
                data[syn_id]["relations"][relation]["edit"]["locality"][
                    locality_relation
                ],
                "ground_truths_loc",
                "ground_truths_loc_mt",
                {
                    lang: merged_df.loc[
                        merged_df["src"]
                        == data[syn_id]["relations"][relation]["edit"]["locality"][
                            locality_relation
                        ]["ground_truths_loc"]["en"],
                        f"tgt_{lang}",
                    ].tolist()[0]
                    if lang != "en"
                    else en_gt_loc
                    for lang in all_langs
                },
            )

            data[syn_id]["relations"][relation]["edit"]["locality"][
                locality_relation
            ] = insert_after(
                data[syn_id]["relations"][relation]["edit"]["locality"][
                    locality_relation
                ],
                "ground_truths_loc_mt",
                "ground_truths_loc_mt_marked",
                {
                    lang: marked_translations.loc[idx + 2, f"object_{lang}"]
                    if lang != "en"
                    else en_gt_loc
                    for lang in all_langs
                },
            )

            en_prompt = data[syn_id]["relations"][relation]["edit"]["locality"][
                locality_relation
            ]["prompts_loc"]["en"]
            gloss_prompts = {
                lang: remove_space(mt_translations.loc[idx + 2, f"tgt_gloss_{lang}"])
                if lang != "en"
                else en_prompt
                for lang in all_langs
            }
            mt_prompts = {
                lang: mt_translations.loc[idx + 2, f"tgt_{lang}"]
                if lang != "en"
                else en_prompt
                for lang in all_langs
            }
            marked_prompts = {
                lang: marked_translations.loc[idx + 2, f"tgt_{lang}"]
                if lang != "en"
                else en_prompt
                for lang in all_langs
            }

            data[syn_id]["relations"][relation]["edit"]["locality"][
                locality_relation
            ] = insert_after(
                data[syn_id]["relations"][relation]["edit"]["locality"][
                    locality_relation
                ],
                "prompts_loc",
                "prompts_loc_mt",
                mt_prompts,
            )

            data[syn_id]["relations"][relation]["edit"]["locality"][
                locality_relation
            ] = insert_after(
                data[syn_id]["relations"][relation]["edit"]["locality"][
                    locality_relation
                ],
                "prompts_loc_mt",
                "prompts_loc_mt_marked",
                marked_prompts,
            )

            data[syn_id]["relations"][relation]["edit"]["locality"][
                locality_relation
            ] = insert_after(
                data[syn_id]["relations"][relation]["edit"]["locality"][
                    locality_relation
                ],
                "prompts_loc_mt_marked",
                "prompts_loc_gloss",
                gloss_prompts,
            )

            data[syn_id]["relations"][relation]["edit"]["locality"][
                locality_relation
            ].pop("prompts_loc")

        idx += NUM_PROMPTS_PER_DATAPOINT


save_path = Path(dataset_path).parent / "translated" / Path(dataset_path).name
save_path.parent.mkdir(parents=True, exist_ok=True)
with open(save_path, "w") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
