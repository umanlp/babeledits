from pathlib import Path
import numpy as np
import sienna
import pandas as pd
import argparse
import json
from utils import read_data, lcs, add_translation


def process_translations(
    tgt_lang, prompts, subjects, en_subjects, translations, glossary
):
    new_prompts = []
    removed_prompts = []
    lcs_repl_prompts = []
    for idx, (subj, en_subj, prompt) in enumerate(zip(subjects, en_subjects, prompts)):
        if subj not in prompt:
            print("-------------------------------")
            print(idx)
            if subj in translations.loc[idx, "tgt"]:
                print("CASE 1")
                rep_prompt = translations.loc[idx, "tgt"]
                assert subj in rep_prompt
            else:
                print("CASE 2")
                res = glossary.loc[
                    glossary["en"] == en_subj, ["synset_id", "en", tgt_lang]
                ]
                print(res, end="\n\n")
                print(f"Subjects: en {en_subjects[idx]} - it {subj}", end="\n\n")
                print(f"Original prompt: {prompt}")
                found = False
                for tgt_term in res[tgt_lang]:
                    tgt_term = str(tgt_term)
                    if tgt_term is not None and tgt_term in prompt:
                        rep_prompt = prompt.replace(tgt_term, subj)
                        found = True
                        assert subj in rep_prompt
                        break
                if not found:
                    for res_idx, en_term in enumerate(res["en"]):
                        en_term = str(en_term)
                        if en_term is not None and en_term in prompt:
                            rep_prompt = prompt.replace(en_term, subj)
                            found = True
                            assert subj in rep_prompt
                            break
                if not found:
                    res_lcs = lcs(prompt, subj)
                    if len(res_lcs) > 0:
                        rep_prompt = prompt.replace(res_lcs, subj)
                        try:
                            assert subj in rep_prompt
                            lcs_repl_prompts.append((prompt, rep_prompt))
                        except AssertionError:
                            print(
                                f"Failed to replace prompt: {prompt}",
                                end="\n-------------------------------",
                            )
                            removed_prompts.append(prompt)
                            continue	
                    else:
                        removed_prompts.append(prompt)
                        print(
                            f"Removed prompt: {prompt}",
                            end="\n-------------------------------",
                        )
                        continue
            print(
                f"Replaced prompt: {rep_prompt}",
                end="\n-------------------------------",
            )
            new_prompts.append(rep_prompt)
        else:
            new_prompts.append(prompt)
    return new_prompts, removed_prompts, lcs_repl_prompts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transl_path",
        type=str,
        default="datasets/v3/tsv/tgt/it/prompts_it.tsv",
        help="Path to translations file",
    )
    parser.add_argument(
        "--gloss_path",
        type=str,
        default="translation/glossaries/v3/glossary.csv",
        help="Path to glossary file",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="datasets/v3/translated/it.json",
        help="Path to dataset file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="datasets/v3/post_proc/it.json",
        help="Path to dataset file",
    )
    parser.add_argument(
        "--tgt_lang", type=str, default="it", help="Language of the dataset"
    )
    args = parser.parse_args()

    translations = pd.read_csv(
        args.transl_path, sep="\t", names=["req_id", "src", "tgt", "tgt_gloss"]
    )
    glossary = pd.read_csv(args.gloss_path)
    translations = translations.sort_values("req_id")
    translations.set_index("req_id", inplace=True)

    data = sienna.load(args.dataset_path)
    subjects, en_subjects, prompts, ground_truth, targets = read_data(
        args.dataset_path, "src"
    )

    new_prompts, removed_prompts, lcs_repl_prompts = process_translations(
        args.tgt_lang, prompts, subjects, en_subjects, translations, glossary
    )
    

    removals = 0 

    for synset in list(data.keys()):
        relation_to_edges = data[synset]['relations']
        for relation in list(relation_to_edges.keys()):
            for edge in list(relation_to_edges[relation]):
                if edge["edit"]["prompt_src"] in removed_prompts:
                    relation_to_edges[relation].remove(edge)
                    removals += 1
                    if len(relation_to_edges[relation]) == 0:
                        del relation_to_edges[relation]


    print(f"Removed prompts: {len(removed_prompts)}, Removals in dataset: {removals}")
    add_translation(data, iter(new_prompts), "prompt_en", "prompt_src")
    
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    f.close()

    subjects, _, prompts, _, _ = read_data(args.output_path, "src")
    subj_check = np.array([s in p for s, p in zip(subjects, prompts)])
    if not subj_check.all():
        print("Some subjects are not in the prompts!")
        print(
            np.array(subjects)[~subj_check], np.array(prompts)[~subj_check], sep="\n\n"
        )
        raise ValueError
    print("DONE!")
