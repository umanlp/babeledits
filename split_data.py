"""
Split the big dataset in train/val/test
"""

import json
from collections import defaultdict

import numpy as np
from sklearn.model_selection import train_test_split

from utils import extract


def split_data(data, random_state=None, test_size=0.1, val_size=0.13):
    keys = list(data.keys())
    rel_types = [next(iter(data[key]["relations"])) for key in keys]

    # rare relationships are merged so that the stratification doesn't fail
    rare_type_to_ids = defaultdict(list)
    non_rare_types = set()
    min_rel_size = 3
    for i, rel_type in enumerate(rel_types):
        rare_type_to_ids[rel_type].append(i)
        if len(rare_type_to_ids[rel_type]) >= min_rel_size:
            non_rare_types.add(rel_type)
            del rare_type_to_ids[rel_type]

    rare_ids = sum(rare_type_to_ids.values(), [])

    print(
        f"Got {len(rare_type_to_ids)} relationship types (over a total of {len(set(rel_types))}) that have less than {min_rel_size} samples, which cannot be stratified.\n"
        + f"They represent a total of {len(rare_ids)} samples, which will merged into a single group."
    )

    for idx in rare_ids:
        rel_types[idx] = "__DEFAULT_RELATIONSHIP__"

    train_keys, test_keys, train_rel_types, _ = train_test_split(
        keys,
        rel_types,
        stratify=rel_types,
        test_size=test_size,
        random_state=random_state,
    )
    train_keys, val_keys = train_test_split(
        train_keys,
        stratify=train_rel_types,
        test_size=val_size,
        random_state=random_state,
    )

    return train_keys, test_keys, val_keys


def get_sample_mask(data, langs, subject_type, prompt_type):
    mask_en = []
    en_subjects = extract(data, "en", subject_type)

    for lang in langs:
        subjects = extract(data, lang, subject_type)
        prompts = extract(data, lang, prompt_type)
        subj_gloss = extract(data, lang, "subjects")

        mask_en.append(
            [
                1 if s in p or s_en in p or s_gl in p else 0
                for s, s_en, s_gl, p in zip(subjects, en_subjects, subj_gloss, prompts)
            ]
        )

    return np.array(mask_en).all(axis=0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="v5")
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the dataset"
    )
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--val_size", type=float, default=0.1)
    args = parser.parse_args()

    train_file = f"datasets/{args.version}/translated/train.json"
    test_file = f"datasets/{args.version}/translated/test.json"
    val_file = f"datasets/{args.version}/translated/val.json"

    with open(args.dataset_path, "r") as f:
        data = json.load(f)

    langs = sorted(list(data[list(data.keys())[0]]["subjects_mt_marked"].keys()))
    mask = get_sample_mask(data, langs, "subjects_mt_marked", "prompts_mt_marked")
    filtered_data = {
        syn_id: val for idx, (syn_id, val) in enumerate(data.items()) if mask[idx]
    }
    train_keys, test_keys, val_keys = split_data(
        filtered_data,
        random_state=args.random_state,
        test_size=args.test_size,
        val_size=args.val_size,
    )

    print(f"Train size: {len(train_keys)}")
    print(f"Val size: {len(val_keys)}")
    print(f"Test size: {len(test_keys)}")

    with open(train_file, "w") as f:
        f.write(json.dumps({k: data[k] for k in train_keys}, indent=4))
    with open(test_file, "w") as f:
        f.write(json.dumps({k: data[k] for k in test_keys}, indent=4))
    with open(val_file, "w") as f:
        f.write(json.dumps({k: data[k] for k in val_keys}, indent=4))


# # %%
# import json
# from utils import extract

# path = "datasets/v8/translated/dataset.json"
# with open(path, "r", encoding="utf-8") as file:
#     data = json.load(file)

# # %%
# import numpy as np

# langs = ["ar", "de", "en", "fr", "hr", "it", "ja", "ka", "my", "qu", "zh"]
# subj_prompt_pairs = [
#     ("subjects_mt", "prompts_mt"),
#     ("subjects_mt_marked", "prompts_mt_marked"),
#     ("subjects", "prompts_gloss"),
# ]
# type_to_array = {pair:[] for pair in subj_prompt_pairs}
# for subject_type, prompt_type in subj_prompt_pairs:
#     data_size = len(data)
#     scores = []
#     scores_en = []
#     en_subjects = extract(data, "en", subject_type)
#     mask_en = []
#     for lang in langs:
#         subjects = extract(data, lang, subject_type)
#         prompts = extract(data, lang, prompt_type)
#         subj_gloss = extract(data, lang, "subjects")
#         score = len([s for s, p in zip(subjects, prompts) if s in p]) / data_size * 100
#         score_en = (
#             len(
#                 [
#                     s
#                     for s, s_en, p in zip(subjects, en_subjects, prompts)
#                     if s in p or s_en in p
#                 ]
#             )
#             / data_size
#             * 100
#         )
#         mask_en.append(
#             [
#                 1 if s in p or s_en in p or s_gl in p else 0
#                 for s, s_en, s_gl, p in zip(subjects, en_subjects, subj_gloss, prompts)
#             ]
#         )
#         scores.append(score)
#         scores_en.append(score_en)
#         print(
#             f"Lang: {lang}, Covered pairs {score:.2f}, Covered pairs with English {score_en:.2f}"
#         )
#     mask_en = np.array(mask_en).all(axis=0)
#     type_to_array[(subject_type, prompt_type)] = mask_en
#     print("Number of points in the dataset ", mask_en.sum())

#     print()
#     print(subject_type, prompt_type)
#     print(f"Average: {np.mean(scores)}")
#     print(f"Average with English: {np.mean(scores_en)}", end="\n--------------------\n")

# print(
#     f"Conjunction across prompt types {np.stack(list(type_to_array.values()), axis=0).all(axis=0).sum().item()}"
# )
