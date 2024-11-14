# %%
import json
from utils import extract

path = "datasets/v8/translated/test.json"
with open(path, "r", encoding="utf-8") as file:
    data = json.load(file)

# %%
import numpy as np

langs = ["ar", "de", "en", "fr", "hr", "it", "ja", "ka", "my", "qu", "zh"]
subj_prompt_pairs = [
    ("subjects_mt", "prompts_mt"),
    ("subjects_mt_marked", "prompts_mt_marked"),
    ("subjects", "prompts_gloss"),
]
type_to_array = {pair:[] for pair in subj_prompt_pairs}
for subject_type, prompt_type in subj_prompt_pairs:
    data_size = len(data)
    scores = []
    scores_en = []
    en_subjects = extract(data, "en", subject_type)
    mask_en = []
    for lang in langs:
        subjects = extract(data, lang, subject_type)
        prompts = extract(data, lang, prompt_type)
        subj_gloss = extract(data, lang, "subjects")
        score = len([s for s, p in zip(subjects, prompts) if s in p]) / data_size * 100
        score_en = (
            len(
                [
                    s
                    for s, s_en, p in zip(subjects, en_subjects, prompts)
                    if s in p or s_en in p
                ]
            )
            / data_size
            * 100
        )
        mask_en.append(
            [
                1 if s in p or s_en in p or s_gl in p else 0
                for s, s_en, s_gl, p in zip(subjects, en_subjects, subj_gloss, prompts)
            ]
        )
        scores.append(score)
        scores_en.append(score_en)
        print(
            f"Lang: {lang}, Covered pairs {score:.2f}, Covered pairs with English {score_en:.2f}"
        )
    mask_en = np.array(mask_en).all(axis=0)
    type_to_array[(subject_type, prompt_type)] = mask_en
    print("Number of points in the dataset ", mask_en.sum())

    print()
    print(subject_type, prompt_type)
    print(f"Average: {np.mean(scores)}")
    print(f"Average with English: {np.mean(scores_en)}", end="\n--------------------\n")

print(
    f"Conjunction across prompt types {np.stack(list(type_to_array.values()), axis=0).all(axis=0).sum().item()}"
)

# %%

import numpy as np

m = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1]])
m, m.all(axis=0)