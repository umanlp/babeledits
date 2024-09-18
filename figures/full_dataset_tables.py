# %%
import sienna
import numpy as np
import pandas as pd

# data = list(sienna.load('datasets/v5/translated/test.json').values())


def get_langs_from_summary(summary_results, src_lang, prompt_type):
    port_res = summary_results["post"]["portability"]
    return sorted(
        list(
            set(
                [
                    metric.split("_")[-2].split("-")[-1]
                    for metric in port_res
                    if metric.startswith(f"{prompt_type}-prompts_mt_{src_lang}")
                ]
            )
        )
    )


def get_port_scores(results, lang):
    mt_res = []
    gloss_res = []
    marked_mt_res = []
    for elem in results:
        mt_res.append(
            elem["post"]["portability"][f"prompts-prompts_mt_{edit_lang}-{lang}_acc"]
        )
        marked_mt_res.append(
            elem["post"]["portability"][
                f"prompts-prompts_mt_marked{edit_lang}-{lang}_acc"
            ]
        )

        gloss_res.append(
            elem["post"]["portability"][f"prompts-prompts_gloss_{edit_lang}-{lang}_acc"]
        )
    return np.array(mt_res), np.array(marked_mt_res), np.array(gloss_res)


edit_lang = "en"
prompt_type = "prompts_gloss"
version_folder = "v6_4"
method = "FT"
model = "CohereForAI_aya-23-8B"
# model = "meta-llama_Meta-Llama-3.1-8B-Instruct"
summary_res_path = (
    f"../logs/{version_folder}/{model}/{method}/{edit_lang}/{prompt_type}/summary.json"
)
summary_results = sienna.load(summary_res_path)
langs = get_langs_from_summary(summary_results, edit_lang, prompt_type)


line_names_before = [
    "reliability",
    "generality pure-mt",
    "generality mt-marked",
    "generality w/ gloss",
    "X-portab. pure-mt",
    "X-portab. mt-marked",
    "X-portab. w/ gloss",
    "w/ gloss - mt-marked",
]
line_names_after = [
    "reliability",
    "generality pure-mt",
    "generality mt-marked",
    "generality w/ gloss",
    "locality pure-mt",
    "locality mt-marked",
    "locality w/ gloss",
    "X-portab. pure-mt",
    "X-portab. mt-marked",
    "X-portab. w/ gloss",
    "w/ gloss - mt-marked",
]
getters_before = [
    lambda x, lang: x["rewrite_acc"],
    lambda x, lang: x["rephrase_acc"][f"prompts_gen_mt_{edit_lang}-{lang}_acc"],
    lambda x, lang: x["rephrase_acc"][f"prompts_gen_mt_marked_{edit_lang}-{lang}_acc"],
    lambda x, lang: x["rephrase_acc"][f"prompts_gen_gloss_{edit_lang}-{lang}_acc"],
    lambda x, lang: x["portability"][
        f"{prompt_type}-prompts_mt_{edit_lang}-{lang}_acc"
    ],
    lambda x, lang: x["portability"][
        f"{prompt_type}-prompts_mt_marked_{edit_lang}-{lang}_acc"
    ],
    lambda x, lang: x["portability"][
        f"{prompt_type}-prompts_gloss_{edit_lang}-{lang}_acc"
    ],
    lambda x, lang: x["portability"][
        f"{prompt_type}-prompts_gloss_{edit_lang}-{lang}_acc"
    ]
    - x["portability"][f"{prompt_type}-prompts_mt_marked_{edit_lang}-{lang}_acc"],
]
getters_after = [
    lambda x, lang: x["rewrite_acc"],
    lambda x, lang: x["rephrase_acc"][f"prompts_gen_mt_{edit_lang}-{lang}_acc"],
    lambda x, lang: x["rephrase_acc"][f"prompts_gen_mt_marked_{edit_lang}-{lang}_acc"],
    lambda x, lang: x["rephrase_acc"][f"prompts_gen_gloss_{edit_lang}-{lang}_acc"],
    lambda x, lang: x["locality"][f"prompts_loc_mt_{edit_lang}-{lang}_acc"],
    lambda x, lang: x["locality"][f"prompts_loc_mt_marked_{edit_lang}-{lang}_acc"],
    lambda x, lang: x["locality"][f"prompts_loc_gloss_{edit_lang}-{lang}_acc"],
    lambda x, lang: x["portability"][
        f"{prompt_type}-prompts_mt_{edit_lang}-{lang}_acc"
    ],
    lambda x, lang: x["portability"][
        f"{prompt_type}-prompts_mt_marked_{edit_lang}-{lang}_acc"
    ],
    lambda x, lang: x["portability"][
        f"{prompt_type}-prompts_gloss_{edit_lang}-{lang}_acc"
    ],
    lambda x, lang: x["portability"][
        f"{prompt_type}-prompts_gloss_{edit_lang}-{lang}_acc"
    ] - x["portability"][f"{prompt_type}-prompts_mt_marked_{edit_lang}-{lang}_acc"],
]

data = {
    "index": [
        *(("before", name) for name in line_names_before),
        *(("after", name) for name in line_names_after),
    ],
    "columns": langs,
    "index_names": ["", ""],
    "column_names": ["langs"],
    "data": [[] for _ in range(len(line_names_before) + len(line_names_after))],
}

print(len(data["data"]))
for lang in langs:
    for i, getter in enumerate(getters_before):
        data["data"][i].append(getter(summary_results["pre"], lang))
    for i, getter in enumerate(getters_after):
        data["data"][i + len(getters_before)].append(
            getter(summary_results["post"], lang)
        )

df = pd.DataFrame.from_dict(data, orient="tight")
formatter = {index: lambda x: f"{100 * x:.1f}" for index in data["index"]}


def format_row_wise(styler, formatter):
    for row, row_formatter in formatter.items():
        row_num = styler.index.get_loc(row)

        for col_num in range(len(styler.columns)):
            styler._display_funcs[(row_num, col_num)] = row_formatter
    return styler


print(summary_res_path)
print(
    format_row_wise(df.style, formatter).to_latex(
        position="!h", position_float="centering", hrules=True, clines="skip-last;data"
    )
)


# %%

res_path = "../logs/v6/CohereForAI_aya-23-8B/FT/en/prompts/results.json"
results = sienna.load(res_path)


def get_metric(all_metrics, eval):
    metrics = dict()
    metrics[eval] = dict()
    for key in ["rewrite_acc"]:
        if key in all_metrics[0][eval].keys():
            metrics[eval][key] = [metric[eval][key] for metric in all_metrics]

    for key in ["rephrase_acc", "locality", "portability"]:
        if key in all_metrics[0][eval].keys() and all_metrics[0][eval][key] != {}:
            metrics[eval][key] = dict()
            for lkey in all_metrics[0][eval][key].keys():
                if lkey.endswith("acc"):
                    metrics[eval][key][lkey] = [
                        metric[eval][key][lkey] for metric in all_metrics
                    ]
    return metrics["post"]


m = get_metric(results, "post")

gloss_it_scores = np.array(
    m["portability"]["prompts-prompts_gloss_{edit_lang}-it_acc"]
).squeeze()
it_scores = np.array(m["portability"]["prompts-prompts_{edit_lang}-it_acc"]).squeeze()
diff = gloss_it_scores - it_scores
diff_idx = np.argsort(diff)

for idx in diff_idx[:100]:
    print(results[idx]["requested_rewrite"]["target_new"])
    print((gloss_it_scores - it_scores)[idx])
    print(
        results[idx]["requested_rewrite"]["portability"][
            "prompts-prompts_{edit_lang}-it"
        ]
    )
    print(
        results[idx]["requested_rewrite"]["portability"][
            "prompts-prompts_gloss_{edit_lang}-it"
        ],
        end="\n\n",
    )

# %%

m = get_metric(results, "post")

gloss_it_scores = np.array(
    m["rephrase_acc"]["prompts_gen_{edit_lang}-it_acc"]
).squeeze()
it_scores = np.array(
    m["rephrase_acc"]["prompts_gen_gloss_{edit_lang}-it_acc"]
).squeeze()
print(gloss_it_scores.mean(), it_scores.mean())
diff = gloss_it_scores - it_scores
diff_idx = np.argsort(diff)[::-1]

for idx in diff_idx[:100]:
    print(results[idx]["requested_rewrite"]["target_new"])
    print((gloss_it_scores - it_scores)[idx])
    print(
        results[idx]["requested_rewrite"]["rephrase_prompt"][
            "prompts_gen_{edit_lang}-it"
        ]
    )
    print(
        results[idx]["requested_rewrite"]["rephrase_prompt"][
            "prompts_gen_gloss_{edit_lang}-it"
        ],
        end="\n\n",
    )
