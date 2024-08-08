
import json
import argparse

import pandas as pd


def create_table(langs: list[str], method: str):
    line_names = ["accuracy", "X-portab. w/o gloss", "X-portab. w/ gloss"]
    getters = [
        lambda x, lang: x["rewrite_acc"],
        lambda x, lang: x["portability"][f"prompts-prompts_en-{lang}_acc"],
        lambda x, lang: x["portability"][f"prompts-prompts_gloss_en-{lang}_acc"]
    ]
    data = {
        "index": [*(("before", name) for name in line_names), *(("after", name) for name in line_names)],
        "columns": langs,
        "index_names": ["", ""],
        "column_names": ["langs"],
        "data": [[] for _ in range(len(line_names) * 2)]
    }
    for lang in langs:
        with open(f"logs/v5_hard_prompts_{method}_{lang}/summary.json", "r") as f:
            raw_result = json.load(f)
        
        for i, getter in enumerate(getters):
            data["data"][i].append(getter(raw_result["pre"], lang))
        for i, getter in enumerate(getters):
            data["data"][i + len(getters)].append(getter(raw_result["post"], lang))

    df = pd.DataFrame.from_dict(data, orient="tight")

    return df.style.format(formatter=lambda x: f"{100 * x:.1f}").to_latex(hrules=False, clines="skip-last;index")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", type=str, nargs="+", default=["FT", "ROME"])
    parser.add_argument("--langs", type=str, nargs="+", default=["ar", "de", "es", "fr", "hr", "it", "ja", "nl", "zh"])
    parser.add_argument("--output_dir", type=str, default="logs")
    args = parser.parse_args()


    for method in args.methods:
        table_text = create_table(args.langs, method)
        with open(f"logs/v5_hard_prompts_table_{method}.tex", "w") as f:
            f.write(table_text)
