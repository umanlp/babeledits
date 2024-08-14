
import json
import argparse
import itertools

import pandas as pd

def format_row_wise(styler, formatter):
    for row, row_formatter in formatter.items():
        row_num = styler.index.get_loc(row)

        for col_num in range(len(styler.columns)):
            styler._display_funcs[(row_num, col_num)] = row_formatter
    return styler

def create_table(langs: list[str], method: str):
    line_names = ["accuracy", "X-portab. w/o gloss", "X-portab. w/ gloss", "w/ - w/o"]
    getters = [
        lambda x, lang: x["rewrite_acc"],
        lambda x, lang: x["portability"][f"prompts-prompts_en-{lang}_acc"],
        lambda x, lang: x["portability"][f"prompts-prompts_gloss_en-{lang}_acc"],
        lambda x, lang: x["portability"][f"prompts-prompts_gloss_en-{lang}_acc"] - x["portability"][f"prompts-prompts_en-{lang}_acc"]
    ]
    data = {
        "index": [("samples", ""), *(("before", name) for name in line_names), *(("after", name) for name in line_names)],
        "columns": langs,
        "index_names": ["", ""],
        "column_names": ["langs"],
        "data": [[] for _ in range(len(line_names) * 2 + 1)]
    }
    for lang in langs:
        with open(f"logs/v5_hard_prompts_{method}_{lang}/summary.json", "r") as f:
            raw_result = json.load(f)
        with open(f"logs/v5_hard_prompts_{method}_{lang}/results.json", "r") as f:
            n_samples = len(json.load(f))
        
        data["data"][0].append(n_samples)
        for i, getter in enumerate(getters):
            data["data"][i + 1].append(getter(raw_result["pre"], lang))
        for i, getter in enumerate(getters):
            data["data"][i + 1 + len(getters)].append(getter(raw_result["post"], lang))

    df = pd.DataFrame.from_dict(data, orient="tight")

    class LogGetitemDict(dict):

        def __getitem__(self, value):
            print(f"retrieving key: {value}")
            return super(LogGetitemDict, self).__getitem__(value)
        
        def __contains__(self, key: object) -> bool:
            print(f"checking key: {key}")
            return super().__contains__(key)

    formatter = {
        index: (lambda x: f"{int(x)}") if index == ("samples", "") else (lambda x: f"{100 * x:.1f}")
        for index in data["index"]
    }

    print(df.index)

    return format_row_wise(df.style, formatter).to_latex(hrules=False, clines="skip-last;index")


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
