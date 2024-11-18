# %%
import sienna
import numpy as np
import pandas as pd


def get_langs_from_summary(summary_results):
    return sorted(list(summary_results["pre"]["ppl"].keys()))


def create_metrics_table(summary_results, langs):
    n_rows = 18  # Updated to match actual number of rows from getters

    # Level 0: before/after
    level0 = ["before"] * 8 + ["after"] * 10  # Updated after section count

    print(f"Level 0: {len(level0)}")

    # Level 1: metric category
    level1 = (
        ["reliability"] * 2
        + ["generality"] * 2
        + ["multi-hop portability"] * 2
        + ["subj-alias portability"] * 1
        + ["PPL"] * 1  # Before section (8)
        + ["reliability"] * 2
        + ["generality"] * 2
        + ["locality"] * 2
        + ["multi-hop portability"] * 2
        + ["subj-alias portability"] * 1
        + ["PPL"] * 1  # After section (10)
    )
    print(f"Level 1: {len(level1)}")

    # Level 2: prompt type
    level2 = (
        ["mt"] * 1
        + ["mt-marked"] * 1  # reliability
        + ["mt"] * 1
        + ["mt-marked"] * 1  # rephrase_acc
        + ["mt"] * 1
        + ["mt-marked"] * 1  # multi-hop portability
        + ["-"] * 1  # subj-alias portability
        + ["-"] * 1  # PPL (before section)
        + ["mt"] * 1
        + ["mt-marked"] * 1  # portability
        + ["mt"] * 1
        + ["mt-marked"] * 1  # rephrase_acc
        + ["mt"] * 1
        + ["mt-marked"] * 1  # locality
        + ["mt"] * 1
        + ["mt-marked"] * 1  # multi-hop portability
        + ["-"] * 1  # subj-alias portability
        + ["-"] * 1  # PPL
    )

    print(f"Level 2: {len(level2)}")

    # Verify lengths
    arrays = [level0, level1, level2]
    for i, arr in enumerate(arrays):
        if len(arr) != n_rows:
            raise ValueError(f"Array {i} has length {len(arr)}, expected {n_rows}")

    # Create multi-index
    index = pd.MultiIndex.from_arrays(arrays, names=["phase", "metric", "prompt"])

    # Keep existing getters
    getters_before = [
        lambda x, lang: x["portability"][f"xlt-prompts_mt-{lang}"]["token_em_lm_eval"],
        lambda x, lang: x["portability"][f"xlt-prompts_mt_marked-{lang}"][
            "token_em_lm_eval"
        ],
        lambda x, lang: x["rephrase_acc"][f"prompts_gen_mt-{lang}"]["token_em_lm_eval"],
        lambda x, lang: x["rephrase_acc"][f"prompts_gen_mt_marked-{lang}"][
            "token_em_lm_eval"
        ],
        lambda x, lang: x["portability"][f"multi-hop_prompts_port_mt-{lang}"][
            "token_em_lm_eval"
        ],
        lambda x, lang: x["portability"][f"multi-hop_prompts_port_mt_marked-{lang}"][
            "token_em_lm_eval"
        ],
        lambda x, lang: x["portability"]
        .get(f"subj-alias_prompts_subj_alias-{lang}", {})
        .get("token_em_lm_eval", np.nan),
        lambda x, lang: x["ppl"][lang],
    ]

    getters_after = [
        lambda x, lang: x["portability"][f"xlt-prompts_mt-{lang}"]["token_em_lm_eval"],
        lambda x, lang: x["portability"][f"xlt-prompts_mt_marked-{lang}"][
            "token_em_lm_eval"
        ],
        lambda x, lang: x["rephrase_acc"][f"prompts_gen_mt-{lang}"]["token_em_lm_eval"],
        lambda x, lang: x["rephrase_acc"][f"prompts_gen_mt_marked-{lang}"][
            "token_em_lm_eval"
        ],
        lambda x, lang: x["locality"][f"prompts_loc_mt-{lang}"]["nkl"],
        lambda x, lang: x["locality"][f"prompts_loc_mt_marked-{lang}"]["nkl"],
        lambda x, lang: x["portability"][f"multi-hop_prompts_port_mt-{lang}"][
            "token_em_lm_eval"
        ],
        lambda x, lang: x["portability"][f"multi-hop_prompts_port_mt_marked-{lang}"][
            "token_em_lm_eval"
        ],
        lambda x, lang: x["portability"]
        .get(f"subj-alias_prompts_subj_alias-{lang}", {})
        .get("token_em_lm_eval", np.nan),
        lambda x, lang: x["ppl"][lang],
    ]

    # Create DataFrame with multi-index
    data = []
    for lang in langs:
        col_data = []
        for getter in getters_before:
            try:
                col_data.append(getter(summary_results["pre"], lang))
            except KeyError:
                col_data.append(np.nan)

        for getter in getters_after:
            try:
                col_data.append(getter(summary_results["post"], lang))
            except KeyError:
                col_data.append(np.nan)
        data.append(col_data)

    # Create initial DataFrame
    df = pd.DataFrame(np.array(data).T, index=index, columns=langs)

    # Calculate averages and insert as first column
    avg_series = df.mean(axis=1)
    df.insert(0, "avg", avg_series)

    df_style = df.style
    # Apply formatting for PPL rows
    df_style = df_style.format(
        formatter=lambda x: "-" if pd.isna(x) else f"{x:.2f}",
        subset=pd.IndexSlice[:, "PPL", :],
    )

    # Apply formatting for nkl rows
    df_style = df_style.format(
        formatter=lambda x: "-" if pd.isna(x) else f"{x:.2f}",
        subset=pd.IndexSlice[:, "locality", :],
    )

    # Apply formatting for all other rows
    df_style = df_style.format(
        formatter=lambda x: "-" if pd.isna(x) else f"{x * 100:.2f}",
        subset=pd.IndexSlice[
            :,
            (df.index.get_level_values("metric") != "PPL")
            & (df.index.get_level_values("metric") != "locality"),
            :,
        ],
    )
    return df_style


# %%
summary_res_path = "../logs/v8_rev3/meta-llama_Meta-Llama-3.1-8B-Instruct/FT-M/en/prompts_mt_marked/summary.json"
summary_results = sienna.load(summary_res_path)
langs = get_langs_from_summary(summary_results)

styled_df = create_metrics_table(summary_results, langs)

# First get just the tabular environment
latex_table = styled_df.to_latex(
    position="!h",
    position_float="centering",
    hrules=True,
    column_format="l" * 4 + "c" * (len(langs) + 1),  # +1 for avg column
    clines="skip-last;data",
)

# Wrap with proper encoding and size adjustments
# latex_output = f"""\\begin{{table}}[!h]
# \\centering
# \\footnotesize
# \\adjustbox{{max width=\\textwidth}}{{
# {latex_table}
# }}"""
latex_output = latex_table
# Write to file instead of printing directly
with open("table_output_v8_rev3.tex", "w", encoding="utf-8") as f:
    f.write(latex_output)

print("Table written to table_output_v8_rev3.tex")


# %%
