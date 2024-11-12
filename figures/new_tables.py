# %%
import sienna
import numpy as np
import pandas as pd


def get_langs_from_summary(summary_results):
    return sorted(list(summary_results["pre"]["ppl"].keys()))


def create_metrics_table(summary_results, langs):
    n_rows = 48  # Updated to match actual number of rows from getters

    # Level 0: before/after
    level0 = ["before"] * 21 + ["after"] * 27  # Updated after section count

    # Level 1: metric category
    level1 = (
        ["reliability"] * 6
        + ["generality"] * 6
        + ["multi-hop portability"] * 6
        + ["subj-alias portability"] * 2
        + ["PPL"] * 1  # Before section (21)
        + ["reliability"] * 6
        + ["generality"] * 6
        + ["locality"] * 6  # Updated to 6 for locality (2 metrics × 3 prompt types)
        + ["multi-hop portability"] * 6
        + ["subj-alias portability"] * 2
        + ["perplexity"] * 1  # After section (27)
    )

    # Level 2: prompt type
    level2 = (
        ["pure-mt"] * 2
        + ["mt-marked"] * 2
        + ["glossary"] * 2  # reliability
        + ["pure-mt"] * 2
        + ["mt-marked"] * 2
        + ["glossary"] * 2  # generality
        + ["pure-mt"] * 2
        + ["mt-marked"] * 2
        + ["glossary"] * 2  # multi-hop
        + ["-"] * 2  # subj-alias
        + ["-"] * 1  # PPL (before section)
        + ["pure-mt"] * 2
        + ["mt-marked"] * 2
        + ["glossary"] * 2  # reliability
        + ["pure-mt"] * 2
        + ["mt-marked"] * 2
        + ["glossary"] * 2  # generality
        + ["pure-mt"] * 2
        + ["mt-marked"] * 2
        + ["glossary"] * 2  # locality
        + ["pure-mt"] * 2
        + ["mt-marked"] * 2
        + ["glossary"] * 2  # multi-hop
        + ["-"] * 2  # subj-alias
        + ["-"] * 1  # PPL
    )

    # Level 3: metric type
    level3 = (
        ["token EM", "first token EM"] * 3  # reliability (6)
        + ["token EM", "first token EM"] * 3  # generality (6)
        + ["token EM", "first token EM"] * 3  # multi-hop (6)
        + ["token EM", "first token EM"]  # subj-alias (2)
        + ["PPL"]  # PPL (1) (before section)
        + ["token EM", "first token EM"] * 3  # reliability (6)
        + ["token EM", "first token EM"] * 3  # generality (6)
        + ["loc. token EM", "NKL"] * 3  # locality (6)
        + ["token EM", "first token EM"] * 3  # multi-hop (6)
        + ["token EM", "first token EM"]  # subj-alias (2)
        + ["PPL"]  # PPL (1)
    )

    # Verify lengths
    arrays = [level0, level1, level2, level3]
    for i, arr in enumerate(arrays):
        if len(arr) != n_rows:
            raise ValueError(f"Array {i} has length {len(arr)}, expected {n_rows}")

    # Create multi-index
    index = pd.MultiIndex.from_arrays(
        arrays, names=["phase", "metric", "prompt", "type"]
    )

    # Keep existing getters
    getters_before = [
        lambda x, lang: x["portability"][f"xlt-prompts_mt-{lang}"]["token_em"],
        lambda x, lang: x["portability"][f"xlt-prompts_mt-{lang}"]["first_token_em"],
        lambda x, lang: x["portability"][f"xlt-prompts_mt_marked-{lang}"]["token_em"],
        lambda x, lang: x["portability"][f"xlt-prompts_mt_marked-{lang}"][
            "first_token_em"
        ],
        lambda x, lang: x["portability"][f"xlt-prompts_gloss-{lang}"]["token_em"],
        lambda x, lang: x["portability"][f"xlt-prompts_gloss-{lang}"]["first_token_em"],
        lambda x, lang: x["rephrase_acc"][f"prompts_gen_mt-{lang}"]["token_em"],
        lambda x, lang: x["rephrase_acc"][f"prompts_gen_mt-{lang}"]["first_token_em"],
        lambda x, lang: x["rephrase_acc"][f"prompts_gen_mt_marked-{lang}"]["token_em"],
        lambda x, lang: x["rephrase_acc"][f"prompts_gen_mt_marked-{lang}"][
            "first_token_em"
        ],
        lambda x, lang: x["rephrase_acc"][f"prompts_gen_gloss-{lang}"]["token_em"],
        lambda x, lang: x["rephrase_acc"][f"prompts_gen_gloss-{lang}"][
            "first_token_em"
        ],
        lambda x, lang: x["portability"][f"multi-hop_prompts_port_mt-{lang}"][
            "token_em"
        ],
        lambda x, lang: x["portability"][f"multi-hop_prompts_port_mt-{lang}"][
            "first_token_em"
        ],
        lambda x, lang: x["portability"][f"multi-hop_prompts_port_mt_marked-{lang}"][
            "token_em"
        ],
        lambda x, lang: x["portability"][f"multi-hop_prompts_port_mt_marked-{lang}"][
            "first_token_em"
        ],
        lambda x, lang: x["portability"][f"multi-hop_prompts_port_gloss-{lang}"][
            "token_em"
        ],
        lambda x, lang: x["portability"][f"multi-hop_prompts_port_gloss-{lang}"][
            "first_token_em"
        ],
        lambda x, lang: x["portability"]
        .get(f"subj-alias_prompts_subj_alias-{lang}", {})
        .get("token_em", np.nan),
        lambda x, lang: x["portability"]
        .get(f"subj-alias_prompts_subj_alias-{lang}", {})
        .get("first_token_em", np.nan),
        lambda x, lang: x["ppl"][lang],
    ]

    getters_after = [
        lambda x, lang: x["portability"][f"xlt-prompts_mt-{lang}"]["token_em"],
        lambda x, lang: x["portability"][f"xlt-prompts_mt-{lang}"]["first_token_em"],
        lambda x, lang: x["portability"][f"xlt-prompts_mt_marked-{lang}"]["token_em"],
        lambda x, lang: x["portability"][f"xlt-prompts_mt_marked-{lang}"][
            "first_token_em"
        ],
        lambda x, lang: x["portability"][f"xlt-prompts_gloss-{lang}"]["token_em"],
        lambda x, lang: x["portability"][f"xlt-prompts_gloss-{lang}"]["first_token_em"],
        lambda x, lang: x["rephrase_acc"][f"prompts_gen_mt-{lang}"]["token_em"],
        lambda x, lang: x["rephrase_acc"][f"prompts_gen_mt-{lang}"]["first_token_em"],
        lambda x, lang: x["rephrase_acc"][f"prompts_gen_mt_marked-{lang}"]["token_em"],
        lambda x, lang: x["rephrase_acc"][f"prompts_gen_mt_marked-{lang}"][
            "first_token_em"
        ],
        lambda x, lang: x["rephrase_acc"][f"prompts_gen_gloss-{lang}"]["token_em"],
        lambda x, lang: x["rephrase_acc"][f"prompts_gen_gloss-{lang}"][
            "first_token_em"
        ],
        lambda x, lang: x["locality"][f"prompts_loc_mt-{lang}"]["token_em"],
        lambda x, lang: x["locality"][f"prompts_loc_mt-{lang}"]["nkl"],
        lambda x, lang: x["locality"][f"prompts_loc_mt_marked-{lang}"]["token_em"],
        lambda x, lang: x["locality"][f"prompts_loc_mt_marked-{lang}"]["nkl"],
        lambda x, lang: x["locality"][f"prompts_loc_gloss-{lang}"]["token_em"],
        lambda x, lang: x["locality"][f"prompts_loc_gloss-{lang}"]["nkl"],
        lambda x, lang: x["portability"][f"multi-hop_prompts_port_mt-{lang}"][
            "token_em"
        ],
        lambda x, lang: x["portability"][f"multi-hop_prompts_port_mt-{lang}"][
            "first_token_em"
        ],
        lambda x, lang: x["portability"][f"multi-hop_prompts_port_mt_marked-{lang}"][
            "token_em"
        ],
        lambda x, lang: x["portability"][f"multi-hop_prompts_port_mt_marked-{lang}"][
            "first_token_em"
        ],
        lambda x, lang: x["portability"][f"multi-hop_prompts_port_gloss-{lang}"][
            "token_em"
        ],
        lambda x, lang: x["portability"][f"multi-hop_prompts_port_gloss-{lang}"][
            "first_token_em"
        ],
        lambda x, lang: x["portability"]
        .get(f"subj-alias_prompts_subj_alias-{lang}", {})
        .get("token_em", np.nan),
        lambda x, lang: x["portability"]
        .get(f"subj-alias_prompts_subj_alias-{lang}", {})
        .get("first_token_em", np.nan),
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
    df = pd.DataFrame(
        np.array(data).T, index=index, columns=langs
    )
    
    # Calculate averages and insert as first column
    avg_series = df.mean(axis=1)
    df.insert(0, 'avg', avg_series)

    idx = pd.IndexSlice

    df_style = df.style
    # Apply formatting for PPL rows
    df_style = df_style.format(
        formatter=lambda x: "-" if pd.isna(x) else f"{x:.2f}",
        subset=pd.IndexSlice[:, :, :, 'PPL']
    )

    # Apply formatting for nkl rows
    df_style = df_style.format(
        formatter=lambda x: "-" if pd.isna(x) else f"{x:.2f}",
        subset=idx[:, :, :, 'NKL']
    )

    # Apply formatting for all other rows
    df_style = df_style.format(
        formatter=lambda x: "-" if pd.isna(x) else f"{x * 100:.2f}",
        subset=idx[:, (df.index.get_level_values('type') != 'PPL') & (df.index.get_level_values('type') != 'NKL'), :, :]
    )
    return df_style

# %%
summary_res_path = (
    "../logs/v8/meta-llama_Meta-Llama-3.1-8B-Instruct/FT/en/prompts_mt/summary_new.json"
)
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
with open("table_output.tex", "w", encoding="utf-8") as f:
    f.write(latex_output)

print("Table written to table_output.tex")



# %%
