# %%
import sienna
import numpy as np
import pandas as pd


def get_langs_from_summary(summary_results):
    return sorted(list(summary_results["pre"]["ppl"].keys()))

method_list = ["FT-L", "FT-M", "R-ROME", "GRACE", "BabelReFT"]
model = "meta-llama_Meta-Llama-3.1-8B-Instruct" 
# model = "google_gemma-2-9b-it" 
logdir="logs/v8_rev7"
edit_lang="en"
prompt_type="prompts_mt_marked"
eval_metric="rewrite_score"
summary_res_path = f"../{logdir}/{model}/{{}}/{edit_lang}/{prompt_type}/summary.json"

method_to_summary = {method:sienna.load(summary_res_path.format(method)) for method in method_list if method != "BabelReFT"}
method_to_summary["BabelReFT"] = sienna.load(
    f"../logs/v8_rev7_subloreft_lm/meta-llama_Meta-Llama-3.1-8B-Instruct/BabelReFT/{edit_lang}/prompts_mt_marked/summary.json"
)
print(method_to_summary.keys())

def create_metrics_table(method_to_summary, langs):
    metrics = [
        "reliability",
        "generality",
        "locality",
        "multi-hop portability",
        "subj-alias portability",
        "delta PPL",
    ]

    # Getters for the metrics
    getters = {
        "reliability": lambda x_post, x_pre, lang: x_post["portability"][
            f"xlt-prompts_mt_marked-{lang}"
        ][f"{eval_metric}"],
        "generality": lambda x_post, x_pre, lang: x_post["rephrase_acc"][
            f"prompts_gen_mt_marked-{lang}"
        ][f"{eval_metric}"],
        "locality": lambda x_post, x_pre, lang: x_post["locality"][
            f"prompts_loc_mt_marked-{lang}"
        ]["nkl"],
        "multi-hop portability": lambda x_post, x_pre, lang: x_post["portability"][
            f"multi-hop_prompts_port_mt_marked-{lang}"
        ][f"{eval_metric}"],
        "subj-alias portability": lambda x_post, x_pre, lang: x_post["portability"]
        .get(f"subj-alias_prompts_subj_alias-{lang}", {})
        .get(f"{eval_metric}", np.nan),
        "delta PPL": lambda x_post, x_pre, lang: x_post["ppl"][lang] - x_pre["ppl"][lang],
    }

    # Create a MultiIndex with metrics and methods
    index = pd.MultiIndex.from_product(
        [metrics, method_list],
        names=['metric', 'method']
    )
    columns = pd.Index(langs + ['avg'], name='language')
    df = pd.DataFrame(index=index, columns=columns)

    # Populate DataFrame
    for metric in metrics:
        for method in method_list:
            summaries = method_to_summary[method]
            for lang in langs:
                try:
                    value = getters[metric](summaries["post"], summaries["pre"], lang)
                except KeyError:
                    value = np.nan
                df.loc[(metric, method), lang] = value
            # Calculate average across languages for each method
            df.loc[(metric, method), 'avg'] = df.loc[(metric, method), langs].mean()
    
    # Reorder columns to have 'avg' as the first column
    df = df[['avg'] + langs]

    # Formatting
    df_style = df.style

    # Apply formatting for delta PPL row
    df_style = df_style.format(
        formatter=lambda x: "-" if pd.isna(x) else f"{x:.2f}",
        subset=pd.IndexSlice[("delta PPL", slice(None)), :]
    )

    # Apply formatting for locality row
    df_style = df_style.format(
        formatter=lambda x: "-" if pd.isna(x) else f"{x:.2f}",
        subset=pd.IndexSlice[("locality", slice(None)), :]
    )

    # Apply formatting for other rows
    metrics_to_format = [m for m in metrics if m not in ["delta PPL", "locality"]]
    df_style = df_style.format(
        formatter=lambda x: "-" if pd.isna(x) else f"{x * 100:.2f}",
        subset=pd.IndexSlice[(metrics_to_format, slice(None)), :]
    )

    return df, df_style

# Get languages
langs = get_langs_from_summary(next(iter(method_to_summary.values())))

# Create the table
df, styled_df = create_metrics_table(method_to_summary, langs)

styled_df
# %%
# Generate LaTeX output with adjustments for MultiIndex
latex_table = styled_df.to_latex(
    position="!h",
    position_float="centering",
    hrules=True,
    column_format="llc|" + "c" * (len(langs)),  # 'll' for 'metric' and 'method' columns
    clines="skip-last;data",
)

if model == "google_gemma-2-9b-it":
    filename = "multi_method_table_output_gemma.tex"
if model == "meta-llama_Meta-Llama-3.1-8B-Instruct":
    filename = "multi_method_table_output_llama.tex"

# Write the table to a file
with open(filename, "w", encoding="utf-8") as f:
    f.write(latex_table)

print(f"Table written to {filename}")
styled_df
# %%


