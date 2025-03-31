# %%
import os
import json
import sienna
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
main_dir_template = "../logs/v8_rev7_seq_{}"
method_list = ["FT-L", "FT-M", "R-ROME", "GRACE", "BabelReFT"]
model = "meta-llama_Meta-Llama-3.1-8B-Instruct"
edit_lang = "en"
prompt_type = "prompts_mt_marked"
eval_metric = "rewrite_score"
sample_sizes = [100, 250, 500, 1042]


def get_langs_from_summary(summary_results):
    return sorted(list(summary_results["pre"]["ppl"].keys()))


def get_benchmark_scores(res_path, model):
    # Find the results file in the directory
    # Find the first directory in the results path and use it for baseline metrics
    dirs = [d for d in os.listdir(res_path) if os.path.isdir(os.path.join(res_path, d))]
    if not dirs:
        return {"belebele": np.nan, "xquad": np.nan}
    res_dir = dirs[0]
    json_files = [f for f in os.listdir(os.path.join(res_path, res_dir)) if f.endswith('.json') and f.startswith('result')] 
    def extract_timestamp(filename):
        # Remove the prefix 'results_' and the suffix '.json'
        timestamp_str = filename[len("results_"):-len(".json")]
        return datetime.strptime(timestamp_str, "%Y-%m-%dT%H-%M-%S.%f")

    latest_result = max(json_files, key=extract_timestamp)
    print("Looking for downstream results in:", os.path.join(res_path, res_dir))
    print("Available results:", ",".join(json_files))
    print("Using latest results:", latest_result, end="\n\n")
    with open(os.path.join(res_path, res_dir, latest_result)) as f:
        results = json.load(f)

        # Extract Belebele scores
        belebele_scores = []
        for key in results["results"]:
            if key.startswith("belebele_"):
                belebele_scores.append(results["results"][key]["acc,none"])

        # Extract XQuAD scores
        xquad_scores = []
        for key in results["results"]:
            if key.startswith("xquad_"):
                xquad_scores.append(results["results"][key]["exact_match,none"])

        return {
            "belebele": np.mean(belebele_scores),
            "xquad": np.mean(xquad_scores),
                }
    return {"belebele": np.nan, "xquad": np.nan}


def create_metrics_table(sample_sizes):
    metrics = [
        "reliability",
        "generality",
        "locality",
        "multi-hop portability",
        "subj-alias portability",
        "delta PPL",
        "Belebele",
        "XQuAD",
    ]

    # Create a MultiIndex with metrics and methods
    index = pd.MultiIndex.from_product(
        [metrics, method_list], names=["metric", "method"]
    )
    columns = pd.Index([str(size) for size in sample_sizes], name="sample_size")
    df = pd.DataFrame(index=index, columns=columns)

    # Populate DataFrame
    for size in sample_sizes:
        size_dir = main_dir_template.format(size)
        summary_res_path = (
            f"{size_dir}/{model}/{{}}/{edit_lang}/{prompt_type}/summary.json"
        )
        res_path = f"{size_dir}/{model}/{{}}/{edit_lang}/{prompt_type}"

        method_to_summary = {}
        for method in method_list:
            try:
                method_to_summary[method] = sienna.load(summary_res_path.format(method))
            except:
                continue

        # Get languages for this size
        if method_to_summary:
            langs = get_langs_from_summary(next(iter(method_to_summary.values())))

            # Get benchmark scores
            benchmark_scores = {
                method: get_benchmark_scores(res_path.format(method), model)
                for method in method_list
            }

            # Calculate metrics for each method
            for method in method_list:
                if method in method_to_summary:
                    summaries = method_to_summary[method]

                    # Calculate language-averaged metrics
                    reliability = np.mean(
                        [
                            summaries["post"]["portability"][
                                f"xlt-prompts_mt_marked-{lang}"
                            ][eval_metric]
                            for lang in langs
                        ]
                    )
                    generality = np.mean(
                        [
                            summaries["post"]["rephrase_acc"][
                                f"prompts_gen_mt_marked-{lang}"
                            ][eval_metric]
                            for lang in langs
                        ]
                    )
                    locality = np.mean(
                        [
                            summaries["post"]["locality"][
                                f"prompts_loc_mt_marked-{lang}"
                            ]["nkl"]
                            for lang in langs
                        ]
                    )
                    multi_hop = np.mean(
                        [
                            summaries["post"]["portability"][
                                f"multi-hop_prompts_port_mt_marked-{lang}"
                            ][eval_metric]
                            for lang in langs
                        ]
                    )

                    # Handle potential missing subj-alias data
                    subj_alias_scores = []
                    for lang in langs:
                        try:
                            score = summaries["post"]["portability"][
                                f"subj-alias_prompts_subj_alias-{lang}"
                            ][eval_metric]
                            subj_alias_scores.append(score)
                        except KeyError:
                            print("Missing subj-alias data for", lang, "in", method)
                            continue
                    subj_alias = (
                        np.mean(subj_alias_scores) if subj_alias_scores else np.nan
                    )

                    # Handle delta PPL calculation with special case for BabelReFT
                    if method == "BabelReFT":
                        delta_ppl = np.nan
                    else:
                        delta_ppl = np.mean(
                            [
                                summaries["post"]["ppl"][lang]
                                - summaries["pre"]["ppl"][lang]
                                for lang in langs
                            ]
                        )

                    # Store values in DataFrame
                    df.loc[("reliability", method), str(size)] = reliability
                    df.loc[("generality", method), str(size)] = generality
                    df.loc[("locality", method), str(size)] = locality
                    df.loc[("multi-hop portability", method), str(size)] = multi_hop
                    df.loc[("subj-alias portability", method), str(size)] = subj_alias
                    df.loc[("delta PPL", method), str(size)] = delta_ppl
                    df.loc[("Belebele", method), str(size)] = benchmark_scores[method][
                        "belebele"
                    ]
                    df.loc[("XQuAD", method), str(size)] = benchmark_scores[method][
                        "xquad"
                    ]

    return df

if __name__ == "__main__":

    # Create the table
    df = create_metrics_table(sample_sizes)

    # Style the DataFrame
    df_style = df.style

    # Format different metrics appropriately
    df_style = df_style.format(
        formatter=lambda x: "-" if pd.isna(x) else f"{x:.2f}",
        subset=pd.IndexSlice[("delta PPL", slice(None)), :],
    )
    df_style = df_style.format(
        formatter=lambda x: "-" if pd.isna(x) else f"{x:.2f}",
        subset=pd.IndexSlice[("locality", slice(None)), :],
    )

    metrics_to_format = [
        "reliability",
        "generality",
        "multi-hop portability",
        "subj-alias portability",
        "Belebele",
        "XQuAD",
    ]
    df_style = df_style.format(
        formatter=lambda x: "-" if pd.isna(x) else f"{x * 100:.2f}",
        subset=pd.IndexSlice[(metrics_to_format, slice(None)), :],
    )

    # Generate LaTeX output
    latex_table = df_style.to_latex(
        position="!h",
        position_float="centering",
        hrules=True,
        column_format="llc|" + "c" * len(sample_sizes),
        clines="skip-last;data",
    )

    if model == "google_gemma-2-9b-it":
        filename = "seq_table_output_gemma.tex"
    if model == "meta-llama_Meta-Llama-3.1-8B-Instruct":
        filename = "seq_table_output_llama.tex"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(latex_table)

    print(f"Table written to {filename}")

# %%
