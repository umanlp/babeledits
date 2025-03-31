# %%
import json
import os
import numpy as np

langs = ["ar", "de", "en", "fr", "hr", "it", "ja", "ka", "my", "qu", "zh"]
eval_langs = langs  # [l for l in langs if l != 'qu']
method = "FT-M"
model = "meta-llama_Meta-Llama-3.1-8B-Instruct"
prompt_type = "prompts_mt_marked"
version_folder = "v8_rev7"
# model = "google_gemma-2-9b-it"
print(f"Method: {method}")
print(f"Model: {model}")
print(f"Prompt type: {prompt_type}")
print(f"Version folder: {version_folder}")
print("Language : % of collapses")
all_collapses = []
t_clp = 1000

all_ppls = []
for lang in eval_langs:
    results_res_path = (
        f"logs/{version_folder}/{model}/{method}/{lang}/{prompt_type}/results.json"
    )

    if not os.path.exists(results_res_path):
        print(f"Path {results_res_path} does not exist.")
        continue
    with open(results_res_path, "r") as f:
        results = json.load(f)
    ppl_per_lang_post = {
        lang: [res["post"]["ppl"][lang] for res in results] for lang in eval_langs
    }
    perc_collapse = (
        np.concat(list(ppl_per_lang_post.values())).reshape(len(eval_langs), -1) > t_clp
    ).any(axis=0).mean() * 100
    all_ppls.append(
        np.concat(list(ppl_per_lang_post.values())).reshape(len(eval_langs), -1)
    )
    all_collapses.append(
        np.concat(list(ppl_per_lang_post.values())).reshape(len(eval_langs), -1)
    )
    print(
        f"{lang} : {perc_collapse:.2f}%",
        f"(Max English PPL: {max(ppl_per_lang_post['en'])})",
    )
# %%

import numpy as np
import pandas as pd

langs = ["ar", "de", "en", "fr", "hr", "it", "ja", "ka", "my", "qu", "zh"]
eval_langs = langs


def find_top_k_coordinates(array, eval_langs, k):
    num_train_langs, num_eval_langs, num_samples = array.shape
    top_k_coordinates = {}

    for eval_lang_idx, eval_lang in enumerate(eval_langs):
        slice_2d = array[:, eval_lang_idx, :]
        flat_indices = np.argpartition(slice_2d.flatten(), -k)[-k:]
        coords = np.array(np.unravel_index(flat_indices, slice_2d.shape)).T
        coords = coords[::-1, :]
        top_k_coordinates[eval_lang] = coords

    return top_k_coordinates


top_k = find_top_k_coordinates(np.stack(all_ppls, axis=0), eval_langs, 5)
top_k
# %%
all_ppls = np.stack(all_ppls, axis=0)


def group_by_train_lang(top_k_coordinates):
    grouped = {}
    for eval_lang, coords in top_k_coordinates.items():
        for train_lang, sample in coords:
            if train_lang not in grouped:
                grouped[train_lang] = []
            grouped[train_lang].append((eval_lang, sample.item()))
    return grouped


# Assuming top_k_coordinates is your original output
grouped_coords = group_by_train_lang(
    find_top_k_coordinates(np.stack(all_ppls, axis=0), eval_langs, 5)
)

# Print the grouped coordinates
lang_to_indexes = {lang: [] for lang in langs}
for train_lang, occurrences in grouped_coords.items():
    occ = []
    for o in occurrences:
        o2 = list(o)
        o2.append(all_ppls[train_lang, eval_langs.index(o[0]), o[1]].item())
        occ.append(o2)
    print(f"{eval_langs[train_lang]} ({train_lang}): {occ}")
    lang_to_indexes[eval_langs[train_lang]].append([x[1] for x in occ])

lang_to_indexes = {
    lang: lang_to_indexes[lang][0]
    for lang in lang_to_indexes
    if len(lang_to_indexes[lang]) > 0
}
lang_to_indexes
# %%


lang_to_indexes_filtered = {
    lang: sorted(list(set(lang_to_indexes[lang]))) for lang in lang_to_indexes
}

# %%
folder = "dst_res"
path_to_res = {}


def load_json_files_recursively(path):
    results = []
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            results.extend(load_json_files_recursively(full_path))
        elif entry.endswith(".json"):
            with open(full_path, "r") as f:
                results.append(json.load(f)["results"])
    return results


for dir_name in os.listdir(folder):
    dir_path = os.path.join(folder, dir_name)
    if os.path.isdir(dir_path):
        path_to_res[dir_name] = load_json_files_recursively(dir_path)
path_to_res


# %%
def extract_scores(d):
    res = {}
    for k in d:
        if k.startswith("belebele"):
            res[k] = d[k]["acc_norm,none"]
        if k.startswith("xquad"):
            res[k] = d[k]["exact_match,none"]
    return res


path_to_res_filtered = {}
for path, res_list in path_to_res.items():
    path_to_res_filtered[path] = []
    for res in res_list:
        path_to_res_filtered[path].append(extract_scores(res))
path_to_res_filtered
# %%
import pandas as pd
import numpy as np

# Create a list of tuples (path, dict) to handle multiple dictionaries per path
data = [(path, d) for path, res_list in path_to_res_filtered.items() for d in res_list]

# Create DataFrame with index being path names automatically repeated
df = pd.DataFrame([d for _, d in data], index=[path for path, _ in data])
print(df)
# %%
fdf = df.drop("vanilla_gemma")
# Get vanilla_llama row values
vanilla_values = df.loc["vanilla_llama"]

# Subtract vanilla values from each row
# fdf = fdf.subtract(vanilla_values)

# Drop the vanilla_llama row
fdf = fdf[fdf.index != 'vanilla_llama']
fdf = fdf.round(2)
fdf['langs'] = [x.split("_")[1] for x in fdf.index]
fdf['avg_xquad'] = fdf[[x for x in fdf.columns if x.startswith("xquad")]].mean(axis=1)
fdf['idx'] = list(range(len(fdf)))
# %%
selected_rows = [11, 16, 8, 21]
import matplotlib.pyplot as plt

# Select only the rows defined in selected_rows
subset = fdf.iloc[selected_rows]

# Exclude the 'lang' column if it exists
cols = [col for col in subset.columns if col.startswith("xquad")]

# Transpose the data so that columns become the x-axis
data = subset[cols].T

# Get vanilla values for the same columns
vanilla_data = vanilla_values[cols]

# Define x positions for each column
x = np.arange(len(cols))
bar_width = 0.8 / (len(selected_rows) + 1)  # +1 for vanilla values

fig, ax = plt.subplots(figsize=(20, 3))

# Plot vanilla values first and add text annotations
vanilla_bars = ax.bar(x, vanilla_data, width=bar_width, label="vanilla")
for i, v in enumerate(vanilla_data):
    ax.text(x[i], v, f"{v:.2f} (vanilla)", ha="center", va="bottom")

# Plot other rows
for i, row_label in enumerate(data.columns):
    bars = ax.bar(
        x + (i + 1) * bar_width, data[row_label], width=bar_width, label=row_label
    )
    # Add text annotations for each bar
    for j, v in enumerate(data[row_label]):
        ax.text(
            x[j] + (i + 1) * bar_width,
            v,
            f"{v:.2f} ({row_label.split('_')[1]})",
            ha="center",
            va="bottom",
        )

ax.set_xticks(x + bar_width * len(selected_rows) / 2)
ax.set_xticklabels(cols)
ax.set_ylabel("Score Differences")
ax.set_title("Bar Plot with Column Names as X-axis")
ax.legend(title="Row Index")

plt.tight_layout()
plt.show()

# %%
selected_rows = [11, 16, 8, 21]
import plotly.graph_objects as go

# Select only the rows defined in selected_rows
subset = fdf.iloc[selected_rows]

# Exclude the 'lang' column if it exists
cols = [col for col in subset.columns if col.startswith("xquad")]
xticks = ["xQuAD (AR)", "xQuAD (DE)", "xQuAD (EN)", "xQuAD (ZH)"]

# Transpose the data so that columns become the x-axis
data = subset[cols].T

# Get vanilla values for the same columns
vanilla_data = vanilla_values[cols]

# Define x positions for each column
x = np.arange(len(cols))
bar_width = 0.8 / (len(selected_rows) + 1)  # +1 for vanilla values

# Create figure
fig = go.Figure()

# Add vanilla bars with color
vanilla_color = "#1f77b4"  # Default plotly color
fig = go.Figure(layout=dict(
    plot_bgcolor='white',
    width=800,  # Increase width
    height=600,  # Increase height
    margin=dict(t=50, b=50, l=50, r=50)  # Add margins
))

fig.add_trace(
    go.Bar(
        x=xticks,
        y=vanilla_data,
        name="vanilla",
        width=bar_width,
        # marker_color=vanilla_color,
        text=[f"{v:.2f}" for v in vanilla_data],
        textposition="outside",
    )
)

# Colors for other bars
colors = ["#ff7f0e", "#2ca02c", "#d62728"]  # Default plotly colors

# Add other bars
for i, row_label in enumerate(data.columns):
    lang_code = row_label.split("_")[1]
    fig.add_trace(
        go.Bar(
            x=xticks,
            y=data[row_label],
            name=lang_code,
            width=bar_width,
            # marker_color=colors[i],
            text=[f"{v:.2f}" if v > 0.01 else 0.0 for v in data[row_label]],
            textposition="outside",
        )
    )

fig.show()
# fig.write_image("collapse_plot.png")
# %%

fig.write_image("collapse_plot.png")
# %%
