# %%
from pathlib import Path
import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt


def uniform_sample_from_buckets(buckets, K, langs):
    N = len(buckets)
    base_sample_size = K // N
    extra_samples = K % N

    samples = []
    global_sampled_elements = set()

    for i in range(N):
        if i < extra_samples:
            num_samples_from_bucket = base_sample_size + 1
        else:
            num_samples_from_bucket = base_sample_size

        bucket = buckets[i]
        lang = langs[i]
        sampled_elements = sample_from_bucket(
            bucket, num_samples_from_bucket, global_sampled_elements, lang
        )
        samples.extend(sampled_elements)
        global_sampled_elements.update([elem[0] for elem in sampled_elements])

    return samples


def sample_from_bucket(bucket, num_samples, global_sampled_elements, lang):
    unique_elements = bucket[~bucket.isin(global_sampled_elements).any(axis=1)]
    num_samples = min(num_samples, len(unique_elements))
    sampled_elements = unique_elements.sample(n=num_samples)
    sampled_elements = [
        (row["English Title"], lang) for _, row in sampled_elements.iterrows()
    ]
    return sampled_elements


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wiki_path",
        default="wikipedia_data/v5/processed",
        help="Path to the wiki file",
    )
    parser.add_argument(
        "--save_path",
        default="wikipedia_data/v5",
        help="Path to save the merged dataset",
    )
    parser.add_argument(
        "--num_samples", default=13000, type=int, help="Number of samples to draw"
    )

    args, _ = parser.parse_known_args()

    lang_to_df = {}
    for file in os.listdir(args.wiki_path):
        if file.endswith(".csv"):
            lang = file.split(".")[0]
            df = pd.read_csv(os.path.join(args.wiki_path, file))
            if len(df) > 0:
                lang_to_df[lang] = df

    buckets = list(lang_to_df.values())
    langs = list(lang_to_df.keys())

    # Sort langs and buckets accordingly
    sorted_langs_and_buckets = sorted(zip(langs, buckets), key=lambda x: x[0])
    sorted_langs, sorted_buckets = zip(*sorted_langs_and_buckets)

    samples = uniform_sample_from_buckets(
        sorted_buckets, args.num_samples, sorted_langs
    )

    # Extract languages from the samples
    sample_langs = [sample[1] for sample in samples]

    # TODO if not enough samples are sampled, continue sampling until we have enough samples
    # Plotting the histogram
    plt.figure(figsize=(20, 8))  # Set the figure size to 20x8 inches
    bars = plt.hist(sample_langs, bins=len(sorted_langs), edgecolor="black")
    plt.xlabel("Language")
    plt.ylabel("Number of Samples")
    plt.title("Distribution of Samples Across Languages")
    plt.xticks(range(len(sorted_langs)), sorted_langs, ha="center", rotation=45)
    plt.tight_layout()

    lang_counts = pd.Series(sample_langs).value_counts().sort_index()
    for bar, count in zip(bars[2], lang_counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
        )


    # Save the plot as a PNG file
    plot_save_path = Path(args.save_path) / "sample_distribution.png"
    plt.savefig(plot_save_path)
    
    plt.show()
    # Save samples to a CSV file if save_path is provided
    save_path = Path(args.save_path) / "all_langs.csv"
    sample_df = pd.DataFrame(samples, columns=["English Title", "Wikipedia Language"])
    sample_df.to_csv(save_path, index=False)

# %%
