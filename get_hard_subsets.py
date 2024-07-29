import argparse
import os
from pathlib import Path
from random import sample
import pandas as pd
from get_pages import process_multiple_pages


def main():
    parser = argparse.ArgumentParser(description='Get hard subsets of Wikipedia pages.')
    parser.add_argument('--lang', type=str, help='Language code')
    parser.add_argument('--min_views', type=int, default=10000, help='Minimum number of views')
    parser.add_argument('--max_views', type=int, default=100000, help='Maximum number of views')
    parser.add_argument('--sample_size', type=int, default=10000, help='Number of pages to sample')
    parser.add_argument('--data_path', type=str, default="wikipedia_data/v5/raw", help='Path to the input data')
    parser.add_argument('--save_path', type=str, default="wikipedia_data/v5/hard/processed", help='Path to save the output data')
    args = parser.parse_args()

    lang = args.lang
    user_agent = os.getenv("WIKI_AGENT")

    load_path = Path(f"{args.data_path}/{lang}.csv")
    df = pd.read_csv(load_path, sep=" ", header=0, names=["Title", "Views"])
    df = df.dropna().reset_index(drop=True)
    df = df[pd.to_numeric(df['Views'], errors='coerce').notnull()]
    df['Views'] = df['Views'].astype(int)

    filtered_df = df[(df['Views'] >= args.min_views) & (df['Views'] <= args.max_views)]
    sample_size = min(args.sample_size, len(filtered_df))
    if sample_size < len(filtered_df):
        print(f"Sampling only {sample_size} pages from {len(filtered_df)} pages")
    filtered_df = filtered_df.sample(sample_size)

    langs = ["en", lang]
    filtered_list = list(zip(filtered_df['Title'], filtered_df['Views']))
    res = process_multiple_pages(filtered_list, lang, user_agent, langs)
    res = res.dropna().sort_values("Views")

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    res.to_csv(save_path / f"{lang}.csv", index=False)

if __name__ == "__main__":
    main()

