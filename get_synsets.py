import os
import pickle
import time

import argparse
import babelnet as bn
import pandas as pd
from datetime import datetime
from babelnet.language import Language
from babelnet.resources import WikipediaID
from collections import Counter
from itertools import chain

if __name__ == "__main__":
    ## Params
    parser = argparse.ArgumentParser(description="Process Wikipedia pageviews.")
    parser.add_argument(
        "--langs",
        nargs="+",
        default=[
            "af",
            "ar",
            "az",
            "bg",
            "bn",
            "de",
            "el",
            "en",
            "es",
            "et",
            "eu",
            "fa",
            "fi",
            "fr",
            "gu",
            "he",
            "hi",
            "ht",
            "hu",
            "id",
            "it",
            "ja",
            "jv",
            "ka",
            "kk",
            "ko",
            "lt",
            "ml",
            "mr",
            "ms",
            "my",
            "nl",
            "pa",
            "pl",
            "pt",
            "qu",
            "ro",
            "ru",
            "sw",
            "ta",
            "te",
            "th",
            "tl",
            "tr",
            "uk",
            "ur",
            "vi",
            "wo",
            "yo",
            "zh",
        ],
        help="List of languages",
    )
    parser.add_argument("--year", type=int, default=2021, help="The year to process")
    parser.add_argument(
        "--save_dir", type=str, default="synsets/v3", help="Save dir of the synsets"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="wikipedia_data/v3/processed",
        help="Path to the Wikipedia processed data",
    )
    args = parser.parse_args()

    langs = args.langs
    year = args.year

    lang_to_df = {}
    for lang in langs:
        t_start = time.time()
        save_dir = f"{args.save_dir}/{lang}"
        save_path = f"{args.data_path}/{lang}_{year}_df.csv"
        df = pd.read_csv(save_path).dropna()

        wiki_ids = [WikipediaID(title, Language.EN) for title in df["English Title"]]
        results = []

        print(f"> Getting synsets for {lang}")
        synsets = bn.get_synsets(*wiki_ids)
        results = [
            (title, synset) for title, synset in zip(df["English Title"], synsets)
        ]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Write results to pickle file
        print(f"> Writing synsets for {lang} to {save_dir}/{lang}_syns.pkl")
        with open(f"{save_dir}/{lang}_syns.pkl", "wb") as f:
            pickle.dump(results, f)

        filtered_synsets = [
            (title, synset) for title, synset in results if synset is not None
        ]
        edges = [
            [e.pointer.name for e in synset.outgoing_edges()]
            for _, synset in filtered_synsets
        ]

        flattened_edges = list(chain.from_iterable(edges))
        counter = Counter(flattened_edges)
        sorted_relations = counter.most_common()
        print(
            f"> Writing relations for {lang} to {save_dir}/{lang}_relations.txt",
            end="\n\n",
        )
        with open(f"{save_dir}/{lang}_relations.txt", "w") as file:
            for relation, count in sorted_relations:
                file.write(f"{relation}:{count}\n")

        print(
            f"Time taken for {lang}: {(time.time()-t_start)/60} minutes for {len(synsets)} synsets"
        )
