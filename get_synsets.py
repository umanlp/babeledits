import os
from pathlib import Path
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


def check_langs(synset, babel_langs):
    return babel_langs.issubset(set(synset.languages))


def all_true(synset):
    if isinstance(synset, bn.BabelSynset):
        return True


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
            "hr",
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
            "yo",
            "zh",
        ],
        help="List of languages",
    )
    parser.add_argument(
        "--save_dir", type=str, default="synsets/v3", help="Save dir of the synsets"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="wikipedia_data/v4/all_langs.csv",
        help="Path to the Wikipedia processed csv with all languages",
    )
    args, _ = parser.parse_known_args()

    langs = args.langs

    babel_langs = set([Language.from_iso(lang) for lang in langs])

    df = pd.read_csv(args.data_path).dropna()
    wiki_ids = [WikipediaID(title, Language.EN) for title in df["English Title"]]

    print("Starting synset extraction")
    t_synsets_start = time.time()
    synsets = [bn.get_synset(w) for w in wiki_ids]
    t_synsets_end = time.time()
    print(
        f"> Time taken to get synsets: {(t_synsets_end - t_synsets_start)/60} minutes"
    )
    results = [
        (title, synset)
        for title, synset in zip(df["English Title"], synsets)
        if synset is not None
    ]
    print(f"> {len(results)} not-null synsets found.")
    results = [
        (title, synset) for title, synset in results if check_langs(synset, babel_langs)
    ]
    print(f"> {len(results)} (filtered) synsets found.")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # Write results to pickle file
    print(f"> Writing synsets to {save_dir / 'syns.pkl'}")
    with open(save_dir / "syns.pkl", "wb") as f:
        pickle.dump(results, f)

    edges = [[e.pointer.name for e in synset.outgoing_edges()] for _, synset in results]

    flattened_edges = list(chain.from_iterable(edges))
    counter = Counter(flattened_edges)
    sorted_relations = counter.most_common()

    print(f"> Writing relations to {save_dir / 'relations.txt'}")
    with open(save_dir / "relations.txt", "w") as file:
        for relation, count in sorted_relations:
            file.write(f"{relation}:{count}\n")
