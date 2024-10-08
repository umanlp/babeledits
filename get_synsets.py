import argparse
import os
import pickle
import random
import time
from collections import Counter, defaultdict
from datetime import datetime
from itertools import chain
from pathlib import Path

import babelnet as bn
import pandas as pd
from babelnet import BabelSynsetID
from babelnet.language import Language
from babelnet.resources import WikipediaID

from utils import clean


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
        "--wiki_path",
        type=str,
        default="wikipedia_data/v4/all_langs.csv",
        help="Path to the Wikipedia processed csv with all languages",
    )
    parser.add_argument(
        "--max_rel", type=int, default=200, help="maximum number of relations"
    )
    parser.add_argument(
        "--dataset_path", default="datasets/v7", help="dataset path"
    )
    args, _ = parser.parse_known_args()

    langs = args.langs

    babel_langs = set([Language.from_iso(lang) for lang in langs])

    df = pd.read_csv(args.wiki_path).dropna()
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
    print(f"> Writing synsets to {save_dir / 'synsets.pkl'}")
    with open(save_dir / "synsets.pkl", "wb") as f:
        pickle.dump(results, f)

    print("Starting relation extraction")
    edges = [[e.pointer.name for e in synset.outgoing_edges()] for _, synset in results]

    flattened_edges = list(chain.from_iterable(edges))
    rel_counter = Counter(flattened_edges)
    sorted_relations = rel_counter.most_common()

    rel_df = pd.DataFrame(rel_counter.items(), columns=["relation_name", "count"])
    rel_df.sort_values(by="count", ascending=False, inplace=True)


    # remove all relations whose name ends with ym or YM, or that have some symbol derived from wordnet
    rel_df = rel_df[~rel_df.relation_name.str.contains("%|#|~|@|%|\+")]
    rel_df = rel_df[
        ~rel_df.relation_name.str.endswith("ym") & ~rel_df.relation_name.str.endswith("YM")
    ]
    rel_df = rel_df.head(args.max_rel).reset_index(drop=True)
    print(rel_df)
    # Save all relations with their counts
    rel_df.to_csv(f"{args.dataset_path}/agg_relations_all.tsv", index=False, sep="\t")

    subj_and_obj = defaultdict(dict)

    relations = rel_df["relation_name"].tolist()
    print(f"Loaded {len(results)} en synsets")
    random.shuffle(results)
    print("Relations:")
    for relation in relations:
        print(relation, end=",")
        count = 0
        found = False
        random.shuffle(results)
        syn_iter = iter(results)
        while not found:
            count += 1
            try:
                _, synset = next(syn_iter)
            except StopIteration:
                print("StopIteration")
                break
            if synset is not None:
                for edge in synset.outgoing_edges():
                    if edge.pointer.name == relation:
                        subject_sense = synset.main_sense(Language.EN)
                        target_sense = bn.get_synset(BabelSynsetID(edge.target)).main_sense(
                            Language.EN
                        )
                        if all(
                            [subject_sense, target_sense]
                        ):  # if both senses are not None
                            subject = clean(subject_sense.full_lemma)
                            object = clean(target_sense.full_lemma)
                            subj_and_obj[relation]["subject"] = subject
                            subj_and_obj[relation]["object"] = object
                            found = True
                            break

    print(subj_and_obj)


    # for each value in the column relation_name of rel_df, get the corresponding subject and object from subj_and_obj and add them to the dataframe
    rel_df["subject"] = rel_df["relation_name"].apply(
        lambda x: subj_and_obj[x]["subject"] if x in subj_and_obj else None
    )
    rel_df["object"] = rel_df["relation_name"].apply(
        lambda x: subj_and_obj[x]["object"] if x in subj_and_obj else None
    )
    rel_df.to_csv(f"{args.dataset_path}/agg_relations_with_subj_obj.tsv", sep="\t", index=False)
