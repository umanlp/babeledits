import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import babelnet as bn
import pandas as pd
import wikipediaapi
from babelnet import Language
from babelnet.resources import WikipediaID

from get_synsets import check_langs


def process_page(record, src_lang, user_agent):
    # Initialize the Wikipedia API for the source language
    wiki_src = wikipediaapi.Wikipedia(user_agent, src_lang)
    title, views = record
    try:
        # Get the source language page
        page_src = wiki_src.page(title)

        # Check if the page exists
        try:
            page_existance = page_src.exists()
            if not page_existance:
                return title, views, None, None
        except Exception:
            return title, views, None, None

        # Check for an English version of the page

        if src_lang == "en":
            langlinks = page_src.langlinks
            return title, views, title, sorted([src_lang] + list(langlinks.keys()))

        try:
            langlinks = page_src.langlinks
            english_title = langlinks["en"].title if "en" in langlinks else None
            lang_list = sorted([src_lang] + list(langlinks.keys()))
        except Exception:
            english_title = None
            lang_list = [src_lang]
        return page_src.title, views, english_title, lang_list
    except json.JSONDecodeError:
        print(f"JSON Decode Error for {record}")
        return record, "JSONDecodeError"


def process_multiple_pages(records, src_lang, user_agent, langs):
    results = []

    # Use ThreadPoolExecutor to handle multithreading
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit tasks to the executor
        future_to_title = [
            executor.submit(process_page, record, src_lang, user_agent)
            for record in records
        ]

        # Collect the results as they complete
        for future in as_completed(future_to_title):
            results.append(future.result())

    results = (
        pd.DataFrame(results, columns=["Title", "Views", "English Title", "Languages"])
        .drop_duplicates(subset=["English Title"])
        .dropna()
    )

    titles = [str(x) for x in results["English Title"]]

    babel_langs = set([Language.from_iso(lang) for lang in langs])
    wiki_ids = [WikipediaID(title, Language.EN) for title in titles]
    print(f"Checking Babelnet presence for {len(titles)} pages")
    synsets = [bn.get_synset(w) for w in wiki_ids]
    selected_pages = [
        title
        for title, synset in zip(titles, synsets)
        if synset is not None and check_langs(synset, babel_langs)
    ]
    mask = results["English Title"].isin(selected_pages)
    print(f"Num Wikipedia pages {len(titles)}, Num selected pages {mask.sum()}")
    results = results[mask]
    return results


if __name__ == "__main__":
    ## Params
    # XTREME-R langs
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
        "--top_k", type=int, default=None, help="The number of top pages to retrieve"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="wikipedia_data/v5",
        help="The main path to save the data",
    )
    parser.add_argument(
        "--user_agent",
        type=str,
        default=os.environ["WIKI_AGENT"],
        help="The user agent to use for Wikipedia API requests",
    )
    args = parser.parse_args()

    print(args.user_agent)
    top_k = args.top_k
    langs = args.langs
    user_agent = args.user_agent

    processed_dir = Path(f"{args.save_path}/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    for lang in langs:
        save_path_wiki = Path(f"{args.save_path}/raw/{lang}.csv")
        if not save_path_wiki.exists():
            raise ValueError(f"File {save_path_wiki} does not exist")

    for lang in langs:
        save_path_wiki = Path(f"{args.save_path}/raw/{lang}.csv")
        save_path_csv = Path(f"{args.save_path}/processed/{lang}.csv")

        if save_path_csv.exists():
            # Load the data from the existing file
            print(f"Data for {lang} from {save_path_csv} already exists.")
        else:
            # Process the pages and save the data to a new file
            print(f"Processing data for {lang}")

            wiki_df = (
                pd.read_csv(save_path_wiki, sep=" ", header=0, names=["Title", "Views"])
                .dropna()
                .reset_index(drop=True)
            )
            wiki_df = wiki_df[
                pd.to_numeric(wiki_df["Views"], errors="coerce").notnull()
            ]
            wiki_df["Views"] = wiki_df["Views"].astype(int)
            wiki_df = wiki_df.sort_values("Views", ascending=False).iloc[: int(1.5 * top_k)]

            # Extracting and saving wikipedia pages 
            records = list(zip(wiki_df["Title"], wiki_df["Views"]))
            print(f"Post-processing {len(records)} pages for {lang}")
            df = process_multiple_pages(records, lang, user_agent, langs).iloc[
                :top_k
            ]
            print(f"Saving {len(df)} pages for {lang}")
            df.to_csv(save_path_csv, index=False, encoding="utf-8")
