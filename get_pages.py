import wikipediaapi
import pandas as pd
import json
import sienna
import os
import datetime
import requests
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta, datetime
import argparse


def get_top_pageviews(day, lang, user_agent):
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/{lang}.wikipedia/all-access/{day.year}/{day.month:02}/{day.day:02}"
    headers = {"User-Agent": user_agent}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return data["items"][0]["articles"]
    else:
        print(f"Failed to get pageviews for {day} (Status code {response.status_code})")
        return []


def aggregate_pageviews(start_date, end_date, lang, top_k, user_agent):
    delta = end_date - start_date
    pageview_counts = defaultdict(int)

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [
            executor.submit(
                get_top_pageviews, start_date + timedelta(days=i), lang, user_agent
            )
            for i in range(delta.days + 1)
        ]
        for future in as_completed(futures):
            top_pages = future.result()
            for page in top_pages:
                title = page["article"]
                views = page["views"]
                pageview_counts[title] += views

    # Sort pages by views and get the top-k
    sorted_pageviews = sorted(
        pageview_counts.items(), key=lambda item: item[1], reverse=True
    )

    # we return double the top-k, since we will have to filter later
    return sorted_pageviews[: top_k]


def get_top_pages(start_date, end_date, lang, top_k, save_path, user_agent):
    print(f"Downloading wikipedia data for {lang}")
    top_pages = aggregate_pageviews(start_date, end_date, lang, top_k, user_agent)
    with open(save_path, "w") as f:
        json.dump(top_pages, f, indent=2)


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
        except:
            return title, views, None, None

        # Check for an English version of the page

        if src_lang == "en":
            langlinks = page_src.langlinks
            return title, views, title, sorted([src_lang] + list(langlinks.keys()))

        try:
            langlinks = page_src.langlinks
            english_title = langlinks["en"].title if "en" in langlinks else None
            lang_list = sorted([src_lang] + list(langlinks.keys()))
        except:
            english_title = None
            lang_list = [src_lang]
        return title, views, english_title, lang_list
    except json.JSONDecodeError:
        print(f"JSON Decode Error for {record}")
        return record, "JSONDecodeError"


def process_multiple_pages(records, lang, user_agent):
    results = []

    # Use ThreadPoolExecutor to handle multithreading
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit tasks to the executor
        future_to_title = [
            executor.submit(process_page, record, lang, user_agent)
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
    # results["Languages filtered"] = results["Languages"].apply(
        # lambda x: list(set(x) & set(langs))
    # )
    # results["Selected Language Count"] = results["Languages filtered"].apply(
        # lambda x: len(x)
    # )
    # results = results.dropna()
    # results = results.sort_values(
        # by="Languages filtered", key=lambda x: x.str.len(), ascending=False
    # )
    titles = [str(x) for x in results["English Title"]]
    from get_synsets import check_langs
    import babelnet as bn
    from babelnet import Language
    from babelnet.resources import WikipediaID

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
    results = results[results["English Title"].isin(selected_pages)]
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
        "--start_date",
        type=str,
        default="2021-01-01",
        help="The start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default="2021-12-31",
        help="The end date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--top_k", type=int, default=10000, help="The number of top pages to retrieve"
    )
    parser.add_argument("--year", type=int, default=2021, help="The year to process")
    parser.add_argument(
        "--save_path",
        type=str,
        default="wikipedia_data/v3",
        help="The main path to save the data",
    )
    parser.add_argument(
        "--user_agent",
        type=str,
        default=os.environ["WIKI_AGENT"],
        help="The user agent to use for Wikipedia API requests",
    )
    args = parser.parse_args()

    import logging
    import sys

    file_handler = logging.FileHandler(filename='logs/get_pages.log')
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]

    logging.basicConfig(
    level=logging.INFO, 
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    handlers=handlers
    )

    logger = logging.getLogger("my logger")
    logging.getLogger('wikipediaapi').setLevel(logging.WARN)
    print = logger.info
    
    print(args.user_agent)
    year = args.year
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    top_k = args.top_k
    langs = args.langs
    user_agent = args.user_agent

    lang_to_df = {}

    if not os.path.exists(f"{args.save_path}/raw"):
        os.makedirs(f"{args.save_path}/raw")
    if not os.path.exists(f"{args.save_path}/processed"):
        os.makedirs(f"{args.save_path}/processed")

    for lang in langs:
        save_path_wiki = (
            f"{args.save_path}/raw/top_{top_k}_wikipedia_pages_{lang}_{year}.json"
        )
        save_path_csv = f"{args.save_path}/processed/{lang}_{year}_df.csv"

        if os.path.exists(save_path_csv):
            # Load the data from the existing file
            print(f"Data for {lang} from {save_path_csv} already exists.")
        else:
            # Process the pages and save the data to a new file
            print(f"Getting data for {lang}")
            if not os.path.exists(save_path_wiki):
                # If the data has not been downloaded yet, download it
                print(f"Getting top {top_k} pages for {lang}")
                get_top_pages(
                    start_date, end_date, lang, top_k, save_path_wiki, user_agent
                )
            records = sienna.load(save_path_wiki)
            print(f"Post-processing {len(records)} pages for {lang}")
            df = process_multiple_pages(records, lang, user_agent).iloc[:top_k]
            df.to_csv(save_path_csv, index=False, encoding="utf-8")
