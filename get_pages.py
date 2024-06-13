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

def get_top_pageviews(day, lang):
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/{lang}.wikipedia/all-access/{day.year}/{day.month:02}/{day.day:02}"
    headers = {'User-Agent': 'BabelEdits (tommaso.green@uni-mannheim.de)'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return data['items'][0]['articles']
    else:
        print(f"Failed to get pageviews for {day} (Status code {response.status_code})")
        return []

def aggregate_pageviews(start_date, end_date, lang, top_k):
    delta = end_date - start_date
    pageview_counts = defaultdict(int)

    def fetch_day_pageviews(day):
        top_pages = get_top_pageviews(day, lang)
        return day, top_pages

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(fetch_day_pageviews, start_date + timedelta(days=i)) for i in range(delta.days + 1)]
        for future in as_completed(futures):
            day, top_pages = future.result()
            for page in top_pages:
                title = page['article']
                views = page['views']
                pageview_counts[title] += views

    # Sort pages by views and get the top 10,000
    sorted_pageviews = sorted(pageview_counts.items(), key=lambda item: item[1], reverse=True)
    return sorted_pageviews[:top_k]

def get_top_pages(start_date, end_date, lang, top_k, save_path):
    lang_to_stats = {}
    for lang in langs:

        if not os.path.exists(save_path):
            print(f"Processing {lang}")
            top_pages = aggregate_pageviews(start_date, end_date, lang, top_k)
            lang_to_stats[lang] = top_pages
            # Save to JSON file
            with open(save_path, 'w') as f:
                json.dump(top_pages, f, indent=2)
        else:
            print(f"Skipping {lang}, {save_path} already exists")


def process_page(record, src_lang):
    # Initialize the Wikipedia API for the source language
    wiki_src = wikipediaapi.Wikipedia('BabelEdits (tommaso.green@uni-mannheim.de)', src_lang)
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


def process_multiple_pages(records, lang):
    results = []

    # Use ThreadPoolExecutor to handle multithreading
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit tasks to the executor
        future_to_title = [executor.submit(process_page, record, lang)   for record in records]

        # Collect the results as they complete
        for future in as_completed(future_to_title):
            results.append(future.result())

    results = pd.DataFrame(results, columns=["Title", "Views", "English Title", "Languages"])
    results = results.dropna(subset=['Languages']).sort_values(by='Languages', key=lambda x: x.str.len(), ascending=False)
    return results

if __name__ == "__main__":
    ## Params
    # XTREME-R langs
    parser = argparse.ArgumentParser(description='Process Wikipedia pageviews.')
    parser.add_argument('--langs', nargs='+', default=["af","ar","az","bg","bn","de","el","en","es","et","eu","fa","fi","fr","gu","he","hi","ht","hu","id","it","ja","jv","ka","kk","ko","lt","ml","mr","ms","my","nl","pa","pl","pt","qu","ro","ru","sw","ta","te","th","tl","tr","uk","ur","vi","wo","yo","zh"], help='List of languages')
    parser.add_argument('--year', type=int, default=2022, help='The year to process')
    parser.add_argument('--start_date', type=str, default='2022-01-01', help='The start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, default='2022-12-31', help='The end date in YYYY-MM-DD format')
    parser.add_argument('--top_k', type=int, default=10000, help='The number of top pages to retrieve')
    parser.add_argument('--save_path', type=str, default='wikipedia_data/v1', help='The main path to save the data')
    args = parser.parse_args()

    year = args.year
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()
    top_k = args.top_k
    langs = args.langs
    
    lang_to_df = {}
    for lang in langs:
        save_path_wiki = f'{args.save_path}/raw/top_{top_k}_wikipedia_pages_{lang}_{year}.json'
        save_path_csv = f'{args.save_path}/processed/{lang}_{year}_df.csv'
        
        if os.path.exists(save_path_csv):
            # Load the data from the existing file
            print(f"Data for {lang} from {save_path_csv} already exists.")
        else:
            # Process the pages and save the data to a new file
            print(f"Getting data for {lang}")
            if not os.path.exists(save_path_wiki):
                # If the data has not been downloaded yet, download it
                print(f"Getting top {top_k} pages for {lang}")
                get_top_pages(start_date, end_date, lang, top_k, save_path_wiki)
            records = sienna.load_json(save_path_wiki)
            df = process_multiple_pages(records, lang)
            df = df.drop_duplicates(subset=['English Title']).dropna(subset=['English Title'])
            df.to_csv(save_path_csv, index=False)