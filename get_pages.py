import wikipediaapi
import pandas as pd
import json
import sienna
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta, date

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
    year = 2022
    # XTREME-R langs
    langs = ["af","ar","az","bg","bn","de","el","en","es","et","eu","fa","fi","fr","gu","he","hi","ht","hu","id","it","ja","jv","ka","kk","ko","lt","ml","mr","ms","my","nl","pa","pl","pt","qu","ro","ru","sw","ta","te","th","tl","tr","uk","ur","vi","wo","yo","zh"]
    start_date = date(year, 1, 1)
    end_date = date(year, 12, 31)
    top_k = 10000


    lang_to_df = {}
    for lang in langs:

        save_path = f'wikipedia_stats/processed_data2/{lang}_{year}_df.csv'

        if os.path.exists(save_path):
            # Load the data from the existing file
            print(f"Loading data for {lang} from {save_path}")
            lang_to_df[lang] = pd.read_csv(save_path)
        else:
            # Process the pages and save the data to a new file
            print(f"Getting data for {lang}")
            records = sienna.load(f"wikipedia_stats/top_{top_k}_wikipedia_pages_{lang}_{year}.json")
            df = process_multiple_pages(records, lang)
            lang_to_df[lang] = df
            lang_to_df[lang].to_csv(save_path, index=False)
        lang_to_df[lang].to_csv(save_path, index=False)