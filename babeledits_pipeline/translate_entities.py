from pathlib import Path
import pandas as pd
from utils import extract
import sienna
import argparse
from translate_utils import translate_file_custom
import concurrent.futures
from typing import List


def translate_single_language_entities(src_path: str, tgt_path: str, lang: str, num_threads: int, limit: int, timeout: int):
    """
    Translate entities to a single target language and process the results.
    
    Returns:
        tuple: (lang, success_bool, error_message)
    """
    try:
        print(f"Starting entity translation to {lang}...")
        
        # Use translate_file_custom for translation
        result = translate_file_custom(
            src_path,
            tgt_path,
            lang,
            num_threads=num_threads,
            limit=limit,
            timeout=timeout
        )
        
        if result:
            # Read the translated file and process for entities (no prompt_type needed)
            df = pd.read_csv(tgt_path, sep="\t")
            
            # Reorder columns to match expected format for entities
            df = df[["req_id", "src", f"tgt_{lang}"]]
            
            # Check for NaN values
            if df.isnull().values.any():
                print(f"Warning: Entity data for {lang} has some NaN values. Please check.")
            
            # Sort by req_id for consistency
            df = df.sort_values("req_id", ascending=True)
            
            # Save the final file
            df.to_csv(tgt_path, sep="\t", index=False)
            print(f"Entity translation to {lang} completed successfully!")
            return (lang, True, None)
        else:
            error_msg = f"Translation function returned None for {lang}"
            print(f"Entity translation to {lang} failed!")
            return (lang, False, error_msg)
            
    except Exception as e:
        error_msg = f"Exception occurred during {lang} entity translation: {str(e)}"
        print(f"Entity translation to {lang} failed with error: {error_msg}")
        return (lang, False, error_msg)


def chunk_list(lst: List, chunk_size: int):
    """Split a list into chunks of specified size."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate entities using local translation")
    parser.add_argument("--src_lang", type=str, default="en", help="Source language code")
    parser.add_argument(
        "--tgt_langs", default=["it", "de", "fr"], nargs="+", help="Target language code(s)"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="datasets/v6/translated/test.json",
        help="Dataset path",
    )
    parser.add_argument(
        "--num_threads", type=int, default=4, help="Number of threads for translation"
    )
    parser.add_argument(
        "--timeout", type=int, default=30, help="Timeout for translation requests"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit processing to the first N entities (saves to debug folder)"
    )
    parser.add_argument(
        "--max_parallel_langs", type=int, default=10, help="Maximum number of target languages to translate in parallel"
    )

    args = parser.parse_args()

    src_lang = args.src_lang
    tgt_langs = args.tgt_langs
    dataset_path = args.dataset_path

    # Load data and extract entities
    data = sienna.load(dataset_path)
    subjects = extract(data, "en", "subjects")
    objects = extract(data, "en", "targets")
    ground_truths = extract(data, "en", "ground_truths")
    ground_truths_port = extract(data, "en", "ground_truths_port", strict=False)
    ground_truths_port = [e for e in ground_truths_port if e]
    ground_truths_loc = extract(data, "en", "ground_truths_loc", strict=False)
    ground_truths_loc = [e for e in ground_truths_loc if e]
    entities = subjects + objects + ground_truths_loc + ground_truths + ground_truths_port

    # Apply limit if specified
    if args.limit is not None:
        entities = entities[:args.limit]
        print(f"Limited to first {len(entities)} entities")

    # Determine output paths based on whether limit is used
    base_path = Path(dataset_path).parent
    if args.limit is not None:
        # Use debug folder when limit is specified
        tsv_base_path = base_path / "debug" / f"limit_{args.limit}" / "tsv_entities"
        print(f"Using debug folder for limited entity translation: {tsv_base_path}")
    else:
        # Use normal path when no limit
        tsv_base_path = base_path / "tsv_entities"

    # Convert entities to tsv, save locally
    df = pd.DataFrame(enumerate(entities), columns=["req_id", "entities"])
    tsv_src_path = tsv_base_path / "src"
    tsv_src_path.mkdir(parents=True, exist_ok=True)
    entities_src_path = tsv_src_path / "entities_en.tsv"
    df.to_csv(entities_src_path, sep="\t", index=False)

    print(f"Created source file at {entities_src_path} with {len(entities)} entities")
    print(f"Translating entities from {src_lang} to {tgt_langs}")
    print(f"Will process up to {args.max_parallel_langs} languages in parallel")

    # Create target directory
    tsv_tgt_path = tsv_base_path / "tgt"
    tsv_tgt_path.mkdir(parents=True, exist_ok=True)

    # Split target languages into chunks for parallel processing
    lang_chunks = list(chunk_list(tgt_langs, args.max_parallel_langs))
    total_chunks = len(lang_chunks)
    
    successful_translations = []
    failed_translations = []

    # Process each chunk of languages in parallel
    for chunk_idx, lang_chunk in enumerate(lang_chunks, 1):
        print(f"\n=== Processing chunk {chunk_idx}/{total_chunks}: {lang_chunk} ===")
        
        # Prepare futures for parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(lang_chunk)) as executor:
            futures = {}
            
            for lang in lang_chunk:
                entities_tgt_path = tsv_tgt_path / f"entities_{lang}.tsv"
                
                # Submit translation task
                future = executor.submit(
                    translate_single_language_entities,
                    str(entities_src_path),
                    str(entities_tgt_path),
                    lang,
                    args.num_threads,
                    args.limit,
                    args.timeout
                )
                futures[future] = lang
            
            # Wait for all translations in this chunk to complete
            for future in concurrent.futures.as_completed(futures):
                lang, success, error_msg = future.result()
                if success:
                    successful_translations.append(lang)
                else:
                    failed_translations.append((lang, error_msg))
        
        print(f"Chunk {chunk_idx} completed.")

    # Print final summary
    print(f"\n=== Entity Translation Summary ===")
    print(f"Successful translations ({len(successful_translations)}): {successful_translations}")
    if failed_translations:
        print(f"Failed translations ({len(failed_translations)}):")
        for lang, error in failed_translations:
            print(f"  - {lang}: {error}")
    
    if args.limit is not None:
        print(f"\nLimited entity translation completed! Results saved to: {tsv_base_path}")
    else:
        print("\nAll entity translations completed!")
