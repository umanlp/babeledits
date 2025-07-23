# %%
import sienna
import pandas as pd
from pathlib import Path
import argparse
from utils import extract
from translate_utils import translate_file_custom
import concurrent.futures
from typing import List


def translate_single_language(src_path: str, tgt_path: str, lang: str, prompt_pattern: List[str], num_threads: int, limit: int, timeout: int):
    """
    Translate to a single target language and process the results.
    
    Returns:
        tuple: (lang, success_bool, error_message)
    """
    try:
        print(f"Starting translation to {lang}...")
        
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
            # Read the translated file and add prompt_type information
            df = pd.read_csv(tgt_path, sep="\t")
            
            # Add prompt_type column based on the pattern we created earlier
            df["prompt_type"] = prompt_pattern
            
            # Reorder columns to match expected format
            df = df[["req_id", "prompt_type", "src", f"tgt_{lang}"]]
            
            # Check for NaN values
            if df.isnull().values.any():
                print(f"Warning: Data for {lang} has some NaN values. Please check.")
            
            # Save the final file
            df.to_csv(tgt_path, sep="\t", index=False)
            print(f"Translation to {lang} completed successfully!")
            return (lang, True, None)
        else:
            error_msg = f"Translation function returned None for {lang}"
            print(f"Translation to {lang} failed!")
            return (lang, False, error_msg)
            
    except Exception as e:
        error_msg = f"Exception occurred during {lang} translation: {str(e)}"
        print(f"Translation to {lang} failed with error: {error_msg}")
        return (lang, False, error_msg)


def chunk_list(lst: List, chunk_size: int):
    """Split a list into chunks of specified size."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate text using local translation")
    parser.add_argument(
        "--dataset_path", default="datasets/v5/dataset.json", help="Path to the dataset"
    )
    parser.add_argument("--src_lang", default="en", help="Source language code")
    parser.add_argument(
        "--tgt_langs",
        default=["it", "de", "fr"],
        nargs="+",
        help="Target language code(s)",
    )
    parser.add_argument(
        "--output_dir", default="datasets/v5/translated", help="Output directory"
    )
    parser.add_argument(
        "--rephrase", action="store_true", help="rephrase the questions"
    )
    parser.add_argument(
        "--locality", action="store_true", help="whether to also get locality translated"
    )
    parser.add_argument(
        "--portability", action="store_true", help="whether to also get portability translated"
    )
    parser.add_argument(
        "--num_threads", type=int, default=4, help="Number of threads for translation"
    )
    parser.add_argument(
        "--timeout", type=int, default=30, help="Timeout for translation requests"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit processing to the first N prompts (saves to debug folder)"
    )
    parser.add_argument(
        "--max_parallel_langs", type=int, default=10, help="Maximum number of target languages to translate in parallel"
    )
    args, _ = parser.parse_known_args()

    dataset_path = args.dataset_path
    src_lang = args.src_lang
    tgt_langs = args.tgt_langs
    output_dir = args.output_dir

    data = sienna.load(dataset_path)
    print(f"Reading dataset from {dataset_path}...")

    prompt_types = ["prompts"]
    if args.rephrase:
        prompt_types.append("prompts_gen")
    if args.locality:
        prompt_types.append("prompts_loc")
    if args.portability:
        prompt_types.append("prompts_port")

    prompt_types_with_strictness = [
        (x, True) if x not in ["prompts_loc", "prompts_port"] else (x, False) for x in prompt_types
    ]
    extracted_prompts = [
        extract(data, args.src_lang, prompt_type, strict=strict_val)
        for prompt_type, strict_val in prompt_types_with_strictness
    ]

    all_prompts = [
        item
        for sublist in zip(*extracted_prompts)
        for item in sublist
        if item is not None
    ]

    # Apply limit if specified
    if args.limit is not None:
        all_prompts = all_prompts[:args.limit]
        print(f"Limited to first {len(all_prompts)} prompts")
        print(all_prompts)

    # Create prompt pattern for later assignment
    prompt_pattern = []
    for syn_id, example in data.items():
        relation = list(example["relations"].keys())[0]
        if "targets" in example["relations"][relation]["edit"]:
            prompt_pattern.append("prompt")
        if "generality" in example["relations"][relation]["edit"]:
            prompt_pattern.append("prompt_gen")
        if "locality" in example["relations"][relation]["edit"]:
            prompt_pattern.append("prompt_loc")
        if "portability" in example["relations"][relation]["edit"]:
            prompt_pattern.append("prompt_port")

    # Apply same limit to prompt_pattern to keep them aligned
    if args.limit is not None:
        prompt_pattern = prompt_pattern[:args.limit]

    # Determine output paths based on whether limit is used
    base_path = Path(dataset_path).parent
    if args.limit is not None:
        # Use debug folder when limit is specified
        tsv_base_path = base_path / "debug" / f"limit_{args.limit}" / "tsv"
        print(f"Using debug folder for limited translation: {tsv_base_path}")
    else:
        # Use normal path when no limit
        tsv_base_path = base_path / "tsv"

    # Convert prompts to tsv, save locally
    df = pd.DataFrame(enumerate(all_prompts), columns=["req_id", "prompt"])
    tsv_src_path = tsv_base_path / "src"
    tsv_src_path.mkdir(parents=True, exist_ok=True)
    prompt_src_path = tsv_src_path / "prompts_en.tsv"
    df.to_csv(prompt_src_path, sep="\t", index=False)

    print(f"Created source file at {prompt_src_path} with {len(all_prompts)} prompts")
    print(f"Translating from {src_lang} to {tgt_langs}")
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
                prompt_tgt_path = tsv_tgt_path / f"prompts_{lang}.tsv"
                
                # Submit translation task
                future = executor.submit(
                    translate_single_language,
                    str(prompt_src_path),
                    str(prompt_tgt_path),
                    lang,
                    prompt_pattern,
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
    print(f"\n=== Translation Summary ===")
    print(f"Successful translations ({len(successful_translations)}): {successful_translations}")
    if failed_translations:
        print(f"Failed translations ({len(failed_translations)}):")
        for lang, error in failed_translations:
            print(f"  - {lang}: {error}")
    
    if args.limit is not None:
        print(f"\nLimited translation completed! Results saved to: {tsv_base_path}")
    else:
        print("\nAll translations completed!")
