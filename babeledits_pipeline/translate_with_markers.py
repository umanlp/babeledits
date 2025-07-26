# %%
import sienna
import pandas as pd
from pathlib import Path
import argparse
from utils import (
    extract,
    format_prompt,
    extract_target,
    extract_subject,
    clean_prompt,
)
from translate_utils import translate_file_custom
import concurrent.futures
from typing import List


def translate_single_language_markers(src_path: str, tgt_path: str, lang: str, prompt_pattern: List[str], data: dict, src_lang: str, num_threads: int, limit: int, timeout: int):
    """
    Translate marked prompts to a single target language and process the results.
    
    Returns:
        tuple: (lang, success_bool, error_message)
    """
    try:
        print(f"Starting marked translation to {lang}...")
        
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
            # Read the translated file and add marker-specific processing
            df = pd.read_csv(tgt_path, sep="\t")
            
            # Add prompt_type column based on the pattern we created earlier
            df["prompt_type"] = prompt_pattern
            
            # Extract subjects and objects from marked translations
            df[f"subject_{lang}"] = [
                extract_subject(x) if prompt_type != "prompt_port" else "-"
                for x, prompt_type in zip(df[f"tgt_{lang}"], df["prompt_type"])
            ]
            df[f"object_{lang}"] = [extract_target(x) for x in df[f"tgt_{lang}"]]
            df[f"tgt_raw_{lang}"] = df[f"tgt_{lang}"]
            df[f"tgt_{lang}"] = [clean_prompt(x) for x in df[f"tgt_{lang}"]]
            
            # Reorder columns to match expected format for marked translations
            df = df[
                [
                    "req_id",
                    "prompt_type",
                    "src",
                    f"tgt_raw_{lang}",
                    f"tgt_{lang}",
                    f"subject_{lang}",
                    f"object_{lang}",
                ]
            ]
            
            # Check for NaN values
            if df.isnull().values.any():
                print(f"⚠️ Data for {lang} has some problems with NaN values. Please check.")
            
            # Sort by req_id for consistency
            df = df.sort_values("req_id", ascending=True)
            
            # Save the final file
            df.to_csv(tgt_path, sep="\t", index=False)
            print(f"Marked translation to {lang} completed successfully!")
            return (lang, True, None)
        else:
            error_msg = f"Translation function returned None for {lang}"
            print(f"Marked translation to {lang} failed!")
            return (lang, False, error_msg)
            
    except Exception as e:
        error_msg = f"Exception occurred during {lang} marked translation: {str(e)}"
        print(f"Marked translation to {lang} failed with error: {error_msg}")
        return (lang, False, error_msg)


def chunk_list(lst: List, chunk_size: int):
    """Split a list into chunks of specified size."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate text with markers using local translation")
    parser.add_argument(
        "--dataset_path",
        default="datasets/debug/dataset.json",
        help="Path to the dataset",
    )
    parser.add_argument("--src_lang", default="en", help="Source language code")
    parser.add_argument(
        "--tgt_langs",
        default=["it", "ja", "ar"],
        nargs="+",
        help="Target language code(s)",
    )
    parser.add_argument(
        "--rephrase", action="store_true", help="rephrase the questions"
    )
    parser.add_argument(
        "--locality", action="store_true", help="whether to also get locality"
    )
    parser.add_argument(
        "--portability", action="store_true", help="whether to also get portability"
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

    data = sienna.load(dataset_path)
    print(f"Reading dataset from {dataset_path}...")
    prompts = extract(data, args.src_lang, "prompts")
    subjects = extract(data, args.src_lang, "subjects")
    targets = extract(data, args.src_lang, "targets")

    prompts = [format_prompt(p, s, t) for (p, s, t) in zip(prompts, subjects, targets)]

    prompt_types = ["prompts"]
    if args.rephrase:
        prompt_types.append("prompts_gen")
    if args.locality:
        prompt_types.append("prompts_loc")
    if args.portability:
        prompt_types.append("prompts_port")

    all_prompts = []

    for syn_id, example in data.items():
        relation = list(example["relations"].keys())[0]
        all_prompts.append(
            format_prompt(
                example["relations"][relation]["edit"]["prompts"][args.src_lang],
                example["subjects"][args.src_lang],
                example["relations"][relation]["edit"]["targets"][args.src_lang],
            )
        )
        if "generality" in example["relations"][relation]["edit"]:
            all_prompts.append(
                format_prompt(
                    example["relations"][relation]["edit"]["generality"]["prompts_gen"][
                        args.src_lang
                    ],
                    example["subjects"][args.src_lang],
                    example["relations"][relation]["edit"]["targets"][args.src_lang],
                )
            )
        if "locality" in example["relations"][relation]["edit"]:
            loc_relation = list(
                example["relations"][relation]["edit"]["locality"].keys()
            )[0]
            all_prompts.append(
                format_prompt(
                    example["relations"][relation]["edit"]["locality"][loc_relation][
                        "prompts_loc"
                    ][args.src_lang],
                    example["subjects"][args.src_lang],
                    example["relations"][relation]["edit"]["locality"][loc_relation][
                        "ground_truths_loc"
                    ][args.src_lang],
                )
            )
        if "portability" in example["relations"][relation]["edit"]: # no need for formatting, we do not really need the subject for portability
            port_relation = list(
                example["relations"][relation]["edit"]["portability"]["multi_hop"].keys()
            )[0]
            port_target = example["relations"][relation]["edit"]["portability"]["multi_hop"][port_relation][
                "ground_truths_port"
            ][args.src_lang]
            all_prompts.append(
                example["relations"][relation]["edit"]["portability"]["multi_hop"][port_relation][
                    "prompts_port"
                ][args.src_lang] + f" <o>{port_target}</o>"
            )

    # Apply limit if specified
    if args.limit is not None:
        all_prompts = all_prompts[:args.limit]
        print(f"Limited to first {len(all_prompts)} prompts")

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
        tsv_base_path = base_path / "debug" / f"limit_{args.limit}" / "tsv_marked"
        print(f"Using debug folder for limited marked translation: {tsv_base_path}")
    else:
        # Use normal path when no limit
        tsv_base_path = base_path / "tsv_marked"

    # Convert prompts to tsv, save locally
    df = pd.DataFrame(enumerate(all_prompts), columns=["req_id", "prompt"])
    tsv_src_path = tsv_base_path / "src"
    tsv_src_path.mkdir(parents=True, exist_ok=True)
    prompt_src_path = tsv_src_path / "prompts_marked_en.tsv"
    df.to_csv(prompt_src_path, sep="\t", index=False)

    print(f"Created source file at {prompt_src_path} with {len(all_prompts)} marked prompts")
    print(f"Translating marked prompts from {src_lang} to {tgt_langs}")
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
                prompt_tgt_path = tsv_tgt_path / f"prompts_marked_{lang}.tsv"
                
                # Submit translation task
                future = executor.submit(
                    translate_single_language_markers,
                    str(prompt_src_path),
                    str(prompt_tgt_path),
                    lang,
                    prompt_pattern,
                    data,
                    args.src_lang,
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
    print(f"\n=== Marked Translation Summary ===")
    print(f"Successful translations ({len(successful_translations)}): {successful_translations}")
    if failed_translations:
        print(f"Failed translations ({len(failed_translations)}):")
        for lang, error in failed_translations:
            print(f"  - {lang}: {error}")
    
    if args.limit is not None:
        print(f"\nLimited marked translation completed! Results saved to: {tsv_base_path}")
    else:
        print("\nAll marked translations completed!")
