# %%
import sienna
import pandas as pd
from pathlib import Path
import argparse
from utils import extract
from translate_gt import translate_file_custom


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

    # Create target directory
    tsv_tgt_path = tsv_base_path / "tgt"
    tsv_tgt_path.mkdir(parents=True, exist_ok=True)

    # Translate to each target language
    for lang in tgt_langs:
        print(f"\nTranslating to {lang}...")
        prompt_tgt_path = tsv_tgt_path / f"prompts_{lang}.tsv"
        
        # Use translate_file_custom for translation
        result = translate_file_custom(
            str(prompt_src_path),
            str(prompt_tgt_path),
            lang,
            num_threads=args.num_threads,
            limit=args.limit,  # Pass limit to the translation function
            timeout=args.timeout
        )
        
        if result:
            # Read the translated file and add prompt_type information
            df = pd.read_csv(prompt_tgt_path, sep="\t")
            
            # Add prompt_type column based on the pattern we created earlier
            df["prompt_type"] = prompt_pattern
            
            # Reorder columns to match expected format
            df = df[["req_id", "prompt_type", "src", f"tgt_{lang}"]]
            
            # Check for NaN values
            if df.isnull().values.any():
                print(f"Warning: Data for {lang} has some NaN values. Please check.")
            
            # Save the final file
            df.to_csv(prompt_tgt_path, sep="\t", index=False)
            print(f"Translation to {lang} completed successfully!")
        else:
            print(f"Translation to {lang} failed!")

    if args.limit is not None:
        print(f"\nLimited translation completed! Results saved to: {tsv_base_path}")
    else:
        print("\nAll translations completed!")
