import argparse
from translatepy.translators.google import GoogleTranslate
from translatepy.exceptions import TranslatepyException, UnknownLanguage
from translatepy import Language
import concurrent.futures
import csv
import itertools
import os
from datetime import datetime
from tqdm import tqdm
import time
import sys

def translate_sentence(translator, sentence, dest_lang):
    try:
        translation = translator.translate(sentence, dest_lang, source_language="en")
        if translation and hasattr(translation, "result"):
            return translation.result
        else:
            raise TranslatepyException("Translator returned an invalid or None result.")
    except TranslatepyException:
        raise


def translate_file_custom(input_file, output_file, dest_lang, num_threads=4, limit=None, timeout=30):
    """
    Custom translation function that writes to a specified output file.
    This version is designed to be used by translate.py
    """

    try:
        language = Language(dest_lang)
        print(f"Target language code: {language.alpha2}")
    except UnknownLanguage as e:
        print(f"Error: Could not resolve language '{dest_lang}' (original: '{dest_lang}').")
        if e.guessed_language:
            print(
                f"Did you mean: {e.guessed_language} (Similarity: {round(e.similarity, 2)}%)?"
            )
        return None

    translator = GoogleTranslate()
    translator._supported_languages.update(["jv", "qu", "zh"])
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)
            rows_to_process = list(itertools.islice(reader, limit))
            sentences = [row[1] for row in rows_to_process if row]

        translations = [None] * len(sentences)
        failed_indices = []

        future_to_index = {
            executor.submit(translate_sentence, translator, sentences[i], language): i
            for i in range(len(sentences))
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_index),
            total=len(sentences),
            desc=f"Translating to {language.alpha2}",
        ):
            index = future_to_index[future]
            try:
                translations[index] = future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                print(
                    f"Translation timed out for sentence at index {index} after {timeout} seconds."
                )
                failed_indices.append(index)
            except TranslatepyException as e:
                print(f"Translation API error for sentence at index {index}: {e}")
                failed_indices.append(index)
            except Exception as e:
                print(
                    f"Unexpected error in thread for index {index}: {type(e).__name__}: {e}"
                )
                failed_indices.append(index)

        # Retry logic
        retries = 3
        for i in range(retries):
            if not failed_indices:
                break

            print(
                f"\nAttempt {i + 1}/{retries} to retry failed translations after a 60-second delay..."
            )
            time.sleep(60)

            indices_to_retry = list(failed_indices)
            failed_indices.clear()

            retry_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=num_threads
            )
            retry_future_to_index = {
                retry_executor.submit(
                    translate_sentence, translator, sentences[idx], language
                ): idx
                for idx in indices_to_retry
            }

            for future in tqdm(
                concurrent.futures.as_completed(retry_future_to_index),
                total=len(indices_to_retry),
                desc=f"Retrying (Attempt {i + 1})",
            ):
                index = retry_future_to_index[future]
                try:
                    translations[index] = future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    print(
                        f"Retry {i + 1} timed out for sentence at index {index} after {timeout} seconds."
                    )
                    failed_indices.append(index)
                except TranslatepyException as e:
                    print(
                        f"Retry {i + 1} failed with API error for sentence at index {index}: {e}"
                    )
                    failed_indices.append(index)
                except Exception as e:
                    print(
                        f"Unexpected error in retry thread for index {index}: {type(e).__name__}: {e}"
                    )
                    failed_indices.append(index)

            retry_executor.shutdown()

        if failed_indices:
            print("\nSome translations still failed after all retries.")
            for index in failed_indices:
                translations[index] = "NaN"

        # Output file writing to specified path
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Use the original language code for output file naming, not the mapped one
        with open(output_file, "w", encoding="utf-8", newline="") as f_out:
            writer = csv.writer(f_out, delimiter="\t")
            # Write the new header format without tgt_gloss column
            new_header = ["req_id", "prompt_type", "src", f"tgt_{dest_lang}"]
            writer.writerow(new_header)

            for i, data in enumerate(zip(rows_to_process, translations)):
                try:
                    row, translation = data
                    # Map data to new column format
                    req_id = row[0] if len(row) > 0 else str(i)
                    prompt_type = None  # Will be set later in translate.py
                    src = row[1] if len(row) > 1 else ""
                    tgt = translation if translation else "NaN"
                    
                    writer.writerow([req_id, prompt_type, src, tgt])
                except TypeError:
                    print(f"FATAL ERROR at row index {i}: Could not unpack data.")
                    print(f"         Problematic data item: {data}")
                    print(f"         Type of this item: {type(data)}")
                    # Continue to next item to see if there are more errors
                    writer.writerow([f"ERROR_{i}", "error", "Could not process this row.", "NaN"])

        print(f"Translation complete. Output written to: {output_file}")
        return dest_lang  # Return original language code, not mapped one

    except KeyboardInterrupt:
        print("\nTranslation interrupted by user. Shutting down...")
        return None
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    finally:
        if sys.version_info >= (3, 9):
            executor.shutdown(wait=False, cancel_futures=True)
        else:
            executor.shutdown(wait=False)


def translate_file(input_file, num_threads, dest_lang, limit, timeout):
    try:
        language = Language(dest_lang)
        print(f"Target language code: {language.alpha2}")
    except UnknownLanguage as e:
        print(f"Error: Could not resolve language '{dest_lang}'.")
        if e.guessed_language:
            print(
                f"Did you mean: {e.guessed_language} (Similarity: {round(e.similarity, 2)}%)?"
            )
        return

    translator = GoogleTranslate()

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)
            rows_to_process = list(itertools.islice(reader, limit))
            sentences = [row[1] for row in rows_to_process if row]

        translations = [None] * len(sentences)
        failed_indices = []

        future_to_index = {
            executor.submit(translate_sentence, translator, sentences[i], language): i
            for i in range(len(sentences))
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_index),
            total=len(sentences),
            desc="Translating",
        ):
            index = future_to_index[future]
            try:
                translations[index] = future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                print(
                    f"Translation timed out for sentence at index {index} after {timeout} seconds."
                )
                failed_indices.append(index)
            except TranslatepyException as e:
                print(f"Translation API error for sentence at index {index}: {e}")
                failed_indices.append(index)
            except Exception as e:
                print(
                    f"Unexpected error in thread for index {index}: {type(e).__name__}: {e}"
                )
                failed_indices.append(index)

        # Retry logic
        retries = 3
        for i in range(retries):
            if not failed_indices:
                break

            print(
                f"\nAttempt {i + 1}/{retries} to retry failed translations after a 60-second delay..."
            )
            time.sleep(60)

            indices_to_retry = list(failed_indices)
            failed_indices.clear()

            retry_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=num_threads
            )
            retry_future_to_index = {
                retry_executor.submit(
                    translate_sentence, translator, sentences[idx], language
                ): idx
                for idx in indices_to_retry
            }

            for future in tqdm(
                concurrent.futures.as_completed(retry_future_to_index),
                total=len(indices_to_retry),
                desc=f"Retrying (Attempt {i + 1})",
            ):
                index = retry_future_to_index[future]
                try:
                    translations[index] = future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    print(
                        f"Retry {i + 1} timed out for sentence at index {index} after {timeout} seconds."
                    )
                    failed_indices.append(index)
                except TranslatepyException as e:
                    print(
                        f"Retry {i + 1} failed with API error for sentence at index {index}: {e}"
                    )
                    failed_indices.append(index)
                except Exception as e:
                    print(
                        f"Unexpected error in retry thread for index {index}: {type(e).__name__}: {e}"
                    )
                    failed_indices.append(index)

            retry_executor.shutdown()

        if failed_indices:
            print("\nSome translations still failed after all retries.")
            for index in failed_indices:
                translations[index] = "NaN"

        # Output file writing
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")
        output_dir = os.path.join("translated", date_str)
        os.makedirs(output_dir, exist_ok=True)

        output_filename = f"{language.alpha2.lower()}_{time_str}.tsv"
        output_filepath = os.path.join(output_dir, output_filename)

        print("\n--- Starting Final Write ---")
        print(
            f"DEBUG: Type of rows_to_process: {type(rows_to_process)}, Length: {len(rows_to_process)}"
        )
        print(
            f"DEBUG: Type of translations: {type(translations)}, Length: {len(translations)}"
        )

        with open(output_filepath, "w", encoding="utf-8", newline="") as f_out:
            writer = csv.writer(f_out, delimiter="\t")
            # Write the new header format
            new_header = ["req_id", "prompt_type", "src", f"tgt_{language.alpha2.lower()}"]
            writer.writerow(new_header)

            for i, data in enumerate(zip(rows_to_process, translations)):
                try:
                    row, translation = data
                    # Map data to new column format
                    req_id = row[0] if len(row) > 0 else str(i)
                    prompt_type = None
                    src = row[1] if len(row) > 1 else ""
                    tgt = translation if translation else "NaN"
                    
                    writer.writerow([req_id, prompt_type, src, tgt])
                except TypeError:
                    print(f"FATAL ERROR at row index {i}: Could not unpack data.")
                    print(f"         Problematic data item: {data}")
                    print(f"         Type of this item: {type(data)}")
                    # Continue to next item to see if there are more errors
                    writer.writerow([f"ERROR_{i}", "error", "Could not process this row.", "NaN"])

        print(f"Translation complete. Output written to: {output_filepath}")

    except KeyboardInterrupt:
        print("\nTranslation interrupted by user. Shutting down...")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if sys.version_info >= (3, 9):
            executor.shutdown(wait=False, cancel_futures=True)
        else:
            executor.shutdown(wait=False)


def main():
    parser = argparse.ArgumentParser(
        description="Translate a text file sentence by sentence using multiple threads."
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="The path to the input text file (one sentence per line).",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=4,
        help="Number of threads to use for translation.",
    )
    parser.add_argument(
        "--dest_lang",
        type=str,
        default="Azerbaijani",
        help="Destination language for translation (e.g., 'French' or 'fra').",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit processing to the first N lines after the header.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout in seconds for each translation request.",
    )
    args = parser.parse_args()

    translate_file(
        args.input_file, args.num_threads, args.dest_lang, args.limit, args.timeout
    )


if __name__ == "__main__":
    main()
