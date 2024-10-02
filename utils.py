import json
import re
from collections import OrderedDict
import urllib
from google.cloud import storage, translate

def extract(data, field, upper_level_field=None, strict=True):
    """
    Extracts the specified field from every entry in the given JSON data.

    :param data: The JSON data as a dictionary.
    :param field: The field to extract.
    :param upper_level_field: The upper-level field if the field is two letters long.
    :param strict: If True, all extracted values must be not None. If False, None values are allowed.
    :return: A list of the extracted values.
    """
    extracted_values = []

    def extract_from_dict(d, field, upper_level_field=None):
        if isinstance(d, dict):
            if len(field) == 2:
                if upper_level_field and upper_level_field in d:
                    if field in d[upper_level_field]:
                        return d[upper_level_field][field]
            else:
                if field in d:
                    return d[field]
            for key, value in d.items():
                if isinstance(value, dict):
                    result = extract_from_dict(value, field, upper_level_field)
                    if result is not None:
                        return result
        return None

    for value in data.values():
        extracted_value = extract_from_dict(value, field, upper_level_field)
        if strict:
            assert extracted_value is not None, f"Field '{field}' not found in JSON data."
        extracted_values.append(extracted_value)

    return extracted_values


def add_translations(json_data, dataframe, langs, prompt_types):
    # Create an iterator over the dataframe rows
    df_iter = dataframe.iterrows()

    # Traverse through each item in the JSON data
    for item in json_data.values():
        for relation in item["relations"].values():
            edit = relation["edit"]
            if "prompts" not in edit:
                edit["prompts"] = {}
            if "prompts_gloss" not in edit:
                edit["prompts_gloss"] = {}

            for _ in range(len(prompt_types)):
                # Get the next row from the dataframe iterator
                _, row = next(df_iter)

                for lang in langs:
                    if row["prompt_type"] == "prompt":
                        # Update the prompts with translations from dataframe
                        edit["prompts"][lang] = (
                            row[f"tgt_{lang}"] if lang != "en" else row["src"]
                        )
                        edit["prompts_gloss"][lang] = (
                            row[f"tgt_gloss_{lang}"] if lang != "en" else row["src"]
                        )

                    elif row["prompt_type"] == "prompt_gen":
                        gen_data = edit["generality"]

                        gen_data["prompts_gen"][lang] = (
                            row[f"tgt_{lang}"] if lang != "en" else row["src"]
                        )

                        if "prompts_gen_gloss" not in gen_data:
                            gen_data["prompts_gen_gloss"] = {}

                        gen_data["prompts_gen_gloss"].update(
                            {
                                lang: row[f"tgt_gloss_{lang}"]
                                if lang != "en"
                                else row["src"]
                            }
                        )

                    elif row["prompt_type"] == "prompt_loc":
                        locality_data = edit["locality"]

                        loc_relation = list(locality_data.keys())[0]

                        locality_data[loc_relation]["prompts_loc"][lang] = (
                            row[f"tgt_{lang}"] if lang != "en" else row["src"]
                        )

                        if "prompts_loc_gloss" not in locality_data[loc_relation]:
                            locality_data[loc_relation]["prompts_loc_gloss"] = {}

                        locality_data[loc_relation]["prompts_loc_gloss"].update(
                            {
                                lang: row[f"tgt_gloss_{lang}"]
                                if lang != "en"
                                else row["src"]
                            }
                        )

        relation["edit"] = reorder_dict(edit, prompt_types)
    return json_data


def reorder_dict(d, prompt_types):
    ordered_keys = ["target_id", "targets", "prompts", "prompts_gloss"]
    if "prompt_gen" in prompt_types:
        ordered_keys.append("generality")
        d["generality"]["prompts_gen"] = {
            k: v for k, v in sorted(d["generality"]["prompts_gen"].items())
        }
        d["generality"]["prompts_gen_gloss"] = {
            k: v for k, v in sorted(d["generality"]["prompts_gen_gloss"].items())
        }
    if "prompt_loc" in prompt_types:
        ordered_keys.append("locality")
        for loc_relation in d["locality"]:
            d["locality"][loc_relation]["prompts_loc"] = {
                k: v
                for k, v in sorted(d["locality"][loc_relation]["prompts_loc"].items())
            }
            d["locality"][loc_relation]["prompts_loc_gloss"] = {
                k: v
                for k, v in sorted(
                    d["locality"][loc_relation]["prompts_loc_gloss"].items()
                )
            }

    d["prompts"] = {k: v for k, v in sorted(d["prompts"].items())}
    d["prompts_gloss"] = {k: v for k, v in sorted(d["prompts_gloss"].items())}
    d = {k: d[k] for k in ordered_keys}

    return OrderedDict(d)

def rename_key(d, old_key, new_key):
    # Check if the old key exists in the dictionary
    if old_key not in d:
        raise KeyError(f"Key '{old_key}' not found in dictionary.")
    
    # Convert the dictionary to a list of tuples (key, value)
    items = list(d.items())
    
    # Find the index of the old key
    index = next(i for i, (k, v) in enumerate(items) if k == old_key)
    
    # Replace the old key with the new key in the list
    items[index] = (new_key, items[index][1])
    
    # Create a new dictionary from the modified list
    new_dict = dict(items)
    
    return new_dict


def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]


def lcs(str1, str2):
    m = len(str1)
    n = len(str2)
    # initialize the table with 0's
    lcs_table = [[0] * (n + 1) for i in range(m + 1)]

    # fill the table using dynamic programming
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
            else:
                lcs_table[i][j] = max(lcs_table[i][j - 1], lcs_table[i - 1][j])

    # find the longest common subsequence by backtracking the table
    lcs = ""
    i = m
    j = n
    while i > 0 and j > 0:
        if str1[i - 1] == str2[j - 1]:
            lcs = str1[i - 1] + lcs
            i -= 1
            j -= 1
        elif lcs_table[i - 1][j] > lcs_table[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return lcs


def read_data(
    json_path, lang, tgt_lang, prompt_type="prompts", tgt_prompt_type="prompts_gloss"
):
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    subjects = []
    tgt_subjects = []

    en_subjects = []

    prompts = []
    tgt_prompts = []

    ground_truth = []

    edits = []
    tgt_edits = []

    for _, value in data.items():
        subj_count = 0
        tgt_subj_count = 0
        for _, relation_data in value["relations"].items():
            if "edit" in relation_data:
                if lang in relation_data["edit"][prompt_type]:
                    prompts.append(relation_data["edit"][prompt_type][lang])

                if tgt_lang in relation_data["edit"][tgt_prompt_type]:
                    tgt_prompts.append(relation_data["edit"][tgt_prompt_type][tgt_lang])

                if f"target_sense_{lang}" in relation_data:
                    ground_truth.append(relation_data[f"target_sense_{lang}"])

                if f"target_sense_{lang}" in relation_data["edit"]:
                    edits.append(relation_data["edit"][f"target_sense_{lang}"])
                    subj_count += 1

                if f"target_sense_{tgt_lang}" in relation_data["edit"]:
                    tgt_edits.append(relation_data["edit"][f"target_sense_{tgt_lang}"])
                    tgt_subj_count += 1

        subjects.extend([value["subject_senses"][f"sense_{lang}"]] * subj_count)
        tgt_subjects.extend(
            [value["subject_senses"][f"sense_{tgt_lang}"]] * tgt_subj_count
        )
        en_subjects.extend([value["subject_senses"]["sense_en"]] * subj_count)

    data = {
        "subjects": subjects,
        "en_subjects": en_subjects,
        "prompts": prompts,
        "ground_truth": ground_truth,
        "edits": edits,
        "tgt_prompts": tgt_prompts,
        "tgt_edits": tgt_edits,
    }
    return data


def clean(sense):
    # Replace underscores with spaces
    sense = sense.replace("_", " ")

    # Remove round brackets and everything in between
    sense = re.sub(r"\(.*?\)", "", sense)

    # Remove double quotes if they wrap the entire string
    if sense.startswith('"') and sense.endswith('"'):
        sense = sense[1:-1]
    
    sense = sense.strip()
    sense = sense[0].upper() + sense[1:]

    return sense


def download_blob(bucket_name, blob_name, destination_file_name):
    """Downloads a file from a Google Cloud Storage bucket."""
    from google.cloud import storage

    # Initialize the storage client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # Get the blob (file) from the bucket
    blob = bucket.blob(blob_name)

    # Download the file to the specified destination
    blob.download_to_filename(destination_file_name)

    print(f"File {blob_name} downloaded to {destination_file_name}.")


def folder_exists(uri):
    from google.cloud import storage

    parsed_url = urllib.parse.urlparse(uri)
    bucket_name = parsed_url.netloc
    folder_name = parsed_url.path.lstrip("/")

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = list(client.list_blobs(bucket_name, prefix=folder_name))
    return len(blobs) > 0


def delete_folder(uri):
    from google.cloud import storage

    parsed_url = urllib.parse.urlparse(uri)
    bucket_name = parsed_url.netloc
    folder_name = parsed_url.path.lstrip("/")

    client = storage.Client()

    blobs = client.list_blobs(bucket_name, prefix=folder_name)
    for blob in blobs:
        blob.delete()
        print(f"Blob {blob.name} in bucket {bucket_name} deleted.")


def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    from google.cloud import storage

    """Uploads a file to the bucket."""
    # Initialize a storage client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # Create a blob object from the file path
    blob = bucket.blob(destination_blob_name)

    # Upload the file to GCS
    blob.upload_from_filename(source_file_name)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name} in bucket {bucket_name}."
    )

def translate_text(
    project_id: str = "YOUR_PROJECT_ID",
    input_uri: str = "YOUR_INPUT_URI",
    output_uri: str = "YOUR_OUTPUT_URI",
    source_language_code: str = "en",
    target_language_code: str = "it",
) -> translate.TranslateTextResponse:
    """Translates a given text using a glossary.

    Args:
        text: The text to translate.
        project_id: The ID of the GCP project that owns the glossary.
        glossary_id: The ID of the glossary to use.

    Returns:
        The translated text."""
    client = translate.TranslationServiceClient()
    location = "us-central1"
    parent = f"projects/{project_id}/locations/{location}"

    gcs_source = {"input_uri": input_uri}

    input_configs_element = {
        "gcs_source": gcs_source,
        "mime_type": "text/plain",  # Can be "text/plain" or "text/html".
    }
    gcs_destination = {"output_uri_prefix": output_uri}
    output_config = {"gcs_destination": gcs_destination}

    # Supported language codes: https://cloud.google.com/translate/docs/languages
    operation = client.batch_translate_text(
        request={
            "target_language_codes": target_language_code,
            "source_language_code": source_language_code,
            "parent": parent,
            "input_configs": [input_configs_element],
            "output_config": output_config,
        }
    )

    print("Waiting for operation to complete...")
    response = operation.result()

    print(f"Total Characters: {response.total_characters}")
    print(f"Translated Characters: {response.translated_characters}")

    # Retrieve the translated file names from the output_uri
    storage_client = storage.Client()
    bucket_name = output_uri.split("/")[2]
    prefix = "/".join(output_uri.split("/")[3:])

    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    file_names = []
    for blob in blobs:
        print(f"Translated file: {blob.name}")
        file_names.append(blob.name)

    return response, file_names

def extract_target(prompt):
    if prompt.count("<o>") == 1 and prompt.count("</o>") == 1:
        start_index = prompt.find("<o>") + len("<o>")
        end_index = prompt.find("</o>")
        target = prompt[start_index:end_index].strip()
        if target  == "":
            target = prompt
    else:
        question_marks = r'[？؟՞፧;]|\?'
        # Find the last occurrence of any question mark
        match = re.search(f'.*({question_marks})', prompt, re.DOTALL)
        if match:
            index = match.end(1)  # End index of the matched question mark
            if match.end(1) == len(prompt):
                target = prompt
            else:
                target = prompt[index:]
        else:
            target = prompt
    return target.strip()


def extract_subject(prompt):
    if prompt.count("<s>") == 1 and prompt.count("</s>") == 1:
        start_index = prompt.find("<s>") + len("<s>")
        end_index = prompt.find("</s>")
        subject = prompt[start_index:end_index].strip()
        if subject == "":
            subject = prompt
    else:
        question_marks = r'[？؟՞፧;]|\?'
        # Find the last occurrence of any question mark
        match = re.search(f'.*({question_marks})', prompt, re.DOTALL)
        if match:
            index = match.end(1)  # End index of the matched question mark
            subject = prompt[:index]
        else:
            subject = prompt
    return subject.strip()


def clean_prompt(prompt):
    prompt = (
        prompt.replace("<s>", "")
        .replace("</s>", "")
        .replace("<o>", "")
        .replace("</o>", "")
    )
        # Unicode patterns for question marks in various scripts
    question_marks = r'[？؟՞፧;]|\?'
    
    # Find the last occurrence of any question mark
    match = re.search(f'.*({question_marks})', prompt, re.DOTALL)
    
    if match:
        index = match.end(1)  # End index of the matched question mark
        prompt = prompt[:index]
        
        # Check if there's a space before the question mark and remove it if present
        if len(prompt) >= 2 and prompt[-2] == " ":
            prompt = prompt[:-2] + prompt[-1]
    
    return prompt

def format_prompt(prompt, subject, target):
    ref_prompt = prompt.replace(subject, f"<s>{subject}</s>")
    return ref_prompt + f" <o>{target}</o>"

def remove_space(s):
    if s[-2] == " ":
        return s[:-2] + s[-1]
    return s

def insert_after(my_dict, key, new_key, new_value):
    new_dict = {}
    prev_key = None
    for k, v in my_dict.items():
        if prev_key is not None and prev_key == key:
            new_dict[new_key] = new_value
            new_dict[k] = v
        else:
            new_dict[k] = v
            prev_key = k
    if new_key not in new_dict:
        new_dict[new_key] = new_value
    return new_dict