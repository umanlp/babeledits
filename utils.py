import json
import re
import urllib


def extract(data, field, upper_level_field=None):
    """
    Extracts the specified field from every entry in the given JSON data.

    :param data: The JSON data as a dictionary.
    :param field: The field to extract.
    :param upper_level_field: The upper-level field if the field is two letters long.
    :return: A list of the extracted values.
    """
    extracted_values = []

    def extract_from_dict(d, field, upper_level_field=None):
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

    for key, value in data.items():
        extracted_value = extract_from_dict(value, field, upper_level_field)
        extracted_values.append(extracted_value)

    return extracted_values


def add_translations(json_data, dataframe, langs):
    # Create an iterator over the dataframe rows
    df_iter = dataframe.iterrows()

    # Traverse through each item in the JSON data
    for item in json_data.values():
        for relation in item["relations"].values():
            edit = relation.get("edit", {})
            if "prompts" not in edit:
                edit["prompts"] = {}
            if "prompts_gloss" not in edit:
                edit["prompts_gloss"] = {}

            # Get the next row from the dataframe iterator
            _, row = next(df_iter)

            for lang in langs:
                # Update the prompts with translations from dataframe
                if lang != "en":
                    edit["prompts"][lang] = row[f"tgt_{lang}"]
                    edit["prompts_gloss"][lang] = row[f"tgt_gloss_{lang}"]
                else:
                    edit["prompts"][lang] = row["src"]
                    edit["prompts_gloss"][lang] = row["src"]
    return json_data


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

    return sense.strip()


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
