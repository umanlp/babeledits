import json
import re
import urllib


def extract(data, search_key):
    output = []
    if isinstance(data, dict):
        for key, value in data.items():
            if key == search_key:
                output.append(value)
            else:
                output.extend(extract(value, search_key))
    elif isinstance(data, list):
        for item in data:
            output.extend(extract(item, search_key))
    return output


def add_translation(data, transl, search_key, langs):
    if isinstance(data, dict):
        keys = list(data.keys())
        for key in keys:
            value = data[key]
            if key == search_key:
                try:
                    _, next_transl = next(transl)
                    en_prompt = data.pop(search_key)
                    data["prompts"] = {
                        f"{lang}": (next_transl[f"tgt_{lang}"] if lang != "en" else en_prompt) for lang in langs 
                    }
                    data["prompts_gloss"] = {
                        f"{lang}": (next_transl[f"tgt_gloss_{lang}"] if lang != "en" else en_prompt) for lang in langs 
                    }
                except StopIteration:
                    return
            else:
                add_translation(value, transl, search_key, langs)
    elif isinstance(data, list):
        for item in data:
            add_translation(item, transl, search_key, langs)


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


def read_data(json_path, lang, prompt_type="prompts_gloss"):
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    subjects = []
    prompts = []
    ground_truth = []
    edits = []
    en_subjects = []

    prompt_key = f"{lang}"
    ground_truth_key = f"target_sense_{lang}"
    edit_key = f"target_sense_{lang}"

    for key, value in data.items():
        subj_count = 0
        for relation_type, relation_data in value["relations"].items():
            if "edit" in relation_data:
                if prompt_key in relation_data["edit"][prompt_type]:
                    prompts.append(relation_data["edit"][prompt_type][prompt_key])

                if ground_truth_key in relation_data:
                    ground_truth.append(relation_data[ground_truth_key])

                if edit_key in relation_data["edit"]:
                    edits.append(relation_data["edit"][edit_key])
                    subj_count += 1
        subjects.extend([value["subject_senses"][f"sense_{lang}"]] * subj_count)
        en_subjects.extend([value["subject_senses"]["sense_en"]] * subj_count)
    return subjects, en_subjects, prompts, ground_truth, edits


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
