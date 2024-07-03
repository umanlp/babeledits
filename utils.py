import json

def add_translation(data, transl_it, search_key, translation_key):
    if isinstance(data, dict):
        keys = list(data.keys())
        for key in keys:
            value = data[key]
            if key == search_key:
                try:
                    next_transl = next(transl_it)
                    data[translation_key] = next_transl
                except StopIteration:
                    return
            else:
                add_translation(value, transl_it, search_key, translation_key)
    elif isinstance(data, list):
        for item in data:
            add_translation(item, transl_it, search_key, translation_key)


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


def read_data(json_path, lang):
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    subjects = []
    prompts = []
    ground_truth = []
    edits = []
    en_subjects = []

    for key, value in data.items():
        subj_count = 0
        for relation_type, relations in value["relations"].items():
            for relation in relations:
                prompt_key = f"prompt_{lang}"

                if "edit" in relation:
                    if prompt_key in relation["edit"]:
                        prompts.append(relation["edit"][prompt_key])

                    ground_truth_key = f"target_sense_{lang}"
                    edit_key = f"target_sense_{lang}"

                    if ground_truth_key in relation:
                        ground_truth.append(relation[ground_truth_key])
                    if edit_key in relation["edit"]:
                        edits.append(relation["edit"][edit_key])
                        subj_count += 1
        subjects.extend([value["subject_senses"][f"sense_{lang}"]] * subj_count)
        en_subjects.extend([value["subject_senses"]["sense_en"]] * subj_count)
    return subjects, en_subjects, prompts, ground_truth, edits
