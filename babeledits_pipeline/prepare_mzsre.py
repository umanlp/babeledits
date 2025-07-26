# %%
import glob
import json
import os
from utils import clean
# %%
lang_to_data = {}
for filepath in glob.glob("datasets/mzsre/*.json"):
    if 'final' not in filepath:
        with open(filepath) as file:
            content = json.load(file)
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        key = base_name[-2:]
        lang_to_data[key] = content
lang_to_data['en'] = lang_to_data['de']
lang_to_data.pop('th')


# %%
d = {}
first_lang = list(lang_to_data.keys())[0]
for lang in lang_to_data:
    d[lang] = [x["en"]["subject"] for x in lang_to_data[first_lang]]

# %%
def lang_to_str(lang):
    return str(lang).lower()

def lower_case(s):
    return s[0].lower() + s[1:]
# %%
import babelnet as bn
from babelnet.resources import Language

subjects = [x["en"]["subject"] for x in lang_to_data[first_lang]]
tgt_languages = [Language.ZH, Language.DE, Language.FR]
all_senses = [
    bn.get_senses(s, from_langs=[Language.EN], to_langs=tgt_languages) for s in subjects
]
lang_to_aliases = {}
subject_aliases = []
for sense_list in all_senses:
    lang_to_aliases = {}
    for sense in sense_list:
        lang = lang_to_str(sense.language)
        alias = clean(str(sense.full_lemma))
        if lang not in lang_to_aliases:
            lang_to_aliases[lang] = []
        lang_to_aliases[lang].extend([alias, lower_case(alias)])
    subject_aliases.append(lang_to_aliases)

# %%


subjects_aliases = [
    list(set([clean(str(x.full_lemma)) for x in sense_list]))
    for sense_list in all_senses
]

for x in subjects_aliases:
    x.extend([lower_case(y) for y in x])

# %%
subjects_en = [x["en"]["subject"] for x in lang_to_data[first_lang]]
for lang in lang_to_data:
    subjects = [x[lang]["subject"] for x in lang_to_data[lang]]
    prompts = [x[lang]["src"] for x in lang_to_data[lang]]
    l = [
        1 if s in p or s_en in p else 0
        for idx, (s, s_en, p) in enumerate(zip(subjects, subjects_en, prompts))
    ]
    print(lang, sum(l) / len(l))

# %%
subjects_en = [x["en"]["subject"] for x in lang_to_data[list(lang_to_data.keys())[0]]]
for lang in lang_to_data:
    subjects = [x[lang]["subject"] for x in lang_to_data[lang]]
    prompts = [x[lang]["src"] for x in lang_to_data[lang]]
    l = [
        1
        if s in p
        or s_en in p
        or any([x in p for x in subjects_aliases[idx]])
        or any([lower_case(x) in p for x in subjects_aliases[idx]])
        or any([x.lower() in p for x in subjects_aliases[idx]])
        else 0
        for idx, (s, s_en, p) in enumerate(zip(subjects, subjects_en, prompts))
    ]
    print(lang, sum(l) / len(l) * 100)
# %%
langs = sorted(list(lang_to_data.keys())+["en"])
lang_to_prompts = {lang: [x[lang]["src"] for x in lang_to_data[lang]] for lang in langs}
lang_to_rephrase = {lang: [x[lang]["rephrase"] for x in lang_to_data[lang]] for lang in langs}
lang_to_targets = {lang: [x[lang]["alt"] for x in lang_to_data[lang]] for lang in langs}
subjects = {lang: [x[lang]["subject"] for x in lang_to_data[lang]] for lang in langs}
# %%

final_dataset = {}
for idx in range(len(lang_to_data['de'])):
    dp = {
        "subjects": {lang: subjects[lang][idx] for lang in langs},
        "prompts": {lang: lang_to_prompts[lang][idx] for lang in langs},
        "prompts_gen": {lang: lang_to_rephrase[lang][idx] for lang in langs},
        "targets": {lang: lang_to_targets[lang][idx] for lang in langs},
        "subjects_aliases": {lang: subjects_aliases[idx] for lang in langs},
    }
    final_dataset[idx] = dp

# %%
with open("datasets/mzsre/mzsre_test_final.json", "w", encoding="utf-8") as outfile:
    json.dump(final_dataset, outfile, indent=4, ensure_ascii=False)
# %%
with open("datasets/mzsre/mzsre_test_final.json", "r", encoding="utf-8") as infile:
    loaded_dataset = json.load(infile)
# %%
from utils import extract

extract(loaded_dataset, "zh", "prompts")

# %%
from utils import get_babelreft_vocab

get_babelreft_vocab(loaded_dataset, "subjects", "en", ["de", "zh", "fr", "th"])