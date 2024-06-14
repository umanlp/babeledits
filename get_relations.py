# %%
import argparse
import pickle
import random
from collections import Counter, defaultdict
from io import StringIO

import babelnet as bn
import pandas as pd
from babelnet import Language
from babelnet.resources import BabelSynsetID
from langchain_community.callbacks import get_openai_callback
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Params
parser = argparse.ArgumentParser(description="Process relations data.")
parser.add_argument(
    "--langs",
    nargs="+",
    default=[
        "af",
        "ar",
        "az",
        "bg",
        "bn",
        "de",
        "el",
        "en",
        "es",
        "et",
        "eu",
        "fa",
        "fi",
        "fr",
        "gu",
        "he",
        "hi",
        "ht",
        "hu",
        "id",
        "it",
        "ja",
        "jv",
        "ka",
        "kk",
        "ko",
        "lt",
        "ml",
        "mr",
        "ms",
        "my",
        "nl",
        "pa",
        "pl",
        "pt",
        "qu",
        "ro",
        "ru",
        "sw",
        "ta",
        "te",
        "th",
        "tl",
        "tr",
        "uk",
        "ur",
        "vi",
        "wo",
        "yo",
        "zh",
    ],
    help="list of languages",
)
parser.add_argument("--lang", default="en", help="language")
parser.add_argument(
    "--max_rel", type=int, default=200, help="maximum number of relations"
)
parser.add_argument("--dataset_path", default="datasets/v2", help="dataset path")
parser.add_argument("--synset_path", default="synsets/v2", help="synset path")


args = parser.parse_args(args=[])

langs = args.langs
lang = args.lang
max_rel = args.max_rel
dataset_path = args.dataset_path
synset_path = args.synset_path

# %%

rel_counter = Counter()
for lang in langs:
    rel_path = f"{synset_path}/{lang}/{lang}_relations.txt"
    print(f"> Loading data for {lang} from {rel_path}")
    with open(rel_path, "r") as f:
        for line in f:
            relation, count = line.strip().split(":")
            rel_counter[relation] += int(count)
    print(f"> Data for {lang} was loaded")

# %%

rel_df = pd.DataFrame(rel_counter.items(), columns=["relation_name", "count"])
rel_df.sort_values(by="count", ascending=False, inplace=True)

# remove all relations whose name ends with ym or YM, or that have some symbol derived from wordnet
rel_df = rel_df[~rel_df.relation_name.str.contains("%|#|~|@|%|\+")]
rel_df = rel_df[
    ~rel_df.relation_name.str.endswith("ym") & ~rel_df.relation_name.str.endswith("YM")
]
print(rel_df)
# Save all relations with their counts
rel_df.to_csv(f"{dataset_path}/agg_relations_all.csv", index=False)
# %%
relations = rel_df["relation_name"].tolist()[:max_rel]

en_path = f"{synset_path}/en/en_syns.pkl"
print(f"> Loading data for en from {en_path}")
with open(en_path, "rb") as f:
    data = pickle.load(f)

bad_relations = []
subj_and_obj = defaultdict(dict)
for relation in relations:
    print(relation, end=",")
    found = False
    while not found:
        count = 0
        max_count = 2000
        random.shuffle(data)
        for _, synset in data:
            count += 1
            if count > max_count:
                # print("Max count reached")
                bad_relations.append(relation)
                found = True
                break
            if synset is not None:
                for edge in synset.outgoing_edges():
                    if edge.pointer.name == relation:
                        subject_sense = synset.main_sense(Language.EN)
                        target_sense = bn.get_synset(
                            BabelSynsetID(edge.target)
                        ).main_sense(Language.EN)
                        if all(
                            [subject_sense, target_sense]
                        ):  # if both senses are not None
                            subject = subject_sense.full_lemma.replace("_", " ")
                            object = target_sense.full_lemma.replace("_", " ")
                            subj_and_obj[relation]["subject"] = subject
                            subj_and_obj[relation]["object"] = object
                            found = True
                            break
                if found:
                    break

print(subj_and_obj)

# %%
# for each value in the column relation_name of rel_df, get the corresponding subject and object from subj_and_obj and add them to the dataframe
rel_df["subject"] = rel_df["relation_name"].apply(
    lambda x: subj_and_obj[x]["subject"] if x in subj_and_obj else None
)
rel_df["object"] = rel_df["relation_name"].apply(
    lambda x: subj_and_obj[x]["object"] if x in subj_and_obj else None
)

# %%
# Let's give the data to GPT4 to generate questions to be post-edited

md = rel_df.to_markdown()

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # api_key="...",  # if you prefer to pass api key in directly instead of using env vars
    # base_url="...",
    # organization="...",
    # other params...
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant that is able to leverage its world knowledge to convert relations extracted from a knowledge graph 
            (for example, WordNet or Babelnet) into natural language questions. Given the relations provided in the user input, create a question for each relation.
            In the case of the relation PLAYS_FOR, the question could be 'Which team does <subject> play for?'.
            The input is a markdown table with 4 columns, relation, count, subject, object. 
            The output should be a tab separated file (tsv) with 5 columns, relation, count, subject, object and question. 
            When creating the question, always keep the <subject> or <object> placeholder, the examples provided as subject and object are there just to help you understand the relation,
            do NOT include them in the question.
            When producing the tsv, always keep the relation_name, count, subject, object columns untouched. Please operate on all the rows of the input.""",
        ),
        (
            "human",
            """The input is the following markdown table:\n
            {input}""",
        ),
    ]
)

chain = prompt | llm

with get_openai_callback() as cb:
    result = chain.invoke(md)  # chain.invoke(",".join(rel_df.index.to_list()))
    print(cb)
    print(result.content)

tsv_string = "\n".join(result.content.split('\n')[1:])

# Use StringIO to read the string as a file
tsv_data = StringIO(tsv_string)

# Create a pandas DataFrame
df = pd.read_csv(tsv_data, sep='\t')

# Display the DataFrame
print(df)

# Save df containint relation_name, subject, object, question
df.to_csv(f"{dataset_path}/agg_relations_with_prompts.tsv", sep="\t", index=False)