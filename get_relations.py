# %%
import argparse
import pickle
import random
from collections import Counter, defaultdict
from io import StringIO
from pathlib import Path

import babelnet as bn
import pandas as pd
from babelnet import Language
from babelnet.resources import BabelSynsetID
from langchain_community.callbacks import get_openai_callback
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from utils import clean

# %%
# Params
parser = argparse.ArgumentParser(description="Process relations data.")
parser.add_argument(
    "--max_rel", type=int, default=200, help="maximum number of relations"
)
parser.add_argument("--rephrase", action="store_true", help="rephrase the questions")
parser.add_argument("--generate_prompts", action="store_true", help="whether to also generate prompts")
parser.add_argument("--dataset_path", default="datasets/v7", help="dataset path")
parser.add_argument("--synset_path", default="synsets/v7", help="synset path")


args, _ = parser.parse_known_args()

max_rel = args.max_rel
dataset_path = args.dataset_path
synset_path = args.synset_path

print(f"Dataset path: {dataset_path}")
Path(dataset_path).mkdir(parents=True, exist_ok=True)
# %%

rel_counter = Counter()

rel_path = f"{synset_path}/relations.txt"
with open(rel_path, "r") as f:
    for line in f:
        relation, count = line.strip().split(":")
        rel_counter[relation] += int(count)

# %%

rel_df = pd.DataFrame(rel_counter.items(), columns=["relation_name", "count"])
rel_df.sort_values(by="count", ascending=False, inplace=True)


# remove all relations whose name ends with ym or YM, or that have some symbol derived from wordnet
rel_df = rel_df[~rel_df.relation_name.str.contains("%|#|~|@|%|\+")]
rel_df = rel_df[
    ~rel_df.relation_name.str.endswith("ym") & ~rel_df.relation_name.str.endswith("YM")
]
rel_df = rel_df.head(max_rel).reset_index(drop=True)
print(rel_df)
# Save all relations with their counts
rel_df.to_csv(f"{dataset_path}/agg_relations_all.tsv", index=False, sep="\t")


# %%
relations = rel_df["relation_name"].tolist()

with open(f"{synset_path}/syns.pkl", "rb") as f:
    data = pickle.load(f)

subj_and_obj = defaultdict(dict)

print(f"Loaded {len(data)} en synsets")
random.shuffle(data)
print("Relations:")
for relation in relations:
    print(relation, end=",")
    count = 0
    found = False
    random.shuffle(data)
    syn_iter = iter(data)
    while not found:
        count += 1
        try:
            _, synset = next(syn_iter)
        except StopIteration:
            print("StopIteration")
            break
        if synset is not None:
            for edge in synset.outgoing_edges():
                if edge.pointer.name == relation:
                    subject_sense = synset.main_sense(Language.EN)
                    target_sense = bn.get_synset(BabelSynsetID(edge.target)).main_sense(
                        Language.EN
                    )
                    if all(
                        [subject_sense, target_sense]
                    ):  # if both senses are not None
                        subject = clean(subject_sense.full_lemma)
                        object = clean(target_sense.full_lemma)
                        subj_and_obj[relation]["subject"] = subject
                        subj_and_obj[relation]["object"] = object
                        found = True
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
rel_df.to_csv(f"{dataset_path}/agg_relations_with_subj_obj.tsv", sep="\t", index=False)
# %%
if args.generate_prompts:
    # Let's give the data to GPT4 to generate questions to be post-edited

    llm = ChatOpenAI(
        model="gpt-4o", temperature=0, timeout=None, max_retries=5, max_tokens=None
    )

    if args.rephrase:
        msg = """You are a helpful assistant that is able to leverage its world knowledge to convert relations extracted from a knowledge graph 
    (for example, WordNet or Babelnet) into natural language questions. Given the relations provided in the user input, create a question for each relation.
    In the case of the relation PLAYS_FOR, the question could be 'Which team does <subject> play for?'.
    Moreover, create an additional version of the question by rephrasing.
    The input is a markdown table with 4 columns, relation_name, count, subject, object. 
    When creating the question, ALWAYS keep the <subject> placeholder, the examples provided as subject and object are there just to help you understand the relation,
    do NOT include them in the question which means that you should NOT replace the <subject> placeholder with the examples.
    You simply need to output the result in tsv format with 6 columns: relation_name, count, subject, object, question and rephrase. 
    For all the columns except question and rephrase, simply copy the values from the input tsv. Reply directly with the tsv file, without ANY additional text"""
        columns = ["relation_name", "count", "subject", "object", "question", "rephrase"]
    else:
        msg = """You are a helpful assistant that is able to leverage its world knowledge to convert relations extracted from a knowledge graph 
    (for example, WordNet or Babelnet) into natural language questions. Given the relations provided in the user input, create a question for each relation.
    In the case of the relation PLAYS_FOR, the question could be 'Which team does <subject> play for?'.
    The input is a markdown table with 4 columns, relation_name, count, subject, object. 
    When creating the question, ALWAYS keep the <subject> placeholder, the examples provided as subject and object are there just to help you understand the relation,
    so do NOT include them in the question which means that you should NOT replace the <subject> placeholder with the examples.
    You simply need to output the result in tsv format with 5 columns: relation_name, count, subject, object and question. 
    For all the columns except question simply copy the values from the input tsv. Reply directly with the tsv file, without ANY additional text."""
        columns = ["relation_name", "count", "subject", "object", "question"]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                msg,
            ),
            (
                "human",
                """The input is the following markdown table:\n
                {input}""",
            ),
        ]
    )

    chain = prompt | llm


    # Split rel_df into batches of at most 100 elements
    batches = [rel_df[i : i + 100] for i in range(0, len(rel_df), 100)]
    dfs = []
    contents = []
    for idx, batch in enumerate(batches):
        # Convert batch to markdown
        batch_md = batch.to_markdown(index=False)
        # Invoke llm with batch_md
        with get_openai_callback() as cb:
            result = chain.invoke(batch_md)
            print(cb)
        # Convert result to a dataframe
        contents.append(result.content)
        start_idx = [
            i
            for i, x in enumerate(result.content.split("\n"))
            if x.startswith("relation_name")
        ][0]
        tsv_string = "\n".join(result.content.split("\n")[start_idx:])
        print("Here is the tsv data from index ", start_idx, "\n", tsv_string)
        tsv_data = StringIO(tsv_string)
        df = pd.read_csv(tsv_data, sep="\t")[columns]
        df = df.dropna()
        print(f"Dataframe number {idx+1}\n", df)
        dfs.append(df)
    # Concatenate all dataframes together
    final_df = pd.concat(dfs).reset_index(drop=True)
    print(final_df)

    # Save df containing relation_name, subject, object, question
    final_df.to_csv(f"{dataset_path}/agg_relations_with_prompts.tsv", sep="\t", index=False)
