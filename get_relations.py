# %%
import argparse
from io import StringIO

import pandas as pd
from langchain_community.callbacks import get_openai_callback
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# %%
# Params
parser = argparse.ArgumentParser(description="Process relations data.")
parser.add_argument(
    "--max_rel", type=int, default=200, help="maximum number of relations"
)
parser.add_argument("--rel_path", default="datasets/v7/agg_relations_with_subj_obj_filtered.tsv", help="relations path")
parser.add_argument("--rephrase", action="store_true", help="rephrase the questions")
parser.add_argument("--dataset_path", default="datasets/v7", help="dataset path")


args, _ = parser.parse_known_args()

max_rel = args.max_rel
dataset_path = args.dataset_path
rel_df = pd.read_csv(args.rel_path, sep="\t")
relations = rel_df["relation_name"].tolist()

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
    with open(f"{dataset_path}/test.tsv", "w") as file:
        file.write(tsv_string)
    df = pd.read_csv(tsv_data, sep="\t")[columns]
    df = df.dropna()
    print(f"Dataframe number {idx+1}\n", df)
    dfs.append(df)
# Concatenate all dataframes together
final_df = pd.concat(dfs).reset_index(drop=True)
assert len(final_df) == len(rel_df) or "Error, the number of rows in the final dataframe is different from the original dataframe"
print(final_df)

# Save df containing relation_name, subject, object, question
final_df.to_csv(f"{dataset_path}/agg_relations_with_prompts.tsv", sep="\t", index=False)
