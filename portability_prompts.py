# %%

import langchain
import sienna
from utils import extract
import argparse

parser = argparse.ArgumentParser(description="Process dataset path.")
parser.add_argument("--dataset_path", type=str, help="Path to the dataset file")
args = parser.parse_args()

langchain.verbose = False
langchain.debug = False
langchain.llm_cache = False

# dataset_path = "datasets/v8/dataset.json" # args.dataset_path
dataset_path = args.dataset_path
data = sienna.load(dataset_path)

# %%

portability_data = {}
subjects = {}
targets = {}
relations = {}
prompts = {}
for syn_id in data:
    relation = list(data[syn_id]["relations"].keys())[0]
    if "portability" in data[syn_id]["relations"][relation]["edit"]:
        subjects[syn_id] = data[syn_id]["subjects"]["en"]
        relations[syn_id] = relation
        prompts[syn_id] = data[syn_id]["relations"][relation]["edit"]["prompts"]["en"]
        targets[syn_id] = data[syn_id]["relations"][relation]["edit"]["targets"]["en"]
        port_relation = list(
            data[syn_id]["relations"][relation]["edit"]["portability"][
                "multi_hop"
            ].keys()
        )[0]
        port_target = data[syn_id]["relations"][relation]["edit"]["portability"][
            "multi_hop"
        ][port_relation]["ground_truths_port"]["en"]
        portability_data[syn_id] = {"relation": port_relation, "target": port_target}
# %%

import pandas as pd

table = {"subject": [], "relation": [], "object": [], "relation_2": [], "object_2": []}
for syn_id in list(portability_data.keys()):
    table["subject"].append(subjects[syn_id])
    table["relation"].append(relations[syn_id])
    table["object"].append(targets[syn_id])
    table["relation_2"].append(portability_data[syn_id]["relation"])
    table["object_2"].append(portability_data[syn_id]["target"])
    # subjects[syn_id], relations[syn_id], targets[syn_id], portability_data[syn_id]["relation"], portability_data[syn_id]["target"])

df = pd.DataFrame(table)
print(df.to_markdown(index=False))

# %%
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o", temperature=0, timeout=None, max_retries=5, max_tokens=None
)

msg = """You are a helpful assistant that is able to leverage its world knowledge to convert relations extracted from a knowledge graph 
    (for example, WordNet or Babelnet) into natural language questions. In this case we are dealing with joined triples of the form (subject, relation, object, relation_2, object_2).
    You need to formulate a natural language question which should be answered with object 2. Consider the case of (Messi, PLAYS_FOR, Barcelona, LOCATED_IN, Spain).
    The question could be 'In which country is the team that Messi plays for located?'. In the generated question, NEVER mention the object (in this case, Barcelona).
    Let me repeat: Do NOT INCLUDE the object in the question.
    The input will be a markdown table, with five columns: subject, relation, object, relation_2, object_2. 
    Please reply directly without any additional text, one question per line, no special characters at the beginning of each line and separate each line with a SINGLE newline character and not two.
    Just a reminder: only one question per line, only one newline character at the end of each line.
"""
# msg = """You are a helpful assistant that is able to leverage its world knowledge to convert relations extracted from a knowledge graph
#     (for example, WordNet or Babelnet) into natural language questions. In this case we are dealing with data of the form (subject, relation, object, prompt, relation_2, object_2).
#     Prompt is a natural language question that verbalizes the relation (subject, relation, object).
#     You need to formulate a natural language question which should be answered with object 2 by chaining the two triples (subject, relation, object, relation_2, object_2).
#     Consider the case of (Messi, PLAYS_FOR, Barcelona, Which team does Messi play for, LOCATED_IN, Spain).
#     The question that chains the two triples could be 'In which country is the team Messi plays for located?'.
#     In the generated question, NEVER mention the object (in this case, Barcelona). Let me repeat it: Do NOT INCLUDE the object in the question.
#     The input will be a markdown table, with five columns: subject, relation, object, relation_2, object_2.
#     Reply directly without any additional text, one question per line, no special charachters at the begining of each line.
# """

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            msg,
        ),
        (
            "human",
            """Input:\n
            {input}""",
        ),
    ]
)

chain = prompt | llm

batch_size = 10
batches = [df[i : i + batch_size] for i in range(0, len(df), batch_size)]
dfs = []
contents = []
max_retries = 10
for idx, batch in enumerate(batches):
    # Convert batch to markdown
    batch_md = batch.to_markdown(index=False)
    # Invoke llm with batch_md
    attempt_idx = 0
    while attempt_idx < max_retries:
        result = chain.invoke(batch_md)
        if len(result.content.split("\n")) == len(batch):
            print(f"Batch {idx+1}/{len(batches)} generated successfully")
            break
        elif len(result.content.split("\n")) > len(batch):
            num_lines = len(result.content.split("\n"))
            print(f"result.content has {num_lines} lines")
            if len(result.content.split("\n\n")) == len(batch):
                print(result.content)
                break
            print(batch_md)
            print("----- //// -----")
            print(result.content)
            raise ValueError("Generated more prompts than expected")
        else:
            print(
                f"Batch {idx+1}/{len(batches)} was NOT generated successfully...retrying ({attempt_idx+1}/{max_retries})"
            )
            attempt_idx += 1
            processed_elements = len(result.content.split("\n"))
            # Add a message to the LLM chain
            new_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        msg,
                    ),
                    (
                        "human",
                        """Input:\n
                        {input}""",
                    ),
                    ("assistant", result.content),
                    (
                        "human",
                        f"You forgot a few elements in your response above, I am only getting {processed_elements} out of {batch_size} elements from your output. REMEMBER TO PROCESS ALL THE ROWS IN THE INPUT TABLE.",
                    ),
                ]
            )
            print(f"Retrying with message {new_prompt}")
            chain = new_prompt | llm
    if attempt_idx == max_retries:
        raise ValueError(f"Failed to generate prompts for batch {idx}")
    # Convert result to a dataframe
    contents.append(result.content)
    chain = prompt | llm

contents = [sublist.replace("\n\n", "\n") for sublist in contents]
flattened_contents = [item for sublist in contents for item in sublist.split("\n")]
assert len(flattened_contents) == len(df)
# %%
idx = 0
for syn_id in data:
    relation = list(data[syn_id]["relations"].keys())[0]
    if "portability" in data[syn_id]["relations"][relation]["edit"]:
        port_relation = list(
            data[syn_id]["relations"][relation]["edit"]["portability"][
                "multi_hop"
            ].keys()
        )[0]
        data[syn_id]["relations"][relation]["edit"]["portability"]["multi_hop"][
            port_relation
        ]["prompts_port"] = {"en": flattened_contents[idx].strip()}
        idx += 1

import json

with open(dataset_path, "w") as f:
    json.dump(data, f, indent=4)
# %%
