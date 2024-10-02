# %%

import sienna
from utils import extract

dataset_path = "datasets/trash/dataset.json"
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
            data[syn_id]["relations"][relation]["edit"]["portability"].keys()
        )[0]
        port_target = data[syn_id]["relations"][relation]["edit"]["portability"][
            port_relation
        ]["ground_truths_id_port"]
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
from langchain_community.callbacks import get_openai_callback
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o", temperature=0, timeout=None, max_retries=5, max_tokens=None
)

msg = """You are a helpful assistant that is able to leverage its world knowledge to convert relations extracted from a knowledge graph 
    (for example, WordNet or Babelnet) into natural language questions. In this case we are dealing with joined triples of the form (subject, relation, object, relation_2, object_2).
    You need to formulate a natural language question which should be answered with object 2. Consider the case of (Messi, PLAYS_FOR, Barcelona, LOCATED_IN, Spain).
    The question could be 'In which country is the team Messi plays for located?'. In the generated question, NEVER mention the object (in this case, Barcelona).
    Let me repeat: Do NOT INCLUDE the object in the question.
    The input will be a markdown table, with five columns: subject, relation, object, relation_2, object_2. 
    Reply directly without any additional text, one question per line, no special charachters at the begining of each line.
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

batch_size = 100
batches = [df[i : i + batch_size] for i in range(0, len(df), batch_size)]
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

flattened_contents = [item for sublist in contents for item in sublist.split("\n")]
assert len(flattened_contents) == len(df)
# %%
idx = 0
for syn_id in data:
    relation = list(data[syn_id]["relations"].keys())[0]
    if "portability" in data[syn_id]["relations"][relation]["edit"]:
        port_relation = list(
            data[syn_id]["relations"][relation]["edit"]["portability"].keys()
        )[0]
        data[syn_id]["relations"][relation]["edit"]["portability"][port_relation][
            "prompts_port"
        ] = {"en" : flattened_contents[idx]}
        idx += 1

import json

with open(dataset_path, "w") as f:
    json.dump(data, f, indent=4)
# %%
