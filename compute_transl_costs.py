# %% 
import sienna
from utils import extract
dataset_path = "datasets/v8/dataset.json" 


data = sienna.load(dataset_path)
src_lang = "en"
print(f"Reading dataset from {dataset_path}...")
prompts = extract(data,"en", "prompts")

prompt_types = ["prompts", "prompts_gen", "prompts_loc", "prompts_port"]

prompt_types_with_strictness = [
    (x, True) if x not in ["prompts_loc", "prompts_port"] else (x, False) for x in prompt_types
]
extracted_prompts = [
    extract(data, "en", prompt_type, strict=strict_val)
    for prompt_type, strict_val in prompt_types_with_strictness
]

all_prompts = [
    item
    for sublist in zip(*extracted_prompts)
    for item in sublist
    if item is not None
]
subjects = extract(data, "en", "subjects")
objects = extract(data, "en", "targets")
ground_truths = extract(data, "en", "ground_truths")
ground_truths_port = extract(data, "en", "ground_truths_port", strict=False)
ground_truths_port = [e for e in ground_truths_port if e]
ground_truths_loc = extract(data, "en", "ground_truths_loc", strict=False)
ground_truths_loc = [e for e in ground_truths_loc if e]
entities = subjects + objects + ground_truths_loc + ground_truths + ground_truths_port

# %%

from utils import format_prompt
prompts = [format_prompt(p, s, t) for (p, s, t) in zip(prompts, subjects, objects)]

all_prompts_marked = []

for syn_id, example in data.items():
    relation = list(example["relations"].keys())[0]
    all_prompts_marked.append(
        format_prompt(
            example["relations"][relation]["edit"]["prompts"][src_lang],
            example["subjects"][src_lang],
            example["relations"][relation]["edit"]["targets"][src_lang],
        )
    )
    if "generality" in example["relations"][relation]["edit"]:
        all_prompts_marked.append(
            format_prompt(
                example["relations"][relation]["edit"]["generality"]["prompts_gen"][
                    src_lang
                ],
                example["subjects"][src_lang],
                example["relations"][relation]["edit"]["targets"][src_lang],
            )
        )
    if "locality" in example["relations"][relation]["edit"]:
        loc_relation = list(
            example["relations"][relation]["edit"]["locality"].keys()
        )[0]
        all_prompts_marked.append(
            format_prompt(
                example["relations"][relation]["edit"]["locality"][loc_relation][
                    "prompts_loc"
                ][src_lang],
                example["subjects"][src_lang],
                example["relations"][relation]["edit"]["locality"][loc_relation][
                    "ground_truths_loc"
                ][src_lang],
            )
        )
    if "portability" in example["relations"][relation]["edit"]:
        port_relation = list(
            example["relations"][relation]["edit"]["portability"]["multi_hop"].keys()
        )[0]
        port_target = example["relations"][relation]["edit"]["portability"]["multi_hop"][port_relation][
            "ground_truths_port"
        ][src_lang]
        all_prompts_marked.append(
            example["relations"][relation]["edit"]["portability"]["multi_hop"][port_relation][
                "prompts_port"
            ][src_lang] + f" <o>{port_target}</o>"
        )

# %%

cost_per_char = 20/1e6
num_langs = 60
cost_all_prompts = sum([len(prompt) for prompt in all_prompts]) * cost_per_char * num_langs
cost_all_prompts_marked = sum([len(prompt) for prompt in all_prompts_marked]) * cost_per_char * num_langs
cost_entities = sum([len(entity) for entity in entities]) * cost_per_char * num_langs
print(cost_all_prompts)
print(cost_all_prompts_marked)
print(cost_entities)
print(f"Total cost: {cost_all_prompts + cost_all_prompts_marked + cost_entities:.2f}")