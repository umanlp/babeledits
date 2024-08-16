"""
Split the big dataset in train/val/test
"""
from collections import defaultdict

import json
from sklearn.model_selection import train_test_split

def split_data(data, random_state=None, test_size=0.1, val_size=0.13):
    keys = list(data.keys())
    rel_types = [next(iter(data[key]["relations"])) for key in keys]

    # rare relationships are merged so that the stratification doesn't fail
    rare_type_to_ids = defaultdict(list)
    non_rare_types = set()
    min_rel_size = 3
    for i, rel_type in enumerate(rel_types):
        rare_type_to_ids[rel_type].append(i)
        if len(rare_type_to_ids[rel_type]) >= min_rel_size:
            non_rare_types.add(rel_type)
            del rare_type_to_ids[rel_type]
    
    rare_ids = sum(rare_type_to_ids.values(), [])
    
    print(
        f"Got {len(rare_type_to_ids)} relationship types (over a total of {len(set(rel_types))}) that have less than {min_rel_size} samples, which cannot be stratified.\n"
        + f"They represent a total of {len(rare_ids)} samples, which will merged into a single group."
    )

    for idx in rare_ids:
        rel_types[idx] = "__DEFAULT_RELATIONSHIP__"


    train_keys, test_keys, train_rel_types, _ = train_test_split(keys, rel_types, stratify=rel_types, test_size=test_size, random_state=random_state)
    train_keys, val_keys = train_test_split(train_keys, stratify=train_rel_types, test_size=val_size, random_state=random_state)

    return train_keys, test_keys, val_keys


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="v5")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--val_size", type=float, default=0.1)
    args = parser.parse_args()

    input_file = f"datasets/{args.version}/dataset.json"

    train_file = f"datasets/{args.version}/train.json"
    test_file = f"datasets/{args.version}/test.json"
    val_file = f"datasets/{args.version}/val.json"

    with open(input_file, "r") as f:
        data = json.load(f)
    
    train_keys, test_keys, val_keys = split_data(data, random_state=args.random_state, test_size=args.test_size, val_size=args.val_size)

    with open(train_file, "w") as f:
        f.write(json.dumps({k: data[k] for k in train_keys}, indent=4))
    with open(test_file, "w") as f:
        f.write(json.dumps({k: data[k] for k in test_keys}, indent=4))
    with open(val_file, "w") as f:
        f.write(json.dumps({k: data[k] for k in val_keys}, indent=4))