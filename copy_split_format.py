import json
import os

# Directory path
directory = "datasets/v8/translated"

# Files to process
dataset_file = os.path.join(directory, "dataset_2.json")
split_files = [
    os.path.join(directory, "train.json"),
    os.path.join(directory, "val.json"),
    os.path.join(directory, "test.json"),
    os.path.join(directory, "val_100.json"),
]

# Load the main dataset
print(f"Loading dataset from {dataset_file}...")
with open(dataset_file, "r", encoding="utf-8") as f:
    dataset = json.load(f)
print(f"Dataset loaded with {len(dataset)} entries")

# Process each split file
for split_file in split_files:
    split_name = os.path.basename(split_file).replace(".json", "")
    output_file = os.path.join(directory, f"{split_name}_2.json")

    print(f"Processing {split_file}...")

    # Load the split file to get its keys
    with open(split_file, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    # Extract the keys
    split_keys = list(split_data.keys())
    print(f"Found {len(split_keys)} keys in {split_name}")

    # Create a new dictionary with only the keys from the split file
    new_split = {k: dataset[k] for k in split_keys if k in dataset}
    print(f"Created new split with {len(new_split)} entries")

    # Save the new split file
    print(f"Saving to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(new_split, f, ensure_ascii=False, indent=None)

    print(f"Saved {output_file}")

print("All splits processed successfully!")
