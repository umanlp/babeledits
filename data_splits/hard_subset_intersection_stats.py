"""
Script to evaluate the proportion of babelnet ids from the hard subset that can be found in the whole dataset
"""
import os
import json


def check_sample_against_keys(key, sample, keys: set, same_relation=False, exclude_object=False) -> bool:
    if same_relation:
        return (key, next(iter(sample["relations"])), sample["relations"][next(iter(sample["relations"]))]["ground_truth_id"]) in keys
    elif exclude_object:
        return key in keys
    else:
        return key in keys or any(
            map(
                lambda x: x["ground_truth_id"] in keys or x["edit"]["target_id"] in keys,
                sample["relations"].values()
            )
        )

def get_keys(data, keys=None, same_relation=False, exclude_object=False):
    keys = keys or set()
    if same_relation:
        keys.update([(key, next(iter(sample["relations"])), sample["relations"][next(iter(sample["relations"]))]["ground_truth_id"]) for key, sample in data.items()])
    else:
        keys.update(data.keys())
        if not exclude_object:
            keys.update([sample["relations"][next(iter(sample["relations"]))]["ground_truth_id"] for sample in data.values()])
            keys.update([sample["relations"][next(iter(sample["relations"]))]["edit"]["target_id"] for sample in data.values()])
    return keys

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="v5")
    parser.add_argument("--languages", type=str, nargs="+", default=["ar", "de", "es", "fr", "hr", "it", "ja", "nl", "zh"])
    parser.add_argument("--same_relation", action="store_true", dest="same_relation", help="Add this flag if you want to check the intersection of (subject,relationship,object) triples instead of simply the individual babelnet ids of subjects and objects")
    parser.add_argument("--exclude_object", action="store_true", dest="exclude_object", help="Add this flag if you want to filter only based on subject (will not matter if you also used the --same_relation flag)")
    args = parser.parse_args()

    hard_subset_dir = f"datasets/{args.version}/hard"
    dataset_file = f"datasets/{args.version}/dataset.json"

    big_dataset_keys = set()
    with open(dataset_file, "r") as f:
        data = json.load(f)
        big_dataset_keys = get_keys(data, same_relation=args.same_relation, exclude_object=args.exclude_object)

    all_hard_keys = set()
    for lang in args.languages:
        n_samples_to_discard = 0
        n_samples = 0
        with open(os.path.join(hard_subset_dir, lang, "dataset.json"), "r") as f:
            hard_data = json.load(f)
            all_hard_keys = get_keys(hard_data, keys=all_hard_keys, same_relation=args.same_relation, exclude_object=args.exclude_object)
            for key, sample in hard_data.items():
                if check_sample_against_keys(key, sample, big_dataset_keys, same_relation=args.same_relation, exclude_object=args.exclude_object):
                    n_samples_to_discard += 1
                n_samples += 1

        print(f"Number of samples in hard subset in language {lang} that use a babelnet id found in the big dataset: {n_samples_to_discard} (over a total of {n_samples})")

    big_n_samples = 0
    big_n_samples_in_hard = 0
    for key, sample in data.items():
        if check_sample_against_keys(key, sample, all_hard_keys, same_relation=args.same_relation, exclude_object=args.exclude_object):
            big_n_samples_in_hard += 1
        big_n_samples += 1
    print(f"Number of keys from the big dataset found in hard subsets: {big_n_samples_in_hard} (over {big_n_samples} samples in the big dataset)")