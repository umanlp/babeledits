"""
Create a test subset set of the hard set keeping only the keys that cannot be
found in train or val set of the big dataset
"""
import json

from data_splits.hard_subset_intersection_stats import check_sample_against_keys, get_keys

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="v5")
    parser.add_argument("--languages", type=str, nargs="+", default=["ar", "de", "es", "fr", "hr", "it", "ja", "nl", "zh"])
    parser.add_argument("--same_relation", action="store_true", dest="same_relation", help="Add this flag if you want to check the intersection of (subject,relationship,object) triples instead of simply the individual babelnet ids of subjects and objects")
    parser.add_argument("--exclude_object", action="store_true", dest="exclude_object", help="Add this flag if you want to filter only based on subject (will not matter if you also used the --same_relation flag)")
    args = parser.parse_args()

    with open(f"datasets/{args.version}/train.json") as f:
        keys = get_keys(json.load(f), same_relation=args.same_relation, exclude_object=args.exclude_object)
    with open(f"datasets/{args.version}/val.json") as f:
        keys = get_keys(json.load(f), keys=keys, same_relation=args.same_relation, exclude_object=args.exclude_object)

    for lang in args.languages:
        with open(f"datasets/{args.version}/hard/{lang}/translated/dataset.json") as f:
            hard_data = json.load(f)

        test_hard_data = {}
        for key, sample in hard_data.items():
            if check_sample_against_keys(key, sample, keys, same_relation=args.same_relation, exclude_object=args.exclude_object):
                continue
            test_hard_data[key] = sample
        
        with open(f"datasets/{args.version}/hard/{lang}/translated/test.json", "w") as f:
            json.dump(test_hard_data, f, indent=4)
        
        print(f"{lang}: could keep only {len(test_hard_data)} samples over a total of {len(hard_data)}")