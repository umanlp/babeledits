import argparse
import pandas as pd

def filter_relations(orig_rel, gold_rel):
    common_relations = set(orig_rel['relation_name']).intersection(set(gold_rel['relation_name']))
    filtered_gold_rel = gold_rel[gold_rel['relation_name'].isin(common_relations)]
    return filtered_gold_rel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter relations based on common relation names')
    parser.add_argument('--orig_rel', help='Path to the original data frame')
    parser.add_argument('--gold_rel', help='Path to the gold data frame')
    parser.add_argument('--output_path', help='Path for the output frame')
    args = parser.parse_args()
    
    orig_rel_path = args.orig_rel
    gold_rel_path = args.gold_rel
    
    # Load the data frames from the provided paths
    orig_rel = pd.read_csv(orig_rel_path, sep='\t')
    gold_rel = pd.read_csv(gold_rel_path, sep='\t')
    
    # Filter the gold data frame based on common relation names
    filtered_gold_rel = filter_relations(orig_rel, gold_rel)
    
    # Print the filtered gold data frame
    print(filtered_gold_rel)

    # Save the filtered gold data frame to the output path
    filtered_gold_rel.to_csv(args.output_path, sep='\t', index=False)
