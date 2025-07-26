# %%

from typing import OrderedDict
import sienna
from collections import Counter
data = sienna.load("datasets/v2/it.json")


relation_counts = Counter()

# Iterate through the dataset
for item in data.values():
    if 'relations' in item:
        for relation, relation_list in item['relations'].items():
            relation_counts[relation] += len(relation_list)

relation_counts = OrderedDict(relation_counts.most_common())
# Print the counts of each relation
for relation, count in relation_counts.items():
    print(f"{relation}: {count}")


# %%

for key, item in data.items():
    if 'relations' in item and 'CAST_MEMBER' in item['relations']:
        print(data[key]["subject_senses"]["sense_en"])