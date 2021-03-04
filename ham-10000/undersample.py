import random
import json
from collections import Counter
from collections import defaultdict

with open('ham-10000/json/validation_db.json') as f:
    data = json.load(f)

targets = [i['target'] for i in data]

print(Counter(targets))

random.seed(0)

groups = defaultdict(list)
for item in data:
    groups[item['target']].append(item)

grouped_list = list(groups.values())

undersampled_list = []
for i in grouped_list:
    undersampled_list += random.sample(i, 23)

print(Counter([i['target'] for i in undersampled_list]))

with open('ham-10000/json/validation_us.json', 'w') as f:
    json.dump(undersampled_list, f)