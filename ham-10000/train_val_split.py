import json
import random
from collections import Counter
data = 'ham-10000/json/ham_debug_data.json'
with open(data) as f:
    data = json.load(f)

random.seed(0)
random.shuffle(data)

train = data[:int(len(data) * 0.8) + 1]
val = data[int(len(data) * 0.8) + 1:]

train_tar = [i['target'] for i in train]
val_tar = [i['target'] for i in val]

print(Counter(train_tar))
print(Counter(val_tar))

print(len(train), len(val))

with open('ham-10000/json/train_db.json', 'w') as f:
    json.dump(train, f)

with open('ham-10000/json/validation_db.json', 'w') as f:
    json.dump(val, f)