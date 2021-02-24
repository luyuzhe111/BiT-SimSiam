import os
import json
import random
from collections import Counter
import shutil

data = 'ham-10000/json/ham_data.json'
with open(data) as f:
    data = json.load(f)

random.seed(3)
random.shuffle(data)

# five fold split
split = int(0.2 * len(data))
folds = []
for i in range(5):
    fold = data[i * split:(i + 1) * split]
    folds.append(fold)
    print(Counter([i['target'] for i in fold]))
    with open(f'ham-10000/json/fold{i+1}.json', 'w') as f:
        json.dump(fold, f)

for idx, fold in enumerate(folds):
    train_set = folds[idx % 5] + folds[(idx + 1) % 5] + folds[(idx + 2) % 5]
    print(Counter([i['target'] for i in train_set]))

    root_dir = 'ham-10000/json'
    exp_dir = os.path.join(root_dir, f'exp_{idx + 1}')
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    with open(os.path.join(exp_dir, 'train.json'), 'w') as f:
        json.dump(train_set, f)

    val_set = os.path.join(root_dir, f'fold{(idx + 3) % 5 + 1}.json')
    test_set = os.path.join(root_dir, f'fold{(idx + 4) % 5 + 1}.json')

    shutil.copy(val_set, os.path.join(exp_dir, 'validation.json'))
    shutil.copy(val_set, os.path.join(exp_dir, 'test.json'))


# 80% / 20% train / val split

# train = data[:int(len(data) * 0.8) + 1]
# val = data[int(len(data) * 0.8) + 1:int(len(data) * 0.9) + 1]
# test = data[int(len(data) * 0.9) + 1:]
#
# train_tar = [i['target'] for i in train]
# val_tar = [i['target'] for i in val]
# test_tar = [i['target'] for i in test]
#
# print(Counter(train_tar))
# print(Counter(val_tar))
# print(Counter(test_tar))
#
# print(len(train), len(val), len(test))
#
# with open('ham-10000/json/train.json', 'w') as f:
#     json.dump(train, f)
#
# with open('ham-10000/json/validation.json', 'w') as f:
#     json.dump(val, f)
#
# with open('ham-10000/json/test.json', 'w') as f:
#     json.dump(test, f)