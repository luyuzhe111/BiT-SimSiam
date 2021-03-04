import json

with open('ham-10000/json/exp_1/test.json') as f:
    test = json.load(f)
    test = [i['img_dir'] for i in test]

with open('ham-10000/json/exp_1/validation.json') as f:
    val = json.load(f)
    val = [i['img_dir'] for i in val]

with open('ham-10000/json/exp_1/train.json') as f:
    train = json.load(f)
    train = [i['img_dir'] for i in train]

with open('ham-10000/json/fold1.json') as f:
    fold1 = json.load(f)
    fold1 = [i['img_dir'] for i in fold1]

with open('ham-10000/json/fold2.json') as f:
    fold2 = json.load(f)
    fold2 = [i['img_dir'] for i in fold2]

with open('ham-10000/json/fold3.json') as f:
    fold3 = json.load(f)
    fold3 = [i['img_dir'] for i in fold3]

print(all(item in train for item in fold1 + fold2 + fold3))

print(any(item in test for item in val))
