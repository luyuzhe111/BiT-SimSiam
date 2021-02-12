import json
import pandas as pd
import os

df = pd.read_csv('ham-10000/csv/ham_data.csv')
label_dict = {'MEL': 0, 'NV': 1, 'BCC': 2, 'AKIEC': 3, 'BKL': 4, 'DF': 5, 'VASC': 6}
targets = []

data_list = []
data_root_dir = '/Data/luy8/data/ham-10000/resized_debug_data'
for index, row in df.iterrows():
    # make csv for ref
    series = row['MEL':'VASC']
    tar_series = series[series == 1]
    label = label_dict[tar_series.index[0]]
    targets.append(label)

    # make json
    img = row['image']
    img_dir = os.path.join(data_root_dir, img + '.png')
    img_label = label
    img_dict = {'image': img, 'img_dir': img_dir, 'target': label}
    data_list.append(img_dict)

df.insert(8, 'target', targets)
df.to_csv('ham-10000/csv/ham_data_with_label.csv')

with open('ham-10000/json/ham_debug_data.json', 'w') as f:
    json.dump(data_list, f)




