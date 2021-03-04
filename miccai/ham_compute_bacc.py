import pandas as pd
from sklearn.metrics import balanced_accuracy_score

exp = 'simsiam-ham-resnet50_common_pretrain-exp1'
epochs_dir = f'logs/{exp}/epochs'
output_dir = 'miccai/loggers'

bacc = []
for i in range(200):
    df = pd.read_csv(f'{epochs_dir}/epoch{i}.csv', index_col=0)
    pred = df['prediction']
    tar = df['target']
    bacc.append(balanced_accuracy_score(tar, pred))

epochs = [i for i in range(1, 201)]
df_out = pd.DataFrame({'epoch': epochs, 'bacc': bacc})
df_out.to_csv(f'{output_dir}/{exp}_knn.csv')