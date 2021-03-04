import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score


exp_name = 'simsiam-ham-resnet50_small_scratch-exp1'
eval_logs = f'eval_logs/{exp_name}'

b_acc = []
b_f1 = []
for i in range(25, 401, 25):
    df = pd.read_csv(f'{eval_logs}/epoch_{i}_fl_ens.csv', index_col=0)
    gh = df['gh']
    pred = df['pred']
    b_acc.append(balanced_accuracy_score(gh, pred))
    f1_for_each = f1_score(gh, pred, average=None)
    b_f1.append(sum(f1_for_each) / len(f1_for_each))

df_out = pd.DataFrame({'b_acc': b_acc, 'b_f1': b_f1})
df_out.to_csv(f'miccai/linear_eval/{exp_name}_fl_ens3.csv')



