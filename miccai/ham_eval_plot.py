import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix

plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams['font.size'] = 12

bit_s = pd.read_csv('eval_logs/bit-ham-experiment-resnet50_small-exp1/epoch_fl_ens.csv', index_col=0)
bit_s_acc = balanced_accuracy_score(bit_s['gh'], bit_s['pred'])
bit_s_f1 = sum(f1_score(bit_s['gh'], bit_s['pred'], average=None)) / len(f1_score(bit_s['gh'], bit_s['pred'], average=None))
print(bit_s_acc, bit_s_f1)
print(confusion_matrix(bit_s['gh'], bit_s['pred']))
print()
bit_m = pd.read_csv('eval_logs/bit-ham-experiment-resnet50_medium-exp1/epoch_fl_ens.csv', index_col=0)
bit_m_acc = balanced_accuracy_score(bit_m['gh'], bit_m['pred'])
bit_m_f1 = sum(f1_score(bit_m['gh'], bit_m['pred'], average=None)) / len(f1_score(bit_s['gh'], bit_s['pred'], average=None))
print(bit_m_acc, bit_m_f1)
print(confusion_matrix(bit_m['gh'], bit_m['pred']))

df_scr = pd.read_csv('miccai/linear_eval/simsiam-ham-resnet50_small_scratch-exp1_fl_ens3.csv', index_col=0)
df_pre_s = pd.read_csv('miccai/linear_eval/simsiam-ham-resnet50_small_pretrain-exp1_fl_ens3.csv', index_col=0)
df_pre_m = pd.read_csv('miccai/linear_eval/simsiam-ham-resnet50_medium_pretrain-exp1_fl_ens3.csv', index_col=0)

epochs = [i for i in range(25, 401, 25)]
plt.plot(epochs, df_scr['b_acc'] * 100, label='SimSiam', marker='s')
plt.plot(epochs, df_pre_s['b_acc'] * 100, label='BiT-S + SimSiam', marker='s')
plt.plot(epochs, df_pre_m['b_acc'] * 100, label='BiT-M + SimSiam', marker='s')
plt.ylim(30, 55)
plt.xlabel('epochs')
plt.ylabel('balanced accuracy')
plt.xticks(np.arange(0, max(epochs)+1, 50))

plt.axhline(y=bit_s_acc * 100, color='r', linestyle='--', label='BiT-S')
plt.axhline(y=bit_m_acc * 100, color='b', linestyle='--', label='BiT-M')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('miccai/plots/ham_exp.svg')
plt.show()