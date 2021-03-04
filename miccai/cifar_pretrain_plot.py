import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams['font.size'] = 14

gn = pd.read_csv('logs/simsiam-cifar10-experiment-resnet50_bn_small_pretrain/logger_resume.csv', index_col=0)
bn = pd.read_csv('logs/simsiam-cifar10-experiment-resnet50_gn_small_pretrain/logger_resume.csv', index_col=0)

epochs = [i for i in range(1, 201)]
plt.plot(epochs, gn['acc'], label='Group Norm')
plt.plot(epochs, bn['acc'], label='Batch Norm')
plt.locator_params(axis="x", nbins=8, tight=True)
plt.xlabel('epochs', fontsize=16)
plt.ylabel('kNN acc', fontsize=16)
plt.legend(fontsize=14, loc='center right')
plt.tight_layout()
plt.savefig('miccai/plots/cifar_pre_acc.svg')
plt.show()
