import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams['font.size'] = 14
df_bn_scr = pd.read_csv('logs/simsiam-cifar10-experiment-resnet50_bn_small_scratch/logger.csv', index_col=0)
df_gn_scr = pd.read_csv('logs/simsiam-cifar10-experiment-resnet50_gn_small_scratch/logger.csv', index_col=0)

epochs = [i for i in range(1, 201)]
plt.plot(epochs, df_bn_scr['loss'], label='Batch Norm', color='tab:blue')
plt.plot(epochs, df_gn_scr['loss'], label='Group Norm', color='tab:orange')
plt.locator_params(axis="x", nbins=8, tight=True)
plt.locator_params(axis="y", nbins=5, tight=True)
plt.xlabel('epochs', fontsize=16)
plt.ylabel('training loss', fontsize=16)
plt.legend(fontsize=14)
# plt.show()
plt.tight_layout()
plt.savefig('miccai/plots/cifar_scr_loss.svg')
plt.close()

epochs = [i for i in range(1, 201)]
plt.plot(epochs, df_bn_scr['acc'], label='Batch Norm', color='tab:blue')
plt.plot(epochs, df_gn_scr['acc'], label='Group Norm', color='tab:orange')
plt.locator_params(axis="x", nbins=8, tight=True)
plt.locator_params(axis="y", nbins=5, tight=True)
plt.xlabel('epochs', fontsize=16)
plt.ylabel('kNN acc', fontsize=16)
plt.legend(fontsize=14, loc='center right')
plt.tight_layout()
# plt.show()
plt.savefig('miccai/plots/cifar_scr_acc.svg')
