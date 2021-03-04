import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams['font.size'] = 14

df_scr = pd.read_csv('miccai/loggers/simsiam-ham-resnet50_small_scratch-exp1_knn.csv')
df_pre_s = pd.read_csv('miccai/loggers/simsiam-ham-resnet50_small_pretrain-exp1_knn.csv')
df_pre_m = pd.read_csv('miccai/loggers/simsiam-ham-resnet50_medium_pretrain-exp1_knn.csv')

epochs = [i for i in range(1, 401)]

plt.plot(epochs, df_scr['bacc'] * 100, label='SimSiam', color='tab:blue')
plt.plot(epochs, df_pre_s['bacc'] * 100, label='BiT-S + SimSiam',  color='tab:orange')
plt.plot(epochs, df_pre_m['bacc'] * 100, label='BiT-M + SimSiam',  color='tab:green')
plt.locator_params(axis="x", nbins=8, tight=True)
plt.locator_params(axis="y", nbins=8, tight=True)
plt.xlabel('epochs', fontsize=16)
plt.ylabel('kNN balanced acc', fontsize=16)
plt.legend(fontsize=14)
plt.tight_layout()
#plt.show()
plt.savefig('miccai/plots/ham_bacc.svg')
plt.close()


df_scr = pd.read_csv('logs/simsiam-ham-resnet50_small_scratch-exp1/logger_resume.csv')
# df_pre_c = pd.read_csv('logs/simsiam-ham-resnet50_common_pretrain-exp1/logger_resume.csv')
df_pre_s = pd.read_csv('logs/simsiam-ham-resnet50_small_pretrain-exp1/logger_resume.csv')
df_pre_m = pd.read_csv('logs/simsiam-ham-resnet50_medium_pretrain-exp1/logger_resume.csv')

plt.plot(epochs, df_scr['loss'], label='SimSiam', color='tab:blue')
# plt.plot(epochs[:200], df_pre_c['loss'], label='SimSiam + PreC',  color='tab:brown')
plt.plot(epochs, df_pre_s['loss'], label='BiT-S + SimSiam',  color='tab:orange')
plt.plot(epochs, df_pre_m['loss'], label='BiT-M + SimSiam',  color='tab:green')
plt.locator_params(axis="x", nbins=8, tight=True)
plt.locator_params(axis="y", nbins=8, tight=True)
plt.xlabel('epochs', fontsize=16)
plt.ylabel('training loss', fontsize=16)
plt.legend(fontsize=14)
plt.tight_layout()
#plt.show()
plt.savefig('miccai/plots/ham_loss.svg')
