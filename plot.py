import pandas as pd
import os
import matplotlib.pyplot as plt

df_scr = pd.read_csv('logs/previous/simsiam-ham-resnet50_small_scratch-exp1/logger.csv')
df_pre = pd.read_csv('logs/previous/ready/simsiam-ham-resnet50_small_pretrain-exp1/logger.csv')
df_pre_ad = pd.read_csv('logs/previous/ready/simsiam-ham-resnet50_small_pretrain_adapt-exp1/logger.csv')

plt.plot(df_scr['epoch'], df_scr['acc'], label='scratch')
plt.plot(df_pre['epoch'], df_pre['acc'], label='pretrain')
plt.plot(df_pre_ad['epoch'], df_pre_ad['acc'], label='pretrain + adapt')
plt.legend()
plt.title('acc')
plt.show()

plt.close()

plt.plot(df_scr['epoch'], df_scr['loss'], label='scratch')
plt.plot(df_pre['epoch'], df_pre['loss'], label='pretrain')
plt.plot(df_pre_ad['epoch'], df_pre_ad['loss'], label='pretrain + adapt')
plt.legend()
plt.title('loss')
plt.show()





