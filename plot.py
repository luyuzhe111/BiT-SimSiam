import pandas as pd
import os
import matplotlib.pyplot as plt

df_scr = pd.read_csv('logs/simsiam-ham-resnet50_small_scratch-exp1/logger_resume.csv')
df_pre_s = pd.read_csv('logs/simsiam-ham-resnet50_small_pretrain-exp1/logger_resume.csv')
df_pre_m = pd.read_csv('logs/simsiam-ham-resnet50_medium_pretrain-exp1/logger_resume.csv')
df_pre_cb = pd.read_csv('logs/simsiam-ham-resnet50_cifar_backbone_pretrain-exp1/logger_resume.csv')
df_pre_c = pd.read_csv('logs/simsiam-ham-resnet50_cifar_pretrain-exp1/logger_resume.csv')

plt.plot(df_scr['epoch'], df_scr['acc'], label='scratch')
plt.plot(df_pre_cb['epoch'], df_pre_cb['acc'], label='pretrain cifar backbone')
plt.plot(df_pre_c['epoch'], df_pre_c['acc'], label='pretrain cifar')
plt.plot(df_pre_s['epoch'], df_pre_s['acc'], label='pretrain small')
# plt.plot(df_pre_m['epoch'], df_pre_m['acc'], label='pretrain medium')
plt.legend()
plt.title('acc')
plt.show()

plt.close()

plt.plot(df_scr['epoch'], df_scr['loss'], label='scratch')
plt.plot(df_pre_cb['epoch'], df_pre_cb['loss'], label='pretrain cifar backbone')
plt.plot(df_pre_c['epoch'], df_pre_c['loss'], label='pretrain cifar')
plt.plot(df_pre_s['epoch'], df_pre_s['loss'], label='pretrain small')
# plt.plot(df_pre_m['epoch'], df_pre_m['loss'], label='pretrain medium')
plt.legend()
plt.title('loss')
plt.show()





