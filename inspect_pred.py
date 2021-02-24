import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

# simsiam_baseline = 'eval_logs/simsiam-ham-experiment-resnet50_cifar_small_scratch_lr0.3.csv'
# simsiam_bit_small_ep100 = 'eval_logs/simsiam-ham-experiment-resnet50_cifar_small_pretrain_epoch100_best.csv'
# simsiam_bit_small_ep200 = 'eval_logs/simsiam-ham-experiment-resnet50_cifar_small_pretrain_epoch200_best.csv'
# simsiam_bit_medium = 'eval_logs/simsiam-ham-experiment-resnet50_cifar_medium_pretrain_lr0.3.csv'
#
# sup_bit_small = 'eval_logs/supervised-ham-experiment-resnet50_small_pretrain_lr0.3.csv'
# sup_bit_medium = 'eval_logs/supervised-ham-experiment-resnet50_medium_pretrain_lr0.3.csv'
#
# df_bl = pd.read_csv(simsiam_baseline)
# df_bit_s_ep100 = pd.read_csv(simsiam_bit_small_ep100)
# df_bit_s_ep200 = pd.read_csv(simsiam_bit_small_ep200)
# df_bit_m = pd.read_csv(simsiam_bit_medium)
#
# df_sup_bit_s = pd.read_csv(sup_bit_small)
# df_sup_bit_m = pd.read_csv(sup_bit_medium)

simsiam_scratch = 'eval_logs/simsiam-ham-resnet50_small_scratch-exp1/epoch50.csv'
simsiam_pre_ad = 'eval_logs/simsiam-ham-resnet50_small_pretrain_adapt-exp1/epoch50.csv'
simsiam_pre = 'eval_logs/simsiam-ham-resnet50_small_pretrain-exp1/epoch50.csv'

df_sim_scr = pd.read_csv(simsiam_scratch)
df_sim_pre = pd.read_csv(simsiam_pre)
df_sim_pre_ad = pd.read_csv(simsiam_pre_ad)

bit_small = 'eval_logs/bit-ham-experiment-resnet50_small-exp1/epoch_none.csv'
bit_medium = 'eval_logs/bit-ham-experiment-resnet50_medium-exp1/epoch_none.csv'

df_bit_small = pd.read_csv(bit_small)
df_bit_medium = pd.read_csv(bit_medium)


def make_confusion_matrix(df, exp):
    gt = df['gh']
    pred = df['pred']
    print(exp, f'acc: {accuracy_score(gt, pred)}', f'balanced acc: {balanced_accuracy_score(gt, pred)}', f'macro f1: {sum(f1_score(gt, pred, average=None)) / 7}')
    print(confusion_matrix(gt, pred))


# make_confusion_matrix(df_sup_bit_s, 'sup_bit-s')
# make_confusion_matrix(df_sup_bit_m, 'sup_bit-m')
#
# make_confusion_matrix(df_bl, 'simsiam_baseline')
# make_confusion_matrix(df_bit_s_ep100, 'simsiam_bit-s epoch100')
# make_confusion_matrix(df_bit_s_ep200, 'simsiam_bit-s epoch200')
# make_confusion_matrix(df_bit_m, 'simsiam_bit-m')

make_confusion_matrix(df_bit_small, 'bit-s')
make_confusion_matrix(df_bit_medium, 'bit-m')

make_confusion_matrix(df_sim_scr, 'simsiam-scratch')
make_confusion_matrix(df_sim_pre, 'simsiam-pre')
make_confusion_matrix(df_sim_pre_ad, 'simsiam-pre-ad')

