import pandas as pd


exp_name = 'bit-ham-experiment-resnet50_medium-exp1'
# exp_name = 'simsiam-ham-resnet50_small_scratch-exp1'
eval_logs = f'eval_logs/{exp_name}'

if 'bit' in exp_name:
    df1 = pd.read_csv(f'{eval_logs}/epochNone_fl.csv', index_col=0)
    df2 = pd.read_csv(f'{eval_logs}/epochNone_fl_2.csv', index_col=0)
    df3 = pd.read_csv(f'{eval_logs}/epochNone_fl_3.csv', index_col=0)
    gh = list(df1['gh']) + list(df2['gh']) + list(df3['gh'])
    pred = list(df1['pred']) + list(df2['pred']) + list(df3['pred'])
    df = pd.DataFrame({'gh': gh, 'pred': pred})
    df.to_csv(f'{eval_logs}/epoch_fl_ens.csv')

else:
    for i in range(25, 401, 25):
        df1 = pd.read_csv(f'{eval_logs}/epoch{i}_fl.csv', index_col=0)
        df2 = pd.read_csv(f'{eval_logs}/epoch{i}_fl_2.csv', index_col=0)
        df3 = pd.read_csv(f'{eval_logs}/epoch{i}_fl_3.csv', index_col=0)
        gh = list(df1['gh']) + list(df2['gh']) + list(df3['gh'])
        pred = list(df1['pred']) + list(df2['pred']) + list(df3['pred'])
        df = pd.DataFrame({'gh': gh, 'pred': pred})
        df.to_csv(f'{eval_logs}/epoch_{i}_fl_ens.csv')



