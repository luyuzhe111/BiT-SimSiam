import os
import time
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams['font.size'] = 14

ckpt_epochs = [25, 400]
# ckpt_epochs = [i for i in range(400, 401, 25)]
# ckpt_epochs = [200]
# exp_name = 'simsiam-ham-resnet50_medium_pretrain-exp1'
exp_name = 'bit-ham-experiment-resnet50_medium-exp1'
# exp_name = 'simsiam-cifar10-experiment-resnet50_gn_small_scratch'
tsne_dir = f'miccai/plots/{exp_name}_tsne'
if not os.path.exists(tsne_dir):
    os.mkdir(tsne_dir)

for ckpt_epoch in ckpt_epochs:
    if 'bit' in exp_name:
        df = pd.read_csv(f'logs/{exp_name}/avgpool_features/features.csv', index_col=0)
    else:
        df = pd.read_csv(f'logs/{exp_name}/avgpool_features/features{ckpt_epoch}.csv', index_col=0)
    ft_columns = ['f' + str(i) for i in range(df.shape[1] - 1)]

    if 'cifar' in exp_name:
        df = df.sample(frac=1)
        df = df.sample(n=2000)

    if 'ham' in exp_name:
        label_dict = {0: 'MEL', 1: 'NV', 2:'BCC', 3:'AKIEC',  4:'BKL', 5:'DF', 6:'VASC'}
        df['y'] = df['y'].apply(lambda x:label_dict[x])

    # df = df[df['y'] != 'NV']
    df['y'] = df['y'].apply(lambda x: 'NV' if x == 'NV' else 'Non-NV')
    data_subset = df[ft_columns].values

    time_start = time.time()
    # perplexity 30 for cifar & ham 7 / 2 ; 20 for ham 6
    perplexity = 30
    tsne = TSNE(n_components=2, verbose=1, random_state=0, perplexity=perplexity, n_iter=500)
    tsne_results = tsne.fit_transform(data_subset)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    df['tsne-2d-one'] = (tsne_results[:,0] - tsne_results[:,0].min()) / (tsne_results[:,0].max() - tsne_results[:,0].min())
    df['tsne-2d-two'] = (tsne_results[:,1] - tsne_results[:,1].min()) / (tsne_results[:,1].max() - tsne_results[:,1].min())

    # df['label'] = df['y'].apply(lambda x:1 if x != 1 else 0)
    plt.figure(figsize=(10, 10))
    ax = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        #palette=['r', 'b', 'c', 'y', 'g', 'm'],
        #palette=['b', 'g', 'r', 'c', 'm', 'y'],
        # palette=sns.color_palette('hls', 6),
        palette=['r', 'b'],
        data=df,
        s=50,
        legend="full",
        alpha=0.8
    )
    # plt.title(perplexity)
    ax.set(xticklabels=[])
    ax.set(yticklabels=[])
    ax.set(xlabel=None)
    ax.set(ylabel=None)
    ax.tick_params(bottom=False, left=False)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[0:], labels=labels[0:])

    plt.tight_layout()
    # plt.title(f'{exp_name}_{perplexity}')
    plt.savefig(f'{tsne_dir}/{ckpt_epoch}.svg')
    plt.show()

    if 'bit' in exp_name:
        break
