# SimSiam
We extend the PyTorch implementation by [Patrick Hua](https://github.com/PatrickHua/SimSiam.git.) for the paper [**Exploring Simple Siamese Representation Learning**](https://arxiv.org/abs/2011.10566) and evaluate the impact of supervised pretraing on SimSiam using the [HAM10000](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000.) dataset. Detailed results are included in our work [**Contrastive Learning Meets Transfer Learning: A Case Study In Medical Image Analysis**](https://arxiv.org/abs/2103.03166).



### Dependencies

If you don't have python 3 environment:
```
conda create -n simsiam python=3.8
conda activate simsiam
```
Then install the required packages:
```
pip install -r requirements.txt
```

### Run SimSiam

```
CUDA_VISIBLE_DEVICES=0 python main.py --data_dir ../Data/ --log_dir ../logs/ -c configs/simsiam_cifar.yaml --ckpt_dir ~/.cache/ --hide_progress
```

### Run SimCLR

```
CUDA_VISIBLE_DEVICES=1 python main.py --data_dir ../Data/ --log_dir ../logs/ -c configs/simclr_cifar.yaml --ckpt_dir ~/.cache/ --hide_progress
```

### Run BYOL
```
CUDA_VISIBLE_DEVICES=2 python main.py --data_dir ../Data/ --log_dir ../logs/ -c configs/byol_cifar.yaml --ckpt_dir ~/.cache/ --hide_progress
```

### TODO

- convert from data-parallel (DP) to distributed data-parallel (DDP)
- create PyPI package `pip install simsiam-pytorch`


If you find this repo helpful, please consider star so that I have the motivation to improve it.



