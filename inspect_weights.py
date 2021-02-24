import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import get_model
from arguments import get_args
from augmentations import get_aug
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler
from tools import AverageMeter
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import pandas as pd
import sys
from copy import deepcopy
from tools.focal_loss import FocalLoss

args = get_args()
ckpt_epoch = args.ckpt_epoch

if 'bit' not in args.name:
    checkpoint = os.path.join(args.log_dir, 'models', f'{args.name}_epoch{ckpt_epoch}.pth')
    model = get_model(args.model)
    model.load_state_dict(torch.load(checkpoint)['state_dict'])
    print(f'Loaded {args.name}_epoch{ckpt_epoch}.pth')
    model = model.backbone
else:
    model = get_model(args.model)
    model = model.backbone

print(model.body.block1[0].conv1.weight.min(), model.body.block1[0].conv1.weight.max())
print(model.body.block1[0].gn1.weight.min(), model.body.block1[0].gn1.weight.max())
