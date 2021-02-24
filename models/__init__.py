from .simsiam import SimSiam
from .byol import BYOL
from .simclr import SimCLR
from torchvision.models import resnet50, resnet18
import numpy as np
import os
from collections import OrderedDict
import torch
import torch.nn as nn
# from .backbones import resnet18_cifar_bn, resnet18_cifar_bn_ws, resnet18_cifar_gn, resnet18_cifar_gn_ws
from .backbones import simsiam_resnet50_small_scratch, simsiam_resnet50_medium_scratch
from .backbones import simsiam_cifar_resnet50_gn, simsiam_cifar_resnet50_bn
from .backbones import bit_s_resnet50, bit_m_resnet50

def get_backbone(backbone, pretrain=False, adapt=False, variant=None, castrate=True):
    backbone = eval(f"{backbone}(pretrain={pretrain}, adapt={adapt})")
    # backbone = resnet18_cifar_variant1()
    # backbone = eval(f"{backbone}()")

    if castrate:
        backbone.output_dim = 2048
        backbone.head = backbone.head[:3]
        backbone.fc = torch.nn.Identity()

        # backbone.output_dim = backbone.fc.in_features
        # backbone.fc = torch.nn.Identity()

    return backbone


def get_model(model_cfg):
    if model_cfg.name == 'simsiam':
        model = SimSiam(get_backbone(model_cfg.backbone, pretrain=model_cfg.pretrain, adapt=model_cfg.adapt))
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers)

    elif model_cfg.name == 'byol':
        model = BYOL(get_backbone(model_cfg.backbone))
    elif model_cfg.name == 'simclr':
        model = SimCLR(get_backbone(model_cfg.backbone))
    elif model_cfg.name == 'swav':
        raise NotImplementedError
    else:
        raise NotImplementedError
    return model






