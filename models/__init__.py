from .simsiam import SimSiam
from .byol import BYOL
from .simclr import SimCLR
from torchvision.models import resnet50, resnet18
import numpy as np
import os
import torch
from .backbones import resnet18_cifar_variant1, resnet18_cifar_variant2, resnet50_cifar_small_scratch

def get_backbone(backbone, pretrain=False, pretrain_dir=None, variant=None, castrate=True):
    backbone = eval(f"{backbone}(pretrain={pretrain})")
    # backbone = resnet18_cifar_variant1()
    # backbone = eval(f"{backbone}()")

    if castrate:
        backbone.output_dim = backbone.head.conv.out_channels
        backbone.fc = torch.nn.Identity()

        # backbone.output_dim = backbone.fc.in_features
        # backbone.fc = torch.nn.Identity()

    return backbone


def get_model(model_cfg):    

    if model_cfg.name == 'simsiam':
        model = SimSiam(get_backbone(model_cfg.backbone, pretrain=model_cfg.pretrain))
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






