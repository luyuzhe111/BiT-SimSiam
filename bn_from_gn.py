from models.backbones import bit_s_resnet50, bit_m_resnet50
from models.backbones import simsiam_resnet50_small_scratch, simsiam_resnet50_medium_scratch
from datasets.ham_dataset import DataLoader
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from tools import AverageMeter
from torch.cuda.amp import autocast, GradScaler

bit_model = bit_m_resnet50(pretrain=True).cuda()
bit_model.head = bit_model.head[0:2]
simsiam_model = simsiam_resnet50_medium_scratch(pretrain=True).cuda()
simsiam_model.head = simsiam_model.head[0:2]
model = simsiam_model

imagenet_mean_std = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(*imagenet_mean_std)
])

data_dir = 'ham-10000/json/exp_1'
train_set = DataLoader(data_dir, transform=transform_train, split='train')
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss().cuda()

losses = AverageMeter(name='loss')

model.train()

model.root.conv.weight.requires_grad = False
for bname, block in model.body.named_children():
    for uname, unit in block.named_children():
        unit.conv1.weight.requires_grad = False
        unit.conv2.weight.requires_grad = False
        unit.conv3.weight.requires_grad = False

bit_model.eval()
scalar = GradScaler()
for i in range(1, 201):
    tbar = tqdm(train_loader, desc='\r')
    for batch_idx, (inputs, targets) in enumerate(tbar):
        inputs = inputs.cuda()
        model.zero_grad()
        with autocast():
            features = model(inputs)
            bit_features = bit_model(inputs)
            loss = criterion(features, bit_features)

        scalar.scale(loss).backward()
        scalar.step(optimizer)
        scalar.update()

        losses.update(loss.item(), inputs.size(0))
        tbar.set_description(f'\r Epoch: {i} | Train Loss: {losses.avg:.3f}')

    if i % 1 == 0:
        model.head.avg = nn.AdaptiveAvgPool2d(output_size=1)
        model.head.conv = nn.Conv2d(2048 * 1, 2048, kernel_size=1, bias=True)
        torch.save(model.state_dict(), os.path.join('models/sup_pretrain', 'Simsiam_Bit-M-R50x1.pth'))
        model.head = model.head[0:2]


model.head.conv = nn.Conv2d(2048 * 1, 2048, kernel_size=1, bias=True)
torch.save(model.state_dict(), os.path.join('models/sup_pretrain', 'Simsiam_Bit-M-R50x1.pth'))