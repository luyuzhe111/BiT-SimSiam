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

args = get_args()

checkpoint = os.path.join('checkpoint', f'{args.name}.pth')

model = get_model(args.model).to(args.device)
model.load_state_dict(torch.load(checkpoint)['state_dict'])
model = model.backbone

train_loader = torch.utils.data.DataLoader(
    dataset=get_dataset(
        transform=get_aug(train=False, train_classifier=True, **args.aug_kwargs),
        train=True,
        **args.dataset_kwargs
    ),
    batch_size=args.eval.batch_size,
    shuffle=True,
    **args.dataloader_kwargs
)

test_loader = torch.utils.data.DataLoader(
    dataset=get_dataset(
        transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs),
        train=False,
        **args.dataset_kwargs
    ),
    batch_size=args.eval.batch_size,
    shuffle=False,
    **args.dataloader_kwargs
)


classifier = nn.Linear(in_features=model.output_dim, out_features=7, bias=True).to(args.device)

assert args.eval_from is not None

classifier = torch.nn.DataParallel(classifier)
# define optimizer
optimizer = get_optimizer(
    args.eval.optimizer.name, classifier,
    lr=args.eval.base_lr * args.eval.batch_size / 256,
    momentum=args.eval.optimizer.momentum,
    weight_decay=args.eval.optimizer.weight_decay)

# define lr scheduler
lr_scheduler = LR_Scheduler(
    optimizer,
    args.eval.warmup_epochs, args.eval.warmup_lr * args.eval.batch_size / 256,
    args.eval.num_epochs, args.eval.base_lr * args.eval.batch_size / 256,
                             args.eval.final_lr * args.eval.batch_size / 256,
    len(train_loader),
)

loss_meter = AverageMeter(name='Loss')
acc_meter = AverageMeter(name='Accuracy')

# Start training
global_progress = tqdm(range(0, args.eval.num_epochs), desc=f'Evaluating')
for epoch in global_progress:
    loss_meter.reset()
    model.eval()
    classifier.train()
    local_progress = tqdm(train_loader, desc=f'Epoch {epoch}/{args.eval.num_epochs}', disable=True)

    for idx, (images, labels) in enumerate(local_progress):
        classifier.zero_grad()
        with torch.no_grad():
            feature = model(images.to(args.device))

        preds = classifier(feature)

        loss = F.cross_entropy(preds, labels.to(args.device))

        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())
        lr = lr_scheduler.step()
        local_progress.set_postfix({'lr': lr, "loss": loss_meter.val, 'loss_avg': loss_meter.avg})


classifier.eval()
correct, total = 0, 0
acc_meter.reset()
pred_hist = []
tar_hist = []
for idx, (images, labels) in enumerate(test_loader):
    with torch.no_grad():
        feature = model(images.to(args.device))
        preds = classifier(feature).argmax(dim=1)

        pred_hist = np.concatenate((pred_hist, preds.data.cpu().numpy()), axis=0)
        tar_hist = np.concatenate((tar_hist, labels.data.numpy()), axis=0)

        correct = (preds == labels.to(args.device)).sum().item()
        acc_meter.update(correct / preds.shape[0])


print(f'Accuracy = {acc_meter.avg * 100:.2f}')
print(f'Balanced Accuracy = {balanced_accuracy_score(tar_hist, pred_hist) * 100}')
print(confusion_matrix(tar_hist, pred_hist))

df = pd.DataFrame(columns=['gh', 'pred'])
df['gh'] = tar_hist
df['pred'] = pred_hist
df.to_csv(os.path.join('eval_logs', f'{args.name}.csv'))