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
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
import pandas as pd
from copy import deepcopy
from tools.focal_loss import FocalLoss

args = get_args()
ckpt_epoch = args.ckpt_epoch

if 'bit' in args.name or 'common' in args.name:
    model = get_model(args.model).to(args.device)
    model = model.backbone

else:
    checkpoint = os.path.join(args.log_dir, 'models', f'{args.name}_epoch{ckpt_epoch}.pth')
    model = get_model(args.model).to(args.device)
    model.load_state_dict(torch.load(checkpoint)['state_dict'])
    print(f'Loaded {args.name}_epoch{ckpt_epoch}.pth')
    model = model.backbone

dataset = get_dataset(
        transform=get_aug(train=False, train_classifier=True, **args.aug_kwargs),
        train=False,
        validation=True,
        finetune=True,
        **args.dataset_kwargs
        )

train_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    # sampler=ImbalancedDatasetSampler(dataset),
    batch_size=args.eval.batch_size,
    shuffle=True,
    num_workers=4
)

val_loader = torch.utils.data.DataLoader(
    dataset=get_dataset(
        transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs),
        train=False,
        validation=True,
        **args.dataset_kwargs
    ),
    batch_size=args.eval.batch_size,
    shuffle=False,
    num_workers=4
)

test_loader = torch.utils.data.DataLoader(
    dataset=get_dataset(
        transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs),
        train=False,
        **args.dataset_kwargs
    ),
    batch_size=args.eval.batch_size,
    shuffle=False,
    num_workers=4
)


classifier = nn.Linear(in_features=model.output_dim, out_features=7, bias=True).to(args.device)

# assert args.eval_from is not None

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

criterion = FocalLoss(gamma=2)

loss_meter = AverageMeter(name='Loss')
acc_meter = AverageMeter(name='Accuracy')

# Start training
global_progress = tqdm(range(0, args.eval.num_epochs), desc=f'Evaluating')
best_b_acc = 0
best_classifier = deepcopy(classifier)

for epoch in global_progress:
    # fine-tuning
    loss_meter.reset()
    model.eval()
    classifier.train()
    local_progress = tqdm(train_loader, desc=f'Epoch {epoch}/{args.eval.num_epochs}')
    pred_hist = []
    tar_hist = []
    for idx, (images, labels) in enumerate(local_progress):
        classifier.zero_grad()
        with torch.no_grad():
            feature = model(images.to(args.device))

        preds = classifier(feature)
        # loss = F.cross_entropy(preds, labels.to(args.device))
        loss = criterion(preds, labels.to(args.device))

        sm = nn.Softmax(dim=1)
        pred_tar = sm(preds).max(dim=1)[1].cpu()

        pred_hist = np.concatenate((pred_hist, pred_tar.numpy()), axis=0)
        tar_hist = np.concatenate((tar_hist, labels.data.numpy()), axis=0)

        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())
        lr = lr_scheduler.step()
        local_progress.set_postfix({'lr': lr, "loss": loss_meter.val, 'loss_avg': loss_meter.avg})

    print(f'loss: {loss_meter.val} | loss_avg: {loss_meter.avg} | b_acc: {balanced_accuracy_score(tar_hist, pred_hist) * 100}')

    # validation
    loss_meter.reset()
    pred_hist = []
    tar_hist = []
    classifier.eval()
    val_progress = tqdm(val_loader, desc=f'Validation {epoch}/{args.eval.num_epochs}')
    for idx, (images, labels) in enumerate(val_progress):
        with torch.no_grad():
            feature = model(images.to(args.device))
            preds = classifier(feature)

            # loss = F.cross_entropy(preds, labels.to(args.device))
            loss = criterion(preds, labels.to(args.device))

            sm = nn.Softmax(dim=1)
            pred_tar = sm(preds).max(dim=1)[1].cpu()

            pred_hist = np.concatenate((pred_hist, pred_tar.numpy()), axis=0)
            tar_hist = np.concatenate((tar_hist, labels.data.numpy()), axis=0)

            loss_meter.update(loss.item())
            val_progress.set_postfix({"loss": loss_meter.val, 'loss_avg': loss_meter.avg})

    b_acc = balanced_accuracy_score(tar_hist, pred_hist)
    best_b_acc = max(best_b_acc, b_acc)
    if best_b_acc == b_acc:
        best_classifier = deepcopy(classifier)

    print(f'loss: {loss_meter.val} | loss_avg: {loss_meter.avg} | b_acc: {b_acc * 100}\n')


best_classifier.eval()
correct, total = 0, 0
acc_meter.reset()
pred_hist = []
tar_hist = []
for idx, (images, labels) in enumerate(test_loader):
    with torch.no_grad():
        feature = model(images.to(args.device))
        preds = best_classifier(feature).argmax(dim=1)

        pred_hist = np.concatenate((pred_hist, preds.data.cpu().numpy()), axis=0)
        tar_hist = np.concatenate((tar_hist, labels.data.numpy()), axis=0)

        correct = (preds == labels.to(args.device)).sum().item()
        acc_meter.update(correct / preds.shape[0])


print(f'Balanced Accuracy = {balanced_accuracy_score(tar_hist, pred_hist) * 100}')
print(f'Balanced f1 = {sum(f1_score(tar_hist, pred_hist, average=None)) / len(f1_score(tar_hist, pred_hist, average=None)) * 100}')
print(confusion_matrix(tar_hist, pred_hist))

df = pd.DataFrame(columns=['gh', 'pred'])
df['gh'] = tar_hist
df['pred'] = pred_hist
root_dir = os.path.join('eval_logs', args.name)
if not os.path.exists(root_dir):
    os.mkdir(root_dir)
df.to_csv(os.path.join(root_dir, f'epoch{ckpt_epoch}_fl_2.csv'))