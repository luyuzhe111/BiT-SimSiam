import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
import pandas as pd
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from models import get_model
from tools import AverageMeter, knn_monitor, Logger, file_exist_check
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler
from linear_eval import main as linear_eval
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
from copy import deepcopy

def main(device, args):

    train_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=True, **args.aug_kwargs), 
            train=True,
            **args.dataset_kwargs),
        shuffle=True,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )
    memory_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs), 
            train=True,
            **args.dataset_kwargs),
        shuffle=False,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs),
            train=False,
            validation=True,
            **args.dataset_kwargs),
        shuffle=False,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )

    # define model
    model = get_model(args.model).to(device)

    # resume training
    if args.resume_from is not None:
        print(f'Resume from epoch: {args.resume_from}')
        checkpoint = os.path.join(args.log_dir, 'models', f'{args.name}_epoch{args.resume_from}.pth')
        model.load_state_dict(torch.load(checkpoint)['state_dict'])

    # if args.model.cifar_pretrain is True:
    #     cifar_epoch = 100
    #     pretrain = f'logs/simsiam-cifar10-experiment-resnet50_for_ham/models/simsiam-cifar10-experiment-resnet50_for_ham_epoch{cifar_epoch}.pth'
    #     if 'backbone' in args.name:
    #         pre_model = get_model(args.model).to(device)
    #         pre_model.load_state_dict(torch.load(pretrain)['state_dict'])
    #         # model.encoder = deepcopy(pre_model.encoder)
    #         # model.predictor = deepcopy(pre_model.predictor)
    #         # model.backbone = deepcopy(pre_model.backbone)
    #         model.backbone.root = deepcopy(pre_model.backbone.root)
    #         model.backbone.body = deepcopy(pre_model.backbone.body)
    #         model.backbone.head = deepcopy(pre_model.backbone.head)
    #         print('Load cifar pretrained model backbone')
    #     else:
    #         model.load_state_dict(torch.load(pretrain)['state_dict'])
    #         print('Load cifar pretrained model')

    model = torch.nn.DataParallel(model)
    scalar = GradScaler()

    # define optimizer
    optimizer = get_optimizer(
        args.train.optimizer.name, model, 
        lr=args.train.base_lr*args.train.batch_size/256, 
        momentum=args.train.optimizer.momentum,
        weight_decay=args.train.optimizer.weight_decay
    )

    lr_scheduler = LR_Scheduler(
        optimizer,
        args.train.warmup_epochs, args.train.warmup_lr*args.train.batch_size/256,
        args.train.num_epochs, args.train.base_lr*args.train.batch_size/256, args.train.final_lr*args.train.batch_size/256,
        len(train_loader),
        resume_from=args.resume_from,
        constant_predictor_lr=True # see the end of section 4.2 predictor
    )

    logger = Logger(tensorboard=args.logger.tensorboard, matplotlib=args.logger.matplotlib, log_dir=args.log_dir)

    epoch_hist = []
    loss_hist = []
    acc_hist = []

    accuracy = 0
    loss = 0
    # Start training
    start_epoch = 0 if args.resume_from is None else args.resume_from
    global_progress = tqdm(range(start_epoch, args.train.stop_at_epoch), desc=f'Training')
    for epoch in global_progress:
        model.train()
        local_progress = tqdm(train_loader, desc=f'Epoch {epoch}/{args.train.num_epochs}', disable=args.hide_progress)
        for idx, ((images1, images2), labels) in enumerate(local_progress):
            model.zero_grad()
            with torch.cuda.amp.autocast():
                data_dict = model.forward(images1.to(device, non_blocking=True), images2.to(device, non_blocking=True))
                loss = data_dict['loss'].mean() # ddp

            scalar.scale(loss).backward()
            scalar.step(optimizer)
            scalar.update()
            # loss.backward()
            # optimizer.step()
            lr_scheduler.step()
            data_dict.update({'lr':lr_scheduler.get_lr()})
            
            local_progress.set_postfix(data_dict)
            logger.update_scalers(data_dict)

        if args.train.knn_monitor and epoch % args.train.knn_interval == 0: 
            accuracy = knn_monitor(model.module.backbone, memory_loader, val_loader, epoch, logger_dir=args.log_dir, dataset=args.dataset.name, k=min(args.train.knn_k, len(memory_loader.dataset)), hide_progress=args.hide_progress)

        epoch_dict = {"epoch": epoch, "accuracy": accuracy}
        global_progress.set_postfix(epoch_dict)
        logger.update_scalers(epoch_dict)

        epoch_hist.append(epoch)
        acc_hist.append(accuracy)
        loss_hist.append(loss.item())

        if (epoch + 1) % 25 == 0:
            ckpt_dir = os.path.join(args.log_dir, 'models')
            if not os.path.exists(ckpt_dir):
                os.mkdir(ckpt_dir)
            model_path = os.path.join(ckpt_dir, f"{args.name}_epoch{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict()
            }, model_path)
            print(f"Model saved to {model_path}")

        # save csv logger
        if args.resume_from is not None:
            logger_csv = pd.read_csv(f'logs/{args.name}/logger.csv', index_col=0)
            logger_csv_new = pd.DataFrame({'epoch': epoch_hist, 'loss': loss_hist, 'acc': acc_hist})
            logger_csv = logger_csv.append(logger_csv_new, ignore_index=True)
        else:
            logger_csv = pd.DataFrame({'epoch': epoch_hist, 'loss': loss_hist, 'acc': acc_hist})

        logger_csv.to_csv(os.path.join(args.log_dir, 'logger_resume.csv'))
    
    # Save checkpoint
    ckpt_dir = os.path.join(args.log_dir, 'models')
    model_path = os.path.join(ckpt_dir, f"{args.name}.pth")  # datetime.now().strftime('%Y%m%d_%H%M%S')
    torch.save({
        'state_dict': model.module.state_dict()
    }, model_path)
    print(f"Model saved to {model_path}")

    # linear eval
    # if args.eval is not False:
    #     args.eval_from = model_path
    #     linear_eval(args)


if __name__ == "__main__":
    args = get_args()

    main(device=args.device, args=args)

    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')

    os.rename(args.log_dir, completed_log_dir)
    print(f'Log file has been saved to {completed_log_dir}')














