import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from models import get_model
from arguments import get_args
from augmentations import get_aug
from datasets import get_dataset


args = get_args()
ckpt_epochs = [i * 25 for i in range(1, 400 // 25 + 1)]

for ckpt_epoch in ckpt_epochs:
    if 'bit' not in args.name:
        checkpoint = os.path.join(args.log_dir, 'models', f'{args.name}_epoch{ckpt_epoch}.pth')
        model = get_model(args.model).cuda()
        model.load_state_dict(torch.load(checkpoint)['state_dict'])
        print(f'Loaded {args.name}_epoch{ckpt_epoch}.pth')
        model = model.backbone
    else:
        model = get_model(args.model)
        model = model.backbone

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

    # Start training
    y = test_loader.dataset.targets
    features = None
    # fine-tuning
    model.eval()
    local_progress = tqdm(test_loader)
    for idx, (images, labels) in enumerate(local_progress):
        with torch.no_grad():
            feature = model(images.cuda())
            features = feature if features is None else torch.cat((features, feature), 0)

    ft_columns = ['f' + str(i) for i in range(features.shape[1])]
    df = pd.DataFrame(features.cpu().numpy(), columns=ft_columns)
    df['y'] = y

    root_dir = os.path.join('logs', args.name, 'avgpool_features')
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    if 'bit' in args.name:
        df.to_csv(os.path.join(root_dir, 'features.csv'))
        break
    else:
        df.to_csv(os.path.join(root_dir, f'features{ckpt_epoch}.csv'))