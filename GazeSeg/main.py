import os
import argparse
import datetime
import random
from pathlib import Path
from tensorboardX import SummaryWriter
import numpy as np
import torch
from torch import nn
from Data import get_dataloader
from Gaze_Seg.engine_GNN import train_one_epoch
import time
from Gaze_Seg.models.ViGUNet import ViGUNet


def get_args_parser():
    parser = argparse.ArgumentParser('Gaze', add_help=False)
    parser.add_argument('--lr', default=0.00007, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--in_channels', default=3, type=int) # 3, 1
    parser.add_argument('--dataset', default='Kvasir', type=str) # Kvasir, NCI
    parser.add_argument('--output_dir', default='output/Kvasir_test/')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--GPU_ids', type=str, default='0', help='Ids of GPUs')
    parser.add_argument('--seed', default=3407, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    return parser

def main(args):
    writer = SummaryWriter(log_dir=args.output_dir + '/summary')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device(args.device)
    model = ViGUNet()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=0.5)

    print('Building dataset...')
    train_loader = get_dataloader(args, split="train", resize_label=True)
    print('Number of training images: {}'.format(len(train_loader) * args.batch_size))
    test_loader = get_dataloader(args, split="test", resize_label=True)
    print('Number of validation images: {}'.format(len(test_loader)))

    criterion = nn.BCEWithLogitsLoss(reduction="none")

    output_dir = Path(args.output_dir)
    start_time = time.time()
    for epoch in range(args.epochs):
        train_one_epoch(model, criterion, train_loader, optimizer, device, epoch, args, writer)

        lr_scheduler.step()

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']

            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 5 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                torch.save(model.state_dict(), checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.GPU_ids)
    main(args)