import datetime
import os
import time
import random
import argparse
import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
import torch.nn.functional as F
import utils.misc as utils
from medpy.metric.binary import dc
from Data import get_dataloader
from GazeSeg.models.ViGUNet import ViGUNet


def get_args_parser():
    parser = argparse.ArgumentParser('Gaze', add_help=False)
    parser.add_argument('--in_channels', default=1, type=int) # 3, 1
    parser.add_argument('--dataset', default='NCI', type=str) # Kvasir, NCI
    parser.add_argument('--output_dir', default='output/NCI/')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format('0')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = ViGUNet()

    model_path = args.output_dir + '0.8077_13_checkpoint.pth'

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(args.device)
    model.eval()
    test_loader = get_dataloader(args, split="test", resize_label=False)
    output_save_dir = args.output_dir + "/output/"
    if not os.path.exists(output_save_dir):
        os.makedirs(output_save_dir)

    dice_score_list = []
    my_dice_score_list = []
    for data in test_loader:
        start = time.time()
        img, label = data['image'], data['label']
        img, label = img.to(args.device), label.to(args.device)

        _, _, output = model(img)
        output = F.interpolate(output, size=label.shape[2:], mode="bilinear")
        output = nn.Sigmoid()(output)
        output = torch.where(output > 0.5, 1, 0)

        dice_score_list.append(dc(label, output))

        reduce_axis = list(range(2, len(img.shape)))
        intersection = torch.sum(output * label, dim=reduce_axis)
        input_o = torch.sum(output, dim=reduce_axis)
        target_o = torch.sum(label, dim=reduce_axis)
        my_dice = torch.mean(2 * intersection / (input_o + target_o + 1e-10), dim=1)
        my_dice_score_list.append(my_dice.item())

        # save output
        output_img = np.array(output[0, 0].detach().cpu(), np.uint8) * 255
        output_img = Image.fromarray(output_img).convert('L')
        img_name = data['path'][0]
        output_img.save(output_save_dir + img_name.split('.')[0] + '.png')

    dice_score = np.array(dice_score_list).mean()
    dice_std = np.array(dice_score_list).std()
    print(dice_score, dice_std)

    dice_score = np.array(my_dice_score_list).mean()
    dice_std = np.array(my_dice_score_list).std()
    print(dice_score, dice_std)
