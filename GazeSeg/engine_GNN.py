import datetime
import time

import numpy as np
import torch
import torchvision
from torch import nn
import utils.misc as utils
from Gaze_Seg.utils.loss import inner_contrastive_loss, cross_entropy_loss, uncertain_consistency_loss, refine_label, \
    deep_supervision_loss, compute_stable_region_cosine
from Gaze_Seg.utils.strong_aug import apply_strong_augmentations
from medpy.metric.binary import dc


class Visualize_train(nn.Module):
    def __init__(self):
        super().__init__()

    def save_image(self, image, tag, epoch, writer):
        image = (image - image.min()) / (image.max() - image.min() + 1e-6)
        grid = torchvision.utils.make_grid(torch.tensor(image), nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    def forward(self, img_list, img_aug_list, label_list, gaze_list, gaze_binary_list,
                output_list, output_aug_list, output_sig_list, output_d3_list,
                gaze_b_list, gaze_d1_list, gaze_d2_list, gaze_d3_list,
                PL_d3_list,
                    epoch, writer):
        self.save_image(img_list.float(), 'img', epoch, writer)
        self.save_image(img_aug_list.float(), 'img_aug', epoch, writer)
        self.save_image(label_list.float(), 'label', epoch, writer)
        self.save_image(gaze_list.float(), 'gaze', epoch, writer)
        self.save_image(gaze_binary_list.float(), 'gaze_binary', epoch, writer)
        self.save_image(output_list.float(), 'output', epoch, writer)
        self.save_image(output_aug_list.float(), 'output_aug', epoch, writer)
        self.save_image(output_sig_list.float(), 'output_sig', epoch, writer)
        self.save_image(output_d3_list.float(), 'output_d3', epoch, writer)
        self.save_image(gaze_b_list.float(), 'gaze_b', epoch, writer)
        self.save_image(gaze_d1_list.float(), 'gaze_d1', epoch, writer)
        self.save_image(gaze_d2_list.float(), 'gaze_d2', epoch, writer)
        self.save_image(gaze_d3_list.float(), 'gaze_d3', epoch, writer)
        self.save_image(PL_d3_list.float(), 'PL_d3', epoch, writer)


def train_one_epoch(model, criterion, train_loader, optimizer, device, epoch, args, writer):
    start_time = time.time()

    model.train()
    criterion.train()
    print('-' * 40)
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    total_steps = len(train_loader)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    step = 0

    img_list, img_aug_list, label_list, gaze_list, gaze_binary_list = [], [], [], [], []
    output_list, output_aug_list, output_sig_list, output_d3_list = [], [], [], []
    gaze_b_list, gaze_d1_list, gaze_d2_list, gaze_d3_list = [], [], [], []
    PL_d3_list = []

    dice_score_list = []
    for data in train_loader:
        start = time.time()
        img, label, gaze = data['image'], data['label'], data['pseudo_label']
        img, label, gaze = img.to(device), label.to(device), gaze.to(device)
        datatime = time.time() - start
        # -------------------------------------------
        # hyperparameter
        # -------------------------------------------
        t1, t2 = 0.3, 0.6
        temperature = 0.1
        stable_threshold = 0.981
        lambda_ = 0.5
        # -------------------------------------------
        # forward
        # -------------------------------------------
        features, outputs, output = model(img)
        f1_b, f1_d1, f1_d2, f1_d3 = features
        img_aug = apply_strong_augmentations(img, device)
        features_aug, outputs_aug, output_aug = model(img_aug)
        f2_b, f2_d1, f2_d2, f2_d3 = features_aug
        output_b, output_d1, output_d2, output_d3 = outputs
        # -------------------------------------------
        # resize gaze
        # -------------------------------------------
        gaze_binary = torch.where(gaze < t1, -1, gaze)
        gaze_binary = torch.where(gaze > t2, 1, gaze_binary)
        gaze_binary = torch.where((gaze >= t1) & (gaze <= t2), 0, gaze_binary)
        gaze_b = torch.nn.functional.interpolate(gaze_binary, size=(f1_b.shape[2], f1_b.shape[3]), mode='nearest')
        gaze_d1 = torch.nn.functional.interpolate(gaze_binary, size=(f1_d1.shape[2], f1_d1.shape[3]), mode='nearest')
        gaze_d2 = torch.nn.functional.interpolate(gaze_binary, size=(f1_d2.shape[2], f1_d2.shape[3]), mode='nearest')
        gaze_d3 = torch.nn.functional.interpolate(gaze_binary, size=(f1_d3.shape[2], f1_d3.shape[3]), mode='nearest')
        # -------------------------------------------
        # ce loss
        # -------------------------------------------
        ce_loss = cross_entropy_loss(output, output_aug, gaze, t1, t2)
        # -------------------------------------------
        # consistency loss for uncertain region
        # -------------------------------------------
        cons_loss_b = uncertain_consistency_loss(f1_b, f2_b, gaze_b)
        cons_loss_d1 = uncertain_consistency_loss(f1_d1, f2_d1, gaze_d1)
        cons_loss_d2 = uncertain_consistency_loss(f1_d2, f2_d2, gaze_d2)
        cons_loss_d3 = uncertain_consistency_loss(f1_d3, f2_d3, gaze_d3)
        cons_loss = 0.1 * cons_loss_b + 0.2 * cons_loss_d1 + 0.3 * cons_loss_d2 + 0.4 * cons_loss_d3
        # -------------------------------------------
        # pseudo label generation
        # -------------------------------------------
        PL_b = refine_label(f1_b, f2_b, gaze_b, stable_threshold)
        PL_d1 = refine_label(f1_d1, f2_d1, gaze_d1, stable_threshold)
        PL_d2 = refine_label(f1_d2, f2_d2, gaze_d2, stable_threshold)
        PL_d3 = refine_label(f1_d3, f2_d3, gaze_d3, stable_threshold)
        # -------------------------------------------
        # deep supervision loss
        # -------------------------------------------
        ds_loss_b = deep_supervision_loss(output_b, PL_b)
        ds_loss_d1 = deep_supervision_loss(output_d1, PL_d1)
        ds_loss_d2 = deep_supervision_loss(output_d2, PL_d2)
        ds_loss_d3 = deep_supervision_loss(output_d3, PL_d3)
        ds_loss = 0.1 * ds_loss_b + 0.2 * ds_loss_d1 + 0.3 * ds_loss_d2 + 0.4 * ds_loss_d3
        # -------------------------------------------
        # contrastive loss
        # -------------------------------------------
        loss_contrast_b = inner_contrastive_loss(f1_b, PL_b, temperature)
        loss_contrast_d1 = inner_contrastive_loss(f1_d1, PL_d1, temperature)
        loss_contrast_d2 = inner_contrastive_loss(f1_d2, PL_d2, temperature)
        loss_contrast_d3 = inner_contrastive_loss(f1_d3, PL_d3, temperature)
        loss_contrast = 0.1 * loss_contrast_b + 0.2 * loss_contrast_d1 + 0.3 * loss_contrast_d2 + 0.4 * loss_contrast_d3
        # -------------------------------------------
        # total loss
        # -------------------------------------------
        loss = ce_loss + lambda_ * cons_loss + lambda_ * ds_loss + lambda_ * loss_contrast

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 30 == 0:
            img_list.append(img[0].detach())
            img_aug_list.append(img_aug[0].detach())
            label_list.append(label[0].detach())
            gaze_list.append(gaze[0].detach())
            output_list.append(output[0].detach())
            output_aug_list.append(output_aug[0].detach())
            output_d3_list.append(torch.where(nn.Sigmoid()(output_d3[0]) > 0.5, 1, 0).detach())
            gaze_b_list.append(gaze_b[0].detach())
            gaze_d1_list.append(gaze_d1[0].detach())
            gaze_d2_list.append(gaze_d2[0].detach())
            gaze_d3_list.append(gaze_d3[0].detach())
            PL_d3_list.append(PL_d3[0].detach())
            output_sig_list.append(torch.where(nn.Sigmoid()(output[0]) > 0.5, 1, 0).detach())
            gaze_binary_list.append(gaze_binary[0].detach())

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss)
        metric_logger.update(ce_loss=ce_loss)
        metric_logger.update(cons_loss=cons_loss)
        metric_logger.update(ds_loss=ds_loss)
        metric_logger.update(contrast_loss=loss_contrast)

        itertime = time.time() - start
        metric_logger.log_every(step, total_steps, datatime, itertime, print_freq, header)
        step = step + 1

        output = torch.where(nn.Sigmoid()(output) > 0.5, 1, 0)
        dice_score_list.append(dc(label, output))

    # gather the stats from all processes
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str, total_time / total_steps))
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    visual_train = Visualize_train()
    dice_score = np.array(dice_score_list).mean()

    writer.add_scalar('loss', loss.item(), epoch)
    writer.add_scalar('ce_loss', ce_loss.item(), epoch)
    writer.add_scalar('cons_loss', cons_loss.item(), epoch)
    writer.add_scalar('ds_loss', ds_loss.item(), epoch)
    writer.add_scalar('contrast_loss', loss_contrast.item(), epoch)
    writer.add_scalar('Train Dice Score', dice_score, epoch)

    visual_train(torch.stack(img_list), torch.stack(img_aug_list), torch.stack(label_list), torch.stack(gaze_list), torch.stack(gaze_binary_list),
                 torch.stack(output_list), torch.stack(output_aug_list), torch.stack(output_sig_list), torch.stack(output_d3_list),
                 torch.stack(gaze_b_list), torch.stack(gaze_d1_list), torch.stack(gaze_d2_list), torch.stack(gaze_d3_list),
                 torch.stack(PL_d3_list),
                  epoch, writer)