#!/usr/bin/python3
# coding=utf-8
import pdb
import sys
import datetime
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from data import dataset
from lib import dataset as val_dataset
# from lib.CEANet_1 import Net
from lib.CODNet import PolypNet_v2
from data.data_prefetcher import DataPrefetcher
from data.data_prefetcher import DataPrefetcher_val
import logging as logger
from skimage import img_as_ubyte
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from lib.lscloss import *
import numpy as np
from lib.tools import *
import matplotlib.pyplot as plt
# from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import argparse
import json
from typing import Any, Dict, List
# from segment_anything.util import get_boxes_from_mask
import time

torch.autograd.set_detect_anomaly(True)
sys.dont_write_bytecode = True
torch.backends.cudnn.enable = True
torch.backends.cudnn.benchmark = True
from scipy.fftpack import dct

# train_v1: CEAnet backbone
# CEA_3:no maxpool
# 1117 pixel =25 over

TAG = "Codnet"
Val_PATH = './data/ValDB/'
SAVE_PATH = './Models/{}/'.format(TAG)
temp_pic = './out/{}/net/'.format(TAG)
temp_sam = './out/{}/sam/'.format(TAG)

logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S', \
                   filename="train_%s.log" % (TAG), filemode="w")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

""" set lr """


def get_triangle_lr(base_lr, max_lr, total_steps, cur, ratio=1., \
                    annealing_decay=1e-2, momentums=[0.95, 0.85]):
    first = int(total_steps * ratio)
    last = total_steps - first
    min_lr = base_lr * annealing_decay
    cycle = np.floor(1 + cur / total_steps)
    x = np.abs(cur * 2.0 / total_steps - 2.0 * cycle + 1)
    if cur < first:
        lr = base_lr + (max_lr - base_lr) * np.maximum(0., 1.0 - x)
    else:
        lr = ((base_lr - min_lr) * cur + min_lr * first - base_lr * total_steps) / (first - total_steps)
    if isinstance(momentums, int):
        momentum = momentums
    else:
        if cur < first:
            momentum = momentums[0] + (momentums[1] - momentums[0]) * np.maximum(0., 1. - x)
        else:
            momentum = momentums[0]

    return lr, momentum


def Dice(pred, mask):
    inter = (pred * mask).sum()
    union = pred.sum() + mask.sum()
    dice = (2.0 * inter) / (union + 1e-10)
    return dice


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy(pred, mask, reduce=None, reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


BASE_LR = 1e-5
MAX_LR = 1e-2
batch = 12
l = 0.3

def train(Dataset, Network):
    ## dataset
    cfg = Dataset.Config(datapath='./data/TrainDB', savepath=SAVE_PATH, mode='train', batch=batch, lr=1e-3,
                         momen=0.9,
                         decay=5e-4, epoch=100)
    data = Dataset.Data(cfg)
    train_loader = DataLoader(data, batch_size=cfg.batch, shuffle=True, num_workers=8)

    ## network
    net = PolypNet_v2(channel=64).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=None, ignore_index=255, reduction='mean')
    net.train()
    net.cuda()
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone1' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer = torch.optim.SGD([{'params': base}, {'params': head}], lr=cfg.lr, momentum=cfg.momen,
                                weight_decay=cfg.decay, nesterov=True)
    global_step = 0
    db_size = len(train_loader)
    criterion.cuda()
    best_epoch = -1
    best_dice = 0
    # -------------------------- training ------------------------------------#
    for epoch in range(cfg.epoch):
        net.train()
        # sam.train()
        prefetcher = DataPrefetcher(train_loader)
        batch_idx = -1
        image, mask, img_sam, size, names = prefetcher.next()
        while image is not None:
            niter = epoch * db_size + batch_idx
            lr, momentum = get_triangle_lr(BASE_LR, MAX_LR, cfg.epoch * db_size, niter, ratio=1.)
            optimizer.param_groups[0]['lr'] = 0.1 * lr  # for backbone
            optimizer.param_groups[1]['lr'] = lr
            optimizer.momentum = momentum
            batch_idx += 1
            global_step += 1

            ######   pseudo label   ######
            gt = mask.squeeze(1).long()
            bg_label = gt.clone()
            fg_label = gt.clone()
            sam_label = gt.clone()
            bg_label[gt != 0] = 255
            fg_label[gt == 0] = 255
            sam_label[gt == 255] = 0

            ######  saliency structure consistency loss  ######
            image_scale = F.interpolate(image, scale_factor=0.3, mode='bilinear', align_corners=True)
            out2, out3, out4, out5 = net(image)
            out2_s, _, _, _ = net(image_scale)
            out2_scale = F.interpolate(out2[:, 1:2], scale_factor=0.3, mode='bilinear', align_corners=True)
            loss_ssc = SaliencyStructureConsistency(out2_s[:, 1:2], out2_scale, 0.85)
            loss2 = criterion(out2, fg_label) + criterion(out2, bg_label)
            loss3 = criterion(out3, fg_label) + criterion(out3, bg_label)
            loss4 = criterion(out4, fg_label) + criterion(out4, bg_label)
            loss5 = criterion(out5, fg_label) + criterion(out5, bg_label)

            pred_label = torch.squeeze(out2[:, 1:2], dim=1)
            # list_s = []
            for i, pack in enumerate(zip(img_sam, sam_label, pred_label)):
                img, scribble_box, pred_box = pack
                if (epoch % 20 == 0):
                    head = temp_pic + '{}/'.format(str(epoch))
                    if not os.path.exists(head):
                        os.makedirs(head)
                    cv2.imwrite(head + names[i], pred_box.detach().cpu().numpy() * 255)

            # ######  objective function  ######
            loss = loss2 + loss_ssc + loss3 * 0.6 + loss4 * 0.4 + loss5 * 0.2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                msg = '%s| %s | step:%d/%d/%d | lr=%.6f | loss=%.6f | loss2=%.6f  | loss_ssc =%.6f| loss3=%.6f  | loss4=%.6f | loss5 =%.6f  ' % (
                    SAVE_PATH, datetime.datetime.now(), global_step, epoch + 1, cfg.epoch,
                    optimizer.param_groups[1]['lr'],
                    loss.item(), loss2.item(),
                    loss_ssc.item(), loss3.item(), loss4.item(), loss5.item())
                print(msg)
                logger.info(msg)
            image, mask, img_sam, size, names = prefetcher.next()

        if (epoch + 1) % 1 == 0 or (epoch + 1) == cfg.epoch:
            latest_dice = val(val_dataset, net)
            msg_ = 'latest-epoch: %d | dice: %.6f' % ((epoch + 1), latest_dice)
            logger.info(msg_)
            if not os.path.exists(cfg.savepath):
                os.makedirs(cfg.savepath)
            if latest_dice > best_dice:
                best_dice = latest_dice
                best_epoch = epoch + 1
                torch.save(net.state_dict(), cfg.savepath + 'best.pt')
                msg_best = 'best-epoch: %d |best-dice: %.6f' % (best_epoch, best_dice)
                logger.info(msg_best)
        if not os.path.exists(cfg.savepath):
            os.makedirs(cfg.savepath)
        if (epoch + 1) % 2 == 0:
            torch.save(net.state_dict(), cfg.savepath + 'model-' + str(epoch + 1) + '.pt')


def Union_box(box1, box2):
    max_noisy = 25
    x0 = max(box1[0, 0].item() - max_noisy, box2[0, 0].item() - max_noisy / 4)
    y0 = max(box1[0, 1].item() - max_noisy, box2[0, 1].item() - max_noisy / 4)
    x1 = min(box1[0, 2].item() + max_noisy, box2[0, 2].item() + max_noisy / 4)
    y1 = min(box1[0, 3].item() + max_noisy, box2[0, 3].item() + max_noisy / 4)
    return torch.tensor([x0, y0, x1, y1], dtype=torch.float).unsqueeze(0)


def Filter(mask, label):
    overlap = mask[0, 0] & label
    similarity = torch.count_nonzero(overlap).item() / torch.count_nonzero(label).item()
    if similarity < 0.5:
        return torch.zeros_like(mask), 0
    else:
        return mask, 1


def val(Dataset, net):
    cfg = Dataset.Config(datapath=Val_PATH, mode='val')
    data = Dataset.Data(cfg)
    val_loader = DataLoader(data, batch_size=1, shuffle=True, num_workers=8)
    net.train(False)
    dice = 0
    cnt = 0
    prefetcher = DataPrefetcher_val(val_loader)
    image, mask, size, names = prefetcher.next()
    while image is not None:
        with torch.no_grad():
            out2, _, _, _ = net(image)
        pred = out2[:, 1:2]
        Dice_is = Dice(pred, mask)
        dice += Dice_is
        cnt += 1
        image, mask, size, names = prefetcher.next()
    print("dice_all =", dice)
    dice /= cnt
    return dice


if __name__ == '__main__':
    net = PolypNet_v2()
    train(dataset, net)
