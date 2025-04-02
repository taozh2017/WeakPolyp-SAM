#!/usr/bin/python3
# coding=utf-8

import os
import pdb
import sys

# sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.ion()
from skimage import img_as_ubyte
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lib import dataset
from lib.CODNet import PolypNet_v2
import time
import logging as logger
from Eval1.eval_functions import *
import pdb

TAG = "Test-CFA"
SAVE_PATH = TAG
GPU_ID = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)

logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S', \
                   filename="test_%s.log" % (TAG), filemode="w")

DATASETS = ['./data/TestDataset/CVC-ClinicDB',
            './data/TestDataset/CVC-300',
            './data/TestDataset/CVC-ColonDB',
            './data/TestDataset/Kvasir',
            './data/TestDataset/ETIS-LaribPolypDB']


def Dice(pred, mask):
    inter = (pred * mask).sum()
    union = (pred + mask).sum()
    dice1 = (2 * inter + 1e-10) / (union + 1e-10)
    return dice1


class Test(object):
    def __init__(self, Dataset, datapath, Network, iee):
        ## dataset
        self.datapath = datapath.split("/")[-1]

        self.cfg = Dataset.Config(datapath=datapath, mode='test')
        self.data = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=True, num_workers=8)
        ## network
        self.net = PolypNet_v2()
        path = './Models/CFA' + '/model-' + str(iee) + '.pt'
        state_dict = torch.load(path)
        self.net.load_state_dict(state_dict)
        self.net.train(False)
        self.net.cuda()

    def accuracy(self):
        with torch.no_grad():
            mae, Dice1, cnt, number, dice, Sm, IoU = 0, 0, 0, 256, 0, 0, 0
            for image, mask, (H, W), maskpath in self.loader:
                image, mask = image.cuda().float(), mask.cuda().float()
                pred, _, _, _ = self.net(image)
                # pred = torch.sigmoid(out2)
                pred = pred[:, 1:2]
                torch.cuda.synchronize()
                ## MAE
                cnt += 1
                dice += Dice(pred, mask)
                mae += (pred - mask).abs().mean()
                Sm += StructureMeasure(pred, mask)
                _, _, _, dice1, _, iou = Fmeasure_calu(pred.cpu().numpy(), mask.cpu().numpy(), 0.5)
                Dice1 += dice1
                IoU += iou
                if cnt % 20 == 0:
                    print('Epoch=%.1f, Dice =%.4f, IoU =%.4f, Sm=%.4f,MAE=%.4f,dice=%.4f' % (
                        iee, dice / cnt, IoU / cnt, Sm / cnt, mae / cnt, Dice1 / cnt))
            msg = 'Epoch=%.1f, %s, Dice =%.4f, IoU =%.4f, Sm=%.4f,MAE=%.4f,dice=%.4f' % (
                iee, self.datapath, dice / cnt, IoU / cnt, Sm / cnt, mae / cnt, Dice1 / cnt)
            logger.info(msg)

    def save(self, iee):
        with torch.no_grad():
            s = str(iee)
            for image, mask, (H, W), name in self.loader:
                _, out2 = self.net(image.cuda().float())
                pred = F.interpolate(out2, size=(H, W), mode='bilinear', align_corners=False)
                # pred = (torch.sigmoid(out2[0, 0])).cpu().numpy()
                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                head = './map/Eval/{}/{}/'.format(TAG, s) + self.cfg.datapath.split('/')[-1]
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head + '/' + name[0], img_as_ubyte(pred))

    def save1(self):
        with torch.no_grad():
            for image, mask, (H, W), name in self.loader:
                _, out2 = self.net(image.cuda().float())
                pred = F.interpolate(out2, size=(H, W), mode='bilinear', align_corners=False)
                # pred = (torch.sigmoid(out2[0, 0])).cpu().numpy()
                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                head = './map/eval/{}/'.format(TAG) + self.cfg.datapath.split('/')[-1]
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head + '/' + name[0], img_as_ubyte(pred))


if __name__ == '__main__':
    # iee = 50
    for iee in range(50, 101, 10):
        for e in DATASETS:
            t = Test(dataset, e, PolypNet_v2, iee)
            t.accuracy()
