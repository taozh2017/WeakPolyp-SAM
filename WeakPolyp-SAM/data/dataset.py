#!/usr/bin/python3
# coding=utf-8

import os
import os.path as osp
import cv2
import torch
import numpy as np

try:
    from . import transform
except:
    import transform

from torch.utils.data import Dataset, DataLoader
from lib.data_prefetcher import DataPrefetcher


class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for k, v in self.kwargs.items():
            print('%-10s: %s' % (k, v))
        if 'TrainDB1' in self.kwargs['datapath']:
            self.mean = np.array([[[126.57, 76.71, 54.92]]])
            self.std = np.array([[[81.53, 55.24, 43.98]]])
        elif 'CVC-300' in self.kwargs['datapath']:
            self.mean = np.array([[[117.20, 77.57, 61.88]]])
            self.std = np.array([[[78.79, 58.32, 48.90]]])
        elif 'CVC-ClinicDB' in self.kwargs['datapath']:
            self.mean = np.array([[[106.54, 70.57, 47.06]]])
            self.std = np.array([[[76.42, 50.91, 33.18]]])
        elif 'CVC-ColonDB' in self.kwargs['datapath']:
            self.mean = np.array([[[112.41, 73.09, 47.80]]])
            self.std = np.array([[[79.22, 58.37, 42.32]]])
        elif 'ETIS-LaribPolypDB' in self.kwargs['datapath']:
            self.mean = np.array([[[153.37, 109.91, 94.88]]])
            self.std = np.array([[[67.40, 60.72, 56.59]]])
        elif 'Kvasir' in self.kwargs['datapath']:
            self.mean = np.array([[[143.38, 83.18, 61.45]]])
            self.std = np.array([[[81.50, 57.41, 48.67]]])
        else:
            # raise ValueError
            self.mean = np.array([[[0.485 * 256, 0.296 * 256, 0.210 * 256]]])
            self.std = np.array([[[0.316 * 256, 0.214 * 256, 0.205 * 256]]])

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


class Data(Dataset):
    def __init__(self, cfg):
        image_extensions = ['.png', '.jpg']
        self.samples = []
        with open(cfg.datapath + '/' + cfg.mode + '.txt', 'r') as lines:
            for line in lines:
                imagepath_base = os.path.join(cfg.datapath, 'image', line.strip())
                maskpath_base = os.path.join(cfg.datapath, 'scribble', line.strip())
                sampath_base = os.path.join(cfg.datapath, 'sam', line.strip())

                imagepath = None
                maskpath = None
                sampath = None

                for ext in image_extensions:
                    if os.path.exists(imagepath_base + ext):
                        imagepath = imagepath_base + ext
                    if os.path.exists(maskpath_base + ext):
                        maskpath = maskpath_base + ext
                    if os.path.exists(sampath_base + ext):
                        sampath = sampath_base + ext

                if imagepath and maskpath and sampath:
                    self.samples.append([imagepath, maskpath, sampath])

        if cfg.mode == 'train':
            self.transform = transform.Compose(transform.Normalize(mean=cfg.mean, std=cfg.std),
                                               transform.Resize(320, 320),
                                               # transform.RandomHorizontalFlip(),
                                               transform.RandomCrop(320, 320),
                                               transform.ToTensor())
        elif cfg.mode == 'test':
            self.transform = transform.Compose(transform.Normalize(mean=cfg.mean, std=cfg.std),
                                               transform.Resize(320, 320),
                                               transform.ToTensor())
        else:
            raise ValueError

    def __getitem__(self, idx):
        imagepath, maskpath, labelpath = self.samples[idx]
        image = cv2.imread(imagepath).astype(np.float32)[:, :, ::-1]
        mask = cv2.imread(maskpath).astype(np.float32)[:, :, ::-1]
        label = cv2.imread(labelpath).astype(np.float32)[:, :, ::-1]
        H, W, C = mask.shape
        image, mask, label = self.transform(image, mask, label)
        mask[mask == 0.] = 255.
        mask[mask == 2.] = 0.
        # label = torch.where(label > 127, torch.ones_like(label), torch.zeros_like(label))
        return image, mask, label, (H, W), maskpath.split('/')[-1]

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plt.ion()

    cfg = Config(mode='train', datapath='./data/TrainDB')
    data = Data(cfg)
    loader = DataLoader(data, batch_size=1, shuffle=True, num_workers=8)
    prefetcher = DataPrefetcher(loader)
    batch_idx = -1
    image, mask = prefetcher.next()
    image = image[0].permute(1, 2, 0).cpu().numpy() * cfg.std + cfg.mean
    mask = mask[0].cpu().numpy()
    plt.subplot(121)
    plt.imshow(np.uint8(image))
    plt.subplot(122)
    plt.imshow(mask)
    input()
