#!/usr/bin/python3
# coding=utf-8

import cv2
import torch
import numpy as np


class Compose(object):
    def __init__(self, *ops):
        self.ops = ops

    def __call__(self, image, mask, label):
        for op in self.ops:
            image, mask, label = op(image, mask, label)
        return image, mask, label


class RGBDCompose(object):
    def __init__(self, *ops):
        self.ops = ops

    def __call__(self, image, depth, mask):
        for op in self.ops:
            image, depth, mask = op(image, depth, mask)
        return image, depth, mask


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask, label):
        image = (image - self.mean) / self.std
        return image, mask, label


class RGBDNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, depth, mask):
        image = (image - self.mean) / self.std
        depth = (depth - self.mean) / self.std
        mask /= 255
        return image, mask


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask, label):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_NEAREST)
        label = cv2.resize(label, dsize=(self.W, self.H), interpolation=cv2.INTER_NEAREST)
        return image, mask, label


class RandomCrop(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask, label):
        H, W, _ = image.shape
        xmin = np.random.randint(W - self.W + 1)
        ymin = np.random.randint(H - self.H + 1)
        image = image[ymin:ymin + self.H, xmin:xmin + self.W, :]
        mask = mask[ymin:ymin + self.H, xmin:xmin + self.W, :]
        label = label[ymin:ymin + self.H, xmin:xmin + self.W, :]
        return image, mask, label


class RandomHorizontalFlip(object):
    def __call__(self, image, mask, label):
        if np.random.randint(2) == 1:
            image = image[:, ::-1, :].copy()
            mask = mask[:, ::-1, :].copy()
            label = label[:, ::-1, :].copy()
        return image, mask, label


class ToTensor(object):
    def __call__(self, image, mask, label):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        mask = torch.from_numpy(mask)
        mask = mask.permute(2, 0, 1)
        label = torch.from_numpy(label)
        label = label.permute(2, 0, 1)
        return image, mask.mean(dim=0, keepdim=True), label
