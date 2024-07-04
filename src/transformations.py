#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import cv2
import torch
import numpy as np


class Illumination:
    def __init__(self, hsv_range):
        self.hsv_range = hsv_range

    def __call__(self, sample):
        # Convert to HSV colorspace from BGR colorspace
        hsv = cv2.cvtColor(sample['img'], cv2.COLOR_RGB2HSV)
        # Generate new random values
        rnd_hue = np.random.uniform(-self.hsv_range[0], self.hsv_range[0]) + 1.0
        rnd_sat = np.random.uniform(-self.hsv_range[1], self.hsv_range[1]) + 1.0
        rnd_val = np.random.uniform(-self.hsv_range[2], self.hsv_range[2]) + 1.0
        hsv[:, :, 0] = np.clip(rnd_hue*hsv[:, :, 0], 0, 255)
        hsv[:, :, 1] = np.clip(rnd_sat*hsv[:, :, 1], 0, 255)
        hsv[:, :, 2] = np.clip(rnd_val*hsv[:, :, 2], 0, 255)
        # Convert back to BGR colorspace
        sample['img'] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return sample


class CropBbox:
    def __init__(self, width, height, bbox_scale):
        self.width = width
        self.height = height
        self.bbox_scale = bbox_scale

    def __call__(self, sample):
        bbox = sample['bbox']
        bbox_width = bbox[2]-bbox[0]
        bbox_height = bbox[3]-bbox[1]
        # Squared bbox required
        max_size = max(bbox_width, bbox_height)
        shift = (float(max_size-bbox_width)/2.0, float(max_size-bbox_height)/2.0)
        bbox_squared = (bbox[0]-shift[0], bbox[1]-shift[1], bbox[2]+shift[0], bbox[3]+shift[1])
        # import copy
        # aux = copy.deepcopy(sample['img'])
        # cv2.rectangle(aux, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0))
        # Enlarge bounding box
        shift = max_size*self.bbox_scale
        bbox_enlarged = (bbox_squared[0]-shift, bbox_squared[1]-shift, bbox_squared[2]+shift, bbox_squared[3]+shift)
        sample['bbox_enlarged'] = torch.tensor(bbox_enlarged)
        # cv2.rectangle(aux, (int(bbox_enlarged[0]), int(bbox_enlarged[1])), (int(bbox_enlarged[2]), int(bbox_enlarged[3])), (255, 255, 0))
        # cv2.imshow('aa', cv2.cvtColor(aux, cv2.COLOR_BGR2RGB))
        # Project image
        T = np.zeros((2, 3), dtype=float)
        T[0, 0], T[0, 1], T[0, 2] = 1, 0, -bbox_enlarged[0]
        T[1, 0], T[1, 1], T[1, 2] = 0, 1, -bbox_enlarged[1]
        bbox_width = bbox_enlarged[2]-bbox_enlarged[0]
        bbox_height = bbox_enlarged[3]-bbox_enlarged[1]
        S = np.matrix([[self.width/bbox_width, 0, 0], [0, self.height/bbox_height, 0]], dtype=float)
        face_translated = cv2.warpAffine(sample['img'], T, (int(round(bbox_width)), int(round(bbox_height))))
        sample['img'] = cv2.warpAffine(face_translated, S, (self.width, self.height))
        # Project landmarks
        num_landmarks = len(sample['landmarks'])
        if num_landmarks > 0:
            lnds_translated = np.transpose(T.dot(np.transpose(cv2.convertPointsToHomogeneous(sample['landmarks']).reshape(num_landmarks, 3))))
            sample['landmarks'] = np.transpose(S.dot(np.transpose(cv2.convertPointsToHomogeneous(lnds_translated).reshape(num_landmarks, 3))))
        # for lnd in sample['landmarks']:
        #     pt = np.array(lnd, dtype=int)[0]
        #     cv2.circle(sample['img'], pt, 3, (0, 255, 0))
        # cv2.imshow('img', cv2.cvtColor(sample['img'], cv2.COLOR_BGR2RGB))
        # cv2.waitKey(0)
        return sample


class ImgPermute:
    def __call__(self, sample):
        # Converts a numpy image in H x W x C format to C x W x H format and changes the range to [0, 1]
        sample['img'] = sample['img'].transpose(2, 0, 1) / 255.0
        return sample


class Heatmaps:
    def __init__(self, num_maps):
        map_size = (128, 128)
        sigma = 1.5
        stride = 1
        self.num_maps = num_maps
        self.sigma = sigma
        self.double_sigma = 2*sigma*sigma
        self.double_sigma_pi = self.double_sigma*np.pi
        self.stride = stride
        self.norm = False
        if isinstance(map_size, (tuple, list)):
            self.width = map_size[0]
            self.height = map_size[1]
        else:
            self.width = map_size
            self.height = map_size
        grid_x = np.arange(self.width)*stride + stride/2 - 0.5
        grid_y = np.arange(self.height)*stride + stride/2 - 0.5
        self.grid_x = np.repeat(grid_x.reshape(1, self.width), self.num_maps, axis=0)
        self.grid_y = np.repeat(grid_y.reshape(1, self.height), self.num_maps, axis=0)

    def __call__(self, sample):
        landmarks = sample['landmarks']
        landmarks = landmarks[-self.num_maps:]
        # Heatmap generation
        exp_x = np.exp(-(self.grid_x-landmarks[:, 0].reshape(-1, 1))**2/self.double_sigma)
        exp_y = np.exp(-(self.grid_y-landmarks[:, 1].reshape(-1, 1))**2/self.double_sigma)
        heatmaps = np.matmul(exp_y.reshape(self.num_maps, self.height, 1), exp_x.reshape(self.num_maps, 1, self.width))
        if self.norm:
            heatmaps = heatmaps/self.double_sigma_pi
        sample['heatmaps'] = heatmaps
        return sample
