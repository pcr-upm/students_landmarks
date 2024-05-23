#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import cv2
import numpy as np
from PIL import Image


class GeometryBase:
    def __call__(self, sample):
        raise NotImplementedError('Inheritance __call__ not defined')

    def map_affine_transformation(self, sample, affine_transf, new_size=None):
        sample['img'] = self._image_affine_trans(sample['img'], affine_transf, new_size)
        sample['bbox_res'] = self._bbox_affine_trans(sample['bbox'], affine_transf)
        if sample['landmarks'].size != 0:
            sample['landmarks'] = self._landmarks_affine_trans(sample['landmarks'], affine_transf)
        return sample

    def _image_affine_trans(self, image, affine_transf, new_size=None):
        def get_inverse_transf(affine_transf):
            A = affine_transf[0:2, 0:2]
            b = affine_transf[:, 2]
            inv_A = np.linalg.inv(A)  # we assume A invertible!
            inv_affine = np.zeros((2, 3))
            inv_affine[0:2, 0:2] = inv_A
            inv_affine[:, 2] = -inv_A.dot(b)
            return inv_affine
        if not new_size:
            new_size = image.size
        inv_affine_transf = get_inverse_transf(affine_transf)
        new_image = image.transform(new_size, Image.AFFINE, inv_affine_transf.flatten())
        return new_image

    def _bbox_affine_trans(self, bbox, affine_transf):
        x, y, w, h = bbox
        images_bb = []
        for point in ([x, y, 1], [x + w, y, 1], [x, y + h, 1], [x + w, y + h, 1]):
            images_bb.append(affine_transf.dot(point))
        images_bb = np.array(images_bb)
        new_corner0 = np.min(images_bb, axis=0)
        new_corner1 = np.max(images_bb, axis=0)
        new_x, new_y = new_corner0
        new_w, new_h = new_corner1 - new_corner0
        new_bbox = np.array((new_x, new_y, new_w, new_h))
        return new_bbox

    def _landmarks_affine_trans(self, landmarks, affine_transf):
        def affine2homogeneous(points):
            num_points = points.shape[0]
            hpoints = np.hstack((points, np.repeat(1, num_points).reshape(num_points, 1)))
            return hpoints
        homog_landmarks = affine2homogeneous(landmarks)
        new_landmarks = affine_transf.dot(homog_landmarks.T).T
        return new_landmarks


class CropBbox(GeometryBase):
    def __init__(self, img_new_size=(256, 256), map_new_size=(128, 128), target_dist=1.6):
        self.target_dist = target_dist
        self.new_size_x, self.new_size_y = self._convert_shapes(img_new_size)
        self.map_size_x, self.map_size_y = self._convert_shapes(map_new_size)
        self.img2map_scale = False
        # Mismatch between img shape and feature map shape
        if self.map_size_x != self.new_size_x or self.map_size_y != self.new_size_y:
            self.img2map_scale = True
            self.map_scale_x = self.map_size_x / self.new_size_x
            self.map_scale_y = self.map_size_y / self.new_size_y
            self.map_scale_xx = self.map_scale_x * self.map_scale_x
            self.map_scale_xy = self.map_scale_x * self.map_scale_y
            self.map_scale_yy = self.map_scale_y * self.map_scale_y

    def _convert_shapes(self, new_size):
        if isinstance(new_size, (tuple, list)):
            new_size_x = new_size[0]
            new_size_y = new_size[1]
        else:
            new_size_x = new_size
            new_size_y = new_size
        return new_size_x, new_size_y

    def __call__(self, sample):
        x, y, w, h = sample['bbox']
        # We enlarge the area taken around the bounding box it is necessary to change the bottom left point of the
        # bounding box according to the previous enlargement. Note this will NOT be the new bounding box!
        # Return square images, which is necessary since all the images must have the same size in order to form batches
        side = max(w, h) * self.target_dist
        x -= (side-w)/2
        y -= (side-h)/2
        # center of the enlarged bounding box
        x0, y0 = x+side/2, y+side/2
        # homothety factor, chosen so the new horizontal dimension will coincide with new_size
        mu_x = self.new_size_x/side
        mu_y = self.new_size_y/side
        # new_w, new_h = new_size, int(h * mu)
        new_w = self.new_size_x
        new_h = self.new_size_y
        new_x0, new_y0 = new_w/2, new_h/2
        # dilatation + translation
        affine_transf = np.array([[mu_x, 0, new_x0-mu_x*x0], [0, mu_y, new_y0-mu_y*y0]])
        sample = self.map_affine_transformation(sample, affine_transf, (new_w, new_h))
        sample['landmarks'] = np.round(sample['landmarks'])
        return sample


class Occlusion:
    def __init__(self, min_length=0.1, max_length=0.4, covar_scale_ratio=1., num_maps=1):
        import torch
        self.min_length = min_length
        self.max_length = max_length
        self.num_maps = num_maps
        self.covar_ratio = covar_scale_ratio
        self.covar_ratio_square = covar_scale_ratio*covar_scale_ratio
        self.covar_scale = torch.tensor([[self.covar_ratio, 0], [0, self.covar_ratio]]).repeat(self.num_maps, 1, 1).type(torch.DoubleTensor)

    def __call__(self, sample):
        x, y, w, h = sample['bbox']
        rnd_width = np.random.randint(int(w*self.min_length), int(w*self.max_length))
        rnd_height = np.random.randint(int(h*self.min_length), int(h*self.max_length))
        # (xi, yi) and (xf, yf) are the lower left points of the occlusion rectangle and the upper right point
        xi = int(x + np.random.randint(0, w-rnd_width))
        xf = int(xi + rnd_width)
        yi = int(y + np.random.randint(0, h-rnd_height))
        yf = int(yi + rnd_height)
        pixels = np.array(sample['img'])
        pixels[yi:yf, xi:xf, :] = np.random.uniform(0, 255, size=3)
        image = Image.fromarray(pixels)
        sample['img'] = image
        return sample


class Illumination:
    def __init__(self, hsv_range_min=(-0.5, -0.5, -0.5), hsv_range_max=(0.5, 0.5, 0.5)):
        self.hsv_range_min = hsv_range_min
        self.hsv_range_max = hsv_range_max

    def __call__(self, sample):
        # Convert to HSV colorspace from RGB colorspace
        image = np.array(sample['img'])
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # Generate new random values
        H = 1 + np.random.uniform(self.hsv_range_min[0], self.hsv_range_max[0])
        S = 1 + np.random.uniform(self.hsv_range_min[1], self.hsv_range_max[1])
        V = 1 + np.random.uniform(self.hsv_range_min[2], self.hsv_range_max[2])
        hsv[:, :, 0] = np.clip(H*hsv[:, :, 0], 0, 179)
        hsv[:, :, 1] = np.clip(S*hsv[:, :, 1], 0, 255)
        hsv[:, :, 2] = np.clip(V*hsv[:, :, 2], 0, 255)
        # Convert back to BGR colorspace
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        sample['img'] = Image.fromarray(image)
        return sample


class Blur:
    def __init__(self, blur_prob=0.5, blur_kernel_range=(0, 2)):
        self.blur_prob = blur_prob
        self.kernel_range = blur_kernel_range

    def __call__(self, sample):
        # Smooth image
        image = np.array(sample['img'])
        if np.random.uniform(0.0, 1.0) < self.blur_prob:
            kernel = np.random.random_integers(self.kernel_range[0], self.kernel_range[1])*2 + 1
            image = cv2.GaussianBlur(image, (kernel, kernel), 0, 0)
        sample['img'] = Image.fromarray(image)
        return sample


class ImgPermute:
    def __call__(self, sample):
        # Convert in a numpy array and change to BGR
        image = cv2.cvtColor(np.array(sample['img']), cv2.COLOR_RGB2BGR)
        # Converts a numpy image in H x W x C format to C x W x H format and changes the range to [0, 1]
        sample['img'] = image.transpose(2, 0, 1) / 255.0
        return sample


class Heatmaps:
    def __init__(self, num_maps, map_size=(128, 128), sigma=1.5, stride=1, norm=False):
        self.num_maps = num_maps
        self.sigma = sigma
        self.double_sigma_pw2 = 2*sigma*sigma
        self.doublepi_sigma_pw2 = self.double_sigma_pw2*np.pi
        self.stride = stride
        self.norm = norm
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
        exp_x = np.exp(-(self.grid_x-landmarks[:, 0].reshape(-1, 1))**2/self.double_sigma_pw2)
        exp_y = np.exp(-(self.grid_y-landmarks[:, 1].reshape(-1, 1))**2/self.double_sigma_pw2)
        heatmaps = np.matmul(exp_y.reshape(self.num_maps, self.height, 1), exp_x.reshape(self.num_maps, 1, self.width))
        if self.norm:
            heatmaps = heatmaps/self.doublepi_sigma_pw2
        sample['heatmaps'] = heatmaps
        return sample
