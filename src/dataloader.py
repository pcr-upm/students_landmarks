#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TargetCropAug:
    def __init__(self, img_new_size, map_new_size, target_dist):
        self.new_size_x, self.new_size_y = img_new_size
        self.map_size_x, self.map_size_y = map_new_size
        self.target_dist = target_dist
        self.img2map_scale = False
        # Mismatch btween img shape and featuremap shape
        if self.map_size_x != self.new_size_x or self.map_size_y != self.new_size_y:
            self.img2map_scale = True
            self.map_scale_x = self.map_size_x / self.new_size_x
            self.map_scale_y = self.map_size_y / self.new_size_y
            self.map_scale_xx = self.map_scale_x * self.map_scale_x
            self.map_scale_xy = self.map_scale_x * self.map_scale_y
            self.map_scale_yy = self.map_scale_y * self.map_scale_y

    def __call__(self, sample):
        def _image_affine_trans(image, affine_transf, new_size=None):
            if not new_size:
                new_size = image.size
            A = affine_transf[0:2, 0:2]
            b = affine_transf[:, 2]
            inv_A = np.linalg.inv(A)  # we assume A invertible!
            inv_affine = np.zeros((2, 3))
            inv_affine[0:2, 0:2] = inv_A
            inv_affine[:, 2] = -inv_A.dot(b)
            new_image = image.transform(new_size, Image.AFFINE, inv_affine.flatten())
            return new_image

        def _bbox_affine_trans(bbox, affine_transf):
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

        x, y, w, h = sample['bbox']
        # we enlarge the area taken around the bounding box
        # it is neccesary to change the botton left point of the bounding box
        # according to the previous enlargement. Note this will NOT be the new
        # bounding box!
        # We return square images, which is neccesary since
        # all the images must have the same size in order to form batches
        side = max(w, h) * self.target_dist
        x -= (side - w) / 2
        y -= (side - h) / 2
        # center of the enlarged bounding box
        x0, y0 = x + side/2, y + side/2
        # homothety factor, chosen so the new horizontal dimension will
        # coincide with new_size
        mu_x = self.new_size_x / side
        mu_y = self.new_size_y / side
        # new_w, new_h = new_size, int(h * mu)
        new_w = self.new_size_x
        new_h = self.new_size_y
        new_x0, new_y0 = new_w / 2, new_h / 2
        # dilatation + translation
        affine_transf = np.array([[mu_x, 0, new_x0 - mu_x * x0], [0, mu_y, new_y0 - mu_y * y0]])
        sample['img'] = _image_affine_trans(sample['img'], affine_transf, (new_w, new_h))
        sample['bbox_res'] = _bbox_affine_trans(sample['bbox'], affine_transf)
        return sample


class ToOpencv:
    def __call__(self, sample):
        image = np.array(sample['img'])
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        sample['img'] = image
        return sample


class MyDataset(Dataset):
    def __init__(self, anns, image_size):
        self.idx_img = []
        self.idx_obj = []
        self.filepaths = []
        self.bboxes = []
        self.targets = []
        self.image_size = image_size
        for ann in anns:
            for idx_img, img_ann in enumerate(ann.images):
                for idx_obj, obj_ann in enumerate(img_ann.objects):
                    self.idx_img.append(idx_img)
                    self.idx_obj.append(idx_obj)
                    self.filepaths.append(img_ann.filename)
                    self.bboxes.append(np.array([obj_ann.bb[0], obj_ann.bb[1], obj_ann.bb[2]-obj_ann.bb[0], obj_ann.bb[3]-obj_ann.bb[1]]))
                    self.targets.append(obj_ann.landmarks)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, sample_idx):
        # Load image
        image = cv2.imread(self.filepaths[sample_idx], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        sample = {'img': image, 'idx_img': 0, 'idx_obj': 0, 'bbox': self.bboxes[sample_idx], 'bbox_res': self.bboxes[sample_idx]}
        sample = transforms.Compose([TargetCropAug(self.image_size, (128, 128), 1.6), ToOpencv()])(sample)
        return sample
