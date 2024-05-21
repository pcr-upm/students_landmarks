#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import cv2
import numpy as np
from PIL import Image
from enum import Enum
from torch.utils.data import Dataset
from torchvision import transforms
from images_framework.alignment.students_landmarks.src.transformations import HorizontalFlipAug, RSTAug, OcclusionAug, LightingAug, BlurAug, TargetCropAug, ToOpencv, Heatmaps


class Mode(Enum):
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'


class MyDataset(Dataset):
    """
    Create a dataset class for our face landmarks data sets.
    """
    def __init__(self, anns, database, indices, image_size, mode: Mode):
        self.database = database
        self.indices = indices
        self.image_size = image_size
        self.mode = mode
        # Set data information
        self.img_indices, self.obj_indices = [], []
        self.filepaths, self.bboxes, self.landmarks = [], [], []
        for ann in anns:
            for img_idx, img_ann in enumerate(ann.images):
                for obj_idx, obj_ann in enumerate(img_ann.objects):
                    self.img_indices.append(img_idx)
                    self.obj_indices.append(obj_idx)
                    self.filepaths.append(img_ann.filename)
                    self.bboxes.append(np.array([obj_ann.bb[0], obj_ann.bb[1], obj_ann.bb[2]-obj_ann.bb[0], obj_ann.bb[3]-obj_ann.bb[1]]))
                    self.landmarks.append(obj_ann.landmarks)
                    #landmarks = map(str, [';'.join(map(str, [lnd.label, ';'.join(map(str, lnd.pos)), lnd.visible, lnd.confidence])) for lnds in [landmarks for lps in obj.landmarks.values() for landmarks in lps.values()] for lnd in lnds])

    def __len__(self):
        # Returns the length of the dataset
        return len(self.filepaths)

    def __getitem__(self, idx):
        # Load image
        # This is memory efficient because all the images are not stored in the memory at once but read as required
        image = cv2.imread(self.filepaths[idx], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        sample = {'img': image, 'idx_img': self.img_indices[idx], 'idx_obj': self.obj_indices[idx], 'bbox': self.bboxes[idx]}
        # Composes several transforms together
        if self.mode == Mode.TRAIN:
            ops = [HorizontalFlipAug(self.database), RSTAug(), OcclusionAug(), LightingAug(), BlurAug(), TargetCropAug(self.image_size), ToOpencv(), Heatmaps(len(self.indices))]
        elif self.mode == Mode.VALID:
            ops = [TargetCropAug(self.image_size), ToOpencv(), Heatmaps(len(self.indices))]
        else:
            ops = [TargetCropAug(self.image_size, (128, 128), 1.6), ToOpencv()]
        sample = transforms.Compose(ops)(sample)
        return sample
