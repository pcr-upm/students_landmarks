#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import os
import torch
import numpy as np
from enum import Enum
from images_framework.src.alignment import Alignment
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)


class Backbone(Enum):
    SHG = 'SHG'


class StudentsLandmarks(Alignment):
    """
    Face alignment using an Stacked Hourglass algorithm
    """
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.model = None
        self.device = None
        self.backbone = None
        self.indices = None
        self.width = 256
        self.height = 256

    def parse_options(self, params):
        unknown = super().parse_options(params)
        import argparse
        parser = argparse.ArgumentParser(prog='StudentsLandmarks', add_help=False)
        parser.add_argument('--gpu', dest='gpu', type=int, action='append',
                            help='GPU ID (negative value indicates CPU).')
        parser.add_argument('--backbone', dest='backbone', required=True, choices=[x.value for x in Backbone],
                            help='Select backbone model.')
        parser.add_argument('--batch-size', dest='batch_size', type=int, default=16,
                            help='Number of images in each mini-batch.')
        parser.add_argument('--epochs', dest='epochs', type=int, default=100000,
                            help='Number of sweeps over the dataset to train.')
        parser.add_argument('--patience', dest='patience', type=int, default=40,
                            help='Number of epochs with no improvement after which training will be stopped.')
        args, unknown = parser.parse_known_args(unknown)
        print(parser.format_usage())
        mode_gpu = torch.cuda.is_available() and -1 not in args.gpu
        self.device = torch.device('cuda:{}'.format(args.gpu[0]) if mode_gpu else 'cpu')
        self.backbone = args.backbone
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.patience = args.patience
        if self.database in ['300w_public', '300w_private', '300wlp']:
            self.indices = [101, 102, 103, 104, 105, 106, 107, 108, 24, 110, 111, 112, 113, 114, 115, 116, 117, 1, 119, 2, 121, 3, 4, 124, 5, 126, 6, 128, 129, 130, 17, 16, 133, 134, 135, 18, 7, 138, 139, 8, 141, 142, 11, 144, 145, 12, 147, 148, 20, 150, 151, 22, 153, 154, 21, 156, 157, 23, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168]
        elif self.database in 'wflw':
            self.indices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 24, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 1, 134, 2, 136, 3, 138, 139, 140, 141, 4, 143, 5, 145, 6, 147, 148, 149, 150, 151, 152, 153, 17, 16, 156, 157, 158, 18, 7, 161, 9, 163, 8, 165, 10, 167, 11, 169, 13, 171, 12, 173, 14, 175, 20, 177, 178, 22, 180, 181, 21, 183, 184, 23, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197]
        else:
            raise ValueError('Database is not implemented')

    def train(self, anns_train, anns_valid):
        print('Training')

    def load(self, mode):
        from images_framework.src.constants import Modes
        from images_framework.alignment.students_landmarks.src.SHG.StackedHourglass_pl import LitSHG
        # Set up a neural network to train
        print('Load model')
        #self.model = LitSHG(num_modules=1, num_landmarks=97)
        if mode is Modes.TEST:
            model_file = self.path + 'data/' + self.database + '/ckpt/' + self.backbone + '.ckpt'
            print('Loading model from {}'.format(model_file))
            self.model = LitSHG.load_from_checkpoint(model_file).to(self.device)
            self.model.eval()

    def process(self, ann, pred):
        from torch.utils.data import DataLoader
        from images_framework.src.datasets import Database
        from images_framework.src.annotations import GenericLandmark
        from images_framework.alignment.landmarks import lps
        from images_framework.alignment.students_landmarks.src.utils import get_landmarks_local_softmax
        from images_framework.alignment.students_landmarks.src.dataloader import MyDataset
        datasets = [subclass().get_names() for subclass in Database.__subclasses__()]
        idx = [datasets.index(subset) for subset in datasets if self.database in subset]
        parts = Database.__subclasses__()[idx[0]]().get_landmarks()
        dataset = MyDataset([pred], image_size=(self.width, self.height))
        dl_test = DataLoader(dataset, batch_size=self.batch_size)
        with torch.no_grad():
            for index, batch in enumerate(dl_test):
                # Generate prediction
                input = batch['img'].float().permute(0, 3, 1, 2).to(self.device) / 255
                output = self.model(input)
                landmarks = get_landmarks_local_softmax(output, temperature=10, window=5, device=self.device).squeeze().cpu()
                # Save prediction
                obj_pred = pred.images[batch['idx_img']].objects[batch['idx_obj']]
                bbox_res = batch['bbox_res'][0]
                bbox = batch['bbox'][0]
                landmarks = landmarks / 0.5
                landmarks = (landmarks - bbox_res[0:2]) / bbox_res[2:4]
                landmarks = (landmarks * bbox[2:4]) + bbox[0:2]
                for idx, pt in enumerate(landmarks):
                    label = self.indices[idx]
                    lp = list(parts.keys())[next((ids for ids, xs in enumerate(parts.values()) for x in xs if x == label), None)]
                    obj_pred.add_landmark(GenericLandmark(label, lp, pt.numpy().tolist(), True), lps[type(lp)])
