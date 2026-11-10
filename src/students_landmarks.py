#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
import torch
import itertools
import numpy as np
from torch.utils.data import DataLoader
from pcr_framework.src.alignment import Alignment
from pcr_framework.src.datasets import Database
from src.pcrlogger import PCRLogger
from src.dataloader import Mode, Regressor, Backbone, MyDataset
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)


class StudentsLandmarks(Alignment):
    """
    Face alignment using a popular algorithm
    """
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.model = None
        self.gpus = None
        self.device = None
        self.regressor = None
        self.backbone = None
        self.parts = None
        self.indices = None
        self.batch_size = None
        self.epochs = None
        self.patience = None
        self.order = None
        self.width = None
        self.height = None

    def parse_options(self, params):
        unknown = super().parse_options(params)
        import argparse
        parser = argparse.ArgumentParser(prog='StudentsLandmarks', add_help=False)
        parser.add_argument('--gpu', dest='gpu', type=int, action='append',
                            help='GPU ID (negative value indicates CPU).')
        parser.add_argument('--regressor', dest='regressor', required=True, choices=[x.value for x in Regressor],
                            help='Select regressor model.')
        parser.add_argument('--backbone', dest='backbone', required=True, choices=[x.value for x in Backbone],
                            help='Select backbone architecture.')
        parser.add_argument('--batch-size', dest='batch_size', type=int, default=8,
                            help='Number of images in each mini-batch.')
        parser.add_argument('--epochs', dest='epochs', type=int, default=100,
                            help='Number of sweeps over the dataset to train.')
        parser.add_argument('--patience', dest='patience', type=int, default=20,
                            help='Number of epochs with no improvement after which training will be stopped.')
        args, unknown = parser.parse_known_args(unknown)
        print(parser.format_usage())
        mode_gpu = torch.cuda.is_available() and -1 not in args.gpu
        self.gpus = args.gpu
        self.device = torch.device('cuda' if mode_gpu else 'cpu')
        self.regressor = Regressor(args.regressor)
        self.backbone = Backbone(args.backbone)
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.patience = args.patience
        self.width, self.height = (224, 224) if self.regressor is Regressor.ENCODER and self.backbone in [Backbone.VIT] else (256, 256)
        datasets = [subclass().get_names() for subclass in Database.__subclasses__()]
        self.parts = Database.__subclasses__()[next(i for i, subset in enumerate(datasets) if self.database in subset)]().get_landmarks()
        self.indices = list(itertools.chain.from_iterable(self.parts.values()))

    def train(self, anns_train, anns_valid):
        import pytorch_lightning as pl
        from pytorch_lightning import loggers as pl_loggers
        from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
        # Prepare dataloaders
        dataset_train = MyDataset(anns_train, self.indices, self.regressor, self.width, self.height, Mode.TRAIN)
        dataset_valid = MyDataset(anns_valid, self.indices, self.regressor, self.width, self.height, Mode.VALID)
        drop_last = (len(dataset_train) % self.batch_size) == 1  # discard a last iteration with a single sample
        dl_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=drop_last)
        dl_valid = DataLoader(dataset_valid, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
        # Train the model
        print('Train model')
        accelerator = 'gpu' if 'cuda' in str(self.device) else 'cpu'
        model_path = self.path + 'data/' + self.database + '/' + self.regressor.value + '/' + self.backbone.value + '/'
        ckpt_path = os.path.join(model_path+'ckpt/', 'last.ckpt')
        loggers = [pl_loggers.TensorBoardLogger(save_dir=model_path+'logs/', default_hp_metric=False), PCRLogger()]
        early_callback = EarlyStopping(monitor='val_loss', mode='min', patience=self.patience)
        ckpt_callback = ModelCheckpoint(dirpath=model_path+'ckpt/', filename='{epoch}-{val_loss:.5f}', monitor='val_loss', save_last=True, save_top_k=1)
        trainer = pl.Trainer(accelerator=accelerator, devices=self.gpus, enable_progress_bar=False, max_epochs=self.epochs, precision=16, deterministic=True, gradient_clip_val=None, logger=loggers, callbacks=[early_callback, ckpt_callback])
        trainer.fit(model=self.model, train_dataloaders=dl_train, val_dataloaders=dl_valid, ckpt_path=ckpt_path if os.path.isfile(ckpt_path) else None)

    def load(self, mode):
        import torchinfo
        from pcr_framework.src.constants import Modes
        from src.lit_encoder import LitEncoder
        from src.lit_unet import LitUNet
        # Set up the neural network to train
        print('Load model')
        torch.set_float32_matmul_precision('medium')
        common_params = {'num_classes': len(self.indices), 'backbone': self.backbone, 'epochs': self.epochs, 'batch_size': self.batch_size, 'transfer': True, 'tune_fc_only': False}
        regressors = {Regressor.ENCODER: LitEncoder, Regressor.UNET: LitUNet}
        ModelClass = regressors[self.regressor]
        self.model = ModelClass(**common_params)
        self.model.to(self.device)
        torchinfo.summary(self.model, input_size=(self.batch_size, 3, self.width, self.height), depth=5, device=self.device.type, col_names=['input_size', 'output_size', 'num_params', 'kernel_size'])
        # Set up the neural network to test
        if mode is Modes.TEST:
            model_path = self.path + 'data/' + self.database + '/' + self.regressor.value + '/' + self.backbone.value + '/'
            print('Loading model from {}'.format(model_path))
            self.model = ModelClass.load_from_checkpoint(os.path.join(model_path+'ckpt/', 'best.ckpt'), num_classes=len(self.indices), backbone=self.backbone)
            self.model.to(self.device)
            self.model.eval()

    def process(self, ann, pred):
        import cv2
        from pcr_framework.src.annotations import PersonObject, GenericLandmark
        from pcr_framework.regression.alignment.landmarks import lps
        # Prepare dataloader
        dataset_test = MyDataset([pred], self.indices, self.regressor, self.width, self.height, Mode.TEST)
        dl_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
        with torch.no_grad():
            for batch in dl_test:
                # Generate prediction
                outputs = self.model(batch['img'].float().to(self.device))
                if self.regressor is Regressor.ENCODER:  # [batch_size, num_landmarks*2]
                    outputs = outputs.view(-1, len(self.indices), 2)
                    landmarks = outputs.squeeze().cpu().numpy()
                    landmarks[:, 0] *= float(self.width)
                    landmarks[:, 1] *= float(self.height)
                elif self.regressor is Regressor.UNET:  # [batch_size, num_landmarks, height_heatmap, width_heatmap]
                    heatmaps = outputs.squeeze().cpu().numpy()
                    landmarks = np.array([tuple(np.unravel_index(np.argmax(heatmaps[idx]), heatmaps[idx].shape)[::-1]) for idx in range(len(self.indices))])
                    # cv2.imshow('img', cv2.cvtColor((batch['img']*255).squeeze().cpu().numpy().astype('uint8').transpose(1, 2, 0), cv2.COLOR_BGR2RGB))
                    # for idx in range(len(self.indices)):
                    #     aux = cv2.normalize(heatmaps[idx][:, :, np.newaxis], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
                    #     cv2.circle(aux, landmarks[idx], 3, (0, 0, 0))
                    #     cv2.imshow('pred'+str(idx), aux)
                    #     cv2.waitKey(0)
                # Save prediction
                obj_pred = pred.images[batch['idx_img']].objects[batch['idx_obj']]
                if not isinstance(obj_pred, PersonObject):
                    continue
                bbox_enlarged = batch['bbox_enlarged'].squeeze().cpu().numpy()
                bbox_width = bbox_enlarged[2] - bbox_enlarged[0]
                bbox_height = bbox_enlarged[3] - bbox_enlarged[1]
                scale = np.array([self.width/bbox_width, self.height/bbox_height])
                landmarks = (landmarks/scale) + bbox_enlarged[0:2]
                for idx, pt in enumerate(landmarks):
                    label = self.indices[idx]
                    lp = list(self.parts.keys())[next((ids for ids, xs in enumerate(self.parts.values()) for x in xs if x == label), None)]
                    obj_pred.add_landmark(GenericLandmark(label, lp, pt, True), lps[type(lp)])
