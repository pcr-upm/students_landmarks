#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from images_framework.alignment.students_landmarks.src.dataloader import Backbone


class LitUNet(pl.LightningModule):
    """
    Pytorch Lightning wrapper to turn an encoder-decoder into a heatmap regressor.
    """
    def __init__(self, num_classes, backbone, epochs=100, batch_size=16, transfer=True, tune_fc_only=True):
        super().__init__()
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        # Loss criterion
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([(256*256)-1]))
        # Using a pretrained UNet architecture
        self.model = smp.Unet(encoder_name='mit_b2' if backbone in [Backbone.VITB, Backbone.VITL] else backbone.value, encoder_weights='imagenet' if transfer else None, decoder_channels=list([256, 128, 64, 64, 64]), in_channels=3)
        # Replace final layer
        self.model.segmentation_head = nn.Sequential(nn.Conv2d(64, num_classes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.Flatten(start_dim=2, end_dim=3))

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=3e-4 , weight_decay=0.05)
        scheduler = CosineAnnealingLR(opt, T_max=self.epochs)
        return {'optimizer': opt, 'lr_scheduler': scheduler}

    def _step(self, batch):
        inputs = batch['img'].float()
        targets = batch['heatmaps'].float()
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        # import cv2
        # import numpy as np
        # with torch.no_grad():
        #     cv2.imshow('img', cv2.cvtColor((batch['img'][0]*255).cpu().numpy().astype('uint8').transpose(1, 2, 0), cv2.COLOR_BGR2RGB))
        #     anno_heatmaps = torch.unflatten(targets[0], 1, (256, 256)).squeeze().cpu().numpy()
        #     pred_heatmaps = torch.unflatten(torch.sigmoid(outputs[0]), 1, (256, 256)).squeeze().cpu().numpy()
        #     for idx in range(outputs.shape[1]):
        #         anno = cv2.normalize(anno_heatmaps[idx][:, :, np.newaxis], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        #         pred = cv2.normalize(pred_heatmaps[idx][:, :, np.newaxis], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        #         cv2.circle(pred, cv2.minMaxLoc(pred)[3], 3, (0, 0, 0))
        #         cv2.imshow('anno'+str(idx), anno)
        #         cv2.imshow('pred'+str(idx), pred)
        #         cv2.waitKey(0)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        # Perform logging
        self.log('train_loss', loss, batch_size=self.batch_size, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        # Perform logging
        self.log('val_loss', loss, batch_size=self.batch_size, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        # Perform logging
        self.log('learning_rate', lr, batch_size=self.batch_size, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss = self._step(batch)
        # Perform logging
        self.log('test_loss', loss, batch_size=self.batch_size, on_step=False, on_epoch=True)
