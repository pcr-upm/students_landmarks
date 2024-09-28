#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import torch.nn as nn
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau


class LitUNet(pl.LightningModule):
    """
    Pytorch Lightning wrapper to the UNet network
    """
    def __init__(self, num_classes, version, lr=1e-3, patience=20, batch_size=16, transfer=True, tune_fc_only=True):
        super().__init__()
        self.num_classes = num_classes
        self.lr = lr
        self.patience = patience
        self.batch_size = batch_size
        # Loss criterion
        self.loss_fn = nn.CrossEntropyLoss()
        # Using a UNet architecture
        print('resnet'+str(version))
        self.model = smp.Unet(encoder_name='resnet'+str(version), encoder_weights='imagenet' if transfer else None, in_channels=3, classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt = SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-6, nesterov=True)
        scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=int(round(self.patience/4)))
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}}

    def _step(self, batch):
        inputs = batch['img'].float()
        targets = batch['heatmaps'].float()
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
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
