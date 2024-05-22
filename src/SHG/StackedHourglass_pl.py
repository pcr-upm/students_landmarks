#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import pytorch_lightning as pl
from torch import nn, optim
from .StackedHourglass import create_StackedHG


class LitSHG(pl.LightningModule):
    """
    Pytorch Lightning wrapper to the Stacked Hourglass network
    """
    def __init__(self, num_modules, num_landmarks, batch_size, lr, weight_decay):
        super().__init__()
        self.model = create_StackedHG(num_modules=num_modules, num_landmarks=num_landmarks)
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.mse_loss = nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)[0]

    def training_step(self, batch, batch_idx):
        inputs = batch['img'].float()
        targets = batch['heatmap'].float()
        outputs = self.model(inputs)[0]
        loss = self.mse_loss(outputs, targets)
        self.log('train_loss', loss, batch_size=self.batch_size, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch['img'].float()
        targets = batch['heatmaps'].float()
        outputs = self.model(inputs)[0]
        loss = self.mse_loss(outputs, targets)
        self.log('val_loss', loss, batch_size=self.batch_size, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
