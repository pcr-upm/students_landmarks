#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models
from torch.optim import SGD, Adam


class LitResNet(pl.LightningModule):
    """
    Define a new class to turn the ResNet model that we want to use as a feature extractor.
    """
    resnets = {18: models.resnet18, 34: models.resnet34, 50: models.resnet50, 101: models.resnet101, 152: models.resnet152,}
    optimizers = {'adam': Adam, 'sgd': SGD}

    def __init__(self, num_classes, resnet_version, optimizer='adam', lr=1e-3, batch_size=16, transfer=True, tune_fc_only=True):
        super().__init__()
        self.num_classes = num_classes
        self.lr = lr
        self.batch_size = batch_size
        self.optimizer = self.optimizers[optimizer]
        # Loss criterion
        self.loss_fn = nn.MSELoss()
        # Using a pretrained ResNet backbone
        self.model = self.resnets[resnet_version](pretrained=transfer)
        # Replace old FC layer with Identity, so we can train our own
        linear_size = list(self.model.children())[-1].in_features
        # Replace final layer for fine-tuning
        self.model.fc = nn.Linear(in_features=linear_size, out_features=num_classes*2)
        # Option to only tune the fully-connected layers
        if tune_fc_only:
            for child in list(self.model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)

    def _step(self, batch):
        inputs = batch['img'].float()
        targets = batch['landmarks'].float()
        outputs = self.model(inputs)
        outputs = outputs.view(-1, self.num_classes, 2)
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

    def test_step(self, batch, batch_idx):
        loss = self._step(batch)
        # Perform logging
        self.log('test_loss', loss, batch_size=self.batch_size, on_step=False, on_epoch=True)
