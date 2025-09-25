#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from images_framework.alignment.students_landmarks.src.dataloader import Backbone


class LitEncoder(pl.LightningModule):
    """
    Pytorch Lightning wrapper to turn an encoder into a coordinates regressor.
    """
    encoders = {
        Backbone.RESNET18: models.resnet18, 
        Backbone.RESNET34: models.resnet34, 
        Backbone.RESNET50: models.resnet50, 
        Backbone.RESNET101: models.resnet101, 
        Backbone.RESNET152: models.resnet152,
        Backbone.EFFICIENTNETB0: models.efficientnet_b0,
        Backbone.EFFICIENTNETB1: models.efficientnet_b1,
        Backbone.EFFICIENTNETB2: models.efficientnet_b2,
        Backbone.EFFICIENTNETB3: models.efficientnet_b3,
        Backbone.EFFICIENTNETB4: models.efficientnet_b4,
        Backbone.EFFICIENTNETB5: models.efficientnet_b5,
        Backbone.EFFICIENTNETB6: models.efficientnet_b6,
        Backbone.EFFICIENTNETB7: models.efficientnet_b7,
        Backbone.VITB: models.vit_b_16,
        Backbone.VITL: models.vit_l_16,
        Backbone.VITH: models.vit_h_14
    }

    def __init__(self, num_classes, backbone, lr=1e-3, patience=20, batch_size=16, transfer=True, tune_fc_only=True):
        super().__init__()
        self.num_classes = num_classes
        self.lr = lr
        self.patience = patience
        self.batch_size = batch_size
        # Loss criterion
        self.loss_fn = nn.L1Loss()
        # Ecnoder architecture
        self.model = self.encoders[backbone](pretrained=transfer)
        # Replace final layer
        linear_size = list(self.model.children())[-1].in_features
        self.model.fc = nn.Linear(in_features=linear_size, out_features=num_classes*2)
        # Option to only tune the fully-connected layers
        if tune_fc_only:
            for child in list(self.model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt = SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-6, nesterov=True)
        scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=int(round(self.patience/4)))
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}}

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

    def on_validation_epoch_end(self):
        lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        # Perform logging
        self.log('learning_rate', lr, batch_size=self.batch_size, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss = self._step(batch)
        # Perform logging
        self.log('test_loss', loss, batch_size=self.batch_size, on_step=False, on_epoch=True)
