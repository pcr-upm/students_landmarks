#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.dataloader import Backbone


class WingLoss(nn.Module):
    """
    Wing Loss (CVPR 2018)
    Feng et al., "Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks"
    """
    def __init__(self, omega=10.0, epsilon=2.0, reduction='mean'):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.reduction = reduction
        self.constant = omega - omega * math.log(1 + omega / epsilon)

    def forward(self, pred, gt):
        w, e = self.omega, self.epsilon
        abs_diff = torch.abs(gt - pred)
        loss = w * torch.log(1 + abs_diff / e) * (abs_diff < w)
        loss = loss + (abs_diff - self.constant) * (abs_diff >= w)
        # Reduce loss
        loss = loss.sum(-1)
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            raise NotImplementedError(f"Reduction type '{self.reduction}' is not implemented.")

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
        Backbone.VIT: models.maxvit_t
    }

    def __init__(self, num_classes, backbone, epochs=100, batch_size=16, transfer=True, tune_fc_only=True):
        super().__init__()
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        # Encoder architecture
        self.model = self.encoders[backbone](weights='IMAGENET1K_V1' if transfer else None)
        # Replace final layer
        if backbone in [Backbone.RESNET18, Backbone.RESNET34, Backbone.RESNET50, Backbone.RESNET101, Backbone.RESNET152]:
            classifier = 'fc'
            linear_size = list(self.model.children())[-1].in_features
            self.model.fc = nn.Linear(in_features=linear_size, out_features=num_classes*2)
        elif backbone in [Backbone.EFFICIENTNETB0, Backbone.EFFICIENTNETB1, Backbone.EFFICIENTNETB2, Backbone.EFFICIENTNETB3, Backbone.EFFICIENTNETB4, Backbone.EFFICIENTNETB5, Backbone.EFFICIENTNETB6, Backbone.EFFICIENTNETB7]:
            classifier = 'classifier.1'
            linear_size = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features=linear_size, out_features=num_classes*2)
        else:
            classifier = 'classifier.5'
            linear_size = self.model.classifier[5].in_features
            self.model.classifier[5] = nn.Linear(in_features=linear_size, out_features=num_classes*2)
        if tune_fc_only:
            for name, param in self.model.named_parameters():
                if not any(sub in name for sub in [classifier]):
                    param.requires_grad = False

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=3e-4 , weight_decay=0.05)
        scheduler = CosineAnnealingLR(opt, T_max=self.epochs)
        return {'optimizer': opt, 'lr_scheduler': scheduler}

    def _step(self, batch):
        inputs = batch['img'].float()
        targets = batch['landmarks'].float()
        outputs = self.model(inputs)
        outputs = outputs.view(-1, self.num_classes, 2)
        loss = WingLoss()(outputs, targets)
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
