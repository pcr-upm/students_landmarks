#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from images_framework.alignment.students_landmarks.src.dataloader import Backbone


class ViTRegressor(nn.Module):
    """
    ViT -> MLP -> coords
    """
    def __init__(self, vit_model, num_classes):
        super().__init__()
        embed_dim = vit_model.heads.head.in_features
        vit_model.heads = nn.Identity()
        self.vit = vit_model
        self.regressor = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes*2)
        )

    def forward(self, x):
        # Patchify + cls + pos embedding
        x = self.vit._process_input(x) # [batch_size, tokens, embed_dim]
        # Añadir class token y positional embedding
        cls_token = self.vit.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1) # [batch_size, tokens+1, embed_dim]
        x = x + self.vit.encoder.pos_embedding
        x = self.vit.encoder.dropout(x)
        # Pasar por el encoder
        x = self.vit.encoder.layers(x)
        x = self.vit.encoder.ln(x)
        pooled = x.mean(dim=1) # [batch_size, embed_dim]
        return self.regressor(pooled)


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
        Backbone.VITL: models.vit_l_16
    }

    def __init__(self, num_classes, backbone, epochs=100, batch_size=16, transfer=True, tune_fc_only=True):
        super().__init__()
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        # Loss criterion
        self.loss_fn = nn.L1Loss()
        # Encoder architecture
        self.model = self.encoders[backbone](weights='IMAGENET1K_V1' if transfer else None)
        # Replace final layer
        if backbone in [Backbone.RESNET18, Backbone.RESNET34, Backbone.RESNET50, Backbone.RESNET101, Backbone.RESNET152]:
            linear_size = list(self.model.children())[-1].in_features
            self.model.fc = nn.Linear(in_features=linear_size, out_features=num_classes*2)
            if tune_fc_only:
                for name, param in self.model.named_parameters():
                    if not any(sub in name for sub in ["fc"]):
                        param.requires_grad = False
        elif backbone in [Backbone.EFFICIENTNETB0, Backbone.EFFICIENTNETB1, Backbone.EFFICIENTNETB2, Backbone.EFFICIENTNETB3, Backbone.EFFICIENTNETB4, Backbone.EFFICIENTNETB5, Backbone.EFFICIENTNETB6, Backbone.EFFICIENTNETB7]:
            linear_size = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features=linear_size, out_features=num_classes*2)
            if tune_fc_only:
                for name, param in self.model.named_parameters():
                    if not any(sub in name for sub in ["classifier.1"]):
                        param.requires_grad = False
        else:
            self.model = ViTRegressor(self.model, num_classes)
            if tune_fc_only:
                for p in self.model.vit.parameters():
                    p.requires_grad = False

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
