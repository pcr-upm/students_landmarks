#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.dataloader import Backbone


class AdaptiveWingLoss(nn.Module):
    """
    Adaptive Wing Loss (ICCV 2019)
    Wang et al., "Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression"
    """
    def __init__(self, omega=14.0, theta=0.5, epsilon=1.0, alpha=2.1, map_weight=10.0, map_threshold=0.2, reduction='mean'):
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
        self.map_weight = map_weight
        self.map_threshold = map_threshold
        self.reduction = reduction

    def forward(self, pred, gt):
        w, t, e, a = self.omega, self.theta, self.epsilon, self.alpha
        abs_diff = torch.abs(gt - pred)
        a_minus_gt = a - gt
        t_div_e = t / e
        cont_constant = w * (1 / (1 + t_div_e ** a_minus_gt)) * a_minus_gt * (t_div_e ** (a_minus_gt - 1)) * (1 / e)
        smooth_constant = (t * cont_constant - w * torch.log(1 + t_div_e ** a_minus_gt))
        loss = w * torch.log(1 + (abs_diff / e) ** a_minus_gt) * (abs_diff < t)
        loss = loss + (cont_constant * abs_diff - smooth_constant) * (abs_diff >= t)
        # Optional weighting: emphasize regions near landmarks
        if self.map_weight > 0:
            dilated_heatmaps = F.max_pool2d(gt, kernel_size=3, stride=1, padding=1)
            weight_map = (dilated_heatmaps >= self.map_threshold).float()
            loss = loss * (self.map_weight * weight_map + 1)
        # Reduce loss
        loss = loss.mean(dim=(-1, -2))
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            raise NotImplementedError(f"Reduction type '{self.reduction}' is not implemented.")

class LitUNet(pl.LightningModule):
    """
    Pytorch Lightning wrapper to turn an encoder-decoder into a heatmap regressor.
    """
    def __init__(self, num_classes, backbone, epochs=100, batch_size=16, transfer=True, tune_fc_only=True):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        # Using a pretrained UNet architecture
        self.model = smp.Unet(encoder_name='mit_b2' if backbone in [Backbone.VIT] else backbone.value, encoder_weights='imagenet' if transfer else None, encoder_depth=5, decoder_channels=list([256, 128, 96, 80, 64]), in_channels=3)
        # Replace final layer
        self.model.segmentation_head = nn.Conv2d(64, num_classes, kernel_size=(1, 1))

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
        loss = AdaptiveWingLoss()(outputs, targets)
        # import cv2
        # import torch
        # import numpy as np
        # with torch.no_grad():
        #     cv2.imshow('img', cv2.cvtColor((batch['img'][0]*255).cpu().numpy().astype('uint8').transpose(1, 2, 0), cv2.COLOR_BGR2RGB))
        #     anno_heatmaps = targets[0].squeeze().cpu().numpy()
        #     pred_heatmaps = outputs[0].squeeze().cpu().numpy()
        #     for idx in range(outputs.shape[1]):
        #         anno = tuple(np.unravel_index(np.argmax(anno_heatmaps[idx]), anno_heatmaps[idx].shape)[::-1])
        #         pred = tuple(np.unravel_index(np.argmax(pred_heatmaps[idx]), pred_heatmaps[idx].shape)[::-1])
        #         anno_heatmap = cv2.normalize(anno_heatmaps[idx][:, :, np.newaxis], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        #         pred_heatmap = cv2.normalize(pred_heatmaps[idx][:, :, np.newaxis], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        #         cv2.circle(anno_heatmap, anno, 3, (255, 255, 255))
        #         cv2.circle(pred_heatmap, pred, 3, (255, 255, 255))
        #         cv2.imshow('anno'+str(idx), anno_heatmap)
        #         cv2.imshow('pred'+str(idx), pred_heatmap)
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
