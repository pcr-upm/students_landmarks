#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
import torch
import numpy as np
from torch.utils.data import DataLoader
from images_framework.src.alignment import Alignment
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
        if self.database in ['300w_public', '300w_private', '300wlp']:
            self.indices = [101, 102, 103, 104, 105, 106, 107, 108, 24, 110, 111, 112, 113, 114, 115, 116, 117, 1, 119, 2, 121, 3, 4, 124, 5, 126, 6, 128, 129, 130, 17, 16, 133, 134, 135, 18, 7, 138, 139, 8, 141, 142, 11, 144, 145, 12, 147, 148, 20, 150, 151, 22, 153, 154, 21, 156, 157, 23, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168]
        elif self.database in 'cofw':
            self.indices = [1, 6, 3, 4, 101, 102, 103, 104, 7, 12, 8, 11, 9, 10, 13, 14, 105, 106, 16, 18, 17, 107, 20, 21, 22, 108, 109, 23, 24]
        elif self.database in 'wflw':
            self.indices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 24, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 1, 134, 2, 136, 3, 138, 139, 140, 141, 4, 143, 5, 145, 6, 147, 148, 149, 150, 151, 152, 153, 17, 16, 156, 157, 158, 18, 7, 161, 9, 163, 8, 165, 10, 167, 11, 169, 13, 171, 12, 173, 14, 175, 20, 177, 178, 22, 180, 181, 21, 183, 184, 23, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197]
        elif self.database in 'face_synthetics':
            self.indices = [101, 102, 103, 104, 105, 106, 107, 108, 24, 110, 111, 112, 113, 114, 115, 116, 117, 1, 119, 2, 121, 3, 4, 124, 5, 126, 6, 128, 129, 130, 17, 16, 133, 134, 135, 18, 7, 138, 139, 8, 141, 142, 11, 144, 145, 12, 147, 148, 20, 150, 151, 22, 153, 154, 21, 156, 157, 23, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170]
        elif self.database in 'dad':
            self.indices = [1983, 2189, 3708, 336, 335, 3153, 3705, 2178, 3684, 3741, 3148, 3696, 2585, 2565, 2567, 3764, 570, 694, 3865, 17, 16, 2134, 3863, 673, 3851, 3880, 2121, 3859, 1448, 1428, 1430, 3893, 2441, 2446, 2382, 2381, 2383, 2496, 3690, 2493, 2491, 2465, 3619, 3632, 2505, 2273, 2276, 2355, 2295, 2359, 2267, 2271, 2403, 2437, 1183, 1194, 1033, 1023, 1034, 1345, 3856, 1342, 1340, 1243, 3827, 3833, 1354, 824, 827, 991, 883, 995, 814, 822, 1096, 1175, 3540, 3704, 3555, 3560, 3561, 3501, 3526, 3563, 2793, 2751, 3092, 3099, 3102, 2205, 2193, 2973, 2868, 2921, 2920, 1676, 1623, 2057, 2064, 2067, 723, 702, 1895, 1757, 1818, 1817, 3515, 3541, 2828, 2832, 2833, 2850, 2813, 2811, 2774, 3546, 1657, 1694, 1696, 1735, 1716, 1715, 1711, 1719, 1748, 1740, 1667, 1668, 3533, 2785, 2784, 2855, 2863, 2836, 2891, 2890, 2892, 2928, 2937, 3509, 1848, 1826, 1789, 1787, 1788, 1579, 1773, 1774, 1795, 1802, 1865, 3503, 2948, 2905, 2898, 2881, 2880, 2715, 3386, 3381, 1962, 2213, 2259, 2257, 2954, 3171, 2003, 3554, 576, 2159, 1872, 798, 802, 731, 567, 3577, 3582, 3390, 3391, 3396, 3400, 3599, 3593, 3588, 3068, 2196, 2091, 3524, 628, 705, 2030]
        elif self.database in 'agora':
            # self.indices = [101, 11, 12, 102, 13, 14, 103, 15, 16, 104, 105, 106, 17, 107, 108, 112, 5, 6, 7, 8, 9, 10, 109, 110, 111]
            self.indices = [4, 124, 5, 126, 6, 1, 119, 2, 121, 3, 128, 129, 130, 17, 16, 133, 134, 135, 18, 11, 144, 145, 12, 147, 148, 7, 138, 139, 8, 141, 142, 20, 150, 151, 22, 153, 154, 21, 165, 164, 163, 162, 161, 156, 157, 23, 159, 160, 168, 167, 166]
        else:
            raise ValueError('Database is not implemented')

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
        from images_framework.src.constants import Modes
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
        from images_framework.src.datasets import Database
        from images_framework.src.annotations import GenericLandmark
        from images_framework.alignment.landmarks import lps
        datasets = [subclass().get_names() for subclass in Database.__subclasses__()]
        idx = [datasets.index(subset) for subset in datasets if self.database in subset]
        parts = Database.__subclasses__()[idx[0]]().get_landmarks()
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
                bbox_enlarged = batch['bbox_enlarged'].squeeze().cpu().numpy()
                bbox_width = bbox_enlarged[2] - bbox_enlarged[0]
                bbox_height = bbox_enlarged[3] - bbox_enlarged[1]
                scale = np.array([self.width/bbox_width, self.height/bbox_height])
                landmarks = (landmarks/scale) + bbox_enlarged[0:2]
                for idx, pt in enumerate(landmarks):
                    label = self.indices[idx]
                    lp = list(parts.keys())[next((ids for ids, xs in enumerate(parts.values()) for x in xs if x == label), None)]
                    obj_pred.add_landmark(GenericLandmark(label, lp, pt, True), lps[type(lp)])
