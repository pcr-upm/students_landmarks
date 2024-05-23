#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import logging
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities import rank_zero_only


class PCRLogger(Logger):
    """ Prints model metrics using Python's standard logging library """
    def __init__(self):
        super().__init__()
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')

    @property
    def name(self):
        return "PCRLogger"

    @property
    def version(self):
        pass

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        msg = f'Epoch {metrics["epoch"]}:'
        for k, v in metrics.items():
            if k != 'epoch':
                msg += f' {k}: {v}'
        logging.info(msg)
