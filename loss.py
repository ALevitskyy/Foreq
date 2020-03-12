#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 17:13:09 2019

@author: andriylevitskyy
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, lr_scheduler
import numpy as np
from datamanager import DataManager, train_start_end_func, val_start_end_func
from neural_net import make_net
from copy import deepcopy, copy
import collections

# from catalyst.dl.callbacks import (
#   LossCallback, TensorboardLogger, OptimizerCallback, CheckpointCallback,  ConsoleLogger)
# from catalyst.dl.experiments.runner import SupervisedRunner
import os


class MaximumLikelyhoodLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        outputs = torch.log(outputs)
        return -torch.sum(outputs * targets)


class Currency_Dataset(Dataset):
    def __init__(self, dataManager, size):
        self.dataManager = dataManager
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        sample = self.dataManager.get_random_processed_sample()
        return {
            "features": torch.from_numpy(sample["input"]),
            "targets": torch.from_numpy(sample["output"]).float(),
        }


def get_callbacks(log_dir):
    callbacks = collections.OrderedDict()
    callbacks["saver"] = CheckpointCallback(
        os.path.join(log_dir, "checkpoints/best.pth")
    )
    callbacks["loss"] = LossCallback()
    callbacks["optimizer"] = OptimizerCallback()
    callbacks["logger"] = ConsoleLogger()
    callbacks["tflogger"] = TensorboardLogger()
    return callbacks


log_dir = "logs"
workers = 4
batch_size = 10
epoch_size_train = 200
epoch_size_val = 200
num_epochs = 3

dataManager = DataManager(data_path="./reduced", start_end_func=train_start_end_func)
dataManager.load_all()
dataManager.init_norm_params()
dataManager.init_splits()
dataManagerVal = DataManager(data_path="./reduced", start_end_func=val_start_end_func)
dataManagerVal.load_all()
dataManagerVal.init_norm_params()
dataManagerVal.init_splits()
dataset_train = Currency_Dataset(dataManager, epoch_size_train)
# Trick because deepcopy does not work with generators
# doesn`t work... need 2 datamanagers (load data twice... idea,
# load once, then pass) - manual deepcopy
dataset_val = Currency_Dataset(dataManagerVal, epoch_size_val)
dataloader_train = DataLoader(
    dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=workers
)
dataloader_val = DataLoader(
    dataset=dataset_val, batch_size=batch_size, shuffle=True, num_workers=workers
)
"""
loaders = collections.OrderedDict()
loaders['train'] = dataloader_train
loaders['valid'] = dataloader_val
model = make_net()
optimizer = Adam(model.parameters(), lr=1e-4)
runner = SupervisedRunner()
runner.train(
            model = model,
            criterion = MaximumLikelyhoodLoss(),
            loaders = loaders,
            logdir = log_dir,
            optimizer = optimizer,
            scheduler = lr_scheduler.MultiStepLR(optimizer, 
                                                 milestones=[10, 20, 40],
                                                 gamma=0.3),
            num_epochs = num_epochs,
            verbose = True,
            callbackas = get_callbacks(log_dir)
        )
"""
