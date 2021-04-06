# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from box import Box
import yaml

from src.dataset import init_test_dataloaders, init_train_dataloaders
from src.trainer import train_model
from src.tester import test_model
from src.models.model_runner import init_runner
from src.models.bayesian_Unet import calculate_kl

# load config
config = Box.from_yaml(filename="./config.yaml", Loader=yaml.FullLoader)

# Initialize the runner for the selected model
runner = init_runner(config)

# Create training and validation datasets + dataloader
image_datasets, dataloaders_dict = init_train_dataloaders(config)

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Send the model to GPU
runner.model = runner.model.to(device)

# Gather the parameters to be optimized/updated in this run. 
params_to_update = runner.model.parameters()

# Specify layers where we want to freeze weights
if config.freeze:
    print("Freeze layers, set trainable layers")
    params_to_update = []
    for name,param in runner.model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(params_to_update, lr=0.001)

# Train and evaluate model
if config.runs.train_run:
    runner, hist = train_model(runner, dataloaders_dict, optimizer_ft, num_epochs = config.num_epochs, config = config, device = device)


# Test model
if config.runs.test_run:
    # Create test datasets + dataloader
    image_datasets, dataloaders_dict = init_test_dataloaders(config)

    # Test model
    test_model(runner, dataloaders_dict, device, config)
