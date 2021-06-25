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
import os
from box import Box
import yaml
import getpass
import random

# load config
config = Box.from_yaml(filename="./config.yaml", Loader=yaml.FullLoader)

from src.seed import seed_all
# fix seed
seed_all(config.seed)

from src.dataset import init_test_dataloaders
from src.tester import test_model
from src.models.model_runner import init_runner
from src.paths import paths_setter


# update paths based on user name
username = getpass.getuser()
config.paths = paths_setter(username=username)

# Initialize the runner for the selected model
runner = init_runner(config)

# Create test datasets + dataloader
image_datasets, dataloaders_dict_test = init_test_dataloaders(config)

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load trained model
if not os.path.exists(config.paths.model_store + "/" + config.checkpoint_name + ".pth"):
    print("Error: Unable to load model, path does not exist:", config.paths.model_store + "/" + config.checkpoint_name + ".pth")
    exit()
    
runner.model.load_state_dict(torch.load(config.paths.model_store + "/" + config.checkpoint_name + ".pth"))

# Send the model to GPU
runner.model = runner.model.to(device)


# Test model
test_model(runner, dataloaders_dict_test, device, config)
