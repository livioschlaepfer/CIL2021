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

from model import initialize_model
from dataset import init_train_dataloaders
from trainer import train_model


# load config
config = Box.from_yaml(filename="./config.yaml", Loader=yaml.FullLoader)

# Create training and validation datasets + dataloader
image_datasets, dataloaders_dict = init_train_dataloaders(config)

# Initialize the model for this run
model_ft = initialize_model(config)

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Send the model to GPU
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. 
params_to_update = model_ft.parameters()

# Specify layers where we want to freeze weights
if config.freeze:
    print("Freeze layers, set trainable layers")
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(params_to_update, lr=0.001)

# Setup the loss
# criterion = torch.nn.MSELoss(reduction='mean')
# criterion = nn.CrossEntropyLoss()
criterion = nn.NLLLoss()


# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs = config.num_epochs, config = config, device = device)

