# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
from box import Box
import pickle
import json
import yaml
import getpass
import random
import argparse
from src.seed import seed_all

# parser to select desired
parser = argparse.ArgumentParser()
parser.add_argument('--config', 
    default = 'custom',
    choices = ['custom', 'baseline_fcn', 'baseline_unet', 'baseline_deeplab'],
    help = 'Select one of the experiments described in our report or setup a custom config file'
)
args = parser.parse_args()

# load config
try: 
    config = Box.from_yaml(filename="./configs/"+ args.config + ".yaml", Loader=yaml.FullLoader)
except:
    raise OSError("Does not exist", args.config)

# fix seed
seed_all(config.seed)

from src.dataset import init_train_dataloaders
from src.trainer import train_model
from src.models.model_runner import init_runner
from src.paths import paths_setter
from src.scheduler import get_scheduler

# update paths based on user name
username = getpass.getuser()
config.paths = paths_setter(username=username, pretrain=config.pretrain)

# Initialize the runner for the selected model
runner = init_runner(config)

# Create training and validation datasets + dataloader
image_datasets, dataloaders_dict_train = init_train_dataloaders(config)

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
optimizer_ft = optim.Adam(params_to_update, lr=config.lr.init_lr)

# set a lr schedule
scheduler_ft = get_scheduler(optimizer_ft, config)

if config.continue_training:
    print("Continue training on", config.continue_training_on_checkpoint)
    # Load trained model
    if not os.path.exists(config.paths.model_store + "/" + config.continue_training_on_checkpoint + "/weights/weights.pth"):
        print("Error: Unable to load model, path does not exist:", config.paths.model_store + "/" + config.continue_training_on_checkpoint + "/weights/weights.pth")
        exit()

    checkpoint = torch.load(config.paths.model_store + "/" + config.continue_training_on_checkpoint + "/weights/weights.pth")
    runner.model.load_state_dict(checkpoint['model_state_dict'])

# Train and evaluate model
runner, hist = train_model(runner, dataloaders_dict_train, optimizer_ft, scheduler_ft, num_epochs = config.num_epochs, config = config, device = device)

# Store model
if not os.path.exists(config.paths.model_store):
    os.makedirs(config.paths.model_store)
if not os.path.exists(config.paths.model_store+ '/' + config.checkpoint_name):
    os.makedirs(config.paths.model_store+ '/' + config.checkpoint_name)
if not os.path.exists(config.paths.model_store+ '/' + config.checkpoint_name +'/weights_seed_' +str(config.seed_run)):
    os.makedirs(config.paths.model_store+ '/' + config.checkpoint_name +'/weights_seed_' +str(config.seed_run))
if not os.path.exists(config.paths.model_store+ '/' + config.checkpoint_name +'/config_seed_' +str(config.seed_run)):
    os.makedirs(config.paths.model_store+ '/' + config.checkpoint_name +'/config_seed_' +str(config.seed_run))
    

torch.save({
    'model_state_dict': runner.model.state_dict()
    }, 
    config.paths.model_store + '/' + config.checkpoint_name +'/weights_seed_' + str(config.seed_run)+ '/weights.pth')


with open(config.paths.model_store + '/' + config.checkpoint_name +'/config_seed_' +str(config.seed_run) +'/config.txt', 'w') as f: # config.paths.model_store + '/' + config.checkpoint_name +'/config/'+
    f.write(json.dumps(config.to_dict()))

print("Stored model statedict under:", config.paths.model_store + config.checkpoint_name  +'/weights_seed_' + str(config.seed_run)+'/weights.pth')