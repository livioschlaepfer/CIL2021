# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import torch
from torchvision import models
import os
from box import Box
import yaml
import getpass
import argparse
from src.seed import seed_all

# parser to select desired
parser = argparse.ArgumentParser()
parser.add_argument('--config', 
    default = 'custom',
    choices = ['custom', 'baseline_fcn', 'baseline_unet', 'baseline_deeplab'],
    help = 'Select on of the experiments described in our report or setup a custom config file'
)
args = parser.parse_args()

# load config
try: 
    config = Box.from_yaml(filename="./configs/"+ args.config + ".yaml", Loader=yaml.FullLoader)
except:
    raise OSError("Does not exist", args.config)

# fix seed
seed_all(config.seed)

from src.dataset import init_test_dataloaders
from src.tester import test_model
from src.models.model_runner import init_runner
from src.paths import paths_setter


# update paths based on user name
username = getpass.getuser()
config.paths = paths_setter(username=username)

# check if dir exists, otherwise create
if not os.path.exists(config.paths.model_store + "/" + config.checkpoint_name +"/"+ 'predictions_seed_'+str(config.seed_run)+'/'):
    os.makedirs(config.paths.model_store + "/" + config.checkpoint_name +"/"+ 'predictions_seed_'+str(config.seed_run)+'/')

# Initialize the runner for the selected model
runner = init_runner(config)

# Load trained model
if not os.path.exists(config.paths.model_store + "/" + config.checkpoint_name + '/weights_seed_'+str(config.seed_run)+'/weights.pth'):
    print("Error: Unable to load model, path does not exist:", config.paths.model_store + "/" + config.checkpoint_name + '/weights_seed_'+str(config.seed_run)+'/weights.pth')
    exit()

checkpoint = torch.load(config.paths.model_store + "/" + config.checkpoint_name + '/weights_seed_'+str(config.seed_run)+'/weights.pth')
runner.model.load_state_dict(checkpoint['model_state_dict'])
# Create test datasets + dataloader
image_datasets, dataloaders_dict_test = init_test_dataloaders(config)

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Send the model to GPU
runner.model = runner.model.to(device)

# Test model
test_model(runner, dataloaders_dict_test, device, config, model=config.checkpoint_name)

