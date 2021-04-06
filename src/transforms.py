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
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

def init_data_transforms(config):

    data_transforms = None

    # Data augmentation and normalization for training
    # Just normalization for validation

    if config.transforms.apply_transforms:
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=15),
                transforms.RandomResizedCrop(config.transforms.crop_size),
            ]),
            'val': transforms.Compose([
                transforms.CenterCrop(config.transforms.crop_size),
            ]),
        }

    return data_transforms
