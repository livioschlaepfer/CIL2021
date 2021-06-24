from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import time
import os
import copy
from src.edge_det import canny
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

def init_data_transforms(config):

    data_transforms = None

    # Data augmentation and normalization for training
    # Just normalization for validation

    def transform(image, mask):

        # Random horizontal flipping
        if np.random.uniform(0,1) > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if np.random.uniform(0,1) > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Random rotation
        if np.random.uniform(0,1) > 0.9:
            angle = np.random.uniform(low=5, high=355)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

            # Center crop to avoid black edges the best we can 
            cc = transforms.CenterCrop(config.transforms.crop_size)
            image = cc(image)
            mask = cc(mask) 

            # Random color jitter
            color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0, hue=0)
            image = color_jitter(image)
            
        else:
            # Five crop image
            fc = transforms.FiveCrop(config.transforms.crop_size)
            images = fc(image)
            masks = fc(mask)

            # Select random crop
            rand = np.random.randint(0,4)

            image = images[rand]
            mask = masks[rand]

            # Random 90 degree rotation
            rot90 = [0, 90, 180, 270]
            rand = np.random.randint(0,4)

            image = TF.rotate(image, rot90[rand])
            mask = TF.rotate(mask, rot90[rand])

            # Random color jitter
            color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0, hue=0)
            image = color_jitter(image)

        image = canny(image, filter_size=3, threshold=250, use_cuda=True)

        return image, mask

    if config.transforms.apply_transforms:
        return transform

    return None
