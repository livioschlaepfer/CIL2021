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

def transform_test(image):
    total_image = []
    total_image.append(image)

    # Horizontal flipping
    cur_image = TF.hflip(image)

    print("image shape", image.shape)
    print("cur_image shape", cur_image.shape)

    total_image.append(cur_image)

    # Vertical flipping
    cur_image = TF.vflip(image)
    total_image.append(cur_image)

    # Apply color jitter twice
    jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0)

    cur_image =jitter(image)
    total_image.append(cur_image)

    # Stack transforms
    total_image = torch.stack(total_image)
    print("total image", total_image.shape)
    
    return total_image

# Reverse transforms on output
def transform_test_back(output):
    output[1] = TF.hflip(output[1])    
    output[2] = TF.vflip(output[2])

    return output

# Aggregate outputs
def transform_test_aggregate(output):

    # Average output
    dims = (0)
    output = torch.sum(output, dims)
    output = output / 4

    output = torch.unsqueeze(output, 0)

    return output