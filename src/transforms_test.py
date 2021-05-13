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
    total_image.append(cur_image)

    # Vertical flipping
    cur_image = TF.vflip(image)
    total_image.append(cur_image)

    # Apply color jitter twice
    # cur_image = TF.adjust_brightness(image, 1.3)
    # total_image.append(cur_image)

    # cur_image = TF.adjust_brightness(image, 0.7)
    # total_image.append(cur_image)

    # Stack transforms
    total_image = torch.stack(total_image)
    
    return total_image

# Reverse transforms on output
def transform_test_back(output):
    output[1] = TF.hflip(output[1])    
    output[2] = TF.vflip(output[2])

    return output

# Aggregate outputs
def transform_test_aggregate(output):

    # Average output
    # dims = (0)
    # output = torch.sum(output, dims)
    # output = output / 3

    # output = torch.unsqueeze(output, 0)

    # Take max output
    dims = (0)
    output = torch.max(output, dims, keepdim = True)[0]

    return output