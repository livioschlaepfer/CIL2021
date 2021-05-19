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
    crop_size = 304

    fc = transforms.FiveCrop(crop_size)
    images = fc(image)

    for image in images:
        total_image.append(image)

        # Horizontal flipping
        cur_image = TF.hflip(image)
        total_image.append(cur_image)

        # Vertical flipping
        cur_image = TF.vflip(image)
        total_image.append(cur_image)

        # Vertical horizontal flipping
        cur_image = TF.hflip(cur_image)
        total_image.append(cur_image)

    # Stack transforms
    total_image = torch.stack(total_image)
    
    return total_image

# Reverse transforms on output
def transform_test_back(output):
    for i in range(0,19,5):
        output[i+1] = TF.hflip(output[i+1])    
        output[i+2] = TF.vflip(output[i+2])
        output[i+3] = TF.hflip(TF.vflip(output[i+3]))

    return output

# Aggregate outputs
def transform_test_aggregate(output):

    # Average output
    # dims = (0)
    # output = torch.sum(output, dims)
    # output = output / 4

    # output = torch.unsqueeze(output, 0)

    # Take max output
    dims = (0)
    for i in range(0,19,5):
        output[i] = torch.max(output[i:i+4], dims, keepdim = True)[0]
    
    # Rebuild image from 5 crop
    output = output.view(1,5,-1).avg(1)

    print("output shape", output.shape)
 
    return output