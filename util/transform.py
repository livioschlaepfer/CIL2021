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
#print("PyTorch Version: ",torch.__version__)
#print("Torchvision Version: ",torchvision.__version__)

class init_data_transforms:
    def __init__(self, opt):
        self.opt = opt
        self.prep_image = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                            ])
        self.prep_target = transforms.Compose([
                                transforms.ToTensor()
                            ])

    def prep(self, image, target):
        return self.prep_image(image), self.prep_target(target)

    def transform(self, image, target):
        
        # Random horizontal flipping
        if np.random.uniform(0,1) > 0.5:
            image = TF.hflip(image)
            target = TF.hflip(target)

        # Random vertical flipping
        if np.random.uniform(0,1) > 0.5:
            image = TF.vflip(image)
            target = TF.vflip(target)

        # Random rotation
        if np.random.uniform(0,1) > 0.5:
            angle = np.random.uniform(low=-45, high=45)
            image = TF.rotate(image, angle)
            target = TF.rotate(target, angle)

            # Center crop to avoid black edges the best we can 
            cc = transforms.CenterCrop(self.opt.crop_size)
            image = cc(image)
            target = cc(target) 
        else:
            # Five crop image
            fc = transforms.FiveCrop(self.opt.crop_size)
            images = fc(image)
            targets = fc(target)

            # Select random crop
            rand = np.random.randint(0,4)

            image = images[rand]
            target = targets[rand]

        return image, target