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
import getpass

from src.dataset import init_test_dataloaders, init_train_dataloaders
from src.trainer import train_model
from src.models.model_runner import init_runner
from src.models.bayesian_Unet import B_Unet
from src.models.bm_parts import composite_bay_conv

from src.criterion.custom_losses import DiceLoss

import torch
from PIL import Image
import matplotlib.pyplot as pl

test1 = torch.rand(2, 5,5, requires_grad=True)
test2 = torch.rand(2,5,5, requires_grad=True)
print(test1.shape, test2.shape)


print("ok")

print(getpass.getuser())

# load config
""" config = Box.from_yaml(filename="./config.yaml", Loader=yaml.FullLoader)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

lay = composite_bay_conv(3,3,3,2)
lay.to(device)

inputs = torch.Tensor(2,3,200,200).to(device)


print(lay.forward(inputs))

print(lay.kl_loss_()) """

""" x = torch.rand(2,3,400,400)
y = torch.rand(2,400,400)
z = torch.rand(2,400,400)


for (i,j,k) in zip(x,y,z):
    print(i.shape,j.shape,k.shape) """




""" output_predictions = torch.rand(400,400)
output_predictions1 = torch.rand(3,400,400)
print(output_predictions)
new = output_predictions.detach().clone()
new[new>0.5]=1
new[new<=0.5]=0
print(new)

out = transforms.ToPILImage(mode="L")(output_predictions).convert("RGB")
out1 = transforms.ToPILImage(mode="RGB")(output_predictions1)
new = transforms.ToPILImage(mode="L")(new).convert("RGB")



out.show()
out1.show()
new.show()

Image.fromarray(np.hstack((np.array(out),np.array(new), np.array(out1)))).show() """

""" # load config
config = Box.from_yaml(filename="./config.yaml", Loader=yaml.FullLoader)

# Initialize the runner for the selected model
runner = init_runner(config)

# Create training and validation datasets + dataloader
image_datasets, dataloaders_dict = init_train_dataloaders(config)

print(dataloaders_dict)

bucket = torch.zeros(3,400,400)
counter =0
for inputs, labels in dataloaders_dict["train"]:
    bucket+=torch.squeeze(inputs)
    counter +=1
bucket = bucket/counter

bucket2 = torch.zeros(90,3,400,400)
counter2 =0
for inputs, labels in dataloaders_dict["train"]:
    bucket2[counter2,:,:,:] =inputs
    counter2 +=1


mean = torch.mean(bucket, dim=[1,2])
std = torch.std(bucket2, dim=[0,2,3])
print(mean, std) """