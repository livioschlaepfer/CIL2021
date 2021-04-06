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

from src.dataset import init_test_dataloaders, init_train_dataloaders
from src.trainer import train_model
from src.models.model_runner import init_runner
from src.models.bayesian_Unet import B_Unet
from src.models.bm_parts import composite_bay_conv

import torch
from PIL import Image
import matplotlib.pyplot as pl

# load config
""" config = Box.from_yaml(filename="./config.yaml", Loader=yaml.FullLoader)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

lay = composite_bay_conv(3,3,3,2)
lay.to(device)

inputs = torch.Tensor(2,3,200,200).to(device)


print(lay.forward(inputs))

print(lay.kl_loss_()) """

x = torch.rand(2,3,400,400)
y = torch.rand(2,400,400)
z = torch.rand(2,400,400)


for (i,j,k) in zip(x,y,z):
    print(i.shape,j.shape,k.shape)




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