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

output_predictions = torch.rand(400,400)
print(output_predictions)
new = output_predictions.detach().clone()
new[new>0.5]=1
new[new<=0.5]=0
print(output_predictions.ToPILImage())
print(new)


palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
print(palette)
colors = torch.as_tensor([i for i in range(2)])[:, None] * palette
print(colors)
colors = (colors % 255).numpy().astype("uint8")
print(colors)

# plot the semantic segmentation predictions per class
r = Image.fromarray(output_predictions.cpu().numpy(), mode="1")
r.show()

bw = Image.fromarray(new.cpu().numpy(), mode="1")
bw.show()

two_img = np.hstack([new.cpu().numpy(), output_predictions.cpu().numpy()])
r_two_img = Image.fromarray(two_img, mode="1")

r_two_img.show()
#r.putpalette(colors)

#plt.imshow(r)
#r.show()