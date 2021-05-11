import torch
import torch.nn as nn
from torchvision import transforms
from crfseg import CRF

from PIL import Image
import glob

model = nn.Sequential(
    nn.Identity(),  # your NN
    CRF(n_spatial_dims=2, requires_grad=False, filter_size=11, n_iter=5, smoothness_theta=1, smoothness_weight=1)
)

img = Image.open("C:/Users/svenk/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data_GAN/predictions/test_10.png")
tt = transforms.ToTensor()
img_tensor = tt(img)
print(img_tensor.shape[1:])

batch_size, n_channels, spatial = 1, 1, img_tensor.shape[1:]
x = torch.zeros(batch_size, n_channels, *spatial)
log_proba = model(x)
print(log_proba.max())