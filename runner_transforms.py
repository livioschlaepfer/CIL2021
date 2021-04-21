# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from sys import path_hooks
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
import glob
from PIL import Image
import torchvision.transforms.functional as TF


from src.dataset import init_test_dataloaders, init_train_dataloaders
from src.trainer import train_model
from src.tester import test_model
from src.models.model_runner import init_runner


# load config
config = Box.from_yaml(filename="./config.yaml", Loader=yaml.FullLoader)

to_tensor = transforms.ToTensor()
transform_to_png = transforms.ToPILImage()

def main():

    # fix seed
    np.random.seed(config.seed)

    # Load data paths
    if not os.path.exists(config.paths.train_image_dir):
        raise OSError("Does not exist", config.paths.train_image_dir)

    if not os.path.exists(config.paths.train_mask_dir):
        raise OSError("Does not exist", config.paths.train_mask_dir)

    image_paths = glob.glob(config.paths.train_image_dir + '/*.png')
    mask_paths = glob.glob(config.paths.train_mask_dir + '/*.png')

    print("image paths", len(image_paths))
    print("mask paths", len(mask_paths))

    # Create output folder if not existing
    if not os.path.exists(config.paths.train_mask_dir_aug):
        os.makedirs(config.paths.train_mask_dir_aug)

    if not os.path.exists(config.paths.train_image_dir_aug):
        os.makedirs(config.paths.train_image_dir_aug)

    for index in range(len(image_paths)):
        path_tail = os.path.splitext(os.path.basename(image_paths[index]))[0]

        print("Started augmentation for", path_tail)

        image = to_tensor(Image.open(image_paths[index]))
        mask = to_tensor(Image.open(mask_paths[index]))

        rotation(image, mask, path_tail)
        five_crop(image, mask, path_tail)

        
# Center crop rotation
def rotation(image, mask, path_tail):
    for angle in [30, 60, 120, 150, 210, 240, 300, 330]:

        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)

        # Center crop to avoid black edges the best we can 
        cc = transforms.CenterCrop(config.transforms.crop_size)
        image = cc(image)
        mask = cc(mask) 

        transform_to_png(image).convert("RGB").save(config.paths.train_image_dir_aug + "/" + path_tail + "_" + "rot" + str(angle) + ".png")
        transform_to_png(mask).convert("RGB").save(config.paths.train_mask_dir_aug + "/" + path_tail + "_" + "rot" + str(angle) + ".png")

# Five crop augmentation
def five_crop(image, mask, path_tail):
    # Five crop image
    fc = transforms.FiveCrop(config.transforms.crop_size)
    images = fc(image)
    masks = fc(mask)

    for index in range(len(images)):

        image = images[index]
        mask = masks[index]

        # normal five crop
        transform_to_png(image).convert("RGB").save(config.paths.train_image_dir_aug + "/" + path_tail + "_" + str(index) + ".png")
        transform_to_png(mask).convert("RGB").save(config.paths.train_mask_dir_aug + "/" + path_tail + "_" + str(index) + ".png")

        v_flip(image, mask, path_tail, index)
        h_flip(image, mask, path_tail, index)
        vh_flip(image, mask, path_tail, index)
        hv_flip(image, mask, path_tail, index)

# Horizontal flipping      
def h_flip(image, mask, path_tail, index):
    image = TF.hflip(image)
    mask = TF.hflip(mask)

    transform_to_png(image).convert("RGB").save(config.paths.train_image_dir_aug + "/" + path_tail + "_" + str(index) + "_h_flip" + ".png")
    transform_to_png(mask).convert("RGB").save(config.paths.train_mask_dir_aug + "/" + path_tail + "_" + str(index) + "_h_flip" + ".png")

# Hertical flipping
def v_flip(image, mask, path_tail, index):
    image = TF.vflip(image)
    mask = TF.vflip(mask)

    transform_to_png(image).convert("RGB").save(config.paths.train_image_dir_aug + "/" + path_tail + "_" + str(index) + "_v_flip" + ".png")
    transform_to_png(mask).convert("RGB").save(config.paths.train_mask_dir_aug + "/" + path_tail + "_" + str(index) + "_v_flip" + ".png")

# Vertical Horizontal flipping
def vh_flip(image, mask, path_tail, index):
    image = TF.vflip(image)
    mask = TF.vflip(mask)

    image = TF.hflip(image)
    mask = TF.hflip(mask)

    transform_to_png(image).convert("RGB").save(config.paths.train_image_dir_aug + "/" + path_tail + "_" + str(index) + "_vh_flip" + ".png")
    transform_to_png(mask).convert("RGB").save(config.paths.train_mask_dir_aug + "/" + path_tail + "_" + str(index) + "_vh_flip" + ".png")

# Horizontal Hertical flipping
def hv_flip(image, mask, path_tail, index):
    image = TF.hflip(image)
    mask = TF.hflip(mask)

    image = TF.vflip(image)
    mask = TF.vflip(mask)

    transform_to_png(image).convert("RGB").save(config.paths.train_image_dir_aug + "/" + path_tail + "_" + str(index) + "_hv_flip" + ".png")
    transform_to_png(mask).convert("RGB").save(config.paths.train_mask_dir_aug + "/" + path_tail + "_" + str(index) + "_hv_flip" + ".png")


if __name__ == "__main__":
    main()

