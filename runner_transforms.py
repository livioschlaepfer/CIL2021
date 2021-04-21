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
import getpass
import torchvision.transforms.functional as TF

from src.paths import paths_setter


# load config
config = Box.from_yaml(filename="./config.yaml", Loader=yaml.FullLoader)

# update paths based on user name
username = getpass.getuser()
config.paths = paths_setter(username=username)

to_tensor = transforms.ToTensor()
transform_to_png = transforms.ToPILImage()

def main():

    # fix seed
    np.random.seed(config.seed)

    # Load data paths for input
    if not os.path.exists(config.paths.train_image_dir_aug_input):
        raise OSError("Does not exist", config.paths.train_image_dir_aug_input)

    if not os.path.exists(config.paths.train_mask_dir_aug_input):
        raise OSError("Does not exist", config.paths.train_mask_dir_aug_input)

    image_paths = glob.glob(config.paths.train_image_dir_aug_input + '/*.png')
    mask_paths = glob.glob(config.paths.train_mask_dir_aug_input + '/*.png')

    print("image paths", len(image_paths))
    print("mask paths", len(mask_paths))

    # Create output folder if not existing
    if not os.path.exists(config.paths.train_mask_dir_aug_output):
        os.makedirs(config.paths.train_mask_dir_aug_output)

    if not os.path.exists(config.paths.train_image_dir_aug_output):
        os.makedirs(config.paths.train_image_dir_aug_output)

    for index in range(len(image_paths)):
        path_tail = os.path.splitext(os.path.basename(image_paths[index]))[0]

        print("Started augmentation for", path_tail)

        image = to_tensor(Image.open(image_paths[index]))
        mask = to_tensor(Image.open(mask_paths[index]))

        center_rotation(image, mask, path_tail)
        five_crop(image, mask, path_tail)

        
# Center crop rotation
def center_rotation(image, mask, path_tail):
    for angle in [30, 60, 120, 150, 210, 240, 300, 330]:

        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)

        # Center crop to avoid black edges the best we can 
        cc = transforms.CenterCrop(config.transforms.crop_size)
        image = cc(image)
        mask = cc(mask) 

        transform_to_png(image).convert("RGB").save(config.paths.train_image_dir_aug_output + "/" + path_tail + "_" + "rot" + str(angle) + ".png")
        transform_to_png(mask).save(config.paths.train_mask_dir_aug_output + "/" + path_tail + "_" + "rot" + str(angle) + ".png")
        
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
        transform_to_png(image).convert("RGB").save(config.paths.train_image_dir_aug_output + "/" + path_tail + "_" + str(index) + ".png")
        transform_to_png(mask).save(config.paths.train_mask_dir_aug_output + "/" + path_tail + "_" + str(index) + ".png")

        for angle in [90, 180, 270]:
    
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

            # Center crop to avoid black edges the best we can 
            cc = transforms.CenterCrop(config.transforms.crop_size)
            image = cc(image)
            mask = cc(mask) 

            transform_to_png(image).convert("RGB").save(config.paths.train_image_dir_aug_output + "/" + path_tail + "_" + str(index) + "_rot" + str(angle) + ".png")
            transform_to_png(mask).save(config.paths.train_mask_dir_aug_output + "/" + path_tail + "_" + str(index) + "_rot" + str(angle) + ".png")


if __name__ == "__main__":
    main()

