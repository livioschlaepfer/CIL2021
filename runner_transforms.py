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

    for angle in [30, 35, 40, 50, 55, 60, 65, 70, 75, 80, 85, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 185, 190, 195, 200, 205, 210, 240, 300, 330]:

        image_rot = TF.rotate(image, angle)
        mask_rot = TF.rotate(mask, angle)

        # Center crop to avoid black edges the best we can 
        cc = transforms.CenterCrop(config.transforms.crop_size)
        image_rot = cc(image_rot)
        mask_rot = cc(mask_rot) 

        transform_to_png(image_rot).convert("RGB").save(config.paths.train_image_dir_aug_output + "/" + path_tail + "_" + "rot" + str(angle) + ".png")
        transform_to_png(mask_rot).save(config.paths.train_mask_dir_aug_output + "/" + path_tail + "_" + "rot" + str(angle) + ".png")

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

        h_flip(image, mask, path_tail + "_" + str(index), index)
        rotate_90(image, mask, path_tail + "_" + str(index), index)
        color_jitter(image, mask, path_tail + "_" + str(index), index)

# Horizontal flipping      
def h_flip(image, mask, path_tail, index):
    image = TF.hflip(image)
    mask = TF.hflip(mask)

    transform_to_png(image).convert("RGB").save(config.paths.train_image_dir_aug_output + "/" + path_tail + "_h_flip" + ".png")
    transform_to_png(mask).save(config.paths.train_mask_dir_aug_output + "/" + path_tail + "_h_flip" + ".png")

    rotate_90(image, mask, path_tail + "_h_flip", index)
    color_jitter(image, mask, path_tail + "_h_flip", index)


# Rotation by 90s multiply
def rotate_90(image, mask, path_tail, index):
    for angle in [90, 180, 270]:
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)

        transform_to_png(image).convert("RGB").save(config.paths.train_image_dir_aug_output + "/" + path_tail + "_rot" + str(angle) + ".png")
        transform_to_png(mask).save(config.paths.train_mask_dir_aug_output + "/" + path_tail + "_rot" + str(angle) + ".png")

# Apply random color jitter
def color_jitter(image, mask, path_tail, index):
    jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0)

    image = jitter(image)

    transform_to_png(image).convert("RGB").save(config.paths.train_image_dir_aug_output + "/" + path_tail + "_jitter" + ".png")
    transform_to_png(mask).save(config.paths.train_mask_dir_aug_output + "/" + path_tail + "_jitter" + ".png")


if __name__ == "__main__":
    main()

