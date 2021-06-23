import os
import numpy as np
import torch
from torchvision import datasets, models, transforms
import torchvision.transforms.functional as TF
from PIL import Image
from cv2 import cv2
import time
import sys
from tqdm import tqdm

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.size[1] + im2.size[1], im1.size[0]))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.size[1], 0))
    return dst

folder_A = '/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/training/images/'
folder_B = '/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/training/groundtruth/'
folder_AB = '/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data_GAN/data/train/'

folder_val_orig = '/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/test_images/test_images'
folder_val = '/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data_GAN/data/val/'

folder_livio_A = '/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data_GAN/data/val/'
folder_livio_B = '/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data_GAN/predictions/Livio/'
folder_livio_AB = '/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data_GAN/data/livio/'

filtering = True

train = False

val = False

livio = True

assert len(os.listdir(folder_A)) == len(os.listdir(folder_B)), "make sure number of images is equal to number of masks"
if train:
    assert len(os.listdir(folder_AB)) == 0, "please delete old images"
if val:
    assert len(os.listdir(folder_val)) == 0, "please delete old images in val"
if livio:
    assert len(os.listdir(folder_livio_AB)) == 0, "please delete old images in livio"

img_names = os.listdir(folder_A) # suffices to look at folder A since corresponding images have the same name

img_names_val = os.listdir(folder_val_orig)

img_names_livio = os.listdir(folder_livio_A)

if val:
    for name in img_names_val:
        img_val = Image.open(os.path.join(folder_val_orig, name))
        if filtering:
            img_val = cv2.bilateralFilter(np.array(img_val), 15, 150, 150)
        img_val = transforms.ToTensor()(img_val)
        img_val = transforms.ToPILImage()(img_val)
        path_val = folder_val  + name
        img_val.save(path_val)

if train:
    for name in tqdm(img_names):
        img_A = Image.open(os.path.join(folder_A, name))
        if filtering:
            img_A = cv2.bilateralFilter(np.array(img_A), 15, 150, 150)
        img_B = Image.open(os.path.join(folder_B, name)).convert("RGB")

        img_A_ten = transforms.ToTensor()(img_A)
        img_B_ten = transforms.ToTensor()(img_B)
        
        # define the centercrop
        cc = transforms.CenterCrop(256)
        # define the Colorjitter
        cj = transforms.ColorJitter(0.5, 0.5, 0.5, 0.5)
        for angle in [30, 60, 120, 150, 210, 240, 300, 330]:
            # rotate image and mask
            img_A_rot = TF.rotate(img_A_ten, angle)
            img_B_rot = TF.rotate(img_B_ten, angle)
            # colorjitter image only
            img_A_rot_jit = cj(img_A_rot)
            img_A_rot_cc = transforms.ToPILImage()(cc(img_A_rot))
            img_A_rot_cc_jit = transforms.ToPILImage()(cc(img_A_rot_jit))
            img_B_rot_cc = transforms.ToPILImage()(cc(img_B_rot))
            img_AB = get_concat_h(img_A_rot_cc, img_B_rot_cc)
            img_AB_jit = get_concat_h(img_A_rot_cc_jit, img_B_rot_cc)
            path_AB = folder_AB  + str(angle) + "_" + "cc" + "_" + name
            path_AB_jit = folder_AB  + str(angle) + "_" + "cc" + "_jit_" + name
            img_AB.save(path_AB)
            img_AB_jit.save(path_AB_jit)

        fc = transforms.FiveCrop(256)
        img_A_fc = fc(img_A_ten)
        img_B_fc = fc(img_B_ten)
        for i in range(len(img_A_fc)):
            A = img_A_fc[i]
            B = img_B_fc[i]
            A_1 = transforms.ToPILImage()(A)
            B_1 = transforms.ToPILImage()(B)
            A_1_jit = transforms.ToPILImage()(cj(A))
            AB = get_concat_h(A_1, B_1)
            AB_jit = get_concat_h(A_1_jit, B_1)
            path_AB = folder_AB + "fc_" + str(i) + "_noflip"  + name
            path_AB_jit = folder_AB + "fc_" + str(i) + "_noflip_jit_"  + name
            AB.save(path_AB)
            AB_jit.save(path_AB_jit)
            for angle in [90, 180, 270]:
                A_rot = transforms.ToPILImage()(TF.rotate(A, angle))
                B_rot = transforms.ToPILImage()(TF.rotate(B, angle))
                A_rot_jit = cj(A_rot)
                AB_rot = get_concat_h(A_rot, B_rot)
                AB_rot_jit = get_concat_h(A_rot_jit, B_rot)
                path_AB = folder_AB + "fc_" + str(i) + "_" + str(angle)  + name
                path_AB_jit = folder_AB + "fc_" + str(i) + "_" + str(angle) + "_jit_" + name
                AB_rot.save(path_AB)
                AB_rot_jit.save(path_AB_jit)  

if livio:
    for name in tqdm(img_names_livio):
        img_A = Image.open(os.path.join(folder_livio_A, name))
        if filtering:
            img_A = cv2.bilateralFilter(np.array(img_A), 15, 150, 150)
        img_B = Image.open(os.path.join(folder_livio_B, name)).convert("RGB")

        img_A_ten = transforms.ToTensor()(img_A)
        img_B_ten = transforms.ToTensor()(img_B)
        
        # define the centercrop
        cc = transforms.CenterCrop(256)
        # define the Colorjitter
        cj = transforms.ColorJitter(0.5, 0.5, 0.5, 0.5)
        for angle in [30, 60, 120, 150, 210, 240, 300, 330]:
            # rotate image and mask
            img_A_rot = TF.rotate(img_A_ten, angle)
            img_B_rot = TF.rotate(img_B_ten, angle)
            # colorjitter image only
            img_A_rot_jit = cj(img_A_rot)
            img_A_rot_cc = transforms.ToPILImage()(cc(img_A_rot))
            img_A_rot_cc_jit = transforms.ToPILImage()(cc(img_A_rot_jit))
            img_B_rot_cc = transforms.ToPILImage()(cc(img_B_rot))
            img_AB = get_concat_h(img_A_rot_cc, img_B_rot_cc)
            img_AB_jit = get_concat_h(img_A_rot_cc_jit, img_B_rot_cc)
            path_AB = folder_livio_AB  + str(angle) + "_" + "cc" + "_" + name
            path_AB_jit = folder_livio_AB  + str(angle) + "_" + "cc" + "_jit_" + name
            img_AB.save(path_AB)
            img_AB_jit.save(path_AB_jit)

        fc = transforms.FiveCrop(256)
        img_A_fc = fc(img_A_ten)
        img_B_fc = fc(img_B_ten)
        for i in range(len(img_A_fc)):
            A = img_A_fc[i]
            B = img_B_fc[i]
            A_1 = transforms.ToPILImage()(A)
            B_1 = transforms.ToPILImage()(B)
            A_1_jit = transforms.ToPILImage()(cj(A))
            AB = get_concat_h(A_1, B_1)
            AB_jit = get_concat_h(A_1_jit, B_1)
            path_AB = folder_livio_AB + "fc_" + str(i) + "_noflip"  + name
            path_AB_jit = folder_livio_AB + "fc_" + str(i) + "_noflip_jit_"  + name
            AB.save(path_AB)
            AB_jit.save(path_AB_jit)
            for angle in [90, 180, 270]:
                A_rot = transforms.ToPILImage()(TF.rotate(A, angle))
                B_rot = transforms.ToPILImage()(TF.rotate(B, angle))
                A_rot_jit = cj(A_rot)
                AB_rot = get_concat_h(A_rot, B_rot)
                AB_rot_jit = get_concat_h(A_rot_jit, B_rot)
                path_AB = folder_livio_AB + "fc_" + str(i) + "_" + str(angle)  + name
                path_AB_jit = folder_livio_AB + "fc_" + str(i) + "_" + str(angle) + "_jit_" + name
                AB_rot.save(path_AB)
                AB_rot_jit.save(path_AB_jit)


        #print(name)

print("done!")