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

folder_A = 'C:/Users/svenk/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/training/images/'
folder_B = 'C:/Users/svenk/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/training/groundtruth/'
folder_AB = 'C:/Users/svenk/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data_GAN/train/'

assert len(os.listdir(folder_A)) == len(os.listdir(folder_B)), "make sure number of images is equal to number of masks"

img_names = os.listdir(folder_A) # suffices to look at folder A since corresponding images have the same name

for name in img_names:
    img_A = Image.open(os.path.join(folder_A, name))
    img_B = Image.open(os.path.join(folder_B, name)).convert("RGB")

    img_A_ten = transforms.ToTensor()(img_A)
    img_B_ten = transforms.ToTensor()(img_B)
    
    # define the centercrop
    cc = transforms.CenterCrop(256)
    for angle in [30, 60, 120, 150, 210, 240, 300, 330]:
        img_A_rot = TF.rotate(img_A_ten, angle)
        img_B_rot = TF.rotate(img_B_ten, angle)
        img_A_rot_cc = transforms.ToPILImage()(cc(img_A_rot))
        img_B_rot_cc = transforms.ToPILImage()(cc(img_B_rot))
        img_AB = get_concat_h(img_A_rot_cc, img_B_rot_cc)
        path_AB = folder_AB  + str(angle) + "_" + "cc" + "_" + name
        img_AB.save(path_AB)

    fc = transforms.FiveCrop(256)
    img_A_fc = fc(img_A_ten)
    img_B_fc = fc(img_B_ten)
    for i in range(len(img_A_fc)):
        A = img_A_fc[i]
        B = img_B_fc[i]
        A_1 = transforms.ToPILImage()(A)
        B_1 = transforms.ToPILImage()(B)
        AB = get_concat_h(A_1, B_1)
        path_AB = folder_AB + "fc_" + str(i) + "_noflip"  + name
        AB.save(path_AB)
        for angle in [90, 180, 270]:
            A_rot = transforms.ToPILImage()(TF.rotate(A, angle))
            B_rot = transforms.ToPILImage()(TF.rotate(B, angle))
            AB_rot = get_concat_h(A_rot, B_rot)
            path_AB = folder_AB + "fc_" + str(i) + "_" + str(angle)  + name
            AB_rot.save(path_AB)  

    print(name)  

print("done!")