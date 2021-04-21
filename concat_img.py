import os
import numpy as np
from PIL import Image
import cv2

# define paths for translation from domain A (images in folderA) -> domain B (images in folderB)
folderA = 'C:/Users/svenk/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/training/images/1'
folderB = 'C:/Users/svenk/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data/training/groundtruth/1'
dest_path = 'C:/Users/svenk/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data_GAN/train'

splits = os.listdir(folderA)

for sp in splits:
    img_fold_A = os.path.join(folderA, sp)
    img_fold_B = os.path.join(folderB, sp)
    img_list = os.listdir(img_fold_A)
    num_imgs = len(img_list)
    img_fold_AB = os.path.join(dest_path, sp)
    if not os.path.isdir(img_fold_AB):
        os.makedirs(img_fold_AB)
    print('split = %s, number of images = %d' % (sp, num_imgs))
    for n in range(num_imgs):
        name_A = img_list[n]
        path_A = os.path.join(img_fold_A, name_A)
        name_B = name_A
        path_B = os.path.join(img_fold_B, name_B)
        if os.path.isfile(path_A) and os.path.isfile(path_B):
            name_AB = name_A
            path_AB = os.path.join(img_fold_AB, name_AB)
            im_A1 = Image.open(path_A)
            im_A = np.array(im_A1)
            im_B1 = Image.open(path_B)
            im_B1 = im_B1.convert(mode='RGB')
            im_B = np.array(im_B1)
            im_AB = np.concatenate([im_A, im_B], 1)
            cv2.imwrite(path_AB, im_AB)