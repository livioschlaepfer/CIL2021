import glob
import keyword
from box import Box
import yaml
import getpass
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch
import argparse
import os

from src.paths import paths_setter

parser = argparse.ArgumentParser()
parser.add_argument("name", help="name of the image production run")
parser.add_argument("--small_then_zoom", action="store_true", help = "option to pick a smaller section of the crop and resize it to 400x400")
parser.add_argument("--make_multiple", action="store_true", help="option to produce multiple random images from one crop")
parser.add_argument("--n_multiple", default=3, help="number of random images per crop")
args = parser.parse_args()

config = Box.from_yaml(filename="./configs/custom.yaml", Loader=yaml.FullLoader)

# update paths based on user name
username = getpass.getuser()
config.paths = paths_setter(username=username)

image_paths = glob.glob(config.paths.massa_images + '/*.png')
mask_paths = glob.glob(config.paths.massa_masks + '/*.png')

if not os.path.exists(config.paths.massa_images_aug + '/' + args.name):
    os.makedirs(config.paths.massa_images_aug + '/' + args.name)

if not os.path.exists(config.paths.massa_masks_aug + '/' + args.name):
    os.makedirs(config.paths.massa_masks_aug + '/' + args.name)

for i, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
    image = transforms.ToTensor()(Image.open(img_path))
    mask = transforms.ToTensor()(Image.open(mask_path))

    compound = torch.cat((image, mask), dim=0)

    fc = transforms.FiveCrop(750)
    if args.small_then_zoom:
        rc = transforms.RandomCrop(size=256)
    else:
        rc = transforms.RandomCrop(size=400)

    if args.make_multiple:
        n = args.n_multiple
    else:
        n = 1

    for k in range(n):
        for j, com in enumerate(fc(compound)):
            cropped = rc(com)
            img_crop= transforms.ToPILImage()(cropped[0:3])
            mask_crop= transforms.ToPILImage()(cropped[3])
            if args.small_then_zoom:
                img_crop = img_crop.resize((400, 400))
                mask_crop = mask_crop.resize((400, 400))
                mask_crop = mask_crop.point(lambda x: 0 if x<128 else 255, '1')
            img_crop.save(config.paths.massa_images_aug + '/' + args.name + '/' + str(i) + '_' + str(k) + '_' + str(j) + '.png')
            mask_crop.save(config.paths.massa_masks_aug + '/' + args.name + '/' + str(i) + '_' + str(k) + '_' + str(j) + '.png')

print("done")