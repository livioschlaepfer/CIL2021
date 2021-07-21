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
import cv2
import numpy as np
from skimage.morphology import area_closing, area_opening, binary_closing, binary_dilation, binary_erosion, disk, square
import matplotlib.pyplot as plt


from src.paths import paths_setter
from src.seed import seed_all

parser = argparse.ArgumentParser()
parser.add_argument('--config', 
    default = 'custom',
    choices = ['custom', 'baseline_fcn', 'baseline_unet', 'baseline_deeplab'],
    help = 'Select one of the experiments described in our report or setup a custom config file'
)
parser.add_argument("name", help="name of the majority voting instance")
parser.add_argument("--voting_rule", default="majority", type=str, help = "voting rule between the results of the different models. Choices: ['majority', 'max']")
parser.add_argument("--morph_post", action="store_true", help = "flag to apply morphological postprocessing")
parser.add_argument("--models", type=str, help="provide list of models you want to take part in majority voting, delimited by a comma. Example: 'model1,model2' ")
parser.add_argument("--model_seeds", type=str, help="per model in models, please indicate which seed should be used, delimited by a comma. Example: '1,3' ")
args = parser.parse_args()

# load config
try: 
    config = Box.from_yaml(filename="./configs/"+ args.config + ".yaml", Loader=yaml.FullLoader)
except:
    raise OSError("Does not exist", args.config)

# update paths based on user name
username = getpass.getuser()
config.paths = paths_setter(username=username, pretrain=config.pretrain, mixed_train=config.mixed_train)

# start of majority voting
models = [str(item) for item in args.models.split(',')]
seeds = [str(item) for item in args.model_seeds.split(',')]

# length check
assert(len(models)==len(seeds)), "make sure number of models and seeds are the same"

# stack the images
images = []
for model, seed in zip(models, seeds):
    images_paths = os.listdir(config.paths.model_store +"/"+model+"/predictions_seed_"+seed)
    imgs = []
    for img in images_paths:
        im = cv2.imread(os.path.join(config.paths.model_store +"/"+model+"/predictions_seed_"+seed+"/",img), cv2.IMREAD_GRAYSCALE)
        imgs.append(im)
    images.append(imgs)
images = np.array(images)

# majority voting
if args.voting_rule == "majority":
    # sum up over first dimension
    sum = np.sum(images, axis=0)
    sum_copy = sum.copy()
    sum_copy[sum>=255*((images.shape[0]/2)+1)] = 1
    sum_copy[sum<255*((images.shape[0]/2)+1)] = 0
    binary_imgs = sum_copy
elif args.voting_rule == "max":
    # sum up over first dimension
    sum = np.sum(images, axis=0)
    sum_copy = sum.copy()
    sum_copy[sum>=255] = 1
    sum_copy[sum<255] = 0
    binary_imgs = sum_copy

# plt.imshow(binary_imgs[0,:,:])
# plt.show()

# apply morphological postprocessing if chosen
if args.morph_post:
    for img in range(binary_imgs.shape[0]):
        binary = area_closing(binary_imgs[img,:,:], area_threshold=500)
        binary = area_opening(binary, area_threshold=500)
        footprint = disk(10)
        footprint_1= disk(14)
        binary = binary_dilation(binary, footprint)
        binary_imgs[img,:,:] = binary_erosion(binary, footprint_1)

# save the images

# check if dir exists, otherwise create
if not os.path.exists(config.paths.model_store +"/"+args.name+"/predictions/"):
    os.makedirs(config.paths.model_store +"/"+args.name+"/predictions/")

for img, name in zip(range(binary_imgs.shape[0]), images_paths):
    pil = Image.fromarray((binary_imgs[img,:,:]*255).astype(np.uint8), mode="L")
    pil.save(config.paths.model_store +"/"+args.name+"/predictions/" +name)

print("successfull")