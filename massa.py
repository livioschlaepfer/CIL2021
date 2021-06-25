import glob
from box import Box
import yaml
import getpass
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch

from src.paths import paths_setter

config = Box.from_yaml(filename="./config.yaml", Loader=yaml.FullLoader)

# update paths based on user name
username = getpass.getuser()
config.paths = paths_setter(username=username)

image_paths = glob.glob(config.paths.massa_images + '/*.png')
mask_paths = glob.glob(config.paths.massa_masks + '/*.png')

for i, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
    image = transforms.ToTensor()(Image.open(img_path))
    mask = transforms.ToTensor()(Image.open(mask_path))

    compound = torch.cat((image, mask), dim=0)

    fc = transforms.FiveCrop(750)
    rc = transforms.RandomCrop(size=400)

    for j, com in enumerate(fc(compound)):
        cropped = rc(com)
        img_crop= transforms.ToPILImage()(cropped[0:3])
        mask_crop= transforms.ToPILImage()(cropped[3])
        img_crop.save(config.paths["massa_images_aug"] + '/' + str(i) + '_' + str(j) + '.png')
        mask_crop.save(config.paths["massa_masks_aug"] + '/' + str(i) + '_' + str(j) + '.png')

print("done")