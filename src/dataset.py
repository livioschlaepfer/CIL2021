import os
from torch.utils import data
from torchvision import transforms
import numpy as np
from sklearn.model_selection import train_test_split
import glob
from PIL import Image
import torchvision.transforms.functional as TF
from torch import random

from src.transforms import init_data_transforms

#from src.transforms import init_data_transforms


def init_train_dataloaders(paths, config):

    # Load data paths
    if not os.path.exists(paths.path_dict["train_image_dir"]):
        raise OSError("Does not exist", config.paths.train_image_dir)

    if not os.path.exists(paths.path_dict["train_mask_dir"]):
        raise OSError("Does not exist", config.paths.train_mask_dir)
    
    image_paths = glob.glob(paths.path_dict["train_image_dir"] + '/*.png')
    mask_paths = glob.glob(paths.path_dict["train_mask_dir"] + '/*.png')

    print('Creating training and validation splits')

    # Create training and validation splits
    image_paths_train, image_paths_val, mask_paths_train, mask_paths_val = train_test_split(image_paths, mask_paths, test_size = config.val_size, random_state = config.seed)
    
    image_paths = {'train': image_paths_train, 'val': image_paths_val}
    mask_paths = {'train': mask_paths_train, 'val': mask_paths_val}

    # Get transforms
    data_transforms = init_data_transforms(config)

    print('Initializing datasets and dataloader for training')

    # Create training and validation datasets
    image_datasets = {x: SegmentationDataSet(image_paths=image_paths[x], mask_paths=mask_paths[x], transform=data_transforms) for x in ['train', 'val']}
    
    # Create training and validation dataloaders
    dataloaders_dict = {x: data.DataLoader(image_datasets[x], batch_size=config.batch_size, shuffle=True) for x in ['train', 'val']}

    return image_datasets, dataloaders_dict


def init_test_dataloaders(paths):
    
    # Load data paths
    if not os.path.exists(paths.path_dict["test_image_dir"]):
        raise OSError("Does not exist", paths.path_dict["test_image_dir"])

    image_paths = glob.glob(paths.path_dict["test_image_dir"] + '/*.png')

    print('Initializing datasets and dataloader for testing')

    # Create training and validation datasets
    image_datasets = {'test': SegmentationDataSet(image_paths=image_paths)}
    
    # Create training and validation dataloaders
    dataloaders_dict = {'test': data.DataLoader(image_datasets['test'], shuffle=False)}

    return image_datasets, dataloaders_dict

class SegmentationDataSet(data.Dataset):
    def __init__(self, image_paths, mask_paths=None, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.prep_image =   transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])
        # self.prep_image =   transforms.Compose([
        #                         transforms.ToTensor(),
        #                     ])
        self.prep_mask =    transforms.Compose([
                                transforms.ToTensor()
                            ])
        if (mask_paths == None): 
            self.training_run = False
        else: 
            self.training_run = True

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        # Load input and target
        image = Image.open(self.image_paths[index])
        if self.training_run:
            mask = Image.open(self.mask_paths[index])

        # image.show() # For testing only
        # mask.show() # For testing only

        # Normalize image
        image = self.prep_image(image)

        if self.training_run:
            # Transformation / Augmentation
            if self.transform is not None:
                image, mask =  self.transform(image, mask)
        
            # One hot encode segmentation classes based on segmentation class colors
            mask = np.array(mask)
            
            mask[mask > 0.1] = 1. # One hot encode road #TODO: determine optimal threshold
            mask[mask <= 0.1] = 0. # One hot encode background #TODO: determine optimal threshold

            print(mask.shape)

        if self.training_run:
            return image, mask
        else:
            return transforms.ToTensor()(image), self.image_paths[index]