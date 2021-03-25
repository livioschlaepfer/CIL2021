from torch.utils import data
from torchvision import transforms
import numpy as np
from sklearn.model_selection import train_test_split

import glob
from PIL import Image


def init_train_dataloaders(config):

    # Load data paths
    image_paths = glob.glob(config.image_dir + '/*.png')
    mask_paths = glob.glob(config.mask_dir + '/*.png')

    print('Creating training and validation splits')

    # Create training and validation splits
    image_paths_train, image_paths_val, mask_paths_train, mask_paths_val = train_test_split(image_paths, mask_paths, test_size = config.val_size, random_state = config.seed)
    
    image_paths = {'train': image_paths_train, 'val': image_paths_val}
    mask_paths = {'train': mask_paths_train, 'val': mask_paths_val}

    print('Initializing datasets and dataloader')

    # Create training and validation datasets
    image_datasets = {x: SegmentationDataSet(image_paths=image_paths[x], mask_paths=mask_paths[x]) for x in ['train', 'val']}
    
    # Create training and validation dataloaders
    dataloaders_dict = {x: data.DataLoader(image_datasets[x], batch_size=config.batch_size, shuffle=True) for x in ['train', 'val']}

    return image_datasets, dataloaders_dict

class SegmentationDataSet(data.Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.prep_image =   transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])
        self.prep_mask =    transforms.Compose([
                                transforms.ToTensor()
                            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        # Load input and target
        image = Image.open(self.image_paths[index])
        mask = Image.open(self.mask_paths[index])

        # image.show() # For testing only
        # mask.show() # For testing only

        # Fix mask segmentation class colors
        mask = np.array(mask)
        mask[mask > 35] = 255 # set road to white #TODO: check threshold
        mask[mask <= 35] = 0 # set backgournd to dark
        mask = Image.fromarray(np.uint8(mask))

        # Transformation / Augmentation
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        # Preprocessing
        image = self.prep_image(image)
        mask = self.prep_mask(mask)

        return image, mask