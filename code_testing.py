from box import Box
import yaml

from torch.utils import data

from dataset import init_train_dataloaders

config = Box.from_yaml(filename="./config.yaml", Loader=yaml.FullLoader)

print("loaded config")

image_datasets, dataloaders_dict = init_train_dataloaders(config)


x, y = next(iter(dataloaders_dict['train']))
print(f'x = shape: {x.shape}; type: {x.dtype}')
print(f'x = min: {x.min()}; max: {x.max()}')
print(f'y = shape: {y.shape}; class: {y.unique()}; type: {y.dtype}')