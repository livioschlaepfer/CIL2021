from box import Box
import yaml
import glob

from torch.utils import data

config = Box.from_yaml(filename="./config.yaml", Loader=yaml.FullLoader)

print("loaded config")

# Get names of test images
test_image_paths = glob.glob(config.paths.test_image_dir + '/*.png')

image_names = [x.split("/")[-1] for x in test_image_paths]

print(test_output_paths)