from box import Box
import yaml
import glob
import json

from torch.utils import data

config = Box.from_yaml(filename="./configs/custom.yaml", Loader=yaml.FullLoader)

print("loaded config")

with open('/home/svkohler/OneDrive/Desktop/'+'config.txt', 'w') as f: # config.paths.model_store + '/' + config.checkpoint_name +'/config/'+
    f.write(json.dumps(config.to_dict()))