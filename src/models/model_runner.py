

from src.models.deeplabv3 import DeepLabv3RunnerClass
from src.models.bayesian_Unet import B_Unet

def init_runner(config):
    if config.model_name == "deeplabv3":
        runner = DeepLabv3RunnerClass(config)
        return runner

    if config.model_name == "bayesian_Unet":
        runner = B_Unet(config)
        return runner