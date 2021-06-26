from src.models.deeplabv3 import DeepLabv3RunnerClass
from src.models.deeplabv3_regularizer import DeepLabv3RegularizerRunnerClass
from models.trivial_baseline import TrivialRunnerClass
from models.fcn_resnet50_baseline import FCNResnet50RunnerClass
from models.unet_baseline import UNetRunnerClass


def init_runner(config):
    if config.model_name == "deeplabv3":
        runner = DeepLabv3RunnerClass(config)
        return runner

    if config.model_name == "trivial_baseline":
        runner = TrivialRunnerClass(config)
        return runner

    if config.model_name == "fcnres50_baseline": #pretrained
        runner = FCNResnet50RunnerClass(config)
        return runner
    
    if config.model_name == "unet_baseline": #not pretrained
        runner = UNetRunnerClass(config)
        return runner