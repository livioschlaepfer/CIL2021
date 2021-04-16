

from src.models.deeplabv3_regularizer import DeepLabv3RegularizerRunnerClass
from src.models.deeplabv3 import DeepLabv3RunnerClass


def init_runner(config):
    if config.model_name == "deeplabv3":
        runner = DeepLabv3RunnerClass(config)
        return runner

    if config.model_name == "deeplabv3_regularizer":
        runner = DeepLabv3RegularizerRunnerClass(config)
        return runner