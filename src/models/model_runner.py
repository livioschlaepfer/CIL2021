

from models.deeplabv3 import DeepLabv3RunnerClass


def init_runner(config):
    if config.model_name == "deeplabv3":
        runner = DeepLabv3RunnerClass(config)
        return runner