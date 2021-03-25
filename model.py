import torch.nn as nn
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead


import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def initialize_model(config):
    print('Initializing model')

    model = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True, aux_loss=None)
    
    # Update number of segmentation classes in classifier and auxillary classifier
    model.classifier = DeepLabHead(2048, config.num_classes)
    model.aux_classifier = FCNHead(1024, config.num_classes)
    
    # Print the model we just instantiated
    print(model)

    return model

