import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead


import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class DeepLabv3RunnerClass:
    def __init__(self, config):
        self.config = config

        self.init_model()
        self.init_criterion()
    
    def forward(self, inputs):
        outputs = self.model(inputs)['out']
        return outputs

    def loss(self, outputs, labels):
        loss = self.criterion(outputs.float(), labels.float())
        return loss

    def init_model(self):
        print('Initializing model')

        model = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True, aux_loss=None)
        
        # Update number of segmentation classes in classifier and auxillary classifier
        model.classifier = DeepLabHead(2048, self.config.num_classes)
        model.aux_classifier = FCNHead(1024, self.config.num_classes)
        
        # Print the model we just instantiated
        print(model)

        self.model = model

    def init_criterion(self):
        # Setup the loss
        self.criterion = torch.nn.MSELoss(reduction='mean')
        # criterion = nn.CrossEntropyLoss()
        # criterion = nn.NLLLoss()