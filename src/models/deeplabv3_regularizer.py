from typing import ClassVar
from PIL import Image
import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead
from torchvision import transforms
from torch.autograd import Variable



from src.criterion.dice_loss import dice_loss


import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class DeepLabv3RegularizerRunnerClass:
    def __init__(self, config):
        self.config = config

        self.init_model()
        self.init_criterion()

    def init_model(self):
        print('Initializing model')

        model = DeepLabv3WithRegularizer(self.config)
        
        # Print the model we just instantiated
        print(model)

        self.model = model
    

    def init_criterion(self):
        # Setup the loss
        # self.criterion = torch.nn.MSELoss(reduction='mean')
        # self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.BCELoss()
        # self.criterion = nn.NLLLoss()
        # self.criterion = DiceLoss()

        def forward(input, target):
            bce = nn.BCELoss()
            dice = dice_loss()

            print("BCE", bce(input, target))
            print("Dice", dice(input, target))

            return 0.2 * bce(input, target) + 0.8 * dice(input, target)

        self.criterion = forward

    def forward(self, inputs):
        outputs = self.model(inputs)
        outputs = torch.sigmoid(outputs)

        return outputs

    def loss(self, outputs, labels):
        loss = self.criterion(outputs.float(), labels.float())
        return loss

    def convert_to_png(self, output):

        binary = output.argmin(0)
        binary = torch.tensor(binary, dtype=torch.float64)
        binary = transforms.ToPILImage(mode="L")(binary).convert("RGB")

        return binary



class DeepLabv3WithRegularizer(nn.Module):
    def __init__(self, config):
        super().__init__() 

        self.config = config

        self.model_segmentation = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True, aux_loss=None)
        
        # Update number of segmentation classes in classifier and auxillary classifier
        self.model_segmentation.classifier = DeepLabHead(2048, self.config.num_classes)
        self.model_segmentation.aux_classifier = FCNHead(1024, self.config.num_classes)


        self.model_regularizer = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True, aux_loss=None)
        
        # Update number of segmentation classes in classifier and auxillary classifier
        self.model_regularizer.classifier = DeepLabHead(2048, self.config.num_classes)
        self.model_regularizer.aux_classifier = FCNHead(1024, self.config.num_classes)

    def forward(self, inputs):
        outputs = self.model_segmentation(inputs)['out']
        outputs = torch.sigmoid(outputs)

        # Padding to obtain 3 input channels
        zeros = Variable(torch.zeros(outputs.shape)[:,0].unsqueeze(1))

        print("zeros shape", zeros.shape)
        print("outputs shape", outputs.shape)

        outputs = torch.cat((outputs, zeros), 1)

        outputs = self.model_regularizer(outputs)['out']

        outputs = torch.sigmoid(outputs)

        return outputs
    


