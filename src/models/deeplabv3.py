from PIL import Image
import os
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead
from torchvision import transforms


from src.criterion.dice_loss import dice_loss


import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class DeepLabv3RunnerClass:
    def __init__(self, config):
        self.config = config

        self.init_model()
        self.init_criterion()

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

            return bce(input, target) + 1.5*dice(input, target)

        self.criterion = forward


    
    def forward(self, inputs):
        outputs = self.model(inputs)['out']
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


