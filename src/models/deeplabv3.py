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

            input = self.image_to_patched(input)
            target = self.mask_to_patched(target)

            bce = nn.BCELoss()
            dice = dice_loss()

            loss = 1 * bce(input, target) + 0.5 * dice(input, target)

            print("loss", loss)

            return loss

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

    # assign a label to a patch
    def mask_to_patched(self, mask):
        foreground_threshold = 0.25
        patch_size = 16
        patcher = nn.AvgPool2d(patch_size)

        mask = patcher(mask)

        thresholded_mask = mask
        thresholded_mask[:,0][mask[:,0] > foreground_threshold] = 1
        thresholded_mask[:,1][mask[:,1] <= foreground_threshold] = 0
        
        return thresholded_mask


    def image_to_patched(self, image):
        patch_size = 16
        patcher = nn.AvgPool2d(patch_size)

        return patcher(image)
        


