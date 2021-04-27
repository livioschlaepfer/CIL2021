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

            loss1 = 0 * bce(input, target) + 1 * dice(input, target)

            input = self.image_to_patched(input, 8)
            target = self.mask_to_patched(target, 8)

            loss2 = 0 * bce(input, target) + 1 * dice(input, target)

            input = self.image_to_patched(input, 16)
            target = self.mask_to_patched(target, 16)

            loss3 = 0 * bce(input, target) + 1 * dice(input, target)

            input = self.image_to_patched(input, 4)
            target = self.mask_to_patched(target, 4)

            loss4 = 0 * bce(input, target) + 1 * dice(input, target)

            print("loss1", loss1, "loss2", loss2, "loss3", loss3, "loss4", loss4 )

            return loss1 + 0.2 * loss2 + 0.2 * loss3 + 0.2 * loss4

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
    def mask_to_patched(self, mask, patch_size = 16):
        foreground_threshold = 0.25
        patcher = nn.AvgPool2d(patch_size)

        mask = patcher(mask)

        thresholded_mask = mask
        thresholded_mask[:,0][mask[:,0] > foreground_threshold] = 1
        thresholded_mask[:,1][mask[:,1] <= foreground_threshold] = 0
        
        return thresholded_mask


    def image_to_patched(self, image, patch_size):
        patcher = nn.AvgPool2d(patch_size)

        return patcher(image)
        


