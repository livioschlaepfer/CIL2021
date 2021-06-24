from PIL import Image
import os
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
import torch.nn.functional as F

from torchvision.models.segmentation.fcn import FCNHead

from src.criterion.dice_loss import DiceLoss
from src.criterion.cldice_loss import SoftDiceCLDice
from src.criterion.focal_loss import FocalLoss

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class DeepLabv3RunnerClass:
    def __init__(self, config):
        self.config = config

        self.init_model()
        self.init_criterion()

    def init_model(self):
        print('Initializing model')

        model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True, aux_loss=None)
        
        # Adapted deeplab head for given classification problem
        model.classifier = DeepLabHead(2048, self.config.num_classes)
        model.aux_classifier = FCNHead(1024, self.config.num_classes)
        
        # Print the instantiated model 
        print(model)

        self.model = model

    def init_criterion(self):
        # Setup the loss
        # self.criterion = torch.nn.MSELoss(reduction='mean')
        # self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.BCELoss()
        # self.criterion = nn.NLLLoss()

        def forward(input, target):

            if self.config.loss.name == "bce":
                loss = nn.BCELoss()

            elif self.config.loss.name == "dice":
                loss = DiceLoss()
                
            elif self.config.loss.name == "focal":
                loss = FocalLoss()

            elif self.config.loss.name == "cl_dice":
                loss = SoftDiceCLDice(iter_=self.config.loss.iter, smooth=self.config.loss.smooth, alpha=self.config.loss.alpha)

            elif self.config.loss.name == "bce_dice_with_patch":
                bce = nn.BCELoss()
                dice = DiceLoss()

                loss1 = 0.2 * bce(input, target) + 0.8 * dice(input, target)

                input_16 = self.image_to_patched(input, 16)
                target_16 = self.mask_to_patched(target, 16)

                loss3 = 0.2 * bce(input_16, target_16) + 0.8 * dice(input_16, target_16)

                print("loss1", loss1,"loss3", loss3)

                return loss1 + 0.5 * loss3 

            else: 
                raise OSError("Loss not exist:", self.config.loss.name)
        
            bce = nn.BCELoss()
            return bce(input, target) + loss(input, target)

        self.criterion = forward
  
    def forward(self, inputs):
        outputs = self.model(inputs)['out']
        outputs = torch.sigmoid(outputs)

        return outputs

    def loss(self, outputs, labels):
        print(outputs.shape)
        loss = self.criterion(outputs.float(), labels.float())
        return loss

    def convert_to_png(self, output):
        binary = output.argmin(0)
        #binary = output[0]
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
        


