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

class TrivialRunnerClass:
    def __init__(self, config):
        self.config = config

        self.init_model()
        self.init_criterion()

    def init_model(self):
        print('Initializing model')

        model = Trivial_baseline(output_prob=True) # Mathias: adjusted model here compared to deeplabrunner
        
        # Print the model we just instantiated
        print(model)

        self.model = model

    def init_criterion(self):
        # Setup the loss
        # self.criterion = torch.nn.MSELoss(reduction='mean')
        # self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.BCELoss()
        # self.criterion = nn.NLLLoss()
        # self.criterion = DiceLoss()

        def forward(input, target):

            bce = nn.BCELoss()
            dice = dice_loss()

            loss1 = 0.0 * bce(input, target) + 0.8 * dice(input, target)

            input_16 = self.image_to_patched(input, 16)
            target_16 = self.mask_to_patched(target, 16)

            loss3 = 0.0 * bce(input_16, target_16) + 0.8 * dice(input_16, target_16)

            # print("loss1", loss1,"loss3", loss3, )

            return loss1 + 0.2 * loss3 

        self.criterion = forward
  
    def forward(self, inputs):
        outputs = self.model(inputs) # Mathias: removed ["out"] compared to deeplab runner -> not needed
        # outputs = torch.sigmoid(outputs) # Mathias: already outputting sigmoids if output_prob = True (by default)

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
        



        
class Trivial_baseline(nn.Module):
    """ Trivial Baseline that predicts all background/ zeros in both channels.
    By default outputs probabilites. If Logits are needed, change output_prob to False"""
    def __init__(self, output_prob=True):
        super().__init__()
        # Dummy parameters needed, otherwise optimizer gives error. But can directly predict without training obviously...
        self.dummy_parameters = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1))
        self.output_prob = output_prob

    def forward(self, x):
        # Predict background for every pixel

        if self.output_prob:
            # To output probabilities:
            y_hat = torch.zeros_like(x)[:, 0:2] #select C dimension via "two-element slice" to keep dim "2" in [B, 2, H, W]
        else:
            # To output logits:
            # has to be very large negative number (-10 enough actually)
            y_hat = torch.ones_like(x)[:, 0:2] * -1e1
            
        # Need dummy grad, otherwise "training" gives error. In inference, with torch.no_grad() deactivates this anyway.
        y_hat.requires_grad = True

        return y_hat 

