from PIL import Image
import os
import torch
import torch.nn as nn
from torchvision import transforms


from src.criterion.dice_loss import dice_loss


import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class UNetRunnerClass:
    def __init__(self, config):
        self.config = config

        self.init_model()
        self.init_criterion()

    def init_model(self):
        print('Initializing model')

        model = UNet_baseline(output_prob=True) # Mathias: adjusted model here compared to deeplabrunner
        
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
        


# Below code adapted from CIL Tutorial3: https://colab.research.google.com/github/dalab/lecture_cil_public/blob/master/exercises/2021/Project_3.ipynb#scrollTo=ZFpPqQccO241
# Comment Mathias:
# Baseline-UNet. 
# ~0.9 patch_accuracy on 10 val. images after 50 epochs, with these hyperparameters: (batch, image size should be increased with larger memory on Leonhard maybe?)
# batchsize: 2
# image size: 224
# learning rate: 0.001
# optimizer: Adam
# loss: BCEDiceLoss (each w/ weight of 0.5)
# train augmentation: randomresizecrop, rnd. rotation, hflip, vflip, colorjitter, sharpness
# val augmentation: only resized to train image size

class Block(nn.Module):
    # a repeating structure composed of two convolutional layers with batch normalization and ReLU activations
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(out_ch),
                                   nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1),
                                   nn.ReLU())

    def forward(self, x):
        return self.block(x)

        
class UNet_baseline(nn.Module):
    # UNet-like architecture for single class semantic segmentation.
    def __init__(self, chs=(3,64,128,256,512,1024), output_prob=True):
        super().__init__()
        enc_chs = chs  # number of channels in the encoder
        dec_chs = chs[::-1][:-1]  # number of channels in the decoder
        self.enc_blocks = nn.ModuleList([Block(in_ch, out_ch) for in_ch, out_ch in zip(enc_chs[:-1], enc_chs[1:])])  # encoder blocks
        self.pool = nn.MaxPool2d(2)  # pooling layer (can be reused as it will not be trained)
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(in_ch, out_ch, 2, 2) for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])])  # deconvolution
        self.dec_blocks = nn.ModuleList([Block(in_ch, out_ch) for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])])  # decoder blocks
        # To output PROBABILITES (sigmoid at the end) or to output LOGITS: (no sigmoid at the end)
        if output_prob:
            self.head = nn.Sequential(nn.Conv2d(dec_chs[-1], 2, 1), nn.Sigmoid()) # 1x1 convolution for producing the 2 output classes # Sigmoid for PROBABILITES
        else:
            self.head = nn.Sequential(nn.Conv2d(dec_chs[-1], 2, 1)) # 1x1 convolution for producing the 2 output classes # LOGITS (no sigmoid at the end)

    def forward(self, x):
        # encode
        enc_features = []
        for block in self.enc_blocks[:-1]:
            x = block(x)  # pass through the block
            enc_features.append(x)  # save features for skip connections
            x = self.pool(x)  # decrease resolution, #Mathias: resolution is decreased 2**4 = 16 fold. Ex.: Img size of (,..., 400, 400) will be compressed to (...,...,25, 25)
        x = self.enc_blocks[-1](x)
        # decode
        for block, upconv, feature in zip(self.dec_blocks, self.upconvs, enc_features[::-1]):
            x = upconv(x)  # increase resolution
            x = torch.cat([x, feature], dim=1)  # concatenate skip features
            x = block(x)  # pass through the block
        return self.head(x)  # reduce to 2 channels (road, background)