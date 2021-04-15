from PIL import Image
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torchvision import transforms
from src.models.bm_parts import composite_bay_conv, composite_bay_trans_conv, Bayesian_Unet
from src.criterion.custom_losses import DiceLoss, weighted_Dice_BCE

def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl

class B_Unet:
    def __init__(self, config):
        self.config = config
        self.init_model()
        self.init_criterion()

    def init_model(self, verbose=True):
        print('Initializing model')

        model = Bayesian_Unet(in_channels=self.config.bm.in_channels, batch_size=self.config.batch_size)
        
        # Print the model we just instantiated
        if verbose:
            print(model)

        self.model = model

    def init_criterion(self):
        if self.config.criterion == "BCE":
            self.criterion = nn.BCELoss(reduction=self.config.criterion.BCE_reduction)
        if self.config.criterion == "Dice":
            self.criterion = DiceLoss()
        if self.config.criterion == "BCE_with_logits":
            self.criterion = nn.BCEWithLogitsLoss(reduction=self.config.criterion.BCE_reduction, pos_weight=self.config.criterion.BCE_reduction)
        if self.config.criterion == "weighted_Dice_BCE":
            self.criterion = weighted_Dice_BCE()
    
    def forward(self, inputs):
        outputs = self.model(inputs)
        return torch.squeeze(outputs)

    def loss(self, outputs, labels):
        #print(self.criterion(outputs.float(), labels.float()))
        #print(self.config.bm.kl_weight*self.model.kl_loss())
        loss = self.criterion(outputs.float(), labels.float()) + self.config.bm.kl_weight*self.model.kl_loss()
        return loss

    def convert_to_png(self, output):

        output_predictions = output.argmax(0)

        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = torch.as_tensor([i for i in range(2)])[:, None] * palette
        colors = (colors % 255).numpy().astype("uint8")

        # Plot the semantic segmentation predictions per class
        r = Image.fromarray(output_predictions.byte().cpu().numpy())
        r.putpalette(colors)

        return r