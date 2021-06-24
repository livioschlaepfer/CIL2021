from torch import nn
import torch


        
class Trivial_baseline(nn.Module):
    """ Trivial Baseline that predicts all background/ zeros.
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
            y_hat = torch.zeros_like(x)[:, 0:1] #select C dimension via "one-element slice" to keep dim "1" in [B, 1, H, W]
        else:
            # To output logits:
            # has to be very large negative number (-10 enough actually)
            y_hat = torch.ones_like(x)[:, 0:1] * -1e1
            
        # Need dummy grad, otherwise "training" gives error. In inference, with torch.no_grad() deactivates this anyway.
        y_hat.requires_grad = True

        return y_hat 