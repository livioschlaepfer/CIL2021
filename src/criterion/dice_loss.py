import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Implementation of dice loss
    """


    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        # Compute area of intersection
        dims = (1,2,3)
        intersection = torch.sum(input * target, dims)

        # Compute area of union
        union = torch.sum(input + target, dims)

        # Compute dice coefficient
        eps = 0.000001
        dice_score = 2. * intersection / (union + eps)

        return torch.mean(1. - dice_score)