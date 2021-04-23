from typing import Optional

from numpy.lib.function_base import interp

import torch

def dice_loss():

    def forward(input, target):
        
        # Compute area of intersection
        dims = (2,3)
        intersection = torch.sum(input * target, dims)

        # Compute area of union
        union = torch.sum(input + target, dims)

        # Compute dice coefficient
        dims = (1)
        eps = 0.000001
        dice_score = (intersection + eps)  / union
        dice_score = torch.mean(dice_score)

        return 1. - dice_score
    
    return forward