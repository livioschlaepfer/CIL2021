from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_loss():

    def forward(input, target):
        
        # Compute area of intersection
        dims = (2,3)
        intersection = torch.sum(input * target, dims)
        print("intersection shape", intersection.shape)

        # Compute area of union
        union = torch.numel(input)

        # Compute dice coefficient
        dims = (1)
        dice_score = intersection * 2. / union

        print("shape dice", dice_score.shape)

        print("dice_score", torch.mean(dice_score))

        print("loss:", torch.mean(1. - dice_score))

        return torch.mean(1. - dice_score)
    
    return forward