import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Implementation of focal loss based on https://arxiv.org/abs/1708.02002

    """

    def __init__(self, alpha=0.1, gamma=2, smooth=1e-5):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, input, target):

        input = input + self.smooth
        
        p_1 = input * target
        p_0 = (1. - input)[target == 1] = 0
        p_t = p_1 + p_0

        p_t = torch.clamp(p_t, self.smooth, 1. - self.smooth)

        focal_1 = - self.alpha * torch.pow((1 - p_t), self.gamma) * torch.log(p_t)
        focal_0 = - (1. - self.alpha) * torch.pow((1 - p_t), self.gamma) * torch.log(p_t)

        focal_1[target == 0] = 0
        focal_0[target == 1] = 0

        focal_1 = torch.clamp(focal_1, self.smooth, 1. - self.smooth)
        focal_1 = torch.clamp(focal_0, self.smooth, 1. - self.smooth)

        return torch.sum(focal_1 + focal_0)     
