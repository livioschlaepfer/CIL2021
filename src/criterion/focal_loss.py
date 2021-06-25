import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Implementation of focal loss based on https://arxiv.org/abs/1708.02002

    """

    def __init__(self, alpha=40, gamma=2, smooth=1e-5):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, input, target):
        p_1 = input[target == 1]
        p_0 = (1. - input)[target == 0]
        p_t = p_0 + p_1
        focal = - self.alpha * (1 - p_t) ** self.gamma * torch.log(p_t)

        return torch.mean(focal)     
