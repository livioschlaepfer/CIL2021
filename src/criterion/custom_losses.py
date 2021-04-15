import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6, squared=False):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
        self.squared = squared

    def forward(self, output, target):
        assert(output.shape == target.shape)

        numerator = 2*torch.sum(torch.mul(output, target))
        if squared:
            denominator = torch.sum(torch.square(output) + torch.square(target))
        else:
            denominator = torch.sum(output + target)

        return 1- ((numerator + epsilon)/(denominator + epsilon))

class weighted_Dice_BCE(nn.Module):
    def __init__(self, weight_Dice=0.5, weight_BCE=0.5, epsilon=1e-6, squared=False, reduction="mean"):
        super(weighted_Dice_BCE, self).__init__()
        self.Dice = DiceLoss(epsilon=epsilon, squared=squared)
        self.BCE = nn.BCELoss(reduction=reduction)
        self.weight_dice = weight_Dice
        self.weight_BCE = weight_BCE

    def forward(self, output, target):
        return self.weight_BCE*self.BCE(output, target) + self.weight_dice*self.Dice(output, target)


