import torch
import torch.nn as nn
from monai.losses import DiceLoss

class DiceWeightedBCELoss(nn.Module):
    def __init__(self, dice_weight=1.0, bce_weight=1.0, pos_weight=None):
        """
        Args:
            dice_weight (float): weight for Dice loss component
            bce_weight (float): weight for BCE loss component
            pos_weight (Tensor): a weight of positive examples. Used for imbalanced data.
                                 Should be a 1D tensor with a single value (for binary).
        """
        super(DiceWeightedBCELoss, self).__init__()
        self.dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True) 
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        bce = self.bce_loss(inputs, targets)
        return self.dice_weight * dice + self.bce_weight * bce
