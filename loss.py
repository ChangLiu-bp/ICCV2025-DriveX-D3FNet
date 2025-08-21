import torch
import torch.nn as nn

class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()
        
    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # 可根据需要修改
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(dim=(1, 2, 3))
            j = y_pred.sum(dim=(1, 2, 3))
            intersection = (y_true * y_pred).sum(dim=(1, 2, 3))
        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss
        
    def forward(self, y_true, y_pred):
        bce_loss = self.bce_loss(y_pred, y_true)
        dice_loss = self.soft_dice_loss(y_true, y_pred)
        return bce_loss + dice_loss
