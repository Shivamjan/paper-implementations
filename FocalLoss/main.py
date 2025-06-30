import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=1.0, reduction = 'mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        logpt = F.log_softmax(y_pred, dim=1)
        logpt = logpt.gather(1, y_true.unsqueeze(1)).squeeze(1)
        pt = logpt.exp()

        modulating_factor = (1 - pt) ** self.gamma
        loss = - self.alpha * modulating_factor * logpt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
