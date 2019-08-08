import torch.nn as nn


class CustomLoss(nn.Module):
    def __init__(self, loss_weight):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, logit, target):
        bce_loss_1 = nn.BCEWithLogitsLoss(weight=target[:, 1:2])(logit[:, :1], target[:, :1])
        bce_loss_2 = nn.BCEWithLogitsLoss()(logit[:, 1:], target[:, 2:])
        return (bce_loss_1 * self.loss_weight) + bce_loss_2
