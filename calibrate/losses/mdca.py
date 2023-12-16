import torch
import torch.nn as nn
import torch.nn.functional as F
from .label_smoothing import LabelSmoothingCrossEntropy


class MDCA(nn.Module):
    def __init__(self):
        super(MDCA, self).__init__()
        self.ls = LabelSmoothingCrossEntropy()

    @property
    def names(self):
        return "loss", "loss_ce",  "loss_mdca"

    def forward(self, output, target):
        output = torch.softmax(output, dim=1)
        # [batch, classes]
        loss_mdca = torch.tensor(0.0).cuda()
        batch, classes = output.shape
        for c in range(classes):
            avg_count = (target == c).float().mean()
            avg_conf = torch.mean(output[:,c])
            loss_mdca += torch.abs(avg_conf - avg_count)
        denom = classes
        loss_mdca /= denom
        loss_mdca *= 1.0
        loss_ce = self.ls(output, target)
        loss = loss_ce + loss_mdca
        
        return loss, loss_ce, loss_mdca
