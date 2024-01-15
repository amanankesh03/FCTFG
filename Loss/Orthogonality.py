import torch
import torch.nn as nn


class OrthogonalityLoss(nn.Module):
    def __init__(self):
        super(OrthogonalityLoss, self).__init__()


    def forward(self, src, tgts, device):
        src = src.unsqueeze(1).repeat(1, tgts.shape[1], 1, 1)
        res = src * tgts
        res = torch.sum(res, dim = 3)        
        res = torch.abs(res)
        res = res.mean()
        return res
    



