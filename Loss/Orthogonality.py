import torch
import torch.nn as nn


class OrthogonalityLoss(nn.Module):
    def __init__(self):
        super(OrthogonalityLoss, self).__init__()


    def forward(self, s_latents, d_latents):
        res = s_latents * d_latents
        res = torch.sum(res, dim=2)        
        res = torch.abs(res)

        return res.mean()
    



