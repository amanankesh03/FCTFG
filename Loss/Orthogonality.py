import torch
import torch.nn as nn


class OrthogonalityLoss(nn.Module):
    def __init__(self):
        super(OrthogonalityLoss, self).__init__()


    def forward(self, src, tgts, device):
        loss = torch.zeros([1]).to(device)
        for i, batch in enumerate(tgts):
            s = src[i]
            # print(src.shape, tgts.shape)
            
            for tgt in batch:
                for (sv, tv) in zip(s, tgt):
                    # print(sv.shape, tv.shape)
                    loss += torch.abs(torch.dot(sv, tv))
        return loss