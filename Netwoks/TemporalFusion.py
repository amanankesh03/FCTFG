import torch
import torch.nn as nn

class TemporalFusion(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(TemporalFusion, self).__init__()

    def forward(self, x):
        return x