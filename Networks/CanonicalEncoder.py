import torch
import torch.nn as nn


## 2 layer MLP

class CanonicalEncoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(CanonicalEncoder, self).__init__()

    def forward(self, x):
        return x