import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Discriminator, self).__init__()

    def forward(self, x):
        return x