import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Generator, self).__init__()

    def forward(self, x):
        return x