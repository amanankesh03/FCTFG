import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Decoder, self).__init__()

    def forward(self, x):
        return x