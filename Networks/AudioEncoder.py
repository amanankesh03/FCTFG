import torch
import torch.nn as nn

class AudioEncoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(AudioEncoder, self).__init__()

    def forward(self, x):
        return x