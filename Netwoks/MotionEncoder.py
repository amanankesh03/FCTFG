import torch
import torch.nn as nn

class MotionEncoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(MotionEncoder, self).__init__()

    def forward(x):
        return x