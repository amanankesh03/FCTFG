import torch
import torch.nn as nn

# 3 Layer MLP

class MotionEncoder(nn.Module):
    def __init__(self, opts) -> None:
        super(MotionEncoder, self).__init__()

    def forward(self, x):
        return x