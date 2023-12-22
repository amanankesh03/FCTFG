import torch
import torch.nn as nn
from Networks.AudioEncoder import AudioEncoder
from Networks.VisualEncoder import *
from Networks.CanonicalEncoder import CanonicalEncoder
from Networks.TemporalFusion import TemporalFusion
from Networks.MotionEncoder import MotionEncoder
#num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'

class Generator(nn.Module):
    def __init__(self, opts) -> None:
        super(Generator, self).__init__()
        self.AudioEncoder = AudioEncoder(opts)
        self.VisualEncoder = VisualEncoder(opts)
        self.CanonicalEncoder = CanonicalEncoder(opts)
        self.TemporalFusion = TemporalFusion(opts)
        self.MotionEncoder = MotionEncoder(opts)

    def forward(self, x_s, x_d, x_a):
        z_s = self.VisualEncoder(x_s)
        z_d = self.VisualEncoder(x_d)
        z_a = self.AudioEncoder(x_a)

        z_s_c = self.CanonicalEncoder(z_s)
        z_c_d = self.MotionEncoder(torch.cat(z_a, z_d, dim=1))

        z_s_d = z_s_c + z_c_d
        z_f = self.TemporalFusion(z_s_d)
        #############decoder
        return z_f