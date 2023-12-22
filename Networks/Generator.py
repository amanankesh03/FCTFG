import torch
import torch.nn as nn
from Networks.AudioEncoder import AudioEncoder
from Networks.VisualEncoder import *
from Networks.CanonicalEncoder import CanonicalEncoder
from Networks.TemporalFusion import TemporalFusion
from Networks.MotionEncoder import MotionEncoder
from Networks.Decoder import Decoder


class Generator(nn.Module):
    def __init__(self, opts):
        super(Generator, self).__init__()
        self.VisualEncoder = VisualEncoder(opts)

        self.AudioEncoder = AudioEncoder(opts)
        self.CanonicalEncoder = CanonicalEncoder(opts)
        self.TemporalFusion = TemporalFusion(opts)
        self.MotionEncoder = MotionEncoder(opts)
        self.Decoder = Decoder(opts)

    def forward(self, x_s, x_d, x_a):
        z_s = self.VisualEncoder(x_s)
        print(z_s.shape)

        z_s_c = self.CanonicalEncoder(z_s)
        print(z_s_c.shape)
        
        z_a = self.AudioEncoder(x_a)
        print(z_a.shape)
        

        z_d = self.VisualEncoder(x_d)
        print(z_d.shape)
        z_c_d = self.MotionEncoder(torch.cat(z_a, z_d, dim=1))

        z_s_d = z_s_c + z_c_d
        z_f = self.TemporalFusion(z_s_d)
        
        im = self.Decoder(z_f)
        return im
    
if __name__ == "__main__":

    from Options.BaseOptions import opts
    device = "cuda:0"
    x_s = torch.randn([1, 3, 512, 512])
    x_d = torch.randn([1, 3, 512, 512])
    x_a = torch.randn([1, 1, 512, 512])
    # print(opts)
    gen = Generator(opts)
    # for i, p in enumerate(gen.parameters()):
    #     try:
    #         print(i)
    #         print(p.shape)
    #     except Exception as e:
    #         print(e)

    print(gen(x_s, x_d, x_a))

         