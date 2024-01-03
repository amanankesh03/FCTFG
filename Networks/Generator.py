import torch
import torch.nn as nn
from Networks.VisualEncoder import *
from Networks.Decoder import Decoder
from Networks.AudioEncoder import AudioEncoder
from Networks.MotionEncoder import MotionEncoder
from Networks.TemporalFusion import TemporalFusion
from Networks.CanonicalEncoder import CanonicalEncoder


class Generator(nn.Module):
    def __init__(self, opts):
        super(Generator, self).__init__()

        self.VisualEncoder = VisualEncoder(opts).to(opts.device)
        self.AudioEncoder = AudioEncoder(opts).to(opts.device)
        self.CanonicalEncoder = CanonicalEncoder(opts).to(opts.device)
        self.TemporalFusion = TemporalFusion(opts).to(opts.device)
        self.MotionEncoder = MotionEncoder(opts).to(opts.device)
        self.Decoder = Decoder(opts)
        self.flatten = nn.Flatten()

    def forward(self, x_s, x_d, x_a):
        z_s = self.VisualEncoder(x_s)
        print(f'visualEncoder s out : {z_s.shape}')
        assert z_s.shape == torch.Size([z_s.shape[0], 18, 512])

        z_d = self.VisualEncoder(x_d)
        print(f'visualEncoder d out : {z_d.shape}')
        assert z_d.shape == torch.Size([z_d.shape[0], 18, 512])
        
        z_s = self.flatten(z_s)
        assert z_s.shape == torch.Size([z_s.shape[0], 18 * 512])

        z_s_c = self.CanonicalEncoder(z_s)
        print(f'canonicalEncoder out : {z_s_c.shape}')
        assert z_s_c.shape == torch.Size([z_s_c.shape[0], 18 * 512])

        z_a = self.AudioEncoder(x_a)
        print(f'audioEncoder out : {z_a.shape}')
        assert z_a.shape == torch.Size([z_a.shape[0], 18, 512])
        
        z_a_d = torch.cat([z_a, z_d], dim=1)
        z_a_d = self.flatten(z_a_d)
        z_c_d = self.MotionEncoder(z_a_d)

        print(f'motionEncoder out : {z_c_d.shape}')
        assert z_c_d.shape == torch.Size([z_c_d.shape[0], 18 *512])

        z_c_d = z_c_d.view(z_c_d.shape[0], 18, 512)
        z_s_c = z_s_c.view(z_s_c.shape[0], 18, 512)
        z_s_d = z_s_c + z_c_d
        z_f = self.TemporalFusion(z_s_d)
        
        im, latents = self.Decoder(z_f.to('cpu'))
        return im, latents
    
if __name__ == "__main__":

    from Options.BaseOptions import opts
    opts.size = 512
    device = "cuda:0"
    x_s = torch.randn([1, 3, 512, 512]).to(opts.device)
    x_d = torch.randn([1, 3, 512, 512]).to(opts.device)
    x_a = torch.randn([1, 1, 80, 321]).to(opts.device)
    # print(opts)
    gen = Generator(opts)
    # for i, p in enumerate(gen.parameters()):
    #     try:
    #         print(i)
    #         print(p.shape)
    #     except Exception as e:
    #         print(e)
    im, latents = gen(x_s, x_d, x_a)
    print(im.shape)

         