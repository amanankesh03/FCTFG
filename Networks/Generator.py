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

        self.Decoder = Decoder(opts).to(opts.device)
        self.flatten = nn.Flatten()

    def forward(self, x_s, x_d, x_a):
        z_s = self.VisualEncoder(x_s)
        bs = x_s.shape[0]
        print(f'visualEncoder s out : {z_s.shape}')
        assert z_s.shape == torch.Size([bs, 18, 512])

        z_d = self.VisualEncoder(x_d)
        print(f'visualEncoder d out : {z_d.shape}')
        assert z_d.shape == torch.Size([bs, 18, 512])
        
        z_s = self.flatten(z_s)
        assert z_s.shape == torch.Size([bs, 18 * 512])

        z_s_c = self.CanonicalEncoder(z_s)
        print(f'canonicalEncoder out : {z_s_c.shape}')
        assert z_s_c.shape == torch.Size([bs, 18 * 512])

        z_a = self.AudioEncoder(x_a)
        print(f'audioEncoder out : {z_a.shape}')
        assert z_a.shape == torch.Size([bs, 1, 512])
        
        z_a_d = torch.cat([z_a, z_d], dim=1)
        z_a_d = self.flatten(z_a_d)
        z_c_d = self.MotionEncoder(z_a_d)

        print(f'motionEncoder out : {z_c_d.shape}')
        assert z_c_d.shape == torch.Size([bs, 18 *512])

        z_c_d = z_c_d.view(bs, 18, 512)
        z_s_c = z_s_c.view(bs, 18, 512)
        z_s_d = z_s_c + z_c_d
        
        z_f = self.TemporalFusion(z_s_d)
        
        im, latents = self.Decoder(z_f)

        # return [z_s_c, z_c_d] for orthogonality loss
        return im, latents, z_s_c, z_c_d
    
if __name__ == "__main__":

    from Options.BaseOptions import opts
    opts.size = 256
    s = opts.size
    bs = 2
    device = "cuda:0"

    x_s = torch.randn([bs, 3, s, s]).to(opts.device)
    x_d = torch.randn([bs, 3, s, s]).to(opts.device)
    x_a = torch.randn([bs, 1, 80, 16]).to(opts.device)
    # print(opts)
    # for i, p in enumerate(gen.parameters()):
    gen = Generator(opts)
    # while(1):
    #     pass

    #     try:
    #         print(i)
    #     except Exception as e:
    #         print(p.shape)
    #         print(e)
    im, latents, z_s_c, z_c_d  = gen(x_s, x_d, x_a)
    print(im.shape, z_s_c.shape, z_c_d.shape)

         