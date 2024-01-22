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
        self.n_styles = int(math.log(opts.size, 2) * 2 - 2)
        
        self.latent_dim = opts.latent_dim
        self.VisualEncoder = VisualEncoder(opts)
        self.AudioEncoder = AudioEncoder(opts)
        self.CanonicalEncoder = CanonicalEncoder(opts)
        self.TemporalFusion = TemporalFusion(opts)
        self.MotionEncoder = MotionEncoder(opts)

        self.Decoder = Decoder(opts)
        self.flatten = nn.Flatten()

    def forward(self, xss, xas):

        bs = xas.shape[0]
        zas = self.AudioEncoder(xas)
        # self.details(zas)
        #print(f'audioEncoder out : {zas.shape}')
        #assert zas.shape == torch.Size([bs, 1, self.latent_dim])

        zfs = []
        zcds = []
        zscs = []
        for i, batch in enumerate(xss):
            nf = batch.shape[0]
            xs = xss[i]
            za = zas[i]

            zs = self.VisualEncoder(xs)
            #print(f'visualEncoder s out : {zs.shape}')
            #assert zs.shape == torch.Size([nf, self.n_styles, self.latent_dim])
            # source latents
            # self.details(zs)e
            z_s = self.flatten(zs[:1])
            #print(f'z_s flatten {z_s.shape}')
            #assert z_s.shape == torch.Size([1, self.n_styles * self.latent_dim])
           
            z_s_c = self.CanonicalEncoder(z_s)
            
            #print(f'canonicalEncoder out : {z_s_c.shape}')
            #assert z_s_c.shape == torch.Size([1, self.n_styles * self.latent_dim])
            
            zscs.append(z_s_c)
            ################
            
            # driving latents
            z_d = self.flatten(zs[1:]) 
            #print(f'z_d flatten {z_d.shape}')
            #assert z_d.shape == torch.Size([(nf - 1), self.n_stylesgc * self.latent_dim])
            
            z_a = za.repeat(nf-1, 1)
            #print(f'z_a : {z_a.shape}')            
            z_d_a = torch.cat([z_d, z_a], dim=1)

            #print(f'z_d_a {z_d_a.shape}')

            z_d_a = self.flatten(z_d_a)
            z_c_d = self.MotionEncoder(z_d_a)

            #print(f'motionEncoder out : {z_c_d.shape}')
            #assert z_c_d.shape == torch.Size([nf - 1, self.n_styles * self.latent_dim])

            z_c_d = z_c_d.view(nf - 1, self.n_styles, self.latent_dim)
            z_s_c = z_s_c.view(1, self.n_styles, self.latent_dim)
            z_s_c = z_s_c.repeat(nf - 1, 1, 1)
            z_s_d = z_s_c + z_c_d
            
            #print(f'add out : {z_s_d.shape}')
            z_s_d = z_s_d.view(1, (nf - 1) * self.n_styles, self.latent_dim)

            #print(f'z_s_d {z_s_d.shape}')
            z_f = self.TemporalFusion(z_s_d)

            #print(f'temporal fusion out {z_f.shape}')
            
            zfs.append(z_f.squeeze(0))
            zcds.append(z_c_d)

        zfs = torch.stack(zfs, dim=0)
        zcds = torch.stack(zcds, dim=0)
        zscs = torch.stack(zscs, dim=0)

        im, latents = self.Decoder(zfs)
        # #print(f'Decoder out {im.shape}')
        # self.details(im)
        # return [z_s_c, z_c_d] for orthogonality loss
        return im, zscs.view(bs, self.n_styles, self.latent_dim), zcds, latents
    
    def details(self, tensor):
        print(f'shape of tensor : {tensor.shape}')
        print(f'min, max : {torch.min(tensor), torch.max(tensor)}') 

    
if __name__ == "__main__":

    from Options.BaseOptions import opts
    from Dataset import FCTFG
    from VideoDataset import FCTFG_VIDEO
    from torch.utils import data
    import torchvision
    # import torchsummary
    import torchvision.transforms as transforms

    opts.size = 128 * 2
    device = "cuda:0"
   

    def sample_data(loader):
        while True:
            for batch in loader:
                yield batch
  
    dataset_train = FCTFG_VIDEO('train', opts)

    loader = data.DataLoader(
        dataset_train,
        num_workers=8,
        batch_size=opts.batch_size,
        pin_memory=True,
        drop_last=False,
    )
    
    loader = sample_data(loader)
    gen = Generator(opts).to(device)
    print('here')
    sample = next(loader)
    for (x_s, x_a) in sample:
    
        x_s = x_s.to(device)
        # x_d = x_d.to(device)
        x_a = x_a.to(device)
        # #print(x_s.shape)
        # torchsummary.summary(gen, x_s, x_d, x_a)
        im, z_s_c, z_c_d, latents  = gen(x_s, x_a)
        print(im.shape, z_s_c.shape, z_c_d.shape)
        break

         