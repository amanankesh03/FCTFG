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

    def forward(self, xss, xds, xas):
        zss = self.VisualEncoder(xss)
        bs = zss.shape[0]
        print(f'visualEncoder s out : {zss.shape}')
        assert zss.shape == torch.Size([bs, self.n_styles, self.latent_dim])

        zss = self.flatten(zss)
        print(f'z_s flatten {zss.shape}')
        assert zss.shape == torch.Size([bs, self.n_styles * self.latent_dim])

        zscs = self.CanonicalEncoder(zss)
        print(f'canonicalEncoder out : {zscs.shape}')
        assert zscs.shape == torch.Size([bs, self.n_styles * self.latent_dim])

        zas = self.AudioEncoder(xas)
        print(f'audioEncoder out : {zas.shape}')
        assert zas.shape == torch.Size([bs, 1, self.latent_dim])

        zfs = []
        zcds = []
        for i, batch in enumerate(xds):
            z_s_c = zscs[i].unsqueeze(0)
            z_a = zas[i].unsqueeze(0)
            nf = batch.shape[0]

            z_d = self.VisualEncoder(batch)
            print(f'visualEncoder d out : {z_d.shape}')
            assert z_d.shape == torch.Size([nf, self.n_styles, self.latent_dim])
            
            z_a = z_a.repeat(nf, 1, 1)
            z_d_a = torch.cat([z_d, z_a], dim=1)

            print(f'z_d, z_a {z_d_a.shape}')

            z_d_a = self.flatten(z_d_a)
            z_c_d = self.MotionEncoder(z_d_a)

            print(f'motionEncoder out : {z_c_d.shape}')
            assert z_c_d.shape == torch.Size([nf, self.n_styles * self.latent_dim])

            z_c_d = z_c_d.view(nf, self.n_styles, self.latent_dim)
            z_s_c = z_s_c.view(1, self.n_styles, self.latent_dim)
            z_s_c = z_s_c.repeat(nf, 1, 1)
            z_s_d = z_s_c + z_c_d
            
            print(f'add out : {z_s_d.shape}')
            z_s_d = z_s_d.view(1, nf * self.n_styles, self.latent_dim)
            z_f = self.TemporalFusion(z_s_d)

            print(f'temporal fusion out {z_f.shape}')
            
            zfs.append(z_f.squeeze(0))
            zcds.append(z_c_d)

        zfs = torch.stack(zfs, dim=0)
        zcds = torch.stack(zcds, dim=0)

        im, latents = self.Decoder(zfs)
        print(f'Decoder out {im.shape}')

        # return [z_s_c, z_c_d] for orthogonality loss
        return im, zscs.view(bs, self.n_styles, self.latent_dim), zcds, latents
    
if __name__ == "__main__":

    from Options.BaseOptions import opts
    from Dataset import FCTFG
    from torch.utils import data
    import torchvision
    # import torchsummary
    import torchvision.transforms as transforms

    opts.size = 128 * 2
    device = "cuda:0"
    
    transform = torchvision.transforms.Compose([
        transforms.Resize((opts.size, opts.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    )

    def sample_data(loader):
        while True:
            for batch in loader:
                yield batch
  
    dataset_train = FCTFG('train', transform, opts)

    loader = data.DataLoader(
        dataset_train,
        num_workers=8,
        batch_size=opts.batch_size,
        pin_memory=True,
        drop_last=False,
    )
    
    loader = sample_data(loader)
    gen = Generator(opts).to(device)
    x_s, x_d, x_a = next(loader)
    x_s = x_s.to(device)
    x_d = x_d.to(device)
    x_a = x_a.to(device)
    print(x_s.shape)
    # torchsummary.summary(gen, x_s, x_d, x_a)
    im, z_s_c, z_c_d, latents  = gen(x_s, x_d, x_a)
    print(im.shape, z_s_c.shape, z_c_d.shape)

         