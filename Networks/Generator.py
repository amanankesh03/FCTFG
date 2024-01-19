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

    def forward(self, src, drv, aud):

        nf = drv.shape[0]
        
        ############ audio ###################
        aud_z = self.AudioEncoder(aud)
        # print(f'aud_z shape : {aud_z.shape}')
        # assert aud_z.shape == torch.Size([nf, 1, self.latent_dim])
        ############ source #################
    
        src_z = self.VisualEncoder(src)
        # print(f'src_z shape : {src_z.shape}')
        # assert src_z.shape == torch.Size([1, self.n_styles, self.latent_dim])
        
        src_zf = self.flatten(src_z)
        # print(f'z_s flatten {src_zf.shape}')
        src_zfc = self.CanonicalEncoder(src_zf)
        
        # print(f'canonicalEncoder out : {src_zfc.shape}')
        # assert src_zfc.shape == torch.Size([1, self.n_styles * self.latent_dim])
        
        src_zc = src_zfc.view(1, self.n_styles, self.latent_dim)

        ################### driving ##########################
        drv_z = self.VisualEncoder(drv)
        # print(f'visualEncoder drv out : {drv_z.shape}')
        # assert drv_z.shape == torch.Size([nf, self.n_styles, self.latent_dim])
                            
        drv_aud_z = torch.cat([drv_z, aud_z], dim=1)

        # print(f'drv_aud_z {drv_aud_z.shape}')
        # assert drv_aud_z.shape == torch.Size([nf, self.n_styles + 1, self.latent_dim])

        drv_aud_zf = self.flatten(drv_aud_z)
        # print(f'drv_aud_z {drv_aud_z.shape}')
        # assert drv_aud_zf.shape == torch.Size([nf, (self.n_styles + 1) * self.latent_dim])

        drv_aud_zfc = self.MotionEncoder(drv_aud_zf)

        # print(f'motionEncoder out : {drv_aud_zfc.shape}')
        # assert drv_aud_zfc.shape == torch.Size([nf, self.n_styles * self.latent_dim])

        drv_aud_zc = drv_aud_zfc.view(nf, self.n_styles, self.latent_dim)

        src_zc = src_zc.repeat(nf, 1, 1)
        src_drv = src_zc + drv_aud_zc 
        # print(f'add out : {src_drv.shape}')

        fused_style = self.TemporalFusion(src_drv)
        # print(f'fused_style shape {fused_style.shape}')

        im, latents = self.Decoder(fused_style)
    
        return im, src_zc, drv_aud_zc, latents
    
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

    # opts.size = 128 * 2
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

    sample = next(loader)
    for (src, drv, aud) in sample:
    
        src = src.to(device)
        drv = drv.to(device)
        aud = aud.to(device)
        
        im, src_zc, drv_zc, latents  = gen(src, drv, aud)
        print(im.shape, drv_zc.shape, src_zc.shape)
        break

         