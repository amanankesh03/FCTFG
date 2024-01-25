import torch
import torch.nn as nn
from Networks.VisualEncoder import *
from Networks.StyleDecoder import Decoder
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
            
        aud_z = self.AudioEncoder(aud).squeeze(1)
         
        src_drv = torch.cat([src, drv], dim=0)
        src_drv_z = self.VisualEncoder(src_drv)

        src = src_drv_z[:1]
        src_feats = src 
        src_z = src[:, -1]

        
        drv = src_drv_z[1:]
        drv_z = drv[:, -1]
        drv_aud_z = torch.cat([drv_z, aud_z], dim=1)

        src_zc = self.CanonicalEncoder(src_z)
        
        drv_aud_zc = self.MotionEncoder(drv_aud_z)

        print(f'src_feat : {src_feats.shape}, src_z : {src_z.shape}, drv : {aud_z.shape}, drv_z : {drv_z.shape}') 
        print(f'{drv_aud_z.shape}, drv_aud_zc : {drv_aud_zc.shape}')

        src_zc = src_zc.repeat(nf, 1)
        src_drv = src_zc + drv_aud_zc 

        print(f'add out : {src_drv.shape}')

        fused_style = self.TemporalFusion(src_drv)
        print(f'fused_style shape {fused_style.shape}')

        # alpha = torch.cat([fused_style.unsqueeze(1), src_zc.unsqueeze(1)], dim=1)

        im = self.Decoder(fused_style, src_feats.repeat(nf, 1, 1))
    
        return im
    
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
  
    dataset_train = FCTFG_VIDEO('test', opts)

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
    
        src = src[0].to(device)
        drv = drv[0].to(device)
        aud = aud[0].to(device)

        # src_drv = torch.cat([src, drv], dim=0)
        # print(src_drv.shape)
        
        im = gen(src, drv, aud)
        print(im.shape)
        # break

         