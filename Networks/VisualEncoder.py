import torch
import math
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module
from VisualEncoderHelper import get_blocks, Flatten, bottleneck_IR, bottleneck_IR_SE
from utils import EqualLinear


## StyleGAN Inversion Network

class VisualEncoder(nn.Module):
    def __init__(self, opts) -> None:
        super(VisualEncoder, self).__init__()
        self.opts = opts
        self.VisualEncoder = self.set_encoder()

    def set_encoder(self):
        if self.opts.visual_encoder_type == 'GradualStyleEncoder':
            encoder = GradualStyleEncoder(self.opts.visual_encoder_layers, 'ir_se', self.opts)
        elif self.opts.visual_encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
            encoder = BackboneEncoderUsingLastLayerIntoW(self.opts.visual_encoder_layers, 'ir_se', self.opts)
        elif self.opts.visual_encoder_type == 'BackboneEncoderUsingLastLayerIntoWPlus':
            encoder = BackboneEncoderUsingLastLayerIntoWPlus(self.opts.visual_encoder_layers, 'ir_se', self.opts)    
        else:
            raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
        return encoder

    def forward(self, x):
        return self.VisualEncoder(x)


class GradualStyleBlock(Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()

        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(math.ceil(np.log2(spatial)))
        # print(f'num_pools : {num_pools}')
        modules = []
        modules += [Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        # print(f'in shape : {x.shape}')
        x = self.convs(x)
        # print(f'out shape : {x.shape}')
        # assert x.shape[1] == 1
        x = x.view(x.shape[0], -1, self.out_c)
        x = self.linear(x)
        # print(f'final shape : {x.shape}')
        return x


class GradualStyleEncoder(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(GradualStyleEncoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50, 100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'

        if mode == 'ir':
            unit_module = bottleneck_IR

        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE

        self.input_layer = Sequential(Conv2d(opts.visual_input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        
        blocks = get_blocks(num_layers)
        modules = []
        
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = opts.visual_n_styles
        self.coarse_ind = 3
        self.middle_ind = 7
        
        self.spatial_mul = int(opts.size/256)
        # style blocks
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16 * self.spatial_mul)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32 * self.spatial_mul)
            else:
                style = GradualStyleBlock(512, 512, 64 * self.spatial_mul)
            self.styles.append(style)

        # Latent Layers
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        x = self.input_layer(x)

        # print(f'x : {x.shape}')

        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x
        # print(c1.shape, c2.shape, c3.shape)
        # print(afdfadf)

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))
        
        p2 = self._upsample_add(c3, self.latlayer1(c2))
 

        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))
            # print(f'latents mid : {latents[0].shape, len(latents)}')
           
        p1 = self._upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))
           

        out = torch.stack(latents, dim=1)
        return out


class BackboneEncoderUsingLastLayerIntoW(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderUsingLastLayerIntoW, self).__init__()
        print('Using BackboneEncoderUsingLastLayerIntoW')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(opts.visual_input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = EqualLinear(512, 512, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_pool(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        return x


class BackboneEncoderUsingLastLayerIntoWPlus(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderUsingLastLayerIntoWPlus, self).__init__()
        print('Using BackboneEncoderUsingLastLayerIntoWPlus')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.visual_n_styles = opts.visual_n_styles
        self.input_layer = Sequential(Conv2d(opts.visual_input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_layer_2 = Sequential(BatchNorm2d(512),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(512 * 7 * 7, 512))
        self.linear = EqualLinear(512, 512 * self.visual_n_styles, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer_2(x)
        x = self.linear(x)
        x = x.view(-1, self.visual_n_styles, 512)
        return x
    

if __name__ == '__main__':
    from Options.BaseOptions import opts
    device = 'cuda:0'
    opts.size *= 2
    ve = VisualEncoder(opts).to(device)
    s = opts.size 
    xs = torch.randn([1, 3, s, s]).to(device)
    # try :
    ls = ve(xs)
    print(ls.shape)
        # while(1):
        #     print() 
    # except:
    #     while(1):
    #         print()