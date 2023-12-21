import math
import torch
from torch import nn
from torch.nn import functional as F
from util import *

class EncoderApp(nn.Module):
    def __init__(self, size, w_dim=512):
        super(EncoderApp, self).__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256,
            128: 128,
            256: 64,
            512: 32,
            1024: 16
        }

        self.w_dim = w_dim
        log_size = int(math.log(size, 2))

        self.convs = nn.ModuleList()
        self.convs.append(ConvLayer(3, channels[size], 1))

        in_channel = channels[size]
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            self.convs.append(ResBlock(in_channel, out_channel))
            in_channel = out_channel
        self.convs.append(EqualConv2d(in_channel, self.w_dim, 4, padding=0, bias=False))

    def forward(self, x):
        res = []
        h = x
        for conv in self.convs:
            h = conv(h)
            res.append(h)

        return res[-1].squeeze(-1).squeeze(-1), res[::-1][2:]

class Encoder(nn.Module):
    def __init__(self, size, dim=512, dim_motion=20):
        super(Encoder, self).__init__()

        # appearance netmork
        self.net_app = EncoderApp(size, dim)

        # motion network
        fc = [EqualLinear(dim, dim)]
        for i in range(3):
            fc.append(EqualLinear(dim, dim))

        fc.append(EqualLinear(dim, dim_motion))
        self.fc = nn.Sequential(*fc)

    def enc_app(self, x):

        h_source = self.net_app(x)

        return h_source

    def enc_motion(self, x):

        h, _ = self.net_app(x)
        h_motion = self.fc(h)

        return h_motion

    def forward(self, input_source, input_target, h_start=None):

        if input_target is not None:

            h_source, feats = self.net_app(input_source)
            h_target, _ = self.net_app(input_target)

            h_motion_target = self.fc(h_target)

            if h_start is not None:
                h_motion_source = self.fc(h_source)
                h_motion = [h_motion_target, h_motion_source, h_start]
            else:
                h_motion = [h_motion_target]

            return h_source, h_motion, feats
        else:
            h_source, feats = self.net_app(input_source)

            return h_source, None, feats
