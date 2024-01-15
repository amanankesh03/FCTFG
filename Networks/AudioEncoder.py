import torch
import torch.nn as nn

###Try to use Conv2d Layer From utils

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False):
        super().__init__()
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class AudioEncoder(nn.Module):
    def __init__(self, opts):
        super(AudioEncoder, self).__init__()
        self.audio_channels = opts.audio_encoder_input_channels
        self.convlist = [
            Conv2d(self.audio_channels, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            
            ######### without batchnorm2d ###########
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        ]
        
        self.convlist = [conv.to(opts.device) for conv in self.convlist]
    
    def forward(self, x):
        for i, conv in enumerate(self.convlist):
            x = conv(x)
        
        x = x.view(x.shape[0], -1, 512)
        return x
    
if __name__ == "__main__":
    from Options.BaseOptions import opts
    ae = AudioEncoder(opts)
    a = torch.randn([1, 2, 80, 17]).to(opts.device)
    print(ae(a).shape)
    print()
    
    #git token
    "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"