import torch
from torch import nn
from torch.nn import functional as F

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

class SyncNet(nn.Module):
    def __init__(self):
        super(SyncNet, self).__init__()
            
        self.face_encoder = nn.Sequential(
            Conv2d(15, 16, kernel_size=(7, 7), stride=1, padding=3),               # 256x512 -> 256x512

            Conv2d(16, 32, kernel_size=5, stride=(1, 2), padding=(2,1)),           # 256x512 -> 256x256
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=2, padding=1),                   # 256x256 -> 128x128
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),                  # 128x128 ->  64x64
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),                  # 64x64 -> 32x32
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 384, kernel_size=3, stride=2, padding=1),                  # 32x32 -> 16x16
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(384, 512, kernel_size=3, stride=2, padding=1),                  # 16x16 -> 8x8
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=5, stride=2, padding=1),                  # 8x8 -> 3x3
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),                  # 3x3 -> 1x1
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)                 # 1x1 -> 1x1

        self.audio_encoder = nn.Sequential(                                     # Same as 144x192 model
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
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
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

    def forward(self, audio_sequences, face_sequences): # audio_sequences := (B, dim, T)

        face_embedding = self.face_encoder(face_sequences)       # #torch.Size([18, 15, 256, 512])  # B, 3 * T, H//2, W
        audio_embedding = self.audio_encoder(audio_sequences)    # B, 1, 80, 16

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)

        return self.loss(audio_embedding, face_embedding)
    
    def loss(self, ae, fe):
        return  1 - F.cosine_similarity(ae, fe)

if __name__ == "__main__":
    from VideoDataset import FCTFG_VIDEO
    from Options.BaseOptions import opts

    dataset = FCTFG_VIDEO('test', opts)

    pth = '/home/amanankesh/working_dir/Extras/checkpoint_step004595000_512_fixed_audio.pth'
    ckpt = torch.load(pth)
    syncNet = SyncNet()    
    syncNet.load_state_dict(ckpt['state_dict'])

    syncNet = syncNet.to(opts.device).eval()

    mel = torch.randn([1, 1, 80, 16]).to(opts.device)
    imgs = torch.randn([1, 15, 256, 512]).to(opts.device)

    loss = syncNet(mel, imgs)

    print(loss)

