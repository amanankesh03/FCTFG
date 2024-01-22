import torch
from torch.utils import data
from Networks.Generator import Generator
from VideoDataset import FCTFG_VIDEO
from Options.BaseOptions import opts
import matplotlib.pyplot as plt

pth = ""
def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def display_direct(img):

    plt.imshow(img,  aspect='auto', origin='upper')
    plt.show()
    
def display_img(img):
    img = torch.permute(img, (1, 2, 0))
    img = img.clamp(-1, 1)
    img = ((img - img.min()) / (img.max() - img.min())).data
    plt.imshow(img,  aspect='auto', origin='upper')
    plt.show()


# ckpt = torch.load(pth)
# gen.module.load_state_dict(ckpt["gen"])

dataset = FCTFG_VIDEO('test', opts).to(opts.device)
print(len(dataset))

gen = Generator(opts)
samples = dataset[0]
for sample in samples:
    imgs, mel = sample
    imgs = imgs.to(opts.device)
    mel = mel.to(opts.device)
    print(imgs.shape, mel.shape)

    gen(imgs, mel)

