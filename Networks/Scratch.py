import torch

recon = torch.randn([1, 3, 512, 512])
recon = recon.unsqueeze(1)
print(recon.shape)

imgs = torch.randn([1, 5, 3, 512, 512])
seq = torch.cat((imgs[:, : -1], recon), dim=1)
print(f'seq : {seq.shape}')

seq = seq.view(imgs.shape[0], -1, 512, 512)
print(seq.shape)