import os
import torch
from torch import nn, optim
import torch.nn.functional as F
from Loss.VGG_loss import VGGLoss
from Networks.Generator import Generator
from Networks.Discriminator import Discriminator
from torch.nn.parallel import DistributedDataParallel as DDP
from Loss.Orthogonality import OrthogonalityLoss

def requires_grad(net, flag=True):
    for p in net.parameters():
        p.requires_grad = flag


class Trainer(nn.Module):
    def __init__(self, args, device, rank):
        super(Trainer, self).__init__()

        self.w_vgg = 1
        self.w_l1 = 1
        self.w_ortholoss = 1
        self.w_gan_g_loss = 1
        self.w_Id_loss = 1
        self.w_sync_loss = 1

        self.args = args
        self.batch_size = args.batch_size
        self.device = device
        self.gen = Generator(args).to(
            device)
        self.dis = Discriminator(args).to(device)

        # distributed computing
        self.gen = DDP(self.gen, device_ids=[rank], find_unused_parameters=True)
        self.dis = DDP(self.dis, device_ids=[rank], find_unused_parameters=True)

        g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
        d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

        self.g_optim = optim.Adam(
            self.gen.parameters(),
            lr=args.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio)
        )

        self.d_optim = optim.Adam(
            self.dis.parameters(),
            lr=args.lr * d_reg_ratio,
            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio)
        )
        self.ortholoss = OrthogonalityLoss()
        self.criterion_vgg = VGGLoss().to(rank)

    def g_nonsaturating_loss(self, fake_pred):
        return F.softplus(-fake_pred).mean()

    def d_nonsaturating_loss(self, fake_pred, real_pred):
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)

        return real_loss.mean() + fake_loss.mean()

    def gen_update(self, imgs, spectrogram):
        
        self.gen.train()
        self.gen.zero_grad()

        requires_grad(self.gen, True)
        requires_grad(self.dis, False)

        img_target_recon, z_s_c, z_c_d, _ = self.gen(imgs, spectrogram)
        img_recon_pred = self.dis(img_target_recon)

        # print(f'trainer 69 : {z_c_d.shape, z_s_c.shape}')
        
        ortholoss = self.ortholoss(z_s_c, z_c_d, self.device)
        vgg_loss = self.criterion_vgg(img_target_recon, imgs[:, -1]).mean()
        
        l1_loss = F.l1_loss(img_target_recon, imgs[:, -1])
        gan_g_loss = self.g_nonsaturating_loss(img_recon_pred)

        g_loss = self.w_vgg *vgg_loss 
        g_loss += self.w_l1 * l1_loss 
        g_loss += self.w_gan_g_loss * gan_g_loss 
        g_loss += self.w_ortholoss * ortholoss

        loss_dict = {

            "vgg_loss" : vgg_loss.item(),
            "l1_loss" : l1_loss.item(),
            "gan_g_loss" : gan_g_loss.item(),
            "ortholoss" : ortholoss.item(),
        }

        g_loss.backward()
        self.g_optim.step()

        return loss_dict, img_target_recon

    def dis_update(self, img_real, img_recon):
        self.dis.zero_grad()

        requires_grad(self.gen, False)
        requires_grad(self.dis, True)

        real_img_pred = self.dis(img_real)
        recon_img_pred = self.dis(img_recon.detach())

        d_loss = self.d_nonsaturating_loss(recon_img_pred, real_img_pred)
        d_loss.backward()
        self.d_optim.step()

        return d_loss

    def sample(self, imgs, spectrogram):
        with torch.no_grad():
            self.gen.eval()
            img_recon, _, _, _ = self.gen(imgs, spectrogram)
        return img_recon

    def resume(self, resume_ckpt):
        print("load model:", resume_ckpt)
        ckpt = torch.load(resume_ckpt)
        ckpt_name = os.path.basename(resume_ckpt)
        start_iter = int(os.path.splitext(ckpt_name)[0])

        self.gen.module.load_state_dict(ckpt["gen"])
        self.dis.module.load_state_dict(ckpt["dis"])
        self.g_optim.load_state_dict(ckpt["g_optim"])
        self.d_optim.load_state_dict(ckpt["d_optim"])

        return start_iter

    def save(self, idx, checkpoint_path):
        torch.save(
            {
                "gen": self.gen.module.state_dict(),
                "dis": self.dis.module.state_dict(),
                "g_optim": self.g_optim.state_dict(),
                "d_optim": self.d_optim.state_dict(),
                "args": self.args
            },
            f"{checkpoint_path}/{str(idx).zfill(6)}.pt"
        )
