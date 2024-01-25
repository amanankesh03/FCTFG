import argparse
import os
import torch
from tqdm import tqdm
from torch.utils import data
# from Dataset import FCTFG
from Decoder_Dataset import DecoderDataset
import torchvision
import torchvision.transforms as transforms
from Decoder_Trainer import Trainer

from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def display_img(idx, img, name, writer):
    img = img.clamp(-1, 1)
    img = ((img - img.min()) / (img.max() - img.min())).data
    writer.add_images(tag='%s' % (name), global_step=idx, img_tensor=img)

def log_loss(i, loss_dict, writer):
    for key in loss_dict.keys():
        writer.add_scalar(key, loss_dict[key], i)
    writer.flush()

def ddp_setup(args, rank, world_size):
    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = args.port

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def log_test_samples(noise, imgs, trainer, writer, i):
    fake = trainer.sample(noise)
    real = imgs.detach()
    display_img(i, fake, 'fake', writer)
    display_img(i, real, 'real', writer)
    
    writer.flush()


def main(rank, world_size, args):
    # init distributed computing
    ddp_setup(args, rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda")

    # make logging folder
    log_path = os.path.join(args.exp_path, args.exp_name + '/log')
    checkpoint_path = os.path.join(args.exp_path, args.exp_name + '/checkpoint')

    os.makedirs(log_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    writer = SummaryWriter(log_path)

    print('==> preparing dataset')
    
    dataset_train = DecoderDataset('train', args)

    print(f'length of dataset_train, dataset_test : {len(dataset_train)}')
    
    loader = data.DataLoader(
        dataset_train,
        num_workers=4,
        batch_size=2,
        sampler=data.distributed.DistributedSampler(dataset_train, num_replicas=world_size, rank=rank, shuffle=True),
        pin_memory=True,
        drop_last=True,
    )

    loader = sample_data(loader)

    print('==> initializing trainer')
    
    # Trainer
    trainer = Trainer(args, device, rank)

    # resume
    if args.resume_ckpt is not None:
        args.start_iter = trainer.resume(args.resume_ckpt)
        print('==> resume from iteration %d' % (args.start_iter))

    print('==> training')
    pbar = tqdm(args.iter)
    i = args.start_iter
    for idx in range(args.iter):
        # try:
        samples = next(loader)
        sample = torch.cat([samples[0], samples[1]], dim=0)

        r_idx = torch.randperm(len(sample))

        # Use the permutation to shuffle the tensor along the first dimension
        imgs = sample[r_idx]
        imgs = imgs[:args.batch_size]

        imgs = imgs.to(rank)
        noise = torch.randn((imgs.shape[0], args.visual_n_styles, args.decoder_latent_dim_style))
        noise = noise.to(rank)

        # update generator
        loss_dict, img_recon = trainer.gen_update(noise)
        # details(img_recon)

        # update discriminator
        if i%args.dis_update_every == 0:
            gan_d_loss = trainer.dis_update(imgs, img_recon)
            loss_dict['gan_d_loss'] = gan_d_loss.item()

        # write to log
        if rank == 0:
            log_loss(idx, loss_dict, writer)

        # display
        if i % args.display_freq == 0 and rank == 0:
            print(f' Loss :  {loss_dict}')
            noise = torch.randn((imgs.shape[0], args.visual_n_styles, args.decoder_latent_dim_style)).to(rank, non_blocking=True)
            log_test_samples(noise, imgs, trainer, writer, i)

        if i % args.save_freq == 0 and rank == 0:
            trainer.save(i, checkpoint_path)

        i+=1
        pbar.update(1)
    return

def details(tensor):
    print(f'shape of tensor : {tensor.shape}')
    print(f'min, max : {torch.min(tensor), torch.max(tensor)}') 


if __name__ == "__main__":
    # training params
    import math
    from Options.BaseOptions import opts

    log_size = int(math.log(opts.size, 2))
    num_layers = log_size  * 2 - 2
    opts.visual_n_styles = num_layers
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 1

    world_size = n_gpus
    print('==> training on %d gpus' % n_gpus)
    mp.spawn(main, args=(world_size, opts,), nprocs=world_size, join=True)