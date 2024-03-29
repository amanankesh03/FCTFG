import argparse
import os
import torch
from tqdm import tqdm
from torch.utils import data
# from Dataset import FCTFG
from VideoDataset import FCTFG_VIDEO
import torchvision
import torchvision.transforms as transforms

from Trainer import Trainer

from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


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


def write_loss(i, vgg_loss, l1_loss, g_loss, d_loss, writer):
    writer.add_scalar('vgg_loss', vgg_loss, i)
    writer.add_scalar('l1_loss', l1_loss, i)
    writer.add_scalar('gen_loss', g_loss, i)
    writer.add_scalar('dis_loss', d_loss, i)
    writer.flush()

def log_loss(i, loss_dict, writer):
    for key in loss_dict.keys():
        writer.add_scalar(key, loss_dict[key], i)
    writer.flush()

def ddp_setup(args, rank, world_size):
    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = args.port

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def log_test_samples(loader_test, rank, trainer, writer, i):
    
    test_samples = next(loader_test)

    for test_sample in test_samples:
        src, drv, mel  = test_sample
        src = src[0].to(rank, non_blocking=True)
        drv = drv[0].to(rank, non_blocking=True)
        mel = mel[0].to(rank, non_blocking=True)

        recon = trainer.sample(src, drv, mel)
        # print(f'test_imgs shape : {src.shape}, {recon.shape}')
        display_img(i, src, 'source', writer)
        display_img(i, drv, 'target', writer)
        display_img(i, recon, 'recon', writer)


        # Create a figure using matplotlib
        mel_np = mel.squeeze(1)[0].cpu().numpy()
        plt.figure(figsize=(10, 5))
        plt.imshow(mel_np, cmap='viridis', origin='lower', aspect='auto')

        # Log the mel spectrogram figure
        writer.add_figure('Mel Spectrogram', plt.gcf())

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
    
    dataset_train = FCTFG_VIDEO('train', args)
    dataset_test = FCTFG_VIDEO('test', args)

    loader = data.DataLoader(
        dataset_train,
        num_workers=4,
        batch_size=args.batch_size // world_size,
        sampler=data.distributed.DistributedSampler(dataset_train, num_replicas=world_size, rank=rank, shuffle=True),
        pin_memory=True,
        drop_last=True,
    )

    loader_test = data.DataLoader(
        dataset_test,
        num_workers=4,
        batch_size=1,
        sampler=data.distributed.DistributedSampler(dataset_test, num_replicas=world_size, rank=rank, shuffle=True),
        pin_memory=True,
        drop_last=True,
    )

    loader = sample_data(loader)
    loader_test = sample_data(loader_test)

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
        for sample in next(loader):
            src, drv, mel = sample

            src = src[0].to(rank)
            drv = drv[0].to(rank)
            mel = mel[0].to(rank)

            loss_dict, recon = trainer.gen_update(src, drv, mel)

            if i%args.dis_update_every == 0:
                gan_d_loss = trainer.dis_update(drv, recon)
                loss_dict['gan_d_loss'] = gan_d_loss.item()

            if rank == 0:
                log_loss(idx, loss_dict, writer)

            if i % args.display_freq == 0 and rank == 0:
                print(f'{i} :  {loss_dict}')

                if rank == 0:
                    log_test_samples(loader_test, rank, trainer, writer, i)
            if i % args.save_freq == 0 and rank == 0:
                trainer.save(i, checkpoint_path)
            
            i+=1
            
        # except Exception as e:
        #     print(e)
        #     if rank == 0:
        #         trainer.save(i, checkpoint_path)
        #         break

        pbar.update(1)
    
    return

def details(tensor):
    print(f'shape of tensor : {tensor.shape}')
    print(f'min, max : {torch.min(tensor), torch.max(tensor)}') 


if __name__ == "__main__":
    # training params
    from Options.BaseOptions import opts

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 1

    world_size = n_gpus
    print('==> training on %d gpus' % n_gpus)
    mp.spawn(main, args=(world_size, opts,), nprocs=world_size, join=True)
