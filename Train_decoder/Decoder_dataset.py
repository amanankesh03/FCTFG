import os
import glob
import random
from PIL import Image, ImageFile
import torch 
from torch.utils.data import Dataset
import subprocess

from Utils.AudioUtils import *
from Utils.VideoUtils import *
import torchaudio.transforms as AudioTransforms
from torchaudio.transforms import Resample 

import torchvision.transforms as VisionTransforms
from torchvision.transforms import functional as F

from torchvision.io import read_video, write_video

import warnings
warnings.filterwarnings("ignore")

ImageFile.LOAD_TRUNCATED_IMAGES = True

class DecoderDataset(Dataset):
    def __init__(self, split, args, vtransform = None):
        super(DecoderDataset, self).__init__()

        self.args = args
        if split == 'train':
            self.Data_path = self.args.train_dataset_path
            self.num_of_samples_per_video = 2
        elif split == 'test':
            self.Data_path = self.args.test_dataset_path
            self.num_of_samples_per_video = 1
        else:
            raise NotImplementedError

        self.video_list = glob.glob(f'{self.Data_path }/**/*.mov', recursive=True) 
        self.video_list += glob.glob(f'{self.Data_path }/**/*.mp4', recursive=True)
    
        self.video_fr = 25
        self.video_size = args.size

        self.vtransform = vtransform

        if self.vtransform is None:
            self.vtransform = VisionTransforms.Compose([
                VisionTransforms.Resize((self.args.size, self.args.size)),
                VisionTransforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
            )

        ###############################

    def __len__(self):
        return len(self.video_list)

    def read_video(self, idx):

        video_path = self.video_list[idx]
        print(video_path)
        try:
            video, audio, info = read_video(video_path)
            print(video.shape)
        except Exception as e:
            print(e)
        return video, audio, info


    def __getitem__(self, idx):
        samples = []
        video, audio, info = self.read_video(idx)
        print(video.shape)

        for i in range(self.num_of_samples_per_video):
            s = random.randint(0, len(video)-1)
            frame = torch.permute(video[s]/255.0, (2,0,1))
            frame = self.vtransform(frame)  
            # print(s, frame.shape)
            samples.append(frame)
            
        if len(samples) != self.num_of_samples_per_video:
            print(f'samples {len(samples)}')
        print(len(samples))
        return samples
    

    def display_direct(self, img):
        plt.imshow(img,  aspect='auto', origin='upper')
        plt.show()
    

    def display_img(self, img):
        img = torch.permute(img, (1, 2, 0))
        img = img.clamp(-1, 1)
        img = ((img - img.min()) / (img.max() - img.min())).data
        plt.imshow(img,  aspect='auto', origin='upper')
        plt.show()

if __name__ == "__main__":
        
    import numpy as np
    from Options.BaseOptions import opts
    import matplotlib.pyplot as plt
    from torch.utils import data
    import torchvision
    import torchvision.transforms as transforms

    opts.num_of_samples_per_video = 1
    
    def sample_data(loader):
        while True:
            for batch in loader:
                yield batch


    dataset = DecoderDataset('train', opts)
    # print(len(dataset))
    # # print(dataset[0])

    loader = data.DataLoader(
        dataset,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    loader = sample_data(loader)
    for sample in next(loader):
        print('here1')
        for img in sample:
            print(img.shape)
            print('here')
      