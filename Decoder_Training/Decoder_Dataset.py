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

from torchvision.io.video import read_video, write_video

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

        ########## Video #############
        self.video_fr = 25
        self.video_size = args.size
        self.window_size = args.num_frames            # num of frames to take in a batch (1 for src, k for driving, total k + 1 )
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

        video_path = self.video_list[idx%(len(self.video_list))]
        video, audio, info = read_video(video_path)

        return video.float(), audio, info


    def __getitem__(self, idx):
        samples = []
        video, audio, info = self.read_video(idx)
        
        while len(video) < 10:
            self.video_list.remove(self.video_list[idx])
            video, audio, info = self.read_video(idx)            

        for i in range(self.num_of_samples_per_video):
            s = random.randint(0, len(video) - self.window_size -1)
            e = s + self.window_size
            v_window = torch.permute(video[s:e]/255.0, (0,3,1,2))
            
            v_window = self.vtransform(v_window)  
                     
            mel_sgram = self.mel_sgram_from_window(audio, info, start_frame=s, end_frame=e)
            
            if mel_sgram.shape != torch.Size([1, 80, 16]):
                continue

            samples.append((v_window, mel_sgram))
        if len(samples) != self.num_of_samples_per_video:
            print(f'samples {len(samples)}', )
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


    dataset = FCTFG_VIDEO('train', opts)
    print(len(dataset))






   