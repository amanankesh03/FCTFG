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
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

ImageFile.LOAD_TRUNCATED_IMAGES = True

class DecoderDataset(Dataset):
    def __init__(self, split, args, vtransform = None):
        super(DecoderDataset, self).__init__()

        self.args = args
        if split == 'train':
            self.Data_path = self.args.train_dataset_path
            self.num_of_samples_per_video = 20
        elif split == 'test':
            self.Data_path = self.args.test_dataset_path
            self.num_of_samples_per_video = 1
        else:
            raise NotImplementedError

        self.video_list = []
        for path in self.Data_path:
            self.video_list += glob.glob(f'{self.Data_path }/**/*.mp4', recursive=True)
            self.video_list += glob.glob(f'{self.Data_path }/**/*.mov', recursive=True) 

        ########## Video #############
        self.video_fr = 25
        self.video_size = args.size
        self.window_size = args.num_frames  # num of frames to take in a batch (1 for src, k for driving, total k + 1 )
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

        video, audio, info = self.read_video(idx)
        
        while len(video) <= 20:
            self.video_list.remove(self.video_list[idx])
            video, audio, info = self.read_video(idx)            

        random_idx = random.sample(range(len(video)), self.num_of_samples_per_video)
        imgs = torch.permute(video[random_idx]/255.0, (0,3,1,2))
        
        imgs = self.vtransform(imgs)  
        if len(imgs) != self.num_of_samples_per_video:
            print(f'samples {len(imgs)}')
        return imgs
    
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

    from Options.BaseOptions import opts
    dataset = DecoderDataset('train', opts)

    print(dataset[0].shape)
    dataset.display_img(dataset[10][0])




   