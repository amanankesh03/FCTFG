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

class FCTFG_VIDEO(Dataset):
    def __init__(self, split, args, vtransform = None):
        super(FCTFG_VIDEO, self).__init__()

        ########## Audio ##############
        self.duration = 200                           # time in milliseconds
        self.aud_sr = 16000                           # in Hertz
        self.channel = 1                              # mono 1, streo 2
        self.n_mels = 80                              # num of melodies
        self.n_fft = 1000                              # num of samples in one window or window length
        self.hop_length = 200                         # num of samples to skip, stride
        self.top_db = 80
        
        ###############################

        self.args = args
        if split == 'train':
            self.Data_path = self.args.train_dataset_path
            self.num_of_samples_per_video = 20
        elif split == 'test':
            self.Data_path = self.args.test_dataset_path
            self.num_of_samples_per_video = 1
        else:
            raise NotImplementedError
        
        self.video_list = glob.glob(f'{self.Data_path }/**/*.mov', recursive=True) 
        self.video_list += glob.glob(f'{self.Data_path }/**/*.mp4', recursive=True)
        # print(self.video_list)

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
        h = video.shape[1]
        w = video.shape[2]
        
        try:
            if abs(info['video_fps'] - 25.0) > 0:    
                video = convert_video_to_tensor(video_path, self.video_fr, h, w)
                info['video_fps'] = self.video_fr
        except Exception as e:
            print(e)
            print("\n\n check the video : ", self.video_list[idx])
            return [], audio, info 
        
        return video.float(), audio, info


    def mel_sgram_from_window(self, audio, info, start_frame=5, end_frame=10):
        audio = Resample(info['audio_fps'], self.aud_sr)(audio)
        
        mel_list = []
        for i in range(start_frame, end_frame):

            aud_s = int((start_frame - 2) / info['video_fps'] * self.aud_sr)
            aud_e = int((start_frame + 3) / info['video_fps'] * self.aud_sr)
            # print(aud_e - aud_s)
            audio_window = audio[:, aud_s:aud_e-1]

            if audio_window.size(0) > 1:
                audio_window = torch.mean(audio_window, dim=0, keepdim=True)

            mel_spectrogram = AudioTransforms.MelSpectrogram(n_fft=800, n_mels=80, hop_length=200)(audio_window)
            mel_spectrogram = AudioTransforms.AmplitudeToDB(top_db=self.top_db)(mel_spectrogram)
            mel_list.append(mel_spectrogram)
        
        return torch.stack(mel_list, dim=0)

    def __getitem__(self, idx):
        samples = []
        video, audio, info = self.read_video(idx)
        
        while len(video) < 10:
            self.video_list.remove(self.video_list[idx])
            video, audio, info = self.read_video(idx)            

 
        for i in range(self.num_of_samples_per_video):
            idx = random.randint(0, len(video)-1)
            src = video[idx:idx+1]/255.0
        
            src = torch.permute(src, (0, 3, 1, 2))
            s = random.randint(2, len(video) - self.window_size - 3)   #5 frames so 2 to last-3
            
            e = s + self.window_size
            v_window = torch.permute(video[s:e]/255.0, (0,3,1,2))
            
            v_window = self.vtransform(v_window)
            src = self.vtransform(src)  
                     
            mel_sgram = self.mel_sgram_from_window(audio, info, start_frame=s, end_frame=e)
            
            if mel_sgram.shape != torch.Size([self.window_size, 1, 80, 16]):
                print(mel_sgram.shape)
                continue

            samples.append((src, v_window, mel_sgram))

        if len(samples) != self.num_of_samples_per_video:
            print(f'samples {len(samples)}', samples )

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

    loader = data.DataLoader(
        dataset,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )


    loader = sample_data(loader)
    for i in range(2):
        for sample in next(loader):
            (src, drv, mel) = sample
            print(src.shape, drv.shape, mel.shape)
            
    # for i in range(2):
    #     sample = dataset[i]
    #     # for (im, ms) in sample: 
    #         # display_img(im[0])




   