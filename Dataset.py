import os
import glob
import random
from PIL import Image, ImageFile
import torch 
from torch.utils.data import Dataset
from Utils.AudioUtils import *
from torchvision import transforms as vtramsforms
from torchaudio import transforms as atransfroms

ImageFile.LOAD_TRUNCATED_IMAGES = True

class FCTFG(Dataset):
    def __init__(self, split, vtransform, opts):
        super(FCTFG, self).__init__()

        if split == 'train':
            self.Data_path = opts.train_dataset_path
        elif split == 'test':
            self.Data_path = opts.test_dataset_path
        else:
            raise NotImplementedError
        
        self.video_path = os.path.join(self.Data_path, 'video')
        self.video_list = os.listdir(self.video_path)
        self.num_frames = opts.num_frames                     # num of frames to take in a batch (1 for src, k for driving, total k + 1 )
        self.vtransform = vtransform

        ######## Audio
        self.audio_path = os.path.join(self.Data_path, 'audio')
        self.audio_list = os.listdir(self.audio_path)
        
        self.duration = 200                           # time in milliseconds
        self.sr = 16000                               # in Hertz
        self.channel = 1                              # mono 1, streo 2
        self.n_mels = 80                              # num of melodies
        self.n_fft = 800                              # num of samples in one window or window length
        self.hop_len = 200                            # num of samples to skip, stride
    

    def __len__(self):
        return len(self.video_list)
    
    def get_spectrogram(self, idx):
        
        audio_path = os.path.join(self.audio_path, self.audio_list[idx])
        aud = AudioUtil.open(audio_path)
        
        reaud = AudioUtil.resample(aud, self.sr)
        dur_aud = AudioUtil.pad_trunc(reaud, self.duration)
        sgram = AudioUtil.spectro_gram(dur_aud, n_mels=self.n_mels, n_fft=self.n_fft, hop_len=self.hop_len)
        return sgram
    
    def get_frames(self, idx):

        video_path = os.path.join(self.video_path, self.video_list[idx])
        frames_paths = sorted(glob.glob(video_path + '/*.png', recursive=True)) 
        nframes = len(frames_paths)

        idx = torch.randint(0, nframes - self.num_frames, (1,))
        selected_frames_paths = frames_paths[idx : idx + self.num_frames]
        
        img_source = Image.open(selected_frames_paths[0]).convert('RGB')
        if self.vtransform is not None:
            img_source = self.vtransform(img_source)
        
        img_targets = []
        for i in range(1, self.num_frames):
            img_target = Image.open(selected_frames_paths[i]).convert('RGB')
            
            if self.vtransform is not None:
                img_target = self.vtransform(img_target)

            img_targets.append(img_target)
        
        return img_source, torch.stack(img_targets, dim=0)

    def __getitem__(self, idx):
        f_src, f_tgts = self.get_frames(idx)
        sgram = self.get_spectrogram(idx)    
        return f_src, f_tgts, sgram
    
   
if __name__ == "__main__":

    from Options.BaseOptions import opts
    opts.Dataset = '/home/amanankesh/working_dir/FCTFG/DATA/'
    opts.num_frames = 6

    import torchvision
    import torchvision.transforms as transforms

    dataset = FCTFG('train', opts)
    src, tgts, sgram = dataset[0]
    print(src.shape, tgts.shape, sgram.shape)
    

