import os
import glob
import random
from PIL import Image, ImageFile
import torch 
from torch.utils.data import Dataset

from Utils.AudioUtils import *
import torchaudio.transforms as AudioTransforms
from torchaudio.transforms import Resample 

import torchvision.transforms as VisionTransforms
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
from torchvision.io.video import read_video, write_video

ImageFile.LOAD_TRUNCATED_IMAGES = True

class FCTFG_VIDEO(Dataset):
    def __init__(self, split, args, vtransform = None):
        super(FCTFG_VIDEO, self).__init__()

        ########## Audio ##############
        self.duration = 200                           # time in milliseconds
        self.aud_sr = 16000                           # in Hertz
        self.channel = 1                              # mono 1, streo 2
        self.n_mels = 80                              # num of melodies
        self.n_fft = 800                              # num of samples in one window or window length
        self.hop_length = 200                         # num of samples to skip, stride
        self.top_db = 80
        
        ###############################

        self.args = args
        if split == 'train':
            self.Data_path = self.args.train_dataset_path
            self.num_of_samples_per_video = 10
        elif split == 'test':
            self.Data_path = self.args.test_dataset_path
            self.num_of_samples_per_video = 1
        else:
            raise NotImplementedError

        self.video_path = os.path.join(self.Data_path, 'video')
        self.video_list = os.listdir(self.video_path)


        ########## Video #############
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
        video_path = os.path.join(self.video_path, self.video_list[idx])
        video, audio, info = read_video(video_path)
        # print(video_path)
        # print(info)
        if abs(info['video_fps'] - 25.0) > 0:
            video = self.change_frame_rate(video, info, self.video_fr)
            info['video_fps'] = self.video_fr
        
        return video.float(), audio, info


    def change_frame_rate(self, video, info, target_frame_rate):
       
        original_frame_rate = info['video_fps']
        frame_rate_ratio = original_frame_rate / target_frame_rate
        new_num_frames = int(video.size(0) / frame_rate_ratio)

        resized_video = F.resize(video, (new_num_frames,), InterpolationMode.NEAREST)

        return resized_video

    def mel_sgram_from_window(self, audio, info, start_frame=5, end_frame=10):
        audio = Resample(info['audio_fps'], self.aud_sr)(audio)

        audio_start_sample = int(start_frame / info['video_fps'] * self.aud_sr)
        audio_end_sample = int(end_frame / info['video_fps'] * self.aud_sr) - 1

        # print(start_frame, end_frame)
        audio_window = audio[:, audio_start_sample:audio_end_sample]

        if audio_window.size(0) > 1:
            audio_window = torch.mean(audio_window, dim=0, keepdim=True)
        
        mel_spectrogram = AudioTransforms.MelSpectrogram(n_fft=self.n_fft, n_mels=self.n_mels, hop_length=self.hop_length)(audio_window)
        mel_spectrogram = AudioTransforms.AmplitudeToDB(top_db=self.top_db)(mel_spectrogram)

        return mel_spectrogram


    def __getitem__(self, idx):
        samples = []
        video, audio, info = self.read_video(idx)
        for i in range(self.num_of_samples_per_video):
            s = random.randint(0, len(video) - self.window_size)
            e = min(s + self.window_size, len(video))
            v_window = torch.permute(video[s:e], (0, 3, 1, 2))
            v_window = self.vtransform(v_window)  

            mel_sgram = self.mel_sgram_from_window(audio, info, start_frame=s, end_frame=e)
            samples.append((v_window, mel_sgram))

        return samples
    

if __name__ == "__main__":

    from Options.BaseOptions import opts
    import matplotlib.pyplot as plt
    opts.train_dataset_path= '/home/zottang/working_dir/Videos'
    

    import torchvision
    import torchvision.transforms as transforms

    dataset = FCTFG_VIDEO('train', opts, None)
    print(len(dataset))
    v_window, mel_spectrogram = dataset[1]
    
    print(v_window.shape, mel_spectrogram.shape)

    # Display the mel spectrogram
    plt.imshow(mel_spectrogram[0].numpy(),  aspect='auto', origin='lower')
    plt.title('Mel Spectrogram for Video Frame Window')
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')
    plt.colorbar(format='%+2.0f dB')
    plt.show()
