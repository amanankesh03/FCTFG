
import os
from AudioUtils import AudioUtil
from torch.utils.data import DataLoader, Dataset, random_split

class AudioDataset(Dataset):
  def __init__(self, data_path):
    
    self.data_list = os.listdir(data_path)
    self.data_list = [os.path.join(data_path, name) for name in self.data_list]
    self.duration = 200
    self.sr = 16000
    self.channel = 1
    self.shift_pct = 0.4
            
  def __len__(self):
    return len(self.data_list)    
    
  def __getitem__(self, idx):
    # Absolute file path of the audio file - concatenate the audio directory with
    # the relative path
    audio_file = self.data_list[idx % self.__len__()]
    

    aud = AudioUtil.open(audio_file)
    reaud = AudioUtil.resample(aud, self.sr)
    print(reaud.shape)
    rechan = AudioUtil.rechannel(reaud, self.channel)
    print(rechan.shape)
    dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
    print(dur_aud.shape)
    shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
    print(shift_aud.shape)
    sgram = AudioUtil.spectro_gram(dur_aud, n_mels=80, n_fft=800, hop_len=200)
    
    aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

    return aug_sgram