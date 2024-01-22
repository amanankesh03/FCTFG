audio_path = '/home/amanankesh/working_dir/FCTFG/Preprocess/n2.wav'
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import random
import torch

"""
For the audio source, we downsample the audio to 16kHz, then convert the downsampled audio to mel-spectrograms 
with a window size of 800, a hop length of 200, and 80 Mel filter banks
"""

def plot_mel_spectrogram(audio_path, hop_length=200, max_ms = 4):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=16000)
    y = torch.from_numpy(y).unsqueeze(0)
    y = torch.cat([y, y], dim=0)
    y, sr = pad_trunc((y, sr), max_ms)
    print(y.shape, sr)
    y = y.numpy()
    # Calculate mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y, sr=sr, n_fft = 800, hop_length=hop_length, n_mels=80)

    # Convert to decibels (log scale)
    mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
    print(mel_spec_db.shape)

    # Plot mel spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.show()


def pad_trunc(aud, max_s):
    sig, sr = aud
    num_rows, sig_len = sig.shape
    max_len = sr * max_s
    print(num_rows, sig_len, max_len)
    if (sig_len > max_len):
      # Truncate the signal to the given length
      sig = sig[:,:max_len]

    elif (sig_len < max_len):
      # Length of padding to add at the beginning and end of the signal
      pad_begin_len = random.randint(0, max_len - sig_len)
      pad_end_len = max_len - sig_len - pad_begin_len

      # Pad with 0s
      pad_begin = torch.zeros((num_rows, pad_begin_len))
      pad_end = torch.zeros((num_rows, pad_end_len))

      sig = torch.cat((pad_begin, sig, pad_end), 1)
      
    return (sig, sr)

plot_mel_spectrogram(audio_path, hop_length=200, max_ms=4)