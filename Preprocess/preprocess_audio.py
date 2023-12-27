audio_path = '/home/amanankesh/working_dir/FCTFG/Preprocess/n2.wav'
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load audio file

y, sr = librosa.load(audio_path, sr=None)  # Load with the original sample rate

# Compute spectrogram
spectrogram = librosa.feature.melspectrogram(y, sr=sr, n_fft=1024, hop_length=256, n_mels=128)
print(spectrogram.shape)
spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
print(spectrogram_db.shape)

# Plot and save spectrogram as an image
plt.figure(figsize=(10, 4))
librosa.display.specshow(spectrogram_db, x_axis='time', y_axis='mel', sr=sr, hop_length=256, cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.savefig('spectrogram.png')  # Save the plot as an image file

# Show the plot (optional)
plt.show()
