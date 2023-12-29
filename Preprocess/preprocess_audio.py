audio_path = '/home/amanankesh/working_dir/FCTFG/Preprocess/n2.wav'
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def plot_mel_spectrogram(audio_path, hop_length=200):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=16000)

    # Calculate mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y, sr=sr, hop_length=hop_length, n_mels=80)

    # Convert to decibels (log scale)
    mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
    print(mel_spec_db.shape)

    # Plot mel spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.show()

# Example usage
# audio_path = "path/to/your/audio/file.wav"
plot_mel_spectrogram(audio_path)
