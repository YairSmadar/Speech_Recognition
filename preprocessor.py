'''
    Preprocess data and create MFCC or log_mel_spectrograms
'''

import numpy as np
import librosa
import random
import scipy.signal
import matplotlib.pyplot as plt
import librosa.display

class AudioProcessor:
    def __init__(self, opt):
        self.num_features = opt.num_features
        self.window_duration_ms = opt.window_duration_ms
        self.window_step_ms = opt.window_step_ms
        # Setting properties
        self.sample_rate = opt.sample_rate
        self.sample_length = opt.sample_length
        self.window_duration_samples = int(self.window_duration_ms * self.sample_rate / 1000)
        self.window_step_samples = int(self.window_step_ms * self.sample_rate / 1000)
        # Compute derived properties
        self.num_windows = 1 + int(self.sample_length / self.window_step_samples)
        self.feature_vector_size = self.num_windows * self.num_features

    def ensure_correct_length(self, audio_data):
        if len(audio_data) > self.sample_length:
            # Cut to fixed length
            audio_data = audio_data[:self.sample_length]
        elif len(audio_data) < self.sample_length:
            # Pad with zeros at the end
            padding = self.sample_length - len(audio_data)
            audio_data = np.pad(audio_data, (0,padding), 'constant', constant_values=0)
        return audio_data

    def add_random_noise(self, data, noise_data, noise_strength=0.3):
        # Choose a random noise
        # Add it to the sound
        noise_energy = np.sqrt(np.sum(noise_data ** 2))
        data_energy = np.sqrt(np.sum(data ** 2))
        if data_energy > 0:
            result = data + (noise_strength * noise_data * data_energy / noise_energy).astype(np.int16)
        else:
            result = noise_strength * noise_data * 0.01 #Reduce more
        return result

    def shift_time(self, data, shift=0):
        result = data
        if shift > 0:
            # First check that the first part has low volume
            if np.sum(data[:shift]**2) < 0.01 * np.sum(data**2):
                # Pad and slice
                result = np.pad(data, (0, shift), 'constant', constant_values=0)[shift:]
        else:
            # Check if the last part has low volume
            if np.sum(data[shift:]**2) < 0.01 * np.sum(data**2):
                # Pad and slice
                result = np.pad(data, (-shift, 0), 'constant', constant_values=0)[:shift]
        return result
    
    def plot_spectogram(self, spectogram, _class):
        # plt.figure()
        # librosa.display.specshow(librosa.power_to_db(spectogram.reshape(4096,1), ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
        # plt.colorbar(format='%+2.0f dB')
        # plt.title('Mel spectrogram')
        # plt.tight_layout()
        # plt.show()

        plt.specgram(spectogram.reshape(4096), Fs=16000)
        plt.title(_class)
        plt.xlabel('Time')
        plt.ylabel('Frequency')

        plt.show()

        

    def get_log_mel_spectra(self, audio_data):
        spectrogram = librosa.feature.melspectrogram(y = audio_data, sr=self.sample_rate, n_mels=128,fmax=8000)
        spectrogram = librosa.power_to_db(spectrogram)
        spectrogram = spectrogram.astype(np.float32)
    
        return spectrogram.flatten()
