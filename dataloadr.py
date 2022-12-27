import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io.wavfile import read as read_wav
import glob

BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
SILENCE_LABEL = 'silence'
UNKNOWN_LABEL = 'unknown'
WANTED_WORDS = ['yes','no','up','down','left','right','on','off','stop','go']
CLASSES = [SILENCE_LABEL, UNKNOWN_LABEL] + WANTED_WORDS


def split_data(data_dir, path):
    dir_names = ['train','test','validation']
    # Get all wav files in data_dir
    wav_files = glob.glob(os.path.join(data_dir, "*.wav"))
    # Randomly shuffle the files
    random.shuffle(wav_files)
    # Split the files into train, test, and validation sets
    train_size = int(0.8 * len(wav_files))
    test_size = int(0.2 * len(wav_files))
    files = [wav_files[:train_size], wav_files[train_size:train_size+test_size], wav_files[train_size+test_size:]]

    # Create the directories if they don't already exist
    for dir_name in dir_names:
        path_to_dir = os.path.join(path, dir_name)
        if not os.path.exists(path_to_dir):
            os.makedirs(path_to_dir)

    # Move the files to their respective directories
    for idx, dir_files in enumerate(files):
        dir_name = dir_names[idx]
        for file in dir_files:
            os.rename(file, os.path.join(dir_name, os.path.basename(file)))


class TestDataset(Dataset):
    def __init__(self, data_dir, preprocessor):
        self.data_dir = data_dir
        self.preprocessor = preprocessor
        self.create_data_index()
        self.classes = CLASSES

    def create_data_index(self):
        self.test_set = []
        search_path = os.path.join(self.data_dir, "*.wav")
        for path in gfile.Glob(search_path):
            filename = path.split("/")[-1]
            self.test_set.append({"path": path, "filename": filename})

    def __len__(self):
        return len(self.test_set)

    def __getitem__(self, idx):
        sample = self.test_set[idx]
        sr, signal = read_wav(sample["path"])
        signal = self.preprocessor.check_audio_length(signal)
        data = self.preprocessor.get_log_mel_spectrograms(signal)
        return {"data": data, "filename": sample["filename"]}





