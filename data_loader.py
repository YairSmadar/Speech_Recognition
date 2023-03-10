import os
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.io.wavfile import read as read_wav
import glob
from preprocess import AudioProcessor
import warnings

warnings.filterwarnings("ignore")
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'


class SpectogramDataset(Dataset):
    def __init__(self, opt, set_type: str = 'train'):
        super(SpectogramDataset, self).__init__()
        self.opt = opt
        self.set_type = set_type
        self.preprocessor = AudioProcessor(opt)
        self.class_label_dict = self.get_class_label_dict(opt)
        self.data_files_list = self.get_data_file_list()
        self.noise_files = self.get_noise_samples()
        self.is_train = True

    def __len__(self):
        return len(self.data_files_list)

    def __getitem__(self, index):

        is_zeros = False
        path = self.data_files_list[index]

        cls = get_class(path)

        sample_rate, data = read_wav(path)
        data = self.preprocessor.ensure_correct_length(data)

        if self.is_train:
            data, is_zeros = self.add_augmentations(data)

        cls = 'silence' if is_zeros else cls

        mel_spectogram = self.preprocessor.get_log_mel_spectra(data.astype(np.float32))

        H,W = mel_spectogram.shape

        data_dict = [mel_spectogram.reshape(1, H, W), self.class_label_dict[cls]]
        return data_dict

    def add_augmentations(self, data):
        is_zeros = False
        # replace data with zeros
        if random.random() < self.opt.silence_freq:
            data = np.zeros(self.preprocessor.sample_length)
            is_zeros = True

        # add noise
        if random.random() < self.opt.noise_freq:
            assert (len(self.noise_files) != 1 or len(self.noise_files) != 0)
            noise_idx = random.randint(0, len(self.noise_files) - 1)
            noise = self.noise_files[noise_idx]
            data = self.preprocessor.add_random_noise(data, noise, noise_strength=self.opt.noise_strength)

        # shift signal
        if random.random() < self.opt.shift_signal_freq:
            shift = random.randint(-self.opt.max_signal_shift_samples, self.opt.max_signal_shift_samples)
            data = self.preprocessor.shift_time(data, shift)

        return data, is_zeros

    def get_class_label_dict(self, opt):
        classes = [d for d in os.listdir(opt.dataroot) if os.path.isdir(os.path.join(opt.dataroot, d))]
        classes.append('silence')
        classes.remove(BACKGROUND_NOISE_DIR_NAME)
        class_label_dict = {}
        for idx, cls in enumerate(classes):
            class_label_dict[cls] = idx

        return class_label_dict

    def get_data_file_list(self):
        if self.set_type == 'train':
            data_path = os.path.join(self.opt.save_directory, 'train.txt')
        elif self.set_type == 'test':
            data_path = os.path.join(self.opt.save_directory, 'test.txt')
        else:
            data_path = os.path.join(self.opt.save_directory, 'validation.txt')
        return read_file_to_list('./' + data_path)

    def get_noise_files_list(self):
        noise_path = os.path.join(self.opt.dataroot, BACKGROUND_NOISE_DIR_NAME) if \
            self.opt.noise_dir_path is None else self.opt.noise_dir_path
        noise_files_paths = glob.glob(os.path.join(noise_path, '*.wav'))
        return noise_files_paths

    def get_noise_samples(self):
        ret_noise_list = []
        noise_list = self.get_noise_files_list()

        # crop noise signals in the length of input data for future augmentation
        for noise_path in noise_list:
            sample_rate, signal = read_wav(noise_path)
            for i in range(0, len(signal) - sample_rate, sample_rate):
                ret_noise_list.append(signal[i:i + sample_rate])
        return ret_noise_list

    def plot_spectogram(self, spectogram, lable):
        _class = list(filter(lambda x: self.class_label_dict[x] == lable, self.class_label_dict))[0]
        self.preprocessor.plot_spectogram(spectogram, _class)


def read_file_to_list(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]
    return lines


def get_class(path):
    return os.path.basename(os.path.dirname(path))


def getDataLoader(spec_dataset: SpectogramDataset, batch_size: int = 128, shuffle: bool = True):
    return DataLoader(spec_dataset, batch_size=batch_size, shuffle=shuffle)
