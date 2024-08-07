from torch.utils.data import Dataset
from .feature import gccphat, mel_spec
from .label import Gaussian, filter_label
import os
import librosa
import json
import numpy as np


# compute features
n_fft = 1024
num_sectors = 8
sector_degree = 360 / 8
num_range = 5
sector_range = 1
max_source = 3


class Main_dataset(Dataset):
    def __init__(self, dataset, config=None, sr=16000):
        self.config = config
        self.encoding = globals()[self.config['encoding']]
        self.root_dir = dataset
        with open(os.path.join(self.root_dir, 'label.json')) as f:
            self.labels = json.load(f)
        self.labels = [label for label in self.labels]
        self.labels = filter_label(self.labels, self.config['classifier']['max_azimuth'], self.config['classifier']['min_azimuth'])
        self.sr = sr
        self.duration = self.config['duration']
        #  self.pre_compute_feature()

    def prune_extend(self, audio):
        length = int(self.sr * self.duration)
        if audio.shape[1] > length:
            start_idx = np.random.randint(0, audio.shape[1] - length)
            audio = audio[:, start_idx : start_idx + length]
        # zero padding
        elif audio.shape[1] < length:
            pad_len = length - audio.shape[1]
            audio = np.pad(audio, ((0, 0), (0, pad_len)))
        return audio
    def __len__(self):
        return len(self.labels)
   
    def preprocess(self, audio):
        audio_feature = {}
        for key, value in self.config['backbone']['features'].items():
            if value == 0:
                continue
            else:
                feat = globals()[key](audio)
                audio_feature[key] = feat
        return audio_feature
    def pre_compute_feature(self, save_folder='features'):
        from tqdm import tqdm
        # clear the folder
        os.system('rm -rf ' + save_folder)
        os.makedirs(save_folder, exist_ok=True)
        for idx in tqdm(range(self.__len__())):
            audio, _ = self.__getitem__(idx, on_the_fly=True)
            np.save(os.path.join(save_folder, str(idx)), audio)
        

    def __getitem__(self, idx, on_the_fly=False): 
        '''
        By default we use the pre-computed features
        '''
        label = self.labels[idx]
        file = os.path.join(self.root_dir, label['fname'])
        if on_the_fly:
            audio = librosa.load(file + '.wav', sr=self.sr, mono=False)[0]
            audio = self.prune_extend(audio)
            audio_features = self.preprocess(audio)
        else:
            audio_features = np.load(os.path.join('features', label['fname'] + '.npy'), allow_pickle=True).item()
        label = self.encoding(label['doa_degree'], label['range'], self.config)
        return audio_features, label
    
if __name__ == '__main__':
    dataset = Main_dataset('simulate/TIMIT/1/hrtf_TEST', config=json.load(open('configs/binaural_TIMIT.json', 'r')))



  