from torch.utils.data import Dataset
import os
import librosa
import json
import numpy as np
from spafe.features import gfcc, mfcc

n_fft = 1024
num_sectors = 8
sector_degree = 360 / 8
num_range = 5
sector_range = 1
max_source = 3
def DeepEAR(doa, ranges):
    label = np.zeros((num_sectors, 7), dtype=np.float32)
    for d, r in zip(doa, ranges):
        if type(d) == list:
            d_azimuth = d[0]
            r_float = r[0]
        else:
            d_azimuth = d
            r_float = r
        idx = int((d_azimuth + 179) // (sector_degree))
        res = ((d_azimuth + 179) % (sector_degree)) / sector_degree
        label[idx, 0] = 1
        label[idx, 1] = res
        range_idx = int(min(4, r_float // sector_range))
        label[idx, 2 + range_idx] = 1
    return label
def DeepBSL(doa, ranges):
    label = np.zeros((max_source, 2), dtype=np.float32)
    for i, (d, r) in enumerate(zip(doa, ranges)):
        label[i, 0] = (d[0] + 180) / 360
        label[i, 1] = (d[1] + 180) / 360
    return label
def label_organize(label, label_func):
    '''
    split the area into sectors, assume each sector at most one sound source
    E.g., we get 1 (yes or no) + 1 (0-1) + 5 (one-hot distance) = 7 for each sector
    '''
    doa = label['doa_degree']; ranges = label['range']
    assert len(doa) == len(ranges)
    label = label_func(doa, ranges)
    return label
    

class TIMIT_dataset(Dataset):
    def __init__(self, split, sr=16000):
        self.root_dir = 'TIMIT/' + split
        self.data = []
        self.sr = sr
        for DR in os.listdir(self.root_dir):
            for P in os.listdir(os.path.join(self.root_dir, DR)):
                folder = os.path.join(self.root_dir, DR, P)
                files = os.listdir(folder)
                files = [os.path.join(folder, file) for file in files if file.endswith('.WAV')]
                self.data += (files)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        audio = librosa.load(self.data[idx], sr=self.sr)[0]
        # audio_trim, index = librosa.effects.trim(audio)
        return audio
class Main_dataset(Dataset):
    def __init__(self, dataset='TIMIT/HRTF', split='TRAIN', label_func='DeepEAR', users=None, sr=16000, set_length=1):
        self.root_dir = os.path.join(dataset, split)
        self.data = []
        with open(os.path.join(self.root_dir, 'label.json')) as f:
            self.labels = json.load(f)

        self.data = [label['fname'] for label in self.labels]
        if users is not None:
            self.data = [d for d in self.data if d.split('/')[-1].split('_')[0] in users]            
        self.sr = sr
        self.gcc_phat_len = int(self.sr * 0.003)
        self.set_length = set_length
        self.label_func = globals()[label_func]
    def prune_extend(self, audio_file):
        if audio_file.shape[1] > (self.sr * self.set_length):
            start_idx = np.random.randint(0, audio_file.shape[1] - self.sr * self.set_length)
            audio_file = audio_file[:, start_idx : start_idx + self.sr * self.set_length]
        # zero padding
        elif audio_file.shape[1] < (self.sr * self.set_length):
            pad_len = (self.sr * self.set_length) - audio_file.shape[1]
            audio_file = np.pad(audio_file, ((0, 0), (0, pad_len)))
        return audio_file
    def __len__(self):
        return len(self.data)
    def preprocess(self, audio):
        n = audio.shape[1] * 2 
        X = np.fft.rfft(audio[0], n=n)
        Y = np.fft.rfft(audio[1], n=n)
        R = X * np.conj(Y)
        R /= (np.abs(R) + 1e-6)
        gccphat = np.fft.irfft(R)[:self.gcc_phat_len].astype(np.float32)
        gammatone_left = gfcc.gfcc(audio[0], fs=self.sr, num_ceps=36, nfilts=50, nfft=int(self.sr * 0.05), low_freq=0, high_freq=8000, )
        gammatone_right = gfcc.gfcc(audio[1], fs=self.sr, num_ceps=36, nfilts=50, nfft=int(self.sr * 0.05),low_freq=0, high_freq=8000,)
        # gammatone_left = mfcc.mfcc(audio[0], fs=self.sr, num_ceps=100, nfilts=100, low_freq=0, high_freq=8000,)
        # gammatone_right = mfcc.mfcc(audio[1], fs=self.sr, num_ceps=100, nfilts=100, low_freq=0, high_freq=8000,)
        gammatone = np.concatenate((gammatone_left[np.newaxis, ...], gammatone_right[np.newaxis, ...]), axis=0, dtype=np.float32)
        audio_feature = {'gcc_phat': gccphat, 'gammatone': gammatone}
        return audio_feature
    def __getitem__(self, idx):
        audio = librosa.load(self.data[idx], sr=self.sr, mono=False)[0]
        audio = self.prune_extend(audio)
        label = self.labels[idx]
        label = label_organize(label, self.label_func)
        audio_feature = self.preprocess(audio)
        return audio_feature, label
    def get_raw(self, idx):
        audio = librosa.load(self.data[idx], sr=self.sr, mono=False)[0]
        label = self.labels[idx]
        return audio, label
class RAW_dataset(Dataset):
    def __init__(self, dataset='RAW_HRTF', split='TRAIN'):
        self.root_dir = os.path.join(dataset, split)
        self.data = []
        with open(os.path.join(self.root_dir, 'label.json')) as f:
            self.labels = json.load(f)
        self.data = [label['fname'] for label in self.labels]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        hrtf = np.load(self.data[idx]).astype(np.float32)
        label = self.labels[idx]
        label = label_organize(label)
        return {'HRTF': hrtf}, label




  