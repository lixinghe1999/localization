from torch.utils.data import Dataset
import os
import librosa
import json
import numpy as np
from spafe.features.gfcc import gfcc
from spafe.features.mfcc import mfcc


# compute features
n_fft = 1024
num_sectors = 8
sector_degree = 360 / 8
num_range = 5
sector_range = 1
max_source = 3

def DeepBSL(doa, ranges, config):
    label = np.zeros((max_source, 2), dtype=np.float32)
    for i, (d, r) in enumerate(zip(doa, ranges)):
        label[i, 0] = (d[0] - config['classifier']['min_azimuth']) / (config['classifier']['max_azimuth']- config['classifier']['min_azimuth'])
        label[i, 1] = (d[1] - config['classifier']['min_elevation']) / (config['classifier']['max_elevation']- config['classifier']['min_elevation'])
    return label

def Gaussian(doa, ranges, config):
    N = int(config['classifier']['max_azimuth'] - config['classifier']['min_azimuth'])
    azimuth = []
    y = np.linspace(config['classifier']['min_azimuth'], config['classifier']['max_azimuth'], N)
    for d, r in zip(doa, ranges):
        azimuth.append(np.exp(-((y - d[0]) ** 2) / (2 * 10 ** 2)))
    azimuth = np.max(np.array(azimuth), axis=0).astype(np.float32)

    N = int(config['classifier']['max_elevation'] - config['classifier']['min_elevation'])
    elevation = []
    y = np.linspace(config['classifier']['min_elevation'], config['classifier']['max_elevation'], N)
    for d, r in zip(doa, ranges):
        elevation.append(np.exp(-((y - d[1]) ** 2) / (2 * 10 ** 2)))
    elevation = np.max(np.array(elevation), axis=0).astype(np.float32)

    return (azimuth, elevation)

def filter_label(labels, max_azimuth=90, min_azimuth=-90):
    new_labels = []
    for label in labels:
        if label['doa_degree'][0][0] >= min_azimuth and label['doa_degree'][0][0] <= max_azimuth:
            new_labels.append(label)
    return new_labels

class Main_dataset(Dataset):
    def __init__(self, dataset, config=None, users=None, sr=16000, set_length=0.5):
        self.config = config
        self.encoding = globals()[self.config['encoding']]
        self.root_dir = dataset
        with open(os.path.join(self.root_dir, 'label.json')) as f:
            self.labels = json.load(f)
        self.labels = [label for label in self.labels]
        self.labels = filter_label(self.labels, self.config['classifier']['max_azimuth'], self.config['classifier']['min_azimuth'])
        if users is not None:
            self.labels = [d for d in self.data if d['fname'].split('/')[-1].split('_')[0] in users]            
        self.sr = sr
        self.gcc_phat_len = int(self.sr * 0.002)
        self.set_length = set_length
    def prune_extend(self, audio_file):
        length = int(self.sr * self.set_length)
        if audio_file.shape[1] > length:
            start_idx = np.random.randint(0, audio_file.shape[1] - length)
            audio_file = audio_file[:, start_idx : start_idx + length]
        # zero padding
        elif audio_file.shape[1] < length:
            pad_len = length - audio_file.shape[1]
            audio_file = np.pad(audio_file, ((0, 0), (0, pad_len)))
        return audio_file
    def __len__(self):
        return len(self.labels)
    def gccphat(self, audio1, audio2, interp=1):
        n = audio1.shape[0] + audio2.shape[0] 
        X = np.fft.rfft(audio1, n=n)
        Y = np.fft.rfft(audio2, n=n)
        R = X * np.conj(Y)
        cc = np.fft.irfft(R / (1e-6 + np.abs(R)),  n=(interp * n))
        cc = np.concatenate((cc[-self.gcc_phat_len:], cc[:self.gcc_phat_len+1])).astype(np.float32)
        return cc
    def mel_gccphat(self, audio1, audio2):
        n = audio1.shape[0] + audio2.shape[0]
        melfb = librosa.filters.mel(sr=self.sr, n_fft=n, n_mels=40)
        X = np.fft.rfft(audio1, n=n)
        Y = np.fft.rfft(audio2, n=n)
        R = X * np.conj(Y)
        R_mel = melfb * R
        cc = np.fft.irfft(R_mel / (np.abs(R_mel) + 1e-6), n=n)
        cc = np.concatenate((cc[:, -self.gcc_phat_len:], cc[:, :self.gcc_phat_len+1]), axis=1).astype(np.float32)
        cc = np.expand_dims(cc, axis=0)
        return cc
    def stft(self, audio):
        S = librosa.stft(audio, n_fft=n_fft)
        S_real = np.real(S)
        S_imag = np.imag(S)
        S = np.concatenate((S_real[np.newaxis, :], S_imag[np.newaxis, :]), axis=0, dtype=np.float32)
        return S 
    def gtcc(self, audio):
        # Define the Gammatone filterbank parameters
        gfccs = gfcc(audio, fs=self.sr, num_ceps=36, nfilts=48, nfft=n_fft).astype(np.float32)
        return gfccs[np.newaxis, :]
    def preprocess(self, audio):
        audio_feature = {}
        for key, value in self.config['backbone']['features'].items():
            if value == 0:
                continue
            else:
                if key in ['gccphat', 'mel_gccphat']: # pair-wise features
                    audio_feature[key] = []
                    key_features = []
                    for i in range(audio.shape[0]):
                        for j in range(i+1, audio.shape[0]):
                            audio1, audio2 = audio[i], audio[j]
                            key_features.append(getattr(self, key)(audio1, audio2))
                    key_features = np.concatenate(key_features, axis=0)
                    audio_feature[key] = key_features
                elif key in ['stft', 'gtcc', 'mfcc',]:
                    audio_feature[key] = []
                    key_features = []
                    for i in range(audio.shape[0]):
                        key_features.append(getattr(self, key)(audio[i]))
                    key_features = np.concatenate(key_features, axis=0)
                    audio_feature[key] = key_features
                else:
                    pass
        audio_feature['raw'] = audio
        return audio_feature
    def __getitem__(self, idx): 
        label = self.labels[idx]
        file = os.path.join('data', label['fname'])
        label = self.encoding(label['doa_degree'], label['range'], self.config)
        if self.config['data'] == 'binaural':
            audio = librosa.load(file + '.wav', sr=self.sr, mono=False)[0]
        elif self.config['data'] == 'micarray':
            audio = librosa.load(file + '_array.wav', sr=self.sr, mono=False)[0]
        else:
            raise NotImplementedError
        audio = self.prune_extend(audio)
        audio_features = self.preprocess(audio)
        if self.config['backbone']['features']['hrtf']:
            hrtf = np.load(file + '.npy').astype(np.float32)
            audio_features['hrtf'] = hrtf 

        return audio_features, label


  