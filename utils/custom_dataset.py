from torch.utils.data import Dataset
from .window_label import Gaussian_label, ACCDOA_label
from .window_feature import spectrogram, gcc_mel_spec 
import os
import librosa
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

def prune_extend(audio, length):
    if len(audio.shape) == 1:
        audio = audio[np.newaxis, :]
    if audio.shape[1] > length:
        # start_idx = np.random.randint(0, audio.shape[1] - length)
        audio = audio[:, :length]
    # zero padding
    elif audio.shape[1] < length:
        pad_len = length - audio.shape[1]
        audio = np.pad(audio, ((0, 0), (0, pad_len)))
    return audio

class Localization_dataset(Dataset):
    '''
    1. self-collected
    2. simulation by HRTF
    3. simulation by ISM (PRA)
    '''
    def __init__(self, root_dir, config=None, sr=16000):
        self.config = config
        self.encoding = self.config['encoding']
        if self.encoding == 'Gaussian':
            self.encoding = Gaussian_label
        elif self.encoding == 'ACCDOA':
            self.encoding = ACCDOA_label

        self.root_dir = root_dir
        self.data_folder = os.path.join(self.root_dir, 'audio')
        self.label_folder = os.path.join(self.root_dir, 'meta')
        self.labels = os.listdir(self.label_folder)
        # make sure 
        self.sr = sr
        self.duration = self.config['duration']
        self.crop_dataset()
        self.class_names = ['female', 'male', 'clapping', 'telephone', 'laughter', 'domestic sound', 'walk, footsteps', 'door, open or close', 
                            'music', 'music instrument', 'water tap', 'bell', 'knock']
    def __len__(self):
        return len(self.crop_labels)
    def crop_dataset(self):
        '''
        split the full audio into small chunks, defined by the duration
        '''
        self.crop_labels = []
        for i, label_name in enumerate(tqdm(self.labels)):
            label = np.loadtxt(os.path.join(self.label_folder, label_name), delimiter=',', dtype=np.int32, skiprows=1)
            # print(os.path.join(self.label_folder, label_name))
            if len(label.shape) == 1:
                label = label[np.newaxis, :]
            audio_file = os.path.join(self.data_folder, label_name[:-4] + '.wav')
            # max_frame is controlled by both the audio duration and the last annotation
            max_frame = int(librosa.get_duration(path=audio_file) * 10)
            # label: [[frame, xx, xx ,xx, xx], ...], unit of frame 100ms=0.1s
            # annotation_max_frame = label[-1, 0] if len(label) > 0 else 0
            # max_frame = min(audio_max_frame, annotation_max_frame)

            frame_duration = int(self.duration * 10)
            for start_frame in range(0, max_frame, frame_duration):
                end_frame = min(start_frame + frame_duration, max_frame)
                mini_chunk = []
                for l in label:
                    if len(l) == 0:
                        continue
                    if l[0] >= start_frame and l[0] < end_frame:
                        mini_chunk.append(l)
                # print('start_frame:', start_frame, 'end_frame:', end_frame, 'mini_chunk:', len(mini_chunk))
                self.crop_labels.append((label_name, start_frame, end_frame, np.array(mini_chunk)))
        print('Total crop labels:', len(self.crop_labels))

    def __getitem__(self, index, encoding=True):
        '''
        label = [frame, class, source/instance, azimuth, elevation, distance]
        '''
        label_name, start_frame, end_frame, label = self.crop_labels[index]
        start_frame_audio = start_frame / 10
        audio_name = os.path.join(self.data_folder, label_name)
        audio, sr = librosa.load(audio_name[:-4] + '.wav', sr=self.sr, mono=False, offset=start_frame_audio, duration=self.duration)
        if audio.shape[-1] < self.duration * self.sr:
                audio = np.pad(audio, ((0, 0), (0, self.duration * self.sr - audio.shape[-1])))
        if encoding:
            label = self.encoding(label, self.config).astype(np.float32)
            spec = spectrogram(audio)
            gcc = gcc_mel_spec(spec).astype(np.float32)
            return gcc, label
        else:
            return audio, label
        
if __name__ == "__main__":
    print(1)