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

class STARSS23_dataset(Dataset):
    def __init__(self, dataset_name, config=None, sr=16000):
        self.config = config
        print('config:', self.config['encoding'], )
        self.encoding = globals()[self.config['encoding']]
        self.root_dir = '/home/lixing/localization/dataset/STARSS23' 

        assert dataset_name in ['dev-test-sony', 'dev-test-tau', 'dev-train-sony', 'dev-train-tau']

        self.label_folder = os.path.join(self.root_dir, 'metadata_dev', dataset_name)
        self.data_folder = os.path.join(self.root_dir, 'mic_dev', dataset_name)
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
        for i, label_name in tqdm(enumerate(self.labels)):
            label = np.loadtxt(os.path.join(self.label_folder, label_name), delimiter=',', dtype=np.int32)
            audio_file = os.path.join(self.data_folder, label_name[:-4] + '.wav')
            # max_frame is controlled by both the audio duration and the last annotation
            audio_max_frame = int(librosa.get_duration(path=audio_file) * 10)
            # label: [[frame, xx, xx ,xx, xx], ...], unit of frame 100ms=0.1s
            annotation_max_frame = label[-1, 0]
            max_frame = min(audio_max_frame, annotation_max_frame)

            frame_duration = int(self.duration * 10)
            for start_frame in range(0, max_frame, frame_duration):
                end_frame = min(start_frame + frame_duration, max_frame)
                mini_chunk = []
                for l in label:
                    if l[0] >= start_frame and l[0] < end_frame:
                        mini_chunk.append(l)
                # print('start_frame:', start_frame, 'end_frame:', end_frame, 'mini_chunk:', len(mini_chunk))
                self.crop_labels.append((label_name, start_frame, end_frame, np.array(mini_chunk)))
        print('Total crop labels:', len(self.crop_labels))
    def label_convert(self, label, start_frame,):
        '''
        label: [[frame, class, source/instance, azimuth, elevation, distance], ...]
        cluster the label based on the index except frame

        return: {'class, source/instance': {'frame': [frame1, frame2, ...], 'location (average)': [azimuth, elevation, distance]}}
        '''
        label_dict = {}
        for l in label:
            key = tuple(l[1:3]) # class, source/instance
            if key not in label_dict:
                label_dict[key] = {'frame': [], 'location': []}
            label_dict[key]['frame'].append(l[0]-start_frame)
            label_dict[key]['location'].append(l[3:6])
        for key in label_dict:
            label_dict[key]['location'] = np.mean(label_dict[key]['location'], axis=0)
        return label_dict

    def __getitem__(self, index, encoding=False):
        '''
        label = [frame, class, source/instance, azimuth, elevation, distance]
        '''
        label_name, start_frame, end_frame, label = self.crop_labels[index]
        label = self.label_convert(label, start_frame)
        start_frame_audio = start_frame /10
        audio_name = os.path.join(self.data_folder, label_name)
        audio, sr = librosa.load(audio_name[:-4] + '.wav', sr=self.sr, mono=False, offset=start_frame_audio, duration=self.duration)
        if audio.shape[-1] < self.duration * self.sr:
                audio = np.pad(audio, ((0, 0), (0, self.duration * self.sr - audio.shape[-1])))
        if encoding:
            label = self.encoding(label, self.config).astype(np.float32)
            spec = spectrogram(audio)
            gcc = get_gcc(spec).astype(np.float32)
            return gcc, label
        else:
            return audio, label
    
    def __baseline__(self,):
        '''
        Run signal processing baseline directly
        '''
        from .doa import pra_doa
        from .parameter import mic_array_starss23
        distances = []
        for index in range(self.__len__()):
            label_name, start_frame, end_frame, label = self.crop_labels[index]
            start_frame_audio = start_frame /10
            audio_name = os.path.join(self.data_folder, label_name)
            audio, sr = librosa.load(audio_name[:-4] + '.wav', sr=self.sr, mono=False, offset=start_frame_audio, duration=self.duration)
            if audio.shape[-1] < self.duration * self.sr:
                audio = np.pad(audio, ((0, 0), (0, self.duration * self.sr - audio.shape[-1])))

            label = self.label_convert(label, start_frame)
            for _, values in label.items():
                azimuth = values['location'][0]
                # convert frames to audio segment
                min_frame = min(values['frame'])
                max_frame = max(values['frame'])+1
                audio_segment = audio[:, min_frame * 1600: max_frame * 1600]
                predictions, _ = pra_doa(audio_segment, mic_array_starss23, sr, 1024, intervals=[[0, audio_segment.shape[-1]]])
                prediction = np.mean(predictions)
                distance = distance = min(abs(prediction - azimuth), 360 - abs(prediction - azimuth))
                # print('predictions:', predictions, 'azimuth:', azimuth, 'distance:', distance)
                distances.append(distance)
        print('Mean error for baseline:', np.mean(distances))

class Mobile_dataset(Dataset):
    def __init__(self, dataset_name, config=None, sr=16000):
        self.config = config
        print('config:', self.config['encoding'], )
        self.encoding = globals()[self.config['encoding']]
        self.root_dir = '/home/lixing/localization/dataset/6DoFSELDDataset' 

        assert dataset_name in ['train', 'test']

        self.label_folder = os.path.join(self.root_dir, dataset_name, 'metadata/metadata_rel')
        self.data_folder = os.path.join(self.root_dir, dataset_name, 'audio')
        self.labels = os.listdir(self.label_folder)
        # make sure 
        self.sr = sr
        self.duration = self.config['duration']
        self.crop_dataset()
        self.class_names = ['female', 'male', 'clapping', 'telephone', 'laughter', 'domestic sound', 'walk, footsteps', 'door, open or close', 
                            'music', 'music instrument', 'water tap', 'bell', 'knock']
    def crop_dataset(self):
        '''
        split the full audio into small chunks, defined by the duration
        '''
        self.crop_labels = []
        for i, label_name in tqdm(enumerate(self.labels)):
            label = np.loadtxt(os.path.join(self.label_folder, label_name), delimiter=',', dtype=np.int32, skiprows=1)
            audio_file = os.path.join(self.data_folder, label_name[:-4] + '.wav')
            # max_frame is controlled by both the audio duration and the last annotation
            audio_max_frame = int(librosa.get_duration(path=audio_file) * 10)
            # label: [[frame, xx, xx ,xx, xx], ...], unit of frame 100ms=0.1s
            annotation_max_frame = label[-1, 0]
            max_frame = min(audio_max_frame, annotation_max_frame)

            frame_duration = int(self.duration * 10)
            for start_frame in range(0, max_frame, frame_duration):
                end_frame = min(start_frame + frame_duration, max_frame)
                mini_chunk = []
                for l in label:
                    if l[0] >= start_frame and l[0] < end_frame:
                        mini_chunk.append(l)
                self.crop_labels.append((label_name, start_frame, end_frame, np.array(mini_chunk)))
        print('Total crop labels:', len(self.crop_labels))
    def label_convert(self, label, start_frame,):
        '''
        label: [[frame, class, track, azimuth, elevation, distance], ...]
        cluster the label based on the index except frame

        return: {'class, source/instance': {'frame': [frame1, frame2, ...], 'location (average)': [azimuth, elevation, distance]}}
        '''
        label_dict = {}
        for l in label:
            key = tuple(l[1:3]) # class, source/instance
            if key not in label_dict:
                label_dict[key] = {'frame': [], 'location': []}
            label_dict[key]['frame'].append(l[0]-start_frame)
            label_dict[key]['location'].append(l[3:6])
        for key in label_dict:
            label_dict[key]['location'] = np.mean(label_dict[key]['location'], axis=0)
        return label_dict

    def __len__(self):
        return len(self.crop_labels)
    def __getitem__(self, index, encoding=True):
        '''
        label = [frame, class, source/instance, azimuth, elevation, distance]
        '''
        label_name, start_frame, end_frame, label = self.crop_labels[index]
        label = self.label_convert(label, start_frame)
        start_frame_audio = start_frame /10
        audio_name = os.path.join(self.data_folder, label_name)
        audio, sr = librosa.load(audio_name[:-4] + '.wav', sr=self.sr, mono=False, offset=start_frame_audio, duration=self.duration)
        audio = audio[[0, 4, 8, 12], :]
        if audio.shape[-1] < self.duration * self.sr:
                audio = np.pad(audio, ((0, 0), (0, self.duration * self.sr - audio.shape[-1])))
        if encoding:
            label = self.encoding(label, self.config).astype(np.float32)
            spec = spectrogram(audio)
            gcc = get_gcc(spec).astype(np.float32)
            return gcc, label
        else:
            return audio, label
    
        