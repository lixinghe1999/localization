from torch.utils.data import Dataset, DataLoader
from .window_label import ACCDOA_label, Multi_ACCDOA_label, Region_label
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
        if self.encoding == 'ACCDOA':
            self.encoding = ACCDOA_label
        elif self.encoding == 'Multi_ACCDOA':
            self.encoding = Multi_ACCDOA_label
        elif self.encoding == 'Region':
            self.encoding = Region_label
        elif self.encoding == 'Gaussian':
            self.encoding = Gaussian_label

        self.root_dir = root_dir
        self.data_folder = os.path.join(self.root_dir, 'audio')
        self.label_folder = os.path.join(self.root_dir, 'meta')
        self.labels = os.listdir(self.label_folder)
        # make sure 
        self.sr = sr
        self.duration = self.config['duration']
        self.frame_duration = self.config['frame_duration']
        self.class_names = ['alarm', 'baby', 'blender', 'cat', 'crash', 'dishes', 'dog', 'engine', 'fire', 'footsteps', 
                            'glassbreak', 'gunshot', 'knock', 'phone', 'piano', 'scream', 'speech', 'water']

        self.raw_audio = self.config['raw_audio']
        self.label_type = self.config['label_type']

        self.crop_dataset()

    def __len__(self):
        return len(self.crop_labels)
    
    def framewise_meta(self, label_name):
        '''
        DCASE-SELD Format
        '''
        if isinstance(label_name, str):
            label = np.loadtxt(os.path.join(self.label_folder, label_name), delimiter=',', dtype=np.int32, skiprows=1)
        else:
            label, label_name = label_name
        # print(os.path.join(self.label_folder, label_name))
        if len(label.shape) == 1:
            label = label[np.newaxis, :]
        audio_file = os.path.join(self.data_folder, label_name[:-4] + '.wav')
        max_frame = int(librosa.get_duration(path=audio_file) / 0.1)

        frame_duration = int(self.duration / 0.1)

        for start_frame in range(0, max_frame, frame_duration):
            end_frame = min(start_frame + frame_duration, max_frame)
            mini_chunk = []
            for l in label:
                if len(l) == 0:
                    continue
                if l[0] >= start_frame and l[0] < end_frame:
                    l[0] = l[0] - start_frame
                    mini_chunk.append(l)
            self.crop_labels.append((label_name, start_frame, end_frame, np.array(mini_chunk)))
    
    def eventwise_meta(self, label_name):
        # load the label, the first column is the classname
        label = pd.read_csv(os.path.join(self.label_folder, label_name), delimiter=',', header=0, converters={0: str})
        # sound_event_recording,start_time,end_time,azi,ele,dist
        # convert to framewise: frame,class,source,azimuth,elevation,distance
        frame_metadata = []
        frame_source = {}
        for entry in label.iterrows():
            sound_event_recording, start_time, end_time, azi, ele, dist = entry[1]
            # Calculate the frame range
            start_frame = int(start_time / 0.1)
            end_frame = int(end_time / 0.1)
            class_idx = self.class_names.index(sound_event_recording)

            # Create entries for each frame in the range
            for frame in range(start_frame, end_frame):
                if frame not in frame_source:
                    frame_source[frame] = 0

                frame_metadata.append([
                    int(frame),
                    class_idx,  # class
                    frame_source[frame],  
                    azi,
                    ele,
                    dist
                ])
                frame_source[frame] += 1
        frame_metadata = np.array(frame_metadata)
        self.framewise_meta((frame_metadata, label_name))
            
    def crop_dataset(self):
        '''
        split the full audio into small chunks, defined by the duration
        '''
        self.crop_labels = []
        for i, label_name in enumerate(tqdm(self.labels)):
            if self.label_type == 'framewise':
                self.framewise_meta(label_name)
            else:
                self.eventwise_meta(label_name)
        print('Total crop labels:', len(self.crop_labels))
    
    def _cache_(self, cache_folder):
        print('Caching the dataset')
        os.makedirs(cache_folder, exist_ok=True)

        num_cached = len(os.listdir(cache_folder))
        if self.__len__() == num_cached:
            print('Already cached')
        else:
            batch_size = 8
            loader = DataLoader(self, batch_size=batch_size, shuffle=False, num_workers=batch_size)
            for i, (data, _, _) in enumerate(tqdm(loader)):
                data = data.numpy()
                for j in range(data.shape[0]):
                    np.save(os.path.join(cache_folder, f'{i * batch_size + j}.npy'), data[j])
        self.cache_folder = cache_folder
    
    def __getitem__(self, index):
        '''
        label = [frame, class, source/instance, azimuth, elevation, distance]
        '''
        label_name, start_frame, end_frame, label = self.crop_labels[index]
        label = self.encoding(label, self.config).astype(np.float32)

        start_frame_audio = start_frame / 10
        audio_name = os.path.join(self.data_folder, label_name)
        if hasattr(self, 'cache_folder'):
            data = np.load(os.path.join(self.cache_folder, f'{index}.npy'))
            audio, sr = librosa.load(audio_name[:-4] + '.wav', sr=self.sr, mono=True, offset=start_frame_audio, duration=self.duration)
            if audio.shape[-1] < self.duration * self.sr:
                audio = np.pad(audio, ((0, self.duration * self.sr - audio.shape[-1])))
        else:
            audio, sr = librosa.load(audio_name[:-4] + '.wav', sr=self.sr, mono=False, offset=start_frame_audio, duration=self.duration)
            if audio.shape[-1] < self.duration * self.sr:
                audio = np.pad(audio, ((0, 0), (0, self.duration * self.sr - audio.shape[-1])))
            spec = spectrogram(audio)
            data = gcc_mel_spec(spec).astype(np.float32)
            audio = audio[0]
        return data, audio, label
            
        
