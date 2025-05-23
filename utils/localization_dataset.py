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
    def __init__(self, root_dir, config=None):
        self.config = config
        self.encoding = self.config['encoding']
        if self.encoding == 'ACCDOA':
            self.encoding = ACCDOA_label
        elif self.encoding == 'Multi_ACCDOA':
            self.encoding = Multi_ACCDOA_label
        elif self.encoding == 'Region':
            self.encoding = Region_label

        self.root_dir = root_dir
        self.data_folder = os.path.join(self.root_dir, 'audio')
        self.motion_folder = os.path.join(self.root_dir, 'imu')
        self.label_folder = os.path.join(self.root_dir, 'meta')
        self.labels = os.listdir(self.label_folder)
        # make sure 
        self.sr = self.config['sr']
        self.duration = self.config['duration']
        self.frame_duration = self.config['frame_duration']
        self.class_names = config['class_names']
        self.num_classes = len(self.class_names)
        self.raw_audio = self.config['raw_audio']
        self.label_type = self.config['label_type']
        self.motion = self.config['motion']
        self.mixture = self.config['mixture']

        self.crop_dataset()
        self._cache_(skip_cache=False)

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
        if len(label.shape) == 1:
            label = label[np.newaxis, :]

        if self.mixture:
            audio_file = os.path.join(self.data_folder, label_name[:-4] + '.wav')
        else:
            audio_file = os.path.join(self.data_folder, label_name[:-4] + '/0.wav')
        max_frame = int(librosa.get_duration(path=audio_file) / 0.1)
        num_frame = int(self.duration / 0.1)

        for start_frame in range(0, max_frame, num_frame):
            end_frame = min(start_frame + num_frame, max_frame)
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
    
    def _cache_(self, skip_cache=False):
        print('Caching the dataset')
        cache_folder = os.path.join(self.root_dir, 'cache')
        os.makedirs(cache_folder, exist_ok=True)
        if skip_cache:
            print('Already cached')
        else:
            batch_size = 8
            loader = DataLoader(self, batch_size=batch_size, shuffle=False, num_workers=batch_size)
            for i, data in enumerate(tqdm(loader)):
                data = data['spatial_feature'].numpy()
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

        if self.mixture:
            audio, sr = librosa.load(audio_name[:-4] + '.wav', sr=self.sr, mono=False, offset=start_frame_audio, duration=self.duration)
            if audio.shape[-1] < self.duration * self.sr:
                audio = np.pad(audio, ((0, 0), (0, self.duration * self.sr - audio.shape[-1])))
            else:
                audio = audio[:, :int(self.duration * self.sr)]
        else:
            source_audio = []
            source_files = os.listdir(audio_name[:-4])
            for i in range(len(source_files)):
                source_file = source_files[i]
                source, sr = librosa.load(os.path.join(audio_name[:-4], source_file), sr=self.sr, mono=False, 
                                        offset=start_frame_audio, duration=self.duration)
                if source.shape[-1] < self.duration * self.sr:
                    source = np.pad(source, ((0, 0), (0, self.duration * self.sr - source.shape[-1])))
                else:
                    source = source[:, :int(self.duration * self.sr)]
                source_audio.append(source)
            source_audio = np.array(source_audio)
            audio = np.sum(source_audio, axis=0).astype(np.float32)

        if hasattr(self, 'cache_folder'):
            spatial_feature = np.load(os.path.join(self.cache_folder, f'{index}.npy'))
        else:
            spec = spectrogram(audio, sr=self.sr)
            spatial_feature = gcc_mel_spec(spec).astype(np.float32)

        if self.motion:
            imu_name = os.path.join(self.motion_folder, label_name.replace('.txt', '.npy'))
            imu = np.load(imu_name).astype(np.float32)
            start_frame_imu = int(start_frame * 5) # 50Hz
            imu = imu[start_frame_imu:start_frame_imu + int(self.duration * 50)]
            if imu.shape[0] < int(self.duration * 50):
                imu = np.pad(imu, ((0, int(self.duration * 50 - imu.shape[0])), (0, 0)))
        else:
            imu = np.zeros((int(self.duration * 50), 6)).astype(np.float32)
        return {'spatial_feature': spatial_feature, 'audio': audio, 'label': label, 'imu': imu}
            
        
