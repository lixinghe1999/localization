from torch.utils.data import Dataset
import os
import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from random import sample

from .parameter import SPEED_OF_SOUND, MIC_ARRAY_SIMULATION

def shift_mixture(input_data, target_position, sr, inverse=False):
    """
    Shifts the input according to the voice position. This
    lines up the voice samples in the time domain coming from a target_angle
    Args:
        input_data - M x T numpy array or torch tensor
        target_position - The location where the data should be aligned
        mic_radius - In meters. The number of mics is inferred from
            the input_Data
        sr - Sample Rate in samples/sec
        inverse - Whether to align or undo a previous alignment

    Returns: shifted data and a list of the shifts
    """
    # elevation_angle = 0.0 * np.pi / 180
    # target_height = 3.0 * np.tan(elevation_angle)
    # target_position = np.append(target_position, target_height)

    num_channels = input_data.shape[0]

    # Must match exactly the generated or captured data
    # mic_array = [[
    #     mic_radius * np.cos(2 * np.pi / num_channels * i),
    #     mic_radius * np.sin(2 * np.pi / num_channels * i),
    # ] for i in range(num_channels)]
    mic_array = MIC_ARRAY_SIMULATION.T[:, :2]
    # Mic 0 is the canonical position
    distance_mic0 = np.linalg.norm(mic_array[0] - target_position)
    shifts = [0]

    # Check if numpy or torch
    if isinstance(input_data, np.ndarray):
        shift_fn = np.roll
    elif isinstance(input_data, torch.Tensor):
        shift_fn = torch.roll
    else:
        raise TypeError("Unknown input data type: {}".format(type(input_data)))

    # Shift each channel of the mixture to align with mic0
    for channel_idx in range(1, num_channels):
        distance = np.linalg.norm(mic_array[channel_idx] - target_position)
        distance_diff = distance - distance_mic0
        shift_time = distance_diff / SPEED_OF_SOUND
        shift_samples = int(round(sr * shift_time))
        if inverse:
            input_data[channel_idx] = shift_fn(input_data[channel_idx],
                                               shift_samples)
        else:
            input_data[channel_idx] = shift_fn(input_data[channel_idx],
                                               -shift_samples)
        shifts.append(shift_samples)

    return input_data, shifts

class Separation_dataset(Dataset):
    def __init__(self, root_dir, config=None):
        self.config = config
        self.root_dir = root_dir
        self.data_folder = os.path.join(self.root_dir, 'audio')
        self.label_folder = os.path.join(self.root_dir, 'meta')
        self.labels = os.listdir(self.label_folder)
        self.sr = self.config['sample_rate']
        self.duration = self.config['duration']
        self.output_format = self.config['output_format']
        self.max_sources = self.config['max_sources']
        self.crop_dataset()
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
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index): 

        label_name, start_frame, end_frame, label = self.crop_labels[index]
        start_frame_audio = start_frame / 10
        audio_name = os.path.join(self.data_folder, label_name)
        mixture_audio, sr = librosa.load(audio_name[:-4] + '.wav', sr=self.sr, mono=False, offset=start_frame_audio, duration=self.duration)
        if mixture_audio.shape[-1] < self.duration * self.sr:
                mixture_audio = np.pad(mixture_audio, ((0, 0), (0, self.duration * self.sr - mixture_audio.shape[-1])))

        source_audio = []
        source_files = os.listdir(audio_name[:-4])
        for i in range(self.max_sources):
            if i < len(source_files):
                source_file = source_files[i]
                source, sr = librosa.load(os.path.join(audio_name[:-4], source_file), sr=self.sr, mono=False, 
                                        offset=start_frame_audio, duration=self.duration)
                if source.shape[-1] < self.duration * self.sr:
                    source = np.pad(source, ((0, 0), (0, self.duration * self.sr - source.shape[-1])))
            else:
                n_channel = mixture_audio.shape[0]
                source = np.zeros((n_channel, self.duration * self.sr))
            source_audio.append(source)
        source_audio = np.array(source_audio)
        
        # print('mixture_audio shape:', mixture_audio.shape, 'source_audio shape:', source_audio.shape)
        if self.output_format == 'separation':
            # fix to the first channel
            mixture_audio = mixture_audio[0] # [left, T]
            source_audio = source_audio[:, 0] # [source, left, T]
            return mixture_audio, source_audio
        elif self.output_format == 'beamforming':
            source_active_azimuth = np.ones(360, dtype=np.int32) * -1
            for l in label:
                azimuth = l[3]
                azimuth_min = azimuth - 10; azimuth_max = azimuth + 10
                if azimuth_min < 0:
                    source_active_azimuth[azimuth_min + 360:360] = l[2]
                    source_active_azimuth[0:azimuth_max] = l[2]
                elif azimuth_max >= 360:
                    source_active_azimuth[azimuth_min:360] = l[2]
                    source_active_azimuth[0:azimuth_max - 360] = l[2]
                else:
                    source_active_azimuth[azimuth_min:azimuth_max] = l[2]
            if np.random.uniform() < 0 or len(source_files) < 1 or len(label) < 1: # negative example, output 0
                # return index where the source is not active
                negative_azimuth = np.where(source_active_azimuth == -1)[0]
                azimuth = np.random.choice(negative_azimuth)
                source_audio = np.zeros_like(mixture_audio)
            else:
                postive_azimuth = np.where(source_active_azimuth != -1)[0]
                azimuth = np.random.choice(postive_azimuth)
                source_idx = source_active_azimuth[azimuth]
                source_audio = source_audio[source_idx]
            target_pos = np.array([np.cos(np.deg2rad(azimuth)), np.sin(np.deg2rad(azimuth))])
            mixture_audio, shifts = shift_mixture(mixture_audio, target_pos, self.sr)

            mixture_audio = mixture_audio[:, 8:]
            source_audio = source_audio[:1, 8:]
            # print('mixture_audio shape:', mixture_audio.shape, 'source_audio shape:', source_audio.shape, shifts)
            return mixture_audio, source_audio
        elif self.output_format == 'region':
            number_of_regions = self.config['model']['n_src'] # note that it is the number of regions, speakers < regions
            regions = np.linspace(self.config['min_azimuth'], self.config['max_azimuth'], number_of_regions + 1)
            
            region_audio = np.zeros((number_of_regions, self.config['sample_rate'] * self.config['duration']), dtype=np.float32)
            region_active = np.zeros(number_of_regions, dtype=np.int32)
            # label: [frame, class, source/instance, azimuth, elevation, distance]
            for i, source in enumerate(source_audio):
                if len(label) == 0: # no label
                    continue
                # pick the source_i from label, get the average location, for fixed object, no need to average
                source_i_loc = label[label[:, 2] == i, 3:6]
                if len(source_i_loc) == 0:
                    continue
                else:
                    source_i_loc = np.mean(source_i_loc, axis=0)
                    azimuth, elevation, distance = source_i_loc
                    region_idx = np.digitize(azimuth, regions) - 1
                    region_active[region_idx] = 1
                    region_audio[region_idx] += source[0] # left channel
            # print(mixture_audio.shape, 'region_audio shape:', region_audio.shape, 'region_active:', region_active)
            return mixture_audio, region_audio
       
