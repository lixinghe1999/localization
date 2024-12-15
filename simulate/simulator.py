import matplotlib.pyplot as plt
import numpy as np
import sofa
import os
from random import uniform, sample 
import scipy.signal as signal
from tqdm import tqdm
import soundfile as sf
import pandas as pd
from pyroomacoustics import doa, Room, ShoeBox
from parameter import SMARTGLASS, EARPHONE
from hrtf_utils import HRTF_simulator


def random_room(num_room=100, sr=16000):
    rooms = []
    for _ in range(num_room):
        width = uniform(3, 20)
        length = uniform(3, 20)
        height = uniform(2, 10)
        room_dim = [width, length, height]
        absorption = np.random.uniform(low=0.1, high=0.99)
        rooms.append([room_dim, absorption])
    return rooms

class simulator():
    def __init__(self, device, sr) -> None:
        if device == 'smartglass':
            self.mic_array = SMARTGLASS
            self.num_channel = np.shape(self.mic_array)[1]
            self.HRTF = False
            self.Reverb = True
        elif device == 'earphone':
            self.mic_array = EARPHONE
            self.HRTF = True
            self.Reverb = True
            self.num_channel = 2
            self.hrtf_simulator = HRTF_simulator()

        self.fs = sr 
        self.max_order = 10
        self.snr_lb, self.snr_ub = 20, 30
        self.offset = 0.5
        self.min_diff = 45
        self.room_dims = random_room()
    
    def simulate(self, data, room, mic_center, max_range, out_folder):
        signals, class_names, active_masks = [], [], []
        doa_degree, ranges = [], []
        for (audio, class_name, active_mask) in data:
            signals.append(audio)
            class_names.append(class_name)
            active_masks.append(active_mask)

            while 1:
                random_azimuth = uniform(0, 360)
                for doa in doa_degree:
                    diff = np.abs(doa[0] - random_azimuth)
                    if diff < self.min_diff:
                        continue
                break
            doa_degree.append([random_azimuth, 0])
            ranges.append(uniform(0.3, max_range))
        
        assert len(doa_degree) == len(ranges)

        if self.Reverb:
            for (azimuth, elevation), r, s in zip(doa_degree, ranges, signals):
                azimuth_rad = np.deg2rad(azimuth); elevation_rad = np.deg2rad(elevation)
                source_loc = mic_center + np.array([r * np.cos(azimuth_rad), r * np.sin(azimuth_rad), r * np.sin(elevation_rad)])
                room.add_source(source_loc, signal=s)
            signals = room.simulate(return_premix=True)
        else: # reverb off
            signals = np.array(signals)
            signals = np.repeat(signals[:, np.newaxis, :], 2, axis=1)

        if self.HRTF:
            signals = self.hrtf_simulator.apply_HRTF(signals, doa_degree)

        for i, s in enumerate(signals):
            os.makedirs(out_folder, exist_ok=True)
            sf.write(out_folder + '/{}.wav'.format(i), s.T, self.fs)
        label = []
        for k in range(len(class_names)):
            active_mask = active_masks[k]
            active_frames = np.where(active_mask)[0].tolist()
            for frame in active_frames:
                label.append([frame, class_names[k], k, doa_degree[k][0], doa_degree[k][1], ranges[k]])
        df = pd.DataFrame(label, columns=['frame', 'class', 'source', 'azimuth', 'elevation', 'distance'])
        df = df.sort_values(by=['frame'])
        return df  
    def simulate_all(self, save_folder, dataset, num_data=None, max_source=1):
        smartglass_folder = save_folder + '/audio'
        os.makedirs(smartglass_folder, exist_ok=True)
        meta_folder = save_folder + '/meta'
        os.makedirs(meta_folder, exist_ok=True)

        if num_data is None:
            num_data = len(dataset)
        else: 
            num_data = num_data
        for i in tqdm(range(num_data)):
            room_dim, absorption = sample(self.room_dims, 1)[0]
            room = ShoeBox(room_dim, fs=self.fs, max_order=10, absorption=absorption)

            mic_center = np.array([uniform(0 + self.offset, room_dim[0] - self.offset), 
                                   uniform(0 + self.offset, room_dim[1] - self.offset), uniform(1.5, 1.8)])
            room.add_microphone_array(mic_center[:, np.newaxis] + self.mic_array)

            max_range = min(room_dim[0]-mic_center[0], mic_center[0], room_dim[1]-mic_center[1], mic_center[1])
            sig_index = sample(range(len(dataset)), max_source)
            data = [dataset[i] for i in sig_index]
            
            df = self.simulate(data, room, mic_center, max_range, f"{smartglass_folder}/{i}")
            meta_file = f"{meta_folder}/{i}.csv"
            df.to_csv(meta_file, index=False)
