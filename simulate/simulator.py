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

def plot_coordinates(HRTF, title):
    coords = HRTF.Source.Position.get_values(system="cartesian")
    x0 = coords
    n0 = coords
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    q = ax.quiver(x0[:, 0], x0[:, 1], x0[:, 2], n0[:, 0],
                  n0[:, 1], n0[:, 2], length=0.1)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(title)
    return q
def access_IR(HRTF, measurement, emitter, plot=False):
    IRs = []
    t = np.arange(0, HRTF.Dimensions.N) * HRTF.Data.SamplingRate.get_values(indices={"M": measurement})
    for receiver in np.arange(HRTF.Dimensions.R):
        IR = HRTF.Data.IR.get_values(indices={"M": measurement, "R": receiver, "E": emitter})
        IRs.append(IR)
    IRs = np.array(IRs)
    # relative_loc_car = (np.round(HRTF.Emitter.Position.get_relative_values(HRTF.Listener, indices={"M":measurement}, angle_unit="degree"),2))
    relative_loc_sph = (np.round(HRTF.Emitter.Position.get_relative_values(HRTF.Listener, indices={"M":measurement}, system="spherical", angle_unit="degree"),2))
    relative_loc_sph = relative_loc_sph[0]
    relative_loc_sph[0] = relative_loc_sph[0] % 360 # convert to 0-360
    if plot:
        plt.figure(figsize=(15, 5))
        plt.plot(t, IRs.T)
        plt.title('HRIR at M={0} for emitter {1}'.format(measurement, emitter))
        plt.xlabel('$t$ in s')
        plt.ylabel(r'$h(t)$')
        plt.grid()
        plt.savefig('HRTF.png')
    return IRs, relative_loc_sph

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
    def __init__(self, device,) -> None:
        if device == 'smartglass':
            mic_array = SMARTGLASS
            self.HRTF = False
        elif device == 'earphone':
            mic_array = EARPHONE
            self.init_HRTF('HRTF-Database/ITA HRTF-database/SOFA')
        self.fs = 16000 
        self.max_order = 10
        self.snr_lb, self.snr_ub = 20, 30
        self.offset = 0.5
        self.mic_array = mic_array
        self.room_dims = random_room()
        
    def init_HRTF(self, HRTF_folder, user=range(1)):
        self.azimuth_interval = 10
        self.HRTF_folder = HRTF_folder
        self.sofas = os.listdir(self.HRTF_folder)
        if user is not None:
            self.sofas = [self.sofas[i] for i in user]
        self.HRTFs = []
        for i, s in enumerate(self.sofas):
            HRTF_path = os.path.join(self.HRTF_folder, s)
            HRTF = sofa.Database.open(HRTF_path)
            M = HRTF.Dimensions.M
            HRTF_list = [[]] * (360 // self.azimuth_interval)
            print(f"Loading user {i} with {M} measurements")
            for m in tqdm(range(M)):
                IR, relative_loc_sph = access_IR(HRTF, m, 0)
                azimuth_idx = int(relative_loc_sph[0] // self.azimuth_interval)
                HRTF_list[azimuth_idx].append([relative_loc_sph, IR])
            self.HRTFs.append(HRTF_list)
        self.HRTF = True
    def apply_HRTF(self, signals, doa_degree):
        HRTF_list = sample(self.HRTFs, 1)[0]
        assert len(signals) == len(doa_degree)
        HRTF_signals = []
        for (s, doa) in zip(signals, doa_degree):
            doa_idx = int(doa[0] // self.azimuth_interval)
            IRs = HRTF_list[doa_idx]
            IR = sample(IRs, 1)[0][1]

            left = signal.convolve(s[0], IR[0])
            right = signal.convolve(s[1], IR[1])
            HRTF_signals.append(np.c_[left, right].T)
        HRTF_signals = np.array(HRTF_signals)
        return HRTF_signals  
    
    def simulate(self, data, room, mic_center, min_diff, max_range, out_folder):
        signals, class_names, active_masks = [], [], []
        doa_degree, ranges = [], []
        for (audio, class_name, active_mask) in data:
            # if active_mask is None:
            #     audio, active_mask = active_frame(audio, frame=0.1)
            signals.append(audio)
            class_names.append(class_name)
            active_masks.append(active_mask)

            while 1:
                random_azimuth = uniform(0, 360)
                for doa in doa_degree:
                    diff = np.abs(doa[0] - random_azimuth)
                    if diff < min_diff:
                        continue
                break
            doa_degree.append([random_azimuth, 0])
            ranges.append(uniform(0.3, max_range))

        assert len(doa_degree) == len(ranges)
        for (azimuth, elevation), r, s in zip(doa_degree, ranges, signals):
            azimuth_rad = np.deg2rad(azimuth); elevation_rad = np.deg2rad(elevation)
            source_loc = mic_center + np.array([r * np.cos(azimuth_rad), r * np.sin(azimuth_rad), r * np.sin(elevation_rad)])
            room.add_source(source_loc, signal=s)
        signals = room.simulate(return_premix=True)

        if self.HRTF:
            signals = self.apply_HRTF(signals, doa_degree)

        for i, s in enumerate(signals):
            os.makedirs(out_folder, exist_ok=True)
            sf.write(out_folder + '/{}.wav'.format(i), s.T, 16000)
        label = []
        for k in range(len(class_names)):
            active_mask = active_masks[k]
            active_frames = np.where(active_mask)[0].tolist()
            for frame in active_frames:
                label.append([frame, class_names[k], k, doa_degree[k][0], doa_degree[k][1], ranges[k]])
        df = pd.DataFrame(label, columns=['frame', 'class', 'source', 'azimuth', 'elevation', 'distance'])
        df = df.sort_values(by=['frame'])
        return df
    
    def simulate_all(self, save_folder, dataset, num_data=None, max_source=1, min_diff=45):
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
            # num_source = sample(range(1, max_source + 1), 1)[0]
            num_source = max_source
            sig_index = sample(range(len(dataset)), num_source)
            data = [dataset[i] for i in sig_index]
            
            df = self.simulate(data, room, mic_center, min_diff, max_range, f"{smartglass_folder}/{i}")
            meta_file = f"{meta_folder}/{i}.csv"
            df.to_csv(meta_file, index=False)
        