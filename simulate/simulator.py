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
from parameter import SMARTGLASS, DUALDEVICE

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
def active_frame(audio, frame=0.1, sr=16000):
    '''
    segment the audio by frame_length, and return a mask to denote the active frame
    '''
    frame_length = int(sr * frame)
    if len(audio) % frame_length != 0:
        pad_length = frame_length - len(audio) % frame_length
        audio = np.pad(audio, (0, pad_length))
    audio_reshape = audio.reshape(-1, frame_length)
    energy = np.sum(audio_reshape ** 2, axis=1)
    energy = energy / np.max(energy)
    mask = energy > 0.1
    return audio, mask
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

class ISM_simulator():
    def __init__(self, device, 
                 ) -> None:
        if device == 'smartglass':
            mic_array = SMARTGLASS
        else: # dual-device
            mic_array = DUALDEVICE
        self.fs = 16000 
        self.max_order = 10
        self.snr_lb, self.snr_ub = 20, 30
        self.offset = 0.5
        self.mic_array = mic_array
        self.room_dims = random_room()
        self.HRTF = False
    def init_HRTF(self, HRTF_folder, user=None):
        self.HRTF_folder = HRTF_folder
        self.sofas = os.listdir(self.HRTF_folder)
        if user is not None:
            self.sofas = [self.sofas[i] for i in user]
        self.HRTFs = []
        for i, s in enumerate(self.sofas):
            HRTF_path = os.path.join(self.HRTF_folder, s)
            HRTF = sofa.Database.open(HRTF_path)
            M = HRTF.Dimensions.M
            HRTF_list = [[]] * (360 // 10)
            print(f"Loading user {i} with {M} measurements")
            for m in tqdm(range(M)):
                IR, relative_loc_sph = access_IR(HRTF, m, 0)
                azimuth_idx = int(relative_loc_sph[0] // 10)
                HRTF_list[azimuth_idx].append([relative_loc_sph, IR])
            self.HRTFs.append(HRTF_list)
            break
        self.HRTF = True
    def apply_HRTF(self, signals, doa_degree, ranges):
        HRTF_list = sample(self.HRTFs, 1)[0]
        assert len(signals) == len(doa_degree)
        HRTF_signals = []
        for i in range(len(signals)):
            s = signals[i]
            doa = doa_degree[i]
            r = ranges[i]
            doa_idx = int(doa[0] // 10)
            IRs = HRTF_list[doa_idx]
            IR = sample(IRs, 1)[0][1]

            left = signal.convolve(s[0], IR[0])
            right = signal.convolve(s[1], IR[1])
            HRTF_signals.append(np.c_[left, right].T)
        HRTF_signals = np.array(HRTF_signals)
        return HRTF_signals  
    def simulate(self, room, mic_center, dataset, sig_index, min_diff, max_range, out_folder):
        signals, doa_degree, ranges, class_names, active_masks = [], [], [], [], []
        for idx in sig_index:
            audio, class_name, active_mask = dataset[idx]
            if active_mask is None:
                audio, active_mask = active_frame(audio, frame=0.1)
            active_masks.append(active_mask)
            signals.append(audio)
            class_names.append(class_name)

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
            azimuth_rad = np.deg2rad(azimuth)
            source_loc = mic_center + np.array([r * np.cos(azimuth_rad), r * np.sin(azimuth_rad), 0])
            room.add_source(source_loc, signal=s)
        signals = room.simulate(return_premix=True)

        if self.HRTF:
            signals = self.apply_HRTF(signals, doa_degree, ranges)

        # mixed_signal = np.sum(signals, axis=0)
        # sf.write(out_folder + '.wav', mixed_signal.T, 16000)
        for i, s in enumerate(signals):
            os.makedirs(out_folder, exist_ok=True)
            sf.write(out_folder + '/{}.wav'.format(i), s.T, 16000)
        label = []
        for k in range(len(class_names)):
            active_mask = active_masks[k]
            # convert mask to list of active frames, pick the True index
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
            

            df = self.simulate(room, mic_center, dataset, sig_index, min_diff, max_range, f"{smartglass_folder}/{i}")
            meta_file = f"{meta_folder}/{i}.csv"
            df.to_csv(meta_file, index=False)
            
class HRTF_simulator():
    def __init__(self, HRTF_folder, split, sr=16000):
        self.HRTF_folder = HRTF_folder
        self.sr = sr
        sofas = os.listdir(self.HRTF_folder)

        if split == 'TRAIN':
            self.sofas = sofas[:40]
        else:
            self.sofas = sofas[40:]
    
    def simulate(self, HRTF, dataset, max_source, min_diff, out_folder):
        '''
        return source < max_source with class_name, signals, locations
        '''
        class_names, signals, doa_degree, ranges, active_masks = [], [], [], [], []
        num_source = sample(range(1, max_source + 1), 1)[0] # number of sources
        max_sample = 0
        for _ in range(num_source):
            data_index = sample(range(len(dataset)), 1)[0]
            while 1: # make sure the doa is not too close
                m = sample(range(HRTF.Dimensions.M), 1)[0]
                IR, relative_loc_sph = access_IR(HRTF, m, 0)
                ok_flag = True
                for pre in doa_degree:
                    diff = np.abs(pre[0] - relative_loc_sph[0])
                    if diff < min_diff:
                        ok_flag = False
                if ok_flag:
                    break
                
            audio, class_name, (start, end) = dataset[data_index]

            if start is None or end is None:
                audio, active_mask = active_frame(audio, frame=0.1)
            else:
                frame_number = int(len(audio) / self.fs / 0.1)
                active_mask = np.zeros(frame_number)
                start_frame = int(start / 0.1)
                end_frame = int(end / 0.1)
                active_mask[start_frame:end_frame] = 1

            class_names.append(class_name)
            left = signal.convolve(audio, IR[0])
            right = signal.convolve(audio, IR[1])
            binaural_signal = np.c_[left, right]

            signals.append(binaural_signal)
            doa_degree.append(relative_loc_sph[:2])
            ranges.append(relative_loc_sph[2])
            active_masks.append(active_mask)
            if np.shape(binaural_signal)[0] > max_sample:
                max_sample = np.shape(binaural_signal)[0]

        mixed_signal = np.zeros((max_sample, 2))
        label = []
        source_folder = out_folder + '_source'
        os.makedirs(source_folder, exist_ok=True)
        for k in range(len(class_names)):
            out_filename = source_folder + "/{}".format(k)
            sf.write(out_filename + '.wav', signals[k], 16000) # save the binaural signal first
            mixed_signal[:np.shape(signals[k])[0], :] += signals[k]

            active_mask = active_masks[k]
            # convert mask to list of active frames, pick the True index
            active_frames = np.where(active_mask)[0].tolist()
            for frame in active_frames:
                label.append([frame, class_names[k], k, doa_degree[k][0], doa_degree[k][1], ranges[k]])
        sf.write(out_folder + '.wav', mixed_signal, 16000) # save the mixed
        # save as csv file
        df = pd.DataFrame(label, columns=['frame', 'class', 'source', 'azimuth', 'elevation', 'distance'])
        # sort by frame
        df = df.sort_values(by=['frame'])
        return df

    def simulate_all(self, save_folder, dataset, num_data_per_user=1000, max_source=1, min_diff=45):
        '''
        Store format:
        [frame number][active class][source index][azimuth, elevation, distance]
        '''
        binaural_folder = save_folder + '/audio'
        os.makedirs(binaural_folder, exist_ok=True)
        meta_folder = save_folder + '/meta'
        os.makedirs(meta_folder, exist_ok=True)
        for i, s in enumerate(tqdm(self.sofas)):
            HRTF_path = os.path.join(self.HRTF_folder, s)
            HRTF = sofa.Database.open(HRTF_path)
            out_folder_list = []
            num_data_per_user = HRTF.Dimensions.M if num_data_per_user is None else num_data_per_user
            print(f"Processing {i} with {num_data_per_user} data per user)")
            for j in range(num_data_per_user):
                out_folder = f"{binaural_folder}/{i}_{j}"
                out_folder_list.append(out_folder)
                df = self.simulate(HRTF, dataset, max_source, min_diff, out_folder)
                meta_file = f"{meta_folder}/{i}_{j}.csv"
                df.to_csv(meta_file, index=False)




    