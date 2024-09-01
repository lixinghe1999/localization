import matplotlib.pyplot as plt
import numpy as np
import sofa
import os
from random import uniform, sample 
import scipy.signal as signal
from tqdm import tqdm
import soundfile as sf
import json

from pyroomacoustics import doa, Room, ShoeBox

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
    relative_loc_car = (np.round(HRTF.Emitter.Position.get_relative_values(HRTF.Listener, indices={"M":measurement}, angle_unit="degree"),2))
    relative_loc_sph = (np.round(HRTF.Emitter.Position.get_relative_values(HRTF.Listener, indices={"M":measurement}, system="spherical", angle_unit="degree"),2))
    # r = np.sum(relative_loc_car ** 2) ** 0.5
    #   print(IRs.shape, relative_loc_car, relative_loc_sph, r)
    if plot:
        plt.figure(figsize=(15, 5))
        plt.plot(t, IRs.T)
        plt.title('HRIR at M={0} for emitter {1}'.format(measurement, emitter))
        plt.xlabel('$t$ in s')
        plt.ylabel(r'$h(t)$')
        plt.grid()
        plt.savefig('HRTF.png')
    return IRs, relative_loc_sph




class ISM_simulator():
    def __init__(self) -> None:
        self.fs = 16000 
        self.max_order = 10
        self.snr_lb, self.snr_ub = 0, 30
        self.offset = 0.5
        self.mic_array =  np.c_[[ 0.06,  0.0, 0.0],
                        [ -0.06,  0.0, 0.0],
                            [ 0.03,  0.06, 0.0],
                            [ 0.03,  -0.06, 0.0],
                            [ -0.03,  0.06, 0.0],
                            [ -0.03,  -0.06, 0.0],]
        self.room_dims = [[5, 5, 3], [10, 10, 3], [20, 20, 3], [40, 20, 3]]
    def simulate(self, room_dim, mic_center, doa_degree, range, signal):
        room = ShoeBox(room_dim, fs=self.fs, max_order=self.max_order)
        assert len(doa_degree) == len(range)
        room.add_microphone_array(mic_center[:, np.newaxis] + self.mic_array)
        for (doa, elevation), r, s in zip(doa_degree, range, signal):
            doa_rad = np.deg2rad(doa)
            if type(doa) == list: #both azimuth + elevation
                source_loc = mic_center + np.array([r[0]*np.cos(doa_rad[0]), r[0]*np.sin(doa_rad[0]), r[0]*np.sin(doa_rad[1])])
            else:
                source_loc = mic_center + np.array([r*np.cos(doa_rad), r*np.sin(doa_rad), 0])   
        
            room.add_source(source_loc, signal=s)
        room.simulate(snr=uniform(self.snr_lb, self.snr_ub))
        signals = room.mic_array.signals
        return signals
    def simulate_all(self, save_folder, dataset, num_data=1000, max_source=1, min_diff=45):
        f = open(save_folder + '/label.json', 'w')
        labels = []
        for i in tqdm(range(num_data)):
            room_dim = sample(self.room_dims, 1)[0]
            mic_center = np.array([uniform(0 + self.offset, room_dim[0] - self.offset), 
                                   uniform(0 + self.offset, room_dim[1] - self.offset), uniform(1.5, 1.8)])
            max_range = min(room_dim[0]-mic_center[0], mic_center[0], room_dim[1]-mic_center[1], mic_center[1])
            num_source = sample(range(1, max_source + 1), 1)[0]
            sig_index = sample(range(len(dataset)), num_source)
            signal, doa_degree, ranges, file_names = [], [], [], []
            for idx in sig_index:
                audio, file_name = dataset[idx]
                signal.append(audio)
                file_names.append(file_name)
                while 1:
                    random_doa = uniform(0, 360)
                    for doa in doa_degree:
                        diff = np.abs(doa - random_doa)
                        if diff < min_diff:
                            continue
                    break
                doa_degree.append([random_doa, 0])
                ranges.append(uniform(0.3, max_range))
            data = self.simulate(room_dim = room_dim, mic_center = mic_center, doa_degree = doa_degree, range=ranges, signal=signal)
            out_filename = save_folder + "/{}".format(i)
            sf.write(out_filename + '.wav', data.T, 16000)
            labels.append({'fname': os.path.basename(out_filename), 'room_dim': room_dim, 'mic_center': list(mic_center), 
                           'doa_degree': doa_degree, 'range': ranges, 'file_names': file_names})
        json.dump(labels, f, indent=4)
        f.close()

class HRTF_simulator():
    def __init__(self, HRTF_folder, split):
        self.HRTF_folder = HRTF_folder
        sofas = os.listdir(self.HRTF_folder)
        if split == 'TRAIN':
            self.sofas = sofas[:40]
        else:
            self.sofas = sofas[40:]
    def simulate(self, HRTF, dataset, max_source, min_diff):
        file_names, signals, IRs,  doa_degree, ranges = [], [], [], [], []
        for _ in range(max_source):
            data_index = sample(range(len(dataset)), 1)[0]
            while 1: # make sure the doa is not too close
                m = sample(range(HRTF.Dimensions.M), 1)[0]
                IR, relative_loc_sph = access_IR(HRTF, m, 0)
                ok_flag = True
                for pre in doa_degree:
                    diff = np.abs(pre[0] - relative_loc_sph[0, 0])
                    if diff < min_diff:
                        ok_flag = False
                if ok_flag:
                    break
                
            audio, file_name = dataset[data_index]
            file_names.append(file_name)
            left = signal.convolve(audio, IR[0])
            right = signal.convolve(audio, IR[1])
            binaural_signal = np.c_[left, right]

            signals.append(binaural_signal)
            IRs.append(IR)
            doa_degree.append(relative_loc_sph[0, :2].tolist())
            ranges.append(relative_loc_sph[0, 2:].tolist())

        # binaural_signal = binaural_signal / np.max(np.abs(binaural_signal), axis=0)
        # add white noise based on SNR  
        return file_names, signals, IRs, doa_degree, ranges

    def simulate_all(self, save_folder, dataset, num_data_per_user=1000, max_source=1, min_diff=45):
        f = open(save_folder + '/label.json', 'w')
        labels = []
        for i, s in enumerate(self.sofas):
            HRTF_path = os.path.join(self.HRTF_folder, s)
            HRTF = sofa.Database.open(HRTF_path)
            for j in tqdm(range(num_data_per_user)):
                out_folder = f"{save_folder}/{i}_{j}"
                os.makedirs(out_folder, exist_ok=True)
                file_names, signals, IRs, doa_degree, ranges = self.simulate(HRTF, dataset, max_source, min_diff)
                for k, signal in enumerate(signals):
                    out_filename = f"{out_folder}/{k}.wav"
                    sf.write(out_filename, signal, 16000)
                labels.append({'fname': os.path.basename(out_folder), 'room_dim': [], 'mic_center': [], 
                               'doa_degree': doa_degree, 'range': ranges, 'file_names': file_names})   
            HRTF.close()
        json.dump(labels, f, indent=4)


    