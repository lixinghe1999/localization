import numpy as np
from random import uniform, sample
from pyroomacoustics import doa, Room, ShoeBox
import soundfile as sf
import json
import os
from tqdm import tqdm
# constants / config
fs = 16000 
max_order = 10
snr_lb, snr_ub = 0, 30
offset = 0.5
mic_array =  np.c_[[ 0.06,  0.0, 0.0],
                   [ -0.06,  0.0, 0.0],
                    [ 0.03,  0.06, 0.0],
                    [ 0.03,  -0.06, 0.0],
                    [ -0.03,  0.06, 0.0],
                    [ -0.03,  -0.06, 0.0],]
room_dims = [[5, 5, 3], [10, 10, 3], [20, 20, 3], [40, 20, 3]]

def simulate_all(save_folder, dataset, num_data=1000, max_source=1, min_diff=45):
    f = open(save_folder + '/label.json', 'w')
    labels = []
    for i in tqdm(range(num_data)):
        room_dim = sample(room_dims, 1)[0]
        mic_center = np.array([uniform(0 + offset, room_dim[0] - offset), uniform(0 + offset, room_dim[1] - offset), uniform(1.5, 1.8)])
        max_range = min(room_dim[0]-mic_center[0], mic_center[0], room_dim[1]-mic_center[1], mic_center[1])
        num_source = sample(range(1, max_source + 1), 1)[0]
        sig_index = sample(range(len(dataset)), num_source)
        signal, doa_degree, ranges = [], [], []
        for idx in sig_index:
            signal.append(dataset[idx])
            while 1:
                random_doa = uniform(0, 360)
                for doa in doa_degree:
                    diff = np.abs(doa - random_doa)
                    if diff < min_diff:
                        continue
                break
            doa_degree.append(random_doa)
            ranges.append(uniform(0.3, max_range))
        data = simulate(room_dim = room_dim, mic_center = mic_center, doa_degree = doa_degree, range=ranges, signal=signal)
        out_filename = save_folder + "/{}".format(i)
        sf.write(out_filename + '.wav', data.T, 16000)
        labels.append({'fname': out_filename, 'room_dim': room_dim, 'mic_center': list(mic_center), 'doa_degree': doa_degree, 'range': ranges})
    json.dump(labels, f, indent=4)
    f.close()
def simulate(room_dim = [5, 5, 3], mic_center=np.array([2,2,1.5]), doa_degree=[60], range=[1], signal=[np.random.random(fs*5)]):
    room = ShoeBox(room_dim, fs=fs, max_order=max_order)
    assert len(doa_degree) == len(range)
    room.add_microphone_array(mic_center[:, np.newaxis] + mic_array)
    for doa, r, s in zip(doa_degree, range, signal):
        doa_rad = np.deg2rad(doa)
        if type(doa) == list: #both azimuth + elevation
            source_loc = mic_center + np.array([r[0]*np.cos(doa_rad[0]), r[0]*np.sin(doa_rad[0]), r[0]*np.sin(doa_rad[1])])
        else:
            source_loc = mic_center + np.array([r*np.cos(doa_rad), r*np.sin(doa_rad), 0])   
    
        room.add_source(source_loc, signal=s)
    room.simulate(snr=uniform(snr_lb, snr_ub))
    signals = room.mic_array.signals
    return signals

