import numpy as np
from random import uniform, sample
from pyroomacoustics import doa, Room, ShoeBox
from dataset import TIMIT_dataset
import soundfile as sf
# constants / config
fs = 16000 
max_order = 10
snr_lb, snr_ub = 0, 30

def simulate(room_dim = [5, 5, 3], mic_center=np.c_[[2,2,1]], doa_degree=[60], range=[1], signal=[np.random.random(fs*5)]):
    room = ShoeBox(room_dim, fs=fs, max_order=max_order)
    # room simulation
    assert len(doa_degree) == len(range)
    room.add_microphone_array(mic_center +  
                             np.c_[[ 0.06,  0.0, 0.0],
                                    [ -0.06,  0.0, 0.0],
                                    #[ 0.05,  0.0, 0.0]
                                    ])
    for doa, r, s in zip(doa_degree, range, signal):
        doa_rad = np.deg2rad(doa)
        source_loc = mic_center[:,0] + np.c_[r*np.cos(doa_rad), r*np.sin(doa_rad), 0][0]    
        room.add_source(source_loc, signal=s)
    room.simulate(snr=uniform(snr_lb, snr_ub))
    signals = room.mic_array.signals
    return signals
if __name__ == "__main__":
    import json
    import os
    from tqdm import tqdm
    split = 'TEST'
    dataset = TIMIT_dataset(split)
    save_folder = 'TIMIT/pra/' + split
    os.makedirs(save_folder, exist_ok=True)
    max_source = 1
    offset = 0.5
    room_dims = [[5, 5, 3], [10, 10, 3], [20, 20, 3], [40, 20, 3]]
    mic_center = np.c_[[2,2,1]]
    f = open(save_folder + '/label.json', 'w')
    labels = []
    for i in tqdm(range(1000)):
        room_dim = sample(room_dims, 1)[0]
        mic_center = np.c_[[uniform(0 + offset, room_dim[0] - offset), uniform(0 + offset, room_dim[1] - offset), uniform(1.5, 1.8)]]
        max_range = min(room_dim[0]-mic_center[0, 0], mic_center[0, 0], room_dim[1]-mic_center[1, 0], mic_center[1, 0])
        num_source = sample(range(1, max_source + 1), 1)[0]
        sig_index = sample(range(len(dataset)), num_source)
        signal, doa_degree, ranges = [], [], []
        for idx in sig_index:
            signal.append(dataset[idx])
            doa_degree.append(uniform(0, 180))
            ranges.append(uniform(0.3, max_range))
        data = simulate(room_dim = room_dim, mic_center = mic_center, doa_degree = doa_degree, range=ranges, signal=signal)
        # print(num_source, max_range, doa_degree, ranges)
        out_filename = save_folder + "/{}.wav".format(i)
        sf.write(out_filename, data.T, 16000)
        labels.append({'fname': out_filename, 'room_dim': room_dim, 'mic_center': list(mic_center[:, 0]), 'doa_degree': doa_degree, 'range': ranges})
        #print(out_filename)
    json.dump(labels, f, indent=4)
    f.close()
