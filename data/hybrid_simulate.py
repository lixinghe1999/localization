import numpy as np
import sofa
import os
from tqdm import tqdm
import soundfile as sf
import json
from hrtf_simulate import random_simulate
from ism_simulate import simulate

def simulate_all(HRTF_folder, split, save_folder, dataset, num_data_per_user=1000, max_source=1, min_diff=45):
    f = open(save_folder + '/label.json', 'w')
    sofas = os.listdir(HRTF_folder)
    if split == 'TRAIN':
        sofas = sofas[:40]
    else:
        sofas = sofas[40:]
    labels = []
    for i, s in enumerate(sofas):
        HRTF_path = os.path.join(HRTF_folder, s)
        HRTF = sofa.Database.open(HRTF_path)
        for j in tqdm(range(num_data_per_user)):
            out_filename = f"{save_folder}/{i}_{j}"
            raw_signal, binaural_signal, IRs, doa_degree, ranges = random_simulate(HRTF, dataset, max_source, min_diff)
            sf.write(out_filename + '.wav', binaural_signal, 16000)

            array_signal = simulate(doa_degree=doa_degree, range=ranges, signal=raw_signal)
            sf.write(out_filename + '_array.wav', array_signal.T, 16000)

            out_filename = f"{save_folder}/{i}_{j}"
            np.save(out_filename + '.npy', IRs)
            labels.append({'fname': out_filename, 'room_dim': [], 'mic_center': [], 'doa_degree': doa_degree, 'range': ranges})
        HRTF.close()
        # break
    json.dump(labels, f, indent=4)
    f.close()

    