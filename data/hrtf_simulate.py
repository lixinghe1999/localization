import matplotlib.pyplot as plt
import numpy as np
import sofa
import os
from random import uniform, sample 
import scipy.signal as signal
from tqdm import tqdm
import soundfile as sf
import json
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

def random_simulate(HRTF, dataset, max_source, min_diff):
    num_source = sample(range(1, max_source + 1), 1)[0]
    raw_signals, signals, IRs,  doa_degree, ranges = [], [], [], [], []
    for _ in range(num_source):
        data_index = sample(range(len(dataset)), 1)[0]
        while 1:
            m = sample(range(HRTF.Dimensions.M), 1)[0]
            IR, relative_loc_sph = access_IR(HRTF, m, 0)
            ok_flag = True
            for pre in doa_degree:
                diff = np.abs(pre[0] - relative_loc_sph[0, 0])
                if diff < min_diff:
                    ok_flag = False
            if ok_flag:
                break
            
        raw_signal = dataset[data_index]
        raw_signals.append(raw_signal)
        left = signal.convolve(raw_signal, IR[0])
        right = signal.convolve(raw_signal, IR[1])
        binaural_signal = np.c_[left, right]

        signals.append(binaural_signal)
        IRs.append(IR)
        doa_degree.append(relative_loc_sph[0, :2].tolist())
        ranges.append(relative_loc_sph[0, 2:].tolist())
    max_length = 0
    for sig in signals:
        max_length = max(max_length, sig.shape[0])
    binaural_signal = np.zeros((max_length, 2))
    for sig in signals:
        binaural_signal[:sig.shape[0], :] += sig
    # normalize 
    binaural_signal = binaural_signal / np.max(np.abs(binaural_signal), axis=0)
    return raw_signals, binaural_signal, IRs, doa_degree, ranges
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
            raw_signal, binaural_signal, IRs, doa_degree, ranges = random_simulate(HRTF, dataset, max_source, min_diff)
            out_filename = f"{save_folder}/{i}_{j}"
            sf.write(out_filename + '.wav', binaural_signal, 16000)

            out_filename = f"{save_folder}/{i}_{j}"
            np.save(out_filename + '.npy', IRs)
            labels.append({'fname': out_filename, 'room_dim': [], 'mic_center': [], 'doa_degree': doa_degree, 'range': ranges})
        HRTF.close()
        # break
    json.dump(labels, f, indent=4)
    f.close()

    