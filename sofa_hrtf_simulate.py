import matplotlib.pyplot as plt
import numpy as np
import sofa
from torch_dataset import TIMIT_dataset
import os
from random import uniform, sample 
import scipy.signal as signal
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
def save_hrtf(sofas):
    labels = []
    for i, s in enumerate(sofas):
        HRTF_path = os.path.join(HRTF_folder, s)
        HRTF = sofa.Database.open(HRTF_path)
        print(HRTF_path, 'receiver {}, Emitter {}, Measurement {}'.format(HRTF.Dimensions.R, HRTF.Dimensions.E, HRTF.Dimensions.M))
        num_measurements = HRTF.Dimensions.M
        for j in tqdm(range(num_measurements)):
            doa_degree, ranges = [], []
            IR, relative_loc_sph = access_IR(HRTF, j, 0)
            out_filename = f"{save_folder}/{i}_{j}_IR.npy"
            np.save(out_filename, IR)            
            doa_degree.append(relative_loc_sph[0, :2].tolist())
            ranges.append(relative_loc_sph[0, 2:].tolist())
        
            labels.append({'fname': out_filename, 'room_dim': [], 'mic_center': [], 'doa_degree': doa_degree, 'range': ranges})
        HRTF.close()
    json.dump(labels, f, indent=4)
    f.close()
def simulate_all(sofas):
    labels = []
    for i, s in enumerate(sofas):
        HRTF_path = os.path.join(HRTF_folder, s)
        HRTF = sofa.Database.open(HRTF_path)
        print(HRTF_path, 'receiver {}, Emitter {}, Measurement {}'.format(HRTF.Dimensions.R, HRTF.Dimensions.E, HRTF.Dimensions.M))
        # num_measurements = 200
        num_measurements = HRTF.Dimensions.M
        for j in tqdm(range(num_measurements)):
            num_source = sample(range(1, max_source + 1), 1)[0]
            signals, doa_degree, ranges = [], [], []
            for _ in range(num_source):
                data_index = sample(range(len(dataset)), 1)[0]
                while 1:
                    # m = sample(range(HRTF.Dimensions.M), 1)[0]
                    m = j
                    IR, relative_loc_sph = access_IR(HRTF, m, 0)
                    ok_flag = True
                    for pre in doa_degree:
                        diff = np.abs(pre[0] - relative_loc_sph[0, 0])
                        if diff < min_diff:
                            ok_flag = False
                    if ok_flag:
                        break
                    
                raw_signal = dataset[data_index]
                left = signal.convolve(raw_signal, IR[0])
                right = signal.convolve(raw_signal, IR[1])
                binaural_signal = np.c_[left, right]

                signals.append(binaural_signal)
                doa_degree.append(relative_loc_sph[0, :2].tolist())
                ranges.append(relative_loc_sph[0, 2:].tolist())
            max_length = 0
            for sig in signals:
                max_length = max(max_length, sig.shape[0])
            binaural_signal = np.zeros((max_length, 2))
            for sig in signals:
                binaural_signal[:sig.shape[0], :] += sig

            out_filename = f"{save_folder}/{i}_{j}.wav"
            sf.write(out_filename, binaural_signal, 16000)
            labels.append({'fname': out_filename, 'room_dim': [], 'mic_center': [], 'doa_degree': doa_degree, 'range': ranges})
        HRTF.close()
        # break
    json.dump(labels, f, indent=4)
    f.close()
  
if __name__ == "__main__":
    import json
    from tqdm import tqdm
    import soundfile as sf
    split = 'TRAIN'
    dataset = TIMIT_dataset(split)
    max_source = 1
    min_diff = 45 # minimum difference between two sources (in degree)
    
    HRTF_folder = "HRTF-Database/SOFA"
    sofas = os.listdir(HRTF_folder)
    if split == 'TRAIN':
        sofas = sofas[:40]
    else:
        sofas = sofas[40:]

    # save_folder = 'TIMIT/HRTF_{}/'.format(max_source) + split
    # os.makedirs(save_folder, exist_ok=True)
    # f = open(save_folder + '/label.json', 'w')
    # simulate_all(sofas)

    save_folder = 'RAW_HRTF/' + split
    os.makedirs(save_folder, exist_ok=True)
    f = open(save_folder + '/label.json', 'w')
    save_hrtf(sofas)


    