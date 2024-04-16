import numpy as np
import pandas as pd
from scipy.signal import stft
from pyroomacoustics import doa
from dataset import TIMIT_dataset, Main_dataset
from pyroomacoustic_simulate import simulate
def output(data, mic_center):
    nfft = 1024
    n_frames = 30
    fs = 16000
    kwargs = {'L': mic_center +  np.c_[[ 0.06,  0.0, 0.0],
                                    [ -0.06,  0.0, 0.0],
                                    # [ 0.05,  0.0, 0.0]
                                    ],
            'fs': fs, 
            'nfft': nfft,
            'azimuth': np.deg2rad(np.arange(180)),
            'num_src': 1
    }
    algorithms = {
        'MUSIC': doa.music.MUSIC(**kwargs),
        'NormMUSIC': doa.normmusic.NormMUSIC(**kwargs),

    }

    predictions = {}
    stft_signals = stft(data[:,fs:fs+n_frames*nfft], fs=fs, nperseg=nfft, noverlap=0, boundary=None)[2]
    for algo_name, algo in algorithms.items():
        algo.locate_sources(stft_signals)
        predictions[algo_name] = np.rad2deg(algo.azimuth_recon[0])
    return predictions
def calc_ae(a,b):
    x = np.abs(a-b)
    return np.min(np.array((x, np.abs(360-x))), axis=0)

if __name__ == "__main__":
    # mic_center = np.c_[[2,2,1]]
    # dataset = TIMIT_dataset('TRAIN', sr=16000)
    # signal = dataset[0]
    # data = simulate(mic_center = mic_center, doa_degree = [60], range=[1], signal=[signal])
    # print(data.shape)
    # pred = output(data, mic_center)
    # print(pred)
    # for algo, pred in pred.items():
    #     error = calc_ae(60, pred)
    #     print(f"Error of {algo}: {error}")

    dataset = Main_dataset('TIMIT/pra', 'TRAIN')
    for i in range(len(dataset)):
        signal, label = dataset.get_raw(0)
        print(signal.shape, label)
        preds = output(signal, np.c_[[2,2,1]])
        for algo, pred in preds.items():
            error = calc_ae(label['doa_degree'], pred)
            print(f"Error of {algo}: {error}")
        break