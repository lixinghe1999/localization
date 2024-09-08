'''
organize the MUSIC DOA algorithm
'''

import numpy as np
from scipy.signal import stft
from pyroomacoustics.doa import MUSIC, SRP, TOPS
import librosa
import matplotlib.pyplot as plt

def init(mic_array, fs, nfft, mic_center=0):
    kwargs = {'L': mic_center + mic_array,
            'fs': fs, 
            'nfft': nfft,
            'azimuth': np.deg2rad(np.arange(180)),
            'num_src': 1
    }
    # algo = MUSIC(**kwargs)
    algo = SRP(**kwargs)
    # algo = TOPS(**kwargs)
    return algo

def pra_doa(audio, mic_array, fs, nfft, intervals=None, plot=False):
    algo = init(mic_array, fs, nfft, mic_center=0)
    nfft = algo.nfft
    fs = algo.fs
    predictions = []
    if intervals is None:
        intervals = librosa.effects.split(y=audio, top_db=35, ref=1)
        # remove too short intervals
        intervals = [interval for interval in intervals if interval[1] - interval[0] > nfft]
    n_windows = np.shape(intervals)[0]
    for i in range(n_windows):
        start = intervals[i][0]
        end = intervals[i][1]
        data = audio[:, start:end]
        # detect voice activity
        stft_signals = stft(data, fs=fs, nperseg=nfft, noverlap=0, boundary=None)[2]
        # M, F, T = stft_signals.shape
        # for T in range(0, T, 10):
        #     stft_signal = stft_signals[:, :, T:T+10]
        #     algo.locate_sources(stft_signal)
        #     predictions.append(np.rad2deg(algo.azimuth_recon[0]))
        algo.locate_sources(stft_signals)
        predictions.append(np.rad2deg(algo.azimuth_recon[0]))
    predictions = np.array(predictions)

    if plot:
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        axs[0].plot(audio[0])
        for prediction, interval in zip(predictions, intervals):
            line_x = np.arange(interval[0], interval[1]) / fs
            line_y = np.ones_like(line_x) * prediction
            axs[1].plot(line_x, line_y, 'r')
        axs[1].set_xlim(0, audio.shape[-1] / fs)
        plt.savefig('doa.png')
        plt.close()
    return predictions, intervals