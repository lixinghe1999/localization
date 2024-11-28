'''
organize the MUSIC DOA algorithm
'''

import numpy as np
from scipy.signal import stft
from pyroomacoustics.doa import MUSIC, SRP, TOPS
import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import noisereduce as nr

def init(mic_array, fs, nfft, algorithm='music'):
    kwargs = {'L': mic_array,
            'fs': fs, 
            'nfft': nfft,
            'azimuth': np.deg2rad(np.arange(360)),
            'num_src': 1
    }
    if algorithm == 'music':
        algo = MUSIC(**kwargs)
    elif algorithm == 'srp':
        algo = SRP(**kwargs)
    elif algorithm == 'tops':
        algo = TOPS(**kwargs)
    return algo

def inference(algo, data):
    audio = data
    nfft = algo.nfft
    fs = algo.fs
    predictions = []

    intervals = librosa.effects.split(y=audio, top_db=30, ref=1)
    print(intervals)
    # remove too short intervals
    intervals = [interval for interval in intervals if interval[1] - interval[0] > nfft]
    # convert to nfft time dimension
    intervals = [(int(interval[0] / nfft), int(interval[1] / nfft)) for interval in intervals]

    stft_signals = stft(audio, fs=fs, nperseg=nfft, noverlap=0, boundary=None)[2] # 
    num_channels, num_freqs, num_time = stft_signals.shape
    # intervals = [[0, num_time]] 

    predictions = np.zeros((num_time, 3))
    # predictions_short = np.ones(num_time) * -1
    # for interval in intervals:
    #     intervals_map[interval[0]:interval[1]] = 1

    # for i in range(num_time):
    #     active_sound = intervals_map[i]
    #     if active_sound:
    #         stft_segment = stft_signals[:, :, i:i+1]
    #         algo.locate_sources(stft_segment)
    #         predictions_short[i] = np.rad2deg(algo.azimuth_recon[0])
    predictions = []
    for interval in intervals:
        stft_segment = stft_signals[:, :, interval[0]:interval[1]]
        algo.locate_sources(stft_segment)
        azimuth = np.rad2deg(algo.azimuth_recon[0])
        # xyz = np.array([np.cos(np.deg2rad(azimuth)), np.sin(np.deg2rad(azimuth)), 0])
        predictions.append([azimuth, interval[0], interval[1]])
    return predictions