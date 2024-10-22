'''
organize the MUSIC DOA algorithm
'''

import numpy as np
from scipy.signal import stft
from pyroomacoustics.doa import MUSIC, SRP, TOPS
import librosa
import matplotlib.pyplot as plt
import numpy as np
from .imu import pyIMU
import scipy.signal as signal
import noisereduce as nr

def init(mic_array, fs, nfft, algorithm='music'):
    kwargs = {'L': mic_array,
            'fs': fs, 
            'nfft': nfft,
            'azimuth': np.deg2rad(np.arange(180)),
            'num_src': 1
    }
    if algorithm == 'music':
        algo = MUSIC(**kwargs)
    elif algorithm == 'srp':
        algo = SRP(**kwargs)
    elif algorithm == 'tops':
        algo = TOPS(**kwargs)
    return algo

def inference(algo, data, plot=''):
    audio, imu = data
    audio = nr.reduce_noise(audio, sr=16000, stationary=True)
    print('Inferencing DOA...')
    euler = pyIMU(imu[:, :6])

    nfft = algo.nfft
    fs = algo.fs
    predictions = []
    audio_sample = audio.shape[-1]
    for i in range(0, audio_sample, 2*fs):
        audio[:, i:i+2*fs] = librosa.util.normalize(audio[:, i:i+2*fs], axis=1)

    intervals = librosa.effects.split(y=audio, top_db=30, ref=1)
    # remove too short intervals
    intervals = [interval for interval in intervals if interval[1] - interval[0] > nfft]
    # convert to nfft time dimension
    intervals = [(int(interval[0] / nfft), int(interval[1] / nfft)) for interval in intervals]

    stft_signals = stft(audio, fs=fs, nperseg=nfft, noverlap=0, boundary=None)[2] # 
    num_channels, num_freqs, num_time = stft_signals.shape
    # intervals = [[0, num_time]] 

    intervals_map = np.zeros(num_time)
    predictions = np.ones(num_time) * -1
    predictions_short = np.ones(num_time) * -1
    for interval in intervals:
        intervals_map[interval[0]:interval[1]] = 1

    for i in range(num_time):
        active_sound = intervals_map[i]
        if active_sound:
            stft_segment = stft_signals[:, :, i:i+1]
            algo.locate_sources(stft_segment)
            predictions_short[i] = np.rad2deg(algo.azimuth_recon[0])
    for interval in intervals:
        stft_segment = stft_signals[:, :, interval[0]:interval[1]]
        algo.locate_sources(stft_segment)
        predictions[interval[0]:interval[1]] = np.rad2deg(algo.azimuth_recon[0])

    if len(plot) > 0:
        print('Plotting...')
        fig, axs = plt.subplots(4, 1, figsize=(10, 10))
        axs[0].plot(audio[0])
        axs[0].plot(np.repeat(intervals_map, nfft)* 0.5)

        axs[1].plot(euler, label=['yaw', 'pitch', 'roll'])
        axs[1].legend()        

        axs[2].plot(predictions_short)
        axs[2].plot(predictions)
        
        b, a = signal.butter(2, 0.5, 'low', fs=10)
        predictions_short = signal.filtfilt(b, a, predictions_short)
        
        yaw = euler[::5, 0]
        if len(yaw) < num_time:
            yaw = np.concatenate([yaw, np.zeros(num_time - len(yaw))])
        else:
            yaw = yaw[:num_time]
        axs[3].plot(predictions_short)
        axs[3].plot(yaw * intervals_map)
        print()
        plt.savefig(f'{plot}.png')
        plt.close()

    return predictions