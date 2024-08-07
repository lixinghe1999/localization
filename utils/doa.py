import numpy as np
from scipy.signal import stft
from pyroomacoustics.doa import MUSIC
import librosa
import matplotlib.pyplot as plt

def init_music(mic_center, mic_array, fs, nfft):
    kwargs = {'L': mic_center + mic_array,
            'fs': fs, 
            'nfft': nfft,
            'azimuth': np.deg2rad(np.arange(360)),
            'num_src': 1
    }
    algo = MUSIC(**kwargs)
    return algo

def inference_music(algo, audio, intervals=None, plot=False):
    nfft = algo.nfft
    fs = algo.fs
    predictions = []
    if intervals is None:
        intervals = librosa.effects.split(y=audio, top_db=30, ref=1)
        # remove too short intervals
        intervals = [interval for interval in intervals if interval[1] - interval[0] > nfft]
    n_windows = np.shape(intervals)[0]
    if plot:    
        fig, axs = plt.subplots(n_windows, 1, figsize=(10, 10))
    for i in range(n_windows):
        start = intervals[i][0]
        end = intervals[i][1]
        data = audio[:, start:end]
        # detect voice activity
        stft_signals = stft(data, fs=fs, nperseg=nfft, noverlap=0, boundary=None)[2]
        algo.locate_sources(stft_signals)
        predictions.append(np.rad2deg(algo.azimuth_recon[0]))
        if plot:
            axs[i].plot(data[0])
            axs[i].set_title('Predicted angle: {}'.format(predictions[-1]))
    predictions = np.array(predictions)

    if plot:
        plt.savefig('music.png')
        plt.close()
    return predictions, intervals