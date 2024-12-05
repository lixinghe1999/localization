'''
organize the MUSIC DOA algorithm
'''

import numpy as np
from scipy.signal import stft
from pyroomacoustics.doa import MUSIC, SRP, TOPS
import librosa
import numpy as np

def doa_inference(audio, mic_array, fs, nfft, algorithm, mode='window'):
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

    stft_signals = stft(audio, fs=fs, nperseg=nfft, noverlap=0, boundary=None)[2]
    num_time = stft_signals.shape[-1]
    predictions = np.zeros((num_time))
    if mode == 'active':
        intervals = librosa.effects.split(y=audio, top_db=30, ref=1)
        intervals = [interval for interval in intervals if interval[1] - interval[0] > nfft]
        intervals = [(int(interval[0] / nfft), int(interval[1] / nfft)) for interval in intervals]
        for interval in intervals:
            stft_segment = stft_signals[:, :, interval[0]:interval[1]]
            algo.locate_sources(stft_segment)
            azimuth = np.rad2deg(algo.azimuth_recon[0])
            predictions[interval[0]:interval[1]] = azimuth
    elif mode == 'window':
        for i in range(num_time):
            stft_segment = stft_signals[:, :, i:i+1]
            algo.locate_sources(stft_segment)
            azimuth = np.rad2deg(algo.azimuth_recon[0])
            predictions[i] = azimuth
    return predictions