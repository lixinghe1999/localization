import librosa
import numpy as np
import math

# target output is 0.1s window, we expect the input to be 0.02s window (5 times downsampling)
_nfft = 640
_win_len = 640
_hop_len = 320
_nb_mel_bins = 64

def nCr(n, r):
    return math.factorial(n) // math.factorial(r) // math.factorial(n-r)

def spectrogram(audio_input):
    '''
    input: audio_input: (nb_channels, nb_samples)
    output: (nb_frames, nb_bins, _nb_frames)
    '''
    _nb_ch, _nb_sampels = audio_input.shape
    _nb_frames = _nb_sampels // _hop_len
    nb_bins = _nfft // 2
    spectra = []
    for ch_cnt in range(_nb_ch):
        stft_ch = librosa.core.stft(np.asfortranarray(audio_input[ch_cnt, :]), n_fft=_nfft, hop_length=_hop_len,
                                    win_length=_win_len, window='hann')
        spectra.append(stft_ch[:, :_nb_frames])
    return np.array(spectra).T

def gcc_mel_spec(linear_spectra):
    '''
    input: linear_spectra: (nb_frames, nb_bins, nb_channels)
    '''
    gcc_channels = nCr(linear_spectra.shape[-1], 2)
    gcc_feat = np.zeros((linear_spectra.shape[0], _nb_mel_bins, gcc_channels))
    cnt = 0
    for m in range(linear_spectra.shape[-1]):
        for n in range(m+1, linear_spectra.shape[-1]):
            R = np.conj(linear_spectra[:, :, m]) * linear_spectra[:, :, n]
            cc = np.fft.irfft(np.exp(1.j*np.angle(R)))
            cc = np.concatenate((cc[:, -_nb_mel_bins//2:], cc[:, :_nb_mel_bins//2]), axis=-1)
            gcc_feat[:, :, cnt] = cc
            cnt += 1
    
    log_mel_spec = librosa.feature.melspectrogram(S=np.abs(linear_spectra)**2, n_mels=_nb_mel_bins)
    feat = np.concatenate((log_mel_spec, gcc_feat), axis=-1).transpose(2, 0, 1)
    return feat