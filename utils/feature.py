'''
Multi-channel audio may need to extract the features before
'''
import numpy as np
import librosa
import noisereduce as nr
import matplotlib.pyplot as plt

def gccphat(audio, gcc_phat_len=32):
    N_channel = audio.shape[0]
    gccs = []
    for i in range(N_channel):
        for j in range(i+1, N_channel):
            audio1 = audio[i]
            audio2 = audio[j]
            n = audio1.shape[0] + audio2.shape[0] 
            X = np.fft.rfft(audio1, n=n)
            Y = np.fft.rfft(audio2, n=n)
            R = X * np.conj(Y)
            cc = np.fft.irfft(R / (1e-6 + np.abs(R)),  n=n)
            cc = np.concatenate((cc[-gcc_phat_len:], cc[:gcc_phat_len+1])).astype(np.float32)
            gccs.append(cc)
    output = np.array(gccs).reshape(-1)
    return output

def mel_spec(audio, n_fft=1024, n_mels=64):
    mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_fft=n_fft, n_mels=n_mels)
    mel = librosa.power_to_db(mel, ref=np.max)
    return mel

def wave_form(audio):
    return audio

def noise_reduce(audio):
    return nr.reduce_noise(y=audio, sr=16000)


def source_separation(audio, fs, n_fft=512, hop_length=16):
    '''
    reference: https://proceedings.mlr.press/v162/xu22b/xu22b.pdf, 'Learning to Separate Voices by Spatial Regions'
    '''
    F_aliasing = 1000
    l_audio, r_audio = audio[0], audio[1] 
    l_stft = librosa.stft(l_audio, n_fft=n_fft, hop_length=hop_length)[1:, :-1]
    r_stft = librosa.stft(r_audio, n_fft=n_fft, hop_length=hop_length)[1:, :-1]
    phase_diff = np.angle(l_stft) - np.angle(r_stft)

    freq_bin = np.linspace(0, fs/2, n_fft//2 + 1)[1:, np.newaxis]
    ITD = phase_diff / (2 * np.pi * freq_bin + 1e-6)
    ILD = np.abs(l_stft) / (np.abs(r_stft)+ 1e-6)
    feature_L = np.concatenate((ITD, ILD), axis=0).astype(np.float32)
    feature_R = np.concatenate((ITD, ILD), axis=0).astype(np.float32)
    return {'L': feature_L, 'R': feature_R, 'Binaural': audio}

    aliasing_index = np.argmin(np.abs(freq_bin - F_aliasing))
    print(aliasing_index, fs, F_aliasing)
    ITD_no_aliasing = ITD[1:aliasing_index]
    ITD_aliasing = ITD[aliasing_index:]

    ILD_no_aliasing = ILD[1:aliasing_index]
    ILD_aliasing = ILD[aliasing_index:]

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].imshow(ITD_no_aliasing, aspect='auto', )
    axs[1].imshow(ILD_no_aliasing, aspect='auto')

    axs[2].hist(ITD_no_aliasing.flatten(), bins=1000)
    axs[2].set_xlabel('ITD')
    
    # fit ITD_no_aliasing with guassian distribution
    peak = ITD_no_aliasing.mean()
    std = ITD_no_aliasing.std()
    print(peak, std)

    plt.savefig('figs/source_separation.png')

    
