'''
Multi-channel feature
'''
import numpy as np
import librosa
import noisereduce as nr
import matplotlib.pyplot as plt
import os
import torch
SPEED_OF_SOUND = 343.2

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
    feature= np.concatenate((ITD, ILD), axis=0).astype(np.float32)
    return {'feature': feature, 'Binaural': audio}

def shift_mixture(input_data, target_position, mic_radius, sr, inverse=False):
    """
    Shifts the input according to the voice position. This
    lines up the voice samples in the time domain coming from a target_angle
    Args:
        input_data - M x T numpy array or torch tensor
        target_position - The location where the data should be aligned
        mic_radius - In meters. The number of mics is inferred from
            the input_Data
        sr - Sample Rate in samples/sec
        inverse - Whether to align or undo a previous alignment

    Returns: shifted data and a list of the shifts
    """
    # elevation_angle = 0.0 * np.pi / 180
    # target_height = 3.0 * np.tan(elevation_angle)
    # target_position = np.append(target_position, target_height)

    num_channels = input_data.shape[0]

    # Must match exactly the generated or captured data
    mic_array = [[
        mic_radius * np.cos(2 * np.pi / num_channels * i),
        mic_radius * np.sin(2 * np.pi / num_channels * i),
    ] for i in range(num_channels)]

    # Mic 0 is the canonical position
    distance_mic0 = np.linalg.norm(mic_array[0] - target_position)
    shifts = [0]

    # Check if numpy or torch
    if isinstance(input_data, np.ndarray):
        shift_fn = np.roll
    elif isinstance(input_data, torch.Tensor):
        shift_fn = torch.roll
    else:
        raise TypeError("Unknown input data type: {}".format(type(input_data)))

    # Shift each channel of the mixture to align with mic0
    for channel_idx in range(1, num_channels):
        distance = np.linalg.norm(mic_array[channel_idx] - target_position)
        distance_diff = distance - distance_mic0
        shift_time = distance_diff / SPEED_OF_SOUND
        shift_samples = int(round(sr * shift_time))
        if inverse:
            input_data[channel_idx] = shift_fn(input_data[channel_idx],
                                               shift_samples)
        else:
            input_data[channel_idx] = shift_fn(input_data[channel_idx],
                                               -shift_samples)
        shifts.append(shift_samples)

    return input_data, shifts



    
