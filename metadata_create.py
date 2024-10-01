import os
import librosa
import noisereduce as nr
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import pandas as pd

from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

def azimuth_parser(azimuth):
    azimuth = int(azimuth)
    if azimuth == 0:
        return 0
    elif azimuth == 1:
        return 90
    elif azimuth == 2:
        return -90
    else:
        return 180
    
def distance_parser(distance):
    distance = int(distance)
    if distance == 0:
        return 0.5
    elif distance == 1:
        return 1.5
    else:
        return 3


datadir = 'dataset/earphone/20240927'
audio_dir = os.path.join(datadir, 'audio')
meta_dir = os.path.join(datadir, 'meta')

model = load_silero_vad()
for audio_file in os.listdir(audio_dir):
    audio_path = os.path.join(audio_dir, audio_file)
    meta_path = os.path.join(meta_dir, audio_file.replace('.wav', '.csv'))
    meta_info, time_str = audio_file.split('-')
    subject, distance, direction, place, content = meta_info.split('_')

    audio, sr = librosa.load(audio_path, sr=16000)
    audio = audio / np.max(np.abs(audio))
    reduced_noise = nr.reduce_noise(y=audio, sr=sr)

    speech_timestamps = get_speech_timestamps(reduced_noise, model)

    meta_data = {'sound_event_recording': [],'start_time': [],'end_time': [], 'azi': [], 'ele': [], 'dist': []}
    for speech_timestamp in speech_timestamps:
        start = speech_timestamp['start']
        end = speech_timestamp['end']

        start_sec = start / sr
        end_sec = end / sr
        meta_data['sound_event_recording'].append('speech')
        meta_data['start_time'].append(start_sec)
        meta_data['end_time'].append(end_sec)

        azi = azimuth_parser(direction)
        dist = distance_parser(distance)

        meta_data['azi'].append(azi)
        meta_data['ele'].append(0)
        meta_data['dist'].append(dist)
    meta_data = pd.DataFrame(meta_data)
    meta_data.to_csv(meta_path, index=False)

    # frame_length = int(sr * 0.5)
    # audio_frame = librosa.util.frame(x=reduced_noise, frame_length=frame_length, hop_length=frame_length, axis=0)
    # print('frame shape:', audio_frame.shape)
    # energy = np.mean(audio_frame ** 2, axis=1)
    # energy = energy / np.max(energy)
    # active_frame = energy > 0.05
    # print('active frame:', np.sum(active_frame), 'total frame:', len(active_frame))

    # active_frame = np.zeros_like(reduced_noise)
    # plt.plot(reduced_noise)
    # for timestamp in speech_timestamps:
    #     start = timestamp['start']
    #     end = timestamp['end']
    #     active_frame[start:end] = 1
    # plt.plot(active_frame)

    # plt.savefig('test.png')
    # sf.write('test.wav', reduced_noise, sr)

