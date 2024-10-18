import os
import librosa
import noisereduce as nr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

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

def get_sound_timestamps(audio):
    frame_length = int(sr * 0.5)
    audio_frame = librosa.util.frame(x=audio, frame_length=frame_length, hop_length=frame_length, axis=0)
    print('audio frame shape:', audio_frame.shape)
    energy = np.mean(audio_frame ** 2, axis=1)
    energy = energy / np.max(energy)
    active_frame = energy > 0.02
    print('active frame:', np.sum(active_frame), 'total frame:', len(active_frame))

    # convert to timestamp: [[start, end], [...]]
    sound_timestamps = []
    start = 0
    for i in range(1, len(active_frame)):
        if active_frame[i] and not active_frame[i-1]:
            start = i
        if not active_frame[i] and active_frame[i-1]:
            end = i
            sound_timestamps.append({'start': start, 'end': end})
    return sound_timestamps

def pcm_to_wav(pcm_file, wav_file, sample_rate=48000, num_channels=1):
    import scipy.io.wavfile as wavfile
    # Load PCM data
    with open(pcm_file, 'rb') as f:
        pcm_data = f.read()
    
    # Convert PCM data to numpy array
    # Assuming 16-bit PCM for this example
    num_samples = len(pcm_data) // 2  # 2 bytes per sample for 16-bit PCM
    pcm_array = np.frombuffer(pcm_data, dtype=np.int16)

    # Reshape if stereo
    if num_channels > 1:
        pcm_array = pcm_array.reshape((-1, num_channels))

    # Write to WAV file
    wavfile.write(wav_file, sample_rate, pcm_array)

datadir = 'dataset/earphone/20241017'
audio_dir = os.path.join(datadir, 'audio')
meta_dir = os.path.join(datadir, 'meta')

model = load_silero_vad()
full_files = os.listdir(audio_dir)

audio_files = [audio_file for audio_file in full_files if audio_file.endswith('.pcm')]
imu_files = [imu_file for imu_file in full_files if imu_file.endswith('.csv')]
audio_files.sort()
imu_files.sort()
assert len(audio_files) == len(imu_files)
for audio_file, imu_file in zip(audio_files, imu_files):
    print('processing:', audio_file, imu_files)

    imu = np.loadtxt(os.path.join(audio_dir, imu_file), delimiter=',', skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6,))
    imu_timestamp = np.loadtxt(os.path.join(audio_dir, imu_file), delimiter=',', skiprows=1, usecols=(7,), dtype=str)
    imu_timestamp = [datetime.datetime.strptime(time, '%Y%m%d_%H%M%S_%f') for time in imu_timestamp]
    # set the start time to 0 and convert to seconds
    imu_timestamp = [(time - imu_timestamp[0]).total_seconds() for time in imu_timestamp]
    imu_sr = len(imu_timestamp)/imu_timestamp[-1]

    imu = librosa.resample(imu[:, :6].T, orig_sr=imu_sr, target_sr=50).T
    np.save(os.path.join(audio_dir, audio_file.replace('.pcm', '.npy')), imu)

    pcm_path = os.path.join(audio_dir, audio_file)
    # convert pcm to wav
    audio_path = pcm_path.replace('.pcm', '.wav')
    pcm_to_wav(pcm_path, audio_path, sample_rate=48000, num_channels=2)

    meta_path = os.path.join(meta_dir, audio_file.replace('.pcm', '.csv'))
    meta_info, time_str = audio_file.split('-')
    subject, distance, direction, place, content = meta_info.split('_')

    audio, sr = librosa.load(audio_path, sr=16000)

    print(imu_timestamp[-1], len(audio) / sr, sr)

    audio = audio / np.max(np.abs(audio))
    reduced_noise = nr.reduce_noise(y=audio, sr=sr)

    # speech_timestamps = get_speech_timestamps(reduced_noise, model)
    sound_timestamps = get_sound_timestamps(reduced_noise)

    meta_data = {'sound_event_recording': [],'start_time': [],'end_time': [], 'azi': [], 'ele': [], 'dist': []}
    for speech_timestamp in sound_timestamps:
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

    

