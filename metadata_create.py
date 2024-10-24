import os
import librosa
import noisereduce as nr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import scipy.signal as signal

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
    return pcm_array

datadir = 'dataset/earphone/20241023'
data_dir = os.path.join(datadir, 'data')
log_dir = os.path.join(datadir, 'log')

audio_dir = os.path.join(datadir, 'audio')
imu_dir = os.path.join(datadir, 'imu')
meta_dir = os.path.join(datadir, 'meta')
os.makedirs(audio_dir, exist_ok=True); os.makedirs(imu_dir, exist_ok=True); os.makedirs(meta_dir, exist_ok=True)

datas = os.listdir(data_dir)
audios = [data for data in datas if data.endswith('.pcm')]; audios.sort()
imus = [data for data in datas if data.endswith('.csv')]; imus.sort()
logs = os.listdir(log_dir); logs.sort()


def get_chirp(sample_rate=44100, duration=5.0, min_freq=100, max_freq=8000):
    import numpy as np
    from scipy.io import wavfile
    t = np.linspace(0, duration, int(duration * sample_rate), False)
    chirp = np.sin(2 * np.pi * (min_freq + (max_freq - min_freq) * t / duration) * t)
    return chirp

chirp_template = get_chirp(sample_rate=48000, duration=1, min_freq=2000, max_freq=4000)

for audio, imu, log in zip(audios, imus, logs):
    print('processing:', audio, imu, log)
    imu_data = np.loadtxt(os.path.join(data_dir, imu), delimiter=',', skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6,))
    imu_timestamp = np.loadtxt(os.path.join(data_dir, imu), delimiter=',', skiprows=1, usecols=(7,), dtype=str)
    imu_timestamp = [datetime.datetime.strptime(time, '%Y%m%d_%H%M%S_%f') for time in imu_timestamp]
    # set the start time to 0 and convert to seconds
    imu_timestamp = [(time - imu_timestamp[0]).total_seconds() for time in imu_timestamp]
    imu_sr = len(imu_timestamp)/imu_timestamp[-1]
    imu_data = librosa.resample(imu_data[:, :6].T, orig_sr=imu_sr, target_sr=50).T
    np.save(os.path.join(imu_dir, audio.replace('.pcm', '.npy')), imu_data)

    audio = pcm_to_wav(os.path.join(data_dir, audio), os.path.join(audio_dir, audio.replace('.pcm', '.wav')), sample_rate=48000, num_channels=2)
    # 16bit PCM to float
    audio = audio[:, 0] / 2 ** 15

    # with open(os.path.join(log_dir, log), 'r') as f:
    #     log = f.readlines()
    # print(log)
    # reference_dataset = 'simulate/NIGENS'
    # for original_audio in log:
    #     annotation = original_audio.rstrip()
    #     base_name = annotation.split('/')[-1].replace('\\', '/')

    #     ref_audio = os.path.join(reference_dataset, base_name)
    #     annotation = os.path.join(reference_dataset, base_name) + '.txt'
    #     if not os.path.exists(annotation):
    #         # no annotation file
    #         continue
    #     else:
    #         with open(annotation, 'r') as f:
    #             annotation = f.readlines()
    #         annotation = [line.rstrip() for line in annotation]
    #         print(annotation, librosa.get_duration(filename=ref_audio))

    corr = np.correlate(audio, chirp_template, mode='valid')
    peaks = signal.find_peaks(corr, height=100, distance=40000)[0]
    print(corr.shape, audio.shape, chirp_template.shape)
    fig, axs = plt.subplots(2, 1)

    axs[0].plot(corr)
    for peak in peaks:
        axs[0].axvline(peak, color='r')

    axs[1].plot(audio)
    for peak in peaks:
        axs[1].axvline(peak, color='r')
        axs[1].axvline(peak +   48000, color='r')

    plt.savefig('test.png')
    print('log:', log, audio.shape)
    break

