import os
import librosa
import noisereduce as nr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.io.wavfile as wavfile


from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

def direction_parser(direction):
    direction = int(direction)
    
    if direction == 0:
        return (0, -30)
    elif direction == 1:
        return (90, -30)
    elif direction == 2:
        return (180, -30)
    elif direction == 3:
        return (270, -30)
    elif direction == 4:
        return (0, 30)
    elif direction == 5:
        return (90, 30)
    elif direction == 6:
        return (180, 30)
    elif direction == 7:
        return (270, 30)

def distance_parser(distance):
    distance = int(distance)
    if distance == 0:
        return 1
    elif distance == 1:
        return 2
    else:
        return 3

def pcm_to_wav(pcm_file, num_channels=1):
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
    return pcm_array

def get_chirp(sample_rate=44100, duration=5.0, min_freq=100, max_freq=8000):
    import numpy as np
    from scipy.io import wavfile
    t = np.linspace(0, duration, int(duration * sample_rate), False)
    chirp = np.sin(2 * np.pi * (min_freq + (max_freq - min_freq) * t / duration) * t)
    return chirp

def chirp_preamble(mono_audio, plot=False):
    print('start matching the chirp, may take times')
    chirp_template = get_chirp(sample_rate=48000, duration=1, min_freq=2000, max_freq=4000)
    corr = np.correlate(mono_audio[:], chirp_template[:], mode='valid')
    peaks = signal.find_peaks(corr, height=50, distance=40000)[0]
    assert len(peaks) == 1
    if plot:    
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(corr)
        for peak in peaks:
            axs[0].axvline(peak, color='r')

        axs[1].plot(mono_audio)
        for peak in peaks:
            axs[1].axvline(peak, color='r')
            axs[1].axvline(peak + 48000, color='r')
        plt.savefig('test.png')
    return peaks[0]

datadir = 'dataset/earphone/haozheng_human'
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

assert len(audios) == len(imus)
if len(logs) == 0: # no log file, create a dummy log file
    logs = [None] * len(audios)
for audio, imu, log in zip(audios, imus, logs):
    print('processing:', audio, imu, log)

    meta, time = audio[:-4].split('-')
    user, distance, direction, location, sound = meta.split('_')
    azimuth, elevation = direction_parser(direction)
    distance = distance_parser(distance)

    imu_data = np.loadtxt(os.path.join(data_dir, imu), delimiter=',', skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6,))
    imu_timestamp = np.loadtxt(os.path.join(data_dir, imu), delimiter=',', skiprows=1, usecols=(7,), dtype=str)
    imu_timestamp = [datetime.datetime.strptime(time, '%Y%m%d_%H%M%S_%f') for time in imu_timestamp]
    # set the start time to 0 and convert to seconds
    imu_timestamp = [(time - imu_timestamp[0]).total_seconds() for time in imu_timestamp]
    imu_sr = len(imu_timestamp)/imu_timestamp[-1]
    imu_data = librosa.resample(imu_data[:, :6].T, orig_sr=imu_sr, target_sr=50).T

    stereo_audio = pcm_to_wav(os.path.join(data_dir, audio), num_channels=2)
    mono_audio = stereo_audio[:, 0] / 2 ** 15

    reference_dataset = 'simulate/NIGENS'
    if log is None:
        # auto generate the sound event
        mono_audio /= np.max(np.abs(mono_audio))
        # reshape the audio to 2D with 0.1s window
        # timestamps = librosa.effects.split(mono_audio, top_db=20, frame_length=2048, hop_length=2048)
        sound_event_list = [[0, len(mono_audio) / 48000]]
        peak = 0;recording_start = len(mono_audio) / 48000
    else:
        peak = chirp_preamble(mono_audio, plot=False)

        log = pd.read_csv(os.path.join(log_dir, log), sep=' ')
        recording_start = 0
        sound_event_list = []
        for (i, row) in log.iterrows():
            # each row: left_audio, right_audio, start, end
            left_audio = row['left_audio']; right_audio = row['right_audio']
            start = float(row['start']); end = float(row['end'])

            annotation = left_audio.rstrip()
            base_name = annotation.split('/')[-1].replace('\\', '/')

            ref_audio = os.path.join(reference_dataset, base_name)
            annotation = os.path.join(reference_dataset, base_name) + '.txt'
            annotation = pd.read_csv(annotation, sep='\t', names=['start', 'end'])

            for (j, anno) in annotation.iterrows():
                if anno['end'] <= start or anno['start'] >= end:
                    continue
                else:
                    new_start = max(anno['start'], start)
                    new_end = min(anno['end'], end)
                sound_event_list.append([new_start - start + recording_start, new_end - start + recording_start])
            recording_start += end - start

    # sound_event_list = [[start, end], [start, end], ...]
    # fuse the event that are too close

    sound_event_list.sort(key=lambda x: x[0])
    new_sound_event_list = []
    for sound_event in sound_event_list:
        if len(new_sound_event_list) == 0:
            new_sound_event_list.append(sound_event)
        else:
            if sound_event[0] - new_sound_event_list[-1][1] < 0.1:
                new_sound_event_list[-1][1] = sound_event[1]
            else:
                new_sound_event_list.append(sound_event)
    sound_event_list = new_sound_event_list

    with open(os.path.join(meta_dir, audio.replace('.pcm', '.txt')), 'w') as f:
        f.write('sound_event_recording,start_time,end_time,azi,ele,dist\n')
        for sound_event in sound_event_list:
            f.write(f'{sound},{sound_event[0]},{sound_event[1]},{azimuth},{elevation},{distance}\n')


    chirp_end = peak + 48000
    recording_end = chirp_end + int(recording_start * 48000)
    stereo_audio = stereo_audio[chirp_end:recording_end]
    wavfile.write(os.path.join(audio_dir, audio.replace('.pcm', '.wav')), 48000, stereo_audio)

    chirp_end_imu = int(chirp_end / 48000 * 50)
    recording_end = int(recording_end / 48000 * 50)
    imu_data = imu_data[chirp_end_imu:recording_end]
    np.save(os.path.join(imu_dir, audio.replace('.pcm', '.npy')), imu_data)

  
