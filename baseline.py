from models.localization.doa import doa_inference
from models.localization.imu_utils import pyIMU, imu_loading
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=False)
plt.rcParams.update({'font.size': 20})
import os

dataset_dir = 'dataset/adhoc/measurement'
data_files = os.listdir(dataset_dir)
audio_files = [file for file in data_files if file.endswith('.wav')]; imu_files = [file for file in data_files if file.endswith('.csv')]
audio_files.sort(); imu_files.sort()
for (audio_file, imu_file) in zip(audio_files, imu_files):
    audio_file = os.path.join(dataset_dir, audio_file)
    imu_file = os.path.join(dataset_dir, imu_file)

    audio, fs = librosa.load(audio_file, sr=None, mono=False)
    imu = imu_loading(imu_file)
    print(audio.shape, fs,  imu.shape)
    mic_array = np.c_[[0.1, 0, 0], [-0.1, 0, 0.0]]

    predictions = doa_inference(audio, mic_array, fs, 2048, 'music', mode='window')
    t_predictions = np.arange(len(predictions)) * 512 / fs

    eulers, positions = pyIMU(imu, frequency=50)
    t_eulers = np.arange(len(eulers)) * 1 / 50

    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    axs[0].plot(t_predictions, predictions, label='DOA')
    axs[0].set_ylabel('Azimuth (deg)')
    axs[0].set_xlabel('Time (s)')

    axs[1].plot(t_eulers, eulers[:, 0], label='Yaw')
    axs[1].plot(t_eulers, eulers[:, 1], label='Pitch')
    axs[1].plot(t_eulers, eulers[:, 2], label='Roll')
    axs[1].set_ylabel('Euler angles (deg)')
    axs[1].set_xlabel('Time (s)')
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig('resources/baseline_result.pdf')
    break