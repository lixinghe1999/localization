import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from utils.doa import doa
from utils.parameter import mic_array_binaural as mic_array
import scipy

def preprocess(imu):
    '''
    imu: numpy array of shape (n, 7)
    [gx, gy, gz, ax, ay, az, t]
    '''
    def low_high_seperate(data):
        b_l, a_l = scipy.signal.butter(3, 20, 'low', fs=200)
        b_h, a_h = scipy.signal.butter(3, 20, 'high', fs=200)
        low = scipy.signal.filtfilt(b_l, a_l, data, axis=0)
        high = scipy.signal.filtfilt(b_h, a_h, data, axis=0)
        return low, high
    imu = imu[:, :6]
    gyro = imu[:, :3]
    gyro_low, gyro_high = low_high_seperate(gyro)
    acc = imu[:, 3:6]
    acc_low, acc_high = low_high_seperate(acc)

    return gyro_low, acc_low, gyro_high, acc_high
folder = 'dataset/dataset_cuhk/2024-08-14/'
segments = os.listdir(folder)
segments.sort()
segments = segments[0:]
for segment in segments:
    print(segment)
    dataset_folder = os.path.join(folder, segment)
    files = os.listdir(dataset_folder)
    audio_file = [file for file in files if file.endswith('.wav')][0]
    assert 'bmi160_0.txt' in files and 'bmi160_1.txt' in files

    audio, sr = librosa.load(os.path.join(dataset_folder, audio_file), sr=None, mono=False)
    predictions, intervals = doa(audio, mic_array, sr, 1024, intervals=None, plot=False)
    print(predictions, intervals)

    imu0 = np.loadtxt(os.path.join(dataset_folder, 'bmi160_0.txt'))
    imu1 = np.loadtxt(os.path.join(dataset_folder, 'bmi160_1.txt'))

    gyro0_low, acc0_low, gyro0_high, acc0_high = preprocess(imu0)

    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    axs[0, 0].plot(audio.T)
    axs[0, 0].set_title('Audio')
    axs[1, 0].plot(gyro0_low)
    axs[1, 0].set_title('Gyro low')
    axs[2, 0].plot(acc0_low)
    axs[2, 0].set_title('Acc low')

    acc_abs = np.max(np.abs(acc0_high), axis=1)
    acc_peaks = scipy.signal.find_peaks(acc_abs, height=100, distance=50)[0]
    gyro_abs= np.max(np.abs(gyro0_high), axis=1)
    gyro_peaks = scipy.signal.find_peaks(gyro_abs, height=100, distance=50)[0]
    print(acc_peaks, gyro_peaks)
    axs[1, 1].plot(gyro0_high)
    axs[1, 1].scatter(gyro_peaks, gyro_abs[gyro_peaks], c='r')
    axs[1, 1].set_title('Gyro high')
    axs[2, 1].plot(acc0_high)
    axs[2, 1].scatter(acc_peaks, acc_abs[acc_peaks], c='r')
    axs[2, 1].set_title('Acc high')
    plt.savefig(f'figs/{segment}.png')
    plt.cla()
    break

