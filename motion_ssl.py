import os
import numpy as np
import matplotlib.pyplot as plt
from utils.doa import pra_doa
from utils.parameter import mic_array_binaural as mic_array
from utils.loading import load_folder

folder = 'dataset/dataset_cuhk/2024-08-21/'
segments = os.listdir(folder)
segments.sort()
segments.remove('meta.txt')
segments = segments[5:10]
for segment in segments:
    dataset_folder = os.path.join(folder, segment)
    data_dict = load_folder(dataset_folder)
    predictions, intervals = pra_doa(data_dict['audio_2']['data'].T, mic_array, fs=16000, nfft=512, plot=False)
    print(predictions, intervals)

    audio = data_dict['audio_2']['data']
    imu = data_dict['imu_1']['data']
    fig, axs = plt.subplots(5, 1, figsize=(10, 10))
    # audio, g_low, a_low, g_high, a_high
    axs[0].plot(audio)
    axs[1].plot(imu[:, :3])
    axs[2].plot(imu[:, 3:6])
    axs[3].plot(imu[:, 6:9])
    axs[4].plot(imu[:, 9:12])
    plt.savefig(f'figs/{segment}.png')

    # acc_abs = np.max(np.abs(acc0_high), axis=1)
    # acc_peaks = scipy.signal.find_peaks(acc_abs, height=100, distance=50)[0]
    # gyro_abs= np.max(np.abs(gyro0_high), axis=1)
    # gyro_peaks = scipy.signal.find_peaks(gyro_abs, height=100, distance=50)[0]
    # print(acc_peaks, gyro_peaks)
    # axs[1, 1].plot(gyro0_high)
    # axs[1, 1].scatter(gyro_peaks, gyro_abs[gyro_peaks], c='r')
    # axs[1, 1].set_title('Gyro high')
    # axs[2, 1].plot(acc0_high)
    # axs[2, 1].scatter(acc_peaks, acc_abs[acc_peaks], c='r')
    # axs[2, 1].set_title('Acc high')
    # plt.savefig(f'figs/{segment}.png')
    # plt.cla()
    # break

