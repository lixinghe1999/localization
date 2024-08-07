'''
With paired aoa or tdoa as input, estimate the position of the source.
'''
from doa import init_music, inference_music
import numpy as np
import os
import librosa

def doa_to_xyz(doa_preds, mic_centers, plot=False):
    from scipy.optimize import minimize
    # Define the objective function to minimize
    def objective_function(coordinate):
        x, y, z = coordinate
        azimuths_calculated = np.arctan2(y - mic_centers[:, 1], x - mic_centers[:, 0]) * 180 / np.pi
        n_windows = doa_preds.shape[1]
        azimuths_calculated = np.tile(azimuths_calculated, (n_windows, 1)).T
        return np.sum((doa_preds - azimuths_calculated) ** 2)
    # Set the initial guess for the coordinates
    initial_guess = np.array([0.1, 0.1, 0.0])

    # Optimize the coordinates using the Nelder-Mead method
    result = minimize(objective_function, initial_guess)
    print('Estimated source position:', result.x)

    if plot:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(mic_centers[:, 0], mic_centers[:, 1], c='r', marker='o')
        ax.scatter(result.x[0], result.x[1], c='b', marker='x')
        for doa_pred, mic_center in zip(doa_preds, mic_centers):
            # plot the doa line
            x = np.cos(np.deg2rad(doa_pred)) + mic_center[0]
            y = np.sin(np.deg2rad(doa_pred)) + mic_center[1]
            ax.plot([mic_center[0], x[0]], [mic_center[1], y[0]], 'g')
            
        plt.savefig('doa_to_xyz.png')

def main(mic_array, audio, ):
    N_channel = audio.shape[0]
    algo = init_music(0, mic_array, fs=16000, nfft=1024)
    predictions, intervals = inference_music(algo, audio)
    print('Predictions:', predictions)

    doa_preds = []; mic_centers = []
    for i in range(N_channel):
        for j in range(i+1, N_channel):
            tiny_microphone = mic_array[:, [i, j]]
            tiny_audio = audio[[i, j]]
            algo = init_music(0, tiny_microphone, fs=16000, nfft=1024)
            predictions, _ = inference_music(algo, tiny_audio, intervals)
            # predictions = [np.mean(predictions)]
            doa_preds.append(predictions)
            mic_centers.append(tiny_microphone.mean(axis=1))
    doa_preds = np.array(doa_preds)
    mic_centers = np.array(mic_centers)
    doa_to_xyz(doa_preds, mic_centers, plot=True)


if __name__ == '__main__':
    mic_array_seeed = np.c_[              
                    [ -0.03,  0.06, 0.0],
                    [ 0.03,  0.06, 0.0],
                    [ 0.06,  0.0, 0.0],
                    [ 0.03,  -0.06, 0.0],
                    [ -0.03,  -0.06, 0.0],
                    [ -0.06,  0, 0.0], 
                    ]
    dataset_folder = '../dataset/dataset_cuhk'
    audio_names = os.listdir(dataset_folder)    
    # audio_names = ['dataset/20240731_154148_micarray.wav']
    for audio_name in audio_names:
        audio_name = os.path.join(dataset_folder, audio_name)
        audio, sr = librosa.load(audio_name, sr=None, mono=False)
        audio = audio[:6, :]
        main(mic_array_seeed, audio)
        break