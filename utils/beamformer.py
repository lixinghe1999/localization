import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
def simulation(mic_array, fs, azimuth):
    # Create a 4 by 6 metres shoe box room
    mic_array = mic_array[:2, :] # use only 2D
    mic_center = np.c_[[2, 3]]
    room = pra.ShoeBox([4,6], fs)
    
    source = mic_center + np.mean(mic_array, axis=1) + 1 * np.array([np.cos(np.deg2rad(azimuth)), np.sin(np.deg2rad(azimuth))])
    print(mic_center+mic_array)
    # room.add_source(source)
    room.add_microphone_array(pra.Beamformer(mic_center+mic_array, room.fs))
    # Now compute the delay and sum weights for the beamformer

    room.mic_array.far_field_weights(np.deg2rad(azimuth))
    # plot the room and resulting beamformer
    room.plot(freq=[1000, 2000], img_order=0)
    plt.savefig('figs/beamformer_{}.png'.format(azimuth))
        
def beamforming(audio, mic_array, fs, nfft, azimuth, intervals=None, plot=False):
    mics = pra.Beamformer(mic_array, fs, nfft)  # set the beamformer object as you like
    mics.far_field_weights(np.deg2rad(azimuth))
    if intervals is None:
        intervals = librosa.effects.split(y=audio, top_db=30, ref=1)
        # remove too short intervals
        intervals = [interval for interval in intervals if interval[1] - interval[0] > nfft]
    intervals = [[0, 220500]]
    n_windows = np.shape(intervals)[0]
    plt.subplot(n_windows, 1, 1)
    mics.plot_beam_response()
    for i in range(n_windows):
        start = intervals[i][0]
        end = intervals[i][1]
        data = audio[:, start:end]
        mics.signals = data
        output = mics.process()
        print(data.shape, output.shape)
        plt.subplot(n_windows, 1, i+1)
        plt.plot(data[0])
        plt.plot(output)
    plt.savefig('beamformer.png')
    plt.close()
    sf.write('output.wav', output, fs)
    return output
