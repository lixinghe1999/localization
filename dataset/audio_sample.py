'''
select a random audio sample from the ESC50 and TIMIT by type
'''
import pandas as pd
import random
import os
def get_chirp():
    import numpy as np
    from scipy.io import wavfile

    # Generate a chirp signal
    sample_rate = 44100  # 44.1 kHz sample rate
    duration = 5.0  # 1 second duration
    t = np.linspace(0, duration, int(duration * sample_rate), False)
    min_freq = 100
    max_freq = 8000
    chirp = np.sin(2 * np.pi * (min_freq + (max_freq - min_freq) * t / duration) * t)
    # Save the chirp to a WAV file
    wavfile.write('chirp.wav', sample_rate, (chirp * 32767).astype(np.int16))
def audio_sample(category):
    if category == 'speech':
        folder = 'TIMIT/TRAIN/'
        filtered_audio_list = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith(".WAV"):
                    filtered_audio_list.append(os.path.join(root, file))
    elif category == 'chirp':
        if not os.path.exists('chirp.wav'):
            get_chirp()
        filtered_audio_list = ['chirp.wav']
    else:
        folder = 'ESC-50-master/audio/'
        meta_esc50 = pd.read_csv(folder + '../meta/esc50.csv')
        filtered_audio_list = meta_esc50[meta_esc50.category == category].filename.tolist()
        filtered_audio_list = [folder + audio for audio in filtered_audio_list]

    random_audio = random.choice(filtered_audio_list)
    return random_audio