'''
Run this script to play random audio from the dataset with speakers
'''
import os
import librosa
import sounddevice as sd
import numpy as np
import datetime
def random_audio(audio_type):
    import random
    if audio_type == 'ESC50':
        audio_path = 'ESC-50-master/audio/'
    elif audio_type == 'TIMIT':
        audio_path = 'TIMIT/TRAIN'        
    audio_files = os.listdir(audio_path)
    random_audio = random.choice(audio_files)
    return os.path.join(audio_path, random_audio)
def play_audio(audio_type, duration=5):
    # Importing the necessary libraries
    log = {}
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log['time'] = current_time
    log['audio_type'] = audio_type
    log['selected_audio'] = []
    for _audio_type in audio_type:
        log['selected_audio'].append(random_audio(_audio_type))
    audios = []
    assert len(log['selected_audio']) <=2 # Only support up to 2 audio files: Two sound source
    for audio_file in log['selected_audio']:
        audio, fs = librosa.load(audio_file, sr=16000)
        if len(audio) < duration*fs:
            zeros = np.zeros(duration*fs - len(audio))
            audio = np.concatenate((audio, zeros))
        else:
            audio = audio[:duration*fs]
        audios.append(audio[:, None])    
    audio = np.concatenate(audios, axis=1)
    print('Playing audio')
    sd.play(audio, fs, blocking=True)
    print('Finished playing audio')
    return log
def main():
    import json
    number_of_audio = 1
    logs = []
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    for i in range(number_of_audio):
        log = play_audio(['ESC50', 'ESC50'])
        logs.append(log)

    with open(f'log/{current_time}.json', 'w') as f:
        json.dump(logs, f)
if __name__ == '__main__':
    main()