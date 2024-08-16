'''
Run this script to play random audio from the dataset with speakers
'''
import os
import librosa
import sounddevice as sd
import numpy as np
import datetime
from audio_sample import audio_sample
from ssh_control import create_connection, execute_command

def play_audio(log):
    # Importing the necessary libraries
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log['time'] = current_time
    log['left_audio'] = audio_sample(log['left_type'])
    log['right_audio'] = audio_sample(log['right_type'])
    if log['mono']:
        log['right_audio'] = log['left_audio']
    print(f'Playing audio: {log["left_audio"]} and {log["right_audio"]}')
    audios = []
    for audio_file in [log['left_audio'], log['right_audio']]:
        audio, fs = librosa.load(audio_file, sr=16000)
        audios.append(audio[:, None])   
    n_sample = 0
    for audio in audios:
        n_sample = max(audio.shape[0], n_sample)
    for i, audio in enumerate(audios):
        if audio.shape[0] < n_sample:
            audios[i] = np.pad(audio, ((0, n_sample - audio.shape[0]), (0, 0)), 'constant')    
    audio = np.concatenate(audios, axis=1)
    print('Playing audio... with shape:', audio.shape)
    sd.play(audio, fs, blocking=True)
    print('Finished playing audio')
    return log
    
if __name__ == '__main__':
    import json
    while True:
        client = create_connection()
        print('Connected to the Raspberry Pi!')
        a = input('press 1 to play audio (edit the config first), press 2 to exit')
        if a == '1':
            config_file = 'collect_config.json'
            log = json.load(open(config_file, 'r'))
            current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

            execute_command(client, f'python3 /home/pi/collect_data.py {current_time}')

            log = play_audio(log)
            with open(f'log/{current_time}.json', 'w') as f:
                json.dump(log, f, indent=4)
        elif a == '2':
            exit()
        
        