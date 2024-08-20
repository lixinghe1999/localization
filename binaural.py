import librosa
from utils.feature import noise_reduce, source_separation
import matplotlib.pyplot as plt
import os

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
    print(audio.shape, sr)
    audio = noise_reduce(audio)
    source_separation(audio, sr)
    break

