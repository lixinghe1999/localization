import pyroomacoustics as pra
from parameter import SMARTGLASS, DUALDEVICE
import os 
import numpy as np
import librosa
from doa import init, inference

DATASET = '../dataset/dualdevice/TIMIT_1/test'
audio_folder = os.path.join(DATASET, 'audio')
meta_folder = os.path.join(DATASET, 'meta')
audios = os.listdir(audio_folder); audios.sort()
metas = os.listdir(meta_folder); metas.sort()

algo = init(DUALDEVICE, fs=16000, nfft=1600, algorithm='music')
for audio, meta in zip(audios, metas):
    print(audio, meta)
    audio_path = os.path.join(audio_folder, audio)
    audio_sources = os.listdir(audio_path)
    sources = []
    for source in audio_sources:
        source_path = os.path.join(audio_path, source)
        source = librosa.load(source_path, sr=16000, mono=False)[0]
        sources.append(source)
    sources = np.concatenate(sources)
    prediction = inference(algo, sources)
    print(prediction)
    meta_path = os.path.join(meta_folder, meta)
    
    break
