'''
Multi-channel audio may need to extract the features before
'''
import numpy as np
import librosa
def gccphat(audio, gcc_phat_len=32):
    N_channel = audio.shape[0]
    gccs = []
    for i in range(N_channel):
        for j in range(i+1, N_channel):
            audio1 = audio[i]
            audio2 = audio[j]
            audio[i] = audio[i] / np.max(np.abs(audio[i]))
            n = audio1.shape[0] + audio2.shape[0] 
            X = np.fft.rfft(audio1, n=n)
            Y = np.fft.rfft(audio2, n=n)
            R = X * np.conj(Y)
            cc = np.fft.irfft(R / (1e-6 + np.abs(R)),  n=n)
            cc = np.concatenate((cc[-gcc_phat_len:], cc[:gcc_phat_len+1])).astype(np.float32)
            gccs.append(cc)
    output = np.array(gccs).reshape(-1)
    return output

def mel_spec(audio, n_fft=1024, n_mels=64):
    mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_fft=n_fft, n_mels=n_mels)
    mel = librosa.power_to_db(mel, ref=np.max)
    return mel