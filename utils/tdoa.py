import numpy as np

def tdoa(audio):
    N_channel = audio.shape[0]
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