import numpy as np
import librosa
def si_snr(s, s_hat):
    """
    Compute the Scale-Invariant Signal-to-Noise Ratio (SI-SNR)
    
    Args:
    s (numpy.ndarray): Clean signal
    s_hat (numpy.ndarray): Estimated signal
    
    Returns:
    float: SI-SNR value in dB
    """
    s = s - np.mean(s)
    s_hat = s_hat - np.mean(s_hat)
    
    s_target = np.dot(s, s_hat) / np.dot(s, s) * s
    e_noise = s_hat - s_target
    
    si_snr_value = 10 * np.log10(np.sum(s_target ** 2) / np.sum(e_noise ** 2))
    return si_snr_value


if __name__ == '__main__':
    t_cut = 0.83 
    Fs = 8000

    import os
    input_samples = os.listdir('input_samples')

    ouput_samples = os.listdir('output_samples')


    input_german = librosa.load('input_samples/german_speech_8000.wav', sr=None)[0]
    input_singing = librosa.load('input_samples/singing_8000.wav', sr=None)[0]

    # input_german = librosa.load('input_samples/signal1.wav', sr=None)[0]
    # input_singing = librosa.load('input_samples/signal2.wav', sr=None)[0]


    input_mixed = librosa.load('output_samples/input.wav', sr=None)[0]

    n_lim = int(np.ceil(len(input_mixed) - t_cut * Fs))
    input_german = input_german[:n_lim]
    input_singing = input_singing[:n_lim]
    input_mixed = input_mixed[:n_lim]

    print(n_lim, input_german.shape, input_singing.shape, input_mixed.shape)
    # sisnr_german = si_snr(input_german, input_mixed)
    sisnr_singing = si_snr(input_singing, input_mixed)
    print(sisnr_singing)

    # output_DirectMVDR = librosa.load('output_samples/output_DirectMVDR.wav', sr=None)[0]
    # offset = 799
    # sisnr_german = si_snr(input_german, output_DirectMVDR[offset:])
    # sisnr_singing = si_snr(input_singing, output_DirectMVDR[offset:])

    # sisnr_german = si_snr(input_german, output_DirectMVDR[:-offset])
    # sisnr_singing = si_snr(input_singing, output_DirectMVDR[:-offset])
    # print(sisnr_german, sisnr_singing)