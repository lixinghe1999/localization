import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):
    '''
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    '''
    
    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))

    # find max cross correlation index
    shift = np.argmax(cc) - max_shift

    # Sometimes, there is a 180-degree phase difference between the two microphones.
    # shift = np.argmax(np.abs(cc)) - max_shift

    tau = shift / float(interp * fs)
    
    return tau, cc

def compute_itd_ild(left_signal, right_signal, sample_rate):
    """
    Compute interaural time difference (ITD) and interaural level difference (ILD)
    from left and right audio signals.
    
    Parameters:
    left_signal (numpy.ndarray): Left audio signal.
    right_signal (numpy.ndarray): Right audio signal.
    sample_rate (float): Sampling rate of the audio signals.
    
    Returns:
    itd (float): Interaural time difference in seconds.
    ild (float): Interaural level difference in decibels.
    """
    # Compute the analytic signal using the Hilbert transform
    left_analytic = hilbert(left_signal)
    right_analytic = hilbert(right_signal)
    
    # Compute the instantaneous phase of the analytic signals
    left_phase = np.unwrap(np.angle(left_analytic))
    right_phase = np.unwrap(np.angle(right_analytic))
    
    # Compute the interaural time difference (ITD)
    itd = np.mean(np.diff(left_phase - right_phase)) / (2 * np.pi * sample_rate)
    
    # Compute the interaural level difference (ILD)
    ild = 20 * np.log10(np.abs(left_analytic) / np.abs(right_analytic))
    ild = np.mean(ild)
    
    return itd, ild
def tdoa(audio, plot=False):
    N_channel, N_sample = audio.shape[0], audio.shape[1]
    
    mono_audio = audio[0]
    mono_spec = np.fft.rfft(mono_audio, n=N_sample)
    results = []
    for i in range(N_channel):
        for j in range(i+1, N_channel):
            audio1 = audio[i]
            audio2 = audio[j]
            tau, cc = gcc_phat(audio1, audio2, fs=24000, max_tau=0.0002, interp=2)
            print(f'Channel {i} and {j} tau: {tau}')
            results.append(cc)
            # X = np.fft.rfft(audio1, n=N_sample)
            # Y = np.fft.rfft(audio2, n=N_sample)
            # R = X * np.conj(Y)
            # mag_R, phase_R = np.abs(R), np.angle(R)
            # phase_Rs.append(phase_R)
            # print(f'Channel {i} and {j} phase_R shape: {phase_R.shape}')
    if plot:
        results = np.array(results).T
        fig, axs = plt.subplots(3, 1)
        axs[0].plot(mono_audio)
        axs[1].plot(np.abs(mono_spec))
        axs[2].plot(results)
        plt.savefig('tdoa.png')
