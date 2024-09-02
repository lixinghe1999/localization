'''
label for every 0.1s window
'''
import numpy as np
import scipy.signal as signal
def Gaussian_window(label, config):
    '''
    label: {(class_idx, source_idx): {'frame':[0, 1, ...], 'location':(azimuth, elevation, distance)},  (class_idx, source_idx): ...}
    '''
    total_num_frames = config['duration'] * 10
    y = np.arange(config['min_azimuth'], config['max_azimuth'], 1, dtype=float)
    y_window = np.zeros((total_num_frames, y.shape[0]))
    for _, values in label.items():
        azimuth = values['location'][0]

        y_gaussian = np.exp(-((y - azimuth) ** 2) / (2 * 10 ** 2))
        for frame in values['frame']:
            y_window[frame] += y_gaussian

        # for frame in values['frame']:
        #     y_window[frame][azimuth] = 1
    return y_window

def Gaussian_window_evaluation(pred, label, threshold=0.5):
    '''
    pred: [batch, time, azimuth]
    label: [batch, time, azimuth]
    '''
    # use threshold to select the peak
    B, T, A = pred.shape
    pred = pred.reshape(-1, pred.shape[-1])
    label = label.reshape(-1, label.shape[-1])
    
    pred_peaks = []; label_peaks = []
    for (p, l) in zip(pred, label):
        pred_peak, _ = signal.find_peaks(p, height=threshold, distance=10)
        label_peak, _ = signal.find_peaks(l, height=threshold, distance=10)

        pred_peaks.append(pred_peak)
        label_peaks.append(label_peak)
    # calculate the distance between the peaks
    min_distances = []
    for l_peak, p_peak in zip(label_peaks, pred_peaks):
        if len(l_peak) == 0:
            min_distances.append(0)
        else:
            if len(p_peak) == 0:
                min_distances.append(180)
            else:
                _min_distances = []
                for l in l_peak:
                    min_distance = 180
                    for p in p_peak:
                        # circular distance calucation, 360 = 0
                        distance = min(abs(p_peak - l_peak), 360 - abs(p_peak - l_peak))
                        if distance < min_distance:
                            min_distance = distance
                    _min_distances.append(min_distance)
                min_distances.append(np.mean(_min_distances))
    min_distances = np.array(min_distances).reshape(B, T)
    return min_distances
        

            


