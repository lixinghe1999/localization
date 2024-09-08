import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
def ACCDOA_evaluation(pred, label, implicit=False, vis=False):
    def distance_between_cartesian_coordinates(x1, y1, z1, x2, y2, z2):
        """
        Angular distance between two cartesian coordinates
        MORE: https://en.wikipedia.org/wiki/Great-circle_distance
        Check 'From chord length' section

        :return: angular distance in degrees
        """
        # Normalize the Cartesian vectors
        N1 = np.sqrt(x1**2 + y1**2 + z1**2 + 1e-10)
        N2 = np.sqrt(x2**2 + y2**2 + z2**2 + 1e-10)
        x1, y1, z1, x2, y2, z2 = x1/N1, y1/N1, z1/N1, x2/N2, y2/N2, z2/N2

        #Compute the distance
        dist = x1*x2 + y1*y2 + z1*z2
        dist = np.clip(dist, -1, 1)
        dist = np.arccos(dist) * 180 / np.pi
        return dist
    if implicit:
        '''
        pred: [batch, time, n_class * 3 (x, y, z)]
        label: [batch, time, n_class * 4 (class_active, x, y, z)]
        '''
        N1 = pred.shape[-1]; N2 = label.shape[-1]
        n_class = N1 // 3
        assert N2 == n_class * 4
        pred = pred.reshape(-1, n_class, 3)
        pred_sed = np.sqrt(np.sum(pred**2, axis=-1)) > 0.5  # [batch*time, n_class]
        label = label.reshape(-1, n_class, 4)
        label_sed = label[..., 0] > 0.5 # [batch*time, n_class]
    else:
        '''
        pred: [batch, time, n_class * 4 (x, y, z)]
        label: [batch, time, n_class * 4 (class_active, x, y, z)]
        '''
        n_class = pred.shape[-1] // 4
        pred = pred.reshape(-1, n_class, 4); label = label.reshape(-1, n_class, 4)
        pred_sed = pred[..., 0] > 0.5 # [batch*time, n_class]
        label_sed = label[..., 0] > 0.5 # [batch*time, n_class]
    

    if vis:
        batch, time = pred.shape[0], pred.shape[1]  
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    else:
        # correct sed prediction: the pred and label are both active
        sed_TP = np.sum(np.logical_and(pred_sed, label_sed))
        sed_FP = np.sum(np.logical_and(pred_sed, np.logical_not(label_sed)))
        sed_FN = np.sum(np.logical_and(np.logical_not(pred_sed), label_sed))
        sed_precision = sed_TP / (sed_TP + sed_FP + 1e-10)
        sed_recall = sed_TP / (sed_TP + sed_FN + 1e-10)
        sed_F1 = 2 * sed_precision * sed_recall / (sed_precision + sed_recall + 1e-10)


        # correct: the pred and label are both active and the distance is less than 20
        distances = []
        TP, FP, FN, TN = 0, 0, 0, 0
        for i in range(pred.shape[0]): # sample
            for c in range(n_class):
                if label_sed[i, c] == pred_sed[i, c]:
                    if label_sed[i, c]: # positive sample
                        dist = distance_between_cartesian_coordinates(pred[i, c, 0], pred[i, c, 1], pred[i, c, 2], label[i, c, 1], label[i, c, 2], label[i, c, 3])
                        distances.append(dist)
                        if dist < 20:
                            TP += 1
                    else:
                        TN += 1
                else:
                    if label_sed[i, c]:
                        FN += 1
                    else:
                        FP += 1
        precision = TP / (TP + FP + 1e-10)
        recall = TP / (TP + FN + 1e-10)
        F1 = 2 * precision * recall / (precision + recall + 1e-10)
        return {
            'precision': precision,
            'recall': recall,
            'F1': F1,
            'distance': np.mean(distances) if len(distances) > 0 else 0,
            'sed_precision': sed_precision,
            'sed_recall': sed_recall,
            'sed_F1': sed_F1
        }


def Gaussian_evaluation(pred, label, threshold=0.5, plot=False):
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
            if len(p_peak) == 0:
                min_distances.append(0)
            else:
                min_distances.append(180)
        else:
            if len(p_peak) == 0:
                min_distances.append(180)
            else:
                _min_distances = []
                for l in l_peak:
                    min_distance = 180
                    for p in p_peak:
                        # circular distance calucation, 360 = 0
                        distance = min(abs(l - p), 360 - abs(l - p))
                        if distance < min_distance:
                            min_distance = distance
                    _min_distances.append(min_distance)
                min_distances.append(np.mean(_min_distances))
    min_distances = np.array(min_distances).reshape(B, T)
    if plot:
        return min_distances, pred_peaks, label_peaks
    else:
        return min_distances
        
