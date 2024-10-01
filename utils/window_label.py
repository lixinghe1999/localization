'''
label for every 0.1s window
'''
import numpy as np
import scipy.signal as signal
def Gaussian_label(labels, config):
    '''
    label: [(frame, class_idx, source_idx, location), ...]
    Note: no classification at all
    '''
    total_num_frames = config['duration'] * 10
    y = np.arange(config['min_azimuth'], config['max_azimuth'], 1, dtype=float)
    y_window = np.zeros((total_num_frames, y.shape[0]))
    for frame, _, _, azimuth, _, _ in labels:
        y_gaussian = np.exp(-((y - azimuth) ** 2) / (2 * 10 ** 2))
        y_window[frame] += y_gaussian
    return y_window

def doa2idx(azimuth, elevation):
    # [front, left, right, back, up, down]
    if elevation >= -35 and elevation <= 35:
        if azimuth >= -45 and azimuth <= 45:
            return 0
        elif azimuth >= 45 and azimuth <= 135:
            return 1
        elif azimuth >= -135 and azimuth <= -45:
            return 2
        else:
            return 3
    else:
        if elevation > 35:
            return 4
        else:
            return 5
        
def dist2idx(distance):
    if distance <= 1:
        return 0
    elif distance <= 2:
        return 1
    else:
        return 2

def Region_label(labels, config):
    total_num_frames = config['duration'] * 10
    label_window = np.zeros((total_num_frames, 9))
    for frame, _, _, azimuth, elevation, distance in labels:
        doa_idx = doa2idx(azimuth, elevation)
        label_window[int(frame), doa_idx] = 1

        dist_idx = dist2idx(distance)
        label_window[int(frame), 6 + dist_idx] = 1
    return label_window


def ACCDOA_label(labels, config, sed=False):
    '''
    label: [(frame, class_idx, source_idx, location), ...]
    config
    sed: bool, if True, do sed+doa, else doa only
    '''
    if not sed:
        num_class = 1
    else:
        num_class = config['num_class']
    _nb_label_frames = config['duration'] * 10
    se_label = np.zeros((_nb_label_frames, num_class))
    x_label = np.zeros((_nb_label_frames, num_class))
    y_label = np.zeros((_nb_label_frames, num_class))
    z_label = np.zeros((_nb_label_frames, num_class))

    for frame, class_idx, _, azimuth, elevation, _  in labels:
        if not sed:
            class_idx = 0
        if frame < _nb_label_frames:
            x = np.cos(np.radians(azimuth)) * np.cos(np.radians(elevation))
            y = np.sin(np.radians(azimuth)) * np.cos(np.radians(elevation))
            z = np.sin(np.radians(elevation))

            se_label[frame, class_idx] = 1
            x_label[frame, class_idx] = x
            y_label[frame, class_idx] = y
            z_label[frame, class_idx] = z
    label_mat = np.concatenate((se_label, x_label, y_label, z_label), axis=1)
    return label_mat

def Multi_ACCDOA_label(labels, config):
    '''
    label: [(frame, class_idx, source_idx, location), ...]
    config
    sed: bool, if True, do sed+doa, else doa only
    '''
    num_source = 3
    _nb_label_frames = config['duration'] * 10
    se_label = np.zeros((_nb_label_frames, num_source))
    x_label = np.zeros((_nb_label_frames, num_source))
    y_label = np.zeros((_nb_label_frames, num_source))
    z_label = np.zeros((_nb_label_frames, num_source))

    for frame, _, source_idx, azimuth, elevation, _  in labels:
        if frame < _nb_label_frames:
            x = np.cos(np.radians(azimuth)) * np.cos(np.radians(elevation))
            y = np.sin(np.radians(azimuth)) * np.cos(np.radians(elevation))
            z = np.sin(np.radians(elevation))

            se_label[frame, source_idx] = 1
            x_label[frame, source_idx] = x
            y_label[frame, source_idx] = y
            z_label[frame, source_idx] = z
    label_mat = np.concatenate((se_label, x_label, y_label, z_label), axis=1)
    return label_mat

