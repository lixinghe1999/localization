'''
label for every 0.1s window (by default)
'''
import numpy as np

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
    if distance <= 1.5:
        return 0
    else:
        return 1

def Region_label(labels, config):
    total_num_frames = int(config['duration'] / config['frame_duration'])
    num_class = config['num_class']

    label_window = np.zeros((total_num_frames, 8 + num_class))
    for frame, class_idx, _, azimuth, elevation, distance in labels:
        doa_idx = doa2idx(azimuth, elevation)
        dist_idx = dist2idx(distance)        
        frame = int(frame * 0.1 / config['frame_duration'])
        label_window[frame, doa_idx] = 1
        label_window[frame, 6 + dist_idx] = 1
        label_window[frame, 8 + int(class_idx)] = 1
    return label_window


def ACCDOA_label(labels, config, sed=False):
    '''
    label: [(frame, class_idx, source_idx, location), ...]
    config
    sed: bool, if True, do sed+doa, else doa only
    '''
    num_class = config['num_class']
    _nb_label_frames = config['duration'] * 10
    label_mat = np.zeros((_nb_label_frames, num_class, 4))
    for frame, class_idx, _, azimuth, elevation, _  in labels:
        if num_class == 1:
            class_idx = 0
        if frame < _nb_label_frames:
            x = np.cos(np.radians(azimuth)) * np.cos(np.radians(elevation))
            y = np.sin(np.radians(azimuth)) * np.cos(np.radians(elevation))
            z = np.sin(np.radians(elevation))

            label_mat[frame, class_idx, 0] = 1
            label_mat[frame, class_idx, 1] = x
            label_mat[frame, class_idx, 2] = y
            label_mat[frame, class_idx, 3] = z
    label_mat = label_mat.reshape(-1, num_class * 4)
    return label_mat

def Multi_ACCDOA_label(labels, config):
    '''
    label: [(frame, class_idx, source_idx, location), ...]
    config
    sed: bool, if True, do sed+doa, else doa only
    '''
    num_source = 2
    _nb_label_frames = config['duration'] * 10
    label_mat = np.zeros((_nb_label_frames, num_source, 4))

    for frame, _, source_idx, azimuth, elevation, _  in labels:
        if frame < _nb_label_frames:
            x = np.cos(np.radians(azimuth)) * np.cos(np.radians(elevation))
            y = np.sin(np.radians(azimuth)) * np.cos(np.radians(elevation))
            z = np.sin(np.radians(elevation))

            label_mat[frame, source_idx, 0] = 1
            label_mat[frame, source_idx, 1] = x
            label_mat[frame, source_idx, 2] = y
            label_mat[frame, source_idx, 3] = z
    label_mat = label_mat.reshape(-1, num_source * 4)
    return label_mat

def Gaussian_label(labels, config):
    _nb_label_frames = config['duration'] * 10
    label_mat = np.zeros((_nb_label_frames, 360))
    x = np.arange(360)
    for frame, _, _, azimuth, elevation, _  in labels:
        if frame < _nb_label_frames:
            gaussian = np.exp(-((x - azimuth) ** 2) / (2 * 10 ** 2))
            label_mat[frame] += gaussian
            label_mat[frame] /= np.max(label_mat[frame])    
    return label_mat


