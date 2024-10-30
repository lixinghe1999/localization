'''
label for every 0.1s window (by default)
'''
import numpy as np

def azimuth2idx(azimuth):
    # [front, left, right, back, up, down]
    if azimuth >= -45 and azimuth <= 45:
        return 0
    elif azimuth >= 45 and azimuth <= 135:
        return 1
    elif azimuth >= -135 and azimuth <= -45:
        return 2
    else:
        return 3
def elevation2idx(elevation):
    if elevation >= 0:
        return 4
    else:
        return 5
        
def dist2idx(distance):
    if distance <= 1.5:
        return 6
    else:
        return 7

def Region_label(labels, config):
    total_num_frames = int(config['duration'] / config['frame_duration'])
    num_class = config['num_class']

    label_window = np.zeros((total_num_frames, 8))
    for frame, class_idx, _, azimuth, elevation, distance in labels:
        azimuth_idx = azimuth2idx(azimuth)
        elevation_idx = elevation2idx(elevation)
        dist_idx = dist2idx(distance)        
        frame = int(frame * 0.1 / config['frame_duration'])
        label_window[frame, azimuth_idx] = 1
        label_window[frame, elevation_idx] = 1
        label_window[frame, dist_idx] = 1
        # label_window[frame, 8 + int(class_idx)] = 1
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
    num_source = 3
    _nb_label_frames = config['duration'] * 10
    label_mat = np.zeros((_nb_label_frames, num_source, 5))
    frame_source_count = np.zeros((_nb_label_frames))
    for frame, class_idx, _, azimuth, elevation, _  in labels:
        if frame < _nb_label_frames:
            x = np.cos(np.radians(azimuth)) * np.cos(np.radians(elevation))
            y = np.sin(np.radians(azimuth)) * np.cos(np.radians(elevation))
            z = np.sin(np.radians(elevation))

            source_idx = int(frame_source_count[frame])
            label_mat[frame, source_idx, 0] = 1
            label_mat[frame, source_idx, 1] = x
            label_mat[frame, source_idx, 2] = y
            label_mat[frame, source_idx, 3] = z
            label_mat[frame, source_idx, 4] = class_idx

            frame_source_count[frame] += 1
    label_mat = label_mat.reshape(-1, num_source * 5)
    return label_mat



