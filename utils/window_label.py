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

def ACCDOA_label(labels, config, sed=True):
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

def get_adpit_labels_for_file(_desc_file, _nb_label_frames, _nb_unique_classes=13):
    """
    Reads description file and returns classification based SED labels and regression based DOA labels
    for multi-ACCDOA with Auxiliary Duplicating Permutation Invariant Training (ADPIT)

    :param _desc_file: metadata description file
    :return: label_mat: of dimension [nb_frames, 6, 4(=act+XYZ), max_classes]
    """

    se_label = np.zeros((_nb_label_frames, 6, _nb_unique_classes))  # [nb_frames, 6, max_classes]
    x_label = np.zeros((_nb_label_frames, 6, _nb_unique_classes))
    y_label = np.zeros((_nb_label_frames, 6, _nb_unique_classes))
    z_label = np.zeros((_nb_label_frames, 6, _nb_unique_classes))

    for frame_ind, active_event_list in _desc_file.items():
        if frame_ind < _nb_label_frames:
            active_event_list.sort(key=lambda x: x[0])  # sort for ov from the same class
            active_event_list_per_class = []
            for i, active_event in enumerate(active_event_list):
                active_event_list_per_class.append(active_event)
                if i == len(active_event_list) - 1:  # if the last
                    if len(active_event_list_per_class) == 1:  # if no ov from the same class
                        # a0----
                        active_event_a0 = active_event_list_per_class[0]
                        se_label[frame_ind, 0, active_event_a0[0]] = 1
                        x_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[2]
                        y_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[3]
                        z_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[4]
                    elif len(active_event_list_per_class) == 2:  # if ov with 2 sources from the same class
                        # --b0--
                        active_event_b0 = active_event_list_per_class[0]
                        se_label[frame_ind, 1, active_event_b0[0]] = 1
                        x_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[2]
                        y_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[3]
                        z_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[4]
                        # --b1--
                        active_event_b1 = active_event_list_per_class[1]
                        se_label[frame_ind, 2, active_event_b1[0]] = 1
                        x_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[2]
                        y_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[3]
                        z_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[4]
                    else:  # if ov with more than 2 sources from the same class
                        # ----c0
                        active_event_c0 = active_event_list_per_class[0]
                        se_label[frame_ind, 3, active_event_c0[0]] = 1
                        x_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[2]
                        y_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[3]
                        z_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[4]
                        # ----c1
                        active_event_c1 = active_event_list_per_class[1]
                        se_label[frame_ind, 4, active_event_c1[0]] = 1
                        x_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[2]
                        y_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[3]
                        z_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[4]
                        # ----c2
                        active_event_c2 = active_event_list_per_class[2]
                        se_label[frame_ind, 5, active_event_c2[0]] = 1
                        x_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[2]
                        y_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[3]
                        z_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[4]

                elif active_event[0] != active_event_list[i + 1][0]:  # if the next is not the same class
                    if len(active_event_list_per_class) == 1:  # if no ov from the same class
                        # a0----
                        active_event_a0 = active_event_list_per_class[0]
                        se_label[frame_ind, 0, active_event_a0[0]] = 1
                        x_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[2]
                        y_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[3]
                        z_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[4]
                    elif len(active_event_list_per_class) == 2:  # if ov with 2 sources from the same class
                        # --b0--
                        active_event_b0 = active_event_list_per_class[0]
                        se_label[frame_ind, 1, active_event_b0[0]] = 1
                        x_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[2]
                        y_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[3]
                        z_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[4]
                        # --b1--
                        active_event_b1 = active_event_list_per_class[1]
                        se_label[frame_ind, 2, active_event_b1[0]] = 1
                        x_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[2]
                        y_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[3]
                        z_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[4]
                    else:  # if ov with more than 2 sources from the same class
                        # ----c0
                        active_event_c0 = active_event_list_per_class[0]
                        se_label[frame_ind, 3, active_event_c0[0]] = 1
                        x_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[2]
                        y_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[3]
                        z_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[4]
                        # ----c1
                        active_event_c1 = active_event_list_per_class[1]
                        se_label[frame_ind, 4, active_event_c1[0]] = 1
                        x_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[2]
                        y_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[3]
                        z_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[4]
                        # ----c2
                        active_event_c2 = active_event_list_per_class[2]
                        se_label[frame_ind, 5, active_event_c2[0]] = 1
                        x_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[2]
                        y_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[3]
                        z_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[4]
                    active_event_list_per_class = []

    label_mat = np.stack((se_label, x_label, y_label, z_label), axis=2)  # [nb_frames, 6, 4(=act+XYZ), max_classes]
    return label_mat


            


