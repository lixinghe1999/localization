import numpy as np

def filter_label(labels, max_azimuth=90, min_azimuth=-90):
    new_labels = []
    for label in labels:
        doa_degree = label['doa_degree']
        for doa in doa_degree:
            if type(doa) == list: # azimuth + elevation
                if doa[0] >= min_azimuth and doa[0] <= max_azimuth:
                    new_labels.append(label)
            else:
                if doa >= min_azimuth and doa <= max_azimuth:
                    new_labels.append(label)
    return new_labels

# Gaussian, distance and Cartesian encoding are for SSL
def Gaussian(doas, ranges, config):
    '''
    Output format: Gaussian distribution of doa, where the peak refer to the doa
    '''
    N = int(config['classifier']['max_azimuth'] - config['classifier']['min_azimuth'])
    azimuth = []
    y = np.linspace(config['classifier']['min_azimuth'], config['classifier']['max_azimuth'], N)
    for d, r in zip(doas, ranges):
        azimuth.append(np.exp(-((y - d[0]) ** 2) / (2 * 10 ** 2)))
    azimuth = np.max(np.array(azimuth), axis=0).astype(np.float32)

    N = int(config['classifier']['max_elevation'] - config['classifier']['min_elevation'])
    elevation = []
    y = np.linspace(config['classifier']['min_elevation'], config['classifier']['max_elevation'], N)
    for d, r in zip(doas, ranges):
        elevation.append(np.exp(-((y - d[1]) ** 2) / (2 * 10 ** 2)))
    elevation = np.max(np.array(elevation), axis=0).astype(np.float32)

    return (azimuth, elevation)
def Distance(doas, ranges, config):
    max_range = config['classifier']['max_range']
    ranges = np.array(ranges) / max_range
    ranges = np.clip(ranges, 0, 1).astype(np.float32)
    return ranges
def Cartesian(doas, ranges, config):
    '''
    Directly output the xyz
    '''
    xyz = []
    for doa, r in zip(doas, ranges):
        x = r * np.cos(np.deg2rad(doa[0])) * np.cos(np.deg2rad(doa[1]))
        y = r * np.sin(np.deg2rad(doa[0])) * np.cos(np.deg2rad(doa[1]))
        z = r * np.sin(np.deg2rad(doa[1]))
        xyz.append([x, y, z])
    xyz = np.array(xyz).astype(np.float32).reshape(-1)
    return xyz

def get_labels_for_file(_desc_file, _nb_label_frames, _nb_unique_classes=13):
    """
    Reads description file and returns classification based SED labels and regression based DOA labels

    :param _desc_file: metadata description file
    :return: label_mat: of dimension [nb_frames, 3*max_classes], max_classes each for x, y, z axis,
    """

    # If using Hungarian net set default DOA value to a fixed value greater than 1 for all axis. We are choosing a fixed value of 10
    # If not using Hungarian net use a deafult DOA, which is a unit vector. We are choosing (x, y, z) = (0, 0, 1)
    se_label = np.zeros((_nb_label_frames, _nb_unique_classes))
    x_label = np.zeros((_nb_label_frames, _nb_unique_classes))
    y_label = np.zeros((_nb_label_frames, _nb_unique_classes))
    z_label = np.zeros((_nb_label_frames, _nb_unique_classes))

    for frame_ind, active_event_list in _desc_file.items():
        if frame_ind < _nb_label_frames:
            for active_event in active_event_list:
                se_label[frame_ind, active_event[0]] = 1
                x_label[frame_ind, active_event[0]] = active_event[2]
                y_label[frame_ind, active_event[0]] = active_event[3]
                z_label[frame_ind, active_event[0]] = active_event[4]

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