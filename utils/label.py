import numpy as np
import librosa
import os
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