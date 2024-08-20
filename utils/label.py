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



def region_wise_separation(doas, ranges, file_names, config):
    number_of_regions = config['classifier']['num_regions']
    way_of_region = config['classifier']['way_of_region']

    if way_of_region == 'azimuth':
        regions = np.linspace(config['classifier']['min_azimuth'], config['classifier']['max_azimuth'], number_of_regions + 1)
    elif way_of_region == 'distance':
        regions = np.linspace(0, config['classifier']['max_range'], number_of_regions + 1)
    else:
        raise NotImplementedError
    
    spatial_audio = np.zeros((number_of_regions, 16000 * config['duration']), dtype=np.float32)
    for doa, range, file_name in zip(doas, ranges, file_names):
        if way_of_region == 'azimuth':
            region = np.digitize(doa[0], regions)
        elif way_of_region == 'distance':
            region = np.digitize(range, regions)
        if region == number_of_regions:
            region = number_of_regions - 1

        mono_audio, _ = librosa.load(os.path.join(config['source_dataset'], file_name), sr=16000, duration=config['duration'])
        if len(mono_audio) < 16000 * config['duration']:
            mono_audio = np.pad(mono_audio, (0, 16000 * config['duration'] - len(mono_audio)))
        spatial_audio[region] += mono_audio.astype(np.float32)
    return spatial_audio