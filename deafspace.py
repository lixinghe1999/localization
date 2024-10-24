from utils.localization_dataset import Localization_dataset

import numpy as np

class DeafSpace():
    def __init__(self, ):
        pass
    def track(self, localization, classification):
        '''
        localization: [T, 3(xyz) * 2 (tracks)]
        classification: [T, C]
        1. visulization
        2. post-processing
        '''
    
if __name__ == '__main__':
    deafspace = DeafSpace()

    config = {
        "dataset": "smartglass",
        "train_datafolder": "/home/lixing/localization/dataset/smartglass/NIGENS_1/train",
        "test_datafolder": "/home/lixing/localization/dataset/smartglass/NIGENS_1/test",
        "cache_folder": "cache/nigens_1/",
        "encoding": "ACCDOA",
        "duration": 5,
        "frame_duration": 0.1,
        "batch_size": 64,
        "epochs": 50,
        "model": "seldnet",
        "label_type": "framewise",
        "raw_audio": False,
        'num_channel': 15,
        'num_class': 1, # no need to do classification now
        "pretrained": False,
        "test": False,
    }
    print(config)

    train_dataset = Localization_dataset(config['train_datafolder'], config)
    train_dataset._cache_(config['cache_folder'] + '/train')
    for data, audio, labels in train_dataset:
        print(data.shape, audio.shape, labels.shape)
        break
    