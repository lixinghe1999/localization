'''
Create separation dataset for FSD50K
Difference to FUSS: We know the class label, so that it is ok to do semantichearing
'''
import sys
sys.path.append('..')
from utils.recognition_dataset import FSD50K_dataset, Singleclass_dataset, ESC50_dataset
import numpy as np
import os
import soundfile as sf
import json
from tqdm import tqdm

for split in ['dev', 'eval']:
    # dataset_dir = './separation/FSD50K/' + split
    # os.makedirs(dataset_dir, exist_ok=True)
    # dataset = FSD50K_dataset('./FSD50K', split=split)
    
    dataset_dir = './separation/ESC50/' + split
    os.makedirs(dataset_dir, exist_ok=True)
    dataset = ESC50_dataset('./audio/ESC-50-master', split=split)
    singleclass_dataset, classes_index = Singleclass_dataset(dataset, keep_classes=[0, 5, 10, 16, 20, 25, 30, 37, 43, 46])
    num_repeat = 25
    for i in tqdm(range(len(singleclass_dataset))):
        output_dict = singleclass_dataset.__getitem__(i)
        audio = output_dict['audio']
        label = output_dict['cls_label']
        label = np.where(label == 1)[0][0]

        for j in range(num_repeat):
            save_folder = os.path.join(dataset_dir, str(i) + '_' + str(j))
            os.makedirs(save_folder, exist_ok=True)
            # save: 1) audio, 2) noise, 3) label
            sf.write(os.path.join(save_folder, 'audio.wav'), audio, 16000)
            remained_index = list(set(range(len(dataset))) - set(classes_index[label]))
            random_noise_index = np.random.choice(remained_index)

            output_dict = dataset.__getitem__(random_noise_index)
            noise = output_dict['audio']; noise_label = output_dict['cls_label']; noise_label = np.where(noise_label == 1)[0]
            sf.write(os.path.join(save_folder, 'noise.wav'), noise, 16000)

            meta_json = {'label': label.tolist(), 'noise_label': noise_label.tolist()}
            with open(os.path.join(save_folder, 'meta.json'), 'w') as f:
                json.dump(meta_json, f)
