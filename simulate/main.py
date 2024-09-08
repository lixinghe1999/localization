import librosa
from torch.utils.data import Dataset, ConcatDataset
import os
from simulator import HRTF_simulator, ISM_simulator
from Freesound_selection import Freesound_dataset
import numpy as np
class TIMIT_dataset(Dataset):
    def __init__(self, root='TIMIT', split='TRAIN', sr=16000):
        self.root_dir = os.path.join(root, split)
        self.data = []
        self.sr = sr
        for DR in os.listdir(self.root_dir):
            for P in os.listdir(os.path.join(self.root_dir, DR)):
                folder = os.path.join(self.root_dir, DR, P)
                files = os.listdir(folder)
                files = [os.path.join(folder, file) for file in files if file.endswith('.WAV')]
                self.data += (files)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        audio = librosa.load(self.data[idx], sr=self.sr)[0]
        return audio, 0
class ESC50(Dataset):
    def __init__(self, root='ESC-50-master', split='TRAIN', sr=16000):
        root = os.path.join(root, 'audio')
        self.data = []
        self.sr = sr
        audio_list = os.listdir(root)
        if split == 'TRAIN':
            audio_list = audio_list[:int(len(audio_list)* 0.8)]
        else:
            audio_list = audio_list[int(len(audio_list)* 0.8):]
        for audio in audio_list:
            class_idx = int(audio[:-4].split('-')[-1])
            if class_idx < 100:
                self.data.append(os.path.join(root, audio))
        self.class_names = ['dog', 'rooster', 'pig', 'cow', 'frog', 'cat', 'hen', 'insects', 'sheep', 'crow',
                            'rain', 'sea_waves', 'crackling_fire', 'crickets', 'chirping_birds', 'water_drops', 'wind', 'pouring_water', 'toilet_flush', 'thunderstorm',
                            'crying_baby', 'sneezing', 'clapping', 'breathing', 'coughing', 'footsteps', 'laughing', 'brushing_teeth', 'snoring', 'drinking_sipping',
                            'door knock', 'mouse click', 'keyboard typing', 'door_wood_knock', 'can_opening', 'washing_machine', 'vacuum_cleaner', 'clock_alarm', 'clock_tick', 'glass_breaking',
                            'helicopter', 'chainsaw', 'siren', 'car_horn', 'engine', 'train', 'church_bells', 'airplane', 'fireworks', 'hand_saw']
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        file_name = self.data[idx]
        audio = librosa.load(file_name, sr=self.sr)[0]
        class_idx = int(file_name[:-4].split('-')[-1])
        return audio, class_idx


def Mixed_dataset(split):
    timit = TIMIT_dataset(split=split)
    esc50 = ESC50(split=split)
    datasets_list = [timit, esc50]
    for dataset in datasets_list:
        print(len(dataset))
    dataset = ConcatDataset(datasets_list)
    return dataset

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='TIMIT', choices=['TIMIT', 'ESC50', 'Freesound', 'mixed'])
    parser.add_argument('--root', type=str, default='TIMIT', choices=['TIMIT', 'ESC-50-master'])
    parser.add_argument('--save_folder', type=str, required=True)
    parser.add_argument('--method', type=str, default='hrtf', choices=['hrtf', 'ism'])
    parser.add_argument('--max_source', type=int, default=1)
    parser.add_argument('--min_diff', type=int, default=45)
    parser.add_argument('--num_data', type=int, default=500)
    parser.add_argument('--sr', type=int, default=16000)

    args = parser.parse_args()
    if args.dataset == 'TIMIT':
        train_dataset = TIMIT_dataset(root='TIMIT', split='TRAIN')
        test_dataset = TIMIT_dataset(root='TIMIT', split='TEST')
    elif args.dataset == 'ESC50':
        train_dataset = ESC50(root='ESC-50-master', split='TRAIN')
        test_dataset = ESC50(root='ESC-50-master', split='TEST')
    elif args.dataset == 'Freesound':
        train_dataset = Freesound_dataset(split='dev')
        test_dataset = Freesound_dataset(split='eval')
    else:
        train_dataset = Mixed_dataset('TRAIN')
        test_dataset = Mixed_dataset('TEST')
        

    # HRTF_folder = "HRTF-Database/SOFA"
    # train_folder = args.save_folder + '/earphone/{}_{}/{}'.format(args.dataset, args.max_source, 'train')
    # os.makedirs(train_folder, exist_ok=True)
    # test_folder = args.save_folder + '/earphone/{}_{}/{}'.format(args.dataset, args.max_source, 'test')
    # os.makedirs(test_folder, exist_ok=True)

    # simulator = HRTF_simulator(HRTF_folder, 'TRAIN', sr=args.sr)
    # simulator.simulate_all(train_folder, train_dataset, num_data_per_user=args.num_data, max_source=args.max_source, min_diff=args.min_diff)

    # simulator = HRTF_simulator(HRTF_folder, 'TEST', sr=args.sr)
    # simulator.simulate_all(test_folder, test_dataset, num_data_per_user=args.num_data, 
    #                        max_source=args.max_source, min_diff=args.min_diff)

    train_folder = args.save_folder + '/smartglass/{}_{}/{}'.format(args.dataset, args.max_source, 'train')
    os.makedirs(train_folder, exist_ok=True)
    test_folder = args.save_folder + '/smartglass/{}_{}/{}'.format(args.dataset, args.max_source, 'test')
    os.makedirs(test_folder, exist_ok=True)

    simulator = ISM_simulator()
    simulator.simulate_all(train_folder, train_dataset, num_data=None, max_source=args.max_source, min_diff=args.min_diff)
    simulator.simulate_all(test_folder, test_dataset, num_data=None, max_source=args.max_source, min_diff=args.min_diff)
