import librosa
from torch.utils.data import Dataset
import os
from simulator import HRTF_simulator, ISM_simulator
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
        # random dBFS from -30 to -10
        audio = audio / np.max(np.abs(audio)) * 10 ** (np.random.uniform(-35, -15) / 20)
        return audio, self.data[idx]
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
            self.data.append(os.path.join(root, audio))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        audio = librosa.load(self.data[idx], sr=self.sr)[0]
        # random dBFS from -30 to -10
        audio = audio / np.max(np.abs(audio)) * 10 ** (np.random.uniform(-35, -15) / 20)
        return audio, self.data[idx]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='TIMIT', choices=['TIMIT', 'ESC50'])
    parser.add_argument('--root', type=str, default='TIMIT', choices=['TIMIT', 'ESC-50-master'])
    parser.add_argument('--method', type=str, default='hrtf', choices=['hrtf', 'ism'])
    parser.add_argument('--max_source', type=int, default=2)
    parser.add_argument('--min_diff', type=int, default=45)
    parser.add_argument('--num_data', type=int, default=500)

    args = parser.parse_args()
    HRTF_folder = "HRTF-Database/SOFA"
    for split in ['TEST', 'TRAIN']:
        if args.dataset == 'TIMIT':
            dataset = TIMIT_dataset(root=args.root, split=split)
        elif args.dataset == 'ESC50':
            dataset = ESC50(root=args.root, split=split)

        save_folder = args.root + '/{}/hrtf_{}'.format(args.max_source, split)
        os.makedirs(save_folder, exist_ok=True)

        simulator = HRTF_simulator(HRTF_folder, split)
        simulator.simulate_all(save_folder, dataset, num_data_per_user=args.num_data, max_source=args.max_source, min_diff=args.min_diff)

        # save_folder = args.root + '/{}/ism_{}'.format(args.max_source, split)
        # os.makedirs(save_folder, exist_ok=True)

        # simulator = ISM_simulator()
        # simulator.simulate_all(save_folder, dataset, num_data=1000, max_source=args.max_source, min_diff=args.min_diff)
