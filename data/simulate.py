import librosa
from torch.utils.data import Dataset
import os
from hrtf_simulate import simulate_all as simulate_all_hrtf
from ism_simulate import simulate_all as simulate_all_ism
from hybrid_simulate import simulate_all as simulate_all_hybrid
import numpy as np
class TIMIT_dataset(Dataset):
    def __init__(self, root, split, sr=16000):
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
        return audio

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='TIMIT')
    parser.add_argument('--method', type=str, default='hybrid')
    parser.add_argument('--max_source', type=int, default=2)
    parser.add_argument('--min_diff', type=int, default=45)
    parser.add_argument('--num_data', type=int, default=2000)

    args = parser.parse_args()
    HRTF_folder = "HRTF-Database/SOFA"
    for split in ['TRAIN', 'TEST']:
        dataset = TIMIT_dataset(args.root, split)
        save_folder = args.root + '/{}_{}_{}'.format(args.method, args.max_source, split)
        os.makedirs(save_folder, exist_ok=True)
        f = open(save_folder + '/label.json', 'w')
        if args.method == 'hrtf':
            simulate_all_hrtf(HRTF_folder, split, save_folder, dataset, num_data_per_user=args.num_data, max_source=args.max_source, min_diff=args.min_diff)
        elif args.method == 'ism':
            simulate_all_ism(save_folder, dataset, num_data=args.num_data * 40, max_source=args.max_source, min_diff=args.min_diff)
        elif args.method == 'hybrid':
            simulate_all_hybrid(HRTF_folder, split, save_folder, dataset, num_data_per_user=args.num_data, max_source=args.max_source, min_diff=args.min_diff)
        else:
            raise NotImplementedError