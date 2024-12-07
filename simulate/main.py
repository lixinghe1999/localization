import os
from simulator import simulator
from audio_dataset import dataset_parser
import numpy as np

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='NIGENS', choices=['TIMIT', 'VCTK', 'NIGENS', 'AudioSet', 'FUSS', 'FSD50K'])
    parser.add_argument('--save_folder', type=str, required=False, default='../dataset')
    parser.add_argument('--device', type=str, default='earphone', choices=['earphone', 'smartglass'])
    parser.add_argument('--max_source', type=int, default=2)
    parser.add_argument('--num_data', type=int, default=5000)
    parser.add_argument('--sr', type=int, default=16000)

    args = parser.parse_args()
    train_dataset, test_dataset = dataset_parser(args.dataset, '../dataset/audio')  
    train_folder = args.save_folder + '/{}/{}_{}/{}'.format(args.device, args.dataset, args.max_source, 'train')
    os.makedirs(train_folder, exist_ok=True)
    test_folder = args.save_folder + '/{}/{}_{}/{}'.format(args.device, args.dataset, args.max_source, 'test')
    os.makedirs(test_folder, exist_ok=True) 
    simulator_ = simulator(args.device)

    simulator_.simulate_all(train_folder, train_dataset, num_data=args.num_data, max_source=args.max_source)
    simulator_.simulate_all(test_folder, test_dataset, num_data= None if args.num_data is None else args.num_data//5, max_source=args.max_source)
