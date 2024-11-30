import os
from simulator import HRTF_simulator, ISM_simulator
from audio_dataset import dataset_parser
import numpy as np

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='FUSS', choices=['TIMIT', 'VCTK', 'NIGENS', 'AudioSet', 'FUSS'])
    parser.add_argument('--save_folder', type=str, required=False, default='../dataset')
    parser.add_argument('--device', type=str, default='smartglass', choices=['earphone', 'smartglass', 'dualdevice'])
    parser.add_argument('--max_source', type=int, default=2)
    parser.add_argument('--min_diff', type=int, default=45)
    parser.add_argument('--num_data', type=int, default=20000)
    parser.add_argument('--sr', type=int, default=16000)

    args = parser.parse_args()
    train_dataset, test_dataset = dataset_parser(args.dataset, '../dataset')  
    train_folder = args.save_folder + '/{}/{}_{}/{}'.format(args.device, args.dataset, args.max_source, 'train')
    os.makedirs(train_folder, exist_ok=True)
    test_folder = args.save_folder + '/{}/{}_{}/{}'.format(args.device, args.dataset, args.max_source, 'test')
    os.makedirs(test_folder, exist_ok=True)      
    if args.device == 'earphone':
        HRTF_folder = "HRTF-Database/SOFA"
        simulator = ISM_simulator(mic_array=np.c_[[ 0.08,  0.0, 0.0],
                                                    [ -0.08,  0.0, 0.0],])
        simulator.init_HRTF(HRTF_folder, user=range(40))
        simulator.simulate_all(train_folder, train_dataset, num_data=args.num_data, max_source=args.max_source, min_diff=args.min_diff)
        
        simulator.init_HRTF(HRTF_folder, user=range(40, 48))
        simulator.simulate_all(test_folder, test_dataset, num_data= None if args.num_data is None else args.num_data//5, max_source=args.max_source, min_diff=args.min_diff)
        
        # simulator = HRTF_simulator(HRTF_folder, 'TRAIN', sr=args.sr)
        # simulator.simulate_all(train_folder, train_dataset, num_data_per_user=args.num_data, max_source=args.max_source, min_diff=args.min_diff)

        # simulator = HRTF_simulator(HRTF_folder, 'TEST', sr=args.sr)
        # simulator.simulate_all(test_folder, test_dataset, num_data_per_user=args.num_data, 
        #                        max_source=args.max_source, min_diff=args.min_diff)
    else:
        simulator = ISM_simulator(args.device)
        simulator.simulate_all(train_folder, train_dataset, num_data=args.num_data, max_source=args.max_source, min_diff=args.min_diff)
        simulator.simulate_all(test_folder, test_dataset, num_data= None if args.num_data is None else args.num_data//5, max_source=args.max_source, min_diff=args.min_diff)
