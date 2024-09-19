DHH_sounds = ["Microwave", "Hazard alarm", "Baby crying", "Alarm clock", "Cutlery", "Water running", "Door knock", "Cat Meow", "Dishwasher", 
          "Car horn", "Phone ringing", "Washer/dryer", "Bird chirp", "Vehicle", "Door open/close", "Doorbell", "Dog bark", "Kettle whistle", 
          "Siren", "Cough", "Snore", "Speech"]
from torch.utils.data import Dataset, ConcatDataset
import os
import numpy as np
import pandas as pd
import librosa

class FSD50K_dataset(Dataset):
    def __init__(self, root='FSD50K', split='eval', sr=16000):
        self.root_dir = os.path.join(root, 'FSD50K.{}_audio'.format(split))
        self.label = os.path.join(root, 'FSD50K.ground_truth', '{}.csv'.format(split))
        self.labels = pd.read_csv(self.label)

        self.labels_vocabulary = pd.read_csv(os.path.join(root, 'FSD50K.ground_truth', 'vocabulary.csv'))
        # convert the vocabulary to dict
        self.vocabulary = {}
        for i in range(len(self.labels_vocabulary)):
            self.vocabulary[self.labels_vocabulary.iloc[i, 1]] = self.labels_vocabulary.iloc[i, 0]

        self.class_name = ['Speech', 'Microwave_oven', 'Alarm', 'Crying_and_sobbing', 'Clock', 'Cutlery_and_silverware', 'Water', 'Door', 'Cat', 
                               'Dishes_and_pots_and_pans', 'Vehicle_horn_and_car_horn_and_honking', 'Telephone', 'Bird', 'Vehicle', 
                               'Sliding_door', 'Doorbell', 'Dog', 'Siren', 'Cough']
        # filter the labels_vocabulary (class1/class2/class3) by containing the selected_vocabulary
        vocab_num = {}
        self.selected_labels = []
        # only keep fname and labels
        self.labels = self.labels[['fname', 'labels']]
        for i, vocab in enumerate(self.class_name):
            vocab_labels = self.labels[self.labels['labels'].str.contains(vocab)]
            vocab_labels.loc[:, 'labels'] = i 
            vocab_num[vocab] = len(vocab_labels)
            # convert to list of (file, class)
            vocab_labels = vocab_labels.to_numpy()
            self.selected_labels += vocab_labels.tolist()
        print(vocab_num)
        self.labels = self.selected_labels
        self.sr = sr
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sound, class_idx = self.labels[idx]
        sound_file = os.path.join(self.root_dir, str(sound) + '.wav')
        audio, _ = librosa.load(sound_file, sr=self.sr)
        return audio, class_idx
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
        self.class_name = ['speech']
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
        self.class_name = ['dog', 'rooster', 'pig', 'cow', 'frog', 'cat', 'hen', 'insects', 'sheep', 'crow',
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


def dataset_parser(dataset, relative_path):
    if dataset == 'TIMIT':
        root = os.path.join(relative_path, 'TIMIT')
        train_dataset = TIMIT_dataset(root=root, split='TRAIN')
        test_dataset = TIMIT_dataset(root=root, split='TEST')
    elif dataset == 'ESC50':
        root = os.path.join(relative_path, 'ESC-50-master')
        train_dataset = ESC50(root=root, split='TRAIN')
        test_dataset = ESC50(root=root, split='TEST')
    elif dataset == 'FSD50K':
        root = os.path.join(relative_path, 'FSD50K')
        train_dataset = FSD50K_dataset(root=root, split='dev')
        test_dataset = FSD50K_dataset(root=root, split='eval')

    return train_dataset, test_dataset

if __name__ == '__main__':
    train_dataset, test_dataset = dataset_parser('ESC50')
    print(len(train_dataset), len(test_dataset))
