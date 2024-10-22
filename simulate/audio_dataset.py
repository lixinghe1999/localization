DHH_sounds = ["Microwave", "Hazard alarm", "Baby crying", "Alarm clock", "Cutlery", "Water running", "Door knock", "Cat Meow", "Dishwasher", 
          "Car horn", "Phone ringing", "Washer/dryer", "Bird chirp", "Vehicle", "Door open/close", "Doorbell", "Dog bark", "Kettle whistle", 
          "Siren", "Cough", "Snore", "Speech"]
from torch.utils.data import Dataset, ConcatDataset, random_split
import os
import numpy as np
import pandas as pd
import librosa

class NIGENS_dataset(Dataset):
    def __init__(self, root='NIGENS', split='train', sr=16000):
        self.root_dir = root
        self.data = []
        self.sr = sr
        self.class_names = ['alarm', 'baby', 'crash', 'dog', 'engine', 'femaleScream', 'femaleSpeech', 'fire', 'footsteps',
                            'knock', 'maleScream', 'maleSpeech', 'phone', 'piano',]
        for sound_class in (self.class_names):
            folder = os.path.join(self.root_dir, sound_class)
            files = os.listdir(folder)
            audio_files = [os.path.join(folder, file) for file in files if file.endswith('.wav')]
            txt_files = [os.path.join(folder, file) for file in files if file.endswith('.txt')]
            audio_files = [audio for audio in audio_files if audio + '.txt' in txt_files]
            txt_files = [audio + '.txt' for audio in audio_files]
            assert len(audio_files) == len(txt_files)
            
            sound_class = [sound_class] * len(audio_files)
            data = [(audio, txt, class_name) for audio, txt, class_name in zip(audio_files, txt_files, sound_class)]   
            if split == 'train':
                self.data += data[:int(len(data) * 0.8)]
            else:
                self.data += data[int(len(data) * 0.8):]   
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        audio, txt, class_name = self.data[idx]
        audio = librosa.load(audio, sr=self.sr)[0]
        active_frame = np.zeros(int(len(audio) / self.sr / 0.1))

        with open(txt, 'r') as f:
            lines = f.readlines()
        for line in lines:
            start, end = line.strip().split()
            start_frame = int(float(start) / 0.1)
            end_frame = int(float(end) / 0.1)
            active_frame[start_frame:end_frame] = 1
        class_idx = self.class_names.index(class_name)
        return audio, class_idx, active_frame
    


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
        return audio, 0, (None, None) 
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
        return audio, class_idx, (0, 5)
class FUSS_Reverb(Dataset):
    def __init__(self, root='FUSS_reverb', split='TRAIN', sr=16000):
        root = os.path.join(root, split)
        self.data = []
        self.sr = sr
        audio_list = os.listdir(root)
        audio_list = [audio for audio in audio_list if audio.endswith('sources')]

        self.data = []; self.annotation = []
        for audio in audio_list:
            source_dir = os.path.join(root, audio)
            source_list = os.listdir(source_dir)
            source_list = [source for source in source_list if source.startswith('foreground')]

            example_name = audio.split('_')[0]
            active_txt = example_name + '.txt'
            active_txt = os.path.join(root, active_txt)
            with open(active_txt, 'r') as f:
                lines = f.readlines()

            for i, source in enumerate(source_list):
                if source.startswith('foreground'):
                    self.data.append(os.path.join(source_dir, source))
                    self.annotation.append(lines[i].strip())

        print(len(self.data), len(self.annotation))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        file_name = self.data[idx]
        annotation = self.annotation[idx]
        start, end, _ = annotation.split()
        start, end = float(start), float(end)

        audio = librosa.load(file_name, sr=self.sr)[0]
        return audio, 0, (start, end)

def dataset_parser(dataset, relative_path):
    if dataset == 'TIMIT':
        root = os.path.join(relative_path, 'TIMIT')
        train_dataset = TIMIT_dataset(root=root, split='TRAIN')
        test_dataset = TIMIT_dataset(root=root, split='TEST')
    elif dataset == 'ESC50':
        root = os.path.join(relative_path, 'ESC-50-master')
        train_dataset = ESC50(root=root, split='TRAIN')
        test_dataset = ESC50(root=root, split='TEST')
    elif dataset == 'FUSS_Reverb':
        root = os.path.join(relative_path, 'ssdata_reverb')
        train_dataset = FUSS_Reverb(root=root, split='train')
        test_dataset = FUSS_Reverb(root=root, split='eval')
    elif dataset == 'NIGENS':
        root = os.path.join(relative_path, 'NIGENS')
        train_dataset = NIGENS_dataset(root=root, split='train')
        test_dataset = NIGENS_dataset(root=root, split='test')
    return train_dataset, test_dataset

if __name__ == '__main__':
    # train_dataset, test_dataset = dataset_parser('FUSS_Reverb', '.')

    # data = train_dataset[0]
    dataset = NIGENS_dataset()
    data = dataset[0]