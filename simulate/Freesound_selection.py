sounds = ["Microwave", "Hazard alarm", "Baby crying", "Alarm clock", "Cutlery", "Water running", "Door knock", "Cat Meow", "Dishwasher", 
          "Car horn", "Phone ringing", "Washer/dryer", "Bird chirp", "Vehicle", "Door open/close", "Doorbell", "Dog bark", "Kettle whistle", 
          "Siren", "Cough", "Snore", "Speech"]

from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
import librosa
class Freesound_dataset(Dataset):
    def __init__(self, root='FSD50K', split='eval', sr=16000):
        self.root_dir = os.path.join(root, 'FSD50K.{}_audio'.format(split))
        self.label = os.path.join(root, 'FSD50K.ground_truth', '{}.csv'.format(split))
        self.labels = pd.read_csv(self.label)

        self.labels_vocabulary = pd.read_csv(os.path.join(root, 'FSD50K.ground_truth', 'vocabulary.csv'))
        # convert the vocabulary to dict
        self.vocabulary = {}
        for i in range(len(self.labels_vocabulary)):
            self.vocabulary[self.labels_vocabulary.iloc[i, 1]] = self.labels_vocabulary.iloc[i, 0]

        self.selected_vocabulary = ['Speech', 'Microwave_oven', 'Alarm', 'Crying_and_sobbing', 'Clock', 'Cutlery_and_silverware', 'Water', 'Door', 'Cat', 
                               'Dishes_and_pots_and_pans', 'Vehicle_horn_and_car_horn_and_honking', 'Telephone', 'Bird', 'Vehicle', 
                               'Sliding_door', 'Doorbell', 'Dog', 'Siren', 'Cough']
        # filter the labels_vocabulary (class1/class2/class3) by containing the selected_vocabulary
        vocab_num = {}
        self.selected_labels = []
        # only keep fname and labels
        self.labels = self.labels[['fname', 'labels']]
        for i, vocab in enumerate(self.selected_vocabulary):
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

if __name__ == '__main__':
    dataset = Freesound_dataset()
    print(dataset.__len__())
    for i in range(10):
        audio, label = dataset.__getitem__(i)
