'''
Audioset Strong frame dataset
'''
from torch.utils.data import Dataset, ConcatDataset
import os
import numpy as np
import pandas as pd
import librosa
from PIL import Image

class AudioSet_dataset(Dataset):
    def __init__(self, root='audioset', split='eval', sr=16000):
        self.image_dir = os.path.join(root, 'audioset_{}_strong_images'.format(split))
        self.audio_dir = os.path.join(root, 'audioset_{}_strong_audios'.format(split))
        label_file = os.path.join(root, 'audioset_{}_strong.tsv'.format(split))

        label = pd.read_csv(label_file, sep='\t')
        labels = {}
        self.label_map = {}
        for row in label.itertuples():
            segment_id = row.segment_id 
            if row.label not in self.label_map:
                self.label_map[row.label] = len(self.label_map)
            if segment_id not in labels:
                labels[segment_id] = []
            labels[segment_id].append([row.start_time_seconds, row.end_time_seconds, row.label])
        self.num_classes = len(self.label_map)
        # clip-level
        self.clip_labels = []; self.frame_labels = []
        for segment_id in labels:
            clip_label = np.zeros(self.num_classes)
            frame_label = np.zeros((10, self.num_classes))
            for start, end, label in labels[segment_id]:
                clip_label[self.label_map[label]] = 1
                start_frame = int(start)
                end_frame = int(end)
                frame_label[start_frame:end_frame, self.label_map[label]] = 1
            self.frame_labels.append([segment_id, frame_label])
            self.clip_labels.append([segment_id, clip_label])
        self.sr = sr

    def __len__(self):
        return len(self.clip_labels)

    def __getitem__(self, idx): 
        segment_id, frame_label = self.clip_labels[idx]
        audio_file = os.path.join(self.audio_dir, segment_id + '.flac')
        image_file = os.path.join(self.image_dir, segment_id + '.jpg')

        audio, _ = librosa.load(audio_file, sr=self.sr)
        image = np.array(Image.open(image_file))
        return audio, image, frame_label

if __name__ == '__main__':
    dataset = AudioSet_dataset('../dataset/audioset')
    print(len(dataset))
    for i in range(1):
        audio, image, frame_label = dataset[i]
        print(audio.shape, image.shape, frame_label.shape)

        


