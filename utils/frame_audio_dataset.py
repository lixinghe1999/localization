'''
Audioset Strong frame dataset
'''
from torch.utils.data import Dataset, ConcatDataset
import os
import numpy as np
import pandas as pd
import librosa
from PIL import Image
import json

class AudioSet_dataset(Dataset):
    def __init__(self, root='audioset', split='eval', sr=16000, duration=10, frame_duration=0.5):
        self.image_dir = os.path.join(root, 'audioset_{}_strong_images'.format(split))
        self.audio_dir = os.path.join(root, 'audioset_{}_strong_audios'.format(split))
        self.embeddings_dir = os.path.join(root, 'audioset_{}_strong_embeddings'.format(split))
        label_file = os.path.join(root, 'audioset_{}_strong.tsv'.format(split))

        label = pd.read_csv(label_file, sep='\t')
        labels = {}
        self.label_map = {}; self.class_name = []
        for row in label.itertuples():
            segment_id = row.segment_id 
            if row.label not in self.label_map:
                self.label_map[row.label] = len(self.label_map)
                self.class_name.append(row.label)
            if segment_id not in labels:
                labels[segment_id] = []
            labels[segment_id].append([row.start_time_seconds, row.end_time_seconds, row.label])
        self.num_classes = len(self.label_map)
        print('Number of classes:', self.num_classes)
        ontology = json.load(open(os.path.join(root, 'ontology.json')))
        self.ontology = {}
        for item in ontology:
            self.ontology[item['id']] = item['name']

        self.sr = sr
        self.duration = duration
        # clip-level
        self.clip_labels = []; self.frame_labels = []
        self.num_frames_per_clip = int(duration / frame_duration)
        for segment_id in labels:
            clip_label = np.zeros(self.num_classes, dtype=np.float32)
            frame_label = np.zeros((self.num_frames_per_clip, self.num_classes), dtype=np.float32)
            for start, end, label in labels[segment_id]:
                clip_label[self.label_map[label]] = 1
                start_frame = int(start/ frame_duration)
                end_frame = int(end/ frame_duration)
                frame_label[start_frame:end_frame, self.label_map[label]] = 1
            self.frame_labels.append([segment_id, frame_label])
            self.clip_labels.append([segment_id, clip_label])

    def filter_modal(self, modal):
        keep_index = []
        for i in range(self.__len__()):
            segment_id, _ = self.clip_labels[i]

            audio_file = os.path.join(self.audio_dir, segment_id + '.flac')
            image_file = os.path.join(self.image_dir, segment_id + '.jpg')
            embeddings_file = os.path.join(self.embeddings_dir, segment_id + '.npy')
            if 'audio' in modal and not os.path.exists(audio_file):
                continue
            if 'image' in modal and not os.path.exists(image_file):
                continue
            if 'embeddings' in modal and not os.path.exists(embeddings_file):
                continue
            keep_index.append(i)
        self.clip_labels = [self.clip_labels[i] for i in keep_index]
        self.frame_labels = [self.frame_labels[i] for i in keep_index]
        
    def __len__(self):
        return len(self.clip_labels)

    def __getitem__(self, idx): 
        segment_id, label = self.frame_labels[idx]
        _, clip_label = self.clip_labels[idx]

        audio_file = os.path.join(self.audio_dir, segment_id + '.flac')
        # duration = librosa.get_duration(path=audio_file)
        audio, _ = librosa.load(audio_file, sr=self.sr, duration=self.duration)
        if len(audio) < self.duration * self.sr:
            audio = np.pad(audio, (0, self.duration*self.sr - len(audio)))
        else:
            audio = audio[:self.duration*self.sr]

        # image_file = os.path.join(self.image_dir, segment_id + '.jpg')
        embeddings_file = os.path.join(self.embeddings_dir, segment_id + '.npy')

        # image = np.array(Image.open(image_file))
        image = np.load(embeddings_file).astype(np.float32)

        return (audio, image), label

if __name__ == '__main__':
    import torchmetrics
    import torch
    dataset = AudioSet_dataset('../dataset/audioset')
    dataset.filter_modal(['audio',])
    print(len(dataset))

    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    for i, data in enumerate(dataset_loader):
        (audio, clip_label), frame_label = data


        accuracy = torchmetrics.AveragePrecision(task='multilabel', num_labels=dataset.num_classes, average='macro')
    
        pseduo_frame_label = torch.repeat_interleave(clip_label, 10, dim=0)
        # rint(pseduo_frame_label[:, :2], clip_label[:, :2])
        frame_label = frame_label.reshape(-1, dataset.num_classes)
        # print(frame_label[:, :2])
        acc = accuracy(pseduo_frame_label[:, :], frame_label.long()[:, :])
        print(acc)
        if i > 0:
            break



