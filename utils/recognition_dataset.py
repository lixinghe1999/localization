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

    def __init__(self, root='audioset', modality=[], split='eval', sr=16000, duration=10, frame_duration=0.1, label_level='clip'):
        self.image_dir = os.path.join(root, 'audioset_{}_strong_images'.format(split))
        self.audio_dir = os.path.join(root, 'audioset_{}_strong_audios'.format(split))
        self.text_dir = os.path.join(root, 'audioset_{}_strong_text'.format(split))
        self.embeddings_dir = os.path.join(root, 'audioset_{}_strong_embeddings'.format(split))
        label_file = os.path.join(root, 'audioset_{}_strong.tsv'.format(split))

        label = pd.read_csv(label_file, sep='\t')
        labels = {}
        self.label_map = {}
        for row in label.itertuples():
            start_seconds = row.start_time_seconds
            end_seconds = row.end_time_seconds
            label = row.label
            segment_id = row.segment_id 
            if row.label not in self.label_map:
                self.label_map[row.label] = len(self.label_map)
            if segment_id not in labels:
                labels[segment_id] = []
            
            labels[segment_id].append([start_seconds, end_seconds, label])
        self.num_classes = len(self.label_map)
        print('Number of classes:', self.num_classes)

        # only keep 10 classes
        num_segment = len(labels)
        select_classes = range(10)
        for segment_id in list(labels.keys()):
            new_labels = []
            for start, end, label in labels[segment_id]:
                label_idx = self.label_map[label]
                if label_idx in select_classes:
                    new_labels.append([start, end, label])
            if len(new_labels) > 0:
                labels[segment_id] = new_labels   
            else:
                del labels[segment_id]
        print('select classes from', select_classes, 'Number of segments:', len(labels), 'before filtering', num_segment)
        

        ontology = pd.read_csv(os.path.join(root, 'mid_to_display_name.tsv'), sep='\t', names=['mid', 'display_name'])
        self.ontology = {}
        for row in ontology.itertuples():
            self.ontology[row.mid] = row.display_name
        self.class_map = ['unknown'] * self.num_classes
        for k in self.label_map:
            cls_idx = self.label_map[k]
            if k in self.ontology:
                self.class_map[cls_idx] = self.ontology[k]

        self.sr = sr
        self.duration = duration
        self.frame_duration = frame_duration
        self.modality = modality
        self.label_level = label_level

        self.clip_labels = []; self.frame_labels = []
        self.num_frames_per_clip = int(duration / self.frame_duration)
        for segment_id in labels:
            clip_label = np.zeros(self.num_classes, dtype=np.float32)
            frame_label = np.zeros((self.num_frames_per_clip, self.num_classes), dtype=np.float32)
            for start, end, label in labels[segment_id]:
                clip_label[self.label_map[label]] = 1
                start_frame = int(start/ self.frame_duration)
                end_frame = int(end/ self.frame_duration)
                frame_label[start_frame:end_frame, self.label_map[label]] = 1
            self.frame_labels.append([segment_id, frame_label])
            self.clip_labels.append([segment_id, clip_label])
    
        self.filter_modal(modality)
    def filter_modal(self, modal):
        keep_index = []
        for i in range(self.__len__()):
            segment_id, _ = self.clip_labels[i]

            audio_file = os.path.join(self.audio_dir, segment_id + '.flac')
            image_file = os.path.join(self.image_dir, segment_id + '.jpg')
            embeddings_file = os.path.join(self.embeddings_dir, segment_id + '.npy')
            text_file = os.path.join(self.text_dir, segment_id + '.json')
            if 'audio' in modal and not os.path.exists(audio_file):
                continue
            if 'image' in modal and not os.path.exists(image_file):
                continue
            if 'embeddings' in modal and not os.path.exists(embeddings_file):
                continue
            if 'text' in modal and not os.path.exists(text_file):
                continue
            keep_index.append(i)
        print('Number of samples before filtering:', len(self.clip_labels))
        print('Number of samples after filtering:', len(keep_index))
        self.clip_labels = [self.clip_labels[i] for i in keep_index]
        self.frame_labels = [self.frame_labels[i] for i in keep_index]
        
    def __len__(self):
        return len(self.clip_labels)

    def __getitem__(self, idx): 
        output_dict = {}
        if self.label_level == 'frame':
            segment_id, label = self.frame_labels[idx]
        else:
            segment_id, label = self.clip_labels[idx]

        audio_file = os.path.join(self.audio_dir, segment_id + '.flac')
        # duration = librosa.get_duration(path=audio_file)
        audio, _ = librosa.load(audio_file, sr=self.sr, duration=self.duration)
        if len(audio) < self.duration * self.sr:
            audio = np.pad(audio, (0, self.duration*self.sr - len(audio)))
        else:
            audio = audio[:self.duration*self.sr]
        output_dict['audio'] = audio
        output_dict['cls_label'] = label
        if 'embeddings' in self.modality:
            embeddings_file = os.path.join(self.embeddings_dir, segment_id + '.npy')
            embeddings = np.load(embeddings_file).astype(np.float32)
            output_dict['embeddings'] = embeddings
        if 'image' in self.modality:
            image_file = os.path.join(self.image_dir, segment_id + '.jpg')
            image = np.array(Image.open(image_file))
            output_dict['image'] = image
        if 'text' in self.modality:
            text_file = os.path.join(self.text_dir, segment_id + '.json')
            with open(text_file, 'r') as f:
                text = json.load(f)
            output_dict['text'] = text
        return output_dict

def dataset_sample(dataset):
    count = 0
    inactive_frames = []; single_class_frames = []; overlap_frames = []
    for segment_id, label in dataset.frame_labels:
        print(segment_id, label.shape)
        inactive_frame = np.sum(label, axis=1) == 0
        single_class_frame = np.sum(label, axis=1) == 1
        overlap_frame = np.sum(label, axis=1) > 1

        inactive_frames.append(np.mean(inactive_frame))
        single_class_frames.append(np.mean(single_class_frame))
        overlap_frames.append(np.mean(overlap_frame))
        if np.mean(single_class_frame) == 1:
            count += 1
    print('Inactive frames:', np.mean(inactive_frames))
    print('Single class frames:', np.mean(single_class_frames))
    print('Overlap frames:', np.mean(overlap_frames))
    print('Number of single label clips:', count, 'total clips:', len(dataset.frame_labels))
if __name__ == '__main__':
    dataset = AudioSet_dataset('dataset/audioset', split='eval', modality=['audio'], label_level='clip')
    print(len(dataset))
    dataset_sample(dataset)




