'''
Audioset Strong frame dataset
'''
from torch.utils.data import Dataset, ConcatDataset, Subset
import os
import numpy as np
import pandas as pd
import librosa
from PIL import Image
import json

class Ontology():
    def __init__(self, root):
        ontolog = os.path.join(root, 'ontology.json')
        ontolog = json.load(open(ontolog))
        self.ontology = {v['id']: v for v in ontolog}
        self.mid_to_name = {mid: self.ontology[mid]['name'] for mid in self.ontology}
        self.leaf_to_parent = {mid: [] for mid in self.ontology}
        for mid in self.ontology:
            for child_id in self.ontology[mid]['child_ids']:
                self.leaf_to_parent[child_id].append(mid)
        for mid in self.leaf_to_parent:
            if len(self.leaf_to_parent[mid]) == 0:
                self.leaf_to_parent[mid].append(mid)

    def convert_classes(self, vocabulary, num_inherit=1):
        '''
        vocabulary: {mid: idx}
        according to the self.leaf_to_parent, perform num_inherit times of inheritance
        output:
        {'parent1': [mid1, mid2, mid3], 'parent2': [mid4, mid5, mid6]}
        '''
        self.classes_mapping = {}
        mids = list(vocabulary.keys())
        for mid in mids:
            parent_ids = [mid]
            for i in range(num_inherit):
                new_parent_ids = []
                for parent_id in parent_ids:
                    new_parent_ids += self.leaf_to_parent[parent_id]
                parent_ids = new_parent_ids
            for parent_id in parent_ids:
                if parent_id not in self.classes_mapping:
                    self.classes_mapping[parent_id] = []
                self.classes_mapping[parent_id].append(mid)

        new_vocabulary = {}
        for i, parent_mid in enumerate(self.classes_mapping):
            for mid in self.classes_mapping[parent_mid]:
                new_vocabulary[mid] = i
        return new_vocabulary

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
        elif self.label_level == 'clip':
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
    
class FSD50K_dataset(Dataset):
    def __init__(self, root_dir, split='eval', label_level='clip', sr=16000, duration=10):
        """
        Args:
            root_dir (string): Directory with all the audio files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = os.path.join(root_dir, f'FSD50K.{split}_audio')
        self.sr = sr
        self.duration = duration
        self.label_level = label_level
        self.label = os.path.join(root_dir, f'FSD50K.ground_truth/{split}.csv')
        vocabulary = os.path.join(root_dir, 'FSD50K.ground_truth/vocabulary.csv')
        self.vocabulary = pd.read_csv(vocabulary, names=['name', 'mid'], delimiter=',')
        # convert to dictionary, key = mid, value = index
        self.vocabulary = {mid: idx for idx, mid in enumerate(self.vocabulary['mid'])}
        self.ontolog = Ontology(root_dir)
        self.vocabulary = self.ontolog.convert_classes(self.vocabulary, num_inherit=1)
        self.num_classes = max(self.vocabulary.values()) + 1 # incase multiple mid may have the same index

        self.clip_labels = []
        self.label = pd.read_csv(self.label, delimiter=',',)
        for i in range(len(self.label)):
            audio_file = os.path.join(self.root_dir, str(self.label.iloc[i, 0]) + '.wav')
            labels = self.label.iloc[i, 2]
            labels = labels.split(',')
            labels = [self.vocabulary[label] for label in labels]
            cls_label = np.zeros(self.num_classes, dtype=np.float32)
            cls_label[labels] = 1
            self.clip_labels.append([audio_file, cls_label])

    def __len__(self):
        return len(self.clip_labels)

    def __getitem__(self, idx):
        output_dict = {}
        audio_path, label = self.clip_labels[idx]
        # convert [cls1, cls2] to one-hot encoding by the number of classes
        audio, _ = librosa.load(audio_path, sr=self.sr)
        if len(audio) < self.duration * self.sr:
            audio = np.pad(audio, (0, self.duration*self.sr - len(audio)))
        else:
            audio = audio[:self.duration*self.sr]

        output_dict['audio'] = audio
        output_dict['cls_label'] = label
        return output_dict

class ESC50_dataset(Dataset):
    def __init__(self, root_dir, split='eval', label_level='clip', sr=16000, duration=5):
        """
        Args:
            root_dir (string): Directory with all the audio files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = os.path.join(root_dir, f'audio')
        self.sr = sr
        self.duration = duration
        self.label_level = label_level
        self.label = os.path.join(root_dir, f'meta/esc50.csv')
        self.num_classes = 50

        self.clip_labels = []
        self.label = pd.read_csv(self.label, delimiter=',',)
        for i in range(len(self.label)):
            audio_file = os.path.join(self.root_dir, self.label.iloc[i, 0])
            label = self.label.iloc[i, 2]; fold = self.label.iloc[i, 1]
            if split == 'eval' and fold != 5:
                continue
            if split == 'dev' and fold == 5:
                continue
             
            cls_label = np.zeros(self.num_classes, dtype=np.float32)
            cls_label[label] = 1
            self.clip_labels.append([audio_file, cls_label])
    def __len__(self):
        return len(self.clip_labels)
    def __getitem__(self, idx):
        output_dict = {}
        audio_path, label = self.clip_labels[idx]
        # convert [cls1, cls2] to one-hot encoding by the number of classes
        audio, _ = librosa.load(audio_path, sr=self.sr)
        if len(audio) < self.duration * self.sr:
            audio = np.pad(audio, (0, self.duration*self.sr - len(audio)))
        else:
            audio = audio[:self.duration*self.sr]

        output_dict['audio'] = audio
        output_dict['cls_label'] = label
        return output_dict


def Singleclass_dataset(dataset, max_num_each_class=500):
    keep_idx = []; classes_index = {}
    count = 0
    for i in range(len(dataset)):
        _, label = dataset.clip_labels[i]
        class_index = np.where(label == 1)[0]
        if len(class_index) == 1: # no multi-label
            class_index = class_index[0]
            keep_idx.append(i)
            if class_index not in classes_index:
                classes_index[class_index] = []
            classes_index[class_index].append(count)
            count += 1

    new_keep_idx = []
    for class_index in classes_index:
        if len(classes_index[class_index]) > max_num_each_class:
            classes_index[class_index] = classes_index[class_index][:max_num_each_class]
        else: # resample
            classes_index[class_index] = classes_index[class_index] * (max_num_each_class // len(classes_index[class_index])) + classes_index[class_index][:max_num_each_class % len(classes_index[class_index])]
        new_keep_idx += [keep_idx[idx] for idx in classes_index[class_index]]
    keep_idx = new_keep_idx

    print('Number of samples before filtering:', len(dataset),  'after filtering:', len(keep_idx))
    dataset = Subset(dataset, keep_idx)
    return dataset, classes_index



def find_similar_classes(classes_name):
    import laion_clap
    import numpy as np
    '''
    Given a list of classes, find the similar clusters = 20
    '''
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt() 
    text_embed = model.get_text_embedding(classes_name)
    
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=20, random_state=0).fit(text_embed)
    labels = kmeans.labels_
    return labels

            




if __name__ == '__main__':
    # dataset = FSD50K_dataset('../dataset/FSD50K/', split='eval')
    # dataset = Singleclass_dataset(dataset)

    dataset = ESC50_dataset('../dataset/ESC-50-master/', split='eval'); dataset = Singleclass_dataset(dataset)
    dataset = ESC50_dataset('../dataset/ESC-50-master/', split='dev'); dataset = Singleclass_dataset(dataset)
