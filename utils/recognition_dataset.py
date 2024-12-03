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

        ontolog = os.path.join(root, 'ontology.json')
        ontolog = json.load(open(ontolog))
        self.ontology = {v['id']: v for v in ontolog}
        self.mid_to_name = {mid: self.ontology[mid]['name'] for mid in self.ontology}


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
        self.label = os.path.join(root_dir, f'FSD50K.ground_truth/{split}.csv')
        vocabulary = os.path.join(root_dir, 'FSD50K.ground_truth/vocabulary.csv')
        self.vocabulary = pd.read_csv(vocabulary, names=['name', 'mid'], delimiter=',')
        # convert to dictionary, key = mid, value = index
        self.vocabulary = {mid: idx for idx, mid in enumerate(self.vocabulary['mid'])}
        self.num_classes = len(self.vocabulary)

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

def Singleclass_dataset(dataset):
    keep_idx = []
    for i, data in enumerate(dataset.clip_labels):
        _, label = data
        total_num_classes = np.sum(label, axis=-1)
        if total_num_classes == 1:
            keep_idx.append(i)
    dataset = Subset(dataset, keep_idx)
    return dataset



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

def split_audioset_into_classwise():
    import shutil

    dataset = AudioSet_dataset('../dataset/audioset', split='eval', modality=['audio'], label_level='clip')

    classes_name = dataset.class_map
    # classes_label = find_similar_classes(classes_name)
    # cluster_info = {}
    # for cluster in range(20):
    #     classes_cluster = np.where(classes_label == cluster)[0]
    #     print([classes_name[class_idx] for class_idx in classes_cluster])
    #     data_segments = []
    #     for segment_id, label in dataset.clip_labels:
    #         if np.sum(label[classes_cluster]) > 0:
    #             data_segments.append(segment_id,)
    #     print('Number of segments for cluster', cluster, len(data_segments), classes_cluster)
    # dataset_folder = '../dataset/audioset/audioset_classwise/' + str(cluster)
    
    cluster_info = {} # one class one cluster
    for class_idx in range(len(classes_name)):
        class_name = classes_name[class_idx]
        data_segments = []
        for segment_id, label in dataset.clip_labels:
            if np.sum(label[class_idx]) > 0:
                data_segments.append(segment_id,)
        print('Number of segments for cluster', class_idx, class_name, len(data_segments))
        dataset_folder = '../dataset/audioset/audioset_classwise/' + class_name
        os.makedirs(dataset_folder, exist_ok=True) 
        for segment_id in data_segments: 
            audio_file = os.path.join(dataset.audio_dir, segment_id + '.flac')
            output_file = os.path.join(dataset_folder, segment_id + '.flac')
            # copy the audio file
            shutil.copy(audio_file, output_file)
    #     cluster_info[class_idx] = class_name
    # with open('../dataset/audioset/audioset_classwise/cluster_info.json', 'w') as f:
    #     json.dump(cluster_info, f, indent=4)

            




if __name__ == '__main__':
    # dataset = AudioSet_dataset('dataset/audioset', split='eval', modality=['audio'], label_level='clip')
    # print(len(dataset))
    # dataset_sample(dataset)
    # dataset = AudioSet_Singleclass_dataset('dataset/audioset', split='eval', modality=['audio'], label_level='frame')

    # for i in range(100):
    #     data = dataset.__getitem__(i)
    #     frame_num_class = np.sum(data['cls_label'], axis=1)
    #     print(np.mean(frame_num_class == 1), np.mean(frame_num_class > 1), np.mean(frame_num_class == 0))

    split_audioset_into_classwise()
