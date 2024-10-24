from utils.frame_audio_dataset import AudioSet_dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

audioset_label = pd.read_csv('dataset/audioset/unbalanced_train_segments.csv', sep=', ', skiprows=3, names=['YTID', 'start_seconds', 'end_seconds', 'positive_labels'])

label_map = {}
clip_labels = []
for label in audioset_label['positive_labels']:
    label = label.replace('"', '').split(',')
    for l in label:
        if l not in label_map:
            label_map[l] = len(label_map)
    clip_labels.append([label_map[l] for l in label])
coexist_audio = np.zeros((len(label_map), len(label_map)), dtype=np.int32)
for label in clip_labels:
    for i in label:
        for j in label:
            if i != j:
                coexist_audio[i, j] += 1



# dataset = AudioSet_dataset(root='dataset/audioset', split='train', frame_duration=1, vision=False, label_level='clip')

# coexist_audio = np.zeros((dataset.num_classes, dataset.num_classes), dtype=np.int32)

# for segment_id, label in dataset.clip_labels:
#     label_idx = np.where(label > 0)[0]
#     for i in label_idx:
#         for j in label_idx:
#             if j != i:
#                 coexist_audio[i, j] += 1
# print(len(dataset.clip_labels), np.mean(coexist_audio))



# contrastive plot
plt.figure(figsize=(10, 10))
plt.imshow(np.log10(coexist_audio, where=coexist_audio>0), cmap='viridis')
plt.savefig('coexist_audio.png')
