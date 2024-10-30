import sys
sys.path.append('..')
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.frame_audio_dataset import AudioSet_dataset

dataset = AudioSet_dataset(root='../dataset/audioset', vision=True, split='eval', sr=16000, duration=10, frame_duration=0.1, label_level='clip')
dataset.filter_modal(['audio', 'embeddings'])

clip_labels = []; images = []

for segment_id, clip_label in tqdm(dataset.clip_labels):
    embeddings_file = os.path.join(dataset.embeddings_dir, segment_id + '.npy')
    image = np.load(embeddings_file).astype(np.float32)[0]
    clip_labels.append(clip_label)
    images.append(image)

clip_labels = np.array(clip_labels); images = np.array(images)

# cluster the image embeddings
from sklearn.cluster import KMeans
n_clusters = 100

kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(images)
# convert to 2d for visualization with cluster label
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
images_2d = tsne.fit_transform(images)

fig, ax = plt.subplots()
plt.scatter(images_2d[:, 0], images_2d[:, 1], c=kmeans.labels_)
plt.title('Embeddings Cluster of {} classes'.format(n_clusters))
plt.savefig('embeddings_cluster.png')


fig, ax = plt.subplots()
for i in range(dataset.num_classes):
    vision_cluster = kmeans.labels_ == i
    clip_labels_cluster = clip_labels[vision_cluster]
    clip_labels_cluster = np.sum(clip_labels_cluster, axis=0)
    clip_labels_cluster = clip_labels_cluster / np.sum(clip_labels_cluster)

    clip_labels_cluster = np.sort(clip_labels_cluster)[::-1]
    # threshold = 0.9
    cumsum = np.cumsum(clip_labels_cluster)
    cluster_max_idx = np.argmax(cumsum > 0.8)
    cumsum = cumsum[:cluster_max_idx]
    plt.plot(cumsum)
plt.title('Class Distribution in each cluster with 80% threshold')
plt.savefig('class_distribution.png')
