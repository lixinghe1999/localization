from utils.localization_dataset import Localization_dataset
import pytorch_lightning as pl
from models.seldnet_model import SeldModel, SeldModel_Mobile
from torch.utils.data import random_split
import torch
import torchmetrics
import numpy as np


class Coarse_grained_localization(pl.LightningModule):
    '''
    Localize by region-wise classification
    '''
    def __init__(self,  lr=1e-3):
        super().__init__()
        # root_dir = 'dataset/earphone/TAU-SEBin/bin_prox_dir_one'
        # num_class = 1
        # config={'duration': 5, 'frame_duration':1, 'encoding': 'Region', 'num_class': num_class, 
        #             'raw_audio': False, 'label_type': 'eventwise', 'motion': False, 'class_names': ['alarm', 'baby', 'blender', 'cat', 'crash', 'dishes', 'dog', 'engine', 'fire', 'footsteps', 
        #                     'glassbreak', 'gunshot', 'knock', 'phone', 'piano', 'scream', 'speech', 'water']}
        # dataset = Localization_dataset(root_dir=root_dir, config=config, sr=16000)
        # self.model = SeldModel_Mobile(mic_channels=3, unique_classes=9, activation='sigmoid', t_pool_size=[50, 1, 1])
        # self.train_dataset, self.test_dataset = random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)],
        #                                                      generator=torch.Generator().manual_seed(42))


        config={'duration': 5, 'frame_duration':1, 'encoding': 'Region', 'num_class': 1, 
                    'raw_audio': False, 'label_type': 'eventwise', 'motion': True, 'class_names': ['nigens']}
        datasets = []
        root_dirs = ['dataset/earphone/lixing', 'dataset/earphone/shangcheng', 'dataset/earphone/jingfei', 'dataset/earphone/kaiwei', 'dataset/earphone/shaoyang',
                         'dataset/earphone/haozheng']
        # root_dirs = ['dataset/earphone/kaiwei']
        for root_dir in root_dirs:
            dataset = Localization_dataset(root_dir=root_dir, config=config, sr=16000)
            datasets.append(dataset)
        dataset = torch.utils.data.ConcatDataset(datasets)
        self.model = SeldModel_Mobile(mic_channels=3, unique_classes=8, activation='sigmoid', t_pool_size=[50, 1, 1])
        self.train_dataset, self.test_dataset = random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)],
                                                             generator=torch.Generator().manual_seed(42))
        

        print('number of training samples: ', len(self.train_dataset), 'number of testing samples: ', len(self.test_dataset))

        self.lr = lr
        self.direction_accuracy = torchmetrics.AveragePrecision(task='multilabel', num_labels=4, average='macro') 
        self.elevation_accuracy = torchmetrics.AveragePrecision(task='multilabel', num_labels=2, average='macro')
        self.distance_accuracy = torchmetrics.AveragePrecision(task='multilabel', num_labels=2, average='macro')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch['spatial_feature']; y = batch['label']; imu = batch['imu']
        y_hat = self.model(x, imu)
        if len(y_hat.shape) == 3:
            y_hat = y_hat.reshape(-1, y_hat.shape[-1])
            y = y.reshape(-1, y.shape[-1])
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch['spatial_feature']; y = batch['label']; imu = batch['imu']
        y_hat = self.model(x, imu)
        assert y_hat.shape == y.shape
        if len(y_hat.shape) == 3:
            y_hat = y_hat.reshape(-1, y_hat.shape[-1])
            y = y.reshape(-1, y.shape[-1])
        direction_acc = self.direction_accuracy(y_hat[:, :4], y[:, :4].long())
        elevation_acc = self.elevation_accuracy(y_hat[:, 4:6], y[:, 4:6].long())
        dist_acc = self.distance_accuracy(y_hat[:, 6:8], y[:, 6:8].long())
        
        self.log('azimuth', direction_acc)
        self.log('elevation', elevation_acc)
        self.log('distance', dist_acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def loss(self, y_hat, y):
        return torch.nn.functional.binary_cross_entropy(y_hat, y.float())
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=8, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=8, shuffle=False, num_workers=4)  

def distance_binary_cls():
    import matplotlib.pyplot as plt
    root_dir = 'dataset/earphone/TAU-SEBin/bin_prox_dir'
    dataset = Localization_dataset(root_dir=root_dir, 
            config={'duration': 5, 'encoding': 'Region', 'num_class': 18, 'raw_audio': True}, sr=16000)
    train_dataset, test_dataset = random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
    distance_energy = [[], [], [], []] # nothing, near, far, both
    for x, y in train_dataset:
        dist_y = y[:, 6:8]
        x = x[0].reshape(-1, 1600)**2
        x_energy = np.mean(x ** 2, axis=1) ** 0.5
        for dist, energy in zip(dist_y, x_energy):
            # append the energy according to the distance
            if dist[0] == 0 and dist[1] == 0:
                distance_energy[0].append(energy)
            elif dist[0] == 1 and dist[1] == 0:
                distance_energy[1].append(energy)
            elif dist[0] == 0 and dist[1] == 1:
                distance_energy[2].append(energy)
            elif dist[0] == 1 and dist[1] == 1:
                distance_energy[3].append(energy)
    for i, energy in enumerate(distance_energy):
        plt.scatter([i]*len(energy), energy)
        if len(energy) > 0:
            distance_energy[i] = np.mean(energy)
        else:
            distance_energy[i] = 0
    plt.xticks([0, 1, 2, 3], ['nothing', 'near', 'far', 'both'])
    plt.ylabel('Energy')
    plt.savefig('distance_energy.png')

def direction_vis():
    import matplotlib.pyplot as plt
    f1score = [71.4, 39.9, 28.7, 51, 82.5]
    labels = ['left/right', 'front/back', 'up/down', 'direction', 'distance']
    plt.bar(labels, f1score, width=0.3)
    plt.ylabel('F1 score')
    plt.savefig('direction_f1.png')


if __name__ == "__main__":
    model = Coarse_grained_localization()
    trainer = pl.Trainer(max_epochs=20, devices=1)
    trainer.fit(model)
    # trainer.validate(model)

    # distance_binary_cls()
    # direction_vis()