from utils.localization_dataset import Localization_dataset
import pytorch_lightning as pl
from models.localization.seldnet_model import SeldModel, SeldModel_Mobile
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
        root_dir = 'dataset/SELD/TAU-SEBin/bin_prox_dir_one'
        num_class = 1
        config={'duration': 5, 'frame_duration':5, 'encoding': 'Region', 'num_class': num_class, 
                    'raw_audio': False, 'label_type': 'eventwise', 'motion': False, 'class_names': ['alarm', 'baby', 'blender', 'cat', 'crash', 'dishes', 'dog', 'engine', 'fire', 'footsteps', 
                    'glassbreak', 'gunshot', 'knock', 'phone', 'piano', 'scream', 'speech', 'water'], "sr": 24000, "mixture": True}
        dataset = Localization_dataset(root_dir=root_dir, config=config)


        # config={'duration': 5, 'frame_duration':0.2, 'encoding': 'Region', 'num_class': 1, 
        #             'raw_audio': False, 'label_type': 'eventwise', 'motion': True, 'class_names': ['real', 'nigens', 'human']}
        # datasets = []
        # root_dirs = ['dataset/earphone/lixing', 'dataset/earphone/shangcheng', 'dataset/earphone/jingfei', 'dataset/earphone/kaiwei', 'dataset/earphone/shaoyang', 'dataset/earphone/haozheng', 
        #             'dataset/earphone/lixing_human', 'dataset/earphone/bufang_human', 'dataset/earphone/shaoyang_human', 'dataset/earphone/haozheng_human']
        # root_dirs = ['dataset/earphone/lixing', 'dataset/earphone/shangcheng', 'dataset/earphone/jingfei', 'dataset/earphone/kaiwei', 'dataset/earphone/shaoyang', 'dataset/earphone/haozheng']
        # for root_dir in root_dirs:
        #     dataset = Localization_dataset(root_dir=root_dir, config=config, sr=16000)
        #     datasets.append(dataset)
        # dataset = torch.utils.data.ConcatDataset(datasets)

        self.model = SeldModel_Mobile(mic_channels=3, unique_classes=8, activation='sigmoid', t_pool_size=[25, 10, 1])

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
        
        self.log('azimuth', direction_acc, on_epoch=True, prog_bar=True)
        self.log('elevation', elevation_acc, on_epoch=True, prog_bar=True)
        self.log('distance', dist_acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def loss(self, y_hat, y):
        return torch.nn.functional.binary_cross_entropy(y_hat, y.float())
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=4, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=4, shuffle=False, num_workers=4)  


if __name__ == "__main__":
    model = Coarse_grained_localization()
    trainer = pl.Trainer(max_epochs=10, devices=1, log_every_n_steps=10)
    trainer.fit(model)
    # trainer.validate(model)