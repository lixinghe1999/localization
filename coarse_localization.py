from utils.localization_dataset import Localization_dataset
import pytorch_lightning as pl
from models.seldnet_model import SeldModel
from torch.utils.data import random_split
import torch
import torchmetrics


class Coarse_grained_localization(pl.LightningModule):
    '''
    Localize by region-wise classification
    '''
    def __init__(self,  lr=1e-3):
        super().__init__()
        root_dir = 'dataset/earphone/TAU-SEBin/bin_prox_dir_one'
        root_dir = 'dataset/earphone/20240927/'
        dataset = Localization_dataset(root_dir=root_dir, 
                               config={'duration': 5, 'encoding': 'Region', 'min_azimuth':-180, 'max_azimuth':180}, sr=16000)
        self.train_dataset, self.test_dataset = random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
        # self.train_dataset = dataset
        # self.test_dataset = dataset
        print('number of training samples: ', len(self.train_dataset), 'number of testing samples: ', len(self.test_dataset))
        self.model = SeldModel(mic_channels=3, unique_classes=9, activation=None)
        self.lr = lr
        self.doa_accuracy = torchmetrics.AveragePrecision(task='multilabel', num_labels=6, average='macro')
        self.distance_accuracy = torchmetrics.AveragePrecision(task='multilabel', num_labels=3, average='macro')
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        if len(y_hat.shape) == 3:
            y_hat = y_hat.reshape(-1, y_hat.shape[-1])
            y = y.reshape(-1, y.shape[-1])
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        assert y_hat.shape == y.shape
        if len(y_hat.shape) == 3:
            y_hat = y_hat.reshape(-1, y_hat.shape[-1])
            y = y.reshape(-1, y.shape[-1])
        doa_acc = self.doa_accuracy(y_hat[:, :6], y[:, :6].long())
        self.log('doa_accuracy', doa_acc)

        dist_acc = self.distance_accuracy(y_hat[:, 6:], y[:, 6:].long())
        self.log('distance_accuracy', dist_acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def loss(self, y_hat, y):
        doa_loss = torch.nn.BCEWithLogitsLoss()(y_hat[..., :6], y[..., :6].float())
        dist_loss = torch.nn.BCEWithLogitsLoss()(y_hat[:, 6:], y[:, 6:].float())
        return doa_loss + dist_loss
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=8, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=8, shuffle=False, num_workers=4)  

if __name__ == "__main__":
    model = Coarse_grained_localization()
    trainer = pl.Trainer(max_epochs=50, devices=1)
    trainer.fit(model)