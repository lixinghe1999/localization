import torch
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.public_dataset import STARSS23_dataset, Mobile_dataset
from utils.localization_dataset import Localization_dataset
from models.seldnet_model import SeldModel
from utils.window_evaluation import ACCDOA_evaluation, Multi_ACCDOA_evaluation, Guassian_evaluation
from utils.window_loss import ACCDOA_loss, Multi_ACCDOA_loss, Gaussian_loss
import numpy as np
import json
import argparse
import os
import time

class SeldNetLightningModule(pl.LightningModule):
    def __init__(self, config):
        super(SeldNetLightningModule, self).__init__()
        self.config = config
        self.model = SeldModel(mic_channels=config['num_channel'], unique_classes=3 * config['num_class'], activation='tanh')
        self.criterion = ACCDOA_loss if config['encoding'] == 'ACCDOA' else Multi_ACCDOA_loss
        self.evaluation = ACCDOA_evaluation if config['encoding'] == 'ACCDOA' else ACCDOA_evaluation

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, _, labels = batch
        outputs = self(data)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, _, labels = batch
        outputs = self(data)
        update_labels = self.criterion(outputs, labels, training=False)
        eval_dict = self.evaluation(outputs.cpu().numpy(), update_labels.cpu().numpy())

        self.log('val_sed_F1', eval_dict['sed_F1'])

        self.log('val_F1', eval_dict['F1'])
        self.log('val_distance', eval_dict['distance'])
        return eval_dict

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=0.0001)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/smartglass.json')
    args = parser.parse_args()

    config = {
        "dataset": "smartglass",
        "train_datafolder": "/home/lixing/localization/dataset/smartglass/FUSS_Reverb_2/train",
        "test_datafolder": "/home/lixing/localization/dataset/smartglass/FUSS_Reverb_2/test",
        "cache_folder": "cache/fuss_2/",
        "encoding": "Multi_ACCDOA",
        "duration": 5,
        "frame_duration": 0.1,
        "batch_size": 64,
        "epochs": 50,
        "model": "seldnet",
        "label_type": "framewise",
        "raw_audio": False,
        'num_channel': 15,
        'num_class': 2, # no need to do classification now
        "pretrained": False,
        "test": False,
    }
    print(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = Localization_dataset(config['train_datafolder'], config)
    test_dataset = Localization_dataset(config['test_datafolder'], config)

    train_dataset._cache_(config['cache_folder'] + '/train')
    test_dataset._cache_(config['cache_folder'] + '/test')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    model = SeldNetLightningModule(config)

    if config['pretrained']:
        ckpt = torch.load(config['pretrained'])['state_dict']
        model.load_state_dict(ckpt)
        print('load pretrained model from', config['pretrained'])


    trainer = Trainer(
        max_epochs=config['epochs'], devices=1)

    if config['test']:
        trainer.validate(model, test_loader)
    else:
        trainer.fit(model, train_loader, test_loader)