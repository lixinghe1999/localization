# Asteroid is based on PyTorch and PyTorch-Lightning.
from torch import optim
import pytorch_lightning as pl
from utils.beamforming_dataset import Beamforming_dataset
from models.spatialcodec import SpatialCodec_Model
import torch.nn as nn
import torch    
import numpy as np


class CodecLightningModule(pl.LightningModule):
    def __init__(self, config):
        super(CodecLightningModule, self).__init__()
        self.config = config
        self.model = SpatialCodec_Model()

 
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, labels = batch
        outputs = self(data)
        loss = self.criterion(outputs, data)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        print('data', data.shape)
        outputs = self(data)
        print('outputs', outputs.shape)
        eval_dict = self.evaluation(outputs.cpu().numpy(), labels.cpu().numpy())

        return eval_dict

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=0.0001)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    config = { "train_datafolder": "dataset/smartglass/AudioSet_2/train",
                "test_datafolder": "dataset/smartglass/AudioSet_2/test",
                "ckpt": "",
                "duration": 5,
                "batch_size": 4,
                "output_format": "codec",
                "sample_rate": 16000,
                "max_sources": 8,
            }
    train_dataset = Beamforming_dataset(config['train_datafolder'], config,)
    val_dataset = Beamforming_dataset(config['test_datafolder'], config,)
    print('train dataset {}, test dataset {}'.format(len(train_dataset), len(val_dataset)))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    model = CodecLightningModule(config)
    trainer = pl.Trainer(max_epochs=1, devices=[0])
    trainer.fit(model, train_loader, val_loader)
    

  