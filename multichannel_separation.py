# Asteroid is based on PyTorch and PyTorch-Lightning.
from torch import optim
from pytorch_lightning import Trainer
import pytorch_lightning as pl
# We train the same model architecture that we used for inference above.

from models.separation.multichannel_sep import MultiChannel_Sep
from asteroid.models import DPRNNTasNet, ConvTasNet, SuDORMRFNet

from asteroid.losses import singlesrc_neg_sisdr

from utils.beamforming_dataset import Beamforming_dataset
import torch.nn as nn
import torch    
import numpy as np


class BeamformingLightningModule(pl.LightningModule):
    def __init__(self, config):
        super(BeamformingLightningModule, self).__init__()
        self.config = config
        separator = ConvTasNet(n_src=2, sample_rate=8000)
        ckpt_path = 'lightning_logs/separation/convasnet_8k/checkpoints/epoch=19-step=49960.ckpt'
        ckpt = torch.load(ckpt_path, weights_only=True)
        ckpt = {k.replace('model.', ''): v for k, v in ckpt['state_dict'].items()}
        separator.load_state_dict(ckpt)
        print('Loaded separator from:', ckpt_path)

        self.model = MultiChannel_Sep(separator, n_channel=2, n_src=2, sample_rate=config['sample_rate'], sample_rate_separator=8000)

        self.loss = singlesrc_neg_sisdr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, label = batch
        outputs = self.model(data)

        loss = self.loss(outputs, label)
        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        print(data.shape, label.shape)
        outputs = self.model(data)
        loss = self.loss(outputs, label)
        self.log('validataion', loss, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=0.001)
    
    def visualize(self, test_dataloader):
        import matplotlib.pyplot as plt
        import soundfile as sf
        self.eval()
        # sample random batch
        batch = next(iter(test_dataloader))
        data, label = batch
        outputs = self(data)
        loss = self.loss(outputs, label, neg_weight=1)
        print('loss:', loss)
        B, C, T = label.shape
        for b in range(B):
            label_sample = label[b]; outputs_sample = outputs[b] # (N_channel, T), (1, T), (1, T)             
            max_value = max(label_sample.max(), outputs_sample.max()).item()
            fig, axs = plt.subplots(C, 2, figsize=(10, 10))
            for i in range(C):
                axs[i, 0].plot(label_sample[i, :].numpy(), c='b')
                axs[i, 1].plot(outputs_sample[i, :].detach().numpy(), c='g')
                axs[i, 0].set_ylim(-max_value, max_value)
                axs[i, 1].set_ylim(-max_value, max_value)
            plt.savefig(f'./resources/beamforming_vis_{b}.png')


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    config = { 
                "train_datafolder": "dataset/earphone/VCTK_2/train",
                "test_datafolder": "dataset/earphone/VCTK_2/test",
                "ckpt": "",
                "duration": 5,
                "epochs": 20,
                "batch_size": 4,
                "output_format": "multichannel_separation",
                "sample_rate": 44100,
                "max_sources": 2,
                "num_channel": 2, 
                "num_region": 4,
            }
    train_dataset = Beamforming_dataset(config['train_datafolder'], config,)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8)

    test_dataset = Beamforming_dataset(config['test_datafolder'], config,)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    
    model = BeamformingLightningModule(config)

    trainer = Trainer(max_epochs=config['epochs'], devices=[0])
    
    # trainer.fit(model, train_loader, test_loader)  
    # model.visualize(test_loader)
    trainer.validate(model, test_loader)

