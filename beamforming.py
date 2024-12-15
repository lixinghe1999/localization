# Asteroid is based on PyTorch and PyTorch-Lightning.
from torch import optim
from pytorch_lightning import Trainer
import pytorch_lightning as pl
# We train the same model architecture that we used for inference above.

from models.beamforming import BeamformerModel, ConvTas_Net, Net
from models.adaptive_loss import SNRLPLoss

from asteroid.losses import pairwise_neg_sisdr, singlesrc_neg_sisdr, multisrc_neg_sdsdr, PITLossWrapper
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio

from utils.beamforming_dataset import Beamforming_dataset
import torch.nn as nn
import torch    
import numpy as np


class BeamformingLightningModule(pl.LightningModule):
    def __init__(self, config):
        super(BeamformingLightningModule, self).__init__()
        self.config = config
        self.model = Net(num_ch=config['num_channel'], num_src=config['num_region'])

        self.loss = SNRLPLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, label = batch
        outputs = self(data)

        if self.current_epoch < 15:
            loss = self.loss(outputs, label, neg_weight=0)
        else:
            loss = self.loss(outputs, label, neg_weight=20)
        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        outputs = self(data)
        positive_loss, negative_loss = self.loss(outputs, label, neg_weight=1)
        self.log('validataion/positive', positive_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('validataion/negative', negative_loss, on_epoch=True, prog_bar=True, logger=True)

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
                "output_format": "region",
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
    

    # ckpt_path = 'lightning_logs/beamforming/vctk_12/checkpoints/epoch=19-step=50000.ckpt'
    # ckpt_path = 'lightning_logs/beamforming/vctk_8/checkpoints/epoch=19-step=50000.ckpt'

    # ckpt_path = 'lightning_logs/version_0/checkpoints/epoch=19-step=25000.ckpt'
    # ckpt_path = 'lightning_logs/version_1/checkpoints/epoch=4-step=6250.ckpt'
    # ckpt_path = 'lightning_logs/beamforming/nigens_12/checkpoints/epoch=4-step=6250.ckpt'
    ckpt_path = 'lightning_logs/version_2/checkpoints/epoch=19-step=28240.ckpt'
    model.load_state_dict(torch.load(ckpt_path, weights_only=True)['state_dict'])    

    # trainer.fit(model, train_loader, test_loader)  
    model.visualize(test_loader)
    # trainer.validate(model, test_loader)

