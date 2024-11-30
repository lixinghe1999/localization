# Asteroid is based on PyTorch and PyTorch-Lightning.
from torch import optim
from pytorch_lightning import Trainer
import pytorch_lightning as pl
# We train the same model architecture that we used for inference above.

from models.beamforming import BeamformerModel, ConvTas_Net, Net
from asteroid.models import FasNetTAC
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
        # self.model = BeamformerModel(ch_in=5, synth_mid=64, synth_hid=96, block_size=16, kernel=3, synth_layer=4, synth_rep=4, lookahead=0)
        # self.model = FasNetTAC(n_src=config['max_sources'], sample_rate=config['sample_rate'])
        self.model = Net(num_src=config['num_region'])

        self.loss = SNRLPLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, label = batch
        outputs = self(data)

        loss = self.loss(outputs, label)
        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        outputs = self(data)

        positive_loss, negative_loss = self.loss(outputs, label)
        self.log('validataion/positive_loss', positive_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('validataion/negative_loss', negative_loss, on_epoch=True, prog_bar=True, logger=True)

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
        loss = self.loss(outputs, label)
        print('loss:', loss)
        B, C, T = label.shape
        print('label shape:', label.shape, 'outputs shape:', outputs.shape)
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
                "train_datafolder": "dataset/smartglass/VCTK_2/train",
                "test_datafolder": "dataset/smartglass/VCTK_2/test",
                "ckpt": "",
                "duration": 5,
                "epochs": 20,
                "batch_size": 4,
                "output_format": "region",
                "sample_rate": 16000,
                "max_sources": 2,
                "num_region": 8,
            }
    train_dataset = Beamforming_dataset(config['train_datafolder'], config,)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)

    test_dataset = Beamforming_dataset(config['test_datafolder'], config,)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    
    model = BeamformingLightningModule(config)

    # trainer = Trainer(max_epochs=config['epochs'], devices=[1])
    # trainer.fit(model, train_loader, test_loader)  

    ckpt_path = 'lightning_logs/vctk_8/checkpoints/epoch=19-step=50000.ckpt'
    model.load_state_dict(torch.load(ckpt_path, weights_only=True)['state_dict'])    
    model.visualize(test_loader)

