# Asteroid is based on PyTorch and PyTorch-Lightning.
from torch import optim
from pytorch_lightning import Trainer
import pytorch_lightning as pl
# We train the same model architecture that we used for inference above.

from models.beamforming import BeamformerModel, ConvTas_Net, Net
from models.adaptive_loss import SNRLPLoss
from asteroid.models import FasNetTAC
from models.localization.cos import CoSNetwork

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
        self.model = Net()
        # self.model = ConvTas_Net(num_mic=5, L=128, N=64, B=16, H=128, P=64, X=8, R=4, causal=True, norm_type='cLN')
        # self.model = CoSNetwork()
        self.loss = SNRLPLoss()
        # self.criterion = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx')
        # self.criterion = Penalized_PIT_Wrapper(PairwiseNegSDR_Loss("sisdr"))
        # self.criterion = ScaleInvariantSignalNoiseRatio()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, label = batch
        outputs = self(data)

        loss = self.loss(outputs, label)
        # B, S, C, T = outputs.shape
        # outputs = outputs.permute(0, 2, 1, 3).reshape(B * C, S, T)
        # label = label.permute(0, 2, 1, 3).reshape(B * C, S, T)
        # loss = self.criterion(outputs, label)
        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        outputs = self(data)

        # B, S, C, T = outputs.shape
        # mixture = data[:, None].repeat(1, 2, 1, 1) # [batch, source, channel, time]
        # # convert all to [batch * channel, source, time]
        # mixture = mixture.permute(0, 2, 1, 3).reshape(B * C, S, T)
        # outputs = outputs.permute(0, 2, 1, 3).reshape(B * C, S, T)
        # label = label.permute(0, 2, 1, 3).reshape(B * C, S, T)
        # mixture_loss = -self.criterion(data, label)
        # self.log('validataion/mixture', mixture_loss, on_epoch=True, prog_bar=True, logger=True)

        positive_loss, negative_loss = self.loss(outputs, label)
        self.log('validataion/positive_loss', positive_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('validataion/negative_loss', negative_loss, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=0.001)
    
    def visualize(self, test_dataloader):
        import matplotlib.pyplot as plt
        import soundfile as sf
        self.eval()
        for batch in test_dataloader:
            data, label = batch
            outputs = self(data)
            loss = self.loss(outputs, label)
            print('loss:', loss)
            for b in range(len(data)):
                label_sample = label[b]; outputs_sample = outputs[b] # (N_channel, T), (1, T), (1, T)             

                max_value = max(label_sample.max(), outputs_sample.max()).item()
                fig, axs = plt.subplots(self.config['max_sources'], 2, figsize=(10, 10))
                for i in range(self.config['max_sources']):
                    axs[i, 0].plot(label_sample[i, :].numpy(), c='b')
                    axs[i, 1].plot(outputs_sample[i, :].detach().numpy(), c='g')
                    axs[i, 0].set_ylim(-max_value, max_value)
                    axs[i, 1].set_ylim(-max_value, max_value)

                # sf.write('data.wav', data[0, 0, :].numpy(), 16000)
                # sf.write('label.wav', label[0, 0, :].numpy(), 16000)
                # sf.write('outputs.wav', outputs[0, 0, :].detach().numpy(), 16000)
        
                plt.savefig('./resources/beamforming_vis.png')
                break
            break

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
                "max_sources": 4,
            }
    train_dataset = Beamforming_dataset(config['train_datafolder'], config,)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)

    test_dataset = Beamforming_dataset(config['test_datafolder'], config,)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    
    model = BeamformingLightningModule(config)

    trainer = Trainer(max_epochs=config['epochs'], devices=[1])
    trainer.fit(model, train_loader, test_loader)  

    # ckpt_path = 'lightning_logs/version_2/checkpoints/epoch=19-step=10000.ckpt'
    # model.load_state_dict(torch.load(ckpt_path, weights_only=True)['state_dict'])    
    # model.visualize(test_loader)

