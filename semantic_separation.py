# Asteroid is based on PyTorch and PyTorch-Lightning.
from torch import optim
from pytorch_lightning import Trainer

# We train the same model architecture that we used for inference above.
from asteroid.models import DPRNNTasNet, ConvTasNet, SuDORMRFNet
from models.separation import SemanticHearingNet
from models.separation.multichannel_sep import MultiChannel_Sep

# In this example we use Permutation Invariant Training (PIT) and the SI-SDR loss.
from asteroid.losses import pairwise_neg_sisdr, PITLossWrapper, singlesrc_neg_sisdr, multisrc_neg_sisdr

# MiniLibriMix is a tiny version of LibriMix (https://github.com/JorisCos/LibriMix),
# which is a free speech separation dataset.

# Asteroid's System is a convenience wrapper for PyTorch-Lightning.
from asteroid.engine import System
from utils.beamforming_dataset import Beamforming_dataset
from utils.separation_dataset import FUSSDataset
import torch
import pytorch_lightning as pl

class SeparationLightningModule(pl.LightningModule):
    def __init__(self, config):
        super(SeparationLightningModule, self).__init__()
        self.config = config
        if self.config['output_format'] == 'semantic':
            self.model = SemanticHearingNet(label_len=512)
            self.loss = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
        else:
            self.model = MultiChannel_Sep('SuDORMRFNet', n_channel=1, n_src=2, sample_rate=config['sample_rate'])
            if config['end2end']:
                ckpt = 'lightning_logs/separation/fuss_sudormrfnet_16k/checkpoints/epoch=19-step=100000.ckpt'
                ckpt = torch.load(ckpt, weights_only=True)['state_dict']
                # remove model. prefix
                ckpt = {k[6:]: v for k, v in ckpt.items()}
                self.model.separator.load_state_dict(ckpt, strict=True)
                print('load pretrained separator')
                # freeze the separator
                for param in self.model.separator.parameters():
                    param.requires_grad = False
                self.loss = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
            else:
                self.loss = multisrc_neg_sisdr

    def training_step(self, batch, batch_idx):
        data, label = batch
        if self.config['output_format'] == 'semantic':
            outputs = self.model(data)
            loss = self.loss(outputs, label)
            self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)
        else:
            if self.config['end2end']:
                outputs = self.model(data)
                B, S, C, T = outputs.shape
                outputs = outputs.permute(0, 2, 1, 3).reshape(B * C, S, T)
                label = label.permute(0, 2, 1, 3).reshape(B * C, S, T)
                loss = self.loss(outputs, label)
            else:
                outputs = self.model(data, label[:, :, 0])
                B, S, C, T = outputs.shape
                outputs = outputs.reshape(B, S * C, T); label = label.reshape(B, S * C, T)
                loss = self.loss(outputs, label).mean()
            self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)         
        return loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        if self.config['output_format'] == 'semantic':
            outputs = self.model(data)
            loss = self.loss(outputs, label)
        else:
            if self.config['end2end']:
                outputs = self.model(data)
                B, S, C, T = outputs.shape
                outputs = outputs.permute(0, 2, 1, 3).reshape(B * C, S, T)
                label = label.permute(0, 2, 1, 3).reshape(B * C, S, T)
                loss = self.loss(outputs, label)
            else:
                outputs = self.model(data, label[:, :, 0])
                B, S, C, T = outputs.shape
                outputs = outputs.reshape(B, S * C, T); label = label.reshape(B, S * C, T)
                loss = self.loss(outputs, label).mean()
        print(loss.item())
        self.log('validation_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)      

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=0.001)
  

if __name__ == '__main__':

    config = { 
                "train_datafolder": "dataset/smartglass/FSD50K_2/train",
                "test_datafolder": "dataset/smartglass/FSD50K_2/test",
                "duration": 10,
                "epochs": 20,
                "batch_size": 4,
                "output_format": "semantic",
                "sample_rate": 16000,
                "max_sources": 2,
                "num_region": 8,
                "end2end": True
            }
    train_dataset = Beamforming_dataset(config['train_datafolder'], config,)
    test_dataset = Beamforming_dataset(config['test_datafolder'], config,)

  
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    model = SeparationLightningModule(config)
    trainer = Trainer(max_epochs=config['epochs'], devices=[0])
    trainer.fit(model, train_loader, test_loader)  
    # trainer.validate(model, test_loader)


