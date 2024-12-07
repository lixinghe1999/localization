# Asteroid is based on PyTorch and PyTorch-Lightning.
from torch import optim
from pytorch_lightning import Trainer

# We train the same model architecture that we used for inference above.
from asteroid.models import DPRNNTasNet, ConvTasNet, SuDORMRFNet
from models.separation import SemanticHearingNet, SemanticHearingNetBinaural
from models.separation.multichannel_sep import MultiChannel_Sep

# In this example we use Permutation Invariant Training (PIT) and the SI-SDR loss.
from asteroid.losses import pairwise_neg_sisdr, PITLossWrapper, singlesrc_neg_sisdr, multisrc_neg_sisdr

from utils.separation_dataset import FUSSDataset, LabelDataset
import torch
import pytorch_lightning as pl

class SeparationLightningModule(pl.LightningModule):
    def __init__(self, config):
        super(SeparationLightningModule, self).__init__()
        self.config = config
        if self.config['output_format'] == 'semantic':
            self.model = SemanticHearingNet(label_len=config['num_class'])
            self.loss = singlesrc_neg_sisdr

    def training_step(self, batch, batch_idx):
        data, label = batch
        if self.config['output_format'] == 'semantic':
            outputs = self.model(data)
            loss = self.loss(outputs.squeeze(), label).mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
          
        return loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        if self.config['output_format'] == 'semantic':
            outputs = self.model(data)
            noisy_loss = self.loss(data[0].squeeze(), label).mean()
            loss = self.loss(outputs.squeeze(), label).mean()
            self.log('noisy_val_loss', noisy_loss, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)      

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=0.001)
  

if __name__ == '__main__':

    config = { 
                "train_datafolder": "dataset/separation/ESC50/dev",
                "test_datafolder": "dataset/separation/ESC50/eval",
                "duration": 5,
                "epochs": 50,
                "batch_size": 8,
                "output_format": "semantic",
                "sample_rate": 16000,
                "max_sources": 2,
                "num_class": 50,
            }
    train_dataset = LabelDataset(config['train_datafolder'], config,)
    test_dataset = LabelDataset(config['test_datafolder'], config,)

  
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    model = SeparationLightningModule(config)
    trainer = Trainer(max_epochs=config['epochs'], devices=[0])
    trainer.fit(model, train_loader, test_loader)  

    # ckpt = torch.load('lightning_logs/version_0/checkpoints/epoch=49-step=50000.ckpt')
    # model.load_state_dict(ckpt['state_dict'])
    # trainer.validate(model, test_loader)




