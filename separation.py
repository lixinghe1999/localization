# Asteroid is based on PyTorch and PyTorch-Lightning.
from torch import optim
from pytorch_lightning import Trainer

# We train the same model architecture that we used for inference above.
from asteroid.models import DPRNNTasNet, ConvTasNet, SuDORMRFNet

# In this example we use Permutation Invariant Training (PIT) and the SI-SDR loss.
from asteroid.losses import pairwise_neg_sisdr, PITLossWrapper, singlesrc_neg_sisdr, multisrc_neg_sisdr

# MiniLibriMix is a tiny version of LibriMix (https://github.com/JorisCos/LibriMix),
# which is a free speech separation dataset.

from utils.separation_dataset import FUSSDataset
import torch
import pytorch_lightning as pl

class SeparationLightningModule(pl.LightningModule):
    def __init__(self, config):
        super(SeparationLightningModule, self).__init__()
        self.config = config
        if self.config['output_format'] == 'separation':
            # self.model = ConvTasNet(n_src=2, sample_rate=config['sample_rate'])
            self.model = SuDORMRFNet(n_src=2, sample_rate=config['sample_rate'])
            self.loss = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

    def training_step(self, batch, batch_idx):
        data, label = batch
        if self.config['output_format'] == 'separation':
            outputs = self.model(data)
            loss = self.loss(outputs, label)
            self.log('train', loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        if self.config['output_format'] == 'separation':
            outputs = self.model(data)
            loss = self.loss(outputs, label)
        self.log('val', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)      

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=0.001)
  

if __name__ == '__main__':
    
    config = { 
                "train_datafolder": 'dataset/audio/FUSS/ssdata/train_example_list.txt',
                "test_datafolder": 'dataset/audio/FUSS/ssdata/validation_example_list.txt',
                "duration": 10,
                "epochs": 20,
                "batch_size": 4,
                "output_format": "separation",
                "sample_rate": 8000,
                "max_sources": 2,
            }
    train_dataset = FUSSDataset(config['train_datafolder'], n_src=config['max_sources'], 
                                duration=config['duration'], sample_rate=config['sample_rate'])
    test_dataset = FUSSDataset(config['train_datafolder'], n_src=config['max_sources'], 
                               duration=config['duration'], sample_rate=config['sample_rate'])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    model = SeparationLightningModule(config)
    trainer = Trainer(max_epochs=config['epochs'], devices=[1])
    trainer.fit(model, train_loader, test_loader)  

    # trainer.validate(model, test_loader)

    # ckpt = 'lightning_logs/separation/fuss_sudormrfnet8/checkpoints/epoch=49-step=62450.ckpt'
    # ckpt = torch.load(ckpt, weights_only=True)['state_dict']
    # model.load_state_dict(ckpt, strict=True)
    # trainer.validate(model, test_loader)

