# Asteroid is based on PyTorch and PyTorch-Lightning.
from torch import optim
from pytorch_lightning import Trainer

# We train the same model architecture that we used for inference above.
from asteroid.models import DPRNNTasNet, ConvTasNet, SuDORMRFNet
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
        if self.config['output_format'] == 'separation':
            self.model = SuDORMRFNet(n_src=2, sample_rate=config['sample_rate'])
            # self.model = ConvTasNet.from_pretrained('JorisCos/ConvTasNet_Libri2Mix_sepclean_16k')
            self.loss = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
        else:
            self.model = MultiChannel_Sep('ConvTasNet', n_channel=5, n_src=2, sample_rate=config['sample_rate'])
            if config['end2end']:
                # ckpt = 'lightning_logs/multichannel_sep_vctk_ref/checkpoints/epoch=19-step=50000.ckpt'
                # ckpt = torch.load(ckpt, weights_only=True)['state_dict']
                # # remove model.separator
                # ckpt = {k: v for k, v in ckpt.items() if 'separator' not in k}
                # self.load_state_dict(ckpt, strict=False)

                self.model.separator = ConvTasNet.from_pretrained('JorisCos/ConvTasNet_Libri2Mix_sepclean_16k')
                # freeze the separator
                # for param in self.model.separator.parameters():
                #     param.requires_grad = False
                self.loss = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
            else:
                self.loss = multisrc_neg_sisdr

    def training_step(self, batch, batch_idx):
        data, label = batch
        if self.config['output_format'] == 'separation':
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
        if self.config['output_format'] == 'separation':
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
        self.log('validation_loss', loss, on_epoch=True, prog_bar=True, logger=True)      

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=0.001)
  

if __name__ == '__main__':

    config = { 
                "train_datafolder": "dataset/smartglass/VCTK_2/train",
                "test_datafolder": "dataset/smartglass/VCTK_2/test",
                "duration": 5,
                "epochs": 20,
                "batch_size": 4,
                "output_format": "multichannel_separation",
                "sample_rate": 16000,
                "max_sources": 2,
                "num_region": 8,
                "end2end": True
            }
    train_dataset = Beamforming_dataset(config['train_datafolder'], config,)
    test_dataset = Beamforming_dataset(config['test_datafolder'], config,)

    
    # config = { 
    #             "train_datafolder": 'dataset/FUSS/ssdata/train_example_list.txt',
    #             "test_datafolder": 'dataset/FUSS/ssdata/validation_example_list.txt',
    #             "duration": 10,
    #             "epochs": 20,
    #             "batch_size": 4,
    #             "output_format": "separation",
    #             "sample_rate": 8000,
    #             "max_sources": 2,
    #         }
    # train_dataset = FUSSDataset(config['train_datafolder'], n_src=config['max_sources'], 
    #                             duration=config['duration'], sample_rate=8000, mode='separation')
    # test_dataset = FUSSDataset(config['train_datafolder'], n_src=config['max_sources'], 
    #                            duration=config['duration'], sample_rate=8000, mode='separation')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    model = SeparationLightningModule(config)
    trainer = Trainer(max_epochs=config['epochs'], devices=[1])
    trainer.fit(model, train_loader, test_loader)  
    # trainer.validate(model, test_loader)

    # ckpt = 'lightning_logs/fuss_sudormrfnet/checkpoints/epoch=49-step=62450.ckpt'
    # ckpt = torch.load(ckpt, weights_only=True)['state_dict']
    # model.load_state_dict(ckpt, strict=True)
    # trainer.validate(model, test_loader)

