# Asteroid is based on PyTorch and PyTorch-Lightning.
from torch import optim
from pytorch_lightning import Trainer
import pytorch_lightning as pl
# We train the same model architecture that we used for inference above.
from models.deepbeam import BeamformerModel
from asteroid.models import FasNetTAC
from asteroid.losses import pairwise_neg_sisdr, singlesrc_neg_sisdr, multisrc_neg_sdsdr, PITLossWrapper

from utils.beamforming_dataset import Beamforming_dataset
import torch.nn as nn
import torch    
import numpy as np




class BeamformingLightningModule(pl.LightningModule):
    def __init__(self, config):
        super(BeamformingLightningModule, self).__init__()
        self.config = config
        # self.model = BeamformerModel(ch_in=5, synth_mid=64, synth_hid=96, block_size=16, kernel=3, synth_layer=4, synth_rep=4, lookahead=0)
        self.model = FasNetTAC(n_src=1, sample_rate=config['sample_rate'])
        
        self.loss = singlesrc_neg_sisdr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, label = batch
        outputs = self(data)
        outputs = outputs.squeeze(); label = label.squeeze()
        loss = self.loss(outputs, label).mean()
        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        outputs = self(data)

        mixture_loss = -self.loss(data[:, :1].squeeze(), label.squeeze()).mean()

        output_loss = -self.loss(outputs.squeeze(), label.squeeze()).mean()
        self.log('validataion/mixture', mixture_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('validataion/output', output_loss, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=0.0001)
    
    def visualize(self, test_dataloader):
        import matplotlib.pyplot as plt
        import soundfile as sf
        for batch in test_dataloader:
            data, label = batch
            before_snr = -self.loss(data[:, 0], label[:, 0]).mean()
            outputs = self(data)
            after_snr = -self.loss(outputs.squeeze(), label.squeeze()).mean()
            print('before_snr:', before_snr, 'after_snr:', after_snr)
            break
            # for b in range(len(data)):
            #     data_sample = data[b]; label_sample = label[b]; outputs_sample = outputs[b] # (N_channel, T), (1, T), (1, T)             
            #     N_channel = data_sample.shape[1]
            #     before_snr = self.loss(data_sample[None, :, :], label_sample[None, :, :])

            #     fig, axs = plt.subplots(3, 1, figsize=(10, 10))
            #     axs[0].plot(data_sample[0, :].numpy(), c='r')
            #     axs[1].plot(label_sample[0, :].numpy(), c='b')
            #     axs[2].plot(outputs_sample[0, :].detach().numpy(), c='g')

            #     sf.write('data.wav', data[0, 0, :].numpy(), 16000)
            #     sf.write('label.wav', label[0, 0, :].numpy(), 16000)
            #     sf.write('outputs.wav', outputs[0, 0, :].detach().numpy(), 16000)

        
            # plt.savefig('test.png')

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    config = { "train_datafolder": "dataset/smartglass/AudioSet_2/train",
                "test_datafolder": "dataset/smartglass/AudioSet_2/test",
                "ckpt": "",
                "duration": 10,
                "epochs": 20,
                "batch_size": 2,
                "output_format": "beamforming",
                "sample_rate": 16000,
                "max_sources": 2,
            }
    train_dataset = Beamforming_dataset(config['train_datafolder'], config,)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)

    test_dataset = Beamforming_dataset(config['test_datafolder'], config,)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=1)
    
    model = BeamformingLightningModule(config)

    trainer = Trainer(max_epochs=config['epochs'], devices=[1])
    trainer.fit(model, train_loader, test_loader)  

    # ckpt_path = 'lightning_logs/version_3/checkpoints/epoch=4-step=3125.ckpt'
    # copy the weight
    # model.load_state_dict(torch.load(ckpt_path, weights_only=True)['state_dict']) 

    # trainer.validate(model, test_loader)
   
    # model.visualize(test_loader)

