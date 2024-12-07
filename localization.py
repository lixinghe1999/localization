import torch
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.localization_dataset import Localization_dataset
from models.localization.seldnet_model import SeldModel
from utils.window_evaluation import ACCDOA_evaluation, Multi_ACCDOA_evaluation
from utils.window_loss import ACCDOA_loss, Multi_ACCDOA_loss
import numpy as np
import argparse

from recognition import AudioRecognition

def sed_vis(audio, output, label, save_path):
    '''
    audio: (batch, time)
    output: (batch, time, xyz)
    label: (batch, time, sed+xyz)
    '''
    import matplotlib.pyplot as plt
    batch, time = audio.shape
    batch, T, N = output.shape
    num_source = N // 3
    audio = audio/np.max(np.abs(audio), axis=-1, keepdims=True)
    fig, axs = plt.subplots(batch//2, 2, figsize=(10, 10))

    output = output.reshape(batch, T, num_source, 3)
    label = label.reshape(batch, T, num_source, 4)
    for i in range(batch):
        plt_idx = [i//2, i%2]
        output_sed = np.sqrt(np.sum(output[i]**2, axis=-1)) > 0.5
        label_sed = label[i, :, :, 0] > 0.5
        axs[plt_idx[0], plt_idx[1]].plot(audio[i])
        # upsample the sed
        output_sed = np.repeat(output_sed, time // T, axis=0)
        label_sed = np.repeat(label_sed, time // T, axis=0)
        axs[plt_idx[0], plt_idx[1]].plot(output_sed, label='output')
        axs[plt_idx[0], plt_idx[1]].plot(label_sed, label='label')
        axs[plt_idx[0], plt_idx[1]].legend()
        axs[plt_idx[0], plt_idx[1]].set_ylim(-1.2, 1.2)
    plt.savefig(save_path)


class SeldNetLightningModule(pl.LightningModule):
    def __init__(self, config):
        super(SeldNetLightningModule, self).__init__()
        self.config = config
        self.model = SeldModel(mic_channels=config['num_channel'], unique_classes=config['output_channel'], activation='tanh')
        self.criterion = ACCDOA_loss if config['encoding'] == 'ACCDOA' else Multi_ACCDOA_loss
        self.evaluation = ACCDOA_evaluation if config['encoding'] == 'ACCDOA' else Multi_ACCDOA_evaluation

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data = batch['spatial_feature']
        labels = batch['label']

        outputs = self(data)
        loss = self.criterion(outputs, labels)
        self.log('loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch['spatial_feature']
        labels = batch['label']

        outputs = self(data)
        eval_dict = self.evaluation(outputs.cpu().numpy(), labels.cpu().numpy())

        self.log('sed_F1', eval_dict['sed_F1'], on_epoch=True, prog_bar=True, logger=True)
        self.log('F1', eval_dict['F1'], on_epoch=True, prog_bar=True, logger=True)
        self.log('precision', eval_dict['precision'])
        self.log('recall', eval_dict['recall'])
        self.log('distance', eval_dict['distance'], on_epoch=True, prog_bar=True, logger=True)
        return eval_dict

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=0.0001)


if __name__ == '__main__':

    config = {
        "train_datafolder": "/home/lixing/localization/dataset/smartglass/NIGENS_2/train",
        "test_datafolder": "/home/lixing/localization/dataset/smartglass/NIGENS_2/test",
        "encoding": "ACCDOA",
        "duration": 5,
        "frame_duration": 0.1,
        "batch_size": 16,
        "epochs": 10,
        "model": "seldnet",
        "label_type": "framewise",
        "raw_audio": False,
        'num_channel': 15,
        'output_channel': 3, # no need to do classification now
        "pretrained": False,
        "test": False,
        'class_names':["sound"],
        'motion': False,
        "sr": 16000,
        'mixture': False,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = Localization_dataset(config['train_datafolder'], config)
    test_dataset = Localization_dataset(config['test_datafolder'], config)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    model = SeldNetLightningModule(config)

    if config['pretrained']:
        ckpt = torch.load(config['pretrained'])['state_dict']
        model.load_state_dict(ckpt)
        print('load pretrained model from', config['pretrained'])

    trainer = Trainer(max_epochs=config['epochs'], devices=1)

    if config['test']:
        trainer.validate(model, test_loader)
    else:
        trainer.fit(model, train_loader, test_loader)