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
        self.model = SeldModel(mic_channels=config['num_channel'], unique_classes=config['output_dimension'], activation='tanh')
        self.criterion = ACCDOA_loss if config['encoding'] == 'ACCDOA' else Multi_ACCDOA_loss
        self.evaluation = ACCDOA_evaluation if config['encoding'] == 'ACCDOA' else Multi_ACCDOA_evaluation

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data = batch['spatial_feature']
        labels = batch['label']

        outputs = self(data)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch['spatial_feature']
        labels = batch['label']

        outputs = self(data)
        eval_dict = self.evaluation(outputs.cpu().numpy(), labels.cpu().numpy())

        self.log('val_sed_F1', eval_dict['sed_F1'])
        self.log('val_F1', eval_dict['F1'])
        self.log('val_precision', eval_dict['precision'])
        self.log('val_recall', eval_dict['recall'])
        self.log('val_distance', eval_dict['distance'])
        return eval_dict

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=0.0001)

def run_MUSIC(dataset):
    from utils.doa import inference, init
    from simulate.parameter import MIC_ARRAY_SIMULATION
    algo = init(MIC_ARRAY_SIMULATION, fs=16000, nfft=1600, algorithm='music')

    for i in range(len(dataset)):
        data = dataset[i]
        audio, imu = data['audio'], data['imu']
        predictions = inference(algo, [audio, None])
        print(predictions.shape, data['label'].shape)
        acc = ACCDOA_evaluation(predictions, data['label'])
        print(acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/smartglass.json')
    args = parser.parse_args()
    
    # config = {
    #     "dataset": "smartglass",
    #     "train_datafolder": "/home/lixing/localization/dataset/starss23/dev-train-sony",
    #     "test_datafolder": "/home/lixing/localization/dataset/starss23/dev-test-sony",
    #     "cache_folder": "cache/starss23/",
    #     "encoding": "Multi_ACCDOA",
    #     "duration": 5,
    #     "frame_duration": 0.1,
    #     "batch_size": 64,
    #     "epochs": 50,
    #     "model": "seldnet",
    #     "label_type": "framewise",
    #     "raw_audio": False,
    #     'num_channel': 10,
    #     'output_dimension': 6, # no need to do classification now
    #     "pretrained": False,
    #     "test": False,
    #     "class_names": [
    #             "Female speech, woman speaking",
    #             "Male speech, man speaking",
    #             "Clapping",
    #             "Telephone",
    #             "Laughter",
    #             "Domestic sounds",
    #             "Walk, footsteps",
    #             "Door, open or close",
    #             "Music",
    #             "Musical instrument",
    #             "Water tap, faucet",
    #             "Bell",
    #             "Knock"
    #         ],
    #     "motion": False,
    # }

    config = {
        "train_datafolder": "/home/lixing/localization/dataset/smartglass/AudioSet_2/train",
        "test_datafolder": "/home/lixing/localization/dataset/smartglass/AudioSet_2/test",
        "cache_folder": "cache/audioset_2/",
        "encoding": "ACCDOA",
        "duration": 5,
        "frame_duration": 0.1,
        "batch_size": 16,
        "epochs": 10,
        "model": "seldnet",
        "label_type": "framewise",
        "raw_audio": False,
        'num_channel': 15,
        'output_dimension': 6, # no need to do classification now
        "pretrained": False,
        "test": False,
        'class_names':["sound"],
        'motion': False
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # train_dataset = Localization_dataset(config['train_datafolder'], config)
    test_dataset = Localization_dataset(config['test_datafolder'], config)
    
    # recognition_model = AudioRecognition(train_dataset=train_dataset, test_dataset=test_dataset)
    # trainer = pl.Trainer(max_epochs=10, devices=1)
    # trainer.fit(recognition_model)


    # train_dataset._cache_(config['cache_folder'] + '/train')
    # test_dataset._cache_(config['cache_folder'] + '/test')

    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    # model = SeldNetLightningModule(config)

    # if config['pretrained']:
    #     ckpt = torch.load(config['pretrained'])['state_dict']
    #     model.load_state_dict(ckpt)
    #     print('load pretrained model from', config['pretrained'])

    # trainer = Trainer(
    #     max_epochs=config['epochs'], devices=1)

    # if config['test']:
    #     trainer.validate(model, test_loader)
    # else:
    #     trainer.fit(model, train_loader, test_loader)