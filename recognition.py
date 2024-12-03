'''
Audio recognition script
'''
import pytorch_lightning as pl
from models.recognition.frame_mn import Sound_Event_Detector
from utils.recognition_dataset import AudioSet_dataset, FSD50K_dataset
from torch.utils.data import random_split
import torch
import torchmetrics
import os 
import warnings
warnings.filterwarnings("ignore")

class AudioRecognition(pl.LightningModule):
    def __init__(self, train_dataset=None, test_dataset=None, model_name='mn10_as', lr=1e-3, ):
        super().__init__()
        self.label_level = 'clip'
        if train_dataset is None or test_dataset is None: # load default dataset
            # root = os.path.join('dataset', 'audioset')
            # dataset = AudioSet_dataset(root=root, split='eval', frame_duration=0.1, modality=['audio'], label_level=self.label_level)
            # self.train_dataset, self.test_dataset = random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
            # self.train_dataset.num_classes = dataset.num_classes
            # self.test_dataset.num_classes = dataset.num_classes

            # self.train_dataset = AudioSet_dataset(root=root, split='train', frame_duration=1, vision=False, label_level=label_level)
            # self.test_dataset = AudioSet_dataset(root=root, split='eval', frame_duration=1, vision=False, label_level=label_level)
            self.train_dataset = FSD50K_dataset('dataset/FSD50K', split='dev', label_level=self.label_level)
            self.test_dataset = FSD50K_dataset('dataset/FSD50K', split='eval', label_level=self.label_level)
        else:
            self.train_dataset = train_dataset
            self.test_dataset = test_dataset
       
        print('number of training samples: ', len(self.train_dataset), 'number of testing samples: ', len(self.test_dataset))

        self.model = Sound_Event_Detector(model_name, self.train_dataset.num_classes, frame_duration=None)
        self.lr = lr
        self.task = 'multilabel' # ['multilabel', 'multiclass']
        if self.task == 'multiclass':
            self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.train_dataset.num_classes)
        else:
            self.accuracy = torchmetrics.AveragePrecision(task='multilabel', num_labels=self.train_dataset.num_classes, average='micro')
        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):
        x = batch['audio']
        y = batch['cls_label']
        y_hat, _ = self.model(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)           
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch['audio']
        y = batch['cls_label']
        y_hat, _ = self.model(x)
        assert y_hat.shape == y.shape
        if len(y_hat.shape) == 3:
            y_hat = y_hat.reshape(-1, y_hat.shape[-1])
            y = y.reshape(-1, y.shape[-1])
        y_hat = torch.sigmoid(y_hat)
        val_acc = self.accuracy(y_hat, y.long())
        if torch.isnan(val_acc):
            print('nan detected, no positive samples')
        else:
            self.log('validation_loss', val_acc, on_epoch=True, prog_bar=True, logger=True)   

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def loss(self, y_hat, y):
        # single-label classification
        if self.task == 'multiclass':
            return torch.nn.CrossEntropyLoss()(y_hat, y)
        elif self.task == 'multilabel':
            return torch.nn.BCEWithLogitsLoss()(y_hat, y.float())
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=16, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=4, shuffle=False, num_workers=4)  

if __name__ == "__main__":
    trainer = pl.Trainer(max_epochs=50, devices=[0])

    model = AudioRecognition()
    trainer.fit(model)
    
    # ckpt = 'lightning_logs/frame_1/checkpoints/epoch=9-step=6700.ckpt'
    # model = AudioRecognition.load_from_checkpoint(ckpt)
    # trainer.validate(model)