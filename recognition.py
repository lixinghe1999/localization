'''
Audio recognition script
'''
import pytorch_lightning as pl
from models.audio_models import Sound_Event_Detector
from utils.frame_audio_dataset import AudioSet_dataset
from torch.utils.data import random_split
import torch
import torchmetrics
import os 
import warnings
warnings.filterwarnings("ignore")

class AudioRecognition(pl.LightningModule):
    def __init__(self, model_name='mn40_as', lr=1e-3):
        super().__init__()
        root = os.path.join('dataset', 'audioset')
        dataset = AudioSet_dataset(root=root, split='train', frame_duration=1, vision=False, label_level='frame')
        dataset.filter_modal(['audio', 'embeddings'])
        self.train_dataset, self.test_dataset = random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
        # self.train_dataset.class_name = dataset.class_name
        # self.test_dataset.class_name = dataset.class_name

        # self.train_dataset = AudioSet_dataset(root=root, split='train', frame_duration=1, vision=False, label_level='frame')
        # self.test_dataset = AudioSet_dataset(root=root, split='eval', frame_duration=1, vision=False, label_level='frame')

        print('number of training samples: ', len(self.train_dataset), 'number of testing samples: ', len(self.test_dataset))
        print('number of classes: ', dataset.num_classes)
        self.model = Sound_Event_Detector(model_name, dataset.num_classes, frame_duration=1)
        self.lr = lr
        self.task = 'multilabel' # ['multilabel', 'multiclass']
        if self.task == 'multiclass':
            self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=dataset.num_classes)
        else:
            self.accuracy = torchmetrics.AveragePrecision(task='multilabel', num_labels=dataset.num_classes, average='macro')
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.model(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch       
        y_hat, _ = self.model(x)
        assert y_hat.shape == y.shape
        if len(y_hat.shape) == 3:
            y_hat = y_hat.reshape(-1, y_hat.shape[-1])
            y = y.reshape(-1, y.shape[-1])
        y_hat = torch.sigmoid(y_hat)
        acc = self.accuracy(y_hat, y.long())
        
        self.log('val_acc', acc)

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
    trainer = pl.Trainer(max_epochs=10, devices=1)

    # model = AudioRecognition()
    # trainer.fit(model)
    
    ckpt = 'lightning_logs/frame_1_mn40_full/checkpoints/epoch=9-step=44160.ckpt'
    model = AudioRecognition.load_from_checkpoint(ckpt)
    trainer.validate(model)