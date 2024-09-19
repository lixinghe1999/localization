'''
Audio recognition script
'''
import pytorch_lightning as pl
from simulate.audio_dataset import dataset_parser
from models.audio_models import Sound_Event_Detector
import torch
import torchmetrics
class AudioRecognition(pl.LightningModule):
    def __init__(self, model_name='mn10_as', lr=1e-3):
        super().__init__()
        self.train_dataset, self.test_dataset = dataset_parser('ESC50', 'simulate')
        print(len(self.train_dataset), len(self.test_dataset))
        print('number of classes: ', len(self.train_dataset.class_name))
        self.model = Sound_Event_Detector(model_name, len(self.train_dataset.class_name))
        self.lr = lr
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=len(self.train_dataset.class_name))

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
        # random crop the audio

        B, T = x.shape
        crop_size = 8000
        start = torch.randint(0, T-crop_size, (B,))
        x = torch.stack([x[b,s:s+crop_size] for b, s in enumerate(start)])

        y_hat, _ = self.model(x)
        acc = self.accuracy(y_hat, y)
        self.log('val_acc', acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def loss(self, y_hat, y):
        return torch.nn.CrossEntropyLoss()(y_hat, y)
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=32, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=32, shuffle=False, num_workers=4)  

if __name__ == "__main__":
    model = AudioRecognition()
    trainer = pl.Trainer(max_epochs=30, devices=1)
    
    # trainer.fit(model)

    ckpt = 'lightning_logs/version_4/checkpoints/epoch=29-step=1500.ckpt'
    model = AudioRecognition.load_from_checkpoint(ckpt)
    trainer.validate(model)