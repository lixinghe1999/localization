# Asteroid is based on PyTorch and PyTorch-Lightning.
from torch import optim
from pytorch_lightning import Trainer
# We train the same model architecture that we used for inference above.
from models.deepbeam import BeamformerModel
# from models.AmbiSep import AmbiSep
# from asteroid.models import FasNetTAC

from asteroid.losses import pairwise_neg_sisdr, singlesrc_neg_sdsdr, multisrc_neg_sdsdr, PITLossWrapper
from asteroid.engine import System
# This will automatically download MiniLibriMix from Zenodo on the first run.
from utils.beamforming_dataset import Beamforming_dataset
import torch.nn as nn
import torch    
import numpy as np

class pwactive_wrapper(nn.Module):
    def __init__(self, loss_func, evaluation=False):
        super(pwactive_wrapper, self,).__init__()
        self.loss_func = loss_func
        self.evaluation = evaluation
    def forward(self, est_targets, targets):
        num_sources = targets.shape[1]
        active_sources = torch.sum(targets ** 2, dim=-1, keepdim=True) > 0
        num_active_sources = torch.sum(active_sources, dim=(1, 2))

        active_sources = active_sources.permute(0, 2, 1) # [batch, 1, targrt]
        active_sources = torch.tile(active_sources, (1, num_sources, 1)) # [batch, pred, target]
        negative_sources = ~active_sources
        pw_loss = self.loss_func(est_targets, targets) # [batch, pred, target(num of sources)]
        if not self.evaluation:
            l2_loss = torch.mean(est_targets ** 2, dim=-1, keepdim=True) ** 0.5
            l2_loss = torch.tile(l2_loss, (1, 1, num_sources)) 

            # print(pw_loss.shape, l2_loss.shape, active_sources.shape, negative_sources.shape)
            # print('pw_loss before:', pw_loss)
            # print(negative_sources)
            # set the loss of negative sources to 0
            pw_loss[negative_sources] = l2_loss[negative_sources]
            
            # only keep the loss of diagonal, pw_loss: [batch, pred, target]
            diag = torch.eye(num_sources, device=pw_loss.device)
            pw_loss = pw_loss * diag[None, :, :]
            pw_loss = torch.sum(pw_loss) / (num_sources * est_targets.shape[0])

            return pw_loss
        else:
            # only calculate the loss of active sources, get the minimum loss
            pw_loss = pw_loss * active_sources
            pw_loss = torch.min(pw_loss, dim=1)[0]
            pw_loss = torch.sum(pw_loss, dim=-1)

            pw_loss = pw_loss / (num_active_sources + 0.01)
            return pw_loss.mean()

def visualize(dataset):
    import matplotlib.pyplot as plt
    mixture, source = dataset[0]
    print(mixture.shape, source.shape)
    fake_output = np.zeros_like(source)
    fake_output[:5] = mixture
    mixture = fake_output
    fig, axs = plt.subplots(1, 3, figsize=(10, 10))

    for i, mix in enumerate(mixture):
        axs[0].plot(mix + i, c='r')
    for i, src in enumerate(source):
        axs[1].plot(src + i, c='b')
    
    loss_pw = loss(torch.tensor(mixture)[None, :, :], torch.tensor(source)[None, :, :])
    num_sources = source.shape[0]
    for i in range(num_sources):
        axs[2].plot(loss_pw[0, i, :], label='predict {}'.format(i))
    axs[2].legend()
    print('loss_pw:', loss_pw)
    plt.savefig('test.png')


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    config = { "train_datafolder": "dataset/smartglass/AudioSet_2/train",
                "test_datafolder": "dataset/smartglass/AudioSet_2/test",
                "ckpt": "",
                "duration": 5,
                "batch_size": 4,
                "output_format": "beamforming",
                "sample_rate": 16000,
                "max_sources": 8,
            }
    train_dataset = Beamforming_dataset(config['train_datafolder'], config,)
    val_dataset = Beamforming_dataset(config['test_datafolder'], config,)
    print('train dataset {}, test dataset {}'.format(len(train_dataset), len(val_dataset)))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    
    model = BeamformerModel(ch_in=5, synth_mid=64, synth_hid=96, block_size=16, kernel=3, synth_layer=4, synth_rep=4, lookahead=0)

    loss = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    system = System(model, optimizer, loss, train_loader, val_loader)
    trainer = Trainer(max_epochs=10, devices=1, num_nodes=1)
    trainer.fit(system)
    # trainer.validate(system)
  