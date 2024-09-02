# Asteroid is based on PyTorch and PyTorch-Lightning.
from torch import optim
from pytorch_lightning import Trainer
# We train the same model architecture that we used for inference above.

from con_asteroid.models.dprnn_tasnet import DPRNNTasNet_multichannel
from con_asteroid.multichannel_wrapper import Loss_Wrapper
from con_asteroid.variable_source_wrapper import PairwiseNegSDR_active
from asteroid.models import DPRNNTasNet
from asteroid.models import SuDORMRFNet

# In this example we use Permutation Invariant Training (PIT) and the SI-SDR loss.
from asteroid.losses import pairwise_neg_sisdr, multisrc_neg_sisdr, PITLossWrapper
# MiniLibriMix is a tiny version of LibriMix (https://github.com/JorisCos/LibriMix),
# which is a free speech separation dataset.
# from asteroid.data import LibriMix
# Asteroid's System is a convenience wrapper for PyTorch-Lightning.
from asteroid.engine import System
# This will automatically download MiniLibriMix from Zenodo on the first run.
from utils.torch_dataset import Separation_dataset, FUSSDataset
import torch

if __name__ == '__main__':
    import json 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/esc50_separation.json')
    args = parser.parse_args()
    config = json.load(open(args.config, 'r'))

    if config['output_format'] == 'separation': # the universal sound separation dataset
        # train_dataset = FUSSDataset(config['train_datafolder'], config['model']['sample_rate'])
        # val_dataset = FUSSDataset(config['train_datafolder'], config['model']['sample_rate'])

        train_dataset = Separation_dataset(config['train_datafolder'], config,)
        val_dataset = Separation_dataset(config['test_datafolder'], config,)

        print('train dataset {}, test dataset {}'.format(len(train_dataset), len(val_dataset)))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=8)

        model = DPRNNTasNet(**config['model'])
        loss = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        system = System(model, optimizer, loss, train_loader, val_loader)

        # Train for 1 epoch using a single GPU. If you're running this on Google Colab,
        # be sure to select a GPU runtime (Runtime → Change runtime type → Hardware accelarator).
        trainer = Trainer(max_epochs=50)
        trainer.fit(system)
    else:
        raise NotImplementedError('Only support universal sound separation dataset')

    # train_dataset = Separation_dataset(config['train_datafolder'], config,)
    # val_dataset = Separation_dataset(config['test_datafolder'], config,)

    

    

    # Tell DPRNN that we want to separate to 2 sources.
    # model = DPRNNTasNet(**config['model'])

    # ckpt = 'lightning_logs/version_0/checkpoints/epoch=49-step=23950.ckpt'
    # ckpt = torch.load(ckpt)['state_dict']
    # # remove 'model.' from the keys
    # new_ckpt = {}
    # for k in ckpt:
    #     new_ckpt[k[6:]] = ckpt[k]
    # model.load_state_dict(new_ckpt)
    # model.encoder = Multichannel_Encoder(model.encoder)

    # PITLossWrapper works with any loss function.
    # no need for PIT
    # loss = Loss_Wrapper(multisrc_neg_sisdr)

    
