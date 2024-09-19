# Asteroid is based on PyTorch and PyTorch-Lightning.
from torch import optim
from pytorch_lightning import Trainer
# We train the same model architecture that we used for inference above.
from utils.variable_source_loss import singlesrc_neg_sisdr

from models.deepbeam import BeamformerModel

# In this example we use Permutation Invariant Training (PIT) and the SI-SDR loss.
from asteroid.losses import pairwise_neg_sisdr, PITLossWrapper
from asteroid.engine import System
# This will automatically download MiniLibriMix from Zenodo on the first run.
from utils.separation_dataset import Separation_dataset, FUSSDataset
import torch

if __name__ == '__main__':
    import json 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/TIMIT_beamforming.json')
    args = parser.parse_args()
    config = json.load(open(args.config, 'r'))

    if config['dataset'] == 'FUSS': # the universal sound separation dataset
        train_dataset = FUSSDataset(config['train_datafolder'], config['model']['sample_rate'])
        val_dataset = FUSSDataset(config['test_datafolder'], config['model']['sample_rate'])
        print('train dataset {}, test dataset {}'.format(len(train_dataset), len(val_dataset)))
    elif config['dataset'] == 'TIMIT':
        train_dataset = Separation_dataset(config['train_datafolder'], config,)
        val_dataset = Separation_dataset(config['test_datafolder'], config,)
        print('train dataset {}, test dataset {}'.format(len(train_dataset), len(val_dataset)))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    
    if config['model_name'] == 'DeepBeam':
        model = BeamformerModel(ch_in=4, synth_mid=64, synth_hid=96, block_size=16, kernel=3, synth_layer=4, synth_rep=4, lookahead=0)

    #loss = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    # loss = PITLossWrapper(PairwiseNegSDR_active("sisdr"), pit_from="pw_mtx") 
    # loss = torch.nn.functional.l1_loss
    loss = singlesrc_neg_sisdr


    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    system = System(model, optimizer, loss, train_loader, val_loader)
    # Train for 1 epoch using a single GPU. If you're running this on Google Colab,
    # be sure to select a GPU runtime (Runtime → Change runtime type → Hardware accelarator).
    trainer = Trainer(max_epochs=20, devices=1, num_nodes=1)
    trainer.fit(system)

    # ckpt = 'lightning_logs/version_1/checkpoints/epoch=19-step=10500.ckpt'
    # trainer.validate(system, val_loader,)
  