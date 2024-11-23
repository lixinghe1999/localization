# Asteroid is based on PyTorch and PyTorch-Lightning.
from torch import optim
from pytorch_lightning import Trainer

# We train the same model architecture that we used for inference above.
from asteroid.models import DPRNNTasNet

# In this example we use Permutation Invariant Training (PIT) and the SI-SDR loss.
from asteroid.losses import pairwise_neg_sisdr, PITLossWrapper

# MiniLibriMix is a tiny version of LibriMix (https://github.com/JorisCos/LibriMix),
# which is a free speech separation dataset.
from asteroid.data import LibriMix

# Asteroid's System is a convenience wrapper for PyTorch-Lightning.
from asteroid.engine import System
from utils.beamforming_dataset import Beamforming_dataset
import torch
if __name__ == '__main__':
    config = { "train_datafolder": "dataset/smartglass/AudioSet_2/train",
                "test_datafolder": "dataset/smartglass/AudioSet_2/test",
                "ckpt": "",
                "duration": 5,
                "epochs": 20,
                "batch_size": 2,
                "output_format": "separation",
                "sample_rate": 8000,
                "max_sources": 2,
            }
    train_dataset = Beamforming_dataset(config['train_datafolder'], config,)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)

    test_dataset = Beamforming_dataset(config['test_datafolder'], config,)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=1)
    # This will automatically download MiniLibriMix from Zenodo on the first run.

    # Tell DPRNN that we want to separate to 2 sources.
    # model = DPRNNTasNet(n_src=2)
    model = DPRNNTasNet.from_pretrained('mpariente/DPRNNTasNet-ks2_WHAM_sepclean')


    # PITLossWrapper works with any loss function.
    loss = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    system = System(model, optimizer, loss, train_loader, test_loader)

    # Train for 1 epoch using a single GPU. If you're running this on Google Colab,
    # be sure to select a GPU runtime (Runtime → Change runtime type → Hardware accelarator).
    trainer = Trainer(max_epochs=10, devices=[0, 1])
    trainer.fit(system)