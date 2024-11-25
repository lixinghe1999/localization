# Asteroid is based on PyTorch and PyTorch-Lightning.
from torch import optim
from pytorch_lightning import Trainer

# We train the same model architecture that we used for inference above.
from asteroid.models import DPRNNTasNet, ConvTasNet, SuDORMRFNet

# In this example we use Permutation Invariant Training (PIT) and the SI-SDR loss.
from asteroid.losses import pairwise_neg_sisdr, PITLossWrapper

# MiniLibriMix is a tiny version of LibriMix (https://github.com/JorisCos/LibriMix),
# which is a free speech separation dataset.
from asteroid.data import LibriMix

# Asteroid's System is a convenience wrapper for PyTorch-Lightning.
from asteroid.engine import System
from utils.beamforming_dataset import Beamforming_dataset
from utils.fuss_dataset import FUSSDataset
import torch

class dynamic_source_wrapper(PITLossWrapper):
    def __init__(self, loss_func, pit_from="pw_mtx", perm_reduce=None):
        super().__init__(loss_func, pit_from, perm_reduce)

    def forward(self, est_targets, targets, return_est=False, reduce_kwargs=None, **kwargs):
        num_sources = targets.shape[1]
        active_sources = torch.sum(targets ** 2, dim=-1, keepdim=True) > 0

        active_sources = active_sources.permute(0, 2, 1) # [batch, 1, targrt]
        active_sources = torch.tile(active_sources, (1, num_sources, 1)) # [batch, pred, target]
        inactive_sources = ~active_sources
        pw_loss = self.loss_func(est_targets, targets) # [batch, pred, target(num of sources)]
        if self.training:
            inactive_loss = 10 * torch.log10(torch.mean(est_targets ** 2, dim=-1, keepdim=True) + 1e-8)
            inactive_loss = torch.tile(inactive_loss, (1, 1, num_sources)) 
            pw_loss[inactive_sources] = inactive_loss[inactive_sources]
            reduce_kwargs = reduce_kwargs if reduce_kwargs is not None else dict()
            min_loss, batch_indices = self.find_best_perm(
                pw_loss, perm_reduce=self.perm_reduce, **reduce_kwargs
            )
            mean_loss = torch.mean(min_loss)
            if not return_est:
                return mean_loss
            reordered = self.reorder_source(est_targets, batch_indices)
            return mean_loss, reordered
        else:
            # only calculate the loss of active sources, get the minimum loss
            pw_loss = torch.min(pw_loss, dim=1, keepdim=True)[0]
            pw_loss = torch.mean(pw_loss[active_sources[:, :1, :]], dim=-1)
            active_loss = torch.mean(pw_loss)
            return active_loss
        

        
        
if __name__ == '__main__':
    # config = { "train_datafolder": "dataset/smartglass/TIMIT_2/train",
    #             "test_datafolder": "dataset/smartglass/TIMIT_2/test",
    #             "ckpt": "",
    #             "duration": 5,
    #             "epochs": 20,
    #             "batch_size": 8,
    #             "output_format": "separation",
    #             "sample_rate": 8000,
    #             "max_sources": 2,
    #         }
    # train_dataset = Beamforming_dataset(config['train_datafolder'], config,)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)

    # test_dataset = Beamforming_dataset(config['test_datafolder'], config,)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    train_dataset = FUSSDataset('dataset/FUSS/ssdata/', 'dataset/FUSS/ssdata/train_example_list.txt', n_src=2, duration=10, sample_rate=8000)
    test_dataset = FUSSDataset('dataset/FUSS/ssdata/', 'dataset/FUSS/ssdata/validation_example_list.txt', n_src=2, duration=10, sample_rate=8000)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

    # model = DPRNNTasNet(n_src=3, sample_rate=16000)
    # model = DPRNNTasNet.from_pretrained('mpariente/DPRNNTasNet-ks2_WHAM_sepclean')
    # model = ConvTasNet.from_pretrained('JorisCos/ConvTasNet_Libri2Mix_sepclean_8k')
    # model = ConvTasNet(n_src=2, sample_rate=8000)
    model = SuDORMRFNet(n_src=2, sample_rate=8000)


    # PITLossWrapper works with any loss function.
    loss = dynamic_source_wrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    # loss = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    system = System(model, optimizer, loss, train_loader, test_loader)

    # Train for 1 epoch using a single GPU. If you're running this on Google Colab,
    # be sure to select a GPU runtime (Runtime → Change runtime type → Hardware accelarator).
    trainer = Trainer(max_epochs=50, devices=[1])
    trainer.fit(system)
