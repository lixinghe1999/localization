# Asteroid is based on PyTorch and PyTorch-Lightning.
from torch import optim
from pytorch_lightning import Trainer

# We train the same model architecture that we used for inference above.
from asteroid.models import DPRNNTasNet, ConvTasNet, SuDORMRFNet
from models.voicefilter import VoiceFilterNet

# In this example we use Permutation Invariant Training (PIT) and the SI-SDR loss.
from asteroid.losses import pairwise_neg_sisdr, PITLossWrapper, singlesrc_neg_sisdr

# MiniLibriMix is a tiny version of LibriMix (https://github.com/JorisCos/LibriMix),
# which is a free speech separation dataset.
from asteroid.data import LibriMix

# Asteroid's System is a convenience wrapper for PyTorch-Lightning.
from asteroid.engine import System
from utils.beamforming_dataset import Beamforming_dataset
from utils.separation_dataset import FUSSDataset
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
    train_dataset = FUSSDataset('dataset/FUSS/ssdata/', 'dataset/FUSS/ssdata/train_example_list.txt', n_src=2, 
                                duration=10, sample_rate=8000, mode='separation')
    test_dataset = FUSSDataset('dataset/FUSS/ssdata/', 'dataset/FUSS/ssdata/validation_example_list.txt', n_src=2, 
                               duration=10, sample_rate=8000, mode='separation')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

    model = SuDORMRFNet(n_src=2, num_blocks=4, sample_rate=8000)
    # model = CLAP_SuDORMRFNet(n_src=1, num_blocks=8, sample_rate=8000)
    # model = VoiceFilterNet(n_src=1, sample_rate=8000)

    # PITLossWrapper works with any loss function.
    # loss = dynamic_source_wrapper(pai  rwise_neg_sisdr, pit_from="pw_mtx")
    # loss = Penalized_PIT_Wrapper(pairwise_neg_sisdr_loss_v2, penalty=30)
    loss = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    system = System(model, optimizer, loss, train_loader, test_loader)
    trainer = Trainer(max_epochs=50, devices=[0])
    trainer.fit(system)

