from asteroid.models import DPRNNTasNet, ConvTasNet, SuDORMRFNet
from ..beamforming.tfgridnet_realtime.net import Net
import torch
import torch.nn as nn

class MultiChannel_Sep(nn.Module):
    def __init__(self, separator, n_channel, n_src, sample_rate=8000):
        super(MultiChannel_Sep, self).__init__()
        if separator == 'DPRNNTasNet':
            self.separator = DPRNNTasNet(n_src=n_src)
        elif separator == 'ConvTasNet':
            self.separator = ConvTasNet(n_src=n_src)
        elif separator == 'SuDORMRFNet':
            self.separator = SuDORMRFNet(n_src=n_src)
        else:
            raise ValueError(f"Unknown separator: {separator}")
        self.n_channel = n_channel
        self.n_src = n_src
        self.sample_rate = sample_rate
        self.refine_net = Net(num_ch=n_channel + n_src, num_src=n_channel * n_src)
    def forward(self, x, separated=None):
        # x: [batch, channel, time]
        B, C, T = x.shape
        if separated is None:
            ref_channel = x[:, :1]
            separated = self.separator(ref_channel) # [batch, source, time]

        _x = torch.cat([x, separated], dim=1) # [batch, channel + source, time]
        _x = self.refine_net(_x)
        _x = _x.reshape(B, self.n_src, self.n_channel, T)
        return _x
        