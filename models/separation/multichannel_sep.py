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
        self.refine_net = Net(num_ch=1 + n_src, num_src=1 * n_src, D=16, B=4)

    def _slow_forward(self, x):
        B, C, T = x.shape
        _xs = []
        for c in range(C):
            ref_channel = x[:, c:c+1]
            separated = self.separator(ref_channel)
            _xs.append(separated.unsqueeze(2))
        _x = torch.cat(_xs, dim=2)
        return _x
    def forward(self, x, separated=None):
        return self._slow_forward(x)
        # x: [batch, channel, time]
        if separated is None:
            ref_channel = x[:, :1]
            separated = self.separator(ref_channel)
            
        B, C, T = x.shape
        _xs = [separated.unsqueeze(2)] # [batch, source, 1, time]
        for c in range(1, C):
            _x = torch.cat([x[:, c:c+1], separated], dim=1) # [batch, 1 + source, time]
            _x = self.refine_net(_x) # [batch, source, time]
            _x = _x.unsqueeze(2)
            _xs.append(_x)
        _x = torch.cat(_xs, dim=2) # [batch, source, channel, time]
        return _x
        