from asteroid.models import DPRNNTasNet, ConvTasNet, SuDORMRFNet
import torch

class CLAP_SuDORMRFNet(SuDORMRFNet):
    def __init__(self, n_src, bn_chan=128, num_blocks=16, upsampling_depth=4, mask_act="softmax", in_chan=None, fb_name="free", kernel_size=21, n_filters=512, stride=None, sample_rate=8000, **fb_kwargs):
        super().__init__(n_src, bn_chan, num_blocks, upsampling_depth, mask_act, in_chan, fb_name, kernel_size, n_filters, stride, sample_rate, **fb_kwargs)
        self.clap_projector = torch.nn.Conv1d(n_src, n_src, 1)

    def forward(self, data):
        mix, clap_embedding = data
        output = super().forward(mix)
        return output
