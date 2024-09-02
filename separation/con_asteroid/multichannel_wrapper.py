'''
Wrappers for the encoder/decoder/loss of asteroid
'''
import torch.nn as nn
import torch
from asteroid_filterbanks import make_enc_dec

def multichannel_make_enc_dec(num_channels = 2, **kwargs):
    '''
    Replace the encoder/decoder with multichannel wrapper
    '''
    encoder, decoder = make_enc_dec(**kwargs)
    encoder = Encoder_Wrapper(encoder, merge_type='cat', num_channels=num_channels)
    decoder = Decoder_Wrapper(decoder, merge_type='cat', num_channels=num_channels)
    return encoder, decoder
class Loss_Wrapper(nn.Module):
    def __init__(self, loss):
        super().__init__()
        self.loss = loss
    def forward(self, est_targets, targets):
        loss = self.loss(est_targets, targets)
        return loss.mean()


class Encoder_Wrapper(nn.Module):
    def __init__(self, module, merge_type='cat', num_channels=2):
        super().__init__()
        self.module = module
        if merge_type == 'cat':
            self.merge = torch.cat  
        elif merge_type == 'mean':
            self.merge = torch.mean
        self.num_channels = num_channels
        self.n_feats_out = module.n_feats_out * num_channels
    def forward(self, inputs):
        '''
        inputs: [B, C, T] for waveform,
        outpus: [B, F*C, T] for feature/spectrogram
        '''
        B, C = inputs.shape[:2]
        outputs = []
        assert C == self.num_channels
        for i in range(C):
            _outputs = self.module(inputs[:, i, :])
            outputs.append(_outputs)
        outputs = self.merge(outputs, dim=1)
        return outputs
    
class Decoder_Wrapper(nn.Module):
    def __init__(self, module,  merge_type='cat', num_channels=2):
        super().__init__()
        self.module = module
        self.num_channels = num_channels
        if merge_type == 'cat':
            self.merge = torch.cat  
        elif merge_type == 'mean':
            self.merge = torch.mean
    def forward(self, inputs):
        '''
        inputs: [B, num_spks, F*C, T] for feature/spectrogram,
        outpus: [B, C, T] for waveform
        '''
        B, num_spks, F = inputs.shape[:3]
        _F = F // self.num_channels
        inputs = inputs.reshape(B, num_spks * self.num_channels, _F, -1)
        outputs = []
        for i in range(self.num_channels):
            _outputs = self.module(inputs[:, i, :])
            outputs.append(_outputs)
        outputs = self.merge(outputs, dim=1)
        return outputs