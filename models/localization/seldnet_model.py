# The SELDnet architecture

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x


class PositionalEmbedding(nn.Module):  # Not used in the baseline
    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self,rnn_size, nb_heads, dropout_rate):
        super().__init__()
        self.mhsa_block_list = nn.ModuleList()
        self.layer_norm_list = nn.ModuleList()
        for mhsa_cnt in range(params['nb_self_attn_layers']):
            self.mhsa_block_list.append(nn.MultiheadAttention(embed_dim=rnn_size, num_heads=nb_heads, dropout=dropout_rate,  batch_first=True))
            self.layer_norm_list.append(nn.LayerNorm(rnn_size))
    def forward(self, x):
        for mhsa_cnt in range(len(self.mhsa_block_list)):
            x_attn_in = x 
            x, _ = self.mhsa_block_list[mhsa_cnt](x_attn_in, x_attn_in, x_attn_in)
            x = x + x_attn_in
            x = self.layer_norm_list[mhsa_cnt](x)
        return x

params = dict(
        dropout_rate=0.05,           # Dropout rate, constant for all layers
        nb_cnn2d_filt=128,           # Number of CNN nodes, constant for each layer
        f_pool_size=[4, 4, 2],      # CNN frequency pooling, length of list = number of CNN layers, list value = pooling per layer

        self_attn=True,
        nb_heads=8,
        nb_self_attn_layers=2,
        
        nb_rnn_layers=2,
        rnn_size=128,

        nb_fnn_layers=1,
        fnn_size=128,
        )

class SeldModel(torch.nn.Module):
    def __init__(self, mic_channels=10, unique_classes=9, mel_bins=64, t_pool_size=[5, 1, 1], activation='sigmoid',
                  params=params):
        '''
        mic_chanels: 10 - 4mics, 3 - 2mics
        unique_classes: 9 - 3 classes * 3 axes for fine-grained, 6 - 4+2 (doa + distance)
        '''
        super().__init__()

        self.nb_classes = unique_classes
        self.params=params
        self.conv_block_list = nn.ModuleList()
        if len(params['f_pool_size']):
            for conv_cnt in range(len(params['f_pool_size'])):
                self.conv_block_list.append(ConvBlock(in_channels=params['nb_cnn2d_filt'] if conv_cnt else mic_channels, out_channels=params['nb_cnn2d_filt']))
                self.conv_block_list.append(nn.MaxPool2d((t_pool_size[conv_cnt], params['f_pool_size'][conv_cnt])))
                self.conv_block_list.append(nn.Dropout2d(p=params['dropout_rate']))

        self.gru_input_dim = params['nb_cnn2d_filt'] * int(np.floor(mel_bins / np.prod(params['f_pool_size'])))
        self.gru = torch.nn.GRU(input_size=self.gru_input_dim, hidden_size=params['rnn_size'],
                                num_layers=params['nb_rnn_layers'], batch_first=True,
                                dropout=params['dropout_rate'], bidirectional=True)
        self.pos_embedder = PositionalEmbedding(self.params['rnn_size'])

        self.multiheadattention = MultiHeadSelfAttention(params['rnn_size'], params['nb_heads'], params['dropout_rate'])

        self.fnn_list_doa = torch.nn.ModuleList()
        if params['nb_fnn_layers']:
            for fc_cnt in range(params['nb_fnn_layers']):
                self.fnn_list_doa.append(nn.Linear(params['fnn_size'] if fc_cnt else self.params['rnn_size'], params['fnn_size'], bias=True))
        self.fnn_list_doa.append(nn.Linear(params['fnn_size'] if params['nb_fnn_layers'] else self.params['rnn_size'], unique_classes, bias=True))


        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()
    def forward(self, x):
        """input: (batch_size, mic_channels, time_steps, mel_bins)"""
        for conv_cnt in range(len(self.conv_block_list)):
            x = self.conv_block_list[conv_cnt](x)

        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        (x, _) = self.gru(x)
        x = torch.tanh(x)
        x = x[:, :, x.shape[-1]//2:] * x[:, :, :x.shape[-1]//2]

        pos_embedding = self.pos_embedder(x)
        x = x + pos_embedding
        
        x = self.multiheadattention(x)
       
        for fnn_cnt in range(len(self.fnn_list_doa)):
            x = self.fnn_list_doa[fnn_cnt](x)

        doa = self.activation(x)
        return doa

class SeldModel_Mobile(SeldModel):
    def __init__(self, mic_channels=10, unique_classes=9, mel_bins=64, t_pool_size=[5, 1, 1], activation='sigmoid',
                  params=params):
        '''
        mic_chanels: 10 - 4mics, 3 - 2mics
        unique_classes: 9 - 3 classes * 3 axes for fine-grained, 6 - 4+2 (doa + distance)
        '''
        super().__init__(mic_channels, unique_classes, mel_bins, t_pool_size, activation, params) # Inherit from SeldModel

        self.imu_conv_block_list = nn.ModuleList()
        # the sampling rate of imu is 50Hz, so we need to same downsample as the audio (suppose the window is 0.02s)

        self.imu_proj = nn.Linear(6, params['nb_cnn2d_filt'])
        transformer = nn.TransformerEncoderLayer(d_model=params['nb_cnn2d_filt'], nhead=params['nb_heads'], dim_feedforward=params['fnn_size'], dropout=params['dropout_rate'])
        self.imu_transformer = nn.TransformerEncoder(transformer, num_layers=4)
        self.imu_norm = nn.GroupNorm(2, 6)
        self.imu_audio_proj = nn.Linear(params['nb_cnn2d_filt'], params['nb_cnn2d_filt'])
        self.t_downsample = t_pool_size[0] * t_pool_size[1] * t_pool_size[2]

    def forward(self, x, imu=None):
        """
        input: 
        audio - (batch_size, mic_channels, time_steps, mel_bins)
        imu - (batch_size, time_steps, 6)
        output:
        doa - (batch_size, time_steps, unique_classes)
        """
        for conv_cnt in range(len(self.conv_block_list)):
            x = self.conv_block_list[conv_cnt](x)
        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        (x, _) = self.gru(x)
        x = torch.tanh(x)
        x = x[:, :, x.shape[-1]//2:] * x[:, :, :x.shape[-1]//2]

        if imu is not None:
            imu = self.imu_norm(imu.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
            imu = self.imu_proj(imu)
            imu = self.imu_transformer(imu)
            # max_pooling by 50
            imu = F.max_pool1d(imu.transpose(1, 2).contiguous(), kernel_size=self.t_downsample).transpose(1, 2).contiguous()
            x = x + self.imu_audio_proj(imu)

        pos_embedding = self.pos_embedder(x)
        x = x + pos_embedding
        
        x = self.multiheadattention(x)
       
        for fnn_cnt in range(len(self.fnn_list_doa)):
            x = self.fnn_list_doa[fnn_cnt](x)
        doa = self.activation(x)
        return doa

if __name__ == '__main__':
    
    batch_size, mic_channels, time_steps, mel_bins = 64, 10, 500, 64
    model = SeldModel()
    x = torch.randn(batch_size, mic_channels, time_steps, mel_bins, )
    y = model(x)
    print(y.shape)  # Expected output torch.Size([64, 50, 13])