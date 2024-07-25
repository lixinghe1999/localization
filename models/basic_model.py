'''
all the models are data to feature model
'''

import torch
import torch.nn as nn
from models.tcnn import TCNN_Encoder
class CNN(nn.Module):
    def __init__(self, feature):
        super(CNN, self).__init__()
        self.layers = []
        for i in range(feature['num_layer']):
            if i == 0:
                i_feature = feature['input_feature']; o_feature = feature['hidden_feature']
            elif i == (feature['num_layer'] - 1):
                i_feature = feature['hidden_feature'] * 2 ** (i-1); o_feature = feature['output_feature']
            else:
                i_feature = feature['hidden_feature'] * 2 ** (i-1); o_feature = feature['hidden_feature'] * 2 ** i
            self.layers.append(nn.Sequential(
                nn.Conv2d(i_feature, o_feature, (3, 3), padding=1),
                nn.BatchNorm2d(o_feature),
                nn.ReLU(),))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.layers(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))[:, :, 0, 0]
        return x
class MLP(nn.Module):
    def __init__(self, feature):
        super(MLP, self).__init__()
        self.fc_layer1 = nn.Sequential(
            nn.Linear(feature['input_feature'], feature['hidden_feature']),
            nn.ReLU(),
            nn.BatchNorm1d(feature['hidden_feature']))
        self.fc_layer2 = nn.Sequential(
            nn.Linear(feature['hidden_feature'], feature['hidden_feature']),
            nn.ReLU(),
            nn.BatchNorm1d(feature['hidden_feature']),
            nn.Dropout(0.2))
        self.fc_layer3 = nn.Sequential(
            nn.Linear(feature['hidden_feature'], feature['output_feature']),
            nn.ReLU(),
            nn.BatchNorm1d(feature['output_feature']),
            nn.Dropout(0.2))
    def forward(self, x):
        x = self.fc_layer1(x)
        x = self.fc_layer2(x)
        x = self.fc_layer3(x)
        return x

# class raw(torch.nn.Module):
#     def __init__(self, feature):
#         super().__init__()
#         self.cnn = CNN(feature)
#     def forward(self, x):
#         batch, channel, time = x.shape
#         if len(x.shape) == 3:
#             x = x.reshape(-1, x.shape[2])
#         stft = torch.stft(x, n_fft=512, hop_length=160, win_length=400, window=torch.hann_window(400).cuda(), return_complex=False)
#         stft = torch.sqrt(stft[..., 0] ** 2 + stft[..., 1] ** 2)
#         stft = torch.log(stft + 1e-5).reshape(batch, channel, 257, -1)
#         return self.cnn(stft)

class raw(torch.nn.Module):
    def __init__(self, feature):
        super().__init__()
        self.tcnn = TCNN_Encoder(feature['input_feature'], feature['output_feature'])
        self.audio_chunk = 320
    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], -1, self.audio_chunk)
        x = self.tcnn(x)
        return x


class stft(CNN):
    def __init__(self, feature):
        super(stft, self).__init__(feature)
class mel_gccphat(CNN):
    def __init__(self, feature):
        super(mel_gccphat, self).__init__(feature)
class gtcc(CNN):
    def __init__(self, feature):
        super(gtcc, self).__init__(feature)
class gccphat(MLP):
    def __init__(self, feature):
        super(gccphat, self).__init__(feature)

class hrtf(nn.Module):
    def __init__(self, feature):
        super(hrtf, self).__init__()
        self.fc_layer1 = nn.Sequential(
            nn.Linear(feature['input_feature'], feature['hidden_feature']),
            nn.ReLU(),)
        self.fc_layer2 = nn.Sequential(
            nn.Linear(feature['hidden_feature'], feature['hidden_feature']),
            nn.ReLU(),)
        self.fc_layer3 = nn.Sequential(
            nn.Linear(feature['hidden_feature'], feature['output_feature']),
            nn.ReLU(),)
        
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        assert x.shape[1] == 512
        x = self.fc_layer1(x)
        x = self.fc_layer2(x)
        x = self.fc_layer3(x)
        return x