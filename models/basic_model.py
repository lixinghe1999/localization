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

class raw(torch.nn.Module):
    def __init__(self, feature):
        super().__init__()
        self.tcnn = TCNN_Encoder(feature['input_feature'], feature['output_feature'])
        self.audio_chunk = 320
    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], -1, self.audio_chunk)
        x = self.tcnn(x)
        return x


class mel_spec(CNN):
    def __init__(self, feature):
        super(mel_spec, self).__init__(feature)
class gccphat(MLP):
    def __init__(self, feature):
        super(gccphat, self).__init__(feature)