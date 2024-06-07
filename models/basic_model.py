import torch
import torch.nn as nn
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
class Transformer(torch.nn.Module):
    def __init__(self, feature):
        super().__init__()
        self.fc = nn.Linear(feature['input_feature'], feature['hidden_feature'])
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature['hidden_feature'], nhead=4, dim_feedforward=feature['hidden_feature'])
        self.model = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)
    def forward(self, batch):
        return NotImplementedError

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