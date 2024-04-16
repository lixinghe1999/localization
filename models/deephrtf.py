import torch.nn as nn
import torch
class DeepHRTF_classifier(nn.Module):
    def __init__(self,):
        super(DeepHRTF_classifier, self).__init__()
        self.fc_layers = nn.Sequential(
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.Dropout(0.2),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.Dropout(0.2),
                        nn.ReLU(),
                        nn.Linear(128, 2),
                        nn.Sigmoid(),)
    def forward(self, x):
        x = self.fc_layers(x)
        return x
    def get_loss(self, pred, labels):      
        '''
        we normalize 0-360 to 0-1
        '''  
        diff = torch.abs(pred - labels[:, 0, :])
        diff = torch.min(diff, 1 - diff)
        sound_loss = diff.mean()
        return sound_loss
    def eval(self, pred, labels, single_source=True):
        if single_source:
            labels = labels[:, 0, :]
            sound_loss = torch.abs(pred - labels)
            sound_loss = torch.min(sound_loss, 1 - sound_loss)
            azimuth_loss = sound_loss[:, 0].mean() * 360
            elevation_loss = sound_loss[:, 1].mean() * 360
        metric_dict = {'azimuth': azimuth_loss.item(), 'elevation': elevation_loss.item()}
        return metric_dict
class DeepHRTF_backbone(nn.Module):
    def __init__(self,):
        super(DeepHRTF_backbone, self).__init__()
        self.fc_layer1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),)
        self.fc_layer2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),)
        self.fc_layer3 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),)
        
    def forward(self, x):
        x = x['HRTF']
        left = x[:, 0]
        right = x[:, 1]
        # left = self.fc_layer1(left)
        # left = self.fc_layer2(left)
        # left = self.fc_layer3(left)

        # right = self.fc_layer1(right)
        # right = self.fc_layer2(right)
        # right = self.fc_layer3(right)

        feat = torch.cat([left, right], dim=1)
        return feat