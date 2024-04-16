import torch.nn as nn
import torch
class DeepBSL_classifier(nn.Module):
    def __init__(self,):
        super(DeepBSL_classifier, self).__init__()
        self.fc_layers = nn.Sequential(
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
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
        # pred =  pred * 2 * 3.1415926
        # labels = labels * 2 * 3.1415926
        # A1 = torch.sin((pred[..., 0] - labels[:, 0, 0])/2) ** 2 
        # A2 = torch.cos(pred[..., 0]) * torch.cos(labels[:, 0, 0]) * torch.sin((pred[..., 1] - labels[:, 0, 1])/2) ** 2
        # A = A1 + A2
        # sound_loss = 2 * torch.atan(torch.sqrt(A)/torch.sqrt(1 - A)).mean()
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
class DeepBSL_backbone(nn.Module):
    def __init__(self,):
        super(DeepBSL_backbone, self).__init__()
        self.gtcc = DeepBSL_GTCC()
        self.gccphat = DeepBSL_GCCPHAT()
    def forward(self, x):
        gccphat = x['gcc_phat']
        gtcc = x['gammatone']
        gccphat = self.gccphat(gccphat)
        gtcc = self.gtcc(gtcc)
        gtcc = torch.nn.functional.adaptive_avg_pool2d(gtcc, (1, 1)).squeeze()
        # gtcc = torch.flatten(gtcc, 1)
        feat = torch.cat([gtcc, gccphat], dim=1)
        return feat
class DeepBSL_GTCC(nn.Module):
    def __init__(self,):
        super(DeepBSL_GTCC, self).__init__()
        self.layer_1 = nn.Sequential(
                nn.Conv2d(2, 32, (3, 3), padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),)
        self.layer_2 = nn.Sequential(
                nn.Conv2d(32, 64, (3, 3), padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),)
        self.layer_3 = nn.Sequential(
                nn.Conv2d(64, 64, (3, 3), padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),)
        self.maxpool = nn.MaxPool2d((2, 2))
    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.maxpool(x)
        return x
class DeepBSL_GCCPHAT(nn.Module):
    def __init__(self,):
        super(DeepBSL_GCCPHAT, self).__init__()
        self.fc_layer1 = nn.Sequential(
            nn.Linear(48, 64),
            nn.ReLU(),)
        self.fc_layer2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2))
        self.fc_layer3 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2))
    def forward(self, x):
        x = self.fc_layer1(x)
        x = self.fc_layer2(x)
        x = self.fc_layer3(x)
        return x

