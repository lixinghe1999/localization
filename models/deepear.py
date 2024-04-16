import torch.nn as nn
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
n_fft = 1024
num_sectors = 8
sector_degree = 360 / 8
num_range = 5
sector_range = 1
max_source = 3
class sector_subnet(nn.Module):
    def __init__(self,):
        super(sector_subnet, self).__init__()
        self.fc_layer = nn.Linear(348, 256)
        self.sound_net = nn.Sequential(
                nn.Linear(256, 100),
                nn.ReLU(),
                nn.Linear(100, 10),
                nn.ReLU(),
                nn.Linear(10, 1),
                nn.Sigmoid(),
        )
        self.aoa_net = nn.Sequential(
                nn.Linear(256, 100),
                nn.ReLU(),
                nn.Linear(100, 10),
                nn.ReLU(),
                nn.Linear(10, 1),
                nn.Sigmoid(),
        )
        self.dis_net = nn.Sequential(
                nn.Linear(256, 100),
                nn.ReLU(),
                nn.Linear(100, 10),
                nn.ReLU(),
                nn.Linear(10, 5),
                nn.Softmax(dim=1),
        )
    def forward(self, x):
        x = self.fc_layer(x)
        sound = self.sound_net(x)
        aoa = self.aoa_net(x)
        dis = self.dis_net(x)
        out = torch.cat((sound, aoa, dis), 1)
        return out
class DeepEAR_classifier(nn.Module):
    def __init__(self,):
        super(DeepEAR_classifier, self).__init__()
        # self.sector_classifier = nn.ModuleList()
        # for i in range(num_sectors):
        #     self.sector_classifier.append(sector_subnet())
        self.fc_layer = nn.Linear(348, 256)
        self.sound_net = nn.Sequential(
                nn.Linear(256, 100),
                nn.ReLU(),
                nn.Linear(100, 10),
                nn.ReLU(),
                nn.Linear(10, 8),
        )
    def forward(self, x):
        # sectors = []
        # for i in range(num_sectors):
        #     sector_output = self.sector_classifier[i](x)
        #     sectors.append(sector_output)
        # sectors = torch.stack(sectors, 1)
        sectors = self.sound_net(self.fc_layer(x))
        return sectors
    def get_loss(self, sectors, labels):      
        sound_loss = torch.nn.functional.cross_entropy(sectors, labels[..., 0])
        # sound_loss = torch.nn.functional.binary_cross_entropy_with_logits(sectors, labels[..., 0])
        # sound_loss = torch.nn.functional.cross_entropy(sectors[..., 0], labels[..., 0])
        # aoa_loss = torch.nn.functional.mse_loss(sectors[..., 1], labels[..., 1])
        # dis_loss = torch.nn.functional.cross_entropy(sectors[..., 2:].reshape(-1, 5), labels[..., 2:].reshape(-1, 5))
        # return sound_loss * 0.4 + aoa_loss * 0.35 + dis_loss * 0
        return sound_loss
    def eval(self, sectors, labels, single_source=False):
        if single_source:
            pred_sec = sectors.argmax(1)
            target_sec = labels[..., 0].argmax(1)
            sound_acc = (pred_sec == target_sec).sum() / sectors.shape[0]
        else:
            #print(sectors, labels[..., 0])
            sectors = (torch.sigmoid(sectors) > 0.5).float()
            sound_acc = average_precision_score(sectors.detach().cpu().numpy(), labels[..., 0].cpu().numpy())
        return sound_acc, 0, 0
    def visualize_feature(self, audio_feature, name):
        import matplotlib.pyplot as plt
        r = audio_feature['gcc_phat'][0]
        gammatone = audio_feature['gammatone'][0]
        plt.subplot(311)
        plt.plot(r[0])
        plt.subplot(312)
        plt.imshow(gammatone[0].T, aspect='auto', cmap='jet')
        plt.subplot(313)
        plt.imshow(gammatone[1].T, aspect='auto')
        plt.savefig(name)
class DeepEAR_backbone(nn.Module):
    def __init__(self,):
        super(DeepEAR_backbone, self).__init__()
        self.gru = nn.GRU(50, 100, 2, batch_first=True)
        # self.gru2 = nn.GRU(200, 100, 1, batch_first=True)
        # self.mobilenet = mobilenet_v2()
        # self.fc_layer = nn.Linear(1280, 100)
    def forward(self, x):
        gccphat = x['gcc_phat']
        gtcc = x['gammatone']
        batch, channel, T, F = gtcc.shape
        gtcc = gtcc.reshape(-1, T, F)
        gtcc, _ = self.gru(gtcc)
        # gtcc, _ = self.gru2(gtcc)
        gtcc = gtcc[:, -1, :]
        gtcc = gtcc.reshape(batch, channel, 100)
        left_feat = gtcc[:, 0]
        right_feat = gtcc[:, 1]

        # left_feat = self.fc_layer(self.mobilenet(gtcc[:, :1]))
        # right_feat = self.fc_layer(self.mobilenet(gtcc[:, 1:]))

        res_feat = left_feat - right_feat
        feat = torch.cat((left_feat, right_feat, res_feat, gccphat), 1)
        return feat