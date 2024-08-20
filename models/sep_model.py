'''
We separate the audio by regions
'''

from models.tcnn import TCNN_Block
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

class SEP_Model(nn.Module):
    def __init__(self, backbone_config, classifier_config):
        super(SEP_Model, self).__init__()
        self.encoder = globals()[backbone_config['name']](backbone_config)
        self.decoder = globals()[classifier_config['name']](classifier_config)   
    def pretrained(self, fname):
        if not fname:
            return
        ckpt = torch.load('ckpts/' + fname + '/best.pth')
        self.load_state_dict(ckpt, strict=False)
        print('Pretrained model loaded {}'.format(fname))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



class Baseline_Encoder(nn.Module):
    def __init__(self, config):
        super(Baseline_Encoder, self).__init__()

        self.left_encoder = []; self.right_encoder = []
        channels = [1, 16, 32, 64, 128]
        for i in range(4):
            self.left_encoder.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=channels[i], out_channels=channels[i+1], kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm1d(num_features=channels[i+1]),
                    nn.PReLU()
                ))
            self.right_encoder.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=channels[i], out_channels=channels[i+1], kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm1d(num_features=channels[i+1]),
                    nn.PReLU()
                ))
        self.left_encoder = nn.Sequential(*self.left_encoder)
        self.right_encoder = nn.Sequential(*self.right_encoder)
    

        self.left_TCN = nn.Sequential(*[TCNN_Block(config['num_feature'], config['hidden_feature']) for _ in range(config['num_layer'])])
        self.right_TCN = nn.Sequential(*[TCNN_Block(config['num_feature'], config['hidden_feature']) for _ in range(config['num_layer'])])
    def forward(self, x):
        '''
        Binaural: [B, 2, T] with sr = 16000
        L, R: [B, 512, T/16]
        1. use linear transformation to reduce the dimension by 16
        '''
        left, right = x['Binaural'][:, :1], x['Binaural'][:, 1:]
        left = self.left_encoder(left)
        right = self.right_encoder(right)
        left = torch.cat([left, x['L']], dim=1)
        right = torch.cat([right, x['R']], dim=1)
        # use TCN to estimate the mask
        left = self.left_TCN(left) * left
        right = self.right_TCN(right) * right
        return left, right    

class Baseline_Decoder(nn.Module):
    def __init__(self, config):
        super(Baseline_Decoder, self).__init__()
        self.num_regions = config['num_regions']
        self.decoder = []
        channels = [640, 320, 160, 80] + [self.num_regions]
        for i in range(4):
            # 4 times upsample and keep the same length
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose1d(in_channels=channels[i], out_channels=channels[i+1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm1d(num_features=channels[i+1]),
                    nn.PReLU()
                ))
        self.decoder = nn.Sequential(*self.decoder)
    def get_loss(self, outputs, labels):
        '''
        L, R: [B, num_region, T]
        '''
        L, R = outputs
        loss = torch.nn.functional.mse_loss(L, labels.to(L.device)) + torch.nn.functional.mse_loss(R, labels.to(R.device))
        return loss
    def vis(self, outputs, labels, epoch, i):
        '''
        L, R: [B, num_region, T]
        '''
        B, num_region, T = outputs[0].shape
        fig, axs = plt.subplots(num_region, 1)
        for i in range(num_region):
            axs[i].plot(outputs[0][0, i].cpu().detach().numpy())
            axs[i].plot(labels[0, i].cpu().detach().numpy())
        plt.savefig('figs/{}_{}.png'.format(epoch, i))
        plt.close()

    def eval(self, outputs, labels):
        '''
        L, R: [B, num_region, T]
        '''
        L, R = outputs
        loss_L = torch.nn.functional.mse_loss(L, labels.to(L.device)) 
        loss_R = torch.nn.functional.mse_loss(R, labels.to(R.device))
        return {'loss_L': loss_L.item(), 'loss_R': loss_R.item()}
    def forward(self, x):
        '''
        L, R = X
        L/R: [B, C, T/16]
        Output: [B, num_region, T]
        '''  
        L, R = x
        L = self.decoder(L)
        R = self.decoder(R)
        return L, R
        
