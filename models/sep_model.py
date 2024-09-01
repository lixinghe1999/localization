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

        self.encoders = []
        channels = [2, 16, 32, 64, 128]
        for i in range(4):
            self.encoders.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=channels[i], out_channels=channels[i+1], kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm1d(num_features=channels[i+1]),
                    nn.PReLU()
                ))
        self.encoders = nn.Sequential(*self.encoders)
        self.TCN = TCNN_Block(config['num_feature'], config['hidden_feature'], num_layers=config['num_layer'])
        self.mapping = nn.Linear(config['num_feature'], 128)
    def forward(self, x):
        '''
        Binaural: [B, 2, T] with sr = 16000
        encode: [B, 512, T/16]
        cat_encode_audio: [B, 512 + 128, T/16]
        '''
        encode_audio = self.encoders(x['Binaural'])
        # cat_encode_audio = torch.cat([encode_audio, x['feature']], dim=1)
        # # use TCN to estimate the mask
        # cat_encode_audio = self.TCN(cat_encode_audio)
        # encode_audio_mask = self.mapping(cat_encode_audio.permute(0, 2, 1)).permute(0, 2, 1)
        # encode_audio = torch.sigmoid(encode_audio_mask) * encode_audio
        return encode_audio    

class Baseline_Decoder(nn.Module):
    def __init__(self, config):
        super(Baseline_Decoder, self).__init__()
        self.num_regions = config['num_regions']
        self.decoder = []
        channels = [128, 64, 32, 16] + [self.num_regions]
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
        loss = torch.nn.functional.mse_loss(outputs, labels.to(outputs.device)) 
        return loss
    def vis(self, outputs, labels, epoch, idx):
        B, num_region, T = outputs.shape
        fig, axs = plt.subplots(num_region, 1)
        for i in range(num_region):
            axs[i].plot(outputs[0, i].cpu().detach().numpy())
            axs[i].plot(labels[0, i].cpu().detach().numpy())
        plt.savefig('figs/{}_{}.png'.format(epoch, idx))
        plt.close()

    def eval(self, outputs, labels):
        loss = torch.nn.functional.mse_loss(outputs, labels.to(outputs.device))
        return {'l2_loss': loss.item()}
    def forward(self, x):
        x = self.decoder(x)
        x = torch.tanh(x)
        return x
        
