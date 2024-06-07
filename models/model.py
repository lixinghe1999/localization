import torch.nn as nn
from models.basic_model import *
import torch
import numpy as np
from scipy.signal import find_peaks
from utils import minimal_distance, good_error
class Model(nn.Module):
    def __init__(self, backbone_config, classifier_config):
        super(Model, self).__init__()
        self.audiobackbone = globals()[backbone_config['name'] + '_backbone'](backbone_config)
        self.classifier = globals()[classifier_config['name'] + '_classifier'](classifier_config)
    def pretrained(self, fname):
        if not fname:
            return
        ckpt = torch.load('ckpts/' + fname + '/best.pth')
        self.load_state_dict(ckpt)
        for param in self.audiobackbone.parameters():
            param.requires_grad = False
    def forward(self, x):
        x = self.audiobackbone(x)
        x = self.classifier(x)
        return x
class Gaussian_classifier(nn.Module):
    def __init__(self, config):
        super(Gaussian_classifier, self).__init__()
        self.config = config
        self.fc_layers = nn.Sequential(
                        nn.Linear(self.config["num_feature"], self.config["hidden_feature"]),
                        nn.ReLU(),
                        nn.Linear(self.config["hidden_feature"], self.config["hidden_feature"]),
                        nn.Dropout(0.2),
                        nn.ReLU(),
                        nn.Linear(self.config["hidden_feature"], self.config["hidden_feature"]),
                        nn.Dropout(0.2),
                        nn.ReLU(),)
        self.azimuth = nn.Linear(self.config["hidden_feature"], int(config['max_azimuth'] - config['min_azimuth']))
        self.elevation = nn.Linear(self.config["hidden_feature"], int(config['max_elevation'] - config['min_elevation']))
        self.num_source = config['num_source']
    def forward(self, x):
        x = self.fc_layers(x)
        azimuth = torch.sigmoid(self.azimuth(x))
        elevation = torch.sigmoid(self.elevation(x))
        return (azimuth, elevation)
    def get_loss(self, pred, labels):      
        loss1 = nn.functional.mse_loss(pred[0], labels[0].to(pred[0].device))
        loss2 = nn.functional.mse_loss(pred[1], labels[1].to(pred[1].device))
        return loss1+loss2
    def vis(self, pred, labels, epoch, i):
        import matplotlib.pyplot as plt
        azimuth = pred[0].cpu().detach().numpy(); elevation = pred[1].cpu().detach().numpy()
        azimuth_label = labels[0].cpu().detach().numpy(); elevation_label = labels[1].cpu().detach().numpy()
        fig = plt.figure()
        a = azimuth[0]; e = elevation[0]; a_l = azimuth_label[0]; e_l = elevation_label[0]
        plt.plot(a/np.max(a)); 
        plt.plot(a_l)
        plt.savefig('figs/{}_{}.png'.format(epoch, i))
        plt.close()

    def eval(self, preds, labels):
        metric_dict = {'azimuth': [], 'elevation': []}
        for (azimuth, elevation), (azimuth_label, elevation_label) in zip(preds, labels):
            azimuth = azimuth.cpu().detach().numpy(); elevation = elevation.cpu().detach().numpy()
            azimuth_label = azimuth_label.cpu().detach().numpy(); elevation_label = elevation_label.cpu().detach().numpy()
            if self.num_source == 1:
                peak_azimuth, peak_elevation = np.argmax(azimuth, axis=1), np.argmax(elevation, axis=1)
                peak_azimuth_label, peak_elevation_label = np.argmax(azimuth_label, axis=1), np.argmax(elevation_label, axis=1)
                azimuth_loss = np.mean(good_error(peak_azimuth, peak_azimuth_label, 360))
                elevation_loss = np.mean(good_error(peak_elevation, peak_elevation_label, 180))
            else:
                azimuth_loss = []; elevation_loss = []
                for a, e, a_l, e_l in zip(azimuth, elevation, azimuth_label, elevation_label):
                    a = a/np.max(a); e = e/np.max(e)
                    azimuth_peaks = find_peaks(a, height=0.5, distance=10)[0]
                    elevation_peaks = find_peaks(e, height=0.5, distance=10)[0]
                    azimuth_peak_labels = find_peaks(a_l, height=0.5, distance=10)[0]
                    elevation_peak_labels = find_peaks(e_l, height=0.5, distance=10)[0]
                    match_azimuth = minimal_distance(azimuth_peaks, azimuth_peak_labels, 360)
                    match_elevation = minimal_distance(elevation_peaks, elevation_peak_labels, 180)
 
                    azimuth_loss.append(match_azimuth)
                    elevation_loss.append(match_elevation)
                azimuth_loss = np.mean(azimuth_loss)
                elevation_loss = np.mean(elevation_loss)
            metric_dict['azimuth'].append(azimuth_loss)
            metric_dict['elevation'].append(elevation_loss)

        metric_dict['azimuth'] = sum(metric_dict['azimuth']) / len(metric_dict['azimuth'])
        metric_dict['elevation'] = sum(metric_dict['elevation']) / len(metric_dict['elevation'])

        return metric_dict
class DeepBSL_classifier(nn.Module):
    def __init__(self, config):
        super(DeepBSL_classifier, self).__init__()
        self.config = config
        self.fc_layers = nn.Sequential(
                        nn.Linear(self.config["num_feature"], 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.Dropout(0.2),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.Dropout(0.2),
                        nn.ReLU(),
                        nn.Linear(128, 2),
                        nn.Sigmoid(),)
        self.num_source = config['num_source']
    def forward(self, x):
        x = self.fc_layers(x)
        return x
    def get_loss(self, pred, labels):      
        diff = torch.abs(pred - labels[:, 0, :].to(pred.device))
        diff = torch.min(diff, 1 - diff)
        sound_loss = diff.mean()
        return sound_loss
    def eval(self, preds, labels):
        metric_dict = {'azimuth': [], 'elevation': []}
        for pred, label in zip(preds, labels):
            if self.num_source == 1:
                labels = labels[:, 0, :].to(pred.device)
                sound_loss = torch.abs(pred - label)
                sound_loss = torch.min(sound_loss, 1 - sound_loss)
                azimuth_loss = sound_loss[:, 0].mean() * (self.config['max_azimuth'] - self.config['min_azimuth'])
                elevation_loss = sound_loss[:, 1].mean() * (self.config['max_elevation'] - self.config['min_elevation'])
                metric_dict['azimuth'].append(azimuth_loss.item())
                metric_dict['elevation'].append(elevation_loss.item())
            else:
                raise NotImplementedError

        metric_dict['azimuth'] = sum(metric_dict['azimuth']) / len(metric_dict['azimuth'])
        metric_dict['elevation'] = sum(metric_dict['elevation']) / len(metric_dict['elevation'])
        return metric_dict
class DeepBSL_backbone(nn.Module):
    def __init__(self, config):
        super(DeepBSL_backbone, self).__init__()
        for k, v in config['features'].items():
            if v == 0:
                continue
            else:
                setattr(self, k, globals()[k](v))
    def forward(self, x):
        feature_list = []
        for k, v in x.items():
            x = getattr(self, k)(v)
            feature_list.append(x)
        feat = torch.cat(feature_list, dim=1)
        return feat

