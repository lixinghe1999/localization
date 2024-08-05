import torch.nn as nn
from models.basic_model import *
import torch
import numpy as np
from scipy.signal import find_peaks
from models.tcnn import TCNN

class Model(nn.Module):
    def __init__(self, backbone_config, classifier_config):
        super(Model, self).__init__()
        self.audiobackbone = globals()[backbone_config['name']](backbone_config)
        self.classifier = globals()[classifier_config['name']](classifier_config)
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
class Translate_Model(nn.Module):
    def __init__(self, backbone_config, classifier_config):
        super(Translate_Model, self).__init__()
        self.audio_chunk = 320
        self.translate_backbone = TCNN(backbone_config['translation']['input_channel'], backbone_config['translation']['output_channel'])
        self.audiobackbone = globals()[backbone_config['name'] + '_backbone'](backbone_config)
        self.classifier = globals()[classifier_config['name'] + '_classifier'](classifier_config)
    def pretrained(self, fname):
        if not fname:
            return
        ckpt = torch.load('ckpts/' + fname + '/best.pth')
        self.load_state_dict(ckpt, strict=False)
        for param in self.audiobackbone.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = False
        print('Pretrained model loaded {}'.format(fname))

    def forward(self, x):
        audio = x['raw']
        audio = audio.reshape(audio.shape[0], audio.shape[1], -1, self.audio_chunk)
        audio = self.translate_backbone(audio)

        audio = audio.reshape(audio.shape[0], audio.shape[1], -1)
        x['raw'] = audio
        x = self.audiobackbone(x)
        x = self.classifier(x)
        return x
class Gaussian(nn.Module):
    def __init__(self, config):
        super(Gaussian, self).__init__()
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
    def calculate_error(self, pred, labels, saturation):
        error = np.min([np.abs(pred - labels), saturation - np.abs(pred - labels)], axis=0)
        return np.mean(error)
    def best_match(self, pred, labels, saturation):
        errors = []
        for l in labels:
            dis_min = 1000
            for p in pred:
                dis = self.calculate_error(p, l, saturation)
                if dis < dis_min:
                    dis_min = dis
            errors.append(dis_min)
        return np.mean(errors)
            

    def eval(self, preds, labels):
        metric_dict = {'azimuth': [], 'elevation': []}
        for (azimuth, elevation), (azimuth_label, elevation_label) in zip(preds, labels):
            azimuth = azimuth.cpu().detach().numpy(); elevation = elevation.cpu().detach().numpy()
            azimuth_label = azimuth_label.cpu().detach().numpy(); elevation_label = elevation_label.cpu().detach().numpy()
            if self.num_source == 1:
                peak_azimuth, peak_elevation = np.argmax(azimuth, axis=1), np.argmax(elevation, axis=1)
                peak_azimuth_label, peak_elevation_label = np.argmax(azimuth_label, axis=1), np.argmax(elevation_label, axis=1)
                azimuth_loss = self.calculate_error(peak_azimuth, peak_azimuth_label, 360)
                elevation_loss = self.calculate_error(peak_elevation, peak_elevation_label, 180)
            else:
                azimuth_loss = []; elevation_loss = []
                for a, e, a_l, e_l in zip(azimuth, elevation, azimuth_label, elevation_label):
                    a = a/np.max(a); e = e/np.max(e)
                    azimuth_peaks = find_peaks(a, height=0.5, distance=10)[0]
                    elevation_peaks = find_peaks(e, height=0.5, distance=10)[0]
                    azimuth_peak_labels = find_peaks(a_l, height=0.5, distance=10)[0]
                    elevation_peak_labels = find_peaks(e_l, height=0.5, distance=10)[0]
                    match_azimuth = self.best_match(azimuth_peaks, azimuth_peak_labels, 360)
                    match_elevation = self.best_match(elevation_peaks, elevation_peak_labels, 180)
 
                    azimuth_loss.append(match_azimuth)
                    elevation_loss.append(match_elevation)
                azimuth_loss = np.mean(azimuth_loss)
                elevation_loss = np.mean(elevation_loss)
            metric_dict['azimuth'].append(azimuth_loss)
            metric_dict['elevation'].append(elevation_loss)

        metric_dict['azimuth'] = sum(metric_dict['azimuth']) / len(metric_dict['azimuth'])
        metric_dict['elevation'] = sum(metric_dict['elevation']) / len(metric_dict['elevation'])

        return metric_dict
class Distance(nn.Module):
    def __init__(self, config):
        super(Distance, self).__init__()
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
        self.distance = nn.Linear(self.config["hidden_feature"], 1)
    def forward(self, x):
        x = self.fc_layers(x)
        distance = torch.sigmoid(self.distance(x))
        return distance
    def get_loss(self, pred, labels):      
        loss = nn.functional.mse_loss(pred, labels.to(pred.device))
        return loss
    def vis(self, pred, labels, epoch, i):
        import matplotlib.pyplot as plt
        distance = pred.cpu().detach().numpy()
        distance_label = labels.cpu().detach().numpy()
        fig = plt.figure()
        d = distance[0]; d_l = distance_label[0]
        plt.plot(d/np.max(d)); 
        plt.plot(d_l)
        plt.savefig('figs/{}_{}.png'.format(epoch, i))
        plt.close()
    def eval(self, preds, labels):
        metric_dict = {'distance': []}
        for distance, distance_label in zip(preds, labels):
            distance = distance.cpu().detach()
            distance_label = distance_label.cpu().detach()
            distance_loss = nn.functional.l1_loss(distance, distance_label).item()
            metric_dict['distance'].append(distance_loss)
        metric_dict['distance'] = sum(metric_dict['distance']) / len(metric_dict['distance'])
        return metric_dict
class Cartesian(nn.Module):
    def __init__(self, config):
        super(Cartesian, self).__init__()
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
        self.xyz = nn.Linear(self.config["hidden_feature"], 3)
    def forward(self, x):
        x = self.fc_layers(x)
        xyz = self.xyz(x)
        return xyz
    def get_loss(self, pred, labels):
        loss = nn.functional.mse_loss(pred, labels.to(pred.device))
        return loss
    def vis(self, pred, labels, epoch, i):
        import matplotlib.pyplot as plt
        xyz = pred.cpu().detach().numpy()
        xyz_label = labels.cpu().detach().numpy()
        fig = plt.figure()
        x = xyz[0]; x_l = xyz_label[0]
        plt.plot(x/np.max(x)); 
        plt.plot(x_l)
        plt.savefig('figs/{}_{}.png'.format(epoch, i))
        plt.close()
    def eval(self, preds, labels):
        metric_dict = {'xyz': []}
        for xyz, xyz_label in zip(preds, labels):
            xyz = xyz.cpu().detach()
            xyz_label = xyz_label.cpu().detach()
            xyz_loss = nn.functional.l1_loss(xyz, xyz_label).item()
            metric_dict['xyz'].append(xyz_loss)
        metric_dict['xyz'] = sum(metric_dict['xyz']) / len(metric_dict['xyz'])
        return metric_dict

class Vanilla(nn.Module):
    def __init__(self, config):
        super(Vanilla, self).__init__()
        self.encoders = nn.ModuleDict()
        for k, v in config['features'].items():
            if v == 0:
                continue
            else:
                self.encoders[k] = globals()[k](v)

    def forward(self, x):
        feature_list = []
        for k, encoder in self.encoders.items():
            feature_list.append(encoder(x[k]))
        feat = torch.cat(feature_list, dim=1)
        return feat

