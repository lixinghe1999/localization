import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import itertools
import torchmetrics
import torch
import torchmetrics.classification
from .window_loss import Multi_ACCDOA_loss

def distance_between_cartesian_coordinates(x1, y1, z1, x2, y2, z2):
    """
    Angular distance between two cartesian coordinates
    MORE: https://en.wikipedia.org/wiki/Great-circle_distance
    Check 'From chord length' section

    :return: angular distance in degrees
    """
    # Normalize the Cartesian vectors
    N1 = np.sqrt(x1**2 + y1**2 + z1**2 + 1e-10)
    N2 = np.sqrt(x2**2 + y2**2 + z2**2 + 1e-10)
    x1, y1, z1, x2, y2, z2 = x1/N1, y1/N1, z1/N1, x2/N2, y2/N2, z2/N2

    #Compute the distance
    dist = x1*x2 + y1*y2 + z1*z2
    dist = np.clip(dist, -1, 1)
    dist = np.arccos(dist) * 180 / np.pi
    return dist

def distance_between_cartesian_coordinates_batch(xyz1, xyz2):
    '''
    xyz1: (batch, 3)
    xyz2: (batch, 3)
    '''
    N1 = np.sqrt(np.sum(xyz1**2, axis=-1, keepdims=True) + 1e-10)
    N2 = np.sqrt(np.sum(xyz2**2, axis=-1, keepdims=True) + 1e-10)
    xyz1 = xyz1 / N1; xyz2 = xyz2 / N2
    dist = np.sum(xyz1 * xyz2, axis=-1)
    dist = np.clip(dist, -1, 1)
    dist = np.arccos(dist) * 180 / np.pi
    return dist

def ACCDOA_evaluation(pred, label, implicit=True):
    if implicit:
        '''
        pred: [batch, time, n_class * 3 (x, y, z)]
        label: [batch, time, n_class * 4 (class_active, x, y, z)]
        '''
        N1 = pred.shape[-1]; N2 = label.shape[-1]
        n_class = N1 // 3
        assert N2 == n_class * 4
        pred = pred.reshape(-1, n_class, 3)
        pred_sed = np.sqrt(np.sum(pred**2, axis=-1)) # [batch*time, n_class]
        label = label.reshape(-1, n_class, 4)
        label_sed = label[..., 0] > 0.5 # [batch*time, n_class]
    else:
        '''
        pred: [batch, time, n_class * 4 (x, y, z)]
        label: [batch, time, n_class * 4 (class_active, x, y, z)]
        '''
        n_class = pred.shape[-1] // 4
        pred = pred.reshape(-1, n_class, 4); label = label.reshape(-1, n_class, 4)
        pred_sed = pred[..., 0] # [batch*time, n_class]
        label_sed = label[..., 0]  > 0.5 # [batch*time, n_class]
    pred_sed_binary = pred_sed > 0.5
    if n_class == 1:
        sed_F1 = torchmetrics.F1Score(task='binary') (torch.from_numpy(pred_sed_binary), torch.from_numpy(label_sed))
    else:
        sed_F1 = torchmetrics.F1Score(task='multilabel', num_labels=n_class)(torch.from_numpy(pred_sed_binary), torch.from_numpy(label_sed).long())

    distances = distance_between_cartesian_coordinates_batch(pred.reshape(-1, 3), label[..., 1:].reshape(-1, 3)).reshape(-1, n_class)
    mean_distance = np.mean(distances * pred_sed_binary)
    distances = distances < 20
    
    pred_sed_binary[~ (pred_sed_binary & label_sed & distances)] = 0
    if n_class == 1:
        precision = torchmetrics.Precision(task='binary')(torch.from_numpy(pred_sed_binary), torch.from_numpy(label_sed))
        recall = torchmetrics.Recall(task='binary')(torch.from_numpy(pred_sed_binary), torch.from_numpy(label_sed))
        F1_score = torchmetrics.F1Score(task='binary')(torch.from_numpy(pred_sed_binary), torch.from_numpy(label_sed))
    else:
        precision = torchmetrics.Precision(task='multilabel', num_labels=n_class)(torch.from_numpy(pred_sed_binary), torch.from_numpy(label_sed))
        recall = torchmetrics.Recall(task='multilabel', num_labels=n_class)(torch.from_numpy(pred_sed_binary), torch.from_numpy(label_sed))
        F1_score = torchmetrics.F1Score(task='multilabel', num_labels=n_class)(torch.from_numpy(pred_sed_binary), torch.from_numpy(label_sed))

    return {
        'precision': precision,
        'recall': recall,
        'F1': F1_score,
        'distance': mean_distance,
        # 'sed_precision': sed_precision,
        # 'sed_recall': sed_recall,
        'sed_F1': sed_F1,
    }

def Multi_ACCDOA_evaluation(pred, label):
    '''
    permutation-aware loss
    pred: (batch, time, source*3(xyz))
    labels: (batch, time, source*4(sed+xyz))
    '''
    label = Multi_ACCDOA_loss(torch.from_numpy(pred).to('cuda'), torch.from_numpy(label).to('cuda'), training=False)
    metric = ACCDOA_evaluation(pred, label.cpu().numpy())
    return metric





