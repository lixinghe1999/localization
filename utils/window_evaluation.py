import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import itertools
import torchmetrics
import torch
import torchmetrics.classification
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
        label_sed = label[..., 0] # [batch*time, n_class]
    else:
        '''
        pred: [batch, time, n_class * 4 (x, y, z)]
        label: [batch, time, n_class * 4 (class_active, x, y, z)]
        '''
        n_class = pred.shape[-1] // 4
        pred = pred.reshape(-1, n_class, 4); label = label.reshape(-1, n_class, 4)
        pred_sed = pred[..., 0]# [batch*time, n_class]
        label_sed = label[..., 0]# [batch*time, n_class]
    pred_sed_binary = pred_sed > 0.5
    if n_class == 1:
        sed_F1 = torchmetrics.F1Score(task='binary') (torch.from_numpy(pred_sed_binary), torch.from_numpy(label_sed))
    else:
        sed_F1 = torchmetrics.F1Score(task='multilabel', num_labels=n_class)(torch.from_numpy(pred_sed), torch.from_numpy(label_sed).long())
    distances = []
    TP, FP, FN, TN = 0, 0, 0, 0
    for i in range(pred.shape[0]): # sample
        for c in range(n_class):  
            if label_sed[i, c] == pred_sed_binary[i, c]:
                if label_sed[i, c]: # positive sample
                    dist = distance_between_cartesian_coordinates(pred[i, c, 0], pred[i, c, 1], pred[i, c, 2], label[i, c, 1], label[i, c, 2], label[i, c, 3])
                    distances.append(dist)
                    if dist < 20:
                        TP += 1
                else:
                    TN += 1
            else:
                if label_sed[i, c]:
                    FN += 1
                else:
                    FP += 1
    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    F1 = 2 * precision * recall / (precision + recall + 1e-10)
    return {
        'precision': precision,
        'recall': recall,
        'F1': F1,
        'distance': np.mean(distances) if len(distances) > 0 else 180,
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
    batch, time, N = label.shape
    num_source = N // 4
    pred = pred.reshape(batch, time, num_source, 3); label = label.reshape(batch, time, num_source, 4)
    # compute all possible permutations and use the one with the smallest loss
    perms = list(itertools.permutations(range(num_source)))
    for p, l in zip(pred, label):
        best_metric_dict = {'distance': 180, 'precision': 0, 'recall': 0, 'F1': 0, 'sed_F1': 0, 'sed_roc_auc': 0}
        for perm in perms:
            pred_perm = p[:, perm, :] # [time, source, 3]
            label_perm = l[:, perm, :] # [time, source, 4]
            pred_perm = pred_perm.reshape(1, time, num_source*3)
            label_perm = label_perm.reshape(1, time, num_source*4)
            metric_dict = ACCDOA_evaluation(pred_perm, label_perm)
            if metric_dict['sed_F1'] > best_metric_dict['sed_F1']:
                best_metric_dict = metric_dict
    return best_metric_dict


def Guassian_evaluation(preds, labels, threshold=0.5, distance=10, vis=False):
    '''
    pred: [batch, time, 360]
    label: [batch, time, 360]
    '''

    batch, time, _ = preds.shape
    preds = preds.reshape(batch*time, -1)
    labels = labels.reshape(batch*time, -1)        

    pred_peaks = [signal.find_peaks(p, height=threshold, distance=distance)[0] for p in preds]
    
    # Initialize counts for TP, FP, FN
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for i in range(len(pred_peaks)):
        # Get the true directions (where labels are 1)
        true_indices = np.where(labels[i] == 1)[0]  # Indices of true directions
        
        # Set to track if we found a correct prediction for each true direction
        found_correct_prediction = np.zeros_like(true_indices, dtype=bool)

        # Check each of the top-k predicted directions
        for pred_index in pred_peaks[i]:
            # Calculate angular error for each true direction
            for j, true_index in enumerate(true_indices):
                # Calculate the angular difference
                error = np.abs(pred_index - true_index)
                # Normalize the error to [0, 180]
                angular_error = min(error, 360 - error)

                # If the angular error is below the threshold, it's a true positive
                if angular_error < threshold:
                    true_positives += 1
                    found_correct_prediction[j] = True  # Mark this true direction as found
                    break  # No need to check further true directions for this prediction

        # Count false negatives: true directions that were not found
        false_negatives += np.sum(~found_correct_prediction)  # Count those not found

    # Count false positives: any top-k prediction that does not match
    false_positives = (len(pred_peaks) - true_positives)

    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'F1': f1_score,
        'distance': 0,
        'sed_F1': 0,
        'sed_roc_auc': 0
    }




