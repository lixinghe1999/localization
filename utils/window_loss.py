import torch
import itertools
def ACCDOA_loss(pred, labels, implicit=True):
    if implicit:
        '''
        pred: (batch, time, nb_class * 3)
        labels: (batch, time, nb_class * 4)
        '''
        num_class = labels.shape[-1] // 4
        pred = pred.reshape(-1, num_class, 3); labels = labels.reshape(-1, num_class, 4)
        loss = torch.nn.functional.mse_loss(pred, labels[..., 1:])
    else:
        '''
        pred: (batch, time, nb_class * 4) [class_active, x, y, z]
        labels: (batch, time, nb_class * 4)
        '''
        num_class = labels.shape[-1] // 4
        pred = pred.reshape(-1, num_class, 4); labels = labels.reshape(-1, num_class, 4)
        pred_sed = pred[..., 0].reshape(-1, num_class); label_sed = labels[..., 0].reshape(-1, num_class)
        # classification loss for sed
        sed_loss = torch.nn.functional.cross_entropy(pred_sed, label_sed)
        # regression loss for doa
        doa_loss = torch.nn.functional.mse_loss(pred[..., 1:], labels[..., 1:])
        loss = sed_loss
    return loss

def Multi_ACCDOA_loss(pred, labels):
    '''
    permutation-aware loss
    pred: (batch, time, source*3(xyz))
    labels: (batch, time, source*4(sed+xyz))
    '''
    num_source = labels.shape[-1] // 4
    pred = pred.reshape(-1, num_source, 3); labels = labels.reshape(-1, num_source, 4)[..., 1:]
    # compute all possible permutations and use the one with the smallest loss
    perms = list(itertools.permutations(range(num_source)))
    min_loss = float('inf')
    for perm in perms:
        perm_loss = torch.nn.functional.mse_loss(pred[:, perm], labels)
        min_loss = min(min_loss, perm_loss)
    return min_loss

