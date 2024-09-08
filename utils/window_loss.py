import torch
def ACCDOA_loss(pred, labels, implicit=False):
    if implicit:
        '''
        pred: (nb_frames, nb_bins, nb_class * 3)
        labels: (nb_frames, nb_bins, nb_class * 4)
        '''
        num_class = labels.shape[-1] // 4
        pred = pred.reshape(-1, num_class, 3); labels = labels.reshape(-1, num_class, 4)
        loss = torch.nn.functional.mse_loss(pred, labels[..., 1:])
    else:
        '''
        pred: (nb_frames, nb_bins, nb_class * 4) [class_active, x, y, z]
        labels: (nb_frames, nb_bins, nb_class * 4)
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
