import torch
import itertools

def Region_loss(pred, labels):
    '''
    pred: (batch, time, nb_class (azimuth + distance))
    labels: (batch, time, nb_class (azimuth + distance))
    '''
    loss = torch.nn.functional.binary_cross_entropy(pred, labels)
    return loss

def ACCDOA_loss(pred, labels, implicit=True, training=True):
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
        pred_sed = pred[..., 0]; label_sed = labels[..., 0]
        # classification loss for sed
        sed_loss = torch.nn.BCEWithLogitsLoss()(pred_sed, label_sed.float())
        doa_loss = torch.nn.functional.mse_loss(pred[..., 1:], labels[..., 1:])
        loss = sed_loss + doa_loss
    return loss

# def Multi_ACCDOA_loss(pred, label):
#     '''
#     permutation-aware loss
#     pred: (batch, time, source*3(xyz))
#     label: (batch, time, source*4(sed+xyz))
#     '''
#     batch, time, N = label.shape
#     num_source = N // 4
#     pred = pred.reshape(batch, time, num_source, 3); label = label.reshape(batch, time, num_source, 4)
#     # compute all possible permutations and use the one with the smallest loss
#     perms = list(itertools.permutations(range(num_source)))
#     batch_loss = 0
#     for p, l in zip(pred, label):
#         min_loss = float('inf')
#         for perm in perms:
#             pred_perm = p[:, perm, :] # [time, source, 3]
#             label_perm = l[:, perm, :] # [time, source, 4]
#             pred_perm = pred_perm.reshape(1, time, num_source*3)
#             label_perm = label_perm.reshape(1, time, num_source*4)
#             loss = ACCDOA_loss(pred_perm, label_perm)
#             min_loss = min(min_loss, loss)
#         batch_loss += min_loss
#     return batch_loss/batch


class MSELoss:
    def __init__(self, reduction='mean'):
        self.reduction = reduction
        self.name = 'loss_MSE'
        if self.reduction != 'PIT':
            self.loss = torch.nn.MSELoss(reduction='mean')
        else:
            self.loss = torch.nn.MSELoss(reduction='none')
    def calculate_loss(self, pred, target):
        if self.reduction != 'PIT':
            return self.loss(pred, target)
        else:
            return self.loss(pred, target).mean(dim=tuple(range(2, pred.ndim)))
        
def Multi_ACCDOA_loss(pred, target, training=True):
    '''
    pred: [batch, T, num_tracks=2, doas=3]
    target: [batch, T, num_tracks=2, SED+DOA=4]

    return: updated target with the minimum loss frame-wisely
    '''
    batch, T, N = target.shape
    pred = pred.reshape(batch, T, 2, 3); target = target.reshape(batch, T, 2, 4)
    target_flipped = target.flip(dims=[2])

    target_sed = target[..., :1]; target_flipped_sed = target_flipped[..., :1]
    target_doa = target[..., 1:]; target_flipped_doa = target_flipped[..., 1:]

    loss1 = MSELoss(reduction='PIT').calculate_loss(pred, target_doa)
    loss2 = MSELoss(reduction='PIT').calculate_loss(pred, target_flipped_doa)
    loss = (loss1 * (loss1 <= loss2) + loss2 * (loss1 > loss2)).mean()

    if training:
        return loss
    else:
        updated_sed = target_sed.clone() * (loss1[:, :, None, None] <= loss2[:, :, None, None]) + target_flipped_sed.clone() * (loss1[:, :, None, None] > loss2[:, :, None, None])
        updated_doa = target_doa.clone() * (loss1[:, :, None, None] <= loss2[:, :, None, None]) + target_flipped_doa.clone() * (loss1[:, :, None, None] > loss2[:, :, None, None])                                    
        updated_target = torch.cat([updated_sed, updated_doa], dim=-1).reshape(batch, T, N)
        return updated_target

