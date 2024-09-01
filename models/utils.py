import numpy as np
import torch

def beamforming_loss(pred, label, eps=1e-8):
    label = label.to(pred.device).unsqueeze(1)
    _sisnr = sisnr(pred, label, eps=eps)
    loss = -torch.mean(_sisnr)
    return loss
def region_loss(self, pred, label, eps=1e-8):
    _sisnr = sisnr(pred, label['spatial_audio'].to(pred.device), eps=eps)
    _sisnr *= label['region_active'].to(pred.device)
    active_loss = -torch.mean(_sisnr)
    inactive_loss = ((1-label['region_active']).to(pred.device)* (torch.norm(pred, dim=-1))).mean() 
    loss = active_loss + inactive_loss
    return loss
def permutation_invaiant_loss(pred, label, eps=1e-8):
    print('pred', pred.shape, 'label', label.shape)
    from itertools import permutations
    B, num_region, T = pred.shape
    permute = permutations(range(num_region))
    _loss = []
    for perm in permute:
        _sisnr = sisnr(pred[:, perm], label.to(pred.device), eps=eps)
        _loss.append(_sisnr)       
    _loss = torch.stack(_loss, dim=1) # [B, num_perm, num_region]
    _loss = torch.max(_loss, dim=1)[0]
    loss = -torch.mean(_loss)
    return loss


def sisnr(x, s, eps=1e-8):
    """
    calculate training loss
    input:
          x: separated signal, N x S tensor
          s: reference signal, N x S tensor
    Return:
          sisnr: N tensor
    """

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)
    x = x.squeeze(); s = s.squeeze()
    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = torch.sum(
        x_zm * s_zm, dim=-1,
        keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))

def minimal_distance(list1, list2, full_scale=360):
    if len(list1) == 0 or len(list2) == 0:
        return full_scale
    # Convert lists to numpy arrays
    array1 = np.array(list1)
    array2 = np.array(list2)
    
    # Determine the sizes of the lists
    len1 = len(array1)
    len2 = len(array2)
    
    error_matrix = np.empty((len1, len2))
    for i in range(len1):
        for j in range(len2):
            error_matrix[i, j] = good_error(array1[i], array2[j], full_scale)
    # Find the minimum error
    min_error = np.mean(np.min(error_matrix, axis=1))
    return min_error
def good_error(a, b, full_scale=360):
    abs_error = np.abs(a - b)
    error = np.minimum(abs_error, full_scale - abs_error)
    error = np.mean(error)
    return error
