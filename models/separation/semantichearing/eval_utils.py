import numpy as np
def compute_ild(s_left, s_right):
    sum_sq_left = np.sum(s_left ** 2, axis=-1)
    sum_sq_right = np.sum(s_right ** 2, axis=-1)
    # print(sum_sq_left)
    # print(sum_sq_right)
    return 10 * np.log10(sum_sq_left / sum_sq_right)
def compute_itd(s_left, s_right, sr, t_max = None):
    corr = signal.correlate(s_left, s_right)
    lags = signal.correlation_lags(len(s_left), len(s_right))
    corr /= np.max(corr)

    mid = len(corr)//2 + 1
        
    # print(corr[-t_max:])
    cc = np.concatenate((corr[-mid:], corr[:mid]))

    if t_max is not None:
    # if False:
        # print(cc[-t_max:].shape)
        cc = np.concatenate([cc[-t_max+1:], cc[:t_max+1]])
    else:
        t_max = mid

    # print("OKKK", cc.shape)
    # t = np.arange(-t_max/sr, (t_max)/sr, 1/sr) * 1e6
    # plt.plot(t, np.abs(cc))
    # plt.show()
    tau = np.argmax(np.abs(cc))
    tau -= t_max
    # tau = lags[x]
    # print(tau/ sr * 1e6)

    return tau / sr * 1e6
def itd_diff(s_est, s_gt, sr):
    """
    Computes the ITD error between model estimate and ground truth
    input: (*, 2, T), (*, 2, T)
    """
    TMAX = int(round(1e-3 * sr))
    itd_est = compute_itd(s_est[..., 0, :], s_est[..., 1, :], sr, TMAX)
    itd_gt = compute_itd(s_gt[..., 0, :], s_gt[..., 1, :], sr, TMAX)
    return np.abs(itd_est - itd_gt)
def ild_diff(s_est, s_gt):
    """
    Computes the ILD error between model estimate and ground truth
    input: (*, 2, T), (*, 2, T)
    """
    ild_est = compute_ild(s_est[..., 0, :], s_est[..., 1, :])
    ild_gt = compute_ild(s_gt[..., 0, :], s_gt[..., 1, :])
    return np.abs(ild_est - ild_gt)