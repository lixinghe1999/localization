import torch
from torch.nn.modules.loss import _Loss


class SingleSrcNegSDR(_Loss):
    r"""Base class for single-source negative SI-SDR, SD-SDR and SNR.

    Args:
        sdr_type (str): choose between ``snr`` for plain SNR, ``sisdr`` for
            SI-SDR and ``sdsdr`` for SD-SDR [1].
        zero_mean (bool, optional): by default it zero mean the target and
            estimate before computing the loss.
        take_log (bool, optional): by default the log10 of sdr is returned.
        reduction (string, optional): Specifies the reduction to apply to
            the output:
            ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output.

    Shape:
        - est_targets : :math:`(batch, time)`.
        - targets: :math:`(batch, time)`.

    Returns:
        :class:`torch.Tensor`: with shape :math:`(batch)` if ``reduction='none'`` else
        [] scalar if ``reduction='mean'``.

    Examples
        >>> import torch
        >>> from asteroid.losses import PITLossWrapper
        >>> targets = torch.randn(10, 2, 32000)
        >>> est_targets = torch.randn(10, 2, 32000)
        >>> loss_func = PITLossWrapper(SingleSrcNegSDR("sisdr"),
        >>>                            pit_from='pw_pt')
        >>> loss = loss_func(est_targets, targets)

    References
        [1] Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE
        International Conference on Acoustics, Speech and Signal
        Processing (ICASSP) 2019.
    """

    def __init__(self, sdr_type, zero_mean=True, take_log=True, reduction="none", active_aware=True, EPS=1e-8):
        assert reduction != "sum", NotImplementedError
        super().__init__(reduction=reduction)

        assert sdr_type in ["snr", "sisdr", "sdsdr"]
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log
        self.EPS = 1e-8
        self.active_aware = active_aware

    def forward(self, est_target, target):
        if target.size() != est_target.size() or target.ndim != 2:
            if target.shape[1] == 1:
                target = target.squeeze(1)
                est_target = est_target.squeeze(1)
            else:
                raise TypeError(
                    f"Inputs must be of shape [batch, time], got {target.size()} and {est_target.size()} instead"
                )
        # Step 1. Zero-mean norm
        if self.zero_mean:
            mean_source = torch.mean(target, dim=1, keepdim=True)
            mean_estimate = torch.mean(est_target, dim=1, keepdim=True)
            target = target - mean_source
            est_target = est_target - mean_estimate
        # Step 2. Pair-wise SI-SDR.
        if self.sdr_type in ["sisdr", "sdsdr"]:
            # [batch, 1]
            dot = torch.sum(est_target * target, dim=1, keepdim=True)
            # [batch, 1]
            s_target_energy = torch.sum(target**2, dim=1, keepdim=True) + self.EPS
            # [batch, time]
            scaled_target = dot * target / s_target_energy
        else:
            # [batch, time]
            scaled_target = target
        if self.sdr_type in ["sdsdr", "snr"]:
            e_noise = est_target - target
        else:
            e_noise = est_target - scaled_target
        # [batch]
        losses = torch.sum(scaled_target**2, dim=1) / (torch.sum(e_noise**2, dim=1) + self.EPS)
        if self.take_log:
            losses = 10 * torch.log10(losses + self.EPS)

        if self.active_aware:
            active = torch.sum(target**2, dim=1) > 0
            active_loss = losses[active].to(dtype=torch.float32)
            inactive_est = est_target[~active].to(dtype=torch.float32)
            # l2_loss for the inactive sources
            if len(inactive_est) == 0:
                inactive_loss = torch.tensor(0.0)
            else:
                inactive_loss = torch.mean(inactive_est**2, dim=1)
            # print(losses[active].mean().dtype, l2_loss.dtype, losses[active].shape, l2_loss.shape)
            if self.training:
                losses = active_loss.mean() + inactive_loss.mean() if self.reduction == "mean" else torch.cat([active_loss, inactive_loss])
            else: 
                # only active sources are considered for validation
                losses = active_loss.mean() if self.reduction == "mean" else active_loss
        else:
            losses = losses.mean() if self.reduction == "mean" else losses
        return -losses

singlesrc_neg_sisdr = SingleSrcNegSDR("sisdr", reduction="mean")
singlesrc_neg_sdsdr = SingleSrcNegSDR("sdsdr", reduction="mean")
singlesrc_neg_snr = SingleSrcNegSDR("snr", reduction="mean")
