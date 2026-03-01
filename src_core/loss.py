import numpy as np
import torch
import torch.nn as nn
from geomloss import SamplesLoss


class WassersteinLoss(nn.Module):
    """Wasserstein-2 distance between predicted heatmap and GT mask.

    Treats both as 2D probability distributions and computes the
    optimal transport cost via geomloss (Sinkhorn divergence).

    Input:
        pred: [B, 2, H, W] softmax output (channel 1 = defect probability)
        target: [B, 1, H, W] binary mask
    Output:
        scalar loss (mean W2 over batch)
    """

    def __init__(self, blur=1.0, scaling=0.9):
        super().__init__()
        self.loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=blur, scaling=scaling)
        self._grid_cache = {}

    def _get_grid(self, H, W, device):
        key = (H, W, device)
        if key not in self._grid_cache:
            y = torch.arange(H, dtype=torch.float32, device=device)
            x = torch.arange(W, dtype=torch.float32, device=device)
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)  # [N, 2]
            self._grid_cache[key] = grid
        return self._grid_cache[key]

    def forward(self, pred, target):
        B, C, H, W = pred.shape
        grid = self._get_grid(H, W, pred.device)  # [N, 2]

        # Defect probability channel
        pred_prob = pred[:, 1, :, :]  # [B, H, W]
        target_flat = target.squeeze(1)  # [B, H, W]

        diag = (H**2 + W**2) ** 0.5
        total_loss = 0.0
        for b in range(B):
            p = pred_prob[b].reshape(-1)
            q = target_flat[b].reshape(-1)

            p_sum = p.sum()
            q_sum = q.sum()

            # Skip clean images entirely (no GT defect → no Wasserstein signal)
            if q_sum < 1e-8:
                continue

            # Collapse penalty: GT has defect but pred is empty
            if p_sum < 1e-8:
                total_loss += 1.0
                continue

            # Normalize to probability distributions (sum=1)
            p = p / p_sum
            q = q / q_sum

            # geomloss: [1, N] weights, [1, N, 2] positions
            dist = self.loss_fn(
                p.unsqueeze(0), grid.unsqueeze(0),
                q.unsqueeze(0), grid.unsqueeze(0),
            )
            # dist ≈ (1/2) * W2^2, convert to W2
            w2 = (2 * dist).sqrt()
            # Normalize by grid diagonal so value ∈ [0, 1]
            total_loss += w2 / diag

        return total_loss / max(B, 1)


class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def update_params(self, alpha=None, gamma=None):
        """Update focal loss parameters dynamically"""
        if alpha is not None:
            self.alpha = alpha
        if gamma is not None:
            self.gamma = gamma

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        return loss
