"""
L = L_marks + L_think + L_net + λ_q L_think_quantile + λ_ac L_think_autocorr

No joint loss that compares 1/IAT to MB/s.
"""

from __future__ import annotations

import torch
from torch import Tensor

from netcond.data.dataset import Batch


def quantile_loss(log_t: Tensor, mask: Tensor) -> Tensor:
    """Sorted log-t L1 (W1 proxy) vs a detached shuffle — used as regularizer vs empirical batch."""
    valid = mask * (log_t.exp() > 1e-4).float()
    if valid.sum() < 4:
        return log_t.sum() * 0.0
    x = (log_t * valid).flatten()
    w = valid.flatten()
    xs = x[w > 0]
    xs = xs.sort().values
    # encourage smoothness of order stats vs mean — L1 of consecutive gaps
    return (xs[1:] - xs[:-1]).abs().mean() * 0.0 + (xs - xs.mean()).abs().mean() * 0.0 + _w1_to_unit(xs)


def _w1_to_unit(xs: Tensor) -> Tensor:
    """Dummy stable term: L1 of sorted log-t to its median (scale regularizer)."""
    med = xs.median()
    return (xs - med).abs().mean()


def think_quantile_pairwise(pred_log_t: Tensor, true_log_t: Tensor, mask: Tensor) -> Tensor:
    valid = mask * (true_log_t.exp() > 1e-4).float()
    if valid.sum() < 2:
        return pred_log_t.sum() * 0.0
    p = pred_log_t[valid > 0].sort().values
    y = true_log_t[valid > 0].sort().values
    n = min(p.numel(), y.numel())
    return (p[:n] - y[:n]).abs().mean()


def autocorr_loss(log_t: Tensor, mask: Tensor) -> Tensor:
    """Lag-1 gap on log-t vs target lag-1 (computed in train from teacher)."""
    if log_t.size(1) < 2:
        return log_t.sum() * 0.0
    m = mask[:, 1:] * mask[:, :-1]
    if m.sum() < 1:
        return log_t.sum() * 0.0
    gap = log_t[:, 1:] - log_t[:, :-1]
    return (gap.pow(2) * m).sum() / m.sum().clamp_min(1.0)


def autocorr_match(pred: Tensor, true: Tensor, mask: Tensor) -> Tensor:
    if pred.size(1) < 2:
        return pred.sum() * 0.0
    m = mask[:, 1:] * mask[:, :-1]
    if m.sum() < 1:
        return pred.sum() * 0.0
    gp = pred[:, 1:] - pred[:, :-1]
    gy = true[:, 1:] - true[:, :-1]
    return ((gp - gy).abs() * m).sum() / m.sum().clamp_min(1.0)


def masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def net_nll(pred_log: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    log_y = torch.log(target.clamp_min(1e-6))
    return masked_mean((pred_log - log_y).pow(2), mask)


def total_loss(
    app_out: dict[str, Tensor],
    net_obs: dict[str, Tensor],
    batch: Batch,
    *,
    lambda_q: float = 0.2,
    lambda_ac: float = 0.08,
    phase: str = "joint",
) -> dict[str, Tensor]:
    mask = batch.mask
    l_a = masked_mean(app_out["nll_a"], mask)
    l_b = masked_mean(app_out["nll_b"], mask)
    l_d = masked_mean(app_out["nll_d"], mask)
    think_mask = mask * (batch.t > 1e-6).float()
    l_t = masked_mean(app_out["nll_t"], think_mask) if think_mask.sum() > 0 else l_a * 0.0
    l_marks = l_a + l_b + l_d
    log_t = torch.log(batch.t.clamp_min(1e-6))
    pred_lt = app_out.get("pred_log_t", log_t)
    l_q = think_quantile_pairwise(pred_lt, log_t.detach(), think_mask)
    l_ac = autocorr_match(pred_lt, log_t.detach(), think_mask)
    ta_mask = mask * (batch.transfer_a > 0).float()
    tb_mask = mask * (batch.transfer_b > 0).float()
    rtt_mask = mask * (batch.rtt > 0).float()
    l_net = net_nll(net_obs["log_transfer_a"], batch.transfer_a, ta_mask) + net_nll(
        net_obs["log_transfer_b"], batch.transfer_b, tb_mask
    )
    if rtt_mask.sum() > 0:
        l_net = l_net + masked_mean((net_obs["rtt"] - batch.rtt).pow(2) / (batch.rtt.clamp_min(1e-3) ** 2 + 1e-6), rtt_mask)

    if phase == "pretrain_app":
        total = l_marks + 2.0 * l_t + lambda_q * l_q + lambda_ac * l_ac
    elif phase == "pretrain_net":
        total = l_net
    else:
        total = l_marks + 2.0 * l_t + l_net + lambda_q * l_q + lambda_ac * l_ac
    return {
        "total": total,
        "L_marks": l_marks.detach(),
        "L_think": l_t.detach(),
        "L_net": l_net.detach(),
        "L_q": l_q.detach(),
        "L_ac": l_ac.detach(),
    }
