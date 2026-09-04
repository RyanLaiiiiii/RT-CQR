"""Risk-aligned composite quantile loss, eq. (16)-(18).

    L(x, SoC; theta) = sum_tau rho_tau(SoC - q_tau(x; theta))                (pinball)
                      + lambda_nc * sum_{tau < tau'} [q_tau - q_tau']_+      (crossing penalty)
                      + lambda_l  * [SoC_min - q_{tau_l}(x; theta)]_+        (lower-tail regularizer)

with tau_l = min(T), and rho_tau(a) = a * (tau - 1{a<0}) the pinball loss.
"""
from __future__ import annotations

from typing import Sequence

import torch


def pinball_loss(soc_true: torch.Tensor, q_pred: torch.Tensor, tau: float) -> torch.Tensor:
    diff = soc_true - q_pred
    return torch.maximum(tau * diff, (tau - 1.0) * diff)


def quantile_crossing_penalty(q_pred: torch.Tensor) -> torch.Tensor:
    """sum_{j<k} relu(q_j - q_k), assuming quantile_levels are sorted ascending
    so that q_pred[:, j] should be <= q_pred[:, k] for j < k."""
    num_q = q_pred.shape[1]
    penalty = q_pred.new_zeros(q_pred.shape[0])
    for j in range(num_q):
        for k in range(j + 1, num_q):
            penalty = penalty + torch.relu(q_pred[:, j] - q_pred[:, k])
    return penalty


def composite_quantile_loss(
    soc_true: torch.Tensor,
    q_pred: torch.Tensor,
    quantile_levels: Sequence[float],
    soc_min: float,
    lambda_nc: float,
    lambda_l: float,
    tau_l_index: int,
) -> torch.Tensor:
    """eq. (17), averaged over the batch. `quantile_levels` must be sorted ascending,
    matching the column order of `q_pred`."""
    pinball_sum = sum(
        pinball_loss(soc_true, q_pred[:, j], tau) for j, tau in enumerate(quantile_levels)
    )
    crossing = quantile_crossing_penalty(q_pred)
    lower_tail = torch.relu(soc_min - q_pred[:, tau_l_index])

    loss_per_sample = pinball_sum + lambda_nc * crossing + lambda_l * lower_tail
    return loss_per_sample.mean()
