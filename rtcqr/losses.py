"""Risk-aligned composite quantile loss, eq. (16)-(18).

    L(x, SoC; theta) = sum_tau rho_tau(SoC - q_tau(x; theta))                (pinball)
                      + lambda_nc * sum_{tau < tau'} [q_tau - q_tau']_+      (crossing penalty)
                      + lambda_l  * [SoC_min - q_{tau_l}(x; theta)]_+        (lower-tail regularizer)

with tau_l = min(T), and rho_tau(a) = a * (tau - 1{a<0}) the pinball loss.

Two notes on how these terms behave against the monotone head in
`model.TCNQuantileNet`, which did not hold against the original
unconstrained linear head:

  * The crossing penalty is identically zero (see `composite_quantile_loss`).

  * The lower-tail term reaches the network only through the head's `base`
    unit, because that head emits q[0] = base and q[k] = base + cumsum of
    non-negative increments. `base` is added to *every* quantile, so a
    gradient step on this term translates the whole quantile fan rather than
    moving q_{tau_l} alone as it did before. The loss is still exactly
    eq. (17); what changed is the optimisation geometry, and the pinball
    terms on the upper quantiles are what hold the fan down against it. This
    matters more since SoC started genuinely reaching 0 (see rtcqr/data.py's
    per-condition capacity normalization): q_{tau_l} now falls below
    `soc_min` on a large fraction of samples, so the term is active far more
    often than it used to be. Ablate it with train.py --no-ltr.

Also note eq. (17) and the LVR metric of eq. (29) pull in opposite
directions by construction: LVR counts samples where the true SoC is below
SoC_min *while the predicted lower bound is not*, so pushing q_{tau_l} up --
which is exactly what this regularizer does, per eq. (18) -- mechanically
increases it. That tension is the paper's, not this implementation's; it is
reproduced faithfully here rather than silently "corrected".
"""
from __future__ import annotations

from typing import Sequence

import torch


def pinball_loss(soc_true: torch.Tensor, q_pred: torch.Tensor, tau: float) -> torch.Tensor:
    diff = soc_true - q_pred
    return torch.maximum(tau * diff, (tau - 1.0) * diff)


def quantile_crossing_penalty(q_pred: torch.Tensor) -> torch.Tensor:
    """sum_{j<k} relu(q_j - q_k), assuming quantile_levels are sorted ascending
    so that q_pred[:, j] should be <= q_pred[:, k] for j < k.

    Kept as a diagnostic (and for anyone swapping in an unconstrained head),
    but note it is identically zero against `model.TCNQuantileNet`, whose head
    is monotone by construction -- see `composite_quantile_loss`.

    Vectorised rather than looped over the O(|T|^2) pairs, which built that
    many autograd nodes per batch for a term that cannot be positive.
    """
    diffs = q_pred[:, :, None] - q_pred[:, None, :]      # diffs[:, j, k] = q_j - q_k
    return torch.relu(torch.triu(diffs, diagonal=1)).sum(dim=(1, 2))


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
    loss_per_sample = pinball_sum
    # `model.TCNQuantileNet` emits q[k] = q[k-1] + softplus(.), so crossing is
    # structurally impossible and this term is *exactly* zero -- zero value and
    # zero gradient -- for every input. Config therefore ships lambda_nc=0.0 and
    # this branch is skipped; set lambda_nc>0 only alongside an unconstrained
    # head, where the penalty does real work.
    if lambda_nc:
        loss_per_sample = loss_per_sample + lambda_nc * quantile_crossing_penalty(q_pred)
    if lambda_l:
        loss_per_sample = loss_per_sample + lambda_l * torch.relu(soc_min - q_pred[:, tau_l_index])
    return loss_per_sample.mean()
