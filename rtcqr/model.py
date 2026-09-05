"""Temporal convolutional network backbone with a multi-quantile output head.

Matches the RT-CQR* backbone in Table I: 4 residual blocks, 64 channels per
block, kernel size 3, dropout 0.1, causal dilated convolutions, a Chomp1d
layer to keep convolutions causal, and a linear quantile head applied to
the representation at the final time step.

Table I fixes the block count but not the dilation schedule, and the two
interact through the causal receptive field
`1 + 2*(kernel_size-1)*sum(dilations)`. With the textbook doubling
schedule {1,2,4,8} four blocks reach only 61 steps, far short of the
k = 400 window used by the protocol RT-CQR cites for its data split
(Hannan et al., Sci. Rep. 11:19541, 2021). `dilation_base` therefore
defaults to 4, giving {1,4,16,64} and a 341-step field, which keeps
Table I's four blocks while getting within reach of that window.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn

try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:  # pragma: no cover - older torch versions
    from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    """Removes the trailing padding introduced by a causal convolution."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.drop1,
            self.conv2, self.chomp2, self.relu2, self.drop2,
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNQuantileNet(nn.Module):
    """TCN backbone -> multi-quantile head, per eq. (14)-(15).

    Input:  (batch, in_channels, seq_len)
    Output: (batch, len(quantile_levels)), the estimated conditional
            quantiles of SoC at the *last* time step of the input window.
    """

    def __init__(
        self,
        in_channels: int,
        quantile_levels: Sequence[float],
        num_blocks: int = 4,
        channels: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.1,
        dilation_base: int = 4,
    ):
        super().__init__()
        self.quantile_levels = list(quantile_levels)
        self.dilation_base = dilation_base

        layers = []
        c_in = in_channels
        for b in range(num_blocks):
            dilation = dilation_base ** b
            layers.append(TemporalBlock(c_in, channels, kernel_size, dilation, dropout))
            c_in = channels
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(channels, len(self.quantile_levels))
        self.receptive_field = 1 + 2 * (kernel_size - 1) * sum(
            dilation_base ** b for b in range(num_blocks)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tcn(x)          # (batch, channels, seq_len)
        h_last = h[:, :, -1]     # representation at the current (last) time step
        return self.head(h_last)  # (batch, num_quantiles)

    def quantile_index(self, tau: float, tol: float = 1e-6) -> int:
        """Column of `forward`'s output holding the tau-quantile."""
        for i, level in enumerate(self.quantile_levels):
            if abs(level - tau) < tol:
                return i
        raise ValueError(
            f"Quantile level {tau} is not in this model's trained set {self.quantile_levels}. "
            "The head emits a fixed set of levels, so an interval can only be formed at a "
            "confidence level whose two boundary quantiles were trained."
        )

    def predict_interval(self, x: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """The model's *uncalibrated* prediction interval at nominal
        coverage 1 - alpha: [q_{alpha/2}, q_{1-alpha/2}].

        This is the confidence level entering as an inference-time input.
        The network itself is not conditioned on alpha -- it emits the whole
        trained quantile set in one pass (eq. 14-15) and alpha selects which
        pair bounds the interval -- so the same forward pass serves every
        alpha whose boundary quantiles are in `quantile_levels`.

        Note this is the interval *before* conformal calibration; RT-CQR's
        VW-TAC step then widens it to [q_l - c_alpha, q_u + c_alpha].
        """
        q = self(x)
        lower = q[:, self.quantile_index(alpha / 2.0)]
        upper = q[:, self.quantile_index(1.0 - alpha / 2.0)]
        return lower, upper
