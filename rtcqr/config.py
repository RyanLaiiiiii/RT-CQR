"""Hyperparameters for RT-CQR on the LG 18650HG2 dataset.

Defaults follow the RT-CQR* row of Table I in the paper
("Risk-Aware Time-Adaptive Conformal Quantile Regression for SoC Interval
Estimation in Battery Energy Storage Systems"): a 4-block, 64-channel,
kernel-size-3 TCN with dropout 0.1, Adam (lr=1e-3, batch size 64), and
lambda_nc=1.0, lambda_l=0.1, zeta=0.98, gamma=1.0.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RTCQRConfig:
    # --- Data / windowing ---
    window_size: int = 100
    stride: int = 1
    resample_dt_s: Optional[float] = 1.0  # uniform resampling interval (s) applied before windowing
    rated_capacity_ah: float = 3.0
    soc_min: float = 0.10
    # tau in eq. (14)/(17): T = {0.025, 0.05, 0.10, 0.15, 0.85, 0.90, 0.95, 0.975}
    quantile_levels: List[float] = field(
        default_factory=lambda: [0.025, 0.05, 0.10, 0.15, 0.85, 0.90, 0.95, 0.975]
    )
    # nominal miscoverage alpha for each evaluated PI (90% and 95%)
    pi_alphas: List[float] = field(default_factory=lambda: [0.10, 0.05])
    val_calib_fraction: float = 0.5  # fraction of the validation split reserved for conformal calibration
    train_frac: float = 0.70
    val_frac: float = 0.15
    # test_frac is implicitly 1 - train_frac - val_frac, per file, in time order

    # --- TCN backbone (Table I) ---
    in_channels: int = 3  # voltage, current, temperature
    num_blocks: int = 4
    channels: int = 64
    kernel_size: int = 3
    dropout: float = 0.1

    # --- Optimization ---
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 64
    max_epochs: int = 200
    # 20, not the original 15: with the ReduceLROnPlateau scheduler (factor=0.5,
    # patience=5) in train_model, 15 epochs of no improvement left room for at
    # most one LR drop before early stopping; 20 lets it try ~2-3 drops, giving
    # the optimizer a chance to settle after the val_loss got noisier once SoC
    # started spanning the full [0,1] range (see rtcqr/data.py's capacity fix).
    patience: int = 20

    # --- Composite loss, eq. (17) ---
    lambda_nc: float = 1.0  # quantile-crossing penalty weight
    lambda_l: float = 0.1  # lower-tail regularization weight

    # --- Violation-weighted time-adaptive conformal calibration, eq. (19)-(28) ---
    zeta: float = 0.98  # temporal decay factor
    gamma: float = 1.0  # lower-bound-violation emphasis in time weights
    wl0: float = 1.5  # w_l^(0): base lower-tail nonconformity weight
    wl1: float = 3.0  # w_l^(1): lower-tail weight on violation samples (wl1 >= wl0 >= wu >= 0)
    wu: float = 1.0  # w_u: upper-tail nonconformity weight
    calib_max_history: int = 2000  # cap on samples kept for online recalibration (efficiency only)

    seed: int = 42

    @property
    def tau_l_index(self) -> int:
        """Index of tau_l = min(T), used by the lower-tail regularizer in eq. (17)."""
        return int(min(range(len(self.quantile_levels)), key=lambda i: self.quantile_levels[i]))

    def quantile_bounds(self, alpha: float):
        """Return (index_lower, index_upper) into quantile_levels for tl=alpha/2, tu=1-alpha/2."""
        tl, tu = alpha / 2.0, 1.0 - alpha / 2.0
        idx_l = _closest_index(self.quantile_levels, tl)
        idx_u = _closest_index(self.quantile_levels, tu)
        return idx_l, idx_u


def _closest_index(levels: List[float], target: float, tol: float = 1e-6) -> int:
    for i, lv in enumerate(levels):
        if abs(lv - target) < tol:
            return i
    raise ValueError(f"Quantile level {target} not found in {levels}; add it to quantile_levels.")
