"""Violation-weighted time-adaptive conformal calibration (VW-TAC), eq. (19)-(28).

Nonconformity score, eq. (20)-(21):
    u_i = 1{SoC_i < SoC_min}
    w_l(u_i) = w_l^(0) + u_i * (w_l^(1) - w_l^(0)),   w_l^(1) >= w_l^(0) >= w_u >= 0
    h_i = max( w_l(u_i) * [q_tl,i - SoC_i]_+ , w_u * [SoC_i - q_tu,i]_+ )

Time-decayed, violation-weighted empirical measure, eq. (22)-(24):
    gamma_{i,t} = zeta^{t-i} * (1 + gamma * u_i),   i <= t
    normalized:  gamma-hat_{i,t} = gamma_{i,t} / sum_j gamma_{j,t}

Calibrated radius as a weighted empirical quantile, eq. (25)-(26):
    c_alpha,t = inf{ c : sum_{i: h_i<=c} gamma-hat_{i,t} >= 1 - alpha }

Calibrated interval via Minkowski addition, eq. (27)-(28):
    PI^cal_t = [ q_tl,t - c_alpha,t , q_tu,t + c_alpha,t ]

WHAT IS AND IS NOT GUARANTEED HERE

The calibration and test sets in this setting are *not* exchangeable, and
the paper says so explicitly (Sec. III.B.2): "Unlike classical conformal
prediction, whose finite-sample coverage guarantee relies on
exchangeability, the proposed calibration uses a time-decayed and
violation-weighted empirical measure to target weighted empirical coverage
rather than distribution-free finite-sample coverage."

So RT-CQR and WCP deliberately trade the distribution-free guarantee for
adaptivity to drift. Nothing in this module should be read as providing
coverage in the split-conformal sense, and a calib/test mismatch does not
"void a guarantee" -- there is none to void. What a mismatch does do is
make the radius fitted on calib the wrong size for test, which shows up
directly as ACE. Exchangeability fails here on several independent axes:

  * stride-1 windows over 1 Hz data share all but one sample with their
    neighbours, so the buffer is nowhere near i.i.d. (see
    `effective_sample_size`);
  * calibration and test come from different drive cycles under the
    paper's fixed protocol (LA92 vs. US06/HWFET), i.e. different load
    distributions by construction;
  * the error distribution is strongly temperature-dependent; and
  * calibration and test were recorded at different times, so any drift
    or aging between them lands squarely in the calibration residuals.

`weighted_quantile`'s finite-sample correction is a split-conformal device
that assumes exchangeability, so here it is a small conservative
adjustment rather than a guarantee. CQR (zeta=1, uniform weights) is the
variant whose textbook guarantee would need exchangeability outright, and
it does not hold on this data either.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def lower_tail_weight(violation: np.ndarray, wl0: float, wl1: float) -> np.ndarray:
    """eq. (21). `violation` is a 0/1 array (or bool array)."""
    return wl0 + violation.astype(np.float64) * (wl1 - wl0)


def nonconformity_scores(
    soc_true: np.ndarray,
    q_lower: np.ndarray,
    q_upper: np.ndarray,
    soc_min: float,
    wl0: float,
    wl1: float,
    wu: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """eq. (20). Returns (scores, violation_indicator)."""
    violation = (soc_true < soc_min).astype(np.float64)
    wl = lower_tail_weight(violation, wl0, wl1)
    lower_excess = np.clip(q_lower - soc_true, 0.0, None)
    upper_excess = np.clip(soc_true - q_upper, 0.0, None)
    scores = np.maximum(wl * lower_excess, wu * upper_excess)
    return scores, violation


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective sample size, (sum w)^2 / sum w^2.

    For the exponential decay of eq. (22) this converges to (1+zeta)/(1-zeta)
    regardless of buffer length -- 99 samples at zeta=0.98. See
    `time_decay_weights`.
    """
    w = np.asarray(weights, dtype=np.float64)
    denom = float(np.sum(w ** 2))
    return float(np.sum(w) ** 2 / denom) if denom > 0 else 0.0


def weighted_quantile(
    scores: np.ndarray, weights: np.ndarray, level: float, finite_sample_correction: bool = True
) -> float:
    """eq. (25)-(26): smallest score c such that the weighted CDF at c is >= level.

    `weights` must be non-negative and sum to (approximately) 1.

    Split conformal prediction needs the ceil((1-alpha)(n+1))-th order
    statistic, not the ceil((1-alpha)n)-th, for its coverage guarantee to
    hold; taking the plain empirical quantile undercovers by roughly 1/(n+1).
    Eq. (25)-(26) is written without that correction, so `finite_sample_correction`
    exists to reproduce the paper exactly (False) or to be statistically
    correct (True, the default). The weighted generalisation inflates the
    requested level by one effective sample's worth of mass -- and the
    *effective* count is what matters here: at zeta=0.98 it is ~99 no matter
    how many calibration windows exist, so the deficit stays around 1% even
    with a six-figure buffer, and feeds straight into the reported ACE.
    """
    scores = np.asarray(scores, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if scores.size == 0:
        return 0.0
    if finite_sample_correction:
        n_eff = effective_sample_size(weights)
        if n_eff > 0:
            level = min(1.0, level * (n_eff + 1.0) / n_eff)
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    cum = np.cumsum(weights[order])
    if cum[-1] <= 0:
        return float(sorted_scores[-1])
    # Renormalise so the CDF ends at exactly 1.0, and compare with a small
    # tolerance. Summing n weights of 1/n drifts by ~1e-13 by the tail, which
    # is enough to push the selected order statistic off by one all on its
    # own (visible from n~2000 up) -- the choice of interval must not depend
    # on cumsum rounding.
    cum /= cum[-1]
    idx = int(np.searchsorted(cum, min(level, 1.0) - 1e-12, side="left"))
    idx = min(idx, len(sorted_scores) - 1)
    return float(sorted_scores[idx])


def time_decay_weights(n: int, zeta: float, gamma: float, violation: np.ndarray) -> np.ndarray:
    """eq. (22)-(23) for a calibration buffer of size n, evaluated at t = n
    (i.e. weights relative to "now", the most recent buffer entry).

    lags[i] = n - 1 - i for i = 0..n-1 (0 = most recent sample), so
    gamma_i = zeta**lags[i] * (1 + gamma * violation[i]).

    NOTE: this treats buffer *position* as time, so the caller must hand over
    a chronologically ordered buffer. Segments come out of the loader grouped
    by measurement and are randomly permuted by `data.segment_split`, so
    train.py sorts the calibration segments with `data.order_chronologically`
    before windowing them. Without that the decay would designate an
    arbitrary segment's tail as "now" and the whole time-adaptive component
    would be weighting noise.

    Note also that the resulting effective sample size converges to
    (1+zeta)/(1-zeta) independently of n -- 99 at zeta=0.98 -- so raising n
    past a few hundred does not make the radius more stable. Use
    `effective_sample_size` to see the number that actually governs it.
    """
    lags = np.arange(n - 1, -1, -1, dtype=np.float64)
    raw = (zeta ** lags) * (1.0 + gamma * violation.astype(np.float64))
    total = raw.sum()
    if total <= 0:
        return np.full(n, 1.0 / n)
    return raw / total


@dataclass
class CalibrationResult:
    radius: float
    n_used: int


class StaticVWTACCalibrator:
    """One-shot calibration on a held-out calibration set (Sec. IV.B: "CQR, WCP,
    and RT-CQR are calibrated on a common subset held out from the validation
    set"). Weights are computed relative to the end of the calibration set
    (eq. 22 with t = N_cal) and the resulting radius c_alpha is reused for
    every test-time prediction -- the practical, static-deployment special
    case of the general time-adaptive calibrator below.
    """

    def __init__(self, soc_min: float, zeta: float, gamma: float, wl0: float, wl1: float, wu: float,
                 finite_sample_correction: bool = True, name: str = "calibrator"):
        self.soc_min = soc_min
        self.zeta = zeta
        self.gamma = gamma
        self.wl0 = wl0
        self.wl1 = wl1
        self.wu = wu
        self.finite_sample_correction = finite_sample_correction
        self.name = name
        self._scores: Optional[np.ndarray] = None
        self._weights: Optional[np.ndarray] = None
        self.n_effective_: Optional[float] = None

    def fit(self, soc_calib: np.ndarray, q_lower_calib: np.ndarray, q_upper_calib: np.ndarray) -> "StaticVWTACCalibrator":
        scores, violation = nonconformity_scores(
            soc_calib, q_lower_calib, q_upper_calib, self.soc_min, self.wl0, self.wl1, self.wu
        )
        n = len(scores)
        self._scores = scores
        self._weights = time_decay_weights(n, self.zeta, self.gamma, violation)
        self.n_effective_ = effective_sample_size(self._weights)
        # The exponential decay caps the effective sample count at
        # (1+zeta)/(1-zeta) -- 99 at zeta=0.98 -- no matter how large the
        # buffer is. Nothing here is wrong, but a radius estimated from ~99
        # effective samples is roughly 15x noisier than one using the whole
        # buffer, so a run whose calibrators differ mostly by seed should be
        # read in the light of this number, not of `n`.
        if n >= 20 and self.n_effective_ < 0.25 * n:
            print(f"[rtcqr.conformal] {self.name}: zeta={self.zeta} reduces {n} calibration samples to "
                  f"an effective {self.n_effective_:.0f}. The radius is set by roughly the most recent "
                  f"{self.n_effective_:.0f} samples; with stride-1 windows those overlap heavily, so the "
                  f"independent information is smaller still.")
        return self

    def radius(self, alpha: float) -> float:
        assert self._scores is not None, "call fit() first"
        return weighted_quantile(self._scores, self._weights, 1.0 - alpha,
                                 finite_sample_correction=self.finite_sample_correction)

    def calibrate_interval(self, q_lower: np.ndarray, q_upper: np.ndarray, alpha: float):
        c = self.radius(alpha)
        return q_lower - c, q_upper + c


class OnlineVWTACCalibrator:
    """Fully time-adaptive variant: at each new prediction time t the
    calibration radius is recomputed from all previously *resolved*
    calibration/test samples i <= t, per eq. (22) with the true t. Suited to
    streaming/rolling deployment where past ground-truth SoC becomes
    available (e.g. from lab reference measurements or coulomb counting)
    before the next prediction is issued.

    `calib_max_history` bounds the buffer for efficiency; since zeta < 1 the
    contribution of samples older than ~ log(eps) / log(zeta) is negligible,
    so a moderate cap has no material effect on the result.
    """

    def __init__(
        self,
        soc_min: float,
        zeta: float,
        gamma: float,
        wl0: float,
        wl1: float,
        wu: float,
        max_history: int = 2000,
        finite_sample_correction: bool = True,
    ):
        self.finite_sample_correction = finite_sample_correction
        self.soc_min = soc_min
        self.zeta = zeta
        self.gamma = gamma
        self.wl0 = wl0
        self.wl1 = wl1
        self.wu = wu
        self.max_history = max_history
        self._scores: list[float] = []
        self._violations: list[float] = []

    def warm_start(self, soc_calib: np.ndarray, q_lower_calib: np.ndarray, q_upper_calib: np.ndarray) -> "OnlineVWTACCalibrator":
        scores, violation = nonconformity_scores(
            soc_calib, q_lower_calib, q_upper_calib, self.soc_min, self.wl0, self.wl1, self.wu
        )
        self._scores = list(scores)
        self._violations = list(violation)
        return self

    def _trim(self):
        if len(self._scores) > self.max_history:
            self._scores = self._scores[-self.max_history:]
            self._violations = self._violations[-self.max_history:]

    def radius(self, alpha: float) -> float:
        n = len(self._scores)
        if n == 0:
            return 0.0
        scores = np.asarray(self._scores)
        violation = np.asarray(self._violations)
        weights = time_decay_weights(n, self.zeta, self.gamma, violation)
        return weighted_quantile(scores, weights, 1.0 - alpha,
                                 finite_sample_correction=self.finite_sample_correction)

    def calibrate_interval(self, q_lower: float, q_upper: float, alpha: float):
        c = self.radius(alpha)
        return q_lower - c, q_upper + c

    def update(self, soc_true: float, q_lower: float, q_upper: float):
        """Reveal the true SoC for the most recent prediction and append it
        to the calibration history for future time steps."""
        score, violation = nonconformity_scores(
            np.array([soc_true]), np.array([q_lower]), np.array([q_upper]),
            self.soc_min, self.wl0, self.wl1, self.wu,
        )
        self._scores.append(float(score[0]))
        self._violations.append(float(violation[0]))
        self._trim()
