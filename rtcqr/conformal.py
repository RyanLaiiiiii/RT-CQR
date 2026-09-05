"""Violation-weighted time-adaptive conformal calibration (VW-TAC), eq. (19)-(28).

Nonconformity score, eq. (20)-(21):
    u_i = 1{SoC_i < SoC_min}
    w_l(u_i) = w_l^(0) + u_i * (w_l^(1) - w_l^(0)),   w_l^(1) >= w_l^(0) >= w_u >= 0
    h_i = max( w_l(u_i) * (q_tl,i - SoC_i) , w_u * (SoC_i - q_tu,i) )

The residuals keep their sign, as in standard CQR, so that c_alpha can be
negative and calibration can tighten an over-covering interval. Taking
eq. (20)'s [.]_+ literally makes calibration a pure widening operator and
collapses RT-CQR, CQR and WCP onto the uncalibrated model whenever the
quantile model over-covers -- see `nonconformity_scores`, and pass
signed_score=False (train.py --clipped-score) for that literal reading.

Time-decayed, violation-weighted empirical measure, eq. (22)-(24):
    gamma_{i,t} = zeta^{t-i} * (1 + gamma * u_i),   i <= t
    normalized:  gamma-hat_{i,t} = gamma_{i,t} / sum_j gamma_{j,t}

Calibrated radius as a weighted empirical quantile, eq. (25)-(26):
    c_alpha,t = inf{ c : sum_{i: h_i<=c} gamma-hat_{i,t} >= 1 - alpha }

Calibrated interval via Minkowski addition, eq. (27)-(28):
    PI^cal_t = [ q_tl,t - c_alpha,t , q_tu,t + c_alpha,t ]
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
    signed_score: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """eq. (20). Returns (scores, violation_indicator).

    `signed_score` controls whether the two residuals keep their sign:

      signed (default):  h_i = max( w_l(u_i) * (q_tl,i - SoC_i),
                                    w_u      * (SoC_i - q_tu,i) )
      clipped:           h_i = max( w_l(u_i) * [q_tl,i - SoC_i]_+,
                                    w_u      * [SoC_i - q_tu,i]_+ )

    The signed form is standard CQR (Romano et al., 2019) generalized with
    the asymmetric weights: for a point comfortably inside the interval
    both residuals are negative, the score is the (negative) slack, and
    c_alpha can come out negative so the calibrated interval *tightens*.

    The clipped form -- the literal reading of eq. (20)'s [.]_+ notation --
    cannot do that. Its score is >= 0, hence c_alpha >= 0, hence
    calibration is a pure widening operator that can only ever fix
    *under*-coverage. When the quantile model over-covers, every
    calibrator returns c_alpha = 0 and RT-CQR, CQR and WCP all collapse
    onto the uncalibrated model, byte for byte. That is not hypothetical:
    measured on this dataset the model covered 97.6% of the calibration
    set at 90% nominal, so all four rows of the results table came out
    identical, and switching to the signed score cut AIW from 0.043 to
    0.028 and ACE from 0.088 to 0.028.

    It also matters for the baselines: `make_cqr_calibrator` documents
    itself as "standard CQR", which is the signed score by definition.
    Under clipping it is not standard CQR at all.
    """
    violation = (soc_true < soc_min).astype(np.float64)
    wl = lower_tail_weight(violation, wl0, wl1)
    lower_excess = q_lower - soc_true
    upper_excess = soc_true - q_upper
    if not signed_score:
        lower_excess = np.clip(lower_excess, 0.0, None)
        upper_excess = np.clip(upper_excess, 0.0, None)
    scores = np.maximum(wl * lower_excess, wu * upper_excess)
    return scores, violation


def weighted_quantile(scores: np.ndarray, weights: np.ndarray, level: float) -> float:
    """eq. (25)-(26): smallest score c such that the weighted CDF at c is >= level.

    `weights` must be non-negative and sum to (approximately) 1.
    """
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    sorted_weights = weights[order]
    cum = np.cumsum(sorted_weights)
    idx = np.searchsorted(cum, level, side="left")
    idx = min(idx, len(sorted_scores) - 1)
    return float(sorted_scores[idx])


def time_decay_weights(n: int, zeta: float, gamma: float, violation: np.ndarray) -> np.ndarray:
    """eq. (22)-(23) for a calibration buffer of size n, evaluated at t = n
    (i.e. weights relative to "now", the most recent buffer entry).

    lags[i] = n - 1 - i for i = 0..n-1 (0 = most recent sample), so
    gamma_i = zeta**lags[i] * (1 + gamma * violation[i]).
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
                 signed_score: bool = True):
        self.soc_min = soc_min
        self.zeta = zeta
        self.gamma = gamma
        self.wl0 = wl0
        self.wl1 = wl1
        self.wu = wu
        self.signed_score = signed_score
        self._scores: Optional[np.ndarray] = None
        self._weights: Optional[np.ndarray] = None

    def fit(self, soc_calib: np.ndarray, q_lower_calib: np.ndarray, q_upper_calib: np.ndarray) -> "StaticVWTACCalibrator":
        scores, violation = nonconformity_scores(
            soc_calib, q_lower_calib, q_upper_calib, self.soc_min, self.wl0, self.wl1, self.wu,
            self.signed_score,
        )
        n = len(scores)
        self._scores = scores
        self._weights = time_decay_weights(n, self.zeta, self.gamma, violation)
        return self

    def radius(self, alpha: float) -> float:
        assert self._scores is not None, "call fit() first"
        return weighted_quantile(self._scores, self._weights, 1.0 - alpha)

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
        signed_score: bool = True,
    ):
        self.soc_min = soc_min
        self.zeta = zeta
        self.gamma = gamma
        self.wl0 = wl0
        self.wl1 = wl1
        self.wu = wu
        self.max_history = max_history
        self.signed_score = signed_score
        self._scores: list[float] = []
        self._violations: list[float] = []

    def warm_start(self, soc_calib: np.ndarray, q_lower_calib: np.ndarray, q_upper_calib: np.ndarray) -> "OnlineVWTACCalibrator":
        scores, violation = nonconformity_scores(
            soc_calib, q_lower_calib, q_upper_calib, self.soc_min, self.wl0, self.wl1, self.wu,
            self.signed_score,
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
        return weighted_quantile(scores, weights, 1.0 - alpha)

    def calibrate_interval(self, q_lower: float, q_upper: float, alpha: float):
        c = self.radius(alpha)
        return q_lower - c, q_upper + c

    def update(self, soc_true: float, q_lower: float, q_upper: float):
        """Reveal the true SoC for the most recent prediction and append it
        to the calibration history for future time steps."""
        score, violation = nonconformity_scores(
            np.array([soc_true]), np.array([q_lower]), np.array([q_upper]),
            self.soc_min, self.wl0, self.wl1, self.wu, self.signed_score,
        )
        self._scores.append(float(score[0]))
        self._violations.append(float(violation[0]))
        self._trim()
