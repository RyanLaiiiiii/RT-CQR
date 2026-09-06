"""Evaluation metrics: LVR, AIW, ACE (Sec. IV.B, eq. 29)."""
from __future__ import annotations

import numpy as np


def lower_violation_rate(soc_true: np.ndarray, y_lower_or_point: np.ndarray, soc_min: float) -> float:
    """eq. (29): LVR = mean( 1{SoC_t < SoC_min} * 1{y_t >= SoC_min} ).

    `y_lower_or_point` is the predicted PI lower bound for interval methods,
    or the point estimate for the deterministic baseline.
    """
    violation = soc_true < soc_min
    predicted_feasible = y_lower_or_point >= soc_min
    return float(np.mean(violation & predicted_feasible))


def average_interval_width(q_lower: np.ndarray, q_upper: np.ndarray) -> float:
    return float(np.mean(q_upper - q_lower))


def average_coverage_error(soc_true: np.ndarray, q_lower: np.ndarray, q_upper: np.ndarray, nominal_coverage: float) -> float:
    empirical = np.mean((soc_true >= q_lower) & (soc_true <= q_upper))
    return float(abs(empirical - nominal_coverage))


def empirical_coverage(soc_true: np.ndarray, q_lower: np.ndarray, q_upper: np.ndarray) -> float:
    """Fraction of samples the interval actually contains.

    ACE is an absolute difference, so it cannot say whether an interval is
    too narrow or too wide -- ACE 0.085 at 90% nominal is either coverage
    0.815 or 0.985, and the corrective actions are opposite. Reported
    alongside so the direction is never inferred.
    """
    return float(np.mean((soc_true >= q_lower) & (soc_true <= q_upper)))


def summarize(soc_true: np.ndarray, q_lower: np.ndarray, q_upper: np.ndarray, alpha: float, soc_min: float) -> dict:
    return {
        "LVR": lower_violation_rate(soc_true, q_lower, soc_min),
        "AIW": average_interval_width(q_lower, q_upper),
        "ACE": average_coverage_error(soc_true, q_lower, q_upper, nominal_coverage=1.0 - alpha),
        "coverage": empirical_coverage(soc_true, q_lower, q_upper),
    }
