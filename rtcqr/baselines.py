"""Conformal-calibration baselines from Table I, expressed as special cases
of the same StaticVWTACCalibrator used by RT-CQR.

Table I assigns each method a *different* nonconformity score, so the flags
below are not cosmetic:

  - CQR:    "standard CQR score" -- the signed residual
            max(q_l - y, y - q_u), unweighted and undecayed.
            zeta=1, gamma=0, all omega=1, signed.
  - WCP:    "standard CQR score; exponential weighting; zeta = 0.98" --
            same signed residual, exponentially time-decayed.
            zeta=0.98, gamma=0, all omega=1, signed.
  - RT-CQR: eq. (20)'s asymmetric score with the [.]_+ clip, plus the
            violation-weighted time decay of eq. (22).
            zeta=0.98, gamma=1.0, omega_l^(1) >= omega_l^(0) >= omega_u,
            clipped.

The signed score can push c_alpha negative and tighten an over-covering
interval; the clipped score cannot (see `conformal.nonconformity_scores`),
so giving all three the same score would misrepresent two of them.
"""
from __future__ import annotations

from .conformal import StaticVWTACCalibrator


def make_cqr_calibrator(soc_min: float) -> StaticVWTACCalibrator:
    """Table I "CQR": standard (signed, unweighted, undecayed) CQR score."""
    return StaticVWTACCalibrator(soc_min=soc_min, zeta=1.0, gamma=0.0, wl0=1.0, wl1=1.0, wu=1.0,
                                 signed_score=True)


def make_wcp_calibrator(soc_min: float, zeta: float = 0.98) -> StaticVWTACCalibrator:
    """Table I "WCP": standard CQR score with exponential time weighting."""
    return StaticVWTACCalibrator(soc_min=soc_min, zeta=zeta, gamma=0.0, wl0=1.0, wl1=1.0, wu=1.0,
                                 signed_score=True)


def make_rtcqr_calibrator(soc_min: float, zeta: float, gamma: float, wl0: float, wl1: float, wu: float,
                          signed_score: bool = False) -> StaticVWTACCalibrator:
    """Table I "RT-CQR": eq. (20) with the [.]_+ clip and eq. (22) weights.

    `signed_score=True` swaps in the standard CQR residual instead; that is
    not eq. (20) and is provided only for comparison.
    """
    return StaticVWTACCalibrator(soc_min=soc_min, zeta=zeta, gamma=gamma, wl0=wl0, wl1=wl1, wu=wu,
                                 signed_score=signed_score)
