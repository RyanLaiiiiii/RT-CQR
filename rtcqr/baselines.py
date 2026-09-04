"""Conformal-calibration baselines from Table I, expressed as special cases
of the same StaticVWTACCalibrator used by RT-CQR:

  - CQR: standard (unweighted, non-decayed) conformal score.
         zeta=1, gamma=0, wl0=wl1=wu=1.
  - WCP: standard CQR score with exponential time weighting only.
         zeta=0.98, gamma=0, wl0=wl1=wu=1.
  - RT-CQR: violation-weighted, time-decayed score (see conformal.py).

Using the same calibrator for all three isolates the effect of RT-CQR's two
extra ingredients (violation weighting and lower-tail loss regularization)
rather than comparing across unrelated implementations.
"""
from __future__ import annotations

from .conformal import StaticVWTACCalibrator


def make_cqr_calibrator(soc_min: float, finite_sample_correction: bool = True) -> StaticVWTACCalibrator:
    return StaticVWTACCalibrator(soc_min=soc_min, zeta=1.0, gamma=0.0, wl0=1.0, wl1=1.0, wu=1.0,
                                 finite_sample_correction=finite_sample_correction, name="cqr")


def make_wcp_calibrator(soc_min: float, zeta: float = 0.98,
                        finite_sample_correction: bool = True) -> StaticVWTACCalibrator:
    return StaticVWTACCalibrator(soc_min=soc_min, zeta=zeta, gamma=0.0, wl0=1.0, wl1=1.0, wu=1.0,
                                 finite_sample_correction=finite_sample_correction, name="wcp")


def make_rtcqr_calibrator(soc_min: float, zeta: float, gamma: float, wl0: float, wl1: float, wu: float,
                          finite_sample_correction: bool = True) -> StaticVWTACCalibrator:
    return StaticVWTACCalibrator(soc_min=soc_min, zeta=zeta, gamma=gamma, wl0=wl0, wl1=wl1, wu=wu,
                                 finite_sample_correction=finite_sample_correction, name="rtcqr")
