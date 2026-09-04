"""Regression tests for VW-TAC calibration."""
from __future__ import annotations

import numpy as np
import pytest

from rtcqr.baselines import make_cqr_calibrator, make_rtcqr_calibrator, make_wcp_calibrator
from rtcqr.conformal import (
    OnlineVWTACCalibrator,
    StaticVWTACCalibrator,
    effective_sample_size,
    lower_tail_weight,
    nonconformity_scores,
    time_decay_weights,
    weighted_quantile,
)
from rtcqr.metrics import average_coverage_error, average_interval_width, lower_violation_rate


# --------------------------------------------------------------------------
# Weighted quantile
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", [50, 100, 500, 2000, 10000])
def test_weighted_quantile_picks_the_right_order_statistic(n):
    """Uniform weights must reproduce the textbook conformal quantile:
    ceil((1-a)n) without the finite-sample correction, ceil((1-a)(n+1)) with
    it. cumsum drift alone was enough to shift this by one from n~2000."""
    scores = np.arange(1.0, n + 1)          # value == rank
    w = np.full(n, 1.0 / n)
    assert weighted_quantile(scores, w, 0.90, finite_sample_correction=False) == np.ceil(0.90 * n)
    assert weighted_quantile(scores, w, 0.90, finite_sample_correction=True) == np.ceil(0.90 * (n + 1))


@pytest.mark.parametrize("n", [50, 200, 1000])
def test_finite_sample_correction_restores_nominal_coverage(n):
    """Without the (n+1) correction split conformal undercovers by ~1/(n+1)."""
    rng = np.random.default_rng(7)
    cov_plain, cov_corrected = [], []
    for _ in range(4000):
        z = np.abs(rng.normal(0, 1, n + 1))
        cal, test = z[:n], z[n]
        w = np.full(n, 1.0 / n)
        cov_plain.append(test <= weighted_quantile(cal, w, 0.90, finite_sample_correction=False))
        cov_corrected.append(test <= weighted_quantile(cal, w, 0.90, finite_sample_correction=True))
    assert np.mean(cov_corrected) >= np.mean(cov_plain)
    assert np.mean(cov_corrected) == pytest.approx(0.90, abs=0.015)


def test_weighted_quantile_handles_degenerate_inputs():
    assert weighted_quantile(np.array([]), np.array([]), 0.9) == 0.0
    assert weighted_quantile(np.zeros(100), np.full(100, 0.01), 0.9) == 0.0
    assert weighted_quantile(np.array([5.0]), np.array([1.0]), 0.9) == 5.0


def test_weighted_quantile_respects_the_weights():
    """A score carrying almost all the mass must dominate the result."""
    scores = np.array([0.0, 1.0])
    assert weighted_quantile(scores, np.array([0.999, 0.001]), 0.5,
                             finite_sample_correction=False) == 0.0
    assert weighted_quantile(scores, np.array([0.001, 0.999]), 0.5,
                             finite_sample_correction=False) == 1.0


# --------------------------------------------------------------------------
# Time decay
# --------------------------------------------------------------------------

def test_time_decay_weights_sum_to_one_and_favour_recency():
    w = time_decay_weights(500, 0.98, 0.0, np.zeros(500))
    assert w.sum() == pytest.approx(1.0)
    assert (np.diff(w) >= 0).all(), "weights must increase toward the most recent entry"


def test_zeta_one_is_uniform():
    w = time_decay_weights(100, 1.0, 0.0, np.zeros(100))
    assert np.allclose(w, 1.0 / 100)


def test_violation_upweighting():
    v = np.zeros(10); v[3] = 1.0
    w = time_decay_weights(10, 1.0, 1.0, v)
    assert w[3] == pytest.approx(2 * w[0]), "gamma=1 doubles a violation sample's weight"


@pytest.mark.parametrize("n", [1000, 20000, 200000])
def test_effective_sample_size_is_capped_by_zeta(n):
    """The exponential decay caps the effective count at (1+z)/(1-z) -- 99 at
    zeta=0.98 -- no matter how large the calibration buffer is. Growing the
    buffer past a few hundred does not stabilise the radius."""
    w = time_decay_weights(n, 0.98, 0.0, np.zeros(n))
    assert effective_sample_size(w) == pytest.approx(99.0, rel=0.02)


def test_effective_sample_size_of_uniform_weights_is_n():
    assert effective_sample_size(np.full(250, 1.0 / 250)) == pytest.approx(250.0)


def test_ess_warning_is_emitted(capsys):
    """The zeta=0.98 buffer collapse is easy to miss; it must be reported."""
    rng = np.random.default_rng(0)
    n = 5000
    make_wcp_calibrator(0.10).fit(rng.random(n), rng.random(n) - 0.5, rng.random(n) + 0.5)
    assert "effective" in capsys.readouterr().out


def test_cqr_emits_no_ess_warning(capsys):
    rng = np.random.default_rng(0)
    n = 5000
    make_cqr_calibrator(0.10).fit(rng.random(n), rng.random(n) - 0.5, rng.random(n) + 0.5)
    assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------
# Nonconformity score
# --------------------------------------------------------------------------

def test_score_is_the_weighted_max_of_both_excesses():
    soc = np.array([0.5, 0.05, 0.5])
    lo = np.array([0.6, 0.10, 0.0])      # sample 0 and 1 miss low, 2 covered
    hi = np.array([0.9, 0.9, 0.4])       # sample 2 misses high
    scores, viol = nonconformity_scores(soc, lo, hi, soc_min=0.10, wl0=1.5, wl1=3.0, wu=1.0)
    assert viol.tolist() == [0.0, 1.0, 0.0]
    assert scores[0] == pytest.approx(1.5 * 0.1)     # wl0 * lower excess
    assert scores[1] == pytest.approx(3.0 * 0.05)    # wl1, since soc < soc_min
    assert scores[2] == pytest.approx(1.0 * 0.1)     # wu * upper excess


def test_covered_samples_score_zero():
    scores, _ = nonconformity_scores(np.array([0.5]), np.array([0.4]), np.array([0.6]),
                                     0.10, 1.5, 3.0, 1.0)
    assert scores[0] == 0.0


def test_lower_tail_weight_interpolates():
    assert lower_tail_weight(np.array([0.0, 1.0]), 1.5, 3.0).tolist() == [1.5, 3.0]


# --------------------------------------------------------------------------
# Calibrators
# --------------------------------------------------------------------------

def test_calibration_widens_symmetrically():
    rng = np.random.default_rng(1)
    n = 400
    soc = rng.random(n)
    lo, hi = soc - 0.05, soc + 0.05
    cal = make_cqr_calibrator(0.10).fit(soc, lo, hi)
    c = cal.radius(0.10)
    out_lo, out_hi = cal.calibrate_interval(np.zeros(3), np.ones(3), 0.10)
    assert np.allclose(out_lo, -c) and np.allclose(out_hi, 1 + c)


def test_calibration_achieves_nominal_coverage_on_exchangeable_data():
    """The end-to-end property the whole module exists for."""
    rng = np.random.default_rng(3)
    n = 4000
    soc = rng.random(2 * n)
    noise = rng.normal(0, 0.05, 2 * n)
    lo, hi = soc + noise - 0.02, soc + noise + 0.02
    cal = make_cqr_calibrator(0.10).fit(soc[:n], lo[:n], hi[:n])
    t_lo, t_hi = cal.calibrate_interval(lo[n:], hi[n:], 0.10)
    covered = ((soc[n:] >= t_lo) & (soc[n:] <= t_hi)).mean()
    assert covered == pytest.approx(0.90, abs=0.02)


def test_rtcqr_differs_from_cqr_when_lower_violations_exist():
    """RT-CQR's violation weighting can only bite when the wl branch wins the
    max on samples whose true SoC is below soc_min. If it never does, all
    three calibrators collapse to the same number."""
    rng = np.random.default_rng(5)
    n = 2000
    soc = rng.uniform(0.0, 0.3, n)
    lo = soc + rng.normal(0.05, 0.02, n)     # lower bound sits above the truth
    hi = soc + 0.2
    r_cqr = make_cqr_calibrator(0.10).fit(soc, lo, hi).radius(0.10)
    r_rt = make_rtcqr_calibrator(0.10, 0.98, 1.0, 1.5, 3.0, 1.0).fit(soc, lo, hi).radius(0.10)
    assert r_rt != pytest.approx(r_cqr)


def test_all_calibrators_agree_when_calibration_is_perfect():
    """Documents the degenerate case diagnose_calibration.py hunts for: with
    every calibration sample covered, all scores are 0 and the radius is 0
    for every method regardless of its weights."""
    soc = np.full(500, 0.5)
    lo, hi = np.zeros(500), np.ones(500)
    for make in (lambda: make_cqr_calibrator(0.10),
                 lambda: make_wcp_calibrator(0.10),
                 lambda: make_rtcqr_calibrator(0.10, 0.98, 1.0, 1.5, 3.0, 1.0)):
        assert make().fit(soc, lo, hi).radius(0.10) == 0.0


def test_online_calibrator_matches_static_after_the_same_warm_start():
    rng = np.random.default_rng(9)
    n = 300
    soc, lo, hi = rng.random(n), rng.random(n) - 0.5, rng.random(n) + 0.5
    static = StaticVWTACCalibrator(0.10, 0.98, 1.0, 1.5, 3.0, 1.0).fit(soc, lo, hi)
    online = OnlineVWTACCalibrator(0.10, 0.98, 1.0, 1.5, 3.0, 1.0).warm_start(soc, lo, hi)
    assert online.radius(0.10) == pytest.approx(static.radius(0.10))


def test_online_calibrator_trims_its_history():
    online = OnlineVWTACCalibrator(0.10, 0.98, 1.0, 1.5, 3.0, 1.0, max_history=50)
    for _ in range(200):
        online.update(0.5, 0.4, 0.6)
    assert len(online._scores) == 50


def test_online_calibrator_with_no_history_returns_zero():
    assert OnlineVWTACCalibrator(0.10, 0.98, 1.0, 1.5, 3.0, 1.0).radius(0.10) == 0.0


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def test_lvr_counts_only_undetected_violations():
    """eq. (29): true SoC below the floor *while the prediction says it is not*."""
    soc = np.array([0.05, 0.05, 0.5, 0.5])
    pred = np.array([0.20, 0.05, 0.20, 0.05])   # only sample 0 is an undetected violation
    assert lower_violation_rate(soc, pred, 0.10) == pytest.approx(0.25)


def test_lvr_is_zero_when_nothing_violates():
    assert lower_violation_rate(np.full(10, 0.5), np.full(10, 0.5), 0.10) == 0.0


def test_aiw_and_ace():
    assert average_interval_width(np.zeros(4), np.full(4, 0.3)) == pytest.approx(0.3)
    soc = np.array([0.1, 0.2, 0.3, 0.9])
    lo, hi = np.zeros(4), np.full(4, 0.5)       # covers 3 of 4
    assert average_coverage_error(soc, lo, hi, 0.90) == pytest.approx(abs(0.75 - 0.90))
