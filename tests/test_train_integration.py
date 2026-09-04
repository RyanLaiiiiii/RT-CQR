"""End-to-end pipeline tests: real files in, trained model and metrics out."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from rtcqr.config import RTCQRConfig
from train import _report_split_conditions, build_windows, evaluate, predict_quantiles, train_model


@pytest.fixture(scope="module")
def small_cfg():
    cfg = RTCQRConfig(seed=0)
    cfg.window_size = 20
    cfg.stride = 40          # keep the window count small; this is a wiring test
    cfg.max_epochs = 2
    cfg.patience = 1
    cfg.batch_size = 32
    cfg.num_blocks = 2
    cfg.channels = 8
    return cfg


@pytest.fixture(scope="module")
def splits(small_cfg, full_dataset):
    return build_windows(small_cfg, full_dataset, current_sign=1.0)


def test_build_windows_shapes_and_dtypes(splits, small_cfg):
    for name in ("train", "val", "calib", "test"):
        X, y = splits[name]
        assert X.ndim == 3 and X.shape[1] == small_cfg.in_channels
        assert X.shape[2] == small_cfg.window_size
        assert X.shape[0] == y.shape[0] > 0
        assert X.dtype == np.float32 and y.dtype == np.float32
        assert np.isfinite(X).all() and np.isfinite(y).all()
        assert ((y >= 0.0) & (y <= 1.0)).all()


def test_standardizer_is_fit_on_train_only(splits):
    """Fitting the scaler on anything but train leaks test statistics."""
    X_train = splits["train"][0]
    assert X_train.mean(axis=(0, 2)) == pytest.approx(np.zeros(3), abs=1e-4)
    assert X_train.std(axis=(0, 2)) == pytest.approx(np.ones(3), abs=1e-3)


def test_calibration_windows_are_chronologically_ordered(small_cfg, full_dataset):
    """time_decay_weights treats buffer position as time, so a permuted calib
    buffer would make the whole time-adaptive component weight noise."""
    from rtcqr.data import load_lg_hg2_dataframe, order_chronologically, segment_split
    files = load_lg_hg2_dataframe(full_dataset)
    _, _, calib, _ = segment_split(files, small_cfg.train_frac, small_cfg.val_frac,
                                   small_cfg.val_calib_fraction, seed=small_cfg.seed)
    ordered = order_chronologically(calib)
    starts = [bf.start_time for bf in ordered]
    assert starts == sorted(starts)


def test_every_test_condition_appears_in_calibration(small_cfg, full_dataset, capsys):
    from rtcqr.data import load_lg_hg2_dataframe, segment_split
    files = load_lg_hg2_dataframe(full_dataset)
    train, val, calib, test = segment_split(files, small_cfg.train_frac, small_cfg.val_frac,
                                            small_cfg.val_calib_fraction, seed=small_cfg.seed)
    _report_split_conditions(train, val, calib, test)
    assert "WARNING" not in capsys.readouterr().out
    assert {b.condition for b in test} <= {b.condition for b in calib}


def test_training_runs_and_predictions_are_monotone(small_cfg, splits):
    device = torch.device("cpu")
    model = train_model(small_cfg, splits, device, num_workers=0)
    q = predict_quantiles(model, splits["test"][0], device)
    assert q.shape == (len(splits["test"][1]), len(small_cfg.quantile_levels))
    assert np.isfinite(q).all()
    crossing_rate = float(np.mean(np.any(np.diff(q, axis=1) < 0, axis=1)))
    assert crossing_rate == 0.0, "the head makes crossing structurally impossible"


def test_evaluate_reports_all_calibrators_and_sane_metrics(small_cfg, splits):
    device = torch.device("cpu")
    model = train_model(small_cfg, splits, device, num_workers=0)
    results = evaluate(small_cfg, model, splits, device, calibrators=["rtcqr", "cqr", "wcp"])
    assert set(results) == {"90%", "95%"}
    for pi, per_calib in results.items():
        assert set(per_calib) == {"rtcqr", "cqr", "wcp"}
        for name, m in per_calib.items():
            assert 0.0 <= m["LVR"] <= 1.0
            assert m["AIW"] >= 0.0
            assert 0.0 <= m["ACE"] <= 1.0


def test_wider_nominal_coverage_gives_a_wider_interval(small_cfg, splits):
    """A 95% PI must never be narrower than the 90% PI from the same model --
    that inversion was the original symptom of a broken split."""
    device = torch.device("cpu")
    model = train_model(small_cfg, splits, device, num_workers=0)
    results = evaluate(small_cfg, model, splits, device, calibrators=["cqr"])
    assert results["95%"]["cqr"]["AIW"] >= results["90%"]["cqr"]["AIW"]
