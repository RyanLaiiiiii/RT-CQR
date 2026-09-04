"""Regression tests for rtcqr/data.py, focused on the SoC label pipeline.

Each test here pins a bug that silently corrupted the training labels: a
wrong capacity denominator does not raise, it just moves every SoC value.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from rtcqr.data import (
    _collapse_duplicate_timestamps,
    _compute_soc_from_current,
    _extract_temperature_c,
    _measured_capacity_ah,
    _split_counts,
    load_lg_hg2_dataframe,
    make_windows,
    order_chronologically,
    segment_split,
    BatteryFile,
)
from tests.conftest import TRUE_CAPACITY_AH, cc, write_section


# --------------------------------------------------------------------------
# Ambient-condition parsing
# --------------------------------------------------------------------------

@pytest.mark.parametrize("folder,expected", [
    ("25degC", 25.0), ("40degC", 40.0), ("0degC", 0.0), ("10degC", 10.0),
    ("n10degC", -10.0), ("n20degC", -20.0), ("N10degC", -10.0),
    ("-10degC", -10.0), ("-20degC", -20.0),      # re-export naming
    ("neg10degC", -10.0), ("minus10degC", -10.0),
    ("LG_HG2_n20degC_Prep", -20.0),
    ("no_temperature_here", None),
])
def test_temperature_sign_conventions(folder, expected):
    """A sub-zero folder read as positive merges two conditions into one
    capacity bucket, since `condition` keys the SoC denominator."""
    assert _extract_temperature_c(folder) == expected


# --------------------------------------------------------------------------
# Capacity measurement
# --------------------------------------------------------------------------

def test_capacity_of_a_clean_1c_check():
    t = np.arange(0.0, 3601.0, 10.0)
    i = np.full(len(t), -2.5)
    assert _measured_capacity_ah(t, i, 1.0, rated_capacity_ah=3.0) == pytest.approx(2.5, rel=0.01)


def test_capacity_sign_conventions_agree():
    """`_measured_capacity_ah` and `_compute_soc_from_current` must read
    `current_sign` the same way, or the denominator and numerator disagree."""
    t = np.arange(0.0, 3601.0, 10.0)
    for sign, current in ((1.0, -2.5), (-1.0, +2.5)):
        i = np.full(len(t), current)
        cap = _measured_capacity_ah(t, i, sign, rated_capacity_ah=3.0)
        assert cap == pytest.approx(2.5, rel=0.01)
        soc = _compute_soc_from_current(t, i, cap, sign, 1.0)
        assert soc[0] == pytest.approx(1.0, abs=0.02)
        assert soc[-1] == pytest.approx(0.0, abs=0.02)


def test_two_capacity_sections_are_not_summed():
    """Two Cap_1C sections in one measurement are separate checks of the same
    cell. Integrating the mask as one trace summed them (and charged
    max_gap_s of phantom current at the seam), roughly doubling the result."""
    seg = np.arange(0.0, 3601.0, 10.0)
    t = np.concatenate([seg, seg + 100000.0])          # hours apart
    i = np.full(len(t), -2.5)
    rows = np.concatenate([np.arange(len(seg)), np.arange(len(seg)) + 5000])  # non-contiguous
    cap = _measured_capacity_ah(t, i, 1.0, rated_capacity_ah=3.0, row_index=rows)
    assert cap == pytest.approx(2.5, rel=0.01), "should be the median of the two, not their sum"


def test_truncated_capacity_check_is_rejected():
    """A 20-minute stub integrates to a plausible-looking 0.83 Ah. Accepting
    it under-sizes the denominator, which drives SoC into the [0,1] clip and
    freezes whole drive-cycle segments at exactly 0."""
    t = np.arange(0.0, 1200.0, 10.0)
    assert _measured_capacity_ah(t, np.full(len(t), -2.5), 1.0, rated_capacity_ah=3.0) is None


def test_capacity_above_rating_is_rejected():
    t = np.arange(0.0, 3601.0, 10.0)
    assert _measured_capacity_ah(t, np.full(len(t), -9.0), 1.0, rated_capacity_ah=3.0) is None


def test_rest_padding_does_not_cancel_discharge():
    """Charge/rest padding inside the section must not net off the discharge."""
    t = np.arange(0.0, 5401.0, 10.0)
    i = np.full(len(t), -2.5)
    i[-180:] = +2.5                                    # 30 min of recharge at the end
    assert _measured_capacity_ah(t, i, 1.0, rated_capacity_ah=3.0) == pytest.approx(2.5, rel=0.02)


# --------------------------------------------------------------------------
# Sub-second rows
# --------------------------------------------------------------------------

def test_duplicate_timestamps_are_averaged_not_dropped():
    """The Time Stamp column has 1 s resolution while drive cycles log at
    ~10 Hz. Keeping one row per second discards 90% of the dynamics and makes
    the coulomb count an instantaneous sample rather than that second's mean."""
    base = dt.datetime(2018, 11, 27, 20, 41, 18)
    stamps = pd.to_datetime([base + dt.timedelta(seconds=k // 10) for k in range(100)])
    current = np.arange(100.0)
    df = pd.DataFrame({"abs_time": stamps, "voltage": current, "current": current,
                       "temperature": current, "soc_raw": np.nan, "test_section": "HWFET"})
    out = _collapse_duplicate_timestamps(df)
    assert len(out) == 10
    assert out["current"].mean() == pytest.approx(current.mean())
    assert out["current"].iloc[0] == pytest.approx(4.5)   # mean of 0..9, not 0.0


# --------------------------------------------------------------------------
# Coulomb counting
# --------------------------------------------------------------------------

def test_soc_clips_every_step_not_once_at_the_end():
    """Repeated charge/discharge cycling: without a per-step clip the running
    sum drifts above 1.0 and the whole span reads as a flat 1.0."""
    t = np.arange(0.0, 4 * 3600.0, 1.0)
    i = np.where((t // 3600) % 2 == 0, +3.0, -3.0)
    soc = _compute_soc_from_current(t, i, 3.0, 1.0, 1.0)
    assert soc.max() <= 1.0 and soc.min() >= 0.0
    assert soc.min() < 0.1, "the discharge halves must actually deplete"


def test_missing_data_gap_is_capped_not_extrapolated():
    t = np.array([0.0, 10.0, 20.0, 100000.0, 100010.0])
    i = np.full(5, -3.0)
    soc = _compute_soc_from_current(t, i, 3.0, 1.0, 1.0, max_gap_s=300.0)
    # The gap contributes at most 300 s of current, not 100000 s.
    assert soc[3] > 0.5


# --------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------

@pytest.mark.parametrize("window,stride", [(100, 1), (100, 3), (50, 7), (20, 1)])
def test_make_windows_matches_the_naive_construction(window, stride):
    rng = np.random.default_rng(0)
    frames = [pd.DataFrame({c: rng.random(n).astype(np.float32)
                            for c in ("voltage", "current", "temperature", "soc")})
              for n in (250, 431, 100, 19)]
    xs, ys = [], []
    for df in frames:
        if len(df) < window:
            continue
        v, i, t, s = (df[c].to_numpy(np.float32) for c in ("voltage", "current", "temperature", "soc"))
        for end in range(window - 1, len(df), stride):
            xs.append(np.stack([v[end - window + 1:end + 1], i[end - window + 1:end + 1],
                                t[end - window + 1:end + 1]], axis=0))
            ys.append(s[end])
    X, y = make_windows(frames, window, stride)
    assert np.array_equal(X, np.stack(xs, 0))
    assert np.array_equal(y, np.asarray(ys, np.float32))
    assert X.flags["C_CONTIGUOUS"]


def test_make_windows_accepts_battery_files():
    df = pd.DataFrame({c: np.zeros(60, np.float32)
                       for c in ("time", "voltage", "current", "temperature", "soc")})
    X, y = make_windows([BatteryFile("m", 25.0, df)], 20, 1)
    assert X.shape == (41, 3, 20) and y.shape == (41,)


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------

def _seg(cond, i, start):
    df = pd.DataFrame({c: np.zeros(200, np.float32)
                       for c in ("time", "voltage", "current", "temperature", "soc")})
    return BatteryFile(f"m{cond}_{i}", cond, df,
                       start_time=pd.Timestamp("2018-11-20") + pd.Timedelta(hours=start))


CONDITIONS = (-20.0, -10.0, 0.0, 10.0, 25.0, 40.0)


@pytest.mark.parametrize("per_condition", [4, 8, 16])
def test_stratified_split_never_leaves_test_condition_unseen_by_calib(per_condition):
    """Conformal coverage assumes calib and test are exchangeable. An
    unstratified split leaves a temperature out of calib in almost every
    seed, and this dataset's error distribution is strongly temperature
    dependent."""
    files = [_seg(c, i, k) for k, (c, i) in
             enumerate((c, i) for c in CONDITIONS for i in range(per_condition))]
    for seed in range(50):
        _, _, calib, test = segment_split(files, 0.70, 0.15, 0.5, seed=seed)
        assert {b.condition for b in test} <= {b.condition for b in calib}
        assert calib and test


def test_splits_are_disjoint_and_cover_everything():
    files = [_seg(c, i, k) for k, (c, i) in
             enumerate((c, i) for c in CONDITIONS for i in range(8))]
    train, val, calib, test = segment_split(files, 0.70, 0.15, 0.5, seed=3)
    ids = [{id(b.frame) for b in split} for split in (train, val, calib, test)]
    assert sum(len(s) for s in ids) == len(files)
    assert set.union(*ids) == {id(b.frame) for b in files}
    for a in range(4):
        for b in range(a + 1, 4):
            assert not ids[a] & ids[b]


def test_split_is_deterministic_for_a_seed():
    files = [_seg(c, i, k) for k, (c, i) in
             enumerate((c, i) for c in CONDITIONS for i in range(8))]
    a = segment_split(files, 0.70, 0.15, 0.5, seed=11)
    b = segment_split(files, 0.70, 0.15, 0.5, seed=11)
    for sa, sb in zip(a, b):
        assert [x.path for x in sa] == [x.path for x in sb]


@pytest.mark.parametrize("n", [4, 5, 8, 20, 60, 120])
def test_split_counts_leave_no_empty_split(n):
    counts = _split_counts(n, 0.70, 0.15, 0.5)
    assert sum(counts) == n
    assert all(c >= 1 for c in counts)


def test_too_few_segments_raises_instead_of_returning_an_empty_split():
    files = [_seg(25.0, i, i) for i in range(3)]
    with pytest.raises(ValueError, match="at least 4"):
        segment_split(files, 0.70, 0.15, 0.5)


def test_order_chronologically_sorts_by_absolute_start():
    """time_decay_weights reads buffer position as time, so the calibration
    buffer must be in chronological order for the decay to mean anything."""
    files = [_seg(25.0, i, start) for i, start in enumerate([5, 1, 9, 3])]
    got = [b.start_time for b in order_chronologically(files)]
    assert got == sorted(got)


def test_order_chronologically_tolerates_missing_start_times():
    files = [_seg(25.0, 0, 5), BatteryFile("no-time", 25.0, pd.DataFrame(), start_time=None)]
    assert len(order_chronologically(files)) == 2


# --------------------------------------------------------------------------
# End-to-end loader
# --------------------------------------------------------------------------

def test_loader_recovers_true_per_condition_capacity(full_dataset, capsys):
    load_lg_hg2_dataframe(full_dataset)
    out = capsys.readouterr().out
    for folder, true_cap in TRUE_CAPACITY_AH.items():
        cond = -float(folder[1:-4]) if folder.startswith("n") else float(folder[:-4])
        line = next(l for l in out.splitlines() if f"condition={cond}:" in l and "measured" in l)
        measured = float(line.split("measured capacity ")[1].split(" Ah")[0])
        assert measured == pytest.approx(true_cap, rel=0.02), f"{folder}: {measured} vs {true_cap}"


def test_loader_produces_usable_segments(full_dataset):
    files = load_lg_hg2_dataframe(full_dataset)
    assert len(files) >= 24
    for bf in files:
        assert bf.condition in (-20.0, -10.0, 0.0, 10.0, 25.0, 40.0)
        assert bf.start_time is not None
        assert bf.frame["soc"].between(0.0, 1.0).all()
        assert bf.frame["time"].is_monotonic_increasing


def test_falls_back_to_rated_capacity_when_no_check_exists(tmp_path):
    t = dt.datetime(2018, 11, 20, 8, 0, 0)
    base = str(tmp_path / "25degC")
    write_section(f"{base}/900_UDDS.csv", "900", "UDDS", t, cc(3600, -1.5), 25.0)
    files = load_lg_hg2_dataframe(str(tmp_path), rated_capacity_ah=3.0)
    # 1.5 A for 1 h = 1.5 Ah out of 3.0 Ah rated -> SoC ends at 0.5.
    assert files[0].frame["soc"].iloc[-1] == pytest.approx(0.5, abs=0.02)


def test_capacity_override_takes_precedence(tmp_path):
    t = dt.datetime(2018, 11, 20, 8, 0, 0)
    base = str(tmp_path / "25degC")
    write_section(f"{base}/901_UDDS.csv", "901", "UDDS", t, cc(3600, -1.0), 25.0)
    files = load_lg_hg2_dataframe(str(tmp_path), rated_capacity_ah=3.0,
                                  capacity_overrides={25.0: 2.0})
    assert files[0].frame["soc"].iloc[-1] == pytest.approx(0.5, abs=0.02)


def test_degenerate_segments_are_dropped_when_asked(tmp_path):
    """A drive cycle sitting in the saturated full-charge region carries a
    near-constant label while V/I/T vary, so it teaches nothing and gives
    the calibrator degenerate nonconformity scores."""
    import datetime as _dt
    t = _dt.datetime(2018, 11, 20, 8, 0, 0)
    base = str(tmp_path / "25degC")
    # Charge well past full, then a drive cycle that barely moves SoC.
    t = write_section(f"{base}/950_Charge1.csv", "950", "Charge1", t, cc(7200, +3.0), 25.0)
    write_section(f"{base}/950_UDDS.csv", "950", "UDDS", t, cc(600, -0.01), 25.0)

    kept = load_lg_hg2_dataframe(str(tmp_path), rated_capacity_ah=3.0)
    assert len(kept) == 1
    span = kept[0].frame["soc"].max() - kept[0].frame["soc"].min()
    assert span < 0.02, "fixture should produce a near-constant-SoC segment"

    with pytest.raises(RuntimeError, match="No usable battery segments"):
        load_lg_hg2_dataframe(str(tmp_path), rated_capacity_ah=3.0, min_soc_range=0.02)


def test_min_soc_range_keeps_healthy_segments(full_dataset):
    assert len(load_lg_hg2_dataframe(full_dataset, min_soc_range=0.02)) == \
           len(load_lg_hg2_dataframe(full_dataset))


# --------------------------------------------------------------------------
# CLI override parsing
# --------------------------------------------------------------------------

@pytest.mark.parametrize("argv,expected", [
    (["40:2.75"], {40.0: 2.75}),
    (["40:2.75", "25:2.71"], {40.0: 2.75, 25.0: 2.71}),
    (["40:2.75,25:2.71"], {40.0: 2.75, 25.0: 2.71}),
    (["n20:1.70"], {-20.0: 1.70}),                 # dataset's own spelling
    (["N20:1.70"], {-20.0: 1.70}),
    (["-20:1.70"], {-20.0: 1.70}),                 # works via --flag=-20:1.70
    (["40.0:2.75"], {40.0: 2.75}),
    ([" 40 : 2.75 "], {40.0: 2.75}),
    ([], {}),
    (None, {}),
])
def test_parse_capacity_overrides(argv, expected):
    from train import parse_capacity_overrides
    assert parse_capacity_overrides(argv) == expected


@pytest.mark.parametrize("bad", [["40"], ["40:abc"], ["x:2.7"], ["40:0"], ["40:-1"]])
def test_parse_capacity_overrides_rejects_junk(bad):
    import argparse
    from train import parse_capacity_overrides
    with pytest.raises(argparse.ArgumentTypeError):
        parse_capacity_overrides(bad)


def test_override_reaches_the_soc_labels(tmp_path):
    """The whole point of the flag: it must change the denominator, not just
    be recorded in the config."""
    import datetime as _dt
    t = _dt.datetime(2018, 11, 20, 8, 0, 0)
    base = str(tmp_path / "40degC")
    # A Cap_1C that stops early: full duration, but only 2.0 Ah delivered.
    t = write_section(f"{base}/700_Cap_1C.csv", "700", "Cap_1C", t, cc(3600, -2.0), 40.0)
    t = write_section(f"{base}/700_Charge1.csv", "700", "Charge1", t, cc(3600, +2.0), 40.0)
    write_section(f"{base}/700_UDDS.csv", "700", "UDDS", t, cc(1800, -2.0), 40.0)

    measured = load_lg_hg2_dataframe(str(tmp_path), rated_capacity_ah=3.0)
    overridden = load_lg_hg2_dataframe(str(tmp_path), rated_capacity_ah=3.0,
                                       capacity_overrides={40.0: 2.75})
    # 1.0 Ah out of the drive cycle: 50% of 2.0 Ah, but only 36% of 2.75 Ah.
    assert measured[0].frame["soc"].iloc[-1] == pytest.approx(0.50, abs=0.02)
    assert overridden[0].frame["soc"].iloc[-1] == pytest.approx(0.636, abs=0.02)
