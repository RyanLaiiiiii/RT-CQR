"""Data loading and windowing for the LG 18650HG2 Li-ion battery dataset.

Real exports of this dataset are battery-cycler exports, not plain flat
CSVs:

    Measurement ID,585
    Battery Name,LG HG2 18650_SN62A4
    ...
    Nominal Capacity, 3
    ...
    Test section,C20DisCh
    ...
    Time Stamp,Step,Status,Prog Time,Step Time,Cycle,Cycle Level,Procedure,Voltage,Current,Temperature,Capacity,WhAccu,Cnt,
    ,,,,,,,,[V],[A],[C],[Ah],[Wh],[Cnt],
    11/27/2018 8:41:18 PM,22,DCH,25:19:08.386,00:01:00.004,0,0,LG_HG2_NN_Char,4.16273,-0.15325,-0.42063,-0.00253,-0.01052,13.00000,

i.e. ~20-30 metadata lines, then the real header row, then a units row,
then the data. `_read_measurement_csv` locates and skips the
preamble/units rows automatically. Some files pad every metadata row
(including the preamble) with trailing commas to match the widest row's
column count (e.g. "Measurement ID,549,,,,,,,,,,,,"); `_extract_metadata`
strips that padding so it doesn't corrupt the Measurement ID / Test
section values used for grouping below.

Critically, one CSV is *not* one independent test: filenames look like
"<measurement_id>_<TestSection>.csv", and files sharing the same
"Measurement ID" are chronologically contiguous slices of a single
continuous cycler run (confirmed via each row's own timestamp -- the
file-level "Start Time"/"End Time" metadata is identical across every
section of a measurement, since it's the measurement's overall span, not
the individual section's). Treating each file as its own independent test
starting from a full charge is therefore wrong for any section after the
first in a run. This module instead:

  1. Groups files by Measurement ID (from metadata; falls back to
     treating each file as its own group if a copy of the dataset lacks
     the metadata preamble, e.g. a flatter CSV re-export).
  2. Concatenates all sections of a group, sorted by each row's own
     timestamp, deduplicated.
  3. Computes SoC via coulomb counting *once*, continuously, across the
     whole reconstructed run, so SoC(0)=1.0 is only assumed at the start
     of the run's earliest available section, with the running value
     clipped to [0, 1] at *every* step (not once at the end -- see
     `_compute_soc_from_current` for why this matters for measurements
     that are repeated charge/discharge cycling runs rather than one
     continuous depleting sweep).
     (The per-row Capacity[Ah] column looked promising for this but is
     unreliable: it resets to 0 at internal step boundaries, so only the
     raw Current signal is used.)
  4. Resamples the continuous run onto a uniform time grid (default 1 Hz),
     since sections are logged at very different native rates (tens of
     seconds between samples during slow characterization/charge
     sections, vs. ~0.1s during dynamic drive-cycle sections), which
     would otherwise make a fixed window_size span wildly different
     real-time durations depending on which section a window falls in.
  5. Splits back into contiguous segments for windowing, breaking wherever
     a test section is excluded (see below) or wherever consecutive rows
     are more than `max_gap_s` apart in real time -- which happens if your
     copy of the dataset is missing an intermediate section, since SoC
     cannot be tracked correctly across an unobserved gap in the current
     signal (a warning is printed when this is detected).

Some test sections are static characterization or charge/rest/maintenance
runs (e.g. "C20DisCh"/"Dis_0p5C"/"Dis_2C" = constant-current discharge
characterization, "HPPC" = pulse power characterization, "Cap_1C" = a 1C
capacity check, "Charge*", "PausCycl") rather than the dynamic drive-cycle
profiles the paper evaluates on. Their current is still used for SoC
continuity (step 3 above), but only sections matching `include_patterns`
(default: the drive-cycle profile names HWFET/UDDS/LA92/US06/Mixed*) are
kept for the windows used in training/evaluation; pass
`include_patterns=None` to `load_lg_hg2_dataframe` (or `--include-all` to
train.py) to keep everything. A blacklist of characterization-test
keywords was tried first but proved fragile (real data includes section
names like "Dis_0p5C" that a blacklist has to keep growing to catch);
whitelisting the small, closed set of real drive-cycle names is more
robust.

Column names can still differ slightly across re-exports, so voltage /
current / temperature / time / SoC columns are auto-detected by keyword
rather than hard-coded. Run `python -m rtcqr.data inspect <path>` to print
a per-measurement-group summary (row count, time span, SoC range/trend,
which test sections were found) before training, and pass
`column_overrides` to `load_lg_hg2_dataframe` to fix any file where
detection guesses wrong.
"""
from __future__ import annotations

import glob
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_KEYWORDS = {
    "time": ["time stamp", "prog time", "test_time", "time"],
    "voltage": ["voltage"],
    "current": ["current"],
    "temperature": ["temp"],
    "capacity": ["capacity", "ah accu", "ah_accu"],
    "soc": ["soc", "state of charge", "stateofcharge"],
}

# Test-section-name substrings (case-insensitive) identifying the dynamic
# drive-cycle profiles the paper evaluates on. Everything else (static
# characterization runs, charge/rest/maintenance sections) is used only
# for SoC continuity across a stitched run, then excluded from windowing.
_DEFAULT_INCLUDE_PATTERNS = ["hwfet", "udds", "la92", "us06", "mixed"]

_KNOWN_DATETIME_FORMATS = ["%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S"]


def download_lg_hg2(dataset_slug: str = "aditya9790/lg-18650hg2-liion-battery-data") -> str:
    """Download the dataset via kagglehub and return the local path.

    Equivalent to the snippet:
        import kagglehub
        path = kagglehub.dataset_download("aditya9790/lg-18650hg2-liion-battery-data")
    Requires Kaggle API credentials to be configured (~/.kaggle/kaggle.json
    or KAGGLE_USERNAME/KAGGLE_KEY env vars).
    """
    import kagglehub

    path = kagglehub.dataset_download(dataset_slug)
    print(f"Path to dataset files: {path}")
    return path


def _match_column(columns: Sequence[str], keywords: Sequence[str]) -> Optional[str]:
    lowered = {c: c.lower() for c in columns}
    for kw in keywords:
        for col, low in lowered.items():
            if kw in low:
                return col
    return None


def detect_columns(df: pd.DataFrame, overrides: Optional[Dict[str, str]] = None) -> Dict[str, Optional[str]]:
    """Best-effort mapping from canonical field name -> actual column name."""
    overrides = overrides or {}
    mapping: Dict[str, Optional[str]] = {}
    for field_name, keywords in _KEYWORDS.items():
        if field_name in overrides:
            mapping[field_name] = overrides[field_name]
        else:
            mapping[field_name] = _match_column(df.columns, keywords)
    return mapping


def _extract_temperature_c(folder_name: str) -> Optional[float]:
    m = re.search(r"(n)?(\d+)\s*deg", folder_name, flags=re.IGNORECASE)
    if not m:
        return None
    sign = -1.0 if m.group(1) else 1.0
    return sign * float(m.group(2))


def _find_header_row(path: str, max_scan: int = 100) -> Optional[int]:
    """Scan the first `max_scan` lines for the real header row, identified as
    the first line containing "voltage", "current", and some form of "time"
    all as separate CSV fields. Cycler exports typically have ~20-30 lines
    of "Key,Value" metadata before this row; a plain flat CSV has it on
    line 0."""
    with open(path, "r", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= max_scan:
                break
            low = line.lower()
            if "voltage" in low and "current" in low and "time" in low:
                return i
    return None


def _extract_metadata(path: str, max_scan: int = 40) -> Dict[str, str]:
    """Best-effort "Key,Value" metadata preamble scan (Measurement ID, Test
    section, ...). Returns {} for a plain flat CSV with no such preamble.

    Some files pad every row (including the metadata preamble) with
    trailing commas to match the widest row's column count, e.g.
    "Measurement ID,549,,,,,,,,,,,,". Taking `line.partition(",")`'s value
    as-is would keep that padding ("549,,,,,,,,,,,,"), which -- since the
    Measurement ID becomes the grouping key in `load_lg_hg2_dataframe` --
    would silently split a measurement's own files into a second, bogus
    group keyed by the padded string instead of merging them with the
    rest of that measurement. The value is therefore also truncated at
    its own next comma.
    """
    meta: Dict[str, str] = {}
    with open(path, "r", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= max_scan:
                break
            if "," in line:
                k, _, v = line.partition(",")
                k = k.strip()
                v = v.partition(",")[0].strip()
                if k and k not in meta:
                    meta[k] = v
    return meta


def _read_measurement_csv(path: str) -> pd.DataFrame:
    header_idx = _find_header_row(path)
    df = (
        pd.read_csv(path, low_memory=False)
        if header_idx is None
        else pd.read_csv(path, skiprows=header_idx, low_memory=False)
    )

    # Drop columns auto-named by pandas from a trailing comma in the header.
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]
    df.columns = [str(c).strip() for c in df.columns]

    # Drop a units row (e.g. ",,,,,,,,[V],[A],[C],[Ah],...") immediately
    # following the header, if present.
    if len(df) > 0:
        first_row_text = " ".join(str(v) for v in df.iloc[0].tolist())
        if "[" in first_row_text and "]" in first_row_text:
            df = df.iloc[1:].reset_index(drop=True)
    return df


def _matches_any(text: str, patterns: Sequence[str]) -> bool:
    low = text.lower()
    return any(p.lower() in low for p in patterns)


def discover_csv_files(root: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {root!r}.")
    return files


def _parse_datetime_column(raw: pd.Series) -> pd.Series:
    """Parse a time column to absolute pandas Timestamps. Handles a real
    "Time Stamp" datetime column (tried against known cycler-export
    formats first, then a generic fallback) and, for plain flat CSVs with
    an already-numeric elapsed-seconds column, synthesizes Timestamps
    anchored at the Unix epoch so downstream code can treat every dataset
    uniformly as absolute time.

    A column that is *already* numeric (float/int dtype, e.g. a plain
    elapsed-seconds column) is routed straight to the seconds-since-epoch
    fallback: pd.to_datetime on a bare numeric Series treats the numbers as
    nanoseconds since epoch rather than failing, which would otherwise look
    like a "successful" (but wrong, near-1970 and nanosecond-spaced) parse.
    """
    if not pd.api.types.is_numeric_dtype(raw):
        for fmt in _KNOWN_DATETIME_FORMATS:
            parsed = pd.to_datetime(raw, format=fmt, errors="coerce")
            if parsed.notna().mean() > 0.5:
                return parsed
        parsed = pd.to_datetime(raw, errors="coerce")
        if parsed.notna().mean() > 0.5:
            return parsed
    numeric = pd.to_numeric(raw, errors="coerce")
    return pd.to_datetime(numeric, unit="s", origin="unix", errors="coerce")


def _compute_soc_from_current(
    time_s: np.ndarray,
    current_a: np.ndarray,
    rated_capacity_ah: float,
    current_sign: float,
    soc_initial: float,
    max_gap_s: float = 300.0,
) -> np.ndarray:
    """Coulomb-count SoC from (time, current). Each step integrates
    current[i] * dt[i] with dt[i] = time[i] - time[i-1] (a right-Riemann
    sum), which is a fine approximation across normal, closely-spaced
    samples but catastrophic across a *missing-data* gap: it would apply
    the post-gap sample's current across the entire gap duration. dt is
    therefore capped at `max_gap_s` -- any gap beyond it contributes at
    most `max_gap_s` worth of charge, rather than being blindly
    extrapolated across time we have no current reading for.

    The running SoC is clipped to [0, 1] at *every* step, not once at the
    end. A real cell physically cannot exceed 100% SoC -- charging current
    that keeps flowing during CV tapering near full charge doesn't store
    energy beyond capacity. This matters for measurements that are
    repeated charge/discharge cycling runs (a "Charge_N" section fully
    recharging the cell, then a "Mixed_N" section discharging it, repeated
    many times) rather than one continuous depleting drive-cycle sweep:
    confirmed on a real sample where 5 repeated Charge/Mixed cycles each
    individually charge/discharge ~2.3-2.5 Ah (roughly 80% of the 3 Ah
    rated capacity) against each other. A single end-of-array clip lets
    the *unclipped* running sum drift arbitrarily far above 1.0 whenever
    several charge segments land close together in the reconstructed
    timeline before their matching discharge segments (order depends on
    each row's real timestamp, not the paper's "Charge_N pairs with
    Mixed_N" naming) -- e.g. repeated +0.8-SoC charges stacking up to +5.6
    before any clipping is ever applied, so that even five subsequent
    ~0.8-SoC discharges only bring the unclipped value down to +1.7,
    which still displays as a flat 1.0 for the entire span once clipped.
    Clipping at every step instead makes each charge segment correctly
    saturate at 1.0 (as a real cell does) before the next discharge
    segment starts, recovering the true sawtooth SoC pattern.
    """
    dt = np.diff(time_s, prepend=time_s[0])
    dt[0] = 0.0
    dt = np.clip(dt, 0.0, max_gap_s)
    delta_soc = current_sign * current_a * dt / 3600.0 / rated_capacity_ah

    soc = np.empty_like(delta_soc)
    running = float(soc_initial)
    for k in range(delta_soc.shape[0]):
        running += float(delta_soc[k])
        if running > 1.0:
            running = 1.0
        elif running < 0.0:
            running = 0.0
        soc[k] = running
    return soc


def _parse_file_raw(path: str, overrides: Optional[Dict[str, str]]):
    """Parse one CSV into a raw per-file frame (abs_time, voltage, current,
    temperature, soc_raw) plus its group/section identity. Returns None if
    the file is unusable."""
    try:
        df = _read_measurement_csv(path)
    except Exception as exc:  # pragma: no cover - defensive I/O guard
        print(f"[rtcqr.data] Skipping unreadable file {path}: {exc}")
        return None
    if df.empty:
        return None

    cols = detect_columns(df, overrides)
    missing = [k for k in ("time", "voltage", "current", "temperature") if cols[k] is None]
    if missing:
        print(f"[rtcqr.data] Skipping {path}: could not detect columns {missing}. "
              f"Detected mapping: {cols}. Pass column_overrides to fix.")
        return None

    abs_time = _parse_datetime_column(df[cols["time"]])
    voltage = pd.to_numeric(df[cols["voltage"]], errors="coerce")
    current = pd.to_numeric(df[cols["current"]], errors="coerce")
    temperature = pd.to_numeric(df[cols["temperature"]], errors="coerce")
    soc_raw = pd.to_numeric(df[cols["soc"]], errors="coerce") if cols["soc"] is not None else pd.Series(
        np.nan, index=df.index
    )

    valid = abs_time.notna() & voltage.notna() & current.notna() & temperature.notna()
    if valid.sum() < 5:
        print(f"[rtcqr.data] Skipping {path}: fewer than 5 valid rows after parsing.")
        return None

    frame = pd.DataFrame({
        "abs_time": abs_time[valid],
        "voltage": voltage[valid],
        "current": current[valid],
        "temperature": temperature[valid],
        "soc_raw": soc_raw[valid],
    }).reset_index(drop=True)

    meta = _extract_metadata(path)
    measurement_id = meta.get("Measurement ID")
    test_section = meta.get("Test section")
    group_key = measurement_id if measurement_id else path  # no metadata -> each file is its own group
    if not test_section:
        test_section = os.path.splitext(os.path.basename(path))[0]
    condition = _extract_temperature_c(os.path.basename(os.path.dirname(path)))
    return frame, group_key, test_section, condition


def _resample_uniform(df: pd.DataFrame, dt_s: float) -> pd.DataFrame:
    """Resample a (time, voltage, current, temperature, soc) frame with
    monotonic but possibly irregular `time` onto a uniform grid via linear
    interpolation."""
    t0, t1 = float(df["time"].iloc[0]), float(df["time"].iloc[-1])
    if t1 - t0 < dt_s or len(df) < 2:
        return df
    new_time = np.arange(t0, t1 + 1e-9, dt_s)
    old_time = df["time"].to_numpy()
    out = {"time": new_time}
    for col in ("voltage", "current", "temperature", "soc"):
        out[col] = np.interp(new_time, old_time, df[col].to_numpy())
    return pd.DataFrame(out)


def _split_contiguous(df: pd.DataFrame, keep_mask: np.ndarray, max_gap_s: float) -> List[pd.DataFrame]:
    """Split `df` into contiguous runs of kept rows, breaking wherever
    excluded rows sit in between or wherever the real-time gap between
    consecutive kept rows exceeds `max_gap_s`."""
    idx = np.where(keep_mask)[0]
    if len(idx) == 0:
        return []
    time_arr = df["time"].to_numpy()
    segments, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i != prev + 1 or (time_arr[i] - time_arr[prev]) > max_gap_s:
            segments.append(df.iloc[start:prev + 1])
            start = i
        prev = i
    segments.append(df.iloc[start:prev + 1])
    return segments


@dataclass
class BatteryFile:
    path: str  # descriptive source label (measurement id / section list), for logging
    condition: Optional[float]  # nominal test temperature in degC, if inferable
    frame: pd.DataFrame  # columns: time, voltage, current, temperature, soc (time-sorted, uniform rate)


def load_lg_hg2_dataframe(
    root: str,
    rated_capacity_ah: float = 3.0,
    current_sign: float = 1.0,
    soc_initial: float = 1.0,
    include_patterns: Optional[Sequence[str]] = _DEFAULT_INCLUDE_PATTERNS,
    resample_dt_s: Optional[float] = 1.0,
    max_gap_s: float = 300.0,
    column_overrides: Optional[Dict[str, Dict[str, str]]] = None,
) -> List[BatteryFile]:
    """Load and reconstruct every measurement run under `root` into a list
    of BatteryFile windowing segments. See the module docstring for the
    grouping/stitching/resampling procedure. Only test sections matching
    `include_patterns` are kept for windowing (pass None to keep all).

    `column_overrides` maps a filename (basename) to a per-field column-name
    override dict, for files where auto-detection needs help, e.g.:
        {"553_Mixed2.csv": {"current": "Current(A)"}}
    """
    column_overrides = column_overrides or {}
    paths = discover_csv_files(root)

    groups: Dict[str, List[dict]] = defaultdict(list)
    for path in paths:
        parsed = _parse_file_raw(path, column_overrides.get(os.path.basename(path)))
        if parsed is None:
            continue
        frame, group_key, test_section, condition = parsed
        frame["test_section"] = test_section
        groups[group_key].append({"frame": frame, "condition": condition, "path": path})

    out: List[BatteryFile] = []
    for group_key, parts in groups.items():
        combined = pd.concat([p["frame"] for p in parts], ignore_index=True)
        combined = combined.sort_values("abs_time").drop_duplicates(subset=["abs_time"], keep="first")
        combined = combined.reset_index(drop=True)
        if len(combined) < 20:
            continue

        combined["time"] = (combined["abs_time"] - combined["abs_time"].iloc[0]).dt.total_seconds()

        gaps = combined["time"].diff().to_numpy()
        big_gaps = gaps[gaps > max_gap_s]
        section_list = ", ".join(sorted({p["frame"]["test_section"].iloc[0] for p in parts}))
        if len(big_gaps) > 0:
            print(f"[rtcqr.data] measurement {group_key} ({section_list}): {len(big_gaps)} gap(s) > "
                  f"{max_gap_s:.0f}s (largest {np.nanmax(big_gaps):.0f}s) -- likely missing intermediate "
                  f"test-section files; SoC will not account for current drawn during the gap(s), and each "
                  f"gap becomes a windowing segment boundary.")

        if combined["soc_raw"].notna().mean() > 0.5:
            soc = combined["soc_raw"].to_numpy()
            if np.nanmax(soc) > 1.5:  # looks like a percentage
                soc = soc / 100.0
            soc = np.clip(soc, 0.0, 1.0)
            if np.isnan(soc).any():
                soc = pd.Series(soc).interpolate(limit_direction="both").to_numpy()
        else:
            soc = _compute_soc_from_current(
                combined["time"].to_numpy(), combined["current"].to_numpy(),
                rated_capacity_ah, current_sign, soc_initial, max_gap_s=max_gap_s,
            )
        combined["soc"] = soc

        condition = next((p["condition"] for p in parts if p["condition"] is not None), None)

        if include_patterns:
            keep_mask = combined["test_section"].apply(lambda s: _matches_any(s, include_patterns)).to_numpy()
        else:
            keep_mask = np.ones(len(combined), dtype=bool)
        if not keep_mask.any():
            continue

        for seg in _split_contiguous(combined, keep_mask, max_gap_s):
            seg = seg[["time", "voltage", "current", "temperature", "soc"]].reset_index(drop=True)
            if resample_dt_s:
                seg = _resample_uniform(seg, resample_dt_s)
            if len(seg) < 20:
                continue
            out.append(BatteryFile(path=f"measurement {group_key} ({section_list})", condition=condition, frame=seg))

    if not out:
        raise RuntimeError(
            f"No usable battery segments parsed under {root!r}. Run "
            "`python -m rtcqr.data inspect <root>` to see per-file/per-group detection."
        )
    return out


def chronological_split(
    files: Sequence[BatteryFile], train_frac: float, val_frac: float
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]:
    """Split each segment's time series into train/val/test slices in time order.

    Splitting within each segment (rather than assigning whole segments to
    a split) keeps every drive cycle and every temperature condition
    represented in all three splits while still respecting chronological
    order, matching the paper's use of a fixed train/val/test partition
    with the test portion held out purely for final evaluation.
    """
    train_parts, val_parts, test_parts = [], [], []
    for bf in files:
        n = len(bf.frame)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        df = bf.frame
        train_parts.append(df.iloc[:n_train].reset_index(drop=True))
        val_parts.append(df.iloc[n_train:n_train + n_val].reset_index(drop=True))
        test_parts.append(df.iloc[n_train + n_val:].reset_index(drop=True))
    return train_parts, val_parts, test_parts


def make_windows(
    frames: Sequence[pd.DataFrame], window_size: int, stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Build sliding windows of (voltage, current, temperature) -> SoC at the window's last step.

    Returns X of shape (N, 3, window_size) and y of shape (N,).
    """
    xs, ys = [], []
    for df in frames:
        if len(df) < window_size:
            continue
        v = df["voltage"].to_numpy(dtype=np.float32)
        i = df["current"].to_numpy(dtype=np.float32)
        t = df["temperature"].to_numpy(dtype=np.float32)
        soc = df["soc"].to_numpy(dtype=np.float32)
        n = len(df)
        for end in range(window_size - 1, n, stride):
            start = end - window_size + 1
            xs.append(np.stack([v[start:end + 1], i[start:end + 1], t[start:end + 1]], axis=0))
            ys.append(soc[end])
    if not xs:
        raise RuntimeError("No windows could be built; window_size may exceed the shortest split length.")
    return np.stack(xs, axis=0), np.asarray(ys, dtype=np.float32)


class Standardizer:
    """Per-channel z-score standardization fit on the training windows only."""

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "Standardizer":
        self.mean_ = x.mean(axis=(0, 2), keepdims=True)
        self.std_ = x.std(axis=(0, 2), keepdims=True)
        self.std_[self.std_ < 1e-6] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.std_


def _inspect(root: str, include_all: bool) -> None:
    include_patterns = None if include_all else _DEFAULT_INCLUDE_PATTERNS
    files = load_lg_hg2_dataframe(root, include_patterns=include_patterns)
    print(f"\n{len(files)} windowing segment(s) after grouping/stitching/splitting:\n")
    for bf in files:
        df = bf.frame
        span_h = (df["time"].iloc[-1] - df["time"].iloc[0]) / 3600.0
        print(f"{bf.path}")
        print(f"  condition={bf.condition}  rows={len(df)}  span_h={span_h:.2f}  "
              f"voltage=[{df.voltage.min():.3f},{df.voltage.max():.3f}]  "
              f"current=[{df.current.min():.3f},{df.current.max():.3f}]  "
              f"soc=[{df.soc.min():.3f},{df.soc.max():.3f}] (start={df.soc.iloc[0]:.3f}, end={df.soc.iloc[-1]:.3f})")


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3 and sys.argv[1] == "inspect":
        include_all = "--include-all" in sys.argv[3:]
        _inspect(sys.argv[2], include_all=include_all)
    else:
        print("Usage: python -m rtcqr.data inspect <dataset_root> [--include-all]")
