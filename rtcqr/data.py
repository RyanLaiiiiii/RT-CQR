"""Data loading and windowing for the LG 18650HG2 Li-ion battery dataset.

Real exports of this dataset (as confirmed from a sample file,
`585_C20DisCh.csv`) are battery-cycler exports, not plain flat CSVs:

    Measurement ID,585
    Battery Name,LG HG2 18650_SN62A4
    ...
    Nominal Capacity, 3
    ...
    Time Stamp,Step,Status,Prog Time,Step Time,Cycle,Cycle Level,Procedure,Voltage,Current,Temperature,Capacity,WhAccu,Cnt,
    ,,,,,,,,[V],[A],[C],[Ah],[Wh],[Cnt],
    11/27/2018 8:41:18 PM,22,DCH,25:19:08.386,00:01:00.004,0,0,LG_HG2_NN_Char,4.16273,-0.15325,-0.42063,-0.00253,-0.01052,13.00000,

i.e. ~20-30 metadata lines, then the real header row, then a units row,
then the data. Each file is one measurement/test section (filenames look
like "<id>_<TestSection>.csv", e.g. "585_C20DisCh.csv" for a C/20 discharge
capacity-characterization run, or "596_LA92.csv" for a dynamic drive-cycle
profile). `_read_measurement_csv` locates and skips the preamble/units rows
automatically; `discover_csv_files` can additionally filter out
non-drive-cycle characterization sections (C/20 charge-discharge, OCV,
HPPC, ...) that aren't representative of the dynamic operating conditions
the paper evaluates on.

Column names can still differ slightly across re-exports, so voltage /
current / temperature / time / capacity / SoC columns are auto-detected by
keyword rather than hard-coded. Run `python -m rtcqr.data inspect <path>`
to print the detected mapping and a parsed summary (row count, time span,
SoC range) for every file, and pass `column_overrides` to
`load_lg_hg2_dataframe` to fix any file where detection guesses wrong.

State of charge:
    1. If the file already has a SoC-like column, it is used directly
       (rescaled to [0, 1] if it looks like a percentage).
    2. Else, if it has a cumulative Capacity[Ah] column (as above), SoC is
       derived from it: SoC(t) = SoC(0) + current_sign * Capacity(t) / Q_rated.
       This is preferred over re-integrating current with a coarse or
       irregular timestamp, since the cycler's own coulomb counter is used.
    3. Else SoC is obtained by coulomb counting the current signal:
           SoC(t) = SoC(0) + (1 / Q_rated) * cumsum(I(t) * dt)

    All three assume SoC(0) = 1.0, i.e. that each exported test-section
    file begins right after a full charge -- the standard protocol for
    these drive-cycle/characterization benchmark logs. Pass a different
    `soc_initial` to `load_lg_hg2_dataframe` if that does not hold for your
    files.

    The default `current_sign=1.0` assumes I > 0 = charging (matches the
    sample file: current and Capacity are both negative during the DCH
    section, so SoC decreases as expected). Set `current_sign=-1` if a
    different export uses the opposite convention -- check with
    `python -m rtcqr.data inspect`, which prints the SoC range and trend
    per file.
"""
from __future__ import annotations

import glob
import os
import re
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

# Filename substrings (case-insensitive) identifying static characterization
# test sections rather than dynamic drive-cycle profiles. These have very
# different current dynamics (slow constant-current or pulse tests) from
# the driving conditions the paper models, so they are excluded by default.
_DEFAULT_EXCLUDE_PATTERNS = ["c20", "c/20", "ocv", "hppc", "pulse", "eis", "reset"]


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


def _read_measurement_csv(path: str) -> pd.DataFrame:
    header_idx = _find_header_row(path)
    df = pd.read_csv(path) if header_idx is None else pd.read_csv(path, skiprows=header_idx)

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


def discover_csv_files(
    root: str,
    exclude_patterns: Optional[Sequence[str]] = _DEFAULT_EXCLUDE_PATTERNS,
) -> List[str]:
    """List CSV files under `root`, optionally filtering out filenames that
    match `exclude_patterns` (case-insensitive substrings). Pass
    `exclude_patterns=None` or `[]` to keep every file, e.g. to include
    static characterization tests alongside dynamic drive cycles."""
    files = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {root!r}.")
    if exclude_patterns:
        kept = [f for f in files if not _matches_any(os.path.basename(f), exclude_patterns)]
        skipped = len(files) - len(kept)
        if skipped:
            print(f"[rtcqr.data] Excluded {skipped} non-drive-cycle file(s) matching {list(exclude_patterns)} "
                  f"(pass exclude_patterns=None to include everything).")
        files = kept
        if not files:
            raise FileNotFoundError(
                f"All CSV files under {root!r} were excluded by exclude_patterns={list(exclude_patterns)}."
            )
    return files


_KNOWN_DATETIME_FORMATS = ["%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S"]


def _parse_time_seconds(raw: pd.Series) -> np.ndarray:
    """Convert a time column to elapsed seconds from the first sample.
    Handles both an absolute "Time Stamp" datetime column and an
    already-numeric seconds column."""
    dt = None
    for fmt in _KNOWN_DATETIME_FORMATS:
        parsed = pd.to_datetime(raw, format=fmt, errors="coerce")
        if parsed.notna().mean() > 0.5:
            dt = parsed
            break
    if dt is None:
        dt = pd.to_datetime(raw, errors="coerce")
    if dt.notna().mean() > 0.5:
        first_valid = dt.dropna().iloc[0]
        return (dt - first_valid).dt.total_seconds().to_numpy(dtype=float)
    return pd.to_numeric(raw, errors="coerce").to_numpy(dtype=float)


def _compute_soc_from_current(
    time_s: np.ndarray,
    current_a: np.ndarray,
    rated_capacity_ah: float,
    current_sign: float,
    soc_initial: float,
) -> np.ndarray:
    dt = np.diff(time_s, prepend=time_s[0])
    dt[0] = 0.0
    dt = np.clip(dt, 0.0, None)  # guard against non-monotonic timestamps
    delta_ah = current_sign * current_a * dt / 3600.0
    soc = soc_initial + np.cumsum(delta_ah) / rated_capacity_ah
    return np.clip(soc, 0.0, 1.0)


def _compute_soc_from_capacity(
    capacity_ah: np.ndarray, rated_capacity_ah: float, current_sign: float, soc_initial: float
) -> np.ndarray:
    capacity_ah = pd.Series(capacity_ah).interpolate(limit_direction="both").to_numpy()
    soc = soc_initial + current_sign * capacity_ah / rated_capacity_ah
    return np.clip(soc, 0.0, 1.0)


@dataclass
class BatteryFile:
    path: str
    condition: Optional[float]  # nominal test temperature in degC, if inferable
    frame: pd.DataFrame  # columns: time, voltage, current, temperature, soc (time-sorted)


def load_lg_hg2_dataframe(
    root: str,
    rated_capacity_ah: float = 3.0,
    current_sign: float = 1.0,
    soc_initial: float = 1.0,
    exclude_patterns: Optional[Sequence[str]] = _DEFAULT_EXCLUDE_PATTERNS,
    column_overrides: Optional[Dict[str, Dict[str, str]]] = None,
) -> List[BatteryFile]:
    """Load every (non-excluded) CSV under `root` into a list of BatteryFile.

    `column_overrides` maps a filename (basename) to a per-field column-name
    override dict, for files where auto-detection needs help, e.g.:
        {"553_Mixed2.csv": {"current": "Current(A)"}}
    """
    column_overrides = column_overrides or {}
    files = discover_csv_files(root, exclude_patterns=exclude_patterns)
    out: List[BatteryFile] = []
    for path in files:
        try:
            df = _read_measurement_csv(path)
        except Exception as exc:  # pragma: no cover - defensive I/O guard
            print(f"[rtcqr.data] Skipping unreadable file {path}: {exc}")
            continue
        if df.empty:
            continue

        overrides = column_overrides.get(os.path.basename(path))
        cols = detect_columns(df, overrides)
        missing = [k for k in ("time", "voltage", "current", "temperature") if cols[k] is None]
        if missing:
            print(f"[rtcqr.data] Skipping {path}: could not detect columns {missing}. "
                  f"Detected mapping: {cols}. Pass column_overrides to fix.")
            continue

        time_s = _parse_time_seconds(df[cols["time"]])
        voltage = pd.to_numeric(df[cols["voltage"]], errors="coerce").to_numpy(dtype=float)
        current = pd.to_numeric(df[cols["current"]], errors="coerce").to_numpy(dtype=float)
        temperature = pd.to_numeric(df[cols["temperature"]], errors="coerce").to_numpy(dtype=float)
        capacity = (
            pd.to_numeric(df[cols["capacity"]], errors="coerce").to_numpy(dtype=float)
            if cols["capacity"] is not None else None
        )
        soc_col = (
            pd.to_numeric(df[cols["soc"]], errors="coerce").to_numpy(dtype=float)
            if cols["soc"] is not None else None
        )

        valid = np.isfinite(time_s) & np.isfinite(voltage) & np.isfinite(current) & np.isfinite(temperature)
        if valid.sum() < 10:
            print(f"[rtcqr.data] Skipping {path}: fewer than 10 valid rows after parsing.")
            continue

        time_s, voltage, current, temperature = (a[valid] for a in (time_s, voltage, current, temperature))
        if capacity is not None:
            capacity = capacity[valid]
        if soc_col is not None:
            soc_col = soc_col[valid]

        order = np.argsort(time_s, kind="stable")
        time_s, voltage, current, temperature = (a[order] for a in (time_s, voltage, current, temperature))
        if capacity is not None:
            capacity = capacity[order]
        if soc_col is not None:
            soc_col = soc_col[order]

        if soc_col is not None and np.isfinite(soc_col).mean() > 0.5:
            soc = soc_col
            if np.nanmax(soc) > 1.5:  # looks like a percentage
                soc = soc / 100.0
            soc = np.clip(soc, 0.0, 1.0)
            if np.isnan(soc).any():
                soc = pd.Series(soc).interpolate(limit_direction="both").to_numpy()
        elif capacity is not None and np.isfinite(capacity).mean() > 0.5:
            soc = _compute_soc_from_capacity(capacity, rated_capacity_ah, current_sign, soc_initial)
        else:
            soc = _compute_soc_from_current(time_s, current, rated_capacity_ah, current_sign, soc_initial)

        condition = _extract_temperature_c(os.path.basename(os.path.dirname(path)))
        frame = pd.DataFrame(
            {"time": time_s, "voltage": voltage, "current": current, "temperature": temperature, "soc": soc}
        )
        out.append(BatteryFile(path=path, condition=condition, frame=frame))

    if not out:
        raise RuntimeError(
            f"No usable battery files parsed under {root!r}. Run "
            "`python -m rtcqr.data inspect <root>` to see per-file column detection."
        )
    return out


def chronological_split(
    files: Sequence[BatteryFile], train_frac: float, val_frac: float
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]:
    """Split each file's time series into train/val/test slices in time order.

    Splitting within each file (rather than assigning whole files to a
    split) keeps every drive cycle and every temperature condition
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
    exclude = None if include_all else _DEFAULT_EXCLUDE_PATTERNS
    files = discover_csv_files(root, exclude_patterns=exclude)
    for path in files:
        print(path)
        try:
            df = _read_measurement_csv(path)
            cols = detect_columns(df)
            print("  columns:", list(df.columns))
            print("  detected:", cols)
            missing = [k for k in ("time", "voltage", "current", "temperature") if cols[k] is None]
            if missing:
                print(f"  WARNING: missing {missing}, this file will be skipped by load_lg_hg2_dataframe")
                continue
            time_s = _parse_time_seconds(df[cols["time"]])
            voltage = pd.to_numeric(df[cols["voltage"]], errors="coerce")
            current = pd.to_numeric(df[cols["current"]], errors="coerce")
            valid = np.isfinite(time_s) & voltage.notna().to_numpy() & current.notna().to_numpy()
            n_valid = int(valid.sum())
            span_h = (np.nanmax(time_s[valid]) - np.nanmin(time_s[valid])) / 3600.0 if n_valid else float("nan")
            print(f"  rows={len(df)} valid={n_valid} time_span_h={span_h:.2f} "
                  f"voltage=[{voltage.min():.3f},{voltage.max():.3f}] current=[{current.min():.3f},{current.max():.3f}]")
        except Exception as exc:  # pragma: no cover - inspection convenience only
            print(f"  ERROR parsing file: {exc}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3 and sys.argv[1] == "inspect":
        include_all = "--include-all" in sys.argv[3:]
        _inspect(sys.argv[2], include_all=include_all)
    else:
        print("Usage: python -m rtcqr.data inspect <dataset_root> [--include-all]")
