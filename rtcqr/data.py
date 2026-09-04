"""Data loading and windowing for the LG 18650HG2 Li-ion battery dataset.

The Kaggle mirror (aditya9790/lg-18650hg2-liion-battery-data) republishes
Kollmeyer et al.'s LG HG2 drive-cycle test logs as one CSV per test file,
grouped into folders by test temperature (e.g. "0degC", "25degC", "n10degC").
Column names differ slightly across re-exports of this dataset, so this
module auto-detects voltage/current/temperature/time/SoC columns by keyword
instead of hard-coding a schema. If auto-detection ever picks the wrong
column for your copy of the dataset, run `python -m rtcqr.data inspect
<path>` to print the detected mapping per file and override it with the
`column_overrides` argument.

State of charge:
    If the file already has a SoC-like column, it is used directly
    (rescaled to [0, 1] if it looks like a percentage). Otherwise SoC is
    obtained by coulomb counting against the cell's rated capacity (3.0 Ah
    for the LG HG2), starting from a full charge at the beginning of each
    test file:

        SoC(t) = SoC(0) + (1 / Q_rated) * cumsum(I(t) * dt)

    using the charge-positive current sign convention (I > 0 = charging).
    Set `current_sign=-1` in `load_lg_hg2_dataframe` if a given file uses
    the opposite convention (I > 0 = discharging) -- this is easy to check
    since the drive-cycle current should be predominantly negative and SoC
    should trend downward over a discharge test when the convention is
    right.
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


def discover_csv_files(root: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {root!r}.")
    return files


def _compute_soc(
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


@dataclass
class BatteryFile:
    path: str
    condition: Optional[float]  # nominal test temperature in degC, if inferable
    frame: pd.DataFrame  # columns: time, voltage, current, temperature, soc (time-sorted)


def load_lg_hg2_dataframe(
    root: str,
    rated_capacity_ah: float = 3.0,
    current_sign: float = 1.0,
    column_overrides: Optional[Dict[str, Dict[str, str]]] = None,
) -> List[BatteryFile]:
    """Load every drive-cycle CSV under `root` into a list of BatteryFile.

    `column_overrides` maps a filename (basename) to a per-field column-name
    override dict, for files where auto-detection needs help, e.g.:
        {"553_Mixed2.csv": {"current": "Current(A)"}}
    """
    column_overrides = column_overrides or {}
    files = discover_csv_files(root)
    out: List[BatteryFile] = []
    for path in files:
        try:
            df = pd.read_csv(path)
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

        time_s = pd.to_numeric(df[cols["time"]], errors="coerce").to_numpy(dtype=float)
        voltage = pd.to_numeric(df[cols["voltage"]], errors="coerce").to_numpy(dtype=float)
        current = pd.to_numeric(df[cols["current"]], errors="coerce").to_numpy(dtype=float)
        temperature = pd.to_numeric(df[cols["temperature"]], errors="coerce").to_numpy(dtype=float)

        valid = np.isfinite(time_s) & np.isfinite(voltage) & np.isfinite(current) & np.isfinite(temperature)
        time_s, voltage, current, temperature = (a[valid] for a in (time_s, voltage, current, temperature))
        if len(time_s) < 10:
            continue

        order = np.argsort(time_s, kind="stable")
        time_s, voltage, current, temperature = (a[order] for a in (time_s, voltage, current, temperature))

        if cols["soc"] is not None:
            soc = pd.to_numeric(df[cols["soc"]], errors="coerce").to_numpy(dtype=float)[valid][order]
            if np.nanmax(soc) > 1.5:  # looks like a percentage
                soc = soc / 100.0
            soc = np.clip(soc, 0.0, 1.0)
            if np.isnan(soc).any():
                soc = pd.Series(soc).interpolate(limit_direction="both").to_numpy()
        else:
            soc = _compute_soc(time_s, current, rated_capacity_ah, current_sign, soc_initial=1.0)

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


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3 and sys.argv[1] == "inspect":
        root = sys.argv[2]
        for path in discover_csv_files(root):
            df = pd.read_csv(path, nrows=5)
            print(path)
            print("  columns:", list(df.columns))
            print("  detected:", detect_columns(df))
    else:
        print("Usage: python -m rtcqr.data inspect <dataset_root>")
