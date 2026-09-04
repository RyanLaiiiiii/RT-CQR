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
     The coulomb count is normalized against that measurement's ambient
     *condition*'s actual measured capacity (integrated from whichever
     measurement group at that condition has a `Cap_1C` capacity-check
     section -- see `_measured_capacity_ah`), not a single fixed
     `rated_capacity_ah` across all temperatures. The LG HG2's usable
     capacity drops substantially in the cold (per the datasheet's
     capacity-vs-temperature curve), so a fixed nominal denominator makes
     the reconstructed SoC floor rise well above 0 at every temperature
     except the warmest, even though the test protocol depletes the same
     ~95% of *actual* capacity at every condition.
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
from dataclasses import dataclass, replace
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

# Test-section name identifying the 1C capacity-check discharge (a
# standalone full discharge at 1C used to measure the cell's *actual*
# capacity at that test's ambient temperature). See `_measured_capacity_ah`
# and its use in `load_lg_hg2_dataframe` for why this matters: the cell's
# usable capacity drops substantially at low temperature (per the LG HG2
# datasheet's discharge-capacity-vs-temperature curve), so normalizing SoC
# against a single fixed nominal capacity across all temperatures
# understates depth of discharge at every temperature except the warmest.
_CAPACITY_CHECK_PATTERN = "cap_1c"

# Plausibility guards for a measured capacity (see `_measured_capacity_ah`).
# A capacity check that fails all of these is discarded in favour of the
# nominal rating, because an under-measured denominator is far more damaging
# than a slightly-wrong one: it drives reconstructed SoC into the [0, 1]
# clip, freezing whole drive-cycle segments at exactly 0.0 and destroying the
# coulomb count for everything after them in the same run.
_MIN_CAPACITY_AH = 0.1
_MIN_CAPACITY_SECTION_ROWS = 5
# A 1C discharge empties the cell in ~1 h *by definition of 1C*, so the
# section's duration is a capacity-independent truncation detector: it is the
# check that catches a section the export cut short, whose integrated Ah is
# otherwise perfectly plausible-looking. The window is wide because the
# cycler sets the current from the *nominal* rating, so a cold cell holding
# less charge finishes sooner (1.64 Ah at 3.0 A = 0.55 h at -20 degC).
_DURATION_PLAUSIBLE_H = (0.4, 2.5)
# True C-rate = mean discharge current / rated capacity. Only checkable when
# the nominal rating is known; catches a section discharged at a rate that
# isn't a capacity check at all (a drive cycle mislabelled, say).
_C_RATE_PLAUSIBLE_RANGE = (0.4, 2.0)
_CAPACITY_PLAUSIBLE_RANGE = (0.3, 1.1)  # multiples of rated_capacity_ah

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


# Sub-zero ambient folders are named with an "n" prefix in the McMaster
# release ("n10degC"), but re-exports and mirrors use "-10degC" / "neg10degC"
# / "minus10degC". All four must map to -10.0: `condition` is the key the
# per-condition capacity denominator is looked up under, so silently reading
# "-10degC" as +10.0 would merge the -10 degC and +10 degC conditions and
# apply one averaged capacity to both.
_NEGATIVE_TEMP_PREFIX = re.compile(r"(?:^|[^a-z0-9])(n|neg|minus|-)\s*$", re.IGNORECASE)


def _extract_temperature_c(folder_name: str) -> Optional[float]:
    m = re.search(r"(\d+)\s*deg", folder_name, flags=re.IGNORECASE)
    if not m:
        return None
    sign = -1.0 if _NEGATIVE_TEMP_PREFIX.search(folder_name[:m.start(1)]) else 1.0
    return sign * float(m.group(1))


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


def _integrate_discharge_ah(
    time_s: np.ndarray, current_a: np.ndarray, current_sign: float, max_gap_s: float,
) -> Tuple[float, float]:
    """Integrate the discharging direction of one *contiguous* current trace.

    Returns (discharged_ah, duration_s). Only the discharging direction is
    summed (`-current_sign * current_a`, clipped to >=0) so that brief
    charge/rest padding inside the section doesn't partially cancel out the
    discharge capacity.
    """
    if len(time_s) < 2:
        return 0.0, 0.0
    dt = np.diff(time_s, prepend=time_s[0])
    dt[0] = 0.0
    dt = np.clip(dt, 0.0, max_gap_s)
    discharge_a = np.clip(-current_sign * current_a, 0.0, None)
    return float(np.sum(discharge_a * dt) / 3600.0), float(time_s[-1] - time_s[0])


def _measured_capacity_ah(
    time_s: np.ndarray,
    current_a: np.ndarray,
    current_sign: float,
    max_gap_s: float = 300.0,
    rated_capacity_ah: Optional[float] = None,
    row_index: Optional[np.ndarray] = None,
    label: str = "",
) -> Optional[float]:
    """Measure a cell's actual capacity from its `Cap_1C` capacity-check
    section(s), for use as the SoC coulomb-counting denominator in place of
    the fixed nominal rating (see `_CAPACITY_CHECK_PATTERN`).

    The rows handed in are selected by a boolean mask over the stitched run,
    and that selection is **not necessarily contiguous**: a measurement can
    contain more than one capacity check (`Cap_1C` and `Cap_1C_2` both match
    the substring pattern), and the mask then jumps across hours of
    intervening drive-cycle/charge sections. Integrating such a selection as
    if it were one trace both sums the separate checks into one impossibly
    large capacity *and* charges up to `max_gap_s` worth of phantom current
    at every discontinuity. `row_index` (the positions the mask selected)
    is therefore used to split the selection back into contiguous runs, each
    integrated on its own; the **median** run is returned, so a truncated or
    corrupted check next to a good one does not drag the answer.

    Every candidate run is validated before it can be returned:

      * it must last at least `_MIN_CAPACITY_SECTION_S` and yield more than
        `_MIN_CAPACITY_AH`, and
      * its implied C-rate (`ah / hours`) must look like a ~1C discharge --
        a *truncated* check (the cycler export cut short, or an intermediate
        file missing from your copy of the dataset) integrates to a
        plausible-looking-but-far-too-small number with nothing in the value
        itself to give it away, and silently under-sizing the denominator
        drives the reconstructed SoC hard into the [0, 1] clip, freezing
        whole drive-cycle segments at exactly 0.0, and
      * when `rated_capacity_ah` is known, it must land within
        `_CAPACITY_PLAUSIBLE_RANGE` of it -- the LG HG2 loses a lot of
        capacity in the cold, but not 10x, and it never *exceeds* its rating.

    Returns None (with a warning naming `label`) if nothing survives, so the
    caller falls back to the nominal rating rather than trusting a bad
    measurement.
    """
    if len(time_s) < _MIN_CAPACITY_SECTION_ROWS:
        return None
    if row_index is None:
        row_index = np.arange(len(time_s))

    # Split the (possibly non-contiguous) selection into contiguous runs.
    breaks = np.where(np.diff(row_index) != 1)[0] + 1
    candidates: List[float] = []
    rejected: List[str] = []
    for chunk in np.split(np.arange(len(row_index)), breaks):
        if len(chunk) < _MIN_CAPACITY_SECTION_ROWS:
            continue
        ah, duration_s = _integrate_discharge_ah(
            time_s[chunk], current_a[chunk], current_sign, max_gap_s
        )
        hours = duration_s / 3600.0
        if ah <= _MIN_CAPACITY_AH:
            rejected.append(f"{ah:.3f} Ah over {hours:.2f} h (near-zero net discharge)")
            continue
        if not (_DURATION_PLAUSIBLE_H[0] <= hours <= _DURATION_PLAUSIBLE_H[1]):
            rejected.append(f"{ah:.3f} Ah over {hours:.2f} h (a 1C check runs ~1 h, expected "
                            f"{_DURATION_PLAUSIBLE_H[0]}-{_DURATION_PLAUSIBLE_H[1]} h -- "
                            f"likely truncated or not a capacity check)")
            continue
        if rated_capacity_ah is not None:
            c_rate = (ah / max(hours, 1e-9)) / rated_capacity_ah
            if not (_C_RATE_PLAUSIBLE_RANGE[0] <= c_rate <= _C_RATE_PLAUSIBLE_RANGE[1]):
                rejected.append(f"{ah:.3f} Ah over {hours:.2f} h (mean {c_rate:.2f}C, not a ~1C check)")
                continue
            lo, hi = _CAPACITY_PLAUSIBLE_RANGE
            if not (lo * rated_capacity_ah <= ah <= hi * rated_capacity_ah):
                rejected.append(f"{ah:.3f} Ah ({ah / rated_capacity_ah:.2f}x rated, outside "
                                f"[{lo}, {hi}]x)")
                continue
        candidates.append(ah)

    if not candidates:
        if rejected:
            print(f"[rtcqr.data] {label}: no usable Cap_1C section -- rejected "
                  f"{len(rejected)} candidate(s): {'; '.join(rejected)}")
        return None
    if len(candidates) > 1:
        print(f"[rtcqr.data] {label}: {len(candidates)} separate Cap_1C section(s) "
              f"({[f'{c:.3f}' for c in candidates]}) -- using the median, not the sum.")
    return float(np.median(candidates))


def _warn_non_monotonic_capacity(capacity_by_condition: Dict[float, float]) -> None:
    """Flag a measured capacity that falls as ambient temperature *rises*.

    A Li-ion cell's usable capacity increases monotonically with temperature
    across this dataset's range (roughly -20 to +40 degC) -- that is the whole
    premise of measuring the denominator per condition. A reading that goes
    the other way (e.g. 40 degC measuring *lower* than 25 degC) is therefore
    not physics, it is a bad capacity check: a truncated section, a missing
    intermediate file, or a sampling gap inside the check. Left unflagged it
    silently under-sizes that condition's denominator and freezes its
    drive-cycle segments at SoC=0.
    """
    conds = sorted(c for c in capacity_by_condition if c is not None)
    for lo, hi in zip(conds, conds[1:]):
        cap_lo, cap_hi = capacity_by_condition[lo], capacity_by_condition[hi]
        if cap_hi < cap_lo * 0.98:
            print(f"[rtcqr.data] WARNING: measured capacity DROPS with rising temperature "
                  f"({lo} degC -> {cap_lo:.3f} Ah, {hi} degC -> {cap_hi:.3f} Ah). A cell's "
                  f"capacity does not fall as it warms, so the {hi} degC Cap_1C section is "
                  f"probably truncated or incomplete. Its SoC will bottom out at 0 too "
                  f"early; consider --exclude-measurement-ids for that group, or supplying "
                  f"the capacity manually via capacity_overrides.")


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


def _collapse_duplicate_timestamps(combined: pd.DataFrame) -> pd.DataFrame:
    """Sort a stitched run by time and collapse rows sharing a timestamp by
    *averaging* them, rather than keeping one and discarding the rest.

    The cycler's "Time Stamp" column has one-second resolution (the
    sub-second field lives in "Prog Time"/"Step Time", which are durations,
    not absolute times, so they cannot order rows across files). The dynamic
    drive-cycle sections are logged at ~0.1 s, so ~10 rows per second share a
    timestamp. Dropping nine of them keeps one *instantaneous* sample and
    then treats it as representative of the whole second -- decimation with
    no anti-aliasing, which both throws away 90% of the V/I/T dynamics the
    model is supposed to learn from and injects noise into the coulomb count,
    since `current[i] * 1s` is then one instantaneous reading rather than
    that second's mean current. Averaging keeps the charge integral right and
    hands the model a properly band-limited 1 Hz signal.
    """
    combined = combined.sort_values("abs_time", kind="stable")
    if not combined["abs_time"].duplicated().any():
        return combined.reset_index(drop=True)
    numeric = ["voltage", "current", "temperature", "soc_raw"]
    agg = {c: "mean" for c in numeric}
    agg["test_section"] = "first"  # a shared second straddling two sections: attribute it to the earlier
    collapsed = combined.groupby("abs_time", as_index=False, sort=True).agg(agg)
    return collapsed.reset_index(drop=True)


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
    # Absolute wall-clock start of this segment. Kept because the conformal
    # calibrator's exponential time decay (conformal.time_decay_weights)
    # assumes its buffer is in chronological order; segments are otherwise
    # emitted grouped by measurement and then randomly permuted by
    # `segment_split`, which would make "recency" meaningless.
    start_time: Optional[pd.Timestamp] = None
    group_key: Optional[str] = None  # Measurement ID this segment came from


def load_lg_hg2_dataframe(
    root: str,
    rated_capacity_ah: float = 3.0,
    current_sign: float = 1.0,
    soc_initial: float = 1.0,
    include_patterns: Optional[Sequence[str]] = _DEFAULT_INCLUDE_PATTERNS,
    resample_dt_s: Optional[float] = 1.0,
    max_gap_s: float = 300.0,
    column_overrides: Optional[Dict[str, Dict[str, str]]] = None,
    capacity_overrides: Optional[Dict[float, float]] = None,
    min_soc_range: float = 0.0,
) -> List[BatteryFile]:
    """Load and reconstruct every measurement run under `root` into a list
    of BatteryFile windowing segments. See the module docstring for the
    grouping/stitching/resampling procedure. Only test sections matching
    `include_patterns` are kept for windowing (pass None to keep all).

    SoC coulomb-counting is normalized per ambient-temperature `condition`
    against that condition's *actual* measured capacity (from whichever
    measurement group at that condition includes a `Cap_1C` capacity-check
    section), not the fixed `rated_capacity_ah`. The LG HG2's usable
    capacity drops substantially at low temperature (per the datasheet's
    capacity-vs-temperature curve), so normalizing every temperature against
    one fixed nominal value understates depth of discharge everywhere except
    the warmest condition -- confirmed on this dataset: the reconstructed
    SoC floor scaled almost exactly with the datasheet's per-temperature 1C
    capacity when using a fixed 3.0 Ah denominator (e.g. floor ~0.45-0.49 at
    -20 degC vs ~0.12-0.15 at 25 degC), even though the test protocol
    targets the same "95% of capacity discharged" stopping point at every
    temperature. `rated_capacity_ah` is used as a fallback only for a
    condition where no measurement group has its own `Cap_1C` section.

    `column_overrides` maps a filename (basename) to a per-field column-name
    override dict, for files where auto-detection needs help, e.g.:
        {"553_Mixed2.csv": {"current": "Current(A)"}}

    `min_soc_range` drops segments whose SoC spans less than this. A drive
    cycle that happens to fall in the saturated full-charge region carries a
    near-constant label while V/I/T vary underneath: it teaches the model
    nothing and, in the calibration split, contributes degenerate
    nonconformity scores. Defaults to 0.0 (keep everything) so the loader
    stays non-destructive; 0.02 is a reasonable value.

    `capacity_overrides` maps an ambient condition (degC) to a capacity in Ah,
    replacing whatever was measured for it. Use it when a condition's Cap_1C
    section is truncated or missing from your copy of the dataset and the
    loader warns about it, e.g. `{40.0: 2.80}`.
    """
    column_overrides = column_overrides or {}
    capacity_overrides = capacity_overrides or {}
    paths = discover_csv_files(root)

    groups: Dict[str, List[dict]] = defaultdict(list)
    for path in paths:
        parsed = _parse_file_raw(path, column_overrides.get(os.path.basename(path)))
        if parsed is None:
            continue
        frame, group_key, test_section, condition = parsed
        frame["test_section"] = test_section
        groups[group_key].append({"frame": frame, "condition": condition, "path": path})

    # Pass 1: reconstruct each group's stitched frame and, where a group
    # includes its own Cap_1C section, measure that condition's actual
    # capacity from it.
    combined_by_group: Dict[str, pd.DataFrame] = {}
    condition_by_group: Dict[str, Optional[float]] = {}
    section_list_by_group: Dict[str, str] = {}
    capacity_readings: Dict[float, List[float]] = defaultdict(list)
    for group_key, parts in groups.items():
        combined = pd.concat([p["frame"] for p in parts], ignore_index=True)
        combined = _collapse_duplicate_timestamps(combined)
        if len(combined) < 20:
            continue

        combined["time"] = (combined["abs_time"] - combined["abs_time"].iloc[0]).dt.total_seconds()
        condition = next((p["condition"] for p in parts if p["condition"] is not None), None)

        gaps = combined["time"].diff().to_numpy()
        big_gaps = gaps[gaps > max_gap_s]
        section_list = ", ".join(sorted({p["frame"]["test_section"].iloc[0] for p in parts}))
        if len(big_gaps) > 0:
            print(f"[rtcqr.data] measurement {group_key} ({section_list}): {len(big_gaps)} gap(s) > "
                  f"{max_gap_s:.0f}s (largest {np.nanmax(big_gaps):.0f}s) -- likely missing intermediate "
                  f"test-section files; SoC will not account for current drawn during the gap(s), and each "
                  f"gap becomes a windowing segment boundary.")

        if condition is not None:
            cap_mask = combined["test_section"].apply(lambda s: _CAPACITY_CHECK_PATTERN in s.lower()).to_numpy()
            if cap_mask.any():
                cap_rows = np.where(cap_mask)[0]
                measured = _measured_capacity_ah(
                    combined["time"].to_numpy()[cap_rows],
                    combined["current"].to_numpy()[cap_rows],
                    current_sign,
                    max_gap_s=max_gap_s,
                    rated_capacity_ah=rated_capacity_ah,
                    row_index=cap_rows,
                    label=f"measurement {group_key} @ {condition} degC",
                )
                if measured is not None:
                    capacity_readings[condition].append(measured)

        combined_by_group[group_key] = combined
        condition_by_group[group_key] = condition
        section_list_by_group[group_key] = section_list

    # Median, not mean: one measurement group with a truncated or doubled
    # capacity check must not drag the denominator that every group at that
    # condition shares.
    capacity_by_condition = {cond: float(np.median(vals)) for cond, vals in capacity_readings.items()}
    for cond, vals in sorted(capacity_readings.items(), key=lambda kv: (kv[0] is None, kv[0])):
        spread = (max(vals) - min(vals)) / max(np.median(vals), 1e-9)
        warn = "  <-- readings disagree by >20%, inspect them" if spread > 0.2 else ""
        print(f"[rtcqr.data] condition={cond}: measured capacity {np.median(vals):.3f} Ah "
              f"(median of {len(vals)} group(s): {[f'{v:.3f}' for v in vals]}), "
              f"vs. rated_capacity_ah={rated_capacity_ah:.3f}{warn}")
    for cond, cap in capacity_overrides.items():
        prev = capacity_by_condition.get(cond)
        capacity_by_condition[cond] = float(cap)
        print(f"[rtcqr.data] condition={cond}: capacity overridden to {cap:.3f} Ah"
              + (f" (was {prev:.3f} Ah measured)" if prev is not None else " (none measured)"))
    _warn_non_monotonic_capacity(capacity_by_condition)
    warned_conditions: set = set()

    # Pass 2: compute SoC (per-group capacity denominator) and split into
    # windowing segments.
    out: List[BatteryFile] = []
    dropped_degenerate: List[tuple] = []
    for group_key, combined in combined_by_group.items():
        condition = condition_by_group[group_key]
        section_list = section_list_by_group[group_key]

        if combined["soc_raw"].notna().mean() > 0.5:
            soc = combined["soc_raw"].to_numpy()
            if np.nanmax(soc) > 1.5:  # looks like a percentage
                soc = soc / 100.0
            soc = np.clip(soc, 0.0, 1.0)
            if np.isnan(soc).any():
                soc = pd.Series(soc).interpolate(limit_direction="both").to_numpy()
        else:
            group_capacity_ah = capacity_by_condition.get(condition, rated_capacity_ah)
            if condition not in capacity_by_condition and condition not in warned_conditions:
                print(f"[rtcqr.data] condition={condition}: no *usable* Cap_1C section at this condition "
                      f"(absent, or present but rejected by the plausibility guards above) -- falling back to "
                      f"rated_capacity_ah={rated_capacity_ah:.3f} Ah for SoC normalization here. If a section "
                      f"was rejected as truncated, prefer capacity_overrides={{{condition}: <Ah>}} over this "
                      f"fallback: the nominal rating overstates a cold cell's real capacity.")
                warned_conditions.add(condition)
            soc = _compute_soc_from_current(
                combined["time"].to_numpy(), combined["current"].to_numpy(),
                group_capacity_ah, current_sign, soc_initial, max_gap_s=max_gap_s,
            )
        combined["soc"] = soc

        if include_patterns:
            keep_mask = combined["test_section"].apply(lambda s: _matches_any(s, include_patterns)).to_numpy()
        else:
            keep_mask = np.ones(len(combined), dtype=bool)
        if not keep_mask.any():
            continue

        for seg in _split_contiguous(combined, keep_mask, max_gap_s):
            seg_start = seg["abs_time"].iloc[0]
            seg = seg[["time", "voltage", "current", "temperature", "soc"]].reset_index(drop=True)
            if resample_dt_s:
                seg = _resample_uniform(seg, resample_dt_s)
            if len(seg) < 20:
                continue
            soc_range = float(seg["soc"].max() - seg["soc"].min())
            if soc_range < min_soc_range:
                dropped_degenerate.append((condition, group_key, soc_range))
                continue
            out.append(BatteryFile(
                path=f"measurement {group_key} ({section_list})", condition=condition, frame=seg,
                start_time=seg_start, group_key=str(group_key),
            ))

    if dropped_degenerate:
        print(f"[rtcqr.data] dropped {len(dropped_degenerate)} segment(s) whose SoC spans less than "
              f"{min_soc_range} (near-constant label): "
              f"{[(c, k, round(r, 4)) for c, k, r in dropped_degenerate[:5]]}"
              + (" ..." if len(dropped_degenerate) > 5 else ""))

    if not out:
        raise RuntimeError(
            f"No usable battery segments parsed under {root!r}. Run "
            "`python -m rtcqr.data inspect <root>` to see per-file/per-group detection."
        )
    return out


def chronological_split(
    files: Sequence[BatteryFile], train_frac: float, val_frac: float
) -> Tuple[List[BatteryFile], List[BatteryFile], List[BatteryFile]]:
    """Split each segment's time series into train/val/test slices in time order.

    Appropriate for a dataset made of a small number of long, continuous
    multi-profile sweeps (SoC declining smoothly from ~1.0 to some low
    point over many hours), where 15% of one such sweep still spans a
    meaningful, representative chunk of the SoC trajectory. Wrong for a
    dataset made of many short, single charge/discharge-cycle segments
    (see `segment_split`): slicing chronologically within a short segment
    systematically gives train the high-SoC early portion and test the
    low-SoC late portion of every cycle, so train rarely sees low-SoC
    examples and the calibration set's SoC distribution differs
    systematically from the test set's -- breaking both generalization
    and the conformal calibration exchangeability assumption.
    """
    train_parts, val_parts, test_parts = [], [], []
    for bf in files:
        n = len(bf.frame)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        df = bf.frame
        for parts, sl in (
            (train_parts, slice(None, n_train)),
            (val_parts, slice(n_train, n_train + n_val)),
            (test_parts, slice(n_train + n_val, None)),
        ):
            parts.append(replace(bf, frame=df.iloc[sl].reset_index(drop=True)))
    return train_parts, val_parts, test_parts


def _split_counts(n: int, train_frac: float, val_frac: float, val_calib_fraction: float):
    """Allocate `n` segments across train/val/calib/test, guaranteeing every
    split gets at least one segment. Requires n >= 4."""
    if n < 4:
        raise ValueError(f"need at least 4 segments to form 4 non-empty splits, got {n}")
    n_val_total = max(2, int(round(n * val_frac)))  # >=2 so val and calib each get one
    n_val = max(1, int(round(n_val_total * (1.0 - val_calib_fraction))))
    n_calib = max(1, n_val_total - n_val)
    n_test = max(1, int(round(n * max(0.0, 1.0 - train_frac - val_frac))))
    n_train = n - n_val - n_calib - n_test
    while n_train < 1:  # tiny n: claw back from the largest non-minimal split
        if n_test > 1:
            n_test -= 1
        elif n_val > 1:
            n_val -= 1
        elif n_calib > 1:
            n_calib -= 1
        else:
            raise ValueError(f"cannot form 4 non-empty splits from {n} segments")
        n_train = n - n_val - n_calib - n_test
    return n_train, n_val, n_calib, n_test


def segment_split(
    files: Sequence[BatteryFile],
    train_frac: float,
    val_frac: float,
    val_calib_fraction: float,
    seed: int = 42,
    stratify_by_condition: bool = True,
) -> Tuple[List[BatteryFile], List[BatteryFile], List[BatteryFile], List[BatteryFile]]:
    """Randomly assign *whole* segments to train / val_model / calib / test,
    instead of slicing each segment chronologically (see `chronological_split`
    for why that's wrong for this dataset). Every split then sees a
    representative mix of full charge/discharge trajectories rather than a
    systematically biased SoC sub-range.

    The assignment is **stratified by ambient condition** by default. A plain
    random split over all segments at once leaves the calibration split
    missing entire temperatures: with ~50-100 segments spread over this
    dataset's six conditions, calib draws only ~7.5% of them, so in the large
    majority of seeds the test set contains a temperature that calib never
    saw. Conformal calibration's coverage guarantee rests on calib and test
    being exchangeable, and this dataset's SoC/error distribution is strongly
    temperature-dependent -- that is precisely why the coulomb-counting
    denominator is measured per condition -- so calibrating -20 degC test
    windows against a buffer containing no -20 degC data silently voids the
    guarantee. Stratifying splits each condition independently, so every
    condition present in test is represented in calib in the same proportion.

    A condition with fewer than 4 segments cannot fill all four splits; those
    segments go to train, which is the safe direction (it can only cost the
    model examples, never break calib/test exchangeability).

    Returns (train, val_model, calib, test) as BatteryFile lists, so callers
    keep each segment's condition and absolute start time -- the latter is
    needed to order the calibration buffer chronologically before the
    conformal time decay is applied to it.
    """
    rng = np.random.default_rng(seed)
    files = list(files)
    if len(files) < 4:
        raise ValueError(
            f"need at least 4 windowing segments to build train/val/calib/test, got {len(files)}. "
            "Loosen --exclude-measurement-ids or include more test sections."
        )

    if stratify_by_condition:
        strata: Dict[object, List[int]] = defaultdict(list)
        for i, bf in enumerate(files):
            strata[bf.condition].append(i)
    else:
        strata = {None: list(range(len(files)))}

    train_idx: List[int] = []
    val_idx: List[int] = []
    calib_idx: List[int] = []
    test_idx: List[int] = []
    undersized: List[object] = []

    for cond in sorted(strata, key=lambda c: (c is None, c)):
        idx = np.array(strata[cond])
        order = idx[rng.permutation(len(idx))]
        if len(order) < 4:
            undersized.append(cond)
            train_idx.extend(order.tolist())
            continue
        n_tr, n_va, n_ca, n_te = _split_counts(len(order), train_frac, val_frac, val_calib_fraction)
        train_idx.extend(order[:n_tr].tolist())
        val_idx.extend(order[n_tr:n_tr + n_va].tolist())
        calib_idx.extend(order[n_tr + n_va:n_tr + n_va + n_ca].tolist())
        test_idx.extend(order[n_tr + n_va + n_ca:].tolist())

    if undersized:
        print(f"[rtcqr.data] condition(s) {undersized} have <4 segments; assigning them all to train "
              f"(cannot fill calib/test without breaking exchangeability).")
    if not test_idx or not calib_idx:
        counts = {c: len(v) for c, v in sorted(strata.items(), key=lambda kv: (kv[0] is None, kv[0]))}
        raise ValueError(
            f"stratified split left calib or test empty: every condition has fewer than 4 segments "
            f"(segments per condition: {counts}). Each condition needs >=4 to fill "
            f"train/val/calib/test independently. Either supply more data (a smaller --window-size or "
            f"--include-all yields more segments), or pass stratify_by_condition=False / "
            f"--no-stratify to split across conditions -- but note that unstratified splits routinely "
            f"put a temperature in test that calib never saw, which voids the conformal coverage "
            f"guarantee for those windows."
        )

    def pick(idx: List[int]) -> List[BatteryFile]:
        return [files[i] for i in idx]

    return pick(train_idx), pick(val_idx), pick(calib_idx), pick(test_idx)


def order_chronologically(files: Sequence[BatteryFile]) -> List[BatteryFile]:
    """Sort segments by their absolute start time.

    `conformal.time_decay_weights` weights a calibration buffer by position,
    treating the last entry as "now" and decaying backwards. That is only
    meaningful if the buffer is in chronological order. Segments come out of
    the loader grouped by measurement (i.e. ordered by file path: temperature
    folder, then measurement id) and are then randomly permuted by
    `segment_split`, so without this the decay would weight an arbitrary
    segment's tail as the most recent evidence. Segments with no known start
    time sort last, preserving their relative order.
    """
    known = [bf for bf in files if bf.start_time is not None]
    unknown = [bf for bf in files if bf.start_time is None]
    return sorted(known, key=lambda bf: bf.start_time) + unknown


def _as_frames(items: Sequence) -> List[pd.DataFrame]:
    """Accept either BatteryFile objects or bare DataFrames."""
    return [it.frame if isinstance(it, BatteryFile) else it for it in items]


def make_windows(
    frames: Sequence, window_size: int, stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Build sliding windows of (voltage, current, temperature) -> SoC at the window's last step.

    Returns X of shape (N, 3, window_size) and y of shape (N,).

    Built with `sliding_window_view` rather than by appending one array per
    window to a Python list: the list-of-arrays form peaks at ~2.3x the size
    of the output array (every window is materialised separately before being
    stacked into a second full copy) and is ~7x slower. At this dataset's
    scale -- 1 Hz resampling with stride 1 over many hours of drive cycles --
    the training split alone runs to hundreds of MB, so that peak is a real
    out-of-memory risk on a modest machine, not just a speed problem.
    """
    xs, ys = [], []
    for df in _as_frames(frames):
        if len(df) < window_size:
            continue
        chans = np.stack([
            df["voltage"].to_numpy(dtype=np.float32),
            df["current"].to_numpy(dtype=np.float32),
            df["temperature"].to_numpy(dtype=np.float32),
        ], axis=0)
        soc = df["soc"].to_numpy(dtype=np.float32)
        # (channels, n, window) -> (n, channels, window); one row per window end.
        win = np.lib.stride_tricks.sliding_window_view(chans, window_size, axis=1)
        win = win.transpose(1, 0, 2)[::stride]
        xs.append(win)
        ys.append(soc[window_size - 1::stride])
    if not xs:
        raise RuntimeError("No windows could be built; window_size may exceed the shortest split length.")
    return np.ascontiguousarray(np.concatenate(xs, axis=0)), np.concatenate(ys).astype(np.float32)


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
