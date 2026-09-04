"""Why does one condition's Cap_1C measure low? Aging or bad data?

Prints, per condition's capacity-check section: when it ran, how long, what
voltage it started and ended at, and how much charge came out. A check that
started well below full charge, or that has an internal time gap, is bad
data. One that looks clean but ran late in the campaign is aging.
"""
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from rtcqr.data import (_CAPACITY_CHECK_PATTERN, _collapse_duplicate_timestamps,
                        _integrate_discharge_ah, _parse_file_raw, discover_csv_files,
                        _extract_temperature_c)
import os

root = sys.argv[1]
groups = defaultdict(list)
for p in discover_csv_files(root):
    parsed = _parse_file_raw(p, None)
    if parsed is None:
        continue
    frame, key, section, cond = parsed
    frame["test_section"] = section
    groups[key].append({"frame": frame, "condition": cond, "path": p})

rows = []
for key, parts in groups.items():
    cond = next((p["condition"] for p in parts if p["condition"] is not None), None)
    combined = _collapse_duplicate_timestamps(pd.concat([p["frame"] for p in parts], ignore_index=True))
    if len(combined) < 20:
        continue
    combined["time"] = (combined["abs_time"] - combined["abs_time"].iloc[0]).dt.total_seconds()
    mask = combined["test_section"].str.lower().str.contains(_CAPACITY_CHECK_PATTERN).to_numpy()
    if not mask.any():
        continue
    idx = np.where(mask)[0]
    for chunk in np.split(idx, np.where(np.diff(idx) != 1)[0] + 1):
        if len(chunk) < 5:
            continue
        sub = combined.iloc[chunk]
        ah, dur = _integrate_discharge_ah(sub["time"].to_numpy(), sub["current"].to_numpy(), 1.0, 300.0)
        gaps = np.diff(sub["time"].to_numpy())
        rows.append(dict(cond=cond, meas=key, when=sub["abs_time"].iloc[0],
                         hours=dur / 3600, ah=ah,
                         v_start=sub["voltage"].iloc[0], v_end=sub["voltage"].iloc[-1],
                         max_gap=float(gaps.max()) if len(gaps) else 0.0,
                         n=len(sub)))

df = pd.DataFrame(rows).sort_values(["cond", "when"])
pd.set_option("display.width", 200)
print("\nEvery Cap_1C section found, per condition:\n")
print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

print("\n\nCampaign order (does the low condition simply come last?):\n")
print(df.sort_values("when")[["when", "cond", "meas", "ah", "hours", "v_start"]]
        .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
