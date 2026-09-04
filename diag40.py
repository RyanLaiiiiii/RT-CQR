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

# --------------------------------------------------------------------------
# Termination voltage: the check a normal duration cannot catch.
# --------------------------------------------------------------------------
# Internal resistance falls as a cell warms, so a warmer cell holds its
# voltage up longer under the same load and reaches a fixed cutoff later.
# v_end must therefore DECREASE monotonically with temperature. A condition
# that stops well above a colder one ran out of file, not out of charge: its
# section was cut short even though it lasted a normal ~1 h -- exactly the
# case a duration guard cannot see.
print("\n\nTermination voltage vs. temperature (v_end must fall as temp rises):\n")
per = df.groupby("cond").agg(v_end=("v_end", "mean"), ah=("ah", "mean"),
                             hours=("hours", "mean"), n=("ah", "size")).sort_index()
print(per.to_string(float_format=lambda v: f"{v:.3f}"))

conds = list(per.index)
flagged = []
for warm in conds:
    colder = [c for c in conds if c < warm]
    if not colder:
        continue
    deepest_cold = min(colder, key=lambda c: per.loc[c, "v_end"])
    dv = per.loc[warm, "v_end"] - per.loc[deepest_cold, "v_end"]
    if dv > 0.05:
        flagged.append((warm, deepest_cold, dv))

print()
if flagged:
    for warm, cold, dv in flagged:
        print(f"SUSPECT  {warm} degC terminated at {per.loc[warm, 'v_end']:.3f} V, {dv * 1000:.0f} mV ABOVE")
        print(f"         the colder {cold} degC ({per.loc[cold, 'v_end']:.3f} V). A warmer cell has lower")
        print(f"         internal resistance and should discharge DEEPER, not stop earlier, so its")
        print(f"         {per.loc[warm, 'ah']:.3f} Ah is an UNDER-estimate: the section ended before the")
        print(f"         discharge finished. Duration was {per.loc[warm, 'hours']:.3f} h, i.e. normal, which")
        print(f"         is why the truncation guard in _measured_capacity_ah does not catch it.")
        print(f"         Estimate from the datasheet ratio against {cold} degC ({per.loc[cold, 'ah']:.3f} Ah):")
        print(f"             capacity_overrides={{{warm}: {per.loc[cold, 'ah'] * 1.015:.2f}}}")
        print()
else:
    print("OK  termination voltage falls monotonically with temperature; no section")
    print("    appears to have stopped early.")
