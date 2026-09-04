"""Synthetic LG-HG2-shaped cycler exports with known-by-construction answers.

Every fixture here writes real files in the raw cycler-export format
(metadata preamble, header row, units row, data) so the tests exercise
`load_lg_hg2_dataframe` end to end rather than poking at internals.
"""
from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pytest

_HDR = ("Time Stamp,Step,Status,Prog Time,Step Time,Cycle,Cycle Level,Procedure,"
        "Voltage,Current,Temperature,Capacity,WhAccu,Cnt,\n")
_UNITS = ",,,,,,,,[V],[A],[C],[Ah],[Wh],[Cnt],\n"


def write_section(path, measurement_id, section, t0, rows, temp_c):
    """Write one cycler-export CSV. `rows` is [(dt_seconds, current_a, voltage), ...].

    Returns the timestamp after the last row, so sections chain in real time.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"Measurement ID,{measurement_id}\n")
        f.write("Battery Name,LG HG2 18650_SN62A4\n")
        f.write("Nominal Capacity, 3\n")
        f.write(f"Test section,{section}\n")
        f.write(_HDR)
        f.write(_UNITS)
        for dt_s, cur, volt in rows:
            t0 = t0 + dt.timedelta(seconds=dt_s)
            f.write(f"{t0.strftime('%m/%d/%Y %I:%M:%S %p')},22,DCH,00:00:00.000,00:00:00.000,0,0,LG,"
                    f"{volt:.5f},{cur:.5f},{temp_c:.5f},0.0,0.0,1.0,\n")
    return t0


def cc(seconds, amps, dt_s=10, volt=3.7):
    """Constant-current rows; negative amps = discharge (this dataset's convention)."""
    return [(dt_s, amps, volt)] * (seconds // dt_s)


# True 1C capacity per ambient condition, mirroring the real cell's
# capacity-vs-temperature behaviour (monotonically rising with temperature).
TRUE_CAPACITY_AH = {
    "n20degC": 1.65, "n10degC": 2.25, "0degC": 2.47,
    "10degC": 2.52, "25degC": 2.71, "40degC": 2.78,
}


def _temp_of(folder):
    return -float(folder[1:-4]) if folder.startswith("n") else float(folder[:-4])


@pytest.fixture(scope="session")
def full_dataset(tmp_path_factory):
    """Six conditions x 6 measurements, each Charge -> Cap_1C -> Charge -> 2 drive
    cycles. Drive cycles are logged at 10 Hz so their one-second timestamps
    repeat, exercising the sub-second collapse path."""
    root = tmp_path_factory.mktemp("lg_hg2_full")
    rng = np.random.default_rng(0)
    for folder, cap in TRUE_CAPACITY_AH.items():
        temp = _temp_of(folder)
        t = dt.datetime(2018, 11, 20, 8, 0, 0)
        for mid in range(6):
            meas = f"{folder}_{mid}"
            base = str(root / folder)
            t = write_section(f"{base}/{meas}_Charge1.csv", meas, "Charge1", t, cc(3600, +cap), temp)
            t = write_section(f"{base}/{meas}_Cap_1C.csv", meas, "Cap_1C", t, cc(3600, -cap), temp)
            t = write_section(f"{base}/{meas}_Charge2.csv", meas, "Charge2", t, cc(3600, +cap), temp)
            for name in ("HWFET", "UDDS"):
                n = 3000  # 300 s at 10 Hz
                prof = np.abs(rng.normal(cap * 0.8, cap * 0.4, n))
                rows = [(0.1, -float(a), 3.7 - 0.3 * k / n) for k, a in enumerate(prof)]
                t = write_section(f"{base}/{meas}_{name}.csv", meas, name, t, rows, temp)
            t += dt.timedelta(hours=2)
    return str(root)
