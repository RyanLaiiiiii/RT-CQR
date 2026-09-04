"""Generates a tiny synthetic dataset with the same folder/column layout as
the LG 18650HG2 Kaggle mirror, purely to smoke-test the rtcqr pipeline
without network access. Not part of the shipped package.
"""
import os
import sys

import numpy as np
import pandas as pd


def make_profile(n, seed, rated_capacity_ah=3.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)  # 1 Hz
    # discharge-dominated current profile (charge-positive convention)
    current = -1.5 + 0.8 * np.sin(t / 37.0) + rng.normal(0, 0.05, n)
    current[:5] = 0.0
    dt = np.ones(n)
    soc = 1.0 + np.cumsum(current * dt / 3600.0) / rated_capacity_ah
    soc = np.clip(soc, 0.0, 1.0)
    ocv = 3.0 + 1.2 * soc
    voltage = ocv + 0.05 * current + rng.normal(0, 0.01, n)
    temperature = 25 + 0.02 * t / n * 5 + rng.normal(0, 0.2, n)
    return pd.DataFrame({
        "Time Stamp": t,
        "Voltage(V)": voltage,
        "Current(A)": current,
        "Temperature (C)_1": temperature,
    })


def main(root):
    conditions = ["0degC", "25degC", "n10degC"]
    profiles = ["UDDS", "LA92", "Mixed1"]
    for cond in conditions:
        d = os.path.join(root, cond)
        os.makedirs(d, exist_ok=True)
        for i, prof in enumerate(profiles):
            df = make_profile(n=1200, seed=hash((cond, prof)) % (2**31))
            df.to_csv(os.path.join(d, f"55{i}_{prof}.csv"), index=False)
    print(f"wrote synthetic dataset to {root}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/synthetic_lg_hg2")
