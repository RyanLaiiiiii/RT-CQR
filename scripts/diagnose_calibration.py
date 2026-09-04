#!/usr/bin/env python3
"""Diagnose why rtcqr/cqr/wcp calibration produced identical LVR/AIW/ACE.

The three calibrators in baselines.py only differ in (zeta, gamma, wl0, wl1);
all three share wu=1.0. The nonconformity score (conformal.py) is

    score_i = max(wl(u_i) * lower_excess_i, wu * upper_excess_i)

so if `lower_excess = max(q_lower - soc_true, 0)` is ~0 for virtually every
calibration/test sample, the `wl(u_i) * lower_excess_i` branch never wins the
max() regardless of wl0/wl1/gamma, and rtcqr collapses to being numerically
identical to cqr/wcp (which is exactly the symptom being investigated). This
script reuses the already-trained model + a deterministic re-derivation of
the same split (same seed, same flags used for training) to check that
without retraining, and prints the breakdown that confirms or rules it out.

Usage: same data/split flags as train.py, plus --model-path.
    python scripts/diagnose_calibration.py --data-root /path/to/lg_hg2
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from rtcqr.config import RTCQRConfig
from rtcqr.model import TCNQuantileNet
from train import build_windows, predict_quantiles, set_seed


def describe(name: str, values: np.ndarray) -> str:
    if values.size == 0:
        return f"{name}: n=0"
    frac_pos = float(np.mean(values > 0))
    mean_pos = float(values[values > 0].mean()) if frac_pos > 0 else 0.0
    return f"{name}: frac>0={frac_pos:.4f}  mean|>0={mean_pos:.5f}  max={values.max():.5f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--dataset-slug", type=str, default="aditya9790/lg-18650hg2-liion-battery-data")
    parser.add_argument("--current-sign", type=float, default=1.0)
    parser.add_argument("--include-all", action="store_true")
    parser.add_argument("--exclude-measurement-ids", nargs="+", default=None)
    parser.add_argument("--split-mode", choices=["segment", "chronological"], default="segment")
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--resample-dt", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=str, default="outputs/rtcqr_model.pt")
    args = parser.parse_args()

    cfg = RTCQRConfig(seed=args.seed)
    if args.window_size is not None:
        cfg.window_size = args.window_size
    if args.resample_dt is not None:
        cfg.resample_dt_s = args.resample_dt or None

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.download:
        from rtcqr.data import download_lg_hg2
        data_root = download_lg_hg2(args.dataset_slug)
    else:
        if args.data_root is None:
            raise SystemExit("Provide --data-root <path> or --download.")
        data_root = args.data_root

    splits = build_windows(
        cfg, data_root, current_sign=args.current_sign, include_all=args.include_all,
        exclude_measurement_ids=args.exclude_measurement_ids, split_mode=args.split_mode,
    )

    model = TCNQuantileNet(
        in_channels=cfg.in_channels, quantile_levels=cfg.quantile_levels, num_blocks=cfg.num_blocks,
        channels=cfg.channels, kernel_size=cfg.kernel_size, dropout=cfg.dropout,
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    (X_calib, y_calib), (X_test, y_test) = splits["calib"], splits["test"]
    q_calib = predict_quantiles(model, X_calib, device)
    q_test = predict_quantiles(model, X_test, device)

    tau_l_idx = cfg.tau_l_index
    print(f"\nquantile_levels = {cfg.quantile_levels}  (tau_l index = {tau_l_idx})")
    for split_name, y, q in [("calib", y_calib, q_calib), ("test", y_test, q_test)]:
        crossing_rate = float(np.mean(np.any(np.diff(q, axis=1) < 0, axis=1)))
        print(f"\n--- {split_name} (n={len(y)}) ---")
        print(f"quantile-crossing rate (any adjacent pair out of order): {crossing_rate:.4f}")
        print(f"q_tau_l (lowest quantile, tau={cfg.quantile_levels[tau_l_idx]}): "
              f"mean={q[:, tau_l_idx].mean():.4f}  frac_above_soc_min({cfg.soc_min})="
              f"{float(np.mean(q[:, tau_l_idx] > cfg.soc_min)):.4f}")
        print(f"soc_true: mean={y.mean():.4f}  frac_below_soc_min={float(np.mean(y < cfg.soc_min)):.4f}")

        for alpha in cfg.pi_alphas:
            idx_l, idx_u = cfg.quantile_bounds(alpha)
            q_lo, q_hi = q[:, idx_l], q[:, idx_u]
            lower_excess = np.clip(q_lo - y, 0.0, None)
            upper_excess = np.clip(y - q_hi, 0.0, None)
            lower_wins = (lower_excess > upper_excess).mean()
            pi_key = f"{int(round((1 - alpha) * 100))}%"
            print(f"  [{pi_key} PI, tl={cfg.quantile_levels[idx_l]}, tu={cfg.quantile_levels[idx_u]}]")
            print(f"    {describe('lower_excess (q_lo - soc_true, if soc under-covered from below)', lower_excess)}")
            print(f"    {describe('upper_excess (soc_true - q_hi, if soc under-covered from above)', upper_excess)}")
            print(f"    frac of samples where lower_excess > upper_excess (i.e. wl-branch would win the max): "
                  f"{lower_wins:.4f}")


if __name__ == "__main__":
    main()
