#!/usr/bin/env python3
"""End-to-end RT-CQR training and evaluation on the LG 18650HG2 dataset.

Usage:
    # Download via kagglehub (requires Kaggle API credentials) and train:
    python train.py --download

    # Or point at an already-downloaded copy of the dataset:
    python train.py --data-root /path/to/lg_hg2

Reproduces the RT-CQR* setting of Table I and evaluates LVR/AIW/ACE
(eq. 29) at 90% and 95% nominal PI coverage (Table II), and also reports
the CQR and WCP calibration baselines using the *same* trained quantile
model, isolating the effect of violation-weighted time-adaptive
calibration. Pass --no-ltr to reproduce the "RT-CQR w/o LTR" ablation row
(Table IV); the "w/o VW-TAC" row is the RT-CQR model with --calibrators cqr.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from rtcqr.baselines import make_cqr_calibrator, make_rtcqr_calibrator, make_wcp_calibrator
from rtcqr.config import RTCQRConfig
from rtcqr.data import (
    Standardizer,
    _DEFAULT_INCLUDE_PATTERNS,
    chronological_split,
    drivecycle_split,
    load_lg_hg2_dataframe,
    make_windows,
    segment_split,
)
from rtcqr.losses import composite_quantile_loss
from rtcqr.metrics import summarize
from rtcqr.model import TCNQuantileNet


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_calib_stride(calib_frames, cfg: RTCQRConfig) -> int:
    """Largest stride (at most window_size, i.e. non-overlapping) that still
    yields cfg.calib_min_windows calibration samples.

    c_alpha is a weighted empirical quantile of the calibration scores, so
    its resolution is bounded by 1/N_cal: 66 samples cannot place a 0.05
    tail to better than 0.015. Overlapping the windows is the lesser evil,
    but only as far as the sample count actually requires.
    """
    usable = sum(max(0, len(df) - cfg.window_size + 1) for df in calib_frames)
    if usable <= 0:
        return cfg.window_size
    stride = max(1, min(cfg.window_size, usable // max(1, cfg.calib_min_windows)))
    print(f"[rtcqr.train] calibration stride {stride} "
          f"(target >={cfg.calib_min_windows} windows from {usable} usable positions; "
          f"non-overlapping would be {cfg.window_size})")
    return stride


def build_windows(cfg: RTCQRConfig, data_root: str, current_sign: float, include_all: bool = False,
                   exclude_measurement_ids: Optional[List[str]] = None, split_mode: str = "segment"):
    include_patterns = None if include_all else _DEFAULT_INCLUDE_PATTERNS
    files = load_lg_hg2_dataframe(
        data_root, rated_capacity_ah=cfg.rated_capacity_ah, current_sign=current_sign,
        include_patterns=include_patterns, resample_dt_s=cfg.resample_dt_s,
    )

    if exclude_measurement_ids:
        before = len(files)
        excluded = {str(x) for x in exclude_measurement_ids}
        files = [bf for bf in files if not any(f"measurement {mid} " in bf.path for mid in excluded)]
        print(f"[rtcqr.train] Excluded {before - len(files)} segment(s) from measurement IDs {sorted(excluded)}")

    print(f"[rtcqr.train] Loaded {len(files)} windowing segment(s) from {data_root}")

    if split_mode == "drivecycle":
        train_frames, val_model_frames, calib_frames, test_frames = drivecycle_split(
            files, cfg.val_frac, cfg.val_calib_fraction, seed=cfg.seed
        )
    elif split_mode == "segment":
        # Most segments in this dataset are short, single charge/discharge
        # cycles (SoC ~1.0 -> some low point over a few hours). Slicing each
        # one chronologically would systematically give train the high-SoC
        # early portion and test the low-SoC late portion of every cycle --
        # confirmed on the full dataset: calib mean SoC 0.33 vs. test mean
        # SoC 0.25, and a 24% quantile-crossing rate on test vs. 3% on
        # calib. Assigning whole segments to a split instead keeps each
        # split's SoC distribution representative.
        train_frames, val_model_frames, calib_frames, test_frames = segment_split(
            files, cfg.train_frac, cfg.val_frac, cfg.val_calib_fraction, seed=cfg.seed
        )
    else:
        train_frames, val_frames, test_frames = chronological_split(files, cfg.train_frac, cfg.val_frac)
        # further split each validation frame chronologically into model-val / calibration
        val_model_frames, calib_frames = [], []
        for df in val_frames:
            n_val = len(df)
            n_cal = int(round(n_val * cfg.val_calib_fraction))
            val_model_frames.append(df.iloc[:n_val - n_cal].reset_index(drop=True))
            calib_frames.append(df.iloc[n_val - n_cal:].reset_index(drop=True))

    X_train, y_train = make_windows(train_frames, cfg.window_size, cfg.stride)
    X_val, y_val = make_windows(val_model_frames, cfg.window_size, cfg.stride)
    calib_stride = cfg.calib_stride or _resolve_calib_stride(calib_frames, cfg)
    X_calib, y_calib = make_windows(calib_frames, cfg.window_size, calib_stride)
    X_test, y_test = make_windows(test_frames, cfg.window_size, stride=1)

    scaler = Standardizer(cfg.normalize).fit(X_train)
    X_train, X_val, X_calib, X_test = (scaler.transform(x) for x in (X_train, X_val, X_calib, X_test))

    print(f"[rtcqr.train] windows: train={len(y_train)} val={len(y_val)} calib={len(y_calib)} test={len(y_test)}")
    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "calib": (X_calib, y_calib),
        "test": (X_test, y_test),
    }


def train_model(cfg: RTCQRConfig, splits, device: torch.device) -> TCNQuantileNet:
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = TCNQuantileNet(
        in_channels=cfg.in_channels,
        quantile_levels=cfg.quantile_levels,
        num_blocks=cfg.num_blocks,
        channels=cfg.channels,
        kernel_size=cfg.kernel_size,
        dropout=cfg.dropout,
        dilation_base=cfg.dilation_base,
    ).to(device)
    dilations = [cfg.dilation_base ** b for b in range(cfg.num_blocks)]
    history_s = cfg.window_size * (cfg.resample_dt_s or 1.0)
    print(f"[rtcqr.train] TCN dilations={dilations} receptive_field={model.receptive_field} "
          f"window_size={cfg.window_size} ({history_s:.0f}s of history at "
          f"{cfg.resample_dt_s or 1.0}s sampling); inputs scaled by '{cfg.normalize}'")
    if model.receptive_field < cfg.window_size:
        print(f"[rtcqr.train] WARNING: the first {cfg.window_size - model.receptive_field} step(s) of every "
              f"window are outside the receptive field and cannot affect the prediction.")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            q_pred = model(xb)
            loss = composite_quantile_loss(
                yb, q_pred, cfg.quantile_levels, cfg.soc_min, cfg.lambda_nc, cfg.lambda_l, cfg.tau_l_index
            )
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                q_pred = model(xb)
                loss = composite_quantile_loss(
                    yb, q_pred, cfg.quantile_levels, cfg.soc_min, cfg.lambda_nc, cfg.lambda_l, cfg.tau_l_index
                )
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_ds)

        print(f"[rtcqr.train] epoch {epoch:03d}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print(f"[rtcqr.train] early stopping at epoch {epoch} (best val_loss={best_val:.5f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_quantiles(model: TCNQuantileNet, X: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    preds = []
    for start in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[start:start + batch_size]).to(device)
        preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds, axis=0)


def report_interval_validity(cfg: RTCQRConfig, q_test: np.ndarray) -> Dict[str, float]:
    """Check that the quantile head's raw output really is an interval.

    The head is `nn.Linear(channels, len(quantile_levels))`: eight
    independent projections with nothing tying them together. Monotonicity
    across tau is encouraged only softly, by the `lambda_nc` crossing
    penalty in eq. (17). When a pair does cross, `[q_tl, q_tu]` is an
    inverted "interval" with negative width, and every metric computed
    from it (AIW, ACE, and the conformal scores that add c_alpha to both
    ends) is meaningless for that sample. Nothing downstream sorts or
    clamps, so this is reported rather than silently repaired.

    SoC is also physically confined to [0, 1] while the head is
    unbounded, so out-of-range bounds are reported too.
    """
    stats = {
        "crossing_any_adjacent_pair": float(np.mean(np.any(np.diff(q_test, axis=1) < 0, axis=1))),
        "below_zero": float(np.mean(q_test < 0.0)),
        "above_one": float(np.mean(q_test > 1.0)),
    }
    print("\n[rtcqr.train] pre-calibration quantile head sanity:")
    print(f"  samples with >=1 crossed adjacent quantile pair: {stats['crossing_any_adjacent_pair'] * 100:.2f}%")
    for alpha in cfg.pi_alphas:
        idx_l, idx_u = cfg.quantile_bounds(alpha)
        inverted = float(np.mean(q_test[:, idx_l] > q_test[:, idx_u]))
        key = f"inverted_interval_{int(round((1 - alpha) * 100))}pct"
        stats[key] = inverted
        print(f"  inverted {int(round((1 - alpha) * 100))}% interval (q_l > q_u):        {inverted * 100:.2f}%")
    print(f"  predicted quantiles outside [0, 1]: {stats['below_zero'] * 100:.2f}% below 0, "
          f"{stats['above_one'] * 100:.2f}% above 1")
    return stats


def evaluate(cfg: RTCQRConfig, model: TCNQuantileNet, splits, device, calibrators: List[str]) -> Dict:
    (X_calib, y_calib), (X_test, y_test) = splits["calib"], splits["test"]
    q_calib = predict_quantiles(model, X_calib, device)
    q_test = predict_quantiles(model, X_test, device)

    report_interval_validity(cfg, q_test)

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for alpha in cfg.pi_alphas:
        idx_l, idx_u = cfg.quantile_bounds(alpha)
        pi_key = f"{int(round((1 - alpha) * 100))}%"
        results[pi_key] = {}

        # The quantile model on its own, before any conformal step. This is
        # the "w/o calibration" baseline the VW-TAC ablation is measured
        # against, so it is always reported.
        results[pi_key]["uncalibrated"] = summarize(
            y_test, q_test[:, idx_l], q_test[:, idx_u], alpha, cfg.soc_min
        )

        for name in calibrators:
            if name == "rtcqr":
                calibrator = make_rtcqr_calibrator(cfg.soc_min, cfg.zeta, cfg.gamma, cfg.wl0, cfg.wl1,
                                                   cfg.wu, cfg.signed_score)
            elif name == "cqr":
                calibrator = make_cqr_calibrator(cfg.soc_min)
            elif name == "wcp":
                calibrator = make_wcp_calibrator(cfg.soc_min, zeta=cfg.zeta)
            else:
                raise ValueError(f"Unknown calibrator {name!r}")

            calibrator.fit(y_calib, q_calib[:, idx_l], q_calib[:, idx_u])
            lo, hi = calibrator.calibrate_interval(q_test[:, idx_l], q_test[:, idx_u], alpha)
            results[pi_key][name] = summarize(y_test, lo, hi, alpha, cfg.soc_min)

    return results


def print_results_table(results: Dict, point_lvr: float = None):
    print("\n=== LG 18650HG2: LVR / AIW / ACE (lower is better) ===")
    for pi_key, per_calib in results.items():
        print(f"\n-- {pi_key} PI --")
        header = f"{'method':<14}{'LVR':>10}{'AIW':>10}{'ACE':>10}"
        print(header)
        if point_lvr is not None:
            print(f"{'Point':<14}{point_lvr:>10.3f}{'-':>10}{'-':>10}")
        for name, m in per_calib.items():
            print(f"{name:<14}{m['LVR']:>10.3f}{m['AIW']:>10.3f}{m['ACE']:>10.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=str, default=None, help="Path to a local copy of the LG 18650HG2 dataset.")
    parser.add_argument("--download", action="store_true", help="Download the dataset via kagglehub first.")
    parser.add_argument("--dataset-slug", type=str, default="aditya9790/lg-18650hg2-liion-battery-data")
    parser.add_argument("--rated-capacity", type=float, default=None,
                         help="Force one coulomb-counting capacity (Ah) for every temperature. "
                              "Default: measure each temperature's usable 1C capacity from its own "
                              "Cap_1C/C20DisCh section.")
    parser.add_argument("--current-sign", type=float, default=1.0, help="1.0 if I>0 means charging, -1.0 if I>0 means discharging.")
    parser.add_argument("--include-all", action="store_true",
                         help="Include static characterization test sections (C/20, OCV, HPPC, ...) instead of "
                              "only dynamic drive-cycle profiles.")
    parser.add_argument("--exclude-measurement-ids", nargs="+", default=None,
                         help="Drop entire Measurement IDs from the windowing segments, "
                              "e.g. --exclude-measurement-ids 590 556")
    parser.add_argument("--split-mode", choices=["drivecycle", "segment", "chronological"], default="drivecycle",
                         help="'drivecycle' (default) holds out every LA92/UDDS/US06 segment as test, per the "
                              "protocol of [6]. 'segment' randomly assigns whole segments to train/val/calib/test, "
                              "appropriate when most segments are short single charge/discharge cycles. "
                              "'chronological' slices each segment by time, appropriate only when segments are "
                              "few, long, continuous multi-profile sweeps.")
    parser.add_argument("--calibrators", nargs="+", default=["rtcqr", "cqr", "wcp"], choices=["rtcqr", "cqr", "wcp"])
    parser.add_argument("--no-ltr", action="store_true", help="Ablation: disable the lower-tail regularizer (lambda_l=0).")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--resample-dt", type=float, default=None,
                         help="Uniform resampling interval in seconds applied to each reconstructed "
                              "measurement run before windowing (default 1.0). Pass 0 to disable resampling.")
    parser.add_argument("--signed-score", action="store_true",
                         help="Give RT-CQR the signed/standard-CQR residual instead of eq. (20)'s "
                              "[.]_+, letting c_alpha go negative so calibration can tighten as well "
                              "as widen. Not eq. (20); for comparison only. CQR/WCP always use the "
                              "signed score, per Table I.")
    parser.add_argument("--normalize", choices=["minmax", "zscore"], default=None,
                         help="Input scaling fit on the training windows (default minmax, eq. (1) of [6]).")
    parser.add_argument("--dilation-base", type=int, default=None,
                         help="Dilation of TCN block b is base**b (default 4 -> {1,4,16,64}, receptive "
                              "field 341). Base 2 gives the textbook {1,2,4,8} and a field of only 61.")
    parser.add_argument("--train-stride", type=int, default=None,
                         help="Stride between training/validation windows (default 1). Consecutive "
                              "stride-1 windows overlap by window_size-1 samples, so a larger stride "
                              "cuts epoch cost with little information loss.")
    parser.add_argument("--calib-min-windows", type=int, default=None,
                         help="Target minimum number of calibration windows; the stride is chosen to "
                              "reach it (default 1000). c_alpha resolves no finer than 1/N_cal.")
    parser.add_argument("--calib-stride", type=int, default=None,
                         help="Stride between conformal calibration windows (default: window_size, i.e. "
                              "non-overlapping). Pass 1 to reproduce the fully overlapping calibration set.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    cfg = RTCQRConfig(seed=args.seed)
    if args.no_ltr:
        cfg.lambda_l = 0.0
    if args.max_epochs is not None:
        cfg.max_epochs = args.max_epochs
    if args.patience is not None:
        cfg.patience = args.patience
    if args.window_size is not None:
        cfg.window_size = args.window_size
    if args.resample_dt is not None:
        cfg.resample_dt_s = args.resample_dt or None
    if args.rated_capacity is not None:
        cfg.rated_capacity_ah = args.rated_capacity
    if args.calib_stride is not None:
        cfg.calib_stride = args.calib_stride
    if args.calib_min_windows is not None:
        cfg.calib_min_windows = args.calib_min_windows
    if args.signed_score:
        cfg.signed_score = True
    if args.train_stride is not None:
        cfg.stride = args.train_stride
    if args.normalize is not None:
        cfg.normalize = args.normalize
    if args.dilation_base is not None:
        cfg.dilation_base = args.dilation_base

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[rtcqr.train] device={device}")

    if args.download:
        from rtcqr.data import download_lg_hg2
        data_root = download_lg_hg2(args.dataset_slug)
    else:
        if args.data_root is None:
            raise SystemExit("Provide --data-root <path> or pass --download to fetch it via kagglehub.")
        data_root = args.data_root

    splits = build_windows(cfg, data_root, current_sign=args.current_sign, include_all=args.include_all,
                            exclude_measurement_ids=args.exclude_measurement_ids, split_mode=args.split_mode)

    t0 = time.time()
    model = train_model(cfg, splits, device)
    print(f"[rtcqr.train] training finished in {time.time() - t0:.1f}s")

    results = evaluate(cfg, model, splits, device, calibrators=args.calibrators)
    print_results_table(results)

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "rtcqr_model.pt"))
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({"config": asdict(cfg), "results": results}, f, indent=2)
    print(f"[rtcqr.train] saved model and results to {args.output_dir}/")


if __name__ == "__main__":
    main()
