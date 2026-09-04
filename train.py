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
from dataclasses import asdict, replace
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
    load_lg_hg2_dataframe,
    make_windows,
    order_chronologically,
    segment_split,
)
from rtcqr.losses import composite_quantile_loss
from rtcqr.metrics import lower_violation_rate, summarize
from rtcqr.model import TCNQuantileNet


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Smallest val_loss decrease treated as real progress. Shared by the LR
# scheduler and the early-stopping counter so they cannot disagree.
_MIN_VAL_IMPROVEMENT = 1e-6
def parse_capacity_overrides(values) -> Dict[float, float]:
    """Parse ``TEMP:AH`` pairs into {condition_degC: capacity_ah}.

    Accepts a repeated flag and/or comma-separated pairs, and takes the
    dataset's own ``n20`` spelling for sub-zero temperatures as well as
    ``-20``. The ``n`` form is worth having: argparse reads a bare
    ``-20:1.70`` as an option rather than a value, so ``--capacity-override
    n20:1.70`` works positionally where the minus sign needs
    ``--capacity-override=-20:1.70``.
    """
    out: Dict[float, float] = {}
    for chunk in values or []:
        for pair in str(chunk).split(","):
            pair = pair.strip()
            if not pair:
                continue
            if ":" not in pair:
                raise argparse.ArgumentTypeError(
                    f"--capacity-override expects TEMP:AH (e.g. 40:2.75), got {pair!r}")
            temp_s, cap_s = pair.split(":", 1)
            temp_s = temp_s.strip()
            if temp_s[:1].lower() == "n":
                temp_s = "-" + temp_s[1:]
            try:
                temp, cap = float(temp_s), float(cap_s)
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f"--capacity-override expects numbers, got {pair!r}")
            if not cap > 0:
                raise argparse.ArgumentTypeError(f"capacity must be positive, got {pair!r}")
            out[temp] = cap
    return out



def _report_split_conditions(train, val, calib, test) -> None:
    """Print each split's per-condition segment counts.

    Conformal coverage assumes calib and test are exchangeable. This
    dataset's error distribution is strongly temperature-dependent, so a
    condition appearing in test but not in calib silently voids the
    guarantee for those windows -- print the table so it is visible rather
    than buried.
    """
    named = [("train", train), ("val", val), ("calib", calib), ("test", test)]
    conds = sorted({bf.condition for _, split in named for bf in split},
                   key=lambda c: (c is None, c))
    header = "".join(f"{str(c) + 'C':>9}" for c in conds)
    print(f"[rtcqr.train] segments per condition\n{'':<8}{header}")
    for name, split in named:
        counts = {c: sum(bf.condition == c for bf in split) for c in conds}
        print(f"{name:<8}" + "".join(f"{counts[c]:>9}" for c in conds))
    missing = {bf.condition for bf in test} - {bf.condition for bf in calib}
    if missing:
        print(f"[rtcqr.train] WARNING: condition(s) {sorted(missing, key=str)} appear in test but not in "
              f"calib. Conformal coverage is not guaranteed for those windows.")


def build_windows(cfg: RTCQRConfig, data_root: str, current_sign: float, include_all: bool = False,
                   exclude_measurement_ids: Optional[List[str]] = None, split_mode: str = "segment"):
    include_patterns = None if include_all else _DEFAULT_INCLUDE_PATTERNS
    files = load_lg_hg2_dataframe(
        data_root, rated_capacity_ah=cfg.rated_capacity_ah, current_sign=current_sign,
        include_patterns=include_patterns, resample_dt_s=cfg.resample_dt_s,
        capacity_overrides=cfg.capacity_overrides, min_soc_range=cfg.min_soc_range,
    )

    if exclude_measurement_ids:
        before = len(files)
        excluded = {str(x) for x in exclude_measurement_ids}
        files = [bf for bf in files if not any(f"measurement {mid} " in bf.path for mid in excluded)]
        print(f"[rtcqr.train] Excluded {before - len(files)} segment(s) from measurement IDs {sorted(excluded)}")

    print(f"[rtcqr.train] Loaded {len(files)} windowing segment(s) from {data_root}")

    if split_mode == "segment":
        # Most segments in this dataset are short, single charge/discharge
        # cycles (SoC ~1.0 -> some low point over a few hours). Slicing each
        # one chronologically would systematically give train the high-SoC
        # early portion and test the low-SoC late portion of every cycle --
        # confirmed on the full dataset: calib mean SoC 0.33 vs. test mean
        # SoC 0.25, and a 24% quantile-crossing rate on test vs. 3% on
        # calib. Assigning whole segments to a split instead keeps each
        # split's SoC distribution representative.
        train_frames, val_model_frames, calib_frames, test_frames = segment_split(
            files, cfg.train_frac, cfg.val_frac, cfg.val_calib_fraction, seed=cfg.seed,
            stratify_by_condition=cfg.stratify_by_condition,
        )
    else:
        train_frames, val_frames, test_frames = chronological_split(files, cfg.train_frac, cfg.val_frac)
        # further split each validation segment chronologically into model-val / calibration
        val_model_frames, calib_frames = [], []
        for bf in val_frames:
            n_val = len(bf.frame)
            n_cal = int(round(n_val * cfg.val_calib_fraction))
            val_model_frames.append(replace(bf, frame=bf.frame.iloc[:n_val - n_cal].reset_index(drop=True)))
            calib_frames.append(replace(bf, frame=bf.frame.iloc[n_val - n_cal:].reset_index(drop=True)))

    # The conformal time decay (conformal.time_decay_weights) reads buffer
    # position as time, weighting the last entries as "now". Segments arrive
    # here grouped by measurement and then randomly permuted by
    # segment_split, so without this sort the decay would treat an arbitrary
    # segment's tail as the most recent evidence and WCP/RT-CQR's
    # time-adaptive component would be weighting an essentially random subset.
    calib_frames = order_chronologically(calib_frames)
    _report_split_conditions(train_frames, val_model_frames, calib_frames, test_frames)

    X_train, y_train = make_windows(train_frames, cfg.window_size, cfg.stride)
    X_val, y_val = make_windows(val_model_frames, cfg.window_size, cfg.stride)
    X_calib, y_calib = make_windows(calib_frames, cfg.window_size, stride=1)
    X_test, y_test = make_windows(test_frames, cfg.window_size, stride=1)

    scaler = Standardizer().fit(X_train)
    X_train, X_val, X_calib, X_test = (scaler.transform(x) for x in (X_train, X_val, X_calib, X_test))

    print(f"[rtcqr.train] windows: train={len(y_train)} val={len(y_val)} calib={len(y_calib)} test={len(y_test)}")
    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "calib": (X_calib, y_calib),
        "test": (X_test, y_test),
    }


def _fit(cfg: RTCQRConfig, splits, device: torch.device, loss_fn, num_workers: int = 2,
         tag: str = "rtcqr") -> TCNQuantileNet:
    """Shared training loop: AMP, LR schedule, early stopping, best-state restore.

    `loss_fn(y_true, model_output) -> scalar tensor` is the only thing that
    differs between the quantile model and the point-estimation baseline.
    """
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    # pin_memory + non_blocking .to() overlap the host->device copy with
    # compute. num_workers>0 moves batch assembly off the main process, but
    # for an already-in-RAM TensorDataset it mostly adds per-batch pickling
    # and IPC, so it defaults off (see --num-workers) -- measure before
    # raising it.
    pin_memory = device.type == "cuda"
    loader_kwargs = dict(num_workers=num_workers, pin_memory=pin_memory,
                         persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)

    model = TCNQuantileNet(
        in_channels=cfg.in_channels,
        quantile_levels=cfg.quantile_levels,
        num_blocks=cfg.num_blocks,
        channels=cfg.channels,
        kernel_size=cfg.kernel_size,
        dropout=cfg.dropout,
    ).to(device)
    model.warn_if_window_exceeds_receptive_field(cfg.window_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Same improvement threshold as the early-stopping test below (absolute
    # 1e-6). With ReduceLROnPlateau's default relative 1e-4 the two disagree:
    # a val_loss around 0.05 improving by 2e-6 counts as progress for early
    # stopping but as a plateau for the scheduler, so the LR keeps halving
    # while the patience counter keeps resetting and the run never stops
    # early -- it just burns all max_epochs at a vanishing learning rate.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
        threshold=_MIN_VAL_IMPROVEMENT, threshold_mode="abs",
    )

    # Mixed precision only applies (and only helps) on CUDA; autocast/GradScaler
    # are harmless no-ops with enabled=False, so this is safe on CPU too.
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        train_loss, train_seen = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss = loss_fn(yb, model(xb))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * xb.size(0)
            train_seen += xb.size(0)
        # Divide by the samples actually iterated, not len(train_ds):
        # drop_last=True discards the final partial batch.
        train_loss /= max(train_seen, 1)

        model.eval()
        val_loss, val_seen = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    loss = loss_fn(yb, model(xb))
                val_loss += loss.item() * xb.size(0)
                val_seen += xb.size(0)
        val_loss /= max(val_seen, 1)
        scheduler.step(val_loss)

        lr = optimizer.param_groups[0]["lr"]
        print(f"[rtcqr.train:{tag}] epoch {epoch:03d}  train_loss={train_loss:.5f}  "
              f"val_loss={val_loss:.5f}  lr={lr:.2e}")

        if val_loss < best_val - _MIN_VAL_IMPROVEMENT:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print(f"[rtcqr.train:{tag}] early stopping at epoch {epoch} (best val_loss={best_val:.5f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_model(cfg: RTCQRConfig, splits, device: torch.device, num_workers: int = 0) -> TCNQuantileNet:
    return _fit(
        cfg, splits, device, num_workers=num_workers, tag="rtcqr",
        loss_fn=lambda y, q: composite_quantile_loss(
            y, q, cfg.quantile_levels, cfg.soc_min, cfg.lambda_nc, cfg.lambda_l, cfg.tau_l_index
        ),
    )


def train_point_model(cfg: RTCQRConfig, splits, device: torch.device, num_workers: int = 0) -> TCNQuantileNet:
    """Train the deterministic point-estimation baseline of Table II.

    Same TCN backbone, one output, MSE loss. The paper reports its LVR
    (0.084 on this dataset) as the motivating comparison -- it is the row
    showing that a model without uncertainty carries a persistent
    minimum-SoC violation risk -- so without it the interval methods'
    LVR numbers have nothing to be better *than*.

    Implemented by reusing TCNQuantileNet with a single "quantile" level:
    with one output the head is just `base` (no softplus increments), so
    this is exactly the same architecture with a scalar output.
    """
    point_cfg = replace(cfg, quantile_levels=[0.5])
    model = _fit(point_cfg, splits, device, num_workers=num_workers,
                 loss_fn=lambda y, q: torch.nn.functional.mse_loss(q[:, 0], y),
                 tag="point")
    return model


@torch.no_grad()
def predict_point(model: TCNQuantileNet, X: np.ndarray, device: torch.device,
                  batch_size: int = 256) -> np.ndarray:
    return predict_quantiles(model, X, device, batch_size)[:, 0]


@torch.no_grad()
def predict_quantiles(model: TCNQuantileNet, X: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    preds = []
    for start in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[start:start + batch_size]).to(device)
        preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds, axis=0)


def evaluate(cfg: RTCQRConfig, model: TCNQuantileNet, splits, device, calibrators: List[str]) -> Dict:
    (X_calib, y_calib), (X_test, y_test) = splits["calib"], splits["test"]
    q_calib = predict_quantiles(model, X_calib, device)
    q_test = predict_quantiles(model, X_test, device)

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for alpha in cfg.pi_alphas:
        idx_l, idx_u = cfg.quantile_bounds(alpha)
        pi_key = f"{int(round((1 - alpha) * 100))}%"
        results[pi_key] = {}

        for name in calibrators:
            fsc = cfg.finite_sample_correction
            if name == "rtcqr":
                calibrator = make_rtcqr_calibrator(cfg.soc_min, cfg.zeta, cfg.gamma, cfg.wl0, cfg.wl1,
                                                   cfg.wu, finite_sample_correction=fsc)
            elif name == "cqr":
                calibrator = make_cqr_calibrator(cfg.soc_min, finite_sample_correction=fsc)
            elif name == "wcp":
                calibrator = make_wcp_calibrator(cfg.soc_min, zeta=cfg.zeta, finite_sample_correction=fsc)
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
        header = f"{'method':<10}{'LVR':>10}{'AIW':>10}{'ACE':>10}"
        print(header)
        if point_lvr is not None:
            print(f"{'Point':<10}{point_lvr:>10.3f}{'-':>10}{'-':>10}")
        for name, m in per_calib.items():
            print(f"{name:<10}{m['LVR']:>10.3f}{m['AIW']:>10.3f}{m['ACE']:>10.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=str, default=None, help="Path to a local copy of the LG 18650HG2 dataset.")
    parser.add_argument("--download", action="store_true", help="Download the dataset via kagglehub first.")
    parser.add_argument("--dataset-slug", type=str, default="aditya9790/lg-18650hg2-liion-battery-data")
    parser.add_argument("--current-sign", type=float, default=1.0, help="1.0 if I>0 means charging, -1.0 if I>0 means discharging.")
    parser.add_argument("--include-all", action="store_true",
                         help="Include static characterization test sections (C/20, OCV, HPPC, ...) instead of "
                              "only dynamic drive-cycle profiles.")
    parser.add_argument("--exclude-measurement-ids", nargs="+", default=None,
                         help="Drop entire Measurement IDs from the windowing segments, "
                              "e.g. --exclude-measurement-ids 590 556")
    parser.add_argument("--split-mode", choices=["segment", "chronological"], default="segment",
                         help="'segment' (default) randomly assigns whole segments to train/val/calib/test, "
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--num-workers", type=int, default=0,
                         help="DataLoader worker processes for batch assembly (default 0). The dataset is "
                              "already a fully in-RAM TensorDataset, so workers mostly add per-batch pickling "
                              "and IPC rather than removing a bottleneck; raise it only if you measure a gain.")
    parser.add_argument("--capacity-override", action="append", metavar="TEMP:AH",
                         help="Override a condition's measured capacity, e.g. --capacity-override 40:2.75. "
                              "Repeatable, and accepts comma-separated pairs. Use for a Cap_1C section that "
                              "ran a normal duration from a full charge but stopped before the discharge "
                              "finished -- run diag40.py to identify one. For sub-zero temperatures write "
                              "n20:1.70, or use the --capacity-override=-20:1.70 form (a bare -20:1.70 is "
                              "read as an option, not a value).")
    parser.add_argument("--min-soc-range", type=float, default=None, metavar="SPAN",
                         help="Drop segments whose SoC spans less than SPAN (try 0.02). A drive cycle sitting "
                              "in the saturated full-charge region has a near-constant label while V/I/T vary, "
                              "so it teaches the model nothing and gives the calibrator degenerate scores.")
    parser.add_argument("--point-baseline", action="store_true",
                         help="Also train the deterministic point-estimation model and report its LVR "
                              "(the 'Point' row of Table II).")
    parser.add_argument("--no-stratify", action="store_true",
                         help="Split segments without stratifying by ambient temperature. Not recommended: "
                              "unstratified splits routinely leave a temperature in test that calib never "
                              "saw, which voids the conformal coverage guarantee for those windows.")
    parser.add_argument("--paper-quantile", action="store_true",
                         help="Use eq. (25)-(26)'s plain empirical quantile instead of split conformal's "
                              "ceil((1-a)(n+1)) order statistic. Reproduces the paper exactly; undercovers.")
    args = parser.parse_args()

    cfg = RTCQRConfig(seed=args.seed)
    cfg.capacity_overrides = parse_capacity_overrides(args.capacity_override)
    if cfg.capacity_overrides:
        print(f"[rtcqr.train] capacity overrides: {cfg.capacity_overrides}")
    if args.min_soc_range is not None:
        cfg.min_soc_range = args.min_soc_range
    if args.no_stratify:
        cfg.stratify_by_condition = False
    if args.paper_quantile:
        cfg.finite_sample_correction = False
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
    model = train_model(cfg, splits, device, num_workers=args.num_workers)
    print(f"[rtcqr.train] training finished in {time.time() - t0:.1f}s")

    point_lvr = None
    if args.point_baseline:
        point_model = train_point_model(cfg, splits, device, num_workers=args.num_workers)
        y_test = splits["test"][1]
        point_pred = predict_point(point_model, splits["test"][0], device)
        point_lvr = lower_violation_rate(y_test, point_pred, cfg.soc_min)

    results = evaluate(cfg, model, splits, device, calibrators=args.calibrators)
    print_results_table(results, point_lvr=point_lvr)

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "rtcqr_model.pt"))
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({"config": asdict(cfg), "results": results, "point_lvr": point_lvr,
                   "torch_version": torch.__version__}, f, indent=2)
    print(f"[rtcqr.train] saved model and results to {args.output_dir}/")


if __name__ == "__main__":
    main()
