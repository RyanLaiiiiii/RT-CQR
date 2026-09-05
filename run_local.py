#!/usr/bin/env python3
"""One-shot local runner: setup, verify, and (optionally) train RT-CQR
outside Colab, on a machine with its own Python and GPU.

Mirrors the notebook flow -- install deps, get the dataset, run the test
suite, check the reconstructed SoC labels, then train -- as a single script
that does not depend on a Colab session's lifetime or its free-tier GPU
quota. Each stage stops the pipeline on failure (pass --force to continue
past a failing test suite or a failing label check anyway).

Usage:
    # Full pipeline: install deps, download via kagglehub, verify, train.
    python run_local.py --train --capacity-override 40:2.75 --min-soc-range 0.02

    # Already have the data locally; skip the download.
    python run_local.py --data-root /path/to/lg_hg2 --train

    # Just verify the environment and labels; print the train.py command
    # to run yourself instead of running it automatically.
    python run_local.py --data-root /path/to/lg_hg2

Do not copy --capacity-override 40:2.75 blindly onto a different copy of the
dataset: it is specific to a Cap_1C section that stopped early on this copy
of the McMaster/Kaggle data, identified with diag40.py (see README.md's
"SoC labels and the capacity denominator" section). Run
`python check_soc.py <data_root>` without it first and only add an override
if that check names a condition as suspect.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def run(cmd: list) -> subprocess.CompletedProcess:
    print(f"\n$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=REPO_ROOT)


def step(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def install_deps(skip: bool) -> bool:
    step("Installing dependencies")
    if skip:
        print("skipped (--skip-install)")
        return True
    return run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"]).returncode == 0


def report_environment() -> None:
    step("Environment")
    print(f"python: {sys.version.split()[0]} ({sys.executable})")
    try:
        import torch
        cuda = torch.cuda.is_available()
        extra = f"  device={torch.cuda.get_device_name(0)}" if cuda else ""
        print(f"torch:  {torch.__version__}  cuda_available={cuda}{extra}")
        if not cuda:
            print("        No GPU visible to torch: training runs on CPU. The RT-CQR* backbone "
                  "is small (~90K params, 4 blocks x 64 channels), so this is workable, just "
                  "slower per epoch than a GPU -- not a reason to stop.")
    except ImportError:
        print("torch:  not importable (install step above should have failed loudly if this "
              "is unexpected)")


def kaggle_credentials_present() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return os.path.isfile(os.path.expanduser("~/.kaggle/kaggle.json"))


def download_dataset(dataset_slug: str) -> str:
    step("Downloading dataset via kagglehub")
    if not kaggle_credentials_present():
        print("No Kaggle API credentials found. Set them up first:")
        print("  1. https://www.kaggle.com/settings -> Create New Token -> downloads kaggle.json")
        print("  2. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && "
              "chmod 600 ~/.kaggle/kaggle.json")
        print("  (or set the KAGGLE_USERNAME / KAGGLE_KEY environment variables instead)")
        print("Or skip this step entirely with --data-root if you already have a local copy.")
        sys.exit(1)
    import kagglehub
    path = kagglehub.dataset_download(dataset_slug)
    print(f"data root: {path}")
    return path


def run_tests(skip: bool, force: bool) -> bool:
    step("Running tests (pytest)")
    if skip:
        print("skipped (--skip-tests)")
        return True
    ok = run([sys.executable, "-m", "pytest"]).returncode == 0
    if not ok and not force:
        print("\nTests failed. Fix the environment before trusting anything downstream "
              "(a broken rtcqr install can silently produce different SoC labels), or pass "
              "--force to continue anyway.")
    return ok or force


def run_check_soc(data_root: str, capacity_overrides, min_soc_range, skip: bool, force: bool) -> bool:
    step("Checking reconstructed SoC labels (check_soc.py)")
    if skip:
        print("skipped (--skip-check)")
        return True
    cmd = [sys.executable, "check_soc.py", data_root]
    for pair in capacity_overrides or []:
        cmd += ["--capacity-override", pair]
    if min_soc_range is not None:
        cmd += ["--min-soc-range", str(min_soc_range)]
    ok = run(cmd).returncode == 0
    if not ok and not force:
        print("\ncheck_soc.py flagged a problem with the labels (see above). For a capacity "
              "that falls as temperature rises, `python diag40.py <data_root>` explains why "
              "and suggests a --capacity-override value. Re-run this check with that override "
              "before training, or pass --force to train on the labels as they are.")
    return ok or force


def build_train_command(args: argparse.Namespace, data_root: str) -> list:
    cmd = [sys.executable, "train.py", "--data-root", data_root,
           "--window-size", str(args.window_size), "--output-dir", args.output_dir]
    for pair in args.capacity_override or []:
        cmd += ["--capacity-override", pair]
    if args.min_soc_range is not None:
        cmd += ["--min-soc-range", str(args.min_soc_range)]
    if args.point_baseline:
        cmd += ["--point-baseline"]
    if args.max_epochs is not None:
        cmd += ["--max-epochs", str(args.max_epochs)]
    if args.patience is not None:
        cmd += ["--patience", str(args.patience)]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=None,
                    help="Local copy of the dataset. Omit to download via kagglehub.")
    ap.add_argument("--dataset-slug", default="aditya9790/lg-18650hg2-liion-battery-data")
    ap.add_argument("--capacity-override", action="append", metavar="TEMP:AH",
                    help="Forwarded to check_soc.py and train.py. Repeatable, comma-separated "
                         "pairs accepted. Dataset-specific -- see the module docstring.")
    ap.add_argument("--min-soc-range", type=float, default=None, metavar="SPAN",
                    help="Forwarded to check_soc.py and train.py; drops near-constant-SoC "
                         "segments (try 0.02).")
    ap.add_argument("--window-size", type=int, default=61,
                    help="Default 61 matches the RT-CQR* backbone's receptive field (4 blocks, "
                         "kernel 3) -- see TCNQuantileNet.receptive_field. A larger window "
                         "carries history the model cannot see.")
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--point-baseline", action="store_true", default=True,
                    help="Also train the deterministic point-estimation baseline (default on).")
    ap.add_argument("--no-point-baseline", dest="point_baseline", action="store_false")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--skip-install", action="store_true", help="Skip `pip install -r requirements.txt`.")
    ap.add_argument("--skip-tests", action="store_true", help="Skip `pytest`.")
    ap.add_argument("--skip-check", action="store_true", help="Skip `check_soc.py`.")
    ap.add_argument("--force", action="store_true",
                    help="Continue past a failing test suite or check_soc.py result instead "
                         "of stopping the pipeline.")
    ap.add_argument("--train", action="store_true",
                    help="Run train.py at the end. Without this flag the script stops after "
                         "verification and prints the train.py command to run yourself.")
    args = ap.parse_args()

    if not install_deps(args.skip_install):
        sys.exit(1)
    report_environment()

    data_root = os.path.abspath(args.data_root) if args.data_root else download_dataset(args.dataset_slug)

    if not run_tests(args.skip_tests, args.force):
        sys.exit(1)
    if not run_check_soc(data_root, args.capacity_override, args.min_soc_range, args.skip_check, args.force):
        sys.exit(1)

    train_cmd = build_train_command(args, data_root)
    if args.train:
        step("Training")
        sys.exit(run(train_cmd).returncode)

    step("Ready")
    print("Environment and labels look good. Run this when ready (or re-run this script with --train):\n")
    print("  " + " ".join(train_cmd))


if __name__ == "__main__":
    main()
