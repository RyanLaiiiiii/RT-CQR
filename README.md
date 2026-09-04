# RT-CQR

PyTorch implementation of **Risk-Aware Time-Adaptive Conformal Quantile
Regression (RT-CQR)** for SoC prediction-interval estimation, applied to the
LG 18650HG2 Li-ion battery dataset, following:

> H.-Y. Su and T.-C. Lin, "Risk-Aware Time-Adaptive Conformal Quantile
> Regression for SoC Interval Estimation in Battery Energy Storage
> Systems," *IEEE Transactions on Energy Conversion*, 2026.

## What's implemented

- `rtcqr/model.py` — TCN backbone (4 residual blocks, 64 channels, kernel
  size 3, dropout 0.1) with a multi-quantile output head, eq. (14)-(15).
- `rtcqr/losses.py` — risk-aligned composite quantile loss: pinball loss +
  quantile-crossing penalty + lower-tail regularizer, eq. (16)-(18).
- `rtcqr/conformal.py` — violation-weighted, time-decayed nonconformity
  score and weighted-empirical-quantile calibration (VW-TAC), eq. (19)-(28),
  with both a static (calibrate-once) and an online/rolling variant.
- `rtcqr/baselines.py` — CQR and WCP as special cases of the same
  calibrator, so the comparison in `train.py` isolates the effect of
  violation weighting rather than comparing unrelated codebases.
- `rtcqr/metrics.py` — LVR, AIW, ACE, eq. (29).
- `rtcqr/data.py` — LG 18650HG2 loading: auto-detected column mapping,
  grouping/stitching multi-file measurement runs in true chronological
  order, coulomb-counting SoC computation, uniform-rate resampling,
  chronological train/val/calib/test split, and sliding-window
  construction.
- `train.py` — end-to-end training + calibration + evaluation CLI.

## Setup

```bash
pip install -r requirements.txt
```

## Getting the data

Configure Kaggle API credentials first (`~/.kaggle/kaggle.json` or the
`KAGGLE_USERNAME`/`KAGGLE_KEY` env vars), then either let `train.py`
download it for you:

```bash
python train.py --download --output-dir outputs
```

or download it yourself and point `train.py` at the local copy:

```python
import kagglehub
path = kagglehub.dataset_download("aditya9790/lg-18650hg2-liion-battery-data")
print("Path to dataset files:", path)
```

```bash
python train.py --data-root "$path" --output-dir outputs
```

### File format

Confirmed against 6 sample files (`585_C20DisCh`, `589_Charge1`,
`589_Cap_1C`, `589_HWFET`, `589_Mixed1`, `590_PausCycl`), each CSV in this
dataset is a raw battery-cycler export: ~20-30 lines of `Key,Value`
metadata (Measurement ID, Test section, battery name, nominal capacity,
...), then the real header row (`Time Stamp,Step,Status,...,Voltage,
Current,Temperature,Capacity,...`), then a units row
(`,,,,,,,,[V],[A],[C],[Ah],...`), then the data. `rtcqr/data.py` locates
and skips the preamble/units rows automatically -- you don't need to
pre-clean the files.

**One CSV is not one independent test.** Filenames follow
`<measurement_id>_<TestSection>.csv`, and files sharing the same
`Measurement ID` are chronologically contiguous slices of a single
continuous cycler run -- e.g. `589_Charge1` (11/29 18:53-20:59) is
immediately followed by `589_Cap_1C` (11/29 20:59-21:59), and later
`589_HWFET` and `589_Mixed1`, all part of measurement 589, in that order.
(The file-level `Start Time`/`End Time` metadata is identical across all
of them -- it's the whole measurement's span, not the individual
section's, so true ordering has to come from each row's own timestamp.)
`load_lg_hg2_dataframe` groups files by Measurement ID, concatenates them
in timestamp order, and computes SoC via coulomb counting *once*, across
the whole reconstructed run -- so `SoC(0) = 1.0` is only assumed at the
start of a run's earliest available section, not at the start of every
individual file. (A tempting shortcut would be to use the per-row
cumulative `Capacity[Ah]` column instead of re-integrating `Current`, but
it turned out to reset to 0 at internal step boundaries -- confirmed on
`589_Charge1`, where it drops from 0.01126 back to 0.00000 partway
through the same file -- so only `Current` is used.)

If your copy of the dataset is missing an intermediate section for some
measurement, there will be a real time gap in the stitched run where SoC
can't be tracked (no current reading for that period); `load_lg_hg2_dataframe`
prints a warning naming the gap and both bounds the coulomb-counting error
(a step's contribution is capped at `max_gap_s`, default 300s, rather than
extrapolating one sample's current across the whole gap) and splits the
run into separate windowing segments there, so a training window never
silently spans the hole.

Some test sections are static characterization tests (e.g. `C20DisCh` =
C/20 constant-current discharge for an OCV-SoC curve, `Cap_1C` = a 1C
capacity check) rather than the dynamic drive-cycle profiles (`HWFET`,
`UDDS`, `LA92`, `US06`, `Mixed*`, ...) the paper evaluates on. Their
current is still used for SoC continuity, but by default they're excluded
from the windows used for training/evaluation -- `load_lg_hg2_dataframe`
excludes test sections matching `c20`, `cap`, `ocv`, `hppc`, `pulse`,
`eis`, `reset` (case-insensitive); pass `--include-all` to `train.py` (or
`exclude_patterns=None` to `load_lg_hg2_dataframe`) to keep everything.

Because sections are logged at very different native rates (~60s between
samples during the slow `Charge1`/`Cap_1C` sections, vs. ~0.1s during the
dynamic `HWFET`/`Mixed1` sections), each reconstructed run is resampled
onto a uniform grid (`resample_dt_s`, default 1.0s / 1 Hz) before
windowing, so `--window-size N` consistently means "N seconds of
history" regardless of which section it falls in. Pass `--resample-dt 0`
to disable resampling and window over native, irregular sampling instead.

Inspect what gets detected, stitched, and parsed before training:

```bash
python -m rtcqr.data inspect /path/to/lg_hg2               # drive-cycle segments only
python -m rtcqr.data inspect /path/to/lg_hg2 --include-all  # + characterization tests
```

This prints, per reconstructed windowing segment: which measurement/test
sections it came from, row count, time span, voltage/current range, and
the SoC range and start/end values, so you can sanity-check the whole
pipeline without opening any CSV by hand. If a file's
voltage/current/temperature/time columns aren't picked up correctly, pass
`column_overrides` to `load_lg_hg2_dataframe` (see `rtcqr/data.py`)
mapping that file's basename to the right column names.

If the dataset already ships a SoC column it is used as-is (rescaled from
percent if needed); otherwise SoC is obtained by coulomb counting the
current signal as described above. This assumes each measurement's
earliest available section starts right after a full charge
(`SoC(0) = 1.0`; confirmed on the sample data: `589_Charge1`, the first
chronological section of measurement 589, starts at ~4.19V, near the LG
HG2's 4.2V max) and that `current > 0` means charging (also confirmed:
current is negative throughout the `585_C20DisCh` discharge test and SoC
decreases as expected). If a different file's computed SoC trends the
wrong way, pass `--current-sign -1.0`; if a measurement's earliest
available section doesn't actually start from a full charge (e.g. because
its true first section wasn't downloaded), pass a different `soc_initial`
to `load_lg_hg2_dataframe`, or accept that the absolute SoC scale for that
measurement will be approximate while its relative dynamics stay
informative.

## Training and evaluation

```bash
python train.py --data-root /path/to/lg_hg2 --output-dir outputs
```

This trains the RT-CQR* configuration from Table I (`lambda_nc=1.0,
lambda_l=0.1, zeta=0.98, gamma=1.0`), calibrates at 90% and 95% nominal
coverage on a held-out calibration split, and reports LVR / AIW / ACE
(Table II-style) for RT-CQR, CQR, and WCP calibration on the same trained
quantile model. Results and the trained weights are written to
`outputs/results.json` and `outputs/rtcqr_model.pt`.

### Reproducing the ablation study (Table IV)

```bash
# RT-CQR w/o lower-tail regularization (LTR): retrain with lambda_l=0
python train.py --data-root /path/to/lg_hg2 --no-ltr --output-dir outputs_no_ltr

# RT-CQR w/o violation-weighted time-adaptive calibration (VW-TAC):
# same trained model, plain CQR calibration
python train.py --data-root /path/to/lg_hg2 --calibrators cqr --output-dir outputs_no_vwtac
```

### Useful flags

- `--window-size N` — length (in resampled seconds) of the input V/I/T
  history window used to predict the current SoC (default 100).
- `--resample-dt S` — uniform resampling interval in seconds before
  windowing (default 1.0); pass 0 to window over native sampling instead.
- `--current-sign {1,-1}` — coulomb-counting sign convention (see above).
- `--include-all` — keep static characterization test sections instead of
  only dynamic drive-cycle profiles.
- `--calibrators rtcqr cqr wcp` — which calibration methods to report.
- `--max-epochs`, `--patience`, `--seed` — training controls.

## Method summary

1. **Risk-aligned quantile learning** (eq. 16-18): a TCN jointly predicts
   the quantile set `T = {0.025, 0.05, 0.10, 0.15, 0.85, 0.90, 0.95, 0.975}`
   using a composite pinball loss with a quantile-crossing penalty and an
   extra term penalizing `[SoC_min - q_{tau_l}]_+` at `tau_l = min(T)`, to
   directly discourage predicted lower bounds from exceeding SoC_min when
   the true SoC does not.
2. **Violation-weighted time-adaptive conformal calibration** (eq. 19-28):
   an asymmetric nonconformity score upweights lower-bound violations
   (`w_l^(1) >= w_l^(0) >= w_u`), and calibration samples are exponentially
   time-decayed (`zeta`) and further upweighted when they were violations
   (`gamma`), so the calibrated interval `[q_tl - c_alpha, q_tu + c_alpha]`
   adapts to recent, safety-critical evidence.
3. Together these target the PI-induced upper bound on minimum-SoC
   violation risk, `E[p_t] <= P(inf(PI_t) < SoC_min) + alpha` (eq. 13),
   by reducing the lower-tail risk term through training and controlling
   the miscoverage term `alpha` through calibration.
