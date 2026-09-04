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
- `rtcqr/data.py` — LG 18650HG2 loading, auto-detected column mapping,
  coulomb-counting SoC computation, chronological train/val/calib/test
  split, and sliding-window construction.
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

Confirmed against an actual sample file (`585_C20DisCh.csv`), each CSV in
this dataset is a raw battery-cycler export: ~20-30 lines of
`Key,Value` metadata (battery name, nominal capacity, ...), then the real
header row (`Time Stamp,Step,Status,...,Voltage,Current,Temperature,
Capacity,...`), then a units row (`,,,,,,,,[V],[A],[C],[Ah],...`), then the
data. `rtcqr/data.py` locates and skips the preamble/units rows
automatically -- you don't need to pre-clean the files.

Filenames follow `<measurement_id>_<TestSection>.csv`. Some sections are
static characterization tests (e.g. `C20DisCh` = C/20 constant-current
discharge, used to build an OCV-SoC curve) rather than the dynamic
drive-cycle profiles (`UDDS`, `LA92`, `US06`, `Mixed*`, ...) the paper
evaluates on. By default, `discover_csv_files`/`load_lg_hg2_dataframe`
exclude filenames matching `c20`, `ocv`, `hppc`, `pulse`, `eis`, `reset`
(case-insensitive); pass `--include-all` to `train.py` (or
`exclude_patterns=None` to `load_lg_hg2_dataframe`) to keep everything.

Inspect what gets detected and parsed, per file, before training:

```bash
python -m rtcqr.data inspect /path/to/lg_hg2            # drive-cycle files only
python -m rtcqr.data inspect /path/to/lg_hg2 --include-all  # + characterization tests
```

This prints the detected column mapping plus a parsed summary (row count,
time span, voltage/current range) so you can sanity-check a file without
opening it. If a file's voltage/current/temperature/time columns aren't
picked up correctly, pass `column_overrides` to `load_lg_hg2_dataframe`
(see `rtcqr/data.py`) mapping that file's basename to the right column
names.

If the dataset already ships a SoC column it is used as-is (rescaled from
percent if needed). Otherwise, if it has a cumulative `Capacity[Ah]` column
(as in the sample file), SoC is derived from that directly, since it's the
cycler's own coulomb counter and more accurate than re-integrating current
against timestamps. Failing that, SoC is obtained by coulomb counting the
current signal. All three assume each test-section file starts right after
a full charge (`SoC(0) = 1.0`, confirmed on the sample file: SoC starts at
0.999 and decreases smoothly to ~0.13 over the discharge) and that
`current > 0` means charging (also confirmed on the sample: current and
`Capacity` are both negative throughout the discharge, and SoC decreases as
expected). If a different file's computed SoC trends the wrong way, pass
`--current-sign -1.0`; if a file doesn't start from a full charge, pass a
different `soc_initial` to `load_lg_hg2_dataframe`.

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

- `--window-size N` — length (in samples) of the input V/I/T history window
  used to predict the current SoC (default 100).
- `--current-sign {1,-1}` — coulomb-counting sign convention (see above).
- `--calibrators rtcqr cqr wcp` — which calibration methods to report.
- `--max-epochs`, `--patience`, `--seed` — training controls.

## Smoke-testing without the real dataset

`tests/make_synthetic_dataset.py` generates a tiny synthetic dataset with
the same folder/column layout as the Kaggle mirror, useful for verifying
the pipeline runs end-to-end without network access:

```bash
python tests/make_synthetic_dataset.py /tmp/synthetic_lg_hg2
python train.py --data-root /tmp/synthetic_lg_hg2 --window-size 20 --max-epochs 3
```

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
