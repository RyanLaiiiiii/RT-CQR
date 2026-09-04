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

### If column auto-detection doesn't match your copy of the dataset

Kaggle re-uploads of this dataset vary slightly in header naming. Inspect
what was detected before training:

```bash
python -m rtcqr.data inspect /path/to/lg_hg2
```

If a file's voltage/current/temperature/time columns aren't picked up
correctly, pass `column_overrides` to `load_lg_hg2_dataframe` (see
`rtcqr/data.py`) mapping that file's basename to the right column names.

If the dataset already ships a SoC column it is used as-is (rescaled from
percent if needed); otherwise SoC is obtained by coulomb counting against
the LG HG2's rated 3.0 Ah capacity, assuming `current > 0` means charging.
If a file's computed SoC trends the wrong way (e.g. increasing SoC during a
drive-cycle discharge), pass `--current-sign -1.0`.

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
