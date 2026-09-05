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
  order on the cycler's millisecond program clock, per-temperature
  capacity measurement, coulomb-counting SoC computation, uniform-rate
  resampling, segment-wise train/val/calib/test split, and sliding-window
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
section's, so true ordering has to come from each row's own clock.)
`load_lg_hg2_dataframe` groups files by Measurement ID, concatenates them
in program-clock order, and computes SoC via coulomb counting *once*,
across the whole reconstructed run -- so `SoC(0) = 1.0` is only assumed at the
start of a run's earliest available section, not at the start of every
individual file. (A tempting shortcut would be to use the per-row
cumulative `Capacity[Ah]` column instead of re-integrating `Current`, but
it turned out to reset to 0 at internal step boundaries -- confirmed on
`589_Charge1`, where it drops from 0.01126 back to 0.00000 partway
through the same file -- so only `Current` is used.)

### Time base: use `Prog Time`, not `Time Stamp`

Order within a measurement comes from the cycler's own program clock
(`Prog Time`, e.g. `06:40:55.195`), not from the wall-clock `Time Stamp`.
Two reasons, both measured on the real dataset:

- **Resolution.** `Time Stamp` is whole-second, but drive cycles are
  logged at 0.1 s, so ten consecutive rows carry the same value.
  Deduplicating on it silently discards 90% of every drive-cycle file
  (`551_UDDS`: 159,646 rows -> 15,966). `Prog Time` is millisecond
  resolution, so the native 0.1 s dynamics survive and `--resample-dt 0.1`
  is meaningful.
- **Integrity.** `Time Stamp` is not always self-consistent. In
  `582_LA92` it jumps between 11/25 21:00 and 11/26 10:11 and back while
  `Prog Time` advances smoothly; `571_Mixed6` jumps 10.4 h mid-file.
  Sorting on it reordered rows and manufactured >300 s "gaps" that split
  single drive cycles into fragments. Sorting on `Prog Time` removes
  those fragments.

The hours field is a running program total, so it exceeds 24 on long runs
(`32:11:19.479`). One file (`549_Charge.csv`) carries an Excel-mangled
`MM:SS.s` variant that cannot represent a multi-hour clock; it is
rejected and that measurement falls back to `Time Stamp`.

Program clocks are per-measurement and *overlap* between measurements
(at 10 degC, `m576` spans 0.018-18.912 h and `m582` spans 7.928-10.451 h),
so the clock is only ever used to order *within* a group -- cross-group
identity still comes from the Measurement ID.

### SoC is referenced to the capacity at the test temperature

`rated_capacity_ah` defaults to `None`, meaning "measure it". For each
temperature the loader integrates that temperature's own `Cap_1C`
section (falling back to `C20DisCh`):

| T (degC) | -20 | -10 | 0 | 10 | 25 | 40 |
|---|---|---|---|---|---|---|
| measured 1C capacity (Ah) | 1.64 | 2.25 | 2.47 | 2.52 | 2.71 | 2.50 |

Using the 3 Ah nameplate instead is not a small error. Every drive cycle
at every temperature ends with the cell at its 2.8 V discharge cut-off --
i.e. empty -- but a fixed 3.0 Ah denominator labels that identical
physical state as SoC 0.12 at 25 degC and SoC **0.44** at -20 degC. With
`soc_min = 0.10` that made SoC dip below `soc_min` *nowhere* in the
dataset, so LVR was identically 0 for every method, the violation
indicator `u_i` was identically 0 (making `wl1` and `gamma` inert, and
RT-CQR's violation weighting a no-op), and the lower-tail regularizer had
nothing to penalize. Normalizing per temperature puts the end of each
drive cycle at a median SoC of 0.041 -- matching the protocol's own
"repeat until 95% of the 1C discharge capacity at the respective
temperature has been discharged" -- and 13.7% of samples now sit below
`soc_min`. Pass `--rated-capacity 3.0` to force the old behaviour.

The running SoC is clipped to `[0, 1]` at *every* step of the coulomb
count, not once at the end. A real cell physically cannot exceed 100%
SoC, and charging current that keeps flowing during CV tapering near full
charge doesn't store energy beyond capacity. This also absorbs the
`SoC(0) = 1.0` assumption being wrong for the three measurements whose
earliest archived section does not start from a full charge
(`m590` at 3.24 V, `m562` at 3.08 V, `m549` at 3.10 V): the first charge
section saturates at 1.0, and by the first drive cycle the trajectory has
re-synchronized. With every segment checked, none now starts below SoC
0.85.

### Sections named as drive cycles that contain no drive cycle

`551_HWFET` is a complete, uncorrupted file whose program clock runs
straight from `551_Charge3` into `551_Charge4` with no room for a drive
cycle in between: the 25 degC HWFET run recorded only its 600 s rest
step, current identically 0 A. A name-based whitelist cannot see this, so
segments are additionally required to carry actual dynamic load
(`min_current_std_a`, default 0.05 A) before being windowed.

If your copy of the dataset is missing an intermediate section for some
measurement, there will be a real time gap in the stitched run where SoC
can't be tracked (no current reading for that period); `load_lg_hg2_dataframe`
prints a warning naming the gap and both bounds the coulomb-counting error
(a step's contribution is capped at `max_gap_s`, default 300s, rather than
extrapolating one sample's current across the whole gap) and splits the
run into separate windowing segments there, so a training window never
silently spans the hole.

Some test sections are static characterization or charge/rest/maintenance
runs (e.g. `C20DisCh`/`Dis_0p5C`/`Dis_2C` = constant-current discharge
characterization, `HPPC` = pulse power characterization, `Cap_1C` = a 1C
capacity check, `Charge*`, `PausCycl`) rather than the dynamic drive-cycle
profiles (`HWFET`, `UDDS`, `LA92`, `US06`, `Mixed*`, ...) the paper
evaluates on. Their current is still used for SoC continuity, but only
sections matching `include_patterns` are kept for the windows used in
training/evaluation -- `load_lg_hg2_dataframe` defaults to a *whitelist*
of the 5 real drive-cycle name patterns above (case-insensitive); pass
`--include-all` to `train.py` (or `include_patterns=None` to
`load_lg_hg2_dataframe`) to keep everything. A blacklist of
characterization-test keywords was tried first but proved fragile (real
data includes section names like `Dis_0p5C` that a blacklist has to keep
growing to catch); whitelisting the small, closed set of real drive-cycle
names is more robust.

If a specific Measurement ID still looks wrong after inspecting it (see
below), `train.py --exclude-measurement-ids 590 556 ...` drops it
entirely rather than fixing the loader for that one case.

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

### Train/val/calib/test split

By default (`--split-mode segment`), whole windowing segments are
randomly assigned to train/val/calib/test. This matters because most
segments in this dataset are short (a few hours), single charge/discharge
cycles whose SoC declines from ~1.0 to some low point over their own
duration. Slicing each segment chronologically (`--split-mode
chronological`, the original approach) would systematically give train
the high-SoC early portion and test the low-SoC late portion of every
cycle -- confirmed on the full dataset: calib mean SoC 0.33 vs. test mean
SoC 0.25, and a 24% quantile-crossing rate on test vs. 3% on calib. That
breaks both generalization (train rarely sees low-SoC examples) and the
conformal calibration exchangeability assumption (calib and test come
from systematically different SoC populations), producing badly
miscalibrated intervals (ACE far above 0, and a 95% PI narrower than the
90% PI). `chronological` is kept for datasets made of a small number of
long, continuous multi-profile sweeps, where 15% of one such sweep still
spans a representative chunk of the SoC trajectory.

### Calibration is a widening-only operator (eq. 20)

Eq. (20) clips both residuals,
`eta_i = max( w_l(u_i) * [q_l - SoC]_+ , w_u * [SoC - q_u]_+ )`, so
`eta_i >= 0` and hence `c_alpha >= 0`. Calibration can only widen the
preliminary interval, never tighten it. It repairs under-coverage; there
is nothing for it to do when the quantile model already over-covers.

A consequence worth knowing before debugging a run where all methods
report identical numbers: **`c_alpha` is exactly 0 whenever the weighted
calibration coverage already reaches `1 - alpha`**, and neither the omega
weights nor gamma can change that. A covered calibration point has both
`[.]_+` terms equal to zero, so `eta_i = 0` for any omega; scaling zeros
leaves zeros, and gamma only redistributes weight among them. Measured
with `wl0/wl1/wu = 1.5/3.0/1.0`, `gamma = 1`:

| calibration coverage | 0.99 | 0.96 | 0.92 | 0.90 | 0.86 | 0.80 | 0.70 |
|---|---|---|---|---|---|---|---|
| `c_alpha` (90% PI) | 0 | 0 | 0 | 0 | 0.013 | 0.028 | 0.047 |

So if RT-CQR, CQR and WCP all come out equal to the uncalibrated model,
the thing to fix is the quantile model's coverage (usually undertraining),
not the score or the weights. `--signed-score` swaps in the standard CQR
residual, which *can* tighten -- useful for diagnosis, but it is not
eq. (20).

Per Table I the three methods do not share one score: CQR and WCP use the
"standard CQR score" (signed), and only RT-CQR uses eq. (20). `baselines.py`
encodes that.

### Backbone size follows from Table I

Table I fixes the TCN at four residual blocks, kernel size 3, so the causal
receptive field is `1 + 2*(k-1)*sum(dilations) = 1 + 2*2*(1+2+4+8) = 61`
steps. `window_size` therefore defaults to 61: at the previous default of
100, the first 39 steps of every window had exactly zero gradient and were
invisible to the head.

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
- `--rated-capacity AH` — force one coulomb-counting capacity for every
  temperature instead of measuring each temperature's own (see above).
- `--signed-score` — give RT-CQR the standard CQR residual instead of
  eq. (20)'s `[.]_+` (see above); diagnostic only.
- `--train-stride N` — stride between training/validation windows.
- `--calib-stride N` — stride between conformal calibration windows
  (default: `window_size`, i.e. non-overlapping). At stride 1 the
  calibration set is 99%-overlapping windows, so `zeta^lag` decays across
  redundant copies of the same instant: 95% of the calibration weight
  then falls inside the last 2.5 minutes of one segment, versus 4.1 hours
  at the default stride.
- `--split-mode {segment,chronological}` — how train/val/calib/test are
  carved out (see above); default `segment`.
- `--include-all` — keep static characterization test sections instead of
  only dynamic drive-cycle profiles.
- `--exclude-measurement-ids ID [ID ...]` — drop specific Measurement IDs
  entirely from the windowing segments.
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
