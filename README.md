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
  Two head parameterizations, see [below](#quantile-head-the-one-deviation-from-the-paper).
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
  order, per-condition measured-capacity SoC coulomb counting, uniform-rate
  resampling, temperature-stratified train/val/calib/test splitting, and
  sliding-window construction.
- `train.py` — end-to-end training + calibration + evaluation CLI.

## Setup

```bash
pip install -r requirements.txt
```

## Getting the data

### SoC labels and the capacity denominator

There is no published SoC formula for this dataset, so the labels are
reconstructed here by coulomb counting, and the denominator that turns
accumulated charge into SoC is **measured per ambient condition** from that
condition's `Cap_1C` capacity-check section rather than fixed at the 3.0 Ah
nominal rating. The cell's usable capacity falls steeply in the cold, while
the test protocol depletes the same fraction of *actual* capacity at every
temperature, so a single fixed denominator lifts the reconstructed SoC floor
well above 0 everywhere except the warmest condition.

Because a wrong denominator moves every label without raising anything, the
measurement is guarded: capacity checks are split into contiguous runs and
reduced by median (a measurement with two `Cap_1C` sections must not have
them summed), each candidate must last roughly the ~1 h a 1C discharge takes
and land within a plausible multiple of the rating, and the loader warns if
measured capacity *falls* as temperature rises -- which is physically
backwards and means that condition's section is truncated. A denominator
that is too small is the damaging direction: it drives SoC into the [0, 1]
clip, freezing whole drive-cycle segments at exactly 0 and, because the clip
is irreversible, corrupting everything after them in the same run. Check the
per-condition capacities the loader prints before trusting a training run.

One failure mode survives all of those guards: a check that starts from a
full charge, has no internal gap, and runs a normal ~1 h, but stops before
the discharge finishes. Nothing about that section alone gives it away --
only the *cross-condition* termination voltage does. Internal resistance
falls as a cell warms, so a warmer cell holds its voltage up under load and
reaches a fixed cutoff later, which means `v_end` must decrease
monotonically with temperature. On the McMaster data this table flags
40 degC, which terminates at 3.254 V --
182 mV *above* the colder 25 degC (3.072 V), while every other condition
falls monotonically from -20 degC's 3.571 V. Its 2.496 Ah is therefore an
under-estimate, not capacity fade (the campaign ran 25 degC before 40 degC,
ten days apart, so aging cannot explain an 8% drop). Pin it instead of
letting it fall back to the nominal rating:

```bash
python train.py --data-root /path/to/lg_hg2 --capacity-override 40:2.75 --min-soc-range 0.02
```

`--capacity-override` takes `TEMP:AH` pairs, is repeatable, and accepts
comma-separated pairs. For sub-zero temperatures write `n20:1.70` (the
dataset's own spelling) or use `--capacity-override=-20:1.70`; a bare
`-20:1.70` is read by argparse as an option rather than a value. Both
settings are recorded in `results.json`, so a run carries the assumptions it
was produced under. Re-derive the per-condition termination-voltage table
above before carrying `--capacity-override 40:2.75` over to a different copy
of the dataset -- the value is specific to this measurement's truncated
section, not a property of the cell.

The dataset is the LG 18650HG2 Li-ion battery cycler data collected by
Dr. Phillip Kollmeyer at McMaster University, Hamilton, Ontario, Canada
(originally released on Mendeley Data as "LG 18650HG2 Li-ion Battery Data
and Example Deep Neural Network xEV SOC Estimator Script"; the dataset's
own README requests that any use of this data be appropriately
referenced). This repo consumes it via the `aditya9790/lg-18650hg2-liion-battery-data`
Kaggle mirror, downloaded at runtime via `kagglehub` -- the raw data is
never committed to this repo.

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

The running SoC is clipped to `[0, 1]` at *every* step of the coulomb
count, not once at the end. This matters for measurements that are
**repeated charge/discharge cycling runs** rather than one continuous
depleting sweep -- confirmed on real data where a `Charge_N` section
recharges the cell (~+2.3-2.5 Ah, ~80% of the 3 Ah rated capacity) and the
following `Mixed_N` section discharges it by a similar amount, repeated
many times. A single end-of-array clip lets the *unclipped* running sum
drift arbitrarily far above 1.0 whenever several charge segments land
close together in the true chronological order before their matching
discharge segments (order follows each row's real timestamp, not the
`Charge_N`-pairs-with-`Mixed_N` naming) -- e.g. several +80%-SoC charges
stacking up before any clipping is applied, so that even several
subsequent ~80%-SoC discharges only bring the unclipped value back down
to some other value still above 1.0, which displays as a flat 1.0 for the
entire span once clipped, masking real depletion. Clipping at every step
instead makes each charge segment correctly saturate at 1.0 (as a real
cell does) before the next discharge segment starts.

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

### Repeating over seeds

```bash
python train.py --data-root /path/to/lg_hg2 --n-seeds 5
```

repeats the whole train → calibrate → evaluate pass at seeds 42..46 and
adds a `mean +/- std` table plus `aggregate` and `per_seed` blocks in
`results.json` (each seed's weights land in
`outputs/rtcqr_model_seed<N>.pt`). `--seeds 1 7 13` takes an explicit
list instead.

Read the spread, not one run. Even with the fixed partition, the
calibration radius is set by an effective ~99 samples (see below), so
LVR/AIW/ACE move from seed to seed on weight init alone; under
`--split-mode segment` they move several-fold. Wall time scales with the
seed count — each repeat retrains from scratch.

### Quantile head: the one deviation from the paper

The composite loss here is eq. (17) term for term. The model is not: the
default head emits the lowest quantile plus a softplus'd (>= 0) increment
per level, so `q[k] = q[k-1] + softplus(...)` cannot cross, whereas the
paper uses |T| unconstrained outputs and *penalizes* crossing with
`lambda_nc = 1.0`. That is why the crossing rate reported here is exactly
0.0000 — it is structural, not something the model learned.

The monotone head was adopted after measuring a ~14–15% crossing rate
under `lambda_nc=1.0`. That measurement is not evidence any more: it was
taken while the reconstructed SoC labels were still being driven into the
`[0, 1]` clip by the truncated 40 °C `Cap_1C` section, before
`--capacity-override` / `--min-soc-range` existed. Samples pinned at
exactly 0.0 have a degenerate conditional distribution — every quantile's
correct answer is the same number — so the fan collapses to zero width and
the ordering is decided by numerical noise; and on exactly those samples
the lower-tail regularizer pushes `q_{tau_l}` alone upward through its
neighbours while `lambda_nc`'s hinge only pushes back after they have
already crossed. With the labels corrected the clip pile-up is ~0%, so
that mechanism is gone.

`--unconstrained-head` runs the paper's version (unconstrained outputs,
`lambda_nc=1.0`) so the question can be settled by measurement rather than
inheritance:

```bash
python train.py --data-root /path/to/lg_hg2 --unconstrained-head
```

Every run prints and records a crossing rate (`crossing_rate` in
`results.json`, `crossing_rate_aggregate` across seeds) for both calib and
test. If it is near 0 under this flag, prefer the paper's head and the
implementation matches it on every axis. If it is still high, keep the
monotone head — and cite *that* number, not the stale one.

### How these numbers compare with the paper

Table II reports, on this dataset, Point LVR 0.084, and for the 90% PI
RT-CQR 0.018 / 0.145 / 0.004 against CQR 0.033 / 0.152 / 0.015 and WCP
0.026 / 0.158 / 0.012. Two structural differences are worth checking
before reading any gap as a bug:

* **Interval width.** The paper's AIW is ~0.145–0.212; a run here that
  reports ~0.03 is not "5x better", it is solving an easier problem —
  usually a split whose test windows share segments (or near-duplicate
  1 Hz stride-1 neighbours) with training. The fixed protocol tests on
  cycles the model never trained on, which is what makes the paper's
  widths the right order of magnitude to compare against.
* **Coverage error.** ACE around 0.09 with a very small AIW means the
  intervals are sharp but badly calibrated, the opposite of Table II.
  Check the calib-vs-test SoC report printed by `train.py` first: a
  calibration set drawn from a different part of the SoC range than test
  fits the radius on evidence test does not resemble, and no calibrator
  recovers from that.

The paper does not state `w_l^(0)`, `w_l^(1)`, or `w_u` — Table I lists
only `lambda_nc`, `lambda_l`, `zeta`, and `gamma`, and eq. (21) only
requires `w_l^(1) >= w_l^(0) >= w_u >= 0`. The values here (1.5 / 3.0 /
1.0) are this implementation's choice, so an exact match to Table II's
AIW should not be expected from them.

### Train/val/calib/test split

By default (`--split-mode fixed`) the partition is the paper's: Sec. IV.A
splits the data "following the protocol in [6]" and Sec. IV.B states that
"all methods follow an identical *fixed* partitioning", i.e. a
deterministic, cycle-name-based split rather than a random one. For this
dataset that protocol is:

| split | drive cycles | source |
|---|---|---|
| train | `Mixed1`–`Mixed8` | this repo's choice |
| val | `HWFET` | this repo's choice |
| test | `US06`, `LA92`, `UDDS` | **stated by [6]** |

over all six ambient conditions −20/−10/0/10/25/**40** °C
(`--fixed-conditions` to change them).

The test role is not a guess: [6] writes "the estimation plot on the test
dataset consisting of US06, LA92 and UDDS drive cycle", and plots exactly
those three at 10, 25, 40, 0, −10 and −20 °C — all six conditions, with
its Methods giving "ambient temperatures ranging from −20 to 40 °C".
HWFET appears only in its Fig. 4, which is the *Panasonic* cell's test
set, not the LG protocol.

What [6] does **not** give in the article is how the remaining cycles
divide between train and validation — it defers that to a Supplementary
Table 3 the PDF does not include. Everything but the three test cycles is
the eight `Mixed` cycles plus `HWFET`, and training on the former while
validating on the latter is this implementation's choice among those.
Override it with `--fixed-sections train=... val=... test=...` if you
obtain that table.

Because 40 °C is inside the protocol, this dataset's truncated 40 °C
`Cap_1C` section is back in play: pass `--capacity-override 40:2.75`, or
that condition's SoC labels bottom out early.

Calibration is carved from validation, per Sec. IV.B ("CQR, WCP, and
RT-CQR are calibrated on a common subset held out from the validation
set"): the last `--val-calib-fraction` (default 0.5) of each validation
cycle. Taking each cycle's tail rather than whole cycles keeps every
temperature present in both halves — with one `LA92` cycle per
temperature, assigning whole segments would leave conditions out of
calibration, and a condition that reaches test without reaching calib gets
a radius fitted on conditions unlike it. `train.py` prints both splits' SoC mean and
range, and warns when they drift more than 0.1 apart, because that gap
inflates ACE for every calibrator.

#### `--split-mode segment` (random, for robustness checks only)

Whole windowing segments are randomly assigned to train/val/calib/test,
**stratified by ambient temperature**. This is *not* the paper's protocol
and its results are strongly seed-dependent on this dataset — measured on
one trained model, re-drawing the split alone moved RT-CQR's AIW from
0.028 to 0.106 and its ACE from 0.019 to 0.094 across four seeds, and in
two of those four seeds the calibration buffer was degenerate enough that
`rtcqr`, `wcp` (and once `cqr`) returned bit-identical radii. Every
condition contributes only a handful of segments, so which single segment
lands in calib dominates the result. Use `--n-seeds` and read the spread,
never a single random-split run.

Whole segments rather than time slices: most segments here are short single
charge/discharge cycles whose SoC declines from ~1.0 to some low point.
Slicing each one chronologically (`--split-mode chronological`)
systematically gives train the high-SoC early portion and test the low-SoC
late portion of every cycle -- confirmed on the full dataset: calib mean SoC
0.33 vs. test mean SoC 0.25, and a 24% quantile-crossing rate on test vs. 3%
on calib. That hurts generalization and fits the radius on the wrong part
of the SoC range.
`chronological` is kept for datasets made of a few long continuous sweeps.

Stratified rather than globally random: calib draws only ~7.5% of segments,
so over this dataset's six conditions an unstratified split leaves a whole
temperature out of calib in almost every seed -- measured over 200 seeds with
48-96 segments, the test set contained a temperature calib had never seen in
93-100% of them. This dataset's error distribution is strongly
temperature-dependent (that is exactly why the capacity denominator is
measured per condition), so those windows get a radius fitted entirely on
conditions that behave differently. Stratifying splits each condition
independently and drops that to 0%. `train.py` prints the per-condition
segment counts for every split and warns if any condition still ends up in
test but not calib. `--no-stratify` restores the old behaviour.

### Calibration-buffer ordering and the effective sample size

Two properties of eq. (22)'s exponential decay are worth knowing before
reading any RT-CQR-vs-WCP-vs-CQR comparison:

* **It reads buffer position as time.** Segments leave the loader grouped by
  measurement and are then randomly permuted by the split, so `train.py`
  sorts calibration segments by absolute start time before windowing them.
  Without that sort the decay designates an arbitrary segment's tail as
  "now", and the time-adaptive component weights an essentially random
  subset.
* **It caps the effective sample size at `(1+zeta)/(1-zeta)`** -- 99 samples
  at `zeta=0.98`, independent of buffer size. 99% of the weight sits on
  roughly the last 228 entries, and with stride-1 windows those overlap
  almost completely, so the independent information is smaller still. Growing
  the calibration set does not stabilise the radius. `train.py` prints the
  effective count; read seed-to-seed differences between calibrators against
  that number, not against the raw sample count.

Relatedly, `weighted_quantile` applies split conformal's `ceil((1-a)(n+1))`
order statistic rather than eq. (25)-(26)'s plain empirical quantile, which
undercovers by ~1/(n_effective+1) -- about 1% here even with a six-figure
buffer, feeding straight into the reported ACE. `--paper-quantile`
reproduces the paper's formula exactly.

### Window size vs. receptive field

With the paper's four residual blocks and kernel size 3, the backbone's
receptive field is **61 time steps**: `1 + 2 * sum_b (k-1) * 2^b`. The
default `--window-size 100` therefore carries 39 steps per window that are
stored, standardized, transferred to the device and convolved but cannot
influence the prediction. Either pass `--window-size 61` to drop the dead
history, or raise `num_blocks` to 5 (receptive field 125) to actually use a
100-step window. `train.py` prints a warning naming both numbers.

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
  history window (default 100; see the receptive-field note above).
- `--resample-dt S` — uniform resampling interval in seconds before
  windowing (default 1.0); pass 0 to window over native sampling instead.
- `--current-sign {1,-1}` — coulomb-counting sign convention (see above).
- `--split-mode {segment,chronological}` — how train/val/calib/test are
  carved out (see above); default `segment`.
- `--no-stratify` — split without stratifying by temperature. Not
  recommended; see above.
- `--paper-quantile` — use eq. (25)-(26)'s uncorrected empirical quantile.
- `--capacity-override TEMP:AH` — pin a condition's capacity (see above).
- `--min-soc-range SPAN` — drop segments whose SoC spans less than SPAN
  (try 0.02); these sit in the saturated full-charge region with a
  near-constant label.
- `--point-baseline` — also train the deterministic point-estimation model
  and report its LVR (the `Point` row of Table II).
- `--include-all` — keep static characterization test sections instead of
  only dynamic drive-cycle profiles.
- `--exclude-measurement-ids ID [ID ...]` — drop specific Measurement IDs
  entirely from the windowing segments.
- `--calibrators rtcqr cqr wcp` — which calibration methods to report.
- `--num-workers N` — DataLoader workers (default 0; the dataset is already
  an in-RAM `TensorDataset`, so workers mostly add per-batch pickling —
  measure before raising it).
- `--max-epochs`, `--patience`, `--seed` — training controls.

## Method summary

1. **Risk-aligned quantile learning** (eq. 16-18): a TCN jointly predicts
   the quantile set `T = {0.025, 0.05, 0.10, 0.15, 0.85, 0.90, 0.95, 0.975}`
   using a composite pinball loss with a quantile-crossing penalty and an
   extra term penalizing `[SoC_min - q_{tau_l}]_+` at `tau_l = min(T)`. Per
   eq. (18) that term is a surrogate for the lower-bound violation event
   `{q_{tau_l} < SoC_min}`, so minimizing it pushes the predicted lower bound
   *up*, toward SoC_min. Note this pulls against the LVR metric of eq. (29),
   which counts samples where the true SoC is below SoC_min while the
   predicted lower bound is not: raising `q_{tau_l}` mechanically increases
   LVR. That tension is the paper's own; it is reproduced faithfully here
   rather than silently "corrected". Ablate the term with `--no-ltr`.
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
