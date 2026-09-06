# Reproducing Table II

Two things are separated here on purpose:

- **§1** — the configuration that takes both papers at their word, used to
  reproduce the published Table II numbers.
- **§2** — implementation defects found while building this, each with the
  measurement that establishes it. These are *corrected* by default and
  deliberately left uncorrected under `--paper-literal`.

Keeping them apart matters: once the reproduction is accepted, §2 is the list
of what to improve, and every entry already says what it is worth.

## 1. The reproduction run

```bash
python train.py --data-root /path/to/lg_hg2 --output-dir outputs_repro \
    --paper-literal --point-baseline
```

`--paper-literal` sets three things that the defaults otherwise correct:

| setting | source | value |
|---|---|---|
| `dilation_base` | Table I: 4 residual blocks, kernel size 3, textbook doubling | 2 → `{1,2,4,8}` |
| `window_size` | [6]: "the width of the window, k was kept at k = 400 timesteps" | 400 |
| `resample_dt_s` | [6] windows the raw data, which is logged at 0.1 s | 0.1 |

**These three combine into a 6.1-second model.** Four kernel-3 blocks with
dilations `{1,2,4,8}` have a causal receptive field of
`1 + 2*(3-1)*(1+2+4+8) = 61` steps. At 0.1 s sampling that is 6.1 seconds, and
the remaining 339 steps of the 400-step window have identically zero gradient —
verified directly, not inferred. Training prints the warning.

This is the most likely explanation for the accuracy gap. The RT-CQR paper
cites [6] for its data split, but [6] reports RMSE 1.19% at varying ambient
temperature, whereas Table II's AIW of 0.145 at 90% implies roughly
`0.145 / (2 * 1.645) = 4.4%` RMSE — 3.7x worse than the protocol it cites. A
model that can only see 6.1 seconds of history would account for that.

The everything-corrected default reaches RMSE 1.32% / MAE 0.99%, i.e. [6]'s
level, and correspondingly narrow intervals (AIW 0.041 at 90%). Those numbers
are *better* than Table II, which is why they do not reproduce it.

### 1.1 What the reproduction reaches

Run above plus `--zeta 0.999` (see §2.5 for why 0.98 does not transfer),
`--train-stride 10 --test-stride 10`, on an RTX 5060:

| 90% PI | LVR | AIW | ACE | coverage |
|---|---|---|---|---|
| paper RT-CQR | 0.018 | **0.145** | 0.004 | — |
| this RT-CQR | 0.00000 | **0.141** | 0.085 | 0.9846 |
| paper CQR | 0.033 | 0.152 | 0.015 | — |
| this CQR | 0.00009 | 0.093 | 0.059 | 0.9594 |
| paper Point | 0.084 | — | — | — |
| this Point | 0.01571 | — | — | — |

| 95% PI | LVR | AIW | ACE | coverage |
|---|---|---|---|---|
| paper RT-CQR | 0.005 | **0.192** | 0.003 | — |
| this RT-CQR | 0.00000 | **0.201** | 0.043 | 0.9932 |

**AIW reproduces**, at 97% and 105% of the published values. The remaining
two columns do not, for reasons that are measurable rather than tunable.

**ACE cannot be matched at the same time as AIW.** Uncalibrated the model
covers 0.8259 at 90% nominal with AIW 0.065; widened to AIW 0.141 it covers
0.9846. The width at which it would cover exactly 0.90 is around 0.08-0.09,
not 0.145. The paper reaches both at once, so its residuals must have a
markedly heavier tail: at the same interval width it covers 90% where this
model covers 98.5%. That is a property of the error distribution's shape,
not its scale, and no amount of degrading the model reproduces it.

**LVR is ~100x smaller** (§2.7), and Point is 0.0157 against the paper's
0.084.

### 1.2 The method ordering is inverted, and why

| | paper AIW (90%) | this AIW (90%) |
|---|---|---|
| CQR | 0.152 | 0.093 |
| WCP | 0.158 | 0.115 |
| RT-CQR | **0.145 (narrowest)** | **0.141 (widest)** |

The paper's claim is that RT-CQR is simultaneously the safest (lowest LVR)
and the tightest. Here it is the widest, and that is structural rather than
a tuning artifact: all three calibrators share one trained model, so they
differ only in their scores, and RT-CQR's omega weights (1.5/3.0/1.0)
inflate its score 1.5x relative to CQR's (1/1/1). A larger score gives a
larger c_alpha, hence necessarily a wider interval.

Table I says otherwise. Its footnote — "each starred method, the TCN
backbone has four residual blocks, 64 channels per block, ..." — gives
CQR*, WCP* and RT-CQR* **each their own backbone**, and only RT-CQR* lists
the composite-loss weights (lambda_nc = 1.0, lambda_l = 0.1). So in the
paper RT-CQR is narrower because its *model* is better trained, not because
its calibration is tighter.

Sharing one model was a deliberate choice here, to isolate the effect of
violation weighting from the effect of a different backbone. It is the
right ablation, but it is not Table II's setup, and it cannot reproduce
Table II's ordering. Reproducing that ordering requires training CQR* and
WCP* as separate models on the plain pinball loss.

## 2. Defects found, corrected by default

Every item below was measured on the full six-temperature LG 18650HG2 archive
(208 CSVs, `-20/-10/0/10/25/40 degC`). The archive itself is intact: all 16
measurements have an unbroken program clock across every section.

### 2.1 SoC referenced to nameplate capacity — corrected

`rated_capacity_ah` defaulted to the cell's 3 Ah rating. Every drive cycle at
every temperature ends at the 2.8 V discharge cut-off — an empty cell — but a
fixed 3.0 Ah denominator labels that same physical state:

| T (degC) | -20 | -10 | 0 | 10 | 25 | 40 |
|---|---|---|---|---|---|---|
| SoC at 2.8 V, 3.0 Ah denominator | **0.517** | 0.318 | 0.255 | 0.167 | 0.120 | 0.193 |
| SoC at 2.8 V, measured 1C capacity | 0.034 | 0.059 | 0.108 | 0.017 | 0.039 | 0.000 |
| measured 1C capacity (Ah) | 1.64 | 2.25 | 2.47 | 2.52 | 2.71 | 2.50 |

With `soc_min = 0.10`, the nameplate version put SoC below `soc_min` *nowhere*
in the dataset. LVR was then identically 0 for every method, the violation
indicator was identically 0 — making `wl1` and `gamma` inert, and RT-CQR's
violation weighting a no-op — and the lower-tail regularizer had nothing to
penalize. Normalizing per temperature puts segment ends at a median SoC of
0.041, matching the protocol's "95% of the 1C discharge capacity at the
respective temperature", and 13.5% of test samples below `soc_min`.

### 2.2 Time base losing 90% of the samples — corrected

Ordering on `Time Stamp` (whole-second) rather than `Prog Time`
(millisecond) discarded nine of every ten rows of each drive-cycle file, which
are logged at 0.1 s: `551_UDDS` went from 159,646 rows to 15,966. `Time Stamp`
is also not always self-consistent — `582_LA92` jumps between 11/25 21:00 and
11/26 10:11 and back while `Prog Time` advances smoothly, and `571_Mixed6`
jumps 10.4 h mid-file — which manufactured >300 s "gaps" that split single
drive cycles into fragments. Switching recovers 449,558 → 4,488,123 rows.

### 2.3 Sections named as drive cycles that contain none — corrected

`551_HWFET` is a complete file whose program clock runs straight from
`551_Charge3` into `551_Charge4` with no room for a drive cycle: the 25 degC
HWFET run recorded only its 600 s rest step, current identically 0 A. It was
being windowed as a drive cycle. Segments now must carry actual dynamic load.

### 2.4 Random split instead of [6]'s protocol — corrected

Whole segments were assigned at random, so a `Mixed4` run could train while
`Mixed5` from the same measurement tested. [6] holds out drive-cycle
*profiles*: its Figs. 1-3 plot LA92, UDDS and US06 as test cycles at each of
the six temperatures. `drivecycle_split` reproduces that exactly — 18 test
segments, one of each profile at each temperature.

### 2.5 zeta = 0.98 does not transfer at this N_cal — flagged, not silently changed

The geometric weights of eq. (22) have an effective sample size of about
`(1+zeta)/(1-zeta)` = 99 at `zeta = 0.98`, *independent of N_cal*. With
N_cal = 1045 spread over four calibration segments whose per-segment coverage
ranges from 0.556 to 0.995, `c_alpha` is decided from roughly one segment — and
which one is a lottery:

| calibration order | zeta-weighted coverage | pooled | c_alpha | ACE (90%) |
|---|---|---|---|---|
| arbitrary (permutation order) | 0.9932 | 0.8804 | 0.00000 | 0.016 |
| chronological (eq. 19's requirement) | 0.7162 | 0.8804 | 0.01118 | 0.083 |
| whole set (`--zeta 1.0`) | 0.8804 | 0.8804 | 0.00124 | **0.007** |

The paper reports ACE 0.004. Only the third row reaches it. The paper does not
report its N_cal, so its 0.98 cannot be carried over as a bare number.
`evaluate` now prints pooled beside zeta-weighted calibration coverage and
flags the case where the decay hides under-coverage, since eq. (20)'s clipped
score renders that as `c_alpha = 0`, indistinguishable from "no correction
needed".

The per-segment spread is wide *because* the intervals are narrow: the model's
mean error varies 8x across those segments (0.0046 at 25 degC, 0.0382 at
-20 degC). Scaling the width to Table II's 0.145 collapses the spread from
0.439 to 0.000 — at the paper's accuracy this failure mode is invisible, which
is why it does not show up there.

### 2.6 Calibration set too small to resolve alpha = 0.05 — corrected

Non-overlapping calibration windows at `window_size = 341` left N_cal = 66.
`c_alpha` is a weighted empirical quantile, so its resolution is bounded by
`1/N_cal` = 0.015 — too coarse to place a 0.05 tail. The stride is now chosen
to reach `calib_min_windows` (1000).

### 2.7 LVR rounded away, and its anchor row never printed — corrected

LVR lands around 1.8e-4 here and the table printed 3 dp, rendering it as a
column of 0.000. At full precision the ordering is intact and matches Table
II's direction (falling from the 90% to the 95% PI; WCP worst because its
interval collapses). `print_results_table`'s `point_lvr` was also dead — `main`
never passed it — so Table II's Point row never appeared. `--point-baseline`
now trains Table I's `Point*` (same backbone, MSE loss).

The scale gap is a property of the fit, not the metric: of 19,704 test samples
below `soc_min`, 27 qualify, because registering one needs the model to
over-predict by more than 0.059 at the median violating point and its median
error there is 0.0057.

### 2.8 Eq. (20)'s `[.]_+` is as published — reverted after checking

An earlier commit replaced the clipped residuals with the signed CQR score,
because clipping forces `c_alpha >= 0` and makes calibration a widening-only
operator. The paper's eq. (20) does carry `[.]_+`, so the default was restored;
`--signed-score` keeps the alternative available for comparison.
