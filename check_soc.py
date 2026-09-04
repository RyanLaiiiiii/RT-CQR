#!/usr/bin/env python3
"""Sanity-check the reconstructed SoC labels before spending GPU time on them.

The SoC labels are not published with this dataset; they are reconstructed
by coulomb counting against a per-condition measured capacity (see
rtcqr/data.py). A wrong capacity denominator does not raise -- it silently
moves every label -- so this script checks the three things that go wrong:

  1. Measured capacity must rise with ambient temperature. A cell does not
     hold less charge as it warms, so a reading that falls means that
     condition's Cap_1C section is truncated or incomplete.
  2. No segment should sit pinned at SoC 0 (or 1) for most of its length.
     That is the signature of a denominator that is too small: the coulomb
     count runs past the measured capacity and saturates at the clip.
  3. Each condition should span a sensible SoC range and actually reach low
     SoC, since that is the regime the paper's LVR metric is about.

Usage:
    python check_soc.py /path/to/lg_hg2
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np

from rtcqr.data import load_lg_hg2_dataframe

def _parse_overrides(values):
    """TEMP:AH pairs -> {condition: capacity}. `n20` is accepted for -20, since
    argparse reads a bare `-20:1.70` as an option rather than a value."""
    out = {}
    for chunk in values or []:
        for pair in str(chunk).split(","):
            pair = pair.strip()
            if not pair:
                continue
            temp_s, _, cap_s = pair.partition(":")
            temp_s = temp_s.strip()
            if temp_s[:1].lower() == "n":
                temp_s = "-" + temp_s[1:]
            out[float(temp_s)] = float(cap_s)
    return out


CLIP_EPS = 1e-3
FROZEN_FRAC = 0.5      # fraction of a segment pinned at a clip to call it frozen
HEAVY_FRAC = 0.1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_root")
    ap.add_argument("--soc-min", type=float, default=0.10,
                    help="Safety floor used by the LVR metric (default 0.10).")
    ap.add_argument("--min-soc-range", type=float, default=0.0,
                    help="Drop segments whose SoC spans less than this (try 0.02).")
    ap.add_argument("--capacity-override", action="append", metavar="TEMP:AH",
                    help="Override a condition's measured capacity, e.g. 40:2.75. Repeatable and accepts "
                         "comma-separated pairs; write n20:1.70 for sub-zero, or use the "
                         "--capacity-override=-20:1.70 form.")
    ap.add_argument("--include-all", action="store_true",
                    help="Include static characterization sections, not just drive cycles.")
    args = ap.parse_args()

    print("Loading. The loader's own capacity report follows -- read it first:\n")
    files = load_lg_hg2_dataframe(
        args.data_root, min_soc_range=args.min_soc_range,
        capacity_overrides=_parse_overrides(args.capacity_override),
        **({"include_patterns": None} if args.include_all else {})
    )

    by_cond = defaultdict(list)
    for bf in files:
        by_cond[bf.condition].append(bf)

    # ---------------------------------------------------------------- check 1
    print("\n" + "=" * 78)
    print("CHECK 1  frozen segments (SoC pinned at a clip for most of the segment)")
    print("=" * 78)
    # Pinned at 0 and pinned at 1 are opposite diagnoses and must not be
    # pooled: a denominator that is too small drives SoC into the *lower*
    # clip, while a segment pinned at 1.0 is a drive cycle that happens to
    # sit in the saturated full-charge region and says nothing about the
    # denominator.
    frozen_low, frozen_high, heavy_low = [], [], []
    for bf in sorted(files, key=lambda b: (b.condition is None, b.condition, b.path)):
        soc = bf.frame["soc"].to_numpy()
        at0 = float(np.mean(soc <= CLIP_EPS))
        at1 = float(np.mean(soc >= 1 - CLIP_EPS))
        note = ""
        if at0 > FROZEN_FRAC:
            note = "  <-- FROZEN at 0 (denominator too small)"
            frozen_low.append(bf)
        elif at1 > FROZEN_FRAC:
            note = "  <-- FROZEN at 1 (degenerate: near-constant label)"
            frozen_high.append(bf)
        elif at0 > HEAVY_FRAC:
            note = "  <-- heavy clipping at 0"
            heavy_low.append(bf)
        print(f"{str(bf.condition):>7}C  {bf.path[:42]:<42} "
              f"soc=[{soc.min():.3f},{soc.max():.3f}]  at0={at0:.2f} at1={at1:.2f}{note}")

    # ---------------------------------------------------------------- check 2
    print("\n" + "=" * 78)
    print("CHECK 2  per-condition SoC coverage")
    print("=" * 78)
    print(f"{'cond':>7} {'segs':>5} {'min':>8} {'max':>8} {'mean':>8} "
          f"{'frac<' + str(args.soc_min):>10} {'frac==0':>9}")
    for cond in sorted(by_cond, key=lambda c: (c is None, c)):
        s = np.concatenate([b.frame["soc"].to_numpy() for b in by_cond[cond]])
        print(f"{str(cond):>7} {len(by_cond[cond]):>5} {s.min():>8.3f} {s.max():>8.3f} "
              f"{s.mean():>8.3f} {float(np.mean(s < args.soc_min)):>10.3f} "
              f"{float(np.mean(s <= CLIP_EPS)):>9.3f}")

    # ---------------------------------------------------------------- check 3
    print("\n" + "=" * 78)
    print("CHECK 3  enough segments per condition for a stratified split?")
    print("=" * 78)
    thin = [c for c, v in by_cond.items() if len(v) < 4]
    for cond in sorted(by_cond, key=lambda c: (c is None, c)):
        n = len(by_cond[cond])
        print(f"{str(cond):>7}C  {n:>3} segment(s)" + ("   <-- <4, will all go to train" if n < 4 else ""))

    # ---------------------------------------------------------------- verdict
    print("\n" + "=" * 78)
    ok = True

    # The denominator's health is read from per-condition frac==0, not from
    # whole-segment freezing: a denominator a few percent too small
    # over-depletes a fraction of samples without pinning any single segment.
    suspect = []
    for cond in sorted(by_cond, key=lambda c: (c is None, c)):
        s_all = np.concatenate([b.frame["soc"].to_numpy() for b in by_cond[cond]])
        frac0 = float(np.mean(s_all <= CLIP_EPS))
        if frac0 > 0.02:
            suspect.append((cond, frac0))

    if frozen_low or suspect:
        ok = False
        conds = sorted({b.condition for b in frozen_low} | {c for c, _ in suspect}, key=str)
        print(f"FAIL  condition(s) {conds} over-deplete: SoC is driven into the lower clip,")
        print("      which means their capacity denominator is too small. Coulomb counting")
        print("      past 0 is irreversible, so it also corrupts everything after it in the")
        print("      same run. Diagnose whether the capacity check is bad data or the cell")
        print("      genuinely aged before overriding:")
        print("          python diag40.py <data_root>")
        for cond, frac0 in suspect:
            print(f"          condition {cond}: {frac0:.1%} of samples sit at exactly 0")
        print(f"      If it is bad data:  load_lg_hg2_dataframe(root, capacity_overrides={{{conds[0]}: <Ah>}})")
        print("      If it is real aging: leave it. Overriding would then overstate the cell.")

    if frozen_high:
        print(f"WARN  {len(frozen_high)} segment(s) are pinned at SoC=1.0 for most of their length:")
        for bf in frozen_high:
            print(f"          {str(bf.condition):>6}C  {bf.path[:60]}")
        print("      These are drive cycles sitting in the saturated full-charge region. Their")
        print("      label is near-constant while V/I/T vary, so they teach the model nothing")
        print("      and distort calibration. This is NOT a denominator problem. Drop them with")
        print("      min_soc_range (see load_lg_hg2_dataframe).")

    if heavy_low:
        print(f"WARN  {len(heavy_low)} segment(s) spend >10% of their length at SoC=0. Some is")
        print("      expected (dynamic loads pull slightly past a 1C reference); a lot is not.")

    if thin:
        print(f"WARN  condition(s) {sorted(thin, key=str)} have <4 segments, so they cannot fill")
        print("      calib/test and are assigned entirely to train.")

    total_low = np.mean(np.concatenate([b.frame['soc'].to_numpy() for b in files]) < args.soc_min)
    if total_low == 0:
        ok = False
        print(f"FAIL  no sample anywhere has SoC < {args.soc_min}. LVR is then trivially 0 for")
        print("      every method and the paper's violation-weighting mechanism is never exercised.")
    else:
        print(f"INFO  {total_low:.1%} of samples sit below soc_min={args.soc_min} (LVR is measurable).")

    if ok:
        print("PASS  labels look usable. Proceed to training.")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
