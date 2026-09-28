# HC #443 FINDING — Pred-Strength Is INVERTED at the Top under Canonical FIFO

**Date**: 2026-05-19 ~23:35 ET
**Source**: `/tmp/hc443_strength_analysis.py` decile sweep of 4 canonical FIFO fills CSVs
**Sample**: 12,616 fills across 4 configs on CNN-Mamba v2 top-0.5% short, 32-36 OOT days

## The Finding

In every one of the 4 canonical runs, the **TOP** decile of pred_strength performs **WORSE** than the **BOTTOM** decile. Monotonic-ish decay.

| Config | Bottom decile mean_tk | Top decile mean_tk | Delta |
|---|---:|---:|---:|
| hc442_v2_canon_c1 (passive, SL=0.5, TP=3, H=1.5s) | −0.181 | **−0.528** | **−0.347** |
| hc442_v2_canon_c10 (passive, cancel=10s) | −0.193 | **−0.444** | **−0.251** |
| hc443_market_entry_tp3 (market, TP=3) | −0.595 | **−0.869** | **−0.274** |
| hc443_wider_sl2_tp3 (passive, SL=2, TP=3) | −0.366 | **−1.043** | **−0.677** |

In `hc442_v2_canon_c1` the top decile has WR=15.2% vs bottom decile WR=27.9%. The model's strongest short signals lose 12.7 percentage points of win rate vs the weakest signals.

## What This Means

**The +21 tick mean MFE we observed at the LABEL level on top-0.5% short does not translate to a tradeable edge — it inverts under FIFO execution.**

Mechanisms (hypotheses, not yet falsified):

1. **Adverse selection at the touch**: When pred_strength is high, price is already moving in our direction. A passive limit at the touch waits in queue. By the time queue clears, the favorable move has already happened — we get filled on the retracement.
2. **Feature staleness**: Strongest predictions correlate with the most violent recent moves, which are also the moves most likely to mean-revert or stall.
3. **Crowding**: Whatever signals the strongest predictions are based on (likely volume bursts, large prints) are also seen by faster participants. We're consistently last in line.

## Implications for Remaining Runs

**The queued tight-geometry run (SL=1 TP=2 H=0.5s top-0.5%)** — likely to fail for the same reason; it still uses top-0.5%. KEEP IT (cheap data point) but adjust expectations.

**The queued multi-h confluence run (1s∧5s∧10s top-30%)** — actually MORE INTERESTING NOW. By requiring agreement at 3 horizons we admit less-confident-at-any-one-horizon signals into the bucket, which may break the inversion.

**A natural new experiment** to add to the queue: **INVERTED top-band sweep** — try the same engine with `top-band` set to "the 5th-15th percentile of confidence" (i.e., admit medium-confidence signals, EXCLUDE the top 5%). If this finding is real, that bucket will outperform top-0.5%.

## What's NOT True

- The model is not noise. IC_1s=0.222 is real. The label-MFE on top-0.5% short within 1s is real.
- But "stronger signal = more profitable trade" is **FALSE** under canonical FIFO. This is the hidden assumption that has been wrong all along.

## Next Step (Autonomous)

Adding two configs to the sequencer:

1. **hc443_band_5to15** — short, h=1, conf band = 5th to 15th percentile (skip the top-5%, keep top-15%-to-5%), TP=3, SL=0.5, H=1.5s passive.
2. **hc443_band_wide_top5pct** — short, h=1, top-5% (10× more permissive than top-0.5%), TP=3, SL=0.5, H=1.5s passive.

These directly test the inversion hypothesis.

Files referenced:
- Per-fill data: `/output/hc432_v342_47day_validation/{run}_fifo_fills.csv`
- Analysis script: `/tmp/hc443_strength_analysis.py`
