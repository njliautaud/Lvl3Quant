# HC #414(b) LGBM Supervised Gate — v2 CNN-Mamba

NPZ: `hc417_v2_full_oot_wrapped_for_hc413.npz`, N=1464715, kept_dates=36
Time-order split: train dates 0..23, test dates 24..35
Train rows: 91132, Test rows: 16883
Target: `target_log_ret_1s < -0.376` (i.e. realised 1s move profitable for short after passive cost)
Best iter: 43

Feature importances (gain):
- slope_5_10: 1978.3
- pred_5s: 1687.2
- pred_10s: 1658.8
- daily_rank: 1205.2
- pred_1s: 1129.5
- slope_1_5: 842.0
- signed_short: 710.6
- abs_p1: 81.0

## Cell metrics: baseline vs gated (full-OOT, not just test split)

**NOTE**: These metrics are computed using a simplified short PnL (-target_1s - 0.376 tk), NOT the HC #413 TP/SL backtester. They are indicative of whether the LGBM gate filters out losing fills, not a replacement for the canonical FIFO replay.

| cell_id | gate | base_n | base_net | base_pdpr | gated_n | gated_net | gated_pdpr | gated_day_conc | delta_net | retention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v2_1s_short_top05 | 0.30 | 5624 | +0.5731 | 0.85 | 5624 | +0.5731 | 0.85 | 0.15 | +0.0000 | 1.00 |
| v2_1s_short_top05 | 0.40 | 5624 | +0.5731 | 0.85 | 5624 | +0.5731 | 0.85 | 0.15 | +0.0000 | 1.00 |
| v2_1s_short_top05 | 0.50 | 5624 | +0.5731 | 0.85 | 5624 | +0.5731 | 0.85 | 0.15 | +0.0000 | 1.00 |
| v2_1s_short_top05 | 0.60 | 5624 | +0.5731 | 0.85 | 5558 | +0.5988 | 0.85 | 0.15 | +0.0257 | 0.99 |
| v2_1s_short_top1 | 0.30 | 11095 | +0.5450 | 0.93 | 11095 | +0.5450 | 0.93 | 0.12 | +0.0000 | 1.00 |
| v2_1s_short_top1 | 0.40 | 11095 | +0.5450 | 0.93 | 11095 | +0.5450 | 0.93 | 0.12 | +0.0000 | 1.00 |
| v2_1s_short_top1 | 0.50 | 11095 | +0.5450 | 0.93 | 11095 | +0.5450 | 0.93 | 0.12 | +0.0000 | 1.00 |
| v2_1s_short_top1 | 0.60 | 11095 | +0.5450 | 0.93 | 11021 | +0.5594 | 0.93 | 0.12 | +0.0144 | 0.99 |
| v2_5s_short_top05 | 0.30 | 5603 | +0.5718 | 0.81 | 5603 | +0.5718 | 0.81 | 0.15 | +0.0000 | 1.00 |
| v2_5s_short_top05 | 0.40 | 5603 | +0.5718 | 0.81 | 5603 | +0.5718 | 0.81 | 0.15 | +0.0000 | 1.00 |
| v2_5s_short_top05 | 0.50 | 5603 | +0.5718 | 0.81 | 5603 | +0.5718 | 0.81 | 0.15 | +0.0000 | 1.00 |
| v2_5s_short_top05 | 0.60 | 5603 | +0.5718 | 0.81 | 5549 | +0.5944 | 0.81 | 0.15 | +0.0226 | 0.99 |
| v2_5s_short_top1 | 0.30 | 11062 | +0.5795 | 0.96 | 11062 | +0.5795 | 0.96 | 0.12 | +0.0000 | 1.00 |
| v2_5s_short_top1 | 0.40 | 11062 | +0.5795 | 0.96 | 11062 | +0.5795 | 0.96 | 0.12 | +0.0000 | 1.00 |
| v2_5s_short_top1 | 0.50 | 11062 | +0.5795 | 0.96 | 11062 | +0.5795 | 0.96 | 0.12 | +0.0000 | 1.00 |
| v2_5s_short_top1 | 0.60 | 11062 | +0.5795 | 0.96 | 10997 | +0.5926 | 0.96 | 0.11 | +0.0131 | 0.99 |
| v2_10s_short_top1 | 0.30 | 11055 | +0.5624 | 0.96 | 11055 | +0.5624 | 0.96 | 0.12 | +0.0000 | 1.00 |
| v2_10s_short_top1 | 0.40 | 11055 | +0.5624 | 0.96 | 11055 | +0.5624 | 0.96 | 0.12 | +0.0000 | 1.00 |
| v2_10s_short_top1 | 0.50 | 11055 | +0.5624 | 0.96 | 11055 | +0.5624 | 0.96 | 0.12 | +0.0000 | 1.00 |
| v2_10s_short_top1 | 0.60 | 11055 | +0.5624 | 0.96 | 11028 | +0.5686 | 0.96 | 0.12 | +0.0063 | 1.00 |

## Honest assessment

- **v2_1s_short_top05**: best gate@0.60 net delta=+0.0257 tk, retention=99%, verdict=PASS
- **v2_1s_short_top1**: best gate@0.60 net delta=+0.0144 tk, retention=99%, verdict=PASS
- **v2_5s_short_top05**: best gate@0.60 net delta=+0.0226 tk, retention=99%, verdict=PASS
- **v2_5s_short_top1**: best gate@0.60 net delta=+0.0131 tk, retention=99%, verdict=PASS
- **v2_10s_short_top1**: best gate@0.60 net delta=+0.0063 tk, retention=100%, verdict=PASS

## Bottom-line honesty (HC #417 honesty mandate)

**The LGBM supervised gate adds essentially no incremental value on top of CNN-Mamba v2's confidence ranking.**

- Out-of-sample AUC = **0.539** (chance = 0.50). The classifier barely distinguishes profitable from unprofitable top-X% shorts.
- Net/fill improvement at the most aggressive gate threshold (0.60) ranges from **+0.006 tk (10s top1) to +0.026 tk (1s top05)** — economically negligible relative to baseline +0.55-0.58 tk.
- `per_day_pass_rate` and `day_conc` are **unchanged or essentially identical** at every gate threshold.
- Retention is 99-100% — meaning the gate barely filters anything out. Below gate threshold 0.50, NOTHING is filtered.

**Interpretation**: The 1s/5s/10s prediction heads are highly correlated; a downstream LGBM trained on these heads has nothing fundamentally new to say. The CNN-Mamba v2 has already learned the discriminative features.

**Recommendation for HC #414(b)**: do NOT add an LGBM gate to the live deployment of `v2_1s_short_top05`. It's not worth the operational complexity for +0.026 tk/fill (which is well within the noise of fill timing in live execution).

**This is a NEGATIVE-RESULT finding and is recorded honestly per HC #417 brief**: "if LGBM boost makes things WORSE, report it" — it doesn't make things worse, but it doesn't help either.

**Note on metric simplification**: this evaluation uses a simplified short PnL formula (`-target_1s - 0.376 tk`), not the canonical HC #413 TP/SL backtester. The baseline net/fill of +0.55-0.58 tk here looks larger than the HC #413 backtester's +0.27 tk because the simplified PnL doesn't apply TP/SL caps. The relative comparison (gate vs no-gate) is still valid: the gate adds ~5% to net at most aggressive setting, which would translate to ~+0.013 tk net/fill under canonical HC #413 — still negligible.
