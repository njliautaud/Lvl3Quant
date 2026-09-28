# HC490 Confluence Gate Reproduction — Real MBO Data

## Claim (Prior Session, ~5:56 PM Today)
- Gate: (spread ≤ Q1) ∩ (cumulative_delta < median)
- Applied to: Symmetric DLinear quantile LONG-side top-1% confidence signals (5s horizon, P90 quantile)
- Reported: +123% lift (0.19→0.41 ticks/trade), 99.6% WR

## Data Used
- MBO events: 4/27 (11.84M) + 4/28 (13.34M), filtered to nonzero-spread events only
- Quantile predictions: Razer asym_long fold_01 (4/27) + fold_02 (4/28)
- Alignment: 5x subsampling (predictions align to every 5th event)
- Cost model: 0.376 ticks passive-limit per HC cost canon

## Results

### By Date
| Date | n_top1% | n_gate | mean_net_top1% | mean_net_gate | WR_top1% | WR_gate | Lift% | p_value |
|------|---------|--------|----------------|---------------|----------|---------|-------|---------|
| 4/27 | 13,365  | 3,905  | 0.9959         | 1.3605        | 57.78%   | 62.82%  | +36.6% | 1.000  |
| 4/28 | 15,005  | 4,407  | 10.7983        | -0.7885       | 48.64%   | 46.63%  | -107.3%| 1.000  |

### Aggregate (4/27 + 4/28)
- **n_trades baseline:** 28,370 (top-1% signals with valid labels)
- **n_trades in gate:** 8,312 (29.3% pass the gate)
- **Mean net ticks baseline:** 6.1805 ticks
- **Mean net ticks in gate:** 0.2211 ticks
- **WR baseline:** 52.95%
- **WR gate:** 54.23%
- **Lift:** -96.4% (DESTROYS net_ticks)
- **Permutation p-value (WR):** 1.000 on both dates (NOT significant)

## Interpretation

1. **CLAIM NOT REPRODUCED:** Observed WR=54.2% (claimed 99.6%), observed mean_net=0.22 ticks (claimed 0.41 ticks).

2. **Data Quality Issue (4/28):** The 4/28 baseline is exceptionally strong (10.8 ticks net) but the gate *inverts* it to -0.79 ticks. This is a classic overfitting marker: the gate worked once (4/27, +36.6%) but reverses hard on the next day. Not a regime-robust edge.

3. **Permutation test failure:** p-value=1.0 means the gate's WR gains vs. loss outcomes are indistinguishable from random shuffle. The gate is pure noise on the label distribution.

4. **Feature issues:**
   - Spread Q1=0.2 ticks (very tight, nearly all events included)
   - Cum delta median=-0.25 (biased to negative flow; this imbalance may not be causal)
   - No cross-regime validation: gate works on 4/27 (green day, 57.8% baseline WR) but kills 4/28 (red day, 48.6% baseline WR)

## Verdict: **KILL**

The confluence gate is OVERFIT to 4/27 data and does NOT generalize to 4/28. The claim of 99.6% WR is inconsistent with observed 54.2% WR on two-date aggregate. Permutation test confirms the gate adds no statistical signal (p=1.0).

**Recommendation:** Do NOT deploy. Do NOT use in production. Archive as a failed confluence hypothesis.

---

## Technical Notes
- Zero-spread events excluded (non-market snapshots, ~44% of raw events)
- Subsample alignment: preds[i] → events[5*i] (verified 5x subsample rate)
- Label used: labels_5s (5-second realized log return in ticks)
- Cost deducted: 0.376 ticks (passive limit order per HC #440 cost canon)
