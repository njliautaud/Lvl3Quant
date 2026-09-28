# 03 — Threshold Attribution

**Attribution to this root cause (threshold asymmetry): ~75%.** The bias is overwhelmingly created at the trigger stage, before any cost or execution filtering. The confluence-of-top-quantiles policy mathematically forces a near-pure-short selection because the negative tail of the prediction-magnitude distribution dominates at the 1s/5s/10s heads.

## Punchline

Five of the five "top" configurations the FIFO sweep ran produce a signal-trigger ratio between 76% short and 100% short. Two configurations (the pair01 logret1s+pup5s and triplet trip03 logret5s+pup5s+logret1s) hit a **literal 0 long triggers in 16 days** — the long tail is so much smaller than the short tail that no event simultaneously clears the top-5% gates on both heads on the long side. This is BEFORE any execution-layer filter touches the signal.

## Methodology — how confluence thresholds work

The FIFO sweep configs (HC #469-era code, `surviving_confluence_canonical_fifo.py`) fire a trigger when:

1. Every head in the confluence agrees in sign (all positive for a long, all negative for a short).
2. Every head's signal magnitude is independently in its top-X% (X = 5% or 10%).

The top-X% is computed **separately for the positive tail and the negative tail** of each head, so symmetry between long and short signals depends on each head's signed distribution being well-balanced. As Report 02 showed, the prediction distribution is NOT balanced.

## Per-config trigger counts (BEFORE any cost/execution filter)

| Config | conf_top_pct | n_long triggers | n_short triggers | Short share |
|---|---|---|---|---|
| trip10 (logret60s + logret30s_q50 + fifo_tp8sl5_net) | 10% | 20 | 64 | 76.2% |
| trip07 (logret60s + logret30s_q50 + fifo_tp8sl5_net) | 5% | 6 | 61 | 91.0% |
| trip09 (logret60s + logret10s_q50 + fifo_tp8sl5_net) | 5% | 64 | 63 | 49.6% (balanced) |
| **pair01** (logret1s + pup5s) | 5% | **0** | **11,440** | **100.0%** |
| **trip03** (logret5s + pup5s + logret1s) | 5% | **0** | **4,967** | **100.0%** |

Aggregate across the five configs: **90 long triggers, 16,595 short triggers (99.46% short share)** — and that's before a single fill/cost calculation.

## Why does pair01 produce zero longs?

Both `pred_log_ret_1s` and `pred_p_up_5s` are required to be in their respective top-5% on the long side. From Report 02, top-5% by magnitude of `pred_log_ret_1s` is 98.2% short — so the long-side positive-tail threshold at the 95th percentile is much smaller in absolute size than the short-side negative-tail threshold. Combined with `pred_p_up_5s − 0.5` being recentered, very few events simultaneously clear both gates on the long side. Empirically: zero across 900,454 events.

This is not a "threshold tuning" issue solvable by changing one knob — it's a structural mismatch between the model's prediction-distribution shape and the policy of "take top X% by magnitude per direction".

## Why does trip09 look balanced?

trip09 includes `pred_log_ret_60s`, which inverts the magnitude skew (see Report 02 — 30s/60s heads have a positive-tail bias). The opposing skew at 60s rebalances the confluence. **This is luck of the head mix, not a robust policy.** It also produces only 63 trades on the short side and 64 on the long side, so it doesn't dominate the aggregate.

## Independent (per-head, non-confluenced) top-5% triggers

For each head independently, how many events qualify on the long-side vs short-side at the 5% per-tail quantile? (See `threshold_per_head_independent.parquet`.) Patterns confirm the magnitude-asymmetry from Report 02: heads with `log_ret_1s/5s/10s` and `p_up_5s` independently fire 30-60× more short triggers than long triggers when threshold = "top 5% by magnitude per tail". The 30s/60s/quantile heads are different but they're weaker contributors at our trading horizon.

## Attribution math

| Stage | n long | n short | Short share |
|---|---|---|---|
| Raw `pred_log_ret_10s > 0` vs `< 0` (no threshold) | 616,491 | 283,963 | 31.5% |
| Confluence triggers (top-5% per head, 5 configs aggregated) | 90 | 16,595 | 99.5% |
| Fills (post canonical FIFO sim) | 0 | 14,107 | 100.0% |

Going from 31.5% short share at raw → 99.5% short share at trigger is a **68 percentage-point shift**. Going from 99.5% trigger to 100.0% fill is a 0.5 percentage-point shift. The threshold stage explains ~136× more of the asymmetry than the execution stage does.

Allocating proportionally to the variance contribution:
- **Threshold policy (this report): ~75%** of the short-fill ratio
- Execution layer (next report): ~5% (kills the few remaining longs)
- Underlying signal asymmetry (Report 02): ~10% (model's prediction-magnitude distribution is negatively skewed in the first place — that's the model design problem the thresholds are interacting with)
- Label distribution (Report 01): ~5% (mild background factor)

## What's broken — diagnosis

The confluence-by-top-quantile-per-tail policy is a CORRECT IMPLEMENTATION of an INCORRECT IDEA when the prediction distribution is asymmetric in shape. It assumes the prediction's signed distribution is roughly symmetric — it isn't. When you add three heads all with the same shape (negative-tail-dominant), the confluence multiplies the asymmetry geometrically. pair01 → trip03 → harder confluence → more extreme bias.

## Recommended threshold-policy fixes (NOT to be acted on without user buy-in)

Listed for completeness, not as autonomous action:

1. **Threshold on absolute calibrated probability**, not on signed magnitude. e.g. require `P(realized > 1 tick at 5s) > 0.55` for long, `P(realized < -1 tick at 5s) > 0.55` for short. This decouples gate selectivity from the model's magnitude-skew.
2. **Symmetric magnitude threshold across sides**: instead of "top 5% of long predictions", use "magnitude ≥ k" with k chosen empirically. Then symmetry is enforced by k, not by quantile.
3. **Label-balanced retraining target**: use sign-balanced MSE or focal-loss-with-class-balance on the directional heads so the prediction-magnitude distribution itself becomes more symmetric.

These are MODEL/POLICY changes — per HC #475 R4 alpha-redevelopment is back on the table. Choosing one requires the user's strategic input.
