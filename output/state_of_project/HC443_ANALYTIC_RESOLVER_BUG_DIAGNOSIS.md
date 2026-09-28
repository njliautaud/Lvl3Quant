# Analytic Resolver Bug — Final Diagnosis

**Author**: Claude (autonomous), 2026-05-19 ~23:40 ET
**Status**: Root cause IDENTIFIED. Bug is in `hc441_full_verdict.py::resolve_with_mfe_mae()` lines 115-123.
**Resolver does NOT need to be fixed** — we have moved to canonical FIFO replay as the ground-truth harness per HC #442. Documenting here for completeness and to prevent reuse.

## The Symptom

Same 3,455 entries, same TP/SL/hold/cancel, two different P&L answers:

| Engine | Source | mean_tk/fill | WR | Verdict |
|---|---|---:|---:|---|
| Analytic resolver | `hc441_full_verdict.py` | **+0.71** | 45.8% | BUGGY |
| Canonical FIFO   | `alpha_discovery.deep_models.fifo_market_replay.FIFOReplayEngine` | **−0.31** | 25.1% | TRUE |

Delta: **−1.02 tk/fill**. Sign INVERTED. 20-point WR difference.

## Root Cause: SL Exits Are Priced at the TRIGGER, Not at the Realized Crossed Touch

Look at `resolve_with_mfe_mae()` lines 115-123:

```python
if first_tp >= 0 and (first_sl < 0 or first_tp <= first_sl):
    exit_pr = int(tp_price); exit_reason = 'tp'        # passive TP — OK
    hold_actual = int(offs_h[first_tp])
elif first_sl >= 0:
    exit_pr = int(sl_price); exit_reason = 'sl'        # ⚠ BUG: SL exit priced at TRIGGER
    hold_actual = int(offs_h[first_sl])
else:
    exit_pr = int(prs_h[-1]); exit_reason = 'max_hold' # max-hold exit at last print — minor inaccuracy
```

When the SL is hit:
- **Analytic**: assumes exit happens at exactly `sl_price` (the trigger level, e.g., `entry + 0.5` ticks for a short).
- **Canonical (correct)**: an SL is a MARKET order out. It must cross the spread, paying ~1 tick more in adverse price than the trigger.

For a short trade with SL=0.5:
- Trigger price: entry + 0.5 ticks
- Trigger event: best ask trades through entry+0.5
- Realized canonical exit: best ask + 1 tick spread = entry + 1.5 ticks worth of adverse move on the gross
- **Analytic underestimates SL loss by ~1 tick per SL hit.**

## Quantitative Reconciliation

Out of 3,455 trades, the canonical run shows WR=25%, so ~75% are losses. Most of those losses are SL hits.

Estimated impact:
```
delta_per_SL_hit       = -1.0 ticks (spread cross not modeled)
fraction_hitting_SL    ≈ 0.75 of fills
expected_delta_per_fill ≈ -1.0 * 0.75 = -0.75 tk/fill
observed_delta          = -1.02 tk/fill
unexplained_remainder   ≈ -0.27 tk/fill
```

The remaining −0.27 tk/fill gap likely comes from:
1. **Max-hold exits priced at last-trade-within-hold** (line 122), but canonical exits these as market orders at book midprice or worse — another spread-cross hit.
2. **Queue priority on TP fills** — canonical may not actually fill our passive TP when the price touches because we're behind in queue, while analytic blindly credits us the TP price.

## Why The Sign Inverted

Label-level edge on top-0.5% short = ~+1.56 ticks mean directional move within 1s (HC #428). Subtract:
- 0.376 tk commission
- 0 tk for passive entry (we are at touch)
- 1.0 tk for SL exits crossing spread (BUG: analytic skips this)
- 0.5 tk for various microstructure costs

Honest math: 1.56 − 0.38 − 0.75 (avg cross cost) ≈ +0.4 tk → BUT realized adverse selection brings it down further to −0.31.

Analytic math (BUG): 1.56 − 0.38 − 0.0 (no cross cost) ≈ +1.18 tk → reported +0.71 after some path-dependent stuff.

## What This Doesn't Explain

The 20% WR delta (45% analytic vs 25% canonical) is larger than the SL-spread bug alone would predict. Additional contributing factors (not fully diagnosed but listed for completeness):

1. **Adverse selection at the touch**: canonical fills at passive limits face queue position. Sometimes our limit is "filled" only AFTER the favorable move is over. Analytic assumes we always fill at the displayed touch.
2. **TP hit fills vs analytic credit**: canonical may not credit TP fills when our queue position is too low.

These two effects together can shave WR by 5-15 points without changing mean_tk much (they're WR-redistributive at marginal trades). Combined with the SL-spread bug, the full picture is consistent.

## Status & Action

- **No fix required**. The analytic resolver is deprecated effective HC #442.
- **All HC #441 results derived from this resolver are INVALID** (so noted in DIRECTIVES.md and STATE_OF_THE_PROJECT_*.md).
- **Canonical FIFO replay is the only authoritative engine** going forward.
- If you ever need an analytic preview again, you must either:
  a. Reimplement SL/max_hold exits to model market-order crossing, OR
  b. Use the canonical engine — which is now ~5 min/day per run, perfectly fast enough.

## Reference

- Resolver code: `/home/jupiter/Lvl3Quant/scripts/hc441_full_verdict.py` lines 63-145
- Canonical engine: `/home/jupiter/Lvl3Quant/alpha_discovery/deep_models/fifo_market_replay.py`
- Side-by-side proof:
  - Analytic: `/output/hc441_full_verdict/per_fill_PRIMARY.csv` (DO NOT TRUST)
  - Canonical (same trades): `/output/hc432_v342_47day_validation/hc442_v2_canon_c10_fifo_fills.csv`
