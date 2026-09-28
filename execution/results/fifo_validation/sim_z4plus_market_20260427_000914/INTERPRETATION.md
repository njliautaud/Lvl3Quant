# Z4+ Market Entry Test on CNN-Mamba — INTERPRETATION

**Run timestamp:** 2026-04-27 00:09:14 ET
**Hypothesis (from HANDOFF):** Market entry at z4+ on CNN-Mamba may absorb the spread because chase reported 63% WR and 13t MFE at that tier.
**Result:** **HYPOTHESIS REJECTED — CATASTROPHIC LOSSES**

## Headline Numbers (Mar 2-5, 2026)

| Threshold | Trades | Fill% | WR    | PF   | PnL          | MFE   | MAE   |
|-----------|--------|-------|-------|------|--------------|-------|-------|
| z3.5+     | 240    | 100%  | 20.8% | 0.02 | -$255,984    | 4.5t  | 3.6t  |
| z4.0+     | 172    | 100%  | 19.2% | 0.02 | -$205,751    | 4.3t  | 3.3t  |
| z4.5+     | 106    | 100%  | 17.9% | 0.02 | -$140,826    | 4.1t  | 2.8t  |
| z5.0+     | 80     | 100%  | 16.2% | 0.01 | -$119,345    | 3.4t  | 2.8t  |

**Win rate gets WORSE at higher confidence** — opposite of every label-based result.

## Why the chase config showed 63% WR but market entry shows 17%

The chase config (--chase-entry --chase-max-ticks 2 --chase-max-reprices 5) achieved
~30-38% fill rate at z3+. **The 70% of signals that didn't fill were the toxic ones.**

When the model emits a high-z signal:
- Sometimes price was already moving in our predicted direction → chase fills passively before BBO moves away → wins.
- More often, the high z-score appears precisely because someone with better info has just hit the BBO → BBO walks AWAY from us → chase cancels → no trade. Or in market-entry mode → we cross the spread INTO the toxic flow → adverse selection bites instantly → loss.

The selectivity of chase **was the alpha** — not just a fill mechanism. We were measuring conditional MFE | filled, which is much higher than unconditional MFE.

## Mar 5 was particularly toxic

z3.5+ on Mar 5 alone lost -$157,688 across 106 trades — likely a regime where the
prediction signal correlated with adverse-selection events (e.g., sweep events,
hidden-liquidity prints). High event-density days are likely worst case.

## Implications for execution research

1. **DELETE the "market entry at z4+" idea.** It's been falsified.
2. **The signal still has alpha** — but only conditional on patient execution.
3. **The chase fill rate (~30%) is a feature, not a bug.** We need execution methods
   that preserve that selectivity while squeezing more out of the fills we do get.

## Better hypotheses to test next (priority order)

1. **Wider chase (3-4 ticks) with --chase-force-cross at z6+ only**
   — Capture more of the long-tail signals that chase currently misses, but only force-cross at extreme conviction where MFE > 15t.

2. **Mid-price entry at z4+** (`--mid-price-entry`)
   — Splits the spread (saves ~0.5t vs market) but still requires the market to come to us. Should retain some selectivity.

3. **Vol-conditional market entry**
   — Only force-cross when spread is exactly 1 tick (cheapest crossing cost) AND z >= 4.5. Reject signals during wide-spread regimes.

4. **Exit optimization on existing chase config**
   — Keep the working entry; tune `--vol-exit-ticks`, `--vol-exit-bars`, `--ratchet-stop`, `--mae-exit-ticks` to capture winners earlier and cut losers faster.

5. **Time-of-day filter**
   — Mar 5 was the killer day. Check if losses cluster in opening 30 min or last hour, vs midday.

## Saved artifacts

- Per-trade JSONs: `sim_z4plus_market_20260427_000914/cnn_mamba_v2_z*.json`
- Summary JSON: `sim_z4plus_market_20260427_000914/summary.json`
- Script: `execution/results/fifo_validation/run_fifo_z4plus_market_cnn_mamba.py`
- Cache symlinks: `execution/results/fifo_validation/pred_cache_z4plus/cnn_mamba_v2/`
