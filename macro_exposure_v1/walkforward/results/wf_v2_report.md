# macro_exposure_v1 Balanced — Overlay v2 Walk-Forward (HC #543 follow-up)

## 3-way comparison

| Variant | Pooled CAGR | DD | Sortino | Sharpe | Hit-rate | Bear CAGR | Bear Sortino | Bull CAGR | Chop CAGR | Trigger freq | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| Original Balanced (no overlay) | 21.5% | 12.4% | 1.55 | 1.43 | 100% | — | 0.39 | — | — | — | **FAIL** (bear Sortino) |
| Overlay v1 (20d VIX percentile) | 16.5% | 12.3% | 1.43 | 1.38 | 90.9% | 5.7% | **0.504** | 22.2% | 11.7% | 37.4% | **PASS** (bear margin 0.004) |
| Overlay v2 (252d VIX percentile) | 15.8% | 12.6% | 1.39 | 1.35 | 86.4% | 5.0% | **0.525** | 22.3% | 10.9% | 27.8% | **PASS** (bear margin 0.025) |

## Verdict

**Both overlays pass all 4 gates** (pooled CAGR ≥ 10%, pooled DD ≤ 18%, hit rate ≥ 65%, all regimes Sortino ≥ 0.5). Neither one is clearly dominant:

- **v1 wins on**: pooled CAGR (+0.6pp), pooled Sortino (+0.04), hit rate (+4.5pp), bull CAGR (basically tied), chop CAGR (+0.8pp).
- **v2 wins on**: bear-regime Sortino margin (0.025 vs 0.004 — 6× wider), trigger frequency (27.8% vs 37.4% — fires less in normal markets).

The hypothesis that a longer (252d) VIX window would preserve more bull/chop CAGR by firing less was **partially correct** — trigger frequency dropped from 37% to 28% — but it did not actually recover bull/chop CAGR (both basically unchanged), and it slightly degraded the headline pooled metrics.

## Recommendation

**v1 is the deployment candidate** for paper-trade (HC #534 R1). Reasoning:
1. Higher pooled CAGR (16.5% vs 15.8%) — strictly better income generation.
2. Higher pooled Sortino (1.43 vs 1.39) — slightly better risk-adjusted profile.
3. Higher hit rate (90.9% vs 86.4%) — fewer losing folds.
4. The bear-Sortino margin is thinner (0.004), but it IS positive — the gate is cleared, and the 3 bear-regime folds were all small-to-modest positives (no losing year in bears).

**v2 is the backup** if paper-trade reveals bear-regime brittleness. The wider bear margin would matter more in a real 2008-style crisis than in the 2018/2022 vol episodes that defined our bear regime here.

## Next steps (per HC #393 — act, don't ask)

1. Treat v1 as the production candidate. File v2 as alternative.
2. Scope paper-trade infrastructure for the v1 config (broker API, daily exposure compute, weekly rebalance executor).
3. Possible overlay-v3 future experiment: AND-intersection gate (`vix_pct_20d > 0.7 AND vix_pct_252d > 0.6`) to require BOTH recent AND historical vol elevation — should fire even less than v2 while still catching genuine crises. Not blocking — only if v1 paper-trade exposes weakness.

## Files

- `walkforward/run_wf_v2.py` — v2 walk-forward harness.
- `walkforward/results/wf_summary_v2.json`, `wf_folds_v2.csv`, `wf_regimes_v2.csv` — raw v2 metrics.
- `backtest/defensive_overlay_v2.py` — the v2 overlay function.

Authorized under HC #420 — user's own quant research codebase.
