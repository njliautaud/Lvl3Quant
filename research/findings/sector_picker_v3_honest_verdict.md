# Sector picker v3 — Honest pooled-OOT verdict (HC #559 / HC #561 R4)

**Date built**: 2026-06-07

## What the model is
Per-sector ridge regression over the master_panel feature shelf (35 features:
multi-horizon vol, intraday proxies, cross-asset state, regime dial, sector
rotation, insider activity, analyst revisions). Walk-forward train 36 months /
OOT 12 months / step 6 months. Long top-3 names, short bottom-3 per sector
on a monthly rebalance. 5 bps round-trip cost per name.

## Per-sector verdicts (walk-forward fold medians, for triage only)

| Sector | Tickers | WF Sharpe (med) | WF CAGR (med) | WF Calmar (med) | Verdict |
|---|---|---|---|---|---|
| **Energy** | 14 | 1.49 | 16.9% | 2.68 | passes floor |
| **Industrials** | 27 | 1.22 | 14.8% | 2.18 | passes floor |
| Basic Materials | 9 | 0.75 | 8.4% | 0.72 | fails floor |
| Real Estate | 14 | 0.57 | 4.7% | 0.72 | fails floor |
| Utilities | 18 | 0.48 | 3.9% | 0.44 | fails floor |
| Communication Services | 11 | 0.18 | 1.2% | 0.16 | fails floor |
| Technology | 40 | -0.10 | -3.3% | -0.13 | fails floor |
| Healthcare | 28 | -0.46 | -7.4% | -0.43 | fails floor |
| Financial Services | 39 | -0.58 | -9.6% | -0.56 | fails floor |
| Consumer Defensive | 17 | -0.65 | -6.7% | -0.63 | fails floor |
| Consumer Cyclical | 25 | -0.27 | -7.7% | -0.32 | fails floor |

WF fold medians can OVERSTATE strength because overlapping windows hide
cross-fold drawdowns. The honest test is below.

## Pooled all-OOT verdict — combined Energy + Industrials (1,563 trading days 2018–2025)

| Metric | Combined book | SPY 1× | Beats SPY? |
|---|---|---|---|
| Sharpe | 0.69 | 0.60 | ✓ marginally |
| Sortino | 0.98 | — | — |
| CAGR | **7.0%** | **10.6%** | **✗** |
| MaxDD | −18.2% | −31.3% | ✓ materially |
| Calmar | 0.39 | — | **✗ fails 1.0 floor** |

## Headline

**Current shelf + ridge picker does NOT beat SPY on return.** It's a lower-risk
lower-return alternative (worse CAGR, better drawdown, slightly higher Sharpe).
Fails the Calmar ≥ 1.0 floor on the pooled-OOT view.

## What's driving the two passing sectors

**Energy** — dominant signal is vol carry within sector (short 60-day Yang-Zhang
high-vol names, long 20-day recent-vol names). Cross-asset + regime + insider
features carry almost no weight in the average ridge coefficients.

**Industrials** — dominant signal is dollar_volume (short heavy-volume names,
i.e. crowded liquid mega-caps) and long-term realised vol (long high 252-day vol).
Again, the macro/news/sector-rotation features are riding along but not driving.

This suggests the new shelf families (cross-asset, regime, sector rotation,
insider, analyst revisions) **aren't yet providing alpha for the ridge model**.
Either the model is too simple to extract them, or the families need to be
combined with text/news signals not yet in the shelf.

## Next steps

1. **Add GDELT news sentiment** (background ingest in flight) — quarter of the
   shelf the regime currently has zero coverage of.
2. **Add 10-K/10-Q text embeddings** (sentence-transformers) — fundamental tone
   signal not captured by quantitative features.
3. **Add Reddit + Google Trends** — retail attention signals.
4. After (1–3): re-run v3 picker. If still under-target, try nonlinear (LightGBM)
   over the same features — interpretability lost but interaction effects gained.

## Files
- `research/master_panel.py` — full panel builder (86 cols × 682k rows).
- `research/walk_forward.py` — harness with Calmar ≥ 1.0 floor.
- `strategy/macro_picker/sector_picker_v3.py` — ridge picker per sector.
- `strategy/macro_picker/combine_passing_sectors.py` — combiner harness.
- `strategy/macro_picker/formulas/energy_ridge_v3.json` — Energy coefficients.
- `strategy/macro_picker/formulas/industrials_ridge_v3.json` — Industrials coefficients.
