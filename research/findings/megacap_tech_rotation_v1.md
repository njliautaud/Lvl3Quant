# Megacap-Tech Rotation v1 — HC #581 R2(c) / HC #586 R1 Dispatch B

**Status: FAIL HC #428 R1 across all K-values. NOT DEPLOYABLE.**
**Date**: 2026-06-09
**Backtest output**: `output/macro_picker/megacap_tech_rotation_v1_20260609_170528/`
**MLflow experiment**: `megacap_tech_rotation_v1` (3 runs, K=1/2/3)

---

## Strategy

- **Universe**: AAPL, MSFT, GOOGL, NVDA, META, AMZN, AVGO, TSLA. (TSLA substituted for TSM — TSM ADR not present in price cache.)
- **Picker**: cross-sectional ridge over `ret_20d`, `ret_60d`, `rel_strength_spy` (same shape as `tech_sub_industry_rotation.py` — momentum-rank, fit per WF fold).
- **Rebalance**: weekly (5 trading days), equal-weight inside top-K. No leverage (1.0x).
- **Regime gate**: hold positions only when SPY > 50d MA AND VIX < 25; else sit in cash. Gate enforced at entry AND intra-hold (daily re-check).
- **Costs**: $0.005/share commission + 1bp slippage per trade.
- **Walk-forward**: sliding 24m train / 6m OOT / 3m step (HC #0). Test window 2018-01-01 → 2025-12-31.
- **Anchor**: $20K (HC #580).

**Time in cash**: ~33% (regime gate kept the book in cash about a third of the time — within the 30-50% target range).

---

## Headline Results (all K, on the same OOT window)

| Metric | K=1 | K=2 | K=3 | SPY B&H | QQQ B&H |
|---|---|---|---|---|---|
| CAGR % | 134.5 | 119.6 | 109.3 | 13.2 | 19.1 |
| Sharpe | 1.69 | 1.99 | 2.12 | 0.70 | 0.82 |
| Sortino | 2.07 | 2.47 | 2.63 | 0.86 | 1.07 |
| Calmar | 2.29 | 3.52 | **4.65** | 0.39 | 0.54 |
| Max DD % | **-58.8** | -34.0 | -23.5 | -34.1 | -35.6 |
| PF | 1.47 | 1.56 | 1.61 | — | — |
| WR % | 34.0 | 34.8 | 34.6 | — | — |
| Final equity (on $20K) | $3.24M | $2.19M | $1.65M | ~$54K | ~$84K |
| OOT days | 1505 | 1505 | 1505 | — | — |

**On absolute risk-adjusted metrics, all three K's BEAT both SPY and QQQ buy-and-hold by ~3x Sharpe and 4-9x CAGR.** K=3 has the best Calmar and the lowest drawdown. The strategy is finding real momentum signal in the megacap basket.

---

## HC #428 R1 Gate (REGIME-AGNOSTIC) — THE KILLER

| K | Green-day Sharpe | Red-day Sharpe | Gap (vs 0.50 ceiling) | Day-conc (vs 0.70 ceiling) | OOT days | **PASS** |
|---|---|---|---|---|---|---|
| K=1 | +6.29 | -4.23 | **1.67** | 0.008 | 1505 | **FAIL** |
| K=2 | +7.95 | -5.94 | **1.75** | (low) | 1505 | **FAIL** |
| K=3 | +8.92 | -6.64 | **1.74** | (low) | 1505 | **FAIL** |

**Diagnosis**: this is structurally the same failure mode the wheel, leader rotation, tech sub-industry rotation, and tech blend hit today. Despite the regime gate (which IS doing real work — it cuts the worst red-day exposure), the strategy still earns its entire CAGR on green days and bleeds heavily on red days. The 1.67-1.75 green/red gap is the carry — it can't be hedged away by a long-only single-stock rotation. The basket is megacap tech, which IS the long-beta engine of the index, so even with weekly turnover and a VIX gate, every red day is a red day for the holdings.

**Day-concentration is fine** (well below 0.70 ceiling) — losses are spread, not from one or two outliers. The failure is purely the green/red asymmetry.

---

## Single-Name Concentration (max % of portfolio over test window)

| K | Worst name | Max time-weight % | Notes |
|---|---|---|---|
| K=1 | **TSLA** | **43.0%** | Picker spent ~43% of bull rebalances holding TSLA alone. Severe single-name risk. |
| K=2 | TSLA | 27.4% | Diversified across 8 names with reasonable spread. |
| K=3 | TSLA | 20.1% | Most even distribution; max name barely above equal-weight 12.5%. |

K=1 concentration risk is unacceptable even setting aside HC #428 R1: a 43% TSLA load is "one earnings miss = -20% portfolio day." K=3 is by far the cleanest from a single-name risk standpoint.

---

## Single-Day -10% or Worse Losses (Earnings / Vol Events)

| K | # days ≤ -10% | Worst day | Worst-day book ret | Driver |
|---|---|---|---|---|
| K=1 | **20** | 2020-02-05 | **-17.2%** | TSLA single-name |
| K=2 | 4 | 2020-03-12 (COVID) | -12.x% | Broad-market |
| K=3 | **2** | 2020-03-x | -10.x% | Broad-market |

K=1 has TWENTY days of -10% or worse, driven overwhelmingly by TSLA single-name moves (8+ of the 20 are TSLA-only earnings/vol events). K=2 cuts to 4 days, K=3 to 2 days. This is consistent with concentration — diversifying across 3 names dilutes any single-name shock to manageable size, but it doesn't fix the regime asymmetry.

---

## Why It Fails (Structural)

This is the SIXTH directional/long-only strategy tested today (after wheel base, QQQ/IWM wheel expansion, hedged wheel, regime-gated wheel, short-DTE wheel, tech sub-industry rotation, tech blend) that fails HC #428 R1 via the same mechanism: **long-only equity-beta-correlated structures earn on green tape and bleed on red tape**. The regime gate at entry helps but doesn't eliminate the asymmetry because (a) the gate is leaky (VIX<25 still includes plenty of red days), (b) holdings opened on bull days carry through gate flips, and (c) the basket IS the beta-engine of the index — there is no version of "long megacap tech only when SPY > MA AND VIX < 25" that produces symmetric green/red performance.

Per HC #585 R3 / HC #586 R3, this is increasingly looking like a fundamental incompatibility between HC #428 R1's symmetric-Sharpe gate and any directional strategy. The user-decision on whether to relax HC #428 R1 to a tail-DD gate is now actively blocking deployment of a strategy that would otherwise be the best deployable candidate from today's research (K=3 Sharpe 2.12, Calmar 4.65, MaxDD -23.5%, beats SPY by 8x CAGR with lower drawdown).

---

## Verdict & Next Step

**No paper engine module written** (per dispatch spec — "If NONE pass, do NOT write the engine module"). All three K-values failed HC #428 R1.

This was the FINAL HC #581 / HC #586 R1 dispatch tonight. Per HC #586 R2, no further quant dispatches without explicit user instruction.

**Recommended escalation** (consistent with HC #585 R3 / HC #586 R3): surface to user that the megacap-tech K=3 variant has materially better absolute risk-adjusted metrics than anything else tested today (Sharpe 2.12, Calmar 4.65, MaxDD -23.5%, beats SPY+QQQ by 3x Sharpe), but is blocked by the symmetric green/red gate. User decides:
- (1) Relax HC #428 R1 → megacap-tech K=3 becomes deployable.
- (2) Keep HC #428 R1, accept tonight's results as "directional lane is structurally exhausted" → pivot exclusively to market-neutral.
- (3) Continue iterating (against HC #586 R2 dispatch bound).
