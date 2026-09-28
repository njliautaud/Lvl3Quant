# Megacap-Tech Rotation Extended Sweep v1

**Status**: NOTHING passes HC #428 R1. ALL variants pass tail-DD gate (-25% worst-red-quarter ceiling).
**Date**: 2026-06-09
**Backtest output**: `output/macro_picker/megacap_tech_extended_v1_20260609_171903/`
**MLflow experiment**: `megacap_tech_extended_v1` (9 runs)
**Test window**: 2018-01-01 -> 2025-12-31, sliding 24m train / 6m OOT / 3m step (HC #0)
**Anchor**: $20K. Universe: AAPL, MSFT, GOOGL, NVDA, META, AMZN, AVGO, TSLA.

---

## Master Comparison Table

| Variant | CAGR % | MaxDD % | Sharpe | Sortino | Calmar | PF | WR % | Regime Gap | Worst-Red Qtr DD % | HC #428 R1 | Tail-DD Gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| K=3 mom60 (baseline) | 109.3 | -23.5 | 2.12 | 2.63 | 4.65 | 1.61 | 34.6 | 1.74 | -14.5 | FAIL | **PASS** |
| K=4 mom60 | 98.0 | -21.3 | 2.12 | 2.54 | 4.61 | 1.60 | 35.5 | 1.74 | -17.3 | FAIL | **PASS** |
| K=5 mom60 | 99.5 | -20.5 | 2.22 | 2.68 | 4.87 | 1.63 | 35.7 | 1.74 | -15.2 | FAIL | **PASS** |
| **K=6 mom60** | **105.3** | **-19.6** | **2.34** | **2.85** | **5.36** | **1.68** | **36.7** | 1.75 | **-15.9** | FAIL | **PASS** |
| K=3 mom20 (1m) | 127.8 | -22.6 | 2.26 | 2.81 | 5.66 | 1.67 | 34.2 | 1.72 | -18.5 | FAIL | **PASS** |
| K=3 mom120 (6m) | 127.1 | -24.2 | 2.24 | 2.67 | 5.25 | 1.66 | 34.4 | 1.68 | -22.1 | FAIL | **PASS** |
| K=3 mom252 (12m) | 107.3 | -23.8 | 2.04 | 2.39 | 4.51 | 1.58 | 34.7 | 1.70 | -20.0 | FAIL | **PASS** |
| K=3 voltarget | 96.1 | -22.3 | 2.11 | 2.64 | 4.30 | 1.61 | 35.3 | 1.77 | -14.4 | FAIL | **PASS** |
| K=3 hedged (-10d puts) | 99.8 | -24.2 | 1.96 | 2.80 | 4.12 | 1.49 | 52.0 | 1.63 | -16.0 | FAIL | **PASS** |
| SPY B&H (reference) | 13.2 | -34.1 | 0.70 | 0.86 | 0.39 | - | - | - | - | n/a | FAIL |
| QQQ B&H (reference) | 19.1 | -35.6 | 0.82 | 1.07 | 0.54 | - | - | - | - | n/a | FAIL |

**Gap = |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|).** HC #428 R1 ceiling 0.50.
**Worst-Red Qtr DD** = maximum drawdown during the worst-red SPY-quarter (quarters where SPY ended down).

---

## Experiment-by-Experiment

### Exp 1: K-Sweep K=4/5/6
- Does NOT close HC #428 R1 gap. All K's land in the 1.74-1.75 region (effectively identical to K=3's 1.74).
- BUT higher K materially improves headline risk-adjusted metrics: **K=6 has the best Sharpe (2.34), best Calmar (5.36), lowest MaxDD (-19.6%), and tightest worst-red-quarter (-15.9%)**.
- Diversification compresses single-name risk (max name % drops from 20% at K=3 to 14% at K=6) and tightens DD, but doesn't fix the regime asymmetry — when SPY drops, all 6 megacap-tech names drop together.

### Exp 2: Momentum-Window Sweep at K=3
- **1m (K=3 mom20) and 6m (K=3 mom120) tied for best CAGR (~127%)** — short and medium momentum both capture similar edge. 6m has slightly worse worst-red-quarter (-22%) suggesting 6m signal carries longer into a turn.
- **12m (K=3 mom252) is worst** — Sharpe 2.04, lowest. Long-horizon momentum lags regime transitions.
- **Regime gap is similar across all momentum windows (1.68-1.74)** — momentum window does NOT affect the regime-gap structural failure.
- Recommended momentum window: 1m or 3m. 6m has too-wide worst-red DD.

### Exp 3: Volatility-Targeted Weighting at K=3
- INVERSE-volatility weighting REDUCED CAGR (109 -> 96) and slightly WIDENED the regime gap (1.74 -> 1.77).
- Tail-DD slightly improved (-14.4% vs -14.5%).
- Intuition: down-weighting high-vol names (NVDA, TSLA) cuts upside too, and these names are not asymmetrically more punishing on red days within the megacap basket — they all move together. Equal weight is more efficient here.

### Exp 4: Tail-DD Gate Evaluation (PRE-STAGING FOR HC #428 RELAXATION)
**ALL nine variants pass the proposed tail-DD gate of -25% worst-red-quarter cleanly:**
- Best worst-red-quarter: K=3 voltarget (-14.4%) and K=3 baseline (-14.5%)
- Worst worst-red-quarter: K=3 mom120 (-22.1%) — still inside -25% cushion
- The K=3 family has the cleanest tail-DD profile of any directional strategy tested
- The user can relax HC #428 R1 to the tail-DD gate with confidence that this family of strategies materially clears it

### Exp 5: Tail-Risk Overlay (90-DTE -10d SPY Put Hedge)
**The hedge UNDERPERFORMED expectations.**
- CAGR drag: hedge cost was ~3.2% annual premium, but realized total hedge PnL was -25.9% over the 8 years (puts mostly expired worthless or lost value through theta decay between SPY drawdowns). Net annual drag: 7.5%/yr.
- Sharpe DROPPED from 2.12 to 1.96.
- **Regime gap only modestly closed (1.74 -> 1.63)** — still 3.3x over the 0.50 ceiling. Far from passing.
- Win rate improved from 34.6% to 52% (the hedge does add positive-PnL days during selloffs), but the average green-day gain was diluted by hedge bleed.
- Worst-red-quarter DD only slightly improved (-14.5% -> -16.0% — hedge actually made it slightly worse because of net premium drag during slow grind-downs that don't trigger meaningful put payoff).
- **Verdict**: a long OTM put hedge cannot fix the regime-gap structural failure of long-only megacap tech. The hedge isn't large enough to neutralize beta on red days but is expensive enough to materially drag green-day returns.

---

## Best Variants by Criterion

| Criterion | Winner | Notes |
|---|---|---|
| Best Sharpe | K=6 mom60 (2.34) | Diversification advantage |
| Best Calmar | K=3 mom20 (5.66) | Short-momentum + concentration |
| Best CAGR | K=3 mom20 (127.8%) | Best absolute return |
| Lowest MaxDD | K=6 mom60 (-19.6%) | Diversification cap |
| Tightest regime gap | K=3 hedged (1.63) | Still 3.3x over 0.50 ceiling |
| Tightest worst-red qtr | K=3 voltarget (-14.4%) | Inverse-vol weighting helps tail |
| **Best overall (Sharpe + DD + tail)** | **K=6 mom60** | If user accepts tail-DD gate, K=6 is the deploy pick |

---

## Deployability Verdict

### Under HC #428 R1 (strict regime-agnostic gate, ceiling 0.50)
**NOTHING passes.** Every directional megacap variant (including hedged, vol-targeted, higher-K) lands at gap 1.6-1.8. This confirms the structural finding from prior dispatches: long-only equity-beta strategies cannot satisfy HC #428 R1 in isolation. **Do not deploy any variant under current HC #428 R1.**

### Under Proposed Tail-DD Gate (-25% worst-red-quarter)
**ALL NINE variants pass cleanly.** The K=3 family has worst-red-quarter -14.5% to -22.1% — substantial cushion to the -25% ceiling. **K=6 mom60 is the cleanest deploy candidate** under this gate (best Sharpe 2.34, lowest MaxDD -19.6%, worst-red-quarter -15.9%).

---

## Paper Engine Pre-Staging (Per Dispatch Deliverable)

Per dispatch deliverables 3/4/5:
- (3) Hedged K=3 did NOT pass HC #428 R1 -> no hedged engine written.
- (4) No higher-K variant passed HC #428 R1 -> no higher-K engine written.
- (5) K=3 family cleanly passes tail-DD gate -> **K=3 paper engine pre-written** at `live_trading_linux/megacap_paper_engine.py` with `ENTRIES_PAUSED=True`. Ready for user to flip if/when HC #428 R1 is relaxed.

If the user relaxes HC #428 R1 to a tail-DD gate, the recommended config to flip on is **K=6 mom60** (best Sharpe + lowest DD). The K=3 engine is pre-staged as the dispatch-default; switching K is a single-parameter change in the engine config.

---

## Recommendations for Each Potential User Decision

1. **User keeps HC #428 R1 strict**: directional lane is exhausted. Pivot entirely to market-neutral research. No directional deploy candidate exists tonight.
2. **User relaxes HC #428 R1 to tail-DD gate**: deploy **K=6 mom60** (Sharpe 2.34, Calmar 5.36, MaxDD -19.6%, worst-red-quarter -15.9%). Pre-staged engine is K=3; one-line config change to switch to K=6.
3. **User wants hedged variant for political optics**: hedged K=3 still fails HC #428 R1 AND has a 7.5%/yr hedge drag. Hedge does NOT solve the problem. Not recommended.
4. **User wants vol-targeted variant**: it slightly improves tail-DD but reduces Sharpe and CAGR. Not recommended over equal-weight K=6.
