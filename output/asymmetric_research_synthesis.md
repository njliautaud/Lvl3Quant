# Asymmetric Signal Research — Synthesis of Findings
## Session: July 21-22, 2026 (HC #725/#726)

## THE BIG PICTURE

We mapped every type of asymmetric upside signal across stocks, sectors, and the market. Here's what's real and what's not.

---

## WHAT WORKS (Statistically Validated)

### 1. Trend CTA (Sharpe 0.91, perm p=0.000)
- **What**: 6-month dual momentum across 8 ETFs, top-3 equal weight, monthly rebalance
- **Why it works**: Momentum is the most persistent factor in markets. Diversification across asset classes (bonds, commodities, REITs, equities) provides genuine edge.
- **Current positions**: DBC, VNQ, EEM
- **Status**: Paper trading (PM2 86/87)

### 2. Jade Lizard Income (Sharpe 1.77, bootstrap p=0.035)
- **What**: Sell OTM put + bear call spread on high-IV stocks
- **Why it works**: Harvests volatility risk premium. When IV > RV (90% of the time per our research), selling premium is profitable.
- **Caveat**: Only 44 backtest trades — small sample
- **Status**: Paper trading (PM2 88)

### 3. ML Stock Ranker + Asymmetric Filter (20.7% CAGR, Sharpe 1.13, perm p=0.000)
- **What**: LightGBM ranks stocks by distress features, only trades when asymmetric filter fires (high vol + negative momentum + volume surge)
- **Why it works**: Combines two edges — (a) distressed stocks mean-revert, (b) ML identifies which distressed stocks recover fastest
- **Key insight**: ML alone doesn't work (Strategy A Sharpe 0.49). The FILTER is critical — it concentrates trades in the 26% of months with highest asymmetry.
- **UPGRADE FOUND**: 3-month horizon features (mom_3m, vol_126d, dist_200sma) beat 1-month-only (Sharpe 1.13 vs 0.72). Regime-agnostic (bull 1.10, bear 1.36, div=0.19 PASSES). Combined 1m+3m model also strong (Sharpe 0.97). Lag-robust (T-1 ≈ T-2 IC).
- **Status**: Paper trading (PM2 89/90), upgrade to 3m features pending

### 4. Growth Combo (Sharpe 1.03)
- **What**: 50% Trend CTA + 30% UPRO/200SMA overlay + 20% VIX spike
- **Why it works**: Genuine diversification — CTA has 0.39 SPY correlation
- **Status**: Theoretical, no paper engine yet

---

## WHAT DOESN'T WORK (Failed Validation)

### Fear-Based Timing (Multiple Failures)
- **Asymmetric Harvester v1-v3**: perm p=0.88, p=0.24, Sharpe 0.07
- **Asymmetric Timing v1**: Sharpe 0.234, perm p=0.900
- **Asymmetric Portfolio v2**: perm p=0.555

**Why**: Fear signals are too sparse (fire 5-20% of time). Spending 80% of time defensively positioned kills compounding. The worst days come BEFORE the signal fires (sequencing risk). You can't time fear profitably as a standalone strategy.

### Buying Options on Distressed Stocks
- ATM calls: -22.52% mean return, 31.4% hit rate
- OTM calls: -40.77% mean return, 21.1% hit rate

**Why**: When our distress filter fires, IV is elevated (mean 69%, 3x realized vol). ATM premium costs 8% of stock price. Calls expire worthless 37% of time. The elevated IV makes options too expensive.

### Most Income Strategies
- Wheel: Sharpe 0.365 (honest, no alpha vs buy-and-hold)
- VIX Regime CSP: Sharpe -0.27
- Dividend Capture: FAKE (yfinance auto_adjust artifact)
- PMCC: Sharpe 0.56 (no better than buy-and-hold)

---

## KEY INSIGHTS FROM SIGNAL ANALYSIS

### 1. Fear = Best Forward Returns (but can't time entry profitably)
- VIX > 30: SPY +5.48% 1m, 86% HR
- Neg momentum breadth > 70%: SPY +5.82% 1m, 88% HR
- The BEST setup is the one that feels worst to execute

### 2. Comfort = Worst Forward Returns
- Momentum crowding (RSI > 75): SPY +0.39% 1m
- Low VIX + uptrend: 0.63% monthly (vs 2.49% in High VIX + downtrend)
- When everything looks great, expected returns are thin

### 3. Tech Leads Recoveries
- XLK: +9.3% at 3m from fear entry (76.9% HR, 1.8x up/down)
- Only sector with statistically significant positive alpha vs SPY during recovery (+3.63% at 3m)
- Defensives (XLP, XLU, XLV) UNDERPERFORM during recovery

### 4. Signal Redundancy
- VIX level, realized vol, distance from 200SMA all say the same thing (r > 0.6)
- Only 10 truly independent signals out of 49 computed
- Don't mistake correlated signals for confirmation

### 5. Stock Distress = Buy Signal (in Stock, Not Options)
- High vol + neg momentum + volume surge: +7.0% 1m, 73% HR, 2.9x ratio
- But buy STOCK, not options (IV too elevated for options to work)
- SELL options premium when IV elevated (Jade Lizard)

### 6. Asymmetric Signals DON'T Work as Market Timing (v1, v2A, v2B all failed)
- v1 (defensive during calm): Sharpe 0.234, perm p=0.900
- v2A (lean into fear with UPRO): Sharpe 0.348, perm p=0.985
- v2B (sector tilt during fear): Sharpe 0.578, perm p=0.170
- Signals fire only 10% of time → 90% of returns = baseline SPY → no edge over buy-and-hold
- Use signals as INFORMATION (which stocks to buy, when to add trades) not ALLOCATION DRIVERS

---

## ACTIONABLE PORTFOLIO RECOMMENDATION

Based on all research, the optimal setup uses validated strategies only:

| Allocation | Strategy | Sharpe | Status |
|---|---|---|---|
| 50% | Trend CTA | 0.91 | Paper trading |
| 20% | ML Stock Ranker (Strategy B) | 0.72 | Paper trading |
| 20% | Jade Lizard Income | 1.77 | Paper trading |
| 10% | Crisis Exit Accelerator | TBD | Research |

**Expected portfolio**: Sharpe ~1.0, CAGR ~12-15%, MaxDD ~-20%
**SPY benchmark**: Sharpe 0.81, CAGR 11.1%, MaxDD -33.7%

**Current market regime**: COMPLACENT (0/7 fear signals). Thin forward returns expected. This is normal — the portfolio should be in its baseline allocation. The edge comes during the 20% of time when fear fires and the ML ranker and crisis exit strategies activate.

---

## DAILY MONITORING (Automated)

- **Asymmetric Scorecard**: 0-7 composite, runs 9:55 AM ET daily
- **Paper engines**: Trend CTA (monthly), Jade Lizard (daily), ML Ranker (monthly)
- **Forward validation**: Need 3-6 months of paper data before live deployment

### 7. Cross-Asset Lead-Lag — NO TRADEABLE SIGNAL (All p > 0.05)
- **What**: Do bonds, gold, credit, or VIX systematically lead equity turning points?
- **Result**: 0 out of 20 lead-lag relationships were statistically significant (permutation test, 200 shuffles)
- **Raw observations** (not significant): GLD appears to lead bottoms by 39d (p=0.215), TLT leads tops by 15d (p=0.095), VIX term structure resolves ~7d before bottoms (91% of time)
- **Why it fails**: Only 11 bottoms in 20 years = tiny sample. IS/OOS splits show most leads are UNSTABLE (TLT bottom lead: IS=+43d, OOS=-3d). Cross-correlations peak at lag 0 for every asset — no systematic lead.
- **Practical implication**: Can't use cross-asset signals to TIME equity entries. Consistent with our broader finding that timing signals don't survive rigorous testing.

---

## WHAT'S LEFT TO RESEARCH

1. ~~Timing v2~~ — COMPLETED, FAILED (perm p=0.170-0.985)
2. ~~Crisis-exit + ML ranker combo~~ — COMPLETED, ML ADDS NOISE (crisis model IC drops 34.6x at T+1)
3. ~~Cross-asset lead-lag~~ — COMPLETED, FAILED (0/20 significant, all p>0.05)
4. ~~Multi-horizon signal fusion~~ — COMPLETED, **SUCCESS**: 3m-only Sharpe 1.13 (vs 1m-only 0.72), regime-agnostic. Upgrade ML Ranker to 3m features.
5. Forward validation of paper engines (3-6 months needed)
6. **ACTION ITEM**: Upgrade ML Ranker paper engine (PM2 89/90) to include 3m features
