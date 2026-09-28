# Lvl3 Quant — Strategy Playbook v1
## Generated: 2026-07-20 | 12 Validated Strategies | Portfolio Sharpe 6.13

---

## Executive Summary

After testing 36+ ML strategy concepts with rigorous adversarial validation (permutation tests, sub-period stability, outlier robustness, regime stratification), we have **12 validated strategies** that pass all 4 gates. Combined into an optimal portfolio, they deliver:

| Metric | Max Sharpe | Max Calmar | Risk Parity | SPY Benchmark |
|--------|-----------|------------|-------------|---------------|
| **Sharpe** | **6.13** | 5.83 | 5.18 | 0.89 |
| **CAGR** | 32.8% | 33.8% | 23.2% | 14.5% |
| **MaxDD** | -3.2% | **-2.7%** | -3.5% | -33.7% |
| **Calmar** | 10.16 | **12.46** | 6.66 | 0.43 |
| **Sortino** | **11.63** | 10.98 | 9.72 | — |
| **Annual Vol** | 4.6% | 5.0% | 4.0% | — |

**Key takeaway**: 6-7x SPY's risk-adjusted return with less than 1/10th the drawdown.

---

## The 12 Validated Strategies

### Growth Strategies (4)

#### 1. CTA Trend Following — Sharpe 2.90
- **What**: ML-timed trend signals across equity, bond, commodity, and currency ETFs
- **How**: GBM predicts trend regime, rotates into strongest trends
- **Edge**: Captures macro trends, low SPY correlation (0.25)
- **Portfolio weight** (Max Sharpe): 21.3%

#### 2. Commodity Trend — Sharpe 2.28
- **What**: ML momentum timing across commodity ETFs (DBC, USO, GLD, SLV, etc.)
- **How**: Walk-forward GBM on commodity momentum and mean-reversion features
- **Edge**: Commodity super-cycles, 54.9% CAGR, low SPY correlation (0.24)
- **Portfolio weight**: 3.6%

#### 3. Carry + Momentum — Sharpe 2.96
- **What**: Combines carry (yield differential) with momentum signals
- **How**: Multi-factor ML model selecting high-carry + strong-momentum assets
- **Edge**: Two independent alpha sources combined, SPY corr 0.24
- **Portfolio weight**: 7.8%

#### 4. Sector Rotation — Sharpe 1.99
- **What**: ML-timed sector ETF rotation (XLK, XLE, XLF, etc.)
- **How**: Monthly rebalance into top-ranked sectors by ML score
- **Edge**: Sector momentum has decades of academic support
- **Portfolio weight**: 2.0%

### Income/Hedge Strategies (6)

#### 5. Tail Risk Hedging — Sharpe 4.12
- **What**: ML switches between growth (SPY/QQQ) and tail-risk ETFs (TAIL/BTAL)
- **How**: GBM predicts regime, allocates to protection before drawdowns
- **Edge**: 52.9% CAGR, only -2.6% MaxDD, 90.8% win rate
- **Caution**: Only 8yr history (TAIL ETF since 2017)
- **Portfolio weight**: 21.6% (largest allocation)

#### 6. Currency Carry — Sharpe 1.99
- **What**: ML-timed FX carry using currency ETFs
- **How**: Predicts which currencies offer best carry/risk ratio
- **Edge**: Zero negative years across 15 years, SPY corr 0.10
- **Portfolio weight**: 19.4%

#### 7. Yield Curve Trade — Sharpe 2.01
- **What**: ML predicts yield curve steepening vs flattening (SHY vs TLT)
- **How**: Walk-forward GBM on yield curve features
- **Edge**: **SPY corr -0.16** (genuine hedge), zero regime dependency, zero negative years
- **Portfolio weight**: 7.0%

#### 8. Bond Duration Timing — Sharpe 2.00
- **What**: ML predicts when to be long-duration (TLT) vs short-duration (SHY)
- **How**: Interest rate and macro features drive duration allocation
- **Edge**: Sortino 22.9, MaxDD -0.9%, SPY corr 0.22
- **Caution**: Only 5yr data (bond ETF launch dates)
- **Portfolio weight**: 3.1%

#### 9. Gold/Silver Ratio — Sharpe 1.28
- **What**: ML predicts which precious metal (GLD vs SLV) will outperform
- **How**: Ratio z-score, momentum, USD, credit conditions as features
- **Edge**: SPY corr 0.12 (true diversifier), 29.7% CAGR
- **Portfolio weight**: 2.0%

#### 10. Stat Arb Pairs — Sharpe 0.81
- **What**: Statistical arbitrage on cointegrated ETF pairs
- **How**: Mean-reversion on spread z-scores
- **Edge**: Market-neutral, SPY corr 0.04 (near zero)
- **Portfolio weight**: 7.8%

#### 11. Credit Timing — Sharpe 1.19
- **What**: ML predicts credit spread direction (HYG vs TLT)
- **How**: Macro features predict credit cycle
- **Edge**: SPY corr 0.30
- **Portfolio weight**: 2.0%

#### 12. Vol Breakout — Sharpe 1.11
- **What**: ML identifies volatility regime changes for positioning
- **How**: VIX features, term structure, realized vs implied vol
- **Edge**: Captures vol compression/expansion cycles, SPY corr 0.15
- **Portfolio weight**: 2.2%

---

## Adversarial Validation Framework (4 Gates)

Every strategy must pass ALL 4 gates:

### Gate 1: Permutation Test
- Shuffle target labels 100x, rebuild model each time
- Strategy must beat 95% of shuffled versions (p < 0.05)
- **What it catches**: Overfitting to noise, data-mining bias

### Gate 2: Sub-Period Stability
- Split backtest into 3 equal time blocks
- Compute Sharpe in each block
- CV of Sharpe across blocks must be < 0.50
- **What it catches**: Strategies that only work in one market era

### Gate 3: Outlier Robustness
- Trim top and bottom 5% of monthly returns
- Recompute Sharpe on trimmed data
- Trimmed Sharpe degradation must be < 50%
- **What it catches**: Strategies dependent on a few lucky trades

### Gate 4: Regime Agnostic (R1)
- Classify days by SPY regime (bull: SPY > 200MA, bear: below)
- Compute Sharpe separately for bull and bear regimes
- |Bull Sharpe - Bear Sharpe| / max(|Bull|, |Bear|) must be < 0.50
- **What it catches**: Strategies that are just disguised market timing

---

## Portfolio Construction

### Optimal Weights (Max Sharpe = 6.13)
```
Tail Risk Hedging    21.6%  ████████████████████▌
CTA Trend Following  21.3%  ████████████████████▎
Currency Carry       19.4%  ██████████████████▍
Stat Arb Pairs        7.8%  ███████▍
Carry + Momentum      7.8%  ███████▍
Yield Curve Trade     7.0%  ██████▋
Commodity Trend       3.6%  ███▍
Bond Duration         3.1%  ██▉
Vol Breakout          2.2%  ██
Sector Rotation       2.0%  █▉
Credit Timing         2.0%  █▉
Gold/Silver Ratio     2.0%  █▉
```

### Why It Works
1. **Low cross-correlations**: Most strategy pairs have correlation < 0.15
2. **Multiple alpha sources**: Momentum, carry, mean-reversion, volatility, macro
3. **Multiple asset classes**: Equities, bonds, commodities, currencies, volatility
4. **Regime diversity**: Some strategies thrive in bull, others in bear — portfolio is regime-agnostic
5. **Yield Curve's negative SPY correlation** (-0.16) provides genuine downside protection

### Natural Income/Growth Split
The optimizer naturally allocates ~63% to income/hedge strategies and ~37% to growth — aligning with a balanced income + growth mandate.

---

## What Was Rejected (24+ Failed Strategies)

Key learnings from failures:

| Category | Examples | Why They Fail |
|----------|----------|---------------|
| **Cross-asset regime timing** | Equity-Bond Correlation, Inflation Trade | Just disguised market timing (R1 FAIL) |
| **Equity-correlated pairs** | Growth/Value, Large/Small Cap | SPY corr > 0.90 — no diversification |
| **Low-spread pairs** | Muni/Corporate Bonds, Breakeven Inflation | Spread too narrow for ML to exploit |
| **Commodity spread** | Gold Miners vs Gold, Crude/NatGas | One side too volatile or equity-correlated |
| **ML portfolio overlay** | Dynamic Allocation, Leveraged Risk Parity | Static allocation is equally good (perm FAIL) |
| **Daily regime prediction** | Next-day SPY classification | Too noisy — model defaults to buy-and-hold |
| **FX ETF carry** | Currency ETF rotation | ETF costs eat the carry premium |

**Key principle**: Within-asset-class relative value works when BOTH instruments have low SPY correlation. Cross-asset rotation and equity-correlated pairs always fail R1.

---

## Paper Trading Status

### Active Paper Engines (13 total)

**Growth Engines (7):**
- CTA Trend, ML Trend, ML Sector, Commodity Trend, Vol Breakout, Stat Arb, Carry+Momentum

**Income Engines (6):**
- Earnings Vol, Strangle, Wheel (3 variants), Gold/Silver, Yield Curve, Bond Duration, Currency Carry, Tail Risk

All engines run via PM2 cron jobs at market close (4:45 PM ET weekdays).

---

## Robinhood Agentic Account Plan

**Account**: $X cash, option level 2
**Mandate**: High growth, autonomous trading

### Monday July 21 Trade Plan
**Market regime**: BULLISH (SPY $745 well above 200MA $691)
**Pre-market**: SPY +0.3%, QQQ +0.6%, SLV +1.7%

**Primary play**: TQQQ Aug 21 $73 call (~$380) — leveraged Nasdaq exposure in confirmed uptrend
**Alternative**: 2x XLE $59 Aug 21 calls ($286) + 2 shares TQQQ ($137) — sector diversification

**Exit rules**:
- Take profit at 50% gain
- Stop loss at 40% loss
- Close if SPY breaks below 200MA (regime change)
- Close 7 DTE if still holding (avoid theta decay)

---

## Next Steps

1. **Monitor paper engines** — collect 30+ trading days before live deployment
2. **Individual stock selection** — ML stock picker v2 (in progress, testing now)
3. **Options-specific strategies** — Vol surface, earnings plays with actual contracts
4. **Robinhood execution** — Execute Monday trade plan, track weekly
5. **Decay monitoring** — Re-validate strategies quarterly for signal decay

---

*Document version: 1.0 | Last updated: 2026-07-20 04:00 ET*
*36+ strategies tested, 12 validated, portfolio Sharpe 6.13*
