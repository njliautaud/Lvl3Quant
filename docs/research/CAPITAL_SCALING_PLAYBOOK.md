# CAPITAL SCALING PLAYBOOK
**Last updated: 2026-07-17**

This is the step-by-step plan for growing the trading account from micro ($500) to institutional ($250K+). Every number comes from validated backtests in STRATEGY_CATALOG.md. Nothing is made up.

---

## HOW TO READ THIS DOCUMENT

- **Validated** = passed permutation test (p < 0.05), regime test (works in bull AND bear), and sub-period consistency. These are real.
- **Paper-tested** = running with real market prices but no real money yet. Numbers are promising but unproven.
- **Backtested** = tested on historical data only. Take with appropriate skepticism.
- All performance numbers assume commission-free equity/ETF trading (Robinhood) unless noted.
- Options strategies assume standard retail commissions (~$0.65/contract).

---

## PHASE 1: MICRO ($500 - $2,000)
**Current phase. Account opened July 2026.**

### What to Trade
**UPRO with VIX Protection (Strategy 1C)** — the only validated strategy tradeable at this size.

Hold 3x leveraged S&P 500 (UPRO, ~$85/share) and scale position based on VIX:
- VIX < 17: fully invested in UPRO
- VIX 17-25: 30% UPRO, 70% cash
- VIX > 25: 100% cash

Plus a 4-signal protection overlay that catches danger before VIX spikes:
1. VIX < 20
2. SPY above its 50-day moving average
3. Credit markets healthy (HYG/LQD not stressed)
4. Market breadth > 50% of stocks above their 50-day average

If 2+ signals flash warning, halve the position. All 4 warning = go to cash.

### Allocation
| Sleeve | Weight | What |
|--------|--------|------|
| UPRO (VIX-gated) | 90% | Growth engine |
| Cash reserve | 10% | Panic-buy dry powder |

### Expected Performance (from validated backtest, 2013-2026)
| Metric | Value |
|--------|-------|
| CAGR | 67.6% |
| Max Drawdown | -9.1% |
| Sharpe | 3.24 |
| Sortino | 4.82 |

**Crisis protection track record:**
- COVID crash (2020): -12.4% vs SPY -33.4%
- 2022 rate hikes: -0.2% vs SPY -24.1%
- Q4 2018 selloff: -8.0% vs SPY -18.7%

### What's Already Built
- VIX daily allocator running on cron (3:30 PM ET daily)
- 4-signal protection overlay integrated
- Live position: 2.88 shares UPRO ($X) on Robinhood since July 16, 2026
- Panic confluence monitor watching for VIX > 30 buy signals

### Transition Criteria to Phase 2
- Account reaches $2,000 (either through deposits or growth)
- At least 3 months of live track record showing the system works as expected

### Kill Switch
- If live MaxDD exceeds -20% (more than 2x backtested worst), pause and review
- If Sharpe drops below 1.0 over rolling 6 months, investigate

---

## PHASE 2: SMALL ($2,000 - $10,000)

### What to Add
**ETF Short-Term Reversal (Strategy 1B)** — a second validated strategy with almost zero correlation to UPRO.

Every week, rank 21 ETFs (sector SPDRs, broad market, fixed income, commodities) by their trailing 5-day return. Buy the 5 worst performers with equal weight. Hold 5 days. Repeat.

This is genuinely independent from UPRO (correlation ~0.06), so adding it actually reduces risk while maintaining returns.

### Allocation
| Sleeve | Weight | Min Capital | What |
|--------|--------|-------------|------|
| UPRO (VIX-gated) | 60% | $1,200 | Growth engine |
| ETF Reversal | 30% | $600 ($120/position x 5) | Weekly rebalance income |
| Cash reserve | 10% | $200 | Panic-buy dry powder |

### Combined Expected Performance
| Metric | UPRO Only | UPRO + Reversal Combined |
|--------|-----------|--------------------------|
| CAGR | 67.6% | ~45-55% (blended, lower but diversified) |
| Max Drawdown | -9.1% | ~-8% (diversification benefit) |
| Sharpe | 3.24 | ~2.5-3.0 (reversal Sharpe 0.83 drags slightly) |

The CAGR drops because 30% of capital is in the lower-returning reversal strategy. The trade-off: better risk-adjusted returns and smaller drawdowns through genuine diversification.

### ETF Reversal Details (validated, 23 years of data)
| Metric | Value |
|--------|-------|
| Sharpe | 0.83 |
| Win Rate | 58% |
| Profit Factor | 1.43 |
| Annual Return | 14.7% |
| Permutation p | 0.000 |
| Regime Gap | 0.03-0.23 (works in ALL conditions) |

### Schedule
- **Daily (3:30 PM ET):** VIX allocator adjusts UPRO position
- **Weekly (Friday close):** Rebalance ETF reversal basket (sell all 5, buy new bottom 5)

### Transition Criteria to Phase 3
- Account reaches $10,000
- At least 6 months combined track record
- Both strategies performing within 1 standard deviation of backtested expectations

---

## PHASE 3: MEDIUM ($10,000 - $50,000)

### What to Add
At this level, two new things become possible:

**1. Panic Buying with Real Firepower (Strategy 1A)**
Reserve 5-10% of capital specifically for VIX > 30 panic-buy signals. These fire 2-6 times per year but are the single most robust edge found in all testing.

| Panic Config | Sharpe | Win Rate | Profit Factor |
|--------------|--------|----------|---------------|
| VIX > 30, hold 5 days | 1.18 | 59% | 1.50 |
| VIX > 30, hold 20 days | 1.38 | 68% | 2.62 |
| Breadth < 30%, hold 10 days | 9.46 | 78% | 4.54 |

**2. Cross-Asset Momentum as Crash Hedge (Strategy 2D)**
Allocate across 13 ETFs spanning 5 asset classes based on trailing momentum. Terrible standalone returns (5.7% CAGR) but almost zero correlation to equities (0.12 to SPY) and made +8.5% during the 2008 crisis. This is portfolio insurance.

### Allocation at $25,000 (midpoint)
| Sleeve | Weight | $ Amount | What |
|--------|--------|----------|------|
| UPRO (VIX-gated) | 45% | $11,250 | Growth engine |
| ETF Reversal | 20% | $5,000 | Weekly rebalance income |
| Cross-Asset Momentum | 15% | $3,750 | Crash protection hedge |
| Panic-buy reserve | 10% | $2,500 | Crisis alpha (cash until VIX > 30) |
| Cash buffer | 10% | $2,500 | General safety |

### Combined Expected Performance at $25K
| Metric | Estimate |
|--------|----------|
| CAGR | ~30-40% |
| Max Drawdown | ~-10 to -13% |
| Sharpe | ~2.0-2.5 |
| Annual Income | ~$7,500-$10,000 |

### At $50K: Options Become Possible
Once capital reaches $50K, cash-secured puts become feasible. This is the bridge to Phase 4. However, the CSP strategy is still in paper testing (bugs fixed July 16, 2026 — needs 60+ clean paper days before deploying real money, target September 2026).

**Do NOT rush into options.** The honest assessment from testing:
- Black-Scholes backtest said CSP returns ~10%+ CAGR
- Real bid/ask pricing showed closer to 6-8% CAGR for full wheel
- Paper trading initially showed +$5,579 but audit revealed 3 critical bugs; real P&L was approximately -$309
- Clean paper data collection restarted July 16, 2026

### Transition Criteria to Phase 4
- Account reaches $50,000
- CSP paper trading shows positive results over 60+ clean days
- At least 1 full market cycle (or significant drawdown event) survived with the system

---

## PHASE 4: LARGE ($50,000 - $250,000)

### The Full Strategy Suite Opens Up

At $50K+, the validated combined portfolio (Strategy 1D) becomes fully deployable. This was backtested over 13.5 years with walk-forward validation.

### Allocation at $50K
| Sleeve | Weight | $ Amount | Strategy |
|--------|--------|----------|----------|
| Protected UPRO | 25% | $12,500 | VIX-gated leveraged growth |
| Megacap Momentum | 35% | $17,500 | QQQ above 200 SMA |
| ETF Rotation | 15% | $7,500 | Rebalancing premium |
| Vol-Selling (proxy) | 15% | $7,500 | When validated: real CSP/wheel |
| Low-Vol Income | 5% | $2,500 | Defensive income |
| Cash | 5% | $2,500 | Panic reserve |

### Validated Portfolio Performance (Strategy 1D, 2013-2026)
| Metric | Portfolio | SPY Buy & Hold |
|--------|-----------|----------------|
| Sharpe | 2.10 | 0.82 |
| Sortino | 3.06 | 1.00 |
| CAGR | 34.3% | 14.9% |
| Max Drawdown | -13.1% | -33.7% |
| Calmar Ratio | 2.61 | 0.44 |

### Income Projections (from validated backtest)
| Account Size | Monthly Income | Annual Income |
|-------------|---------------|---------------|
| $50,000 | ~$1,430 | ~$17,150 |
| $100,000 | ~$2,860 | ~$34,300 |
| $250,000 | ~$7,150 | ~$85,750 |

### Options Integration (when paper-validated)

**Cash-Secured Puts (CSP):** Once 60+ days of clean paper trading confirm profitability (target: September 2026), deploy on high-quality tech names:
- Core universe: NVDA, AMZN, TSLA, GOOGL, SHOP, DDOG, META, NFLX, ORCL, AVGO
- Vol-sizing: larger positions on stable names, smaller on volatile (LGBM vol forecaster, IC=0.752)
- Earnings filter: blocks entries before earnings announcements
- Realistic expectation: Sharpe ~1.1-1.4 (honest range after audit)
- Conservative projection: ~$40K/year on $100K capital

**Covered Calls on UPRO:** Overlay 40-delta calls on the UPRO position.
- Adds +1.4% to +3.4% CAGR with regime gap 0.22 (passes R1)
- This is the "wheel" component — collect premium while holding

**What NOT to deploy (proven failures):**
- Iron condors: real pricing shows 0.86% CAGR vs 207% in backtest
- Strangles without careful management: real Sharpe 0.3-0.8 vs 4.55 in backtest
- Earnings vol selling: real win rate 22.8% vs 86.5% in backtest

### Risk Management at Scale
- **Position limits:** No single CSP position > 5% of portfolio
- **Sector concentration:** Max 30% in any one sector
- **Protection overlay always on:** 4-signal system, never overridden
- **Day-concentration cap:** ≤ 70% of portfolio can be in correlated positions
- **Margin usage:** Never exceed 50% of available margin

### Transition Criteria to Phase 5
- Account reaches $250,000
- 12+ months track record with the full portfolio
- Drawdown never exceeds -20%
- Options strategies paper-validated and live for 6+ months

---

## PHASE 5: INSTITUTIONAL ($250,000+)

### Full Deployment

At this level, all validated strategies run simultaneously with professional risk management.

### Allocation at $250K
| Sleeve | Weight | $ Amount | Strategy |
|--------|--------|----------|----------|
| Protected UPRO | 20% | $50,000 | VIX-gated leveraged growth |
| Megacap Momentum | 25% | $62,500 | QQQ trend following |
| Diversified CSP | 20% | $50,000 | Full 10+ name universe, vol-sized |
| ETF Reversal | 10% | $25,000 | Weekly rebalance |
| Cross-Asset Momentum | 10% | $25,000 | Crash protection hedge |
| Covered Call Overlay | 5% | $12,500 | Income on UPRO/positions |
| Panic Reserve | 5% | $12,500 | VIX > 30 crisis alpha |
| Cash Buffer | 5% | $12,500 | General safety |

### Income Projections at Scale
| Account Size | Monthly Income | Annual Income | Notes |
|-------------|---------------|---------------|-------|
| $250,000 | ~$7,150 | ~$85,750 | Full portfolio, validated CAGR 34.3% |
| $500,000 | ~$14,300 | ~$171,500 | Income strategies generate meaningful cash flow |
| $1,000,000 | ~$28,600 | ~$343,000 | Consider reducing leverage at this level |

### Rebalancing Rule (UPDATED — July 17, 2026)
**Use Threshold 15% rebalancing.** When any sleeve drifts more than 15 percentage points from its target weight, rebalance. Updated from 5% based on comprehensive 10-method test:
- Threshold 15% = best Sharpe (1.096), only 0.6 rebalances/year
- Calendar rebalancing (monthly/weekly) is WORSE — too much turnover, no benefit
- Quarterly is runner-up (Sharpe 1.037, best MaxDD at -15.7%)
- Rebalancing premium is massive: Sharpe +0.531 vs never rebalancing
- TX cost negligible (~0.37% total over 14 years at 10bps round-trip)

**Practical rule:** Check allocations quarterly. Only trade if any position drifted >15% from target.

### Withdrawal Rules (Validated — July 17, 2026)
At Phase 4+ ($50K+), can begin sustainable withdrawals while portfolio continues growing:
- **4% fixed annual** = ~$6,070/yr income at $50K, portfolio grows to 8.3x over 14yr
- **5% fixed annual** = ~$6,927/yr income, portfolio grows to 7.2x
- **Variable (4% good / 2% bad year)** = $5,942/yr, portfolio grows to 9.1x — best risk-adjusted
- **Guardrails (3-6% adaptive)** = $5,278/yr, 8.9x — most stable income
- Zero ruin risk across ALL withdrawal rates tested (up to 10%)
- At $100K: double all income numbers above

### Professional Risk Management Overlay
1. **Portfolio-level VaR:** Daily 95% VaR should not exceed 2% of portfolio
2. **Correlation monitoring:** If UPRO + QQQ correlation exceeds 0.95, reduce combined weight to 40%
3. **Drawdown circuit breaker:** If portfolio draws down 15% from peak, reduce all risk 50% for 10 trading days
4. **Annual rebalancing:** Re-run all strategy validations annually to confirm edges persist
5. **Strategy decay detection:** If any strategy's rolling 12-month Sharpe drops below 0.5, reduce allocation by half and investigate

### At $500K+: Consider Portfolio Margin
Portfolio margin (available at most brokers above $150K) allows more efficient capital usage:
- CSP margin requirements drop ~60%
- Can run more concurrent positions with same capital
- WARNING: More leverage = more risk. Do NOT increase gross exposure just because margin allows it. Use the freed capital for the cash/hedge sleeves.

---

## DRAWDOWN PROTECTION RULES (ALWAYS ON) — UPDATED JUL 17

These rules apply at EVERY phase. Never override them.

### PRIMARY: Vol-Adjusted Leverage (Validated Jul 17, 2026)
**This replaces the old VIX-based gating.** Based on 252-config grid search, walk-forward validated (7% Sharpe degradation OOS), sub-period consistent.

| 21-Day Realized SPY Vol | Allocation | Rationale |
|--------------------------|------------|-----------|
| < 20% | **UPRO** (3x leveraged S&P) | Low vol = leverage delivers 2.9-3.4x effective multiplier |
| 20% - 30% | **SPY** (unleveraged) | Medium vol = leverage decay starts hurting |
| > 30% | **GLD + TLT** (50/50 safe havens) | High vol = leverage is destructive |

**Plus SMA50 protection overlay:** If SPY < 50-day SMA, go to cash regardless of vol level. This single rule BOOSTS effective leverage from 2.1x to 3.6x by avoiding drawdowns.

**Check frequency:** Weekly (or when market moves significantly). ~6 switches per year on average.

**Backtested performance (13.5yr, $500 start + $100/wk):**
- Vol-adjusted: $678K final, Sharpe 1.78, MaxDD -38%
- vs naked UPRO: $800K but MaxDD -77%
- vs SPY: $244K, Sharpe 2.63, MaxDD -33%

### SECONDARY: 6-Factor Risk Score (Early Warning)
| Signal | Green | Red |
|--------|-------|-----|
| Vol ratio (5d/21d) | < 1.2 | > 1.2 (rising short-term vol) |
| VIXY vs 5-day SMA | Below | Above (fear rising) |
| Credit spread (LQD/HYG) | Stable | Widening >0.5% in 5 days |
| SPY drawdown from peak | < 5% | > 5% |
| Breadth (IWM-SPY 21d) | Positive | < -3% (small caps lagging) |
| Gold vs SPY (10d) | Underperforming | Outperforming >2% (flight to safety) |

- **0-1 red:** Normal — follow vol regime allocation
- **2-3 red:** Heightened risk — consider partial deleverage (UPRO → SPY)
- **4+ red:** High risk — defensive allocation even if vol is low

### The Old 4-Signal Protection Overlay (STILL VALID, SECONDARY)
| Signal | Green | Yellow | Red |
|--------|-------|--------|-----|
| VIX | < 20 | 20-25 | > 25 |
| SPY vs 50 SMA | Above | Within 1% | Below |
| Credit (HYG/LQD) | Healthy | Stressed | Very stressed |
| Breadth | > 50% | 30-50% | < 30% |
| 17-25 | 30% | Reduce, watch closely |
| > 25 | 0% | Cash |
| > 30 + breadth < 30% | Deploy panic reserve | BUY signal (crisis alpha) |

### Kill Switches
| Condition | Action |
|-----------|--------|
| Live MaxDD > 2x backtested worst | Pause strategy, investigate |
| Rolling 6-month Sharpe < 0.5 | Reduce allocation 50%, review |
| Single-day loss > 5% of portfolio | Halt new entries for 48 hours |
| System malfunction (stale data, missed signals) | Go to cash until fixed |

---

## WHAT'S AUTOMATED VS MANUAL

### Fully Automated (running now)
- VIX daily allocator (3:30 PM ET cron)
- 4-signal protection overlay
- Panic confluence monitor (alerts when VIX > 30 + breadth + credit align)

### Semi-Automated (alerts generated, human executes)
- ETF reversal weekly rebalance (ranking automated, trades manual)
- Panic buy entries (alert fires, user confirms)
- Earnings gap opportunities (scanner flags, user decides)

### Manual (needs human judgment)
- Phase transitions (increasing capital allocation)
- Options strategy deployment (after paper validation)
- Kill switch responses (system flags, user decides response)

---

## HONEST ASSESSMENT

### What we KNOW works (high confidence)
- UPRO with VIX protection: 13.5 years of data, Sharpe 3.24, passes all adversarial tests
- ETF reversal: 23 years of data, Sharpe 0.83, zero survivorship bias, works in all regimes
- Panic buying: 20+ years of data, every crisis confirms the edge
- Combined portfolio: Sharpe 2.10 over 13.5 years, walk-forward validated

### What we THINK works (medium confidence, needs more live data)
- CSP income on quality names (Sharpe ~1.1-1.4 in honest range, but paper bugs found)
- Covered call overlay on UPRO (+1.4-3.4% CAGR, passes R1)
- Cross-asset momentum as a hedge (low returns but excellent crash protection)

### What DOESN'T work (proven failures, don't retry)
- ML/RL timing of leveraged ETFs (4 architectures tested, all lose to simple VIX rules)
- Iron condors (real pricing kills the edge entirely)
- Sector rotation (fails permutation test — random rotation is equally good)
- Strangles without management (real Sharpe drops 80%+ from backtest)
- Earnings vol selling (data artifact, real win rate 22.8%)
- Pairs trading / stat arb on ETFs (cointegration unstable, 0/10 pairs pass any gate)

### Timeline
| Milestone | Target Date | Dependency |
|-----------|------------|------------|
| Phase 1 live (UPRO) | NOW (July 2026) | Done |
| Add ETF reversal | When account hits $2K | Capital |
| CSP paper validation complete | September 2026 | 60+ clean paper days |
| Phase 3 entry | Account hits $10K | Capital + time |
| Options deployment | After CSP validated | Paper results |
| Full portfolio | Account hits $50K | Capital + options validated |
