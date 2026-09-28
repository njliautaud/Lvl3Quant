# Existing Wheel/Options-Selling Setups Analysis

**Date**: 2026-09-28
**Author**: Claude (Opus 4.6)

---

## 1. Paper Engine Performance Summary

We have **8 wheel/options paper engines** built, with 5 currently running in PM2:

### RUNNING ENGINES (PM2 online):

| Engine | Start Date | NAV Now | Realized PnL | Return | Trades | Status |
|--------|-----------|---------|-------------|--------|--------|--------|
| **BPS Conservative** | 2026-07-09 | $157,581 | $57,912 | **+57.6%** | 620 | online (18d uptime) |
| **BPS Paper** | 2026-07-06 | $139,593 | $40,145 | **+39.6%** | 743 | online (11d uptime) |
| **Wheel Diversified** | 2026-07-03 | $128,900 | $4,374 | **+28.9%** | 246 | online (18d uptime) |
| **Wheel IC** | 2026-07-06 | $111,887 | $11,928 | **+11.9%** | 373 | online (2d uptime) |
| **Wheel Paper Balanced** (Full Wheel) | 2026-06-10 | $104,896 | $1,483 | **+4.9%** | 13 trades | online (11d uptime) |

### STOPPED / LOW-ACTIVITY:

| Engine | NAV | Realized PnL | Trades | Status |
|--------|-----|-------------|--------|--------|
| Wheel Paper Engine (Original Scalp) | $100,961 | $338 | 2 | stopped, idle since Sep 12 |
| Wheel V4 | $62,540 | $644 | 11 | stopped, 33 open positions |
| Wheel V5 | $60,768 | -$746 | 636 | stopped, 26 open positions |
| Strangle Paper | $121,693 | $6,762 | 71 | 15 open positions |

### BEST PERFORMER: BPS Conservative (+57.6% in ~82 days)

- Config: 30-delta short leg, $15-wide bull put spreads, 25% margin cap, 10-day DTE, 65% profit-take
- Risk controls: bear gate (SPY < 50d SMA), VIX hard cutoff at 30, equity curve brake, circuit breaker
- 620 trades in 82 days = ~7.6 trades/day average
- Uses Alpaca pricing bridge for real quotes (BS fallback when market closed)
- **Caveat**: This is defined-risk spreads, not naked CSPs. The high return includes aggressive position sizing.

### Wheel Paper Balanced (the true full-wheel engine):

- Config: 0.22 delta puts/calls, 30-45 DTE, 65% profit-take, roll at DTE=1 (allow assignment)
- SPY-only, $100K paper capital, VIX gate at 32, regime-gated (HC #555 macro overlay)
- Walk-forward validated: Sharpe 1.49, CAGR 13.8%, MaxDD -8.4%, PF 2.47, WR 90%
- **Current position**: Short SPY Oct 23 $742 put, entry premium $4.98, opened Sep 21
- **Trade history**: 13 trades since June 10. 6 profit-takes, 1 stop-loss, currently on trade #7
- The stop-loss (July 29) was during a SPY dip to ~$729, but subsequent trades recovered quickly.
- Annualized return on the $100K paper account: ~17% (extrapolated from 110 days of data)

---

## 2. Current Robinhood Holdings: Covered Call Opportunities

### FRSH (Freshworks) -- THE ONLY CC-ELIGIBLE HOLDING

- **Shares**: 100 (exactly 1 contract worth)
- **Cost basis**: $9.64/share
- **Current price**: ~$12.54
- **Unrealized gain**: ~$X (+30%)
- **Status**: WHEEL-READY

**Covered Call Analysis for FRSH:**

FRSH is a SaaS company (CRM/ITSM) with ~$700M annual revenue, growing ~15% YoY. Market cap ~$3.7B. It is a mid-cap tech name with moderate volatility.

Recommended CC parameters for FRSH:
- **Delta**: 0.20-0.25 (consistent with our validated wheel engine config of 0.22)
- **DTE**: 30-45 days (our validated sweet spot)
- **Strike selection**: ~$14-$15 range for Oct/Nov expiration (OTM by 10-15%)
  - At 0.20 delta, 30 DTE: strike around $14-$14.50
  - Premium estimate: ~$0.20-$0.40 per share ($20-$40 per contract)

**Monthly income expectation**: $20-$40/month per contract = 2-4% monthly yield on current value
- Annualized: ~24-48% yield on premium alone (but this is optimistic -- assignment risk is real)
- With cost basis at $9.64 and selling $14 calls: called away = $14.00 - $9.64 + premium = ~$4.56/share profit = 47% total return

**Should you own FRSH for wheeling?**
- PROS: Above cost basis, 100 shares available, decent IV for SaaS stock
- CONS: Mid-cap SaaS with execution risk, not a dividend aristocrat, limited options liquidity (wider spreads on smaller names)
- VERDICT: Fine to sell 1 covered call NOW for income while holding. If called away at $14+, that is a great exit. If not called, you collect premium. Win-win given the +30% unrealized gain.

### ALL OTHER HOLDINGS: NOT CC-ELIGIBLE

| Ticker | Shares | Need for CC | Issue |
|--------|--------|-------------|-------|
| AVAV | 10 | 100 | Only 10% of needed shares, deeply underwater (-30%) |
| CRDO | 7 | 100 | Only 7 shares, would need 93 more (~$19.6K) |
| CLSK | 60 | 100 | Need 40 more shares (~$558) -- CLOSEST to eligible |
| NOK | 40 | 100 | Need 60 more, underwater (-24%) |
| MNTN | 80 | 100 | Need 20 more shares (~$220) -- second closest |
| AMKR | 16 | 100 | Need 84 more (~$4.7K) |

**Near-CC-eligible**: MNTN (80/100 shares, need ~$220 more) and CLSK (60/100, need ~$558 more) could become CC-eligible with small additional purchases. However, both are mediocre wheel candidates:
- MNTN: $11 stock, low options liquidity, tiny premium
- CLSK: Crypto mining stock, high volatility but directional risk

---

## 3. AVO-Evolved Strategies Relevant to Options Selling

From RUN_HISTORY.md, these are the options-specific evolved strategies:

### Options Execution AVO (v32, CONVERGED, LOCKBOX VALIDATED)

- **Score**: 531.27 (converged)
- **Lockbox result**: +378% return, Sharpe 2.48, 70 trades, 40% WR, PF 1.53
- **Method**: RSI/BB/MACD/dispersion multi-signal, sector-specific hold periods, trailing stop 10%/35%, TP 55%, SL -20%, 16% sizing, max 4 concurrent
- **Key**: This trades LONG options (calls/puts), NOT premium-selling. Directly applicable to the RH agentic account for directional option plays.
- **Paper engine status**: Running, started at $650, was at $701 (+7.8%) as of Aug 26

### IV Regime Options Backtest (VALIDATED)

- 572 trades across 3 signals x 3 IV regimes
- **Key finding**: Only trade options during CHEAP IV (sector-relative)
  - Cheap IV: Bond Yield signal +64% (Sharpe 0.72), Base MR +43% (0.39), IV-RV Gap +45% (0.47)
  - Expensive IV: dramatically worse across all signals
  - Theta cost: 12-14% in cheap IV vs 17-18% in expensive
- **Implication for wheel**: SELL premium when IV is expensive (high IV rank). This is the inverse -- our wheel strategies should ENTER (sell puts/calls) when IV is HIGH, not low.

### BPS (Bull Put Spread) Research

- The BPS Conservative engine is the best-performing options paper engine (+57.6%)
- Uses 30-delta, $15-wide spreads, 10-day DTE, 65% profit-take
- Defined-risk approach limits blowup scenarios
- Validated through extensive d30 research with regime-gated entries

---

## 4. Recommendations

### IMMEDIATE ACTION: Sell 1 Covered Call on FRSH

- Sell 1 FRSH call, strike ~$14, 30-45 DTE (Oct or Nov expiration)
- Expected premium: $20-$40
- Risk: called away at $14 (47% total return from $9.64 cost basis -- excellent outcome)
- Use the Robinhood MCP tool to check the actual option chain and get real premiums

### SHORT-TERM: Build Wheel Positions in Quality Stocks

The paper engines prove the wheel strategy works. To generate meaningful CC income in the real account, you need 100-share positions in quality names. Good wheel candidates criteria:

**What Makes a Good Wheel Stock:**
1. **Price range $15-$80**: Low enough for reasonable collateral, high enough for decent premium
2. **Weekly options available**: Liquidity matters -- need tight bid/ask spreads
3. **IV rank > 30**: Higher IV = more premium collected
4. **Fundamentals**: Profitable or near-profitable, positive revenue growth, not a meme stock
5. **Sector diversification**: Don't wheel 5 tech stocks
6. **You want to own it**: If assigned, you're holding shares -- pick companies you believe in

**Top wheel candidates to BUILD positions in (100 shares each):**

| Ticker | Approx Price | Capital for 100 | Sector | Why |
|--------|-------------|----------------|--------|-----|
| SOFI | ~$17 | $1,700 | Fintech | High IV, growing, weekly options |
| PLTR | ~$40 | $4,000 | Tech/AI | Very liquid options, high IV |
| AMD | ~$155 | $15,500 | Semiconductors | Liquid, volatile, strong fundamentals |
| F | ~$11 | $1,100 | Auto | Low cost, weekly options, dividend |
| HOOD | ~$28 | $2,800 | Fintech | High IV, growing |
| NIO | ~$5 | $500 | EV | Ultra-cheap, high IV (speculative) |

### MEDIUM-TERM: Deploy Validated Paper Strategies to Real

The BPS Conservative engine (+57.6% in 82 days) and Wheel Balanced engine (Sharpe 1.49 validated) are both ready for real deployment consideration. The BPS strategy in particular has enough trade history (620 trades) for statistical confidence.

### WHAT TO AVOID

- Do NOT sell covered calls on positions with fewer than 100 shares
- Do NOT wheel stocks you are underwater on unless you are committed to holding (AVAV at -30%, NOK at -24%)
- Do NOT sell covered calls below your cost basis -- that locks in losses if called away
- Do NOT wheel high-beta/meme stocks that can gap 20%+ overnight

---

## 5. Teleclaude Wheel Strategy Agent Status

The trading agent at `/home/jupiter/teleclaude-main/trading_agents/agents/wheel_strategy_agent.js` is **broken**. All recent runs show failures:
- Last successful morning run: March 30, 2026
- Last successful daily run: April 1, 2026
- All Friday runs since July 2026 fail (no stderr output = likely missing automation.py dependency)
- The agent tries to call `automation.py` but it appears to have an import or environment issue

This agent would need debugging before it can automate real wheel trades through Robinhood.

---

## 6. Summary

| Item | Status | Action |
|------|--------|--------|
| FRSH covered call | READY NOW | Sell 1 Oct/Nov $14 call for ~$20-40 income |
| Paper engines | 5 running, BPS Conservative best (+57.6%) | Continue monitoring |
| Real account wheel positions | Need 100-share blocks | Accumulate shares in quality names |
| Wheel strategy agent | Broken since April | Needs debugging |
| Options Execution AVO | Validated, paper running | Consider real deployment for directional plays |
