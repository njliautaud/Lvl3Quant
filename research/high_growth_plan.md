# High-Growth Strategy Design Plan
**Created: 2026-08-26 | Grounded in validated backtest results from QUANT_KNOWLEDGE_BASE.md and STRATEGY_CATALOG.md**

---

## Context: What We Actually Know

Before designing leverage, a critical truth from our own research:

- **The only fully validated, regime-agnostic growth edge is Strategy 1C: UPRO + VIX protection.** Sharpe 3.24, CAGR 67.6%, MaxDD -9.1%. Already live since 2026-07-16.
- **Sector bull call spreads (V10 Optimal)** are the best options structure: Sharpe 2.96-6.32 (V10 paper config), regime-gated with LGBM ranking. Running at $645 paper.
- **Sentiment Contrarian (AVO v22)** is the best single-stock mean reversion: Sharpe 2.73, lockbox validated. Running at $10K paper.
- **Vol Compression (AVO v17)** is the sector ETF swing strategy: Sharpe 1.28 OOS, tri-regime VIX-gated. Running at $10K paper.
- **Sector Equity Rotation (LGBM top-2)**: Sharpe 1.40, CAGR 24.1%, zero options pricing risk. Running at $645 paper.

**What does NOT work for leverage (already validated as failures):**
- ML timing of leveraged ETFs (best ML Sharpe 0.80 vs baseline 1.64)
- Adaptive regime scoring beyond simple VIX rules (v1 Sharpe 0.36, v2 Sharpe 0.54 — both worse than SPY)
- Adding income overlays (ICs, condors, butterflies) — all lose money at $645
- More risk management rules on top of VIX filter — redundant, adds nothing

---

## Strategy 1: UPRO + VIX Protection — The Core Growth Engine

### Current Implementation
**File:** `research/adaptive_leveraged_growth_v1.py`, `research/adaptive_leveraged_growth_v2.py`
**Live since:** 2026-07-16 with 2.88 shares UPRO ($X)
**Current allocation rules:**
- VIX < 17: 100% UPRO
- VIX 17-25: 30% UPRO + 70% SHY
- VIX > 25: 100% SHY (cash)
- 4-signal protection overlay: credit, breadth, SPY SMA, VIX

### Validated Backtest Results (STRATEGY_CATALOG.md 1C)
| Config | CAGR | MaxDD | Sharpe | Sortino |
|--------|------|-------|--------|---------|
| UPRO + VIX thresholds only | 84.6% | -14.7% | 3.13 | 4.94 |
| UPRO + 4-signal protection | 67.6% | -9.1% | 3.24 | 4.82 |
| TQQQ + 4-signal protection | 87.3% | -16.8% | 2.73 | — |

### High-Growth Variant: UPRO + Concentrated Sizing in Clear Regimes
**The V1/V2 adaptive scripts are underperforming** (Sharpe 0.35-0.54) because they added overly complex regime scoring. Return to the **validated simple rules**.

**Proposed config (already validated in backtest, not yet fully deployed at scale):**
- VIX < 17: 100% UPRO (no splitting, no complexity)
- VIX 17-25: 30% UPRO
- VIX > 25: 100% cash (SHY or money market)
- Keep 4-signal protection overlay (validated add)
- No ML timing — simple rules beat all ML approaches (finding confirmed by 6 ML models)

**Expected outcome vs current partial deployment:**
- Scale from 2.88 shares ($X) to full account deployment as capital grows
- CAGR 67.6-84.6% depending on how aggressive the VIX threshold
- MaxDD contained to -9.1% to -14.7%

**Implementation steps:**
1. Current VIX daily allocator cron (3:30 PM ET) is already running — verify it's using the SIMPLE rule, not the v1/v2 complex scoring
2. As account grows past $1K, $2.5K, $5K — scale UPRO position proportionally
3. At $5K+: add TQQQ as a 20-30% satellite if higher CAGR is wanted (accept -16.8% MaxDD tradeoff)
4. No further ML experimentation on timing — this is closed research

---

## Strategy 2: Sector Bull Call Spreads (V10 Optimal) — Leveraged Options

### Current Implementation
**File:** `paper_engines/sector_combined_v10_optimal_paper.py`
**Capital:** $645 paper
**Mechanics:** LGBM-ranked sector ETFs (XLK, XLF, XLE, XLV, XLY, XLP, XLI, XLB, XLU, XLRE, XLC). Monthly rebalance. 8 positions (top-4 bull + bottom-4 bear). DTE=28, 4% OTM, 30% profit target. $200 max per trade.
**Paper backtest:** Sharpe 6.32 (5/5 gates), honest validated Sharpe 2.96 (per QUANT_KNOWLEDGE_BASE.md Tier 1)

### Why This IS Already a Leveraged Strategy
Bull call spreads at 4% OTM on sector ETFs give approximately 3-5x capital leverage vs buying the ETF outright:
- ETF moves 4% → spread can 3-5x in value
- Defined risk: max loss = premium paid
- This is the correct leverage vehicle at $645 — NOT leveraged ETFs, which decay

### High-Growth Variant A: Replace Sector ETFs with Leveraged ETFs (TQQQ calls, SOXL calls)
**Risk: HIGH — not validated, do not deploy without backtest**

Using TQQQ/UPRO/SOXL as the underlying for bull call spreads would amplify exposure further:
- TQQQ already 3x QQQ, so a call spread on TQQQ = ~9x underlying exposure
- **Problem:** IV on leveraged ETFs is extreme (50-80%). Spread premium is huge relative to defined risk. The structural VRP edge (buying spreads in high IV) may break down.
- **What needs to be tested:** Run the sector spread backtest substituting TQQQ for XLK and SOXL for XLF/XLK on semiconductor-heavy plays. Compare honest hold-to-expiry results.

### High-Growth Variant B: Increase Position Concentration (4 positions instead of 8)
**Less risky — partially validated by V10 vs V9.3 evolution**

V10 moved from 3+3 to 4+4 positions and Sharpe improved from 3.x to 6.32. Next logical step: test 2+2 concentration (top-2 bull, bottom-2 bear) with larger per-position sizing.
- Per-position sizing: $200 → $300 (3 positions instead of 8 at same $645 capital)
- **Risk:** Higher per-position concentration increases drawdown risk
- **Implementation:** Add a V11 config with `N_BULL=2, N_BEAR=2, MAX_POS_SIZE=300` to `sector_combined_v10_optimal_paper.py` and run as second paper engine

### High-Growth Variant C: VIX>25 Spike Trades (Validated, Scarce)
**From QUANT_KNOWLEDGE_BASE.md:** VIX Spike OTM Call Spreads (5% OTM, VIX>30) — Sharpe 1.29, 36x total over 17 years, WR 59.3%. Only 27 trades in 17 years — fires ~2x/year.
- Already captured by the agentic signal aggregator and V10 optimal config's VIX-gating
- Not a standalone high-frequency strategy, but worth 2x normal sizing when VIX>30 occurs
- The `vix_contango_sector_oversold_paper.py` captures the VIX contango version (Sharpe 1.67, WR 62%)

**Implementation steps for V10 High-Growth:**
1. Paper trade V10 Optimal for 60 days (started recently) — collect real pricing data
2. Add V11 concentrated variant (2+2, $300/trade) as parallel paper engine
3. At 60+ days of real data: audit V10 vs V11 on real fills
4. Scale V10 to real money at $X Robinhood account when paper results confirm backtest

---

## Strategy 3: Sentiment Contrarian (AVO v22) — Single-Stock Mean Reversion

### Current Implementation
**File:** `paper_engines/sentiment_contrarian_avo_paper.py`
**Capital:** $10,000 paper (not live yet)
**Mechanics:** Buy large-cap stocks beaten down relative to SPY when macro isn't deteriorating. Tier 1 (deep oversold) and Tier 2 (moderate). Sentiment boost from Reddit polarity. Max 3 concurrent positions at $2,000 each.
**Validated performance:** Sharpe 2.73 at MAX_CONCURRENT=3, AVO lockbox validated

### Current State
As of 2026-08-26, the paper engine is live (started today, 3 positions open). Capital: $10,000 paper. No closed trades yet.

### High-Growth Variant: Increase Trade Velocity + Options Overlay

**Option A: Stock positions → deep ITM calls for capital efficiency**
- Instead of $2,000 in stock, buy deep ITM call (70+ delta) for ~40-60% of that cost
- Same directional exposure, less capital tied up per position
- Allows more simultaneous positions (5-6 instead of 3)
- **Risk:** Time decay works against you; must exit within 5-day max hold window regardless
- **LOCKBOX WARNING:** MAX_CONCURRENT=3 was the validated config. Sharpe dropped sharply at 5 positions (-0.53). Do not increase positions without re-validating.

**Option B: Apply strategy to smaller growth names (not just large-caps)**
- Current universe: 50 large-cap names (AAPL, MSFT, NVDA, etc.)
- High-growth variant: Add SMID-cap momentum names (SHOP, DDOG, MELI, PLTR)
- Higher volatility = bigger mean-reversion bounces
- **Risk:** Higher volatility also = higher failure rate. Current oversold thresholds (-7% rel strength, -7% 5d ret) need recalibration for volatile names

**Option C: Stay the course — this is already 2.73 Sharpe with -0.002 MaxDD at 3 positions**
The lockbox validated that any increase in position count HURTS this strategy. The safest high-growth approach is simply scaling capital while keeping MAX_CONCURRENT=3.
- At $10K: 3 positions × $2,000 each = $6,000 deployed when fully active
- At $30K: 3 positions × $6,000 each = $18,000 deployed
- CAGR scales linearly with capital at same Sharpe

**Recommended approach:** Option C for now. Scale capital as it grows. Revisit Options A/B only after 90+ days of real paper data confirm the AVO parameters hold in live trading.

---

## Strategy 4: Vol Compression (AVO v17) — Sector ETF Swing Trading

### Current Implementation
**File:** `paper_engines/vol_compression_avo_paper.py`
**Capital:** $10,000 paper
**Mechanics:** Buy sector ETFs (XLK, XLF, XLV, XLE, etc.) when vol is compressing (short/long vol ratio < 0.75, Bollinger bandwidth < 20th percentile). Tri-regime: low/mid/high VIX. Max hold 12 days, 4.75% take profit, -3.5% hard stop.
**Validated performance:** Sharpe 1.28 OOS, AVO-evolved over 17 steps

### Current State
As of 2026-08-26, paper engine is live with 0 trades closed (started ~2 days ago).

### High-Growth Variant: Replace ETF Purchases with ATM Calls

Instead of buying XLK outright when vol compresses, buy a 28-DTE ATM call:
- Vol compression = realized vol is low relative to historical
- **If IV is also low** (IV rank < 30), ATM calls are cheap
- The breakout move that follows vol compression = call goes up 3-5x
- Current engine caps at 4.75% profit on the ETF; an ATM call would capture 3-5x that

**Key constraint:** Timing exit is critical. The strategy holds up to 12 days — option theta decay starts hurting after 5-7 days on a 28-DTE option if the move doesn't happen.

**Recommended option structure for vol compression:**
- When signal fires: buy 21-28 DTE ATM or 2% OTM call on the top-ranked sector ETF
- Size: 1 contract per signal (defined risk = premium paid)
- Exit: same triggers as equity version (4.75% underlying move = 2-3x option gain), or at day 7 regardless
- **The agentic signal aggregator (`agentic_signal_aggregator.py`) already generates call signals with greeks** — the vol compression signal can feed directly into the Robinhood options account

**Expected impact:**
- ETF version: 4.75% gain on $2,500 position = $119 per trade
- Options version: 4.75% ETF move on ATM 28-DTE call ≈ 200-300% option gain
- At $150-200 per contract: $300-600 per winning trade vs $119 for the equity version
- MaxDD risk: maximum loss = premium ($150-200 per contract), same as V10 spread risk model

**Implementation steps:**
1. Continue paper trading vol compression equity engine for 30 days to verify signal quality
2. Add options variant as parallel paper engine: when vol compression fires AND IV rank < 35, log the call option signal to `state/vol_compression_options_signals.json`
3. The agentic signal aggregator can consume this as a new signal source
4. After 30 days of signal logging: analyze which signals would have profited on the options side

---

## Strategy 5: VIX Contango + Sector Oversold — Regime-Gated Swing

### Current Implementation
**File:** `paper_engines/vix_contango_sector_oversold_paper.py`
**Capital:** $645 paper
**Mechanics:** Buy sector ETFs when VIX is in contango (VIX < VIX3M) AND RSI(14) < 30. Hold 5 days. Best sectors: XLU (Sharpe 3.16), XLV (2.84), XLF (2.62).
**Validated performance:** 6/6 adversarial gates, Sharpe 1.67, WR 62%, 776 trades over backtest

### High-Growth Variant: Same Signal → Single-Leg Calls

This is the most natural strategy to run through the Robinhood options account because:
1. VIX contango = options are cheaper relative to future realized vol (favorable for buying)
2. RSI < 30 = oversold = directional move likely
3. 5-day hold = 21-28 DTE calls have enough time value not to decay meaningfully
4. The agentic signal aggregator **already converts this to single-leg call signals** as of today's run (XLE scored 0.95 confidence, recommended call at $60 strike, 9/11 expiry)

**Current signal output (2026-08-26):**
- XLE: 95% confidence bull, 6 confirming sources (sector_etf_momentum, cta_trend, strong_momentum, rsi_bullish, v91_monthly, earnings_outperf_e), recommended $60 call at $1.52 estimated cost ($152/contract), IV rank 21% (cheap)
- This IS the high-growth implementation — it's already being generated

**What's needed:**
1. Review the output of `agentic_signal_aggregator.py` daily (already running)
2. When confidence >= 0.75 AND IV rank < 35 AND VIX in contango → execute the single-leg call on Robinhood
3. Stay within $100-150 per trade as specified in signal output
4. Use the exit guidance already embedded in each signal (30% TP, 25% SL, max 5 days)

---

## Signal Feed for Agentic Options Account

### Current Signal Sources (Wired Into Aggregator)
The `agentic_signal_aggregator.py` reads from 20+ validated paper engine state files and produces unified signals at `/home/jupiter/Lvl3Quant/state/agentic_signals.json`.

**As of 2026-08-26 run:**
- Market regime: neutral
- VIX: 15.21 (low, contango mode — favorable for buying options)
- VIX term structure: contango (VIX/VIX3M = 0.845)
- Active signals: XLE bull at 95% confidence (6 confirming sources)

**Key signal state files feeding the aggregator:**
- `state/sector_etf_momentum_paper_state.json` — LGBM-ranked sector momentum
- `state/sector_combined_v10_optimal_paper_state.json` — V10 LGBM + spread model
- `paper_engines/state/sentiment_contrarian_avo_state.json` — AVO mean-reversion
- `paper_engines/state/vol_compression_avo_state.json` — AVO vol compression
- `state/vix_contango_sector_oversold_paper_state.json` — VIX contango oversold
- `state/equity_rotation_paper_state.json` — Sector equity rotation
- `state/rsi_divergence_paper_state.json`, `state/bond_yield_paper_state.json` (7 more validated engines)
- `state/earnings_outperformance_e_paper_state.json` — Earnings outperformance (Sharpe 2.91, WR 62%)

### How Signals Flow to Options Trades
```
Paper engines (20+) → state JSON files
     ↓
agentic_signal_aggregator.py (runs daily at 4:05 PM ET)
     ↓
state/agentic_signals.json
  - ticker, direction, confidence_score
  - confirming_sources (n_confirming)
  - recommended_option (call/put)
  - recommended_strike, recommended_expiry
  - estimated_cost, affordable (within $200 budget)
  - exit_guidance (TP%, SL%, max_hold_days)
  - iv_rank, iv_classification
  - greeks_analysis (greeks_score, delta, theta)
     ↓
Manual review → Robinhood options order placement
```

### What's Missing (Gap to Full Automation)
1. **Auto-execution**: The aggregator generates signals but does NOT place orders. The `v10_agentic_signal_generator.py` has Robinhood MCP integration documented but `ENABLE_RH_TOOLS = False`.
2. **Earnings filter**: Currently using yfinance; `RH_USE_EARNINGS_CALENDAR = False`. Should switch to `get_earnings_calendar` MCP tool to avoid entering before earnings.
3. **Liquidity gate**: `RH_OPTIONS_LIMITS` defined but not enforced (need OI > 50, spread < $0.50).

---

## Portfolio Allocation Plan: Growth Sleeve

### Current State (2026-08-26)
| Strategy | Capital | Status | Validated Sharpe |
|----------|---------|--------|-----------------|
| UPRO + VIX protection | ~$X live | Live | 3.24 |
| V10 sector spreads | $645 paper | Paper | 2.96 |
| Sentiment Contrarian | $10K paper | Paper (day 1) | 2.73 |
| Vol Compression | $10K paper | Paper (day 3) | 1.28 |
| Sector Equity Rotation | $645 paper | Paper | 1.40 |
| Agentic options signals | $X Robinhood | Signals ready, not auto-trading | — |

### Target Allocation as Capital Scales

**Phase 1 ($645-$2,500 total):** Pure UPRO + VIX protection. Proven Sharpe 3.24, already live. Let it compound. No options needed at this size — the leverage is already inside UPRO.

**Phase 2 ($2,500-$10,000):**
- 60% UPRO + VIX protection (core)
- 40% Sector bull call spreads (V10 Optimal at $645 per cycle, roll proceeds into next)
- Begin executing agentic call signals at $100-150 per trade (1-2 per month based on current signal frequency)

**Phase 3 ($10,000-$50,000):**
- 40% UPRO + VIX protection
- 30% V10 sector bull call spreads (scale to $2,000-$3,000 per month deployed)
- 20% Sentiment Contrarian (stock positions, 3-position limit)
- 10% Vol Compression options overlay (calls when IV cheap + vol compressing)

**Phase 4 ($50,000+):**
- 30% UPRO + VIX protection
- 25% V10 sector bull spreads
- 20% Sentiment Contrarian (stock positions)
- 15% CTA trend following diversifier (GLD, TLT, UUP — Sharpe 2.0+ per asset)
- 10% Vol Compression options
- Add covered calls on UPRO position (validated add, +1.4-3.4% CAGR)

### Expected Portfolio CAGR by Phase
| Phase | Capital | Primary Driver | Est. CAGR | Est. MaxDD |
|-------|---------|---------------|-----------|-----------|
| 1 | $645-$2.5K | UPRO+protection | 67-85% | -9 to -15% |
| 2 | $2.5K-$10K | UPRO + spreads | 50-70% | -10 to -18% |
| 3 | $10K-$50K | Diversified | 35-50% | -12 to -20% |
| 4 | $50K+ | Full portfolio | 25-35% | -13% (1D result) |

*All CAGR estimates from validated backtests — real results will differ. Phase 1 is the most reliable given live validation since July 2026.*

---

## Implementation Priority Order

1. **Highest priority — already live, keep running:** UPRO + VIX allocator (cron 3:30 PM ET). Verify it's using simple VIX thresholds, not the complex v1/v2 adaptive scoring that underperformed.

2. **High priority — start executing agentic signals:** When VIX contango + high confidence (>75%) sector call signals fire, execute at $100-150 per trade on Robinhood. XLE signal fired today at 95% confidence. This is the direct output of 6 months of signal research.

3. **Medium priority — V10 paper validation:** Continue paper trading for 30 more days. After 60 total days of paper data: if results track backtest within 30%, start live execution at $X.

4. **Medium priority — Vol compression options log:** Add signal logging to vol compression paper engine so we can see what options trades would have looked like alongside the equity version.

5. **Lower priority — V11 concentrated variant:** After V10 paper completes, test 2+2 concentration (fewer positions, larger size). This is exploratory and should not displace any running engine.

6. **Do not:** Revisit ML timing for UPRO (validated failure x6 models). Do not add IC overlays, butterfly spreads, or calendar spreads (all validated failures at $645). Do not increase Sentiment Contrarian to >3 concurrent positions (Sharpe collapses per lockbox).

---

## Key Risk Constraints (from validated research)

- **VIX circuit breaker:** VIX > 25 → no new options trades, exit to UPRO/cash allocation
- **Per-trade max:** $200 for spreads, $150 for single-leg calls, $2,000 for stock positions
- **No more than 3 concurrent positions** in Sentiment Contrarian (hard lockbox finding)
- **Earnings filter:** Block all options entries within 3 days of earnings (agentic aggregator already implements this)
- **IV rank gate:** Only buy single-leg calls when IV rank < 40 (cheap premium, favorable entry)
- **Hold time discipline:** Single-leg calls max 5 days, no exceptions (time decay accelerates on day 3+)
