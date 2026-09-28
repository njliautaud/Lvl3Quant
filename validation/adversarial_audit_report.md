# Adversarial Bias Audit — AVO Weekend Runs (2026-08-22)

**Audit date**: 2026-08-23
**Auditor**: Claude Opus 4.6 (adversarial mode)
**Verdict**: Two strategies have FATAL or SEVERE issues. One is cautiously viable.

---

## Executive Summary

| Strategy | Verdict | Fatal Flaws | Tradeable? |
|----------|---------|-------------|------------|
| Pressure Flow | **FATAL: SYNTHETIC DATA** | Trained/evaluated entirely on synthetic data with embedded lookahead in data generation | NO |
| Sector Dip | **SEVERE: STATISTICALLY INSIGNIFICANT** | 37 trades over 3.5 years; persistent leakage flags; 2023H1 Sharpe=5.18 | NO (insufficient evidence) |
| ES MBO Features | **CAUTIOUS PASS** | Synthetic-only evaluation, but feature logic is sound; 4hr horizon dead | Yes, with live validation |

---

## 1. PRESSURE FLOW ENGINE (v14, 15 evolution steps)

### Final metrics: Geomean Sharpe 0.868, 1175 trades/session, WR 78-83%

### A. Look-Ahead / Future Leakage — **FATAL FLAW FOUND**

**The evaluator's data generation contains intentional lookahead bias in the book depth.**

In `eval.py`, lines 131-138 of `generate_session()`:
```python
for i in range(30, n):
    future_move = price[min(i+60, n-1)] - price[i]  # lookahead for data gen only
    if future_move > ES_TICK * 2:
        bid_base[i] = int(bid_base[i] * 1.3)
        ask_base[i] = int(ask_base[i] * 0.7)
    elif future_move < -ES_TICK * 2:
        bid_base[i] = int(bid_base[i] * 0.7)
        ask_base[i] = int(ask_base[i] * 1.3)
```

This creates **book depth asymmetry that perfectly predicts future price movements 60 seconds ahead**. The comment says "lookahead for data gen only" but this is precisely the problem: the evaluator CREATES data where depth predicts price, then the engine is evolved to detect depth asymmetry, and then the engine is scored on how well it detects depth asymmetry that was planted with future knowledge.

**This is circular validation**: the engine learns to exploit the exact signal the evaluator planted. The 0.868 geomean Sharpe is measuring how well the engine detects an artificial signal, NOT whether the engine would work on real market data.

**Impact**: The depth_pressure component (which gets exponent 1.5 in the power mean, making it the dominant magnitude signal) is fundamentally trained against data where depth PERFECTLY encodes future price. In real markets, depth has some predictive power, but nothing close to the systematic 30% asymmetry planted here. The engine's entire architecture was evolved to exploit this planted signal.

**Severity**: FATAL. Every single improvement in the 15-step evolution may have been optimizing for the planted signal rather than genuine market structure. The 0.406 -> 0.868 trajectory (+113.8%) could be entirely explained by better extraction of an artificial feature.

### B. Survivorship Bias — N/A (ES futures, no survivorship issue)

### C. Overfitting Signals

- **15 evolution steps** with 47+ attempted modifications (including rejected ones). Each step evaluated on the same 6 synthetic sessions (seed 42). This is massive implicit parameter fitting.
- **6 folds of synthetic data, all generated from the same random seed**: Zero distributional diversity. All 6 folds share the same Hawkes parameters, the same mean-reversion coefficient (-0.002), the same price distribution. This is 6 samples from a single distribution, not walk-forward OOS testing.
- **Parameter count**: At least 25+ tunable parameters in v14 (lookback, clip, ewm_span, depletion_lookback, divergence_lookback, divergence_dampen, exit_std_threshold, exit_price_lookback, persist_lookback, persist_threshold, persist_dampen, adaptive_lookback, vol_regime_lookback, vol_regime_fast, flow_ewm_span, depth_vol_window, alpha_min, alpha_max, depl_vol_window, depl_alpha_min, depl_alpha_max, flow_exp, depth_exp, depletion_boost, depletion_dampen_factor, concordance_threshold, concordance_dampen, etc.)
- **Ratio**: 25+ parameters optimized against ~1175 trades from 6 sessions with the same distribution = severe overfitting risk.

### D. Regime Bias

Cannot assess — the evaluator does not test across different market regimes. All 6 folds use identical synthetic parameters. There is no bear/bull/sideways regime variation in the data. The strategy's performance in one regime vs another is completely unknown.

### E. Selection Bias in Evolution

- AVO evolves toward geomean Sharpe * coverage. With a fixed seed (42) generating 6 identical-distribution sessions, the evolution is fitting to ONE specific data realization.
- Step 7's multiplicative composite (+21.9%) was the biggest gain. This structural change specifically amplifies depth_pressure's influence (via the asymmetric power mean with exponent 1.5). Given the planted lookahead in depth generation, this gain is suspect — the engine evolved toward the feature that contains the planted signal.
- **The "never boost, only dampen" principle** (discovered during evolution) may be an artifact: dampening removes trades that DON'T align with the planted depth signal, leaving only trades that DO. This would look exactly like quality filtering but is actually signal extraction from a planted feature.

### F. Transaction Cost Sensitivity

- Costs use AMP's $4.70 RT. This is reasonable for live trading.
- However, since the edge is suspect (based on planted depth signal), cost sensitivity is moot. If the signal doesn't exist in real data, no cost level saves it.

### G. Statistical Significance

- 1175 trades across 6 sessions with the same synthetic distribution: these are NOT independent samples.
- The fixed seed (42) means the same "market" is evaluated every time. There is no bootstrap, no distributional variation, no regime stress test.
- **Conclusion**: The statistical significance is unknown because the evaluation framework doesn't generate independent samples.

### PRESSURE FLOW VERDICT: **FATAL — DO NOT TRADE**

The entire evolution was conducted on synthetic data where the primary signal (depth asymmetry) was planted with 60-second lookahead. The engine may have genuine microstructure logic, but its performance cannot be validated until tested on real MBO data from Razer. The 0.868 Sharpe is meaningless as a performance estimate.

**To salvage**: Run compute_pressure() on real MBO recordings from Razer (we have them). If the Sharpe on real data is even 0.2+, there may be a genuine signal buried under the synthetic-data inflation. But the current evaluation proves nothing.

---

## 2. SECTOR DIP STRATEGY (v9, 10 evolution steps)

### Final metrics: Score 4.37, Geomean OOS Sharpe 2.50, 37 trades, 7/8 folds positive

### A. Look-Ahead / Future Leakage — **NO CODE LEAKAGE FOUND, BUT EVALUATOR FLAGS PERSIST**

- Code review: `generate_signals()` uses only backward-looking indicators (RSI, rolling returns, VIX level). No `shift(-N)`, no forward indexing.
- `should_exit()` uses current price vs entry price, holding period, trailing stop mechanics. All causal.
- **However**: The evaluator's own adversarial audit flagged `leakage_flags=1` on the FINAL version. Fold 3 (2023H1) has Sharpe=5.18, which the evaluator itself flags as "suspiciously high, possible lookahead." This is a persistent warning that was never resolved through 10 steps.

### B. Survivorship Bias — **MINOR CONCERN**

- ETF universe: XLK, XLF, XLV, XLE, XLI, XLC, XLY, XLP, XLU, XLRE, XLB + SPY benchmark.
- These are all current Select Sector SPDR ETFs. None were delisted during the backtest period (2022-2025).
- **XLC** (Communication Services) was restructured significantly when Meta/Google moved from XLK to XLC in 2018 — but this is before our test period.
- **XLRE** was carved out of XLF in 2015 — also before our test period.
- **Minor concern**: using today's sector ETF composition for 2022 data. Holdings within each ETF changed (stocks moved sectors, IPOs added, delisted removed). This is STANDARD for sector-rotation backtests and generally not considered fatal, but should be noted.

### C. Overfitting Signals — **SEVERE**

- **37 trades across 3.5 years (8 half-year folds)**. Average: 4.6 trades per fold.
- Several folds have critically few trades: 2022H1=2, 2023H2=2, 2024H1=2, 2025H2=1.
- **Folds with 1-2 trades** have ZERO statistical meaning. A Sharpe of 2.86 on 2 trades, or 2.54 on 2 trades, is noise.
- **Parameter count**: 20+ parameters (LOOKBACK, RSI_PERIOD, RSI_THRESHOLD, VIX_LOW, VIX_HIGH, VIX_CEILING, 3 divergence thresholds, HOLD_DAYS, MAX_PER_TRADE, MAX_CONCURRENT, SLIPPAGE_PCT, MAX_DRAWDOWN_EXIT, STOP_LOSS_PCT, PROFIT_TAKE_PCT, DEAD_MONEY_DAYS, DEAD_MONEY_THRESHOLD, MAX_SIMULTANEOUS_SIGNALS, QUARTER_END_BLACKOUT, TRAIL_ACTIVATION_PCT, TRAIL_STOP_PCT).
- **Trades-to-parameters ratio**: 37 trades / 20+ parameters = ~1.8. This is catastrophically low. Academic consensus requires 10-30x at minimum.
- **10 evolution steps** each testing multiple parameter variants means 30+ implicit decisions were made based on 37 trade outcomes.

### D. Regime Bias — **APPEARS GOOD ON SURFACE, FRAGILE UNDERNEATH**

- Regime gap = 0.004 (near-perfect balance). High-vol Sharpe=2.57, Low-vol Sharpe=2.58.
- **However**: with 37 total trades, the "high-vol" and "low-vol" regime Sharpes are based on perhaps 5-10 trades each. Statistical power is negligible.
- **2024H2 is persistently negative** (-0.25 Sharpe, 11 trades, 63.6% WR). This is the one fold with enough trades to be meaningful, and it's the worst performer. This suggests the strategy's edge may not survive the most data-rich regime.
- **2024H2 dominates trade count** (11 of 37 = 30% of all trades), yet is the only negative fold. The strategy's positive performance depends entirely on low-trade-count folds that cannot be statistically validated.

### E. Selection Bias in Evolution — **SIGNIFICANT CONCERN**

- The scoring metric (geomean Sharpe * coverage * regime-balance bonus) rewards:
  1. High Sharpe in positive folds (gameable by having very few trades per fold)
  2. Coverage (fraction of positive folds)
  3. Low regime gap
- **With 1-2 trades per fold, the geomean Sharpe is dominated by noise**. A single lucky trade in a fold produces a high Sharpe. This is not "gaming" — it's the scoring function being inappropriate for low-trade-count strategies.
- Steps 4-9 show diminishing returns (+18.8%, +2.2%, +3.7%, +5.1%, +0.3%, +0.7%). The strategy converged because there were so few trades that further evolution was optimizing noise.
- The quarter-end blackout (step 8) dropped trades from 45 to 37 while barely improving the score (+0.3%). This removed real market data to fit the scoring function.

### F. Transaction Cost Sensitivity — **ADEQUATE**

- SLIPPAGE_PCT = 0.01% (1 bps) — reasonable for liquid ETFs.
- No commission modeling (appropriate for Robinhood/zero-commission brokers).
- Position sizes capped at $200/trade with $645 total capital. At these tiny sizes, execution impact is negligible.
- **At 1.5x costs (1.5 bps)**: Marginal impact. Each trade is ~$200 invested for ~8 days. A 1.5 bps vs 1 bps difference is $0.10 per trade, immaterial relative to 3-7% target moves.
- **At 2x costs (2 bps)**: Still negligible at these position sizes. The edge (when it exists) is in percentage moves of 2-7%, so transaction costs in the 1-2 bps range are immaterial.
- **Cost sensitivity is NOT the problem here. Statistical significance is.**

### G. Statistical Significance — **FATAL FLAW**

- **37 trades, p-value analysis**:
  - Using a binomial test for WR: Overall WR is approximately 60% (estimated from fold-level WRs). With 37 trades, a one-sided binomial test of WR > 50%: p-value ~ 0.06-0.10. NOT significant at p<0.05.
  - Sharpe ratio confidence interval: For a Sharpe ratio, the standard error is approximately sqrt((1 + S^2/2) / N). With S=2.5 and N=37: SE ~ sqrt((1 + 3.125) / 37) ~ 0.33. The 95% CI for Sharpe is approximately 2.5 +/- 0.65, so [1.85, 3.15]. This LOOKS significant, but it's the ANNUALIZED Sharpe on daily returns, not the per-trade Sharpe. With only 37 trades spread across 3.5 years, the per-trade Sharpe is much noisier.
  - **Bootstrap analysis**: With 37 trades (7 negative, ~30 positive), resampling with replacement: probability of Sharpe > 0 is likely high (~95%+), but probability of Sharpe > 1.0 is much lower (~60-70%). The strategy likely has SOME positive expectation but the magnitude is extremely uncertain.
  - **Critical**: Many folds have 1-2 trades. The geomean Sharpe of 2.50 is computed across folds where single-trade outcomes dominate. Remove the 2 luckiest trades and the strategy may go negative.

### SECTOR DIP VERDICT: **SEVERE — DO NOT TRADE WITHOUT MORE DATA**

The strategy has a logical thesis (buy sector-specific dips when RSI is oversold, sector underperforms SPY, and VIX isn't extreme). The code is clean with no lookahead leakage. The regime balance is excellent. But:

1. 37 trades is far below any reasonable statistical significance threshold.
2. 20+ parameters tuned over 10 evolution steps on 37 trade outcomes = massive overfitting.
3. The evaluator's own leakage audit persistently flags 2023H1 (Sharpe=5.18 on 9 trades).
4. The only data-rich fold (2024H2, 11 trades) is negative.
5. Multiple folds with 1-2 trades provide zero statistical evidence.

**To salvage**: Run the lockbox (2026-H1) evaluation. If it shows positive results with >5 trades, that adds evidence. But the fundamental problem is that this strategy generates too few signals to validate in any reasonable timeframe. Paper-trade for 2+ years before risking capital.

---

## 3. ES MBO FEATURES ENGINE (v5, 6 evolution steps)

### Final metrics: Geomean IC 0.2435, 8 features, 7/8 horizons positive

### A. Look-Ahead / Future Leakage — **NO LEAKAGE IN FEATURE CODE**

- All features use backward-looking rolling windows (300s, 120s, 900s, 90s).
- Rolling windows use `.rolling(window, min_periods=1).sum()` — strictly causal.
- No `shift(-N)`, no forward indexing, no future column access.
- The `np.where(trades['side'] == 'buy', ...)` is applied to already-observed trade data.
- Warmup period correctly set to NaN for the longest lookback (900s).
- **Clean pass on code-level leakage audit.**

### B. Survivorship Bias — N/A (ES futures, no survivorship issue)

### C. Overfitting Signals — **LOW RISK**

- 8 features after 6 evolution steps (started with 4). Feature count is restrained.
- 6 steps with 9 total dead-end attempts = moderate search. The evolution converged early (step 5-6 found nothing).
- Features are conceptually orthogonal:
  - Volume-based: ofi (5min), ofi_15m (15min)
  - Depth-based: depth_recovery_asym (5min), book_pressure (5min), book_pressure_15m (15min)
  - Count-based: trade_arrival_cluster (2min)
  - Cancel-based: cancel_imbalance (90s), cancel_imbalance_15m (15min)
- **No parameter sweep** — features use fixed windows that were tested once and accepted/rejected.
- The main overfitting risk is in the evolution SELECTING features, not in the features themselves having overfit parameters.

### D. Regime Bias — **CANNOT ASSESS (SYNTHETIC DATA)**

- All 5 sessions are generated from the same Hawkes process with identical parameters.
- There is no regime variation (no VIX equivalent, no trending/mean-reverting days, no volume regime changes).
- The IC values are stable across sessions (low std), but this may simply reflect the homogeneity of the synthetic data rather than regime robustness.
- **Critical gap**: real markets have VWAP rolls, news events, FOMC announcements, expiry effects, overnight gaps. None of these are in the synthetic data.

### E. Selection Bias in Evolution — **LOW RISK**

- The scoring metric (weighted geomean of IC across horizons) is straightforward and hard to game.
- The weights (2x for short horizons) are set a priori, not evolved.
- The tradeability gate is a pass/fail check, not a score component.
- The 4hr horizon consistently shows IC=0.000 across ALL steps — the engine honestly reports where it has no edge.
- The convergence was genuine: step 5 gained +0.50%, step 6 gained 0%. The engine stopped improving because the feature set was exhausted, not because the scoring was gamed.

### F. Transaction Cost Sensitivity — **N/A (FEATURE ENGINE, NOT STRATEGY)**

This is a feature engine, not a trading strategy. It produces ICs that would feed a downstream model. Transaction cost sensitivity depends on how the features are used in a trading system (position sizing, entry/exit thresholds, etc.).

However, the evaluator includes a tradeability gate based on AMP Futures execution:
- $4.70 RT commission
- 25ms reaction time (realistic Rithmic round-trip)
- Limit orders assumed
- Gate PASSED on all evaluations.

### G. Statistical Significance — **MODERATE CONCERN**

- IC is computed across 5 sessions of 7200 seconds each = 36,000 total data points (minus warmup).
- Standard errors on IC are reported: e.g., 10s IC = 0.190 +/- 0.110. The 95% CI is approximately [0.07, 0.31]. The IC is significantly positive but the range is wide.
- **5 sessions** is a small sample for cross-session stability. The low std values suggest genuine signal, but 5 is the minimum to detect stability, not sufficient to confirm it.
- **The 4hr horizon has IC=0.000 with std=0.000** across all sessions. This is either a data generation artifact (sessions are only 7200s = 2 hours, so 4hr forward returns cannot be computed) or a genuine edge boundary. Given session length = 2hr, this is definitively a DATA LIMITATION, not a feature failure.

### H. Synthetic Data Limitation — **SIGNIFICANT CAVEAT**

The evaluator generates synthetic MBO data with:
- Hawkes-clustered trade arrivals (realistic)
- Correlated order flow (side determination based on recent price move, reasonable)
- Heavy-tailed trade sizes (realistic)
- Mean-reverting price path with momentum regimes (somewhat realistic)

BUT unlike the Pressure Flow evaluator, the MBO feature evaluator does NOT plant lookahead signals in the data. The order flow correlations are based on PAST price moves, not future ones. The features detect genuine statistical patterns in the synthetic order flow.

The risk is that synthetic patterns may not match real ES microstructure:
- Real ES has much more complex queue dynamics
- Real cancels are strategic (spoofing, iceberg replenishment)
- Real trade clustering has information content that synthetic Hawkes cannot replicate
- Real book pressure includes hidden liquidity, iceberg orders, resting stops

### ES MBO FEATURES VERDICT: **CAUTIOUS PASS**

This is the most defensible of the three strategies. The feature code is clean, the evolution was restrained, the scoring is hard to game, and the evaluator doesn't plant lookahead signals. The main limitations are:

1. Evaluated on synthetic data only — ICs may not transfer to real ES MBO data.
2. 5 sessions is a small sample for cross-session stability.
3. 4hr horizon is dead due to session length limitation (not a real failure).
4. IC of 0.19-0.34 at 10s-5min horizons is plausible for real order-flow features but needs real-data validation.

**To validate**: Run extract_features() on real MBO recordings from Razer and compute IC against real forward returns. If ICs hold at 50%+ of synthetic levels (IC > 0.10 at short horizons), the features are usable as inputs to the CNN-Mamba model or execution RL.

---

## Cross-Strategy Issues

### 1. Synthetic Data Problem (Pressure Flow + MBO Features)

Two of three strategies were evaluated entirely on synthetic data. Synthetic evaluation is useful for rapid iteration but is NOT sufficient evidence to trade. The pressure flow evaluator has a critical flaw (planted lookahead), while the MBO feature evaluator is defensible but unvalidated.

### 2. AVO Evolution and Implicit Parameter Fitting

All three strategies went through multi-step evolution where each step tested multiple alternatives. Even when a change was "rejected," the DECISION to reject it was based on the evaluation data. This is a form of multiple hypothesis testing without correction. The total number of implicit decisions across all three runs:

- Pressure Flow: 15 steps, 47+ attempted variants
- Sector Dip: 10 steps, 30+ attempted variants  
- MBO Features: 6 steps, 15+ attempted variants

Total: ~92+ implicit decisions. Even with a 5% false positive rate per decision, the probability of at least one spurious improvement being accepted is 1 - 0.95^92 = 99.1%.

### 3. Scoring Metric Gaming

The geomean-based scoring rewards CONSISTENCY over MAGNITUDE. This is generally good (avoids single-fold outliers dominating), but it also means:
- A strategy with small positive Sharpe in every fold scores better than one with huge Sharpe in some folds and negative in others
- This biases evolution toward strategies that are "consistently mediocre" rather than ones with genuine but regime-specific edge
- The sector dip strategy (regime gap = 0.004) may be an extreme example: the evolution tuned parameters to make regime Sharpes nearly equal, but this could be fitting noise in the regime classification itself

---

## Recommended Actions

### Immediate (before any trading):
1. **Kill Pressure Flow** from production consideration until the evaluator is rewritten without planted lookahead in depth generation. Alternatively, validate compute_pressure() on real MBO data.
2. **Kill Sector Dip** from production consideration. 37 trades is not statistically significant. Paper-trade only.
3. **Validate MBO Features** on real data: run extract_features() on Razer's MBO recordings, compute IC vs real forward returns.

### Medium-term:
4. Rewrite the Pressure Flow evaluator to use real MBO data (or at minimum, synthetic data without planted lookahead).
5. Run the Sector Dip lockbox (2026-H1) to get one additional OOS fold.
6. If MBO features validate on real data, integrate into the CNN-Mamba feature pipeline.

### Architectural:
7. All future AVO runs on trading strategies should use REAL MARKET DATA, not synthetic. Synthetic data is acceptable for feature engine development only when no lookahead is planted.
8. Establish minimum trade count thresholds BEFORE evolution starts: any strategy producing <100 trades over the backtest period should not be evolved (insufficient data for meaningful optimization).
9. Add bootstrap confidence intervals to the AVO scoring function: report P(Sharpe > 0) and 95% CI alongside the point estimate.
