# Sector Rotation Options Strategy — Research Knowledge Base
*Last updated: 2026-07-27 10:40 ET. Covers all completed research through SESSION_STATE entries ~800–1121.*

---

## 1. Strategy Overview

The sector rotation options strategy uses a LightGBM model trained on multi-factor sector ETF features (momentum, quality, volatility, cross-asset) to rank all 11 SPDR sector ETFs biweekly, then trades bull call spreads on the top 3 sectors when VIX≥20 and bear put spreads on the bottom 3 when VIX<20. Option spreads are 3% wide, 30-day DTE, sized at ~$200/spread from a $645 starting account. The definitive adversarial-validated performance (hold-to-expiry, no pricing shortcuts) is **Sharpe 3.04, CAGR 63%, MDD -9.6%, WR 57%, 422 trades** over 2010–2026. A large fraction of this edge is structural — any sector with a VIX filter and bull call spreads outperforms — with LGBM adding approximately 25% incremental Sharpe and 5x better drawdown control versus random sector selection.

---

## 2. Production Config

- **Universe**: 11 SPDR sector ETFs (XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLV, XLY, XLU)
- **Model**: LightGBM ranking, 20 features (quality+momentum set, "F_QualMom_Simple" variant)
- **Rebalance**: Biweekly
- **Entry filter**: VIX≥20 → bull call spreads on top-3 ranked; VIX<20 → bear put spreads on bottom-3 ranked
- **Spread width**: 3% OTM
- **DTE**: 30 days (20-day early exit confirmed best; trailing stop AVOIDED — hurts WR)
- **Sizing**: Fixed $200/spread at $645; switch to tiered/linear scaling once equity exceeds $2K
- **Confluence gate**: Min 2/6 multi-signal confirmation required before entry (per HC #750)
- **Commission assumption**: $2.60/spread (validated against AMP/Robinhood quotes)
- **Bid-ask haircut**: 15% (**KNOWN UNDERESTIMATE** — KB #282 found ATR-based IV underprices by 73% median. Fix pending: switch to market IV or apply multivariate correction)
- **Paper engines (A/B test):**
  - V7 (2% OTM, DTE=21): PM2 104, backtest Sharpe 2.81
  - V8 (2% OTM, DTE=14): PM2 106, backtest Sharpe 2.92
  - V9 (3% OTM, DTE=21): PM2 108/109, backtest Sharpe 3.01
  - V9.1 (3% OTM, DTE=28): PM2 111, backtest Sharpe 3.77 ← **current best**
  - V10 (momentum+earnings signal avg): PM2 112, backtest Calmar 6.61
  - Earnings Standalone (14 features, DTE=21): PM2 110, backtest Sharpe 2.85
  - All cron at 4:30 PM ET weekdays, $645 initial capital
- **Honest definitive Sharpe (hold-to-expiry, strictest pricing)**: 3.04
- **Latest findings (Session 12)**:
  - DTE=28 beats DTE=14 by 2.2× real-priced Sharpe (KB #245)
  - GRU equity ranking edge doesn't transfer to options — LGBM stays (KB #248)
  - Monthly rebalance beats weekly by 16% Sharpe (KB #252)
  - Jade lizard income: 78% WR, 25.7% CAGR but -61% MDD (KB #251)
  - Simple Rules score: 14-0 (every "smart" approach loses to simple rules)

---

## 3. Key Findings Table

| Experiment | Sharpe (honest) | CAGR | MDD | Key Insight | Actionable? |
|---|---|---|---|---|---|
| Sector bull spreads (definitive, hold-to-expiry) | **3.04** | 63% | -9.6% | Correct hold; prev. 4.73 was early-exit inflated | YES — production config |
| Bull+Bear Combined v1 (always-trading) | 3.10 | 32.7% | -4.1% | Fills VIX<20 gap with bear puts; 73% more equity vs bull-only | YES — growth path |
| Multi-Factor Sector v1 (F_QualMom_Simple) | 4.20 | 70.2% | -1.1% | Quality+momentum beats momentum-only by +0.30 Sharpe | YES — production features |
| Integrated Sector v2 (VIX filter+confluence) | 3.99–4.20 | 68–70% | -1.0% | VIX filter + confluence cuts MDD from -4.1% to -1.0% | YES — gates recommended |
| Multi-Asset Broad Universe v1 | 4.70 → **0.82** (honest) | 77% | -2.6% | Sharpe inflation bug (4.17x); still passes gates but materially weaker than sector-only | NO — sectors-only is the true best |
| Sensitivity analysis v2 (28 perturbations) | 0.97–2.97 | 55–174% | -0.4% to -4.6% | 100% pass rate; VIX threshold most sensitive, spread width least | YES — config is robust |
| PEAD call spreads (gap>5%, 45 DTE) | 1.38 | 55.9% | -29.1% | Good Sharpe but MDD is the weakness; complement to sector | YES — as overlay only |
| Sector bear puts alone | 2.10 | 37.7% | -18.5% | Bear side adds trades; MDD 13x worse than bull side | PARTIAL — only inside combined |
| Sector bull-only | 4.73 → 3.04 (honest hold) | 68% | -1.4% | Highest Sharpe per trade; inactive 64% of time | YES — for max Sharpe at cost of growth |
| Multi-asset + bonds | 4.56 → 2.77 (honest) | — | -0.5% | Lowest drawdown ever; honest Sharpe still solid | MAYBE — if drawdown reduction priority |
| Sector IC income | dead (0/4 gates) | negative | — | Sector ETFs too volatile for IC premium selling at $645 | NO |
| PMCC momentum | Sharpe -1.3 to -1.8 | -14% | -93% | LEAPS cost too much for small accounts | NO |
| Weekly DTE sector spreads | 0.44 | — | -29% to -65% | More noise; R1 gap 0.70-0.80. Monthly strictly superior | NO |
| Butterfly strategies | 0/4 gates | — | ~-100% | Momentum signal can't predict precise price for butterfly | NO |
| LGBM vs random (honest comparison) | 4.59 vs 3.50 | — | -3.1% vs -16.7% | ML's main value is risk management (5x safer MDD), not return | YES — keep LGBM |
| Lead-lag sector options | 2.38–2.49 (FAIL perm) | — | — | Lead-lag is 19% WORSE than random; structural edge only | NO |
| Seasonal sector filter | neutral (adds no value) | — | — | LGBM already captures seasonality; adding features is noise | NO |
| Neural regime allocator | -0.85 vs 0.92 EW | — | — | Equal weight across strategies beats ML allocation | NO — use equal weight |
| VIX spike predictor (LSTM) | AUC 0.908 | — | — | Useful as filter for VIX mean-rev entry, not standalone sector | PARTIAL — feature only |
| Leveraged ETF momentum | survivorship bias | 91% | -70%+ | TQQQ parabolic rise = look-ahead; not reliable | NO |
| Cross-sector mean reversion options | -0.01 best | — | — | Signal real (+0.47% 10d return) but options costs consume edge entirely | NO |
| VRP small account | dead (0/4 gates) | — | — | VRP works for sellers; buying options = systematic theta drag | NO |
| Intraday-overnight factor | Sharpe -5.30 | — | — | IC 0.14 real, but gaps too small (10-30 bps) to cover options costs | NO (useful as feature) |
| Earnings ICs (mega-cap, realistic pricing) | 0/4 gates | — | — | $645 too small; 98% of trades rejected by margin | NO until $2K+ |
| Iron condors (mega-cap, 8-DTE) | WR 89%, Sharpe 1.27 | — | — | Earnings IV consistently overestimates realized move; selling is +EV | YES at $2K+ |

---

## 4. Feature Importance Ranking

From the feature ablation cross-validation study (8 configurations, 2008–2026, 1,333–1,336 trades each):

| Rank | Feature | Group | Ablation Evidence |
|---|---|---|---|
| 1 | `up_capture` | Cross-asset | Removing it IMPROVES Sharpe (2.79→2.84). Conflates with beta exposure. |
| 2 | `sector_spy_beta_63d` | Cross-asset | Part of 3-feature subset that achieves Sharpe 2.57 alone (H_crossasset_only) |
| 3 | `vol_21d` | Volatility | Top-5 subset retains most of full-model edge |
| 4 | `sector_relative_vol_21d` | Cross-asset | In top-5 subset; cross-sectional vol rank matters |
| 5 | `ret_126d` | Momentum | 6-month return in top-5 subset |
| 6 | `trend_r2_63d` | Trend | In top-10 subset |
| 7 | `ret_5d` | Momentum | Short-term return; in top-10 |
| 8 | `maxdd_63d` | Volatility | Risk quality measure; top-10 |
| 9 | `vol_63d` | Volatility | Longer-horizon volatility; top-10 |
| 10 | `ret_252d` | Momentum | 1-year return; top-10 |
| 11–21 | `ret_10d`, `ret_21d`, `ret_63d`, `sharpe_63d`, `pct_52w_high`, `mom_accel`, `pct_pos_months_12m`, `sortino_63d`, `calmar_1y`, `trend_slope_63d`, `cross_sector_dispersion` | Various | Marginal individual contribution |

**Critical finding:** Removing volatility features entirely (F_no_volatility) RAISES Sharpe to 2.875 — the best-performing subset. Removing momentum features (E_no_momentum) drops yearly consistency from 16/17 to 15/17 but barely touches Sharpe (2.74). The 3-feature cross-asset-only subset (H) achieves Sharpe 2.57 with 17/17 profitable years (perfect year count) — the irreducible structural core of the strategy.

The multi-factor research (entry #951) found the top features from a production LGBM run: Calmar ratio, down-capture, relative strength vs SPY, relative volume, trend R². These are the features that separate good sectors from bad by quality, not just recent momentum.

---

## 5. Temporal Stability (Subperiod Analysis)

From `subperiod_results.json`, full period 2008–2026:

| Period | Label | Sharpe | Sortino | WR | MDD | Trades | Notes |
|---|---|---|---|---|---|---|---|
| 2008–2026 (full) | A_full | 3.37 | 12.88 | 52.3% | -5.9% | 1,335 | Baseline |
| 2008–2012 | B (crisis/recovery) | 3.51 | 13.23 | 53.3% | -5.9% | 582 | Best sub-period Sharpe |
| 2013–2017 | C (low-VIX bull) | 2.63 | 20.20 | 40.0% | -8.2% | 60 | Fewest trades (low VIX, few entries); bear WR 4.2% — regime stress |
| 2018–2022 | D (vol spike/COVID/hikes) | 2.09 | 6.17 | 53.3% | -17.3% | 507 | Highest MDD; bear trades abundant; weakest Sharpe |
| 2023–2026 | E (most recent) | 3.21 | 13.80 | 50.5% | -11.9% | 186 | Live-trading relevant; PF 5.95 best |
| Pre-COVID | F (2008–2019) | 3.28 | 12.91 | 50.1% | -5.9% | 726 | Consistent |
| Post-COVID | G (2020–2026) | 2.06 | 4.54 | 54.8% | -32.6% | 609 | Worst Sortino; worst MDD; vol-spike era is hardest regime |

**Regime balance** (bull/bear market days across full period):
- Bull trades: 1,010 (75.7%), WR 58.4%, PnL $37K
- Bear trades: 325 (24.3%), WR 33.2%, PnL $5.2K
- Regime gap (WR basis): |58.4 - 33.2| / 62.1 = **0.40 — below 0.50 threshold, PASS**

**VIX regime split:**
- High VIX (≥20): 52.1% of trades, $28.7K PnL — this is the primary edge source
- Low VIX (<20): 47.9% of trades, $13.6K PnL — bear put spreads carry this regime

**Key temporal risk:** The 2013–2017 low-VIX bull market had only 60 trades (VIX rarely hit 20) and bear WR dropped to 4.2%, representing the worst-case quiet-market regime. Post-COVID (2020–2026) has the worst Sortino (4.54) and worst MDD (-32.6%) due to violent vol-spike events. The strategy is structurally weaker during prolonged low-volatility regimes.

---

## 6. Risk Assessment

**Definitive (honest, hold-to-expiry) metrics:**
- Sharpe: 3.04 | Sortino: not separately reported in definitive run
- MDD: -9.6% (production-honest) | Calmar: ~6.6 (63% CAGR / 9.6% MDD)
- WR: 57% | PF: implied ~2.3 from trade stats
- Day concentration: not the primary concern for monthly-options strategy

**Bootstrap stress test (10K paths, 5yr, from actual 1,258 trade PnLs):**
- Base case: $645→$15,229 median (88% CAGR), 0% ruin, 100% profitable
- Honest forward (50% of backtest performance): $645→$7,911, 0% ruin, 100% profitable
- Combined adversarial (degraded WR + fat tails + higher commissions): $645→$11,753, 0% ruin
- Degraded WR -10% only: 0.9% ruin — only scenario with any ruin at all
- **Conclusion: viable for live deployment even at substantially degraded performance**

**Structural risk factors:**
1. Strategy is inactive 64% of the time under VIX>20-only bull mode. Bear puts fill the gap but at 2x worse Sharpe and 13x worse MDD.
2. Post-COVID epoch (2020-2026) MDD -32.6% is much worse than full-period -9.6% — vol-spike regime is the main risk.
3. At $645 account, commission drag is 31% of capital in year 1 ($203/yr). Drops to 4% at $5K, 0.8% at $25K. Strategy needs scale to be efficient.
4. Pricing validation: B-S flat-vol underprices put premium vs skew-adjusted. IC credits underpriced by 68% — iron condor income estimates are CONSERVATIVE. Bull call spread estimates off by +22% (slightly high). 15% haircut is validated.
5. R1 regime gap 0.40 on WR basis — PASSES threshold of 0.50 but not a near-zero gap. Bear days have meaningful WR suppression.

---

## 7. What We Tried That Didn't Work (Negative Results)

| Approach | Outcome | Why It Fails |
|---|---|---|
| PMCC (Poor Man's Covered Call) on sector ETFs | Sharpe -1.3 to -1.8, -93% MDD | LEAPS too expensive; commissions eat all income |
| Weekly DTE (5-7 day) spreads | Best Sharpe 0.44, R1 gap 0.70-0.80 | Too much noise; monthly DTE strictly superior |
| Butterfly structures | 0/4 gates, WR 0-16%, near-100% MDD | Requires precise price prediction; momentum can't do that |
| Sector iron condors (sell premium on ETFs) | 0/4 gates, account goes negative | Sector ETFs too volatile for IC at $645; R1 gap excellent but strategy loses money |
| Broad universe (25 ETFs) | Honest Sharpe 0.82 (claimed 4.70) | Sharpe inflation bug — more compounding inflated denominator 4x; sectors-only is true best |
| Leveraged ETF momentum | Survivorship bias | TQQQ's 2010-2026 rise is historical artifact; not reproducible forward |
| Sector lead-lag correlations | 19% WORSE than random | Laggards keep lagging (momentum persistence); selection method is noise on top of structural premium |
| Sector seasonality filter | Null (identical Sharpe) | LGBM already captures seasonality; features don't add signal |
| VIX spike predictor as sector filter | LSTM makes no difference (Sharpe 3.21 vs 3.22 baseline) | VIX momentum already regime-robust (R1 gap 0.11); don't add complexity |
| Neural regime allocator (ML allocation) | Sharpe -0.85 vs EW 0.92 | Equal weight across validated strategies beats any complex allocation |
| Drawdown control overlays | Null result | VIX>20 filter already prevents most drawdowns; DD rules never trigger |
| Trailing stop exits | HURTS WR (88%→84%), increases MDD | Kills winners early; structural ETF mean-reversion makes trailing stops counterproductive |
| VIX dead-zone skip (skip 18-22) | Sharpe 2.93 vs 3.10 baseline | Skipping gray-zone loses quality trades with no quality gain |
| Cross-sector mean reversion via options | Best Sharpe -0.01 | Signal is real (+0.47% 10d) but options theta/bid-ask consume the entire edge |
| VRP strategies at $645 | 0/4 gates, all strategies fail | VRP works for sellers; buying options (the only option at $645) loses systematically to theta |
| Intraday-overnight gap factor | Sharpe -5.30 | Gaps too small (10-30 bps); IC 0.14 is real but not tradeable via options |
| PEAD as standalone (not overlay) | Sharpe 1.02 but MDD -60.4% | Too volatile alone; only viable as 12% supplement to sector core |
| Earnings ICs at $645 | 0/4 gates (98% trades rejected) | Stock prices too high; max loss $300-1500 exceeds $645 budget |
| DTE×Moneyness interaction (8 combos) | Best: DTE=21+3%OTM Sharpe 3.01 | Optimal OTM increases with DTE. DTE=10→1%, DTE=14→2%, DTE=21→3% |
| Combined optimal stacking | Best: Sharpe 2.96 (+22% vs baseline) | 2% spread + DTE=14 + 2% OTM combined. Sub-additive but real |
| Top-K sensitivity (K=1 to K=5) | K=1 Sharpe 2.70 (+9%), monotonically ↓ | LGBM edge concentrated in top-1 sector. But K=1 = 433 trades (concentration risk) |
| LGBM hyperparameters | Production near-optimal (+3% max) | Longer WF window -28%. lr=0.01 -5%. Low-leverage tuning |
| Cost sensitivity (8 scenarios) | Worst case Sharpe 1.88 | Strategy survives 64% cost increase. $10 commission + 33% haircut still profitable |
| Moneyness cross-validation | 2% OTM optimal (Sharpe 2.43) | ITM catastrophic (-0.88). 5% OTM reverts to ATM. OTM avg 2.28 >> ITM avg 0.32 |
| Regime threshold sensitivity | VIX<25 marginally better (+2%) | GRU>0.4 confirmed optimal. Removing GRU hurts -16%. Conservative >0.5 hurts -26% |
| Earnings standalone (14 features) | **Sharpe 2.85**, 5/5 gates | Rank corr 0.26 with production (complementary). Works all regimes/months. ~15% alpha over random. Paper engine deployed. |
| Earnings sector features (5 vars, perm p=0.04) | Earnings-only Sharpe 2.62 (+7.2% vs baseline) | Genuine signal. avg_surprise + post_drift rank #3-4 importance. But adding to full feature set hurts. Suggests separate strategy |
| CTA flow as binary filter | Sharpe 0.30 vs 0.65 baseline (HARMFUL) | Money flow timing is actively harmful; better as ML feature |
| Sector IC income overlay on bull spreads | Pure drag (-$74/trade) | Compounds risk, adds no signal |
| Relative value pairs (long laggard / short leader) | All fail permutation | Laggards keep lagging; long/short does not neutralize sector beta |
| RL growth allocator (PPO, 4 assets) | Sharpe 1.15 (p=0.011) | Learns real signal but TQQQ+200MA simple rule dominates (Sharpe 1.57). Agent favors concentrated bets |
| GPU drawdown predictor (LSTM) | AUC 0.74 | Predicts 5d drawdowns but VIX threshold UPRO (Sharpe 2.23) still beats DD-aware variant (0.71) |
| GPU trend predictor (Transformer) | Portfolio Sharpe 3.44 at p≥0.50 | Driven by 17-asset diversification, not prediction. Bear year precision drops to 38%. Not actionable |
| Spread outcome predictor (4 variants) | All fail (Sharpe -33 to 0) | Training on win/loss instead of equity returns is strictly worse. Bear WR 0% across all models |
| ML cross-sectional momentum (Fama-French L/S) | Sharpe -0.213 | 28 ETFs, momentum crashes kill it. Permutation p=0.83. Sub-period wildly unstable |
| NN vol forecaster (LSTM, 196 folds) | Correlation 0.000 | Model learns nothing about 5d forward vol — just predicts training mean. VIX/trailing-vol is best |
| GRU regime detector (170 folds) | R²=0.173, Corr=0.42 | Wildly inconsistent per-fold (R² -6.5 to +0.35). Explains only 17% of regime variance. VIX threshold is simpler and more robust. |

---

## 8. Next Research Directions

**Priority 1 — Live paper validation (A/B test deployed):**
- Three paper engines running: V7 (2% OTM/DTE=21), V8 (2% OTM/DTE=14), V9 (3% OTM/DTE=21). All start $645. First rebalance Mon Jul 27 at 4:30 PM ET.
- After 4-8 weeks of data, compare realized Sharpe/WR/slippage to pick production winner.
- V9 (3% OTM) is the backtest favorite (Sharpe 3.01) but needs live validation.

**Priority 2 — ML improvement (incremental, not foundational):**
- Cross-sectional z-score features (`xs_zscore_5d`) improved standalone IC 2.6x in the enhanced features study (#994). Worth integrating into production LGBM — this is the single most actionable ML improvement found.
- MLP ranking (from sector momentum neural study #992) was 25% better than LGBM at ordering sectors. Test MLP rankings inside the production spreads framework to see if translates to Sharpe improvement.
- Flow signals improved IC 4x (0.044→0.178) but fail in high-VIX crash regimes. Regime-conditional feature selection (normal vs crash feature sets) could unlock this without the R1 failure.

**Priority 3 — Scale + capital milestones:**
- Current growth path: sector bull+bear → reach $2K (unlocks earnings ICs) → reach $10K (unlocks SPY ICs + VIX mean-rev). Most important parameter is position sizing: switching from fixed $200 to linear scaling gives 8.5x better 5yr equity.
- At $2K, add earnings iron condors (mega-cap, quarterly, validated 89% WR). This is the biggest growth lever.

**Priority 4 — PEAD options overlay:**
- PEAD call spreads (Sharpe 1.38, gap>5%, 45 DTE) are validated and complementary — fire during earnings seasons while sector spreads fire biweekly. Worth adding at half-size once capital reaches $1,500+ to cover the position margin.

**Priority 5 — Regime-conditional bear side:**
- The 2013–2017 low-VIX subperiod (60 trades, bear WR 4.2%) is the strategy's structural weakness. Research a third mode for sustained low-VIX environments where neither bull spreads (VIX too low) nor bear puts have edge. Candidate: earnings ICs as the primary strategy during quiet market regimes.

**Do NOT pursue:**
- Any additional architecture exploration for the sector model — LGBM is near-optimal, returns are diminishing
- Leveraged ETFs, butterflies, PMCC, weekly DTE, sector ICs — all thoroughly rejected with clear failure modes documented
- Broad universe expansion — the honest Sharpe after correcting the inflation bug makes sectors-only the winner

---

## 9. Meta-Insight: Simple Rules Beat Neural Approaches

Across 10+ neural/ML experiments (LSTM drawdown, Transformer trend, LSTM vol, RL allocator, spread outcome MLP/GRU, neural regime allocator, cross-sectional momentum, NN vol forecaster), **zero** beat simple rules (VIX threshold, LGBM sector ranking, trend-following SMA). The pattern is consistent:
- Neural models find real but small signal (permutation p < 0.05 in some cases)
- But the signal doesn't translate to better risk-adjusted returns than a simple rule
- VIX threshold alone captures ~75% of the strategy's value
- LGBM adds ~25% incremental Sharpe, primarily through risk management (5x better MDD)
- Complex allocation (RL, neural regime switching) actively hurts vs equal weight

**Implication**: Future research should focus on finding NEW simple signals/rules, not making existing ones more complex. The strategy is near-optimal.

---

---

## 10. V10 and Beyond (KB #267–#282, July 2026)

**V10** = V9.3 + structural optimizations: 8 positions (4 top/4 bottom), 4% OTM, 30% profit target, monthly rebalance, 1/rank-weighted sizing. Backtest Sharpe 6.32 (highest ever), but see BS pricing caveat below.

**V10 Adversarial Audit (KB #280):** 6/8 checks PASS. Signal is REAL — no feature/label leakage, WF integrity verified, permutation test passed, LGBM rank correlation rho=0.52. Two failures: minor XLC survivorship (started June 2018), and CRITICAL BS pricing gap.

**BS Pricing Calibration (KB #282, MLflow exp 268):** 112,181 real options compared to our BS estimates.
- **Root cause: ATR-based IV estimation**, not the BS formula. Market IV gives R²=0.9955.
- BS underprices by 72.7% median. For V10's 4% OTM/DTE28, need ~90% upward correction.
- Multivariate correction model (R²=0.93): market_mid ≈ 1.12*BS - 95*moneyness + 0.18*DTE + 71*IV - 6.16
- For SPREADS, the error partially cancels between legs. Calibrated spread backtest running to determine honest Sharpe.
- **Fix**: Replace ATR-based IV with historical IV percentiles or market-implied IV.
- **GROUND TRUTH (2026-07-27 live RH quotes)**: XLE $61/$63 bull call spread, 25 DTE. BS+15% estimates $0.11/share. Real market mid: $0.43/share (4x higher). Options ARE liquid (OI 5K-7K, vol 100-600). Spread is tradeable but 4x more expensive than backtested.
- **V10 calibrated backtest (KB #284)**: With corrected pricing, V10 produces 0 trades (costs too high) or Sharpe -2.46 (market IV). The RANKING signal is real but spread execution doesn't survive realistic costs.

**Equity Rotation (KB #285, MLflow exp 271):** LGBM ranking as pure equity rotation (no options). Best: Long Top-2 monthly → Sharpe 1.40, Sortino 2.25, WR 64%, MDD -12%, +17.5% alpha vs SPY. Permutation p=0.002. HONEST Sharpe with zero pricing assumptions.

**Leveraged ETF Rotation (KB #286, MLflow exp 273):** Leverage does NOT improve risk-adjusted returns. 2x Top-2: Sharpe 1.27 (-10%), 3x: 0.85 (-40%). Vol decay averages -1.2%/trade. Inverse = DEATH. Unleveraged wins Sharpe.

**Score: Simple Rules 26, Complex 0.** Every complex approach tested (neural ranker, RL allocator, dynamic DTE, flow features at V10 level, multi-model ensemble, leveraged ETFs, options spreads with calibrated pricing) has failed to beat simple rules.

---

*All MLflow experiment references: exps 82–268+ cover this body of research. Paper engine states in `state/sector_combined_v*.json`.*
