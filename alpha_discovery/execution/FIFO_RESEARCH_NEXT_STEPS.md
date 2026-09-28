# FIFO Research — Comprehensive Results After Overnight Sweep
# Date: 2026-05-08 (Updated 02:30 ET with v2+v3 COMPLETE results)

## WHAT WE KNOW (Definitive — 40+ configs tested)

1. **CNN-Mamba v2 signal IS directionally correct** — midpoint WR=56-58% at top confidence
2. **Midpoint edges do NOT survive FIFO fills** — adverse selection drops WR by 5-20 points
3. **No execution config is profitable** — tested 40+ configs across v1/v2/v3:
   - Passive limit: WR=43-49%, always negative
   - Symmetric TP/SL (needs 51% WR): WR=43-49%, never crosses breakeven
   - Chase entry: WR=36-46%, WORSE than passive (-0.94 to -2.26 tk/trade)
   - Chase + conviction exit: WR=36-39%, catastrophic
   - Trailing stop: WR=0.000 (fill sim bug, but direction clearly negative)
   - Signal-flip exit: WR=0.000 (fill sim bug)
   - Prime hours: slightly better but still negative
   - Higher confidence: marginally helps passive, doesn't help chase
   - SHORT-only = catastrophic (WR=36%, 0% green)
   - LONG-only chase = equally catastrophic (WR=38%, 0-3% green)
   - LONG-only passive = least bad (WR=47-48%, 15-33% green)
4. **Best result**: passive long c200 tp4/sl4 = 33% green days, Sortino=-0.35 (only 325 trades, 12 dates)
5. **Best high-volume**: passive long c075 tp6/sl3 = 21% green days, Sortino=-0.39
6. **Adverse selection cost**: ~5-10 ticks per trade
7. **Chase entry makes things WORSE** — force-chase costs -2.26tk/trade vs -0.61tk passive
8. **FIFO outcome predictor**: AUC=0.50 (random). Pre-trade features cannot predict FIFO profitability

## COMPLETE RESULTS TABLE

### v2 (Neptune, COMPLETE — 17 configs, passive + chase, long-only + both)
| Config | Entry | Trades | WR | Sortino | Green% | PnL/tr |
|--------|-------|--------|-----|---------|--------|--------|
| c200_tp4_sl4 | PASSIVE | 325 | 0.480 | -0.351 | 33% | -0.45 |
| c075_prime_tp3_sl3 | PASSIVE | 2383 | 0.458 | -0.360 | 13% | -0.64 |
| c075_prime_tp4_sl4 | PASSIVE | 2043 | 0.441 | -0.383 | 15% | -0.68 |
| c075_tp6_sl3 | PASSIVE | 3303 | 0.342 | -0.386 | 21% | -0.78 |
| c075_tp3_sl3 | PASSIVE | 3871 | 0.473 | -0.390 | 15% | -0.61 |
| force_chase_tp4_sl4 | CHASE | 5164 | 0.446 | -0.403 | 5% | -2.26 |
| c075_tp4_sl4 | PASSIVE | 3327 | 0.462 | -0.403 | 15% | -0.69 |
| c075_tp2_sl2 | PASSIVE | 4479 | 0.486 | -0.435 | 3% | -0.62 |
| chase_tp4_sl4 | CHASE | 4662 | 0.463 | -0.436 | 10% | -0.68 |
| c100_tp4_sl4 | PASSIVE | 2861 | 0.447 | -0.436 | 13% | -0.64 |
| c100_flipex_sl4 | PASSIVE | 4294 | 0.000 | -0.445 | 0% | -0.52 |
| c100_tp3_sl3 | PASSIVE | 3263 | 0.428 | -0.449 | 8% | -0.65 |
| c100_tp6_sl3 | PASSIVE | 2817 | 0.319 | -0.451 | 8% | -0.89 |
| c075_flipex_sl6 | PASSIVE | 5200 | 0.000 | -0.456 | 0% | -0.52 |
| c075_trail2_sl4 | PASSIVE | 5160 | 0.000 | -0.461 | 0% | -0.45 |
| c075_trail3_sl6 | PASSIVE | 5152 | 0.000 | -0.463 | 0% | -0.43 |
| both_c075_tp3_sl3 | PASSIVE | 8586 | 0.467 | -0.689 | 3% | -0.66 |

### v3 (Jupiter, 7/16 configs done — chase + conviction exit)
| Config | Side | Trades | WR | Sortino | Green% | PnL/tr |
|--------|------|--------|-----|---------|--------|--------|
| conv30_short_c03 | SHORT | 8924 | 0.362 | -0.709 | 0% | -1.18 |
| conv30_short_c05 | SHORT | 8203 | 0.365 | -0.724 | 0% | -1.11 |
| conv30_short_c075 | SHORT | 5877 | 0.368 | -0.719 | 3% | -1.10 |
| conv30_both_c03 | BOTH | 8918 | 0.363 | -0.702 | 3% | -1.15 |
| conv30_both_c05 | BOTH | 8410 | 0.378 | -0.712 | 0% | -1.00 |
| conv30_long_c03 | LONG | 7896 | 0.376 | -0.701 | 3% | -0.99 |
| conv30_long_c05 | LONG | 6042 | 0.386 | -0.715 | 0% | -0.94 |

## ROOT CAUSE ANALYSIS

The fundamental problem: **our signal predicts direction at the mid-price, but we can only execute at the BBO. Getting filled at the BBO is informative — it means the market is moving THROUGH our price.**

This creates an information paradox:
- Signal says "price will go up" → post buy at bid
- Fill happens when someone sells into our bid → informed seller → price likely to drop
- Our directional signal (positive) conflicts with the fill signal (negative)
- The fill signal wins most of the time

## NEXT RESEARCH DIRECTIONS (Priority Order)

### 1. RETRAIN EXECUTION FILTER ON FIFO OUTCOMES
We have per-trade FIFO data from v1 (16K+ trades across 40 dates with full feature set).
Train XGBoost/LGBM to predict FIFO P&L (not midpoint P&L) from:
- Signal features (confidence, persistence, agreement)
- Microstructure features (bid-ask imbalance, queue depth, trade flow)  
- Temporal features (time of day, volatility regime)
- Order features (queue position at entry)

This would tell us: **are there conditions where FIFO P&L is positive?**

### 2. CONDITIONAL ENTRY: WAIT FOR PULLBACK
Instead of posting immediately when signal fires:
- Signal says "buy" → WAIT
- Wait for 1-2 tick pullback (price drops toward us)
- Then post buy limit at new lower price
- This reverses the adverse selection: fill happens when price has already dropped
- Risk: miss trades if no pullback occurs

### 3. MAGNITUDE-BASED FILTERING
CNN-Mamba v2 predicts "expected tick move." Currently we filter on |pred| > threshold.
But what if we only trade when the PREDICTED MAGNITUDE exceeds the adverse selection cost?
- Avg adverse selection: ~5 ticks
- Only trade when |predicted move| > 5 ticks?
- Problem: max |pred| is only 1.5 for CNN-Mamba v2 (it predicts at most 1.5 ticks)
- This means: the model never predicts a move large enough to overcome FIFO costs
- Implication: need a model that predicts LARGER moves, or needs to be recalibrated

### 4. DIFFERENT HORIZONS / STRIDE
Current: 250ms stride, 1s/5s/10s horizons
- Could a 1-min horizon with 5s stride give larger predicted moves?
- Larger predicted moves → more room for adverse selection → might be profitable
- Trade-off: fewer signals, slower reaction

### 5. MARKET MAKING APPROACH
Instead of directional trading, use the signal for skewed market making:
- Signal says "buy" → widen ask spread, tighten bid spread
- Capture more fills on the side the signal predicts correctly
- This naturally handles adverse selection (you're always in the book)
- Requires co-location for competitive edge

### 6. CROSS-INSTRUMENT
ES has 1 tick = 0.25 pts = $12.50. Spread is always 1 tick during RTH.
NQ has smaller tick ($5), MES has smaller contract ($1.25/tick).
Different adverse selection characteristics.

### 7. FIX FILL SIM MARKET ENTRY BUG
The market entry configs produced >100% fill rates (re-entry after quick SL).
Fix the fill sim to not re-enter for N bars after exit.
Then retest market entry at very high confidence for comparison.
