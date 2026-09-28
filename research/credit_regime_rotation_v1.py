#!/usr/bin/env python3
"""
Credit Regime Rotation Signal v1 — Cross-Asset Stress Regime for Sector ETF Options

THESIS: Credit market stress (HYG/LQD spread), rate volatility (TLT realized vol),
and bank stress (KRE/XLF ratio) create distinct regime states that predict which
sectors will mean-revert vs trend. During credit stress transitions, defensive sectors
get overbought (sell) and beaten-down cyclicals snap back (buy). During credit calm,
quality mean-reversion works best on sectors that overshot.

This signal is NOVEL because:
- We use credit TRANSITIONS (delta of regime), not the level (which is just VIX proxy)
- We combine 3 independent stress axes: credit, rates, banking
- We construct a composite "stress acceleration" that captures regime CHANGES
- Sector selection is conditional on stress direction, not just level

SIGNAL CONSTRUCTION:
1. Credit Stress: z-score of HYG/LQD ratio changes (junk vs investment grade)
2. Rate Vol: z-score of TLT 10d realized vol vs 60d norm
3. Bank Fragility: z-score of KRE/XLF ratio changes (regional vs large banks)
4. Composite: weighted average of the 3 z-scores
5. Stress ACCELERATION: 5d change in composite (captures regime transitions)
6. Sector selection:
   - Stress accelerating UP: buy beaten-down cyclicals (contrarian), sell overbought defensives
   - Stress accelerating DOWN: buy recovering defensives laggards, sell cyclical that ran too far
   - Use sector RSI relative to stress regime to pick specific ETFs

5-day hold, sliding walk-forward, 5-gate validation.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Config ──────────────────────────────────────────────────────────────
SECTOR_TICKERS = ['XLF', 'XLE', 'XLU', 'XLK', 'XLY', 'XLP', 'XLRE', 'XLV', 'XLI', 'XLB', 'XLC']
BENCHMARK = 'SPY'
CREDIT_TICKERS = ['HYG', 'LQD', 'TLT', 'KRE']
ALL_TICKERS = SECTOR_TICKERS + [BENCHMARK] + CREDIT_TICKERS

HOLD_DAYS = 5
RT_COST_PCT = 0.0003  # 0.03% round-trip (conservative for options)
PERM_ITERS = 1000

# Walk-forward params
WF_TRAIN_DAYS = 252  # 1 year training
WF_TEST_DAYS = 21    # 1 month OOT
WF_STEP_DAYS = 21    # slide by 1 month

# Signal params
CREDIT_LOOKBACK = 20    # days for credit ratio z-score
RATE_VOL_FAST = 10      # fast realized vol window
RATE_VOL_SLOW = 60      # slow realized vol normalization
BANK_LOOKBACK = 20      # bank fragility lookback
STRESS_ACCEL_WINDOW = 5 # stress acceleration (transition detection)
RSI_PERIOD = 14         # sector RSI

DEFENSIVE = ['XLU', 'XLP', 'XLV']
CYCLICAL = ['XLF', 'XLE', 'XLK', 'XLY', 'XLI', 'XLB', 'XLC', 'XLRE']

TOP_N_LONG = 2   # how many sectors to go long
TOP_N_SHORT = 2  # how many sectors to go short

print("=" * 70)
print("CREDIT REGIME ROTATION SIGNAL v1")
print("Cross-Asset Stress Transitions for Sector ETF Options")
print("=" * 70)

# ── Data Download ───────────────────────────────────────────────────────
print("\nDownloading data...")
data = yf.download(ALL_TICKERS, start='2014-01-01', end='2026-08-18', auto_adjust=True)

close = data['Close'].copy()
volume = data['Volume'].copy()

close = close.ffill().dropna()
volume = volume.ffill().fillna(0)

print(f"Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} trading days")

# Verify we have credit data
for t in CREDIT_TICKERS:
    if t not in close.columns:
        print(f"WARNING: Missing {t}")
    else:
        valid_pct = close[t].notna().mean() * 100
        print(f"  {t}: {valid_pct:.1f}% valid")

# ── Helper Functions ────────────────────────────────────────────────────
def compute_rsi(series, period=14):
    """Standard RSI calculation."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def z_score(series, window):
    """Rolling z-score."""
    mu = series.rolling(window, min_periods=max(10, window // 2)).mean()
    sigma = series.rolling(window, min_periods=max(10, window // 2)).std()
    return (series - mu) / sigma.replace(0, np.nan)

def sharpe_ratio(returns):
    if len(returns) < 10 or returns.std() == 0:
        return 0.0
    return returns.mean() / returns.std() * np.sqrt(252 / HOLD_DAYS)

def sortino_ratio(returns):
    if len(returns) < 10:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return float('inf') if returns.mean() > 0 else 0.0
    return returns.mean() / downside.std() * np.sqrt(252 / HOLD_DAYS)

def profit_factor(returns):
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    return gross_profit / gross_loss

def win_rate(returns):
    if len(returns) == 0:
        return 0.0
    return (returns > 0).mean()

def max_drawdown(equity_curve):
    """Max drawdown from equity curve."""
    peak = equity_curve.expanding().max()
    dd = (equity_curve - peak) / peak
    return dd.min()

def permutation_test(signal_returns, n_iter=1000):
    """Permutation test: shuffle returns, compare Sharpe distribution."""
    observed_sharpe = sharpe_ratio(signal_returns)
    count_better = 0
    arr = signal_returns.values.copy()
    for _ in range(n_iter):
        np.random.shuffle(arr)
        if sharpe_ratio(pd.Series(arr)) >= observed_sharpe:
            count_better += 1
    return observed_sharpe, count_better / n_iter

def regime_stratify(returns, spy_returns_aligned):
    """Split returns by green (SPY up) vs red (SPY down) periods."""
    # Align indices
    common = returns.index.intersection(spy_returns_aligned.index)
    r = returns.loc[common]
    s = spy_returns_aligned.loc[common]

    green_mask = s > 0
    red_mask = s <= 0
    green_rets = r[green_mask]
    red_rets = r[red_mask]

    g_sharpe = sharpe_ratio(green_rets) if len(green_rets) > 5 else 0.0
    r_sharpe = sharpe_ratio(red_rets) if len(red_rets) > 5 else 0.0

    # Regime gap
    max_abs = max(abs(g_sharpe), abs(r_sharpe), 0.001)
    regime_gap = abs(g_sharpe - r_sharpe) / max_abs

    return {
        'green_sharpe': round(g_sharpe, 3),
        'red_sharpe': round(r_sharpe, 3),
        'regime_gap': round(regime_gap, 3),
        'green_n': int(green_mask.sum()),
        'red_n': int(red_mask.sum()),
        'green_wr': round(win_rate(green_rets), 3),
        'red_wr': round(win_rate(red_rets), 3),
    }

# ── Build Stress Indicators ────────────────────────────────────────────
print("\nBuilding credit regime indicators...")

# 1. Credit Stress: HYG/LQD ratio (high = risk-on, low = stress)
credit_ratio = close['HYG'] / close['LQD']
credit_ratio_chg = credit_ratio.pct_change(5)  # 5-day change
credit_z = z_score(credit_ratio_chg, CREDIT_LOOKBACK)

# 2. Rate Volatility: TLT realized vol (fast vs slow)
tlt_ret = close['TLT'].pct_change()
rate_vol_fast = tlt_ret.rolling(RATE_VOL_FAST).std() * np.sqrt(252)
rate_vol_slow = tlt_ret.rolling(RATE_VOL_SLOW).std() * np.sqrt(252)
rate_vol_ratio = rate_vol_fast / rate_vol_slow.replace(0, np.nan)
rate_vol_z = z_score(rate_vol_ratio, CREDIT_LOOKBACK)

# 3. Bank Fragility: KRE/XLF ratio (regional vs large bank relative performance)
bank_ratio = close['KRE'] / close['XLF']
bank_ratio_chg = bank_ratio.pct_change(5)
bank_z = z_score(bank_ratio_chg, BANK_LOOKBACK)

# 4. Composite Stress Score (equal weight the 3 axes)
# Note: credit_z is INVERTED (negative = stress), rate_vol_z positive = stress, bank_z negative = stress
stress_composite = (-credit_z + rate_vol_z - bank_z) / 3.0
stress_composite = stress_composite.fillna(0)

# 5. Stress ACCELERATION (the key innovation — regime transitions matter more than levels)
stress_accel = stress_composite.diff(STRESS_ACCEL_WINDOW)
stress_accel_z = z_score(stress_accel, 40)  # z-score vs longer history

print(f"  Credit ratio (HYG/LQD) computed: {credit_z.notna().sum()} valid days")
print(f"  Rate vol ratio computed: {rate_vol_z.notna().sum()} valid days")
print(f"  Bank fragility computed: {bank_z.notna().sum()} valid days")
print(f"  Stress acceleration computed: {stress_accel_z.notna().sum()} valid days")

# ── Compute Sector RSI ─────────────────────────────────────────────────
sector_rsi = pd.DataFrame(index=close.index)
for ticker in SECTOR_TICKERS:
    sector_rsi[ticker] = compute_rsi(close[ticker], RSI_PERIOD)

# Relative RSI: each sector's RSI vs cross-sectional median
rsi_median = sector_rsi[SECTOR_TICKERS].median(axis=1)
relative_rsi = sector_rsi[SECTOR_TICKERS].sub(rsi_median, axis=0)

# ── Sector Forward Returns ─────────────────────────────────────────────
sector_close = close[SECTOR_TICKERS]
sector_5d_fwd = sector_close.shift(-HOLD_DAYS) / sector_close - 1

# SPY forward returns for regime classification
spy_close = close[BENCHMARK]
spy_5d_fwd = spy_close.shift(-HOLD_DAYS) / spy_close - 1
spy_20d_ret = spy_close.pct_change(20)  # for regime classification

# ── Walk-Forward Signal Generation ──────────────────────────────────────
print("\n" + "=" * 70)
print("WALK-FORWARD BACKTEST (Sliding Window)")
print(f"  Train: {WF_TRAIN_DAYS}d | Test: {WF_TEST_DAYS}d | Step: {WF_STEP_DAYS}d")
print("=" * 70)

# We need enough history for indicators
start_idx = max(RATE_VOL_SLOW, CREDIT_LOOKBACK, BANK_LOOKBACK, 60) + WF_TRAIN_DAYS
dates = close.index

all_trades = []  # list of dicts: date, ticker, direction, entry_ret, signal_strength

for wf_start in range(start_idx, len(dates) - HOLD_DAYS - WF_TEST_DAYS, WF_STEP_DAYS):
    train_end = wf_start
    train_start = wf_start - WF_TRAIN_DAYS
    test_start = wf_start
    test_end = min(wf_start + WF_TEST_DAYS, len(dates) - HOLD_DAYS)

    if test_end <= test_start:
        continue

    train_dates = dates[train_start:train_end]
    test_dates = dates[test_start:test_end]

    # ── TRAIN: Learn optimal thresholds from training window ──
    # Find stress acceleration thresholds that historically worked
    train_accel = stress_accel_z.loc[train_dates].dropna()

    if len(train_accel) < 60:
        continue

    # Threshold: top/bottom quartile of stress acceleration
    accel_q75 = train_accel.quantile(0.75)
    accel_q25 = train_accel.quantile(0.25)

    # Train: measure which sector rotation works in each regime
    # During stress acceleration (accel > q75):
    #   - Sectors with LOW relative RSI (oversold) tend to bounce (contrarian)
    #   - Sectors with HIGH relative RSI (overbought defensives) tend to give back gains
    # During stress deceleration (accel < q25):
    #   - Recovery phase: sectors that lagged during stress catch up

    # Validate that the signal direction holds in training
    train_fwd = sector_5d_fwd.loc[train_dates]
    train_rel_rsi = relative_rsi.loc[train_dates]
    train_accel_full = stress_accel_z.loc[train_dates]

    # Score sectors: in high-stress-accel periods, does low RSI predict positive returns?
    stress_up_mask = train_accel_full > accel_q75
    stress_down_mask = train_accel_full < accel_q25

    # Compute correlation between relative RSI and forward returns in each regime
    corr_stress_up = 0
    corr_stress_down = 0
    count_up = stress_up_mask.sum()
    count_down = stress_down_mask.sum()

    if count_up > 10:
        # Stack all sector observations
        rsi_vals = train_rel_rsi[stress_up_mask].values.flatten()
        fwd_vals = train_fwd[stress_up_mask].values.flatten()
        valid = ~(np.isnan(rsi_vals) | np.isnan(fwd_vals))
        if valid.sum() > 20:
            corr_stress_up = np.corrcoef(rsi_vals[valid], fwd_vals[valid])[0, 1]

    if count_down > 10:
        rsi_vals = train_rel_rsi[stress_down_mask].values.flatten()
        fwd_vals = train_fwd[stress_down_mask].values.flatten()
        valid = ~(np.isnan(rsi_vals) | np.isnan(fwd_vals))
        if valid.sum() > 20:
            corr_stress_down = np.corrcoef(rsi_vals[valid], fwd_vals[valid])[0, 1]

    # ── TEST: Apply learned thresholds to test window ──
    for test_date in test_dates:
        accel_val = stress_accel_z.get(test_date, np.nan)
        if np.isnan(accel_val):
            continue

        # Get current relative RSI for all sectors
        curr_rsi = relative_rsi.loc[test_date]
        if curr_rsi.isna().all():
            continue

        # Get current stress level for signal strength
        curr_stress = stress_composite.get(test_date, 0)

        signal_direction = None

        if accel_val > accel_q75:
            # Stress accelerating: contrarian on oversold if training showed negative corr
            if corr_stress_up < -0.02:  # RSI negatively correlated with fwd returns
                signal_direction = 'stress_up_contrarian'
            elif corr_stress_up > 0.02:
                signal_direction = 'stress_up_momentum'
        elif accel_val < accel_q25:
            # Stress decelerating: recovery trade
            if corr_stress_down < -0.02:
                signal_direction = 'stress_down_contrarian'
            elif corr_stress_down > 0.02:
                signal_direction = 'stress_down_momentum'
        else:
            # Neutral zone: skip or use pure mean-reversion
            # Use absolute RSI extremes only
            extreme_low = curr_rsi.nsmallest(TOP_N_LONG)
            extreme_high = curr_rsi.nlargest(TOP_N_SHORT)

            # Only trade if extremes are actually extreme
            if extreme_low.min() < -10 and extreme_high.max() > 10:
                signal_direction = 'neutral_meanrev'

        if signal_direction is None:
            continue

        # Select sectors based on signal direction
        if 'contrarian' in signal_direction:
            # Buy most oversold (lowest relative RSI), sell most overbought
            longs = curr_rsi.dropna().nsmallest(TOP_N_LONG).index.tolist()
            shorts = curr_rsi.dropna().nlargest(TOP_N_SHORT).index.tolist()
        elif 'momentum' in signal_direction:
            # Buy strongest RSI, sell weakest (trend continuation)
            longs = curr_rsi.dropna().nlargest(TOP_N_LONG).index.tolist()
            shorts = curr_rsi.dropna().nsmallest(TOP_N_SHORT).index.tolist()
        elif signal_direction == 'neutral_meanrev':
            longs = curr_rsi.dropna().nsmallest(TOP_N_LONG).index.tolist()
            shorts = curr_rsi.dropna().nlargest(TOP_N_SHORT).index.tolist()

        # Record trades
        signal_strength = abs(accel_val)

        for ticker in longs:
            fwd_ret = sector_5d_fwd.loc[test_date, ticker]
            if not np.isnan(fwd_ret):
                all_trades.append({
                    'date': test_date,
                    'ticker': ticker,
                    'direction': 'long',
                    'fwd_ret': fwd_ret,
                    'net_ret': fwd_ret - RT_COST_PCT,
                    'signal_strength': signal_strength,
                    'regime': signal_direction,
                    'stress_accel': accel_val,
                    'rel_rsi': curr_rsi[ticker],
                })

        for ticker in shorts:
            fwd_ret = sector_5d_fwd.loc[test_date, ticker]
            if not np.isnan(fwd_ret):
                all_trades.append({
                    'date': test_date,
                    'ticker': ticker,
                    'direction': 'short',
                    'fwd_ret': -fwd_ret,  # short = inverse
                    'net_ret': -fwd_ret - RT_COST_PCT,
                    'signal_strength': signal_strength,
                    'regime': signal_direction,
                    'stress_accel': accel_val,
                    'rel_rsi': curr_rsi[ticker],
                })

trades_df = pd.DataFrame(all_trades)
print(f"\nTotal trades generated: {len(trades_df)}")

if len(trades_df) < 30:
    print("INSUFFICIENT TRADES — signal too selective or data issue")
    exit(1)

# ── De-duplicate overlapping trades ─────────────────────────────────────
# Only take one trade per ticker per non-overlapping 5-day window
trades_df['date_dt'] = pd.to_datetime(trades_df['date'])
trades_df = trades_df.sort_values('date_dt')

# Group by every 5 days to avoid overlapping positions
trades_df['week_group'] = (trades_df['date_dt'] - trades_df['date_dt'].min()).dt.days // HOLD_DAYS
deduped = trades_df.groupby(['week_group', 'ticker', 'direction']).first().reset_index()
print(f"After deduplication (non-overlapping): {len(deduped)} trades")

# ── Portfolio-Level Returns ─────────────────────────────────────────────
# Aggregate: average return per trading period across all positions
period_returns = deduped.groupby('week_group')['net_ret'].mean()
period_dates = deduped.groupby('week_group')['date'].first()

# Build equity curve
equity = (1 + period_returns).cumprod()

# ── Core Metrics ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESULTS: CREDIT REGIME ROTATION SIGNAL v1")
print("=" * 70)

total_return = equity.iloc[-1] - 1
ann_return = (1 + total_return) ** (252 / (HOLD_DAYS * len(period_returns))) - 1
sharpe = sharpe_ratio(period_returns)
sortino = sortino_ratio(period_returns)
pf = profit_factor(period_returns)
wr = win_rate(period_returns)
mdd = max_drawdown(equity)

print(f"\nTotal return:     {total_return:.2%}")
print(f"Annualized return: {ann_return:.2%}")
print(f"Sharpe ratio:     {sharpe:.3f}")
print(f"Sortino ratio:    {sortino:.3f}")
print(f"Profit factor:    {pf:.3f}")
print(f"Win rate:         {wr:.1%}")
print(f"Max drawdown:     {mdd:.2%}")
print(f"Total periods:    {len(period_returns)}")
print(f"Total trades:     {len(deduped)}")
print(f"Avg trades/period: {len(deduped)/len(period_returns):.1f}")

# ── Long vs Short Breakdown ────────────────────────────────────────────
print("\n--- LONG vs SHORT ---")
for direction in ['long', 'short']:
    sub = deduped[deduped['direction'] == direction]
    if len(sub) > 5:
        dir_rets = sub.groupby('week_group')['net_ret'].mean()
        print(f"  {direction.upper():5s}: n={len(sub):4d}, Sharpe={sharpe_ratio(dir_rets):.3f}, "
              f"WR={win_rate(dir_rets):.1%}, PF={profit_factor(dir_rets):.3f}, "
              f"Avg ret={dir_rets.mean():.4f}")

# ── Regime Breakdown ───────────────────────────────────────────────────
print("\n--- BY SIGNAL REGIME ---")
for regime in deduped['regime'].unique():
    sub = deduped[deduped['regime'] == regime]
    if len(sub) > 5:
        reg_rets = sub.groupby('week_group')['net_ret'].mean()
        print(f"  {regime:30s}: n={len(sub):4d}, Sharpe={sharpe_ratio(reg_rets):.3f}, "
              f"WR={win_rate(reg_rets):.1%}")

# ── Per-Sector Breakdown ──────────────────────────────────────────────
print("\n--- BY SECTOR ---")
for ticker in SECTOR_TICKERS:
    sub = deduped[deduped['ticker'] == ticker]
    if len(sub) > 3:
        avg_ret = sub['net_ret'].mean()
        wr_t = win_rate(sub['net_ret'])
        print(f"  {ticker:5s}: n={len(sub):4d}, avg_ret={avg_ret:.4f}, WR={wr_t:.1%}")

# ── Regime Stratification (SPY green/red) ──────────────────────────────
print("\n--- REGIME STRATIFICATION (SPY GREEN vs RED) ---")
# Map each trade to SPY 20d return for regime
period_spy = deduped.groupby('week_group').apply(
    lambda x: spy_20d_ret.loc[x['date'].iloc[0]] if x['date'].iloc[0] in spy_20d_ret.index else np.nan
)

regime_result = regime_stratify(period_returns, period_spy)
print(f"  Green (SPY up):  Sharpe={regime_result['green_sharpe']:.3f}, "
      f"WR={regime_result['green_wr']:.1%}, n={regime_result['green_n']}")
print(f"  Red (SPY down):  Sharpe={regime_result['red_sharpe']:.3f}, "
      f"WR={regime_result['red_wr']:.1%}, n={regime_result['red_n']}")
print(f"  Regime gap:      {regime_result['regime_gap']:.3f}")

# ── Permutation Test ───────────────────────────────────────────────────
print("\n--- PERMUTATION TEST ---")
obs_sharpe, p_value = permutation_test(period_returns, n_iter=PERM_ITERS)
print(f"  Observed Sharpe: {obs_sharpe:.3f}")
print(f"  p-value:         {p_value:.4f}")

# ── Year-by-Year ───────────────────────────────────────────────────────
print("\n--- YEAR-BY-YEAR ---")
deduped_with_year = deduped.copy()
deduped_with_year['year'] = pd.to_datetime(deduped_with_year['date']).dt.year

for year in sorted(deduped_with_year['year'].unique()):
    year_sub = deduped_with_year[deduped_with_year['year'] == year]
    year_rets = year_sub.groupby('week_group')['net_ret'].mean()
    if len(year_rets) > 3:
        yr_sharpe = sharpe_ratio(year_rets)
        yr_ret = (1 + year_rets).prod() - 1
        print(f"  {year}: ret={yr_ret:.2%}, Sharpe={yr_sharpe:.3f}, WR={win_rate(year_rets):.1%}, n_periods={len(year_rets)}")

# ── Signal Strength Analysis ──────────────────────────────────────────
print("\n--- SIGNAL STRENGTH ANALYSIS ---")
strength_median = deduped['signal_strength'].median()
strong = deduped[deduped['signal_strength'] > strength_median]
weak = deduped[deduped['signal_strength'] <= strength_median]

strong_rets = strong.groupby('week_group')['net_ret'].mean()
weak_rets = weak.groupby('week_group')['net_ret'].mean()

print(f"  Strong signals (>{strength_median:.2f}): n={len(strong)}, "
      f"Sharpe={sharpe_ratio(strong_rets):.3f}, WR={win_rate(strong_rets):.1%}")
print(f"  Weak signals  (<={strength_median:.2f}): n={len(weak)}, "
      f"Sharpe={sharpe_ratio(weak_rets):.3f}, WR={win_rate(weak_rets):.1%}")

# ── 5-GATE VALIDATION ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("5-GATE VALIDATION")
print("=" * 70)

gate1 = sharpe >= 0.5
gate2 = regime_result['regime_gap'] < 0.5
gate3 = p_value < 0.05
gate4 = mdd > -0.30  # max DD not worse than -30%
gate5 = len(period_returns) >= 50  # sufficient trade periods

gates = {
    'G1: Sharpe >= 0.5': (gate1, f"Sharpe = {sharpe:.3f}"),
    'G2: Regime gap < 0.5': (gate2, f"Gap = {regime_result['regime_gap']:.3f}"),
    'G3: Perm test p < 0.05': (gate3, f"p = {p_value:.4f}"),
    'G4: Max DD > -30%': (gate4, f"DD = {mdd:.2%}"),
    'G5: >= 50 trade periods': (gate5, f"Periods = {len(period_returns)}"),
}

all_pass = True
for name, (passed, detail) in gates.items():
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name} — {detail}")
    if not passed:
        all_pass = False

print(f"\n  OVERALL: {'ALL GATES PASSED — SIGNAL VALIDATED' if all_pass else 'NOT ALL GATES PASSED'}")

# ── Variation: Strong Signals Only ──────────────────────────────────────
print("\n" + "=" * 70)
print("VARIATION: STRONG SIGNALS ONLY (top 50% signal strength)")
print("=" * 70)

if len(strong_rets) >= 20:
    s_sharpe = sharpe_ratio(strong_rets)
    s_sortino = sortino_ratio(strong_rets)
    s_pf = profit_factor(strong_rets)
    s_wr = win_rate(strong_rets)
    s_equity = (1 + strong_rets).cumprod()
    s_mdd = max_drawdown(s_equity)
    s_total = s_equity.iloc[-1] - 1

    print(f"  Total return:   {s_total:.2%}")
    print(f"  Sharpe:         {s_sharpe:.3f}")
    print(f"  Sortino:        {s_sortino:.3f}")
    print(f"  Profit factor:  {s_pf:.3f}")
    print(f"  Win rate:       {s_wr:.1%}")
    print(f"  Max drawdown:   {s_mdd:.2%}")
    print(f"  Trade periods:  {len(strong_rets)}")

    # Permutation test on strong only
    _, s_pval = permutation_test(strong_rets, n_iter=PERM_ITERS)
    print(f"  Perm test p:    {s_pval:.4f}")

    # Regime stratification
    strong_spy = deduped[deduped['signal_strength'] > strength_median].groupby('week_group').apply(
        lambda x: spy_20d_ret.loc[x['date'].iloc[0]] if x['date'].iloc[0] in spy_20d_ret.index else np.nan
    )
    strong_regime = regime_stratify(strong_rets, strong_spy)
    print(f"  Green Sharpe:   {strong_regime['green_sharpe']:.3f}")
    print(f"  Red Sharpe:     {strong_regime['red_sharpe']:.3f}")
    print(f"  Regime gap:     {strong_regime['regime_gap']:.3f}")

# ── Variation: Stress Transitions Only (exclude neutral) ───────────────
print("\n" + "=" * 70)
print("VARIATION: STRESS TRANSITIONS ONLY (exclude neutral mean-rev)")
print("=" * 70)

stress_only = deduped[deduped['regime'] != 'neutral_meanrev']
if len(stress_only) > 20:
    so_rets = stress_only.groupby('week_group')['net_ret'].mean()
    so_sharpe = sharpe_ratio(so_rets)
    so_wr = win_rate(so_rets)
    so_pf = profit_factor(so_rets)
    so_equity = (1 + so_rets).cumprod()
    so_mdd = max_drawdown(so_equity)

    print(f"  Sharpe:         {so_sharpe:.3f}")
    print(f"  Win rate:       {so_wr:.1%}")
    print(f"  Profit factor:  {so_pf:.3f}")
    print(f"  Max drawdown:   {so_mdd:.2%}")
    print(f"  Trade periods:  {len(so_rets)}")

# ── Save Results ───────────────────────────────────────────────────────
results = {
    'signal_name': 'credit_regime_rotation_v1',
    'description': 'Cross-asset credit stress acceleration for sector ETF rotation',
    'date_run': datetime.now().isoformat(),
    'data_range': f"{close.index[0].date()} to {close.index[-1].date()}",
    'hold_days': HOLD_DAYS,
    'rt_cost_pct': RT_COST_PCT,
    'wf_train_days': WF_TRAIN_DAYS,
    'wf_test_days': WF_TEST_DAYS,
    'metrics': {
        'total_return': round(total_return, 4),
        'annualized_return': round(ann_return, 4),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'profit_factor': round(pf, 3),
        'win_rate': round(wr, 3),
        'max_drawdown': round(mdd, 4),
        'total_periods': int(len(period_returns)),
        'total_trades': int(len(deduped)),
    },
    'permutation_test': {
        'p_value': round(p_value, 4),
        'observed_sharpe': round(obs_sharpe, 3),
    },
    'regime_stratification': regime_result,
    'gates': {name: passed for name, (passed, _) in gates.items()},
    'all_gates_passed': all_pass,
}

results_path = Path('/home/jupiter/Lvl3Quant/research/credit_regime_rotation_v1_results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to {results_path}")
print("\nDONE.")
