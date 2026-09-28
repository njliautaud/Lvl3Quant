#!/usr/bin/env python3
"""
Sector Volatility-Adjusted Relative Value Signal v1

THESIS: Trade PAIRS of sector ETFs based on volatility-adjusted relative
mispricing. When sector A's vol-adjusted return significantly diverges from
sector B's, the spread tends to mean-revert. This is inherently regime-agnostic
because we're market-neutral (long one sector, short another).

KEY INNOVATION: Instead of raw L/S on individual sectors (which is directional
and regime-dependent), we construct SPREAD trades:
- Long the oversold sector (vol-adjusted relative underperformer)
- Short the overbought sector (vol-adjusted relative outperformer)
- The trade is the SPREAD, not the individual legs

WHY THIS SHOULD PASS REGIME GAP:
- Market-neutral (long+short same dollar amount)
- Regime affects BOTH legs similarly (both sector ETFs)
- Edge comes from RELATIVE mispricing, not market direction
- Vol adjustment normalizes for regime changes in volatility

SIGNAL:
1. For each sector pair, compute vol-adjusted relative performance (20d)
2. Z-score the spread vs 60d history
3. When z-score exceeds threshold: fade the spread
4. Filter: only trade pairs with historical mean-reversion (high negative autocorrelation of spread)

Walk-forward, sliding window, 5-gate validation.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path
from itertools import combinations

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Config ──────────────────────────────────────────────────────────────
SECTOR_TICKERS = ['XLF', 'XLE', 'XLU', 'XLK', 'XLY', 'XLP', 'XLRE', 'XLV', 'XLI', 'XLB', 'XLC']
BENCHMARK = 'SPY'
ALL_TICKERS = SECTOR_TICKERS + [BENCHMARK]

HOLD_DAYS = 5
RT_COST_PCT = 0.0006  # 0.06% RT (two legs: long + short = double cost)
PERM_ITERS = 2000

# Walk-forward
WF_TRAIN_DAYS = 252
WF_TEST_DAYS = 21
WF_STEP_DAYS = 21

# Signal params
RETURN_LOOKBACK = 20     # days for relative performance
VOL_LOOKBACK = 20        # realized vol window
ZSCORE_WINDOW = 60       # z-score normalization window
ZSCORE_THRESHOLD = 1.5   # minimum z-score to trigger (optimized in WF)
MIN_AUTOCORR = 0.10      # autocorrelation threshold (less strict - just needs some mean-reversion)
MAX_PAIRS_PER_DAY = 3    # max concurrent pair trades

print("=" * 70)
print("SECTOR VOL-ADJUSTED RELATIVE VALUE v1")
print("Market-Neutral Pair Spreads with Vol Normalization")
print("=" * 70)

# ── Data Download ───────────────────────────────────────────────────────
print("\nDownloading data...")
data = yf.download(ALL_TICKERS, start='2014-01-01', end='2026-08-18', auto_adjust=True)

close = data['Close'].ffill().dropna()
print(f"Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} trading days")

# ── Helper Functions ────────────────────────────────────────────────────
def sharpe_ratio(returns):
    if len(returns) < 10 or returns.std() == 0:
        return 0.0
    return returns.mean() / returns.std() * np.sqrt(252 / HOLD_DAYS)

def sortino_ratio(returns):
    if len(returns) < 10:
        return 0.0
    ds = returns[returns < 0]
    if len(ds) == 0 or ds.std() == 0:
        return float('inf') if returns.mean() > 0 else 0.0
    return returns.mean() / ds.std() * np.sqrt(252 / HOLD_DAYS)

def profit_factor(returns):
    gp = returns[returns > 0].sum()
    gl = abs(returns[returns < 0].sum())
    return gp / gl if gl > 0 else (float('inf') if gp > 0 else 0.0)

def win_rate(returns):
    return (returns > 0).mean() if len(returns) > 0 else 0.0

def max_drawdown(eq):
    peak = eq.expanding().max()
    return ((eq - peak) / peak).min()

def permutation_test(rets, n_iter=2000):
    observed = sharpe_ratio(rets)
    arr = rets.values.copy()
    count = 0
    for _ in range(n_iter):
        np.random.shuffle(arr)
        if sharpe_ratio(pd.Series(arr)) >= observed:
            count += 1
    return observed, count / n_iter

def regime_stratify(returns, spy_ret):
    common = returns.index.intersection(spy_ret.index)
    r, s = returns.loc[common], spy_ret.loc[common]
    green, red = r[s > 0], r[s <= 0]
    g_sh = sharpe_ratio(green) if len(green) > 5 else 0.0
    r_sh = sharpe_ratio(red) if len(red) > 5 else 0.0
    mx = max(abs(g_sh), abs(r_sh), 0.001)
    return {
        'green_sharpe': round(g_sh, 3), 'red_sharpe': round(r_sh, 3),
        'regime_gap': round(abs(g_sh - r_sh) / mx, 3),
        'green_n': int((s > 0).sum()), 'red_n': int((s <= 0).sum()),
        'green_wr': round(win_rate(green), 3), 'red_wr': round(win_rate(red), 3),
    }

# ── Build Pair Spread Features ─────────────────────────────────────────
print("\nBuilding vol-adjusted relative value features...")

# Realized vol for each sector
sector_ret = close[SECTOR_TICKERS].pct_change()
sector_vol = sector_ret.rolling(VOL_LOOKBACK, min_periods=10).std() * np.sqrt(252)

# Vol-adjusted returns (Sharpe-like daily: return / vol)
# Use 20d cumulative return / 20d vol as the "vol-adjusted performance"
sector_20d_ret = close[SECTOR_TICKERS].pct_change(RETURN_LOOKBACK)
vol_adj_ret = sector_20d_ret / sector_vol.replace(0, np.nan)

# For each pair, compute spread
pairs = list(combinations(SECTOR_TICKERS, 2))
print(f"  Total pairs: {len(pairs)}")

# Pre-compute all spread z-scores
spread_z = {}
spread_autocorr = {}

for a, b in pairs:
    spread = vol_adj_ret[a] - vol_adj_ret[b]
    spread_mu = spread.rolling(ZSCORE_WINDOW, min_periods=20).mean()
    spread_std = spread.rolling(ZSCORE_WINDOW, min_periods=20).std()
    z = (spread - spread_mu) / spread_std.replace(0, np.nan)
    spread_z[(a, b)] = z

# SPY returns for regime classification
spy_20d_ret = close[BENCHMARK].pct_change(20)

# Sector forward returns
sector_5d_fwd = close[SECTOR_TICKERS].shift(-HOLD_DAYS) / close[SECTOR_TICKERS] - 1

print(f"  Spread z-scores computed for {len(spread_z)} pairs")

# ── Walk-Forward Backtest ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("WALK-FORWARD BACKTEST (Sliding Window)")
print(f"  Train: {WF_TRAIN_DAYS}d | Test: {WF_TEST_DAYS}d | Step: {WF_STEP_DAYS}d")
print("=" * 70)

start_idx = max(VOL_LOOKBACK, RETURN_LOOKBACK, ZSCORE_WINDOW) + 20 + WF_TRAIN_DAYS
dates = close.index

all_trades = []

for wf_start in range(start_idx, len(dates) - HOLD_DAYS - WF_TEST_DAYS, WF_STEP_DAYS):
    train_end = wf_start
    train_start = wf_start - WF_TRAIN_DAYS
    test_start = wf_start
    test_end = min(wf_start + WF_TEST_DAYS, len(dates) - HOLD_DAYS)

    if test_end <= test_start:
        continue

    train_dates = dates[train_start:train_end]
    test_dates = dates[test_start:test_end]

    # ── TRAIN: Find pairs with mean-reversion and optimal z-threshold ──
    viable_pairs = []

    for a, b in pairs:
        z_train = spread_z[(a, b)].loc[train_dates].dropna()
        if len(z_train) < 60:
            continue

        # Check autocorrelation of spread (negative = mean-reverting)
        spread_train = (vol_adj_ret[a] - vol_adj_ret[b]).loc[train_dates].dropna()
        if len(spread_train) < 60:
            continue

        autocorr = spread_train.autocorr(lag=5)  # 5-day lag autocorrelation
        if np.isnan(autocorr) or autocorr > MIN_AUTOCORR:
            continue  # Not mean-reverting enough (autocorr should be negative or small)

        # Test signal in training: fade extreme z-scores
        # When z > threshold: short A, long B (spread too high, expect reversion)
        # When z < -threshold: long A, short B
        best_z_thresh = None
        best_z_sharpe = -999

        for z_t in [1.0, 1.5, 2.0]:
            trade_rets = []
            for d in train_dates:
                z_val = spread_z[(a, b)].get(d, np.nan)
                if np.isnan(z_val) or abs(z_val) < z_t:
                    continue

                fwd_a = sector_5d_fwd.loc[d, a] if d in sector_5d_fwd.index else np.nan
                fwd_b = sector_5d_fwd.loc[d, b] if d in sector_5d_fwd.index else np.nan
                if np.isnan(fwd_a) or np.isnan(fwd_b):
                    continue

                if z_val > z_t:
                    # Spread too high: short A, long B
                    spread_ret = fwd_b - fwd_a
                else:
                    # Spread too low: long A, short B
                    spread_ret = fwd_a - fwd_b

                trade_rets.append(spread_ret - RT_COST_PCT)

            if len(trade_rets) > 8:
                sr = sharpe_ratio(pd.Series(trade_rets))
                if sr > best_z_sharpe:
                    best_z_sharpe = sr
                    best_z_thresh = z_t

        if best_z_thresh is not None and best_z_sharpe > 0:
            viable_pairs.append({
                'pair': (a, b),
                'autocorr': autocorr,
                'z_thresh': best_z_thresh,
                'train_sharpe': best_z_sharpe,
            })

    if not viable_pairs:
        continue

    # Sort by train Sharpe, take top pairs
    viable_pairs.sort(key=lambda x: -x['train_sharpe'])
    top_pairs = viable_pairs[:10]  # use up to 10 best pairs

    # ── TEST: Apply to OOT ──
    for test_date in test_dates:
        day_candidates = []

        for vp in top_pairs:
            a, b = vp['pair']
            z_t = vp['z_thresh']

            z_val = spread_z[(a, b)].get(test_date, np.nan)
            if np.isnan(z_val) or abs(z_val) < z_t:
                continue

            fwd_a = sector_5d_fwd.loc[test_date, a] if test_date in sector_5d_fwd.index else np.nan
            fwd_b = sector_5d_fwd.loc[test_date, b] if test_date in sector_5d_fwd.index else np.nan
            if np.isnan(fwd_a) or np.isnan(fwd_b):
                continue

            if z_val > z_t:
                # Short A, Long B
                spread_ret = fwd_b - fwd_a
                long_leg = b
                short_leg = a
            else:
                # Long A, Short B
                spread_ret = fwd_a - fwd_b
                long_leg = a
                short_leg = b

            net_ret = spread_ret - RT_COST_PCT

            day_candidates.append({
                'date': test_date,
                'long_leg': long_leg,
                'short_leg': short_leg,
                'pair': f"{a}/{b}",
                'z_score': z_val,
                'z_thresh': z_t,
                'net_ret': net_ret,
                'spread_ret': spread_ret,
                'signal_strength': abs(z_val),
                'autocorr': vp['autocorr'],
                'train_sharpe': vp['train_sharpe'],
            })

        if not day_candidates:
            continue

        # Sort by signal strength, take top N
        day_candidates.sort(key=lambda x: -x['signal_strength'])
        for c in day_candidates[:MAX_PAIRS_PER_DAY]:
            all_trades.append(c)

trades_df = pd.DataFrame(all_trades)
print(f"\nTotal raw pair trades: {len(trades_df)}")

if len(trades_df) < 30:
    print("INSUFFICIENT TRADES")
    exit(1)

# ── Deduplication ──────────────────────────────────────────────────────
trades_df['date_dt'] = pd.to_datetime(trades_df['date'])
trades_df = trades_df.sort_values('date_dt')
trades_df['week_group'] = (trades_df['date_dt'] - trades_df['date_dt'].min()).dt.days // HOLD_DAYS
deduped = trades_df.groupby(['week_group', 'pair']).first().reset_index()
print(f"After deduplication: {len(deduped)} trades")

# ── Portfolio Returns ───────────────────────────────────────────────────
period_returns = deduped.groupby('week_group')['net_ret'].mean()
equity = (1 + period_returns).cumprod()

# ── Core Metrics ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESULTS: SECTOR VOL-ADJUSTED RELATIVE VALUE v1")
print("=" * 70)

n_periods = len(period_returns)
total_return = equity.iloc[-1] - 1
ann_return = (1 + total_return) ** (252 / (HOLD_DAYS * n_periods)) - 1 if n_periods > 0 else 0
sharpe = sharpe_ratio(period_returns)
sortino = sortino_ratio(period_returns)
pf = profit_factor(period_returns)
wr = win_rate(period_returns)
mdd = max_drawdown(equity)

print(f"\nTotal return:      {total_return:.2%}")
print(f"Annualized return: {ann_return:.2%}")
print(f"Sharpe ratio:      {sharpe:.3f}")
print(f"Sortino ratio:     {sortino:.3f}")
print(f"Profit factor:     {pf:.3f}")
print(f"Win rate:          {wr:.1%}")
print(f"Max drawdown:      {mdd:.2%}")
print(f"Total periods:     {n_periods}")
print(f"Total trades:      {len(deduped)}")
print(f"Avg trades/period: {len(deduped)/max(1,n_periods):.1f}")

# ── Top Pairs ──────────────────────────────────────────────────────────
print("\n--- TOP PAIRS BY FREQUENCY ---")
pair_counts = deduped['pair'].value_counts().head(15)
for pair, count in pair_counts.items():
    sub = deduped[deduped['pair'] == pair]
    avg_ret = sub['net_ret'].mean()
    wr_p = win_rate(sub['net_ret'])
    print(f"  {pair:12s}: n={count:3d}, avg_ret={avg_ret:+.4f}, WR={wr_p:.1%}")

# ── Year-by-Year ───────────────────────────────────────────────────────
print("\n--- YEAR-BY-YEAR ---")
deduped_y = deduped.copy()
deduped_y['year'] = pd.to_datetime(deduped_y['date']).dt.year
for year in sorted(deduped_y['year'].unique()):
    ys = deduped_y[deduped_y['year'] == year]
    yr = ys.groupby('week_group')['net_ret'].mean()
    if len(yr) > 1:
        yr_ret = (1 + yr).prod() - 1
        print(f"  {year}: ret={yr_ret:+.2%}, Sharpe={sharpe_ratio(yr):.3f}, WR={win_rate(yr):.1%}, n={len(yr)}")

# ── Regime Stratification ──────────────────────────────────────────────
print("\n--- REGIME STRATIFICATION ---")
period_spy = deduped.groupby('week_group').apply(
    lambda x: spy_20d_ret.loc[x['date'].iloc[0]] if x['date'].iloc[0] in spy_20d_ret.index else np.nan
)
regime_result = regime_stratify(period_returns, period_spy)
print(f"  Green (SPY up):  Sharpe={regime_result['green_sharpe']:.3f}, WR={regime_result['green_wr']:.1%}, n={regime_result['green_n']}")
print(f"  Red (SPY down):  Sharpe={regime_result['red_sharpe']:.3f}, WR={regime_result['red_wr']:.1%}, n={regime_result['red_n']}")
print(f"  Regime gap:      {regime_result['regime_gap']:.3f}")

# ── Permutation Test ───────────────────────────────────────────────────
print("\n--- PERMUTATION TEST ---")
obs_sharpe, p_value = permutation_test(period_returns, n_iter=PERM_ITERS)
print(f"  Observed Sharpe: {obs_sharpe:.3f}")
print(f"  p-value:         {p_value:.4f}")

# ── Signal Strength Analysis ──────────────────────────────────────────
print("\n--- SIGNAL STRENGTH (Z-SCORE MAGNITUDE) ---")
try:
    deduped['z_q'] = pd.qcut(deduped['signal_strength'], 3, labels=['low','med','high'], duplicates='drop')
    for q in ['low', 'med', 'high']:
        sub = deduped[deduped['z_q'] == q]
        if len(sub) > 5:
            q_rets = sub.groupby('week_group')['net_ret'].mean()
            print(f"  {q:6s}: n={len(sub):4d}, Sharpe={sharpe_ratio(q_rets):.3f}, WR={win_rate(q_rets):.1%}")
except:
    print("  Could not compute z-score groups")

# ── 5-GATE VALIDATION ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("5-GATE VALIDATION")
print("=" * 70)

gate1 = sharpe >= 0.5
gate2 = regime_result['regime_gap'] < 0.5
gate3 = p_value < 0.05
gate4 = mdd > -0.30
gate5 = n_periods >= 50

gates = {
    'G1: Sharpe >= 0.5': (gate1, f"Sharpe = {sharpe:.3f}"),
    'G2: Regime gap < 0.5': (gate2, f"Gap = {regime_result['regime_gap']:.3f}"),
    'G3: Perm test p < 0.05': (gate3, f"p = {p_value:.4f}"),
    'G4: Max DD > -30%': (gate4, f"DD = {mdd:.2%}"),
    'G5: >= 50 trade periods': (gate5, f"Periods = {n_periods}"),
}

all_pass = True
for name, (passed, detail) in gates.items():
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name} — {detail}")
    if not passed:
        all_pass = False

print(f"\n  OVERALL: {'ALL GATES PASSED — SIGNAL VALIDATED' if all_pass else 'NOT ALL GATES PASSED'}")

# ── High Z-Score Only ─────────────────────────────────────────────────
if 'z_q' in deduped.columns:
    high_z = deduped[deduped['z_q'] == 'high']
    if len(high_z) > 20:
        print("\n" + "=" * 70)
        print("FILTERED: HIGH Z-SCORE ONLY (top tercile)")
        print("=" * 70)

        hz_rets = high_z.groupby('week_group')['net_ret'].mean()
        hz_eq = (1 + hz_rets).cumprod()
        hz_sharpe = sharpe_ratio(hz_rets)
        hz_mdd = max_drawdown(hz_eq)
        _, hz_pval = permutation_test(hz_rets, n_iter=PERM_ITERS)
        hz_spy = high_z.groupby('week_group').apply(
            lambda x: spy_20d_ret.loc[x['date'].iloc[0]] if x['date'].iloc[0] in spy_20d_ret.index else np.nan
        )
        hz_regime = regime_stratify(hz_rets, hz_spy)

        print(f"  Sharpe:     {hz_sharpe:.3f}")
        print(f"  Sortino:    {sortino_ratio(hz_rets):.3f}")
        print(f"  PF:         {profit_factor(hz_rets):.3f}")
        print(f"  WR:         {win_rate(hz_rets):.1%}")
        print(f"  Max DD:     {hz_mdd:.2%}")
        print(f"  Perm p:     {hz_pval:.4f}")
        print(f"  Regime gap: {hz_regime['regime_gap']:.3f}")
        print(f"  Periods:    {len(hz_rets)}")

# ── Save Results ───────────────────────────────────────────────────────
results = {
    'signal_name': 'sector_vol_relative_value_v1',
    'description': 'Market-neutral sector pair spreads using vol-adjusted relative value',
    'date_run': datetime.now().isoformat(),
    'data_range': f"{close.index[0].date()} to {close.index[-1].date()}",
    'hold_days': HOLD_DAYS,
    'rt_cost_pct': RT_COST_PCT,
    'metrics': {
        'total_return': round(float(total_return), 4),
        'annualized_return': round(float(ann_return), 4),
        'sharpe': round(float(sharpe), 3),
        'sortino': round(float(sortino), 3),
        'profit_factor': round(float(pf), 3),
        'win_rate': round(float(wr), 3),
        'max_drawdown': round(float(mdd), 4),
        'total_periods': int(n_periods),
        'total_trades': int(len(deduped)),
    },
    'permutation_test': {'p_value': round(float(p_value), 4)},
    'regime_stratification': regime_result,
    'gates': {name: bool(passed) for name, (passed, _) in gates.items()},
    'all_gates_passed': bool(all_pass),
}

results_path = Path('/home/jupiter/Lvl3Quant/research/sector_vol_relative_value_v1_results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to {results_path}")
print("\nDONE.")
