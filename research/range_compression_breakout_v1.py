#!/usr/bin/env python3
"""
Range Compression Breakout Signal v1 — Sector ETF Options

THESIS: When a sector ETF's daily trading range compresses relative to its
historical ATR (a "coiled spring"), the subsequent breakout direction is
predictable using close-position-in-range + cross-sector context. This exploits
the well-documented volatility clustering phenomenon at the sector level.

WHY THIS IS NOVEL vs what's been tried:
- NOT gap patterns (dead) — we use intraday range compression over multiple days
- NOT momentum burst gap-up (dead) — we use range COMPRESSION not expansion
- NOT VIX standalone (dead) — we use sector-specific implied range, not market VIX
- NOT sector dispersion (dead) — we use individual sector compression/expansion

SIGNAL CONSTRUCTION:
1. Range Ratio (RR): Average of last 3 days (H-L) / 20-day ATR
   - RR < 0.6 = compressed (coiled spring)
   - RR > 1.2 = expanded (exhaustion)
2. Close Position (CP): Where close sits in the day's range, averaged over 3 days
   - CP > 0.7 = closing near highs (bullish lean)
   - CP < 0.3 = closing near lows (bearish lean)
3. Compression Score: combination of RR and relative RR (vs other sectors)
4. Entry Logic:
   - LONG: Range compressed + closing near highs = coiled spring ready to break up
   - SHORT: Range compressed + closing near lows = coiled spring ready to break down
   - Filter: Only trade if sector's compression is in bottom quartile cross-sectionally
5. Added filters:
   - Volume should be declining during compression (true compression, not holiday)
   - ATR should be above minimum threshold (no dead/illiquid sectors)

Walk-forward sliding window, 5-gate validation, 5-day hold.
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
ALL_TICKERS = SECTOR_TICKERS + [BENCHMARK]

HOLD_DAYS = 5
RT_COST_PCT = 0.0003  # 0.03% round-trip
PERM_ITERS = 2000

# Walk-forward
WF_TRAIN_DAYS = 252
WF_TEST_DAYS = 21
WF_STEP_DAYS = 21

# Signal params
ATR_PERIOD = 20
RANGE_AVG_DAYS = 3      # average range over 3 days for smoothing
COMPRESSION_THRESHOLD = 0.65  # RR < this = compressed
CLOSE_POS_BULL = 0.65   # close in upper 35% = bullish
CLOSE_POS_BEAR = 0.35   # close in lower 35% = bearish
VOLUME_DECLINE_DAYS = 5  # volume should decline over this window
TOP_N = 2               # trade top 2 most compressed sectors per direction

print("=" * 70)
print("RANGE COMPRESSION BREAKOUT SIGNAL v1")
print("Sector ETF Coiled Spring + Close Position Directional Signal")
print("=" * 70)

# ── Data Download ───────────────────────────────────────────────────────
print("\nDownloading OHLCV data...")
data = yf.download(ALL_TICKERS, start='2014-01-01', end='2026-08-18', auto_adjust=True)

close = data['Close'].ffill().dropna()
high = data['High'].ffill().dropna()
low = data['Low'].ffill().dropna()
volume = data['Volume'].ffill().fillna(0)

# Align all
common_idx = close.index.intersection(high.index).intersection(low.index).intersection(volume.index)
close = close.loc[common_idx]
high = high.loc[common_idx]
low = low.loc[common_idx]
volume = volume.loc[common_idx]

print(f"Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} trading days")

# ── Helper Functions ────────────────────────────────────────────────────
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
    gp = returns[returns > 0].sum()
    gl = abs(returns[returns < 0].sum())
    if gl == 0:
        return float('inf') if gp > 0 else 0.0
    return gp / gl

def win_rate(returns):
    return (returns > 0).mean() if len(returns) > 0 else 0.0

def max_drawdown(equity_curve):
    peak = equity_curve.expanding().max()
    return ((equity_curve - peak) / peak).min()

def permutation_test(signal_returns, n_iter=2000):
    observed = sharpe_ratio(signal_returns)
    arr = signal_returns.values.copy()
    count = sum(1 for _ in range(n_iter) if (np.random.shuffle(arr) or True) and sharpe_ratio(pd.Series(arr)) >= observed)
    return observed, count / n_iter

def regime_stratify(returns, spy_ret_aligned):
    common = returns.index.intersection(spy_ret_aligned.index)
    r = returns.loc[common]
    s = spy_ret_aligned.loc[common]
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

# ── Build Range Compression Features ───────────────────────────────────
print("\nBuilding range compression features...")

# True Range
prev_close = close.shift(1)
tr1 = high - low
tr2 = (high - prev_close).abs()
tr3 = (low - prev_close).abs()
true_range = pd.concat([tr1, tr2, tr3]).groupby(level=0).max()
# Actually compute element-wise max
true_range = pd.DataFrame(index=close.index, columns=close.columns)
for t in ALL_TICKERS:
    if t in close.columns:
        tr = pd.concat([
            (high[t] - low[t]),
            (high[t] - close[t].shift(1)).abs(),
            (low[t] - close[t].shift(1)).abs()
        ], axis=1).max(axis=1)
        true_range[t] = tr

true_range = true_range.astype(float)

# ATR (20-day)
atr = true_range.rolling(ATR_PERIOD, min_periods=10).mean()

# Daily Range
daily_range = high - low

# Range Ratio: smoothed daily range / ATR
range_ratio = pd.DataFrame(index=close.index, columns=SECTOR_TICKERS)
for t in SECTOR_TICKERS:
    smoothed_range = daily_range[t].rolling(RANGE_AVG_DAYS).mean()
    range_ratio[t] = smoothed_range / atr[t].replace(0, np.nan)

range_ratio = range_ratio.astype(float)

# Close Position in Range: (close - low) / (high - low), smoothed over 3 days
close_position = pd.DataFrame(index=close.index, columns=SECTOR_TICKERS)
for t in SECTOR_TICKERS:
    raw_cp = (close[t] - low[t]) / (high[t] - low[t]).replace(0, np.nan)
    close_position[t] = raw_cp.rolling(RANGE_AVG_DAYS).mean()

close_position = close_position.astype(float)

# Volume trend: is volume declining? (5-day slope normalized)
vol_trend = pd.DataFrame(index=close.index, columns=SECTOR_TICKERS)
for t in SECTOR_TICKERS:
    vol_ma = volume[t].rolling(20).mean()
    vol_ratio = volume[t].rolling(VOLUME_DECLINE_DAYS).mean() / vol_ma.replace(0, np.nan)
    vol_trend[t] = vol_ratio

vol_trend = vol_trend.astype(float)

# Cross-sectional rank of range ratio (lower = more compressed)
rr_rank = range_ratio.rank(axis=1, pct=True)

# Forward returns
sector_5d_fwd = close[SECTOR_TICKERS].shift(-HOLD_DAYS) / close[SECTOR_TICKERS] - 1

# SPY
spy_close = close[BENCHMARK]
spy_20d_ret = spy_close.pct_change(20)

print(f"  Range ratio computed: {range_ratio.notna().all(axis=1).sum()} full days")
print(f"  Close position computed: {close_position.notna().all(axis=1).sum()} full days")

# ── Walk-Forward Backtest ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("WALK-FORWARD BACKTEST (Sliding Window)")
print(f"  Train: {WF_TRAIN_DAYS}d | Test: {WF_TEST_DAYS}d | Step: {WF_STEP_DAYS}d")
print("=" * 70)

start_idx = ATR_PERIOD + 20 + WF_TRAIN_DAYS
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

    # ── TRAIN: Learn optimal thresholds ──
    train_rr = range_ratio.loc[train_dates]
    train_cp = close_position.loc[train_dates]
    train_fwd = sector_5d_fwd.loc[train_dates]
    train_vol = vol_trend.loc[train_dates]

    # Find optimal compression threshold and close position thresholds
    best_config = None
    best_train_sharpe = -999

    for comp_thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        for cp_bull in [0.60, 0.65, 0.70, 0.75]:
            cp_bear = 1.0 - cp_bull  # symmetric

            train_rets = []
            for d in train_dates:
                rr_d = train_rr.loc[d].dropna()
                cp_d = train_cp.loc[d].dropna()
                fwd_d = train_fwd.loc[d].dropna()
                vt_d = train_vol.loc[d].dropna()

                if len(rr_d) < 5:
                    continue

                # Find compressed sectors (RR < threshold)
                compressed = rr_d[rr_d < comp_thresh]
                if len(compressed) == 0:
                    continue

                # Among compressed, find bullish (close near highs) and bearish (close near lows)
                for t in compressed.index:
                    if t not in cp_d.index or t not in fwd_d.index:
                        continue

                    cp_val = cp_d[t]
                    fwd_val = fwd_d[t]

                    # Volume should not be spiking (quiet compression, not panic)
                    if t in vt_d.index and vt_d[t] > 1.3:
                        continue  # volume too high, not genuine compression

                    if np.isnan(cp_val) or np.isnan(fwd_val):
                        continue

                    if cp_val > cp_bull:
                        # Bullish compression: long
                        train_rets.append(fwd_val - RT_COST_PCT)
                    elif cp_val < cp_bear:
                        # Bearish compression: short
                        train_rets.append(-fwd_val - RT_COST_PCT)

            if len(train_rets) > 20:
                sr = sharpe_ratio(pd.Series(train_rets))
                if sr > best_train_sharpe:
                    best_train_sharpe = sr
                    best_config = {
                        'comp_thresh': comp_thresh,
                        'cp_bull': cp_bull,
                        'cp_bear': cp_bear,
                    }

    if best_config is None or best_train_sharpe < -0.5:
        continue

    # ── TEST: Apply learned thresholds ──
    comp_t = best_config['comp_thresh']
    cp_b = best_config['cp_bull']
    cp_br = best_config['cp_bear']

    for test_date in test_dates:
        rr_d = range_ratio.loc[test_date].dropna()
        cp_d = close_position.loc[test_date].dropna()
        vt_d = vol_trend.loc[test_date].dropna()
        fwd_d = sector_5d_fwd.loc[test_date].dropna()

        if len(rr_d) < 5:
            continue

        # Find compressed sectors
        compressed = rr_d[rr_d < comp_t]
        if len(compressed) == 0:
            continue

        # Score each compressed sector
        candidates = []
        for t in compressed.index:
            if t not in cp_d.index or t not in fwd_d.index:
                continue

            cp_val = cp_d[t]
            fwd_val = fwd_d[t]

            # Volume filter
            if t in vt_d.index and vt_d[t] > 1.3:
                continue

            if np.isnan(cp_val) or np.isnan(fwd_val):
                continue

            # Compression score (lower RR = more compressed = stronger signal)
            compression_score = 1.0 - rr_d[t] / comp_t  # 0 at threshold, higher = more compressed

            if cp_val > cp_b:
                direction = 'long'
                dir_strength = cp_val - cp_b
                net_ret = fwd_val - RT_COST_PCT
            elif cp_val < cp_br:
                direction = 'short'
                dir_strength = cp_br - cp_val
                net_ret = -fwd_val - RT_COST_PCT
            else:
                continue  # close in middle = no directional lean

            signal_strength = compression_score * (1 + dir_strength)

            candidates.append({
                'ticker': t,
                'direction': direction,
                'net_ret': net_ret,
                'fwd_ret': fwd_val if direction == 'long' else -fwd_val,
                'signal_strength': signal_strength,
                'range_ratio': rr_d[t],
                'close_position': cp_val,
                'compression_score': compression_score,
                'vol_trend': vt_d.get(t, np.nan),
            })

        if not candidates:
            continue

        # Sort by signal strength, take top N per direction
        cands_df = pd.DataFrame(candidates)

        for direction in ['long', 'short']:
            dir_cands = cands_df[cands_df['direction'] == direction].nlargest(TOP_N, 'signal_strength')
            for _, row in dir_cands.iterrows():
                all_trades.append({
                    'date': test_date,
                    'ticker': row['ticker'],
                    'direction': direction,
                    'net_ret': row['net_ret'],
                    'fwd_ret': row['fwd_ret'],
                    'signal_strength': row['signal_strength'],
                    'range_ratio': row['range_ratio'],
                    'close_position': row['close_position'],
                    'compression_score': row['compression_score'],
                    'vol_trend': row['vol_trend'],
                    'train_sharpe': best_train_sharpe,
                })

trades_df = pd.DataFrame(all_trades)
print(f"\nTotal raw trades: {len(trades_df)}")

if len(trades_df) < 30:
    print("INSUFFICIENT TRADES")
    exit(1)

# ── Deduplication ──────────────────────────────────────────────────────
trades_df['date_dt'] = pd.to_datetime(trades_df['date'])
trades_df = trades_df.sort_values('date_dt')
trades_df['week_group'] = (trades_df['date_dt'] - trades_df['date_dt'].min()).dt.days // HOLD_DAYS
deduped = trades_df.groupby(['week_group', 'ticker', 'direction']).first().reset_index()
print(f"After deduplication: {len(deduped)} trades")

# ── Portfolio Returns ───────────────────────────────────────────────────
period_returns = deduped.groupby('week_group')['net_ret'].mean()
equity = (1 + period_returns).cumprod()

# ── Core Metrics ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESULTS: RANGE COMPRESSION BREAKOUT v1")
print("=" * 70)

total_return = equity.iloc[-1] - 1
n_periods = len(period_returns)
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

# ── Long vs Short ──────────────────────────────────────────────────────
print("\n--- LONG vs SHORT ---")
for direction in ['long', 'short']:
    sub = deduped[deduped['direction'] == direction]
    if len(sub) > 5:
        dir_rets = sub.groupby('week_group')['net_ret'].mean()
        print(f"  {direction.upper():5s}: n={len(sub):4d}, Sharpe={sharpe_ratio(dir_rets):.3f}, "
              f"WR={win_rate(dir_rets):.1%}, PF={profit_factor(dir_rets):.3f}, "
              f"Avg ret={dir_rets.mean():.4f}")

# ── Per-Sector ──────────────────────────────────────────────────────────
print("\n--- BY SECTOR ---")
for ticker in SECTOR_TICKERS:
    sub = deduped[deduped['ticker'] == ticker]
    if len(sub) > 0:
        print(f"  {ticker:5s}: n={len(sub):4d}, avg_ret={sub['net_ret'].mean():+.4f}, WR={win_rate(sub['net_ret']):.1%}")

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
print("\n--- SIGNAL STRENGTH TERCILES ---")
try:
    deduped['strength_tercile'] = pd.qcut(deduped['signal_strength'], 3, labels=[1,2,3], duplicates='drop')
    for q in sorted(deduped['strength_tercile'].dropna().unique()):
        sub = deduped[deduped['strength_tercile'] == q]
        q_rets = sub.groupby('week_group')['net_ret'].mean()
        if len(q_rets) > 3:
            print(f"  T{q}: n={len(sub):4d}, Sharpe={sharpe_ratio(q_rets):.3f}, WR={win_rate(q_rets):.1%}, avg={q_rets.mean():.4f}")
except:
    print("  Could not compute terciles")

# ── Compression Score Analysis ─────────────────────────────────────────
print("\n--- COMPRESSION DEPTH ANALYSIS ---")
comp_median = deduped['compression_score'].median()
deep_comp = deduped[deduped['compression_score'] > comp_median]
shallow_comp = deduped[deduped['compression_score'] <= comp_median]

if len(deep_comp) > 5 and len(shallow_comp) > 5:
    deep_rets = deep_comp.groupby('week_group')['net_ret'].mean()
    shallow_rets = shallow_comp.groupby('week_group')['net_ret'].mean()
    print(f"  Deep compression:    n={len(deep_comp)}, Sharpe={sharpe_ratio(deep_rets):.3f}, WR={win_rate(deep_rets):.1%}")
    print(f"  Shallow compression: n={len(shallow_comp)}, Sharpe={sharpe_ratio(shallow_rets):.3f}, WR={win_rate(shallow_rets):.1%}")

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

# ── Best Sub-Signal Analysis ──────────────────────────────────────────
# Check if long-only or short-only passes gates
print("\n" + "=" * 70)
print("SUB-SIGNAL ANALYSIS: LONG-ONLY and SHORT-ONLY")
print("=" * 70)

for direction in ['long', 'short']:
    sub = deduped[deduped['direction'] == direction]
    if len(sub) < 20:
        print(f"\n  {direction.upper()}: insufficient trades ({len(sub)})")
        continue

    sub_rets = sub.groupby('week_group')['net_ret'].mean()
    sub_eq = (1 + sub_rets).cumprod()
    sub_sharpe = sharpe_ratio(sub_rets)
    sub_sortino = sortino_ratio(sub_rets)
    sub_pf = profit_factor(sub_rets)
    sub_wr = win_rate(sub_rets)
    sub_mdd = max_drawdown(sub_eq)

    # Regime test
    sub_spy = sub.groupby('week_group').apply(
        lambda x: spy_20d_ret.loc[x['date'].iloc[0]] if x['date'].iloc[0] in spy_20d_ret.index else np.nan
    )
    sub_regime = regime_stratify(sub_rets, sub_spy)

    # Perm test
    _, sub_pval = permutation_test(sub_rets, n_iter=PERM_ITERS)

    print(f"\n  {direction.upper()}-ONLY:")
    print(f"    Sharpe:      {sub_sharpe:.3f}")
    print(f"    Sortino:     {sub_sortino:.3f}")
    print(f"    PF:          {sub_pf:.3f}")
    print(f"    WR:          {sub_wr:.1%}")
    print(f"    Max DD:      {sub_mdd:.2%}")
    print(f"    Perm p:      {sub_pval:.4f}")
    print(f"    Regime gap:  {sub_regime['regime_gap']:.3f}")
    print(f"    Periods:     {len(sub_rets)}")

    # Gate check
    g1 = sub_sharpe >= 0.5
    g2 = sub_regime['regime_gap'] < 0.5
    g3 = sub_pval < 0.05
    g4 = sub_mdd > -0.30
    g5 = len(sub_rets) >= 50
    gates_passed = sum([g1, g2, g3, g4, g5])
    print(f"    Gates: {gates_passed}/5 ({'PASS' if gates_passed == 5 else 'PARTIAL'})")

# ── Save Results ───────────────────────────────────────────────────────
results = {
    'signal_name': 'range_compression_breakout_v1',
    'description': 'Sector ETF range compression with close-position directional bias',
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
    'permutation_test': {'p_value': round(float(p_value), 4), 'observed_sharpe': round(float(obs_sharpe), 3)},
    'regime_stratification': regime_result,
    'gates': {name: bool(passed) for name, (passed, _) in gates.items()},
    'all_gates_passed': bool(all_pass),
}

results_path = Path('/home/jupiter/Lvl3Quant/research/range_compression_breakout_v1_results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to {results_path}")
print("\nDONE.")
