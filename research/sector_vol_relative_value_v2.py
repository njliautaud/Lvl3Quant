#!/usr/bin/env python3
"""
Sector Vol-Adjusted Relative Value Signal v2 — Simplified Cross-Sectional

THESIS: Rank sectors by vol-adjusted relative performance. Buy the most
undervalued (worst vol-adj return) and sell the most overvalued (best vol-adj return).
This is a cross-sectional mean-reversion signal that should be regime-agnostic
because it's always long AND short sectors simultaneously.

SIMPLIFICATION FROM v1:
- No pair-wise computation (too slow, autocorrelation issues)
- Instead: cross-sectional z-score ranking
- Each day, rank all 11 sectors by vol-adjusted 20d return
- Buy bottom 2, sell top 2
- The long/short portfolio is naturally hedged

SIGNAL:
1. Vol-adjusted return = 20d return / 20d realized vol (like a rolling Sharpe)
2. Cross-sectional z-score: how extreme is each sector's vol-adj return vs its peers
3. Trade when dispersion is above median (there IS a spread to capture)
4. Filter: require confirmation from 5d momentum reversal (3d return opposing 20d return)
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

SECTOR_TICKERS = ['XLF', 'XLE', 'XLU', 'XLK', 'XLY', 'XLP', 'XLRE', 'XLV', 'XLI', 'XLB', 'XLC']
BENCHMARK = 'SPY'
ALL_TICKERS = SECTOR_TICKERS + [BENCHMARK]

HOLD_DAYS = 5
RT_COST_PCT = 0.0006  # 2 legs
PERM_ITERS = 2000

WF_TRAIN_DAYS = 252
WF_TEST_DAYS = 21
WF_STEP_DAYS = 21

RETURN_LOOKBACK = 20
VOL_LOOKBACK = 20
TOP_N = 2

print("=" * 70)
print("SECTOR VOL-ADJUSTED RELATIVE VALUE v2 — CROSS-SECTIONAL L/S")
print("=" * 70)

# ── Data ───────────────────────────────────────────────────────────────
print("\nDownloading data...")
data = yf.download(ALL_TICKERS, start='2014-01-01', end='2026-08-18', auto_adjust=True)
close = data['Close'].ffill().dropna()
print(f"Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} trading days")

# ── Helpers ────────────────────────────────────────────────────────────
def sharpe_ratio(r):
    return r.mean() / r.std() * np.sqrt(252/HOLD_DAYS) if len(r) > 10 and r.std() > 0 else 0.0

def sortino_ratio(r):
    if len(r) < 10: return 0.0
    d = r[r < 0]
    return r.mean() / d.std() * np.sqrt(252/HOLD_DAYS) if len(d) > 0 and d.std() > 0 else (float('inf') if r.mean() > 0 else 0.0)

def profit_factor(r):
    gp, gl = r[r > 0].sum(), abs(r[r < 0].sum())
    return gp / gl if gl > 0 else (float('inf') if gp > 0 else 0.0)

def win_rate(r):
    return (r > 0).mean() if len(r) > 0 else 0.0

def max_drawdown(eq):
    return ((eq - eq.expanding().max()) / eq.expanding().max()).min()

def permutation_test(rets, n_iter=2000):
    obs = sharpe_ratio(rets)
    arr = rets.values.copy()
    count = sum(1 for _ in range(n_iter) if (np.random.shuffle(arr) or True) and sharpe_ratio(pd.Series(arr)) >= obs)
    return obs, count / n_iter

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

# ── Features ───────────────────────────────────────────────────────────
print("\nBuilding features...")

sector_ret = close[SECTOR_TICKERS].pct_change()
sector_vol = sector_ret.rolling(VOL_LOOKBACK, min_periods=10).std() * np.sqrt(252)
sector_20d_ret = close[SECTOR_TICKERS].pct_change(RETURN_LOOKBACK)
sector_3d_ret = close[SECTOR_TICKERS].pct_change(3)

# Vol-adjusted return
vol_adj = sector_20d_ret / sector_vol.replace(0, np.nan)

# Cross-sectional z-score (how extreme vs peers)
cs_mean = vol_adj.mean(axis=1)
cs_std = vol_adj.std(axis=1)
cs_z = vol_adj.sub(cs_mean, axis=0).div(cs_std.replace(0, np.nan), axis=0)

# Dispersion: cross-sectional std of vol-adj returns (high = opportunity)
dispersion = cs_std

# Reversal confirmation: 3d return opposite to 20d return direction
reversal = (sector_3d_ret * sector_20d_ret) < 0  # True when reversing

# Forward returns
sector_5d_fwd = close[SECTOR_TICKERS].shift(-HOLD_DAYS) / close[SECTOR_TICKERS] - 1
spy_20d_ret = close[BENCHMARK].pct_change(20)

print(f"  Vol-adj z-scores: {cs_z.notna().all(axis=1).sum()} full days")

# ── Walk-Forward ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("WALK-FORWARD BACKTEST")
print(f"  Train: {WF_TRAIN_DAYS}d | Test: {WF_TEST_DAYS}d | Step: {WF_STEP_DAYS}d")
print("=" * 70)

start_idx = max(RETURN_LOOKBACK, VOL_LOOKBACK) + 20 + WF_TRAIN_DAYS
dates = close.index
all_trades = []

for wf_start in range(start_idx, len(dates) - HOLD_DAYS - WF_TEST_DAYS, WF_STEP_DAYS):
    train_start = wf_start - WF_TRAIN_DAYS
    test_start = wf_start
    test_end = min(wf_start + WF_TEST_DAYS, len(dates) - HOLD_DAYS)
    if test_end <= test_start:
        continue

    train_dates = dates[train_start:wf_start]
    test_dates = dates[test_start:test_end]

    # ── TRAIN: Determine if L/S or L-only works better ──
    # Strategy A: Pure L/S (long bottom 2, short top 2)
    # Strategy B: L/S with reversal confirmation
    # Strategy C: L/S only when dispersion above median

    train_disp = dispersion.loc[train_dates].dropna()
    disp_median = train_disp.median() if len(train_disp) > 20 else 0

    # Evaluate strategy A in training
    train_rets_a = []
    train_rets_b = []

    for d in train_dates:
        z_d = cs_z.loc[d].dropna()
        fwd_d = sector_5d_fwd.loc[d].dropna()
        rev_d = reversal.loc[d]
        disp_d = dispersion.get(d, 0)

        if len(z_d) < 8 or len(fwd_d) < 8:
            continue

        common = z_d.index.intersection(fwd_d.index)
        z_d = z_d[common]
        fwd_d = fwd_d[common]

        # Strategy A: always trade
        longs = z_d.nsmallest(TOP_N).index
        shorts = z_d.nlargest(TOP_N).index
        long_ret = fwd_d[longs].mean()
        short_ret = -fwd_d[shorts].mean()
        spread_ret = (long_ret + short_ret) / 2 - RT_COST_PCT
        train_rets_a.append(spread_ret)

        # Strategy B: only when dispersion high AND reversal
        if disp_d > disp_median:
            # Require at least one long candidate showing reversal
            long_rev = [t for t in longs if t in rev_d.index and rev_d[t]]
            short_rev = [t for t in shorts if t in rev_d.index and rev_d[t]]
            if long_rev or short_rev:
                train_rets_b.append(spread_ret)

    sharpe_a = sharpe_ratio(pd.Series(train_rets_a)) if len(train_rets_a) > 10 else -999
    sharpe_b = sharpe_ratio(pd.Series(train_rets_b)) if len(train_rets_b) > 10 else -999

    # Pick best strategy
    use_dispersion_filter = sharpe_b > sharpe_a and len(train_rets_b) > 15
    use_reversal = sharpe_b > sharpe_a

    # ── TEST ──
    for test_date in test_dates:
        z_d = cs_z.loc[test_date].dropna()
        fwd_d = sector_5d_fwd.loc[test_date].dropna()
        rev_d = reversal.loc[test_date]
        disp_d = dispersion.get(test_date, 0)

        if len(z_d) < 8:
            continue

        common = z_d.index.intersection(fwd_d.index)
        z_d = z_d[common]
        fwd_d = fwd_d[common]

        # Dispersion filter
        if use_dispersion_filter and disp_d <= disp_median:
            continue

        longs = z_d.nsmallest(TOP_N).index.tolist()
        shorts = z_d.nlargest(TOP_N).index.tolist()

        # Reversal filter (optional)
        if use_reversal:
            longs_rev = [t for t in longs if t in rev_d.index and rev_d[t]]
            shorts_rev = [t for t in shorts if t in rev_d.index and rev_d[t]]
            if not longs_rev and not shorts_rev:
                continue
            # Use reversal-confirmed legs if available, otherwise keep original
            if longs_rev:
                longs = longs_rev
            if shorts_rev:
                shorts = shorts_rev

        long_ret = fwd_d[longs].mean()
        short_ret = -fwd_d[shorts].mean()
        spread_ret = (long_ret + short_ret) / 2

        # Signal strength: max absolute z-score among picks
        signal_strength = max(abs(z_d[longs]).max(), abs(z_d[shorts]).max())

        all_trades.append({
            'date': test_date,
            'longs': ','.join(longs),
            'shorts': ','.join(shorts),
            'long_ret': long_ret,
            'short_ret': short_ret,
            'spread_ret': spread_ret,
            'net_ret': spread_ret - RT_COST_PCT,
            'signal_strength': signal_strength,
            'dispersion': disp_d,
            'used_reversal': use_reversal,
            'used_disp_filter': use_dispersion_filter,
        })

trades_df = pd.DataFrame(all_trades)
print(f"\nTotal raw trades: {len(trades_df)}")

if len(trades_df) < 30:
    print("INSUFFICIENT TRADES")
    exit(1)

# ── Dedup ──────────────────────────────────────────────────────────────
trades_df['date_dt'] = pd.to_datetime(trades_df['date'])
trades_df = trades_df.sort_values('date_dt')
trades_df['week_group'] = (trades_df['date_dt'] - trades_df['date_dt'].min()).dt.days // HOLD_DAYS
deduped = trades_df.groupby('week_group').first().reset_index()
print(f"After deduplication: {len(deduped)} trades")

# ── Portfolio Returns ───────────────────────────────────────────────────
period_returns = deduped['net_ret']
period_returns.index = deduped['week_group']
equity = (1 + period_returns).cumprod()

# ── Metrics ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESULTS: SECTOR VOL-ADJUSTED RELATIVE VALUE v2")
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
print(f"Avg ret/period:    {period_returns.mean():.4f}")

# ── Long vs Short leg ──────────────────────────────────────────────────
print("\n--- LONG vs SHORT LEG ---")
long_rets = deduped['long_ret']
short_rets = deduped['short_ret']
print(f"  Long leg:  avg={long_rets.mean():+.4f}, WR={(long_rets > 0).mean():.1%}")
print(f"  Short leg: avg={short_rets.mean():+.4f}, WR={(short_rets > 0).mean():.1%}")

# ── Most frequent sectors on each side ─────────────────────────────────
print("\n--- SECTOR FREQUENCY ---")
from collections import Counter
long_sectors = Counter()
short_sectors = Counter()
for _, row in deduped.iterrows():
    for s in row['longs'].split(','):
        long_sectors[s] += 1
    for s in row['shorts'].split(','):
        short_sectors[s] += 1

print("  Most frequently LONG:")
for s, c in long_sectors.most_common(5):
    print(f"    {s}: {c}")
print("  Most frequently SHORT:")
for s, c in short_sectors.most_common(5):
    print(f"    {s}: {c}")

# ── Year-by-Year ───────────────────────────────────────────────────────
print("\n--- YEAR-BY-YEAR ---")
deduped_y = deduped.copy()
deduped_y['year'] = pd.to_datetime(deduped_y['date']).dt.year
for year in sorted(deduped_y['year'].unique()):
    ys = deduped_y[deduped_y['year'] == year]
    yr = ys['net_ret']
    if len(yr) > 1:
        yr_ret = (1 + yr).prod() - 1
        print(f"  {year}: ret={yr_ret:+.2%}, Sharpe={sharpe_ratio(yr):.3f}, WR={win_rate(yr):.1%}, n={len(yr)}")

# ── Regime Stratification ──────────────────────────────────────────────
print("\n--- REGIME STRATIFICATION ---")
period_spy_vals = []
for _, row in deduped.iterrows():
    d = row['date']
    period_spy_vals.append(spy_20d_ret.loc[d] if d in spy_20d_ret.index else np.nan)
period_spy = pd.Series(period_spy_vals, index=period_returns.index)
regime_result = regime_stratify(period_returns, period_spy)
print(f"  Green (SPY up):  Sharpe={regime_result['green_sharpe']:.3f}, WR={regime_result['green_wr']:.1%}, n={regime_result['green_n']}")
print(f"  Red (SPY down):  Sharpe={regime_result['red_sharpe']:.3f}, WR={regime_result['red_wr']:.1%}, n={regime_result['red_n']}")
print(f"  Regime gap:      {regime_result['regime_gap']:.3f}")

# ── Permutation Test ───────────────────────────────────────────────────
print("\n--- PERMUTATION TEST ---")
obs_sharpe, p_value = permutation_test(period_returns, n_iter=PERM_ITERS)
print(f"  Observed Sharpe: {obs_sharpe:.3f}")
print(f"  p-value:         {p_value:.4f}")

# ── Signal Strength ──────────────────────────────────────────────────
print("\n--- SIGNAL STRENGTH ---")
str_median = deduped['signal_strength'].median()
strong = deduped[deduped['signal_strength'] > str_median]
weak = deduped[deduped['signal_strength'] <= str_median]
if len(strong) > 5 and len(weak) > 5:
    sr_s = sharpe_ratio(strong['net_ret'])
    sr_w = sharpe_ratio(weak['net_ret'])
    print(f"  Strong (>{str_median:.2f}): n={len(strong)}, Sharpe={sr_s:.3f}, WR={win_rate(strong['net_ret']):.1%}")
    print(f"  Weak  (<={str_median:.2f}): n={len(weak)}, Sharpe={sr_w:.3f}, WR={win_rate(weak['net_ret']):.1%}")

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

# ── Strong signals only ───────────────────────────────────────────────
if len(strong) > 30:
    print("\n" + "=" * 70)
    print("FILTERED: STRONG SIGNALS ONLY")
    print("=" * 70)
    s_rets = strong['net_ret'].reset_index(drop=True)
    s_eq = (1 + s_rets).cumprod()
    s_sharpe = sharpe_ratio(s_rets)
    s_mdd = max_drawdown(s_eq)
    _, s_pval = permutation_test(s_rets, n_iter=PERM_ITERS)

    s_spy_vals = []
    for _, row in strong.iterrows():
        d = row['date']
        s_spy_vals.append(spy_20d_ret.loc[d] if d in spy_20d_ret.index else np.nan)
    s_spy = pd.Series(s_spy_vals, index=s_rets.index)
    s_regime = regime_stratify(s_rets, s_spy)

    print(f"  Sharpe:     {s_sharpe:.3f}")
    print(f"  Sortino:    {sortino_ratio(s_rets):.3f}")
    print(f"  PF:         {profit_factor(s_rets):.3f}")
    print(f"  WR:         {win_rate(s_rets):.1%}")
    print(f"  Max DD:     {s_mdd:.2%}")
    print(f"  Perm p:     {s_pval:.4f}")
    print(f"  Regime gap: {s_regime['regime_gap']:.3f}")
    print(f"  Periods:    {len(s_rets)}")
    sg = [s_sharpe >= 0.5, s_regime['regime_gap'] < 0.5, s_pval < 0.05, s_mdd > -0.30, len(s_rets) >= 30]
    print(f"  Gates: {sum(sg)}/5")

# ── Save ───────────────────────────────────────────────────────────────
results = {
    'signal_name': 'sector_vol_relative_value_v2',
    'description': 'Cross-sectional L/S vol-adjusted sector mean reversion',
    'date_run': datetime.now().isoformat(),
    'metrics': {
        'sharpe': round(float(sharpe), 3),
        'sortino': round(float(sortino), 3),
        'profit_factor': round(float(pf), 3),
        'win_rate': round(float(wr), 3),
        'max_drawdown': round(float(mdd), 4),
        'total_return': round(float(total_return), 4),
        'total_periods': int(n_periods),
    },
    'permutation_test': {'p_value': round(float(p_value), 4)},
    'regime_stratification': regime_result,
    'gates': {name: bool(passed) for name, (passed, _) in gates.items()},
    'all_gates_passed': bool(all_pass),
}

results_path = Path('/home/jupiter/Lvl3Quant/research/sector_vol_relative_value_v2_results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to {results_path}")
print("\nDONE.")
