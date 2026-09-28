#!/usr/bin/env python3
"""
Sector Drawdown Recovery Signal v1 — Long-Only Mean Reversion

THESIS: When a sector ETF draws down significantly relative to SPY (excess drawdown),
and then shows the FIRST signs of recovery (positive 3-day return after the trough),
it tends to snap back over the next 5 days. This is a "pain trade" — the oversold
bounce that happens when forced sellers are exhausted.

WHY THIS IS DIFFERENT from dead signals:
- NOT quality mean reversion (which uses RSI/z-score of returns) — we use DRAWDOWN DEPTH
- NOT sector momentum (dead, harmful) — we're explicitly CONTRARIAN on losers
- NOT gap patterns — we use multi-day drawdown dynamics
- NOT sector dispersion — we measure INDIVIDUAL sector drawdown vs SPY, not cross-sector spread
- Key differentiator: we require RECOVERY INITIATION (first uptick), not just oversold level

SIGNAL CONSTRUCTION:
1. Sector Excess Drawdown: sector 20d return minus SPY 20d return
   - Must be < -X% (sector lagging SPY significantly)
2. Recovery Initiation: sector 3-day return turns positive after excess drawdown
   - This catches the "first green candle" after washout
3. Drawdown Depth Ranking: rank sectors by excess drawdown severity
   - Deeper drawdown = stronger potential snap-back (more compressed spring)
4. Filters:
   - Volume surge on recovery day (buyers stepping in)
   - VIX not extreme (avoid catching falling knives in panic)
   - Minimum absolute drawdown threshold (not just relative)
5. Position sizing proportional to drawdown severity

LONG-ONLY. No shorts. 5-day hold. Walk-forward sliding window.
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
VIX_PROXY = ['UVXY']  # VIX proxy via UVXY for longer history
ALL_TICKERS = SECTOR_TICKERS + [BENCHMARK, '^VIX']

HOLD_DAYS = 5
RT_COST_PCT = 0.0003
PERM_ITERS = 2000

# Walk-forward
WF_TRAIN_DAYS = 252
WF_TEST_DAYS = 21
WF_STEP_DAYS = 21

# Signal params
DRAWDOWN_LOOKBACK = 20      # days to measure drawdown
RECOVERY_WINDOW = 3         # days for recovery detection
TOP_N = 2                   # top 2 most oversold with recovery
MIN_EXCESS_DD = -0.03       # minimum 3% excess drawdown vs SPY (starting point, optimized in WF)

print("=" * 70)
print("SECTOR DRAWDOWN RECOVERY SIGNAL v1")
print("Long-Only: Buy sectors recovering from excess drawdowns vs SPY")
print("=" * 70)

# ── Data Download ───────────────────────────────────────────────────────
print("\nDownloading data...")
data = yf.download(ALL_TICKERS, start='2014-01-01', end='2026-08-18', auto_adjust=True)

close = data['Close'].ffill().dropna()
volume = data['Volume'].ffill().fillna(0)

# Handle VIX name
if '^VIX' in close.columns:
    close = close.rename(columns={'^VIX': 'VIX'})
    volume = volume.rename(columns={'^VIX': 'VIX'})

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

# ── Build Features ─────────────────────────────────────────────────────
print("\nBuilding drawdown recovery features...")

# Sector and SPY trailing returns
sector_20d_ret = close[SECTOR_TICKERS].pct_change(DRAWDOWN_LOOKBACK)
spy_20d_ret = close[BENCHMARK].pct_change(DRAWDOWN_LOOKBACK)

# Excess return vs SPY (negative = underperforming)
excess_ret = sector_20d_ret.sub(spy_20d_ret, axis=0)

# 3-day recent return (recovery signal)
sector_3d_ret = close[SECTOR_TICKERS].pct_change(RECOVERY_WINDOW)

# 5-day trailing return (was it recently falling?)
sector_5d_ret = close[SECTOR_TICKERS].pct_change(5)

# Volume ratio: recent vs average (>1 = volume surge)
vol_ratio = pd.DataFrame(index=close.index, columns=SECTOR_TICKERS)
for t in SECTOR_TICKERS:
    vol_ma = volume[t].rolling(20).mean()
    vol_ratio[t] = volume[t].rolling(3).mean() / vol_ma.replace(0, np.nan)
vol_ratio = vol_ratio.astype(float)

# Sector forward returns
sector_5d_fwd = close[SECTOR_TICKERS].shift(-HOLD_DAYS) / close[SECTOR_TICKERS] - 1

# VIX level (for filter)
vix = close['VIX'] if 'VIX' in close.columns else None

# 10d trailing return (for measuring recovery vs trough)
sector_10d_ret = close[SECTOR_TICKERS].pct_change(10)

# Drawdown from 20d high
sector_dd_from_high = pd.DataFrame(index=close.index, columns=SECTOR_TICKERS)
for t in SECTOR_TICKERS:
    rolling_high = close[t].rolling(20).max()
    sector_dd_from_high[t] = close[t] / rolling_high - 1
sector_dd_from_high = sector_dd_from_high.astype(float)

print(f"  Excess returns computed: {excess_ret.notna().all(axis=1).sum()} days")

# ── Walk-Forward Backtest ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("WALK-FORWARD BACKTEST (Sliding Window, Long-Only)")
print(f"  Train: {WF_TRAIN_DAYS}d | Test: {WF_TEST_DAYS}d | Step: {WF_STEP_DAYS}d")
print("=" * 70)

start_idx = DRAWDOWN_LOOKBACK + 20 + WF_TRAIN_DAYS
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

    # ── TRAIN: Learn optimal excess drawdown threshold ──
    best_config = None
    best_sharpe = -999

    for dd_thresh in [-0.03, -0.05, -0.08]:
        for recovery_min in [0.002, 0.005]:
            for require_dd_from_high in [False]:

                train_rets = []
                for d in train_dates:
                    ex_d = excess_ret.loc[d].dropna()
                    rec_d = sector_3d_ret.loc[d].dropna()
                    fwd_d = sector_5d_fwd.loc[d].dropna()
                    dd_high_d = sector_dd_from_high.loc[d].dropna()

                    # Find sectors with excess drawdown
                    oversold = ex_d[ex_d < dd_thresh]
                    if len(oversold) == 0:
                        continue

                    # Filter: must be showing recovery (positive 3d return)
                    recovering = []
                    for t in oversold.index:
                        if t not in rec_d.index or t not in fwd_d.index:
                            continue
                        if rec_d[t] < recovery_min:
                            continue  # not recovering yet

                        # Optional: must also be in drawdown from 20d high
                        if require_dd_from_high and t in dd_high_d.index:
                            if dd_high_d[t] > -0.02:  # not meaningfully below high
                                continue

                        recovering.append((t, ex_d[t], rec_d[t]))

                    if not recovering:
                        continue

                    # Sort by excess drawdown severity (most oversold first)
                    recovering.sort(key=lambda x: x[1])
                    picks = recovering[:TOP_N]

                    for t, excess, rec in picks:
                        train_rets.append(fwd_d[t] - RT_COST_PCT)

                if len(train_rets) > 15:
                    sr = sharpe_ratio(pd.Series(train_rets))
                    if sr > best_sharpe:
                        best_sharpe = sr
                        best_config = {
                            'dd_thresh': dd_thresh,
                            'recovery_min': recovery_min,
                            'require_dd_from_high': require_dd_from_high,
                        }

    if best_config is None or best_sharpe < -0.3:
        continue

    # ── TEST: Apply learned thresholds ──
    dd_t = best_config['dd_thresh']
    rec_min = best_config['recovery_min']
    req_ddh = best_config['require_dd_from_high']

    for test_date in test_dates:
        ex_d = excess_ret.loc[test_date].dropna()
        rec_d = sector_3d_ret.loc[test_date].dropna()
        fwd_d = sector_5d_fwd.loc[test_date].dropna()
        dd_high_d = sector_dd_from_high.loc[test_date].dropna()
        vol_d = vol_ratio.loc[test_date].dropna()

        # VIX filter: skip extreme panic (VIX > 40)
        if vix is not None and test_date in vix.index:
            if vix.loc[test_date] > 40:
                continue

        oversold = ex_d[ex_d < dd_t]
        if len(oversold) == 0:
            continue

        candidates = []
        for t in oversold.index:
            if t not in rec_d.index or t not in fwd_d.index:
                continue

            # Recovery check
            if rec_d[t] < rec_min:
                continue

            # Drawdown from high check
            if req_ddh and t in dd_high_d.index:
                if dd_high_d[t] > -0.02:
                    continue

            # Signal strength: deeper drawdown + stronger recovery = better
            excess_depth = abs(ex_d[t])
            recovery_strength = rec_d[t]
            signal_strength = excess_depth * 10 + recovery_strength * 20

            # Volume confirmation bonus
            vol_bonus = 0
            if t in vol_d.index and vol_d[t] > 1.1:
                vol_bonus = 0.2  # volume picking up on recovery
                signal_strength += vol_bonus

            candidates.append({
                'ticker': t,
                'net_ret': fwd_d[t] - RT_COST_PCT,
                'fwd_ret': fwd_d[t],
                'signal_strength': signal_strength,
                'excess_dd': ex_d[t],
                'recovery_3d': rec_d[t],
                'dd_from_high': dd_high_d.get(t, np.nan),
                'vol_ratio': vol_d.get(t, np.nan),
            })

        if not candidates:
            continue

        # Sort by signal strength, take top N
        candidates.sort(key=lambda x: -x['signal_strength'])
        for c in candidates[:TOP_N]:
            all_trades.append({
                'date': test_date,
                **c,
                'train_sharpe': best_sharpe,
                'dd_thresh': dd_t,
                'recovery_min': rec_min,
            })

trades_df = pd.DataFrame(all_trades)
print(f"\nTotal raw trades: {len(trades_df)}")

if len(trades_df) < 30:
    print("INSUFFICIENT TRADES — adjusting...")
    exit(1)

# ── Deduplication ──────────────────────────────────────────────────────
trades_df['date_dt'] = pd.to_datetime(trades_df['date'])
trades_df = trades_df.sort_values('date_dt')
trades_df['week_group'] = (trades_df['date_dt'] - trades_df['date_dt'].min()).dt.days // HOLD_DAYS
deduped = trades_df.groupby(['week_group', 'ticker']).first().reset_index()
print(f"After deduplication: {len(deduped)} trades")

# ── Portfolio Returns ───────────────────────────────────────────────────
period_returns = deduped.groupby('week_group')['net_ret'].mean()
equity = (1 + period_returns).cumprod()

# ── Core Metrics ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESULTS: SECTOR DRAWDOWN RECOVERY v1 (Long-Only)")
print("=" * 70)

n_periods = len(period_returns)
total_return = equity.iloc[-1] - 1
ann_return = (1 + total_return) ** (252 / (HOLD_DAYS * n_periods)) - 1 if n_periods > 0 else 0
sharpe = sharpe_ratio(period_returns)
sortino = sortino_ratio(period_returns)
pf = profit_factor(period_returns)
wr = win_rate(period_returns)
mdd = max_drawdown(equity)
avg_ret = period_returns.mean()
avg_win = period_returns[period_returns > 0].mean() if (period_returns > 0).any() else 0
avg_loss = period_returns[period_returns < 0].mean() if (period_returns < 0).any() else 0

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
print(f"Avg return/period: {avg_ret:.4f}")
print(f"Avg winner:        {avg_win:.4f}")
print(f"Avg loser:         {avg_loss:.4f}")
print(f"Win/Loss ratio:    {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "Win/Loss: inf")

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
spy_20d = close[BENCHMARK].pct_change(20)
period_spy = deduped.groupby('week_group').apply(
    lambda x: spy_20d.loc[x['date'].iloc[0]] if x['date'].iloc[0] in spy_20d.index else np.nan
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
print("\n--- SIGNAL STRENGTH ANALYSIS ---")
try:
    deduped['strength_q'] = pd.qcut(deduped['signal_strength'], 3, labels=['weak','med','strong'], duplicates='drop')
    for q in ['weak', 'med', 'strong']:
        sub = deduped[deduped['strength_q'] == q]
        if len(sub) > 5:
            q_rets = sub.groupby('week_group')['net_ret'].mean()
            print(f"  {q:8s}: n={len(sub):4d}, Sharpe={sharpe_ratio(q_rets):.3f}, WR={win_rate(q_rets):.1%}")
except:
    print("  Could not compute strength groups")

# ── Drawdown Depth Analysis ──────────────────────────────────────────
print("\n--- DRAWDOWN DEPTH ANALYSIS ---")
dd_median = deduped['excess_dd'].median()
deep = deduped[deduped['excess_dd'] < dd_median]
shallow = deduped[deduped['excess_dd'] >= dd_median]
if len(deep) > 5 and len(shallow) > 5:
    deep_rets = deep.groupby('week_group')['net_ret'].mean()
    shallow_rets = shallow.groupby('week_group')['net_ret'].mean()
    print(f"  Deep DD (< {dd_median:.3f}):    n={len(deep)}, Sharpe={sharpe_ratio(deep_rets):.3f}, WR={win_rate(deep_rets):.1%}")
    print(f"  Shallow DD (>= {dd_median:.3f}): n={len(shallow)}, Sharpe={sharpe_ratio(shallow_rets):.3f}, WR={win_rate(shallow_rets):.1%}")

# ── Recovery Strength Analysis ─────────────────────────────────────────
print("\n--- RECOVERY STRENGTH ANALYSIS ---")
rec_median = deduped['recovery_3d'].median()
strong_rec = deduped[deduped['recovery_3d'] > rec_median]
weak_rec = deduped[deduped['recovery_3d'] <= rec_median]
if len(strong_rec) > 5 and len(weak_rec) > 5:
    sr_rets = strong_rec.groupby('week_group')['net_ret'].mean()
    wr_rets = weak_rec.groupby('week_group')['net_ret'].mean()
    print(f"  Strong recovery: n={len(strong_rec)}, Sharpe={sharpe_ratio(sr_rets):.3f}, WR={win_rate(sr_rets):.1%}")
    print(f"  Weak recovery:   n={len(weak_rec)}, Sharpe={sharpe_ratio(wr_rets):.3f}, WR={win_rate(wr_rets):.1%}")

# ── Consecutive Analysis ──────────────────────────────────────────────
print("\n--- STREAK ANALYSIS ---")
wins = (period_returns > 0).astype(int)
streak_changes = wins.diff().ne(0).cumsum()
streaks = wins.groupby(streak_changes).agg(['sum', 'count'])
win_streaks = streaks[streaks['sum'] == streaks['count']]['count']
loss_streaks = streaks[streaks['sum'] == 0]['count']
if len(win_streaks) > 0:
    print(f"  Max win streak:  {win_streaks.max()}, Avg: {win_streaks.mean():.1f}")
if len(loss_streaks) > 0:
    print(f"  Max loss streak: {loss_streaks.max()}, Avg: {loss_streaks.mean():.1f}")

# ── Monthly Returns (last 12) ─────────────────────────────────────────
print("\n--- RECENT MONTHLY RETURNS ---")
deduped_m = deduped.copy()
deduped_m['month'] = pd.to_datetime(deduped_m['date']).dt.to_period('M')
monthly = deduped_m.groupby('month')['net_ret'].mean()
for m in monthly.index[-12:]:
    print(f"  {m}: {monthly.loc[m]:+.4f}")

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

# ── Filtered version: strong signals only ─────────────────────────────
if 'strength_q' in deduped.columns:
    strong_only = deduped[deduped['strength_q'] == 'strong']
    if len(strong_only) > 20:
        print("\n" + "=" * 70)
        print("FILTERED: STRONG SIGNALS ONLY")
        print("=" * 70)
        so_rets = strong_only.groupby('week_group')['net_ret'].mean()
        so_eq = (1 + so_rets).cumprod()
        so_sharpe = sharpe_ratio(so_rets)
        so_sortino = sortino_ratio(so_rets)
        so_wr = win_rate(so_rets)
        so_pf = profit_factor(so_rets)
        so_mdd = max_drawdown(so_eq)
        _, so_pval = permutation_test(so_rets, n_iter=PERM_ITERS)

        so_spy = strong_only.groupby('week_group').apply(
            lambda x: spy_20d.loc[x['date'].iloc[0]] if x['date'].iloc[0] in spy_20d.index else np.nan
        )
        so_regime = regime_stratify(so_rets, so_spy)

        print(f"  Sharpe:     {so_sharpe:.3f}")
        print(f"  Sortino:    {so_sortino:.3f}")
        print(f"  PF:         {so_pf:.3f}")
        print(f"  WR:         {so_wr:.1%}")
        print(f"  Max DD:     {so_mdd:.2%}")
        print(f"  Perm p:     {so_pval:.4f}")
        print(f"  Regime gap: {so_regime['regime_gap']:.3f}")
        print(f"  Periods:    {len(so_rets)}")

        # Gate check
        sg = [so_sharpe >= 0.5, so_regime['regime_gap'] < 0.5, so_pval < 0.05, so_mdd > -0.30, len(so_rets) >= 30]
        print(f"  Gates: {sum(sg)}/5")

# ── vs Buy-and-Hold SPY ──────────────────────────────────────────────
print("\n--- vs BUY-AND-HOLD SPY ---")
first_date = pd.to_datetime(deduped['date'].min())
last_date = pd.to_datetime(deduped['date'].max())
spy_bah = close[BENCHMARK].loc[last_date] / close[BENCHMARK].loc[first_date] - 1
spy_days = (last_date - first_date).days
spy_ann = (1 + spy_bah) ** (365 / max(1, spy_days)) - 1
print(f"  Signal:  total={total_return:+.2%}, ann={ann_return:+.2%}")
print(f"  SPY B&H: total={spy_bah:+.2%}, ann={spy_ann:+.2%}")

# ── Save Results ───────────────────────────────────────────────────────
results = {
    'signal_name': 'sector_drawdown_recovery_v1',
    'description': 'Long-only: buy sectors recovering from excess drawdowns vs SPY',
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

results_path = Path('/home/jupiter/Lvl3Quant/research/sector_drawdown_recovery_v1_results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to {results_path}")
print("\nDONE.")
