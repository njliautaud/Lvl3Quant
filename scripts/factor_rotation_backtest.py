#!/usr/bin/env python3
"""
Factor/Style Rotation Backtest — 6 Variants
Walk-forward OOT: Jan 2022 – Jul 2026
5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, ≥20 trades
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Config ──────────────────────────────────────────────────────────────
TICKERS = ['QQQ', 'VUG', 'VTV', 'IWD', 'IWM', 'MTUM', 'QUAL', 'USMV', 'GLD', 'TLT', 'SPY']
OOT_START = '2022-01-01'
OOT_END = '2026-07-30'
DATA_START = '2019-01-01'  # extra history for lookback windows
SLIPPAGE = 0.0002  # 0.02%
RISK_FREE_ANNUAL = 0.04
PERM_ITERS = 500
RESULTS_PATH = '/home/jupiter/Lvl3Quant/data/factor_rotation_results.json'

# ── Data Download ───────────────────────────────────────────────────────
print("Downloading data...")
data = yf.download(TICKERS, start=DATA_START, end=OOT_END, auto_adjust=True, progress=False)

# Handle multi-level columns from yfinance
if isinstance(data.columns, pd.MultiIndex):
    close = data['Close'].copy()
else:
    close = data.copy()

# Forward fill then drop any remaining NaN rows at the start
close = close.ffill().dropna()

# Daily returns
returns = close.pct_change().dropna()

# SPY 200-day SMA for regime
spy_sma200 = close['SPY'].rolling(200).mean()
regime_bull = close['SPY'] > spy_sma200  # True = bull

# OOT mask
oot_mask = returns.index >= OOT_START

# Monthly rebalance dates (first trading day of each month in OOT)
oot_dates = returns.index[oot_mask]
monthly_dates = oot_dates.to_series().groupby([oot_dates.year, oot_dates.month]).first().values
monthly_dates = pd.DatetimeIndex(monthly_dates)

print(f"Data: {close.index[0].date()} to {close.index[-1].date()}")
print(f"OOT: {oot_dates[0].date()} to {oot_dates[-1].date()}")
print(f"Monthly rebalance dates: {len(monthly_dates)}")
regime_bull_oot = regime_bull.reindex(oot_dates).fillna(False)
print(f"Regime: {regime_bull_oot.mean():.1%} bull days in OOT")
print()

# ── Helper Functions ────────────────────────────────────────────────────

def calc_risk_adj_momentum(ticker, date, lookback_months=3):
    """Risk-adjusted momentum = return / volatility over lookback period."""
    lb_days = lookback_months * 21
    idx = returns.index.get_loc(date)
    if idx < lb_days:
        return -999
    r = returns[ticker].iloc[idx - lb_days:idx]
    vol = r.std() * np.sqrt(252)
    if vol < 1e-8:
        return 0
    total_ret = (1 + r).prod() - 1
    return total_ret / vol


def calc_momentum(ticker, date, lookback_months=1):
    """Simple momentum = total return over lookback."""
    lb_days = lookback_months * 21
    idx = returns.index.get_loc(date)
    if idx < lb_days:
        return -999
    r = returns[ticker].iloc[idx - lb_days:idx]
    return (1 + r).prod() - 1


def simulate_strategy(signal_func, name):
    """
    Run a monthly-rebalance strategy.
    signal_func(date) -> ticker to hold for next month.
    Returns daily return series over OOT.
    """
    # Build position series
    holdings = {}
    for i, reb_date in enumerate(monthly_dates):
        chosen = signal_func(reb_date)
        if i + 1 < len(monthly_dates):
            end_date = monthly_dates[i + 1]
        else:
            end_date = oot_dates[-1] + pd.Timedelta(days=1)
        # All days from reb_date (inclusive) to next reb_date (exclusive)
        mask = (returns.index >= reb_date) & (returns.index < end_date)
        for d in returns.index[mask]:
            holdings[d] = chosen

    # Calculate daily returns with slippage on rebalance days
    strat_returns = []
    prev_holding = None
    for d in oot_dates:
        if d not in holdings:
            strat_returns.append(0.0)
            continue
        ticker = holdings[d]
        day_ret = returns.loc[d, ticker]
        # Apply slippage on rebalance (position change)
        if ticker != prev_holding:
            day_ret -= SLIPPAGE
        strat_returns.append(day_ret)
        prev_holding = ticker

    return pd.Series(strat_returns, index=oot_dates, name=name)


def calc_metrics(strat_rets, qqq_rets):
    """Calculate all required metrics."""
    if len(strat_rets) == 0 or strat_rets.std() == 0:
        return None

    # Sharpe
    sharpe = strat_rets.mean() / strat_rets.std() * np.sqrt(252)

    # Sortino
    downside = strat_rets[strat_rets < 0].std() * np.sqrt(252)
    sortino = strat_rets.mean() * 252 / downside if downside > 0 else 0

    # Total return
    total_ret = (1 + strat_rets).prod() - 1

    # Max drawdown
    cum = (1 + strat_rets).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    # Win rate (monthly)
    monthly_rets = strat_rets.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    wr = (monthly_rets > 0).mean()

    # Profit factor
    gross_profit = strat_rets[strat_rets > 0].sum()
    gross_loss = abs(strat_rets[strat_rets < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else 999

    # Number of trades (position changes)
    n_trades = len(monthly_rets)  # monthly rebalances

    # Bull/bear Sharpe
    bull_mask = regime_bull.reindex(strat_rets.index).fillna(False)
    bear_mask = ~bull_mask

    bull_rets = strat_rets[bull_mask]
    bear_rets = strat_rets[bear_mask]

    bull_sharpe = bull_rets.mean() / bull_rets.std() * np.sqrt(252) if len(bull_rets) > 20 and bull_rets.std() > 0 else 0
    bear_sharpe = bear_rets.mean() / bear_rets.std() * np.sqrt(252) if len(bear_rets) > 20 and bear_rets.std() > 0 else 0

    # Regime gap
    denom = max(abs(bull_sharpe), abs(bear_sharpe), 1e-8)
    regime_gap = abs(bull_sharpe - bear_sharpe) / denom

    # QQQ correlation
    aligned = pd.concat([strat_rets, qqq_rets], axis=1).dropna()
    qqq_corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])

    # Annual return
    n_years = len(strat_rets) / 252
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0

    return {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'profit_factor': round(pf, 3),
        'win_rate': round(wr, 3),
        'max_dd': round(max_dd, 3),
        'total_return': round(total_ret, 3),
        'annual_return': round(ann_ret, 3),
        'n_trades': int(n_trades),
        'bull_sharpe': round(bull_sharpe, 3),
        'bear_sharpe': round(bear_sharpe, 3),
        'regime_gap': round(regime_gap, 3),
        'qqq_corr': round(qqq_corr, 3),
    }


def permutation_test(strat_rets, n_iter=PERM_ITERS):
    """
    Shuffle entry timing: randomly permute which monthly return goes with which month.
    If random timing gives same Sharpe, there's no timing alpha.
    """
    actual_sharpe = strat_rets.mean() / strat_rets.std() * np.sqrt(252)

    # Resample to monthly returns for shuffling
    monthly_rets = strat_rets.resample('ME').apply(lambda x: (1 + x).prod() - 1).values

    count_ge = 0
    for _ in range(n_iter):
        shuffled = np.random.permutation(monthly_rets)
        # Reconstruct daily-equivalent Sharpe from shuffled monthly
        shuf_mean = np.mean(shuffled)
        shuf_std = np.std(shuffled)
        if shuf_std > 0:
            # Monthly Sharpe scaled to annual
            shuf_sharpe = shuf_mean / shuf_std * np.sqrt(12)
        else:
            shuf_sharpe = 0
        if shuf_sharpe >= actual_sharpe:
            count_ge += 1

    p_value = count_ge / n_iter
    return round(p_value, 4)


def check_gates(metrics, p_value):
    """Check all 5 gates."""
    gates = {
        'sharpe_gt_0.5': metrics['sharpe'] > 0.5,
        'perm_p_lt_0.05': p_value < 0.05,
        'regime_gap_lt_0.5': metrics['regime_gap'] < 0.5,
        'max_dd_gt_neg50': metrics['max_dd'] > -0.50,
        'trades_ge_20': metrics['n_trades'] >= 20,
    }
    return gates


# ── Strategy Signal Functions ───────────────────────────────────────────

# A) Simple Momentum Rotation
FACTOR_POOL_A = ['QQQ', 'VTV', 'IWM', 'MTUM', 'QUAL', 'USMV']
def signal_a(date):
    scores = {t: calc_risk_adj_momentum(t, date, 3) for t in FACTOR_POOL_A}
    return max(scores, key=scores.get)

# B) Regime-Adaptive Factor
BULL_POOL_B = ['QQQ', 'MTUM', 'VTV']
BEAR_POOL_B = ['USMV', 'GLD', 'TLT']
def signal_b(date):
    is_bull = regime_bull.get(date, True)
    pool = BULL_POOL_B if is_bull else BEAR_POOL_B
    scores = {t: calc_risk_adj_momentum(t, date, 3) for t in pool}
    return max(scores, key=scores.get)

# C) Factor Momentum (Cross-Sectional) — 1-month return, top-1
FACTOR_POOL_C = ['QQQ', 'VTV', 'IWM', 'MTUM', 'QUAL', 'USMV']
def signal_c(date):
    scores = {t: calc_momentum(t, date, 1) for t in FACTOR_POOL_C}
    return max(scores, key=scores.get)

# D) Dual Momentum Factor — 6-month return > risk-free, else GLD
FACTOR_POOL_D = ['QQQ', 'VTV', 'IWM', 'MTUM', 'QUAL', 'USMV']
def signal_d(date):
    rf_6m = (1 + RISK_FREE_ANNUAL) ** 0.5 - 1  # 6-month risk-free return
    candidates = {}
    for t in FACTOR_POOL_D:
        mom = calc_momentum(t, date, 6)
        if mom > rf_6m:
            ram = calc_risk_adj_momentum(t, date, 6)
            candidates[t] = ram
    if not candidates:
        return 'GLD'
    return max(candidates, key=candidates.get)

# E) Mean Reversion Factor — worst 1-month return
FACTOR_POOL_E = ['QQQ', 'VTV', 'IWM', 'MTUM', 'QUAL', 'USMV']
def signal_e(date):
    scores = {t: calc_momentum(t, date, 1) for t in FACTOR_POOL_E}
    return min(scores, key=scores.get)

# F) Value-Growth Timing — VTV/QQQ ratio SMA crossover
def signal_f(date):
    idx = close.index.get_loc(date)
    if idx < 50:
        return 'QQQ'
    ratio = close['VTV'].iloc[:idx+1] / close['QQQ'].iloc[:idx+1]
    sma20 = ratio.rolling(20).mean()
    sma50 = ratio.rolling(50).mean()
    if sma20.iloc[-1] > sma50.iloc[-1]:
        return 'VTV'
    else:
        return 'QQQ'


# ── Run All Variants ────────────────────────────────────────────────────

variants = {
    'A_Simple_Momentum': signal_a,
    'B_Regime_Adaptive': signal_b,
    'C_Factor_Momentum_XS': signal_c,
    'D_Dual_Momentum': signal_d,
    'E_Mean_Reversion': signal_e,
    'F_Value_Growth_Timing': signal_f,
}

# QQQ buy-and-hold returns for correlation
qqq_oot = returns.loc[oot_mask, 'QQQ']
spy_oot = returns.loc[oot_mask, 'SPY']

results = {}

print("=" * 100)
print("FACTOR/STYLE ROTATION BACKTEST — 6 VARIANTS")
print(f"OOT: {OOT_START} to {OOT_END} | Slippage: {SLIPPAGE:.2%} | Permutation iters: {PERM_ITERS}")
print("=" * 100)
print()

for name, signal_func in variants.items():
    print(f"Running {name}...")
    strat_rets = simulate_strategy(signal_func, name)
    metrics = calc_metrics(strat_rets, qqq_oot)

    if metrics is None:
        print(f"  SKIP — no valid returns")
        results[name] = {'status': 'SKIP', 'reason': 'no valid returns'}
        continue

    # Permutation test
    p_value = permutation_test(strat_rets)
    metrics['perm_p_value'] = p_value

    # Gate check
    gates = check_gates(metrics, p_value)
    all_pass = all(gates.values())
    metrics['gates'] = gates
    metrics['all_gates_pass'] = all_pass
    metrics['status'] = 'PASS' if all_pass else 'FAIL'

    failed_gates = [g for g, v in gates.items() if not v]
    metrics['failed_gates'] = failed_gates

    results[name] = metrics

    status = "PASS ✓" if all_pass else f"FAIL ✗ ({', '.join(failed_gates)})"
    print(f"  Sharpe={metrics['sharpe']:.2f} Sortino={metrics['sortino']:.2f} "
          f"PF={metrics['profit_factor']:.2f} WR={metrics['win_rate']:.1%} "
          f"MaxDD={metrics['max_dd']:.1%} Ret={metrics['total_return']:.1%} "
          f"Trades={metrics['n_trades']} p={p_value:.3f} "
          f"RegGap={metrics['regime_gap']:.2f} QQQ_r={metrics['qqq_corr']:.2f} "
          f"→ {status}")
    print()


# ── Benchmarks ──────────────────────────────────────────────────────────
print("\n--- BENCHMARKS ---")
for bench_name, bench_ticker in [('SPY_BuyHold', 'SPY'), ('QQQ_BuyHold', 'QQQ')]:
    bench_rets = returns.loc[oot_mask, bench_ticker]
    bm = calc_metrics(bench_rets, qqq_oot)
    if bm:
        print(f"  {bench_name}: Sharpe={bm['sharpe']:.2f} Sortino={bm['sortino']:.2f} "
              f"MaxDD={bm['max_dd']:.1%} Ret={bm['total_return']:.1%}")
        results[bench_name] = bm


# ── Summary Table ───────────────────────────────────────────────────────
print("\n" + "=" * 100)
print("SUMMARY TABLE")
print("=" * 100)
header = f"{'Variant':<25} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR':>6} {'MaxDD':>7} {'Ret':>8} {'Trades':>7} {'p-val':>6} {'RegGap':>7} {'QQQ_r':>6} {'Status':>8}"
print(header)
print("-" * 100)

for name, m in results.items():
    if isinstance(m, dict) and 'sharpe' in m:
        status = m.get('status', '-')
        p_val = m.get('perm_p_value', '-')
        p_str = f"{p_val:.3f}" if isinstance(p_val, float) else '-'
        rg = m.get('regime_gap', '-')
        rg_str = f"{rg:.2f}" if isinstance(rg, float) else '-'
        print(f"{name:<25} {m['sharpe']:>7.2f} {m['sortino']:>8.2f} {m['profit_factor']:>6.2f} "
              f"{m['win_rate']:>5.1%} {m['max_dd']:>6.1%} {m['total_return']:>7.1%} "
              f"{m['n_trades']:>7} {p_str:>6} {rg_str:>7} {m['qqq_corr']:>6.2f} {status:>8}")

# ── Passing Strategies ──────────────────────────────────────────────────
print("\n" + "=" * 100)
passing = {k: v for k, v in results.items() if isinstance(v, dict) and v.get('all_gates_pass')}
if passing:
    print(f"STRATEGIES PASSING ALL 5 GATES: {len(passing)}")
    for name in passing:
        print(f"  → {name}: Sharpe={passing[name]['sharpe']:.2f}, p={passing[name]['perm_p_value']:.3f}")
else:
    print("NO STRATEGIES PASSED ALL 5 GATES.")
    # Find closest
    candidates = {k: v for k, v in results.items() if isinstance(v, dict) and 'sharpe' in v and 'gates' in v}
    if candidates:
        best = max(candidates, key=lambda k: candidates[k]['sharpe'])
        bm = candidates[best]
        n_pass = sum(bm['gates'].values())
        print(f"  Closest: {best} ({n_pass}/5 gates, Sharpe={bm['sharpe']:.2f}, failed: {bm['failed_gates']})")

print("=" * 100)

# ── Save Results ────────────────────────────────────────────────────────
# Make JSON serializable
def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj

output = {
    'timestamp': datetime.now().isoformat(),
    'oot_period': f"{OOT_START} to {OOT_END}",
    'slippage': SLIPPAGE,
    'perm_iterations': PERM_ITERS,
    'variants': clean_for_json(results),
}

Path(RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_PATH, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to {RESULTS_PATH}")
