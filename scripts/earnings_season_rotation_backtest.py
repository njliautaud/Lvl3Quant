#!/usr/bin/env python3
"""
Earnings Season Portfolio Rotation Backtest
============================================
Thesis: Earnings season (weeks 3-5 after quarter end) is when company-specific
catalysts dominate — growth stocks get re-rated. Between seasons, macro/sentiment
drives. Different optimal portfolio compositions for each phase.

Walk-forward OOT: Jan 2022 – Jul 2026
5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, ≥20 trades
"""

import json
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
START_DATE = '2021-06-01'  # extra lookback for 200-SMA + 60d momentum
END_DATE = '2026-07-29'
OOT_START = '2022-01-01'
N_PERMUTATIONS = 1000
np.random.seed(42)

# Universe
GROWTH_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'NFLX', 'CRM']
SAFE_TICKERS = ['GLD', 'TLT', 'SHY', 'UUP']
INDEX_TICKERS = ['SPY', 'QQQ']
VIX_TICKER = '^VIX'

ALL_TICKERS = list(set(GROWTH_TICKERS + SAFE_TICKERS + INDEX_TICKERS))

# Quarter-end dates
QUARTER_ENDS = []
for year in range(2021, 2027):
    for month, day in [(3, 31), (6, 30), (9, 30), (12, 31)]:
        QUARTER_ENDS.append(pd.Timestamp(year, month, day))


def download_data():
    """Download all needed price data."""
    print("Downloading price data...", flush=True)
    tickers = ALL_TICKERS + [VIX_TICKER]
    data = yf.download(tickers, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']
    else:
        close = data

    vix = close['^VIX'].dropna() if '^VIX' in close.columns else pd.Series(dtype=float)
    print(f"Downloaded {len(close)} days of data", flush=True)
    return close, vix


def compute_regime(spy_series):
    """Bull = SPY > 200-SMA, Bear = SPY < 200-SMA."""
    sma200 = spy_series.rolling(200).mean()
    return (spy_series >= sma200)  # True = bull


def build_earnings_mask(trading_days, offset_days=0):
    """Return boolean array: True = earnings window (td 15-35 after quarter end)."""
    mask = np.zeros(len(trading_days), dtype=bool)
    td_arr = trading_days.values

    for qe in QUARTER_ENDS:
        shifted = qe + pd.Timedelta(days=offset_days)
        future = td_arr[td_arr > shifted]
        if len(future) >= 35:
            start_idx = np.searchsorted(td_arr, future[14])
            end_idx = np.searchsorted(td_arr, future[34])
            mask[start_idx:end_idx + 1] = True
    return mask


def build_pre_earnings_mask(trading_days, offset_days=0):
    """5 days before earnings start to 5 days into it."""
    mask = np.zeros(len(trading_days), dtype=bool)
    td_arr = trading_days.values

    for qe in QUARTER_ENDS:
        shifted = qe + pd.Timedelta(days=offset_days)
        future = td_arr[td_arr > shifted]
        if len(future) >= 20:
            earnings_start_idx = np.searchsorted(td_arr, future[14])
            pre_start_idx = max(0, earnings_start_idx - 5)
            pre_end_idx = np.searchsorted(td_arr, future[19])
            mask[pre_start_idx:pre_end_idx + 1] = True
    return mask


def build_post_earnings_mask(trading_days, offset_days=0):
    """Trading days 36-60 after quarter end."""
    mask = np.zeros(len(trading_days), dtype=bool)
    td_arr = trading_days.values

    for qe in QUARTER_ENDS:
        shifted = qe + pd.Timedelta(days=offset_days)
        future = td_arr[td_arr > shifted]
        if len(future) >= 60:
            start_idx = np.searchsorted(td_arr, future[35])
            end_idx = np.searchsorted(td_arr, future[59])
            mask[start_idx:end_idx + 1] = True
    return mask


def compute_metrics(daily_rets, oot_mask, regime_bull, n_trades):
    """Compute all metrics for a return series."""
    oot = daily_rets[oot_mask]
    if len(oot) == 0:
        return None

    equity = INITIAL_CAPITAL * np.cumprod(1 + oot)
    total_ret = equity[-1] / INITIAL_CAPITAL - 1

    std = oot.std()
    sharpe = (oot.mean() / std * np.sqrt(252)) if std > 0 else 0.0

    down = oot[oot < 0]
    down_std = down.std() if len(down) > 0 else 0
    sortino = (oot.mean() / down_std * np.sqrt(252)) if down_std > 0 else (999.0 if oot.mean() > 0 else 0.0)

    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = dd.min()

    years = len(oot) / 252
    cagr = (equity[-1] / INITIAL_CAPITAL) ** (1 / years) - 1 if years > 0 and equity[-1] > 0 else 0.0

    wins = oot[oot > 0].sum()
    losses = abs(oot[oot < 0].sum())
    pf = wins / losses if losses > 0 else 999.0
    wr = (oot > 0).sum() / len(oot) if len(oot) > 0 else 0.0

    # Regime analysis
    oot_bull = regime_bull[oot_mask]
    bull_rets = oot[oot_bull]
    bear_rets = oot[~oot_bull]

    bull_std = bull_rets.std() if len(bull_rets) > 5 else 0
    bear_std = bear_rets.std() if len(bear_rets) > 5 else 0
    bull_sharpe = (bull_rets.mean() / bull_std * np.sqrt(252)) if bull_std > 0 else 0.0
    bear_sharpe = (bear_rets.mean() / bear_std * np.sqrt(252)) if bear_std > 0 else 0.0
    max_abs = max(abs(bull_sharpe), abs(bear_sharpe))
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs if max_abs > 0 else 0.0

    return {
        'sharpe': round(float(sharpe), 4),
        'sortino': round(float(sortino), 4),
        'total_return': round(float(total_ret), 4),
        'max_drawdown': round(float(max_dd), 4),
        'n_trades': int(n_trades),
        'profit_factor': round(float(pf), 4),
        'win_rate': round(float(wr), 4),
        'cagr': round(float(cagr), 4),
        'final_equity': round(float(equity[-1]), 2),
        'regime_analysis': {
            'bull_sharpe': round(float(bull_sharpe), 4),
            'bear_sharpe': round(float(bear_sharpe), 4),
            'regime_gap': round(float(regime_gap), 4),
            'bull_days': int(oot_bull.sum()),
            'bear_days': int((~oot_bull).sum()),
        }
    }


def count_transitions(mask):
    """Count number of True->False and False->True transitions."""
    return int(np.sum(np.abs(np.diff(mask.astype(int)))))


def sharpe_from_rets(rets):
    """Quick Sharpe calc from numpy array."""
    if len(rets) == 0:
        return 0.0
    s = rets.std()
    return (rets.mean() / s * np.sqrt(252)) if s > 0 else 0.0


# ── Precomputed returns (numpy arrays for speed) ──
class PrecomputedData:
    def __init__(self, close_df, trading_days, vix_series):
        self.trading_days = trading_days
        self.td_arr = trading_days.values
        self.n = len(trading_days)

        # Daily returns as numpy arrays
        self.qqq_ret = close_df['QQQ'].pct_change().reindex(trading_days).fillna(0).values
        self.gld_ret = close_df['GLD'].pct_change().reindex(trading_days).fillna(0).values
        self.shy_ret = close_df['SHY'].pct_change().reindex(trading_days).fillna(0).values

        # Growth stock returns for momentum strategy
        self.growth_rets = {}
        for t in GROWTH_TICKERS:
            if t in close_df.columns:
                self.growth_rets[t] = close_df[t].pct_change().reindex(trading_days).fillna(0).values

        # 60-day momentum for growth stocks
        growth_prices = close_df[GROWTH_TICKERS].reindex(trading_days)
        self.momentum_60d = growth_prices.pct_change(60).values  # shape: (n, 10)
        self.growth_ret_matrix = growth_prices.pct_change().fillna(0).values  # (n, 10)

        # VIX
        if len(vix_series) > 0:
            self.vix = vix_series.reindex(trading_days).ffill().fillna(20).values
        else:
            self.vix = np.full(self.n, 20.0)


def strategy_A_fast(pre, mask):
    """QQQ during earnings, GLD off-season."""
    rets = np.where(mask, pre.qqq_ret, pre.gld_ret)
    # Slippage on transitions
    transitions = np.abs(np.diff(mask.astype(float)))
    slip = np.zeros(pre.n)
    slip[1:] = transitions * SLIPPAGE_PCT
    return rets - slip


def strategy_B_fast(pre, mask):
    """Top 3 momentum growth during earnings, GLD off-season."""
    n = pre.n
    rets = pre.gld_ret.copy()

    # Process earnings windows
    # Find contiguous earnings blocks
    changes = np.diff(mask.astype(int), prepend=0)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    if len(ends) < len(starts):
        ends = np.append(ends, n)

    for s, e in zip(starts, ends):
        # At window start, pick top 3 momentum stocks
        if s > 0:
            mom = pre.momentum_60d[s - 1]  # use previous day's momentum
        else:
            mom = pre.momentum_60d[s]

        valid = ~np.isnan(mom)
        if valid.sum() >= 3:
            top3_idx = np.argsort(mom)[-3:]
            top3_idx = top3_idx[valid[top3_idx]]
            if len(top3_idx) > 0:
                # Average return of top 3
                for i in range(s, e):
                    rets[i] = pre.growth_ret_matrix[i, top3_idx].mean()

        # Slippage on entry/exit
        rets[s] -= SLIPPAGE_PCT
        if e < n:
            rets[min(e, n - 1)] -= SLIPPAGE_PCT

    return rets


def strategy_C_fast(pre, mask):
    """Pre-earnings ramp: QQQ during pre-window, GLD otherwise."""
    rets = np.where(mask, pre.qqq_ret, pre.gld_ret)
    transitions = np.abs(np.diff(mask.astype(float)))
    slip = np.zeros(pre.n)
    slip[1:] = transitions * SLIPPAGE_PCT
    return rets - slip


def strategy_D_fast(pre, mask):
    """Post-earnings drift: QQQ during post-window, GLD otherwise."""
    rets = np.where(mask, pre.qqq_ret, pre.gld_ret)
    transitions = np.abs(np.diff(mask.astype(float)))
    slip = np.zeros(pre.n)
    slip[1:] = transitions * SLIPPAGE_PCT
    return rets - slip


def strategy_E_fast(pre, mask):
    """QQQ during earnings with 10% trailing stop -> SHY. GLD off-season."""
    n = pre.n
    rets = pre.gld_ret.copy()

    changes = np.diff(mask.astype(int), prepend=0)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    if len(ends) < len(starts):
        ends = np.append(ends, n)

    for s, e in zip(starts, ends):
        peak_eq = 1.0
        eq = 1.0
        stopped = False
        rets[s] = pre.qqq_ret[s] - SLIPPAGE_PCT  # entry slippage

        eq *= (1 + pre.qqq_ret[s])
        peak_eq = max(peak_eq, eq)

        for i in range(s + 1, e):
            if stopped:
                rets[i] = pre.shy_ret[i]
            else:
                rets[i] = pre.qqq_ret[i]
                eq *= (1 + pre.qqq_ret[i])
                peak_eq = max(peak_eq, eq)
                dd = (eq - peak_eq) / peak_eq
                if dd <= -0.10:
                    stopped = True
                    rets[i] -= SLIPPAGE_PCT  # exit slippage

        # Exit slippage at end of window
        if e < n:
            rets[min(e, n - 1)] -= SLIPPAGE_PCT

    return rets


def strategy_F_fast(pre, mask):
    """VIX<20 during earnings -> QQQ, VIX>=20 during earnings -> GLD. Off-season GLD."""
    in_qqq = mask & (pre.vix < 20)
    rets = np.where(in_qqq, pre.qqq_ret, pre.gld_ret)
    transitions = np.abs(np.diff(in_qqq.astype(float)))
    slip = np.zeros(pre.n)
    slip[1:] = transitions * SLIPPAGE_PCT
    return rets - slip


def run_permutation_test(pre, oot_mask, mask_builder, strategy_func, actual_sharpe,
                         n_perms=N_PERMUTATIONS):
    """Fast permutation test using precomputed data."""
    count_better = 0
    trading_days = pre.trading_days

    for _ in range(n_perms):
        offset = np.random.randint(1, 31)
        perm_mask = mask_builder(trading_days, offset_days=offset)
        perm_rets = strategy_func(pre, perm_mask)
        perm_sharpe = sharpe_from_rets(perm_rets[oot_mask])
        if perm_sharpe >= actual_sharpe:
            count_better += 1

    return round(count_better / n_perms, 4)


def five_gate_check(metrics, perm_pvalue):
    """Apply 5-gate validation."""
    gates = {
        'sharpe_gt_0.5': metrics['sharpe'] > 0.5,
        'perm_p_lt_0.05': perm_pvalue < 0.05,
        'regime_gap_lt_0.5': metrics['regime_analysis']['regime_gap'] < 0.5,
        'max_dd_gt_neg50': metrics['max_drawdown'] > -0.50,
        'min_20_trades': metrics['n_trades'] >= 20,
    }
    passed = sum(gates.values())
    return gates, passed


def main():
    close_df, vix_series = download_data()

    # Common trading days (all core tickers must have data)
    core = ['SPY', 'QQQ', 'GLD', 'SHY']
    valid = close_df[core].dropna()
    trading_days = valid.index.sort_values()

    # Regime
    spy = close_df['SPY'].reindex(trading_days).ffill()
    regime_bull = compute_regime(spy)
    regime_bull_arr = regime_bull.values

    # OOT mask
    oot_mask = np.array(trading_days >= OOT_START)

    print(f"Trading days: {len(trading_days)}", flush=True)
    print(f"OOT days: {oot_mask.sum()}", flush=True)
    print(f"Bull (OOT): {regime_bull_arr[oot_mask].sum()}, Bear (OOT): {(~regime_bull_arr[oot_mask]).sum()}", flush=True)

    # Earnings window stats
    e_mask = build_earnings_mask(trading_days)
    print(f"Earnings window days (total): {e_mask.sum()}", flush=True)
    print(f"Off-season days (total): {(~e_mask).sum()}", flush=True)

    # Precompute all return arrays
    pre = PrecomputedData(close_df, trading_days, vix_series)

    # Strategy definitions: (name, mask_builder, strategy_func)
    strategies = [
        ('A_GrowthEarnings_SafeOff', build_earnings_mask, strategy_A_fast),
        ('B_MomentumGrowthEarnings', build_earnings_mask, strategy_B_fast),
        ('C_PreEarningsRamp', build_pre_earnings_mask, strategy_C_fast),
        ('D_PostEarningsDrift', build_post_earnings_mask, strategy_D_fast),
        ('E_GrowthRiskManaged', build_earnings_mask, strategy_E_fast),
        ('F_AdaptiveVIX', build_earnings_mask, strategy_F_fast),
    ]

    results = {}

    for name, mask_builder, strat_func in strategies:
        print(f"\n{'='*60}", flush=True)
        print(f"Running {name}...", flush=True)

        mask = mask_builder(trading_days)
        daily_rets = strat_func(pre, mask)
        n_trades = count_transitions(mask)

        metrics = compute_metrics(daily_rets, oot_mask, regime_bull_arr, n_trades)
        if metrics is None:
            print(f"  SKIP — no OOT data", flush=True)
            continue

        # Permutation test
        print(f"  Running permutation test ({N_PERMUTATIONS} iters)...", flush=True)
        perm_p = run_permutation_test(pre, oot_mask, mask_builder, strat_func,
                                       metrics['sharpe'], N_PERMUTATIONS)

        # 5-gate check
        gates, gates_passed = five_gate_check(metrics, perm_p)
        verdict = f"{'PASS' if gates_passed == 5 else 'FAIL'} ({gates_passed}/5)"

        results[name] = {
            **metrics,
            'permutation_pvalue': perm_p,
            'gates': gates,
            'gates_passed': gates_passed,
            'verdict': verdict,
        }

        print(f"  Sharpe: {metrics['sharpe']}", flush=True)
        print(f"  Sortino: {metrics['sortino']}", flush=True)
        print(f"  Total Return: {metrics['total_return']:.2%}", flush=True)
        print(f"  Max DD: {metrics['max_drawdown']:.2%}", flush=True)
        print(f"  Trades: {n_trades}", flush=True)
        print(f"  PF: {metrics['profit_factor']}", flush=True)
        print(f"  WR: {metrics['win_rate']:.2%}", flush=True)
        print(f"  Regime Gap: {metrics['regime_analysis']['regime_gap']}", flush=True)
        print(f"  Perm p: {perm_p}", flush=True)
        print(f"  Verdict: {verdict}", flush=True)

    # Save results
    output_path = Path('/home/jupiter/Lvl3Quant/data/earnings_season_rotation_results.json')

    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return round(float(obj), 6)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, float) and (np.isinf(obj) or np.isnan(obj)):
            return str(obj)
        return obj

    with open(output_path, 'w') as f:
        json.dump(sanitize(results), f, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"Results saved to {output_path}", flush=True)
    print(f"\nSUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    for name, r in results.items():
        print(f"  {name}: {r['verdict']} | Sharpe={r['sharpe']} | Ret={r['total_return']:.2%} | "
              f"DD={r['max_drawdown']:.2%} | PF={r['profit_factor']} | Perm-p={r['permutation_pvalue']}", flush=True)


if __name__ == '__main__':
    main()
