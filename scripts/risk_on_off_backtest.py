#!/usr/bin/env python3
"""
Systematic Risk-On/Risk-Off Market Timing Backtest
===================================================
Tests 6 variants of regime-based timing on QQQ with $645 starting capital.
Walk-forward OOT: Jan 2022 - Jul 2026.

5-Gate Validation:
  1. Sharpe > 0.5
  2. Permutation p-value < 0.05
  3. Regime gap < 0.5
  4. MaxDD > -50%
  5. >= 20 trades
"""

import json
import warnings
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings('ignore')

# ── Constants ────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 645.0
SLIPPAGE_BPS = 0.02 / 100  # 0.02% per trade
COMMISSION = 0.0
OOT_START = "2021-01-01"  # extra lookback for SMA warmup
OOT_TRADE_START = "2022-01-03"
OOT_END = "2026-07-28"
N_PERMUTATIONS = 1000
RANDOM_SEED = 42

# ── Data Download ────────────────────────────────────────────────────────────
def download_data():
    """Download SPY and QQQ daily data with enough lookback for 200-SMA."""
    print("Downloading market data...")
    spy = yf.download("SPY", start=OOT_START, end=OOT_END, progress=False, auto_adjust=True)
    qqq = yf.download("QQQ", start=OOT_START, end=OOT_END, progress=False, auto_adjust=True)

    # Handle multi-level columns from yfinance
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    if isinstance(qqq.columns, pd.MultiIndex):
        qqq.columns = qqq.columns.get_level_values(0)

    df = pd.DataFrame({
        'spy_close': spy['Close'],
        'qqq_close': qqq['Close'],
    })
    df = df.dropna()

    # Compute indicators
    df['spy_sma200'] = df['spy_close'].rolling(200).mean()
    df['spy_sma50'] = df['spy_close'].rolling(50).mean()
    df['spy_mom20'] = df['spy_close'].pct_change(20) * 100  # 20-day return in %
    df['spy_rsi10'] = compute_rsi(df['spy_close'], 10)
    # VIX proxy: 20-day realized vol annualized
    df['vix_proxy'] = df['spy_close'].pct_change().rolling(20).std() * np.sqrt(252) * 100
    df['qqq_ret'] = df['qqq_close'].pct_change()

    # Trim to OOT period
    df = df.loc[OOT_TRADE_START:]
    df = df.dropna()

    print(f"  Data: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}, {len(df)} trading days")
    return df


def compute_rsi(series, period=10):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# ── Signal Generators ────────────────────────────────────────────────────────
def signal_200sma(df):
    """A) 200-SMA Binary: SPY > 200-SMA = 100% QQQ, else cash."""
    alloc = (df['spy_close'] > df['spy_sma200']).astype(float)
    return alloc

def signal_50sma(df):
    """B) 50-SMA Binary: SPY > 50-SMA = 100% QQQ, else cash."""
    alloc = (df['spy_close'] > df['spy_sma50']).astype(float)
    return alloc

def signal_dual_sma(df):
    """C) Dual SMA: both = 100%, > 200 but < 50 = 50%, < 200 = 0%."""
    above200 = df['spy_close'] > df['spy_sma200']
    above50 = df['spy_close'] > df['spy_sma50']
    alloc = pd.Series(0.0, index=df.index)
    alloc[above200 & ~above50] = 0.5
    alloc[above200 & above50] = 1.0
    return alloc

def signal_vix_regime(df):
    """D) VIX Regime: VIX proxy < 18 = 100%, 18-25 = 50%, > 25 = cash."""
    alloc = pd.Series(0.0, index=df.index)
    alloc[df['vix_proxy'] < 18] = 1.0
    alloc[(df['vix_proxy'] >= 18) & (df['vix_proxy'] <= 25)] = 0.5
    return alloc

def signal_momentum(df):
    """E) Momentum Timer: 20d ret > 0 = 100%, < -5% = cash, between = 50%."""
    alloc = pd.Series(0.5, index=df.index)
    alloc[df['spy_mom20'] > 0] = 1.0
    alloc[df['spy_mom20'] < -5] = 0.0
    return alloc

def signal_combined(df):
    """F) Combined: score from 200SMA + 50SMA + momentum. 3=100%, 2=66%, 1=33%, 0=0%."""
    score = ((df['spy_close'] > df['spy_sma200']).astype(int) +
             (df['spy_close'] > df['spy_sma50']).astype(int) +
             (df['spy_mom20'] > 0).astype(int))
    mapping = {0: 0.0, 1: 1/3, 2: 2/3, 3: 1.0}
    alloc = score.map(mapping)
    return alloc


VARIANTS = {
    'A_200SMA_Binary': signal_200sma,
    'B_50SMA_Binary': signal_50sma,
    'C_Dual_SMA': signal_dual_sma,
    'D_VIX_Regime': signal_vix_regime,
    'E_Momentum_Timer': signal_momentum,
    'F_Combined_Signal': signal_combined,
}


# ── Backtest Engine ──────────────────────────────────────────────────────────
def run_backtest(df, alloc_series, initial_capital=INITIAL_CAPITAL):
    """
    Simulate daily rebalancing with fractional shares.
    alloc_series: daily target allocation to QQQ (0.0 to 1.0).
    Returns equity curve, daily returns, trade count.
    """
    equity = initial_capital
    position_value = 0.0  # value in QQQ
    cash = initial_capital
    current_alloc = 0.0

    equity_curve = []
    daily_returns = []
    n_trades = 0
    prev_equity = initial_capital

    for i in range(len(df)):
        # Apply today's QQQ return to position
        if i > 0 and current_alloc > 0:
            qqq_ret = df['qqq_ret'].iloc[i]
            position_value *= (1 + qqq_ret)

        equity = cash + position_value
        target_alloc = alloc_series.iloc[i]

        # Rebalance if allocation changed
        if abs(target_alloc - current_alloc) > 0.01:
            # Cost of rebalancing
            target_position = equity * target_alloc
            trade_value = abs(target_position - position_value)

            if trade_value > 1.0:  # min trade threshold
                slippage_cost = trade_value * SLIPPAGE_BPS
                equity -= slippage_cost
                n_trades += 1

            position_value = equity * target_alloc
            cash = equity - position_value
            current_alloc = target_alloc

        equity = cash + position_value
        daily_ret = (equity / prev_equity - 1) if prev_equity > 0 else 0.0
        equity_curve.append(equity)
        daily_returns.append(daily_ret)
        prev_equity = equity

    return np.array(equity_curve), np.array(daily_returns), n_trades


# ── Performance Metrics ──────────────────────────────────────────────────────
def compute_metrics(daily_returns, equity_curve, n_trades):
    """Compute Sharpe, Sortino, PF, WR, MaxDD from daily returns."""
    dr = daily_returns[1:]  # skip first day
    if len(dr) == 0 or np.std(dr) == 0:
        return {
            'sharpe': 0.0, 'sortino': 0.0, 'profit_factor': 0.0,
            'win_rate': 0.0, 'max_dd_pct': 0.0, 'total_return_pct': 0.0,
            'final_equity': equity_curve[-1], 'n_trades': n_trades,
            'cagr_pct': 0.0,
        }

    ann_factor = np.sqrt(252)

    # Sharpe
    sharpe = np.mean(dr) / np.std(dr) * ann_factor

    # Sortino
    downside = dr[dr < 0]
    downside_std = np.std(downside) if len(downside) > 0 else 1e-8
    sortino = np.mean(dr) / downside_std * ann_factor

    # Profit factor (sum of positive days / abs sum of negative days)
    pos = dr[dr > 0].sum()
    neg = abs(dr[dr < 0].sum())
    pf = pos / neg if neg > 0 else float('inf')

    # Win rate
    wr = (dr > 0).sum() / len(dr) * 100

    # Max drawdown
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak
    max_dd = dd.min() * 100

    # Total return
    total_ret = (equity_curve[-1] / equity_curve[0] - 1) * 100

    # CAGR
    n_years = len(dr) / 252
    cagr = ((equity_curve[-1] / equity_curve[0]) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

    return {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'profit_factor': round(pf, 3),
        'win_rate': round(wr, 1),
        'max_dd_pct': round(max_dd, 2),
        'total_return_pct': round(total_ret, 2),
        'final_equity': round(equity_curve[-1], 2),
        'n_trades': n_trades,
        'cagr_pct': round(cagr, 2),
    }


# ── Permutation Test ─────────────────────────────────────────────────────────
def permutation_test(df, alloc_series, observed_sharpe, n_perms=N_PERMUTATIONS):
    """
    Shuffle the signal dates (preserving frequency of each allocation level)
    to test if timing adds value vs random timing.
    """
    rng = np.random.RandomState(RANDOM_SEED)
    alloc_vals = alloc_series.values.copy()
    count_better = 0

    for _ in range(n_perms):
        shuffled = alloc_vals.copy()
        rng.shuffle(shuffled)
        shuffled_series = pd.Series(shuffled, index=alloc_series.index)
        _, dr, _ = run_backtest(df, shuffled_series)
        dr_clean = dr[1:]
        if len(dr_clean) > 0 and np.std(dr_clean) > 0:
            perm_sharpe = np.mean(dr_clean) / np.std(dr_clean) * np.sqrt(252)
        else:
            perm_sharpe = 0.0
        if perm_sharpe >= observed_sharpe:
            count_better += 1

    p_value = (count_better + 1) / (n_perms + 1)
    return p_value


# ── Regime Stratification ────────────────────────────────────────────────────
def regime_stratification(df, daily_returns):
    """
    Compute Sharpe in bull vs bear regimes.
    Bull = SPY > 200-SMA, Bear = SPY < 200-SMA.
    """
    bull = df['spy_close'] > df['spy_sma200']
    bear = ~bull

    dr = daily_returns
    bull_dr = dr[bull.values]
    bear_dr = dr[bear.values]

    def safe_sharpe(r):
        if len(r) < 10 or np.std(r) == 0:
            return 0.0
        return np.mean(r) / np.std(r) * np.sqrt(252)

    bull_sharpe = safe_sharpe(bull_dr)
    bear_sharpe = safe_sharpe(bear_dr)

    gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 1e-8)

    return {
        'bull_sharpe': round(bull_sharpe, 3),
        'bear_sharpe': round(bear_sharpe, 3),
        'regime_gap': round(gap, 3),
        'bull_days': int(bull.sum()),
        'bear_days': int(bear.sum()),
    }


# ── Max Drawdown Avoided ─────────────────────────────────────────────────────
def max_dd_avoided(benchmark_equity, strategy_equity):
    """
    How much of QQQ's worst drawdown was avoided by the strategy.
    """
    # Benchmark max DD
    bm_peak = np.maximum.accumulate(benchmark_equity)
    bm_dd = (benchmark_equity - bm_peak) / bm_peak
    bm_max_dd = bm_dd.min() * 100

    # Strategy max DD
    st_peak = np.maximum.accumulate(strategy_equity)
    st_dd = (strategy_equity - st_peak) / st_peak
    st_max_dd = st_dd.min() * 100

    avoided = bm_max_dd - st_max_dd  # positive means strategy had less DD
    return {
        'benchmark_max_dd_pct': round(bm_max_dd, 2),
        'strategy_max_dd_pct': round(st_max_dd, 2),
        'dd_avoided_pct': round(avoided, 2),
    }


# ── 5-Gate Validation ─────────────────────────────────────────────────────────
def validate_5gates(metrics, perm_p, regime_gap):
    gates = {
        'sharpe_gt_0.5': metrics['sharpe'] > 0.5,
        'perm_p_lt_0.05': perm_p < 0.05,
        'regime_gap_lt_0.5': regime_gap < 0.5,
        'max_dd_gt_neg50': metrics['max_dd_pct'] > -50,
        'trades_gte_20': metrics['n_trades'] >= 20,
    }
    gates['all_passed'] = all(gates.values())
    return gates


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    df = download_data()

    # ── Benchmark: Buy and Hold QQQ ──
    print("\n=== BENCHMARK: Buy & Hold QQQ ===")
    bh_alloc = pd.Series(1.0, index=df.index)
    bh_equity, bh_returns, bh_trades = run_backtest(df, bh_alloc)
    bh_metrics = compute_metrics(bh_returns, bh_equity, bh_trades)
    print(f"  Final equity: ${bh_metrics['final_equity']:.2f}")
    print(f"  Total return: {bh_metrics['total_return_pct']:.1f}%")
    print(f"  Sharpe: {bh_metrics['sharpe']:.3f}")
    print(f"  MaxDD: {bh_metrics['max_dd_pct']:.1f}%")

    results = {
        'metadata': {
            'strategy': 'Systematic Risk-On/Risk-Off Market Timing',
            'run_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'oot_period': f"{OOT_TRADE_START} to {OOT_END}",
            'initial_capital': INITIAL_CAPITAL,
            'slippage_bps': SLIPPAGE_BPS * 10000,
            'n_permutations': N_PERMUTATIONS,
            'trading_days': len(df),
        },
        'benchmark': bh_metrics,
        'variants': {},
    }

    # ── Run Each Variant ──
    for name, signal_fn in VARIANTS.items():
        print(f"\n=== Variant {name} ===")
        alloc = signal_fn(df)

        # Run backtest
        equity, daily_ret, n_trades = run_backtest(df, alloc)
        metrics = compute_metrics(daily_ret, equity, n_trades)

        print(f"  Final equity: ${metrics['final_equity']:.2f}  |  Trades: {metrics['n_trades']}")
        print(f"  Sharpe: {metrics['sharpe']:.3f}  |  Sortino: {metrics['sortino']:.3f}")
        print(f"  MaxDD: {metrics['max_dd_pct']:.1f}%  |  WR: {metrics['win_rate']:.1f}%")

        # Permutation test
        print(f"  Running permutation test ({N_PERMUTATIONS} iterations)...", end='', flush=True)
        perm_p = permutation_test(df, alloc, metrics['sharpe'])
        print(f" p={perm_p:.4f}")

        # Regime stratification
        regime = regime_stratification(df, daily_ret)
        print(f"  Bull Sharpe: {regime['bull_sharpe']:.3f}  |  Bear Sharpe: {regime['bear_sharpe']:.3f}  |  Gap: {regime['regime_gap']:.3f}")

        # DD avoided
        dd_info = max_dd_avoided(bh_equity, equity)
        print(f"  Benchmark MaxDD: {dd_info['benchmark_max_dd_pct']:.1f}%  |  Strategy MaxDD: {dd_info['strategy_max_dd_pct']:.1f}%  |  Avoided: {dd_info['dd_avoided_pct']:.1f}%")

        # 5-gate validation
        gates = validate_5gates(metrics, perm_p, regime['regime_gap'])
        gate_str = ' | '.join([f"{k}={'PASS' if v else 'FAIL'}" for k, v in gates.items() if k != 'all_passed'])
        status = "ALL PASS" if gates['all_passed'] else "FAILED"
        print(f"  Gates: {status}")
        print(f"    {gate_str}")

        # Excess vs benchmark
        excess_sharpe = round(metrics['sharpe'] - bh_metrics['sharpe'], 3)
        excess_return = round(metrics['total_return_pct'] - bh_metrics['total_return_pct'], 2)

        results['variants'][name] = {
            'metrics': metrics,
            'permutation_p_value': round(perm_p, 4),
            'regime': regime,
            'drawdown_avoided': dd_info,
            'gates': gates,
            'benchmark_comparison': {
                'excess_sharpe': excess_sharpe,
                'excess_return_pct': excess_return,
            },
            'allocation_stats': {
                'pct_time_fully_invested': round((alloc == 1.0).mean() * 100, 1),
                'pct_time_cash': round((alloc == 0.0).mean() * 100, 1),
                'pct_time_partial': round(((alloc > 0) & (alloc < 1.0)).mean() * 100, 1),
            },
        }

    # ── Summary Table ──
    print("\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)
    header = f"{'Variant':<22} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR%':>6} {'MaxDD%':>8} {'Trades':>7} {'Final$':>9} {'Perm-p':>8} {'ExcSh':>7} {'Gates':>7}"
    print(header)
    print("-"*100)

    for name, data in results['variants'].items():
        m = data['metrics']
        p = data['permutation_p_value']
        es = data['benchmark_comparison']['excess_sharpe']
        g = "PASS" if data['gates']['all_passed'] else "FAIL"
        row = f"{name:<22} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['profit_factor']:>6.2f} {m['win_rate']:>5.1f}% {m['max_dd_pct']:>7.1f}% {m['n_trades']:>7d} ${m['final_equity']:>8.2f} {p:>8.4f} {es:>+7.3f} {g:>7}"
        print(row)

    bm = results['benchmark']
    print("-"*100)
    print(f"{'BUY&HOLD QQQ':<22} {bm['sharpe']:>7.3f} {bm['sortino']:>8.3f} {bm['profit_factor']:>6.2f} {bm['win_rate']:>5.1f}% {bm['max_dd_pct']:>7.1f}% {bm['n_trades']:>7d} ${bm['final_equity']:>8.2f} {'N/A':>8} {'+0.000':>7} {'BASE':>7}")
    print("="*100)

    # ── Save results ──
    out_path = Path("/home/jupiter/Lvl3Quant/data/risk_on_off_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # ── Final verdict ──
    passed = [n for n, d in results['variants'].items() if d['gates']['all_passed']]
    if passed:
        print(f"\nVERDICT: {len(passed)} variant(s) passed all 5 gates: {', '.join(passed)}")
        best = max(passed, key=lambda n: results['variants'][n]['metrics']['sharpe'])
        print(f"  Best: {best} (Sharpe={results['variants'][best]['metrics']['sharpe']:.3f})")
    else:
        print("\nVERDICT: No variants passed all 5 gates.")
        # Find the one closest to passing
        best_sharpe_name = max(results['variants'].keys(),
                               key=lambda n: results['variants'][n]['metrics']['sharpe'])
        print(f"  Highest Sharpe: {best_sharpe_name} ({results['variants'][best_sharpe_name]['metrics']['sharpe']:.3f})")


if __name__ == '__main__':
    main()
