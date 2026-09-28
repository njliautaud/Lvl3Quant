#!/usr/bin/env python3
"""
Volatility Regime Switching Backtest
=====================================
6 variants of vol-regime-based allocation switching.
Walk-forward OOT: Jan 2022 - Jul 2026.
5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades.
"""

import json
import warnings
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings('ignore')

# ─── Configuration ───────────────────────────────────────────────────────────
ACCOUNT_SIZE = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02% per trade
OOT_START = '2022-01-01'
OOT_END = '2026-07-30'
DATA_START = '2021-01-01'  # extra history for lookback calcs
N_PERMUTATIONS = 1000
SEED = 42

AGGRESSIVE = ['QQQ', 'TQQQ', 'XLK', 'SMH']
DEFENSIVE = ['TLT', 'GLD', 'SHY', 'UUP']
BENCHMARK = 'SPY'
ALL_TICKERS = list(set(AGGRESSIVE + DEFENSIVE + [BENCHMARK]))

# Vol regime thresholds (annualized %)
VOL_LOW = 15
VOL_NORMAL = 25
VOL_HIGH = 25
VOL_CRISIS = 35

np.random.seed(SEED)


# ─── Data Download ───────────────────────────────────────────────────────────
def download_data():
    """Download adjusted close prices for all tickers."""
    print("Downloading price data...")
    data = yf.download(ALL_TICKERS, start=DATA_START, end=OOT_END, auto_adjust=True, progress=False)

    # Handle multi-level columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        closes = data['Close']
    else:
        closes = data

    # Drop any columns that are entirely NaN
    closes = closes.dropna(axis=1, how='all')

    # Forward fill then backward fill for missing data
    closes = closes.ffill().bfill()

    print(f"  Data: {closes.index[0].strftime('%Y-%m-%d')} to {closes.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Tickers: {list(closes.columns)}")

    return closes


def compute_features(closes):
    """Compute all features needed for strategies."""
    spy = closes[BENCHMARK]

    features = pd.DataFrame(index=closes.index)

    # Realized vol (20-day, annualized)
    spy_returns = spy.pct_change()
    features['rvol_20d'] = spy_returns.rolling(20).std() * np.sqrt(252) * 100
    features['rvol_5d'] = spy_returns.rolling(5).std() * np.sqrt(252) * 100

    # SPY 200-SMA for regime classification
    features['spy_sma200'] = spy.rolling(200).mean()
    features['spy_price'] = spy
    features['bull_regime'] = (spy > features['spy_sma200']).astype(int)

    # Returns for momentum ranking
    for ticker in AGGRESSIVE + DEFENSIVE:
        if ticker in closes.columns:
            features[f'{ticker}_ret20d'] = closes[ticker].pct_change(20)
            features[f'{ticker}_vol20d'] = closes[ticker].pct_change().rolling(20).std() * np.sqrt(252) * 100

    # VIX proxy: we don't have VIX directly, approximate using realized vol ratio
    # For variant F, we use rvol_5d vs rvol_20d as a proxy for VIX vs realized vol
    features['vol_fear_premium'] = features['rvol_5d'] - features['rvol_20d']

    return features


# ─── Strategy Implementations ───────────────────────────────────────────────
def get_vol_regime(rvol):
    """Classify volatility regime."""
    if rvol < VOL_LOW:
        return 'low'
    elif rvol < VOL_NORMAL:
        return 'normal'
    elif rvol < VOL_CRISIS:
        return 'high'
    else:
        return 'crisis'


def strategy_a_binary_switch(features, closes):
    """Binary Vol Switch: Low/Normal -> QQQ, High/Crisis -> TLT."""
    positions = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)

    for i in range(len(features)):
        rvol = features['rvol_20d'].iloc[i]
        if pd.isna(rvol):
            continue
        regime = get_vol_regime(rvol)
        if regime in ('low', 'normal'):
            if 'QQQ' in positions.columns:
                positions.iloc[i, positions.columns.get_loc('QQQ')] = 1.0
        else:
            if 'TLT' in positions.columns:
                positions.iloc[i, positions.columns.get_loc('TLT')] = 1.0

    return positions


def strategy_b_gradual_allocation(features, closes):
    """Gradual Vol Allocation based on regime."""
    positions = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)

    for i in range(len(features)):
        rvol = features['rvol_20d'].iloc[i]
        if pd.isna(rvol):
            continue
        regime = get_vol_regime(rvol)

        if regime == 'low':
            if 'QQQ' in positions.columns:
                positions.iloc[i, positions.columns.get_loc('QQQ')] = 1.0
        elif regime == 'normal':
            if 'QQQ' in positions.columns:
                positions.iloc[i, positions.columns.get_loc('QQQ')] = 0.6
            if 'GLD' in positions.columns:
                positions.iloc[i, positions.columns.get_loc('GLD')] = 0.4
        elif regime == 'high':
            if 'GLD' in positions.columns:
                positions.iloc[i, positions.columns.get_loc('GLD')] = 1.0
        else:  # crisis
            if 'GLD' in positions.columns:
                positions.iloc[i, positions.columns.get_loc('GLD')] = 0.5
            if 'SHY' in positions.columns:
                positions.iloc[i, positions.columns.get_loc('SHY')] = 0.5

    return positions


def strategy_c_vol_breakout(features, closes):
    """Vol Breakout: 5d vol crosses above 20d vol by >50% -> defensive."""
    positions = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)

    for i in range(len(features)):
        rvol_5 = features['rvol_5d'].iloc[i]
        rvol_20 = features['rvol_20d'].iloc[i]
        if pd.isna(rvol_5) or pd.isna(rvol_20) or rvol_20 == 0:
            continue

        ratio = rvol_5 / rvol_20
        if ratio > 1.5:  # 5d vol > 20d vol by 50%
            if 'TLT' in positions.columns:
                positions.iloc[i, positions.columns.get_loc('TLT')] = 0.5
            if 'GLD' in positions.columns:
                positions.iloc[i, positions.columns.get_loc('GLD')] = 0.5
        else:
            if 'QQQ' in positions.columns:
                positions.iloc[i, positions.columns.get_loc('QQQ')] = 1.0

    return positions


def strategy_d_vol_mean_reversion(features, closes):
    """Vol Mean Reversion: After vol spike >30%, wait for drop <20% -> buy QQQ."""
    positions = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)

    spike_detected = False

    for i in range(len(features)):
        rvol = features['rvol_20d'].iloc[i]
        if pd.isna(rvol):
            continue

        if rvol > 30:
            spike_detected = True
            if 'GLD' in positions.columns:
                positions.iloc[i, positions.columns.get_loc('GLD')] = 1.0
        elif spike_detected and rvol < 20:
            spike_detected = False
            if 'QQQ' in positions.columns:
                positions.iloc[i, positions.columns.get_loc('QQQ')] = 1.0
        elif rvol < 20:
            if 'QQQ' in positions.columns:
                positions.iloc[i, positions.columns.get_loc('QQQ')] = 1.0
        else:
            if 'GLD' in positions.columns:
                positions.iloc[i, positions.columns.get_loc('GLD')] = 1.0

    return positions


def strategy_e_vol_weighted_momentum(features, closes):
    """Vol-Weighted Momentum: Low vol -> highest momentum aggressive. High vol -> lowest vol defensive."""
    positions = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)

    agg_tickers = [t for t in AGGRESSIVE if t in closes.columns and f'{t}_ret20d' in features.columns]
    def_tickers = [t for t in DEFENSIVE if t in closes.columns and f'{t}_vol20d' in features.columns]

    for i in range(len(features)):
        rvol = features['rvol_20d'].iloc[i]
        if pd.isna(rvol):
            continue

        if rvol < VOL_NORMAL:
            mom_scores = {}
            for t in agg_tickers:
                ret = features[f'{t}_ret20d'].iloc[i]
                if not pd.isna(ret):
                    mom_scores[t] = ret
            if mom_scores:
                best = max(mom_scores, key=mom_scores.get)
                positions.iloc[i, positions.columns.get_loc(best)] = 1.0
        else:
            vol_scores = {}
            for t in def_tickers:
                vol = features[f'{t}_vol20d'].iloc[i]
                if not pd.isna(vol) and vol > 0:
                    vol_scores[t] = vol
            if vol_scores:
                best = min(vol_scores, key=vol_scores.get)
                positions.iloc[i, positions.columns.get_loc(best)] = 1.0

    return positions


def strategy_f_vix_contango(features, closes):
    """VIX Contango/Backwardation proxy: 5d vol > 20d vol (fear premium) -> defensive."""
    positions = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)

    for i in range(len(features)):
        fear_premium = features['vol_fear_premium'].iloc[i]
        if pd.isna(fear_premium):
            continue

        if fear_premium > 3:
            if 'GLD' in positions.columns:
                positions.iloc[i, positions.columns.get_loc('GLD')] = 0.6
            if 'TLT' in positions.columns:
                positions.iloc[i, positions.columns.get_loc('TLT')] = 0.4
        elif fear_premium < -3:
            if 'TQQQ' in positions.columns:
                positions.iloc[i, positions.columns.get_loc('TQQQ')] = 0.5
            if 'SMH' in positions.columns:
                positions.iloc[i, positions.columns.get_loc('SMH')] = 0.5
        else:
            if 'QQQ' in positions.columns:
                positions.iloc[i, positions.columns.get_loc('QQQ')] = 0.6
            if 'GLD' in positions.columns:
                positions.iloc[i, positions.columns.get_loc('GLD')] = 0.4

    return positions


# ─── Backtest Engine ─────────────────────────────────────────────────────────
def run_backtest(positions, closes, account_size=ACCOUNT_SIZE):
    """Run backtest with slippage costs. Returns portfolio value series and trade count."""
    oot_mask = (closes.index >= OOT_START) & (closes.index <= OOT_END)
    pos = positions.loc[oot_mask].copy()
    px = closes.loc[oot_mask].copy()

    if len(pos) == 0:
        return pd.Series(dtype=float), 0

    # Daily returns per asset
    asset_returns = px.pct_change().fillna(0)

    # Apply slippage on weight changes (rebalances)
    weight_changes = pos.diff().fillna(0).abs()
    daily_slippage = weight_changes.sum(axis=1) * SLIPPAGE_PCT

    port_returns = (pos.shift(1) * asset_returns).sum(axis=1) - daily_slippage
    port_returns.iloc[0] = 0

    # Count rebalances (days where weights change materially)
    rebalance_mask = weight_changes.sum(axis=1) > 0.01
    n_trades = rebalance_mask.sum()

    # Portfolio value
    port_value = account_size * (1 + port_returns).cumprod()

    return port_value, int(n_trades)


def compute_metrics(port_value, closes):
    """Compute performance metrics."""
    if len(port_value) < 10:
        return None

    returns = port_value.pct_change().dropna()
    if len(returns) == 0 or returns.std() == 0:
        return None

    ann_return = (port_value.iloc[-1] / port_value.iloc[0]) ** (252 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else ann_vol
    sortino = ann_return / downside_std if downside_std > 0 else 0

    cummax = port_value.cummax()
    drawdown = (port_value - cummax) / cummax
    max_dd = drawdown.min()

    total_return = (port_value.iloc[-1] / port_value.iloc[0]) - 1

    win_rate = (returns > 0).mean()

    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    final_value = port_value.iloc[-1]

    return {
        'total_return_pct': round(total_return * 100, 2),
        'ann_return_pct': round(ann_return * 100, 2),
        'ann_vol_pct': round(ann_vol * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_drawdown_pct': round(max_dd * 100, 2),
        'profit_factor': round(profit_factor, 3),
        'win_rate_pct': round(win_rate * 100, 1),
        'final_value': round(final_value, 2),
        'n_days': len(returns),
    }


def compute_regime_metrics(port_value, features):
    """Compute Sharpe in bull vs bear regimes."""
    returns = port_value.pct_change().dropna()

    common_idx = returns.index.intersection(features.index)
    returns_aligned = returns.loc[common_idx]
    bull_mask = features.loc[common_idx, 'bull_regime'] == 1

    bull_returns = returns_aligned[bull_mask]
    bear_returns = returns_aligned[~bull_mask]

    def regime_sharpe(r):
        if len(r) < 10 or r.std() == 0:
            return 0.0
        return (r.mean() * 252) / (r.std() * np.sqrt(252))

    sharpe_bull = regime_sharpe(bull_returns)
    sharpe_bear = regime_sharpe(bear_returns)

    max_abs = max(abs(sharpe_bull), abs(sharpe_bear))
    regime_gap = abs(sharpe_bull - sharpe_bear) / max_abs if max_abs > 0 else 0

    return {
        'sharpe_bull': round(sharpe_bull, 3),
        'sharpe_bear': round(sharpe_bear, 3),
        'regime_gap': round(regime_gap, 3),
        'bull_days': int(bull_mask.sum()),
        'bear_days': int((~bull_mask).sum()),
    }


def permutation_test(port_value, features, closes, n_perms=N_PERMUTATIONS):
    """
    Permutation test: shuffle daily returns to test whether timing matters.
    p-value = fraction of permuted Sharpes >= actual Sharpe.
    """
    returns = port_value.pct_change().dropna()
    if len(returns) < 10:
        return 1.0

    actual_sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

    perm_sharpes = []
    for _ in range(n_perms):
        shuffled = returns.sample(frac=1, replace=False).values
        s_mean = np.mean(shuffled) * 252
        s_std = np.std(shuffled) * np.sqrt(252)
        perm_sharpe = s_mean / s_std if s_std > 0 else 0
        perm_sharpes.append(perm_sharpe)

    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= actual_sharpe).mean()

    return float(p_value)


def compute_benchmark(closes):
    """Compute buy-and-hold SPY benchmark."""
    oot_mask = (closes.index >= OOT_START) & (closes.index <= OOT_END)
    spy = closes.loc[oot_mask, BENCHMARK]
    bench_value = ACCOUNT_SIZE * (spy / spy.iloc[0])
    return bench_value


def validate_5_gates(metrics, regime_metrics, p_value, n_trades):
    """Apply 5-gate validation."""
    gates = {}

    gates['sharpe_gt_0.5'] = {
        'pass': metrics['sharpe'] > 0.5,
        'value': metrics['sharpe'],
        'threshold': 0.5,
    }
    gates['perm_p_lt_0.05'] = {
        'pass': p_value < 0.05,
        'value': round(p_value, 4),
        'threshold': 0.05,
    }
    gates['regime_gap_lt_0.5'] = {
        'pass': regime_metrics['regime_gap'] < 0.5,
        'value': regime_metrics['regime_gap'],
        'threshold': 0.5,
    }
    gates['maxdd_gt_neg50'] = {
        'pass': metrics['max_drawdown_pct'] > -50,
        'value': metrics['max_drawdown_pct'],
        'threshold': -50,
    }
    gates['trades_gte_20'] = {
        'pass': n_trades >= 20,
        'value': n_trades,
        'threshold': 20,
    }

    all_pass = all(g['pass'] for g in gates.values())
    n_pass = sum(1 for g in gates.values() if g['pass'])

    return gates, all_pass, n_pass


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("VOLATILITY REGIME SWITCHING BACKTEST")
    print(f"OOT Period: {OOT_START} to {OOT_END}")
    print(f"Account Size: ${ACCOUNT_SIZE}")
    print("=" * 80)

    closes = download_data()
    features = compute_features(closes)

    available = [t for t in ALL_TICKERS if t in closes.columns]
    missing = [t for t in ALL_TICKERS if t not in closes.columns]
    if missing:
        print(f"  WARNING: Missing tickers: {missing}")
    print(f"  Available: {available}")

    bench_value = compute_benchmark(closes)
    bench_metrics = compute_metrics(bench_value, closes)

    strategies = {
        'A) Binary Vol Switch': strategy_a_binary_switch,
        'B) Gradual Vol Allocation': strategy_b_gradual_allocation,
        'C) Vol Breakout': strategy_c_vol_breakout,
        'D) Vol Mean Reversion': strategy_d_vol_mean_reversion,
        'E) Vol-Weighted Momentum': strategy_e_vol_weighted_momentum,
        'F) VIX Contango Proxy': strategy_f_vix_contango,
    }

    results = {}

    for name, strategy_fn in strategies.items():
        print(f"\n{'─' * 60}")
        print(f"Running: {name}")

        positions = strategy_fn(features, closes)
        port_value, n_trades = run_backtest(positions, closes)

        if len(port_value) < 10:
            print(f"  SKIP: Insufficient data")
            continue

        metrics = compute_metrics(port_value, closes)
        if metrics is None:
            print(f"  SKIP: Could not compute metrics")
            continue

        regime_metrics = compute_regime_metrics(port_value, features)

        print(f"  Running permutation test ({N_PERMUTATIONS} iterations)...")
        p_value = permutation_test(port_value, features, closes)

        gates, all_pass, n_pass = validate_5_gates(metrics, regime_metrics, p_value, n_trades)
        metrics['n_trades'] = n_trades

        results[name] = {
            'metrics': metrics,
            'regime': regime_metrics,
            'p_value': round(p_value, 4),
            'gates': gates,
            'all_gates_pass': all_pass,
            'gates_passed': n_pass,
        }

        status = "PASS" if all_pass else f"FAIL ({n_pass}/5)"
        print(f"  Sharpe: {metrics['sharpe']:.3f} | Sortino: {metrics['sortino']:.3f} | "
              f"Return: {metrics['total_return_pct']:.1f}% | MaxDD: {metrics['max_drawdown_pct']:.1f}%")
        print(f"  Bull Sharpe: {regime_metrics['sharpe_bull']:.3f} | Bear Sharpe: {regime_metrics['sharpe_bear']:.3f} | "
              f"Regime Gap: {regime_metrics['regime_gap']:.3f}")
        print(f"  Trades: {n_trades} | p-value: {p_value:.4f} | Final: ${metrics['final_value']:.2f}")
        print(f"  5-Gate: {status}")

    # ─── Summary Table ───────────────────────────────────────────────────
    print("\n" + "=" * 120)
    print("SUMMARY TABLE")
    print("=" * 120)

    header = f"{'Strategy':<30} {'Sharpe':>7} {'Sortino':>8} {'Return%':>8} {'MaxDD%':>7} {'RegGap':>7} {'p-val':>6} {'Trades':>7} {'Final$':>8} {'Gates':>6} {'Status':>6}"
    print(header)
    print("─" * 120)

    if bench_metrics:
        print(f"{'SPY Buy&Hold (benchmark)':<30} {bench_metrics['sharpe']:>7.3f} {bench_metrics['sortino']:>8.3f} "
              f"{bench_metrics['total_return_pct']:>7.1f}% {bench_metrics['max_drawdown_pct']:>6.1f}% "
              f"{'N/A':>7} {'N/A':>6} {'N/A':>7} {bench_metrics['final_value']:>7.2f} {'N/A':>6} {'REF':>6}")

    print("─" * 120)

    passed_strategies = []

    for name, res in results.items():
        m = res['metrics']
        r = res['regime']
        status = "PASS" if res['all_gates_pass'] else "FAIL"
        gates_str = f"{res['gates_passed']}/5"

        print(f"{name:<30} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{m['total_return_pct']:>7.1f}% {m['max_drawdown_pct']:>6.1f}% "
              f"{r['regime_gap']:>7.3f} {res['p_value']:>6.4f} {m['n_trades']:>7} "
              f"{m['final_value']:>7.2f} {gates_str:>6} {status:>6}")

        if res['all_gates_pass']:
            passed_strategies.append(name)

    print("─" * 120)

    # Gate details
    print("\n" + "=" * 80)
    print("GATE DETAILS")
    print("=" * 80)

    for name, res in results.items():
        gates = res['gates']
        gate_strs = []
        for gname, g in gates.items():
            symbol = "Y" if g['pass'] else "X"
            gate_strs.append(f"{symbol} {gname}={g['value']}")
        print(f"{name}: {' | '.join(gate_strs)}")

    # Regime breakdown
    print("\n" + "=" * 80)
    print("REGIME BREAKDOWN")
    print("=" * 80)
    print(f"{'Strategy':<30} {'Bull Sharpe':>12} {'Bear Sharpe':>12} {'Bull Days':>10} {'Bear Days':>10} {'Gap':>6}")
    print("─" * 80)
    for name, res in results.items():
        r = res['regime']
        print(f"{name:<30} {r['sharpe_bull']:>12.3f} {r['sharpe_bear']:>12.3f} "
              f"{r['bull_days']:>10} {r['bear_days']:>10} {r['regime_gap']:>6.3f}")

    # Final verdict
    print("\n" + "=" * 80)
    if passed_strategies:
        print(f"PASSED STRATEGIES ({len(passed_strategies)}/6):")
        for s in passed_strategies:
            m = results[s]['metrics']
            print(f"  * {s} -- Sharpe {m['sharpe']:.3f}, Sortino {m['sortino']:.3f}, "
                  f"Return {m['total_return_pct']:.1f}%, MaxDD {m['max_drawdown_pct']:.1f}%")
    else:
        print("NO STRATEGIES PASSED ALL 5 GATES")
    print("=" * 80)

    # ─── Save Results ────────────────────────────────────────────────────
    output = {
        'metadata': {
            'backtest_type': 'volatility_regime_switching',
            'oot_start': OOT_START,
            'oot_end': OOT_END,
            'account_size': ACCOUNT_SIZE,
            'slippage_pct': SLIPPAGE_PCT,
            'n_permutations': N_PERMUTATIONS,
            'run_timestamp': datetime.now().isoformat(),
        },
        'benchmark': bench_metrics,
        'strategies': results,
        'passed_strategies': passed_strategies,
        'n_passed': len(passed_strategies),
    }

    output_path = Path('/home/jupiter/Lvl3Quant/data/vol_regime_switch_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")

    return output


if __name__ == '__main__':
    main()
