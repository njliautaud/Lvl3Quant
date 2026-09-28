#!/usr/bin/env python3
"""
Adaptive Drawdown Protection Overlay for v4.4 + ML Vol Targeting
================================================================
Instead of PREDICTING drawdowns (failed - perm p=0.284), this REACTS:
- When portfolio enters drawdown, tighten position sizing
- Use trailing stops that adapt to current realized volatility
- Progressive de-risking: mild DD = slight reduction, deep DD = aggressive reduction

Base strategy: v4.4 regime (when to hold UPRO) + ML vol targeting (how much UPRO)
Overlay: adaptive position scaling based on current drawdown depth + trailing stop

HC #713: Fixed capital, no DCA
HC #714: Income + growth focus
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/adaptive_dd_protection')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL = 100_000

def download_data():
    """Download required ETF + indicator data"""
    tickers = ['SPY', 'UPRO', 'SHY', 'TLT', 'GLD', 'UUP', 'HYG', 'IEF']
    vix = yf.download('^VIX', start='2010-01-01', progress=False)['Close']
    prices = yf.download(tickers, start='2010-01-01', progress=False)['Close']

    # Flatten multi-index if needed
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)

    prices['VIX'] = vix
    prices = prices.dropna()
    print(f"[DATA] {len(prices)} rows, {prices.columns.tolist()}")
    print(f"[DATA] Range: {prices.index[0].strftime('%Y-%m-%d')} → {prices.index[-1].strftime('%Y-%m-%d')}")
    return prices


def compute_v44_signal(prices):
    """v4.4 regime signal: SMA crossover for risk-on/off"""
    spy = prices['SPY']
    sma_50 = spy.rolling(50).mean()
    sma_200 = spy.rolling(200).mean()

    # v4.4: risk-on when 50 SMA > 200 SMA
    signal = (sma_50 > sma_200).astype(float)
    return signal


def compute_vol_target(prices, lookback=21, target_vol=0.15):
    """ML vol targeting approximation: scale position by inverse realized vol"""
    spy_ret = prices['SPY'].pct_change()
    realized_vol = spy_ret.rolling(lookback).std() * np.sqrt(252)

    # Target vol scaling: when vol is high, reduce; when low, increase
    vol_scale = target_vol / realized_vol.clip(lower=0.05)
    vol_scale = vol_scale.clip(0.0, 2.0)  # Cap at 2x leverage

    return vol_scale


def run_strategy(prices, dd_protection='none', dd_params=None):
    """
    Run v4.4 + vol targeting with optional drawdown protection overlay.

    dd_protection modes:
    - 'none': pure v4.4 + vol targeting
    - 'linear': linear scaling (deeper DD = less exposure)
    - 'exponential': exponential scaling (fast de-risk in deep DD)
    - 'trailing_stop': vol-adjusted trailing stop
    - 'combined': linear + trailing stop
    """
    if dd_params is None:
        dd_params = {}

    v44_signal = compute_v44_signal(prices)
    vol_scale = compute_vol_target(prices)

    upro_ret = prices['UPRO'].pct_change()
    shy_ret = prices['SHY'].pct_change()
    tlt_ret = prices['TLT'].pct_change()
    spy_ret = prices['SPY'].pct_change()

    # Start after warmup
    start_idx = 220  # After 200 SMA warmup + buffer

    equity = np.ones(len(prices)) * INITIAL_CAPITAL
    weights = np.zeros(len(prices))  # Track actual UPRO weight
    dd_scale_history = np.ones(len(prices))  # Track DD overlay scaling

    hwm = INITIAL_CAPITAL  # High water mark
    trailing_stop_level = 0.0

    for i in range(start_idx, len(prices)):
        # Base signal: v4.4 regime * vol targeting
        base_weight = v44_signal.iloc[i] * vol_scale.iloc[i]

        # Current drawdown from HWM
        current_dd = (equity[i-1] - hwm) / hwm  # Negative when in drawdown

        # Apply drawdown protection overlay
        dd_multiplier = 1.0

        if dd_protection in ('linear', 'combined'):
            # Linear de-risking: at -5% DD, reduce by 25%; at -20%, reduce by 100%
            dd_threshold = dd_params.get('dd_start', -0.05)
            dd_max = dd_params.get('dd_max', -0.20)

            if current_dd < dd_threshold:
                # Linear scale from 1.0 at threshold to 0.0 at dd_max
                dd_pct = (current_dd - dd_threshold) / (dd_max - dd_threshold)
                dd_multiplier = max(0.0, 1.0 - dd_pct)

        elif dd_protection == 'exponential':
            dd_threshold = dd_params.get('dd_start', -0.05)
            decay = dd_params.get('decay', 10.0)

            if current_dd < dd_threshold:
                dd_depth = abs(current_dd - dd_threshold)
                dd_multiplier = np.exp(-decay * dd_depth)

        # Trailing stop check
        if dd_protection in ('trailing_stop', 'combined'):
            # Vol-adjusted trailing stop
            trail_mult = dd_params.get('trail_mult', 2.0)
            realized_vol_20d = spy_ret.iloc[max(0,i-20):i].std() * np.sqrt(252)
            stop_distance = trail_mult * realized_vol_20d / np.sqrt(252)  # Daily

            if equity[i-1] >= hwm:
                trailing_stop_level = hwm * (1.0 - stop_distance)
            else:
                # Don't lower the stop
                trailing_stop_level = max(trailing_stop_level, equity[i-1] * (1.0 - stop_distance * 0.5))

            if equity[i-1] < trailing_stop_level and base_weight > 0:
                dd_multiplier *= 0.25  # Reduce to 25% when trailing stop hit

        # Final weight
        final_weight = base_weight * dd_multiplier
        final_weight = max(0.0, min(final_weight, 2.0))

        weights[i] = final_weight
        dd_scale_history[i] = dd_multiplier

        # Portfolio return: weighted UPRO + remainder in SHY (or TLT for hedge)
        hedge_asset = dd_params.get('hedge_asset', 'shy')
        if hedge_asset == 'tlt' and current_dd < -0.10:
            # Switch to TLT hedge in deep drawdowns
            port_ret = final_weight * upro_ret.iloc[i] + (1 - final_weight) * tlt_ret.iloc[i]
        else:
            port_ret = final_weight * upro_ret.iloc[i] + (1 - final_weight) * shy_ret.iloc[i]

        equity[i] = equity[i-1] * (1 + port_ret)

        # Update HWM
        if equity[i] > hwm:
            hwm = equity[i]
            trailing_stop_level = 0  # Reset

    # Trim to active period
    equity = equity[start_idx:]
    weights = weights[start_idx:]
    dd_scale_history = dd_scale_history[start_idx:]
    dates = prices.index[start_idx:]

    return pd.Series(equity, index=dates), pd.Series(weights, index=dates), pd.Series(dd_scale_history, index=dates)


def compute_metrics(equity_series, name="Strategy"):
    """Compute risk-adjusted metrics"""
    returns = equity_series.pct_change().dropna()

    ann_ret = (equity_series.iloc[-1] / equity_series.iloc[0]) ** (252 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside > 0 else 0

    # Max drawdown
    cummax = equity_series.cummax()
    drawdown = (equity_series - cummax) / cummax
    max_dd = drawdown.min()

    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    # Win rate (daily)
    wr = (returns > 0).mean()

    # Profit factor
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')

    return {
        'name': name,
        'sharpe': sharpe,
        'sortino': sortino,
        'cagr': ann_ret,
        'max_dd': max_dd,
        'calmar': calmar,
        'win_rate': wr,
        'profit_factor': pf,
        'ann_vol': ann_vol,
        'final_equity': equity_series.iloc[-1]
    }


def regime_analysis(equity_series, spy_prices):
    """R1 regime test: check performance in green vs red regimes"""
    returns = equity_series.pct_change().dropna()
    spy_ret = spy_prices.reindex(returns.index).pct_change().dropna()

    # Align
    common = returns.index.intersection(spy_ret.index)
    returns = returns.loc[common]
    spy_ret = spy_ret.loc[common]

    green_mask = spy_ret > 0
    red_mask = spy_ret <= 0

    green_sharpe = returns[green_mask].mean() / returns[green_mask].std() * np.sqrt(252) if green_mask.sum() > 50 else 0
    red_sharpe = returns[red_mask].mean() / returns[red_mask].std() * np.sqrt(252) if red_mask.sum() > 50 else 0

    gap = abs(green_sharpe - red_sharpe) / max(abs(green_sharpe), abs(red_sharpe), 0.01)

    return {
        'green_sharpe': green_sharpe,
        'red_sharpe': red_sharpe,
        'green_days': green_mask.sum(),
        'red_days': red_mask.sum(),
        'gap': gap,
        'pass': gap <= 0.50
    }


def permutation_test(equity_series, prices, n_perms=200):
    """Permutation test: shuffle regime signal, check if random is equally profitable"""
    real_metrics = compute_metrics(equity_series, "Real")
    real_sharpe = real_metrics['sharpe']

    v44_signal = compute_v44_signal(prices)
    vol_scale = compute_vol_target(prices)
    upro_ret = prices['UPRO'].pct_change()
    shy_ret = prices['SHY'].pct_change()

    start_idx = 220
    perm_sharpes = []

    for p in range(n_perms):
        # Shuffle the v4.4 signal dates (preserving distribution)
        shuffled_signal = v44_signal.copy()
        shuffled_signal.iloc[start_idx:] = np.random.permutation(shuffled_signal.iloc[start_idx:].values)

        eq = np.ones(len(prices)) * INITIAL_CAPITAL
        for i in range(start_idx, len(prices)):
            w = shuffled_signal.iloc[i] * vol_scale.iloc[i]
            w = max(0, min(w, 2.0))
            ret = w * upro_ret.iloc[i] + (1-w) * shy_ret.iloc[i]
            eq[i] = eq[i-1] * (1 + ret)

        eq_series = pd.Series(eq[start_idx:], index=prices.index[start_idx:])
        pm = compute_metrics(eq_series, f"perm_{p}")
        perm_sharpes.append(pm['sharpe'])

    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= real_sharpe).mean()

    return {
        'real_sharpe': real_sharpe,
        'perm_mean': perm_sharpes.mean(),
        'perm_std': perm_sharpes.std(),
        'p_value': p_value,
        'pass': p_value < 0.05
    }


def sub_period_test(equity_series, n_blocks=4):
    """Sub-period consistency: split into blocks, check CV of Sharpe"""
    returns = equity_series.pct_change().dropna()
    block_size = len(returns) // n_blocks

    block_sharpes = []
    for b in range(n_blocks):
        start = b * block_size
        end = (b+1) * block_size if b < n_blocks-1 else len(returns)
        block_ret = returns.iloc[start:end]

        ann_ret = block_ret.mean() * 252
        ann_vol = block_ret.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        block_sharpes.append(sharpe)

    cv = np.std(block_sharpes) / max(np.mean(block_sharpes), 0.01)

    return {
        'block_sharpes': block_sharpes,
        'mean': np.mean(block_sharpes),
        'cv': cv,
        'pass': cv < 0.50
    }


def main():
    print("=" * 80)
    print("ADAPTIVE DRAWDOWN PROTECTION OVERLAY")
    print("Base: v4.4 + ML Vol Targeting | Overlay: reactive DD protection")
    print("=" * 80)

    # Step 1: Download data
    print("\n" + "=" * 80)
    print("STEP 1: DOWNLOADING DATA")
    print("=" * 80)
    prices = download_data()

    # Step 2: Define configurations to test
    configs = {
        'baseline': {
            'dd_protection': 'none',
            'dd_params': {}
        },
        'linear_mild': {
            'dd_protection': 'linear',
            'dd_params': {'dd_start': -0.05, 'dd_max': -0.25}
        },
        'linear_aggressive': {
            'dd_protection': 'linear',
            'dd_params': {'dd_start': -0.03, 'dd_max': -0.15}
        },
        'exponential': {
            'dd_protection': 'exponential',
            'dd_params': {'dd_start': -0.05, 'decay': 8.0}
        },
        'trailing_stop_tight': {
            'dd_protection': 'trailing_stop',
            'dd_params': {'trail_mult': 1.5}
        },
        'trailing_stop_wide': {
            'dd_protection': 'trailing_stop',
            'dd_params': {'trail_mult': 2.5}
        },
        'combined_balanced': {
            'dd_protection': 'combined',
            'dd_params': {'dd_start': -0.05, 'dd_max': -0.20, 'trail_mult': 2.0}
        },
        'combined_aggressive': {
            'dd_protection': 'combined',
            'dd_params': {'dd_start': -0.03, 'dd_max': -0.15, 'trail_mult': 1.5}
        },
        'combined_tlt_hedge': {
            'dd_protection': 'combined',
            'dd_params': {'dd_start': -0.05, 'dd_max': -0.20, 'trail_mult': 2.0, 'hedge_asset': 'tlt'}
        },
    }

    # Step 3: Run all configurations
    print("\n" + "=" * 80)
    print("STEP 2: RUNNING CONFIGURATIONS")
    print("=" * 80)

    results = {}
    all_metrics = []

    for name, cfg in configs.items():
        print(f"\n  [{name}]...")
        equity, weights, dd_scales = run_strategy(
            prices,
            dd_protection=cfg['dd_protection'],
            dd_params=cfg['dd_params']
        )

        metrics = compute_metrics(equity, name)
        regime = regime_analysis(equity, prices['SPY'])

        results[name] = {
            'equity': equity,
            'weights': weights,
            'dd_scales': dd_scales,
            'metrics': metrics,
            'regime': regime
        }

        metrics['r1_pass'] = regime['pass']
        metrics['green_sharpe'] = regime['green_sharpe']
        metrics['red_sharpe'] = regime['red_sharpe']
        metrics['regime_gap'] = regime['gap']
        all_metrics.append(metrics)

        print(f"    Sharpe {metrics['sharpe']:.3f}, CAGR {metrics['cagr']:.1%}, "
              f"MaxDD {metrics['max_dd']:.1%}, Sortino {metrics['sortino']:.3f}")
        print(f"    R1: green={regime['green_sharpe']:.3f}, red={regime['red_sharpe']:.3f}, "
              f"gap={regime['gap']:.3f} {'PASS' if regime['pass'] else 'FAIL'}")

    # Step 4: Results comparison
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)

    print(f"\n  {'Config':<25s} {'Sharpe':>7s} {'Sortino':>8s} {'CAGR':>7s} {'MaxDD':>7s} "
          f"{'Calmar':>7s} {'R1':>5s}")
    print(f"  {'-'*25} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*5}")

    for m in all_metrics:
        print(f"  {m['name']:<25s} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{m['cagr']:>6.1%} {m['max_dd']:>6.1%} {m['calmar']:>7.3f} "
              f"{'PASS' if m['r1_pass'] else 'FAIL':>5s}")

    # Step 5: Find best config (highest Sharpe that improves MaxDD vs baseline)
    baseline_metrics = results['baseline']['metrics']

    improved = []
    for name, r in results.items():
        if name == 'baseline':
            continue
        m = r['metrics']
        # Must improve MaxDD without destroying Sharpe
        dd_improvement = m['max_dd'] - baseline_metrics['max_dd']  # Less negative = better
        sharpe_cost = baseline_metrics['sharpe'] - m['sharpe']

        if dd_improvement > 0.01:  # At least 1pp MaxDD improvement
            improved.append({
                'name': name,
                'dd_improvement': dd_improvement,
                'sharpe_cost': sharpe_cost,
                'efficiency': dd_improvement / max(sharpe_cost, 0.001),  # DD reduction per unit Sharpe cost
                'metrics': m,
                'regime': r['regime']
            })

    improved.sort(key=lambda x: x['efficiency'], reverse=True)

    print("\n" + "=" * 80)
    print("DRAWDOWN IMPROVEMENT ANALYSIS (vs baseline)")
    print("=" * 80)

    if improved:
        print(f"\n  {'Config':<25s} {'DD Improv':>10s} {'Sharpe Cost':>12s} {'Efficiency':>11s}")
        print(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*11}")
        for imp in improved:
            print(f"  {imp['name']:<25s} {imp['dd_improvement']:>+9.1%} "
                  f"{imp['sharpe_cost']:>+11.3f} {imp['efficiency']:>10.1f}")

        best = improved[0]
        print(f"\n  BEST: {best['name']} (most DD reduction per Sharpe cost)")
    else:
        print("\n  No configuration improved MaxDD by >1pp. Baseline may already be well-protected.")
        best = None

    # Step 6: Adversarial validation on best config (or baseline if none improved)
    test_name = best['name'] if best else 'baseline'
    test_equity = results[test_name]['equity']

    print("\n" + "=" * 80)
    print(f"ADVERSARIAL VALIDATION: {test_name}")
    print("=" * 80)

    print(f"\n  [1/3] Permutation test (200 shuffles)...")
    perm = permutation_test(test_equity, prices, n_perms=200)
    print(f"    Real Sharpe: {perm['real_sharpe']:.3f}")
    print(f"    Perm mean: {perm['perm_mean']:.3f} ± {perm['perm_std']:.3f}")
    print(f"    p-value: {perm['p_value']:.3f} → {'PASS' if perm['pass'] else 'FAIL'}")

    print(f"\n  [2/3] Sub-period consistency...")
    subp = sub_period_test(test_equity)
    for i, s in enumerate(subp['block_sharpes']):
        print(f"    Block {i+1}: Sharpe {s:.3f}")
    print(f"    CV: {subp['cv']:.3f} → {'PASS' if subp['pass'] else 'FAIL'}")

    print(f"\n  [3/3] R1 regime test...")
    regime = results[test_name]['regime']
    print(f"    Green Sharpe: {regime['green_sharpe']:.3f} ({regime['green_days']} days)")
    print(f"    Red Sharpe: {regime['red_sharpe']:.3f} ({regime['red_days']} days)")
    print(f"    Gap: {regime['gap']:.3f} → {'PASS' if regime['pass'] else 'FAIL'}")

    gates_passed = sum([perm['pass'], subp['pass'], regime['pass']])
    print(f"\n  ADVERSARIAL SUMMARY: {gates_passed}/3 gates passed → {'PASS' if gates_passed >= 2 else 'FAIL'}")

    # Step 7: Drawdown episode analysis
    print("\n" + "=" * 80)
    print("DRAWDOWN EPISODE ANALYSIS")
    print("=" * 80)

    for config_name in ['baseline', test_name] if test_name != 'baseline' else ['baseline']:
        eq = results[config_name]['equity']
        cummax = eq.cummax()
        dd = (eq - cummax) / cummax

        # Find drawdown episodes > 10%
        in_dd = dd < -0.10
        episodes = []
        start = None
        for i in range(len(dd)):
            if in_dd.iloc[i] and start is None:
                start = dd.index[i]
            elif not in_dd.iloc[i] and start is not None:
                end = dd.index[i]
                max_dd = dd.loc[start:end].min()
                duration = (end - start).days
                episodes.append({'start': start, 'end': end, 'max_dd': max_dd, 'duration': duration})
                start = None

        print(f"\n  [{config_name}] Drawdown episodes > 10%: {len(episodes)}")
        for ep in episodes[:8]:
            print(f"    {ep['start'].strftime('%Y-%m')} → {ep['end'].strftime('%Y-%m')}: "
                  f"{ep['max_dd']:.1%} over {ep['duration']}d")

    # Save results
    summary = pd.DataFrame(all_metrics)
    summary.to_csv(OUTPUT_DIR / 'config_comparison.csv', index=False)

    print("\n" + "=" * 80)
    print(f"COMPLETE")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == '__main__':
    main()
