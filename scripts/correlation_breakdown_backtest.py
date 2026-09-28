#!/usr/bin/env python3
"""
Cross-Sector Correlation Breakdown Backtest
============================================
Theory: When two normally-correlated sectors suddenly de-correlate, the lagging
sector typically catches up within 3-10 days. We trade the lagger in the
direction of the leader's move.

Validation: 5-gate system (Sharpe, permutation, regime, trade count, drawdown).
"""

import json
import warnings
import itertools
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

# ── CONFIG ──────────────────────────────────────────────────────────────────
SECTOR_ETFS = ['XLE', 'XLU', 'XLP', 'XLF', 'XLK', 'XLC', 'XLV', 'XLY', 'XLRE', 'XLB', 'XLI']
BENCHMARK = 'SPY'
VIX_TICKER = '^VIX'

CORR_WINDOW = 20       # rolling correlation window
BASELINE_WINDOW = 60   # baseline average correlation window
BREAKDOWN_THRESHOLD = 0.30  # min drop from baseline to trigger signal
LEADER_LOOKBACK = 5    # days to determine leader/lagger

COST_RT_PCT = 0.10 / 100  # 0.10% round-trip cost
LOOKBACK_DAYS = 252    # sliding window for baseline (HC #0)

HOLD_PERIODS = [3, 5, 10]
N_PERMUTATIONS = 1000

# 5-gate thresholds
GATE_SHARPE = 0.5
GATE_PVAL = 0.05
GATE_REGIME_GAP = 0.50
GATE_MIN_TRADES = 50
GATE_MAX_DD = 0.40

SECTOR_GROUPS = {
    'cyclical': ['XLF', 'XLK', 'XLC', 'XLY', 'XLI', 'XLB'],
    'defensive': ['XLU', 'XLP', 'XLV'],
    'other': ['XLE', 'XLRE']
}

OUTPUT_PATH = Path('/home/jupiter/Lvl3Quant/output/correlation_breakdown_results.json')


def download_data():
    """Download 5 years of daily data for sector ETFs + SPY + VIX."""
    tickers = SECTOR_ETFS + [BENCHMARK, VIX_TICKER]
    print(f"Downloading data for {len(tickers)} tickers...")
    data = yf.download(tickers, period='5y', auto_adjust=True, progress=False)

    close = data['Close'].copy()
    volume = data['Volume'].copy()

    # Drop rows with too many NaNs
    close = close.dropna(thresh=len(SECTOR_ETFS))
    volume = volume.reindex(close.index)

    # Rename VIX column
    if '^VIX' in close.columns:
        close = close.rename(columns={'^VIX': 'VIX'})
        volume = volume.rename(columns={'^VIX': 'VIX'})

    print(f"Data range: {close.index[0].date()} to {close.index[-1].date()} ({len(close)} days)")
    return close, volume


def compute_returns(close):
    """Compute daily returns."""
    return close.pct_change()


def compute_rolling_correlations(returns, sector_etfs):
    """Compute 20d rolling pairwise correlations for all 55 pairs."""
    pairs = list(itertools.combinations(sector_etfs, 2))
    print(f"Computing rolling correlations for {len(pairs)} pairs...")

    corr_data = {}
    for s1, s2 in pairs:
        pair_key = f"{s1}_{s2}"
        rolling_corr = returns[s1].rolling(CORR_WINDOW).corr(returns[s2])
        baseline_corr = rolling_corr.rolling(BASELINE_WINDOW).mean()
        corr_data[pair_key] = {
            'rolling': rolling_corr,
            'baseline': baseline_corr,
            'drop': rolling_corr - baseline_corr,
            's1': s1, 's2': s2
        }

    return pairs, corr_data


def detect_breakdowns(corr_data, returns, close, volume, pairs):
    """Detect correlation breakdown events and generate signals."""
    signals = []

    for pair_key, cd in corr_data.items():
        s1, s2 = cd['s1'], cd['s2']
        drop = cd['drop']
        baseline = cd['baseline']
        rolling = cd['rolling']

        # Find breakdown events: rolling corr drops > threshold below baseline
        breakdown_mask = drop < -BREAKDOWN_THRESHOLD
        breakdown_dates = drop.index[breakdown_mask]

        for date in breakdown_dates:
            idx = returns.index.get_loc(date)
            if idx < LEADER_LOOKBACK + BASELINE_WINDOW + CORR_WINDOW:
                continue

            # Determine leader vs lagger
            ret_s1_5d = returns[s1].iloc[idx-LEADER_LOOKBACK+1:idx+1].sum()
            ret_s2_5d = returns[s2].iloc[idx-LEADER_LOOKBACK+1:idx+1].sum()

            if abs(ret_s1_5d) > abs(ret_s2_5d):
                leader, lagger = s1, s2
                leader_ret = ret_s1_5d
                lagger_ret = ret_s2_5d
            else:
                leader, lagger = s2, s1
                leader_ret = ret_s2_5d
                lagger_ret = ret_s1_5d

            # Signal direction: lagger should move in leader's direction
            direction = 1 if leader_ret > 0 else -1  # +1 = long lagger, -1 = short lagger

            # Get volume info for lagger
            vol_20d_avg = volume[lagger].iloc[idx-20:idx].mean() if lagger in volume.columns else np.nan
            vol_current = volume[lagger].iloc[idx] if lagger in volume.columns else np.nan
            low_volume = vol_current < vol_20d_avg if not np.isnan(vol_20d_avg) else False

            # Get VIX
            vix_val = close['VIX'].iloc[idx] if 'VIX' in close.columns else np.nan

            signals.append({
                'date': date,
                'pair': pair_key,
                'leader': leader,
                'lagger': lagger,
                'leader_5d_ret': leader_ret,
                'lagger_5d_ret': lagger_ret,
                'direction': direction,
                'baseline_corr': baseline.iloc[idx] if not np.isnan(baseline.iloc[idx]) else 0,
                'rolling_corr': rolling.iloc[idx] if not np.isnan(rolling.iloc[idx]) else 0,
                'corr_drop': drop.iloc[idx],
                'low_volume': low_volume,
                'vix': vix_val,
                'idx': idx,
            })

    print(f"Total breakdown events detected: {len(signals)}")
    return signals


def deduplicate_signals(signals):
    """Remove duplicate signals for same lagger on same date (keep strongest breakdown)."""
    from collections import defaultdict
    by_date_lagger = defaultdict(list)
    for s in signals:
        key = (s['date'], s['lagger'])
        by_date_lagger[key].append(s)

    deduped = []
    for key, group in by_date_lagger.items():
        # Keep the one with largest absolute correlation drop
        best = min(group, key=lambda x: x['corr_drop'])
        deduped.append(best)

    deduped.sort(key=lambda x: x['date'])
    return deduped


def compute_trade_returns(signals, returns, hold_period):
    """Compute forward returns for each signal."""
    trade_returns = []
    for sig in signals:
        idx = sig['idx']
        lagger = sig['lagger']
        direction = sig['direction']

        # Forward return over hold period
        if idx + hold_period >= len(returns):
            continue

        fwd_ret = returns[lagger].iloc[idx+1:idx+1+hold_period].sum()
        trade_ret = direction * fwd_ret - COST_RT_PCT

        # SPY return for regime classification
        spy_ret = returns['SPY'].iloc[idx+1:idx+1+hold_period].sum() if 'SPY' in returns.columns else 0

        trade_returns.append({
            **sig,
            'fwd_return': fwd_ret,
            'trade_return': trade_ret,
            'spy_fwd_return': spy_ret,
            'hold_period': hold_period,
        })

    return trade_returns


def calc_metrics(trade_rets_list):
    """Calculate performance metrics from list of trade return dicts."""
    if len(trade_rets_list) == 0:
        return None

    rets = np.array([t['trade_return'] for t in trade_rets_list])
    n = len(rets)

    mean_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1) if n > 1 else 1e-9

    # Annualize (assume ~50 trades/yr rough)
    trades_per_year = max(n / 5, 1)  # rough: 5 years of data
    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0

    # Sortino
    downside = rets[rets < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = (mean_ret / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0

    # Win rate
    wins = np.sum(rets > 0)
    wr = wins / n if n > 0 else 0

    # Profit factor
    gross_profit = np.sum(rets[rets > 0])
    gross_loss = abs(np.sum(rets[rets < 0]))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Avg win / avg loss
    avg_win = np.mean(rets[rets > 0]) if np.any(rets > 0) else 0
    avg_loss = np.mean(rets[rets < 0]) if np.any(rets < 0) else 0

    # Max drawdown (cumulative)
    cum = np.cumsum(rets)
    running_max = np.maximum.accumulate(cum)
    dd = cum - running_max
    max_dd = abs(np.min(dd)) if len(dd) > 0 else 0

    # Regime stratification
    green_rets = [t['trade_return'] for t in trade_rets_list if t.get('spy_fwd_return', 0) > 0]
    red_rets = [t['trade_return'] for t in trade_rets_list if t.get('spy_fwd_return', 0) <= 0]

    green_sharpe = 0
    red_sharpe = 0
    if len(green_rets) > 2:
        g_arr = np.array(green_rets)
        g_std = np.std(g_arr, ddof=1)
        green_sharpe = (np.mean(g_arr) / g_std * np.sqrt(max(len(g_arr)/5, 1))) if g_std > 0 else 0
    if len(red_rets) > 2:
        r_arr = np.array(red_rets)
        r_std = np.std(r_arr, ddof=1)
        red_sharpe = (np.mean(r_arr) / r_std * np.sqrt(max(len(r_arr)/5, 1))) if r_std > 0 else 0

    max_abs = max(abs(green_sharpe), abs(red_sharpe), 1e-9)
    regime_gap = abs(green_sharpe - red_sharpe) / max_abs

    return {
        'n_trades': n,
        'mean_return': float(mean_ret),
        'total_return': float(np.sum(rets)),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'win_rate': float(wr),
        'profit_factor': float(pf),
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
        'max_drawdown': float(max_dd),
        'green_sharpe': float(green_sharpe),
        'red_sharpe': float(red_sharpe),
        'regime_gap': float(regime_gap),
        'n_green': len(green_rets),
        'n_red': len(red_rets),
    }


def permutation_test(trade_rets_list, n_perms=N_PERMUTATIONS):
    """Permutation test: shuffle signal directions, compute null Sharpe distribution."""
    if len(trade_rets_list) < 10:
        return 1.0

    actual_rets = np.array([t['trade_return'] for t in trade_rets_list])
    actual_sharpe = np.mean(actual_rets) / (np.std(actual_rets, ddof=1) + 1e-9)

    # For permutation: shuffle which direction we trade (randomize the signal)
    raw_fwd = np.array([t['fwd_return'] for t in trade_rets_list])
    directions = np.array([t['direction'] for t in trade_rets_list])

    null_sharpes = np.zeros(n_perms)
    rng = np.random.default_rng(42)

    for i in range(n_perms):
        shuffled_dirs = rng.choice([-1, 1], size=len(directions))
        perm_rets = shuffled_dirs * raw_fwd - COST_RT_PCT
        perm_std = np.std(perm_rets, ddof=1)
        null_sharpes[i] = np.mean(perm_rets) / (perm_std + 1e-9)

    p_value = np.mean(null_sharpes >= actual_sharpe)
    return float(p_value)


def apply_variant_filter(trade_rets, variant):
    """Filter trades based on variant."""
    if variant == 'A':
        # Pure correlation breakdown - no filter
        return trade_rets
    elif variant == 'B':
        # High-base-correlation pairs only (baseline > 0.60)
        return [t for t in trade_rets if t['baseline_corr'] > 0.60]
    elif variant == 'C':
        # Volume-confirmed (lagger has below-average volume)
        return [t for t in trade_rets if t.get('low_volume', False)]
    elif variant == 'D':
        # Regime-filtered (VIX < 20)
        return [t for t in trade_rets if t.get('vix', 99) < 20]
    return trade_rets


def analyze_pair_performance(trade_rets):
    """Analyze which pairs show strongest mean-reversion."""
    from collections import defaultdict
    pair_stats = defaultdict(list)

    for t in trade_rets:
        pair_stats[t['pair']].append(t['trade_return'])

    results = {}
    for pair, rets in pair_stats.items():
        if len(rets) < 5:
            continue
        arr = np.array(rets)
        results[pair] = {
            'n_trades': len(rets),
            'mean_return': float(np.mean(arr)),
            'win_rate': float(np.mean(arr > 0)),
            'total_return': float(np.sum(arr)),
        }

    return dict(sorted(results.items(), key=lambda x: x[1]['mean_return'], reverse=True))


def analyze_recorrelation_halflife(corr_data, signals):
    """Estimate average time to re-correlate after breakdown."""
    half_lives = []

    for sig in signals[:200]:  # sample for speed
        pair_key = sig['pair']
        idx = sig['idx']
        cd = corr_data[pair_key]
        drop_series = cd['drop']

        # Starting drop magnitude
        start_drop = sig['corr_drop']
        half_target = start_drop / 2  # halfway back to baseline

        # Look forward up to 60 days
        for fwd in range(1, 61):
            if idx + fwd >= len(drop_series):
                break
            if drop_series.iloc[idx + fwd] > half_target:
                half_lives.append(fwd)
                break

    if half_lives:
        return {
            'mean_halflife': float(np.mean(half_lives)),
            'median_halflife': float(np.median(half_lives)),
            'p25': float(np.percentile(half_lives, 25)),
            'p75': float(np.percentile(half_lives, 75)),
            'n_samples': len(half_lives),
        }
    return {'mean_halflife': None, 'n_samples': 0}


def analyze_sector_groupings(trade_rets):
    """Analyze if signal works better for cyclical vs defensive pairs."""
    def get_group(ticker):
        for group, members in SECTOR_GROUPS.items():
            if ticker in members:
                return group
        return 'other'

    grouping_stats = {}
    for combo in ['cyclical-cyclical', 'defensive-defensive', 'cyclical-defensive', 'other']:
        grouping_stats[combo] = []

    for t in trade_rets:
        g1 = get_group(t['leader'])
        g2 = get_group(t['lagger'])
        groups = sorted([g1, g2])
        key = f"{groups[0]}-{groups[1]}"
        if key not in grouping_stats:
            key = 'other'
        grouping_stats.setdefault(key, []).append(t['trade_return'])

    results = {}
    for key, rets in grouping_stats.items():
        if len(rets) < 5:
            results[key] = {'n_trades': len(rets), 'insufficient_data': True}
            continue
        arr = np.array(rets)
        results[key] = {
            'n_trades': len(rets),
            'mean_return': float(np.mean(arr)),
            'win_rate': float(np.mean(arr > 0)),
            'sharpe_approx': float(np.mean(arr) / (np.std(arr, ddof=1) + 1e-9)),
        }

    return results


def validate_5_gates(metrics, p_value):
    """Apply 5-gate validation."""
    if metrics is None:
        return {'passed': False, 'reason': 'no metrics'}

    gates = {
        'gate1_sharpe': metrics['sharpe'] > GATE_SHARPE,
        'gate2_pvalue': p_value < GATE_PVAL,
        'gate3_regime': metrics['regime_gap'] < GATE_REGIME_GAP,
        'gate4_trades': metrics['n_trades'] > GATE_MIN_TRADES,
        'gate5_drawdown': metrics['max_drawdown'] < GATE_MAX_DD,
    }
    gates['all_passed'] = all(gates.values())

    return gates


def main():
    print("=" * 80)
    print("CROSS-SECTOR CORRELATION BREAKDOWN BACKTEST")
    print("=" * 80)
    print()

    # 1. Download data
    close, volume = download_data()
    returns = compute_returns(close)

    # 2. Compute rolling correlations
    pairs, corr_data = compute_rolling_correlations(returns, SECTOR_ETFS)

    # 3. Detect breakdowns
    signals = detect_breakdowns(corr_data, returns, close, volume, pairs)
    signals = deduplicate_signals(signals)
    print(f"After deduplication: {len(signals)} unique signals")
    print()

    # 4. Run all variants x hold periods
    variants = {
        'A': 'Pure correlation breakdown (any pair, drop > 0.30)',
        'B': 'High-base-correlation pairs only (baseline > 0.60)',
        'C': 'Volume-confirmed (lagger below-avg volume)',
        'D': 'Regime-filtered (VIX < 20)',
    }

    all_results = {}
    best_variant = None
    best_sharpe = -999

    for hold in HOLD_PERIODS:
        print(f"\n{'─' * 60}")
        print(f"HOLD PERIOD: {hold} days")
        print(f"{'─' * 60}")

        # Compute base trade returns
        base_trades = compute_trade_returns(signals, returns, hold)

        for var_key, var_desc in variants.items():
            filtered = apply_variant_filter(base_trades, var_key)

            result_key = f"{var_key}_hold{hold}d"

            if len(filtered) < 10:
                print(f"\n  Variant {var_key} ({var_desc}): SKIP ({len(filtered)} trades)")
                all_results[result_key] = {'n_trades': len(filtered), 'skipped': True}
                continue

            metrics = calc_metrics(filtered)
            p_value = permutation_test(filtered)
            gates = validate_5_gates(metrics, p_value)

            all_results[result_key] = {
                'variant': var_key,
                'description': var_desc,
                'hold_period': hold,
                'metrics': metrics,
                'p_value': p_value,
                'gates': gates,
            }

            # Track best
            if metrics and metrics['sharpe'] > best_sharpe:
                best_sharpe = metrics['sharpe']
                best_variant = result_key

            status = "PASS ALL GATES" if gates.get('all_passed') else "FAIL"
            gate_detail = " | ".join([f"G{i+1}:{'Y' if v else 'N'}" for i, (k, v) in enumerate(gates.items()) if k != 'all_passed'])

            print(f"\n  Variant {var_key}: {var_desc}")
            print(f"    Trades: {metrics['n_trades']} | Sharpe: {metrics['sharpe']:.3f} | Sortino: {metrics['sortino']:.3f}")
            print(f"    WR: {metrics['win_rate']:.1%} | PF: {metrics['profit_factor']:.2f} | Total: {metrics['total_return']:.4f}")
            print(f"    Avg Win: {metrics['avg_win']:.4f} | Avg Loss: {metrics['avg_loss']:.4f}")
            print(f"    MaxDD: {metrics['max_drawdown']:.4f} | p-value: {p_value:.4f}")
            print(f"    Regime: Green Sharpe={metrics['green_sharpe']:.3f} Red Sharpe={metrics['red_sharpe']:.3f} Gap={metrics['regime_gap']:.3f}")
            print(f"    Gates: {gate_detail} => {status}")

    # 5. Additional analysis (use best hold period or default 5d)
    print(f"\n\n{'=' * 80}")
    print("ADDITIONAL ANALYSIS")
    print(f"{'=' * 80}")

    base_trades_5d = compute_trade_returns(signals, returns, 5)

    # Pair performance
    print("\n--- Top 10 Pairs by Mean Return (5d hold) ---")
    pair_perf = analyze_pair_performance(base_trades_5d)
    for i, (pair, stats) in enumerate(list(pair_perf.items())[:10]):
        print(f"  {pair}: mean={stats['mean_return']:.4f}, WR={stats['win_rate']:.1%}, n={stats['n_trades']}")

    # Bottom 10
    print("\n--- Bottom 10 Pairs ---")
    for pair, stats in list(reversed(list(pair_perf.items())))[:10]:
        print(f"  {pair}: mean={stats['mean_return']:.4f}, WR={stats['win_rate']:.1%}, n={stats['n_trades']}")

    # Re-correlation half-life
    print("\n--- Re-correlation Half-Life ---")
    halflife = analyze_recorrelation_halflife(corr_data, signals)
    if halflife['mean_halflife']:
        print(f"  Mean: {halflife['mean_halflife']:.1f} days | Median: {halflife['median_halflife']:.1f} days")
        print(f"  IQR: [{halflife['p25']:.0f}, {halflife['p75']:.0f}] days | Samples: {halflife['n_samples']}")
    else:
        print("  Insufficient data")

    # Sector groupings
    print("\n--- Performance by Sector Grouping (5d hold) ---")
    grouping = analyze_sector_groupings(base_trades_5d)
    for grp, stats in grouping.items():
        if stats.get('insufficient_data'):
            print(f"  {grp}: n={stats['n_trades']} (insufficient)")
        else:
            print(f"  {grp}: mean={stats['mean_return']:.4f}, WR={stats['win_rate']:.1%}, sharpe_approx={stats['sharpe_approx']:.3f}, n={stats['n_trades']}")

    # 6. Summary
    print(f"\n\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    passing = [(k, v) for k, v in all_results.items()
               if not v.get('skipped') and v.get('gates', {}).get('all_passed')]

    if passing:
        print(f"\n  ** {len(passing)} VARIANT(S) PASSED ALL 5 GATES **")
        for k, v in passing:
            m = v['metrics']
            print(f"  {k}: Sharpe={m['sharpe']:.3f}, Sortino={m['sortino']:.3f}, "
                  f"WR={m['win_rate']:.1%}, PF={m['profit_factor']:.2f}, "
                  f"Trades={m['n_trades']}, MaxDD={m['max_drawdown']:.4f}, "
                  f"p={v['p_value']:.4f}, RegimeGap={m['regime_gap']:.3f}")
    else:
        print("\n  ** NO VARIANTS PASSED ALL 5 GATES **")
        print(f"\n  Best variant: {best_variant}")
        if best_variant and best_variant in all_results:
            bv = all_results[best_variant]
            if not bv.get('skipped'):
                m = bv['metrics']
                print(f"    Sharpe={m['sharpe']:.3f}, WR={m['win_rate']:.1%}, PF={m['profit_factor']:.2f}")
                print(f"    Failed gates: ", end="")
                failed = [k for k, v in bv['gates'].items() if not v and k != 'all_passed']
                print(", ".join(failed))

    # 7. Save results
    output = {
        'run_timestamp': datetime.now().isoformat(),
        'config': {
            'corr_window': CORR_WINDOW,
            'baseline_window': BASELINE_WINDOW,
            'breakdown_threshold': BREAKDOWN_THRESHOLD,
            'cost_rt_pct': COST_RT_PCT,
            'hold_periods': HOLD_PERIODS,
            'n_permutations': N_PERMUTATIONS,
        },
        'total_signals_detected': len(signals),
        'variants': {},
        'pair_performance_5d': pair_perf,
        'recorrelation_halflife': halflife,
        'sector_grouping_performance': grouping,
        'passing_variants': [k for k, _ in passing],
        'best_variant': best_variant,
    }

    for k, v in all_results.items():
        # Convert for JSON serialization
        if v.get('skipped'):
            output['variants'][k] = v
        else:
            output['variants'][k] = {
                'variant': v.get('variant'),
                'description': v.get('description'),
                'hold_period': v.get('hold_period'),
                'metrics': v.get('metrics'),
                'p_value': v.get('p_value'),
                'gates': v.get('gates'),
            }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results saved to {OUTPUT_PATH}")
    print("=" * 80)


if __name__ == '__main__':
    main()
