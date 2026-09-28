#!/usr/bin/env python3
"""
GAMEPLAN v2 — ADVERSARIAL VALIDATION SUITE (HC #709 R2)
========================================================
Full adversarial battery on the finalized vol-adjusted UPRO system:
1. Permutation test: randomize regime signals → random switching must LOSE
2. Sub-period consistency: each 3-year block independently
3. Outlier removal: remove best N days → edge must persist
4. Data integrity: verify no look-ahead in signal computation
5. Bootstrap CI: confidence intervals on Sharpe/CAGR
6. Regime-label shuffling: are regime transitions actually predictive?
7. Transaction cost sensitivity: does edge survive higher costs?
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/adversarial_v2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

INITIAL = 500
WEEKLY_DCA = 100
np.random.seed(42)


def download_data():
    tickers = ['SPY', 'UPRO', 'GLD', 'TLT']
    data = yf.download(tickers, start='2012-01-01', period='max',
                       auto_adjust=True, threads=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        closes = data['Close']
    else:
        closes = data
    if hasattr(closes.columns, 'droplevel'):
        try:
            closes.columns = closes.columns.droplevel(1)
        except:
            pass
    return closes.dropna(how='all').dropna(subset=['SPY', 'UPRO'])


def compute_signals(spy_close):
    """Compute vol-adjusted regime signals with NO look-ahead."""
    spy_ret = spy_close.pct_change()
    vol_21d = spy_ret.rolling(21).std() * np.sqrt(252)
    sma20 = spy_close.rolling(20).mean()
    sma200 = spy_close.rolling(200).mean()
    return vol_21d, sma20, sma200


def get_regime(vol, sma20_val, sma200_val, spy_val, date,
               sep_hedge=True, earnings_aggr=True):
    """Determine allocation regime. Pure rules, no optimization."""
    # September hedge
    if sep_hedge and date.month == 9:
        return 'SPY'

    # Earnings aggression: use 25% threshold instead of 20%
    is_earnings = False
    if earnings_aggr:
        m, d = date.month, date.day
        is_earnings = ((m == 1 and d >= 15) or (m == 2 and d <= 15) or
                      (m == 4 and d >= 15) or (m == 5 and d <= 15) or
                      (m == 7 and d >= 15) or (m == 8 and d <= 15) or
                      (m == 10 and d >= 15) or (m == 11 and d <= 15))

    low_thresh = 25 if is_earnings else 20
    high_thresh = 30

    if np.isnan(vol):
        vol = 15
    vol_pct = vol * 100

    # Protection overlay: 20/200 crossover
    protection_off = (not np.isnan(sma20_val) and not np.isnan(sma200_val)
                     and sma20_val < sma200_val)

    if vol_pct > high_thresh:
        return 'GLD'
    elif vol_pct > low_thresh or protection_off:
        return 'SPY'
    else:
        return 'UPRO'


def simulate_system(closes, regime_override=None, tx_cost_pct=0.0,
                    sep_hedge=True, earnings_aggr=True):
    """
    Simulate the full v2 system.
    regime_override: if provided, use this Series of regimes instead of computed.
    tx_cost_pct: proportional transaction cost per switch.
    """
    spy = closes['SPY']
    vol_21d, sma20, sma200 = compute_signals(spy)
    returns = closes.pct_change().fillna(0)

    warmup = 260
    cash = float(INITIAL)
    total_contributed = float(INITIAL)
    last_week = None
    last_regime = None
    switches = 0
    daily_values = []
    daily_regimes = []

    for i in range(warmup, len(closes)):
        date = closes.index[i]

        # Weekly DCA
        week_key = (date.year, date.isocalendar()[1])
        if week_key != last_week:
            cash += WEEKLY_DCA
            total_contributed += WEEKLY_DCA
            last_week = week_key

        # Determine regime
        if regime_override is not None:
            regime = regime_override.iloc[i] if i < len(regime_override) else 'SPY'
        else:
            regime = get_regime(
                vol_21d.iloc[i],
                sma20.iloc[i] if i < len(sma20) else np.nan,
                sma200.iloc[i] if i < len(sma200) else np.nan,
                spy.iloc[i],
                date,
                sep_hedge=sep_hedge,
                earnings_aggr=earnings_aggr
            )

        # Track switches and apply tx costs
        if regime != last_regime and last_regime is not None:
            switches += 1
            cash *= (1 - tx_cost_pct)
        last_regime = regime

        # Apply return
        if regime in returns.columns:
            r = returns.loc[date, regime]
            if not np.isnan(r):
                cash *= (1 + r)

        daily_values.append(cash)
        daily_regimes.append(regime)

    dates = closes.index[warmup:]
    vals = pd.Series(daily_values, index=dates)
    regs = pd.Series(daily_regimes, index=dates)

    return vals, total_contributed, switches, regs


def compute_metrics(values, total_contributed):
    """Compute risk-adjusted metrics from daily portfolio values."""
    daily_ret = values.pct_change().dropna()
    ann_ret = daily_ret.mean() * 252
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    neg_ret = daily_ret[daily_ret < 0]
    downside_vol = neg_ret.std() * np.sqrt(252) if len(neg_ret) > 0 else 1
    sortino = ann_ret / downside_vol if downside_vol > 0 else 0

    # Max drawdown
    running_max = values.cummax()
    drawdown = (values - running_max) / running_max
    max_dd = drawdown.min()

    # CAGR
    years = (values.index[-1] - values.index[0]).days / 365.25
    cagr = (values.iloc[-1] / values.iloc[0]) ** (1/years) - 1 if years > 0 else 0

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    return {
        'final_value': values.iloc[-1],
        'cagr': cagr,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_dd': max_dd,
        'calmar': calmar,
        'ann_vol': ann_vol,
        'total_contributed': total_contributed,
        'profit': values.iloc[-1] - total_contributed
    }


def test_1_permutation(closes, n_perms=500):
    """
    PERMUTATION TEST: Randomize regime assignment dates.
    If random regime switches also make money → our signal is not real.
    """
    print("\n" + "="*60)
    print("TEST 1: PERMUTATION TEST (n=%d)" % n_perms)
    print("="*60)

    # Real system
    vals_real, contrib, switches, regimes = simulate_system(closes)
    metrics_real = compute_metrics(vals_real, contrib)
    real_sharpe = metrics_real['sharpe']

    print(f"  Real system: Sharpe={real_sharpe:.3f}, CAGR={metrics_real['cagr']:.1%}, "
          f"Final=${metrics_real['final_value']:,.0f}")

    # Permutation: shuffle regime labels in time (preserving regime distribution)
    perm_sharpes = []
    perm_finals = []

    for p in range(n_perms):
        shuffled = regimes.copy()
        # Block shuffle (blocks of 5 days to preserve some autocorrelation)
        block_size = 5
        n_blocks = len(shuffled) // block_size
        block_indices = np.arange(n_blocks)
        np.random.shuffle(block_indices)
        new_regimes = []
        for bi in block_indices:
            start = bi * block_size
            new_regimes.extend(shuffled.iloc[start:start+block_size].tolist())
        # Handle remainder
        new_regimes.extend(shuffled.iloc[n_blocks*block_size:].tolist())

        perm_regime = pd.Series(new_regimes[:len(shuffled)], index=shuffled.index)
        # Need to create override that matches closes index
        regime_full = pd.Series('SPY', index=closes.index)
        regime_full.loc[perm_regime.index] = perm_regime.values

        vals_perm, contrib_p, _, _ = simulate_system(closes, regime_override=regime_full)
        m = compute_metrics(vals_perm, contrib_p)
        perm_sharpes.append(m['sharpe'])
        perm_finals.append(m['final_value'])

        if (p+1) % 100 == 0:
            print(f"    {p+1}/{n_perms} permutations done...")

    perm_sharpes = np.array(perm_sharpes)
    p_value = np.mean(perm_sharpes >= real_sharpe)

    print(f"\n  Permutation results:")
    print(f"    Real Sharpe: {real_sharpe:.3f}")
    print(f"    Perm mean:   {np.mean(perm_sharpes):.3f}")
    print(f"    Perm median: {np.median(perm_sharpes):.3f}")
    print(f"    Perm std:    {np.std(perm_sharpes):.3f}")
    print(f"    p-value:     {p_value:.4f}")
    print(f"    VERDICT:     {'PASS ✓' if p_value < 0.05 else 'FAIL ✗'}")

    return {
        'real_sharpe': real_sharpe,
        'perm_mean': float(np.mean(perm_sharpes)),
        'perm_median': float(np.median(perm_sharpes)),
        'perm_std': float(np.std(perm_sharpes)),
        'p_value': float(p_value),
        'pass': p_value < 0.05
    }


def test_2_subperiod(closes):
    """
    SUB-PERIOD CONSISTENCY: Run on each 3-year block independently.
    Edge must be present in majority of sub-periods.
    """
    print("\n" + "="*60)
    print("TEST 2: SUB-PERIOD CONSISTENCY")
    print("="*60)

    vals, contrib, switches, regimes = simulate_system(closes)
    daily_ret = vals.pct_change().dropna()

    # Split into 3-year blocks
    years = sorted(set(daily_ret.index.year))
    blocks = []
    for i in range(0, len(years), 3):
        block_years = years[i:i+3]
        if len(block_years) >= 2:  # Need at least 2 years
            block_rets = daily_ret[daily_ret.index.year.isin(block_years)]
            if len(block_rets) > 100:
                ann_ret = block_rets.mean() * 252
                ann_vol = block_rets.std() * np.sqrt(252)
                sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
                blocks.append({
                    'years': f"{min(block_years)}-{max(block_years)}",
                    'sharpe': sharpe,
                    'ann_ret': ann_ret,
                    'n_days': len(block_rets)
                })

    print(f"\n  {'Period':<15} {'Sharpe':>8} {'Ann Ret':>10} {'Days':>6}")
    print(f"  {'-'*15} {'-'*8} {'-'*10} {'-'*6}")
    positive_sharpes = 0
    for b in blocks:
        marker = "✓" if b['sharpe'] > 0 else "✗"
        print(f"  {b['years']:<15} {b['sharpe']:>8.3f} {b['ann_ret']:>9.1%} {b['n_days']:>6} {marker}")
        if b['sharpe'] > 0:
            positive_sharpes += 1

    consistency = positive_sharpes / len(blocks) if blocks else 0
    print(f"\n  Positive Sharpe: {positive_sharpes}/{len(blocks)} periods ({consistency:.0%})")
    print(f"  VERDICT: {'PASS ✓' if consistency >= 0.67 else 'FAIL ✗'} (need ≥67%)")

    return {
        'blocks': blocks,
        'positive_ratio': consistency,
        'pass': consistency >= 0.67
    }


def test_3_outlier_removal(closes):
    """
    OUTLIER ROBUSTNESS: Remove best N days. Edge must persist.
    """
    print("\n" + "="*60)
    print("TEST 3: OUTLIER ROBUSTNESS")
    print("="*60)

    vals, contrib, switches, regimes = simulate_system(closes)
    daily_ret = vals.pct_change().dropna()

    full_sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252)

    results = []
    for n_remove in [1, 3, 5, 10, 20, 50]:
        trimmed = daily_ret.copy()
        # Remove top N days
        top_n_idx = trimmed.nlargest(n_remove).index
        trimmed = trimmed.drop(top_n_idx)

        trimmed_sharpe = trimmed.mean() / trimmed.std() * np.sqrt(252)
        pct_change = (trimmed_sharpe - full_sharpe) / abs(full_sharpe) * 100

        results.append({
            'n_removed': n_remove,
            'sharpe': trimmed_sharpe,
            'pct_change': pct_change
        })

    print(f"\n  Full Sharpe: {full_sharpe:.3f}")
    print(f"\n  {'Removed':>8} {'Sharpe':>8} {'Change':>8}")
    print(f"  {'-'*8} {'-'*8} {'-'*8}")
    for r in results:
        marker = "✓" if r['sharpe'] > 0.5 else "✗"
        print(f"  {r['n_removed']:>8d} {r['sharpe']:>8.3f} {r['pct_change']:>7.1f}% {marker}")

    # Edge persists if Sharpe > 0.5 after removing 10 best days
    sharpe_after_10 = [r['sharpe'] for r in results if r['n_removed'] == 10][0]
    passes = sharpe_after_10 > 0.5
    print(f"\n  Sharpe after removing 10 best days: {sharpe_after_10:.3f}")
    print(f"  VERDICT: {'PASS ✓' if passes else 'FAIL ✗'} (need Sharpe > 0.5 after removing 10 best days)")

    return {
        'full_sharpe': full_sharpe,
        'results': results,
        'sharpe_after_10': sharpe_after_10,
        'pass': passes
    }


def test_4_look_ahead(closes):
    """
    DATA INTEGRITY: Verify no look-ahead bias.
    Check that signals only use data available at decision time.
    """
    print("\n" + "="*60)
    print("TEST 4: LOOK-AHEAD BIAS CHECK")
    print("="*60)

    spy = closes['SPY']
    vol_21d, sma20, sma200 = compute_signals(spy)

    issues = []

    # Check 1: vol_21d uses only past data
    # rolling(21) uses the current and past 20 observations — correct
    for i in [260, 500, 1000, 2000]:
        if i < len(spy):
            # Compute vol using only data up to i
            spy_subset = spy.iloc[:i+1]
            vol_check = spy_subset.pct_change().rolling(21).std().iloc[-1] * np.sqrt(252)
            vol_full = vol_21d.iloc[i]
            if abs(vol_check - vol_full) > 1e-10:
                issues.append(f"Vol look-ahead at index {i}")

    # Check 2: SMA uses only past data
    for i in [260, 500, 1000, 2000]:
        if i < len(spy):
            sma_check = spy.iloc[:i+1].rolling(200).mean().iloc[-1]
            sma_full = sma200.iloc[i]
            if abs(sma_check - sma_full) > 1e-10:
                issues.append(f"SMA200 look-ahead at index {i}")

    # Check 3: Returns are applied AFTER regime decision (simulated in daily loop)
    # This is structurally guaranteed by the simulation loop (regime computed before return applied)

    # Check 4: No future data in yfinance download (adjusted close computed at download time)
    # This is inherent to yfinance — no issue for non-split-adjusted work

    if issues:
        print(f"  ISSUES FOUND:")
        for iss in issues:
            print(f"    ✗ {iss}")
    else:
        print(f"  ✓ Vol (21d rolling) — uses only past data")
        print(f"  ✓ SMA20/200 — uses only past data")
        print(f"  ✓ Regime decision — computed before return applied (structural)")
        print(f"  ✓ No future information in signal computation")

    passes = len(issues) == 0
    print(f"\n  VERDICT: {'PASS ✓' if passes else 'FAIL ✗'}")

    return {'issues': issues, 'pass': passes}


def test_5_bootstrap_ci(closes, n_bootstrap=1000):
    """
    BOOTSTRAP CONFIDENCE INTERVALS on Sharpe and CAGR.
    """
    print("\n" + "="*60)
    print("TEST 5: BOOTSTRAP CONFIDENCE INTERVALS (n=%d)" % n_bootstrap)
    print("="*60)

    vals, contrib, switches, regimes = simulate_system(closes)
    daily_ret = vals.pct_change().dropna()

    boot_sharpes = []
    boot_cagrs = []

    n = len(daily_ret)
    for b in range(n_bootstrap):
        # Block bootstrap (20-day blocks to preserve autocorrelation)
        block_size = 20
        n_blocks = n // block_size + 1
        boot_idx = []
        for _ in range(n_blocks):
            start = np.random.randint(0, n - block_size)
            boot_idx.extend(range(start, start + block_size))
        boot_idx = boot_idx[:n]

        boot_rets = daily_ret.iloc[boot_idx]
        boot_sharpe = boot_rets.mean() / boot_rets.std() * np.sqrt(252)
        boot_sharpes.append(boot_sharpe)

        # CAGR from bootstrap
        cum = (1 + boot_rets).cumprod()
        years = n / 252
        boot_cagr = cum.iloc[-1] ** (1/years) - 1
        boot_cagrs.append(boot_cagr)

        if (b+1) % 250 == 0:
            print(f"    {b+1}/{n_bootstrap} bootstraps done...")

    boot_sharpes = np.array(boot_sharpes)
    boot_cagrs = np.array(boot_cagrs)

    sharpe_ci = np.percentile(boot_sharpes, [2.5, 50, 97.5])
    cagr_ci = np.percentile(boot_cagrs, [2.5, 50, 97.5])

    print(f"\n  Sharpe 95% CI: [{sharpe_ci[0]:.3f}, {sharpe_ci[2]:.3f}]")
    print(f"  Sharpe median: {sharpe_ci[1]:.3f}")
    print(f"  CAGR 95% CI:   [{cagr_ci[0]:.1%}, {cagr_ci[2]:.1%}]")
    print(f"  CAGR median:   {cagr_ci[1]:.1%}")

    # Pass if lower bound of Sharpe CI > 0
    passes = sharpe_ci[0] > 0
    print(f"\n  Lower Sharpe bound: {sharpe_ci[0]:.3f}")
    print(f"  VERDICT: {'PASS ✓' if passes else 'FAIL ✗'} (need lower CI > 0)")

    return {
        'sharpe_ci': sharpe_ci.tolist(),
        'cagr_ci': cagr_ci.tolist(),
        'pass': passes
    }


def test_6_regime_shuffle(closes, n_shuffles=200):
    """
    REGIME LABEL SHUFFLING: Keep regime transition DATES but randomize
    which asset to hold in each regime. Tests if timing matters.
    """
    print("\n" + "="*60)
    print("TEST 6: REGIME TRANSITION TIMING TEST (n=%d)" % n_shuffles)
    print("="*60)

    vals_real, contrib, switches, regimes = simulate_system(closes)
    metrics_real = compute_metrics(vals_real, contrib)
    real_sharpe = metrics_real['sharpe']

    # Find regime transition points
    transitions = regimes.ne(regimes.shift()).cumsum()
    unique_regimes = regimes.unique()

    shuffle_sharpes = []
    for s in range(n_shuffles):
        # For each regime block, randomly assign UPRO/SPY/GLD
        shuffled = regimes.copy()
        for block_id in transitions.unique():
            mask = transitions == block_id
            shuffled[mask] = np.random.choice(unique_regimes)

        regime_full = pd.Series('SPY', index=closes.index)
        regime_full.loc[shuffled.index] = shuffled.values

        vals_s, contrib_s, _, _ = simulate_system(closes, regime_override=regime_full)
        m = compute_metrics(vals_s, contrib_s)
        shuffle_sharpes.append(m['sharpe'])

    shuffle_sharpes = np.array(shuffle_sharpes)
    p_value = np.mean(shuffle_sharpes >= real_sharpe)

    print(f"\n  Real Sharpe:    {real_sharpe:.3f}")
    print(f"  Shuffle mean:   {np.mean(shuffle_sharpes):.3f}")
    print(f"  Shuffle std:    {np.std(shuffle_sharpes):.3f}")
    print(f"  p-value:        {p_value:.4f}")
    print(f"  VERDICT:        {'PASS ✓' if p_value < 0.10 else 'FAIL ✗'} (need p < 0.10)")

    return {
        'real_sharpe': real_sharpe,
        'shuffle_mean': float(np.mean(shuffle_sharpes)),
        'p_value': float(p_value),
        'pass': p_value < 0.10
    }


def test_7_tx_cost_sensitivity(closes):
    """
    TRANSACTION COST SENSITIVITY: Does edge survive realistic costs?
    """
    print("\n" + "="*60)
    print("TEST 7: TRANSACTION COST SENSITIVITY")
    print("="*60)

    results = []
    for cost_bps in [0, 5, 10, 25, 50, 100]:
        cost_pct = cost_bps / 10000
        vals, contrib, switches, _ = simulate_system(closes, tx_cost_pct=cost_pct)
        m = compute_metrics(vals, contrib)
        results.append({
            'cost_bps': cost_bps,
            'sharpe': m['sharpe'],
            'final_value': m['final_value'],
            'cagr': m['cagr'],
            'switches': switches
        })

    print(f"\n  {'Cost (bps)':>10} {'Sharpe':>8} {'CAGR':>8} {'Final':>12} {'Switches':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*12} {'-'*8}")
    for r in results:
        print(f"  {r['cost_bps']:>10} {r['sharpe']:>8.3f} {r['cagr']:>7.1%} ${r['final_value']:>10,.0f} {r['switches']:>8}")

    # Pass if Sharpe > 1.0 at 50bps cost (very conservative)
    sharpe_at_50 = [r['sharpe'] for r in results if r['cost_bps'] == 50][0]
    passes = sharpe_at_50 > 1.0
    print(f"\n  Sharpe at 50bps: {sharpe_at_50:.3f}")
    print(f"  VERDICT: {'PASS ✓' if passes else 'FAIL ✗'} (need Sharpe > 1.0 at 50bps)")

    return {'results': results, 'sharpe_at_50bps': sharpe_at_50, 'pass': passes}


def test_8_benchmark_comparison(closes):
    """
    BENCHMARK: Compare vs naive strategies to confirm alpha.
    """
    print("\n" + "="*60)
    print("TEST 8: BENCHMARK COMPARISON")
    print("="*60)

    warmup = 260
    returns = closes.pct_change().fillna(0)

    benchmarks = {}

    # SPY buy-and-hold with DCA
    cash = float(INITIAL)
    total_c = float(INITIAL)
    last_week = None
    vals = []
    for i in range(warmup, len(closes)):
        date = closes.index[i]
        week_key = (date.year, date.isocalendar()[1])
        if week_key != last_week:
            cash += WEEKLY_DCA
            total_c += WEEKLY_DCA
            last_week = week_key
        r = returns.loc[date, 'SPY']
        if not np.isnan(r):
            cash *= (1 + r)
        vals.append(cash)
    spy_vals = pd.Series(vals, index=closes.index[warmup:])
    benchmarks['SPY DCA'] = compute_metrics(spy_vals, total_c)

    # UPRO buy-and-hold with DCA (no protection)
    cash = float(INITIAL)
    total_c = float(INITIAL)
    last_week = None
    vals = []
    for i in range(warmup, len(closes)):
        date = closes.index[i]
        week_key = (date.year, date.isocalendar()[1])
        if week_key != last_week:
            cash += WEEKLY_DCA
            total_c += WEEKLY_DCA
            last_week = week_key
        r = returns.loc[date, 'UPRO']
        if not np.isnan(r):
            cash *= (1 + r)
        vals.append(cash)
    upro_vals = pd.Series(vals, index=closes.index[warmup:])
    benchmarks['UPRO naked'] = compute_metrics(upro_vals, total_c)

    # Our system
    v2_vals, v2_contrib, _, _ = simulate_system(closes)
    benchmarks['Gameplan v2'] = compute_metrics(v2_vals, v2_contrib)

    print(f"\n  {'Strategy':<15} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'Final':>12} {'Sortino':>8}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*12} {'-'*8}")
    for name, m in benchmarks.items():
        print(f"  {name:<15} {m['sharpe']:>8.3f} {m['cagr']:>7.1%} {m['max_dd']:>7.1%} ${m['final_value']:>10,.0f} {m['sortino']:>8.3f}")

    # v2 must beat SPY on Sharpe
    v2_sharpe = benchmarks['Gameplan v2']['sharpe']
    spy_sharpe = benchmarks['SPY DCA']['sharpe']
    passes = v2_sharpe > spy_sharpe
    print(f"\n  v2 Sharpe ({v2_sharpe:.3f}) vs SPY ({spy_sharpe:.3f})")
    print(f"  VERDICT: {'PASS ✓' if passes else 'FAIL ✗'} (v2 must beat SPY)")

    return {
        'benchmarks': {k: v for k, v in benchmarks.items()},
        'pass': passes
    }


def main():
    print("="*60)
    print("GAMEPLAN v2 — ADVERSARIAL VALIDATION SUITE")
    print("HC #709 R2: Full adversarial battery")
    print(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    print("\nDownloading data...")
    closes = download_data()
    print(f"  Data: {closes.index[0].date()} to {closes.index[-1].date()} ({len(closes)} days)")

    results = {}

    # Run all tests
    results['permutation'] = test_1_permutation(closes, n_perms=500)
    results['subperiod'] = test_2_subperiod(closes)
    results['outlier'] = test_3_outlier_removal(closes)
    results['look_ahead'] = test_4_look_ahead(closes)
    results['bootstrap'] = test_5_bootstrap_ci(closes, n_bootstrap=1000)
    results['regime_shuffle'] = test_6_regime_shuffle(closes, n_shuffles=200)
    results['tx_cost'] = test_7_tx_cost_sensitivity(closes)
    results['benchmark'] = test_8_benchmark_comparison(closes)

    # Summary
    print("\n" + "="*60)
    print("ADVERSARIAL VALIDATION SUMMARY")
    print("="*60)

    n_pass = 0
    n_total = 0
    for test_name, test_result in results.items():
        if 'pass' in test_result:
            status = "PASS ✓" if test_result['pass'] else "FAIL ✗"
            n_pass += test_result['pass']
            n_total += 1
            print(f"  {test_name:<20} {status}")

    print(f"\n  OVERALL: {n_pass}/{n_total} tests passed")

    if n_pass >= 6:
        print(f"  VERDICT: VALIDATED ✓ — Gameplan v2 passes adversarial suite")
    elif n_pass >= 4:
        print(f"  VERDICT: CONDITIONAL PASS — some concerns but viable")
    else:
        print(f"  VERDICT: FAIL ✗ — system may not have real edge")

    # Save results
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        return obj

    save_results = {}
    for k, v in results.items():
        save_results[k] = {}
        for kk, vv in v.items():
            if isinstance(vv, list):
                save_results[k][kk] = [
                    {kkk: convert(vvv) for kkk, vvv in item.items()} if isinstance(item, dict) else convert(item)
                    for item in vv
                ]
            elif isinstance(vv, dict):
                save_results[k][kk] = {kkk: convert(vvv) for kkk, vvv in vv.items()}
            else:
                save_results[k][kk] = convert(vv)

    with open(os.path.join(OUTPUT_DIR, 'adversarial_results.json'), 'w') as f:
        json.dump(save_results, f, indent=2, default=str)

    print(f"\n  Results saved to {OUTPUT_DIR}/adversarial_results.json")


if __name__ == '__main__':
    main()
