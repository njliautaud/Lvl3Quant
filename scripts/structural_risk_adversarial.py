#!/usr/bin/env python3
"""
Adversarial Validation: Structural Risk Score (RSP/GLD/UUP)
6-check adversarial suite for the market structure diversifier strategy.
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────
START = '2022-01-01'
END = '2026-07-29'
INITIAL_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
COMMISSION = 0.0

TICKERS = ['SPY', 'QQQ', 'RSP', 'IWM', 'GLD', 'UUP', '^VIX', 'XLE', 'XLF', 'XLV', 'XLK', 'XLU', 'XLP']
SECTORS = ['XLE', 'XLF', 'XLV', 'XLK', 'XLU', 'XLP']


def download_data():
    """Download all required data via yfinance."""
    print("Downloading data...")
    data = {}
    for t in TICKERS:
        df = yf.download(t, start=START, end=END, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        data[t] = df
    print(f"  Downloaded {len(data)} tickers, date range: {data['SPY'].index[0].date()} to {data['SPY'].index[-1].date()}")
    return data


def compute_risk_score(data, date_idx, vix_threshold=20, sector_breadth_threshold=4, lookback=20):
    """Compute the 4-component structural risk score for a given date index."""
    if date_idx < max(lookback, 50):
        return 0

    spy_close = data['SPY']['Close'].values
    rsp_close = data['RSP']['Close'].values
    iwm_close = data['IWM']['Close'].values
    vix_close = data['^VIX']['Close'].values

    # Component 1: RSP outperforming SPY over lookback days
    rsp_ret = rsp_close[date_idx] / rsp_close[date_idx - lookback] - 1
    spy_ret = spy_close[date_idx] / spy_close[date_idx - lookback] - 1
    c1 = 1 if rsp_ret > spy_ret else 0

    # Component 2: IWM outperforming SPY over lookback days
    iwm_ret = iwm_close[date_idx] / iwm_close[date_idx - lookback] - 1
    c2 = 1 if iwm_ret > spy_ret else 0

    # Component 3: VIX < threshold
    c3 = 1 if vix_close[date_idx] < vix_threshold else 0

    # Component 4: >= sector_breadth_threshold sectors above 50d SMA
    sectors_above = 0
    for sec in SECTORS:
        sec_close = data[sec]['Close'].values
        sma50 = np.mean(sec_close[date_idx - 49:date_idx + 1])
        if sec_close[date_idx] > sma50:
            sectors_above += 1
    c4 = 1 if sectors_above >= sector_breadth_threshold else 0

    return c1 + c2 + c3 + c4


def run_strategy(data, risk_on_thresh=3, risk_off_thresh=0, vix_threshold=20,
                 sector_breadth_threshold=4, rebalance_days=5, slippage=SLIPPAGE_PCT,
                 lag=0, inverse=False):
    """
    Run the structural risk score strategy.
    lag=1 means use T-1 signals for T allocation (look-ahead bias check).
    inverse=True flips risk-on/risk-off allocations.
    """
    spy_idx = data['SPY'].index
    n = len(spy_idx)

    # Pre-compute daily returns for allocation assets
    rsp_ret = data['RSP']['Close'].pct_change().values
    gld_ret = data['GLD']['Close'].pct_change().values
    uup_ret = data['UUP']['Close'].pct_change().values

    capital = INITIAL_CAPITAL
    equity_curve = [capital]
    daily_returns = []
    allocations = []  # track which allocation each day

    current_alloc = 'cash'
    days_since_rebalance = rebalance_days  # trigger rebalance on first valid day

    start_idx = max(50, 1)  # need 50 days for SMA

    for i in range(start_idx, n):
        days_since_rebalance += 1

        if days_since_rebalance >= rebalance_days:
            signal_idx = i - lag if lag > 0 else i
            if signal_idx < 50:
                signal_idx = 50
            score = compute_risk_score(data, signal_idx, vix_threshold, sector_breadth_threshold)

            if not inverse:
                if score >= risk_on_thresh:
                    current_alloc = 'RSP'
                elif score <= risk_off_thresh:
                    current_alloc = 'GLD'
                else:
                    current_alloc = 'UUP'
            else:
                # Inverse: flip risk-on and risk-off
                if score >= risk_on_thresh:
                    current_alloc = 'GLD'  # was RSP
                elif score <= risk_off_thresh:
                    current_alloc = 'RSP'  # was GLD
                else:
                    current_alloc = 'UUP'

            # Apply slippage on rebalance
            capital *= (1 - slippage)
            days_since_rebalance = 0

        # Apply daily return based on allocation
        if current_alloc == 'RSP':
            day_ret = rsp_ret[i] if not np.isnan(rsp_ret[i]) else 0
        elif current_alloc == 'GLD':
            day_ret = gld_ret[i] if not np.isnan(gld_ret[i]) else 0
        elif current_alloc == 'UUP':
            day_ret = uup_ret[i] if not np.isnan(uup_ret[i]) else 0
        else:
            day_ret = 0

        capital *= (1 + day_ret)
        equity_curve.append(capital)
        daily_returns.append(day_ret)
        allocations.append(current_alloc)

    return np.array(equity_curve), np.array(daily_returns), allocations


def compute_metrics(equity_curve, daily_returns):
    """Compute strategy performance metrics."""
    if len(daily_returns) == 0 or np.std(daily_returns) == 0:
        return {'sharpe': 0, 'sortino': 0, 'win_rate': 0, 'profit_factor': 0,
                'num_trades': 0, 'total_return': 0, 'mdd': 0, 'mean_return_pct': 0}

    mean_ret = np.mean(daily_returns)
    std_ret = np.std(daily_returns)
    sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0

    downside = daily_returns[daily_returns < 0]
    downside_std = np.std(downside) if len(downside) > 0 else 1e-10
    sortino = mean_ret / downside_std * np.sqrt(252)

    wins = daily_returns[daily_returns > 0]
    losses = daily_returns[daily_returns < 0]
    win_rate = len(wins) / len(daily_returns) if len(daily_returns) > 0 else 0

    gross_profit = np.sum(wins) if len(wins) > 0 else 0
    gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 1e-10
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999

    total_return = (equity_curve[-1] / equity_curve[0] - 1) * 100

    # Max drawdown
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak
    mdd = np.min(dd) * 100

    return {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'win_rate': round(win_rate, 3),
        'profit_factor': round(profit_factor, 3),
        'num_trades': len(daily_returns),
        'total_return': round(total_return, 1),
        'mdd': round(mdd, 1),
        'mean_return_pct': round(mean_ret * 100, 4)
    }


def check_1_inverse_direction(data, baseline_sharpe):
    """Check 1: Inverse direction test."""
    print("\n[Check 1] Inverse Direction Test...")
    eq, rets, _ = run_strategy(data, inverse=True)
    inv_metrics = compute_metrics(eq, rets)

    passed = inv_metrics['sharpe'] < 0 and baseline_sharpe > 2 * inv_metrics['sharpe']
    verdict = f"Inverse Sharpe={inv_metrics['sharpe']:.3f} vs Baseline={baseline_sharpe:.3f}. "
    if passed:
        verdict += "PASS: Inverse loses money and baseline > 2x inverse."
    else:
        if inv_metrics['sharpe'] >= 0:
            verdict += f"FAIL: Inverse Sharpe >= 0 ({inv_metrics['sharpe']:.3f}), strategy works in both directions."
        else:
            verdict += f"FAIL: Baseline not > 2x inverse Sharpe."

    print(f"  {verdict}")
    return {'pass': bool(passed), 'verdict': verdict, 'inverse_metrics': inv_metrics}


def check_2_random_timing(data, baseline_sharpe, n_iter=1000):
    """Check 2: Random timing test (1000 iterations)."""
    print(f"\n[Check 2] Random Timing Test ({n_iter} iterations)...")

    # Get baseline allocations to know the distribution
    _, _, base_allocs = run_strategy(data)

    # Count allocation periods
    alloc_counts = {}
    for a in base_allocs:
        alloc_counts[a] = alloc_counts.get(a, 0) + 1

    n_days = len(base_allocs)
    rsp_ret = data['RSP']['Close'].pct_change().values
    gld_ret = data['GLD']['Close'].pct_change().values
    uup_ret = data['UUP']['Close'].pct_change().values

    spy_idx = data['SPY'].index
    start_idx = len(spy_idx) - n_days

    # Build return arrays for the period
    period_rsp = rsp_ret[start_idx:start_idx + n_days]
    period_gld = gld_ret[start_idx:start_idx + n_days]
    period_uup = uup_ret[start_idx:start_idx + n_days]

    # Replace NaN
    period_rsp = np.nan_to_num(period_rsp)
    period_gld = np.nan_to_num(period_gld)
    period_uup = np.nan_to_num(period_uup)

    # Create allocation label array
    alloc_labels = np.array(base_allocs)

    random_sharpes = []
    rng = np.random.default_rng(42)

    for _ in range(n_iter):
        # Shuffle allocations randomly
        shuffled = alloc_labels.copy()
        rng.shuffle(shuffled)

        # Compute returns
        rand_rets = np.where(shuffled == 'RSP', period_rsp,
                    np.where(shuffled == 'GLD', period_gld,
                    np.where(shuffled == 'UUP', period_uup, 0.0)))

        if np.std(rand_rets) > 0:
            s = np.mean(rand_rets) / np.std(rand_rets) * np.sqrt(252)
        else:
            s = 0
        random_sharpes.append(s)

    random_sharpes = np.array(random_sharpes)
    percentile = np.mean(random_sharpes < baseline_sharpe) * 100

    passed = percentile >= 90
    verdict = f"Strategy at {percentile:.1f}th percentile of random timing (mean random Sharpe={np.mean(random_sharpes):.3f}, std={np.std(random_sharpes):.3f}). "
    verdict += "PASS" if passed else "FAIL"

    print(f"  {verdict}")
    return {
        'pass': bool(passed),
        'verdict': verdict,
        'percentile': round(percentile, 1),
        'random_sharpe_mean': round(float(np.mean(random_sharpes)), 3),
        'random_sharpe_std': round(float(np.std(random_sharpes)), 3)
    }


def check_3_look_ahead_bias(data, baseline_sharpe):
    """Check 3: Look-ahead bias check (1-day lag on all signals)."""
    print("\n[Check 3] Look-Ahead Bias Check (1-day lag)...")
    eq, rets, _ = run_strategy(data, lag=1)
    pit_metrics = compute_metrics(eq, rets)

    sharpe_ratio = pit_metrics['sharpe'] / baseline_sharpe if baseline_sharpe != 0 else 0
    within_30pct = abs(1 - sharpe_ratio) <= 0.30

    passed = pit_metrics['sharpe'] > 0.5 and within_30pct
    verdict = f"Lagged Sharpe={pit_metrics['sharpe']:.3f} (baseline={baseline_sharpe:.3f}, ratio={sharpe_ratio:.2f}). "
    if passed:
        verdict += "PASS: Lagged > 0.5 and within 30% of baseline."
    else:
        reasons = []
        if pit_metrics['sharpe'] <= 0.5:
            reasons.append(f"lagged Sharpe {pit_metrics['sharpe']:.3f} <= 0.5")
        if not within_30pct:
            reasons.append(f"ratio {sharpe_ratio:.2f} outside 30% band")
        verdict += f"FAIL: {', '.join(reasons)}."

    print(f"  {verdict}")
    return {'pass': bool(passed), 'verdict': verdict, 'point_in_time_metrics': pit_metrics}


def check_4_cost_sensitivity(data):
    """Check 4: Cost sensitivity test at various slippage levels."""
    print("\n[Check 4] Cost Sensitivity Test...")
    slippage_levels = [0.0005, 0.0010, 0.0015, 0.0020]
    results = {}

    for slip in slippage_levels:
        eq, rets, _ = run_strategy(data, slippage=slip)
        m = compute_metrics(eq, rets)
        label = f"{slip*100:.2f}%"
        results[label] = m
        print(f"  Slippage {label}: Sharpe={m['sharpe']:.3f}, Return={m['total_return']:.1f}%")

    # Pass if Sharpe > 0.5 at 0.10% slippage
    passed = results['0.10%']['sharpe'] > 0.5
    verdict = f"Sharpe at 0.10% slippage = {results['0.10%']['sharpe']:.3f}. "
    verdict += "PASS" if passed else "FAIL: Sharpe <= 0.5 at 0.10% slippage"

    print(f"  {verdict}")
    return {'pass': bool(passed), 'verdict': verdict, 'results_by_slippage': results}


def check_5_sub_period_stability(data):
    """Check 5: Sub-period stability test (4 equal periods)."""
    print("\n[Check 5] Sub-Period Stability Test...")

    eq, rets, allocs = run_strategy(data)
    n = len(rets)
    period_size = n // 4

    sub_periods = []
    sharpes_positive = 0

    for i in range(4):
        start = i * period_size
        end = (i + 1) * period_size if i < 3 else n
        sub_rets = rets[start:end]

        # Build mini equity curve
        sub_eq = np.cumprod(1 + sub_rets) * INITIAL_CAPITAL
        sub_eq = np.insert(sub_eq, 0, INITIAL_CAPITAL)

        m = compute_metrics(sub_eq, sub_rets)

        spy_idx = data['SPY'].index
        offset = len(spy_idx) - n
        start_date = str(spy_idx[offset + start].date())
        end_date = str(spy_idx[offset + end - 1].date()) if end <= n else str(spy_idx[-1].date())

        sub_periods.append({
            'period': f"{start_date} to {end_date}",
            'sharpe': m['sharpe'],
            'total_return': m['total_return'],
            'mdd': m['mdd']
        })

        if m['sharpe'] > 0:
            sharpes_positive += 1

        print(f"  Period {i+1} ({start_date} to {end_date}): Sharpe={m['sharpe']:.3f}, Return={m['total_return']:.1f}%")

    passed = sharpes_positive >= 3
    verdict = f"{sharpes_positive}/4 sub-periods have Sharpe > 0. "
    verdict += "PASS" if passed else "FAIL: Need >= 3/4 positive Sharpe periods"

    print(f"  {verdict}")
    return {'pass': bool(passed), 'verdict': verdict, 'sub_periods': sub_periods}


def check_6_parameter_sensitivity(data):
    """Check 6: Parameter sensitivity grid search (216 combinations)."""
    print("\n[Check 6] Parameter Sensitivity Test (216 combinations)...")

    risk_on_thresholds = [2, 3, 4]
    risk_off_thresholds = [0, 1]
    vix_thresholds = [18, 20, 22, 25]
    sector_breadth_thresholds = [3, 4, 5]
    rebalance_periods = [3, 5, 10]

    total = len(risk_on_thresholds) * len(risk_off_thresholds) * len(vix_thresholds) * \
            len(sector_breadth_thresholds) * len(rebalance_periods)

    above_03 = 0
    best_sharpe = -999
    best_params = {}
    all_sharpes = []

    count = 0
    for ron in risk_on_thresholds:
        for roff in risk_off_thresholds:
            if roff >= ron:  # risk-off threshold must be below risk-on
                # Still run it, just note it's degenerate
                pass
            for vix_t in vix_thresholds:
                for sbt in sector_breadth_thresholds:
                    for reb in rebalance_periods:
                        count += 1
                        eq, rets, _ = run_strategy(data, risk_on_thresh=ron, risk_off_thresh=roff,
                                                    vix_threshold=vix_t, sector_breadth_threshold=sbt,
                                                    rebalance_days=reb)
                        m = compute_metrics(eq, rets)
                        s = m['sharpe']
                        all_sharpes.append(s)

                        if s > 0.3:
                            above_03 += 1
                        if s > best_sharpe:
                            best_sharpe = s
                            best_params = {
                                'risk_on_threshold': ron,
                                'risk_off_threshold': roff,
                                'vix_threshold': vix_t,
                                'sector_breadth_threshold': sbt,
                                'rebalance_days': reb,
                                'sharpe': round(s, 3),
                                'total_return': m['total_return'],
                                'mdd': m['mdd']
                            }

    pct_above = above_03 / total * 100
    passed = pct_above >= 30

    verdict = f"{above_03}/{total} ({pct_above:.1f}%) combinations have Sharpe > 0.3. "
    verdict += f"Best Sharpe={best_sharpe:.3f}. "
    verdict += "PASS" if passed else "FAIL: Need >= 30% above 0.3 Sharpe"

    print(f"  {verdict}")
    print(f"  Best params: {best_params}")
    print(f"  Sharpe distribution: mean={np.mean(all_sharpes):.3f}, median={np.median(all_sharpes):.3f}, "
          f"min={np.min(all_sharpes):.3f}, max={np.max(all_sharpes):.3f}")

    return {
        'pass': bool(passed),
        'verdict': verdict,
        'pct_above_0_3_sharpe': round(pct_above, 1),
        'total_combinations': total,
        'above_0_3_count': above_03,
        'best_params': best_params,
        'sharpe_distribution': {
            'mean': round(float(np.mean(all_sharpes)), 3),
            'median': round(float(np.median(all_sharpes)), 3),
            'min': round(float(np.min(all_sharpes)), 3),
            'max': round(float(np.max(all_sharpes)), 3)
        }
    }


def main():
    print("=" * 70)
    print("ADVERSARIAL VALIDATION: Structural Risk Score (RSP/GLD/UUP)")
    print("=" * 70)

    # Download data
    data = download_data()

    # Run baseline strategy
    print("\n[Baseline] Running strategy with default parameters...")
    eq, rets, allocs = run_strategy(data)
    baseline = compute_metrics(eq, rets)
    print(f"  Baseline: Sharpe={baseline['sharpe']}, Sortino={baseline['sortino']}, "
          f"Return={baseline['total_return']}%, MDD={baseline['mdd']}%, WR={baseline['win_rate']}")

    # QQQ correlation
    qqq_ret = data['QQQ']['Close'].pct_change().values
    spy_idx = data['SPY'].index
    n_strat = len(rets)
    qqq_aligned = qqq_ret[len(spy_idx) - n_strat:]
    qqq_aligned = np.nan_to_num(qqq_aligned)
    qqq_corr = float(np.corrcoef(rets, qqq_aligned)[0, 1])
    print(f"  QQQ Correlation: {qqq_corr:.3f}")

    # Run all 6 checks
    c1 = check_1_inverse_direction(data, baseline['sharpe'])
    c2 = check_2_random_timing(data, baseline['sharpe'])
    c3 = check_3_look_ahead_bias(data, baseline['sharpe'])
    c4 = check_4_cost_sensitivity(data)
    c5 = check_5_sub_period_stability(data)
    c6 = check_6_parameter_sensitivity(data)

    checks = {
        '1_inverse_direction': c1,
        '2_random_timing': c2,
        '3_look_ahead_bias': c3,
        '4_cost_sensitivity': c4,
        '5_sub_period_stability': c5,
        '6_parameter_sensitivity': c6
    }

    passed_count = sum(1 for c in checks.values() if c['pass'])
    overall = passed_count == 6

    # OOT period
    oot_start = data['SPY'].index[50].date()
    oot_end = data['SPY'].index[-1].date()

    results = {
        'strategy': 'Structural Risk Score (RSP/GLD/UUP)',
        'run_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'oot_period': f"{oot_start} to {oot_end}",
        'baseline': baseline,
        'checks': checks,
        'qqq_correlation': round(qqq_corr, 3),
        'overall_pass': overall,
        'checks_passed': f"{passed_count}/6"
    }

    # Summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    for name, check in checks.items():
        status = "PASS" if check['pass'] else "FAIL"
        print(f"  [{status}] {name}")
    print(f"\n  Overall: {passed_count}/6 checks passed → {'PASS' if overall else 'FAIL'}")
    print(f"  QQQ Correlation: {qqq_corr:.3f}")

    # Save results
    out_path = '/home/jupiter/Lvl3Quant/data/structural_risk_adversarial_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    return results


if __name__ == '__main__':
    results = main()
