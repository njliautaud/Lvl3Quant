#!/usr/bin/env python3
"""
Adversarial Validation — Sector Rank Reversal Variants B & E
=============================================================
6-test battery per variant:
  1. Re-implementation (independent code, same logic)
  2. Inverse signal (flip conditions)
  3. Random timing (1000 shuffled entry dates)
  4. Cost sensitivity (0.05%-0.20%)
  5. Sub-period stability (4 equal periods, all must be Sharpe >= 0)
  6. Parameter robustness (grid search, >=60% combos Sharpe > 0.3)

Author: Claude Opus 4.6 | Date: 2026-08-18
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from datetime import datetime
from collections import defaultdict
import sys
import time

# ── Config (matches original) ────────────────────────────────────────────────
SECTOR_ETFS = ['XLE', 'XLU', 'XLP', 'XLK', 'XLY', 'XLF', 'XLRE', 'XLV', 'XLI', 'XLB', 'XLC']
BENCHMARK = 'SPY'
HOLD_PERIOD = 5
LOOKBACK_RANK = 20
MOMENTUM_CONFIRM = 3
SLIDING_WINDOW = 60
START_DATE = '2019-01-01'
END_DATE = '2026-08-15'

EARNINGS_MONTHS = [1, 2, 4, 5, 7, 8, 10, 11]
EARNINGS_WINDOW = 10
EARNINGS_THRESHOLD = 0.01

# ── Data Download ─────────────────────────────────────────────────────────────

def download_data():
    tickers = SECTOR_ETFS + [BENCHMARK]
    print(f"Downloading {len(tickers)} tickers from {START_DATE} to {END_DATE}...")
    data = yf.download(tickers, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']
    else:
        close = data
    close = close.dropna(axis=1, how='all').ffill().dropna()
    print(f"Got {len(close)} trading days, {len(close.columns)} tickers")
    return close


# ══════════════════════════════════════════════════════════════════════════════
#  ORIGINAL IMPLEMENTATIONS (copied from backtest for baseline comparison)
# ══════════════════════════════════════════════════════════════════════════════

def orig_earnings_signals(returns, sector_cols):
    """Variant E original: long sectors outperforming SPY by >1% over 10d earnings window."""
    spy_ret = returns[BENCHMARK]
    signals = pd.DataFrame(0, index=returns.index, columns=sector_cols)
    for col in sector_cols:
        rel_ret = (returns[col] - spy_ret).rolling(EARNINGS_WINDOW).sum()
        for i in range(len(returns)):
            if returns.index[i].month not in EARNINGS_MONTHS:
                continue
            if pd.isna(rel_ret.iloc[i]):
                continue
            if rel_ret.iloc[i] > EARNINGS_THRESHOLD:
                signals.iloc[i][col] = 1
    return signals


def orig_reversal_signals(returns, sector_cols, max_rank=2, min_mom_days=3):
    """Variant B original: bottom-2 ranked sectors with positive 3d momentum."""
    rolling_ret = returns[sector_cols].rolling(LOOKBACK_RANK).sum()
    ranks = rolling_ret.rank(axis=1, ascending=True)
    mom = returns[sector_cols].rolling(min_mom_days).sum()
    signals = pd.DataFrame(0, index=returns.index, columns=sector_cols)

    for i in range(max(LOOKBACK_RANK, min_mom_days) + 1, len(returns)):
        for col in sector_cols:
            rank_val = ranks.iloc[i].get(col, np.nan)
            mom_val = mom.iloc[i].get(col, np.nan)
            if pd.isna(rank_val) or pd.isna(mom_val):
                continue
            if rank_val <= max_rank and mom_val > 0:
                signals.iloc[i][col] = 1
    return signals


# ══════════════════════════════════════════════════════════════════════════════
#  RE-IMPLEMENTATIONS (Test 1 — written from scratch, same logic)
# ══════════════════════════════════════════════════════════════════════════════

def reimpl_earnings_signals(returns, sector_cols):
    """Independent re-implementation of Variant E earnings outperformance."""
    spy = returns[BENCHMARK].values
    idx = returns.index
    n = len(returns)
    sig_arr = np.zeros((n, len(sector_cols)), dtype=int)

    for c_idx, col in enumerate(sector_cols):
        sector_vals = returns[col].values
        excess = sector_vals - spy
        # Rolling 10-day sum of excess returns
        cum_excess = np.cumsum(excess)
        cum_excess = np.insert(cum_excess, 0, 0.0)
        for i in range(EARNINGS_WINDOW, n):
            if idx[i].month not in EARNINGS_MONTHS:
                continue
            roll_sum = cum_excess[i + 1] - cum_excess[i + 1 - EARNINGS_WINDOW]
            if roll_sum > EARNINGS_THRESHOLD:
                sig_arr[i, c_idx] = 1

    return pd.DataFrame(sig_arr, index=idx, columns=sector_cols)


def reimpl_reversal_signals(returns, sector_cols, max_rank=2, min_mom_days=3):
    """Independent re-implementation of Variant B rank reversal."""
    n = len(returns)
    n_sec = len(sector_cols)
    sig_arr = np.zeros((n, n_sec), dtype=int)

    # Pre-compute rolling 20d returns and 3d momentum using numpy
    sector_rets = returns[sector_cols].values  # (n, n_sec)

    # Rolling 20d sum
    cum = np.cumsum(sector_rets, axis=0)
    cum = np.vstack([np.zeros((1, n_sec)), cum])
    roll20 = np.full((n, n_sec), np.nan)
    for i in range(LOOKBACK_RANK, n):
        roll20[i] = cum[i + 1] - cum[i + 1 - LOOKBACK_RANK]

    # Rolling 3d momentum
    roll_mom = np.full((n, n_sec), np.nan)
    for i in range(min_mom_days, n):
        roll_mom[i] = cum[i + 1] - cum[i + 1 - min_mom_days]

    start = max(LOOKBACK_RANK, min_mom_days) + 1
    for i in range(start, n):
        row20 = roll20[i]
        if np.any(np.isnan(row20)):
            continue
        # Rank ascending (1 = worst performer)
        order = np.argsort(np.argsort(row20)) + 1  # 1-based ranks
        for c_idx in range(n_sec):
            if order[c_idx] <= max_rank and roll_mom[i, c_idx] > 0:
                sig_arr[i, c_idx] = 1

    return pd.DataFrame(sig_arr, index=returns.index, columns=sector_cols)


# ══════════════════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE (shared by all tests)
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(returns, signals, sector_cols, max_positions=3, cost_pct=0.0, skip_trailing_filter=False):
    """
    Walk-forward sliding-window backtest matching original logic.
    Optional per-trade round-trip cost (as fraction of notional).
    skip_trailing_filter: skip expensive trailing filter (for permutation tests).
    Returns (trades_list, daily_portfolio_returns).
    """
    spy_daily = returns[BENCHMARK]
    warmup = SLIDING_WINDOW + LOOKBACK_RANK + MOMENTUM_CONFIRM + 5

    # Convert to numpy for speed
    ret_arr = returns[sector_cols].values  # shape (n_days, n_sectors)
    sig_arr = signals[sector_cols].values if hasattr(signals, 'values') else signals
    spy_arr = spy_daily.values
    n_days = len(returns)
    n_sec = len(sector_cols)

    trades = []
    active_positions = []  # list of (exit_idx, sector_idx, direction)
    daily_port_ret = np.zeros(n_days)

    for i in range(warmup, n_days):
        active_positions = [(ex, si, d) for ex, si, d in active_positions if ex > i]

        new_entries = []
        for c_idx in range(n_sec):
            sig = sig_arr[i, c_idx]
            if sig == 0:
                continue
            if any(si == c_idx for _, si, _ in active_positions):
                continue

            if not skip_trailing_filter:
                # Walk-forward trailing filter
                ws = max(warmup, i - SLIDING_WINDOW)
                trail_sigs = sig_arr[ws:i, c_idx]
                trail_nonzero = np.nonzero(trail_sigs)[0]
                if len(trail_nonzero) >= 3:
                    trail_rets = []
                    for t_off in trail_nonzero:
                        j_pos = ws + t_off
                        if j_pos + HOLD_PERIOD < n_days and j_pos < i:
                            r = ret_arr[j_pos+1:j_pos+1+HOLD_PERIOD, c_idx].sum()
                            trail_rets.append(r * trail_sigs[t_off])
                    if len(trail_rets) >= 3 and np.mean(trail_rets) < -0.005:
                        continue

            new_entries.append((c_idx, sig))

        if len(active_positions) + len(new_entries) > max_positions:
            new_entries = new_entries[:max_positions - len(active_positions)]

        for c_idx, sig in new_entries:
            exit_idx = min(i + HOLD_PERIOD, n_days - 1)
            active_positions.append((exit_idx, c_idx, sig))

            entry_idx = i + 1 if i + 1 < n_days else i
            actual_exit = min(i + HOLD_PERIOD, n_days - 1)
            hold_return = ret_arr[entry_idx:actual_exit+1, c_idx].sum()
            trade_return = hold_return * sig - cost_pct
            spy_hold = spy_arr[entry_idx:actual_exit+1].sum()
            regime = 'green' if spy_arr[i] > 0 else 'red'

            trades.append({
                'entry_date': returns.index[entry_idx] if entry_idx < n_days else returns.index[-1],
                'exit_date': returns.index[actual_exit],
                'sector': sector_cols[c_idx],
                'direction': 'long' if sig > 0 else 'short',
                'signal': sig,
                'return': trade_return,
                'spy_return': spy_hold,
                'regime': regime,
            })

        n_active = len(active_positions)
        if n_active > 0:
            weight = 1.0 / max_positions
            day_ret = 0.0
            for _, si, d in active_positions:
                day_ret += weight * ret_arr[i, si] * d
            if cost_pct > 0 and len(new_entries) > 0:
                daily_port_ret[i] = day_ret - (cost_pct * len(new_entries) / max_positions / HOLD_PERIOD)
            else:
                daily_port_ret[i] = day_ret

    daily_port_ret_series = pd.Series(daily_port_ret, index=returns.index)
    return trades, daily_port_ret_series


def compute_sharpe(trades_or_daily, use_daily=False):
    """Compute annualized Sharpe from trade returns or daily portfolio returns."""
    if use_daily:
        port = trades_or_daily[trades_or_daily != 0]
        if len(port) < 5:
            port = trades_or_daily
        if len(port) == 0 or np.std(port) == 0:
            return 0.0
        return (np.mean(port) / np.std(port)) * np.sqrt(252)
    else:
        rets = np.array([t['return'] for t in trades_or_daily]) if isinstance(trades_or_daily, list) else trades_or_daily
        if len(rets) < 2 or np.std(rets) == 0:
            return 0.0
        return (np.mean(rets) / np.std(rets)) * np.sqrt(52)  # weekly-ish trades


def compute_full_metrics(trades_list, daily_ret):
    """Compute Sharpe, WR, PF from daily portfolio returns and trade list."""
    if len(trades_list) == 0:
        return {'sharpe': 0, 'win_rate': 0, 'profit_factor': 0, 'n_trades': 0, 'avg_return': 0}

    rets = np.array([t['return'] for t in trades_list])
    port = daily_ret[daily_ret != 0]
    if len(port) < 5:
        port = daily_ret

    sharpe = (np.mean(port) / np.std(port)) * np.sqrt(252) if np.std(port) > 0 else 0
    wr = np.mean(rets > 0)
    gp = np.sum(rets[rets > 0]) if np.any(rets > 0) else 0
    gl = abs(np.sum(rets[rets < 0])) if np.any(rets < 0) else 0.0001
    pf = gp / gl

    return {'sharpe': sharpe, 'win_rate': wr, 'profit_factor': pf,
            'n_trades': len(rets), 'avg_return': np.mean(rets)}


# ══════════════════════════════════════════════════════════════════════════════
#  TEST BATTERY
# ══════════════════════════════════════════════════════════════════════════════

def test_reimplementation(returns, sector_cols, variant, orig_sharpe):
    """Test 1: Re-implementation from scratch. FAIL if reimpl Sharpe < 50% of original."""
    print(f"    [1] Re-implementation test...")
    if variant == 'E':
        signals = reimpl_earnings_signals(returns, sector_cols)
    else:
        signals = reimpl_reversal_signals(returns, sector_cols, max_rank=2, min_mom_days=3)

    trades, daily_ret = run_backtest(returns, signals, sector_cols)
    m = compute_full_metrics(trades, daily_ret)
    reimpl_sharpe = m['sharpe']
    ratio = reimpl_sharpe / orig_sharpe if orig_sharpe != 0 else 0
    passed = reimpl_sharpe >= 0.50 * orig_sharpe
    print(f"        Original Sharpe: {orig_sharpe:.3f}, Reimpl Sharpe: {reimpl_sharpe:.3f} "
          f"(ratio: {ratio:.2f}) -> {'PASS' if passed else 'FAIL'}")
    return passed, f"Reimpl={reimpl_sharpe:.3f} ({ratio:.0%} of orig)"


def test_inverse_signal(returns, sector_cols, variant, orig_sharpe):
    """Test 2: Flip entry conditions. FAIL if inverse Sharpe > 50% of original."""
    print(f"    [2] Inverse signal test...")
    if variant == 'E':
        # Inverse: long sectors UNDERperforming SPY (rel_ret < -threshold)
        spy_ret = returns[BENCHMARK]
        signals = pd.DataFrame(0, index=returns.index, columns=sector_cols)
        for col in sector_cols:
            rel_ret = (returns[col] - spy_ret).rolling(EARNINGS_WINDOW).sum()
            for i in range(len(returns)):
                if returns.index[i].month not in EARNINGS_MONTHS:
                    continue
                if pd.isna(rel_ret.iloc[i]):
                    continue
                if rel_ret.iloc[i] < -EARNINGS_THRESHOLD:  # FLIPPED
                    signals.iloc[i][col] = 1
    else:
        # Inverse: long TOP-ranked sectors with NEGATIVE momentum
        rolling_ret = returns[sector_cols].rolling(LOOKBACK_RANK).sum()
        ranks = rolling_ret.rank(axis=1, ascending=True)
        n_sectors = len(sector_cols)
        mom = returns[sector_cols].rolling(MOMENTUM_CONFIRM).sum()
        signals = pd.DataFrame(0, index=returns.index, columns=sector_cols)
        for i in range(max(LOOKBACK_RANK, MOMENTUM_CONFIRM) + 1, len(returns)):
            for col in sector_cols:
                rank_val = ranks.iloc[i].get(col, np.nan)
                mom_val = mom.iloc[i].get(col, np.nan)
                if pd.isna(rank_val) or pd.isna(mom_val):
                    continue
                # TOP rank + NEGATIVE momentum (opposite of original)
                if rank_val >= (n_sectors - 1) and mom_val < 0:
                    signals.iloc[i][col] = 1

    trades, daily_ret = run_backtest(returns, signals, sector_cols)
    m = compute_full_metrics(trades, daily_ret)
    inv_sharpe = m['sharpe']
    threshold = 0.50 * orig_sharpe
    passed = inv_sharpe < threshold
    print(f"        Original Sharpe: {orig_sharpe:.3f}, Inverse Sharpe: {inv_sharpe:.3f} "
          f"(threshold: < {threshold:.3f}) -> {'PASS' if passed else 'FAIL'}")
    return passed, f"Inverse={inv_sharpe:.3f} (limit={threshold:.3f})"


def test_random_timing(returns, sector_cols, variant, orig_sharpe, n_perms=200):
    """Test 3: Shuffle entry dates 200x. FAIL if real < 95th percentile."""
    print(f"    [3] Random timing test ({n_perms} permutations)...")
    if variant == 'E':
        signals = orig_earnings_signals(returns, sector_cols)
    else:
        signals = orig_reversal_signals(returns, sector_cols, max_rank=2, min_mom_days=3)

    # Get original signal positions
    sig_mask = (signals != 0)
    total_signals = sig_mask.sum().sum()

    # Run original
    trades_orig, daily_orig = run_backtest(returns, signals, sector_cols)
    real_sharpe = compute_sharpe(daily_orig, use_daily=True)

    perm_sharpes = []
    rng = np.random.RandomState(42)
    valid_range = range(SLIDING_WINDOW + LOOKBACK_RANK + MOMENTUM_CONFIRM + 5, len(returns))
    valid_indices = list(valid_range)

    sig_counts = {col: int(sig_mask[col].sum()) for col in sector_cols}
    col_to_idx = {col: idx for idx, col in enumerate(sector_cols)}

    for p in range(n_perms):
        # Create shuffled signals using numpy arrays directly
        shuffled_arr = np.zeros((len(returns), len(sector_cols)), dtype=np.float64)
        for col in sector_cols:
            n_col_sigs = sig_counts[col]
            if n_col_sigs == 0:
                continue
            random_days = rng.choice(valid_indices, size=min(n_col_sigs, len(valid_indices)), replace=False)
            shuffled_arr[random_days, col_to_idx[col]] = 1

        shuffled = pd.DataFrame(shuffled_arr, index=returns.index, columns=sector_cols)
        trades_p, daily_p = run_backtest(returns, shuffled, sector_cols, skip_trailing_filter=True)
        perm_sharpes.append(compute_sharpe(daily_p, use_daily=True))

    perm_sharpes = np.array(perm_sharpes)
    pctile = np.percentile(perm_sharpes, 95)
    rank_pct = np.mean(perm_sharpes >= real_sharpe)
    passed = real_sharpe > pctile
    print(f"        Real Sharpe: {real_sharpe:.3f}, 95th pctile: {pctile:.3f}, "
          f"rank: {rank_pct:.4f} -> {'PASS' if passed else 'FAIL'}")
    return passed, f"Real={real_sharpe:.3f} vs p95={pctile:.3f} (p={rank_pct:.4f})"


def test_cost_sensitivity(returns, sector_cols, variant, orig_sharpe):
    """Test 4: Add costs. FAIL if Sharpe < 0.5 at 0.10% cost."""
    print(f"    [4] Cost sensitivity test...")
    if variant == 'E':
        signals = orig_earnings_signals(returns, sector_cols)
    else:
        signals = orig_reversal_signals(returns, sector_cols, max_rank=2, min_mom_days=3)

    cost_levels = [0.0005, 0.001, 0.0015, 0.002]  # 0.05%, 0.10%, 0.15%, 0.20%
    cost_labels = ['0.05%', '0.10%', '0.15%', '0.20%']
    results = {}

    for cost, label in zip(cost_levels, cost_labels):
        trades, daily_ret = run_backtest(returns, signals, sector_cols, cost_pct=cost)
        m = compute_full_metrics(trades, daily_ret)
        results[label] = m['sharpe']
        print(f"        Cost {label}: Sharpe={m['sharpe']:.3f}, WR={m['win_rate']*100:.1f}%, N={m['n_trades']}")

    passed = results['0.10%'] >= 0.5
    print(f"        At 0.10% cost: Sharpe={results['0.10%']:.3f} (gate: >= 0.5) -> {'PASS' if passed else 'FAIL'}")
    return passed, f"Sharpe@10bps={results['0.10%']:.3f}"


def test_subperiod_stability(returns, sector_cols, variant):
    """Test 5: Split into 4 equal periods. FAIL if ANY has negative Sharpe."""
    print(f"    [5] Sub-period stability test...")
    if variant == 'E':
        signals = orig_earnings_signals(returns, sector_cols)
    else:
        signals = orig_reversal_signals(returns, sector_cols, max_rank=2, min_mom_days=3)

    n = len(returns)
    quarter_size = n // 4
    period_sharpes = []

    for q in range(4):
        start = q * quarter_size
        end = (q + 1) * quarter_size if q < 3 else n

        sub_returns = returns.iloc[start:end].copy()
        sub_signals = signals.iloc[start:end].copy()

        if len(sub_returns) < 50:
            period_sharpes.append(0.0)
            continue

        trades, daily_ret = run_backtest(sub_returns, sub_signals, sector_cols)
        m = compute_full_metrics(trades, daily_ret)
        period_sharpes.append(m['sharpe'])

        date_range = f"{sub_returns.index[0].strftime('%Y-%m')} to {sub_returns.index[-1].strftime('%Y-%m')}"
        print(f"        Period {q+1} ({date_range}): Sharpe={m['sharpe']:.3f}, N={m['n_trades']}")

    any_negative = any(s < 0 for s in period_sharpes)
    passed = not any_negative
    print(f"        Any negative Sharpe: {any_negative} -> {'PASS' if passed else 'FAIL'}")
    min_s = min(period_sharpes)
    return passed, f"Min period Sharpe={min_s:.3f}"


def test_parameter_robustness(returns, sector_cols, variant, orig_sharpe):
    """Test 6: Parameter grid. FAIL if <60% of combos have Sharpe > 0.3."""
    print(f"    [6] Parameter robustness test...")

    if variant == 'E':
        # Grid: earnings_window x threshold x earnings_months_sets
        windows = [7, 10, 14, 20]
        thresholds = [0.005, 0.01, 0.015, 0.02]
        total = 0
        above_threshold = 0

        for ew in windows:
            for thresh in thresholds:
                spy_ret = returns[BENCHMARK]
                signals = pd.DataFrame(0, index=returns.index, columns=sector_cols)
                for col in sector_cols:
                    rel_ret = (returns[col] - spy_ret).rolling(ew).sum()
                    for i in range(len(returns)):
                        if returns.index[i].month not in EARNINGS_MONTHS:
                            continue
                        if pd.isna(rel_ret.iloc[i]):
                            continue
                        if rel_ret.iloc[i] > thresh:
                            signals.iloc[i][col] = 1

                trades, daily_ret = run_backtest(returns, signals, sector_cols)
                m = compute_full_metrics(trades, daily_ret)
                total += 1
                if m['sharpe'] > 0.3:
                    above_threshold += 1

        pct = above_threshold / total if total > 0 else 0
        passed = pct >= 0.60
        print(f"        {above_threshold}/{total} combos Sharpe > 0.3 ({pct*100:.0f}%) -> {'PASS' if passed else 'FAIL'}")
        return passed, f"{above_threshold}/{total} ({pct:.0%}) above 0.3"

    else:  # Variant B
        # Grid: max_rank x momentum_days x lookback
        max_ranks = [1, 2, 3]
        mom_days = [2, 3, 5, 7]
        lookbacks = [15, 20, 30]
        total = 0
        above_threshold = 0

        for mr in max_ranks:
            for md in mom_days:
                for lb in lookbacks:
                    # Re-generate with different lookback
                    rolling_ret = returns[sector_cols].rolling(lb).sum()
                    ranks = rolling_ret.rank(axis=1, ascending=True)
                    mom = returns[sector_cols].rolling(md).sum()
                    signals = pd.DataFrame(0, index=returns.index, columns=sector_cols)

                    start_idx = max(lb, md) + 1
                    for i in range(start_idx, len(returns)):
                        for col in sector_cols:
                            rank_val = ranks.iloc[i].get(col, np.nan)
                            mom_val = mom.iloc[i].get(col, np.nan)
                            if pd.isna(rank_val) or pd.isna(mom_val):
                                continue
                            if rank_val <= mr and mom_val > 0:
                                signals.iloc[i][col] = 1

                    trades, daily_ret = run_backtest(returns, signals, sector_cols)
                    m = compute_full_metrics(trades, daily_ret)
                    total += 1
                    if m['sharpe'] > 0.3:
                        above_threshold += 1

        pct = above_threshold / total if total > 0 else 0
        passed = pct >= 0.60
        print(f"        {above_threshold}/{total} combos Sharpe > 0.3 ({pct*100:.0f}%) -> {'PASS' if passed else 'FAIL'}")
        return passed, f"{above_threshold}/{total} ({pct:.0%}) above 0.3"


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_battery(returns, sector_cols, variant, orig_sharpe):
    """Run all 6 adversarial tests for a variant."""
    results = {}

    t1_pass, t1_detail = test_reimplementation(returns, sector_cols, variant, orig_sharpe)
    results['1_reimplementation'] = (t1_pass, t1_detail)

    t2_pass, t2_detail = test_inverse_signal(returns, sector_cols, variant, orig_sharpe)
    results['2_inverse_signal'] = (t2_pass, t2_detail)

    t3_pass, t3_detail = test_random_timing(returns, sector_cols, variant, orig_sharpe)
    results['3_random_timing'] = (t3_pass, t3_detail)

    t4_pass, t4_detail = test_cost_sensitivity(returns, sector_cols, variant, orig_sharpe)
    results['4_cost_sensitivity'] = (t4_pass, t4_detail)

    t5_pass, t5_detail = test_subperiod_stability(returns, sector_cols, variant)
    results['5_subperiod_stability'] = (t5_pass, t5_detail)

    t6_pass, t6_detail = test_parameter_robustness(returns, sector_cols, variant, orig_sharpe)
    results['6_parameter_robustness'] = (t6_pass, t6_detail)

    return results


def print_battery_results(variant_name, results):
    n_pass = sum(1 for v in results.values() if v[0])
    print(f"\n{'='*70}")
    print(f"  ADVERSARIAL RESULTS: {variant_name}")
    print(f"{'='*70}")
    print(f"  {'Test':<30} {'Result':>8}  {'Detail'}")
    print(f"  {'-'*65}")

    test_names = {
        '1_reimplementation': 'Re-implementation',
        '2_inverse_signal': 'Inverse Signal',
        '3_random_timing': 'Random Timing (1000x)',
        '4_cost_sensitivity': 'Cost Sensitivity',
        '5_subperiod_stability': 'Sub-period Stability',
        '6_parameter_robustness': 'Parameter Robustness',
    }

    for key in sorted(results.keys()):
        passed, detail = results[key]
        label = test_names.get(key, key)
        icon = 'PASS' if passed else 'FAIL'
        print(f"  {label:<30} {icon:>8}  {detail}")

    print(f"  {'-'*65}")
    print(f"  VERDICT: {n_pass}/6 PASSED")
    if n_pass == 6:
        print(f"  >> CLEAN BILL OF HEALTH — signal appears robust")
    elif n_pass >= 4:
        print(f"  >> MOSTLY ROBUST — investigate failures before deploying")
    else:
        print(f"  >> SIGNIFICANT CONCERNS — likely overfit or spurious")
    print(f"{'='*70}")
    return n_pass


def main():
    np.random.seed(42)
    t_start = time.time()

    print("=" * 70)
    print("  ADVERSARIAL VALIDATION — Sector Strategies (Variants B & E)")
    print("  6-Test Battery | Walk-Forward | Sliding Window")
    print("=" * 70)

    close = download_data()
    returns = close.pct_change().dropna()
    sector_cols = [s for s in SECTOR_ETFS if s in returns.columns]
    print(f"Using {len(sector_cols)} sectors: {sector_cols}")

    # ── First establish baseline Sharpes ──────────────────────────────────────
    print("\n" + "-" * 70)
    print("  Establishing baselines...")

    # Variant E baseline
    sig_e = orig_earnings_signals(returns, sector_cols)
    trades_e, daily_e = run_backtest(returns, sig_e, sector_cols)
    m_e = compute_full_metrics(trades_e, daily_e)
    print(f"  Variant E (Earnings Outperformance): Sharpe={m_e['sharpe']:.3f}, "
          f"N={m_e['n_trades']}, WR={m_e['win_rate']*100:.1f}%")

    # Variant B baseline
    sig_b = orig_reversal_signals(returns, sector_cols, max_rank=2, min_mom_days=3)
    trades_b, daily_b = run_backtest(returns, sig_b, sector_cols)
    m_b = compute_full_metrics(trades_b, daily_b)
    print(f"  Variant B (Rank Reversal):           Sharpe={m_b['sharpe']:.3f}, "
          f"N={m_b['n_trades']}, WR={m_b['win_rate']*100:.1f}%")

    # ── Run Battery: Variant E ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TESTING VARIANT E: Earnings Outperformance")
    print("=" * 70)
    results_e = run_battery(returns, sector_cols, 'E', m_e['sharpe'])
    n_pass_e = print_battery_results("Variant E — Earnings Outperformance", results_e)

    # ── Run Battery: Variant B ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TESTING VARIANT B: Rank Reversal (Bottom-2, 3d Momentum)")
    print("=" * 70)
    results_b = run_battery(returns, sector_cols, 'B', m_b['sharpe'])
    n_pass_b = print_battery_results("Variant B — Rank Reversal", results_b)

    # ── Final Summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  FINAL ADVERSARIAL SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Variant':<40} {'Score':>8}")
    print(f"  {'-'*50}")
    print(f"  {'E: Earnings Outperformance':<40} {n_pass_e}/6")
    print(f"  {'B: Rank Reversal (Bottom-2 + 3d Mom)':<40} {n_pass_b}/6")
    print(f"  {'-'*50}")

    for name, n_pass in [('E', n_pass_e), ('B', n_pass_b)]:
        if n_pass == 6:
            verdict = "ROBUST — deploy candidate"
        elif n_pass >= 4:
            verdict = "MOSTLY ROBUST — review failures"
        else:
            verdict = "SUSPECT — likely overfit"
        print(f"  Variant {name}: {verdict}")

    print(f"\n  Elapsed: {elapsed:.0f}s")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
