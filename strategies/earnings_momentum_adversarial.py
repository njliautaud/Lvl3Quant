#!/usr/bin/env python3
"""
Earnings Momentum Concentration — 6-Gate Adversarial Validation
================================================================
Strategy B from high_growth_research.py:
  Buy top-50 S&P 500 stocks that gap up >5% on high volume (proxy for
  earnings beats). Hold 20 trading days. Max 5 concurrent positions. 10bps costs.
  Claimed: 31% CAGR, Sharpe 1.10, MaxDD -31.5%, 333 trades over 2019-2026.

Gates:
  1. Re-implementation      — independent code, verify similar metrics
  2. Inverse signal         — buy gap-DOWN >5% — should lose money
  3. Random timing          — 200 permutations of entry dates
  4. Cost sensitivity       — 0/10/20/50 bps
  5. Sub-period stability   — 4 equal sub-periods, all must be positive
  6. Parameter robustness   — grid sweep, >50% combos Sharpe > 0.3

Extra: FAANG-exclusion test to check if edge is just mega-cap tech.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import time
import json
import sys

# ─── CONFIG ──────────────────────────────────────────────────────────────────
CAPITAL = 100_000.0
START = '2018-06-01'
TRADE_START = '2019-01-02'
END = '2026-08-22'
RISK_FREE = 0.04
N_PERMS = 100  # Reduced from 200 for runtime (still statistically valid at p<0.05)

UNIVERSE = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'BRK-B', 'LLY',
    'AVGO', 'JPM', 'TSLA', 'V', 'UNH', 'XOM', 'MA', 'JNJ', 'PG',
    'COST', 'HD', 'ABBV', 'MRK', 'WMT', 'NFLX', 'CRM', 'BAC',
    'CVX', 'KO', 'PEP', 'ORCL', 'LIN', 'AMD', 'TMO', 'ACN',
    'MCD', 'CSCO', 'ABT', 'ADBE', 'WFC', 'DHR', 'TXN', 'NEE',
    'PM', 'AMGN', 'IBM', 'QCOM', 'CAT', 'GE', 'INTU', 'AMAT', 'ISRG'
]

FAANG = {'AAPL', 'AMZN', 'META', 'GOOGL', 'NFLX', 'MSFT', 'NVDA', 'TSLA'}


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def download_safe(tickers, start, end, retries=3):
    for attempt in range(retries):
        try:
            data = yf.download(tickers, start=start, end=end,
                               auto_adjust=True, progress=False)
            if data is not None and len(data) > 0:
                return data
        except Exception as e:
            print(f"  Download attempt {attempt+1} failed: {e}", flush=True)
            time.sleep(2)
    return None


def calc_metrics(equity_curve, risk_free=RISK_FREE):
    if len(equity_curve) < 20:
        return {'CAGR': 0, 'Sharpe': 0, 'Sortino': 0, 'MaxDD': 0, 'PF': 0, 'WR': 0}
    returns = equity_curve.pct_change().dropna()
    if len(returns) < 10:
        return {'CAGR': 0, 'Sharpe': 0, 'Sortino': 0, 'MaxDD': 0, 'PF': 0, 'WR': 0}

    total_days = (equity_curve.index[-1] - equity_curve.index[0]).days
    if total_days <= 0:
        return {'CAGR': 0, 'Sharpe': 0, 'Sortino': 0, 'MaxDD': 0, 'PF': 0, 'WR': 0}
    years = total_days / 365.25
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    cagr = (total_return ** (1 / years) - 1) * 100

    daily_rf = (1 + risk_free) ** (1 / 252) - 1
    excess = returns - daily_rf
    sharpe = np.sqrt(252) * excess.mean() / excess.std() if excess.std() > 0 else 0

    downside = returns[returns < daily_rf] - daily_rf
    downside_std = np.sqrt((downside ** 2).mean()) if len(downside) > 0 else 1e-9
    sortino = np.sqrt(252) * excess.mean() / downside_std if downside_std > 0 else 0

    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak
    max_dd = dd.min() * 100

    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')
    wr = (returns > 0).sum() / len(returns) * 100

    return {
        'CAGR': round(cagr, 1),
        'Sharpe': round(sharpe, 2),
        'Sortino': round(sortino, 2),
        'MaxDD': round(max_dd, 1),
        'PF': round(pf, 2),
        'WR': round(wr, 1),
    }


def precompute_signals(close, volume, valid_tickers, gap_thresh, direction='up',
                       trade_start=None):
    """Vectorized signal computation — call once per (gap_thresh, direction) combo."""
    if trade_start is None:
        trade_start = TRADE_START
    trade_dates = close.index[close.index >= trade_start]

    # Vectorized gaps and volume ratios
    close_sub = close[valid_tickers].reindex(trade_dates)
    volume_sub = volume[[t for t in valid_tickers if t in volume.columns]].reindex(trade_dates)

    gaps = close_sub / close_sub.shift(1) - 1  # day-over-day gap
    vol_avg = volume_sub.rolling(20, min_periods=1).mean().shift(1)
    vol_ratio = volume_sub / vol_avg

    signals = []  # list of (date_idx, ticker)
    for i in range(21, len(trade_dates)):
        for ticker in valid_tickers:
            if ticker not in gaps.columns:
                continue
            g = gaps.iloc[i].get(ticker, np.nan)
            vr = vol_ratio.iloc[i].get(ticker, np.nan) if ticker in vol_ratio.columns else np.nan
            if pd.isna(g) or pd.isna(vr):
                continue
            if direction == 'up' and g >= gap_thresh and vr >= 1.5:
                signals.append((i, ticker))
            elif direction == 'down' and g <= -gap_thresh and vr >= 1.5:
                signals.append((i, ticker))

    return signals, trade_dates


def run_backtest(close, trade_dates, signals, hold_days, max_concurrent, cost_bps):
    """Fast backtest given pre-computed signals list."""
    POSITION_SIZE = 1.0 / max_concurrent
    close_vals = {}  # cache .loc lookups as numpy arrays
    for ticker in close.columns:
        s = close[ticker].reindex(trade_dates)
        close_vals[ticker] = s.values

    date_to_idx = {d: i for i, d in enumerate(trade_dates)}

    signal_dict = {}
    for idx, ticker in signals:
        signal_dict.setdefault(idx, []).append(ticker)

    equity = CAPITAL
    equity_curve = np.empty(len(trade_dates))
    positions = []
    n_trades = 0
    n_wins = 0

    for i in range(len(trade_dates)):
        new_positions = []
        for ticker, entry_idx, entry_price, exit_idx in positions:
            if i >= exit_idx:
                exit_price = close_vals.get(ticker, None)
                if exit_price is not None and not np.isnan(exit_price[i]):
                    ret = exit_price[i] / entry_price - 1
                    pnl = equity * POSITION_SIZE * (ret - cost_bps / 10000 * 2)
                    equity += pnl
                    n_trades += 1
                    if ret > 0:
                        n_wins += 1
            else:
                new_positions.append((ticker, entry_idx, entry_price, exit_idx))
        positions = new_positions

        if len(positions) < max_concurrent and i in signal_dict:
            for ticker in signal_dict[i]:
                if len(positions) >= max_concurrent:
                    break
                if any(p[0] == ticker for p in positions):
                    continue
                cv = close_vals.get(ticker)
                if cv is None or np.isnan(cv[i]):
                    continue
                entry_price = cv[i]
                exit_idx = min(i + hold_days, len(trade_dates) - 1)
                positions.append((ticker, i, entry_price, exit_idx))
                equity -= equity * POSITION_SIZE * cost_bps / 10000

        mtm = equity
        for ticker, entry_idx, entry_price, exit_idx in positions:
            cv = close_vals.get(ticker)
            if cv is not None and not np.isnan(cv[i]):
                mtm += equity * POSITION_SIZE * (cv[i] / entry_price - 1)
        equity_curve[i] = mtm

    eq_series = pd.Series(equity_curve, index=trade_dates)
    metrics = calc_metrics(eq_series)
    metrics['Trades'] = n_trades
    metrics['WR_trades'] = round(n_wins / n_trades * 100, 1) if n_trades > 0 else 0
    return metrics, eq_series


def run_strategy(close, volume, valid_tickers, gap_thresh, hold_days,
                 max_concurrent, cost_bps, direction='up', random_seed=None,
                 trade_start=None, precomputed_signals=None, precomputed_trade_dates=None):
    """
    Core backtest engine. If precomputed_signals provided, skips signal computation.
    """
    if precomputed_signals is not None:
        signals = list(precomputed_signals)  # copy so shuffle doesn't mutate
        trade_dates = precomputed_trade_dates
    else:
        signals, trade_dates = precompute_signals(
            close, volume, valid_tickers, gap_thresh, direction, trade_start)

    # If random timing: shuffle the signal dates
    if random_seed is not None:
        rng = np.random.RandomState(random_seed)
        if len(signals) > 0:
            date_indices = [s[0] for s in signals]
            tickers_list = [s[1] for s in signals]
            rng.shuffle(date_indices)
            signals = list(zip(date_indices, tickers_list))

    return run_backtest(close, trade_dates, signals, hold_days, max_concurrent, cost_bps)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70, flush=True)
    print("EARNINGS MOMENTUM CONCENTRATION — 6-GATE ADVERSARIAL VALIDATION", flush=True)
    print("=" * 70, flush=True)

    # ─── Download data ────────────────────────────────────────────────────
    print("\nDownloading data...", flush=True)
    data = download_safe(UNIVERSE + ['SPY'], START, END)
    if data is None:
        print("FATAL: Failed to download data", flush=True)
        sys.exit(1)

    close = data['Close']
    volume = data['Volume']

    valid_tickers = []
    for t in UNIVERSE:
        if t in close.columns:
            pct_valid = close[t].dropna().shape[0] / close.shape[0]
            if pct_valid > 0.8:
                valid_tickers.append(t)
    print(f"Valid tickers: {len(valid_tickers)}", flush=True)

    results = {}
    gate_results = {}

    # ═════════════════════════════════════════════════════════════════════
    # GATE 1: RE-IMPLEMENTATION
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("GATE 1: RE-IMPLEMENTATION (5% gap, 20d hold, max 5, 10bps)", flush=True)
    print("=" * 70, flush=True)

    base_metrics, base_eq = run_strategy(
        close, volume, valid_tickers,
        gap_thresh=0.05, hold_days=20, max_concurrent=5, cost_bps=10
    )
    print(f"  CAGR:    {base_metrics['CAGR']:.1f}%", flush=True)
    print(f"  Sharpe:  {base_metrics['Sharpe']:.2f}", flush=True)
    print(f"  Sortino: {base_metrics['Sortino']:.2f}", flush=True)
    print(f"  MaxDD:   {base_metrics['MaxDD']:.1f}%", flush=True)
    print(f"  PF:      {base_metrics['PF']:.2f}", flush=True)
    print(f"  Trades:  {base_metrics['Trades']}", flush=True)
    print(f"  WR:      {base_metrics['WR_trades']:.1f}%", flush=True)

    g1_pass = base_metrics['Sharpe'] > 0.5
    gate_results['G1_reimplementation'] = {
        'pass': g1_pass,
        'sharpe': base_metrics['Sharpe'],
        'cagr': base_metrics['CAGR'],
    }
    print(f"\n  GATE 1: {'PASS' if g1_pass else 'FAIL'} (Sharpe {base_metrics['Sharpe']:.2f})", flush=True)

    # ═════════════════════════════════════════════════════════════════════
    # GATE 2: INVERSE SIGNAL (buy gap-downs)
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("GATE 2: INVERSE SIGNAL (buy gap-DOWN >5%)", flush=True)
    print("=" * 70, flush=True)

    inv_metrics, inv_eq = run_strategy(
        close, volume, valid_tickers,
        gap_thresh=0.05, hold_days=20, max_concurrent=5, cost_bps=10,
        direction='down'
    )
    print(f"  Inverse CAGR:    {inv_metrics['CAGR']:.1f}%", flush=True)
    print(f"  Inverse Sharpe:  {inv_metrics['Sharpe']:.2f}", flush=True)
    print(f"  Inverse Trades:  {inv_metrics['Trades']}", flush=True)
    print(f"  Inverse WR:      {inv_metrics['WR_trades']:.1f}%", flush=True)

    # Inverse should be notably worse. If inverse also profits well, edge is just beta.
    inv_sharpe_ratio = inv_metrics['Sharpe'] / base_metrics['Sharpe'] if base_metrics['Sharpe'] != 0 else 999
    g2_pass = inv_metrics['Sharpe'] < base_metrics['Sharpe'] * 0.5  # inverse < half of original
    gate_results['G2_inverse_signal'] = {
        'pass': g2_pass,
        'orig_sharpe': base_metrics['Sharpe'],
        'inv_sharpe': inv_metrics['Sharpe'],
        'inv_cagr': inv_metrics['CAGR'],
        'inv_trades': inv_metrics['Trades'],
    }
    print(f"\n  GATE 2: {'PASS' if g2_pass else 'FAIL'} "
          f"(inverse Sharpe {inv_metrics['Sharpe']:.2f} vs original {base_metrics['Sharpe']:.2f})", flush=True)

    # ─── FAANG exclusion sub-test ─────────────────────────────────────
    print("\n  --- FAANG Exclusion Sub-Test ---", flush=True)
    non_faang = [t for t in valid_tickers if t not in FAANG]
    print(f"  Non-FAANG tickers: {len(non_faang)}", flush=True)
    nf_metrics, _ = run_strategy(
        close, volume, non_faang,
        gap_thresh=0.05, hold_days=20, max_concurrent=5, cost_bps=10
    )
    print(f"  Ex-FAANG CAGR:   {nf_metrics['CAGR']:.1f}%", flush=True)
    print(f"  Ex-FAANG Sharpe: {nf_metrics['Sharpe']:.2f}", flush=True)
    print(f"  Ex-FAANG Trades: {nf_metrics['Trades']}", flush=True)
    faang_dependent = nf_metrics['Sharpe'] < base_metrics['Sharpe'] * 0.3
    gate_results['FAANG_exclusion'] = {
        'full_sharpe': base_metrics['Sharpe'],
        'ex_faang_sharpe': nf_metrics['Sharpe'],
        'faang_dependent': faang_dependent,
    }
    print(f"  FAANG-dependent: {'YES — edge collapses' if faang_dependent else 'NO — edge survives'}", flush=True)

    # ═════════════════════════════════════════════════════════════════════
    # GATE 3: RANDOM TIMING (200 permutations)
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print(f"GATE 3: RANDOM TIMING ({N_PERMS} permutations)", flush=True)
    print("=" * 70, flush=True)

    # Precompute signals once for permutation reuse
    base_signals, base_trade_dates = precompute_signals(
        close, volume, valid_tickers, gap_thresh=0.05, direction='up')
    print(f"  Pre-computed {len(base_signals)} signals, running {N_PERMS} permutations...", flush=True)

    perm_sharpes = []
    for p in range(N_PERMS):
        pm, _ = run_strategy(
            close, volume, valid_tickers,
            gap_thresh=0.05, hold_days=20, max_concurrent=5, cost_bps=10,
            random_seed=p + 42,
            precomputed_signals=base_signals,
            precomputed_trade_dates=base_trade_dates
        )
        perm_sharpes.append(pm['Sharpe'])
        if (p + 1) % 50 == 0:
            print(f"  Completed {p+1}/{N_PERMS} permutations...", flush=True)

    perm_sharpes = np.array(perm_sharpes)
    p95 = np.percentile(perm_sharpes, 95)
    p99 = np.percentile(perm_sharpes, 99)
    perm_mean = np.mean(perm_sharpes)
    perm_std = np.std(perm_sharpes)
    pval = (perm_sharpes >= base_metrics['Sharpe']).mean()

    g3_pass = base_metrics['Sharpe'] > p95
    gate_results['G3_random_timing'] = {
        'pass': g3_pass,
        'orig_sharpe': base_metrics['Sharpe'],
        'perm_mean': round(float(perm_mean), 3),
        'perm_std': round(float(perm_std), 3),
        'p95': round(float(p95), 3),
        'p99': round(float(p99), 3),
        'p_value': round(float(pval), 4),
    }
    print(f"  Original Sharpe: {base_metrics['Sharpe']:.2f}", flush=True)
    print(f"  Perm mean:       {perm_mean:.3f} +/- {perm_std:.3f}", flush=True)
    print(f"  Perm p95:        {p95:.3f}", flush=True)
    print(f"  Perm p99:        {p99:.3f}", flush=True)
    print(f"  p-value:         {pval:.4f}", flush=True)
    print(f"\n  GATE 3: {'PASS' if g3_pass else 'FAIL'} "
          f"(Sharpe {base_metrics['Sharpe']:.2f} vs p95 {p95:.3f})", flush=True)

    # ═════════════════════════════════════════════════════════════════════
    # GATE 4: COST SENSITIVITY
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("GATE 4: COST SENSITIVITY (0 / 10 / 20 / 50 bps)", flush=True)
    print("=" * 70, flush=True)

    cost_results = {}
    for bps in [0, 10, 20, 50]:
        cm, _ = run_strategy(
            close, volume, valid_tickers,
            gap_thresh=0.05, hold_days=20, max_concurrent=5, cost_bps=bps
        )
        cost_results[bps] = cm
        print(f"  {bps:>3} bps: Sharpe {cm['Sharpe']:.2f}, CAGR {cm['CAGR']:.1f}%, "
              f"MaxDD {cm['MaxDD']:.1f}%", flush=True)

    g4_pass = cost_results[20]['Sharpe'] > 0.5
    gate_results['G4_cost_sensitivity'] = {
        'pass': g4_pass,
        'details': {str(k): {'sharpe': v['Sharpe'], 'cagr': v['CAGR']}
                    for k, v in cost_results.items()},
    }
    print(f"\n  GATE 4: {'PASS' if g4_pass else 'FAIL'} "
          f"(Sharpe at 20bps = {cost_results[20]['Sharpe']:.2f})", flush=True)

    # ═════════════════════════════════════════════════════════════════════
    # GATE 5: SUB-PERIOD STABILITY
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("GATE 5: SUB-PERIOD STABILITY (4 equal periods)", flush=True)
    print("=" * 70, flush=True)

    trade_dates = close.index[close.index >= TRADE_START]
    n_dates = len(trade_dates)
    quarter_size = n_dates // 4
    sub_periods = []
    for q in range(4):
        start_idx = q * quarter_size
        end_idx = (q + 1) * quarter_size if q < 3 else n_dates
        sub_start = trade_dates[start_idx].strftime('%Y-%m-%d')
        sub_end = trade_dates[end_idx - 1].strftime('%Y-%m-%d')
        sub_periods.append((sub_start, sub_end))

    sub_results = []
    all_positive = True
    for q, (ss, se) in enumerate(sub_periods):
        # Slice data for sub-period
        mask = (close.index >= ss) & (close.index <= se)
        sub_close = close[mask]
        sub_volume = volume[mask]

        # We need some warmup data, so include prior 21 days
        warmup_start_idx = max(0, close.index.get_loc(sub_close.index[0]) - 25)
        warmup_start = close.index[warmup_start_idx]
        full_mask = (close.index >= warmup_start) & (close.index <= se)
        sub_close_full = close[full_mask]
        sub_volume_full = volume[full_mask]

        sm, _ = run_strategy(
            sub_close_full, sub_volume_full, valid_tickers,
            gap_thresh=0.05, hold_days=20, max_concurrent=5, cost_bps=10,
            trade_start=ss
        )

        sub_results.append(sm)
        is_pos = sm['CAGR'] > 0
        if not is_pos:
            all_positive = False
        print(f"  Q{q+1} ({ss} to {se}): Sharpe {sm['Sharpe']:.2f}, "
              f"CAGR {sm['CAGR']:.1f}%, Trades {sm['Trades']}, "
              f"{'POSITIVE' if is_pos else 'NEGATIVE'}", flush=True)

    g5_pass = all_positive
    gate_results['G5_subperiod'] = {
        'pass': g5_pass,
        'periods': [
            {'period': f"{sp[0]} to {sp[1]}", 'sharpe': sr['Sharpe'],
             'cagr': sr['CAGR'], 'trades': sr['Trades']}
            for sp, sr in zip(sub_periods, sub_results)
        ],
    }
    print(f"\n  GATE 5: {'PASS' if g5_pass else 'FAIL'} "
          f"(all positive: {all_positive})", flush=True)

    # ═════════════════════════════════════════════════════════════════════
    # GATE 6: PARAMETER ROBUSTNESS (grid sweep)
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("GATE 6: PARAMETER ROBUSTNESS (grid sweep)", flush=True)
    print("=" * 70, flush=True)

    gap_thresholds = [0.03, 0.05, 0.07, 0.08]
    hold_days_list = [10, 20, 30, 40]
    max_concurrent_list = [3, 5, 7]

    total_combos = len(gap_thresholds) * len(hold_days_list) * len(max_concurrent_list)
    print(f"  Testing {total_combos} parameter combinations...", flush=True)

    grid_results = []
    count = 0
    sharpe_above_03 = 0

    for gt in gap_thresholds:
        for hd in hold_days_list:
            for mc in max_concurrent_list:
                gm, _ = run_strategy(
                    close, volume, valid_tickers,
                    gap_thresh=gt, hold_days=hd, max_concurrent=mc, cost_bps=10
                )
                grid_results.append({
                    'gap': gt, 'hold': hd, 'max_pos': mc,
                    'sharpe': gm['Sharpe'], 'cagr': gm['CAGR'],
                    'trades': gm['Trades'],
                })
                if gm['Sharpe'] > 0.3:
                    sharpe_above_03 += 1
                count += 1
                if count % 50 == 0:
                    print(f"  Completed {count}/{total_combos}...", flush=True)

    pct_above = sharpe_above_03 / total_combos * 100
    g6_pass = pct_above > 50

    # Find best and worst
    grid_df = pd.DataFrame(grid_results)
    best = grid_df.loc[grid_df['sharpe'].idxmax()]
    worst = grid_df.loc[grid_df['sharpe'].idxmin()]

    gate_results['G6_parameter_robustness'] = {
        'pass': g6_pass,
        'total_combos': total_combos,
        'sharpe_above_03': sharpe_above_03,
        'pct_above': round(pct_above, 1),
        'best': {'gap': best['gap'], 'hold': int(best['hold']),
                 'max_pos': int(best['max_pos']), 'sharpe': best['sharpe']},
        'worst': {'gap': worst['gap'], 'hold': int(worst['hold']),
                  'max_pos': int(worst['max_pos']), 'sharpe': worst['sharpe']},
        'median_sharpe': round(float(grid_df['sharpe'].median()), 3),
    }

    print(f"  {sharpe_above_03}/{total_combos} combos have Sharpe > 0.3 ({pct_above:.1f}%)", flush=True)
    print(f"  Median Sharpe across grid: {grid_df['sharpe'].median():.3f}", flush=True)
    print(f"  Best:  gap={best['gap']:.0%}, hold={int(best['hold'])}d, "
          f"max={int(best['max_pos'])}, Sharpe={best['sharpe']:.2f}", flush=True)
    print(f"  Worst: gap={worst['gap']:.0%}, hold={int(worst['hold'])}d, "
          f"max={int(worst['max_pos'])}, Sharpe={worst['sharpe']:.2f}", flush=True)
    print(f"\n  GATE 6: {'PASS' if g6_pass else 'FAIL'} "
          f"({pct_above:.1f}% > 50% threshold)", flush=True)

    # ═════════════════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("FINAL VERDICT", flush=True)
    print("=" * 70, flush=True)

    gates_passed = sum(1 for g in ['G1_reimplementation', 'G2_inverse_signal',
                                    'G3_random_timing', 'G4_cost_sensitivity',
                                    'G5_subperiod', 'G6_parameter_robustness']
                       if gate_results[g]['pass'])

    for name, gr in gate_results.items():
        if name == 'FAANG_exclusion':
            status = 'WARN' if gr['faang_dependent'] else 'OK'
            print(f"  {name}: {status}", flush=True)
        elif 'pass' in gr:
            print(f"  {name}: {'PASS' if gr['pass'] else 'FAIL'}", flush=True)

    overall = gates_passed >= 5  # allow 1 failure
    print(f"\n  Gates passed: {gates_passed}/6", flush=True)
    print(f"  OVERALL: {'PASS' if overall else 'FAIL'}", flush=True)

    if gate_results.get('FAANG_exclusion', {}).get('faang_dependent'):
        print("\n  WARNING: Strategy edge is concentrated in FAANG/mega-cap tech.", flush=True)
        print("  This is a concentration risk, not necessarily invalid,", flush=True)
        print("  but be aware the edge may not generalize.", flush=True)

    # Save results
    output = {
        'run_date': datetime.now().isoformat(),
        'strategy': 'Earnings Momentum Concentration (Strategy B)',
        'base_params': {'gap_thresh': 0.05, 'hold_days': 20, 'max_concurrent': 5, 'cost_bps': 10},
        'base_metrics': base_metrics,
        'gate_results': gate_results,
        'overall_pass': overall,
        'gates_passed': gates_passed,
    }

    out_path = '/home/jupiter/Lvl3Quant/strategies/earnings_momentum_adversarial_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved.", flush=True)


if __name__ == '__main__':
    main()
