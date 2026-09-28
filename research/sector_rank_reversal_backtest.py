#!/usr/bin/env python3
"""
Sector-Level Rank Reversal Backtest (v2 — refined)
====================================================
Primary: Rank Reversal at Extremes (long-only, refined filters)
Secondary: Earnings Surprise Proxy (long-only)

Key changes from v1:
- Long-only (short side was -23bps avg, destroying value)
- Portfolio-level drawdown (not sum of all concurrent trades)
- Stricter reversal entry: rank=1 only (worst of 11), with momentum confirm
- Walk-forward SLIDING window with in-sample quality filter

Author: Claude Opus 4.6 | Date: 2026-08-18
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from datetime import datetime, timedelta
from collections import defaultdict
import sys

# ── Config ──────────────────────────────────────────────────────────────────
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

SHARPE_GATE = 0.5
REGIME_GAP_GATE = 0.50
PERM_P_GATE = 0.05
MIN_TRADES = 30


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
    print(f"Date range: {close.index[0].date()} to {close.index[-1].date()}")
    missing = [t for t in SECTOR_ETFS if t not in close.columns]
    if missing:
        print(f"WARNING: Missing tickers: {missing}")
    return close


def compute_returns(close):
    return close.pct_change().dropna()


def compute_sector_ranks(returns, sector_cols):
    rolling_ret = returns[sector_cols].rolling(LOOKBACK_RANK).sum()
    ranks = rolling_ret.rank(axis=1, ascending=True)
    return ranks


def is_earnings_season(date):
    return date.month in EARNINGS_MONTHS


# ── Strategy: Earnings Surprise Proxy (LONG ONLY) ──────────────────────────

def generate_earnings_long_signals(returns, sector_cols):
    spy_ret = returns[BENCHMARK]
    signals = pd.DataFrame(0, index=returns.index, columns=sector_cols)
    for col in sector_cols:
        rel_ret = (returns[col] - spy_ret).rolling(EARNINGS_WINDOW).sum()
        for i in range(len(returns)):
            if not is_earnings_season(returns.index[i]):
                continue
            if pd.isna(rel_ret.iloc[i]):
                continue
            if rel_ret.iloc[i] > EARNINGS_THRESHOLD:
                signals.iloc[i][col] = 1
    return signals


# ── Strategy: Rank Reversal (multiple variants) ────────────────────────────

def generate_reversal_signals(returns, ranks, sector_cols,
                               max_rank=1, min_mom_days=3, long_only=True):
    """
    Long when sector is at bottom rank AND shows positive short-term momentum.
    Optionally short when at top rank with negative momentum.
    """
    n_sectors = len(sector_cols)
    mom = returns[sector_cols].rolling(min_mom_days).sum()
    signals = pd.DataFrame(0, index=returns.index, columns=sector_cols)

    for i in range(max(LOOKBACK_RANK, min_mom_days) + 1, len(returns)):
        for col in sector_cols:
            rank_val = ranks.iloc[i].get(col, np.nan)
            mom_val = mom.iloc[i].get(col, np.nan)
            if pd.isna(rank_val) or pd.isna(mom_val):
                continue
            # Bottom rank + positive momentum → long
            if rank_val <= max_rank and mom_val > 0:
                signals.iloc[i][col] = 1
            # Top rank + negative momentum → short (if enabled)
            if not long_only and rank_val >= (n_sectors - max_rank + 1) and mom_val < 0:
                signals.iloc[i][col] = -1

    return signals


# ── Walk-Forward Backtest (portfolio-level P&L) ─────────────────────────────

def run_backtest(returns, signals, sector_cols, name, max_positions=3):
    """
    Walk-forward sliding-window backtest with portfolio-level tracking.
    - Max concurrent positions capped
    - Equal-weight allocation per position
    - Portfolio-level daily returns for proper drawdown
    """
    spy_daily = returns[BENCHMARK]
    warmup = SLIDING_WINDOW + LOOKBACK_RANK + MOMENTUM_CONFIRM + 5

    trades = []
    # Track active positions: list of (exit_day_idx, sector, direction)
    active_positions = []
    # Daily portfolio returns
    daily_port_ret = pd.Series(0.0, index=returns.index)

    for i in range(warmup, len(returns)):
        # Remove expired positions
        active_positions = [(ex, sec, d) for ex, sec, d in active_positions if ex > i]

        # Check for new signals
        new_entries = []
        for col in sector_cols:
            sig = signals.iloc[i][col]
            if sig == 0:
                continue
            # Don't double up on same sector
            if any(sec == col for _, sec, _ in active_positions):
                continue

            # Walk-forward filter: check trailing window
            ws = max(warmup, i - SLIDING_WINDOW)
            trail_sigs = signals.iloc[ws:i][col]
            trail_sig_days = trail_sigs[trail_sigs != 0]
            if len(trail_sig_days) >= 3:
                trail_rets = []
                for j_idx in trail_sig_days.index:
                    j_pos = returns.index.get_loc(j_idx)
                    if j_pos + HOLD_PERIOD < len(returns) and j_pos < i:
                        r = returns[col].iloc[j_pos+1:j_pos+1+HOLD_PERIOD].sum()
                        trail_rets.append(r * trail_sig_days[j_idx])
                if len(trail_rets) >= 3 and np.mean(trail_rets) < -0.005:
                    continue  # skip if trailing performance terrible

            new_entries.append((col, sig))

        # Cap new entries
        if len(active_positions) + len(new_entries) > max_positions:
            new_entries = new_entries[:max_positions - len(active_positions)]

        # Open new positions
        for col, sig in new_entries:
            exit_idx = min(i + HOLD_PERIOD, len(returns) - 1)
            active_positions.append((exit_idx, col, sig))

            # Record trade
            entry_idx = i + 1 if i + 1 < len(returns) else i
            actual_exit = min(i + HOLD_PERIOD, len(returns) - 1)
            hold_return = returns[col].iloc[entry_idx:actual_exit+1].sum()
            trade_return = hold_return * sig
            spy_hold = spy_daily.iloc[entry_idx:actual_exit+1].sum()
            regime = 'green' if spy_daily.iloc[i] > 0 else 'red'

            trades.append({
                'entry_date': returns.index[entry_idx] if entry_idx < len(returns) else returns.index[-1],
                'exit_date': returns.index[actual_exit],
                'sector': col,
                'direction': 'long' if sig > 0 else 'short',
                'signal': sig,
                'return': trade_return,
                'spy_return': spy_hold,
                'regime': regime,
            })

        # Compute daily portfolio return
        n_active = len(active_positions)
        if n_active > 0:
            weight = 1.0 / max_positions  # fixed allocation per slot
            day_ret = 0.0
            for _, sec, d in active_positions:
                if i < len(returns):
                    day_ret += weight * returns[sec].iloc[i] * d
            daily_port_ret.iloc[i] = day_ret

    trades_df = pd.DataFrame(trades)
    return trades_df, daily_port_ret


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(trades_df, daily_port_ret, label=""):
    if len(trades_df) == 0:
        return None

    rets = trades_df['return'].values
    n_trades = len(rets)
    avg_ret = np.mean(rets)
    win_rate = np.mean(rets > 0)

    # Use daily portfolio returns for Sharpe/Sortino/DD
    port = daily_port_ret[daily_port_ret != 0]  # days with exposure
    if len(port) < 10:
        port = daily_port_ret

    # Annualized Sharpe from daily portfolio returns
    if np.std(port) > 0:
        sharpe = (np.mean(port) / np.std(port)) * np.sqrt(252)
    else:
        sharpe = 0

    # Sortino
    downside = port[port < 0]
    if len(downside) > 0 and np.std(downside) > 0:
        sortino = (np.mean(port) / np.std(downside)) * np.sqrt(252)
    else:
        sortino = np.inf if avg_ret > 0 else 0

    # Profit Factor (from trades)
    gross_profit = np.sum(rets[rets > 0]) if np.any(rets > 0) else 0
    gross_loss = abs(np.sum(rets[rets < 0])) if np.any(rets < 0) else 0.0001
    profit_factor = gross_profit / gross_loss

    # Max Drawdown from cumulative portfolio equity
    cum = (1 + daily_port_ret).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    max_dd = dd.min()

    # Per-regime
    green_trades = trades_df[trades_df['regime'] == 'green']['return'].values
    red_trades = trades_df[trades_df['regime'] == 'red']['return'].values

    def regime_sharpe(r):
        if len(r) < 2 or np.std(r) == 0:
            return 0
        return (np.mean(r) / np.std(r)) * np.sqrt(52)

    sharpe_green = regime_sharpe(green_trades)
    sharpe_red = regime_sharpe(red_trades)
    max_rs = max(abs(sharpe_green), abs(sharpe_red), 0.0001)
    regime_gap = abs(sharpe_green - sharpe_red) / max_rs

    # Permutation test
    observed = np.mean(rets)
    n_perms = 2000
    perm_means = []
    for _ in range(n_perms):
        shuffled = rets * np.random.choice([-1, 1], size=len(rets))
        perm_means.append(np.mean(shuffled))
    perm_means = np.array(perm_means)
    perm_p = np.mean(perm_means >= observed) if observed > 0 else np.mean(perm_means <= observed)

    # Total return
    total_return = cum.iloc[-1] - 1

    return {
        'label': label,
        'n_trades': n_trades,
        'avg_return': avg_ret,
        'win_rate': win_rate,
        'sharpe': sharpe,
        'sortino': sortino,
        'profit_factor': profit_factor,
        'max_dd': max_dd,
        'total_return': total_return,
        'sharpe_green': sharpe_green,
        'sharpe_red': sharpe_red,
        'regime_gap': regime_gap,
        'perm_p': perm_p,
        'n_green': len(green_trades),
        'n_red': len(red_trades),
        'avg_ret_green': np.mean(green_trades) if len(green_trades) > 0 else 0,
        'avg_ret_red': np.mean(red_trades) if len(red_trades) > 0 else 0,
    }


def five_gate_validation(metrics):
    if metrics is None:
        return {'pass': False, 'reason': 'No trades'}
    gates = {}
    gates['sharpe'] = ('PASS' if metrics['sharpe'] > SHARPE_GATE else 'FAIL',
                       f"Sharpe {metrics['sharpe']:.3f} vs gate {SHARPE_GATE}")
    gates['regime_gap'] = ('PASS' if metrics['regime_gap'] < REGIME_GAP_GATE else 'FAIL',
                           f"Regime gap {metrics['regime_gap']:.3f} vs gate {REGIME_GAP_GATE}")
    gates['perm_p'] = ('PASS' if metrics['perm_p'] < PERM_P_GATE else 'FAIL',
                       f"Perm p-value {metrics['perm_p']:.4f} vs gate {PERM_P_GATE}")
    gates['max_dd'] = ('PASS' if metrics['max_dd'] > -0.30 else 'FAIL',
                       f"Max DD {metrics['max_dd']*100:.1f}% vs gate -30%")
    gates['n_trades'] = ('PASS' if metrics['n_trades'] >= MIN_TRADES else 'FAIL',
                         f"Trades {metrics['n_trades']} vs gate {MIN_TRADES}")
    all_pass = all(v[0] == 'PASS' for v in gates.values())
    return {'pass': all_pass, 'gates': gates}


def print_results(metrics, gates_result, strategy_name):
    if metrics is None:
        print(f"\n{'='*60}")
        print(f"  {strategy_name}: NO TRADES")
        print(f"{'='*60}")
        return

    print(f"\n{'='*70}")
    print(f"  {strategy_name}")
    print(f"{'='*70}")
    print(f"  Trades:         {metrics['n_trades']}")
    print(f"  Avg Trade Ret:  {metrics['avg_return']*100:.3f}%")
    print(f"  Win Rate:       {metrics['win_rate']*100:.1f}%")
    print(f"  Total Return:   {metrics['total_return']*100:.2f}%")
    print(f"  Sharpe:         {metrics['sharpe']:.3f}")
    print(f"  Sortino:        {metrics['sortino']:.3f}")
    print(f"  Profit Factor:  {metrics['profit_factor']:.3f}")
    print(f"  Max Drawdown:   {metrics['max_dd']*100:.2f}%")
    print(f"  ")
    print(f"  -- Regime Stratification --")
    print(f"  Green days:  N={metrics['n_green']}, Avg={metrics['avg_ret_green']*100:.3f}%, Sharpe={metrics['sharpe_green']:.3f}")
    print(f"  Red days:    N={metrics['n_red']}, Avg={metrics['avg_ret_red']*100:.3f}%, Sharpe={metrics['sharpe_red']:.3f}")
    print(f"  Regime Gap:  {metrics['regime_gap']:.3f}")
    print(f"  ")
    print(f"  -- Permutation Test --")
    print(f"  Perm p-value:   {metrics['perm_p']:.4f}")
    print(f"  ")
    print(f"  -- 5-Gate Validation --")
    if 'gates' in gates_result:
        for gn, (result, desc) in gates_result['gates'].items():
            icon = '+' if result == 'PASS' else 'X'
            print(f"  [{icon}] {gn}: {desc}")
    verdict = "PASS ALL GATES" if gates_result['pass'] else "FAILED"
    print(f"  VERDICT: {verdict}")
    print(f"{'='*70}")


def print_sector_breakdown(trades_df):
    if len(trades_df) == 0:
        return
    print(f"\n  -- Per-Sector Breakdown --")
    print(f"  {'Sector':<8} {'Trades':>6} {'WR':>6} {'AvgRet':>8} {'Sharpe':>7}")
    print(f"  {'-'*37}")
    for sector in sorted(trades_df['sector'].unique()):
        st = trades_df[trades_df['sector'] == sector]
        r = st['return'].values
        wr = np.mean(r > 0) * 100
        avg = np.mean(r) * 100
        sr = (np.mean(r) / np.std(r) * np.sqrt(52)) if np.std(r) > 0 else 0
        print(f"  {sector:<8} {len(r):>6} {wr:>5.1f}% {avg:>7.3f}% {sr:>7.3f}")


def print_yearly_breakdown(trades_df):
    if len(trades_df) == 0:
        return
    trades_df = trades_df.copy()
    trades_df['year'] = pd.to_datetime(trades_df['entry_date']).dt.year
    print(f"\n  -- Yearly Breakdown --")
    print(f"  {'Year':<6} {'Trades':>6} {'WR':>6} {'AvgRet':>8} {'TotalRet':>9}")
    print(f"  {'-'*37}")
    for year in sorted(trades_df['year'].unique()):
        yt = trades_df[trades_df['year'] == year]
        r = yt['return'].values
        print(f"  {year:<6} {len(r):>6} {np.mean(r>0)*100:>5.1f}% {np.mean(r)*100:>7.3f}% {np.sum(r)*100:>8.2f}%")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)

    print("=" * 70)
    print("  SECTOR RANK REVERSAL BACKTEST (v2 — Refined)")
    print("  Walk-Forward Sliding Window | 5-Gate Validation | Long-Only Focus")
    print("=" * 70)

    close = download_data()
    returns = compute_returns(close)
    sector_cols = [s for s in SECTOR_ETFS if s in returns.columns]
    print(f"\nUsing {len(sector_cols)} sectors: {sector_cols}")

    ranks = compute_sector_ranks(returns, sector_cols)

    all_results = []

    # ── Variant A: Strict Reversal (rank=1 only, long-only) ──
    print("\n" + "-" * 70)
    print("  Variant A: Strict Rank-1 Reversal (long-only, 3d momentum confirm)")
    sig_a = generate_reversal_signals(returns, ranks, sector_cols,
                                       max_rank=1, min_mom_days=3, long_only=True)
    n_sig = (sig_a != 0).sum().sum()
    print(f"  Signals: {n_sig}")
    trades_a, port_a = run_backtest(returns, sig_a, sector_cols, "Strict Reversal", max_positions=3)
    print(f"  Trades: {len(trades_a)}")
    m_a = compute_metrics(trades_a, port_a, "Strict Rank-1 Reversal")
    g_a = five_gate_validation(m_a)
    print_results(m_a, g_a, "Variant A: Strict Rank-1 Reversal (Long-Only)")
    if len(trades_a) > 0:
        print_sector_breakdown(trades_a)
        print_yearly_breakdown(trades_a)
    all_results.append(("A: Strict Rank-1 Reversal", m_a, g_a))

    # ── Variant B: Relaxed Reversal (rank<=2, long-only) ──
    print("\n" + "-" * 70)
    print("  Variant B: Relaxed Rank<=2 Reversal (long-only, 3d momentum)")
    sig_b = generate_reversal_signals(returns, ranks, sector_cols,
                                       max_rank=2, min_mom_days=3, long_only=True)
    n_sig = (sig_b != 0).sum().sum()
    print(f"  Signals: {n_sig}")
    trades_b, port_b = run_backtest(returns, sig_b, sector_cols, "Relaxed Reversal", max_positions=3)
    print(f"  Trades: {len(trades_b)}")
    m_b = compute_metrics(trades_b, port_b, "Relaxed Rank<=2 Reversal")
    g_b = five_gate_validation(m_b)
    print_results(m_b, g_b, "Variant B: Relaxed Rank<=2 Reversal (Long-Only)")
    if len(trades_b) > 0:
        print_sector_breakdown(trades_b)
        print_yearly_breakdown(trades_b)
    all_results.append(("B: Relaxed Rank<=2 Reversal", m_b, g_b))

    # ── Variant C: Rank-1 + 5d momentum confirm (slower confirm) ──
    print("\n" + "-" * 70)
    print("  Variant C: Rank-1 Reversal + 5d momentum confirm")
    sig_c = generate_reversal_signals(returns, ranks, sector_cols,
                                       max_rank=1, min_mom_days=5, long_only=True)
    n_sig = (sig_c != 0).sum().sum()
    print(f"  Signals: {n_sig}")
    trades_c, port_c = run_backtest(returns, sig_c, sector_cols, "Slow Confirm", max_positions=3)
    print(f"  Trades: {len(trades_c)}")
    m_c = compute_metrics(trades_c, port_c, "Rank-1 + 5d Momentum")
    g_c = five_gate_validation(m_c)
    print_results(m_c, g_c, "Variant C: Rank-1 + 5d Momentum Confirm (Long-Only)")
    if len(trades_c) > 0:
        print_sector_breakdown(trades_c)
        print_yearly_breakdown(trades_c)
    all_results.append(("C: Rank-1 + 5d Momentum", m_c, g_c))

    # ── Variant D: Earnings-Season Reversal (rank-1 during earnings + momentum) ──
    print("\n" + "-" * 70)
    print("  Variant D: Earnings-Season Rank-1 Reversal (earnings months only)")
    sig_d = generate_reversal_signals(returns, ranks, sector_cols,
                                       max_rank=1, min_mom_days=3, long_only=True)
    # Filter to earnings season only
    for i in range(len(returns)):
        if not is_earnings_season(returns.index[i]):
            sig_d.iloc[i] = 0
    n_sig = (sig_d != 0).sum().sum()
    print(f"  Signals: {n_sig}")
    trades_d, port_d = run_backtest(returns, sig_d, sector_cols, "Earnings Reversal", max_positions=3)
    print(f"  Trades: {len(trades_d)}")
    m_d = compute_metrics(trades_d, port_d, "Earnings-Season Reversal")
    g_d = five_gate_validation(m_d)
    print_results(m_d, g_d, "Variant D: Earnings-Season Rank-1 Reversal (Long-Only)")
    if len(trades_d) > 0:
        print_sector_breakdown(trades_d)
        print_yearly_breakdown(trades_d)
    all_results.append(("D: Earnings-Season Reversal", m_d, g_d))

    # ── Variant E: Earnings Outperformance (long sectors beating SPY) ──
    print("\n" + "-" * 70)
    print("  Variant E: Earnings Outperformance (long sectors beating SPY by >1%)")
    sig_e = generate_earnings_long_signals(returns, sector_cols)
    n_sig = (sig_e != 0).sum().sum()
    print(f"  Signals: {n_sig}")
    trades_e, port_e = run_backtest(returns, sig_e, sector_cols, "Earnings Beat", max_positions=3)
    print(f"  Trades: {len(trades_e)}")
    m_e = compute_metrics(trades_e, port_e, "Earnings Outperformance")
    g_e = five_gate_validation(m_e)
    print_results(m_e, g_e, "Variant E: Earnings Outperformance (Long-Only)")
    if len(trades_e) > 0:
        print_sector_breakdown(trades_e)
        print_yearly_breakdown(trades_e)
    all_results.append(("E: Earnings Outperformance", m_e, g_e))

    # ── Variant F: Long-short reversal (both sides, rank 1 & 11) ──
    print("\n" + "-" * 70)
    print("  Variant F: Long-Short Rank Reversal (rank 1 long, rank 11 short)")
    sig_f = generate_reversal_signals(returns, ranks, sector_cols,
                                       max_rank=1, min_mom_days=3, long_only=False)
    n_sig = (sig_f != 0).sum().sum()
    print(f"  Signals: {n_sig}")
    trades_f, port_f = run_backtest(returns, sig_f, sector_cols, "Long-Short", max_positions=4)
    print(f"  Trades: {len(trades_f)}")
    m_f = compute_metrics(trades_f, port_f, "Long-Short Reversal")
    g_f = five_gate_validation(m_f)
    print_results(m_f, g_f, "Variant F: Long-Short Rank Reversal")
    if len(trades_f) > 0:
        print_sector_breakdown(trades_f)
        print_yearly_breakdown(trades_f)
        # Direction breakdown
        for d in ['long', 'short']:
            dt = trades_f[trades_f['direction'] == d]
            if len(dt) > 0:
                r = dt['return'].values
                print(f"  {d.upper()}: N={len(r)}, WR={np.mean(r>0)*100:.1f}%, Avg={np.mean(r)*100:.3f}%")
    all_results.append(("F: Long-Short Reversal", m_f, g_f))

    # ── Final Summary ──
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY — ALL VARIANTS")
    print("=" * 70)
    hdr = f"  {'Variant':<32} {'N':>5} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} {'PF':>6} {'MaxDD':>7} {'RGap':>6} {'PermP':>7} {'Gate':>5}"
    print(hdr)
    print(f"  {'-'*len(hdr)}")

    for name, m, g in all_results:
        if m is None:
            print(f"  {name:<32} {'N/A':>5}")
            continue
        v = "YES" if g['pass'] else "NO"
        print(f"  {name:<32} {m['n_trades']:>5} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{m['win_rate']*100:>5.1f}% {m['profit_factor']:>6.3f} "
              f"{m['max_dd']*100:>6.2f}% {m['regime_gap']:>6.3f} {m['perm_p']:>7.4f} {v:>5}")

    print(f"\n  5-Gate: Sharpe>{SHARPE_GATE}, RegimeGap<{REGIME_GAP_GATE}, PermP<{PERM_P_GATE}, MaxDD>-30%, Trades>={MIN_TRADES}")

    viable = [(n, m, g) for n, m, g in all_results if m is not None and g['pass']]
    if viable:
        best = max(viable, key=lambda x: x[1]['sharpe'])
        print(f"\n  BEST PASSING STRATEGY: {best[0]} (Sharpe={best[1]['sharpe']:.3f})")
    else:
        valid = [(n, m, g) for n, m, g in all_results if m is not None and m['n_trades'] >= 10]
        if valid:
            best = max(valid, key=lambda x: x[1]['sharpe'])
            print(f"\n  BEST (no gate pass): {best[0]} (Sharpe={best[1]['sharpe']:.3f})")
            # Show which gates it failed
            if 'gates' in best[2]:
                fails = [gn for gn, (r, _) in best[2]['gates'].items() if r == 'FAIL']
                print(f"  Failed gates: {', '.join(fails)}")
        print(f"\n  CONCLUSION: No variant passed all 5 gates.")

    print()


if __name__ == '__main__':
    main()
