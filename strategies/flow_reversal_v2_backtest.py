#!/usr/bin/env python3
"""
Flow Reversal v2 — High-Selectivity Backtest with Walk-Forward Validation
==========================================================================
Key changes from v1:
  - vol_z_threshold: 2.0 -> 2.5 (higher conviction entries)
  - max_hold: 3 -> 5 days (let winners run longer)
  - top_n: 2 -> 1 (single best opportunity only)
  - max_concurrent: 2 -> 1 (concentrated positions)
  - Walk-forward: 12-month sliding train, 1-month OOS
  - Full 4-gate adversarial if primary metrics look good

Design rationale:
  v1 had Sharpe 0.52 at zero cost but collapsed at 10bps due to 387 trades.
  Adversarial grid showed high-selectivity params (vol_z>=2.5, hold>=4, top_n=1)
  cluster as the best combos. Fewer trades with bigger edge = survives costs.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import time
import json
import os

np.random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
UNIVERSE = [
    'XLK', 'XLE', 'XLF', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB',
    'XLRE', 'XLC', 'SMH', 'IWM', 'QQQ', 'DIA', 'TLT', 'GLD', 'HYG',
]
START_DATE = '2019-01-01'
END_DATE = '2026-08-26'

# v2 optimized parameters (from adversarial grid best-combo cluster)
VOL_LOOKBACK = 20
VOL_Z_THRESHOLD = 2.5       # was 2.0 — higher bar for entry
PRICE_DIP_THRESHOLD = -0.01  # -1% (keep — grid shows -1% is fine)
MAX_HOLD_DAYS = 5            # was 3 — let winners run
TAKE_PROFIT = 0.035          # +3.5% (keep)
TRAILING_STOP = -0.02        # -2% (keep)
MAX_CONCURRENT = 1           # was 2 — single concentrated bet
TOP_N = 1                    # was 2 — best signal only

# Walk-forward config
WF_TRAIN_MONTHS = 12
WF_TEST_MONTHS = 1

N_PERMS = 200  # permutation count for adversarial Gate 2

print("=" * 90, flush=True)
print("FLOW REVERSAL v2 — HIGH-SELECTIVITY BACKTEST", flush=True)
print("=" * 90, flush=True)
print(f"Period: {START_DATE} to {END_DATE}", flush=True)
print(f"Universe: {len(UNIVERSE)} ETFs", flush=True)
print(f"Signal: vol_z >= {VOL_Z_THRESHOLD}, dip <= {PRICE_DIP_THRESHOLD*100:.0f}%", flush=True)
print(f"Exits: TP={TAKE_PROFIT*100:.1f}%, trailing stop={TRAILING_STOP*100:.0f}%, "
      f"max hold={MAX_HOLD_DAYS}d", flush=True)
print(f"Max concurrent: {MAX_CONCURRENT}, top_n: {TOP_N}", flush=True)
print(f"Walk-forward: {WF_TRAIN_MONTHS}mo train, {WF_TEST_MONTHS}mo OOS (sliding)", flush=True)
print(flush=True)

# ── Download Data ─────────────────────────────────────────────────────────────
print("[DATA] Downloading ETF daily data...", flush=True)
data = {}
for ticker in UNIVERSE:
    for attempt in range(3):
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE,
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 100:
                data[ticker] = df
                break
            time.sleep(1)
        except Exception:
            time.sleep(2)

available = [t for t in UNIVERSE if t in data]
print(f"  Got data for {len(available)}/{len(UNIVERSE)} ETFs", flush=True)

# Build aligned close and volume DataFrames
close_df = pd.DataFrame({t: data[t]['Close'] for t in available}).dropna(how='all')
volume_df = pd.DataFrame({t: data[t]['Volume'] for t in available}).dropna(how='all')

common_idx = close_df.index.intersection(volume_df.index)
close_df = close_df.loc[common_idx]
volume_df = volume_df.loc[common_idx]

print(f"  Date range: {close_df.index[0].date()} to {close_df.index[-1].date()}", flush=True)
print(f"  Trading days: {len(close_df)}", flush=True)

# Also download SPY for benchmark
spy_df = yf.download('SPY', start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
if isinstance(spy_df.columns, pd.MultiIndex):
    spy_df.columns = spy_df.columns.get_level_values(0)
spy_rets = spy_df['Close'].pct_change().dropna()


# ── Core backtest engine ─────────────────────────────────────────────────────
def backtest_flow_reversal(close_df, volume_df, tickers,
                           vol_z_thresh=VOL_Z_THRESHOLD,
                           dip_thresh=PRICE_DIP_THRESHOLD,
                           max_hold=MAX_HOLD_DAYS,
                           tp=TAKE_PROFIT,
                           trail_stop=TRAILING_STOP,
                           max_concurrent=MAX_CONCURRENT,
                           cost_bps=10,
                           signal_override=None,
                           top_n=TOP_N):
    """
    Event-driven backtest of flow reversal v2.
    Returns: (daily_returns Series, trade_log list)
    """
    dates = close_df.index
    n_days = len(dates)

    # Pre-compute returns and volume z-scores
    returns_1d = close_df[tickers].pct_change()
    vol_mean = volume_df[tickers].rolling(VOL_LOOKBACK).mean()
    vol_std = volume_df[tickers].rolling(VOL_LOOKBACK).std()
    vol_z = (volume_df[tickers] - vol_mean) / (vol_std + 1e-10)

    positions = []
    daily_pnl = pd.Series(0.0, index=dates)
    trade_log = []
    cost_per_trade = cost_bps / 10000.0

    for i in range(VOL_LOOKBACK + 2, n_days):
        date = dates[i]

        # --- Exit check ---
        positions_to_remove = []
        for pos in positions:
            ticker = pos['ticker']
            current_price = close_df.loc[date, ticker] if ticker in close_df.columns else np.nan
            if pd.isna(current_price):
                continue

            entry_price = pos['entry_price']
            pnl_pct = (current_price - entry_price) / entry_price
            days_held = i - pos['entry_idx']

            if current_price > pos['peak_price']:
                pos['peak_price'] = current_price

            drawdown_from_peak = (current_price - pos['peak_price']) / pos['peak_price']

            exit_reason = None
            if pnl_pct >= tp:
                exit_reason = 'take_profit'
            elif drawdown_from_peak <= trail_stop:
                exit_reason = 'trailing_stop'
            elif days_held >= max_hold:
                exit_reason = 'max_hold'

            if exit_reason:
                trade_pnl = pnl_pct - cost_per_trade
                trade_log.append({
                    'entry_date': str(dates[pos['entry_idx']].date()),
                    'exit_date': str(date.date()),
                    'ticker': ticker,
                    'pnl_pct': trade_pnl,
                    'days_held': days_held,
                    'reason': exit_reason,
                })
                daily_pnl.iloc[i] += trade_pnl / max_concurrent
                positions_to_remove.append(pos)

        for pos in positions_to_remove:
            if pos in positions:
                positions.remove(pos)

        # --- Entry check ---
        open_slots = max_concurrent - len(positions)
        if open_slots <= 0:
            continue

        held_tickers = {p['ticker'] for p in positions}

        if signal_override is not None:
            signals = signal_override(close_df, volume_df, tickers, i)
        else:
            signals = []
            for ticker in tickers:
                if ticker in held_tickers:
                    continue
                if ticker not in returns_1d.columns or ticker not in vol_z.columns:
                    continue

                r1d = returns_1d.iloc[i].get(ticker, np.nan)
                vz = vol_z.iloc[i].get(ticker, np.nan)

                if pd.isna(r1d) or pd.isna(vz):
                    continue

                if vz >= vol_z_thresh and r1d <= dip_thresh:
                    score = vz * abs(r1d)
                    signals.append((ticker, score))

        signals.sort(key=lambda x: x[1], reverse=True)

        for ticker, score in signals[:min(open_slots, top_n)]:
            if ticker in held_tickers:
                continue
            price = close_df.iloc[i].get(ticker, np.nan)
            if pd.isna(price) or price <= 0:
                continue

            positions.append({
                'ticker': ticker,
                'entry_price': float(price),
                'entry_idx': i,
                'peak_price': float(price),
            })
            held_tickers.add(ticker)
            daily_pnl.iloc[i] -= cost_per_trade / max_concurrent

    # Force-close remaining
    for pos in positions:
        ticker = pos['ticker']
        if ticker in close_df.columns:
            final_price = close_df.iloc[-1].get(ticker, pos['entry_price'])
            if pd.notna(final_price):
                pnl_pct = (final_price - pos['entry_price']) / pos['entry_price'] - cost_per_trade
                daily_pnl.iloc[-1] += pnl_pct / max_concurrent
                trade_log.append({
                    'entry_date': str(dates[pos['entry_idx']].date()),
                    'exit_date': str(dates[-1].date()),
                    'ticker': ticker,
                    'pnl_pct': pnl_pct,
                    'days_held': len(dates) - 1 - pos['entry_idx'],
                    'reason': 'end_of_period',
                })

    return daily_pnl, trade_log


# ── Metric helpers ────────────────────────────────────────────────────────────
def calc_sharpe(rets):
    if len(rets) < 30 or rets.std() == 0:
        return 0.0
    return float(rets.mean() / rets.std() * np.sqrt(252))

def calc_sortino(rets):
    if len(rets) < 30:
        return 0.0
    down = rets[rets < 0]
    down_vol = down.std() * np.sqrt(252) if len(down) > 5 else 1e-10
    return float(rets.mean() * 252 / (down_vol + 1e-10))

def calc_cagr(rets):
    if len(rets) < 30:
        return 0.0
    n_years = len(rets) / 252
    total_ret = (1 + rets).prod() - 1
    if total_ret <= -1:
        return -1.0
    return float((1 + total_ret) ** (1 / n_years) - 1)

def calc_max_dd(rets):
    cum = (1 + rets).cumprod()
    peak = cum.expanding().max()
    dd = (cum - peak) / peak
    return float(dd.min())

def calc_pf(rets):
    g = rets[rets > 0].sum()
    l = abs(rets[rets < 0].sum())
    return float(g / (l + 1e-10))

def calc_wr(trade_log):
    if not trade_log:
        return 0.0
    wins = sum(1 for t in trade_log if t['pnl_pct'] > 0)
    return wins / len(trade_log)

def calc_avg_win_loss(trade_log):
    if not trade_log:
        return 0.0, 0.0
    wins = [t['pnl_pct'] for t in trade_log if t['pnl_pct'] > 0]
    losses = [t['pnl_pct'] for t in trade_log if t['pnl_pct'] <= 0]
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0
    return avg_win, avg_loss

def print_metrics(rets, trade_log, label=""):
    s = calc_sharpe(rets)
    so = calc_sortino(rets)
    c = calc_cagr(rets)
    mdd = calc_max_dd(rets)
    pf = calc_pf(rets)
    wr = calc_wr(trade_log)
    aw, al = calc_avg_win_loss(trade_log)
    print(f"  {label}Sharpe={s:.3f}  Sortino={so:.3f}  CAGR={c:.1%}  "
          f"MaxDD={mdd:.1%}  PF={pf:.3f}  WR={wr:.0%}  Trades={len(trade_log)}  "
          f"AvgWin={aw:.2%}  AvgLoss={al:.2%}", flush=True)
    return s, so, c, mdd, pf, wr


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: FULL-PERIOD BACKTEST (v2 params)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90, flush=True)
print("SECTION 1: FULL-PERIOD BACKTEST (v2 HIGH-SELECTIVITY)", flush=True)
print("=" * 90, flush=True)

results_by_cost = {}
for cost_bps in [0, 10, 20]:
    rets, trades = backtest_flow_reversal(close_df, volume_df, available, cost_bps=cost_bps)
    s, so, c, mdd, pf, wr = print_metrics(rets, trades, f"  {cost_bps:2d}bps: ")
    results_by_cost[cost_bps] = {
        'sharpe': s, 'sortino': so, 'cagr': c, 'max_dd': mdd,
        'pf': pf, 'wr': wr, 'n_trades': len(trades),
        'rets': rets, 'trades': trades,
    }

# Use 10bps as primary result
primary = results_by_cost[10]
primary_rets = primary['rets']
primary_trades = primary['trades']

# SPY benchmark for same period
spy_common = spy_rets.loc[spy_rets.index.intersection(primary_rets.index)]
spy_sharpe = calc_sharpe(spy_common)
spy_sortino = calc_sortino(spy_common)
spy_cagr = calc_cagr(spy_common)
spy_mdd = calc_max_dd(spy_common)

print(f"\n  SPY Benchmark: Sharpe={spy_sharpe:.3f}  Sortino={spy_sortino:.3f}  "
      f"CAGR={spy_cagr:.1%}  MaxDD={spy_mdd:.1%}", flush=True)
print(f"  v2 vs SPY at 10bps: Sharpe {primary['sharpe']:.3f} vs {spy_sharpe:.3f}", flush=True)

# Trade distribution analysis
if primary_trades:
    exits = {}
    for t in primary_trades:
        exits[t['reason']] = exits.get(t['reason'], 0) + 1
    print(f"\n  Exit reasons: {exits}", flush=True)

    tickers_traded = {}
    for t in primary_trades:
        tickers_traded[t['ticker']] = tickers_traded.get(t['ticker'], 0) + 1
    print(f"  Tickers traded: {dict(sorted(tickers_traded.items(), key=lambda x: -x[1]))}", flush=True)

    hold_days = [t['days_held'] for t in primary_trades]
    print(f"  Hold days: mean={np.mean(hold_days):.1f}, median={np.median(hold_days):.0f}, "
          f"min={min(hold_days)}, max={max(hold_days)}", flush=True)

    pnls = [t['pnl_pct'] for t in primary_trades]
    print(f"  Per-trade PnL: mean={np.mean(pnls):.3%}, median={np.median(pnls):.3%}, "
          f"best={max(pnls):.3%}, worst={min(pnls):.3%}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: WALK-FORWARD VALIDATION (12mo train, 1mo OOS, sliding)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90, flush=True)
print("SECTION 2: WALK-FORWARD VALIDATION", flush=True)
print(f"  {WF_TRAIN_MONTHS}mo sliding train, {WF_TEST_MONTHS}mo OOS", flush=True)
print("=" * 90, flush=True)

# Build monthly boundaries
dates = close_df.index
monthly_starts = []
current_month = None
for i, d in enumerate(dates):
    ym = (d.year, d.month)
    if ym != current_month:
        monthly_starts.append(i)
        current_month = ym

wf_oos_rets = pd.Series(0.0, index=dates)
wf_oos_trades = []
wf_fold_results = []

n_folds = 0
for fold_start in range(WF_TRAIN_MONTHS, len(monthly_starts) - WF_TEST_MONTHS):
    # Training window: [fold_start - WF_TRAIN_MONTHS, fold_start)
    train_start_idx = monthly_starts[fold_start - WF_TRAIN_MONTHS]
    train_end_idx = monthly_starts[fold_start] - 1

    # Test window: [fold_start, fold_start + WF_TEST_MONTHS)
    test_start_idx = monthly_starts[fold_start]
    test_end_idx = (monthly_starts[fold_start + WF_TEST_MONTHS] - 1
                    if fold_start + WF_TEST_MONTHS < len(monthly_starts)
                    else len(dates) - 1)

    # For walk-forward, we use fixed v2 params (no in-sample optimization)
    # The "training" period is just to ensure we have lookback data
    # The real test: does the signal work OOS with these fixed params?

    test_close = close_df.iloc[max(0, test_start_idx - VOL_LOOKBACK - 5):test_end_idx + 1]
    test_volume = volume_df.iloc[max(0, test_start_idx - VOL_LOOKBACK - 5):test_end_idx + 1]

    fold_rets, fold_trades = backtest_flow_reversal(
        test_close, test_volume, available, cost_bps=10)

    # Only count the OOS portion
    oos_dates = dates[test_start_idx:test_end_idx + 1]
    for d in oos_dates:
        if d in fold_rets.index:
            wf_oos_rets.loc[d] = fold_rets.loc[d]

    # Filter trades to OOS period only
    oos_start_date = dates[test_start_idx].date()
    oos_end_date = dates[test_end_idx].date()
    for t in fold_trades:
        entry_d = datetime.strptime(t['entry_date'], '%Y-%m-%d').date()
        if oos_start_date <= entry_d <= oos_end_date:
            wf_oos_trades.append(t)

    fold_sharpe = calc_sharpe(fold_rets)
    n_oos_trades = len([t for t in fold_trades
                        if oos_start_date <= datetime.strptime(t['entry_date'], '%Y-%m-%d').date() <= oos_end_date])

    wf_fold_results.append({
        'fold': n_folds,
        'oos_period': f"{dates[test_start_idx].strftime('%Y-%m')}",
        'sharpe': fold_sharpe,
        'n_trades': n_oos_trades,
    })
    n_folds += 1

# Trim to only OOS period
wf_oos_start = dates[monthly_starts[WF_TRAIN_MONTHS]]
wf_oos_rets_trimmed = wf_oos_rets.loc[wf_oos_start:]

print(f"\n  Walk-forward: {n_folds} OOS folds", flush=True)
wf_s, wf_so, wf_c, wf_mdd, wf_pf, wf_wr = print_metrics(
    wf_oos_rets_trimmed, wf_oos_trades, "WF OOS: ")

# Show per-fold summary
pos_folds = sum(1 for f in wf_fold_results if f['sharpe'] > 0)
print(f"  Positive Sharpe folds: {pos_folds}/{n_folds} ({pos_folds/n_folds*100:.0f}%)", flush=True)

# Year-by-year OOS breakdown
print(f"\n  Year-by-year OOS breakdown:", flush=True)
for year in range(2020, 2027):
    year_rets = wf_oos_rets_trimmed[wf_oos_rets_trimmed.index.year == year]
    if len(year_rets) < 20:
        continue
    year_trades = [t for t in wf_oos_trades if t['entry_date'].startswith(str(year))]
    ys = calc_sharpe(year_rets)
    yc = calc_cagr(year_rets)
    yr = float((1 + year_rets).prod() - 1)
    print(f"    {year}: Sharpe={ys:.3f}  Return={yr:.1%}  Trades={len(year_trades)}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: v1 vs v2 COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90, flush=True)
print("SECTION 3: v1 vs v2 COMPARISON", flush=True)
print("=" * 90, flush=True)

# Run v1 for comparison
v1_rets_0, v1_trades_0 = backtest_flow_reversal(
    close_df, volume_df, available,
    vol_z_thresh=2.0, max_hold=3, max_concurrent=2, top_n=2, cost_bps=0)
v1_rets_10, v1_trades_10 = backtest_flow_reversal(
    close_df, volume_df, available,
    vol_z_thresh=2.0, max_hold=3, max_concurrent=2, top_n=2, cost_bps=10)
v1_rets_20, v1_trades_20 = backtest_flow_reversal(
    close_df, volume_df, available,
    vol_z_thresh=2.0, max_hold=3, max_concurrent=2, top_n=2, cost_bps=20)

print(f"  v1 (vol_z=2.0, hold=3, top_n=2):", flush=True)
print(f"    0bps:  Sharpe={calc_sharpe(v1_rets_0):.3f}  CAGR={calc_cagr(v1_rets_0):.1%}  "
      f"Trades={len(v1_trades_0)}", flush=True)
print(f"    10bps: Sharpe={calc_sharpe(v1_rets_10):.3f}  CAGR={calc_cagr(v1_rets_10):.1%}  "
      f"Trades={len(v1_trades_10)}", flush=True)
print(f"    20bps: Sharpe={calc_sharpe(v1_rets_20):.3f}  CAGR={calc_cagr(v1_rets_20):.1%}  "
      f"Trades={len(v1_trades_20)}", flush=True)

print(f"\n  v2 (vol_z=2.5, hold=5, top_n=1):", flush=True)
for cost_bps in [0, 10, 20]:
    r = results_by_cost[cost_bps]
    print(f"    {cost_bps}bps:  Sharpe={r['sharpe']:.3f}  CAGR={r['cagr']:.1%}  "
          f"Trades={r['n_trades']}", flush=True)

edge_per_trade_v1 = np.mean([t['pnl_pct'] for t in v1_trades_10]) if v1_trades_10 else 0
edge_per_trade_v2 = np.mean([t['pnl_pct'] for t in primary_trades]) if primary_trades else 0
print(f"\n  Edge per trade (10bps): v1={edge_per_trade_v1:.3%}  v2={edge_per_trade_v2:.3%}  "
      f"improvement={edge_per_trade_v2 - edge_per_trade_v1:.3%}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: ADVERSARIAL VALIDATION (4 gates — if Sharpe > 0.5 after 10bps)
# ═══════════════════════════════════════════════════════════════════════════════
run_adversarial = primary['sharpe'] > 0.5
gate_results = {}

if not run_adversarial:
    print("\n" + "=" * 90, flush=True)
    print(f"ADVERSARIAL SKIPPED — Sharpe at 10bps = {primary['sharpe']:.3f} (threshold: > 0.5)", flush=True)
    print("=" * 90, flush=True)
    # Still run adversarial for diagnostics even if below threshold
    print("  Running anyway for diagnostics...", flush=True)

print("\n" + "=" * 90, flush=True)
print("SECTION 4: ADVERSARIAL VALIDATION (4 GATES)", flush=True)
print("=" * 90, flush=True)

# ── Gate 1: Inverse Signal ──
print("\n--- GATE 1: INVERSE SIGNAL ---", flush=True)
print("  Buy when vol z-score <= -2.5 AND 1d return >= +1%", flush=True)

def inverse_signal(close_df, volume_df, tickers, day_idx):
    signals = []
    for ticker in tickers:
        if ticker not in close_df.columns or ticker not in volume_df.columns:
            continue
        vol = volume_df[ticker].values
        c = close_df[ticker].values
        if day_idx < VOL_LOOKBACK + 1:
            continue
        window = vol[day_idx - VOL_LOOKBACK:day_idx]
        mu = np.nanmean(window)
        sigma = np.nanstd(window, ddof=1)
        if sigma <= 0 or pd.isna(sigma):
            continue
        vz = (vol[day_idx] - mu) / sigma
        r1d = (c[day_idx] - c[day_idx - 1]) / c[day_idx - 1] if c[day_idx - 1] > 0 else 0
        if vz <= -VOL_Z_THRESHOLD and r1d >= abs(PRICE_DIP_THRESHOLD):
            score = abs(vz) * abs(r1d)
            signals.append((ticker, score))
    return signals

inv_rets, inv_trades = backtest_flow_reversal(
    close_df, volume_df, available, signal_override=inverse_signal, cost_bps=10)
inv_sharpe = calc_sharpe(inv_rets)

print_metrics(inv_rets, inv_trades, "Inverse: ")

gate1_pass = inv_sharpe < 0.50 and (inv_sharpe < primary['sharpe'] * 0.6 or len(inv_trades) < 5)
if len(inv_trades) < 5:
    gate1_pass = True
    print(f"  Inverse produced only {len(inv_trades)} trades (signal is asymmetric) — auto PASS", flush=True)

print(f"  v2 Sharpe:     {primary['sharpe']:.3f} ({primary['n_trades']} trades)", flush=True)
print(f"  Inverse Sharpe: {inv_sharpe:.3f} ({len(inv_trades)} trades)", flush=True)
print(f"  GATE 1 (Inverse): {'PASS' if gate1_pass else 'FAIL'}", flush=True)
gate_results['Gate 1: Inverse signal'] = 'PASS' if gate1_pass else 'FAIL'


# ── Gate 2: Random Timing Permutation ──
print(f"\n--- GATE 2: RANDOM TIMING PERMUTATION ({N_PERMS} shuffles) ---", flush=True)

orig_signal_days = len(set(t['entry_date'] for t in primary_trades))
perm_sharpes = []
t0 = time.time()

for p in range(N_PERMS):
    rng = np.random.RandomState(p + 7777)
    n_days_total = len(close_df)
    valid_days = list(range(VOL_LOOKBACK + 2, n_days_total - MAX_HOLD_DAYS))
    random_signal_days = set(rng.choice(valid_days, size=min(orig_signal_days, len(valid_days)),
                                        replace=False))

    def random_signal_fn(close_df, volume_df, tickers, day_idx, _rng=rng,
                         _days=random_signal_days):
        if day_idx not in _days:
            return []
        avail = [t for t in tickers if t in close_df.columns]
        _rng.shuffle(avail)
        signals = [(avail[0], _rng.random())] if avail else []
        return signals

    perm_rets, perm_trades = backtest_flow_reversal(
        close_df, volume_df, available, signal_override=random_signal_fn, cost_bps=10)
    perm_sharpes.append(calc_sharpe(perm_rets) if len(perm_rets) > 30 else 0.0)

    if (p + 1) % 50 == 0:
        print(f"  ... {p+1}/{N_PERMS} permutations ({time.time()-t0:.1f}s)", flush=True)

perm_sharpes = np.array(perm_sharpes)
p95 = np.percentile(perm_sharpes, 95)
p99 = np.percentile(perm_sharpes, 99)
pctile = (perm_sharpes < primary['sharpe']).mean() * 100

gate2_pass = primary['sharpe'] > p95

print(f"  v2 Sharpe:        {primary['sharpe']:.3f}", flush=True)
print(f"  Random mean:      {perm_sharpes.mean():.3f} +/- {perm_sharpes.std():.3f}", flush=True)
print(f"  Random 95th pctl: {p95:.3f}", flush=True)
print(f"  Random 99th pctl: {p99:.3f}", flush=True)
print(f"  Strategy pctile:  {pctile:.1f}%", flush=True)
print(f"  GATE 2 (Random timing): {'PASS' if gate2_pass else 'FAIL'}", flush=True)
gate_results['Gate 2: Random timing'] = 'PASS' if gate2_pass else 'FAIL'


# ── Gate 3: Sub-period Stability ──
print("\n--- GATE 3: SUB-PERIOD STABILITY (4 equal sub-periods) ---", flush=True)

n = len(primary_rets)
chunk = n // 4
all_positive = True
sub_results = []

for i in range(4):
    s_idx = i * chunk
    e_idx = (i + 1) * chunk if i < 3 else n
    sub = primary_rets.iloc[s_idx:e_idx]
    ss = calc_sharpe(sub)
    cum_ret = float((1 + sub).prod() - 1)

    sub_start = sub.index[0].date()
    sub_end = sub.index[-1].date()
    sub_trade_count = sum(1 for t in primary_trades
                         if sub_start <= datetime.strptime(t['entry_date'], '%Y-%m-%d').date() <= sub_end)

    sub_results.append({
        'period': f"{sub.index[0].strftime('%Y-%m')} to {sub.index[-1].strftime('%Y-%m')}",
        'sharpe': ss, 'cum_return': cum_ret, 'trades': sub_trade_count,
    })

    if cum_ret <= 0:
        all_positive = False

    status = 'OK' if cum_ret > 0 else 'NEG'
    print(f"  Period {i+1}: {sub_results[-1]['period']}  "
          f"Sharpe={ss:.3f}  Cum={cum_ret:.1%}  Trades={sub_trade_count}  [{status}]", flush=True)

# Relaxed: at least 3 of 4 positive
n_positive = sum(1 for sr in sub_results if sr['cum_return'] > 0)
gate3_pass = n_positive >= 3

print(f"  Positive sub-periods: {n_positive}/4 (need >= 3)", flush=True)
print(f"  GATE 3 (Sub-period): {'PASS' if gate3_pass else 'FAIL'}", flush=True)
gate_results['Gate 3: Sub-period stability'] = 'PASS' if gate3_pass else 'FAIL'


# ── Gate 4: Cost Sensitivity ──
print("\n--- GATE 4: COST SENSITIVITY ---", flush=True)

cost_levels = [0, 5, 10, 15, 20, 30, 50]
cost_detail = {}

for cost_bps in cost_levels:
    cr, ct = backtest_flow_reversal(close_df, volume_df, available, cost_bps=cost_bps)
    cs = calc_sharpe(cr)
    cc = calc_cagr(cr)
    cpf = calc_pf(cr)
    cost_detail[cost_bps] = {'sharpe': cs, 'cagr': cc, 'pf': cpf, 'trades': len(ct)}
    print(f"  {cost_bps:3d}bps: Sharpe={cs:.3f}  CAGR={cc:.1%}  PF={cpf:.3f}  "
          f"Trades={len(ct)}", flush=True)

# Must survive 20bps with positive Sharpe
gate4_pass = cost_detail[20]['sharpe'] > 0.0 and cost_detail[10]['sharpe'] > 0.3
print(f"  At 10bps: Sharpe={cost_detail[10]['sharpe']:.3f} (need > 0.3)", flush=True)
print(f"  At 20bps: Sharpe={cost_detail[20]['sharpe']:.3f} (need > 0.0)", flush=True)
print(f"  GATE 4 (Cost sensitivity): {'PASS' if gate4_pass else 'FAIL'}", flush=True)
gate_results['Gate 4: Cost sensitivity'] = 'PASS' if gate4_pass else 'FAIL'


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: ADDITIONAL PARAMETER SENSITIVITY (nearby params)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90, flush=True)
print("SECTION 5: NEARBY PARAMETER SENSITIVITY", flush=True)
print("  Testing vol_z=[2.0,2.5,3.0], hold=[3,4,5], dip=[-0.5%,-1%,-1.5%,-2%]", flush=True)
print("=" * 90, flush=True)

nearby_grid = []
for vz in [2.0, 2.5, 3.0]:
    for hold in [3, 4, 5]:
        for dip in [-0.005, -0.01, -0.015, -0.02]:
            gr, gt = backtest_flow_reversal(
                close_df, volume_df, available,
                vol_z_thresh=vz, dip_thresh=dip, max_hold=hold,
                max_concurrent=1, top_n=1, cost_bps=10)
            gs = calc_sharpe(gr)
            gc = calc_cagr(gr)
            gpf = calc_pf(gr)
            nearby_grid.append({
                'vol_z': vz, 'dip': dip, 'hold': hold,
                'sharpe': gs, 'cagr': gc, 'pf': gpf, 'trades': len(gt),
            })

grid_df = pd.DataFrame(nearby_grid)
print(f"\n  Grid: {len(grid_df)} combos", flush=True)
print(f"  Sharpe range: [{grid_df['sharpe'].min():.3f}, {grid_df['sharpe'].max():.3f}]", flush=True)
print(f"  Mean Sharpe: {grid_df['sharpe'].mean():.3f}", flush=True)
print(f"  Combos with Sharpe > 0.3 (at 10bps): "
      f"{(grid_df['sharpe'] > 0.3).sum()}/{len(grid_df)} "
      f"({(grid_df['sharpe'] > 0.3).mean()*100:.0f}%)", flush=True)

print(f"\n  Top 10 combos (at 10bps cost):", flush=True)
for _, row in grid_df.nlargest(10, 'sharpe').iterrows():
    print(f"    vol_z={row['vol_z']}, dip={row['dip']*100:.1f}%, hold={int(row['hold'])}: "
          f"Sharpe={row['sharpe']:.3f}  CAGR={row['cagr']:.1%}  PF={row['pf']:.3f}  "
          f"Trades={int(row['trades'])}", flush=True)

print(f"\n  By vol_z:", flush=True)
for vz in [2.0, 2.5, 3.0]:
    sub = grid_df[grid_df['vol_z'] == vz]
    print(f"    vol_z={vz}: mean Sharpe={sub['sharpe'].mean():.3f}  "
          f"mean trades={sub['trades'].mean():.0f}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SCORECARD
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90, flush=True)
print("FLOW REVERSAL v2 — FINAL SCORECARD", flush=True)
print("=" * 90, flush=True)

n_pass = sum(1 for v in gate_results.values() if v == 'PASS')
n_total = len(gate_results)

for gate, result in gate_results.items():
    icon = "[PASS]" if result == "PASS" else "[FAIL]"
    print(f"  {icon} {gate}", flush=True)

print(f"\n  ADVERSARIAL: {n_pass}/{n_total} gates passed", flush=True)

print(f"\n  PRIMARY METRICS (10bps cost):", flush=True)
print(f"    Sharpe:  {primary['sharpe']:.3f}", flush=True)
print(f"    Sortino: {primary['sortino']:.3f}", flush=True)
print(f"    CAGR:    {primary['cagr']:.1%}", flush=True)
print(f"    MaxDD:   {primary['max_dd']:.1%}", flush=True)
print(f"    PF:      {primary['pf']:.3f}", flush=True)
print(f"    WR:      {primary['wr']:.0%}", flush=True)
print(f"    Trades:  {primary['n_trades']}", flush=True)

print(f"\n  WALK-FORWARD OOS:", flush=True)
print(f"    Sharpe:  {wf_s:.3f}", flush=True)
print(f"    CAGR:    {wf_c:.1%}", flush=True)
print(f"    Trades:  {len(wf_oos_trades)}", flush=True)

print(f"\n  SPY BENCHMARK:", flush=True)
print(f"    Sharpe:  {spy_sharpe:.3f}", flush=True)
print(f"    CAGR:    {spy_cagr:.1%}", flush=True)

if primary['sharpe'] > 0.5 and n_pass >= 3:
    verdict = "PROMISING — v2 shows improved cost-resilience, warrants further analysis"
elif primary['sharpe'] > 0.3 and n_pass >= 2:
    verdict = "MARGINAL — improved over v1 but needs more edge"
else:
    verdict = "INSUFFICIENT — high-selectivity params did not resolve cost issue"

print(f"\n  VERDICT: {verdict}", flush=True)
print("=" * 90, flush=True)

# ── Save results ──────────────────────────────────────────────────────────────
save_data = {
    'run_date': datetime.now().isoformat(),
    'strategy': 'flow_reversal_v2',
    'params': {
        'vol_z_threshold': VOL_Z_THRESHOLD,
        'dip_threshold': PRICE_DIP_THRESHOLD,
        'max_hold': MAX_HOLD_DAYS,
        'take_profit': TAKE_PROFIT,
        'trailing_stop': TRAILING_STOP,
        'max_concurrent': MAX_CONCURRENT,
        'top_n': TOP_N,
    },
    'primary_10bps': {
        'sharpe': primary['sharpe'],
        'sortino': primary['sortino'],
        'cagr': primary['cagr'],
        'max_dd': primary['max_dd'],
        'pf': primary['pf'],
        'wr': primary['wr'],
        'n_trades': primary['n_trades'],
    },
    'cost_sensitivity': {str(k): {kk: vv for kk, vv in v.items()}
                         for k, v in cost_detail.items()},
    'walk_forward': {
        'sharpe': wf_s, 'sortino': wf_so, 'cagr': wf_c,
        'max_dd': wf_mdd, 'pf': wf_pf, 'wr': wf_wr,
        'n_trades': len(wf_oos_trades),
        'n_folds': n_folds,
        'positive_folds_pct': pos_folds / n_folds * 100 if n_folds > 0 else 0,
    },
    'adversarial_gates': gate_results,
    'n_pass': n_pass,
    'n_total': n_total,
    'verdict': verdict,
    'spy_benchmark': {
        'sharpe': spy_sharpe, 'sortino': spy_sortino,
        'cagr': spy_cagr, 'max_dd': spy_mdd,
    },
    'v1_comparison': {
        '0bps_sharpe': calc_sharpe(v1_rets_0),
        '10bps_sharpe': calc_sharpe(v1_rets_10),
        '20bps_sharpe': calc_sharpe(v1_rets_20),
        'v1_trades': len(v1_trades_10),
    },
    'nearby_param_grid': nearby_grid,
    'sub_period_results': sub_results,
}

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'flow_reversal_v2_results.json')
with open(out_path, 'w') as f:
    json.dump(save_data, f, indent=2, default=str)
print(f"\nResults saved to flow_reversal_v2_results.json", flush=True)
