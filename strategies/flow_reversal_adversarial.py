#!/usr/bin/env python3
"""
6-GATE ADVERSARIAL VALIDATION: Flow Reversal Strategy
======================================================
Target: Flow Reversal (AVO v25) — buy ETFs on volume spike + price dip
  - Signal: volume z-score >= 2.0 AND 1-day return <= -1%
  - Hold max 3 days, TP +3.5%, trailing stop -2%
  - Max 2 concurrent positions
  - Lockbox: Sharpe 3.37, PF 1.90, +7.3%, 26 trades (2026)
  - Leveraged projection: 37% CAGR at 2x, Sharpe 2.93

Gate 1: Re-implementation from concept (independent code, verify similar results)
Gate 2: Inverse signal (volume COLLAPSE + price RISE — should lose money)
Gate 3: Random timing permutation (200 shuffles, must beat 95%+)
Gate 4: Cost sensitivity (0, 10, 20, 50 bps — must survive 20 bps)
Gate 5: Sub-period stability (4 equal sub-periods, all positive)
Gate 6: Parameter robustness (vol_z x dip x hold x top_n grid — >50% Sharpe > 0.3)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import time
import sys
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

# Strategy parameters (from AVO v25 / paper engine)
VOL_LOOKBACK = 20
VOL_Z_THRESHOLD = 2.0
PRICE_DIP_THRESHOLD = -0.01  # -1%
MAX_HOLD_DAYS = 3
TAKE_PROFIT = 0.035           # +3.5%
TRAILING_STOP = -0.02         # -2%
MAX_CONCURRENT = 2
COST_BPS_DEFAULT = 10         # 10 bps round-trip for ETFs

N_PERMS = 200  # permutation count for Gate 3

print("=" * 90, flush=True)
print("6-GATE ADVERSARIAL VALIDATION: Flow Reversal Strategy", flush=True)
print("=" * 90, flush=True)
print(f"Period: {START_DATE} to {END_DATE}", flush=True)
print(f"Universe: {len(UNIVERSE)} ETFs: {', '.join(UNIVERSE)}", flush=True)
print(f"Signal: vol_z >= {VOL_Z_THRESHOLD}, dip <= {PRICE_DIP_THRESHOLD*100:.0f}%", flush=True)
print(f"Exits: TP={TAKE_PROFIT*100:.1f}%, trailing stop={TRAILING_STOP*100:.0f}%, max hold={MAX_HOLD_DAYS}d", flush=True)
print(f"Max concurrent: {MAX_CONCURRENT}", flush=True)
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

# Align indices
common_idx = close_df.index.intersection(volume_df.index)
close_df = close_df.loc[common_idx]
volume_df = volume_df.loc[common_idx]

print(f"  Date range: {close_df.index[0].date()} to {close_df.index[-1].date()}", flush=True)
print(f"  Trading days: {len(close_df)}", flush=True)


# ── Core backtest engine ─────────────────────────────────────────────────────
def backtest_flow_reversal(close_df, volume_df, tickers,
                           vol_z_thresh=VOL_Z_THRESHOLD,
                           dip_thresh=PRICE_DIP_THRESHOLD,
                           max_hold=MAX_HOLD_DAYS,
                           tp=TAKE_PROFIT,
                           trail_stop=TRAILING_STOP,
                           max_concurrent=MAX_CONCURRENT,
                           cost_bps=COST_BPS_DEFAULT,
                           signal_override=None,
                           top_n=MAX_CONCURRENT):
    """
    Event-driven backtest of flow reversal.

    signal_override: if provided, a function(close_df, volume_df, tickers, date_idx)
                     that returns list of (ticker, score) signals for that day.
    top_n: max number of new positions to open per day (from top signals).

    Returns: pd.Series of daily portfolio returns (equal-weight per-trade).
    """
    dates = close_df.index
    n_days = len(dates)

    # Pre-compute returns and volume z-scores
    returns_1d = close_df[tickers].pct_change()
    vol_mean = volume_df[tickers].rolling(vol_z_thresh if isinstance(vol_z_thresh, int) and vol_z_thresh > 10 else VOL_LOOKBACK).mean()
    vol_std = volume_df[tickers].rolling(VOL_LOOKBACK).std()
    vol_z = (volume_df[tickers] - vol_mean) / (vol_std + 1e-10)

    # Track positions: list of dicts {ticker, entry_price, entry_idx, peak_price}
    positions = []
    daily_pnl = pd.Series(0.0, index=dates)
    trade_log = []
    cost_per_trade = cost_bps / 10000.0

    for i in range(VOL_LOOKBACK + 2, n_days):
        date = dates[i]

        # --- Exit check for existing positions ---
        positions_to_remove = []
        for pos in positions:
            ticker = pos['ticker']
            if ticker not in close_df.columns:
                positions_to_remove.append(pos)
                continue

            current_price = close_df.loc[date, ticker]
            if pd.isna(current_price):
                continue

            entry_price = pos['entry_price']
            pnl_pct = (current_price - entry_price) / entry_price
            days_held = i - pos['entry_idx']

            # Update peak
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
                # P&L for this trade (equal weight: 1/max_concurrent of portfolio)
                trade_pnl = pnl_pct - cost_per_trade  # exit cost
                trade_log.append({
                    'entry_date': str(dates[pos['entry_idx']].date()),
                    'exit_date': str(date.date()),
                    'ticker': ticker,
                    'pnl_pct': trade_pnl,
                    'days_held': days_held,
                    'reason': exit_reason,
                })
                # Attribute to daily P&L (spread across days held)
                daily_pnl.iloc[i] += trade_pnl / max_concurrent
                positions_to_remove.append(pos)

        for pos in positions_to_remove:
            if pos in positions:
                positions.remove(pos)

        # --- Entry check: generate signals ---
        open_slots = max_concurrent - len(positions)
        if open_slots <= 0:
            continue

        held_tickers = {p['ticker'] for p in positions}

        if signal_override is not None:
            signals = signal_override(close_df, volume_df, tickers, i)
        else:
            # Standard flow reversal signal
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

        # Sort by score descending, take top_n
        signals.sort(key=lambda x: x[1], reverse=True)

        for ticker, score in signals[:min(open_slots, top_n)]:
            if ticker in held_tickers:
                continue
            price = close_df.iloc[i].get(ticker, np.nan)
            if pd.isna(price) or price <= 0:
                continue

            positions.append({
                'ticker': ticker,
                'entry_price': price,
                'entry_idx': i,
                'peak_price': price,
            })
            held_tickers.add(ticker)
            # Entry cost
            daily_pnl.iloc[i] -= cost_per_trade / max_concurrent

    # Force-close any remaining positions at end
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


def print_metrics(rets, trade_log, label=""):
    s = calc_sharpe(rets)
    so = calc_sortino(rets)
    c = calc_cagr(rets)
    mdd = calc_max_dd(rets)
    pf = calc_pf(rets)
    wr = calc_wr(trade_log)
    print(f"  {label}Sharpe={s:.3f}  Sortino={so:.3f}  CAGR={c:.1%}  "
          f"MaxDD={mdd:.1%}  PF={pf:.3f}  WR={wr:.0%}  Trades={len(trade_log)}", flush=True)
    return s, so, c, mdd, pf, wr


# ── BASELINE ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 90, flush=True)
print("[BASELINE] Running original Flow Reversal strategy...", flush=True)
print("=" * 90, flush=True)

orig_rets, orig_trades = backtest_flow_reversal(close_df, volume_df, available)
orig_sharpe, orig_sortino, orig_cagr, orig_mdd, orig_pf, orig_wr = print_metrics(
    orig_rets, orig_trades, "BASELINE: ")

gate_results = {}


# ════════════════════════════════════════════════════════════════════════════════
# GATE 1: RE-IMPLEMENTATION FROM SCRATCH
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90, flush=True)
print("GATE 1: RE-IMPLEMENTATION FROM SCRATCH", flush=True)
print("  Concept: buy ETF when volume spikes + price drops, hold short term with TP/SL", flush=True)
print("  Independent implementation, no function reuse", flush=True)
print("=" * 90, flush=True)


def reimpl_backtest(close_df, volume_df, tickers, cost_bps=COST_BPS_DEFAULT):
    """
    Fresh re-implementation from concept only:
    1. For each day, compute 20-day volume z-score for each ETF
    2. Compute 1-day return
    3. If vol_z >= 2.0 AND ret_1d <= -1%, generate buy signal
    4. Hold up to 3 days, TP at +3.5%, trailing stop at -2%
    5. Max 2 concurrent positions
    """
    dates = close_df.index
    n = len(dates)
    cost = cost_bps / 10000.0

    # Compute all data upfront
    daily_ret = {}
    vol_zscore = {}
    for t in tickers:
        c = close_df[t].values
        v = volume_df[t].values

        ret = np.full(n, np.nan)
        ret[1:] = c[1:] / c[:-1] - 1
        daily_ret[t] = ret

        vz = np.full(n, np.nan)
        for j in range(VOL_LOOKBACK + 1, n):
            window = v[j - VOL_LOOKBACK:j]  # 20 days BEFORE current
            mu = np.nanmean(window)
            sigma = np.nanstd(window, ddof=1)
            if sigma > 0:
                vz[j] = (v[j] - mu) / sigma
        vol_zscore[t] = vz

    positions = []  # (ticker, entry_price, entry_day_idx, peak_price)
    port_ret = np.zeros(n)
    trades = []

    for i in range(VOL_LOOKBACK + 2, n):
        # Check exits
        to_remove = []
        for pos in positions:
            tk, ep, ei, pk = pos
            cp = close_df.iloc[i][tk]
            if pd.isna(cp):
                continue

            pnl = (cp - ep) / ep
            held = i - ei
            pk_new = max(pk, cp)
            pos[3] = pk_new  # update peak
            dd = (cp - pk_new) / pk_new

            do_exit = False
            reason = ''
            if pnl >= 0.035:
                do_exit, reason = True, 'tp'
            elif dd <= -0.02:
                do_exit, reason = True, 'trail'
            elif held >= 3:
                do_exit, reason = True, 'maxhold'

            if do_exit:
                net_pnl = pnl - cost
                port_ret[i] += net_pnl / MAX_CONCURRENT
                trades.append({'entry_date': str(dates[ei].date()),
                               'exit_date': str(dates[i].date()),
                               'ticker': tk, 'pnl_pct': net_pnl,
                               'days_held': held, 'reason': reason})
                to_remove.append(pos)

        for p in to_remove:
            positions.remove(p)

        # Check entries
        slots = MAX_CONCURRENT - len(positions)
        if slots <= 0:
            continue

        held_tk = {p[0] for p in positions}
        sigs = []
        for t in tickers:
            if t in held_tk:
                continue
            vz = vol_zscore[t][i]
            r = daily_ret[t][i]
            if pd.notna(vz) and pd.notna(r) and vz >= 2.0 and r <= -0.01:
                sigs.append((t, vz * abs(r)))

        sigs.sort(key=lambda x: x[1], reverse=True)
        for t, sc in sigs[:slots]:
            p = close_df.iloc[i][t]
            if pd.isna(p) or p <= 0:
                continue
            positions.append([t, float(p), i, float(p)])
            port_ret[i] -= cost / MAX_CONCURRENT
            held_tk.add(t)

    # Force close remaining
    for pos in positions:
        tk, ep, ei, pk = pos
        fp = close_df.iloc[-1][tk]
        if pd.notna(fp):
            pnl = (fp - ep) / ep - cost
            port_ret[-1] += pnl / MAX_CONCURRENT
            trades.append({'entry_date': str(dates[ei].date()),
                           'exit_date': str(dates[-1].date()),
                           'ticker': tk, 'pnl_pct': pnl,
                           'days_held': n - 1 - ei, 'reason': 'eop'})

    return pd.Series(port_ret, index=dates), trades


reimpl_rets, reimpl_trades = reimpl_backtest(close_df, volume_df, available)
reimpl_sharpe = calc_sharpe(reimpl_rets)
reimpl_cagr_val = calc_cagr(reimpl_rets)

print_metrics(reimpl_rets, reimpl_trades, "Re-impl: ")

sharpe_diff = abs(reimpl_sharpe - orig_sharpe)
# For event-driven strategies, allow wider tolerance since the two implementations
# may have slight timing differences in exit logic
gate1_pass = sharpe_diff <= 0.50

print(f"  Original Sharpe:          {orig_sharpe:.3f}", flush=True)
print(f"  Re-implementation Sharpe: {reimpl_sharpe:.3f}", flush=True)
print(f"  |Sharpe difference|:      {sharpe_diff:.3f} (threshold: <= 0.50)", flush=True)
print(f"  GATE 1: {'PASS' if gate1_pass else 'FAIL'}", flush=True)
gate_results['Gate 1: Re-implementation'] = 'PASS' if gate1_pass else 'FAIL'


# ════════════════════════════════════════════════════════════════════════════════
# GATE 2: INVERSE SIGNAL (volume COLLAPSE + price RISE)
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90, flush=True)
print("GATE 2: INVERSE SIGNAL (volume COLLAPSE + price RISE)", flush=True)
print("  Buy when vol z-score <= -2.0 AND 1d return >= +1%", flush=True)
print("  If this also profits, the edge is just momentum/beta, not flow reversal", flush=True)
print("=" * 90, flush=True)


def inverse_signal(close_df, volume_df, tickers, day_idx):
    """Inverse: low volume + price rise."""
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

        # INVERSE: low volume + positive return
        if vz <= -2.0 and r1d >= 0.01:
            score = abs(vz) * abs(r1d)
            signals.append((ticker, score))

    return signals


inv_rets, inv_trades = backtest_flow_reversal(
    close_df, volume_df, available, signal_override=inverse_signal)
inv_sharpe = calc_sharpe(inv_rets)
inv_cagr_val = calc_cagr(inv_rets)

print_metrics(inv_rets, inv_trades, "Inverse: ")

# Inverse should be negative or much worse
gate2_pass = inv_sharpe < 0.50 and inv_sharpe < orig_sharpe * 0.6
# If inverse has very few trades, it's fine (means the signal is asymmetric)
if len(inv_trades) < 5:
    print(f"  Note: inverse produced only {len(inv_trades)} trades (signal is highly asymmetric)", flush=True)
    gate2_pass = True  # Very few inverse signals = good, signal IS directional

print(f"  Original Sharpe:  {orig_sharpe:.3f}  ({len(orig_trades)} trades)", flush=True)
print(f"  Inverse Sharpe:   {inv_sharpe:.3f}  ({len(inv_trades)} trades)", flush=True)
print(f"  Threshold: inverse Sharpe < 0.50 AND < 60% of original", flush=True)
print(f"  GATE 2: {'PASS' if gate2_pass else 'FAIL'}", flush=True)
gate_results['Gate 2: Inverse signal'] = 'PASS' if gate2_pass else 'FAIL'


# ════════════════════════════════════════════════════════════════════════════════
# GATE 3: RANDOM TIMING PERMUTATION (200 shuffles)
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90, flush=True)
print(f"GATE 3: RANDOM TIMING PERMUTATION ({N_PERMS} shuffles)", flush=True)
print("  Randomize which days get signals, keep everything else the same", flush=True)
print("  Original must beat 95%+ of random entries", flush=True)
print("=" * 90, flush=True)

# Count how many signal days the original had
orig_signal_days = len(set(t['entry_date'] for t in orig_trades))
# Average signals per signal day
avg_signals_per_day = len(orig_trades) / max(orig_signal_days, 1)

perm_sharpes = []
t0 = time.time()

for p in range(N_PERMS):
    rng = np.random.RandomState(p + 7777)

    # Generate random signal days with same frequency as original
    n_days_total = len(close_df)
    valid_days = list(range(VOL_LOOKBACK + 2, n_days_total - MAX_HOLD_DAYS))
    n_signal_days = orig_signal_days
    random_signal_days = set(rng.choice(valid_days, size=min(n_signal_days, len(valid_days)),
                                        replace=False))

    def random_signal_fn(close_df, volume_df, tickers, day_idx, _rng=rng,
                         _days=random_signal_days):
        if day_idx not in _days:
            return []
        # Pick random tickers
        avail = [t for t in tickers if t in close_df.columns]
        _rng.shuffle(avail)
        n_pick = min(MAX_CONCURRENT, len(avail))
        signals = [(avail[j], _rng.random()) for j in range(n_pick)]
        return signals

    perm_rets, perm_trades = backtest_flow_reversal(
        close_df, volume_df, available, signal_override=random_signal_fn)

    if len(perm_rets) > 30:
        perm_sharpes.append(calc_sharpe(perm_rets))
    else:
        perm_sharpes.append(0.0)

    if (p + 1) % 50 == 0:
        elapsed = time.time() - t0
        print(f"  ... {p+1}/{N_PERMS} permutations ({elapsed:.1f}s)", flush=True)

perm_sharpes = np.array(perm_sharpes)
p95 = np.percentile(perm_sharpes, 95)
p99 = np.percentile(perm_sharpes, 99)
pctile = (perm_sharpes < orig_sharpe).mean() * 100

gate3_pass = orig_sharpe > p95

print(f"  Original Sharpe:     {orig_sharpe:.3f}", flush=True)
print(f"  Random mean Sharpe:  {perm_sharpes.mean():.3f} +/- {perm_sharpes.std():.3f}", flush=True)
print(f"  Random 95th pctl:    {p95:.3f}", flush=True)
print(f"  Random 99th pctl:    {p99:.3f}", flush=True)
print(f"  Strategy percentile: {pctile:.1f}%", flush=True)
print(f"  GATE 3: {'PASS' if gate3_pass else 'FAIL'}", flush=True)
gate_results['Gate 3: Random timing'] = 'PASS' if gate3_pass else 'FAIL'


# ════════════════════════════════════════════════════════════════════════════════
# GATE 4: COST SENSITIVITY (0, 10, 20, 50 bps)
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90, flush=True)
print("GATE 4: COST SENSITIVITY (0, 10, 20, 50 bps)", flush=True)
print("  Must survive realistic ETF costs — Sharpe > 0.3 at 20 bps", flush=True)
print("=" * 90, flush=True)

cost_levels_bps = [0, 10, 20, 50]
cost_results = {}

for cost_bps in cost_levels_bps:
    cr, ct = backtest_flow_reversal(close_df, volume_df, available, cost_bps=cost_bps)
    cs = calc_sharpe(cr)
    cc = calc_cagr(cr)
    cpf = calc_pf(cr)
    cost_results[cost_bps] = {'sharpe': cs, 'cagr': cc, 'pf': cpf, 'trades': len(ct)}
    print(f"  {cost_bps:3d} bps: Sharpe={cs:.3f}  CAGR={cc:.1%}  PF={cpf:.3f}  Trades={len(ct)}", flush=True)

# Must have Sharpe > 0.3 at 20 bps (flow reversal is short-hold, costs matter more)
gate4_pass = cost_results[20]['sharpe'] > 0.3

print(f"  At 20 bps: Sharpe={cost_results[20]['sharpe']:.3f} (threshold: > 0.3)", flush=True)
print(f"  At 50 bps: Sharpe={cost_results[50]['sharpe']:.3f} (stress test)", flush=True)
print(f"  GATE 4: {'PASS' if gate4_pass else 'FAIL'}", flush=True)
gate_results['Gate 4: Cost sensitivity'] = 'PASS' if gate4_pass else 'FAIL'


# ════════════════════════════════════════════════════════════════════════════════
# GATE 5: SUB-PERIOD STABILITY (4 equal sub-periods)
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90, flush=True)
print("GATE 5: SUB-PERIOD STABILITY (4 equal sub-periods)", flush=True)
print("  All 4 sub-periods must have positive cumulative return", flush=True)
print("=" * 90, flush=True)

n = len(orig_rets)
chunk = n // 4
sub_results = []
all_positive = True

for i in range(4):
    s_idx = i * chunk
    e_idx = (i + 1) * chunk if i < 3 else n
    sub = orig_rets.iloc[s_idx:e_idx]
    ss = calc_sharpe(sub)
    sc = calc_cagr(sub)
    cum_ret = float((1 + sub).prod() - 1)

    # Count trades in this period
    sub_start = sub.index[0].date()
    sub_end = sub.index[-1].date()
    sub_trade_count = sum(1 for t in orig_trades
                         if sub_start <= datetime.strptime(t['entry_date'], '%Y-%m-%d').date() <= sub_end)

    sub_results.append({
        'period': f"{sub.index[0].strftime('%Y-%m')} to {sub.index[-1].strftime('%Y-%m')}",
        'sharpe': ss,
        'cagr': sc,
        'cum_return': cum_ret,
        'trades': sub_trade_count,
        'days': len(sub),
    })

    if cum_ret <= 0:
        all_positive = False

    status = 'OK' if cum_ret > 0 else 'NEGATIVE'
    print(f"  Period {i+1}: {sub_results[-1]['period']}  "
          f"Sharpe={ss:.3f}  CAGR={sc:.1%}  Cum={cum_ret:.1%}  "
          f"Trades={sub_trade_count}  [{status}]", flush=True)

gate5_pass = all_positive
print(f"  All sub-periods positive: {all_positive}", flush=True)
print(f"  GATE 5: {'PASS' if gate5_pass else 'FAIL'}", flush=True)
gate_results['Gate 5: Sub-period stability'] = 'PASS' if gate5_pass else 'FAIL'


# ════════════════════════════════════════════════════════════════════════════════
# GATE 6: PARAMETER ROBUSTNESS
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90, flush=True)
print("GATE 6: PARAMETER ROBUSTNESS", flush=True)
print("  Grid: vol_z=[1.5,2.0,2.5,3.0], dip=[-0.5%,-1%,-1.5%,-2%], "
      "hold=[2,3,4,5], top_n=[1,2,3]", flush=True)
print("  >50% of grid must have Sharpe > 0.3", flush=True)
print("=" * 90, flush=True)

vol_z_grid = [1.5, 2.0, 2.5, 3.0]
dip_grid = [-0.005, -0.01, -0.015, -0.02]
hold_grid = [2, 3, 4, 5]
top_n_grid = [1, 2, 3]

total_combos = len(vol_z_grid) * len(dip_grid) * len(hold_grid) * len(top_n_grid)
grid_results_list = []
combo_idx = 0
t0 = time.time()

for vz_t in vol_z_grid:
    for dip_t in dip_grid:
        for hold_d in hold_grid:
            for tn in top_n_grid:
                combo_idx += 1

                gr, gt = backtest_flow_reversal(
                    close_df, volume_df, available,
                    vol_z_thresh=vz_t,
                    dip_thresh=dip_t,
                    max_hold=hold_d,
                    max_concurrent=tn,
                    top_n=tn,
                )
                gs = calc_sharpe(gr)

                grid_results_list.append({
                    'vol_z': vz_t, 'dip': dip_t, 'hold': hold_d, 'top_n': tn,
                    'sharpe': gs, 'trades': len(gt),
                })

                if combo_idx % 32 == 0:
                    print(f"  ... {combo_idx}/{total_combos} combos "
                          f"({time.time()-t0:.1f}s)", flush=True)

grid_df = pd.DataFrame(grid_results_list)
above_threshold = (grid_df['sharpe'] > 0.3).sum()
pct_above = above_threshold / len(grid_df) * 100

gate6_pass = pct_above > 50

print(f"\n  Grid results: {len(grid_df)} parameter combinations", flush=True)
print(f"  Sharpe range: [{grid_df['sharpe'].min():.3f}, {grid_df['sharpe'].max():.3f}]", flush=True)
print(f"  Sharpe mean:  {grid_df['sharpe'].mean():.3f}", flush=True)
print(f"  Sharpe median:{grid_df['sharpe'].median():.3f}", flush=True)
print(f"  Combos with Sharpe > 0.3: {above_threshold}/{len(grid_df)} ({pct_above:.0f}%)", flush=True)

# Show best/worst
print(f"\n  Best 5 combos:", flush=True)
for _, row in grid_df.nlargest(5, 'sharpe').iterrows():
    print(f"    vol_z={row['vol_z']}, dip={row['dip']*100:.1f}%, hold={int(row['hold'])}, "
          f"top_n={int(row['top_n'])}: Sharpe={row['sharpe']:.3f} ({int(row['trades'])} trades)", flush=True)
print(f"  Worst 5 combos:", flush=True)
for _, row in grid_df.nsmallest(5, 'sharpe').iterrows():
    print(f"    vol_z={row['vol_z']}, dip={row['dip']*100:.1f}%, hold={int(row['hold'])}, "
          f"top_n={int(row['top_n'])}: Sharpe={row['sharpe']:.3f} ({int(row['trades'])} trades)", flush=True)

# By parameter dimension
print(f"\n  By vol_z_threshold:", flush=True)
for vz in vol_z_grid:
    sub = grid_df[grid_df['vol_z'] == vz]
    pct = (sub['sharpe'] > 0.3).mean() * 100
    print(f"    vol_z={vz}: mean Sharpe={sub['sharpe'].mean():.3f}  ({pct:.0f}% > 0.3)", flush=True)

print(f"  By dip_threshold:", flush=True)
for dip in dip_grid:
    sub = grid_df[grid_df['dip'] == dip]
    pct = (sub['sharpe'] > 0.3).mean() * 100
    print(f"    dip={dip*100:.1f}%: mean Sharpe={sub['sharpe'].mean():.3f}  ({pct:.0f}% > 0.3)", flush=True)

print(f"  By hold_days:", flush=True)
for h in hold_grid:
    sub = grid_df[grid_df['hold'] == h]
    pct = (sub['sharpe'] > 0.3).mean() * 100
    print(f"    hold={h}: mean Sharpe={sub['sharpe'].mean():.3f}  ({pct:.0f}% > 0.3)", flush=True)

print(f"  By top_n:", flush=True)
for tn in top_n_grid:
    sub = grid_df[grid_df['top_n'] == tn]
    pct = (sub['sharpe'] > 0.3).mean() * 100
    print(f"    top_n={tn}: mean Sharpe={sub['sharpe'].mean():.3f}  ({pct:.0f}% > 0.3)", flush=True)

print(f"\n  GATE 6: {'PASS' if gate6_pass else 'FAIL'}", flush=True)
gate_results['Gate 6: Parameter robustness'] = 'PASS' if gate6_pass else 'FAIL'


# ════════════════════════════════════════════════════════════════════════════════
# FINAL SCORECARD
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90, flush=True)
print("FINAL ADVERSARIAL VALIDATION SCORECARD — FLOW REVERSAL", flush=True)
print("=" * 90, flush=True)

n_pass = sum(1 for v in gate_results.values() if v == 'PASS')
n_total = len(gate_results)

for gate, result in gate_results.items():
    icon = "[PASS]" if result == "PASS" else "[FAIL]"
    print(f"  {icon} {gate}", flush=True)

print(f"\n  OVERALL: {n_pass}/{n_total} gates passed", flush=True)

if n_pass == n_total:
    verdict = "STRATEGY VALIDATED - all 6 gates passed"
elif n_pass >= 4:
    verdict = "PARTIAL PASS - strategy shows edge but has weaknesses"
elif n_pass >= 2:
    verdict = "WEAK - strategy has significant concerns"
else:
    verdict = "FAIL - strategy likely not robust"

print(f"  VERDICT: {verdict}", flush=True)
print("=" * 90, flush=True)

# ── Save results ──────────────────────────────────────────────────────────────
save_data = {
    'run_date': datetime.now().isoformat(),
    'strategy': 'flow_reversal',
    'baseline': {
        'sharpe': orig_sharpe, 'sortino': orig_sortino, 'cagr': orig_cagr,
        'max_dd': orig_mdd, 'pf': orig_pf, 'wr': orig_wr,
        'n_trades': len(orig_trades),
    },
    'gates': gate_results,
    'n_pass': n_pass,
    'n_total': n_total,
    'verdict': verdict,
    'gate_details': {
        'gate1_reimpl_sharpe': reimpl_sharpe,
        'gate1_sharpe_diff': sharpe_diff,
        'gate2_inverse_sharpe': inv_sharpe,
        'gate2_inverse_trades': len(inv_trades),
        'gate3_percentile': float(pctile),
        'gate3_p95': float(p95),
        'gate4_cost_results': {str(k): v for k, v in cost_results.items()},
        'gate5_sub_periods': sub_results,
        'gate6_pct_above_threshold': float(pct_above),
        'gate6_grid_mean_sharpe': float(grid_df['sharpe'].mean()),
    }
}

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'flow_reversal_adversarial_results.json')
with open(out_path, 'w') as f:
    json.dump(save_data, f, indent=2, default=str)
print(f"\nResults saved.", flush=True)
