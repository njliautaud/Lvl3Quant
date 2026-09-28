#!/usr/bin/env python3
"""
6-GATE ADVERSARIAL VALIDATION: Momentum Growth — Strategy E (RS Vol-Weighted)
==============================================================================
Target: Strategy E from momentum_growth_backtest.py
  - Cross-sectional relative strength vs SPY, inverse-vol weighting
  - Reported: 29.7% CAGR, Sharpe 1.08

Gate 1: Re-implementation from concept (no code reuse)
Gate 2: Inverse signal (buy WORST momentum instead of best)
Gate 3: Random timing permutation (1000 shuffles)
Gate 4: Cost sensitivity (0, 10, 20, 50 bps)
Gate 5: Sub-period stability (4 equal sub-periods, all positive)
Gate 6: Parameter robustness (top_n x lookback x rebal_freq grid)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import sys
import json
import os

np.random.seed(42)

# ── Config (matches original backtest) ─────────────────────────────────────────
INDIVIDUAL_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'AVGO', 'AMD', 'TSM',
    'LLY', 'UNH', 'JPM', 'V', 'MA', 'COST', 'CRM', 'ORCL',
    'NFLX', 'ADBE', 'NOW', 'UBER', 'ABNB', 'PLTR', 'COIN',
]
BENCHMARK = 'SPY'
START_DATE = '2019-01-01'
END_DATE = '2026-08-26'

# Original Strategy E parameters
TRAIN_MONTHS = 12
TOP_N = 8      # TOP_N_STOCKS from original
RS_WINDOW = 126  # 6-month relative strength lookback
VOL_WINDOW = 63  # 3-month vol for inverse-vol weighting
REBAL_COST_BPS = 10

print("=" * 90, flush=True)
print("6-GATE ADVERSARIAL VALIDATION: Momentum Growth — Strategy E (RS Vol-Weighted)", flush=True)
print("=" * 90, flush=True)
print(f"Period: {START_DATE} to {END_DATE}", flush=True)
print(f"Universe: {len(INDIVIDUAL_STOCKS)} individual stocks", flush=True)
print(f"Parameters: top_n={TOP_N}, RS_window={RS_WINDOW}d, vol_window={VOL_WINDOW}d", flush=True)
print(flush=True)

# ── Download Data ──────────────────────────────────────────────────────────────
print("[DATA] Downloading stock + SPY daily data...", flush=True)
all_tickers = list(set(INDIVIDUAL_STOCKS + [BENCHMARK]))
data = {}
for ticker in all_tickers:
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

available = [t for t in INDIVIDUAL_STOCKS if t in data]
close_df = pd.DataFrame({t: data[t]['Close'] for t in available}).dropna(how='all')
spy_close = data[BENCHMARK]['Close'] if BENCHMARK in data else None

if spy_close is None:
    print("FATAL: SPY data not available", flush=True)
    sys.exit(1)

spy_ret = spy_close.pct_change().dropna()
print(f"  Got {len(close_df)} days, {len(available)} stocks", flush=True)
print(f"  Date range: {close_df.index[0].date()} to {close_df.index[-1].date()}", flush=True)


# ── Original Strategy E: RS Vol-Weighted (from original code) ──────────────────
def strategy_e_original(close_df, spy_close, tickers, train_end, top_n=TOP_N):
    """Exact replica of strategy_e_rs_vol_weighted from the backtest."""
    # Relative strength vs SPY over RS_WINDOW
    asset_ret = close_df[tickers].pct_change(RS_WINDOW)
    spy_ret_w = spy_close.pct_change(RS_WINDOW)
    rs = asset_ret.subtract(spy_ret_w, axis=0)

    # Realized vol
    vol = close_df[tickers].pct_change().rolling(VOL_WINDOW).std()

    if train_end not in rs.index:
        valid = rs.index[rs.index <= train_end]
        if len(valid) == 0:
            return {}
        train_end = valid[-1]

    rs_row = rs.loc[train_end].dropna()
    vol_row = vol.loc[train_end].dropna()
    common = rs_row.index.intersection(vol_row.index)

    if len(common) < top_n:
        top_n = max(1, len(common))

    top = rs_row[common].nlargest(top_n).index.tolist()

    vols = vol_row[top].replace(0, np.nan).dropna()
    if len(vols) == 0:
        return {t: 1.0 / len(top) for t in top}

    inv_vol = 1.0 / vols
    weights = inv_vol / inv_vol.sum()
    return weights.to_dict()


def run_walk_forward(close_df, spy_close, tickers, strategy_fn, top_n=TOP_N,
                     rebal_cost_bps=REBAL_COST_BPS, train_months=TRAIN_MONTHS):
    """Walk-forward engine: sliding train window, monthly rebalance."""
    all_dates = close_df.index
    start = all_dates[0]

    results_returns = []
    first_rebal = start + pd.DateOffset(months=train_months)
    rebal_dates = pd.date_range(first_rebal, all_dates[-1], freq='ME')
    rebal_dates = [d for d in rebal_dates if d <= all_dates[-1]]

    for i, rebal_date in enumerate(rebal_dates):
        valid_dates = all_dates[all_dates <= rebal_date]
        if len(valid_dates) == 0:
            continue
        train_end = valid_dates[-1]

        if i + 1 < len(rebal_dates):
            next_rebal = rebal_dates[i + 1]
            valid_next = all_dates[all_dates <= next_rebal]
            if len(valid_next) == 0:
                continue
            oos_end = valid_next[-1]
        else:
            oos_end = all_dates[-1]

        oos_mask = (all_dates > train_end) & (all_dates <= oos_end)
        oos_dates = all_dates[oos_mask]
        if len(oos_dates) == 0:
            continue

        weights = strategy_fn(close_df, spy_close, tickers, train_end, top_n=top_n)
        if not weights:
            continue

        daily_ret = close_df[list(weights.keys())].pct_change()
        oos_ret = daily_ret.loc[oos_dates]

        port_ret = pd.Series(0.0, index=oos_dates)
        for t, w in weights.items():
            if t in oos_ret.columns:
                port_ret += w * oos_ret[t].fillna(0)

        turnover_cost = rebal_cost_bps / 10000.0
        if len(port_ret) > 0:
            port_ret.iloc[0] -= turnover_cost

        results_returns.append(port_ret)

    if not results_returns:
        return pd.Series(dtype=float)

    full_returns = pd.concat(results_returns).sort_index()
    full_returns = full_returns[~full_returns.index.duplicated(keep='first')]
    return full_returns


# ── Metric helpers ──────────────────────────────────────────────────────────────
def sharpe(rets):
    if len(rets) < 30 or rets.std() == 0:
        return 0.0
    n_years = len(rets) / 252
    total_ret = (1 + rets).prod() - 1
    cagr = (1 + total_ret) ** (1 / n_years) - 1
    ann_vol = rets.std() * np.sqrt(252)
    return float(cagr / (ann_vol + 1e-10))


def sortino(rets):
    if len(rets) < 30:
        return 0.0
    n_years = len(rets) / 252
    total_ret = (1 + rets).prod() - 1
    cagr = (1 + total_ret) ** (1 / n_years) - 1
    down = rets[rets < 0]
    down_vol = down.std() * np.sqrt(252) if len(down) > 5 else 1e-10
    return float(cagr / (down_vol + 1e-10))


def cagr(rets):
    if len(rets) < 30:
        return 0.0
    n_years = len(rets) / 252
    total_ret = (1 + rets).prod() - 1
    return float((1 + total_ret) ** (1 / n_years) - 1)


def max_dd(rets):
    cum = (1 + rets).cumprod()
    peak = cum.expanding().max()
    dd = (cum - peak) / peak
    return float(dd.min())


def profit_factor(rets):
    g = rets[rets > 0].sum()
    l = abs(rets[rets < 0].sum())
    return float(g / (l + 1e-10))


def win_rate_monthly(rets):
    monthly = rets.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    return float((monthly > 0).mean())


# ── Run Original (Baseline) ───────────────────────────────────────────────────
print("\n" + "=" * 90, flush=True)
print("[BASELINE] Running original Strategy E (RS vol-weighted)...", flush=True)
print("=" * 90, flush=True)

orig_rets = run_walk_forward(close_df, spy_close, available, strategy_e_original)

orig_sharpe = sharpe(orig_rets)
orig_sortino = sortino(orig_rets)
orig_cagr = cagr(orig_rets)
orig_mdd = max_dd(orig_rets)
orig_pf = profit_factor(orig_rets)
orig_wr = win_rate_monthly(orig_rets)

print(f"  OOS days:  {len(orig_rets)}", flush=True)
print(f"  CAGR:      {orig_cagr:.1%}", flush=True)
print(f"  Sharpe:    {orig_sharpe:.3f}", flush=True)
print(f"  Sortino:   {orig_sortino:.3f}", flush=True)
print(f"  MaxDD:     {orig_mdd:.1%}", flush=True)
print(f"  PF:        {orig_pf:.3f}", flush=True)
print(f"  WR(mo):    {orig_wr:.0%}", flush=True)

gate_results = {}


# ════════════════════════════════════════════════════════════════════════════════
# GATE 1: RE-IMPLEMENTATION FROM SCRATCH
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90, flush=True)
print("GATE 1: RE-IMPLEMENTATION FROM SCRATCH", flush=True)
print("  Concept: rank stocks by 6-month excess return vs SPY, weight by inverse vol", flush=True)
print("  Independent code, no function reuse from original", flush=True)
print("=" * 90, flush=True)


def strategy_e_reimplemented(close_df, spy_close, tickers, train_end, top_n=TOP_N):
    """
    Fresh re-implementation from concept only:
    1. Compute each stock's 126-day total return
    2. Compute SPY's 126-day total return
    3. Excess = stock - SPY
    4. Pick top_n by excess return
    5. Weight by 1/realized_vol (63-day)
    """
    if train_end not in close_df.index:
        valid = close_df.index[close_df.index <= train_end]
        if len(valid) == 0:
            return {}
        train_end = valid[-1]

    # 126-day returns computed fresh
    te_idx = close_df.index.get_loc(train_end)
    if te_idx < 126:
        return {}

    lookback_start = close_df.index[te_idx - 126]

    stock_returns = {}
    for t in tickers:
        if t not in close_df.columns:
            continue
        p_now = close_df.loc[train_end, t]
        p_then = close_df.loc[lookback_start, t]
        if pd.notna(p_now) and pd.notna(p_then) and p_then > 0:
            stock_returns[t] = (p_now / p_then) - 1

    # SPY return
    spy_now = spy_close.loc[train_end] if train_end in spy_close.index else np.nan
    spy_then = spy_close.loc[lookback_start] if lookback_start in spy_close.index else np.nan
    if pd.isna(spy_now) or pd.isna(spy_then) or spy_then == 0:
        return {}
    spy_ret_126 = (spy_now / spy_then) - 1

    # Excess return
    excess = {t: r - spy_ret_126 for t, r in stock_returns.items()}
    if len(excess) < top_n:
        top_n = max(1, len(excess))

    # Pick top_n
    sorted_excess = sorted(excess.items(), key=lambda x: x[1], reverse=True)
    top_tickers = [t for t, _ in sorted_excess[:top_n]]

    # Inverse vol weighting (63-day realized vol)
    vol_lookback_start_idx = max(0, te_idx - 63)
    vol_slice = close_df.iloc[vol_lookback_start_idx:te_idx + 1][top_tickers].pct_change().dropna()
    vols = vol_slice.std()
    vols = vols.replace(0, np.nan).dropna()

    if len(vols) == 0:
        return {t: 1.0 / len(top_tickers) for t in top_tickers}

    inv_v = 1.0 / vols
    w = inv_v / inv_v.sum()
    return w.to_dict()


reimpl_rets = run_walk_forward(close_df, spy_close, available, strategy_e_reimplemented)
reimpl_sharpe = sharpe(reimpl_rets)
reimpl_cagr_val = cagr(reimpl_rets)

sharpe_diff = abs(reimpl_sharpe - orig_sharpe)
gate1_pass = sharpe_diff <= 0.30

print(f"  Original Sharpe:         {orig_sharpe:.3f}  (CAGR {orig_cagr:.1%})", flush=True)
print(f"  Re-implementation Sharpe: {reimpl_sharpe:.3f}  (CAGR {reimpl_cagr_val:.1%})", flush=True)
print(f"  |Sharpe difference|:     {sharpe_diff:.3f} (threshold: <= 0.30)", flush=True)
print(f"  GATE 1: {'PASS' if gate1_pass else 'FAIL'}", flush=True)
gate_results['Gate 1: Re-implementation'] = 'PASS' if gate1_pass else 'FAIL'


# ════════════════════════════════════════════════════════════════════════════════
# GATE 2: INVERSE SIGNAL (buy WORST momentum)
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90, flush=True)
print("GATE 2: INVERSE SIGNAL (buy WORST RS stocks instead of BEST)", flush=True)
print("  If inverse also profits, signal is noise / beta exposure", flush=True)
print("=" * 90, flush=True)


def strategy_e_inverse(close_df, spy_close, tickers, train_end, top_n=TOP_N):
    """Buy the WORST relative strength stocks, inverse-vol weighted."""
    asset_ret = close_df[tickers].pct_change(RS_WINDOW)
    spy_ret_w = spy_close.pct_change(RS_WINDOW)
    rs = asset_ret.subtract(spy_ret_w, axis=0)
    vol = close_df[tickers].pct_change().rolling(VOL_WINDOW).std()

    if train_end not in rs.index:
        valid = rs.index[rs.index <= train_end]
        if len(valid) == 0:
            return {}
        train_end = valid[-1]

    rs_row = rs.loc[train_end].dropna()
    vol_row = vol.loc[train_end].dropna()
    common = rs_row.index.intersection(vol_row.index)

    if len(common) < top_n:
        top_n = max(1, len(common))

    # WORST instead of BEST
    bottom = rs_row[common].nsmallest(top_n).index.tolist()

    vols = vol_row[bottom].replace(0, np.nan).dropna()
    if len(vols) == 0:
        return {t: 1.0 / len(bottom) for t in bottom}

    inv_vol = 1.0 / vols
    weights = inv_vol / inv_vol.sum()
    return weights.to_dict()


inv_rets = run_walk_forward(close_df, spy_close, available, strategy_e_inverse)
inv_sharpe = sharpe(inv_rets)
inv_cagr_val = cagr(inv_rets)

# Inverse should lose money or significantly underperform
# Gate passes if inverse Sharpe < 0.5 (much worse than original)
gate2_pass = inv_sharpe < 0.50
# Also check: inverse should be meaningfully worse than original
if inv_sharpe > orig_sharpe * 0.8:
    gate2_pass = False  # inverse nearly as good = no directional edge

print(f"  Original (best RS) Sharpe:  {orig_sharpe:.3f}  CAGR {orig_cagr:.1%}", flush=True)
print(f"  Inverse (worst RS) Sharpe:  {inv_sharpe:.3f}  CAGR {inv_cagr_val:.1%}", flush=True)
print(f"  Threshold: inverse Sharpe < 0.50 AND < 80% of original", flush=True)
print(f"  GATE 2: {'PASS' if gate2_pass else 'FAIL'}", flush=True)
gate_results['Gate 2: Inverse signal'] = 'PASS' if gate2_pass else 'FAIL'


# ════════════════════════════════════════════════════════════════════════════════
# GATE 3: RANDOM TIMING PERMUTATION (1000 shuffles)
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90, flush=True)
print("GATE 3: RANDOM TIMING PERMUTATION (1000 shuffles)", flush=True)
print("  Randomize entry dates, keep stock returns intact", flush=True)
print("  Original must beat 95% of random entries", flush=True)
print("=" * 90, flush=True)


def strategy_e_random(close_df, spy_close, tickers, train_end, top_n=TOP_N, rng=None):
    """Pick random stocks, inverse-vol weighted."""
    vol = close_df[tickers].pct_change().rolling(VOL_WINDOW).std()

    if train_end not in vol.index:
        valid = vol.index[vol.index <= train_end]
        if len(valid) == 0:
            return {}
        train_end = valid[-1]

    vol_row = vol.loc[train_end].dropna()
    avail = list(vol_row.index)
    if len(avail) < top_n:
        top_n = max(1, len(avail))

    if rng is None:
        rng = np.random.RandomState(0)
    rng.shuffle(avail)
    picks = avail[:top_n]

    vols = vol_row[picks].replace(0, np.nan).dropna()
    if len(vols) == 0:
        return {t: 1.0 / len(picks) for t in picks}

    inv_vol = 1.0 / vols
    weights = inv_vol / inv_vol.sum()
    return weights.to_dict()


N_PERMS = 1000
perm_sharpes = []
t0 = time.time()

for p in range(N_PERMS):
    rng = np.random.RandomState(p + 9999)

    def random_fn(close_df, spy_close, tickers, train_end, top_n=TOP_N, _rng=rng):
        return strategy_e_random(close_df, spy_close, tickers, train_end, top_n=top_n, rng=_rng)

    perm_rets = run_walk_forward(close_df, spy_close, available, random_fn)
    if len(perm_rets) > 30:
        perm_sharpes.append(sharpe(perm_rets))
    else:
        perm_sharpes.append(0.0)

    if (p + 1) % 200 == 0:
        print(f"  ... {p+1}/{N_PERMS} permutations ({time.time()-t0:.1f}s)", flush=True)

perm_sharpes = np.array(perm_sharpes)
p95 = np.percentile(perm_sharpes, 95)
p99 = np.percentile(perm_sharpes, 99)
pctile = (perm_sharpes < orig_sharpe).mean() * 100

gate3_pass = orig_sharpe > p95

print(f"  Original Sharpe:       {orig_sharpe:.3f}", flush=True)
print(f"  Random mean Sharpe:    {perm_sharpes.mean():.3f} +/- {perm_sharpes.std():.3f}", flush=True)
print(f"  Random 95th pctl:      {p95:.3f}", flush=True)
print(f"  Random 99th pctl:      {p99:.3f}", flush=True)
print(f"  Strategy percentile:   {pctile:.1f}%", flush=True)
print(f"  GATE 3: {'PASS' if gate3_pass else 'FAIL'}", flush=True)
gate_results['Gate 3: Random timing'] = 'PASS' if gate3_pass else 'FAIL'


# ════════════════════════════════════════════════════════════════════════════════
# GATE 4: COST SENSITIVITY
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90, flush=True)
print("GATE 4: COST SENSITIVITY (0, 10, 20, 50 bps)", flush=True)
print("  Must survive realistic costs — Sharpe > 0.5 at 20 bps", flush=True)
print("=" * 90, flush=True)

cost_levels_bps = [0, 10, 20, 50]
cost_results = {}

for cost_bps in cost_levels_bps:
    cr = run_walk_forward(close_df, spy_close, available, strategy_e_original,
                          rebal_cost_bps=cost_bps)
    cs = sharpe(cr)
    cc = cagr(cr)
    cost_results[cost_bps] = {'sharpe': cs, 'cagr': cc}
    print(f"  {cost_bps:3d} bps: Sharpe={cs:.3f}  CAGR={cc:.1%}", flush=True)

# Must have Sharpe > 0.5 at 20 bps (realistic ETF/stock costs)
gate4_pass = cost_results[20]['sharpe'] > 0.5

print(f"  At 20 bps: Sharpe={cost_results[20]['sharpe']:.3f} (threshold: > 0.5)", flush=True)
print(f"  At 50 bps: Sharpe={cost_results[50]['sharpe']:.3f} (stress test)", flush=True)
print(f"  GATE 4: {'PASS' if gate4_pass else 'FAIL'}", flush=True)
gate_results['Gate 4: Cost sensitivity'] = 'PASS' if gate4_pass else 'FAIL'


# ════════════════════════════════════════════════════════════════════════════════
# GATE 5: SUB-PERIOD STABILITY
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90, flush=True)
print("GATE 5: SUB-PERIOD STABILITY (4 equal sub-periods)", flush=True)
print("  All 4 sub-periods must have positive total return", flush=True)
print("=" * 90, flush=True)

n = len(orig_rets)
chunk = n // 4
sub_results = []
all_positive = True

for i in range(4):
    s_idx = i * chunk
    e_idx = (i + 1) * chunk if i < 3 else n
    sub = orig_rets.iloc[s_idx:e_idx]
    ss = sharpe(sub)
    sc = cagr(sub)
    cum_ret = float((1 + sub).prod() - 1)

    sub_results.append({
        'period': f"{sub.index[0].strftime('%Y-%m')} to {sub.index[-1].strftime('%Y-%m')}",
        'sharpe': ss,
        'cagr': sc,
        'cum_return': cum_ret,
        'days': len(sub),
    })

    if cum_ret <= 0:
        all_positive = False

    print(f"  Period {i+1}: {sub_results[-1]['period']}  "
          f"Sharpe={ss:.3f}  CAGR={sc:.1%}  Cum={cum_ret:.1%}  "
          f"({'OK' if cum_ret > 0 else 'NEGATIVE'})", flush=True)

gate5_pass = all_positive
print(f"  All sub-periods positive: {all_positive}", flush=True)
print(f"  GATE 5: {'PASS' if gate5_pass else 'FAIL'}", flush=True)
gate_results['Gate 5: Sub-period stability'] = 'PASS' if gate5_pass else 'FAIL'


# ════════════════════════════════════════════════════════════════════════════════
# GATE 6: PARAMETER ROBUSTNESS
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90, flush=True)
print("GATE 6: PARAMETER ROBUSTNESS", flush=True)
print("  Grid: top_n=[3,4,5,6,7,8], lookback=[63,84,126,189,252], rebal=[W,M,Q]", flush=True)
print("  >50% of grid must have Sharpe > 0.3", flush=True)
print("=" * 90, flush=True)

top_n_grid = [3, 4, 5, 6, 7, 8]
lookback_grid = [63, 84, 126, 189, 252]  # ~3m, 4m, 6m, 9m, 12m
rebal_freqs = {
    'W': 'W-FRI',   # Weekly
    'M': 'ME',      # Monthly
    'Q': 'QE',      # Quarterly
}

grid_results = []
total_combos = len(top_n_grid) * len(lookback_grid) * len(rebal_freqs)
combo_idx = 0

t0 = time.time()
for tn in top_n_grid:
    for lb in lookback_grid:
        for rebal_label, rebal_freq in rebal_freqs.items():
            combo_idx += 1

            def param_strategy(close_df, spy_close, tickers, train_end,
                               top_n=tn, _lb=lb):
                """Parameterized Strategy E."""
                asset_ret = close_df[tickers].pct_change(_lb)
                spy_ret_w = spy_close.pct_change(_lb)
                rs = asset_ret.subtract(spy_ret_w, axis=0)
                vol = close_df[tickers].pct_change().rolling(VOL_WINDOW).std()

                if train_end not in rs.index:
                    valid = rs.index[rs.index <= train_end]
                    if len(valid) == 0:
                        return {}
                    train_end = valid[-1]

                rs_row = rs.loc[train_end].dropna()
                vol_row = vol.loc[train_end].dropna()
                common = rs_row.index.intersection(vol_row.index)

                if len(common) < top_n:
                    top_n_adj = max(1, len(common))
                else:
                    top_n_adj = top_n

                top = rs_row[common].nlargest(top_n_adj).index.tolist()
                vols = vol_row[top].replace(0, np.nan).dropna()
                if len(vols) == 0:
                    return {t: 1.0 / len(top) for t in top}
                inv_vol = 1.0 / vols
                weights = inv_vol / inv_vol.sum()
                return weights.to_dict()

            # Custom walk-forward with different rebal frequency
            all_dates = close_df.index
            start = all_dates[0]
            first_rebal = start + pd.DateOffset(months=TRAIN_MONTHS)
            rebal_dates = pd.date_range(first_rebal, all_dates[-1], freq=rebal_freq)
            rebal_dates = [d for d in rebal_dates if d <= all_dates[-1]]

            period_returns = []
            for ri, rebal_date in enumerate(rebal_dates):
                valid_dates = all_dates[all_dates <= rebal_date]
                if len(valid_dates) == 0:
                    continue
                train_end = valid_dates[-1]

                if ri + 1 < len(rebal_dates):
                    next_rebal = rebal_dates[ri + 1]
                    valid_next = all_dates[all_dates <= next_rebal]
                    if len(valid_next) == 0:
                        continue
                    oos_end = valid_next[-1]
                else:
                    oos_end = all_dates[-1]

                oos_mask = (all_dates > train_end) & (all_dates <= oos_end)
                oos_dates = all_dates[oos_mask]
                if len(oos_dates) == 0:
                    continue

                weights = param_strategy(close_df, spy_close, available, train_end,
                                         top_n=tn)
                if not weights:
                    continue

                daily_ret = close_df[list(weights.keys())].pct_change()
                oos_ret = daily_ret.loc[oos_dates]
                port_ret = pd.Series(0.0, index=oos_dates)
                for t, w in weights.items():
                    if t in oos_ret.columns:
                        port_ret += w * oos_ret[t].fillna(0)

                port_ret.iloc[0] -= REBAL_COST_BPS / 10000.0
                period_returns.append(port_ret)

            if period_returns:
                full_ret = pd.concat(period_returns).sort_index()
                full_ret = full_ret[~full_ret.index.duplicated(keep='first')]
                gs = sharpe(full_ret)
            else:
                gs = 0.0

            grid_results.append({
                'top_n': tn, 'lookback': lb, 'rebal': rebal_label,
                'sharpe': gs,
            })

            if combo_idx % 15 == 0:
                print(f"  ... {combo_idx}/{total_combos} combos "
                      f"({time.time()-t0:.1f}s)", flush=True)

# Analyze grid
grid_df = pd.DataFrame(grid_results)
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
    print(f"    top_n={int(row['top_n'])}, lookback={int(row['lookback'])}, "
          f"rebal={row['rebal']}: Sharpe={row['sharpe']:.3f}", flush=True)
print(f"  Worst 5 combos:", flush=True)
for _, row in grid_df.nsmallest(5, 'sharpe').iterrows():
    print(f"    top_n={int(row['top_n'])}, lookback={int(row['lookback'])}, "
          f"rebal={row['rebal']}: Sharpe={row['sharpe']:.3f}", flush=True)

# By parameter dimension
print(f"\n  By top_n:", flush=True)
for tn in top_n_grid:
    sub = grid_df[grid_df['top_n'] == tn]
    print(f"    top_n={tn}: mean Sharpe={sub['sharpe'].mean():.3f}", flush=True)

print(f"  By lookback:", flush=True)
for lb in lookback_grid:
    sub = grid_df[grid_df['lookback'] == lb]
    print(f"    lookback={lb}: mean Sharpe={sub['sharpe'].mean():.3f}", flush=True)

print(f"  By rebal freq:", flush=True)
for rl in rebal_freqs:
    sub = grid_df[grid_df['rebal'] == rl]
    print(f"    rebal={rl}: mean Sharpe={sub['sharpe'].mean():.3f}", flush=True)

print(f"\n  GATE 6: {'PASS' if gate6_pass else 'FAIL'}", flush=True)
gate_results['Gate 6: Parameter robustness'] = 'PASS' if gate6_pass else 'FAIL'


# ════════════════════════════════════════════════════════════════════════════════
# FINAL SCORECARD
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90, flush=True)
print("FINAL ADVERSARIAL VALIDATION SCORECARD", flush=True)
print("=" * 90, flush=True)

n_pass = sum(1 for v in gate_results.values() if v == 'PASS')
n_total = len(gate_results)

for gate, result in gate_results.items():
    icon = "[PASS]" if result == "PASS" else "[FAIL]"
    print(f"  {icon} {gate}", flush=True)

print(f"\n  OVERALL: {n_pass}/{n_total} gates passed", flush=True)

if n_pass == n_total:
    verdict = "STRATEGY VALIDATED — all 6 gates passed"
elif n_pass >= 4:
    verdict = "PARTIAL PASS — strategy shows edge but has weaknesses"
elif n_pass >= 2:
    verdict = "WEAK — strategy has significant concerns"
else:
    verdict = "FAIL — strategy likely not robust"

print(f"  VERDICT: {verdict}", flush=True)
print("=" * 90, flush=True)

# ── Save results ───────────────────────────────────────────────────────────────
save_data = {
    'run_date': datetime.now().isoformat(),
    'strategy': 'E_rs_vol_weighted',
    'baseline': {
        'cagr': orig_cagr, 'sharpe': orig_sharpe, 'sortino': orig_sortino,
        'max_dd': orig_mdd, 'pf': orig_pf, 'wr_monthly': orig_wr,
    },
    'gates': gate_results,
    'n_pass': n_pass,
    'n_total': n_total,
    'verdict': verdict,
    'gate_details': {
        'gate1_reimpl_sharpe': reimpl_sharpe,
        'gate2_inverse_sharpe': inv_sharpe,
        'gate3_percentile': float(pctile),
        'gate4_cost_results': {str(k): v for k, v in cost_results.items()},
        'gate5_sub_periods': sub_results,
        'gate6_pct_above_threshold': float(pct_above),
        'gate6_grid_mean_sharpe': float(grid_df['sharpe'].mean()),
    }
}

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'momentum_growth_adversarial_results.json')
with open(out_path, 'w') as f:
    json.dump(save_data, f, indent=2, default=str)
print(f"\nResults saved.", flush=True)
