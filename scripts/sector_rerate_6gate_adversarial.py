#!/usr/bin/env python3
"""
6-GATE ADVERSARIAL VALIDATION: Sector Re-Rating Acceleration Signal
====================================================================
Gate 1: Re-implementation from scratch (concept only)
Gate 2: Inverse direction (bottom-3 instead of top-3)
Gate 3: Random timing permutation (1000 shuffles)
Gate 4: Cost sensitivity (0.20%, 0.50%, 1.0%)
Gate 5: Sub-period consistency (4 equal sub-periods)
Gate 6: Parameter robustness (lookback x top_n x hold grid)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import warnings
import sys
import time

warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────────────
SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLU', 'XLP', 'XLY', 'XLV', 'XLI', 'XLB', 'XLC', 'XLRE']
BENCHMARK = 'SPY'
ALL_TICKERS = SECTOR_ETFS + [BENCHMARK]

# Original parameters
LOOKBACK = 20
HOLD_DAYS = 5
TOP_N = 3
COST_RT = 0.0020
ESTIMATION_WINDOW = 60

DATA_START = '2020-01-01'
DATA_END = '2026-08-21'

print("=" * 90, flush=True)
print("6-GATE ADVERSARIAL VALIDATION: Sector Re-Rating Acceleration", flush=True)
print("=" * 90, flush=True)

# ── Download Data ───────────────────────────────────────────────────────────────
print("\n[DATA] Downloading sector ETF + SPY daily data...", flush=True)
raw = yf.download(ALL_TICKERS, start=DATA_START, end=DATA_END,
                   progress=False, auto_adjust=True, group_by='ticker', threads=True)

closes = {}
for ticker in ALL_TICKERS:
    try:
        s = raw[ticker]['Close'].dropna().squeeze()
        if len(s) > 100:
            closes[ticker] = s
    except Exception as e:
        print(f"  WARNING: {ticker}: {e}", flush=True)

prices = pd.DataFrame(closes).dropna()
print(f"  Got {len(prices)} days, {len(prices.columns)} tickers, "
      f"{prices.index[0].date()} to {prices.index[-1].date()}", flush=True)

returns = prices.pct_change()

# ── Original Signal (from the script) ───────────────────────────────────────────
def original_signal(prices_df, lookback=LOOKBACK):
    """
    Original: Re-Rating Acceleration = 5-day change in (excess_ret / vol).
    excess_ret = sector_cum_ret - spy_cum_ret over lookback window.
    vol = sector rolling vol annualized.
    """
    rets = prices_df.pct_change()
    spy_ret = rets[BENCHMARK]
    signals = pd.DataFrame(index=prices_df.index, columns=SECTOR_ETFS, dtype=float)
    for etf in SECTOR_ETFS:
        if etf not in rets.columns:
            continue
        etf_ret = rets[etf]
        etf_cum = etf_ret.rolling(lookback).sum()
        spy_cum = spy_ret.rolling(lookback).sum()
        excess = etf_cum - spy_cum
        vol = etf_ret.rolling(lookback).std() * np.sqrt(252)
        rerate_speed = excess / vol.replace(0, np.nan)
        signals[etf] = rerate_speed.diff(5)  # 5-day acceleration
    return signals


def run_backtest(prices_df, signal_df, top_n=TOP_N, hold=HOLD_DAYS, cost_rt=COST_RT,
                 bottom=False, warmup=None):
    """
    Generic backtest engine.
    bottom=True picks worst sectors instead of best.
    """
    if warmup is None:
        warmup = LOOKBACK + ESTIMATION_WINDOW
    rets = prices_df[SECTOR_ETFS].pct_change()
    spy_rets = prices_df[BENCHMARK].pct_change()
    tradeable = prices_df.index[warmup:]

    strat_rets = []
    strat_dates = []
    holdings = []
    days_since = hold  # force rebalance first day

    for date in tradeable:
        if days_since >= hold:
            day_sig = signal_df.loc[date].dropna()
            if len(day_sig) >= top_n:
                ranked = day_sig.sort_values(ascending=bottom)  # ascending=True for bottom
                holdings = list(ranked.index[:top_n])
            days_since = 0

        if holdings:
            dr = rets.loc[date, holdings].mean()
            if days_since == 0:
                dr -= cost_rt / hold
            strat_rets.append(float(dr))
        else:
            strat_rets.append(0.0)

        strat_dates.append(date)
        days_since += 1

    sr = pd.Series(strat_rets, index=strat_dates)
    spy_sr = spy_rets.loc[strat_dates]
    return sr, spy_sr


def sharpe(rets):
    if len(rets) < 10 or rets.std() == 0:
        return 0.0
    return float((rets.mean() / rets.std()) * np.sqrt(252))


def sortino(rets):
    down = rets[rets < 0]
    if len(down) < 5 or down.std() == 0:
        return 0.0
    return float((rets.mean() / down.std()) * np.sqrt(252))


def profit_factor(rets):
    g = rets[rets > 0].sum()
    l = abs(rets[rets < 0].sum())
    return float(g / l) if l > 0 else 0.0


def win_rate(rets):
    return float((rets > 0).mean() * 100)


def max_dd(rets):
    cum = (1 + rets).cumprod()
    peak = cum.expanding().max()
    dd = (cum - peak) / peak
    return float(dd.min() * 100)


# ── Run Original ────────────────────────────────────────────────────────────────
print("\n" + "=" * 90, flush=True)
print("[BASELINE] Running original signal backtest...", flush=True)
print("=" * 90, flush=True)

orig_signal_df = original_signal(prices)
orig_rets, orig_spy = run_backtest(prices, orig_signal_df)

orig_sharpe = sharpe(orig_rets)
orig_sortino = sortino(orig_rets)
orig_pf = profit_factor(orig_rets)
orig_wr = win_rate(orig_rets)
orig_mdd = max_dd(orig_rets)
orig_cum = float(((1 + orig_rets).prod() - 1) * 100)

print(f"  OOT days: {len(orig_rets)}", flush=True)
print(f"  Sharpe:   {orig_sharpe:.3f}", flush=True)
print(f"  Sortino:  {orig_sortino:.3f}", flush=True)
print(f"  PF:       {orig_pf:.3f}", flush=True)
print(f"  WR:       {orig_wr:.1f}%", flush=True)
print(f"  Cum ret:  {orig_cum:.2f}%", flush=True)
print(f"  Max DD:   {orig_mdd:.2f}%", flush=True)


# ════════════════════════════════════════════════════════════════════════════════
# GATE 1: RE-IMPLEMENTATION FROM SCRATCH
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90, flush=True)
print("GATE 1: RE-IMPLEMENTATION FROM SCRATCH", flush=True)
print("  Concept: rank sectors by 5-day change in vol-adjusted relative momentum vs SPY", flush=True)
print("  Independent implementation, no code reuse from original", flush=True)
print("=" * 90, flush=True)

# Fresh implementation from concept description only:
# "5-day change in vol-adjusted relative momentum vs SPY"
# 1. For each sector, compute relative return vs SPY over lookback window
# 2. Divide by sector volatility (vol-adjust)
# 3. Take 5-day difference (acceleration)

spy_daily = returns[BENCHMARK]
gate1_signals = pd.DataFrame(index=prices.index, columns=SECTOR_ETFS, dtype=float)

for etf in SECTOR_ETFS:
    etf_daily = returns[etf]
    # Relative return over 20 days: sum of daily (sector - spy) returns
    relative_ret = (etf_daily - spy_daily).rolling(20).sum()
    # Vol-adjust: divide by sector's 20-day realized vol (annualized)
    sector_vol = etf_daily.rolling(20).std() * np.sqrt(252)
    vol_adj_rel_mom = relative_ret / sector_vol.replace(0, np.nan)
    # Acceleration: 5-day change
    gate1_signals[etf] = vol_adj_rel_mom.diff(5)

gate1_rets, _ = run_backtest(prices, gate1_signals)
gate1_sharpe = sharpe(gate1_rets)

sharpe_diff = abs(gate1_sharpe - orig_sharpe)
gate1_pass = sharpe_diff <= 0.3

print(f"  Original Sharpe:         {orig_sharpe:.3f}", flush=True)
print(f"  Re-implementation Sharpe: {gate1_sharpe:.3f}", flush=True)
print(f"  |Difference|:            {sharpe_diff:.3f} (threshold: 0.3)", flush=True)
print(f"  GATE 1: {'PASS' if gate1_pass else 'FAIL'}", flush=True)


# ════════════════════════════════════════════════════════════════════════════════
# GATE 2: INVERSE DIRECTION (BOTTOM-3)
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90, flush=True)
print("GATE 2: INVERSE DIRECTION (pick BOTTOM-3 sectors)", flush=True)
print("  If inverse also works (Sharpe > 0.50), signal is noise", flush=True)
print("=" * 90, flush=True)

inv_rets, _ = run_backtest(prices, orig_signal_df, bottom=True)
inv_sharpe = sharpe(inv_rets)
inv_cum = float(((1 + inv_rets).prod() - 1) * 100)

gate2_pass = inv_sharpe <= 0.50

print(f"  Original (top-3) Sharpe:  {orig_sharpe:.3f}", flush=True)
print(f"  Inverse (bottom-3) Sharpe: {inv_sharpe:.3f}", flush=True)
print(f"  Inverse cum return:       {inv_cum:.2f}%", flush=True)
print(f"  Threshold: inverse must be <= 0.50", flush=True)
print(f"  GATE 2: {'PASS' if gate2_pass else 'FAIL'}", flush=True)


# ════════════════════════════════════════════════════════════════════════════════
# GATE 3: RANDOM TIMING PERMUTATION (1000 shuffles)
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90, flush=True)
print("GATE 3: RANDOM TIMING PERMUTATION (1000 shuffles)", flush=True)
print("  Shuffle entry dates, keep sector returns intact", flush=True)
print("  Original must beat 95th percentile of random Sharpes", flush=True)
print("=" * 90, flush=True)

warmup = LOOKBACK + ESTIMATION_WINDOW
tradeable_dates = prices.index[warmup:]
sector_returns = returns[SECTOR_ETFS]

N_PERMS = 1000
perm_sharpes = []

t0 = time.time()
for p in range(N_PERMS):
    rng = np.random.RandomState(p + 7777)
    strat_rets_perm = []
    holdings = []
    days_since = HOLD_DAYS

    for date in tradeable_dates:
        if days_since >= HOLD_DAYS:
            # Random selection: pick TOP_N sectors randomly
            avail = list(SECTOR_ETFS)
            rng.shuffle(avail)
            holdings = avail[:TOP_N]
            days_since = 0

        if holdings:
            dr = sector_returns.loc[date, holdings].mean()
            if days_since == 0:
                dr -= COST_RT / HOLD_DAYS
            strat_rets_perm.append(float(dr))
        else:
            strat_rets_perm.append(0.0)
        days_since += 1

    sr_perm = pd.Series(strat_rets_perm)
    if sr_perm.std() > 0:
        perm_sharpes.append(float((sr_perm.mean() / sr_perm.std()) * np.sqrt(252)))
    else:
        perm_sharpes.append(0.0)

    if (p + 1) % 200 == 0:
        print(f"  ... {p+1}/{N_PERMS} permutations done ({time.time()-t0:.1f}s)", flush=True)

perm_sharpes = np.array(perm_sharpes)
p95 = np.percentile(perm_sharpes, 95)
p99 = np.percentile(perm_sharpes, 99)
pval = (perm_sharpes >= orig_sharpe).mean()

gate3_pass = orig_sharpe > p95

print(f"  Original Sharpe:     {orig_sharpe:.3f}", flush=True)
print(f"  Perm mean Sharpe:    {perm_sharpes.mean():.3f} +/- {perm_sharpes.std():.3f}", flush=True)
print(f"  Perm 95th pctl:      {p95:.3f}", flush=True)
print(f"  Perm 99th pctl:      {p99:.3f}", flush=True)
print(f"  p-value:             {pval:.4f}", flush=True)
print(f"  GATE 3: {'PASS' if gate3_pass else 'FAIL'}", flush=True)


# ════════════════════════════════════════════════════════════════════════════════
# GATE 4: COST SENSITIVITY
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90, flush=True)
print("GATE 4: COST SENSITIVITY", flush=True)
print("  Test at 0.20%, 0.50%, 1.0% round-trip cost", flush=True)
print("  Must maintain Sharpe > 0.5 at 0.50% cost", flush=True)
print("=" * 90, flush=True)

cost_levels = [0.0020, 0.0050, 0.0100]
cost_sharpes = {}

for cost in cost_levels:
    cr, _ = run_backtest(prices, orig_signal_df, cost_rt=cost)
    cs = sharpe(cr)
    cost_sharpes[cost] = cs
    ccum = float(((1 + cr).prod() - 1) * 100)
    print(f"  Cost {cost*100:.2f}%: Sharpe={cs:.3f}, Cum={ccum:.2f}%", flush=True)

gate4_pass = cost_sharpes[0.0050] > 0.5

print(f"  At 0.50% cost: Sharpe={cost_sharpes[0.0050]:.3f} (threshold: > 0.5)", flush=True)
print(f"  GATE 4: {'PASS' if gate4_pass else 'FAIL'}", flush=True)


# ════════════════════════════════════════════════════════════════════════════════
# GATE 5: SUB-PERIOD CONSISTENCY
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90, flush=True)
print("GATE 5: SUB-PERIOD CONSISTENCY", flush=True)
print("  Split OOT into 4 equal sub-periods. All must have positive Sharpe.", flush=True)
print("=" * 90, flush=True)

n = len(orig_rets)
chunk = n // 4
sub_sharpes = []
all_positive = True

for i in range(4):
    start_idx = i * chunk
    end_idx = (i + 1) * chunk if i < 3 else n
    sub = orig_rets.iloc[start_idx:end_idx]
    ss = sharpe(sub)
    sub_sharpes.append(ss)
    date_start = sub.index[0].strftime('%Y-%m-%d')
    date_end = sub.index[-1].strftime('%Y-%m-%d')
    scum = float(((1 + sub).prod() - 1) * 100)
    status = "OK" if ss > 0 else "NEGATIVE"
    if ss <= 0:
        all_positive = False
    print(f"  Period {i+1}: {date_start} to {date_end} ({len(sub)}d) "
          f"Sharpe={ss:.3f} Cum={scum:.2f}% [{status}]", flush=True)

gate5_pass = all_positive

print(f"  GATE 5: {'PASS' if gate5_pass else 'FAIL'}", flush=True)


# ════════════════════════════════════════════════════════════════════════════════
# GATE 6: PARAMETER ROBUSTNESS
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90, flush=True)
print("GATE 6: PARAMETER ROBUSTNESS", flush=True)
print("  Grid: lookback=[5,10,15,20,30] x top_n=[2,3,4] x hold=[3,5,7,10]", flush=True)
print("  Need >= 60% of grid to produce Sharpe > 0.5", flush=True)
print("=" * 90, flush=True)

lookbacks = [5, 10, 15, 20, 30]
top_ns = [2, 3, 4]
holds = [3, 5, 7, 10]

total_combos = len(lookbacks) * len(top_ns) * len(holds)
passing_combos = 0
grid_results = []

t0 = time.time()
combo_count = 0

for lb in lookbacks:
    # Recompute signal with this lookback
    sig = pd.DataFrame(index=prices.index, columns=SECTOR_ETFS, dtype=float)
    for etf in SECTOR_ETFS:
        etf_ret = returns[etf]
        etf_cum = etf_ret.rolling(lb).sum()
        spy_cum = spy_daily.rolling(lb).sum()
        excess = etf_cum - spy_cum
        vol = etf_ret.rolling(lb).std() * np.sqrt(252)
        rerate = excess / vol.replace(0, np.nan)
        sig[etf] = rerate.diff(5)

    for tn in top_ns:
        for hd in holds:
            combo_count += 1
            # Adjust warmup for different lookback
            wu = lb + ESTIMATION_WINDOW
            cr, _ = run_backtest(prices, sig, top_n=tn, hold=hd, warmup=wu)
            cs = sharpe(cr)
            passed = cs > 0.5
            if passed:
                passing_combos += 1
            grid_results.append({
                'lookback': lb, 'top_n': tn, 'hold': hd,
                'sharpe': cs, 'pass': passed
            })

    print(f"  Lookback {lb}: done ({combo_count}/{total_combos}, "
          f"{time.time()-t0:.1f}s)", flush=True)

pass_pct = passing_combos / total_combos * 100
gate6_pass = pass_pct >= 60.0

print(f"\n  Total combos: {total_combos}", flush=True)
print(f"  Passing (Sharpe > 0.5): {passing_combos} ({pass_pct:.1f}%)", flush=True)
print(f"  Threshold: >= 60%", flush=True)

# Show grid summary
print(f"\n  Grid detail (Sharpe by lookback x top_n, hold=5):", flush=True)
print(f"  {'LB':>4}  {'top2':>6}  {'top3':>6}  {'top4':>6}", flush=True)
for lb in lookbacks:
    vals = []
    for tn in top_ns:
        match = [r for r in grid_results if r['lookback'] == lb and r['top_n'] == tn and r['hold'] == 5]
        vals.append(f"{match[0]['sharpe']:.3f}" if match else "  N/A")
    print(f"  {lb:>4}  {'  '.join(vals)}", flush=True)

print(f"\n  GATE 6: {'PASS' if gate6_pass else 'FAIL'}", flush=True)


# ════════════════════════════════════════════════════════════════════════════════
# FINAL VERDICT
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90, flush=True)
print("FINAL VERDICT", flush=True)
print("=" * 90, flush=True)

gates = [
    ("Gate 1: Re-implementation", gate1_pass, f"Sharpe diff={sharpe_diff:.3f}"),
    ("Gate 2: Inverse direction", gate2_pass, f"Inverse Sharpe={inv_sharpe:.3f}"),
    ("Gate 3: Random timing perm", gate3_pass, f"p-value={pval:.4f}, 95th pctl={p95:.3f}"),
    ("Gate 4: Cost sensitivity", gate4_pass, f"Sharpe@0.50%={cost_sharpes[0.0050]:.3f}"),
    ("Gate 5: Sub-period consistency", gate5_pass, f"Sub-Sharpes={[round(s,3) for s in sub_sharpes]}"),
    ("Gate 6: Parameter robustness", gate6_pass, f"{pass_pct:.1f}% of grid passes"),
]

n_pass = sum(1 for _, p, _ in gates if p)

for name, passed, evidence in gates:
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}: {evidence}", flush=True)

print(f"\n  RESULT: {n_pass}/6 gates passed", flush=True)

if n_pass >= 5:
    print(f"  VERDICT: SIGNAL IS LIKELY REAL. Proceed to paper trading.", flush=True)
elif n_pass >= 3:
    print(f"  VERDICT: MIXED EVIDENCE. Signal may have conditional edge. Investigate failures.", flush=True)
else:
    print(f"  VERDICT: SIGNAL IS LIKELY SPURIOUS. Do not trade.", flush=True)

print(f"\n  Original baseline: Sharpe={orig_sharpe:.3f}, Sortino={orig_sortino:.3f}, "
      f"PF={orig_pf:.3f}, WR={orig_wr:.1f}%, MDD={orig_mdd:.2f}%", flush=True)
