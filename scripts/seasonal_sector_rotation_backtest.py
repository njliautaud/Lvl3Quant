#!/usr/bin/env python3
"""
Seasonal Sector Rotation Backtest
Academic basis: Jacobsen & Visaltanachoti (2009)
6 variants tested with 5-gate validation.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────
START_DATE = "2020-01-01"  # extra lookback for SMA
OOT_START  = "2022-01-01"
OOT_END    = "2026-07-30"
INITIAL_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
PERM_ITERS = 1000
np.random.seed(42)

# ── Download data ───────────────────────────────────────────────────────
TICKERS = ["SPY", "QQQ", "GLD", "XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP"]

print("Downloading price data...")
raw = yf.download(TICKERS, start=START_DATE, end=OOT_END, auto_adjust=True, progress=False)
# yfinance returns MultiIndex columns (Price, Ticker) — grab Close
if isinstance(raw.columns, pd.MultiIndex):
    close = raw["Close"]
else:
    close = raw
close = close.ffill().dropna()
print(f"  Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} rows")

# Daily returns
rets = close.pct_change().fillna(0)

# SPY 200-SMA for regime
spy_sma200 = close["SPY"].rolling(200).mean()
regime = (close["SPY"] > spy_sma200).astype(int)  # 1=Bull, 0=Bear

# Filter to OOT period
oot_mask = close.index >= OOT_START
close_oot = close.loc[oot_mask]
rets_oot = rets.loc[oot_mask]
regime_oot = regime.loc[oot_mask]

print(f"  OOT period: {close_oot.index[0].date()} to {close_oot.index[-1].date()}, {len(close_oot)} days")

# ── Helper: compute strategy returns from signal series ─────────────────
def strategy_returns(holdings_series):
    """
    holdings_series: pd.Series indexed by date, values are ticker names.
    Returns daily returns series with slippage on switch days.
    """
    strat_ret = pd.Series(0.0, index=rets_oot.index)
    switches = 0
    prev_hold = None
    for i, dt in enumerate(rets_oot.index):
        hold = holdings_series.loc[dt]
        r = rets_oot.loc[dt, hold]
        if prev_hold is not None and hold != prev_hold:
            # Slippage on both legs (sell old, buy new)
            r -= 2 * SLIPPAGE_PCT
            switches += 1
        strat_ret.iloc[i] = r
        prev_hold = hold
    return strat_ret, switches


# ── Variant A: Classic Sell-in-May ──────────────────────────────────────
def variant_a():
    hold = pd.Series(index=rets_oot.index, dtype=str)
    for dt in rets_oot.index:
        m = dt.month
        hold.loc[dt] = "SPY" if m >= 11 or m <= 4 else "GLD"
    ret, sw = strategy_returns(hold)
    return ret, sw, hold


# ── Variant B: Tech Seasonal ───────────────────────────────────────────
def variant_b():
    hold = pd.Series(index=rets_oot.index, dtype=str)
    for dt in rets_oot.index:
        m = dt.month
        hold.loc[dt] = "QQQ" if m >= 11 or m <= 4 else "XLP"
    ret, sw = strategy_returns(hold)
    return ret, sw, hold


# ── Variant C: Sector Calendar ─────────────────────────────────────────
def variant_c():
    sectors = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP"]
    # Compute historical average monthly return per sector (using pre-OOT data)
    pre_oot = rets.loc[rets.index < OOT_START, sectors]
    monthly_avg = {}
    for m in range(1, 13):
        mask = pre_oot.index.month == m
        monthly_avg[m] = pre_oot.loc[mask].mean()

    hold = pd.Series(index=rets_oot.index, dtype=str)
    for dt in rets_oot.index:
        best_sector = monthly_avg[dt.month].idxmax()
        hold.loc[dt] = best_sector
    ret, sw = strategy_returns(hold)
    return ret, sw, hold


# ── Variant D: Seasonal + Momentum Confirm ─────────────────────────────
def variant_d():
    # 3-month momentum of SPY
    spy_mom3m = close["SPY"].pct_change(63)  # ~63 trading days = 3 months

    hold = pd.Series(index=rets_oot.index, dtype=str)
    for dt in rets_oot.index:
        m = dt.month
        if m >= 11 or m <= 4:
            # Winter: SPY only if 3m momentum positive
            mom = spy_mom3m.loc[:dt].iloc[-1] if dt in spy_mom3m.index else spy_mom3m.asof(dt)
            hold.loc[dt] = "SPY" if mom > 0 else "GLD"
        else:
            hold.loc[dt] = "GLD"
    ret, sw = strategy_returns(hold)
    return ret, sw, hold


# ── Variant E: Q4 Rally ────────────────────────────────────────────────
def variant_e():
    hold = pd.Series(index=rets_oot.index, dtype=str)
    for dt in rets_oot.index:
        m = dt.month
        hold.loc[dt] = "SPY" if m in [11, 12, 1] else "GLD"
    ret, sw = strategy_returns(hold)
    return ret, sw, hold


# ── Variant F: Earnings Season ──────────────────────────────────────────
def variant_f():
    hold = pd.Series(index=rets_oot.index, dtype=str)
    for dt in rets_oot.index:
        m = dt.month
        hold.loc[dt] = "QQQ" if m in [1, 4, 7, 10] else "GLD"
    ret, sw = strategy_returns(hold)
    return ret, sw, hold


# ── Metrics ─────────────────────────────────────────────────────────────
def compute_metrics(daily_rets, n_switches, label):
    equity = INITIAL_CAPITAL * (1 + daily_rets).cumprod()
    total_ret = equity.iloc[-1] / INITIAL_CAPITAL - 1
    ann_ret = (1 + total_ret) ** (252 / len(daily_rets)) - 1
    ann_vol = daily_rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    downside = daily_rets[daily_rets < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside > 0 else 0

    # Max drawdown
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    max_dd = dd.min()

    # Profit factor
    gains = daily_rets[daily_rets > 0].sum()
    losses = abs(daily_rets[daily_rets < 0].sum())
    pf = gains / losses if losses > 0 else float("inf")

    # Win rate (daily)
    wr = (daily_rets > 0).sum() / len(daily_rets)

    # Calmar
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    return {
        "variant": label,
        "total_return_pct": round(total_ret * 100, 2),
        "ann_return_pct": round(ann_ret * 100, 2),
        "ann_vol_pct": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(pf, 3),
        "win_rate": round(wr, 4),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "n_switches": n_switches,
        "final_equity": round(equity.iloc[-1], 2),
    }


def permutation_test(daily_rets, holdings_series, n_iters=PERM_ITERS):
    """
    Circular block bootstrap permutation test. We circularly shift the
    holdings assignment relative to the return series by a random offset,
    preserving the autocorrelation structure of both the returns and the
    allocation pattern, but breaking the calendar alignment. This tests
    whether the SPECIFIC seasonal timing matters.
    """
    obs_sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if daily_rets.std() > 0 else 0
    count = 0
    n = len(daily_rets)
    hold_arr = holdings_series.values
    dates = rets_oot.index

    for _ in range(n_iters):
        # Circular shift: shift holdings by random offset
        shift = np.random.randint(1, n)
        shifted_hold = np.roll(hold_arr, shift)
        # Compute returns with shifted allocation
        perm_ret = np.zeros(n)
        prev_h = None
        for i in range(n):
            h = shifted_hold[i]
            r = rets_oot.iloc[i][h]
            if prev_h is not None and h != prev_h:
                r -= 2 * SLIPPAGE_PCT
            perm_ret[i] = r
            prev_h = h
        std = perm_ret.std()
        s = perm_ret.mean() / std * np.sqrt(252) if std > 0 else 0
        if s >= obs_sharpe:
            count += 1
    return count / n_iters


def regime_analysis(daily_rets, regime_series):
    """Compute Sharpe in bull vs bear regimes."""
    bull_mask = regime_series == 1
    bear_mask = regime_series == 0

    def _sharpe(r):
        if len(r) < 10 or r.std() == 0:
            return 0.0
        return r.mean() / r.std() * np.sqrt(252)

    bull_sharpe = _sharpe(daily_rets[bull_mask])
    bear_sharpe = _sharpe(daily_rets[bear_mask])

    denom = max(abs(bull_sharpe), abs(bear_sharpe))
    gap = abs(bull_sharpe - bear_sharpe) / denom if denom > 0 else 0

    return {
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "regime_gap": round(gap, 3),
    }


def five_gate_validation(metrics, perm_p, regime_info):
    """5-gate validation. Returns dict of gate results."""
    gates = {
        "G1_sharpe_gt_0.5": bool(metrics["sharpe"] > 0.5),
        "G2_perm_p_lt_0.05": bool(perm_p < 0.05),
        "G3_regime_gap_lt_0.5": bool(regime_info["regime_gap"] < 0.5),
        "G4_maxdd_gt_neg50": bool(metrics["max_drawdown_pct"] > -50),
        "G5_trades_gte_20": bool(metrics["n_switches"] >= 20),
    }
    gates["all_pass"] = all(gates.values())
    return gates


# ── Run all variants ────────────────────────────────────────────────────
print("\nRunning 6 seasonal rotation variants...\n")

variants = {
    "A_classic_sell_in_may": variant_a,
    "B_tech_seasonal": variant_b,
    "C_sector_calendar": variant_c,
    "D_seasonal_momentum_confirm": variant_d,
    "E_q4_rally": variant_e,
    "F_earnings_season": variant_f,
}

all_results = {}
summary_rows = []

for name, func in variants.items():
    print(f"  {name}...", end=" ", flush=True)
    daily_rets, n_switches, holdings = func()
    metrics = compute_metrics(daily_rets, n_switches, name)
    perm_p = permutation_test(daily_rets, holdings)
    reg = regime_analysis(daily_rets, regime_oot)
    gates = five_gate_validation(metrics, perm_p, reg)

    result = {
        **metrics,
        "perm_p_value": round(perm_p, 4),
        **reg,
        **gates,
    }
    all_results[name] = result
    summary_rows.append(result)
    status = "PASS" if gates["all_pass"] else "FAIL"
    print(f"Sharpe={metrics['sharpe']:.3f}  MaxDD={metrics['max_drawdown_pct']:.1f}%  "
          f"Perm-p={perm_p:.3f}  RegGap={reg['regime_gap']:.3f}  "
          f"Trades={n_switches}  [{status}]")

# ── SPY Buy-and-Hold Benchmark ──────────────────────────────────────────
spy_oot_ret = rets_oot["SPY"]
spy_equity = INITIAL_CAPITAL * (1 + spy_oot_ret).cumprod()
spy_total = spy_equity.iloc[-1] / INITIAL_CAPITAL - 1
spy_ann = (1 + spy_total) ** (252 / len(spy_oot_ret)) - 1
spy_vol = spy_oot_ret.std() * np.sqrt(252)
spy_sharpe = spy_ann / spy_vol if spy_vol > 0 else 0
spy_dd = ((spy_equity - spy_equity.cummax()) / spy_equity.cummax()).min()

benchmark = {
    "total_return_pct": round(spy_total * 100, 2),
    "ann_return_pct": round(spy_ann * 100, 2),
    "sharpe": round(spy_sharpe, 3),
    "max_drawdown_pct": round(spy_dd * 100, 2),
    "final_equity": round(spy_equity.iloc[-1], 2),
}

# ── Output ──────────────────────────────────────────────────────────────
output = {
    "strategy": "Seasonal Sector Rotation",
    "academic_basis": "Jacobsen & Visaltanachoti (2009)",
    "oot_period": f"{OOT_START} to {OOT_END}",
    "initial_capital": INITIAL_CAPITAL,
    "slippage_pct": SLIPPAGE_PCT,
    "commission": 0,
    "regime_definition": "SPY > 200-SMA = Bull, else Bear",
    "perm_test_iterations": PERM_ITERS,
    "variants": all_results,
    "benchmark_spy_bah": benchmark,
    "generated_at": datetime.now().isoformat(),
}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

out_path = "/home/jupiter/Lvl3Quant/data/seasonal_sector_rotation_results.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, cls=NumpyEncoder)

print(f"\nResults saved to {out_path}")

# ── Summary table ───────────────────────────────────────────────────────
print("\n" + "=" * 100)
print(f"{'Variant':<35} {'Sharpe':>7} {'Sortino':>8} {'AnnRet%':>8} {'MaxDD%':>8} {'PF':>6} {'WR':>6} {'Perm-p':>7} {'RegGap':>7} {'5G':>5}")
print("-" * 100)
for r in summary_rows:
    status = "PASS" if r["all_pass"] else "FAIL"
    print(f"{r['variant']:<35} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} {r['ann_return_pct']:>8.2f} "
          f"{r['max_drawdown_pct']:>8.2f} {r['profit_factor']:>6.2f} {r['win_rate']:>6.3f} "
          f"{r['perm_p_value']:>7.3f} {r['regime_gap']:>7.3f} {status:>5}")
print("-" * 100)
print(f"{'SPY B&H (benchmark)':<35} {spy_sharpe:>7.3f} {'--':>8} {spy_ann*100:>8.2f} "
      f"{spy_dd*100:>8.2f} {'--':>6} {'--':>6} {'--':>7} {'--':>7} {'--':>5}")
print("=" * 100)

n_pass = sum(1 for r in summary_rows if r["all_pass"])
print(f"\n{n_pass}/6 variants passed all 5 gates.")
if n_pass > 0:
    best = max([r for r in summary_rows if r["all_pass"]], key=lambda x: x["sharpe"])
    print(f"Best passing variant: {best['variant']} (Sharpe={best['sharpe']:.3f}, "
          f"Sortino={best['sortino']:.3f}, AnnRet={best['ann_return_pct']:.2f}%, "
          f"MaxDD={best['max_drawdown_pct']:.2f}%)")
else:
    best = max(summary_rows, key=lambda x: x["sharpe"])
    print(f"No variant passed all gates. Best by Sharpe: {best['variant']} (Sharpe={best['sharpe']:.3f})")
