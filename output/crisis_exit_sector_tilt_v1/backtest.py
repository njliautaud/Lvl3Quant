"""
Crisis Exit + Sector Recovery Tilt Backtest
============================================
Signal: VIX crosses BELOW 30 after being above 30 → buy signal
Tilt: 40% XLK + 30% XLY + 30% XLB for 3 months, then back to 100% SPY
Baseline: 100% SPY always (never go to cash)
Transaction costs: 10 bps per trade
All signals T-1 (use previous day's VIX to decide today's action)

Validation: permutation test, regime test, lag sensitivity, comparisons
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import timedelta

OUT_DIR = "/home/jupiter/Lvl3Quant/output/crisis_exit_sector_tilt_v1"

# ── 1. Download data ──────────────────────────────────────────────────────
import yfinance as yf

tickers = ["SPY", "XLK", "XLY", "XLB", "^VIX"]
print("Downloading data...")
data = yf.download(tickers, start="2005-01-01", end="2026-07-18", auto_adjust=True)
prices = data["Close"].copy()
prices.columns = [c if c != "^VIX" else "VIX" for c in prices.columns]
prices = prices.dropna()
print(f"Data range: {prices.index[0].date()} to {prices.index[-1].date()}, {len(prices)} days")

# ── 2. Detect crisis exit signals ────────────────────────────────────────
vix = prices["VIX"]

def detect_crisis_exits(vix_series, lag=1):
    """VIX crosses below 30 after being above 30. lag=1 means T-1 signal."""
    above_30 = vix_series > 30
    # Shift by lag: signal available with delay
    shifted_above = above_30.shift(lag)
    shifted_below = (~above_30).shift(lag)  # was below 30 on signal day
    shifted_above_prev = above_30.shift(lag + 1)  # was above 30 day before

    # Cross below: yesterday above 30, today below 30 (with lag applied)
    cross_below = shifted_above_prev & shifted_below
    signal_dates = vix_series.index[cross_below.fillna(False)]
    return signal_dates

signals_t1 = detect_crisis_exits(vix, lag=1)
signals_t0 = detect_crisis_exits(vix, lag=0)
print(f"Crisis exit signals (T-1): {len(signals_t1)} events")
print(f"Crisis exit signals (T-0): {len(signals_t0)} events")

# ── 3. Backtest engine ───────────────────────────────────────────────────
def run_backtest(prices, signal_dates, tilt=True, tcost_bps=10, hold_days=63, label=""):
    """
    Run backtest with crisis exit signals.
    tilt=True: use sector tilt (XLK/XLY/XLB). tilt=False: stay in SPY on signal.
    """
    spy_ret = prices["SPY"].pct_change()
    xlk_ret = prices["XLK"].pct_change()
    xly_ret = prices["XLY"].pct_change()
    xlb_ret = prices["XLB"].pct_change()

    tcost = tcost_bps / 10000.0
    dates = prices.index
    n = len(dates)

    # Track portfolio returns
    port_ret = pd.Series(0.0, index=dates)
    in_tilt = False
    tilt_end = None
    tilt_entries = []

    for i in range(1, n):
        dt = dates[i]

        # Check if tilt period ended
        if in_tilt and dt >= tilt_end:
            in_tilt = False
            # Transaction cost for switching back to SPY
            port_ret.iloc[i] = spy_ret.iloc[i] - tcost
            continue

        # Check if new signal fires today
        if dt in signal_dates and not in_tilt:
            in_tilt = True
            tilt_end = dt + timedelta(days=hold_days)
            tilt_entries.append(dt)
            # Transaction cost for entering tilt
            if tilt:
                port_ret.iloc[i] = (0.4 * xlk_ret.iloc[i] + 0.3 * xly_ret.iloc[i] +
                                     0.3 * xlb_ret.iloc[i]) - tcost
            else:
                # Crisis exit + SPY (no tilt) - just pay tcost for "signal awareness"
                port_ret.iloc[i] = spy_ret.iloc[i] - tcost
            continue

        # Normal day
        if in_tilt and tilt:
            port_ret.iloc[i] = (0.4 * xlk_ret.iloc[i] + 0.3 * xly_ret.iloc[i] +
                                 0.3 * xlb_ret.iloc[i])
        else:
            port_ret.iloc[i] = spy_ret.iloc[i]

    # Build equity curve
    equity = (1 + port_ret).cumprod()

    # Metrics
    ann_ret = equity.iloc[-1] ** (252 / n) - 1
    ann_vol = port_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    downside = port_ret[port_ret < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside > 0 else 0

    # Max drawdown
    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = dd.min()

    # Calmar
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    return {
        "label": label,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "calmar": calmar,
        "total_return": equity.iloc[-1] - 1,
        "n_signals": len(tilt_entries),
        "signal_dates": [str(d.date()) for d in tilt_entries],
        "equity": equity,
        "returns": port_ret,
    }

# ── 4. Run main strategies ───────────────────────────────────────────────
print("\n=== Running backtests ===")

# Pure SPY baseline
res_spy = run_backtest(prices, pd.DatetimeIndex([]), tilt=False, label="Pure SPY")
print(f"Pure SPY: Sharpe={res_spy['sharpe']:.3f}, Return={res_spy['total_return']:.1%}, MaxDD={res_spy['max_dd']:.1%}")

# Crisis exit + SPY (no sector tilt)
res_ce_spy = run_backtest(prices, signals_t1, tilt=False, label="Crisis Exit + SPY")
print(f"Crisis Exit + SPY: Sharpe={res_ce_spy['sharpe']:.3f}, Return={res_ce_spy['total_return']:.1%}, MaxDD={res_ce_spy['max_dd']:.1%}")

# Crisis exit + sector tilt (the main strategy)
res_tilt = run_backtest(prices, signals_t1, tilt=True, label="Crisis Exit + Sector Tilt")
print(f"Crisis Exit + Tilt: Sharpe={res_tilt['sharpe']:.3f}, Return={res_tilt['total_return']:.1%}, MaxDD={res_tilt['max_dd']:.1%}")

# ── 5. Permutation test (200 shuffles of signal timing) ──────────────────
print("\n=== Permutation test (200 shuffles) ===")
np.random.seed(42)
n_perm = 200
perm_sharpes = []

# We shuffle WHEN signals occur (same number of signals, random dates)
n_signals = len(signals_t1)
valid_dates = prices.index[63:-63]  # avoid edges

for i in range(n_perm):
    fake_signals = pd.DatetimeIndex(np.random.choice(valid_dates, size=n_signals, replace=False))
    res_fake = run_backtest(prices, fake_signals, tilt=True, tcost_bps=10, label=f"perm_{i}")
    perm_sharpes.append(res_fake["sharpe"])
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{n_perm} permutations done")

actual_sharpe = res_tilt["sharpe"]
p_value = np.mean([s >= actual_sharpe for s in perm_sharpes])
print(f"Actual Sharpe: {actual_sharpe:.4f}")
print(f"Permutation mean Sharpe: {np.mean(perm_sharpes):.4f} +/- {np.std(perm_sharpes):.4f}")
print(f"p-value: {p_value:.4f}")

# ── 6. Regime test (bull vs bear) ─────────────────────────────────────────
print("\n=== Regime test ===")
spy_200ma = prices["SPY"].rolling(200).mean()
bull_mask = prices["SPY"] > spy_200ma
bear_mask = ~bull_mask

tilt_ret = res_tilt["returns"]
bull_ret = tilt_ret[bull_mask]
bear_ret = tilt_ret[bear_mask]
spy_ret_series = res_spy["returns"]

bull_sharpe = (bull_ret.mean() * 252) / (bull_ret.std() * np.sqrt(252)) if bull_ret.std() > 0 else 0
bear_sharpe = (bear_ret.mean() * 252) / (bear_ret.std() * np.sqrt(252)) if bear_ret.std() > 0 else 0

spy_bull_ret = spy_ret_series[bull_mask]
spy_bear_ret = spy_ret_series[bear_mask]
spy_bull_sharpe = (spy_bull_ret.mean() * 252) / (spy_bull_ret.std() * np.sqrt(252)) if spy_bull_ret.std() > 0 else 0
spy_bear_sharpe = (spy_bear_ret.mean() * 252) / (spy_bear_ret.std() * np.sqrt(252)) if spy_bear_ret.std() > 0 else 0

print(f"Strategy - Bull Sharpe: {bull_sharpe:.3f}, Bear Sharpe: {bear_sharpe:.3f}")
print(f"SPY      - Bull Sharpe: {spy_bull_sharpe:.3f}, Bear Sharpe: {spy_bear_sharpe:.3f}")

sharpe_gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 0.001)
print(f"Regime Sharpe gap (strategy): {sharpe_gap:.3f}")

# ── 7. Lag sensitivity (T-0 vs T-1) ──────────────────────────────────────
print("\n=== Lag sensitivity ===")
res_t0 = run_backtest(prices, signals_t0, tilt=True, label="T-0 (same-day)")
print(f"T-0 Sharpe: {res_t0['sharpe']:.4f}")
print(f"T-1 Sharpe: {res_tilt['sharpe']:.4f}")
print(f"Delta: {res_t0['sharpe'] - res_tilt['sharpe']:.4f}")

# ── 8. Per-signal analysis ────────────────────────────────────────────────
print("\n=== Per-signal forward returns ===")
signal_returns = []
for sd in signals_t1:
    idx = prices.index.get_loc(sd)
    for horizon_name, horizon_days in [("1m", 21), ("3m", 63), ("12m", 252)]:
        end_idx = min(idx + horizon_days, len(prices) - 1)
        spy_fwd = prices["SPY"].iloc[end_idx] / prices["SPY"].iloc[idx] - 1
        if True:  # tilt return
            xlk_fwd = prices["XLK"].iloc[end_idx] / prices["XLK"].iloc[idx] - 1
            xly_fwd = prices["XLY"].iloc[end_idx] / prices["XLY"].iloc[idx] - 1
            xlb_fwd = prices["XLB"].iloc[end_idx] / prices["XLB"].iloc[idx] - 1
            tilt_fwd = 0.4 * xlk_fwd + 0.3 * xly_fwd + 0.3 * xlb_fwd
        signal_returns.append({
            "signal_date": str(sd.date()),
            "horizon": horizon_name,
            "spy_return": spy_fwd,
            "tilt_return": tilt_fwd,
            "tilt_excess": tilt_fwd - spy_fwd,
        })

sr_df = pd.DataFrame(signal_returns)
for h in ["1m", "3m", "12m"]:
    sub = sr_df[sr_df["horizon"] == h]
    print(f"\n{h} horizon ({len(sub)} events):")
    print(f"  SPY avg: {sub['spy_return'].mean():.2%}, WR: {(sub['spy_return'] > 0).mean():.0%}")
    print(f"  Tilt avg: {sub['tilt_return'].mean():.2%}, WR: {(sub['tilt_return'] > 0).mean():.0%}")
    print(f"  Excess avg: {sub['tilt_excess'].mean():.2%}")

# ── 9. Save results ──────────────────────────────────────────────────────
print("\n=== Saving results ===")

# JSON results
results = {
    "test_period": f"{prices.index[0].date()} to {prices.index[-1].date()}",
    "n_trading_days": len(prices),
    "strategies": {},
    "permutation_test": {
        "n_permutations": n_perm,
        "actual_sharpe": round(actual_sharpe, 4),
        "perm_mean_sharpe": round(np.mean(perm_sharpes), 4),
        "perm_std_sharpe": round(np.std(perm_sharpes), 4),
        "p_value": round(p_value, 4),
        "significant_5pct": bool(p_value < 0.05),
    },
    "regime_test": {
        "strategy_bull_sharpe": round(bull_sharpe, 4),
        "strategy_bear_sharpe": round(bear_sharpe, 4),
        "spy_bull_sharpe": round(spy_bull_sharpe, 4),
        "spy_bear_sharpe": round(spy_bear_sharpe, 4),
        "regime_sharpe_gap": round(sharpe_gap, 4),
    },
    "lag_sensitivity": {
        "t0_sharpe": round(res_t0["sharpe"], 4),
        "t1_sharpe": round(res_tilt["sharpe"], 4),
        "delta": round(res_t0["sharpe"] - res_tilt["sharpe"], 4),
    },
    "per_signal_analysis": {},
}

for name, res in [("pure_spy", res_spy), ("crisis_exit_spy", res_ce_spy), ("crisis_exit_sector_tilt", res_tilt)]:
    results["strategies"][name] = {
        "ann_return": round(res["ann_return"], 4),
        "ann_vol": round(res["ann_vol"], 4),
        "sharpe": round(res["sharpe"], 4),
        "sortino": round(res["sortino"], 4),
        "max_drawdown": round(res["max_dd"], 4),
        "calmar": round(res["calmar"], 4),
        "total_return": round(res["total_return"], 4),
        "n_signals": res["n_signals"],
        "signal_dates": res["signal_dates"],
    }

for h in ["1m", "3m", "12m"]:
    sub = sr_df[sr_df["horizon"] == h]
    results["per_signal_analysis"][h] = {
        "n_events": len(sub),
        "spy_avg_return": round(sub["spy_return"].mean(), 4),
        "spy_win_rate": round((sub["spy_return"] > 0).mean(), 4),
        "tilt_avg_return": round(sub["tilt_return"].mean(), 4),
        "tilt_win_rate": round((sub["tilt_return"] > 0).mean(), 4),
        "excess_avg": round(sub["tilt_excess"].mean(), 4),
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

with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2, cls=NumpyEncoder)

# Equity curve parquet
eq_df = pd.DataFrame({
    "date": prices.index,
    "pure_spy": res_spy["equity"].values,
    "crisis_exit_spy": res_ce_spy["equity"].values,
    "crisis_exit_sector_tilt": res_tilt["equity"].values,
})
eq_df.to_parquet(os.path.join(OUT_DIR, "equity_curve.parquet"), index=False)

# Summary report
spy_s = results["strategies"]["pure_spy"]
ce_s = results["strategies"]["crisis_exit_spy"]
tilt_s = results["strategies"]["crisis_exit_sector_tilt"]

report = f"""CRISIS EXIT + SECTOR RECOVERY TILT BACKTEST
=============================================
Period: {results['test_period']} ({results['n_trading_days']} trading days)

STRATEGY COMPARISON
-------------------
                        Pure SPY    Crisis+SPY   Crisis+Tilt
Annual Return:          {spy_s['ann_return']:>8.1%}      {ce_s['ann_return']:>8.1%}      {tilt_s['ann_return']:>8.1%}
Annual Vol:             {spy_s['ann_vol']:>8.1%}      {ce_s['ann_vol']:>8.1%}      {tilt_s['ann_vol']:>8.1%}
Sharpe:                 {spy_s['sharpe']:>8.3f}      {ce_s['sharpe']:>8.3f}      {tilt_s['sharpe']:>8.3f}
Sortino:                {spy_s['sortino']:>8.3f}      {ce_s['sortino']:>8.3f}      {tilt_s['sortino']:>8.3f}
Max Drawdown:           {spy_s['max_drawdown']:>8.1%}     {ce_s['max_drawdown']:>8.1%}     {tilt_s['max_drawdown']:>8.1%}
Calmar:                 {spy_s['calmar']:>8.3f}      {ce_s['calmar']:>8.3f}      {tilt_s['calmar']:>8.3f}
Total Return:           {spy_s['total_return']:>8.1%}     {ce_s['total_return']:>8.1%}     {tilt_s['total_return']:>8.1%}
Signals Fired:          {spy_s['n_signals']:>8d}      {ce_s['n_signals']:>8d}      {tilt_s['n_signals']:>8d}

SIGNAL QUALITY (per-event forward returns)
------------------------------------------
"""
for h in ["1m", "3m", "12m"]:
    psa = results["per_signal_analysis"][h]
    report += f"{h}: SPY avg {psa['spy_avg_return']:+.2%} (WR {psa['spy_win_rate']:.0%}), "
    report += f"Tilt avg {psa['tilt_avg_return']:+.2%} (WR {psa['tilt_win_rate']:.0%}), "
    report += f"Excess {psa['excess_avg']:+.2%}\n"

perm = results["permutation_test"]
report += f"""
PERMUTATION TEST (200 random-timing shuffles)
----------------------------------------------
Actual Sharpe:    {perm['actual_sharpe']:.4f}
Random mean:      {perm['perm_mean_sharpe']:.4f} +/- {perm['perm_std_sharpe']:.4f}
p-value:          {perm['p_value']:.4f}
Significant (5%): {'YES' if perm['significant_5pct'] else 'NO'}

REGIME TEST (bull = SPY > 200MA, bear = SPY < 200MA)
------------------------------------------------------
Strategy bull Sharpe: {results['regime_test']['strategy_bull_sharpe']:.4f}
Strategy bear Sharpe: {results['regime_test']['strategy_bear_sharpe']:.4f}
SPY bull Sharpe:      {results['regime_test']['spy_bull_sharpe']:.4f}
SPY bear Sharpe:      {results['regime_test']['spy_bear_sharpe']:.4f}
Regime gap:           {results['regime_test']['regime_sharpe_gap']:.4f}

LAG SENSITIVITY
----------------
T-0 (same-day) Sharpe: {results['lag_sensitivity']['t0_sharpe']:.4f}
T-1 (next-day) Sharpe: {results['lag_sensitivity']['t1_sharpe']:.4f}
Delta (T0 - T1):       {results['lag_sensitivity']['delta']:.4f}

SIGNAL DATES (T-1)
-------------------
"""
for sd in tilt_s["signal_dates"]:
    report += f"  {sd}\n"

report += f"""
INTERPRETATION
--------------
"""

# Auto-generate interpretation
tilt_excess_3m = results["per_signal_analysis"]["3m"]["excess_avg"]
if tilt_s["sharpe"] > spy_s["sharpe"] + 0.05:
    report += "The sector tilt strategy outperforms pure SPY on a risk-adjusted basis.\n"
elif tilt_s["sharpe"] < spy_s["sharpe"] - 0.05:
    report += "The sector tilt strategy underperforms pure SPY on a risk-adjusted basis.\n"
else:
    report += "The sector tilt strategy performs roughly in line with pure SPY on a risk-adjusted basis.\n"

if perm["p_value"] < 0.05:
    report += "The timing of signals matters (permutation test significant at 5%).\n"
else:
    report += "Signal timing is NOT statistically significant vs random timing.\n"

if results["regime_test"]["regime_sharpe_gap"] > 0.5:
    report += "WARNING: Large regime gap suggests strategy is regime-dependent, not robust.\n"
else:
    report += "Regime gap is acceptable - strategy works across bull and bear markets.\n"

if abs(results["lag_sensitivity"]["delta"]) > 0.05:
    report += f"Lag matters: {'T-0 is better' if results['lag_sensitivity']['delta'] > 0 else 'T-1 is better'} (delta={results['lag_sensitivity']['delta']:.4f}).\n"
else:
    report += "T-0 vs T-1 makes little difference - signal is not front-run sensitive.\n"

if tilt_excess_3m > 0.005:
    report += f"Sector tilt adds value: +{tilt_excess_3m:.2%} excess over SPY at 3m horizon per event.\n"
else:
    report += f"Sector tilt adds minimal/negative value: {tilt_excess_3m:+.2%} at 3m per event. Consider just using SPY.\n"

with open(os.path.join(OUT_DIR, "summary_report.txt"), "w") as f:
    f.write(report)

print(report)
print(f"\nFiles saved to {OUT_DIR}/")
print("Done.")
