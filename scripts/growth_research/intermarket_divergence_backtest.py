#!/usr/bin/env python3
"""
Intermarket Divergence Sector Dip-Buying Backtest
Tests whether cross-asset divergences predict sector ETF recovery opportunities.
"""

import json
import os
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────
SECTOR_ETFS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
INTERMARKET = ["SPY", "TLT", "GLD", "UUP", "HYG", "IEF"]
ALL_TICKERS = SECTOR_ETFS + INTERMARKET
START = "2020-01-01"
END = datetime.now().strftime("%Y-%m-%d")
COST_RT = 0.0010  # 0.10% round-trip
HOLD_DAYS = 5
TP_PCT = 0.03
SL_PCT = -0.05
RSI_THRESH = 35
DEFENSIVE_SECTORS = ["XLU", "XLRE", "XLP"]
N_PERMUTATIONS = 1000

# ── Data Download ──────────────────────────────────────────────────────────
print(f"Downloading {len(ALL_TICKERS)} tickers from {START} to {END}...")
data = yf.download(ALL_TICKERS, start=START, end=END, auto_adjust=True, progress=False)

# Handle multi-level columns from yfinance
if isinstance(data.columns, pd.MultiIndex):
    close = data["Close"]
else:
    close = data

close = close.dropna(how="all")
print(f"Data shape: {close.shape}, date range: {close.index[0].date()} to {close.index[-1].date()}")

# ── Indicator Helpers ──────────────────────────────────────────────────────
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_returns(series, days):
    return series.pct_change(days)

# Pre-compute indicators
rsi = pd.DataFrame({t: calc_rsi(close[t]) for t in SECTOR_ETFS})
ret5 = pd.DataFrame({t: calc_returns(close[t], 5) for t in ALL_TICKERS})
ret10 = pd.DataFrame({t: calc_returns(close[t], 10) for t in ALL_TICKERS})

# Z-scores for intermarket momentum (Signal G)
def rolling_zscore(series, window=60):
    return (series - series.rolling(window).mean()) / series.rolling(window).std()

# ── Signal Generators ─────────────────────────────────────────────────────
def signal_A_bond_equity_divergence(date_idx):
    """Bond-Equity Divergence: TLT up >1% over 5d while sector down >3% AND RSI < 35."""
    trades = []
    if ret5.loc[date_idx, "TLT"] > 0.01:
        for etf in SECTOR_ETFS:
            if ret5.loc[date_idx, etf] < -0.03 and rsi.loc[date_idx, etf] < RSI_THRESH:
                trades.append(etf)
    return trades

def signal_B_credit_tightening(date_idx):
    """Credit Spread Tightening: HYG outperforming IEF over 5d AND sector RSI < 35."""
    trades = []
    hyg_vs_ief = ret5.loc[date_idx, "HYG"] - ret5.loc[date_idx, "IEF"]
    if hyg_vs_ief > 0:
        for etf in SECTOR_ETFS:
            if rsi.loc[date_idx, etf] < RSI_THRESH:
                trades.append(etf)
    return trades

def signal_C_dollar_weakness(date_idx):
    """Dollar Weakness + Sector Dip: UUP down >1% over 10d AND sector RSI < 35."""
    trades = []
    if ret10.loc[date_idx, "UUP"] < -0.01:
        for etf in SECTOR_ETFS:
            if rsi.loc[date_idx, etf] < RSI_THRESH:
                trades.append(etf)
    return trades

def signal_D_gold_divergence(date_idx):
    """Gold Divergence: GLD down >2% over 5d AND sector RSI < 35."""
    trades = []
    if ret5.loc[date_idx, "GLD"] < -0.02:
        for etf in SECTOR_ETFS:
            if rsi.loc[date_idx, etf] < RSI_THRESH:
                trades.append(etf)
    return trades

def signal_E_triple_confirmation(date_idx):
    """Triple Confirmation Risk-On: TLT down + HYG up + UUP down over 5d, buy most oversold sector."""
    if ret5.loc[date_idx, "TLT"] < 0 and ret5.loc[date_idx, "HYG"] > 0 and ret5.loc[date_idx, "UUP"] < 0:
        oversold = [(etf, rsi.loc[date_idx, etf]) for etf in SECTOR_ETFS
                     if rsi.loc[date_idx, etf] < RSI_THRESH]
        if oversold:
            oversold.sort(key=lambda x: x[1])
            return [oversold[0][0]]  # Most oversold only
    return []

def signal_F_bond_yield_spike(date_idx):
    """Bond Yield Spike + Defensive Sector Dip: IEF down >1% over 5d AND defensive RSI < 35."""
    trades = []
    if ret5.loc[date_idx, "IEF"] < -0.01:
        for etf in DEFENSIVE_SECTORS:
            if rsi.loc[date_idx, etf] < RSI_THRESH:
                trades.append(etf)
    return trades

def signal_G_intermarket_momentum(date_idx):
    """Intermarket Momentum Score: z-score composite > 1.5 AND sector RSI < 35."""
    trades = []
    # Inverted: negative TLT/GLD/UUP returns = risk-on
    tlt_z = rolling_zscore(ret5["TLT"]).loc[date_idx]
    gld_z = rolling_zscore(ret5["GLD"]).loc[date_idx]
    uup_z = rolling_zscore(ret5["UUP"]).loc[date_idx]
    if pd.notna(tlt_z) and pd.notna(gld_z) and pd.notna(uup_z):
        score = (-tlt_z) + (-gld_z) + (-uup_z)  # Higher = more risk-on
        if score > 1.5:
            for etf in SECTOR_ETFS:
                if rsi.loc[date_idx, etf] < RSI_THRESH:
                    trades.append(etf)
    return trades

def signal_H_flight_to_safety_reversal(date_idx, idx_pos):
    """Flight-to-Safety Reversal: TLT up >2% + GLD up >1% + SPY down >2% over 5d,
    then buy most oversold sector on first SPY up day."""
    if idx_pos < 1:
        return []
    prev_idx = close.index[idx_pos - 1]
    # Check panic conditions on previous day
    tlt_ok = ret5.loc[prev_idx, "TLT"] > 0.02
    gld_ok = ret5.loc[prev_idx, "GLD"] > 0.01
    spy_down = ret5.loc[prev_idx, "SPY"] < -0.02
    # Check SPY was down yesterday but up today (panic fading)
    spy_today_up = close.loc[date_idx, "SPY"] > close.loc[prev_idx, "SPY"]

    if tlt_ok and gld_ok and spy_down and spy_today_up:
        oversold = [(etf, rsi.loc[date_idx, etf]) for etf in SECTOR_ETFS
                     if rsi.loc[date_idx, etf] < RSI_THRESH]
        if oversold:
            oversold.sort(key=lambda x: x[1])
            return [oversold[0][0]]
    return []

def signal_baseline(date_idx):
    """Baseline: RSI < 35 only, no intermarket filter."""
    return [etf for etf in SECTOR_ETFS if rsi.loc[date_idx, etf] < RSI_THRESH]


# ── Backtest Engine ────────────────────────────────────────────────────────
def run_backtest(signal_func, name, use_idx_pos=False):
    """Run backtest for a signal function. Returns list of trade dicts."""
    trades = []
    dates = close.index.tolist()
    # Skip first 60 days for indicator warmup
    start_idx = 60

    for i in range(start_idx, len(dates) - HOLD_DAYS):
        date = dates[i]
        # Skip if indicators are NaN
        if pd.isna(ret5.loc[date, "TLT"]):
            continue

        if use_idx_pos:
            signals = signal_func(date, i)
        else:
            signals = signal_func(date)

        for etf in signals:
            entry_price = close.loc[date, etf]
            if pd.isna(entry_price) or entry_price <= 0:
                continue

            # Simulate hold with TP/SL
            exit_price = None
            exit_day = None
            for j in range(1, HOLD_DAYS + 1):
                if i + j >= len(dates):
                    break
                future_date = dates[i + j]
                price = close.loc[future_date, etf]
                ret = (price - entry_price) / entry_price
                if ret >= TP_PCT:
                    exit_price = entry_price * (1 + TP_PCT)
                    exit_day = j
                    break
                elif ret <= SL_PCT:
                    exit_price = entry_price * (1 + SL_PCT)
                    exit_day = j
                    break

            if exit_price is None:
                # Exit at end of hold period
                exit_date = dates[min(i + HOLD_DAYS, len(dates) - 1)]
                exit_price = close.loc[exit_date, etf]
                exit_day = HOLD_DAYS

            gross_ret = (exit_price - entry_price) / entry_price
            net_ret = gross_ret - COST_RT

            trades.append({
                "date": date.strftime("%Y-%m-%d"),
                "etf": etf,
                "entry": float(entry_price),
                "exit": float(exit_price),
                "hold_days": exit_day,
                "gross_ret": float(gross_ret),
                "net_ret": float(net_ret),
            })

    return trades


def calc_metrics(trades, name):
    """Calculate performance metrics from trade list."""
    if not trades:
        return {
            "signal": name, "trades": 0, "WR": 0, "avg_ret": 0,
            "sharpe": 0, "sortino": 0, "PF": 0, "max_dd": 0,
            "regime_gap": 0, "perm_p": 1.0, "day_conc": 0, "PASS": False
        }

    rets = np.array([t["net_ret"] for t in trades])
    dates = [t["date"] for t in trades]

    n = len(rets)
    wins = np.sum(rets > 0)
    wr = wins / n
    avg_ret = np.mean(rets)

    # Sharpe (annualized, assuming ~50 trades/yr avg hold 5d)
    if np.std(rets) > 0:
        sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(252 / HOLD_DAYS)
    else:
        sharpe = 0.0

    # Sortino
    downside = rets[rets < 0]
    if len(downside) > 0 and np.std(downside) > 0:
        sortino = (np.mean(rets) / np.std(downside)) * np.sqrt(252 / HOLD_DAYS)
    else:
        sortino = 0.0

    # Profit Factor
    gross_wins = np.sum(rets[rets > 0])
    gross_losses = abs(np.sum(rets[rets < 0]))
    pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    # Max Drawdown (cumulative equity curve)
    cum = np.cumsum(rets)
    running_max = np.maximum.accumulate(cum)
    dd = cum - running_max
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0

    # Regime gap: split by SPY green/red days
    regime_sharpes = {"green": [], "red": []}
    for t in trades:
        d = t["date"]
        if d in close.index.strftime("%Y-%m-%d").tolist():
            didx = pd.Timestamp(d)
            if didx in close.index:
                prev_idx = close.index.get_loc(didx)
                if prev_idx > 0:
                    spy_ret = (close.iloc[prev_idx]["SPY"] - close.iloc[prev_idx - 1]["SPY"]) / close.iloc[prev_idx - 1]["SPY"]
                    if spy_ret >= 0:
                        regime_sharpes["green"].append(t["net_ret"])
                    else:
                        regime_sharpes["red"].append(t["net_ret"])

    g_rets = np.array(regime_sharpes["green"]) if regime_sharpes["green"] else np.array([0.0])
    r_rets = np.array(regime_sharpes["red"]) if regime_sharpes["red"] else np.array([0.0])
    g_sharpe = (np.mean(g_rets) / np.std(g_rets) * np.sqrt(252/HOLD_DAYS)) if np.std(g_rets) > 0 else 0.0
    r_sharpe = (np.mean(r_rets) / np.std(r_rets) * np.sqrt(252/HOLD_DAYS)) if np.std(r_rets) > 0 else 0.0
    max_regime = max(abs(g_sharpe), abs(r_sharpe))
    regime_gap = abs(g_sharpe - r_sharpe) / max_regime if max_regime > 0 else 0.0

    # Permutation test
    observed_mean = np.mean(rets)
    perm_count = 0
    for _ in range(N_PERMUTATIONS):
        shuffled = np.random.permutation(rets)
        if np.mean(shuffled[:n]) >= observed_mean:
            perm_count += 1
    perm_p = perm_count / N_PERMUTATIONS

    # Day concentration
    date_counts = pd.Series(dates).value_counts()
    day_conc = date_counts.max() / n if n > 0 else 0.0

    # PASS/FAIL
    passed = (
        n >= 20
        and wr > 0.50
        and sharpe > 0.5
        and pf > 1.2
        and regime_gap < 0.50
        and perm_p < 0.05
        and day_conc < 0.70
    )

    return {
        "signal": name,
        "trades": n,
        "WR": round(wr, 4),
        "avg_ret": round(avg_ret, 6),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "PF": round(pf, 3),
        "max_dd": round(max_dd, 4),
        "regime_gap": round(regime_gap, 3),
        "perm_p": round(perm_p, 4),
        "day_conc": round(day_conc, 4),
        "PASS": bool(passed)
    }


# ── Run All Signals ───────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("INTERMARKET DIVERGENCE SECTOR DIP-BUYING BACKTEST")
print(f"Period: {START} to {END} | Cost: {COST_RT*100:.2f}% RT | Hold: {HOLD_DAYS}d | TP: {TP_PCT*100:.0f}% / SL: {SL_PCT*100:.0f}%")
print("=" * 90)

signals = {
    "A: Bond-Equity Divergence": (signal_A_bond_equity_divergence, False),
    "B: Credit Spread Tightening": (signal_B_credit_tightening, False),
    "C: Dollar Weakness + Dip": (signal_C_dollar_weakness, False),
    "D: Gold Divergence": (signal_D_gold_divergence, False),
    "E: Triple Confirmation": (signal_E_triple_confirmation, False),
    "F: Bond Yield Spike + Def": (signal_F_bond_yield_spike, False),
    "G: Intermarket Momentum": (signal_G_intermarket_momentum, False),
    "H: Flight-Safety Reversal": (signal_H_flight_to_safety_reversal, True),
    "Baseline: RSI<35 Only": (signal_baseline, False),
}

results = []
for name, (func, use_idx) in signals.items():
    print(f"\nRunning {name}...")
    trades = run_backtest(func, name, use_idx_pos=use_idx)
    metrics = calc_metrics(trades, name)
    results.append(metrics)
    print(f"  Trades: {metrics['trades']:>5} | WR: {metrics['WR']:.1%} | Avg: {metrics['avg_ret']:+.4f} | "
          f"Sharpe: {metrics['sharpe']:>6.2f} | Sortino: {metrics['sortino']:>6.2f} | "
          f"PF: {metrics['PF']:>5.2f} | MaxDD: {metrics['max_dd']:+.4f} | "
          f"RegGap: {metrics['regime_gap']:.3f} | Perm-p: {metrics['perm_p']:.4f} | "
          f"DayConc: {metrics['day_conc']:.3f} | {'PASS' if metrics['PASS'] else 'FAIL'}")

# ── Summary Table ──────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print(f"{'Signal':<32} {'N':>5} {'WR':>6} {'AvgRet':>8} {'Sharpe':>7} {'Sortino':>8} "
      f"{'PF':>6} {'MaxDD':>7} {'RGap':>6} {'Perm-p':>7} {'DConc':>6} {'Result':>6}")
print("-" * 90)

for r in results:
    flag = "PASS" if r["PASS"] else "FAIL"
    print(f"{r['signal']:<32} {r['trades']:>5} {r['WR']:>5.1%} {r['avg_ret']:>+8.4f} "
          f"{r['sharpe']:>7.2f} {r['sortino']:>8.2f} {r['PF']:>6.2f} {r['max_dd']:>+7.4f} "
          f"{r['regime_gap']:>6.3f} {r['perm_p']:>7.4f} {r['day_conc']:>6.3f} {flag:>6}")

print("-" * 90)

# Highlight winners
passing = [r for r in results if r["PASS"]]
if passing:
    print(f"\n✓ {len(passing)} signal(s) PASSED all gates:")
    for r in passing:
        print(f"  → {r['signal']}: Sharpe {r['sharpe']:.2f}, WR {r['WR']:.1%}, PF {r['PF']:.2f}")
else:
    print("\n✗ No signals passed all gates.")

# Compare vs baseline
baseline = [r for r in results if "Baseline" in r["signal"]][0]
print(f"\nBaseline (RSI<35 only): {baseline['trades']} trades, Sharpe {baseline['sharpe']:.2f}, WR {baseline['WR']:.1%}")
above_baseline = [r for r in results if r["sharpe"] > baseline["sharpe"] and "Baseline" not in r["signal"]]
if above_baseline:
    print(f"Signals beating baseline Sharpe:")
    for r in above_baseline:
        delta = r["sharpe"] - baseline["sharpe"]
        print(f"  → {r['signal']}: Sharpe {r['sharpe']:.2f} (+{delta:.2f} vs baseline)")
else:
    print("No intermarket signals beat the baseline.")

# ── Save Results ───────────────────────────────────────────────────────────
results_dir = Path(__file__).parent / "results"
results_dir.mkdir(exist_ok=True)
output = {
    "timestamp": datetime.now().isoformat(),
    "config": {
        "start": START, "end": END, "cost_rt": COST_RT,
        "hold_days": HOLD_DAYS, "tp_pct": TP_PCT, "sl_pct": SL_PCT,
        "rsi_threshold": RSI_THRESH, "n_permutations": N_PERMUTATIONS,
    },
    "results": results,
}
out_path = results_dir / "intermarket_divergence_results.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {out_path}")
