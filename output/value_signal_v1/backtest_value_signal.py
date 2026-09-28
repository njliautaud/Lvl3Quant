#!/usr/bin/env python3
"""
Value Signal v1 — Insider Buying Proxy Backtest (optimized)
===========================================================
All quality gates built into the script per HC #705.
"""

import json, sys, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
np.random.seed(42)

TICKERS = [
    "AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","JPM","V","UNH",
    "JNJ","WMT","PG","HD","MA","BAC","XOM","DIS","NFLX","AMD",
    "CRM","AVGO","COST","ABBV","LLY","PFE","KO","PEP","MRK","CSCO",
]
START, END = "2015-01-01", "2026-07-01"
HOLD_PERIODS = [5, 10, 20]
N_PERMS = 200
OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/value_signal_v1")

# ── Download ─────────────────────────────────────────────────────────────
print("Downloading price data (batch) …", flush=True)
raw = yf.download(TICKERS, start=START, end=END, auto_adjust=True, threads=True)
close_df = raw["Close"].copy()
close_df = close_df.dropna(how="all")
print(f"Close matrix: {close_df.shape[0]} days × {close_df.shape[1]} tickers", flush=True)

# Also grab .info for fundamental filters (cached)
print("Fetching ticker info …", flush=True)
info_cache = {}
for t in TICKERS:
    try:
        info_cache[t] = yf.Ticker(t).info
    except Exception:
        info_cache[t] = {}
    print(f"  {t} info OK", flush=True)

# Forward returns (vectorized)
fwd_ret = {}
for hp in HOLD_PERIODS:
    fwd_ret[hp] = close_df.shift(-hp) / close_df - 1

# SPY for regime
print("Downloading SPY …", flush=True)
spy = yf.download("SPY", start=START, end=END, auto_adjust=True)
spy_close = spy["Close"]
if isinstance(spy_close, pd.DataFrame):
    spy_close = spy_close.iloc[:, 0]  # flatten multi-level
spy_monthly = spy_close.resample("ME").last().pct_change()
regime_map = {}
for dt, ret in spy_monthly.items():
    val = float(ret) if not np.isnan(ret) else 0
    regime_map[(dt.year, dt.month)] = "up" if val > 0 else "down"

# ── Signal generators (return list of (date_idx, ticker_col) tuples) ────

def make_signals():
    """Generate all signals as a DataFrame with columns: date, ticker, signal."""
    records = []

    for ti, t in enumerate(close_df.columns):
        series = close_df[t].dropna()
        if len(series) < 252:
            continue
        info = info_cache.get(t, {})

        # --- high_yield_value: price < 0.7 * 5yr SMA ---
        sma_5y = series.rolling(252 * 5, min_periods=252 * 3).mean()
        mask = series < 0.7 * sma_5y
        for d in mask[mask].index:
            records.append((d, t, "high_yield_value"))

        # --- 52wk_low_value: within 5% of 52wk low, positive earnings ---
        eps = info.get("trailingEps", 1)  # default positive if unknown
        if eps is not None and eps > 0:
            low_52 = series.rolling(252, min_periods=200).min()
            mask = series <= low_52 * 1.05
            cutoff = series.index[252] if len(series) > 252 else series.index[-1]
            for d in mask[mask].index:
                if d >= cutoff:
                    records.append((d, t, "52wk_low_value"))

        # --- post_selloff_quality: down 20%+ from 52wk high, positive ROE ---
        roe = info.get("returnOnEquity", 0.1)  # default positive if unknown
        if roe is not None and roe > 0:
            high_52 = series.rolling(252, min_periods=200).max()
            dd = series / high_52 - 1
            mask = dd <= -0.20
            cutoff = series.index[252] if len(series) > 252 else series.index[-1]
            for d in mask[mask].index:
                if d >= cutoff:
                    records.append((d, t, "post_selloff_quality"))

        # --- buyback_momentum: near 20d low but 60d return positive ---
        if len(series) > 60:
            low_20 = series.rolling(20, min_periods=15).min()
            ret_60 = series / series.shift(60) - 1
            mask = (series <= low_20 * 1.03) & (ret_60 > 0)
            cutoff = series.index[60]
            for d in mask[mask].index:
                if d >= cutoff:
                    records.append((d, t, "buyback_momentum"))

    return pd.DataFrame(records, columns=["date", "ticker", "signal"])

print("Generating signals …", flush=True)
sig_df = make_signals()
sig_df["date"] = pd.to_datetime(sig_df["date"])
sig_df = sig_df.drop_duplicates(subset=["date", "ticker", "signal"])
print(f"Total signals: {len(sig_df)}", flush=True)
for s in sig_df["signal"].unique():
    print(f"  {s}: {(sig_df['signal']==s).sum()}", flush=True)

if len(sig_df) == 0:
    print("ERROR: No signals. Aborting.")
    sys.exit(1)

# ── Attach forward returns (vectorized via merge) ────────────────────────
print("Attaching forward returns …", flush=True)
# Convert close_df to long format for merge
for hp in HOLD_PERIODS:
    col = f"fwd_ret_{hp}d"
    fr_long = fwd_ret[hp].stack().reset_index()
    fr_long.columns = ["date", "ticker", col]
    fr_long["date"] = pd.to_datetime(fr_long["date"])
    sig_df = sig_df.merge(fr_long, on=["date", "ticker"], how="left")

# Drop rows with no valid returns at all
ret_cols = [f"fwd_ret_{hp}d" for hp in HOLD_PERIODS]
sig_df = sig_df.dropna(subset=ret_cols, how="all")
print(f"Signals with returns: {len(sig_df)}", flush=True)

# Regime
sig_df["regime"] = sig_df["date"].apply(lambda d: regime_map.get((d.year, d.month), "unknown"))

# ── Winsorize (G4) ──────────────────────────────────────────────────────
for col in ret_cols:
    valid = sig_df[col].dropna()
    if len(valid) > 10:
        lo, hi = valid.quantile(0.01), valid.quantile(0.99)
        sig_df[col] = sig_df[col].clip(lo, hi)

# ── Helper functions ─────────────────────────────────────────────────────
def sharpe(r):
    if len(r) < 5 or r.std() == 0: return 0.0
    return float(r.mean() / r.std() * np.sqrt(252))

def sortino(r):
    if len(r) < 5: return 0.0
    down = r[r < 0]
    if len(down) == 0 or down.std() == 0: return 10.0
    return float(r.mean() / down.std() * np.sqrt(252))

def profit_factor(r):
    g = r[r > 0].sum()
    l = abs(r[r < 0].sum())
    return float(g / l) if l > 0 else 10.0

# ── Pre-compute fwd_ret matrices as numpy for fast permutation ───────────
# date_index mapping
date_to_idx = {d: i for i, d in enumerate(close_df.index)}
ticker_to_col = {t: i for i, t in enumerate(close_df.columns)}
fwd_ret_np = {}
for hp in HOLD_PERIODS:
    fwd_ret_np[hp] = fwd_ret[hp].values  # shape (n_dates, n_tickers)

n_dates_total = len(close_df.index)

# ── Evaluate one signal/hold combo ───────────────────────────────────────
def evaluate(subset, hp, label):
    col = f"fwd_ret_{hp}d"
    s = subset[subset[col].notna()].copy()
    if len(s) < 30:
        print(f"  {label}: only {len(s)} trades, skip", flush=True)
        return None

    rets = s[col].values
    n = len(rets)
    mean_ret = rets.mean()
    wr = (rets > 0).mean()
    pf = profit_factor(pd.Series(rets))
    sh = sharpe(pd.Series(rets))
    so = sortino(pd.Series(rets))

    gates = {}

    # G1: Permutation test — random DATE entry
    # For each perm: for each trade, pick random date, same ticker, get fwd return
    ticker_indices = s["ticker"].map(ticker_to_col).values
    date_indices = s["date"].map(date_to_idx).values
    valid_mask = ~np.isnan(ticker_indices) & ~np.isnan(date_indices)
    ticker_idx_arr = ticker_indices[valid_mask].astype(int)

    fr_matrix = fwd_ret_np[hp]
    perm_means = np.empty(N_PERMS)
    for p in range(N_PERMS):
        rand_dates = np.random.randint(0, max(1, n_dates_total - hp - 1), size=len(ticker_idx_arr))
        perm_rets = fr_matrix[rand_dates, ticker_idx_arr]
        valid = perm_rets[~np.isnan(perm_rets)]
        perm_means[p] = valid.mean() if len(valid) > 0 else 0.0

    p_value = float((np.sum(perm_means >= mean_ret) + 1) / (N_PERMS + 1))
    gates["G1_perm_p"] = round(p_value, 4)
    gates["G1_pass"] = p_value < 0.05

    # G2: Regime
    up_r = s[s["regime"] == "up"][col]
    dn_r = s[s["regime"] == "down"][col]
    sh_up = sharpe(up_r)
    sh_dn = sharpe(dn_r)
    mx = max(abs(sh_up), abs(sh_dn))
    gap = abs(sh_up - sh_dn) / mx if mx > 0 else 0
    gates["G2_sharpe_up"] = round(sh_up, 3)
    gates["G2_sharpe_down"] = round(sh_dn, 3)
    gates["G2_regime_gap"] = round(gap, 3)
    gates["G2_pass"] = gap < 0.50

    # G3: Sub-period
    mid = n // 2
    sh1 = sharpe(pd.Series(rets[:mid]))
    sh2 = sharpe(pd.Series(rets[mid:]))
    gates["G3_sharpe_first"] = round(sh1, 3)
    gates["G3_sharpe_second"] = round(sh2, 3)
    gates["G3_pass"] = sh1 > 0 and sh2 > 0

    # G5: Ticker concentration
    tc = s["ticker"].value_counts(normalize=True)
    gates["G5_max_conc"] = round(float(tc.max()), 3)
    gates["G5_top_ticker"] = tc.index[0]
    gates["G5_pass"] = float(tc.max()) < 0.25

    # G6: WR > 50%, PF > 1.0
    gates["G6_wr"] = round(float(wr), 4)
    gates["G6_pf"] = round(pf, 3)
    gates["G6_pass"] = wr > 0.50 and pf > 1.0

    all_pass = all(v for k, v in gates.items() if k.endswith("_pass"))
    failed = [k.split("_")[0] for k, v in gates.items() if k.endswith("_pass") and not v]

    print(f"  {label}: n={n} WR={wr:.1%} PF={pf:.2f} Sh={sh:.2f} So={so:.2f} "
          f"perm_p={p_value:.3f} regime_gap={gap:.2f} → {'PASS' if all_pass else 'FAIL'}"
          f"{' ('+','.join(failed)+')' if failed else ''}", flush=True)

    return {
        "signal": label.split("__")[0],
        "hold_days": hp,
        "n_trades": int(n),
        "mean_return_pct": round(float(mean_ret * 100), 4),
        "win_rate": round(float(wr), 4),
        "profit_factor": round(pf, 3),
        "sharpe": round(sh, 3),
        "sortino": round(so, 3),
        "gates": gates,
        "ALL_GATES_PASS": all_pass,
    }

# ── Run evaluations ─────────────────────────────────────────────────────
results = {}

print("\n── Per-signal results ──", flush=True)
for sig_name in sorted(sig_df["signal"].unique()):
    for hp in HOLD_PERIODS:
        key = f"{sig_name}__{hp}d"
        subset = sig_df[sig_df["signal"] == sig_name]
        r = evaluate(subset, hp, key)
        if r:
            results[key] = r

print("\n── Composite (any signal fires) ──", flush=True)
composite = sig_df.drop_duplicates(subset=["date", "ticker"])
for hp in HOLD_PERIODS:
    key = f"composite__{hp}d"
    r = evaluate(composite, hp, key)
    if r:
        results[key] = r

# ── Annual breakdown ─────────────────────────────────────────────────────
print("\n── Annual breakdown (composite 20d) ──", flush=True)
comp20 = composite[composite["fwd_ret_20d"].notna()].copy()
comp20["year"] = comp20["date"].dt.year
for yr in sorted(comp20["year"].unique()):
    yr_r = comp20[comp20["year"] == yr]["fwd_ret_20d"]
    if len(yr_r) >= 5:
        print(f"  {yr}: n={len(yr_r):4d}  mean={yr_r.mean()*100:+.2f}%  "
              f"WR={(yr_r>0).mean():.1%}  Sharpe={sharpe(yr_r):.2f}", flush=True)

# ── Save report ──────────────────────────────────────────────────────────
report = {
    "strategy": "Value Signal v1 — Insider Buying Proxy",
    "generated": datetime.now().isoformat(),
    "universe": TICKERS,
    "date_range": f"{START} to {END}",
    "n_tickers": int(close_df.shape[1]),
    "total_signals": int(len(sig_df)),
    "signals_by_type": {k: int(v) for k, v in sig_df["signal"].value_counts().items()},
    "hold_periods": HOLD_PERIODS,
    "n_permutations": N_PERMS,
    "quality_gates": {
        "G1": "Permutation test (200 random-date perms, p<0.05)",
        "G2": "Regime-agnostic (|Sharpe_up-Sharpe_down|/max < 0.50)",
        "G3": "Sub-period consistency (both halves Sharpe > 0)",
        "G4": "Outlier winsorization (1%/99%)",
        "G5": "Ticker concentration < 25%",
        "G6": "WR > 50% AND PF > 1.0",
    },
    "results": results,
    "passing_strategies": [k for k, v in results.items() if v["ALL_GATES_PASS"]],
}

n_pass = len(report["passing_strategies"])
report["verdict"] = (
    f"{n_pass} signal/hold combos pass ALL gates. "
    "Proceed to paper trading with position sizing for $440 account."
) if n_pass > 0 else (
    "NO signal/hold combo passes all 6 quality gates. "
    "Signal is NOT tradeable as-is. Needs refinement or real insider data."
)

out_path = OUT_DIR / "backtest_report.json"
with open(out_path, "w") as f:
    json.dump(report, f, indent=2, default=str)

print(f"\n{'='*60}")
print(f"VERDICT: {report['verdict']}")
print(f"Passing: {report['passing_strategies']}")
print(f"Report: {out_path}")
print(f"{'='*60}", flush=True)
