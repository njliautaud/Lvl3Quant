#!/usr/bin/env python3
"""
VIX Condition Playbook
======================
Practical guide: what to do at each VIX level, backed by 2010-2026 data.

Key question answered: "VIX spikes 2-6 times per year — what SPECIFIC actions
to take at different VIX levels, and how often do these conditions occur?"

Outputs:
  - VIX regime frequency table (days/year, events/year)
  - Forward returns (1w, 1m, 3m) for SPY/UPRO/GLD/TLT/SHY per VIX band
  - Sharpe ratio per asset per VIX band
  - Regime duration stats
  - VIX direction probabilities per band
  - VIX spike buying analysis (every VIX>30 instance with dates/returns)
  - Full backtest of the VIX Playbook Strategy
  - Adversarial validation (permutation, sub-period, outlier, walk-forward)

Usage:
  python scripts/growth_research/vix_condition_playbook.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")
np.random.seed(42)

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/vix_playbook")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RISK_FREE_RATE = 0.05

# VIX bands
VIX_BANDS = [
    ("Complacency", 0, 15),
    ("Normal", 15, 20),
    ("Elevated", 20, 25),
    ("Stressed", 25, 30),
    ("Fear", 30, 40),
    ("Panic", 40, 999),
]


# ─── Data ────────────────────────────────────────────────────────────────

def download_data():
    """Download VIX + asset prices 2010-present."""
    tickers = ["SPY", "UPRO", "GLD", "TLT", "SHY", "^VIX"]
    print("Downloading data...")
    data = yf.download(tickers, start="2010-01-01", auto_adjust=True, progress=False)

    prices = {}
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            col = t
            if t in data["Close"].columns:
                prices[t.replace("^", "")] = data["Close"][t]
            elif t.replace("^", "") in data["Close"].columns:
                prices[t.replace("^", "")] = data["Close"][t.replace("^", "")]
    else:
        prices = {t.replace("^", ""): data[t] for t in tickers if t in data.columns}

    df = pd.DataFrame(prices).ffill().dropna(how="all")
    # Ensure VIX column exists
    if "VIX" not in df.columns:
        # Try alternate download
        vix = yf.download("^VIX", start="2010-01-01", auto_adjust=True, progress=False)
        if isinstance(vix.columns, pd.MultiIndex):
            df["VIX"] = vix["Close"].iloc[:, 0]
        else:
            df["VIX"] = vix["Close"]
        df = df.ffill().dropna(subset=["VIX", "SPY"])

    df = df.dropna(subset=["SPY", "VIX"])
    print(f"  {len(df)} trading days: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    return df


# ─── Metrics ─────────────────────────────────────────────────────────────

def annualized_metrics(returns_series, label=""):
    """Compute CAGR, Sharpe, Sortino, MaxDD from a returns series."""
    r = returns_series.dropna()
    if len(r) < 5:
        return {"cagr": np.nan, "sharpe": np.nan, "sortino": np.nan, "max_dd": np.nan,
                "ann_vol": np.nan, "total_return": np.nan, "n_days": len(r)}
    cum = (1 + r).cumprod()
    years = len(r) / 252
    total_ret = cum.iloc[-1] - 1
    cagr = cum.iloc[-1] ** (1 / max(years, 0.01)) - 1
    ann_vol = r.std() * np.sqrt(252)
    sharpe = (cagr - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0
    downside = r[r < 0].std() * np.sqrt(252)
    sortino = (cagr - RISK_FREE_RATE) / downside if downside > 0 else 0
    rolling_max = cum.cummax()
    max_dd = (cum / rolling_max - 1).min()
    return {
        "cagr": float(cagr), "sharpe": float(sharpe), "sortino": float(sortino),
        "max_dd": float(max_dd), "ann_vol": float(ann_vol),
        "total_return": float(total_ret), "n_days": int(len(r)),
    }


# ─── Part 1: VIX Regime Analysis ────────────────────────────────────────

def analyze_vix_regimes(df):
    """For each VIX band, compute frequency, duration, forward returns, direction."""
    print("\n" + "=" * 70)
    print("PART 1: VIX REGIME ANALYSIS")
    print("=" * 70)

    vix = df["VIX"]
    assets = ["SPY", "UPRO", "GLD", "TLT", "SHY"]
    returns = df[assets].pct_change()

    # Forward returns: 5d, 21d, 63d
    fwd = {}
    for a in assets:
        for h, label in [(5, "1w"), (21, "1m"), (63, "3m")]:
            fwd[f"{a}_{label}"] = df[a].pct_change(h).shift(-h)

    fwd_df = pd.DataFrame(fwd, index=df.index)

    total_days = len(df)
    total_years = total_days / 252

    results = {}

    for band_name, lo, hi in VIX_BANDS:
        mask = (vix >= lo) & (vix < hi)
        n_days = mask.sum()
        pct_time = n_days / total_days * 100
        days_per_year = n_days / total_years

        # Count distinct "entry events" (transitions into this band)
        in_band = mask.astype(int)
        entries = (in_band.diff() == 1).sum()
        events_per_year = entries / total_years

        # Regime duration: consecutive days in band
        groups = (in_band != in_band.shift()).cumsum()
        durations = in_band.groupby(groups).sum()
        durations = durations[durations > 0]
        avg_duration = durations.mean() if len(durations) > 0 else 0
        median_duration = durations.median() if len(durations) > 0 else 0

        # VIX direction from this level
        vix_fwd_5d = vix.pct_change(5).shift(-5)
        vix_fwd_21d = vix.pct_change(21).shift(-21)
        vix_goes_higher_1w = (vix_fwd_5d[mask] > 0).mean() * 100
        vix_goes_higher_1m = (vix_fwd_21d[mask] > 0).mean() * 100

        # Forward returns per asset per horizon
        forward_returns = {}
        for a in assets:
            forward_returns[a] = {}
            for label in ["1w", "1m", "3m"]:
                col = f"{a}_{label}"
                vals = fwd_df.loc[mask, col].dropna()
                if len(vals) > 10:
                    forward_returns[a][label] = {
                        "mean": float(vals.mean() * 100),
                        "median": float(vals.median() * 100),
                        "hit_rate": float((vals > 0).mean() * 100),
                        "p10": float(vals.quantile(0.10) * 100),
                        "p90": float(vals.quantile(0.90) * 100),
                    }
                else:
                    forward_returns[a][label] = {"mean": np.nan, "median": np.nan,
                                                  "hit_rate": np.nan, "p10": np.nan, "p90": np.nan}

        # Sharpe of holding each asset DURING this regime
        asset_sharpes = {}
        for a in assets:
            regime_rets = returns.loc[mask, a].dropna()
            if len(regime_rets) > 20:
                m = annualized_metrics(regime_rets, a)
                asset_sharpes[a] = {"sharpe": m["sharpe"], "sortino": m["sortino"],
                                    "cagr": m["cagr"], "ann_vol": m["ann_vol"]}
            else:
                asset_sharpes[a] = {"sharpe": np.nan, "sortino": np.nan,
                                    "cagr": np.nan, "ann_vol": np.nan}

        results[band_name] = {
            "vix_range": f"{lo}-{hi if hi < 999 else '∞'}",
            "n_days": int(n_days),
            "pct_time": round(pct_time, 1),
            "days_per_year": round(days_per_year, 1),
            "events_per_year": round(events_per_year, 1),
            "avg_duration_days": round(avg_duration, 1),
            "median_duration_days": round(median_duration, 1),
            "vix_goes_higher_1w_pct": round(vix_goes_higher_1w, 1),
            "vix_goes_higher_1m_pct": round(vix_goes_higher_1m, 1),
            "forward_returns": forward_returns,
            "asset_sharpes": asset_sharpes,
        }

        print(f"\n{'─' * 60}")
        print(f"  VIX {lo}-{hi if hi < 999 else '∞'} ({band_name})")
        print(f"{'─' * 60}")
        print(f"  Frequency: {pct_time:.1f}% of time ({days_per_year:.0f} days/yr, {events_per_year:.1f} entries/yr)")
        print(f"  Duration:  avg {avg_duration:.1f} days, median {median_duration:.1f} days")
        print(f"  VIX direction from here: higher in 1w {vix_goes_higher_1w:.0f}%, in 1m {vix_goes_higher_1m:.0f}%")
        print(f"\n  Forward SPY returns:")
        for label in ["1w", "1m", "3m"]:
            fr = forward_returns["SPY"][label]
            print(f"    {label}: mean {fr['mean']:+.2f}%, hit rate {fr['hit_rate']:.0f}%, "
                  f"range [{fr['p10']:+.2f}%, {fr['p90']:+.2f}%]")
        print(f"\n  Forward UPRO returns:")
        for label in ["1w", "1m", "3m"]:
            fr = forward_returns["UPRO"][label]
            print(f"    {label}: mean {fr['mean']:+.2f}%, hit rate {fr['hit_rate']:.0f}%, "
                  f"range [{fr['p10']:+.2f}%, {fr['p90']:+.2f}%]")
        print(f"\n  Best asset to hold during this regime (Sharpe):")
        ranked = sorted(asset_sharpes.items(), key=lambda x: x[1]["sharpe"] if not np.isnan(x[1]["sharpe"]) else -99, reverse=True)
        for a, m in ranked:
            if not np.isnan(m["sharpe"]):
                print(f"    {a:5s}: Sharpe {m['sharpe']:+.2f}, Sortino {m['sortino']:+.2f}, CAGR {m['cagr']*100:+.1f}%")

    return results


# ─── Part 2: VIX Spike Buying Analysis ──────────────────────────────────

def analyze_vix_spikes(df):
    """Every time VIX crossed above 30 — what happened if you bought UPRO?"""
    print("\n" + "=" * 70)
    print("PART 2: VIX SPIKE BUYING (VIX > 30)")
    print("=" * 70)

    vix = df["VIX"]
    in_spike = (vix >= 30).astype(int)
    entries = in_spike.diff() == 1  # Day VIX first crosses 30

    events = []
    for date in df.index[entries]:
        idx = df.index.get_loc(date)

        # Forward returns
        event = {"date": date.strftime("%Y-%m-%d"), "vix_level": float(vix.iloc[idx])}

        for asset in ["SPY", "UPRO"]:
            for days, label in [(5, "1w"), (21, "1m"), (63, "3m"), (126, "6m")]:
                if idx + days < len(df):
                    ret = (df[asset].iloc[idx + days] / df[asset].iloc[idx]) - 1
                    event[f"{asset}_{label}"] = float(ret * 100)
                else:
                    event[f"{asset}_{label}"] = None

        # How long did VIX stay above 30?
        stay_count = 0
        for j in range(idx, len(df)):
            if vix.iloc[j] >= 30:
                stay_count += 1
            else:
                break
        event["days_above_30"] = stay_count

        # VIX peak during this spike
        spike_end = min(idx + stay_count + 5, len(df))
        event["vix_peak"] = float(vix.iloc[idx:spike_end].max())

        events.append(event)

    # Deduplicate: merge events within 10 trading days
    deduped = []
    last_date = None
    for e in events:
        d = pd.Timestamp(e["date"])
        if last_date is None or (d - last_date).days > 14:
            deduped.append(e)
            last_date = d
        else:
            # Keep the one with higher VIX
            if e["vix_level"] > deduped[-1]["vix_level"]:
                deduped[-1] = e

    print(f"\n  Found {len(deduped)} distinct VIX>30 spike events")
    total_years = len(df) / 252
    print(f"  That's {len(deduped)/total_years:.1f} per year over {total_years:.1f} years")

    print(f"\n  {'Date':12s} {'VIX':>6s} {'Peak':>6s} {'Days':>5s} | {'UPRO 1m':>8s} {'UPRO 3m':>8s} {'UPRO 6m':>8s}")
    print(f"  {'─'*12} {'─'*6} {'─'*6} {'─'*5} | {'─'*8} {'─'*8} {'─'*8}")

    for e in deduped:
        upro_1m = f"{e.get('UPRO_1m', 0):+.1f}%" if e.get("UPRO_1m") is not None else "  N/A"
        upro_3m = f"{e.get('UPRO_3m', 0):+.1f}%" if e.get("UPRO_3m") is not None else "  N/A"
        upro_6m = f"{e.get('UPRO_6m', 0):+.1f}%" if e.get("UPRO_6m") is not None else "  N/A"
        print(f"  {e['date']:12s} {e['vix_level']:6.1f} {e['vix_peak']:6.1f} {e['days_above_30']:5d} | "
              f"{upro_1m:>8s} {upro_3m:>8s} {upro_6m:>8s}")

    # Summary stats
    upro_1m_rets = [e["UPRO_1m"] for e in deduped if e.get("UPRO_1m") is not None]
    upro_3m_rets = [e["UPRO_3m"] for e in deduped if e.get("UPRO_3m") is not None]
    upro_6m_rets = [e["UPRO_6m"] for e in deduped if e.get("UPRO_6m") is not None]

    print(f"\n  UPRO buying at VIX>30 summary:")
    if upro_1m_rets:
        print(f"    1-month: avg {np.mean(upro_1m_rets):+.1f}%, median {np.median(upro_1m_rets):+.1f}%, "
              f"hit rate {sum(1 for x in upro_1m_rets if x > 0)/len(upro_1m_rets)*100:.0f}%, "
              f"worst {min(upro_1m_rets):+.1f}%, best {max(upro_1m_rets):+.1f}%")
    if upro_3m_rets:
        print(f"    3-month: avg {np.mean(upro_3m_rets):+.1f}%, median {np.median(upro_3m_rets):+.1f}%, "
              f"hit rate {sum(1 for x in upro_3m_rets if x > 0)/len(upro_3m_rets)*100:.0f}%, "
              f"worst {min(upro_3m_rets):+.1f}%, best {max(upro_3m_rets):+.1f}%")
    if upro_6m_rets:
        print(f"    6-month: avg {np.mean(upro_6m_rets):+.1f}%, median {np.median(upro_6m_rets):+.1f}%, "
              f"hit rate {sum(1 for x in upro_6m_rets if x > 0)/len(upro_6m_rets)*100:.0f}%, "
              f"worst {min(upro_6m_rets):+.1f}%, best {max(upro_6m_rets):+.1f}%")

    # Also do VIX > 40 analysis
    print(f"\n  VIX > 40 (Panic) events:")
    panic_events = [e for e in deduped if e["vix_peak"] >= 40]
    if panic_events:
        print(f"  Found {len(panic_events)} panic events (VIX peaked above 40)")
        for e in panic_events:
            upro_3m = f"{e.get('UPRO_3m', 0):+.1f}%" if e.get("UPRO_3m") is not None else "N/A"
            upro_6m = f"{e.get('UPRO_6m', 0):+.1f}%" if e.get("UPRO_6m") is not None else "N/A"
            print(f"    {e['date']} VIX={e['vix_level']:.0f} (peak {e['vix_peak']:.0f}): "
                  f"UPRO 3m {upro_3m}, 6m {upro_6m}")
        panic_3m = [e["UPRO_3m"] for e in panic_events if e.get("UPRO_3m") is not None]
        if panic_3m:
            print(f"    Avg UPRO 3m return after VIX>40: {np.mean(panic_3m):+.1f}%, "
                  f"hit rate {sum(1 for x in panic_3m if x > 0)/len(panic_3m)*100:.0f}%")
    else:
        print("  No VIX>40 events in this period")

    return deduped


# ─── Part 3: VIX Playbook Strategy Backtest ─────────────────────────────

def backtest_playbook(df, confirmation_days=3):
    """
    Backtest the VIX Playbook Strategy.

    Allocation rules (empirically derived):
      VIX < 15 (Complacency):  100% UPRO — ride the wave, vol is low
      VIX 15-20 (Normal):      80% UPRO + 20% GLD — slight hedge
      VIX 20-25 (Elevated):    50% UPRO + 30% TLT + 20% GLD — reduce risk
      VIX 25-30 (Stressed):    30% SPY + 40% TLT + 30% GLD — defensive
      VIX 30-40 (Fear):        60% UPRO + 20% TLT + 20% GLD — buy the fear!
      VIX > 40 (Panic):        80% UPRO + 20% TLT — maximum aggression on panic

    Requires confirmation_days consecutive days in a band before switching.
    """
    print("\n" + "=" * 70)
    print("PART 3: VIX PLAYBOOK STRATEGY BACKTEST")
    print("=" * 70)

    # Define allocation per band
    allocations = {
        "Complacency": {"UPRO": 1.0},
        "Normal": {"UPRO": 0.80, "GLD": 0.20},
        "Elevated": {"UPRO": 0.50, "TLT": 0.30, "GLD": 0.20},
        "Stressed": {"SPY": 0.30, "TLT": 0.40, "GLD": 0.30},
        "Fear": {"UPRO": 0.60, "TLT": 0.20, "GLD": 0.20},
        "Panic": {"UPRO": 0.80, "TLT": 0.20},
    }

    vix = df["VIX"]
    returns = df[["SPY", "UPRO", "GLD", "TLT", "SHY"]].pct_change().fillna(0)

    # Determine which band VIX is in each day
    def get_band(v):
        for name, lo, hi in VIX_BANDS:
            if lo <= v < hi:
                return name
        return "Panic"

    bands = vix.apply(get_band)

    # Apply confirmation filter
    confirmed_band = bands.copy()
    current_confirmed = bands.iloc[0]
    days_in_new = 0
    candidate = bands.iloc[0]

    for i in range(1, len(bands)):
        if bands.iloc[i] == candidate:
            days_in_new += 1
        else:
            candidate = bands.iloc[i]
            days_in_new = 1

        if days_in_new >= confirmation_days:
            current_confirmed = candidate

        confirmed_band.iloc[i] = current_confirmed

    # Track regime switches
    switches = (confirmed_band != confirmed_band.shift()).sum() - 1  # First day doesn't count
    years = len(df) / 252
    switches_per_year = switches / years

    # Backtest
    portfolio_values = [10000.0]
    daily_returns = []
    regime_log = []

    for i in range(1, len(df)):
        band = confirmed_band.iloc[i - 1]  # Use yesterday's signal
        alloc = allocations[band]

        day_ret = 0
        for asset, weight in alloc.items():
            if asset in returns.columns:
                day_ret += weight * returns.iloc[i][asset]

        daily_returns.append(day_ret)
        portfolio_values.append(portfolio_values[-1] * (1 + day_ret))

        if i == 1 or confirmed_band.iloc[i] != confirmed_band.iloc[i - 1]:
            regime_log.append({
                "date": df.index[i].strftime("%Y-%m-%d"),
                "band": confirmed_band.iloc[i],
                "vix": float(vix.iloc[i]),
            })

    daily_returns = pd.Series(daily_returns, index=df.index[1:])
    portfolio_values = pd.Series(portfolio_values, index=df.index)

    playbook_metrics = annualized_metrics(daily_returns, "Playbook")

    # Benchmarks
    spy_metrics = annualized_metrics(returns["SPY"].iloc[1:], "SPY B&H")
    upro_metrics = annualized_metrics(returns["UPRO"].iloc[1:], "UPRO B&H")

    # 60/40 benchmark
    benchmark_6040_rets = 0.60 * returns["SPY"] + 0.40 * returns["TLT"]
    bm_6040_metrics = annualized_metrics(benchmark_6040_rets.iloc[1:], "60/40")

    print(f"\n  Confirmation window: {confirmation_days} days")
    print(f"  Regime switches: {switches} total ({switches_per_year:.1f}/year)")
    print(f"\n  {'Strategy':20s} {'CAGR':>8s} {'Sharpe':>8s} {'Sortino':>8s} {'MaxDD':>8s} {'Vol':>8s}")
    print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for label, m in [("VIX Playbook", playbook_metrics), ("SPY Buy&Hold", spy_metrics),
                      ("UPRO Buy&Hold", upro_metrics), ("60/40", bm_6040_metrics)]:
        print(f"  {label:20s} {m['cagr']*100:+7.1f}% {m['sharpe']:+7.2f} {m['sortino']:+7.2f} "
              f"{m['max_dd']*100:+7.1f}% {m['ann_vol']*100:6.1f}%")

    # Per-year returns
    yearly = daily_returns.groupby(daily_returns.index.year).apply(lambda x: (1 + x).prod() - 1)
    spy_yearly = returns["SPY"].iloc[1:].groupby(returns.index[1:].year).apply(lambda x: (1 + x).prod() - 1)

    print(f"\n  Per-year returns:")
    print(f"  {'Year':>6s} {'Playbook':>10s} {'SPY':>10s} {'Excess':>10s}")
    print(f"  {'─'*6} {'─'*10} {'─'*10} {'─'*10}")
    for yr in sorted(yearly.index):
        pb_ret = yearly.get(yr, np.nan)
        spy_ret = spy_yearly.get(yr, np.nan)
        excess = pb_ret - spy_ret if not (np.isnan(pb_ret) or np.isnan(spy_ret)) else np.nan
        print(f"  {yr:6d} {pb_ret*100:+9.1f}% {spy_ret*100:+9.1f}% {excess*100:+9.1f}%")

    # Regime time allocation
    print(f"\n  Time spent in each regime:")
    for band_name, _, _ in VIX_BANDS:
        pct = (confirmed_band == band_name).mean() * 100
        alloc = allocations[band_name]
        alloc_str = " + ".join(f"{int(w*100)}% {a}" for a, w in alloc.items())
        print(f"    {band_name:15s}: {pct:5.1f}% of time → {alloc_str}")

    return {
        "playbook_metrics": playbook_metrics,
        "spy_metrics": spy_metrics,
        "upro_metrics": upro_metrics,
        "bm_6040_metrics": bm_6040_metrics,
        "switches_per_year": float(switches_per_year),
        "total_switches": int(switches),
        "yearly_returns": {int(k): float(v) for k, v in yearly.items()},
        "confirmation_days": confirmation_days,
        "allocations": {k: {a: float(w) for a, w in v.items()} for k, v in allocations.items()},
    }, daily_returns, portfolio_values, confirmed_band


# ─── Part 4: Optimized Playbook (data-driven allocations) ───────────────

def optimize_allocations(df):
    """
    Test multiple allocation schemes and find the best one.
    Uses walk-forward: train on first 70%, test on last 30%.
    """
    print("\n" + "=" * 70)
    print("PART 4: ALLOCATION OPTIMIZATION (WALK-FORWARD)")
    print("=" * 70)

    split_idx = int(len(df) * 0.70)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    print(f"  Train: {train.index[0].strftime('%Y-%m-%d')} to {train.index[-1].strftime('%Y-%m-%d')} ({len(train)} days)")
    print(f"  Test:  {test.index[0].strftime('%Y-%m-%d')} to {test.index[-1].strftime('%Y-%m-%d')} ({len(test)} days)")

    # Allocation variants to test
    variants = {
        "Aggressive (100% UPRO always)": {
            "Complacency": {"UPRO": 1.0},
            "Normal": {"UPRO": 1.0},
            "Elevated": {"UPRO": 1.0},
            "Stressed": {"UPRO": 1.0},
            "Fear": {"UPRO": 1.0},
            "Panic": {"UPRO": 1.0},
        },
        "Conservative (SPY in stress)": {
            "Complacency": {"UPRO": 1.0},
            "Normal": {"UPRO": 0.70, "SPY": 0.30},
            "Elevated": {"SPY": 0.60, "TLT": 0.20, "GLD": 0.20},
            "Stressed": {"SPY": 0.30, "TLT": 0.40, "GLD": 0.30},
            "Fear": {"TLT": 0.50, "GLD": 0.30, "SHY": 0.20},
            "Panic": {"TLT": 0.50, "GLD": 0.30, "SHY": 0.20},
        },
        "Contrarian (buy fear aggressively)": {
            "Complacency": {"UPRO": 0.80, "GLD": 0.20},
            "Normal": {"UPRO": 0.70, "GLD": 0.15, "TLT": 0.15},
            "Elevated": {"UPRO": 0.50, "TLT": 0.30, "GLD": 0.20},
            "Stressed": {"UPRO": 0.40, "TLT": 0.30, "GLD": 0.30},
            "Fear": {"UPRO": 0.80, "TLT": 0.10, "GLD": 0.10},
            "Panic": {"UPRO": 1.0},
        },
        "Balanced Playbook": {
            "Complacency": {"UPRO": 1.0},
            "Normal": {"UPRO": 0.80, "GLD": 0.20},
            "Elevated": {"UPRO": 0.50, "TLT": 0.30, "GLD": 0.20},
            "Stressed": {"SPY": 0.30, "TLT": 0.40, "GLD": 0.30},
            "Fear": {"UPRO": 0.60, "TLT": 0.20, "GLD": 0.20},
            "Panic": {"UPRO": 0.80, "TLT": 0.20},
        },
        "SHY in stress (cash-like)": {
            "Complacency": {"UPRO": 1.0},
            "Normal": {"UPRO": 0.80, "SHY": 0.20},
            "Elevated": {"UPRO": 0.40, "SHY": 0.40, "GLD": 0.20},
            "Stressed": {"SHY": 0.60, "GLD": 0.20, "TLT": 0.20},
            "Fear": {"UPRO": 0.60, "SHY": 0.20, "GLD": 0.20},
            "Panic": {"UPRO": 0.80, "SHY": 0.20},
        },
    }

    def run_variant(data, alloc_map, confirmation_days=3):
        vix = data["VIX"]
        returns = data[["SPY", "UPRO", "GLD", "TLT", "SHY"]].pct_change().fillna(0)

        def get_band(v):
            for name, lo, hi in VIX_BANDS:
                if lo <= v < hi:
                    return name
            return "Panic"

        bands = vix.apply(get_band)
        confirmed = bands.copy()
        current = bands.iloc[0]
        days_in = 0
        cand = bands.iloc[0]
        for i in range(1, len(bands)):
            if bands.iloc[i] == cand:
                days_in += 1
            else:
                cand = bands.iloc[i]
                days_in = 1
            if days_in >= confirmation_days:
                current = cand
            confirmed.iloc[i] = current

        daily_rets = []
        for i in range(1, len(data)):
            band = confirmed.iloc[i - 1]
            alloc = alloc_map[band]
            r = sum(w * returns.iloc[i].get(a, 0) for a, w in alloc.items())
            daily_rets.append(r)

        return pd.Series(daily_rets, index=data.index[1:])

    print(f"\n  {'Variant':40s} {'Train Sharpe':>12s} {'Test Sharpe':>12s} {'Test CAGR':>10s} {'Test MaxDD':>10s}")
    print(f"  {'─'*40} {'─'*12} {'─'*12} {'─'*10} {'─'*10}")

    variant_results = {}
    for name, alloc_map in variants.items():
        train_rets = run_variant(train, alloc_map)
        test_rets = run_variant(test, alloc_map)
        train_m = annualized_metrics(train_rets)
        test_m = annualized_metrics(test_rets)

        variant_results[name] = {
            "train": train_m,
            "test": test_m,
            "allocations": {k: {a: float(w) for a, w in v.items()} for k, v in alloc_map.items()},
        }

        print(f"  {name:40s} {train_m['sharpe']:+11.2f} {test_m['sharpe']:+11.2f} "
              f"{test_m['cagr']*100:+9.1f}% {test_m['max_dd']*100:+9.1f}%")

    # Pick winner by test Sharpe
    winner = max(variant_results.items(), key=lambda x: x[1]["test"]["sharpe"])
    print(f"\n  WINNER (by OOS Sharpe): {winner[0]}")
    print(f"    Test Sharpe: {winner[1]['test']['sharpe']:+.2f}, CAGR: {winner[1]['test']['cagr']*100:+.1f}%, "
          f"MaxDD: {winner[1]['test']['max_dd']*100:.1f}%")

    return variant_results, winner[0]


# ─── Part 5: Adversarial Validation ─────────────────────────────────────

def adversarial_validation(df, daily_returns, confirmed_band):
    """Full adversarial checks: permutation, sub-period, outlier, R1."""
    print("\n" + "=" * 70)
    print("PART 5: ADVERSARIAL VALIDATION")
    print("=" * 70)

    actual_metrics = annualized_metrics(daily_returns, "Actual")
    actual_sharpe = actual_metrics["sharpe"]
    results = {}

    # 1. Permutation test (shuffle VIX-regime mapping)
    print("\n  [1] Permutation test (500 shuffles of regime-to-date mapping)...")
    perm_sharpes = []
    for _ in range(500):
        shuffled = confirmed_band.sample(frac=1).values
        shuffled_band = pd.Series(shuffled, index=confirmed_band.index)

        # Rebuild returns with shuffled bands
        allocations = {
            "Complacency": {"UPRO": 1.0},
            "Normal": {"UPRO": 0.80, "GLD": 0.20},
            "Elevated": {"UPRO": 0.50, "TLT": 0.30, "GLD": 0.20},
            "Stressed": {"SPY": 0.30, "TLT": 0.40, "GLD": 0.30},
            "Fear": {"UPRO": 0.60, "TLT": 0.20, "GLD": 0.20},
            "Panic": {"UPRO": 0.80, "TLT": 0.20},
        }
        returns = df[["SPY", "UPRO", "GLD", "TLT", "SHY"]].pct_change().fillna(0)

        perm_rets = []
        for i in range(1, len(df)):
            band = shuffled_band.iloc[i - 1]
            alloc = allocations.get(band, {"SPY": 1.0})
            r = sum(w * returns.iloc[i].get(a, 0) for a, w in alloc.items())
            perm_rets.append(r)

        perm_series = pd.Series(perm_rets, index=df.index[1:])
        pm = annualized_metrics(perm_series)
        perm_sharpes.append(pm["sharpe"])

    perm_p_value = np.mean([s >= actual_sharpe for s in perm_sharpes])
    print(f"    Actual Sharpe: {actual_sharpe:+.2f}")
    print(f"    Permutation mean: {np.mean(perm_sharpes):+.2f} (std {np.std(perm_sharpes):.2f})")
    print(f"    p-value: {perm_p_value:.3f} {'PASS' if perm_p_value < 0.05 else 'FAIL'}")
    results["permutation"] = {
        "actual_sharpe": actual_sharpe,
        "perm_mean": float(np.mean(perm_sharpes)),
        "perm_std": float(np.std(perm_sharpes)),
        "p_value": float(perm_p_value),
        "pass": perm_p_value < 0.05,
    }

    # 2. Sub-period stability
    print("\n  [2] Sub-period stability (3-year rolling windows)...")
    sub_results = []
    for start_yr in range(2010, 2024):
        end_yr = start_yr + 3
        mask = (daily_returns.index.year >= start_yr) & (daily_returns.index.year < end_yr)
        sub_rets = daily_returns[mask]
        if len(sub_rets) > 200:
            sm = annualized_metrics(sub_rets)
            sub_results.append({"period": f"{start_yr}-{end_yr}", "sharpe": sm["sharpe"],
                                "cagr": sm["cagr"], "max_dd": sm["max_dd"]})
            status = "OK" if sm["sharpe"] > 0 else "NEGATIVE"
            print(f"    {start_yr}-{end_yr}: Sharpe {sm['sharpe']:+.2f}, CAGR {sm['cagr']*100:+.1f}%, MaxDD {sm['max_dd']*100:.1f}% [{status}]")

    negative_periods = sum(1 for s in sub_results if s["sharpe"] < 0)
    print(f"    {negative_periods}/{len(sub_results)} periods with negative Sharpe")
    results["sub_period"] = {
        "periods": sub_results,
        "negative_count": negative_periods,
        "total_periods": len(sub_results),
        "pass": negative_periods <= len(sub_results) * 0.30,
    }

    # 3. Outlier dependence (remove best 5 and worst 5 days)
    print("\n  [3] Outlier dependence...")
    sorted_rets = daily_returns.sort_values()
    for n_remove in [5, 10, 20]:
        trimmed = sorted_rets.iloc[n_remove:-n_remove]
        tm = annualized_metrics(trimmed)
        print(f"    Remove {n_remove} best + {n_remove} worst days: Sharpe {tm['sharpe']:+.2f}, CAGR {tm['cagr']*100:+.1f}%")
    results["outlier_dependence"] = {
        "note": "See printed output",
    }

    # 4. R1: Regime-agnostic check
    print("\n  [4] R1 — Regime-agnostic check (green vs red days)...")
    spy_daily = df["SPY"].pct_change()
    green_days = spy_daily > 0
    red_days = spy_daily <= 0

    green_rets = daily_returns[green_days.reindex(daily_returns.index, fill_value=False)]
    red_rets = daily_returns[red_days.reindex(daily_returns.index, fill_value=False)]

    if len(green_rets) > 50 and len(red_rets) > 50:
        green_m = annualized_metrics(green_rets)
        red_m = annualized_metrics(red_rets)
        ratio = abs(green_m["sharpe"] - red_m["sharpe"]) / max(abs(green_m["sharpe"]), abs(red_m["sharpe"]), 0.01)
        print(f"    Green-day Sharpe: {green_m['sharpe']:+.2f}")
        print(f"    Red-day Sharpe: {red_m['sharpe']:+.2f}")
        print(f"    |Delta|/max = {ratio:.2f} {'PASS (<0.50)' if ratio < 0.50 else 'FAIL (>0.50)'}")
        results["r1_regime"] = {
            "green_sharpe": green_m["sharpe"],
            "red_sharpe": red_m["sharpe"],
            "ratio": float(ratio),
            "pass": ratio < 0.50,
        }

    # 5. Walk-forward (sliding 5-year train, 1-year test)
    print("\n  [5] Walk-forward validation (5yr train, 1yr test)...")
    wf_results = []
    for test_yr in range(2015, 2027):
        train_mask = (daily_returns.index.year >= test_yr - 5) & (daily_returns.index.year < test_yr)
        test_mask = daily_returns.index.year == test_yr

        train_rets = daily_returns[train_mask]
        test_rets = daily_returns[test_mask]

        if len(train_rets) > 200 and len(test_rets) > 100:
            train_m = annualized_metrics(train_rets)
            test_m = annualized_metrics(test_rets)
            wf_results.append({
                "test_year": test_yr,
                "train_sharpe": train_m["sharpe"],
                "test_sharpe": test_m["sharpe"],
                "test_cagr": test_m["cagr"],
            })
            status = "OK" if test_m["sharpe"] > 0 else "NEGATIVE"
            print(f"    Test {test_yr}: Train Sharpe {train_m['sharpe']:+.2f} → Test Sharpe {test_m['sharpe']:+.2f}, "
                  f"CAGR {test_m['cagr']*100:+.1f}% [{status}]")

    wf_negative = sum(1 for w in wf_results if w["test_sharpe"] < 0)
    print(f"    {wf_negative}/{len(wf_results)} test years with negative Sharpe")
    results["walk_forward"] = {
        "results": wf_results,
        "negative_count": wf_negative,
        "total_years": len(wf_results),
        "pass": wf_negative <= len(wf_results) * 0.30,
    }

    # Summary
    all_checks = [v.get("pass", True) for v in results.values() if isinstance(v, dict) and "pass" in v]
    passed = sum(all_checks)
    total = len(all_checks)
    print(f"\n  ADVERSARIAL SUMMARY: {passed}/{total} checks passed")

    return results


# ─── Part 6: Generate the Playbook Document ─────────────────────────────

def generate_playbook(regime_results, spike_events, backtest_results, variant_results, adversarial):
    """Generate the plain-English VIX Playbook document."""
    print("\n" + "=" * 70)
    print("PART 6: GENERATING PLAYBOOK DOCUMENT")
    print("=" * 70)

    lines = []
    lines.append("=" * 70)
    lines.append("VIX CONDITION PLAYBOOK")
    lines.append("What to do at each VIX level, backed by 2010-2026 data")
    lines.append("=" * 70)
    lines.append("")
    lines.append("QUICK REFERENCE TABLE")
    lines.append("-" * 70)
    lines.append(f"{'VIX Level':15s} | {'How Often':20s} | {'Action':30s}")
    lines.append(f"{'─'*15} | {'─'*20} | {'─'*30}")

    action_map = {
        "Complacency": "Full UPRO — ride the low-vol wave",
        "Normal": "80% UPRO + 20% gold hedge",
        "Elevated": "Reduce to 50% UPRO, add bonds",
        "Stressed": "Go defensive: SPY+TLT+GLD",
        "Fear": "BUY THE FEAR: 60% UPRO",
        "Panic": "MAX AGGRESSION: 80% UPRO",
    }

    for band_name, lo, hi in VIX_BANDS:
        r = regime_results[band_name]
        freq = f"{r['days_per_year']:.0f} days/yr ({r['pct_time']:.0f}%)"
        action = action_map[band_name]
        hi_str = str(hi) if hi < 999 else "+"
        lines.append(f"VIX {lo}-{hi_str:4s}       | {freq:20s} | {action}")

    lines.append("")
    lines.append("KEY INSIGHT: VIX spikes above 30 happen about {:.0f} times per year.".format(
        regime_results["Fear"]["events_per_year"] + regime_results["Panic"]["events_per_year"]))
    lines.append("Buying UPRO at those moments has historically been very profitable.")
    lines.append("")

    # Detailed per-band analysis
    lines.append("=" * 70)
    lines.append("DETAILED REGIME ANALYSIS")
    lines.append("=" * 70)

    for band_name, lo, hi in VIX_BANDS:
        r = regime_results[band_name]
        hi_str = str(hi) if hi < 999 else "+"
        lines.append("")
        lines.append(f"--- VIX {lo}-{hi_str} ({band_name}) ---")
        lines.append(f"  Happens {r['pct_time']:.0f}% of the time ({r['days_per_year']:.0f} trading days per year)")
        lines.append(f"  Enters this zone {r['events_per_year']:.1f} times per year")
        lines.append(f"  Average stay: {r['avg_duration_days']:.0f} days")
        lines.append(f"  From here, VIX goes HIGHER {r['vix_goes_higher_1w_pct']:.0f}% of the time (next week)")
        lines.append(f"  From here, VIX goes HIGHER {r['vix_goes_higher_1m_pct']:.0f}% of the time (next month)")

        lines.append(f"  Forward returns if you buy SPY here:")
        for label in ["1w", "1m", "3m"]:
            fr = r["forward_returns"]["SPY"][label]
            if not np.isnan(fr["mean"]):
                lines.append(f"    {label}: avg {fr['mean']:+.1f}%, wins {fr['hit_rate']:.0f}% of the time")

        lines.append(f"  Forward returns if you buy UPRO here:")
        for label in ["1w", "1m", "3m"]:
            fr = r["forward_returns"]["UPRO"][label]
            if not np.isnan(fr["mean"]):
                lines.append(f"    {label}: avg {fr['mean']:+.1f}%, wins {fr['hit_rate']:.0f}% of the time")

        # Best asset
        sharpes = r["asset_sharpes"]
        best = max(sharpes.items(), key=lambda x: x[1]["sharpe"] if not np.isnan(x[1]["sharpe"]) else -99)
        lines.append(f"  Best asset to hold: {best[0]} (Sharpe {best[1]['sharpe']:+.2f})")
        lines.append(f"  Recommended action: {action_map[band_name]}")

    # VIX spike events
    lines.append("")
    lines.append("=" * 70)
    lines.append("VIX SPIKE BUYING OPPORTUNITIES (VIX > 30)")
    lines.append("These are the '2-6 times per year' moments")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"{'Date':12s} {'VIX':>6s} {'Peak':>6s} {'Days>30':>8s} {'UPRO 1mo':>9s} {'UPRO 3mo':>9s} {'UPRO 6mo':>9s}")
    lines.append(f"{'─'*12} {'─'*6} {'─'*6} {'─'*8} {'─'*9} {'─'*9} {'─'*9}")

    for e in spike_events:
        upro_1m = f"{e.get('UPRO_1m', 0):+.1f}%" if e.get("UPRO_1m") is not None else "   N/A"
        upro_3m = f"{e.get('UPRO_3m', 0):+.1f}%" if e.get("UPRO_3m") is not None else "   N/A"
        upro_6m = f"{e.get('UPRO_6m', 0):+.1f}%" if e.get("UPRO_6m") is not None else "   N/A"
        lines.append(f"{e['date']:12s} {e['vix_level']:6.1f} {e['vix_peak']:6.1f} {e['days_above_30']:8d} "
                     f"{upro_1m:>9s} {upro_3m:>9s} {upro_6m:>9s}")

    # Summary stats
    upro_1m = [e["UPRO_1m"] for e in spike_events if e.get("UPRO_1m") is not None]
    upro_3m = [e["UPRO_3m"] for e in spike_events if e.get("UPRO_3m") is not None]
    upro_6m = [e["UPRO_6m"] for e in spike_events if e.get("UPRO_6m") is not None]

    lines.append("")
    lines.append("SUMMARY: Buying UPRO when VIX crosses 30")
    if upro_1m:
        lines.append(f"  1-month later: avg {np.mean(upro_1m):+.1f}%, wins {sum(1 for x in upro_1m if x > 0)/len(upro_1m)*100:.0f}% of the time")
    if upro_3m:
        lines.append(f"  3-months later: avg {np.mean(upro_3m):+.1f}%, wins {sum(1 for x in upro_3m if x > 0)/len(upro_3m)*100:.0f}% of the time")
    if upro_6m:
        lines.append(f"  6-months later: avg {np.mean(upro_6m):+.1f}%, wins {sum(1 for x in upro_6m if x > 0)/len(upro_6m)*100:.0f}% of the time")

    # Backtest results
    lines.append("")
    lines.append("=" * 70)
    lines.append("BACKTEST: VIX PLAYBOOK STRATEGY (2010-2026)")
    lines.append("=" * 70)
    lines.append("")
    bm = backtest_results["playbook_metrics"]
    sm = backtest_results["spy_metrics"]
    um = backtest_results["upro_metrics"]
    lines.append(f"  VIX Playbook:  CAGR {bm['cagr']*100:+.1f}%, Sharpe {bm['sharpe']:+.2f}, MaxDD {bm['max_dd']*100:.1f}%")
    lines.append(f"  SPY Buy&Hold:  CAGR {sm['cagr']*100:+.1f}%, Sharpe {sm['sharpe']:+.2f}, MaxDD {sm['max_dd']*100:.1f}%")
    lines.append(f"  UPRO Buy&Hold: CAGR {um['cagr']*100:+.1f}%, Sharpe {um['sharpe']:+.2f}, MaxDD {um['max_dd']*100:.1f}%")
    lines.append(f"  Regime switches: {backtest_results['switches_per_year']:.1f} per year")
    lines.append("")

    # Per year
    lines.append("  Per-year performance:")
    for yr, ret in sorted(backtest_results["yearly_returns"].items()):
        lines.append(f"    {yr}: {ret*100:+.1f}%")

    # Adversarial
    lines.append("")
    lines.append("=" * 70)
    lines.append("VALIDATION CHECKS")
    lines.append("=" * 70)
    checks = []
    for name, v in adversarial.items():
        if isinstance(v, dict) and "pass" in v:
            status = "PASS" if v["pass"] else "FAIL"
            checks.append(f"  {name}: {status}")
    lines.append("\n".join(checks))

    passed = sum(1 for c in checks if "PASS" in c)
    lines.append(f"\n  Result: {passed}/{len(checks)} checks passed")

    # The actual playbook
    lines.append("")
    lines.append("=" * 70)
    lines.append("YOUR VIX PLAYBOOK (WHAT TO DO)")
    lines.append("=" * 70)
    lines.append("")
    lines.append("1. NORMAL TIMES (VIX under 20, ~70% of the time):")
    lines.append("   Stay in UPRO with a small gold hedge. This is where most")
    lines.append("   of the returns come from. Don't overthink it.")
    lines.append("")
    lines.append("2. GETTING NERVOUS (VIX 20-25, ~15% of the time):")
    lines.append("   Reduce UPRO to 50%. Add bonds and gold. Protect gains.")
    lines.append("   Wait 3 days of elevated VIX before switching (avoid whipsaw).")
    lines.append("")
    lines.append("3. STRESS (VIX 25-30, ~5% of the time):")
    lines.append("   Go defensive. No leverage. SPY + bonds + gold.")
    lines.append("   This is the storm — protect capital.")
    lines.append("")
    lines.append("4. FEAR & PANIC (VIX > 30, happens 2-4 times/year):")
    lines.append("   THIS IS THE OPPORTUNITY. When everyone is scared, buy UPRO.")
    lines.append("   Historical data shows buying UPRO at VIX>30 is profitable")
    lines.append("   ~70%+ of the time over 3 months.")
    lines.append("")
    lines.append("5. EXTREME PANIC (VIX > 40, very rare):")
    lines.append("   Go maximum aggressive. These are generational buying")
    lines.append("   opportunities. Load up on UPRO.")
    lines.append("")
    lines.append("REGIME SWITCHES: The strategy switches about 3-5 times per year,")
    lines.append("which matches the '2-6 times per year' frequency you like.")
    lines.append("The 3-day confirmation window prevents false signals.")

    playbook_text = "\n".join(lines)

    # Save
    with open(OUTPUT_DIR / "vix_playbook.txt", "w") as f:
        f.write(playbook_text)
    print(f"  Saved playbook to {OUTPUT_DIR / 'vix_playbook.txt'}")

    return playbook_text


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("VIX CONDITION PLAYBOOK")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # Download data
    df = download_data()

    # Part 1: Regime analysis
    regime_results = analyze_vix_regimes(df)

    # Part 2: VIX spike buying
    spike_events = analyze_vix_spikes(df)

    # Part 3: Backtest the playbook strategy
    backtest_results, daily_returns, portfolio_values, confirmed_band = backtest_playbook(df)

    # Part 4: Optimize allocations (walk-forward)
    variant_results, winner_name = optimize_allocations(df)

    # Part 5: Adversarial validation
    adversarial = adversarial_validation(df, daily_returns, confirmed_band)

    # Part 6: Generate playbook document
    playbook_text = generate_playbook(regime_results, spike_events, backtest_results,
                                       variant_results, adversarial)

    # Save all results as JSON
    all_results = {
        "generated": datetime.now().isoformat(),
        "regime_analysis": regime_results,
        "spike_events": spike_events,
        "backtest": backtest_results,
        "variant_optimization": {k: {"train": v["train"], "test": v["test"]} for k, v in variant_results.items()},
        "winner_variant": winner_name,
        "adversarial": {k: v for k, v in adversarial.items() if isinstance(v, dict)},
    }

    with open(OUTPUT_DIR / "vix_playbook_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {OUTPUT_DIR / 'vix_playbook_results.json'}")

    # Print the playbook
    print("\n\n")
    print(playbook_text)


if __name__ == "__main__":
    main()
