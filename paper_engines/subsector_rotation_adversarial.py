#!/usr/bin/env python3
"""
Sub-Sector Rotation — Comprehensive Adversarial Validation
=============================================================
8 adversarial tests to determine whether the rotation signal has real edge
or is an artifact of momentum-chasing, concentration, or parameter-fitting.

Tests:
  1. Re-implementation (independent signal + backtest)
  2. Inverse signal (trade the opposite)
  3. Random timing (1000 permutation tests)
  4. Sub-period stability (4 equal periods)
  5. Top-N removal (remove 3 best sub-sectors)
  6. Parameter sensitivity (lookback × top-N × rebalance grid)
  7. Lead-lag (does past rotation predict future returns?)
  8. Transaction cost sensitivity (breakeven cost)

HC #0: Sliding window only (NOT expanding).
Cost model: 0.1% round-trip for ETF trades (default).
Data: 4+ years via yfinance (2022-01-01 to present).
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import spearmanr

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance required. pip install yfinance")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
# TAXONOMY — same 31 sub-industries from the tracker
# Using ETF proxies for simplicity (one ticker per sub-sector)
# ─────────────────────────────────────────────────────────────
SUBSECTOR_ETFS = {
    "semiconductors":        ["NVDA", "AMD", "INTC", "QCOM", "MU"],
    "semicon_equipment":     ["AMAT", "LRCX", "KLAC", "ASML", "TER"],
    "enterprise_software":   ["MSFT", "CRM", "ORCL", "NOW", "ADBE"],
    "cybersecurity":         ["PANW", "CRWD", "FTNT", "ZS", "OKTA"],
    "it_hardware":           ["AAPL", "DELL", "HPQ", "CSCO", "ANET"],
    "biotech":               ["REGN", "GILD", "VRTX", "MRNA", "BIIB"],
    "medtech_devices":       ["ISRG", "ABT", "MDT", "SYK", "EW"],
    "pharma_large":          ["LLY", "JNJ", "ABBV", "MRK", "PFE"],
    "health_services":       ["UNH", "HCA", "CNC", "ELV", "CI"],
    "megabank":              ["JPM", "BAC", "WFC", "C", "GS"],
    "regional_bank":         ["USB", "PNC", "TFC", "FITB", "KEY"],
    "insurance":             ["BRK-B", "PGR", "AIG", "MET", "ALL"],
    "fintech_payments":      ["V", "MA", "PYPL", "AFRM", "FIS"],
    "oil_integrated":        ["XOM", "CVX", "COP", "EOG", "OXY"],
    "oilfield_services":     ["SLB", "HAL", "BKR", "FTI", "NOV"],
    "midstream_pipelines":   ["WMB", "KMI", "OKE", "ET", "MPLX"],
    "aerospace_defense":     ["LMT", "RTX", "NOC", "GD", "LHX"],
    "heavy_equipment":       ["CAT", "DE", "CMI", "PCAR", "TTC"],
    "industrial_automation": ["EMR", "ROK", "ETN", "IR", "AME"],
    "transportation":        ["UNP", "UPS", "FDX", "CSX", "DAL"],
    "ecommerce_retail":      ["AMZN", "HD", "LOW", "TJX", "COST"],
    "autos_ev":              ["TSLA", "GM", "F", "RIVN", "ON"],
    "restaurants_leisure":   ["MCD", "SBUX", "CMG", "DRI", "YUM"],
    "consumer_staples_food": ["PG", "KO", "PEP", "MDLZ", "GIS"],
    "staples_retail":        ["WMT", "COST", "TGT", "DG", "KR"],
    "mining_metals":         ["FCX", "NEM", "GOLD", "SCCO", "CLF"],
    "chemicals":             ["LIN", "APD", "ECL", "SHW", "DD"],
    "reits_data_towers":     ["PLD", "AMT", "EQIX", "DLR", "SPG"],
    "utilities_electric":    ["NEE", "DUK", "SO", "AEP", "SRE"],
    "big_tech_comm":         ["META", "GOOGL", "NFLX", "DIS", "CMCSA"],
    "telecom":               ["T", "VZ", "TMUS"],
}

MACRO_TICKERS = ["SPY", "^VIX"]

# ─────────────────────────────────────────────────────────────
# DATA DOWNLOAD
# ─────────────────────────────────────────────────────────────
def download_all_data():
    """Download 4+ years of daily data for all sub-sector tickers + macro."""
    all_tickers = sorted(set(
        t for tlist in SUBSECTOR_ETFS.values() for t in tlist
    ) | set(MACRO_TICKERS))

    print(f"Downloading {len(all_tickers)} tickers (2022-01-01 to now)...")
    batch_size = 50
    frames = []
    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i+batch_size]
        print(f"  Batch {i//batch_size+1}: {len(batch)} tickers")
        df = yf.download(batch, start="2022-01-01", progress=False, threads=False)
        if df is not None and not df.empty:
            frames.append(df)
        if i + batch_size < len(all_tickers):
            time.sleep(1)

    if not frames:
        print("ERROR: All downloads failed")
        sys.exit(1)

    data = pd.concat(frames, axis=1)
    data = data.loc[:, ~data.columns.duplicated()]
    print(f"  Got {len(data)} trading days")
    return data


def build_subsector_returns(data, universe=None):
    """Build equal-weight daily return panel for each sub-sector.
    Returns DataFrame: index=date, columns=sub-sector names, values=daily returns.
    """
    if universe is None:
        universe = SUBSECTOR_ETFS

    rets = {}
    for name, tickers in universe.items():
        closes = {}
        for t in tickers:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if t in data["Close"].columns:
                        s = data["Close"][t].dropna()
                        if len(s) >= 60:
                            closes[t] = s
                else:
                    if t in data.columns:
                        s = data[t].dropna()
                        if len(s) >= 60:
                            closes[t] = s
            except Exception:
                continue
        if len(closes) < 2:
            continue
        cdf = pd.DataFrame(closes).dropna()
        if len(cdf) < 60:
            continue
        rets[name] = cdf.pct_change().mean(axis=1)

    panel = pd.DataFrame(rets).dropna()
    return panel


def get_vix(data):
    """Extract VIX series."""
    try:
        if isinstance(data.columns, pd.MultiIndex):
            return data["Close"]["^VIX"].dropna()
        elif "^VIX" in data.columns:
            return data["^VIX"].dropna()
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────
# CORE ROTATION STRATEGY (independent re-implementation)
# ─────────────────────────────────────────────────────────────
def rotation_backtest(returns_panel, lookback=21, top_n=5, rebal_freq=21,
                      cost_rt=0.001, inverse=False, random_seed=None):
    """
    Simple momentum-rotation backtest.
    - Each rebalance: rank sub-sectors by lookback-period return.
    - Go long equal-weight top_n sub-sectors.
    - Hold for rebal_freq days, then re-rank.
    - cost_rt: round-trip cost per rebalance (applied at entry+exit).

    Returns dict with: equity_curve, sharpe, sortino, wr, pf, mdd, n_trades
    """
    dates = returns_panel.index
    n_days = len(dates)
    subsectors = list(returns_panel.columns)
    n_subs = len(subsectors)

    if n_days < lookback + rebal_freq + 10:
        return None

    rng = np.random.RandomState(random_seed) if random_seed is not None else None

    # Pre-compute cumulative returns for lookback ranking
    cum = (1 + returns_panel).cumprod()

    equity = [1.0]
    period_returns = []
    holdings = []
    n_trades = 0

    t = lookback  # start after enough lookback
    while t < n_days:
        # Rank sub-sectors by lookback return
        if t - lookback < 0:
            t += rebal_freq
            continue

        lookback_rets = {}
        for s in subsectors:
            r = cum[s].iloc[t] / cum[s].iloc[t - lookback] - 1
            lookback_rets[s] = r

        ranked = sorted(lookback_rets.items(), key=lambda x: x[1], reverse=True)

        if random_seed is not None and rng is not None:
            # Random: shuffle rankings
            rng.shuffle(ranked)

        if inverse:
            # Inverse: pick the WORST sub-sectors
            selected = [r[0] for r in ranked[-top_n:]]
        else:
            selected = [r[0] for r in ranked[:top_n]]

        # Track turnover
        prev_set = set(holdings)
        new_set = set(selected)
        turnover = len(prev_set.symmetric_difference(new_set))
        n_trades += turnover
        holdings = selected

        # Hold for rebal_freq days
        hold_end = min(t + rebal_freq, n_days)
        if hold_end <= t:
            break

        # Equal-weight return over hold period
        hold_rets = returns_panel[selected].iloc[t:hold_end]
        daily_port_ret = hold_rets.mean(axis=1)

        # Apply cost at rebalance (spread across the period)
        cost_per_day = (cost_rt * turnover / top_n) / max(1, hold_end - t)

        for dr in daily_port_ret:
            eq_prev = equity[-1]
            eq_new = eq_prev * (1 + dr - cost_per_day)
            equity.append(eq_new)
            period_returns.append(dr - cost_per_day)

        t = hold_end

    if len(period_returns) < 20:
        return None

    arr = np.array(period_returns)
    eq = np.array(equity)

    # Compute metrics
    ann_factor = 252
    mean_daily = arr.mean()
    std_daily = arr.std()
    sharpe = mean_daily / std_daily * np.sqrt(ann_factor) if std_daily > 0 else 0

    downside = arr[arr < 0]
    down_std = downside.std() if len(downside) > 0 else 1e-9
    sortino = mean_daily / down_std * np.sqrt(ann_factor) if down_std > 0 else 0

    wins = (arr > 0).sum()
    wr = wins / len(arr)

    gains = arr[arr > 0].sum()
    losses = abs(arr[arr < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')

    # Max drawdown
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak
    mdd = dd.max()

    total_ret = eq[-1] / eq[0] - 1
    n_years = len(arr) / 252
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0

    return {
        "equity_curve": eq,
        "returns": arr,
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "wr": round(wr, 3),
        "pf": round(pf, 2),
        "mdd": round(mdd * 100, 1),
        "total_ret_pct": round(total_ret * 100, 1),
        "ann_ret_pct": round(ann_ret * 100, 1),
        "n_trades": n_trades,
        "n_days": len(arr),
    }


# ─────────────────────────────────────────────────────────────
# TEST 1: RE-IMPLEMENTATION
# ─────────────────────────────────────────────────────────────
def test_reimplementation(returns_panel):
    """Independent re-implementation of rotation signal + backtest."""
    print("\n" + "="*60)
    print("TEST 1: RE-IMPLEMENTATION (independent rotation backtest)")
    print("="*60)

    result = rotation_backtest(returns_panel, lookback=21, top_n=5, rebal_freq=21, cost_rt=0.001)
    if result is None:
        return {"pass": False, "reason": "Insufficient data"}

    # Compare with original's Sharpe of 3.94 (long-only, 10 folds)
    original_sharpe = 3.94
    our_sharpe = result["sharpe"]
    diff_pct = abs(our_sharpe - original_sharpe) / max(abs(original_sharpe), 0.01) * 100

    passed = diff_pct <= 30  # within 30%

    print(f"  Re-implemented Sharpe: {our_sharpe:.2f}")
    print(f"  Original Sharpe:       {original_sharpe:.2f}")
    print(f"  Difference:            {diff_pct:.0f}%")
    print(f"  Metrics: Sortino={result['sortino']}, WR={result['wr']:.1%}, PF={result['pf']}, MDD={result['mdd']:.1f}%")
    print(f"  Trades: {result['n_trades']}, Days: {result['n_days']}")

    note = ""
    if not passed:
        note = (f"DIVERGENCE: Re-implemented Sharpe ({our_sharpe:.2f}) differs from original ({original_sharpe:.2f}) by {diff_pct:.0f}%. "
                f"Original was likely overfit (10 folds only) or used different methodology.")
        print(f"  NOTE: {note}")

    return {
        "pass": passed,
        "reimpl_sharpe": our_sharpe,
        "original_sharpe": original_sharpe,
        "diff_pct": round(diff_pct, 1),
        "result": result,
        "note": note,
    }


# ─────────────────────────────────────────────────────────────
# TEST 2: INVERSE SIGNAL
# ─────────────────────────────────────────────────────────────
def test_inverse(returns_panel):
    """Trade the opposite of what the signal says."""
    print("\n" + "="*60)
    print("TEST 2: INVERSE SIGNAL (long worst-ranked, short best)")
    print("="*60)

    normal = rotation_backtest(returns_panel, lookback=21, top_n=5, rebal_freq=21, cost_rt=0.001)
    inverse = rotation_backtest(returns_panel, lookback=21, top_n=5, rebal_freq=21, cost_rt=0.001, inverse=True)

    if normal is None or inverse is None:
        return {"pass": False, "reason": "Insufficient data"}

    ratio = inverse["sharpe"] / max(abs(normal["sharpe"]), 0.01) if normal["sharpe"] != 0 else float('inf')
    passed = ratio < 0.50

    print(f"  Normal Sharpe:  {normal['sharpe']:.2f}")
    print(f"  Inverse Sharpe: {inverse['sharpe']:.2f}")
    print(f"  Ratio:          {ratio:.2f} (threshold: < 0.50)")
    print(f"  Inverse: Sortino={inverse['sortino']}, WR={inverse['wr']:.1%}, PF={inverse['pf']}")

    return {
        "pass": passed,
        "normal_sharpe": normal["sharpe"],
        "inverse_sharpe": inverse["sharpe"],
        "ratio": round(ratio, 2),
    }


# ─────────────────────────────────────────────────────────────
# TEST 3: RANDOM TIMING (permutation test)
# ─────────────────────────────────────────────────────────────
def test_random_timing(returns_panel, n_perms=1000):
    """Run 1000 permutation tests with random rotation signals."""
    print("\n" + "="*60)
    print(f"TEST 3: RANDOM TIMING ({n_perms} permutation tests)")
    print("="*60)

    actual = rotation_backtest(returns_panel, lookback=21, top_n=5, rebal_freq=21, cost_rt=0.001)
    if actual is None:
        return {"pass": False, "reason": "Insufficient data"}

    actual_sharpe = actual["sharpe"]
    random_sharpes = []

    print(f"  Running {n_perms} random permutations...", end="", flush=True)
    for i in range(n_perms):
        r = rotation_backtest(returns_panel, lookback=21, top_n=5, rebal_freq=21,
                              cost_rt=0.001, random_seed=i+42)
        if r is not None:
            random_sharpes.append(r["sharpe"])
        if (i+1) % 200 == 0:
            print(f" {i+1}", end="", flush=True)
    print()

    if not random_sharpes:
        return {"pass": False, "reason": "No random results"}

    random_sharpes = np.array(random_sharpes)
    p_value = (random_sharpes >= actual_sharpe).mean()
    passed = p_value < 0.05

    print(f"  Actual Sharpe: {actual_sharpe:.2f}")
    print(f"  Random mean:   {random_sharpes.mean():.2f} +/- {random_sharpes.std():.2f}")
    print(f"  Random median: {np.median(random_sharpes):.2f}")
    print(f"  Random max:    {random_sharpes.max():.2f}")
    print(f"  p-value:       {p_value:.4f} (threshold: < 0.05)")
    print(f"  Percentile:    {(1-p_value)*100:.1f}th")

    return {
        "pass": passed,
        "actual_sharpe": actual_sharpe,
        "p_value": round(p_value, 4),
        "random_mean": round(random_sharpes.mean(), 2),
        "random_std": round(random_sharpes.std(), 2),
        "random_max": round(random_sharpes.max(), 2),
        "percentile": round((1-p_value)*100, 1),
    }


# ─────────────────────────────────────────────────────────────
# TEST 4: SUB-PERIOD STABILITY
# ─────────────────────────────────────────────────────────────
def test_subperiod_stability(returns_panel):
    """Split into 4 equal sub-periods. All must have positive Sharpe."""
    print("\n" + "="*60)
    print("TEST 4: SUB-PERIOD STABILITY (4 equal periods)")
    print("="*60)

    n = len(returns_panel)
    quarter = n // 4
    periods = []

    for i in range(4):
        start = i * quarter
        end = (i + 1) * quarter if i < 3 else n
        sub = returns_panel.iloc[start:end]
        r = rotation_backtest(sub, lookback=21, top_n=5, rebal_freq=21, cost_rt=0.001)
        date_range = f"{sub.index[0].strftime('%Y-%m-%d')} to {sub.index[-1].strftime('%Y-%m-%d')}"
        if r is not None:
            periods.append({"period": i+1, "dates": date_range, **r})
            print(f"  P{i+1} ({date_range}): Sharpe={r['sharpe']:.2f}, Sortino={r['sortino']:.2f}, "
                  f"WR={r['wr']:.1%}, PF={r['pf']}, MDD={r['mdd']:.1f}%, Ret={r['total_ret_pct']:+.1f}%")
        else:
            periods.append({"period": i+1, "dates": date_range, "sharpe": -999, "note": "insufficient data"})
            print(f"  P{i+1} ({date_range}): INSUFFICIENT DATA")

    sharpes = [p["sharpe"] for p in periods if p["sharpe"] != -999]
    all_positive = all(s > 0 for s in sharpes)
    n_negative = sum(1 for s in sharpes if s <= 0)

    print(f"  All positive Sharpe: {all_positive} ({n_negative} negative)")

    return {
        "pass": all_positive and len(sharpes) == 4,
        "sharpes": [round(s, 2) for s in sharpes],
        "periods": periods,
        "n_negative": n_negative,
    }


# ─────────────────────────────────────────────────────────────
# TEST 5: TOP-N REMOVAL
# ─────────────────────────────────────────────────────────────
def test_topn_removal(returns_panel, data):
    """Remove the 3 best-performing sub-sectors and re-run."""
    print("\n" + "="*60)
    print("TEST 5: TOP-N REMOVAL (remove 3 best sub-sectors)")
    print("="*60)

    # Full backtest
    full = rotation_backtest(returns_panel, lookback=21, top_n=5, rebal_freq=21, cost_rt=0.001)
    if full is None:
        return {"pass": False, "reason": "Insufficient data"}

    # Find top 3 sub-sectors by total return
    total_rets = {}
    for s in returns_panel.columns:
        cum = (1 + returns_panel[s]).cumprod()
        total_rets[s] = cum.iloc[-1] / cum.iloc[0] - 1

    sorted_subs = sorted(total_rets.items(), key=lambda x: x[1], reverse=True)
    top3 = [s[0] for s in sorted_subs[:3]]
    top3_rets = [f"{s[0]}: {s[1]*100:+.1f}%" for s in sorted_subs[:3]]

    print(f"  Top 3 sub-sectors removed: {', '.join(top3)}")
    print(f"  Their total returns: {', '.join(top3_rets)}")

    # Remove and re-run
    reduced = returns_panel.drop(columns=top3, errors="ignore")
    reduced_result = rotation_backtest(reduced, lookback=21, top_n=5, rebal_freq=21, cost_rt=0.001)

    if reduced_result is None:
        return {"pass": False, "reason": "Insufficient data after removal"}

    drop_pct = (full["sharpe"] - reduced_result["sharpe"]) / max(abs(full["sharpe"]), 0.01) * 100
    passed = drop_pct < 50

    print(f"  Full Sharpe:    {full['sharpe']:.2f}")
    print(f"  Reduced Sharpe: {reduced_result['sharpe']:.2f}")
    print(f"  Drop:           {drop_pct:.0f}% (threshold: < 50%)")
    print(f"  Reduced: Sortino={reduced_result['sortino']}, WR={reduced_result['wr']:.1%}")

    return {
        "pass": passed,
        "full_sharpe": full["sharpe"],
        "reduced_sharpe": reduced_result["sharpe"],
        "drop_pct": round(drop_pct, 1),
        "removed": top3,
    }


# ─────────────────────────────────────────────────────────────
# TEST 6: PARAMETER SENSITIVITY
# ─────────────────────────────────────────────────────────────
def test_param_sensitivity(returns_panel):
    """Test grid of parameters. >80% of combos must produce Sharpe > 0.30."""
    print("\n" + "="*60)
    print("TEST 6: PARAMETER SENSITIVITY (lookback x top_n x rebal grid)")
    print("="*60)

    lookbacks = [5, 10, 21, 42, 63]
    top_ns = [3, 5, 7, 10]
    rebal_freqs = [5, 10, 21]  # weekly, biweekly, monthly

    results = []
    total = len(lookbacks) * len(top_ns) * len(rebal_freqs)
    count = 0

    for lb in lookbacks:
        for tn in top_ns:
            for rf in rebal_freqs:
                count += 1
                r = rotation_backtest(returns_panel, lookback=lb, top_n=tn,
                                      rebal_freq=rf, cost_rt=0.001)
                if r is not None:
                    results.append({
                        "lookback": lb, "top_n": tn, "rebal": rf,
                        "sharpe": r["sharpe"], "sortino": r["sortino"],
                        "wr": r["wr"], "pf": r["pf"],
                    })

    if not results:
        return {"pass": False, "reason": "No results"}

    above_threshold = sum(1 for r in results if r["sharpe"] > 0.30)
    pct_above = above_threshold / len(results) * 100
    passed = pct_above >= 80

    # Summary stats
    sharpes = [r["sharpe"] for r in results]
    print(f"  Tested: {len(results)}/{total} parameter combos")
    print(f"  Sharpe > 0.30: {above_threshold}/{len(results)} ({pct_above:.0f}%)")
    print(f"  Sharpe range: [{min(sharpes):.2f}, {max(sharpes):.2f}]")
    print(f"  Sharpe mean: {np.mean(sharpes):.2f}, median: {np.median(sharpes):.2f}")
    print(f"  Threshold: >= 80%")

    # Best and worst combos
    best = max(results, key=lambda x: x["sharpe"])
    worst = min(results, key=lambda x: x["sharpe"])
    print(f"  Best:  lb={best['lookback']}, top_n={best['top_n']}, rebal={best['rebal']} -> Sharpe={best['sharpe']:.2f}")
    print(f"  Worst: lb={worst['lookback']}, top_n={worst['top_n']}, rebal={worst['rebal']} -> Sharpe={worst['sharpe']:.2f}")

    return {
        "pass": passed,
        "pct_above_030": round(pct_above, 1),
        "n_tested": len(results),
        "sharpe_mean": round(np.mean(sharpes), 2),
        "sharpe_median": round(np.median(sharpes), 2),
        "sharpe_min": round(min(sharpes), 2),
        "sharpe_max": round(max(sharpes), 2),
        "best": best,
        "worst": worst,
    }


# ─────────────────────────────────────────────────────────────
# TEST 7: LEAD-LAG (alpha test)
# ─────────────────────────────────────────────────────────────
def test_lead_lag(returns_panel):
    """Does PAST rotation predict FUTURE returns?
    Use rotation scores from T-21d to pick sub-sectors, measure performance T to T+21d.
    """
    print("\n" + "="*60)
    print("TEST 7: LEAD-LAG (does past rotation predict future returns?)")
    print("="*60)

    cum = (1 + returns_panel).cumprod()
    dates = returns_panel.index
    n = len(dates)
    subsectors = list(returns_panel.columns)

    lags = [0, 5, 10, 21, 42]  # lag in days between signal and trade
    lag_results = {}

    for lag in lags:
        ics = []
        for t in range(63, n - 21 - lag, 21):
            # Signal: 21d lookback return at time (t - lag)
            signal_t = t - lag
            if signal_t - 21 < 0:
                continue

            signals = {}
            fwd_rets = {}
            for s in subsectors:
                signals[s] = cum[s].iloc[signal_t] / cum[s].iloc[signal_t - 21] - 1
                if t + 21 < n:
                    fwd_rets[s] = cum[s].iloc[t + 21] / cum[s].iloc[t] - 1

            if len(signals) < 10 or len(fwd_rets) < 10:
                continue

            # Cross-sectional IC
            common = set(signals.keys()) & set(fwd_rets.keys())
            if len(common) < 10:
                continue

            sig_arr = [signals[s] for s in common]
            fwd_arr = [fwd_rets[s] for s in common]
            ic, _ = spearmanr(sig_arr, fwd_arr)
            if not np.isnan(ic):
                ics.append(ic)

        if ics:
            mean_ic = np.mean(ics)
            lag_results[lag] = {
                "mean_ic": round(mean_ic, 3),
                "median_ic": round(np.median(ics), 3),
                "n_obs": len(ics),
                "pct_positive": round(np.mean([1 if ic > 0 else 0 for ic in ics]) * 100, 1),
            }
            print(f"  Lag {lag:2d}d: IC={mean_ic:.3f}, median={np.median(ics):.3f}, "
                  f"positive={np.mean([1 if ic > 0 else 0 for ic in ics])*100:.0f}%, n={len(ics)}")

    # The key question: does lagged signal (lag>0) still have positive IC?
    lag0_ic = lag_results.get(0, {}).get("mean_ic", 0)
    lag21_ic = lag_results.get(21, {}).get("mean_ic", 0)

    # Pass if lag=21 IC is still meaningfully positive (>0.03)
    passed = lag21_ic > 0.03

    print(f"  Lag 0 IC:  {lag0_ic:.3f} (contemporaneous — this is the 'cheating' baseline)")
    print(f"  Lag 21 IC: {lag21_ic:.3f} (this is the alpha test)")
    print(f"  PASS threshold: lagged IC > 0.03")

    return {
        "pass": passed,
        "lag_results": lag_results,
        "lag0_ic": lag0_ic,
        "lag21_ic": lag21_ic,
    }


# ─────────────────────────────────────────────────────────────
# TEST 8: TRANSACTION COST SENSITIVITY
# ─────────────────────────────────────────────────────────────
def test_cost_sensitivity(returns_panel):
    """At what cost level does the strategy break even?"""
    print("\n" + "="*60)
    print("TEST 8: TRANSACTION COST SENSITIVITY (breakeven cost)")
    print("="*60)

    costs = [0.0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01]
    cost_results = []

    for c in costs:
        r = rotation_backtest(returns_panel, lookback=21, top_n=5, rebal_freq=21, cost_rt=c)
        if r is not None:
            cost_results.append({"cost_pct": c * 100, "sharpe": r["sharpe"],
                                 "ann_ret_pct": r["ann_ret_pct"]})
            print(f"  Cost {c*100:.2f}%: Sharpe={r['sharpe']:.2f}, Ann.Ret={r['ann_ret_pct']:+.1f}%")

    # Find breakeven (where Sharpe crosses 0)
    breakeven = None
    for i in range(1, len(cost_results)):
        if cost_results[i-1]["sharpe"] > 0 and cost_results[i]["sharpe"] <= 0:
            # Linear interpolation
            s1 = cost_results[i-1]["sharpe"]
            s2 = cost_results[i]["sharpe"]
            c1 = cost_results[i-1]["cost_pct"]
            c2 = cost_results[i]["cost_pct"]
            breakeven = c1 + (c2 - c1) * s1 / (s1 - s2)
            break

    if breakeven is None:
        if cost_results and cost_results[-1]["sharpe"] > 0:
            breakeven = cost_results[-1]["cost_pct"]
            print(f"  Strategy still profitable at highest tested cost ({breakeven:.2f}%)")
        elif cost_results and cost_results[0]["sharpe"] <= 0:
            breakeven = 0
            print(f"  Strategy not profitable even at zero cost!")

    if breakeven is not None:
        print(f"  BREAKEVEN COST: {breakeven:.3f}% round-trip")

    return {
        "breakeven_pct": round(breakeven, 3) if breakeven is not None else None,
        "cost_results": cost_results,
    }


# ─────────────────────────────────────────────────────────────
# REGIME STRATIFICATION
# ─────────────────────────────────────────────────────────────
def regime_stratification(returns_panel, vix_series):
    """Split backtest by VIX regime and report Sharpe per regime."""
    print("\n" + "="*60)
    print("REGIME STRATIFICATION (VIX: low <15, normal 15-25, high >25)")
    print("="*60)

    if vix_series is None or len(vix_series) < 60:
        print("  VIX data not available")
        return {}

    # Align VIX to returns dates
    common_dates = returns_panel.index.intersection(vix_series.index)
    vix_aligned = vix_series.reindex(common_dates)

    # Classify each day
    regimes = {"low": vix_aligned < 15, "normal": (vix_aligned >= 15) & (vix_aligned < 25), "high": vix_aligned >= 25}

    results = {}
    for regime_name, mask in regimes.items():
        regime_dates = mask[mask].index
        if len(regime_dates) < 50:
            print(f"  {regime_name}: too few days ({len(regime_dates)})")
            continue

        sub = returns_panel.reindex(regime_dates).dropna()
        if len(sub) < 50:
            continue

        r = rotation_backtest(sub, lookback=21, top_n=5, rebal_freq=21, cost_rt=0.001)
        if r is not None:
            results[regime_name] = r
            print(f"  {regime_name:8s} ({len(sub):4d} days): Sharpe={r['sharpe']:.2f}, "
                  f"Sortino={r['sortino']:.2f}, WR={r['wr']:.1%}, Ret={r['total_ret_pct']:+.1f}%")

    return results


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("SUB-SECTOR ROTATION — COMPREHENSIVE ADVERSARIAL VALIDATION")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Download data
    data = download_all_data()

    # Build returns panel
    returns_panel = build_subsector_returns(data)
    print(f"\nReturns panel: {len(returns_panel)} days, {len(returns_panel.columns)} sub-sectors")
    print(f"Date range: {returns_panel.index[0].strftime('%Y-%m-%d')} to {returns_panel.index[-1].strftime('%Y-%m-%d')}")
    print(f"Sub-sectors: {', '.join(sorted(returns_panel.columns))}")

    # VIX for regime stratification
    vix = get_vix(data)

    # ─── Run all 8 tests ───
    t1 = test_reimplementation(returns_panel)
    t2 = test_inverse(returns_panel)
    t3 = test_random_timing(returns_panel, n_perms=1000)
    t4 = test_subperiod_stability(returns_panel)
    t5 = test_topn_removal(returns_panel, data)
    t6 = test_param_sensitivity(returns_panel)
    t7 = test_lead_lag(returns_panel)
    t8 = test_cost_sensitivity(returns_panel)

    # Regime stratification (bonus)
    regime_results = regime_stratification(returns_panel, vix)

    # ─── FINAL SUMMARY ───
    print("\n" + "=" * 70)
    print("=== ADVERSARIAL RESULTS ===")
    print("=" * 70)

    def status(t):
        return "PASS" if t.get("pass", False) else "FAIL"

    results = []

    r1 = f"Test 1 (Re-implementation): {status(t1)} — Sharpe {t1.get('reimpl_sharpe', '?')} vs original {t1.get('original_sharpe', '?')} (diff {t1.get('diff_pct', '?')}%)"
    print(r1)
    results.append(t1.get("pass", False))

    r2 = f"Test 2 (Inverse signal):    {status(t2)} — ratio {t2.get('ratio', '?')}"
    print(r2)
    results.append(t2.get("pass", False))

    r3 = f"Test 3 (Random timing):     {status(t3)} — p={t3.get('p_value', '?')}"
    print(r3)
    results.append(t3.get("pass", False))

    sharpes_str = str(t4.get("sharpes", []))
    r4 = f"Test 4 (Sub-period):        {status(t4)} — {sharpes_str}"
    print(r4)
    results.append(t4.get("pass", False))

    r5 = f"Test 5 (Top-N removal):     {status(t5)} — drop {t5.get('drop_pct', '?')}%"
    print(r5)
    results.append(t5.get("pass", False))

    r6 = f"Test 6 (Param sensitivity): {status(t6)} — {t6.get('pct_above_030', '?')}% > 0.30"
    print(r6)
    results.append(t6.get("pass", False))

    r7 = f"Test 7 (Lead-lag):          {status(t7)} — lagged IC {t7.get('lag21_ic', '?')}"
    print(r7)
    results.append(t7.get("pass", False))

    be = t8.get("breakeven_pct")
    r8 = f"Test 8 (Cost breakeven):    breakeven at {be}%" if be is not None else "Test 8 (Cost breakeven): could not determine"
    print(r8)

    n_pass = sum(results)
    n_total = len(results)
    print(f"\nOVERALL: {n_pass}/{n_total} PASS")

    if regime_results:
        print(f"\nRegime Sharpes: ", end="")
        for rname, rdata in regime_results.items():
            print(f"{rname}={rdata['sharpe']:.2f}  ", end="")
        print()

    # Overall assessment
    print("\n" + "-"*70)
    if n_pass >= 6:
        print("VERDICT: Strategy shows ROBUST edge across adversarial tests.")
    elif n_pass >= 4:
        print("VERDICT: Strategy shows MODERATE edge but has weaknesses. Proceed with caution.")
    else:
        print("VERDICT: Strategy FAILS adversarial validation. Edge is likely spurious.")
    print("-"*70)

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
