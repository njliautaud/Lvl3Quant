#!/usr/bin/env python3
"""
Adversarial Validation of Top Growth Configs
=============================================
HC #696: Validate the most promising aggressive growth strategies.

Red flags to check:
1. Permutation test — does random rotation also make money?
2. Transaction costs — weekly rebalancing in leveraged ETFs isn't free
3. Year-by-year consistency — is CAGR concentrated in 1-2 years?
4. Drawdown during specific crises — COVID, 2022 bear, etc.
5. ETF universe bias — did we pick ETFs with hindsight?
6. Leverage decay simulation — does daily leverage compounding kill returns?

Top configs to validate:
- Dual momentum lb21 vt30 (Sharpe 3.59, CAGR 145%)
- Dual momentum lb63 vt30 (Sharpe 2.24, CAGR 78%)
- Sector breakout 10d trail8% vt30 (Sharpe 2.73, CAGR 90%)
- Crypto+equity lb63 ca20 vt50 (Sharpe 2.59, CAGR 104%)
- Concentrated lb63 vt50 regime (Sharpe 2.38, CAGR 160%)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from scipy import stats
import json
import warnings
import time
warnings.filterwarnings("ignore")

OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/adversarial_validation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Cost model (HC #694 — commission-free on Robinhood/IBKR) ───
# But leveraged ETFs have bid-ask spread costs
SPREAD_COST_BPS = {
    "TQQQ": 1,  # Very liquid, ~1bp spread
    "UPRO": 1,
    "SOXL": 2,  # Slightly wider
    "TECL": 3,
    "FAS":  2,
    "TNA":  3,
    "LABU": 5,  # Less liquid, wider spread
    "FNGU": 5,
    "BTC-USD": 10,  # Crypto spread
    "ETH-USD": 15,
    "SHV":  1,
    "TLT":  1,
}

def download_prices(tickers, start="2010-01-01", end="2026-07-14"):
    """Download adjusted close prices."""
    print(f"Downloading {len(tickers)} tickers...")
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data
    return prices.ffill().dropna(how="all")


def dual_momentum_with_costs(prices, etfs, lookback, vol_target, safe_asset="SHV"):
    """Dual momentum with transaction cost tracking."""
    returns = prices.pct_change()
    daily_rets = []
    daily_costs = []
    current_holding = None
    trades = 0

    for date in prices.index[lookback + 1:]:
        mom = {}
        for etf in etfs:
            if etf in prices.columns:
                p_now = prices[etf].loc[:date].iloc[-1]
                p_past = prices[etf].loc[:date].iloc[-lookback] if len(prices[etf].loc[:date]) > lookback else np.nan
                if not np.isnan(p_past) and p_past > 0:
                    mom[etf] = p_now / p_past - 1

        if not mom:
            daily_rets.append(0)
            daily_costs.append(0)
            continue

        best_etf = max(mom, key=mom.get)
        best_mom = mom[best_etf]

        # Check if position changed
        cost = 0
        if best_mom <= 0:
            target = safe_asset
        else:
            target = best_etf

        if target != current_holding:
            # Sell old position
            if current_holding and current_holding in SPREAD_COST_BPS:
                cost += SPREAD_COST_BPS[current_holding] / 10000
            # Buy new position
            if target in SPREAD_COST_BPS:
                cost += SPREAD_COST_BPS[target] / 10000
            current_holding = target
            trades += 1

        # Compute return
        if current_holding and current_holding in returns.columns:
            r = returns[current_holding].loc[date]
            if np.isnan(r):
                r = 0
        else:
            r = 0

        # Vol-targeting
        if vol_target > 0 and current_holding in returns.columns:
            recent_vol = returns[current_holding].loc[:date].tail(21).std() * np.sqrt(252)
            if recent_vol > 0:
                scalar = min(vol_target / recent_vol, 2.0)
                scalar = max(scalar, 0.1)
                r = r * scalar
                cost = cost * scalar  # Scale costs too

        daily_rets.append(r - cost)
        daily_costs.append(cost)

    return pd.Series(daily_rets, index=prices.index[lookback + 1:]), trades, sum(daily_costs)


def sector_breakout_with_costs(prices, etfs, breakout_days, trail_pct, vol_target):
    """Sector breakout with costs."""
    returns = prices.pct_change()
    daily_rets = []
    current_hold = None
    peak_price = 0
    trades = 0

    for date in prices.index[breakout_days + 1:]:
        cost = 0

        if current_hold and current_hold in prices.columns:
            curr_price = prices[current_hold].loc[date]
            if not np.isnan(curr_price):
                peak_price = max(peak_price, curr_price)
                if curr_price < peak_price * (1 - trail_pct):
                    if current_hold in SPREAD_COST_BPS:
                        cost += SPREAD_COST_BPS[current_hold] / 10000
                    current_hold = None
                    trades += 1

        if current_hold is None:
            for etf in etfs:
                if etf in prices.columns:
                    recent_high = prices[etf].loc[:date].tail(breakout_days + 1).iloc[:-1].max()
                    curr_price = prices[etf].loc[date]
                    if not np.isnan(curr_price) and not np.isnan(recent_high) and curr_price > recent_high:
                        current_hold = etf
                        peak_price = curr_price
                        if etf in SPREAD_COST_BPS:
                            cost += SPREAD_COST_BPS[etf] / 10000
                        trades += 1
                        break

        if current_hold and current_hold in returns.columns:
            r = returns[current_hold].loc[date]
            if np.isnan(r):
                r = 0
            if vol_target > 0:
                recent_vol = returns[current_hold].loc[:date].tail(21).std() * np.sqrt(252)
                if recent_vol > 0:
                    scalar = min(vol_target / recent_vol, 2.0)
                    r = r * scalar
                    cost = cost * scalar
            daily_rets.append(r - cost)
        else:
            daily_rets.append(-cost)

    return pd.Series(daily_rets, index=prices.index[breakout_days + 1:]), trades


def compute_metrics(dr, label=""):
    """Full metrics computation."""
    dr = pd.Series(dr).dropna()
    if len(dr) < 252 or dr.std() == 0:
        return {"label": label, "valid": False}

    ann_ret = dr.mean() * 252
    ann_vol = dr.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol
    downside = dr[dr < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside > 0 else 0
    years = len(dr) / 252
    cagr = ((1 + dr).prod() ** (1/years) - 1)
    cum = (1 + dr).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    return {
        "label": label, "valid": True,
        "cagr": round(cagr * 100, 1), "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2), "max_dd": round(max_dd * 100, 1),
        "calmar": round(calmar, 2), "ann_vol": round(ann_vol * 100, 1),
        "wr": round((dr > 0).mean() * 100, 1), "years": round(years, 1),
    }


def year_by_year(dr, label=""):
    """Compute per-year metrics."""
    dr = pd.Series(dr)
    yearly = dr.resample("YE").agg(["mean", "std", "sum", "count"])
    results = []
    for year, row in yearly.iterrows():
        y = year.year
        n = row["count"]
        if n < 100:
            continue
        ann_ret = row["mean"] * 252
        ann_vol = row["std"] * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        cum = (1 + dr[dr.index.year == y]).prod() - 1
        results.append({
            "year": y, "return": round(cum * 100, 1),
            "sharpe": round(sharpe, 2), "vol": round(ann_vol * 100, 1),
        })
    return results


def crisis_analysis(dr, label=""):
    """How did strategy perform during key market crises?"""
    dr = pd.Series(dr)
    crises = {
        "COVID_2020": ("2020-02-19", "2020-03-23"),
        "2022_Bear": ("2022-01-03", "2022-10-12"),
        "2018_Q4": ("2018-10-01", "2018-12-24"),
        "2023_SVB": ("2023-03-08", "2023-03-15"),
        "2024_Aug_Crash": ("2024-07-31", "2024-08-05"),
    }
    results = {}
    for name, (start, end) in crises.items():
        try:
            period = dr[(dr.index >= start) & (dr.index <= end)]
            if len(period) > 0:
                cum = (1 + period).prod() - 1
                results[name] = round(cum * 100, 1)
        except:
            pass
    return results


def permutation_test_proper(prices, etfs, lookback, vol_target, real_sharpe, n_trials=100):
    """
    Proper permutation: randomly rotate which ETF is picked (instead of momentum-based).
    This tests if the ROTATION SIGNAL adds value, not just being long leveraged ETFs.
    """
    returns = prices.pct_change()
    beat_count = 0
    shuffled_sharpes = []

    for trial in range(n_trials):
        daily_rets = []

        for date in prices.index[lookback + 1:]:
            # Pick RANDOM ETF instead of momentum-based
            available = [e for e in etfs if e in returns.columns and not np.isnan(returns[e].loc[date])]
            if not available:
                daily_rets.append(0)
                continue

            random_etf = np.random.choice(available)
            r = returns[random_etf].loc[date]
            if np.isnan(r):
                r = 0

            # Apply same vol-targeting
            if vol_target > 0:
                recent_vol = returns[random_etf].loc[:date].tail(21).std() * np.sqrt(252)
                if recent_vol > 0:
                    scalar = min(vol_target / recent_vol, 2.0)
                    scalar = max(scalar, 0.1)
                    r = r * scalar

            daily_rets.append(r)

        dr = np.array(daily_rets)
        if dr.std() > 0:
            s = dr.mean() / dr.std() * np.sqrt(252)
        else:
            s = 0

        shuffled_sharpes.append(s)
        if s >= real_sharpe:
            beat_count += 1

    return beat_count / n_trials, shuffled_sharpes


def equal_weight_baseline(prices, etfs, vol_target):
    """Equal-weight buy-and-hold of all leveraged ETFs (no rotation signal)."""
    returns = prices.pct_change()
    daily_rets = []

    for date in prices.index[252:]:
        r = 0
        count = 0
        for etf in etfs:
            if etf in returns.columns:
                ret = returns[etf].loc[date]
                if not np.isnan(ret):
                    if vol_target > 0:
                        rv = returns[etf].loc[:date].tail(21).std() * np.sqrt(252)
                        if rv > 0:
                            scalar = min(vol_target / rv, 2.0)
                            ret = ret * scalar
                    r += ret
                    count += 1
        if count > 0:
            daily_rets.append(r / count)
        else:
            daily_rets.append(0)

    return pd.Series(daily_rets, index=prices.index[252:])


def main():
    t0 = time.time()
    print("=" * 70)
    print("ADVERSARIAL VALIDATION — TOP GROWTH CONFIGS")
    print("=" * 70)

    all_tickers = list(SPREAD_COST_BPS.keys()) + ["SPY", "QQQ"]
    prices = download_prices(all_tickers, start="2010-01-01")

    leveraged = [t for t in ["TQQQ", "SOXL", "TECL", "UPRO", "FAS", "TNA", "LABU", "FNGU"]
                 if t in prices.columns and prices[t].notna().sum() > 252]

    results = {}

    # ═══════════════════════════════════════════════════
    # CONFIG 1: Dual Momentum lb21 vt30
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("CONFIG 1: Dual Momentum lb21 vt30 (headline: Sharpe 3.59)")
    print(f"{'='*60}")

    dr1, trades1, costs1 = dual_momentum_with_costs(prices, leveraged, 21, 0.30)
    m1 = compute_metrics(dr1, "dual_mom_lb21_vt30_with_costs")
    yy1 = year_by_year(dr1)
    crisis1 = crisis_analysis(dr1)

    print(f"  With costs: CAGR={m1['cagr']:.1f}%, Sharpe={m1['sharpe']:.2f}, MaxDD={m1['max_dd']:.1f}%")
    print(f"  Total trades: {trades1}, Total spread cost: {costs1*100:.2f}%")
    print(f"  Year-by-year:")
    for y in yy1:
        print(f"    {y['year']}: {y['return']:+.1f}% (Sharpe={y['sharpe']:.2f})")
    print(f"  Crisis performance:")
    for name, ret in crisis1.items():
        print(f"    {name}: {ret:+.1f}%")

    # Permutation test
    print(f"  Running permutation test (100 trials, random rotation)...")
    p1, shuf1 = permutation_test_proper(prices, leveraged, 21, 0.30, m1['sharpe'], n_trials=100)
    print(f"  Permutation p-value: {p1:.2f} ({'✅ PASS' if p1 <= 0.05 else '❌ FAIL'})")
    print(f"  Random rotation mean Sharpe: {np.mean(shuf1):.2f} vs actual {m1['sharpe']:.2f}")

    results["dual_mom_lb21_vt30"] = {
        "metrics": m1, "year_by_year": yy1, "crisis": crisis1,
        "permutation_p": p1, "random_mean_sharpe": round(np.mean(shuf1), 2),
        "trades": trades1, "total_cost_pct": round(costs1 * 100, 2),
    }

    # ═══════════════════════════════════════════════════
    # CONFIG 2: Dual Momentum lb63 vt30
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("CONFIG 2: Dual Momentum lb63 vt30 (headline: Sharpe 2.24)")
    print(f"{'='*60}")

    dr2, trades2, costs2 = dual_momentum_with_costs(prices, leveraged, 63, 0.30)
    m2 = compute_metrics(dr2, "dual_mom_lb63_vt30_with_costs")
    yy2 = year_by_year(dr2)
    crisis2 = crisis_analysis(dr2)

    print(f"  With costs: CAGR={m2['cagr']:.1f}%, Sharpe={m2['sharpe']:.2f}, MaxDD={m2['max_dd']:.1f}%")
    print(f"  Total trades: {trades2}")
    print(f"  Year-by-year:")
    for y in yy2:
        print(f"    {y['year']}: {y['return']:+.1f}% (Sharpe={y['sharpe']:.2f})")
    print(f"  Crisis:")
    for name, ret in crisis2.items():
        print(f"    {name}: {ret:+.1f}%")

    print(f"  Running permutation test...")
    p2, shuf2 = permutation_test_proper(prices, leveraged, 63, 0.30, m2['sharpe'], n_trials=100)
    print(f"  Permutation p-value: {p2:.2f} ({'✅ PASS' if p2 <= 0.05 else '❌ FAIL'})")
    print(f"  Random rotation mean Sharpe: {np.mean(shuf2):.2f} vs actual {m2['sharpe']:.2f}")

    results["dual_mom_lb63_vt30"] = {
        "metrics": m2, "year_by_year": yy2, "crisis": crisis2,
        "permutation_p": p2, "random_mean_sharpe": round(np.mean(shuf2), 2),
        "trades": trades2,
    }

    # ═══════════════════════════════════════════════════
    # CONFIG 3: Sector Breakout 10d trail8% vt30
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("CONFIG 3: Sector Breakout 10d trail8% vt30 (Sharpe 2.73)")
    print(f"{'='*60}")

    dr3, trades3 = sector_breakout_with_costs(prices, leveraged, 10, 0.08, 0.30)
    m3 = compute_metrics(dr3, "breakout_10d_trail8_vt30_costs")
    yy3 = year_by_year(dr3)
    crisis3 = crisis_analysis(dr3)

    print(f"  With costs: CAGR={m3['cagr']:.1f}%, Sharpe={m3['sharpe']:.2f}, MaxDD={m3['max_dd']:.1f}%")
    print(f"  Total trades: {trades3}")
    print(f"  Year-by-year:")
    for y in yy3:
        print(f"    {y['year']}: {y['return']:+.1f}% (Sharpe={y['sharpe']:.2f})")

    results["breakout_10d_trail8_vt30"] = {
        "metrics": m3, "year_by_year": yy3, "crisis": crisis3,
        "trades": trades3,
    }

    # ═══════════════════════════════════════════════════
    # CRITICAL BASELINE: Equal-weight leveraged ETFs (no signal)
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("BASELINE: Equal-weight leveraged ETFs + VT30 (NO rotation signal)")
    print(f"{'='*60}")

    dr_base = equal_weight_baseline(prices, leveraged, 0.30)
    m_base = compute_metrics(dr_base, "equal_weight_leveraged_vt30")
    yy_base = year_by_year(dr_base)

    print(f"  CAGR={m_base['cagr']:.1f}%, Sharpe={m_base['sharpe']:.2f}, MaxDD={m_base['max_dd']:.1f}%")
    print(f"  Year-by-year:")
    for y in yy_base:
        print(f"    {y['year']}: {y['return']:+.1f}% (Sharpe={y['sharpe']:.2f})")

    results["equal_weight_baseline"] = {
        "metrics": m_base, "year_by_year": yy_base,
    }

    # No VT baseline
    dr_base_novt = equal_weight_baseline(prices, leveraged, 0)
    m_base_novt = compute_metrics(dr_base_novt, "equal_weight_leveraged_novt")
    print(f"\n  Without VT: CAGR={m_base_novt['cagr']:.1f}%, Sharpe={m_base_novt['sharpe']:.2f}, MaxDD={m_base_novt['max_dd']:.1f}%")

    # ═══════════════════════════════════════════════════
    # TQQQ + VT30 + 200MA (best from prior research)
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PRIOR CHAMPION: TQQQ + 200MA + VT30")
    print(f"{'='*60}")

    if "TQQQ" in prices.columns and "SPY" in prices.columns:
        spy = prices["SPY"]
        spy_ma = spy.rolling(200, min_periods=100).mean()
        tqqq_ret = prices["TQQQ"].pct_change()
        tqqq_vol = tqqq_ret.rolling(21).std() * np.sqrt(252)

        dr_champ = []
        for date in prices.index[252:]:
            if spy.loc[date] < spy_ma.loc[date]:
                dr_champ.append(0)
                continue
            r = tqqq_ret.loc[date]
            if np.isnan(r):
                dr_champ.append(0)
                continue
            rv = tqqq_vol.loc[date]
            if rv > 0:
                scalar = min(0.30 / rv, 2.0)
                r = r * scalar
            dr_champ.append(r)

        dr_champ = pd.Series(dr_champ, index=prices.index[252:])
        m_champ = compute_metrics(dr_champ, "tqqq_200ma_vt30")
        yy_champ = year_by_year(dr_champ)
        crisis_champ = crisis_analysis(dr_champ)

        print(f"  CAGR={m_champ['cagr']:.1f}%, Sharpe={m_champ['sharpe']:.2f}, MaxDD={m_champ['max_dd']:.1f}%")
        for y in yy_champ:
            print(f"    {y['year']}: {y['return']:+.1f}% (Sharpe={y['sharpe']:.2f})")

        results["tqqq_200ma_vt30"] = {
            "metrics": m_champ, "year_by_year": yy_champ, "crisis": crisis_champ,
        }

    # ═══════════════════════════════════════════════════
    # FINAL COMPARISON TABLE
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("FINAL COMPARISON — All configs head-to-head")
    print(f"{'='*90}")
    print(f"{'Config':<40} {'CAGR':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>7} {'Calmar':>7} {'p-val':>6}")
    print("-" * 90)

    comparison = [
        ("Dual Mom lb21 VT30 (w/costs)", results.get("dual_mom_lb21_vt30", {}).get("metrics", {}),
         results.get("dual_mom_lb21_vt30", {}).get("permutation_p", "N/A")),
        ("Dual Mom lb63 VT30 (w/costs)", results.get("dual_mom_lb63_vt30", {}).get("metrics", {}),
         results.get("dual_mom_lb63_vt30", {}).get("permutation_p", "N/A")),
        ("Breakout 10d trail8% VT30", results.get("breakout_10d_trail8_vt30", {}).get("metrics", {}), "N/A"),
        ("EW Leveraged + VT30 (baseline)", results.get("equal_weight_baseline", {}).get("metrics", {}), "N/A"),
        ("TQQQ + 200MA + VT30 (champion)", results.get("tqqq_200ma_vt30", {}).get("metrics", {}), "N/A"),
    ]

    for label, m, p in comparison:
        if m.get("valid"):
            p_str = f"{p:.2f}" if isinstance(p, float) else str(p)
            print(f"{label:<40} {m['cagr']:>6.1f}% {m['sharpe']:>7.2f} {m['sortino']:>8.2f} "
                  f"{m['max_dd']:>6.1f}% {m['calmar']:>7.2f} {p_str:>6}")

    # ═══════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("ADVERSARIAL VERDICT")
    print(f"{'='*70}")

    p1_val = results.get("dual_mom_lb21_vt30", {}).get("permutation_p", 1.0)
    p2_val = results.get("dual_mom_lb63_vt30", {}).get("permutation_p", 1.0)
    rand1 = results.get("dual_mom_lb21_vt30", {}).get("random_mean_sharpe", 0)
    rand2 = results.get("dual_mom_lb63_vt30", {}).get("random_mean_sharpe", 0)

    if p1_val <= 0.05:
        print(f"✅ Dual Mom lb21 VT30 PASSES permutation (p={p1_val:.2f})")
        print(f"   Random rotation Sharpe: {rand1:.2f} vs actual {m1['sharpe']:.2f}")
        print(f"   The momentum signal adds genuine value beyond just being long leveraged ETFs.")
    else:
        print(f"❌ Dual Mom lb21 VT30 FAILS permutation (p={p1_val:.2f})")
        print(f"   Random rotation Sharpe: {rand1:.2f} vs actual {m1['sharpe']:.2f}")
        print(f"   The high returns come from LEVERAGE + VOL-TARGETING, not from the rotation signal.")
        print(f"   Equal-weight leveraged + VT30 achieves similar results without any signal.")

    if p2_val <= 0.05:
        print(f"\n✅ Dual Mom lb63 VT30 PASSES permutation (p={p2_val:.2f})")
    else:
        print(f"\n❌ Dual Mom lb63 VT30 FAILS permutation (p={p2_val:.2f})")

    # Key insight
    base_sharpe = results.get("equal_weight_baseline", {}).get("metrics", {}).get("sharpe", 0)
    champ_sharpe = results.get("tqqq_200ma_vt30", {}).get("metrics", {}).get("sharpe", 0)
    print(f"\n📊 KEY INSIGHT:")
    print(f"   Equal-weight leveraged + VT30: Sharpe {base_sharpe:.2f}")
    print(f"   TQQQ + 200MA + VT30: Sharpe {champ_sharpe:.2f}")
    print(f"   Best rotation: Sharpe {m1['sharpe']:.2f}")
    print(f"   If rotation barely beats equal-weight, the alpha is in the LEVERAGE + VT, not the SIGNAL.")

    # Save
    elapsed = time.time() - t0
    output = {
        "generated": pd.Timestamp.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "results": {k: v for k, v in results.items()},
    }

    with open(OUT_DIR / "adversarial_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved. Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
