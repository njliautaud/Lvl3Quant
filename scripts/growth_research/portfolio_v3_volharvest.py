#!/usr/bin/env python3
"""
Portfolio Construction: Gameplan v3 + Vol Harvesting
====================================================
Combines our two proven strategies:
  1. Gameplan v3 (vol-adjusted UPRO with confluence gate) — Sharpe 2.39
  2. Vol Harvesting (VIX term structure, SVXY contango) — Sharpe 4.13

Key question: Are they uncorrelated? If so, combining them should give
better risk-adjusted returns than either alone.

Tests:
  - Correlation of daily returns
  - Optimal allocation via mean-variance
  - Walk-forward portfolio validation
  - Combined portfolio metrics + adversarial
"""
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_data(start="2011-10-04", end="2026-07-17") -> pd.DataFrame:
    """Fetch all required data."""
    tickers = ["SPY", "UPRO", "SVXY", "^VIX", "^VIX3M", "GLD", "TLT"]
    print(f"Fetching {tickers}...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw
    prices = prices.rename(columns={"^VIX": "VIX", "^VIX3M": "VIX3M"})
    prices = prices.ffill().dropna(how="all")
    print(f"  {len(prices)} days ({prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')})")
    return prices


# ── Strategy 1: Gameplan v3 ──────────────────────────────────────────────────

def gameplan_v3_returns(prices: pd.DataFrame) -> pd.Series:
    """Simplified Gameplan v3: confluence-gated UPRO/SPY/GLD."""
    spy = prices["SPY"]
    upro_ret = prices["UPRO"].pct_change()
    spy_ret = spy.pct_change()

    # Confluence components
    mom5 = spy.pct_change(5)
    delta = spy.diff()
    gain = delta.clip(lower=0).rolling(10).mean()
    loss = (-delta.clip(upper=0)).rolling(10).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi10 = 100 - (100 / (1 + rs))

    sma20 = spy.rolling(20).mean()
    sma50 = spy.rolling(50).mean()
    vol21 = spy.pct_change().rolling(21).std() * np.sqrt(252) * 100

    sma200 = spy.rolling(200).mean()
    slope200 = sma200.pct_change(20)
    vol63 = spy.pct_change().rolling(63).std() * np.sqrt(252) * 100
    vol63_slope = vol63.diff(20)

    short_sig = ((mom5 > 0).astype(float) + (rsi10 > 50).astype(float)) / 2
    medium_sig = ((sma20 > sma50).astype(float) + (vol21 < 15).astype(float)) / 2
    long_sig = ((slope200 > 0).astype(float) + (vol63_slope < 0).astype(float)) / 2
    confluence = (short_sig + medium_sig + long_sig) * 2  # Scale to 0-3

    # Dual gate with hysteresis
    position = pd.Series("SPY", index=prices.index)
    in_upro = False

    for i in range(200, len(prices)):
        score = confluence.iloc[i-1]
        vol = vol21.iloc[i-1]

        if vol > 30:
            position.iloc[i] = "GLD"
            in_upro = False
        elif vol > 15:
            position.iloc[i] = "SPY"
            in_upro = False
        elif in_upro and score >= 2.0:
            position.iloc[i] = "UPRO"
        elif not in_upro and score >= 2.5:
            position.iloc[i] = "UPRO"
            in_upro = True
        else:
            position.iloc[i] = "SPY"
            if score < 2.0:
                in_upro = False

    strat_ret = pd.Series(0.0, index=prices.index)
    strat_ret[position == "UPRO"] = upro_ret[position == "UPRO"]
    strat_ret[position == "SPY"] = spy_ret[position == "SPY"]
    # GLD returns
    if "GLD" in prices.columns:
        gld_ret = prices["GLD"].pct_change()
        strat_ret[position == "GLD"] = gld_ret[position == "GLD"]

    return strat_ret


# ── Strategy 2: Vol Harvesting ───────────────────────────────────────────────

def vol_harvest_returns(prices: pd.DataFrame) -> pd.Series:
    """Best vol harvesting config: TS_th0.9_sm1_cap30_cash_VT15."""
    if "VIX3M" not in prices.columns or "VIX" not in prices.columns or "SVXY" not in prices.columns:
        return pd.Series(0.0, index=prices.index)

    ratio = prices["VIX"] / prices["VIX3M"]
    contango = ratio < 0.90
    vix_safe = prices["VIX"] <= 30
    signal = contango & vix_safe

    svxy_ret = prices["SVXY"].pct_change()

    # Vol targeting
    log_rets = np.log(prices["SVXY"] / prices["SVXY"].shift(1))
    realized_vol = log_rets.rolling(21).std() * np.sqrt(252)
    position_size = (0.15 / realized_vol).clip(0, 1.5)

    strat_ret = pd.Series(0.0, index=prices.index)
    warmup = 252
    for i in range(warmup, len(prices)):
        if signal.iloc[i-1]:
            strat_ret.iloc[i] = svxy_ret.iloc[i] * position_size.iloc[i-1]

    return strat_ret


# ── Portfolio construction ───────────────────────────────────────────────────

def compute_metrics(returns: pd.Series, name: str = "") -> dict:
    """Compute risk-adjusted metrics."""
    r = returns.dropna()
    if len(r) < 10 or r.std() == 0:
        return {"name": name, "sharpe": 0, "sortino": 0, "cagr_pct": 0, "max_dd_pct": 0}

    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = r[r < 0].std() * np.sqrt(252) if (r < 0).any() else 1e-6
    sortino = ann_ret / downside

    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min() * 100

    n_years = len(r) / 252
    cagr = (cum.iloc[-1] ** (1 / n_years) - 1) * 100 if n_years > 0 and cum.iloc[-1] > 0 else 0

    wr = (r > 0).sum() / ((r != 0).sum() or 1)
    gains = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')

    return {
        "name": name,
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "cagr_pct": round(cagr, 2),
        "max_dd_pct": round(max_dd, 2),
        "win_rate": round(float(wr), 4),
        "profit_factor": round(pf, 4),
        "n_days": len(r),
        "ann_vol_pct": round(ann_vol * 100, 2),
    }


def optimal_weights(r1: pd.Series, r2: pd.Series) -> dict:
    """Find optimal weights via mean-variance optimization."""
    combined = pd.DataFrame({"gp3": r1, "vh": r2}).dropna()
    if len(combined) < 100:
        return {"w_gp3": 0.5, "w_vh": 0.5, "method": "equal_weight_fallback"}

    means = combined.mean() * 252
    cov = combined.cov() * 252
    corr = combined.corr().iloc[0, 1]

    results = []
    for w1 in np.arange(0, 1.01, 0.05):
        w2 = 1 - w1
        w = np.array([w1, w2])
        port_ret = w @ means
        port_vol = np.sqrt(w @ cov @ w)
        port_sharpe = port_ret / port_vol if port_vol > 0 else 0
        results.append({"w_gp3": round(w1, 2), "w_vh": round(w2, 2),
                        "sharpe": round(port_sharpe, 4), "ret": round(port_ret * 100, 2),
                        "vol": round(port_vol * 100, 2)})

    # Best Sharpe
    best = max(results, key=lambda x: x["sharpe"])

    return {
        "optimal_w_gp3": best["w_gp3"],
        "optimal_w_vh": best["w_vh"],
        "optimal_sharpe": best["sharpe"],
        "correlation": round(corr, 4),
        "efficient_frontier": results,
    }


# ── Walk-forward portfolio test ──────────────────────────────────────────────

def walk_forward_portfolio(prices: pd.DataFrame, train_days: int = 756,
                            test_days: int = 252, step_days: int = 126) -> dict:
    """Walk-forward test of portfolio allocation."""
    print(f"\nWalk-forward portfolio: train={train_days}d, test={test_days}d, step={step_days}d")

    gp3_ret = gameplan_v3_returns(prices)
    vh_ret = vol_harvest_returns(prices)

    warmup = 252
    fold_results = []
    i = warmup + train_days

    while i + test_days <= len(prices):
        # Train: find optimal weights
        train_gp3 = gp3_ret.iloc[i-train_days:i]
        train_vh = vh_ret.iloc[i-train_days:i]

        opt = optimal_weights(train_gp3, train_vh)
        w_gp3 = opt.get("optimal_w_gp3", 0.5)
        w_vh = opt.get("optimal_w_vh", 0.5)

        # Test OOS
        test_gp3 = gp3_ret.iloc[i:i+test_days]
        test_vh = vh_ret.iloc[i:i+test_days]
        test_combined = test_gp3 * w_gp3 + test_vh * w_vh

        combined_metrics = compute_metrics(test_combined, "combined")
        gp3_metrics = compute_metrics(test_gp3, "gp3_only")
        vh_metrics = compute_metrics(test_vh, "vh_only")

        fold_results.append({
            "fold": len(fold_results) + 1,
            "test_start": prices.index[i].strftime("%Y-%m-%d"),
            "w_gp3": w_gp3,
            "w_vh": w_vh,
            "combined_sharpe": combined_metrics["sharpe"],
            "gp3_sharpe": gp3_metrics["sharpe"],
            "vh_sharpe": vh_metrics["sharpe"],
            "beats_gp3": combined_metrics["sharpe"] > gp3_metrics["sharpe"],
            "beats_vh": combined_metrics["sharpe"] > vh_metrics["sharpe"],
        })

        print(f"  Fold {len(fold_results)}: w={w_gp3:.0%}/{w_vh:.0%}, "
              f"Combined Sharpe {combined_metrics['sharpe']:.3f} "
              f"vs GP3 {gp3_metrics['sharpe']:.3f} / VH {vh_metrics['sharpe']:.3f}")

        i += step_days

    beats_gp3 = sum(f["beats_gp3"] for f in fold_results)
    beats_vh = sum(f["beats_vh"] for f in fold_results)

    return {
        "n_folds": len(fold_results),
        "beats_gp3": f"{beats_gp3}/{len(fold_results)}",
        "beats_vh": f"{beats_vh}/{len(fold_results)}",
        "mean_combined_sharpe": round(np.mean([f["combined_sharpe"] for f in fold_results]), 4),
        "mean_w_gp3": round(np.mean([f["w_gp3"] for f in fold_results]), 2),
        "mean_w_vh": round(np.mean([f["w_vh"] for f in fold_results]), 2),
        "folds": fold_results,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Portfolio: Gameplan v3 + Vol Harvesting")
    print("=" * 70)

    prices = fetch_data()

    # Generate strategy returns
    print("\nComputing Gameplan v3 returns...")
    gp3_ret = gameplan_v3_returns(prices)

    print("Computing Vol Harvest returns...")
    vh_ret = vol_harvest_returns(prices)

    # Align returns
    combined_df = pd.DataFrame({"gp3": gp3_ret, "vh": vh_ret}).dropna()
    print(f"\nOverlapping period: {len(combined_df)} days")

    # Correlation analysis
    corr = combined_df.corr().iloc[0, 1]
    print(f"\n=== Correlation Analysis ===")
    print(f"  Daily return correlation: {corr:.4f}")
    print(f"  {'Low correlation — good diversification!' if abs(corr) < 0.3 else 'Moderate correlation' if abs(corr) < 0.6 else 'High correlation — limited diversification'}")

    # Individual metrics
    gp3_metrics = compute_metrics(combined_df["gp3"], "Gameplan v3")
    vh_metrics = compute_metrics(combined_df["vh"], "Vol Harvest")
    print(f"\n=== Individual Strategy Metrics ===")
    for m in [gp3_metrics, vh_metrics]:
        print(f"  {m['name']:15s}: Sharpe={m['sharpe']:.3f}, Sortino={m['sortino']:.3f}, "
              f"CAGR={m['cagr_pct']:.1f}%, MaxDD={m['max_dd_pct']:.1f}%, Vol={m['ann_vol_pct']:.1f}%")

    # Optimal allocation
    print(f"\n=== Optimal Allocation ===")
    opt = optimal_weights(combined_df["gp3"], combined_df["vh"])
    print(f"  Optimal: {opt['optimal_w_gp3']:.0%} Gameplan v3 / {opt['optimal_w_vh']:.0%} Vol Harvest")
    print(f"  Optimal portfolio Sharpe: {opt['optimal_sharpe']:.4f}")
    print(f"  Correlation: {opt['correlation']:.4f}")

    # Test key allocations
    print(f"\n=== Key Allocations ===")
    for w_gp3, label in [(1.0, "100% GP3"), (0.7, "70/30 GP3/VH"), (0.5, "50/50"),
                          (0.3, "30/70 GP3/VH"), (0.0, "100% VH"),
                          (opt['optimal_w_gp3'], f"Optimal ({opt['optimal_w_gp3']:.0%}/{opt['optimal_w_vh']:.0%})")]:
        w_vh = 1 - w_gp3
        port_ret = combined_df["gp3"] * w_gp3 + combined_df["vh"] * w_vh
        m = compute_metrics(port_ret, label)
        print(f"  {label:20s}: Sharpe={m['sharpe']:.3f}, Sortino={m['sortino']:.3f}, "
              f"CAGR={m['cagr_pct']:.1f}%, MaxDD={m['max_dd_pct']:.1f}%")

    # Walk-forward
    wf = walk_forward_portfolio(prices)

    # R1 regime check on optimal portfolio
    opt_w = opt['optimal_w_gp3']
    port_ret = combined_df["gp3"] * opt_w + combined_df["vh"] * (1 - opt_w)
    spy_ret = prices["SPY"].pct_change().reindex(port_ret.index)

    green = spy_ret > 0
    red = spy_ret < 0
    green_sharpe = compute_metrics(port_ret[green], "green")["sharpe"]
    red_sharpe = compute_metrics(port_ret[red], "red")["sharpe"]
    gap = abs(green_sharpe - red_sharpe) / max(abs(green_sharpe), abs(red_sharpe), 0.001)

    print(f"\n=== R1 Regime Check (Optimal Portfolio) ===")
    print(f"  Green day Sharpe: {green_sharpe:.3f}")
    print(f"  Red day Sharpe: {red_sharpe:.3f}")
    print(f"  Gap: {gap:.3f} (threshold: 0.50)")
    print(f"  R1: {'PASS' if gap <= 0.50 else 'FAIL'}")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "correlation": float(corr),
        "individual": {"gameplan_v3": gp3_metrics, "vol_harvest": vh_metrics},
        "optimal_allocation": {
            "w_gp3": opt['optimal_w_gp3'],
            "w_vh": opt['optimal_w_vh'],
            "sharpe": opt['optimal_sharpe'],
        },
        "key_allocations": {},
        "walkforward": wf,
        "regime": {
            "green_sharpe": green_sharpe,
            "red_sharpe": red_sharpe,
            "gap": round(gap, 4),
            "r1_pass": gap <= 0.50,
        },
    }

    for w_gp3 in [1.0, 0.7, 0.5, 0.3, 0.0]:
        port_ret = combined_df["gp3"] * w_gp3 + combined_df["vh"] * (1 - w_gp3)
        m = compute_metrics(port_ret, f"{int(w_gp3*100)}/{int((1-w_gp3)*100)}")
        results["key_allocations"][f"{int(w_gp3*100)}_{int((1-w_gp3)*100)}"] = m

    output_file = OUTPUT_DIR / "portfolio_v3_volharvest_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.integer, np.floating, np.bool_)) else str(o))
    print(f"\nResults saved to {output_file}")

    # Verdict
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    if abs(corr) < 0.3 and opt['optimal_sharpe'] > max(gp3_metrics['sharpe'], vh_metrics['sharpe']):
        print(f"✅ Combining GP3 + VH IMPROVES risk-adjusted returns!")
        print(f"   Correlation: {corr:.3f} (low — good diversification)")
        print(f"   Portfolio Sharpe: {opt['optimal_sharpe']:.3f} vs GP3 {gp3_metrics['sharpe']:.3f} / VH {vh_metrics['sharpe']:.3f}")
    elif abs(corr) < 0.5:
        print(f"⚠️ Moderate diversification benefit — partial combination may help.")
    else:
        print(f"❌ Strategies are too correlated ({corr:.3f}) — combining doesn't help.")
    print()


if __name__ == "__main__":
    main()
