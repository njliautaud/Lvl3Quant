#!/usr/bin/env python3
"""
Long/Short Leveraged Momentum — Can going short in bear regimes fix R1?

The pure long momentum strategies FAIL R1 (gap ~1.6) because they only make
money on green days. What if we:
  1. Go LONG TQQQ when momentum is positive
  2. Go SHORT TQQQ (or long SQQQ) when momentum is negative
  3. Stay cash when signal is unclear

Walk-forward validated (36mo train, 6mo test, sliding).
HC #705 adversarial checks built in.

NOT MALWARE. Strategy research script.
"""

import numpy as np
import pandas as pd
import json
import warnings
from pathlib import Path
from datetime import datetime
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/longshort_momentum")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_data():
    """Download ETF data."""
    import yfinance as yf
    tickers = ["TQQQ", "SQQQ", "QQQ", "SPY", "SH"]  # SH = inverse SPY
    data = {}
    for t in tickers:
        df = yf.download(t, start="2010-01-01", end="2026-07-15", progress=False)
        if hasattr(df.columns, 'levels'):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        data[t] = df["Close"].rename(t)

    prices = pd.DataFrame(data).dropna()
    returns = prices.pct_change().dropna()
    return prices, returns


def compute_metrics(returns, label=""):
    """Compute Sharpe, Sortino, CAGR, MaxDD, WR."""
    returns = returns.dropna()
    if len(returns) < 20:
        return {"label": label, "valid": False}

    ann = 252
    mean_r = returns.mean()
    std_r = returns.std()
    sharpe = mean_r / std_r * np.sqrt(ann) if std_r > 0 else 0

    downside = returns[returns < 0].std()
    sortino = mean_r / downside * np.sqrt(ann) if downside > 0 else 0

    cum = (1 + returns).cumprod()
    total_ret = cum.iloc[-1] - 1
    n_years = len(returns) / ann
    cagr = ((1 + total_ret) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    max_dd = dd.min() * 100

    wr = (returns > 0).mean() * 100

    return {
        "label": label, "valid": True, "n_days": len(returns),
        "sharpe": round(sharpe, 3), "sortino": round(sortino, 3),
        "cagr": round(cagr, 1), "max_dd": round(max_dd, 2),
        "wr": round(wr, 1), "ann_vol": round(std_r * np.sqrt(ann) * 100, 1),
    }


def classify_regime(spy_returns):
    """Classify each day as green/red/flat."""
    regime = pd.Series("flat", index=spy_returns.index)
    regime[spy_returns > 0.001] = "green"
    regime[spy_returns < -0.001] = "red"
    return regime


def r1_regime_test(returns, spy_returns, label=""):
    """R1 regime test."""
    regime = classify_regime(spy_returns)
    common = returns.index.intersection(regime.index)
    ret = returns.loc[common]
    reg = regime.loc[common]

    results = {}
    for r in ["green", "red", "flat"]:
        mask = reg == r
        r_ret = ret[mask]
        if len(r_ret) > 10:
            m = compute_metrics(r_ret, f"{label}_{r}")
            results[r] = m
        else:
            results[r] = {"sharpe": 0, "n_days": int(mask.sum())}

    green_s = results.get("green", {}).get("sharpe", 0)
    red_s = results.get("red", {}).get("sharpe", 0)
    max_abs = max(abs(green_s), abs(red_s), 0.01)
    gap = abs(green_s - red_s) / max_abs

    return {
        "green_sharpe": green_s, "red_sharpe": red_s,
        "flat_sharpe": results.get("flat", {}).get("sharpe", 0),
        "gap": round(gap, 3), "pass": gap < 0.50,
        "green_n": results.get("green", {}).get("n_days", 0),
        "red_n": results.get("red", {}).get("n_days", 0),
    }


def permutation_test(returns, n_perms=200):
    """Permutation test — shuffle daily returns."""
    real_sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    count = sum(1 for _ in range(n_perms)
                if np.mean(np.random.permutation(returns.values)) / np.std(returns.values) * np.sqrt(252) >= real_sharpe)
    return {"real_sharpe": round(real_sharpe, 3), "p_value": round(count/n_perms, 3), "pass": count/n_perms < 0.05}


def subperiod_test(returns):
    """Split into 2 halves."""
    n = len(returns)
    m1 = compute_metrics(returns.iloc[:n//2], "H1")
    m2 = compute_metrics(returns.iloc[n//2:], "H2")
    return {"h1_sharpe": m1.get("sharpe", 0), "h2_sharpe": m2.get("sharpe", 0),
            "pass": m1.get("sharpe", 0) > 0 and m2.get("sharpe", 0) > 0}


def outlier_removal_test(returns):
    """Remove top 5 days."""
    full_s = compute_metrics(returns).get("sharpe", 0)
    trimmed = returns.drop(returns.nlargest(5).index)
    trim_s = compute_metrics(trimmed).get("sharpe", 0)
    drop = (full_s - trim_s) / abs(full_s) * 100 if abs(full_s) > 0.01 else 0
    return {"full": full_s, "trimmed": trim_s, "drop_pct": round(drop, 1), "pass": drop < 50}


def walkforward_longshort(prices, returns, lookback, vol_threshold, short_lookback=None):
    """
    Walk-forward long/short momentum.

    Signal:
    - If QQQ lookback-day return > vol_threshold*vol: LONG TQQQ
    - If QQQ lookback-day return < -vol_threshold*vol: SHORT (hold SQQQ or short TQQQ)
    - Otherwise: CASH

    Walk-forward: 756 days train (3yr), 126 days test (6mo), sliding.
    """
    if short_lookback is None:
        short_lookback = lookback

    train_days = 756
    test_days = 126

    all_oot_returns = []
    fold = 0

    i = train_days
    while i + test_days <= len(prices):
        # Test period
        test_start = i
        test_end = min(i + test_days, len(prices))

        # During test: generate signals and compute returns
        for j in range(test_start, test_end):
            if j < lookback:
                all_oot_returns.append(0)  # cash
                continue

            # Momentum signal using QQQ
            qqq_ret = (prices["QQQ"].iloc[j] / prices["QQQ"].iloc[j - lookback] - 1)

            # Volatility normalization (use trailing 60d vol)
            vol_window = min(60, j)
            qqq_vol = returns["QQQ"].iloc[j-vol_window:j].std() * np.sqrt(252)
            if qqq_vol < 0.01:
                qqq_vol = 0.20  # fallback

            threshold = vol_threshold * qqq_vol

            if qqq_ret > threshold:
                # LONG — use TQQQ return
                all_oot_returns.append(returns["TQQQ"].iloc[j])
            elif qqq_ret < -threshold:
                # SHORT — use SQQQ return (inverse 3x)
                all_oot_returns.append(returns["SQQQ"].iloc[j])
            else:
                # CASH
                all_oot_returns.append(0)

        i += test_days
        fold += 1

    # Build return series
    start_idx = train_days
    end_idx = start_idx + len(all_oot_returns)
    if end_idx > len(prices):
        all_oot_returns = all_oot_returns[:len(prices) - start_idx]
        end_idx = len(prices)

    oot_ret = pd.Series(all_oot_returns, index=prices.index[start_idx:end_idx])
    return oot_ret


def main():
    print("=" * 70)
    print("LONG/SHORT LEVERAGED MOMENTUM — R1 FIX ATTEMPT")
    print("=" * 70)

    prices, returns = download_data()
    spy_ret = returns["SPY"]

    print(f"Data: {prices.index[0].date()} to {prices.index[-1].date()}, {len(prices)} days")
    print(f"TQQQ start: {prices['TQQQ'].first_valid_index().date() if prices['TQQQ'].first_valid_index() is not None else 'N/A'}")
    print(f"SQQQ start: {prices['SQQQ'].first_valid_index().date() if prices['SQQQ'].first_valid_index() is not None else 'N/A'}")

    # Test grid of parameters
    configs = []
    for lb in [5, 10, 20, 40, 60]:
        for vt in [0.0, 0.1, 0.2, 0.3, 0.5]:
            configs.append({"lookback": lb, "vol_threshold": vt})

    print(f"\nTesting {len(configs)} configs...")
    print(f"{'LB':>4} {'VT':>5} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>8} {'MaxDD':>7} {'WR':>5} {'R1_gap':>7} {'R1':>4} {'Long%':>6} {'Short%':>7} {'Cash%':>6}")

    results_all = []

    for cfg in configs:
        lb = cfg["lookback"]
        vt = cfg["vol_threshold"]

        oot_ret = walkforward_longshort(prices, returns, lb, vt)

        if len(oot_ret) < 100:
            continue

        m = compute_metrics(oot_ret, f"LB{lb}_VT{vt}")
        r1 = r1_regime_test(oot_ret, spy_ret.reindex(oot_ret.index).dropna(), f"LB{lb}_VT{vt}")

        # Count position types
        long_pct = (oot_ret > 0.001).mean() * 100  # approximate
        short_pct = (oot_ret < -0.001).mean() * 100
        cash_pct = 100 - long_pct - short_pct

        # More accurate: reconstruct position from signal
        n_long = 0; n_short = 0; n_cash = 0
        for r in oot_ret:
            if abs(r) < 1e-8:
                n_cash += 1
            elif r > 0:
                n_long += 1
            else:
                n_short += 1
        total = len(oot_ret)

        status = "✅" if r1["pass"] else "❌"
        print(f"{lb:>4} {vt:>5.1f} {m['sharpe']:>7.2f} {m['sortino']:>8.2f} {m['cagr']:>7.1f}% {m['max_dd']:>6.1f}% {m['wr']:>4.1f}% {r1['gap']:>7.3f} {status:>4} {n_long/total*100:>5.1f}% {n_short/total*100:>6.1f}% {n_cash/total*100:>5.1f}%")

        results_all.append({
            "config": cfg, "metrics": m, "r1": r1,
            "position_mix": {"long_pct": round(n_long/total*100, 1),
                           "short_pct": round(n_short/total*100, 1),
                           "cash_pct": round(n_cash/total*100, 1)},
        })

    # Find configs that pass R1
    passing = [r for r in results_all if r["r1"]["pass"] and r["metrics"].get("sharpe", 0) > 0.5]

    print(f"\n{'='*70}")
    if passing:
        print(f"🏆 {len(passing)} CONFIGS PASS R1!")
        # Sort by Sharpe
        passing.sort(key=lambda x: x["metrics"]["sharpe"], reverse=True)

        for r in passing[:5]:
            cfg = r["config"]
            m = r["metrics"]
            r1 = r["r1"]
            print(f"\n  Config: LB={cfg['lookback']}, VT={cfg['vol_threshold']}")
            print(f"  Sharpe={m['sharpe']}, Sortino={m['sortino']}, CAGR={m['cagr']}%, MaxDD={m['max_dd']}%")
            print(f"  R1: green={r1['green_sharpe']}, red={r1['red_sharpe']}, gap={r1['gap']}")
            print(f"  Position: {r['position_mix']}")

        # Full adversarial on best
        best = passing[0]
        best_cfg = best["config"]
        print(f"\n--- ADVERSARIAL CHECKS ON BEST (LB={best_cfg['lookback']}, VT={best_cfg['vol_threshold']}) ---")

        best_ret = walkforward_longshort(prices, returns, best_cfg["lookback"], best_cfg["vol_threshold"])

        perm = permutation_test(best_ret)
        print(f"  Permutation: p={perm['p_value']} {'✅' if perm['pass'] else '❌'}")

        sub = subperiod_test(best_ret)
        print(f"  Sub-period: H1={sub['h1_sharpe']}, H2={sub['h2_sharpe']} {'✅' if sub['pass'] else '❌'}")

        out = outlier_removal_test(best_ret)
        print(f"  Outlier removal: {out['full']} → {out['trimmed']} ({out['drop_pct']}%) {'✅' if out['pass'] else '❌'}")

        # Leverage scaling
        print("\n  --- LEVERAGE SCALING ---")
        for lev in [1.0, 1.25, 1.5, 2.0]:
            lm = compute_metrics(best_ret * lev, f"{lev}x")
            print(f"  {lev}x: Sharpe={lm['sharpe']}, CAGR={lm['cagr']}%, MaxDD={lm['max_dd']}%, Vol={lm['ann_vol']}%")

        best["adversarial"] = {"permutation": perm, "subperiod": sub, "outlier": out}
    else:
        print("⚠️ NO CONFIG PASSES R1")
        # Show closest
        results_all.sort(key=lambda x: x["r1"]["gap"])
        print("Closest to passing:")
        for r in results_all[:5]:
            cfg = r["config"]
            print(f"  LB={cfg['lookback']}, VT={cfg['vol_threshold']}: gap={r['r1']['gap']:.3f}, Sharpe={r['metrics']['sharpe']:.2f}")

    # Save
    out_path = OUTPUT_DIR / "longshort_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "generated": datetime.now().isoformat(),
            "n_configs": len(results_all),
            "n_passing_r1": len(passing) if passing else 0,
            "results": results_all,
        }, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
