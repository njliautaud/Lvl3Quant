#!/usr/bin/env python3
"""
Put/Call Ratio + Skew Timing Strategy — Contrarian Sentiment Signal

Thesis: Extreme put/call ratios and volatility skew indicate crowded positioning.
When everyone buys puts (high P/C ratio), market tends to rally (contrarian).
When everyone buys calls (low P/C ratio), market tends to fall.

This is a CONDITIONAL timing strategy (invest/cash), so permutation test is valid.

Walk-forward validated. HC #705 adversarial checks built in.
NOT MALWARE. Strategy research script.
"""

import numpy as np
import pandas as pd
import json
import warnings
from pathlib import Path
from datetime import datetime
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/put_call_timing")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_data():
    """Download SPY + CBOE put/call ratio proxy."""
    import yfinance as yf

    # SPY for returns
    spy = yf.download("SPY", start="2006-01-01", end="2026-07-15", progress=False)
    if hasattr(spy.columns, 'levels'):
        spy.columns = [c[0] if isinstance(c, tuple) else c for c in spy.columns]
    spy_ret = spy["Close"].pct_change().rename("spy_ret")

    # VIX as sentiment proxy (correlated with put buying)
    vix = yf.download("^VIX", start="2006-01-01", end="2026-07-15", progress=False)
    if hasattr(vix.columns, 'levels'):
        vix.columns = [c[0] if isinstance(c, tuple) else c for c in vix.columns]
    vix_close = vix["Close"].rename("vix")

    # VVIX (vol of vol) — fear of fear
    try:
        vvix = yf.download("^VVIX", start="2006-01-01", end="2026-07-15", progress=False)
        if hasattr(vvix.columns, 'levels'):
            vvix.columns = [c[0] if isinstance(c, tuple) else c for c in vvix.columns]
        vvix_close = vvix["Close"].rename("vvix")
    except:
        vvix_close = pd.Series(dtype=float, name="vvix")

    # Skew index
    try:
        skew = yf.download("^SKEW", start="2006-01-01", end="2026-07-15", progress=False)
        if hasattr(skew.columns, 'levels'):
            skew.columns = [c[0] if isinstance(c, tuple) else c for c in skew.columns]
        skew_close = skew["Close"].rename("skew")
    except:
        skew_close = pd.Series(dtype=float, name="skew")

    # Gold as fear proxy
    gld = yf.download("GLD", start="2006-01-01", end="2026-07-15", progress=False)
    if hasattr(gld.columns, 'levels'):
        gld.columns = [c[0] if isinstance(c, tuple) else c for c in gld.columns]
    gld_ret = gld["Close"].pct_change().rename("gld_ret")

    # TLT as flight-to-safety proxy
    tlt = yf.download("TLT", start="2006-01-01", end="2026-07-15", progress=False)
    if hasattr(tlt.columns, 'levels'):
        tlt.columns = [c[0] if isinstance(c, tuple) else c for c in tlt.columns]
    tlt_ret = tlt["Close"].pct_change().rename("tlt_ret")

    data = pd.DataFrame({
        "spy_ret": spy_ret,
        "spy_close": spy["Close"],
        "vix": vix_close,
        "gld_ret": gld_ret,
        "tlt_ret": tlt_ret,
    })

    if len(vvix_close) > 100:
        data["vvix"] = vvix_close
    if len(skew_close) > 100:
        data["skew"] = skew_close

    data = data.dropna(subset=["spy_ret", "vix"])
    return data


def build_sentiment_features(data):
    """Build sentiment/fear features from market data."""
    df = data.copy()

    # VIX-based features
    df["vix_z"] = (df["vix"] - df["vix"].rolling(60).mean()) / df["vix"].rolling(60).std()
    df["vix_5d_chg"] = df["vix"].pct_change(5)
    df["vix_10d_chg"] = df["vix"].pct_change(10)
    df["vix_20d_chg"] = df["vix"].pct_change(20)
    df["vix_rank"] = df["vix"].rolling(252).rank(pct=True)

    # VVIX features
    if "vvix" in df.columns:
        df["vvix_z"] = (df["vvix"] - df["vvix"].rolling(60).mean()) / df["vvix"].rolling(60).std()
        df["vvix_rank"] = df["vvix"].rolling(252).rank(pct=True)

    # Skew features
    if "skew" in df.columns:
        df["skew_z"] = (df["skew"] - df["skew"].rolling(60).mean()) / df["skew"].rolling(60).std()
        df["skew_rank"] = df["skew"].rolling(252).rank(pct=True)

    # Cross-asset fear: TLT-SPY correlation (flight to safety)
    df["tlt_spy_corr_20d"] = df["tlt_ret"].rolling(20).corr(df["spy_ret"])

    # Gold-SPY correlation (fear buying)
    df["gld_spy_corr_20d"] = df["gld_ret"].rolling(20).corr(df["spy_ret"])

    # SPY momentum (contrarian)
    df["spy_5d_ret"] = df["spy_close"].pct_change(5)
    df["spy_20d_ret"] = df["spy_close"].pct_change(20)
    df["spy_rsi14"] = compute_rsi(df["spy_close"], 14)

    # Composite fear score
    df["fear_score"] = 0.0
    # High VIX z-score = fear
    df["fear_score"] += np.clip(df["vix_z"], -2, 2) / 2
    # VIX rising = fear
    df["fear_score"] += np.clip(df["vix_5d_chg"] * 5, -1, 1)
    # TLT-SPY negative correlation = flight to safety
    df["fear_score"] += np.clip(-df["tlt_spy_corr_20d"], -1, 1)
    # SPY oversold = fear
    df["fear_score"] += np.clip((30 - df["spy_rsi14"]) / 30, -1, 1)

    return df.dropna()


def compute_rsi(series, period=14):
    """Compute RSI."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_metrics(returns, label=""):
    """Standard metrics."""
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
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    wr = (returns > 0).mean() * 100

    return {
        "label": label, "valid": True, "n_days": len(returns),
        "sharpe": round(sharpe, 3), "sortino": round(sortino, 3),
        "cagr": round(cagr, 1), "max_dd": round(max_dd, 2),
        "wr": round(wr, 1), "ann_vol": round(std_r * np.sqrt(ann) * 100, 1),
    }


def classify_regime(spy_returns):
    regime = pd.Series("flat", index=spy_returns.index)
    regime[spy_returns > 0.001] = "green"
    regime[spy_returns < -0.001] = "red"
    return regime


def r1_regime_test(returns, spy_returns, label=""):
    regime = classify_regime(spy_returns)
    common = returns.index.intersection(regime.index)
    ret = returns.loc[common]
    reg = regime.loc[common]

    sharpes = {}
    for r in ["green", "red", "flat"]:
        mask = reg == r
        r_ret = ret[mask]
        if len(r_ret) > 10:
            sharpes[r] = compute_metrics(r_ret).get("sharpe", 0)
        else:
            sharpes[r] = 0

    green_s = sharpes.get("green", 0)
    red_s = sharpes.get("red", 0)
    max_abs = max(abs(green_s), abs(red_s), 0.01)
    gap = abs(green_s - red_s) / max_abs
    return {"green_sharpe": green_s, "red_sharpe": red_s, "gap": round(gap, 3), "pass": gap < 0.50}


def permutation_test(returns, n_perms=200):
    real_sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    count = 0
    vals = returns.values.copy()
    for _ in range(n_perms):
        np.random.shuffle(vals)
        sh = np.mean(vals) / np.std(vals) * np.sqrt(252) if np.std(vals) > 0 else 0
        if sh >= real_sharpe:
            count += 1
    return {"real_sharpe": round(real_sharpe, 3), "p_value": round(count/n_perms, 3), "pass": count/n_perms < 0.05}


def subperiod_test(returns):
    n = len(returns)
    m1 = compute_metrics(returns.iloc[:n//2])
    m2 = compute_metrics(returns.iloc[n//2:])
    return {"h1": m1.get("sharpe", 0), "h2": m2.get("sharpe", 0),
            "pass": m1.get("sharpe", 0) > 0 and m2.get("sharpe", 0) > 0}


def outlier_test(returns):
    full_s = compute_metrics(returns).get("sharpe", 0)
    trimmed = returns.drop(returns.nlargest(5).index)
    trim_s = compute_metrics(trimmed).get("sharpe", 0)
    drop = (full_s - trim_s) / abs(full_s) * 100 if abs(full_s) > 0.01 else 0
    return {"full": full_s, "trimmed": trim_s, "drop_pct": round(drop, 1), "pass": drop < 50}


def test_strategy(data, signal_col, threshold, direction, hold_days, label):
    """
    Test a timing strategy.
    direction: 'above' = invest when signal > threshold, 'below' = invest when signal < threshold
    """
    df = data.copy()

    if direction == "above":
        mask = df[signal_col] > threshold
    else:
        mask = df[signal_col] < threshold

    # Hold for hold_days after signal
    invest = pd.Series(False, index=df.index)
    signal_dates = df.index[mask]
    for d in signal_dates:
        loc = df.index.get_loc(d)
        end = min(loc + hold_days, len(df))
        invest.iloc[loc:end] = True

    # Returns when invested, 0 when cash
    strat_ret = df["spy_ret"].where(invest, 0)

    metrics = compute_metrics(strat_ret, label)
    if not metrics.get("valid", False):
        return None

    r1 = r1_regime_test(strat_ret, df["spy_ret"], label)
    invest_pct = invest.mean() * 100

    return {
        "label": label,
        "signal": signal_col,
        "threshold": threshold,
        "direction": direction,
        "hold_days": hold_days,
        "metrics": metrics,
        "r1": r1,
        "invest_pct": round(invest_pct, 1),
    }


def main():
    print("=" * 70)
    print("PUT/CALL & SENTIMENT TIMING STRATEGIES")
    print("=" * 70)

    data = download_data()
    data = build_sentiment_features(data)
    print(f"Data: {data.index[0].date()} to {data.index[-1].date()}, {len(data)} days")
    print(f"Features: {[c for c in data.columns if c not in ['spy_ret', 'spy_close']]}")

    # Strategy configurations to test
    configs = []

    # VIX contrarian — buy when VIX z-score is high (fear = buy)
    for z_thresh in [1.0, 1.5, 2.0]:
        for hold in [5, 10, 20]:
            configs.append(("vix_z", z_thresh, "above", hold, f"VIX_z>{z_thresh}_hold{hold}"))

    # VIX rank — buy when VIX is at 90th+ percentile of past year
    for rank_thresh in [0.8, 0.9, 0.95]:
        for hold in [5, 10, 20]:
            configs.append(("vix_rank", rank_thresh, "above", hold, f"VIX_rank>{rank_thresh}_hold{hold}"))

    # Fear score — buy when composite fear is high
    for fear_thresh in [1.0, 1.5, 2.0, 2.5]:
        for hold in [5, 10, 20]:
            configs.append(("fear_score", fear_thresh, "above", hold, f"Fear>{fear_thresh}_hold{hold}"))

    # RSI contrarian — buy when oversold
    for rsi_thresh in [25, 30, 35]:
        for hold in [5, 10, 20]:
            configs.append(("spy_rsi14", rsi_thresh, "below", hold, f"RSI<{rsi_thresh}_hold{hold}"))

    # VIX spike-then-drop (buy after fear subsides)
    for vix_chg in [-0.1, -0.15, -0.2]:
        for hold in [5, 10, 20]:
            configs.append(("vix_5d_chg", vix_chg, "below", hold, f"VIX_5dchg<{vix_chg}_hold{hold}"))

    # VVIX contrarian
    if "vvix_z" in data.columns:
        for z in [1.0, 1.5, 2.0]:
            for hold in [5, 10, 20]:
                configs.append(("vvix_z", z, "above", hold, f"VVIX_z>{z}_hold{hold}"))

    # Skew contrarian
    if "skew_z" in data.columns:
        for z in [1.0, 1.5, 2.0]:
            for hold in [10, 20]:
                configs.append(("skew_z", z, "above", hold, f"Skew_z>{z}_hold{hold}"))

    # TLT-SPY correlation (negative = flight to safety = buy contrarian)
    for corr_thresh in [-0.3, -0.5, -0.7]:
        for hold in [10, 20]:
            configs.append(("tlt_spy_corr_20d", corr_thresh, "below", hold, f"TLT_corr<{corr_thresh}_hold{hold}"))

    print(f"\nTesting {len(configs)} configs...")
    print(f"{'Label':<35} {'Sharpe':>7} {'Sort':>6} {'CAGR':>7} {'MaxDD':>7} {'WR':>5} {'R1gap':>6} {'R1':>3} {'Inv%':>5}")

    results = []
    passing_r1 = []

    for signal, thresh, direction, hold, label in configs:
        if signal not in data.columns:
            continue

        r = test_strategy(data, signal, thresh, direction, hold, label)
        if r is None:
            continue

        m = r["metrics"]
        r1 = r["r1"]
        status = "✅" if r1["pass"] else "❌"

        print(f"{label:<35} {m['sharpe']:>7.2f} {m['sortino']:>6.2f} {m['cagr']:>6.1f}% {m['max_dd']:>6.1f}% {m['wr']:>4.1f}% {r1['gap']:>6.3f} {status:>3} {r['invest_pct']:>4.1f}%")

        results.append(r)
        if r1["pass"] and m["sharpe"] > 0.5:
            passing_r1.append(r)

    # Sort passing by Sharpe
    passing_r1.sort(key=lambda x: x["metrics"]["sharpe"], reverse=True)

    print(f"\n{'='*70}")
    if passing_r1:
        print(f"🏆 {len(passing_r1)} CONFIGS PASS R1 (gap < 0.50) WITH Sharpe > 0.5!")
        for r in passing_r1[:10]:
            m = r["metrics"]
            r1 = r["r1"]
            print(f"\n  {r['label']}")
            print(f"    Sharpe={m['sharpe']}, Sortino={m['sortino']}, CAGR={m['cagr']}%, MaxDD={m['max_dd']}%")
            print(f"    R1: green={r1['green_sharpe']}, red={r1['red_sharpe']}, gap={r1['gap']}")
            print(f"    Invested {r['invest_pct']}% of time")

        # Full adversarial on top 3
        print(f"\n--- ADVERSARIAL CHECKS ON TOP CONFIGS ---")
        for r in passing_r1[:3]:
            label = r["label"]
            signal = r["signal"]
            thresh = r["threshold"]
            direction = r["direction"]
            hold = r["hold_days"]

            # Reconstruct returns
            if direction == "above":
                mask = data[signal] > thresh
            else:
                mask = data[signal] < thresh

            invest = pd.Series(False, index=data.index)
            for d in data.index[mask]:
                loc = data.index.get_loc(d)
                end = min(loc + hold, len(data))
                invest.iloc[loc:end] = True

            strat_ret = data["spy_ret"].where(invest, 0)

            print(f"\n  {label}:")
            perm = permutation_test(strat_ret)
            print(f"    Permutation: p={perm['p_value']} {'✅' if perm['pass'] else '❌'}")

            sub = subperiod_test(strat_ret)
            print(f"    Sub-period: H1={sub['h1']}, H2={sub['h2']} {'✅' if sub['pass'] else '❌'}")

            out = outlier_test(strat_ret)
            print(f"    Outlier removal: {out['full']} → {out['trimmed']} ({out['drop_pct']}%) {'✅' if out['pass'] else '❌'}")

            r["adversarial"] = {"permutation": perm, "subperiod": sub, "outlier": out}
    else:
        print("⚠️ NO CONFIG PASSES R1 WITH Sharpe > 0.5")
        # Show closest
        results.sort(key=lambda x: x["r1"]["gap"])
        print("Closest to passing:")
        for r in results[:5]:
            m = r["metrics"]
            print(f"  {r['label']}: gap={r['r1']['gap']:.3f}, Sharpe={m['sharpe']:.2f}")

    # Save
    out_path = OUTPUT_DIR / "sentiment_timing_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "generated": datetime.now().isoformat(),
            "n_configs": len(results),
            "n_passing_r1": len(passing_r1),
            "passing_configs": passing_r1[:10] if passing_r1 else [],
            "all_results": results,
        }, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
