#!/usr/bin/env python3
"""
Cross-Asset Regime Dashboard (HC #710)
Tracks equities, commodities, bonds, dollar, crypto, and volatility
to build regime signals and predictive analytics.
Designed for daily cron execution.
"""

import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output" / "growth_research" / "cross_asset_dashboard"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = {
    # Equities
    "SPY": "Equities", "QQQ": "Equities", "IWM": "Equities",
    "EFA": "Equities", "EEM": "Equities",
    # Volatility
    "^VIX": "Volatility", "^VIX3M": "Volatility",
    # Commodities
    "GLD": "Commodities", "SLV": "Commodities", "CPER": "Commodities",
    "USO": "Commodities", "DBA": "Commodities",
    # Bonds
    "TLT": "Bonds", "IEF": "Bonds", "HYG": "Bonds", "LQD": "Bonds",
    # Dollar
    "UUP": "Dollar",
    # Crypto
    "BTC-USD": "Crypto",
}

LOOKBACK_YEARS = 3


# ---------------------------------------------------------------------------
# Data Download
# ---------------------------------------------------------------------------
def download_data() -> pd.DataFrame:
    """Download daily close data for all tickers."""
    end = datetime.today()
    start = end - timedelta(days=LOOKBACK_YEARS * 365 + 30)
    tickers_list = list(TICKERS.keys())

    print(f"Downloading {len(tickers_list)} assets ({LOOKBACK_YEARS}yr)...")
    raw = yf.download(tickers_list, start=start, end=end, auto_adjust=True, progress=False)

    # yfinance returns multi-level columns; extract Close
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"]
    else:
        closes = raw[["Close"]]
        closes.columns = tickers_list

    # Drop rows where ALL are NaN, forward-fill the rest
    closes = closes.dropna(how="all").ffill()
    print(f"  Got {len(closes)} trading days, {closes.shape[1]} assets")
    return closes


# ---------------------------------------------------------------------------
# Per-Asset Metrics
# ---------------------------------------------------------------------------
def compute_asset_metrics(closes: pd.DataFrame) -> dict:
    """For each asset: trend, momentum, vol, relative strength vs SPY."""
    metrics = {}
    spy = closes.get("SPY")

    for ticker in closes.columns:
        s = closes[ticker].dropna()
        if len(s) < 210:
            continue

        last = float(s.iloc[-1])
        sma50 = float(s.iloc[-50:].mean())
        sma200 = float(s.iloc[-200:].mean())

        # Trend status
        if last > sma50 > sma200:
            trend = "strong_uptrend"
        elif last > sma50:
            trend = "uptrend"
        elif last < sma50 < sma200:
            trend = "strong_downtrend"
        elif last < sma50:
            trend = "downtrend"
        else:
            trend = "neutral"

        # Momentum (returns)
        def pct(n):
            if len(s) > n:
                return round(float((s.iloc[-1] / s.iloc[-n] - 1) * 100), 2)
            return None

        mom_1w = pct(5)
        mom_1m = pct(21)
        mom_3m = pct(63)
        mom_6m = pct(126)

        # 20-day realized vol (annualized)
        rets = s.pct_change().dropna()
        vol_20d = round(float(rets.iloc[-20:].std() * np.sqrt(252) * 100), 2) if len(rets) >= 20 else None

        # Relative strength vs SPY
        rs_vs_spy = None
        if spy is not None and ticker != "SPY" and ticker not in ("^VIX", "^VIX3M"):
            try:
                ratio = s / spy.reindex(s.index).ffill()
                ratio = ratio.dropna()
                if len(ratio) >= 63:
                    rs_vs_spy = round(float((ratio.iloc[-1] / ratio.iloc[-63] - 1) * 100), 2)
            except Exception:
                pass

        metrics[ticker] = {
            "group": TICKERS.get(ticker, "Unknown"),
            "price": round(last, 2),
            "sma50": round(sma50, 2),
            "sma200": round(sma200, 2),
            "trend": trend,
            "mom_1w_pct": mom_1w,
            "mom_1m_pct": mom_1m,
            "mom_3m_pct": mom_3m,
            "mom_6m_pct": mom_6m,
            "vol_20d_ann_pct": vol_20d,
            "rs_vs_spy_3m_pct": rs_vs_spy,
        }

    return metrics


# ---------------------------------------------------------------------------
# Regime Signals
# ---------------------------------------------------------------------------
def _trending_up(m: dict) -> bool:
    return m.get("trend", "") in ("uptrend", "strong_uptrend")


def _trending_down(m: dict) -> bool:
    return m.get("trend", "") in ("downtrend", "strong_downtrend")


def _mom_positive(m: dict, period: str = "mom_1m_pct") -> bool:
    v = m.get(period)
    return v is not None and v > 0


def build_regime_signals(metrics: dict) -> dict:
    """Cross-reference asset metrics into regime signals."""
    signals = {}

    # --- Risk-On / Risk-Off ---
    risk_on_assets = ["CPER", "USO", "IWM", "EEM"]
    risk_off_assets = ["GLD", "TLT", "UUP"]

    risk_on_score = sum(1 for t in risk_on_assets if t in metrics and _trending_up(metrics[t]))
    risk_off_score = sum(1 for t in risk_off_assets if t in metrics and _trending_up(metrics[t]))

    if risk_on_score >= 3 and risk_off_score <= 1:
        risk_regime = "RISK-ON"
    elif risk_off_score >= 2 and risk_on_score <= 1:
        risk_regime = "RISK-OFF"
    else:
        risk_regime = "MIXED"

    signals["risk_regime"] = {
        "signal": risk_regime,
        "risk_on_score": f"{risk_on_score}/{len(risk_on_assets)}",
        "risk_off_score": f"{risk_off_score}/{len(risk_off_assets)}",
        "detail": {t: metrics[t]["trend"] for t in risk_on_assets + risk_off_assets if t in metrics},
    }

    # --- Inflation Signal ---
    inflation_assets = ["GLD", "USO", "CPER", "DBA"]
    bond_assets = ["TLT", "IEF"]

    commodities_rising = sum(1 for t in inflation_assets if t in metrics and _mom_positive(metrics[t]))
    bonds_falling = sum(1 for t in bond_assets if t in metrics and not _mom_positive(metrics[t]))

    if commodities_rising >= 3 and bonds_falling >= 1:
        inflation_signal = "INFLATIONARY"
    elif commodities_rising <= 1 and bonds_falling == 0:
        inflation_signal = "DISINFLATIONARY"
    else:
        inflation_signal = "NEUTRAL"

    signals["inflation"] = {
        "signal": inflation_signal,
        "commodities_rising": f"{commodities_rising}/{len(inflation_assets)}",
        "bonds_falling": f"{bonds_falling}/{len(bond_assets)}",
    }

    # --- Credit Stress ---
    hyg = metrics.get("HYG", {})
    lqd = metrics.get("LQD", {})
    hyg_mom = hyg.get("mom_1m_pct")
    lqd_mom = lqd.get("mom_1m_pct")

    if hyg_mom is not None and lqd_mom is not None:
        spread_move = lqd_mom - hyg_mom  # positive = HYG underperforming = stress
        if spread_move > 1.0:
            credit_signal = "CREDIT_STRESS"
        elif spread_move < -0.5:
            credit_signal = "CREDIT_EASING"
        else:
            credit_signal = "NEUTRAL"
        signals["credit"] = {
            "signal": credit_signal,
            "hyg_1m_pct": hyg_mom,
            "lqd_1m_pct": lqd_mom,
            "spread_move": round(spread_move, 2),
        }
    else:
        signals["credit"] = {"signal": "NO_DATA"}

    # --- Dollar Regime ---
    uup = metrics.get("UUP", {})
    dollar_trend = uup.get("trend", "unknown")
    dollar_mom = uup.get("mom_1m_pct")

    if dollar_trend in ("uptrend", "strong_uptrend"):
        dollar_signal = "STRONG_DOLLAR"
    elif dollar_trend in ("downtrend", "strong_downtrend"):
        dollar_signal = "WEAK_DOLLAR"
    else:
        dollar_signal = "NEUTRAL"

    signals["dollar"] = {
        "signal": dollar_signal,
        "uup_trend": dollar_trend,
        "uup_mom_1m_pct": dollar_mom,
        "impact": "Strong USD typically pressures EM equities and commodities"
        if dollar_signal == "STRONG_DOLLAR"
        else "Weak USD supports commodities and EM"
        if dollar_signal == "WEAK_DOLLAR"
        else "No clear dollar bias",
    }

    # --- Intermarket Divergence ---
    spy_m = metrics.get("SPY", {})
    copper_m = metrics.get("CPER", {})
    oil_m = metrics.get("USO", {})

    spy_up = _mom_positive(spy_m)
    copper_down = copper_m.get("mom_1m_pct") is not None and copper_m["mom_1m_pct"] < -2
    oil_down = oil_m.get("mom_1m_pct") is not None and oil_m["mom_1m_pct"] < -2

    if spy_up and (copper_down or oil_down):
        divergence = "WARNING"
        detail = "Stocks rising but industrial commodities falling -- potential divergence"
    elif not spy_up and not copper_down and not oil_down:
        divergence = "CONFIRMED_WEAKNESS"
        detail = "Both stocks and commodities weak"
    else:
        divergence = "NO_DIVERGENCE"
        detail = "Stocks and commodities broadly aligned"

    signals["intermarket_divergence"] = {
        "signal": divergence,
        "spy_1m": spy_m.get("mom_1m_pct"),
        "copper_1m": copper_m.get("mom_1m_pct"),
        "oil_1m": oil_m.get("mom_1m_pct"),
        "detail": detail,
    }

    return signals


# ---------------------------------------------------------------------------
# Correlation Analysis
# ---------------------------------------------------------------------------
def correlation_analysis(closes: pd.DataFrame) -> dict:
    """60-day rolling correlation with SPY + anomaly detection."""
    spy = closes.get("SPY")
    if spy is None:
        return {}

    rets = closes.pct_change().dropna()
    spy_rets = rets.get("SPY")
    if spy_rets is None:
        return {}

    results = {}
    for ticker in rets.columns:
        if ticker in ("SPY", "^VIX", "^VIX3M"):
            continue
        asset_rets = rets[ticker].dropna()
        common = spy_rets.index.intersection(asset_rets.index)
        if len(common) < 252:
            continue

        s = spy_rets.loc[common]
        a = asset_rets.loc[common]

        corr_60d = float(s.iloc[-60:].corr(a.iloc[-60:]))
        corr_1yr = float(s.iloc[-252:].corr(a.iloc[-252:]))
        diff = round(corr_60d - corr_1yr, 3)
        anomaly = abs(diff) > 0.3

        results[ticker] = {
            "corr_60d": round(corr_60d, 3),
            "corr_1yr_avg": round(corr_1yr, 3),
            "diff": diff,
            "anomaly": anomaly,
        }

    # Sort by current correlation
    sorted_by_corr = sorted(results.items(), key=lambda x: x[1]["corr_60d"])
    most_anticorr = sorted_by_corr[:3]
    most_corr = sorted_by_corr[-3:]

    anomalies = {k: v for k, v in results.items() if v["anomaly"]}

    return {
        "all": results,
        "most_correlated_with_spy": {k: v for k, v in most_corr},
        "most_anticorrelated_with_spy": {k: v for k, v in most_anticorr},
        "correlation_anomalies": anomalies,
    }


# ---------------------------------------------------------------------------
# Historical Predictive Analysis
# ---------------------------------------------------------------------------
def predictive_analysis(closes: pd.DataFrame) -> dict:
    """When certain assets lead/lag, what happens to SPY?"""
    spy = closes.get("SPY")
    if spy is None:
        return {}

    rets = closes.pct_change()
    spy_rets = rets["SPY"]

    # Pre-compute 1m returns for ranking
    mom_1m = closes.pct_change(21)

    # Forward SPY returns at various horizons
    fwd_1w = spy.pct_change(5).shift(-5) * 100
    fwd_2w = spy.pct_change(10).shift(-10) * 100
    fwd_1m = spy.pct_change(21).shift(-21) * 100

    results = {}

    # --- Gold Leading ---
    gold_mom = mom_1m.get("GLD")
    if gold_mom is not None:
        # Rank all assets by 1m momentum each day; check when gold is top-3
        ranks = mom_1m.rank(axis=1, ascending=False)
        gold_rank = ranks.get("GLD")
        if gold_rank is not None:
            mask = gold_rank <= 3
            mask = mask & mask.index.isin(fwd_1w.dropna().index)
            n = int(mask.sum())
            if n > 20:
                results["gold_leading"] = {
                    "condition": "GLD in top-3 assets by 1-month return",
                    "sample_size": n,
                    "spy_fwd_1w_avg_pct": round(float(fwd_1w[mask].mean()), 3),
                    "spy_fwd_2w_avg_pct": round(float(fwd_2w[mask].mean()), 3),
                    "spy_fwd_1m_avg_pct": round(float(fwd_1m[mask].mean()), 3),
                    "spy_fwd_1w_wr": round(float((fwd_1w[mask] > 0).mean() * 100), 1),
                    "spy_fwd_1m_wr": round(float((fwd_1m[mask] > 0).mean() * 100), 1),
                }

    # --- Copper Leading vs Lagging ---
    copper_mom = mom_1m.get("CPER")
    if copper_mom is not None:
        copper_up = copper_mom > 0.03
        copper_down = copper_mom < -0.03
        for label, mask_raw in [("copper_leading", copper_up), ("copper_lagging", copper_down)]:
            mask = mask_raw & mask_raw.index.isin(fwd_1w.dropna().index)
            n = int(mask.sum())
            if n > 20:
                results[label] = {
                    "condition": f"CPER 1m return {'> 3%' if 'leading' in label else '< -3%'}",
                    "sample_size": n,
                    "spy_fwd_1w_avg_pct": round(float(fwd_1w[mask].mean()), 3),
                    "spy_fwd_2w_avg_pct": round(float(fwd_2w[mask].mean()), 3),
                    "spy_fwd_1m_avg_pct": round(float(fwd_1m[mask].mean()), 3),
                    "spy_fwd_1w_wr": round(float((fwd_1w[mask] > 0).mean() * 100), 1),
                    "spy_fwd_1m_wr": round(float((fwd_1m[mask] > 0).mean() * 100), 1),
                }

    # --- Dollar Strong vs Weak ---
    uup_mom = mom_1m.get("UUP")
    if uup_mom is not None:
        dollar_strong = uup_mom > 0.01
        dollar_weak = uup_mom < -0.01
        for label, mask_raw in [("dollar_strong", dollar_strong), ("dollar_weak", dollar_weak)]:
            mask = mask_raw & mask_raw.index.isin(fwd_1w.dropna().index)
            n = int(mask.sum())
            if n > 20:
                results[label] = {
                    "condition": f"UUP 1m return {'> 1%' if 'strong' in label else '< -1%'}",
                    "sample_size": n,
                    "spy_fwd_1w_avg_pct": round(float(fwd_1w[mask].mean()), 3),
                    "spy_fwd_2w_avg_pct": round(float(fwd_2w[mask].mean()), 3),
                    "spy_fwd_1m_avg_pct": round(float(fwd_1m[mask].mean()), 3),
                    "spy_fwd_1w_wr": round(float((fwd_1w[mask] > 0).mean() * 100), 1),
                    "spy_fwd_1m_wr": round(float((fwd_1m[mask] > 0).mean() * 100), 1),
                }

    # --- Credit Spread Widening vs Tightening ---
    hyg = closes.get("HYG")
    lqd = closes.get("LQD")
    if hyg is not None and lqd is not None:
        # HYG/LQD ratio as credit spread proxy (falling = stress)
        ratio = hyg / lqd
        ratio_chg = ratio.pct_change(21)
        spread_widening = ratio_chg < -0.01  # HYG underperforming
        spread_tightening = ratio_chg > 0.01

        for label, mask_raw in [("credit_widening", spread_widening), ("credit_tightening", spread_tightening)]:
            mask = mask_raw & mask_raw.index.isin(fwd_1w.dropna().index)
            n = int(mask.sum())
            if n > 20:
                results[label] = {
                    "condition": f"HYG/LQD ratio 1m change {'< -1%' if 'widening' in label else '> +1%'}",
                    "sample_size": n,
                    "spy_fwd_1w_avg_pct": round(float(fwd_1w[mask].mean()), 3),
                    "spy_fwd_2w_avg_pct": round(float(fwd_2w[mask].mean()), 3),
                    "spy_fwd_1m_avg_pct": round(float(fwd_1m[mask].mean()), 3),
                    "spy_fwd_1w_wr": round(float((fwd_1w[mask] > 0).mean() * 100), 1),
                    "spy_fwd_1m_wr": round(float((fwd_1m[mask] > 0).mean() * 100), 1),
                }

    return results


# ---------------------------------------------------------------------------
# Pretty Print
# ---------------------------------------------------------------------------
def print_summary(metrics: dict, signals: dict, corr: dict, predictive: dict):
    """Print a clean summary to stdout."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*70}")
    print(f"  CROSS-ASSET REGIME DASHBOARD  |  {now}")
    print(f"{'='*70}\n")

    # --- Asset Table ---
    print("ASSET SNAPSHOT")
    print(f"{'Ticker':<10} {'Group':<12} {'Price':>9} {'Trend':<18} {'1W%':>7} {'1M%':>7} {'3M%':>7} {'Vol20d':>7} {'RS/SPY':>7}")
    print("-" * 98)

    groups_order = ["Equities", "Volatility", "Commodities", "Bonds", "Dollar", "Crypto"]
    for group in groups_order:
        for ticker, m in sorted(metrics.items()):
            if m["group"] != group:
                continue
            rs = f"{m['rs_vs_spy_3m_pct']:>6.1f}" if m.get("rs_vs_spy_3m_pct") is not None else "   n/a"
            w1 = f"{m['mom_1w_pct']:>6.1f}" if m.get("mom_1w_pct") is not None else "   n/a"
            m1 = f"{m['mom_1m_pct']:>6.1f}" if m.get("mom_1m_pct") is not None else "   n/a"
            m3 = f"{m['mom_3m_pct']:>6.1f}" if m.get("mom_3m_pct") is not None else "   n/a"
            v20 = f"{m['vol_20d_ann_pct']:>6.1f}" if m.get("vol_20d_ann_pct") is not None else "   n/a"
            print(f"{ticker:<10} {m['group']:<12} {m['price']:>9.2f} {m['trend']:<18} {w1} {m1} {m3} {v20} {rs}")
        print()

    # --- Regime Signals ---
    print(f"\n{'='*70}")
    print("  REGIME SIGNALS")
    print(f"{'='*70}\n")

    for name, sig in signals.items():
        label = name.upper().replace("_", " ")
        signal_val = sig.get("signal", "?")
        # Color hint
        emoji = ""
        if signal_val in ("RISK-ON", "CREDIT_EASING", "NO_DIVERGENCE"):
            emoji = "[+]"
        elif signal_val in ("RISK-OFF", "CREDIT_STRESS", "WARNING", "INFLATIONARY", "STRONG_DOLLAR"):
            emoji = "[!]"
        else:
            emoji = "[~]"

        print(f"  {emoji} {label}: {signal_val}")
        for k, v in sig.items():
            if k not in ("signal",):
                print(f"      {k}: {v}")
        print()

    # --- Correlation ---
    if corr:
        print(f"\n{'='*70}")
        print("  SPY CORRELATION ANALYSIS (60-day)")
        print(f"{'='*70}\n")

        if "most_correlated_with_spy" in corr:
            print("  Most correlated with SPY:")
            for t, v in corr["most_correlated_with_spy"].items():
                print(f"    {t:<10} corr={v['corr_60d']:>6.3f}  (1yr avg={v['corr_1yr_avg']:>6.3f})")

        if "most_anticorrelated_with_spy" in corr:
            print("\n  Most anti-correlated with SPY:")
            for t, v in corr["most_anticorrelated_with_spy"].items():
                print(f"    {t:<10} corr={v['corr_60d']:>6.3f}  (1yr avg={v['corr_1yr_avg']:>6.3f})")

        if corr.get("correlation_anomalies"):
            print("\n  CORRELATION ANOMALIES (60d vs 1yr avg diff > 0.3):")
            for t, v in corr["correlation_anomalies"].items():
                print(f"    {t:<10} 60d={v['corr_60d']:>6.3f}  1yr={v['corr_1yr_avg']:>6.3f}  diff={v['diff']:>+6.3f}")
        else:
            print("\n  No correlation anomalies detected.")

    # --- Predictive Analysis ---
    if predictive:
        print(f"\n{'='*70}")
        print("  HISTORICAL PREDICTIVE SIGNALS (SPY forward returns)")
        print(f"{'='*70}\n")

        for name, p in predictive.items():
            label = name.upper().replace("_", " ")
            print(f"  {label}")
            print(f"    Condition: {p['condition']}")
            print(f"    Sample: {p['sample_size']} days")
            print(f"    SPY fwd 1W: {p['spy_fwd_1w_avg_pct']:>+.3f}% (WR {p['spy_fwd_1w_wr']:.0f}%)")
            print(f"    SPY fwd 2W: {p['spy_fwd_2w_avg_pct']:>+.3f}%")
            print(f"    SPY fwd 1M: {p['spy_fwd_1m_avg_pct']:>+.3f}% (WR {p['spy_fwd_1m_wr']:.0f}%)")
            print()

    print(f"{'='*70}")
    print(f"  Snapshot saved to {OUTPUT_DIR / 'latest_snapshot.json'}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    closes = download_data()

    print("Computing asset metrics...")
    metrics = compute_asset_metrics(closes)

    print("Building regime signals...")
    signals = build_regime_signals(metrics)

    print("Running correlation analysis...")
    corr = correlation_analysis(closes)

    print("Running predictive analysis...")
    predictive = predictive_analysis(closes)

    # Build full snapshot
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "asset_metrics": metrics,
        "regime_signals": signals,
        "correlation": corr,
        "predictive_analysis": predictive,
    }

    # Save
    out_path = OUTPUT_DIR / "latest_snapshot.json"
    with open(out_path, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)

    # Print summary
    print_summary(metrics, signals, corr, predictive)

    return snapshot


if __name__ == "__main__":
    main()
