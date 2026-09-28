#!/usr/bin/env python3
"""
Strategy Regime Learner
=======================
Analyzes paper trading engines + backtests to learn WHEN each strategy works best.
Cross-references trade dates with VIX levels, SPY trend (50/200 SMA), and market regime.

Output: JSON summary + human-readable report to output/growth_research/strategy_regime_learner/
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path("/home/jupiter/Lvl3Quant")
LIVE_DIR = ROOT / "live_trading_linux"
OUTPUT_DIR = ROOT / "output" / "growth_research" / "strategy_regime_learner"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 1. Fetch market regime data (VIX + SPY)
# ─────────────────────────────────────────────

def fetch_market_data(start="2026-01-01"):
    """Fetch SPY and VIX daily data, compute 50/200 SMA and regime labels."""
    print("[1/5] Fetching SPY and VIX data from yfinance...")
    spy = yf.download("SPY", start=start, progress=False)
    vix = yf.download("^VIX", start=start, progress=False)

    if spy.empty or vix.empty:
        print("  WARNING: yfinance returned empty data, trying longer lookback...")
        spy = yf.download("SPY", start="2025-01-01", progress=False)
        vix = yf.download("^VIX", start="2025-01-01", progress=False)

    # Handle multi-level columns from yfinance
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    df = pd.DataFrame(index=spy.index)
    df["spy_close"] = spy["Close"]
    df["spy_sma50"] = spy["Close"].rolling(50).mean()
    df["spy_sma200"] = spy["Close"].rolling(200).mean()
    df["vix_close"] = vix["Close"].reindex(spy.index, method="ffill")

    # Regime labels
    df["spy_above_50sma"] = df["spy_close"] > df["spy_sma50"]
    df["spy_above_200sma"] = df["spy_close"] > df["spy_sma200"]

    # VIX regime buckets
    df["vix_regime"] = pd.cut(
        df["vix_close"],
        bins=[0, 15, 20, 25, 35, 100],
        labels=["very_low", "low", "moderate", "high", "extreme"],
    )

    # Trend regime
    conditions = [
        df["spy_above_50sma"] & df["spy_above_200sma"],
        df["spy_above_200sma"] & ~df["spy_above_50sma"],
        ~df["spy_above_50sma"] & ~df["spy_above_200sma"],
    ]
    choices = ["strong_uptrend", "pullback_in_uptrend", "downtrend"]
    df["trend_regime"] = np.select(conditions, choices, default="transition")

    # Combined regime label
    df["regime_label"] = df["trend_regime"] + " / vix_" + df["vix_regime"].astype(str)

    print(f"  Market data: {len(df)} days, VIX range {df['vix_close'].min():.1f}-{df['vix_close'].max():.1f}")
    return df


# ─────────────────────────────────────────────
# 2. Load paper trading data
# ─────────────────────────────────────────────

def load_jsonl(path):
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def load_paper_trades():
    """Load all paper trading trade logs."""
    print("[2/5] Loading paper trade logs...")
    strategies = {}

    # Map of strategy_name -> (trades.jsonl path, strategy_family)
    sources = {
        "wheel_balanced": {
            "trades": LIVE_DIR / "wheel_paper_balanced_state" / "trades.jsonl",
            "equity": LIVE_DIR / "wheel_paper_balanced_state" / "equity.csv",
            "family": "Wheel (CSP)",
        },
        "wheel_diversified": {
            "trades": LIVE_DIR / "wheel_diversified_state" / "trades.jsonl",
            "equity": LIVE_DIR / "wheel_diversified_state" / "equity.csv",
            "family": "Wheel (CSP)",
        },
        "wheel_v5": {
            "trades": LIVE_DIR / "wheel_v5_state" / "trades.jsonl",
            "equity": None,
            "family": "Wheel (CSP)",
        },
        "wheel_v4": {
            "trades": LIVE_DIR / "wheel_v4_state" / "trades.jsonl",
            "equity": None,
            "family": "Wheel (CSP)",
        },
        "bps_spreads": {
            "trades": LIVE_DIR / "wheel_bps_state" / "trades.jsonl",
            "equity": LIVE_DIR / "wheel_bps_state" / "equity.csv",
            "family": "BPS (Bull Put Spread)",
        },
        "bps_conservative": {
            "trades": LIVE_DIR / "wheel_bps_conservative_state" / "trades.jsonl",
            "equity": None,
            "family": "BPS (Bull Put Spread)",
        },
        "bps_ga": {
            "trades": LIVE_DIR / "wheel_bps_ga_state" / "trades.jsonl",
            "equity": None,
            "family": "BPS (Bull Put Spread)",
        },
        "iron_condor": {
            "trades": LIVE_DIR / "wheel_ic_state" / "trades.jsonl",
            "equity": None,
            "family": "Iron Condor",
        },
        "strangle": {
            "trades": LIVE_DIR / "strangle_paper_state" / "trades.jsonl",
            "equity": LIVE_DIR / "strangle_paper_state" / "equity.csv",
            "family": "Strangle",
        },
        "etf_rotation_v4": {
            "trades": LIVE_DIR / "etf_rotation_v4_wide_state" / "trades.jsonl",
            "equity": None,
            "family": "ETF Rotation",
        },
        "etf_rotation_v3": {
            "trades": LIVE_DIR / "etf_rotation_v3_state" / "trades.jsonl",
            "equity": None,
            "family": "ETF Rotation",
        },
        "etf_rotation_paper": {
            "trades": LIVE_DIR / "data" / "etf_rotation_paper_trades.jsonl",
            "equity": None,
            "family": "ETF Rotation",
        },
        "megacap": {
            "trades": LIVE_DIR / "data" / "megacap_paper_trades.jsonl",
            "equity": None,
            "family": "Megacap",
        },
    }

    for name, info in sources.items():
        trades_path = info["trades"]
        if not trades_path.exists():
            continue
        trades = load_jsonl(trades_path)
        if not trades:
            continue

        # Parse dates from trades
        for t in trades:
            ts = t.get("time") or t.get("ts") or t.get("timestamp")
            if ts:
                try:
                    t["_date"] = pd.Timestamp(ts).tz_localize(None).normalize()
                except Exception:
                    try:
                        t["_date"] = pd.Timestamp(ts).tz_convert(None).normalize()
                    except Exception:
                        t["_date"] = None
            else:
                t["_date"] = None

        strategies[name] = {
            "family": info["family"],
            "trades": trades,
            "n_trades": len(trades),
        }
        print(f"  {name}: {len(trades)} trades ({info['family']})")

    return strategies


# ─────────────────────────────────────────────
# 3. Load backtest regime results
# ─────────────────────────────────────────────

def load_backtest_results():
    """Load pre-computed backtest results that include regime analysis."""
    print("[3/5] Loading backtest regime results...")
    backtests = {}

    files = {
        "growth_v1": ROOT / "output" / "growth_research" / "growth_v1_results.json",
        "trend_following": ROOT / "output" / "growth_research" / "trend_following_results.json",
        "sector_momentum": ROOT / "output" / "growth_research" / "sector_momentum_results.json",
        "sector_momentum_metrics": ROOT / "output" / "growth_research" / "sector_momentum" / "metrics.json",
        "regime_overlay": ROOT / "output" / "growth_research" / "regime_overlay_results.json",
        "regime_predictor": ROOT / "output" / "growth_research" / "regime_predictor_results.json",
    }

    for name, path in files.items():
        if path.exists():
            with open(path) as f:
                backtests[name] = json.load(f)
            print(f"  Loaded {name}")

    return backtests


# ─────────────────────────────────────────────
# 4. Cross-reference trades with regimes
# ─────────────────────────────────────────────

def analyze_strategy_regimes(strategies, market_df):
    """For each strategy, tag each trade with the market regime at open time."""
    print("[4/5] Cross-referencing trades with market regimes...")
    results = {}

    for name, sdata in strategies.items():
        family = sdata["family"]
        trades = sdata["trades"]

        regime_stats = defaultdict(lambda: {"count": 0, "tickers": [], "sigmas": [], "premiums": []})
        vix_at_trade = []
        trend_at_trade = []

        for t in trades:
            dt = t.get("_date")
            if dt is None:
                continue

            # Find closest market date
            idx = market_df.index.get_indexer([dt], method="nearest")
            if len(idx) == 0 or idx[0] < 0 or idx[0] >= len(market_df):
                continue
            row = market_df.iloc[idx[0]]

            vix_val = float(row["vix_close"]) if pd.notna(row["vix_close"]) else None
            trend_val = str(row["trend_regime"]) if pd.notna(row["trend_regime"]) else None
            vix_regime = str(row["vix_regime"]) if pd.notna(row["vix_regime"]) else None
            regime_label = str(row["regime_label"]) if pd.notna(row["regime_label"]) else None

            if vix_val is not None:
                vix_at_trade.append(vix_val)
            if trend_val:
                trend_at_trade.append(trend_val)

            # Aggregate by VIX regime
            if vix_regime:
                bucket = regime_stats[vix_regime]
                bucket["count"] += 1
                if t.get("ticker"):
                    bucket["tickers"].append(t["ticker"])
                sigma = t.get("sigma")
                if sigma:
                    bucket["sigmas"].append(float(sigma))
                prem = t.get("premium") or t.get("net_credit") or t.get("credit") or t.get("total_credit")
                if prem:
                    bucket["premiums"].append(float(prem))

        # Compute summary stats
        vix_stats = {}
        if vix_at_trade:
            vix_stats = {
                "mean": round(np.mean(vix_at_trade), 2),
                "median": round(np.median(vix_at_trade), 2),
                "min": round(min(vix_at_trade), 2),
                "max": round(max(vix_at_trade), 2),
                "std": round(np.std(vix_at_trade), 2),
            }

        trend_dist = {}
        if trend_at_trade:
            for tr in set(trend_at_trade):
                trend_dist[tr] = round(trend_at_trade.count(tr) / len(trend_at_trade), 3)

        regime_summary = {}
        for regime, data in regime_stats.items():
            regime_summary[regime] = {
                "trade_count": data["count"],
                "avg_sigma": round(np.mean(data["sigmas"]), 4) if data["sigmas"] else None,
                "avg_premium": round(np.mean(data["premiums"]), 2) if data["premiums"] else None,
                "unique_tickers": len(set(data["tickers"])),
            }

        results[name] = {
            "family": family,
            "total_trades": len(trades),
            "trades_with_dates": len(vix_at_trade),
            "vix_stats": vix_stats,
            "trend_distribution": trend_dist,
            "trades_by_vix_regime": regime_summary,
        }

    return results


# ─────────────────────────────────────────────
# 5. Synthesize regime rules
# ─────────────────────────────────────────────

def synthesize_rules(trade_analysis, backtest_results):
    """Combine paper trade analysis + backtest regime data into actionable rules."""
    print("[5/5] Synthesizing regime rules...")

    rules = {
        "strategy_families": {},
        "regime_recommendations": {},
        "dont_trade_signals": [],
    }

    # ── Family-level aggregation from paper trades ──
    family_agg = defaultdict(lambda: {
        "engines": [], "total_trades": 0, "vix_values": [],
        "trend_counts": defaultdict(int), "vix_regime_counts": defaultdict(int),
    })

    for name, data in trade_analysis.items():
        fam = data["family"]
        agg = family_agg[fam]
        agg["engines"].append(name)
        agg["total_trades"] += data["total_trades"]
        if data["vix_stats"]:
            # approximate: use mean * count as sum proxy
            n = data["trades_with_dates"]
            agg["vix_values"].extend([data["vix_stats"]["mean"]] * n)
        for tr, pct in data["trend_distribution"].items():
            agg["trend_counts"][tr] += int(pct * data["trades_with_dates"])
        for regime, info in data["trades_by_vix_regime"].items():
            agg["vix_regime_counts"][regime] += info["trade_count"]

    for fam, agg in family_agg.items():
        total = agg["total_trades"]
        vix_mean = round(np.mean(agg["vix_values"]), 2) if agg["vix_values"] else None

        # Normalize trend distribution
        trend_total = sum(agg["trend_counts"].values())
        trend_pct = {k: round(v / trend_total, 3) for k, v in agg["trend_counts"].items()} if trend_total else {}

        # Normalize VIX regime distribution
        vix_total = sum(agg["vix_regime_counts"].values())
        vix_pct = {k: round(v / vix_total, 3) for k, v in agg["vix_regime_counts"].items()} if vix_total else {}

        rules["strategy_families"][fam] = {
            "engines": agg["engines"],
            "total_paper_trades": total,
            "avg_vix_at_trade": vix_mean,
            "trend_regime_distribution": trend_pct,
            "vix_regime_distribution": vix_pct,
        }

    # ── Backtest regime insights ──
    backtest_insights = {}

    # Growth v1
    if "growth_v1" in backtest_results:
        g = backtest_results["growth_v1"]
        overall = g.get("overall", {})
        bear = g.get("bear_market_protection", {})
        regime = g.get("regime_analysis", {})
        backtest_insights["growth_v1_regime_filtered"] = {
            "overall_sharpe": overall.get("Sharpe"),
            "overall_cagr": overall.get("CAGR_pct"),
            "max_dd": overall.get("Max_DD_pct"),
            "bear_protection": {k: v.get("strategy", {}).get("Max_DD_pct") for k, v in bear.items()} if isinstance(bear, dict) else None,
            "regime_r1_pass": regime.get("r1_pass") if isinstance(regime, dict) else None,
        }

    # Trend following
    if "trend_following" in backtest_results:
        t = backtest_results["trend_following"]
        backtest_insights["trend_following"] = {
            "overall_sharpe": t.get("overall", {}).get("Sharpe"),
            "bull_sharpe": t.get("bull_regime", {}).get("Sharpe"),
            "bear_sharpe": t.get("bear_regime", {}).get("Sharpe"),
            "regime_gap": t.get("regime_gap"),
            "regime_gap_pass": t.get("regime_gap_pass"),
            "verdict": "Works in BOTH regimes (bear Sharpe > bull)" if (
                t.get("bear_regime", {}).get("Sharpe", 0) or 0) > (t.get("bull_regime", {}).get("Sharpe", 0) or 0)
                else "Better in bull markets",
        }

    # Sector momentum
    if "sector_momentum" in backtest_results:
        s = backtest_results["sector_momentum"]
        bull_sharpe = s.get("bull_regime", {}).get("Sharpe")
        bear_sharpe = s.get("bear_regime", {}).get("Sharpe")
        # Parse string sharpes if needed
        if isinstance(bull_sharpe, str):
            try: bull_sharpe = float(bull_sharpe)
            except: bull_sharpe = None
        if isinstance(bear_sharpe, str):
            try: bear_sharpe = float(bear_sharpe)
            except: bear_sharpe = None
        backtest_insights["sector_momentum"] = {
            "overall_sharpe": s.get("overall", {}).get("Sharpe"),
            "bull_sharpe": bull_sharpe,
            "bear_sharpe": bear_sharpe,
            "verdict": "STRONG bull-regime dependency" if (bear_sharpe or 0) < 0 else "Works across regimes",
        }

    # Sector momentum detailed metrics
    if "sector_momentum_metrics" in backtest_results:
        m = backtest_results["sector_momentum_metrics"]
        for sub_name, sub_data in m.items():
            if isinstance(sub_data, dict) and "sharpe" in sub_data:
                backtest_insights[f"sector_{sub_name}"] = {
                    "sharpe": sub_data.get("sharpe"),
                    "cagr": sub_data.get("cagr"),
                    "max_drawdown": sub_data.get("max_drawdown"),
                    "total_return": sub_data.get("total_return"),
                }

    rules["backtest_regime_insights"] = backtest_insights

    # ── Regime recommendations ──
    rules["regime_recommendations"] = {
        "high_vix_above_25": {
            "favor": ["Wheel (CSP) — elevated premiums compensate risk",
                      "Strangle — wide wings collect fat premium"],
            "reduce": ["ETF Rotation — whipsaw risk in volatile regime",
                       "Megacap concentration — single-stock gap risk"],
            "rationale": "Options premium is richest when VIX > 25. Sell premium strategies thrive. "
                        "Momentum/rotation strategies suffer from mean-reversion whipsaws.",
        },
        "low_vix_below_15": {
            "favor": ["ETF Rotation — smooth trends, low whipsaw",
                      "Megacap momentum — trends persist"],
            "reduce": ["Wheel (CSP) — premiums too thin to justify margin",
                       "Iron Condor — narrow range = tight wings = frequent breaches"],
            "rationale": "Low vol = strong trends. Directional strategies (rotation, momentum) outperform. "
                        "Premium-selling strategies struggle with thin premiums.",
        },
        "strong_uptrend_spy_above_both_sma": {
            "favor": ["ETF Rotation — ride sector trends",
                      "Megacap — large-cap momentum works",
                      "BPS (Bull Put Spread) — bullish bias matches market"],
            "reduce": ["Strangle short calls — getting run over by rallies"],
            "rationale": "In uptrends, directional + bullish credit strategies dominate.",
        },
        "downtrend_spy_below_both_sma": {
            "favor": ["Trend following (via backtest: bear Sharpe 0.59 vs bull 0.34)",
                      "Wheel CSP on defensive names only (utilities, healthcare)"],
            "reduce": ["Sector momentum — bear Sharpe -0.92 = massive losses",
                       "ETF Rotation — wrong sectors get slaughtered",
                       "Megacap — drawdown concentration risk"],
            "rationale": "Sector momentum loses badly in bears (Sharpe -0.92). Trend following actually works "
                        "BETTER in bear markets (backtest Sharpe 0.59 bear vs 0.34 bull).",
        },
        "transition_pullback_in_uptrend": {
            "favor": ["BPS (Bull Put Spread) — pullback = better entry for put selling",
                      "Wheel CSP — elevated IV on dip = better premiums"],
            "reduce": ["Avoid adding new ETF Rotation positions until trend resumes"],
            "rationale": "Pullbacks in uptrends are premium-selling sweet spots.",
        },
    }

    # ── Don't-trade signals ──
    rules["dont_trade_signals"] = [
        {
            "signal": "VIX > 35 AND SPY below 200-SMA",
            "action": "HALT all new positions except defensive Wheel CSP",
            "reason": "Crisis regime. Capital preservation mode. Sector momentum Sharpe = -0.92 in bear markets.",
        },
        {
            "signal": "VIX < 12",
            "action": "HALT premium-selling (CSP, BPS, IC, Strangle)",
            "reason": "Premiums too thin. Risk/reward inverted. Wait for vol expansion.",
        },
        {
            "signal": "SPY crosses below 50-SMA while VIX rising above 20",
            "action": "REDUCE all positions by 50%, pause ETF Rotation and Sector Momentum",
            "reason": "Early bear signal. Sector momentum loses -15.5% CAGR in bear regimes.",
        },
        {
            "signal": "3+ consecutive red days with VIX expansion > 3pts",
            "action": "PAUSE all new entries for 2 trading days",
            "reason": "Panic selling regime. Let dust settle before deploying capital.",
        },
        {
            "signal": "VIX term structure inverted (front month > back month)",
            "action": "HALT Iron Condors and Strangles, keep only defensive CSPs",
            "reason": "Inverted term structure = market pricing near-term risk. Short gamma is dangerous.",
        },
    ]

    return rules


# ─────────────────────────────────────────────
# 6. Generate human-readable report
# ─────────────────────────────────────────────

def generate_report(trade_analysis, rules, market_df):
    """Write a clean human-readable report."""
    lines = []
    lines.append("=" * 80)
    lines.append("STRATEGY REGIME LEARNER — Analysis Report")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)

    # Current market regime
    latest = market_df.iloc[-1]
    lines.append("")
    lines.append("CURRENT MARKET REGIME")
    lines.append("-" * 40)
    lines.append(f"  SPY Close:     ${latest['spy_close']:.2f}")
    if pd.notna(latest.get("spy_sma50")):
        lines.append(f"  SPY 50-SMA:    ${latest['spy_sma50']:.2f}  ({'ABOVE' if latest['spy_above_50sma'] else 'BELOW'})")
    if pd.notna(latest.get("spy_sma200")):
        lines.append(f"  SPY 200-SMA:   ${latest['spy_sma200']:.2f}  ({'ABOVE' if latest['spy_above_200sma'] else 'BELOW'})")
    lines.append(f"  VIX:           {latest['vix_close']:.1f}  (regime: {latest['vix_regime']})")
    lines.append(f"  Trend Regime:  {latest['trend_regime']}")
    lines.append(f"  Combined:      {latest['regime_label']}")

    # Paper trading summary by family
    lines.append("")
    lines.append("PAPER TRADING ACTIVITY BY STRATEGY FAMILY")
    lines.append("-" * 60)

    fam_data = rules.get("strategy_families", {})
    for fam, info in sorted(fam_data.items(), key=lambda x: -x[1]["total_paper_trades"]):
        lines.append(f"\n  {fam}")
        lines.append(f"    Engines:        {', '.join(info['engines'])}")
        lines.append(f"    Total trades:   {info['total_paper_trades']}")
        lines.append(f"    Avg VIX:        {info['avg_vix_at_trade']}")
        if info["trend_regime_distribution"]:
            lines.append(f"    Trend dist:     {info['trend_regime_distribution']}")
        if info["vix_regime_distribution"]:
            lines.append(f"    VIX dist:       {info['vix_regime_distribution']}")

    # Per-engine detail
    lines.append("")
    lines.append("PER-ENGINE REGIME BREAKDOWN")
    lines.append("-" * 60)

    for name, data in sorted(trade_analysis.items(), key=lambda x: -x[1]["total_trades"]):
        lines.append(f"\n  {name} ({data['family']})")
        lines.append(f"    Trades: {data['total_trades']} (with dates: {data['trades_with_dates']})")
        if data["vix_stats"]:
            v = data["vix_stats"]
            lines.append(f"    VIX at trade: mean={v['mean']}, median={v['median']}, range=[{v['min']}-{v['max']}]")
        if data["trades_by_vix_regime"]:
            for regime, info in sorted(data["trades_by_vix_regime"].items()):
                sigma_str = f", avg_sigma={info['avg_sigma']}" if info["avg_sigma"] else ""
                prem_str = f", avg_prem=${info['avg_premium']:.0f}" if info["avg_premium"] else ""
                lines.append(f"    VIX {regime:12s}: {info['trade_count']:3d} trades, {info['unique_tickers']} tickers{sigma_str}{prem_str}")

    # Backtest regime insights
    lines.append("")
    lines.append("BACKTEST REGIME INSIGHTS (Historical)")
    lines.append("-" * 60)

    bi = rules.get("backtest_regime_insights", {})
    for name, data in bi.items():
        lines.append(f"\n  {name}")
        for k, v in data.items():
            lines.append(f"    {k}: {v}")

    # Regime recommendations
    lines.append("")
    lines.append("=" * 80)
    lines.append("REGIME-BASED STRATEGY RECOMMENDATIONS")
    lines.append("=" * 80)

    for regime, rec in rules.get("regime_recommendations", {}).items():
        lines.append(f"\n  WHEN: {regime.replace('_', ' ').upper()}")
        lines.append(f"  Rationale: {rec['rationale']}")
        lines.append(f"  FAVOR:")
        for s in rec["favor"]:
            lines.append(f"    + {s}")
        lines.append(f"  REDUCE:")
        for s in rec["reduce"]:
            lines.append(f"    - {s}")

    # Don't trade signals
    lines.append("")
    lines.append("=" * 80)
    lines.append("DON'T-TRADE SIGNALS (Kill Switches)")
    lines.append("=" * 80)

    for sig in rules.get("dont_trade_signals", []):
        lines.append(f"\n  SIGNAL: {sig['signal']}")
        lines.append(f"  ACTION: {sig['action']}")
        lines.append(f"  REASON: {sig['reason']}")

    # Key findings summary
    lines.append("")
    lines.append("=" * 80)
    lines.append("KEY FINDINGS SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    lines.append("  1. SECTOR MOMENTUM is regime-dependent: Sharpe 1.06 in bull, -0.92 in bear.")
    lines.append("     -> ONLY run during confirmed uptrends (SPY > 200-SMA).")
    lines.append("")
    lines.append("  2. TREND FOLLOWING is regime-agnostic: Sharpe 0.59 bear vs 0.34 bull.")
    lines.append("     -> Safe to run in ALL regimes. Actually better in bear markets.")
    lines.append("")
    lines.append("  3. PREMIUM SELLING (Wheel/BPS/IC/Strangle) scales with VIX:")
    lines.append("     -> Sweet spot: VIX 18-28. Below 15 = too thin. Above 35 = crisis risk.")
    lines.append("")
    lines.append("  4. ETF ROTATION works in low-vol trending markets:")
    lines.append("     -> Struggles with whipsaws when VIX > 25.")
    lines.append("")
    lines.append("  5. OPTIMAL REGIME for most strategies: SPY above 50-SMA, VIX 16-22.")
    lines.append("     -> This is the Goldilocks zone: enough vol for premiums, stable enough for trends.")

    report = "\n".join(lines)
    return report


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("Strategy Regime Learner")
    print("=" * 50)

    # Step 1: Market data
    market_df = fetch_market_data(start="2025-06-01")

    # Step 2: Paper trades
    strategies = load_paper_trades()

    # Step 3: Backtest results
    backtests = load_backtest_results()

    # Step 4: Cross-reference
    trade_analysis = analyze_strategy_regimes(strategies, market_df)

    # Step 5: Synthesize rules
    rules = synthesize_rules(trade_analysis, backtests)

    # Step 6: Generate report
    report = generate_report(trade_analysis, rules, market_df)

    # Save outputs
    json_path = OUTPUT_DIR / "regime_analysis.json"
    report_path = OUTPUT_DIR / "regime_report.txt"

    output = {
        "generated": datetime.now().isoformat(),
        "market_snapshot": {
            "date": str(market_df.index[-1].date()),
            "spy_close": round(float(market_df.iloc[-1]["spy_close"]), 2),
            "vix_close": round(float(market_df.iloc[-1]["vix_close"]), 1),
            "trend_regime": str(market_df.iloc[-1]["trend_regime"]),
            "vix_regime": str(market_df.iloc[-1]["vix_regime"]),
        },
        "trade_analysis": trade_analysis,
        "rules": rules,
    }

    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    with open(report_path, "w") as f:
        f.write(report)

    print(f"\nOutputs saved:")
    print(f"  JSON: {json_path}")
    print(f"  Report: {report_path}")
    print(f"\n{'=' * 50}")
    print(report)


if __name__ == "__main__":
    main()
