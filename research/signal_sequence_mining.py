#!/usr/bin/env python3
"""
Signal Sequence Pattern Mining
==============================
Instead of finding new signal sources, mine TEMPORAL PATTERNS in how existing
signals fire over time. Do certain sequences predict better outcomes?

Patterns tested:
1. Building confluence (signals accumulating over consecutive days)
2. Sudden burst (0 to many signals in one day)
3. Signal persistence (active 3+ consecutive days)
4. Flash signal (fires once, gone next day)
5. Re-ignition (signal fires, goes quiet 2-3 days, fires again)
6. Velocity (increasing vs decreasing confluence day-over-day)

Data sources:
- Paper engine state files (all sector ETF engines)
- Agentic trade log (56 trades with signal types)
- yfinance for actual forward returns
"""

import json
import glob
import os
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timedelta, date
from pathlib import Path
import traceback

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance required. pip install yfinance")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
BASE = Path("/home/jupiter/Lvl3Quant")
STATE = BASE / "state"
OUTPUT = BASE / "research" / "signal_sequence_patterns.json"

SECTOR_ETFS = ["XLF", "XLE", "XLU", "XLK", "XLY", "XLP", "XLRE", "XLV", "XLI", "XLB", "XLC"]
ALL_TICKERS = SECTOR_ETFS + ["SPY"]

# Signal source names (from agentic signal aggregator engines)
SIGNAL_SOURCES = [
    "sector_spreads", "quality_momentum", "pead_drift", "vix_call_spread",
    "sector_etf_momentum", "signal_watcher", "sector_v91", "sector_v93",
    "sector_v10_optimal", "sector_mom_spreads", "gap_fade_spread",
    "subsector_rotation", "subsector_ml_predictions", "subsector_validated_pairs",
    "extreme_idio", "volume_surge", "vol_crush", "vix_mr_spread",
    "bond_yield_inflow", "momentum_options", "rsi_divergence",
    "equity_rotation_rank", "cta_trend", "strong_momentum", "rsi_bullish",
    "rsi_bearish", "mom_accelerating", "lgbm_sector_rotation"
]


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Extract ALL signal firings from paper engine states
# ─────────────────────────────────────────────────────────────────────────────
def extract_signal_events():
    """
    Extract signal events from all paper engines.
    Returns list of dicts: {date, ticker, direction, engine, signals, n_signals, confidence}
    """
    events = []

    # 1a. Sector combined engines — trades have entry dates + modes
    for f in sorted(glob.glob(str(STATE / "sector_combined_*paper_state.json"))):
        engine = os.path.basename(f).replace("_paper_state.json", "")
        d = load_json(f)
        if not d:
            continue

        for t in d.get("closed_trades", []) + d.get("open_positions", []):
            entry_date = t.get("entry_date")
            ticker = t.get("ticker")
            mode = t.get("mode", "")
            direction = "bull" if mode == "bull" else "bear"

            if entry_date and ticker:
                signals = t.get("signals", [])
                events.append({
                    "date": entry_date,
                    "ticker": ticker,
                    "direction": direction,
                    "engine": engine,
                    "signals": signals if signals else [engine],
                    "n_signals": t.get("n_signals", len(signals) if signals else 1),
                    "confidence": t.get("lgbm_score", t.get("confidence", 0.5)),
                    "pnl": t.get("pnl"),
                    "exit_date": t.get("date") if "action" in t else None,
                    "exit_reason": t.get("exit_reason"),
                    "status": "closed" if "action" in t or "pnl" in t else "open",
                })

    # 1b. Other engines with sector ETF trades
    other_engines = [
        "sector_spreads", "sector_spreads_v6", "sector_momentum_spreads",
        "sector_pairs", "sector_reversal", "sector_earnings_standalone",
        "sector_equity_rotation", "subsector_rotation", "sector_etf_momentum",
        "momentum_options", "rsi_divergence", "extreme_idio", "vol_crush",
        "vix_call_spread", "liquidity_signal", "bond_yield", "iv_runup",
    ]
    for eng in other_engines:
        f = STATE / f"{eng}_paper_state.json"
        d = load_json(f)
        if not d:
            continue
        for t in d.get("closed_trades", []) + d.get("open_positions", []):
            ticker = t.get("ticker", "")
            if ticker not in SECTOR_ETFS:
                continue
            entry_date = t.get("entry_date")
            if not entry_date:
                continue
            direction = t.get("direction", t.get("mode", "bull"))
            if direction in ("call", "long"):
                direction = "bull"
            elif direction in ("put", "short"):
                direction = "bear"

            signals = t.get("signals", [])
            events.append({
                "date": entry_date,
                "ticker": ticker,
                "direction": direction,
                "engine": eng,
                "signals": signals if signals else [eng],
                "n_signals": t.get("n_signals", max(1, len(signals))),
                "confidence": t.get("lgbm_score", t.get("confidence", t.get("rank_score", 0.5))),
                "pnl": t.get("pnl"),
                "exit_date": t.get("exit_date", t.get("date")),
                "exit_reason": t.get("exit_reason"),
                "status": "closed" if t.get("pnl") is not None else "open",
            })

    # 1c. Agentic trade log — explicit signal types
    atl = load_json(STATE / "agentic_trade_log.json")
    if atl and "trades" in atl:
        for t in atl["trades"]:
            ticker = t.get("ticker", "")
            if ticker not in SECTOR_ETFS:
                continue
            entry_date = t.get("entry_date")
            if not entry_date:
                continue
            direction = "bull" if t.get("direction") == "call" else "bear"
            events.append({
                "date": entry_date,
                "ticker": ticker,
                "direction": direction,
                "engine": "agentic_" + t.get("setup_type", "unknown"),
                "signals": [t.get("setup_type", "unknown")],
                "n_signals": 1,
                "confidence": t.get("confidence", 0.5),
                "pnl": t.get("exit_pnl"),
                "exit_date": t.get("exit_date"),
                "exit_reason": t.get("status"),
                "status": t.get("status", "unknown"),
            })

    print(f"  Extracted {len(events)} signal events")
    return events


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Build daily signal matrix per ticker
# ─────────────────────────────────────────────────────────────────────────────
def build_daily_signal_matrix(events):
    """
    Build a daily time series per ticker:
      - n_engines_active: how many engines fired that day
      - engines_active: list of engine names
      - direction: consensus direction
      - confidence_avg: average confidence
    """
    # Group by (date, ticker)
    daily = defaultdict(lambda: {"engines": set(), "directions": [], "confidences": [], "signals": set()})

    for e in events:
        key = (e["date"], e["ticker"])
        daily[key]["engines"].add(e["engine"])
        daily[key]["directions"].append(e["direction"])
        daily[key]["confidences"].append(e["confidence"])
        for s in e.get("signals", []):
            daily[key]["signals"].add(s)

    # Convert to DataFrame-friendly format
    rows = []
    for (dt, ticker), info in sorted(daily.items()):
        bull_count = sum(1 for d in info["directions"] if d == "bull")
        bear_count = sum(1 for d in info["directions"] if d == "bear")
        rows.append({
            "date": dt,
            "ticker": ticker,
            "n_engines": len(info["engines"]),
            "n_signals": len(info["signals"]),
            "engines": sorted(info["engines"]),
            "signals": sorted(info["signals"]),
            "confidence_avg": np.mean(info["confidences"]) if info["confidences"] else 0.5,
            "direction": "bull" if bull_count > bear_count else "bear" if bear_count > bull_count else "mixed",
            "bull_count": bull_count,
            "bear_count": bear_count,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], format="mixed", utc=False)
        # Normalize to date only (strip time component)
        df["date"] = df["date"].dt.normalize()
    print(f"  Built daily matrix: {len(df)} rows, {df['ticker'].nunique()} tickers, "
          f"date range {df['date'].min().strftime('%Y-%m-%d') if len(df) > 0 else '?'} to "
          f"{df['date'].max().strftime('%Y-%m-%d') if len(df) > 0 else '?'}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Fetch price data and compute forward returns
# ─────────────────────────────────────────────────────────────────────────────
def fetch_prices(tickers, start_date, end_date):
    """Fetch daily OHLCV from yfinance."""
    print(f"  Fetching prices for {len(tickers)} tickers from {start_date} to {end_date}...")
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if data.empty:
        print("  WARNING: No price data returned!")
        return pd.DataFrame()

    # Get close prices
    if isinstance(data.columns, pd.MultiIndex):
        closes = data["Close"]
    else:
        closes = data[["Close"]]
        closes.columns = [tickers[0]] if isinstance(tickers, list) and len(tickers) == 1 else tickers

    # Compute forward returns
    fwd_1d = closes.pct_change(1).shift(-1)
    fwd_3d = closes.pct_change(3).shift(-3)
    fwd_5d = closes.pct_change(5).shift(-5)
    fwd_10d = closes.pct_change(10).shift(-10)

    print(f"  Got {len(closes)} trading days of price data")
    return closes, fwd_1d, fwd_3d, fwd_5d, fwd_10d


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Compute temporal features for each ticker's signal series
# ─────────────────────────────────────────────────────────────────────────────
def compute_temporal_features(daily_df, all_dates):
    """
    For each ticker, compute temporal signal features across the full date range.
    Returns DataFrame with features per (date, ticker).
    """
    features_list = []

    for ticker in daily_df["ticker"].unique():
        ticker_df = daily_df[daily_df["ticker"] == ticker].set_index("date").sort_index()

        # Create full date range for this ticker
        full_range = pd.DataFrame(index=all_dates)
        full_range["n_engines"] = 0
        full_range["n_signals"] = 0
        full_range["confidence_avg"] = 0.0
        full_range["has_signal"] = False

        # Fill in days with signals
        for dt in ticker_df.index:
            if dt in full_range.index:
                row = ticker_df.loc[dt]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                full_range.loc[dt, "n_engines"] = row["n_engines"]
                full_range.loc[dt, "n_signals"] = row["n_signals"]
                full_range.loc[dt, "confidence_avg"] = row["confidence_avg"]
                full_range.loc[dt, "has_signal"] = True

        # Compute temporal features
        for dt in all_dates:
            if not full_range.loc[dt, "has_signal"]:
                continue

            lookback = full_range.loc[:dt].tail(10)  # 10 day lookback

            # 1. Consecutive days active (streak length)
            streak = 0
            for i in range(len(lookback) - 1, -1, -1):
                if lookback.iloc[i]["has_signal"]:
                    streak += 1
                else:
                    break

            # 2. Signal velocity (change in n_engines over last 3 days)
            last_3 = lookback.tail(3)
            active_3 = last_3[last_3["has_signal"]]
            if len(active_3) >= 2:
                velocity = active_3["n_engines"].iloc[-1] - active_3["n_engines"].iloc[0]
            else:
                velocity = 0

            # 3. Building confluence (more engines each day for 3+ days)
            building = False
            if streak >= 3:
                recent_streak = lookback[lookback["has_signal"]].tail(streak)
                if len(recent_streak) >= 3:
                    engines_seq = recent_streak["n_engines"].values
                    building = all(engines_seq[i+1] >= engines_seq[i] for i in range(len(engines_seq)-1))

            # 4. Sudden burst (from 0 or 1 signal yesterday to 3+ today)
            prev_day = lookback.iloc[-2] if len(lookback) >= 2 else None
            today = lookback.iloc[-1]
            sudden_burst = False
            if prev_day is not None:
                if (not prev_day["has_signal"] or prev_day["n_engines"] <= 1) and today["n_engines"] >= 3:
                    sudden_burst = True

            # 5. Re-ignition detection (fired, quiet 1-3 days, fired again)
            re_ignition = False
            if len(lookback) >= 4:
                signal_pattern = lookback["has_signal"].values[-5:] if len(lookback) >= 5 else lookback["has_signal"].values
                # Look for pattern: True, [False]*1-3, True
                for gap_len in [1, 2, 3]:
                    if len(signal_pattern) >= gap_len + 2:
                        end_idx = len(signal_pattern) - 1
                        start_idx = end_idx - gap_len - 1
                        if start_idx >= 0:
                            if (signal_pattern[start_idx] and
                                not any(signal_pattern[start_idx+1:end_idx]) and
                                signal_pattern[end_idx]):
                                re_ignition = True
                                break

            # 6. Signal persistence (active 3+ consecutive days)
            persistent = streak >= 3

            # 7. Flash signal (no signal yesterday or tomorrow)
            flash = streak == 1

            # 8. Days since last signal for this ticker
            prior_signals = lookback.iloc[:-1][lookback.iloc[:-1]["has_signal"]]
            days_since_last = (dt - prior_signals.index[-1]).days if len(prior_signals) > 0 else 999

            # Get direction from daily_df
            direction = "unknown"
            if dt in ticker_df.index:
                r = ticker_df.loc[dt]
                if isinstance(r, pd.DataFrame):
                    r = r.iloc[0]
                direction = r.get("direction", "unknown")

            features_list.append({
                "date": dt,
                "ticker": ticker,
                "n_engines": int(today["n_engines"]),
                "n_signals": int(today["n_signals"]),
                "confidence": float(today["confidence_avg"]),
                "direction": direction,
                "streak_days": streak,
                "velocity": velocity,
                "building_confluence": building,
                "sudden_burst": sudden_burst,
                "re_ignition": re_ignition,
                "persistent_signal": persistent,
                "flash_signal": flash,
                "days_since_last": days_since_last,
            })

    feat_df = pd.DataFrame(features_list)
    if not feat_df.empty:
        feat_df["date"] = pd.to_datetime(feat_df["date"])
    print(f"  Computed temporal features: {len(feat_df)} signal-days")
    return feat_df


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Test patterns against forward returns
# ─────────────────────────────────────────────────────────────────────────────
def compute_pattern_stats(feat_df, fwd_returns, horizon_name="fwd_5d"):
    """
    For each pattern, compute directional returns and stats.
    """
    results = {}

    # Merge features with forward returns
    merged = feat_df.copy()
    merged["fwd_ret"] = np.nan

    for idx, row in merged.iterrows():
        ticker = row["ticker"]
        dt = row["date"]
        if ticker in fwd_returns.columns and dt in fwd_returns.index:
            ret = fwd_returns.loc[dt, ticker]
            # Directional: if bull signal, positive return is good; if bear, negative is good
            if row["direction"] == "bear":
                ret = -ret  # Flip so positive = signal was right
            merged.at[idx, "fwd_ret"] = ret

    merged = merged.dropna(subset=["fwd_ret"])
    if merged.empty:
        return {"error": "No return data available"}

    print(f"  Merged {len(merged)} signal-days with forward returns ({horizon_name})")

    # Baseline: all signals
    baseline = compute_group_stats(merged["fwd_ret"], "all_signals")
    results["baseline"] = baseline

    # Pattern 1: Building confluence vs not
    building = merged[merged["building_confluence"]]
    not_building = merged[~merged["building_confluence"]]
    results["building_confluence"] = {
        "yes": compute_group_stats(building["fwd_ret"], "building_confluence=True"),
        "no": compute_group_stats(not_building["fwd_ret"], "building_confluence=False"),
    }

    # Pattern 2: Sudden burst
    burst = merged[merged["sudden_burst"]]
    no_burst = merged[~merged["sudden_burst"]]
    results["sudden_burst"] = {
        "yes": compute_group_stats(burst["fwd_ret"], "sudden_burst=True"),
        "no": compute_group_stats(no_burst["fwd_ret"], "sudden_burst=False"),
    }

    # Pattern 3: Persistent (3+ days) vs flash (1 day)
    persistent = merged[merged["persistent_signal"]]
    flash = merged[merged["flash_signal"]]
    results["persistence"] = {
        "persistent_3plus_days": compute_group_stats(persistent["fwd_ret"], "persistent"),
        "flash_1_day": compute_group_stats(flash["fwd_ret"], "flash"),
    }

    # Pattern 4: Re-ignition
    reignite = merged[merged["re_ignition"]]
    no_reignite = merged[~merged["re_ignition"]]
    results["re_ignition"] = {
        "yes": compute_group_stats(reignite["fwd_ret"], "re_ignition=True"),
        "no": compute_group_stats(no_reignite["fwd_ret"], "re_ignition=False"),
    }

    # Pattern 5: Velocity (positive = building, negative = fading)
    positive_vel = merged[merged["velocity"] > 0]
    negative_vel = merged[merged["velocity"] < 0]
    zero_vel = merged[merged["velocity"] == 0]
    results["velocity"] = {
        "increasing": compute_group_stats(positive_vel["fwd_ret"], "velocity>0"),
        "decreasing": compute_group_stats(negative_vel["fwd_ret"], "velocity<0"),
        "flat": compute_group_stats(zero_vel["fwd_ret"], "velocity=0"),
    }

    # Pattern 6: High confidence (top quartile) vs low
    if len(merged) >= 8:
        q75 = merged["confidence"].quantile(0.75)
        q25 = merged["confidence"].quantile(0.25)
        high_conf = merged[merged["confidence"] >= q75]
        low_conf = merged[merged["confidence"] <= q25]
        results["confidence_split"] = {
            "high_q75": compute_group_stats(high_conf["fwd_ret"], f"confidence>={q75:.2f}"),
            "low_q25": compute_group_stats(low_conf["fwd_ret"], f"confidence<={q25:.2f}"),
        }

    # Pattern 7: N_engines buckets
    for threshold in [1, 2, 3, 5]:
        subset = merged[merged["n_engines"] >= threshold]
        results[f"n_engines_ge_{threshold}"] = compute_group_stats(
            subset["fwd_ret"], f"n_engines>={threshold}"
        )

    # Pattern 8: Streak length buckets
    for streak in [1, 2, 3, 5]:
        subset = merged[merged["streak_days"] >= streak]
        results[f"streak_ge_{streak}"] = compute_group_stats(
            subset["fwd_ret"], f"streak>={streak}"
        )

    # Pattern 9: Combined patterns (high value combos)
    # Persistent + high confidence
    if len(merged) >= 4:
        q_med = merged["confidence"].median()
        combo1 = merged[(merged["persistent_signal"]) & (merged["confidence"] >= q_med)]
        results["combo_persistent_highconf"] = compute_group_stats(
            combo1["fwd_ret"], "persistent+high_confidence"
        )

    # Re-ignition + building
    combo2 = merged[(merged["re_ignition"]) & (merged["n_engines"] >= 2)]
    results["combo_reignition_multiengine"] = compute_group_stats(
        combo2["fwd_ret"], "re_ignition+multi_engine"
    )

    # Flash + high confidence (contrarian indicator?)
    if len(merged) >= 4:
        combo3 = merged[(merged["flash_signal"]) & (merged["confidence"] >= q_med)]
        results["combo_flash_highconf"] = compute_group_stats(
            combo3["fwd_ret"], "flash+high_confidence"
        )

    return results


def compute_group_stats(returns, label):
    """Compute trading stats for a group of returns."""
    if len(returns) == 0:
        return {"n": 0, "label": label, "insufficient_data": True}

    rets = returns.values
    n = len(rets)
    mean_ret = float(np.mean(rets))
    std_ret = float(np.std(rets)) if n > 1 else 0.0
    wins = int(np.sum(rets > 0))
    losses = int(np.sum(rets <= 0))
    wr = wins / n if n > 0 else 0.0

    # Annualized Sharpe (assuming 5-day holding = ~50 trades/year)
    sharpe = (mean_ret / std_ret) * np.sqrt(252 / 5) if std_ret > 0 else 0.0

    # Profit factor
    gross_wins = float(np.sum(rets[rets > 0])) if wins > 0 else 0.0
    gross_losses = float(abs(np.sum(rets[rets <= 0]))) if losses > 0 else 0.001
    pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    # Sortino
    downside = rets[rets < 0]
    downside_std = float(np.std(downside)) if len(downside) > 1 else std_ret
    sortino = (mean_ret / downside_std) * np.sqrt(252 / 5) if downside_std > 0 else 0.0

    # Max drawdown (cumulative)
    cum = np.cumsum(rets)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0

    return {
        "label": label,
        "n": n,
        "mean_return_pct": round(mean_ret * 100, 3),
        "std_return_pct": round(std_ret * 100, 3),
        "win_rate": round(wr, 3),
        "wins": wins,
        "losses": losses,
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(pf, 3),
        "max_dd_pct": round(max_dd * 100, 3),
        "median_return_pct": round(float(np.median(rets)) * 100, 3),
        "total_return_pct": round(float(np.sum(rets)) * 100, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Regime stratification
# ─────────────────────────────────────────────────────────────────────────────
def regime_stratification(feat_df, fwd_returns, spy_fwd):
    """Split analysis by market regime (green/red days based on SPY)."""
    results = {}

    # Classify each date as green/red based on SPY 5-day return
    green_dates = set()
    red_dates = set()
    flat_dates = set()

    for dt in spy_fwd.index:
        ret = spy_fwd.loc[dt]
        if isinstance(ret, pd.Series):
            ret = ret.iloc[0]
        if pd.isna(ret):
            continue
        if ret > 0.005:
            green_dates.add(dt)
        elif ret < -0.005:
            red_dates.add(dt)
        else:
            flat_dates.add(dt)

    # Merge features with returns
    merged = feat_df.copy()
    merged["fwd_ret"] = np.nan
    merged["regime"] = "unknown"

    for idx, row in merged.iterrows():
        ticker = row["ticker"]
        dt = row["date"]
        if ticker in fwd_returns.columns and dt in fwd_returns.index:
            ret = fwd_returns.loc[dt, ticker]
            if row["direction"] == "bear":
                ret = -ret
            merged.at[idx, "fwd_ret"] = ret

        if dt in green_dates:
            merged.at[idx, "regime"] = "green"
        elif dt in red_dates:
            merged.at[idx, "regime"] = "red"
        elif dt in flat_dates:
            merged.at[idx, "regime"] = "flat"

    merged = merged.dropna(subset=["fwd_ret"])

    for regime in ["green", "red", "flat"]:
        subset = merged[merged["regime"] == regime]
        if len(subset) < 3:
            results[regime] = {"n": len(subset), "insufficient_data": True}
            continue
        results[regime] = {
            "n": len(subset),
            "baseline": compute_group_stats(subset["fwd_ret"], f"regime={regime}"),
            "persistent": compute_group_stats(
                subset[subset["persistent_signal"]]["fwd_ret"], f"{regime}+persistent"
            ),
            "flash": compute_group_stats(
                subset[subset["flash_signal"]]["fwd_ret"], f"{regime}+flash"
            ),
            "reignition": compute_group_stats(
                subset[subset["re_ignition"]]["fwd_ret"], f"{regime}+reignition"
            ),
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 7: Per-ticker pattern analysis
# ─────────────────────────────────────────────────────────────────────────────
def per_ticker_analysis(feat_df, fwd_returns):
    """Which tickers respond best to which patterns?"""
    results = {}

    for ticker in feat_df["ticker"].unique():
        ticker_feat = feat_df[feat_df["ticker"] == ticker].copy()
        if len(ticker_feat) < 3:
            continue

        # Merge with returns
        ticker_feat["fwd_ret"] = np.nan
        for idx, row in ticker_feat.iterrows():
            dt = row["date"]
            if ticker in fwd_returns.columns and dt in fwd_returns.index:
                ret = fwd_returns.loc[dt, ticker]
                if row["direction"] == "bear":
                    ret = -ret
                ticker_feat.at[idx, "fwd_ret"] = ret

        ticker_feat = ticker_feat.dropna(subset=["fwd_ret"])
        if len(ticker_feat) < 3:
            continue

        results[ticker] = {
            "n_signals": len(ticker_feat),
            "baseline": compute_group_stats(ticker_feat["fwd_ret"], f"{ticker}_all"),
            "patterns": {}
        }

        # Test key patterns per ticker
        for pattern, col in [
            ("persistent", "persistent_signal"),
            ("flash", "flash_signal"),
            ("reignition", "re_ignition"),
            ("burst", "sudden_burst"),
        ]:
            subset = ticker_feat[ticker_feat[col]]
            if len(subset) >= 2:
                results[ticker]["patterns"][pattern] = compute_group_stats(
                    subset["fwd_ret"], f"{ticker}_{pattern}"
                )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("SIGNAL SEQUENCE PATTERN MINING")
    print("=" * 70)

    # Step 1: Extract signal events
    print("\n[1/6] Extracting signal events from paper engines...")
    events = extract_signal_events()

    if not events:
        print("ERROR: No signal events found!")
        return

    # Step 2: Build daily signal matrix
    print("\n[2/6] Building daily signal matrix...")
    daily_df = build_daily_signal_matrix(events)

    if daily_df.empty:
        print("ERROR: Empty daily matrix!")
        return

    # Determine date range
    min_date = daily_df["date"].min()
    max_date = daily_df["date"].max()
    # Extend for forward return calculation
    fetch_start = (min_date - timedelta(days=10)).strftime("%Y-%m-%d")
    fetch_end = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")

    # Step 3: Fetch prices
    print("\n[3/6] Fetching price data...")
    try:
        closes, fwd_1d, fwd_3d, fwd_5d, fwd_10d = fetch_prices(ALL_TICKERS, fetch_start, fetch_end)
    except Exception as e:
        print(f"ERROR fetching prices: {e}")
        traceback.print_exc()
        return

    if closes.empty:
        print("ERROR: No price data!")
        return

    # Trading dates
    all_trading_dates = closes.index

    # Step 4: Compute temporal features
    print("\n[4/6] Computing temporal features...")
    feat_df = compute_temporal_features(daily_df, all_trading_dates)

    if feat_df.empty:
        print("ERROR: No temporal features computed!")
        return

    # Summary of feature distributions
    print(f"\n  Feature distributions:")
    print(f"    Streak days: mean={feat_df['streak_days'].mean():.1f}, max={feat_df['streak_days'].max()}")
    print(f"    Building confluence: {feat_df['building_confluence'].sum()}/{len(feat_df)}")
    print(f"    Sudden burst: {feat_df['sudden_burst'].sum()}/{len(feat_df)}")
    print(f"    Re-ignition: {feat_df['re_ignition'].sum()}/{len(feat_df)}")
    print(f"    Persistent (3+ days): {feat_df['persistent_signal'].sum()}/{len(feat_df)}")
    print(f"    Flash (1 day): {feat_df['flash_signal'].sum()}/{len(feat_df)}")

    # Step 5: Test patterns
    print("\n[5/6] Testing patterns against forward returns...")
    output = {
        "metadata": {
            "generated": datetime.now().isoformat(),
            "date_range": f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}",
            "n_signal_events": len(events),
            "n_signal_days": len(feat_df),
            "n_tickers": int(feat_df["ticker"].nunique()),
            "tickers": sorted(feat_df["ticker"].unique().tolist()),
            "trading_days_covered": int((max_date - min_date).days),
            "note": "SMALL SAMPLE WARNING: Only ~20-30 trading days of signal data. "
                    "Results are directional hints, NOT statistically reliable. "
                    "Need 6+ months for reliable pattern validation.",
        },
        "patterns": {},
        "regime_analysis": {},
        "per_ticker": {},
    }

    # Test at multiple horizons
    for horizon_name, fwd_ret in [("fwd_1d", fwd_1d), ("fwd_3d", fwd_3d),
                                   ("fwd_5d", fwd_5d), ("fwd_10d", fwd_10d)]:
        print(f"\n  --- {horizon_name} ---")
        pattern_results = compute_pattern_stats(feat_df, fwd_ret, horizon_name)
        output["patterns"][horizon_name] = pattern_results

    # Step 6: Regime stratification (using 5d returns)
    print("\n[6/6] Regime stratification...")
    spy_fwd = fwd_5d["SPY"] if "SPY" in fwd_5d.columns else None
    if spy_fwd is not None:
        output["regime_analysis"] = regime_stratification(feat_df, fwd_5d, spy_fwd)

    # Per-ticker analysis (using 5d returns)
    output["per_ticker"] = per_ticker_analysis(feat_df, fwd_5d)

    # ─────────────────────────────────────────────────────────────────────
    # Key findings summary
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    fwd5 = output["patterns"].get("fwd_5d", {})
    baseline = fwd5.get("baseline", {})
    print(f"\nBaseline (all signals, 5d fwd): n={baseline.get('n', 0)}, "
          f"mean={baseline.get('mean_return_pct', 0):.2f}%, "
          f"WR={baseline.get('win_rate', 0):.1%}, "
          f"Sharpe={baseline.get('sharpe', 0):.2f}")

    # Compare patterns
    print("\nPattern comparison (5d forward, sorted by Sharpe):")
    pattern_summary = []
    for key, val in fwd5.items():
        if key == "baseline":
            continue
        if isinstance(val, dict):
            if "n" in val and val.get("n", 0) >= 3:
                pattern_summary.append((key, val))
            else:
                # Nested dict (yes/no or similar)
                for subkey, subval in val.items():
                    if isinstance(subval, dict) and subval.get("n", 0) >= 3:
                        pattern_summary.append((f"{key}.{subkey}", subval))

    pattern_summary.sort(key=lambda x: x[1].get("sharpe", 0), reverse=True)
    for name, stats in pattern_summary[:15]:
        print(f"  {name:40s} n={stats['n']:3d}  mean={stats['mean_return_pct']:+.2f}%  "
              f"WR={stats['win_rate']:.1%}  Sharpe={stats['sharpe']:+.2f}  "
              f"PF={stats['profit_factor']:.2f}")

    # Actionable findings
    print("\n" + "-" * 70)
    print("ACTIONABLE PATTERNS (if sample confirms):")
    print("-" * 70)

    top_patterns = [p for p in pattern_summary if p[1].get("sharpe", 0) > 0.5 and p[1].get("n", 0) >= 5]
    if top_patterns:
        for name, stats in top_patterns:
            print(f"  + {name}: Sharpe {stats['sharpe']:+.2f}, WR {stats['win_rate']:.1%}, "
                  f"PF {stats['profit_factor']:.2f} (n={stats['n']})")
    else:
        print("  No patterns with Sharpe > 0.5 and n >= 5 found.")
        print("  This may indicate sample is too small or patterns need more data.")

    anti_patterns = [p for p in pattern_summary if p[1].get("sharpe", 0) < -0.5 and p[1].get("n", 0) >= 5]
    if anti_patterns:
        print("\n  ANTI-PATTERNS (avoid these):")
        for name, stats in anti_patterns:
            print(f"  - {name}: Sharpe {stats['sharpe']:+.2f}, WR {stats['win_rate']:.1%} (n={stats['n']})")

    # Statistical power warning
    n_total = baseline.get("n", 0)
    print(f"\n  STATISTICAL POWER: {n_total} observations.")
    if n_total < 50:
        print("  WARNING: Very small sample. These are DIRECTIONAL HINTS only.")
        print("  Need 200+ observations for reliable pattern validation.")
        print("  Recommend: wire pattern tracking into daily scanner, revisit in 3 months.")
    elif n_total < 200:
        print("  CAUTION: Moderate sample. Patterns with n<20 are unreliable.")

    # Save output
    with open(OUTPUT, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT}")


if __name__ == "__main__":
    main()
