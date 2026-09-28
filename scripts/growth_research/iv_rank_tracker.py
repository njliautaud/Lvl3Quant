#!/usr/bin/env python3
"""
IV Rank & Percentile Tracker (Sector-Relative)
================================================
Computes IV Rank and IV Percentile for our trading universe using two sources:

  1. Dolt volatility_history table (pre-computed iv_current, iv_year_high, iv_year_low)
  2. Chain parquets with full IV per strike (ATM IV extraction for 252-day rolling)

SECTOR-RELATIVE ANALYSIS (2026-08-06):
  Each ticker is mapped to its sector. IV rank is assessed against sector-specific
  baselines from config/sector_iv_baselines.json. A flat 70% cutoff is WRONG because
  tech (XLK) naturally runs higher IV than utilities (XLU) or staples (XLP).

  Example: XLK IV rank 65% is NORMAL for tech, but XLU IV rank 65% is ELEVATED
  for utilities. The sector baselines define what "expensive" means per sector.

Outputs:
  - state/iv_rank_data.json  -- machine-readable for signal aggregator integration
  - stdout summary           -- human-readable with sector context

Integration:
  The agentic_signal_aggregator.py reads state/iv_rank_data.json and applies
  sector-relative confidence penalties (not a flat 15% penalty).

Data Sources:
  - Dolt volatility_history: direct iv_current / iv_year_high / iv_year_low
    (fast, but only gives rank vs annual extremes -- not rolling percentile)
  - Chain parquets: ATM IV per day -> full 252-day rolling rank + percentile
    (slower, but richer -- true statistical percentile)
  - config/sector_iv_baselines.json: sector-specific IV thresholds

Usage:
  python3 scripts/growth_research/iv_rank_tracker.py                  # full run
  python3 scripts/growth_research/iv_rank_tracker.py --tickers XLE,XLK  # specific
  python3 scripts/growth_research/iv_rank_tracker.py --fast             # Dolt only (quick)
  python3 scripts/growth_research/iv_rank_tracker.py --history          # include 30d history

Schedule (daily after market close):
  30 16 * * 1-5  cd /home/jupiter/Lvl3Quant && python3 scripts/growth_research/iv_rank_tracker.py >> logs/iv_rank_tracker.log 2>&1
"""

import argparse
import json
import shutil
import subprocess
import sys
import warnings
from datetime import datetime, date
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ──
ROOT = Path(__file__).resolve().parents[2]
CHAINS_DIR = ROOT / "wheel_strategy_v1" / "data" / "cache" / "options_real" / "chains"
DOLT_DIR = ROOT / "wheel_strategy_v1" / "data" / "cache" / "options_real" / "options"
STATE_DIR = ROOT / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = STATE_DIR / "iv_rank_data.json"
SECTOR_BASELINES_FILE = ROOT / "config" / "sector_iv_baselines.json"

# ── Universe ──
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLI", "XLP", "XLU", "XLRE", "XLB", "XLC"]
TOP_STOCKS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD",
    "JPM", "BAC", "GS", "WFC",
    "XOM", "CVX", "COP",
    "UNH", "JNJ", "PFE", "ABBV", "MRK",
    "NFLX", "DIS", "CRM",
    "HD", "MCD", "NKE",
    "CAT", "HON", "BA",
    "PG", "KO", "WMT", "COST",
    "SPY",
]
DEFAULT_UNIVERSE = SECTOR_ETFS + TOP_STOCKS

# ── IV thresholds for signal generation ──
IV_CHEAP_RANK = 30.0      # below this = cheap IV, good for buying options
IV_EXPENSIVE_RANK = 70.0  # above this = expensive IV, bad for buying options
IV_VERY_CHEAP = 15.0      # below this = historically cheap, strong buy signal
IV_VERY_EXPENSIVE = 85.0  # above this = historically rich, strong sell signal


def _find_dolt() -> Optional[str]:
    for p in (shutil.which("dolt"), "/home/jupiter/.local/bin/dolt", "/usr/local/bin/dolt"):
        if p and Path(p).exists():
            return p
    return None


def dolt_query(sql: str, dolt_bin: str, timeout: int = 60) -> pd.DataFrame:
    """Run a Dolt SQL query and return DataFrame."""
    proc = subprocess.run(
        [dolt_bin, "sql", "-q", sql, "-r", "csv"],
        cwd=str(DOLT_DIR),
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"dolt sql failed: {proc.stderr[:300]}")
    out = proc.stdout.strip()
    if not out:
        return pd.DataFrame()
    return pd.read_csv(StringIO(out))


def get_iv_from_dolt(tickers: list[str], dolt_bin: str) -> dict:
    """
    Get IV rank from Dolt volatility_history (fast path).

    The table has: iv_current, iv_year_high, iv_year_low per day per symbol.
    IV Rank = (iv_current - iv_year_low) / (iv_year_high - iv_year_low) * 100
    """
    results = {}

    # Build IN clause for all tickers
    # Dolt uses act_symbol which may differ from our tickers (e.g. BRK.B vs BRK-B)
    symbols_str = ", ".join(f"'{t.replace('-', '.')}'" for t in tickers)

    # Query each ticker individually to avoid slow subquery with IN clause
    # First get the max date from a fast query
    try:
        max_date_df = dolt_query(
            "SELECT MAX(date) as md FROM volatility_history WHERE act_symbol = 'SPY'",
            dolt_bin, timeout=15
        )
        if max_date_df.empty:
            return results
        max_date = max_date_df["md"].iloc[0]
    except Exception:
        # Fallback: just use today-ish
        max_date = str(date.today())

    sql = f"""
        SELECT act_symbol, date, iv_current, iv_week_ago, iv_month_ago,
               iv_year_high, iv_year_high_date, iv_year_low, iv_year_low_date,
               hv_current
        FROM volatility_history
        WHERE act_symbol IN ({symbols_str})
          AND date = '{max_date}'
        ORDER BY act_symbol
    """

    try:
        df = dolt_query(sql, dolt_bin, timeout=60)
    except Exception as e:
        print(f"  [WARN] Dolt query failed: {e}", file=sys.stderr)
        return results

    if df.empty:
        return results

    for _, row in df.iterrows():
        sym = str(row["act_symbol"]).replace(".", "-")  # back to our convention
        iv_cur = float(row["iv_current"]) if pd.notna(row["iv_current"]) else None
        iv_hi = float(row["iv_year_high"]) if pd.notna(row["iv_year_high"]) else None
        iv_lo = float(row["iv_year_low"]) if pd.notna(row["iv_year_low"]) else None
        hv_cur = float(row["hv_current"]) if pd.notna(row["hv_current"]) else None

        if iv_cur is None or iv_hi is None or iv_lo is None:
            continue

        iv_range = iv_hi - iv_lo
        iv_rank = ((iv_cur - iv_lo) / iv_range * 100) if iv_range > 0.001 else 50.0

        # IV vs HV ratio (IV premium)
        iv_hv_ratio = (iv_cur / hv_cur) if hv_cur and hv_cur > 0.001 else None

        results[sym] = {
            "source": "dolt_volatility_history",
            "date": str(row["date"]),
            "iv_current": round(iv_cur * 100, 2),  # as percentage
            "iv_year_high": round(iv_hi * 100, 2),
            "iv_year_low": round(iv_lo * 100, 2),
            "iv_rank": round(iv_rank, 1),
            "hv_current": round(hv_cur * 100, 2) if hv_cur else None,
            "iv_hv_ratio": round(iv_hv_ratio, 3) if iv_hv_ratio else None,
            "iv_week_ago": round(float(row["iv_week_ago"]) * 100, 2) if pd.notna(row.get("iv_week_ago")) else None,
            "iv_month_ago": round(float(row["iv_month_ago"]) * 100, 2) if pd.notna(row.get("iv_month_ago")) else None,
        }

    return results


def get_iv_from_chains(ticker: str, lookback_days: int = 252) -> Optional[dict]:
    """
    Compute IV rank and percentile from chain parquets (rich path).

    Extracts daily ATM IV from 25-45 DTE calls, then computes:
      - IV rank: (current - 252d low) / (252d high - 252d low) * 100
      - IV percentile: % of past 252 days where IV was below current
      - IV trend: 5d vs 21d average (rising/falling)
    """
    chain_path = CHAINS_DIR / f"{ticker}.parquet"
    if not chain_path.exists():
        return None

    try:
        df = pd.read_parquet(chain_path)
    except Exception:
        return None

    df["date"] = pd.to_datetime(df["date"])

    # Filter: calls, 25-45 DTE, with valid IV and delta
    mask = (
        (df["type"] == "c") &
        (df["dte"].between(25, 45)) &
        (df["vol"] > 0.01) &
        (df["delta"].between(0.30, 0.70))  # near ATM
    )
    filtered = df[mask].copy()

    if len(filtered) < 60:
        return None

    # Extract daily ATM IV: for each day, pick the call closest to delta=0.50
    daily_records = []
    for dt, grp in filtered.groupby("date"):
        closest_atm = grp.iloc[(grp["delta"] - 0.50).abs().argsort()[:1]]
        if len(closest_atm) > 0:
            daily_records.append({
                "date": dt,
                "atm_iv": float(closest_atm["vol"].iloc[0]),
                "strike": float(closest_atm["strike"].iloc[0]),
                "dte": int(closest_atm["dte"].iloc[0]),
            })

    if len(daily_records) < 60:
        return None

    iv_df = pd.DataFrame(daily_records).set_index("date").sort_index()

    # Rolling 252-day IV rank and percentile
    window = min(lookback_days, len(iv_df) - 1)
    if window < 60:
        return None

    iv_series = iv_df["atm_iv"]

    # Current values
    current_iv = float(iv_series.iloc[-1])

    # 252-day high/low
    recent = iv_series.iloc[-window:]
    iv_high = float(recent.max())
    iv_low = float(recent.min())
    iv_range = iv_high - iv_low

    iv_rank = ((current_iv - iv_low) / iv_range * 100) if iv_range > 0.001 else 50.0

    # Percentile: what % of recent days had IV below current
    iv_percentile = float((recent < current_iv).sum() / len(recent) * 100)

    # Trend: 5d avg vs 21d avg
    iv_5d = float(iv_series.iloc[-5:].mean()) if len(iv_series) >= 5 else current_iv
    iv_21d = float(iv_series.iloc[-21:].mean()) if len(iv_series) >= 21 else current_iv
    iv_trend = "rising" if iv_5d > iv_21d * 1.02 else ("falling" if iv_5d < iv_21d * 0.98 else "stable")

    # 5d change
    if len(iv_series) >= 6:
        iv_5d_ago = float(iv_series.iloc[-6])
        iv_change_5d = ((current_iv / iv_5d_ago) - 1) * 100
    else:
        iv_change_5d = 0.0

    return {
        "source": "chain_parquet_atm",
        "date": str(iv_df.index[-1].date()),
        "iv_current": round(current_iv * 100, 2),
        "iv_year_high": round(iv_high * 100, 2),
        "iv_year_low": round(iv_low * 100, 2),
        "iv_rank": round(iv_rank, 1),
        "iv_percentile": round(iv_percentile, 1),
        "iv_5d_avg": round(iv_5d * 100, 2),
        "iv_21d_avg": round(iv_21d * 100, 2),
        "iv_trend": iv_trend,
        "iv_change_5d_pct": round(iv_change_5d, 2),
        "lookback_days": window,
        "data_points": len(iv_df),
    }


def load_sector_baselines() -> dict:
    """Load sector IV baselines from config file."""
    if SECTOR_BASELINES_FILE.exists():
        try:
            with open(SECTOR_BASELINES_FILE) as f:
                return json.load(f)
        except Exception as e:
            print(f"  [WARN] Could not load sector baselines: {e}", file=sys.stderr)
    return {}


def get_ticker_sector(ticker: str, baselines: dict) -> Optional[str]:
    """Map a ticker to its sector ETF using the baselines config."""
    mapping = baselines.get("ticker_to_sector", {})
    if ticker in mapping:
        return mapping[ticker]
    # ETFs map to themselves
    if ticker in baselines.get("sectors", {}):
        return ticker
    return None


def classify_iv_environment_sector_relative(
    iv_rank: float,
    ticker: str,
    baselines: dict,
) -> dict:
    """
    Classify IV environment relative to the ticker's SECTOR norms.

    Instead of flat thresholds (cheap < 30%, expensive > 70%), uses sector-specific
    cutoffs from config/sector_iv_baselines.json. Tech stocks tolerate higher IV rank
    than utilities/staples because their baseline vol is naturally higher.
    """
    sector = get_ticker_sector(ticker, baselines)
    sector_config = baselines.get("sectors", {}).get(sector, {}) if sector else {}

    # Use sector-specific thresholds if available, else fall back to global defaults
    cheap_rank = sector_config.get("cheap_rank", IV_CHEAP_RANK)
    expensive_rank = sector_config.get("expensive_rank", IV_EXPENSIVE_RANK)
    very_cheap = cheap_rank * 0.6   # scale very_cheap relative to sector cheap threshold
    very_expensive = expensive_rank + (100 - expensive_rank) * 0.5  # midpoint to 100

    sector_name = sector_config.get("name", sector or "Unknown")

    if iv_rank <= very_cheap:
        return {
            "classification": "VERY_CHEAP",
            "action": "STRONG_BUY_OPTIONS",
            "description": f"IV historically very low for {sector_name}. Premium cheap.",
            "sector": sector,
            "sector_name": sector_name,
            "sector_expensive_rank": expensive_rank,
            "sector_cheap_rank": cheap_rank,
            "sector_relative_assessment": "well_below_normal",
        }
    elif iv_rank <= cheap_rank:
        return {
            "classification": "CHEAP",
            "action": "BUY_OPTIONS",
            "description": f"IV below {sector_name} historical average. Good time to buy options.",
            "sector": sector,
            "sector_name": sector_name,
            "sector_expensive_rank": expensive_rank,
            "sector_cheap_rank": cheap_rank,
            "sector_relative_assessment": "below_normal",
        }
    elif iv_rank >= very_expensive:
        return {
            "classification": "VERY_EXPENSIVE",
            "action": "SELL_PREMIUM",
            "description": f"IV historically very high for {sector_name}. Avoid buying, consider selling.",
            "sector": sector,
            "sector_name": sector_name,
            "sector_expensive_rank": expensive_rank,
            "sector_cheap_rank": cheap_rank,
            "sector_relative_assessment": "well_above_normal",
        }
    elif iv_rank >= expensive_rank:
        return {
            "classification": "EXPENSIVE",
            "action": "AVOID_BUYING",
            "description": f"IV elevated for {sector_name} (rank {iv_rank:.0f}% vs sector threshold {expensive_rank}%). Buying has headwind.",
            "sector": sector,
            "sector_name": sector_name,
            "sector_expensive_rank": expensive_rank,
            "sector_cheap_rank": cheap_rank,
            "sector_relative_assessment": "above_normal",
        }
    else:
        return {
            "classification": "NORMAL",
            "action": "NEUTRAL",
            "description": f"IV in normal range for {sector_name}.",
            "sector": sector,
            "sector_name": sector_name,
            "sector_expensive_rank": expensive_rank,
            "sector_cheap_rank": cheap_rank,
            "sector_relative_assessment": "normal",
        }


def classify_iv_environment(iv_rank: float) -> dict:
    """Classify IV environment for options timing (legacy flat thresholds)."""
    if iv_rank <= IV_VERY_CHEAP:
        return {
            "classification": "VERY_CHEAP",
            "action": "STRONG_BUY_OPTIONS",
            "description": "IV historically very low. Premium cheap. Favor buying calls/puts.",
        }
    elif iv_rank <= IV_CHEAP_RANK:
        return {
            "classification": "CHEAP",
            "action": "BUY_OPTIONS",
            "description": "IV below historical average. Good time to buy options.",
        }
    elif iv_rank >= IV_VERY_EXPENSIVE:
        return {
            "classification": "VERY_EXPENSIVE",
            "action": "SELL_PREMIUM",
            "description": "IV historically very high. Premium rich. Avoid buying, consider selling.",
        }
    elif iv_rank >= IV_EXPENSIVE_RANK:
        return {
            "classification": "EXPENSIVE",
            "action": "AVOID_BUYING",
            "description": "IV elevated. Buying options has theta/vega headwind.",
        }
    else:
        return {
            "classification": "NORMAL",
            "action": "NEUTRAL",
            "description": "IV in normal range. No strong timing signal.",
        }


def generate_timing_signals(iv_data: dict) -> list[dict]:
    """
    Generate actionable timing signals from IV data.
    Returns list of signals sorted by strength.
    """
    signals = []

    for ticker, data in iv_data.items():
        iv_rank = data.get("iv_rank")
        if iv_rank is None:
            continue

        env = classify_iv_environment(iv_rank)

        # Only generate signals for actionable levels
        if env["classification"] in ("VERY_CHEAP", "CHEAP", "VERY_EXPENSIVE", "EXPENSIVE"):
            signal = {
                "ticker": ticker,
                "iv_rank": iv_rank,
                "iv_current": data.get("iv_current"),
                "iv_trend": data.get("iv_trend", "unknown"),
                "iv_hv_ratio": data.get("iv_hv_ratio"),
                "classification": env["classification"],
                "action": env["action"],
                "description": env["description"],
            }

            # Strength score: how far from neutral (50) the IV rank is
            signal["strength"] = abs(iv_rank - 50.0) / 50.0

            # Bonus for IV/HV divergence (IV much higher than HV = expensive, vice versa)
            if data.get("iv_hv_ratio"):
                ratio = data["iv_hv_ratio"]
                if ratio > 1.3 and iv_rank > 60:
                    signal["iv_hv_note"] = f"IV is {ratio:.1f}x HV - premium significantly elevated"
                elif ratio < 0.8 and iv_rank < 40:
                    signal["iv_hv_note"] = f"IV is only {ratio:.1f}x HV - premium cheap relative to realized vol"

            signals.append(signal)

    # Sort by strength (strongest first)
    signals.sort(key=lambda x: x["strength"], reverse=True)
    return signals


def get_iv_history(ticker: str, days: int = 30) -> Optional[list[dict]]:
    """Get recent IV history from chain parquets for trend analysis."""
    chain_path = CHAINS_DIR / f"{ticker}.parquet"
    if not chain_path.exists():
        return None

    try:
        df = pd.read_parquet(chain_path)
    except Exception:
        return None

    df["date"] = pd.to_datetime(df["date"])

    mask = (
        (df["type"] == "c") &
        (df["dte"].between(25, 45)) &
        (df["vol"] > 0.01) &
        (df["delta"].between(0.30, 0.70))
    )
    filtered = df[mask].copy()

    history = []
    for dt, grp in filtered.groupby("date"):
        closest_atm = grp.iloc[(grp["delta"] - 0.50).abs().argsort()[:1]]
        if len(closest_atm) > 0:
            history.append({
                "date": str(dt.date()),
                "atm_iv": round(float(closest_atm["vol"].iloc[0]) * 100, 2),
            })

    return history[-days:] if history else None


def run(tickers: list[str], fast: bool = False, include_history: bool = False) -> dict:
    """Main runner. Returns output dict."""
    print("=" * 60)
    print("  IV RANK & PERCENTILE TRACKER (SECTOR-RELATIVE)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print(f"\n  Universe: {len(tickers)} tickers")
    print(f"  Mode: {'FAST (Dolt only)' if fast else 'FULL (Dolt + chains)'}")

    # Load sector baselines
    baselines = load_sector_baselines()
    if baselines:
        print(f"  Sector baselines loaded: {len(baselines.get('sectors', {}))} sectors")
    else:
        print("  [WARN] No sector baselines found, using flat thresholds")

    iv_data = {}

    # ── Step 1: Dolt volatility_history (always try first -- it's fast) ──
    dolt_bin = _find_dolt()
    if dolt_bin and DOLT_DIR.exists():
        print("\n[1] Querying Dolt volatility_history...")
        dolt_results = get_iv_from_dolt(tickers, dolt_bin)
        iv_data.update(dolt_results)
        print(f"  Got {len(dolt_results)} tickers from Dolt")
    else:
        print("\n[1] Dolt not available, skipping...")

    # ── Step 2: Chain parquets (richer data -- IV percentile, trend) ──
    if not fast:
        print("\n[2] Computing ATM IV from chain parquets...")
        chain_count = 0
        for ticker in tickers:
            chain_data = get_iv_from_chains(ticker)
            if chain_data:
                # Merge with Dolt data (chain data is richer, so it adds fields)
                if ticker in iv_data:
                    # Keep Dolt IV/HV ratio but add chain-specific fields
                    dolt_entry = iv_data[ticker]
                    chain_data["hv_current"] = dolt_entry.get("hv_current")
                    chain_data["iv_hv_ratio"] = dolt_entry.get("iv_hv_ratio")
                    chain_data["iv_week_ago"] = dolt_entry.get("iv_week_ago")
                    chain_data["iv_month_ago"] = dolt_entry.get("iv_month_ago")
                    chain_data["source"] = "dolt+chain_merged"
                iv_data[ticker] = chain_data
                chain_count += 1
        print(f"  Computed ATM IV for {chain_count} tickers from chains")

    # ── Step 3: Classify with sector-relative thresholds ──
    print("\n[3] Classifying IV environments (sector-relative)...")
    for ticker, data in iv_data.items():
        if baselines:
            env = classify_iv_environment_sector_relative(data["iv_rank"], ticker, baselines)
        else:
            env = classify_iv_environment(data["iv_rank"])
        data["classification"] = env["classification"]
        data["action"] = env["action"]
        data["sector"] = env.get("sector")
        data["sector_name"] = env.get("sector_name")
        data["sector_expensive_rank"] = env.get("sector_expensive_rank")
        data["sector_cheap_rank"] = env.get("sector_cheap_rank")
        data["sector_relative_assessment"] = env.get("sector_relative_assessment", "unknown")

    timing_signals = generate_timing_signals(iv_data)

    # ── Step 4: Optional history ──
    if include_history:
        print("\n[4] Fetching 30d IV history...")
        for ticker in tickers:
            if ticker in iv_data:
                hist = get_iv_history(ticker, 30)
                if hist:
                    iv_data[ticker]["history_30d"] = hist

    # ── Build output ──
    # Compute sector-relative expensive/cheap lists using sector thresholds
    cheap_tickers = []
    expensive_tickers = []
    for t, d in iv_data.items():
        rank = d.get("iv_rank", 50)
        sector_cheap = d.get("sector_cheap_rank", IV_CHEAP_RANK)
        sector_expensive = d.get("sector_expensive_rank", IV_EXPENSIVE_RANK)
        if rank < sector_cheap:
            cheap_tickers.append(t)
        elif rank > sector_expensive:
            expensive_tickers.append(t)

    output = {
        "timestamp": datetime.now().isoformat(),
        "tickers_tracked": len(iv_data),
        "sector_relative": True,
        "iv_data": iv_data,
        "timing_signals": timing_signals,
        "cheap_iv_tickers": cheap_tickers,
        "expensive_iv_tickers": expensive_tickers,
        "thresholds": {
            "note": "Sector-relative thresholds replace flat cutoffs. See per-ticker sector_expensive_rank and sector_cheap_rank.",
            "fallback_cheap": IV_CHEAP_RANK,
            "fallback_expensive": IV_EXPENSIVE_RANK,
        },
    }

    # ── Print summary ──
    print(f"\n{'='*60}")
    print("  IV RANK SUMMARY (SECTOR-RELATIVE)")
    print(f"{'='*60}")

    # Group by sector for display
    from collections import defaultdict
    sector_groups = defaultdict(list)
    for ticker, data in iv_data.items():
        sector = data.get("sector", "OTHER")
        sector_groups[sector].append((ticker, data))

    for sector in sorted(sector_groups.keys()):
        items = sorted(sector_groups[sector], key=lambda x: x[1].get("iv_rank", 50))
        sector_config = baselines.get("sectors", {}).get(sector, {})
        sector_name = sector_config.get("name", sector)
        exp_rank = sector_config.get("expensive_rank", IV_EXPENSIVE_RANK)
        cheap_rank = sector_config.get("cheap_rank", IV_CHEAP_RANK)
        print(f"\n  -- {sector_name} ({sector}) | cheap < {cheap_rank}% | expensive > {exp_rank}% --")
        print(f"  {'Ticker':<8} {'IV Rank':>8} {'IV%ile':>7} {'IV Curr':>8} {'Trend':>9} {'Sector Assessment':>20}")
        print("  " + "-" * 65)

        for ticker, data in items:
            rank = data.get("iv_rank", 0)
            pctile = data.get("iv_percentile", "N/A")
            iv_cur = data.get("iv_current", 0)
            trend = data.get("iv_trend", "N/A")
            assessment = data.get("sector_relative_assessment", "unknown")
            classification = data.get("classification", "NORMAL")

            if classification in ("VERY_CHEAP", "CHEAP"):
                marker = "<- CHEAP"
            elif classification in ("VERY_EXPENSIVE", "EXPENSIVE"):
                marker = "-> EXPENSIVE"
            else:
                marker = "   normal"

            pctile_str = f"{pctile:.0f}%" if isinstance(pctile, (int, float)) else pctile
            print(f"  {ticker:<8} {rank:>7.1f}% {pctile_str:>7} {iv_cur:>7.1f}% {trend:>9} {marker:>20}")

    # Timing signals
    if timing_signals:
        print(f"\n  TIMING SIGNALS ({len(timing_signals)}):")
        print("  " + "-" * 50)
        for sig in timing_signals[:10]:
            print(f"  {sig['ticker']:<8} IV Rank {sig['iv_rank']:>5.1f}% -> {sig['action']}")
            if sig.get("iv_hv_note"):
                print(f"           {sig['iv_hv_note']}")

    # Sector-relative reclassification summary
    if baselines:
        print(f"\n  SECTOR-RELATIVE RECLASSIFICATIONS:")
        print("  " + "-" * 50)
        reclass_count = 0
        for ticker, data in iv_data.items():
            rank = data.get("iv_rank", 50)
            sector_class = data.get("classification", "NORMAL")
            flat_class = classify_iv_environment(rank)["classification"]
            if sector_class != flat_class:
                reclass_count += 1
                sector_name = data.get("sector_name", "?")
                print(f"  {ticker:<8} IV rank {rank:.0f}%: flat={flat_class}, sector-relative={sector_class} ({sector_name})")
        if reclass_count == 0:
            print("  (none -- flat and sector-relative agree for all tickers)")
        else:
            print(f"  {reclass_count} ticker(s) reclassified using sector-relative thresholds")

    # Save output
    try:
        with open(OUTPUT_FILE, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Output saved to {OUTPUT_FILE.name}")
    except Exception as e:
        print(f"\n  [ERROR] Could not save output: {e}", file=sys.stderr)

    return output


def main():
    ap = argparse.ArgumentParser(description="IV Rank & Percentile Tracker")
    ap.add_argument("--tickers", default=None,
                    help="Comma-separated ticker list (default: sector ETFs + top stocks)")
    ap.add_argument("--fast", action="store_true",
                    help="Fast mode: Dolt only, skip chain parquet computation")
    ap.add_argument("--history", action="store_true",
                    help="Include 30-day IV history per ticker")
    args = ap.parse_args()

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        tickers = DEFAULT_UNIVERSE

    run(tickers, fast=args.fast, include_history=args.history)


if __name__ == "__main__":
    main()
