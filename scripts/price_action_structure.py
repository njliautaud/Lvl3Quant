#!/usr/bin/env python3
"""
Price Action Structure Module
==============================
Calculates where current price sits relative to structural levels:
  - Multi-timeframe highs/lows (5d, 10d, 20d, 52w)
  - Fractal swing highs/lows (support/resistance)
  - Pivot points (classic floor pivots)

Returns a 0-100 structure score:
  - Low score (0-35)  = price near resistance/highs → bad for calls, good for puts
  - Mid score (35-65) = no strong structural signal
  - High score (65-100) = price near support/lows → good for calls, bad for puts

The raw score is DIRECTION-NEUTRAL (measures position in range).
Use get_directional_score() to get a direction-adjusted score for timing.

Usage:
  python3 price_action_structure.py --ticker XLF
  python3 price_action_structure.py --ticker XLF --dir call
  python3 price_action_structure.py --backtest --tickers XLF,XLE,XLU,XLC,XLP,XLK

Author: Claude (HC requested Aug 7 2026)
"""

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

BASE = Path("/home/jupiter/Lvl3Quant")
STATE = BASE / "state"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PriceStructure] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("PriceStructure")


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_daily_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetch daily OHLCV from yfinance."""
    if not YF_AVAILABLE:
        log.warning("yfinance not available")
        return pd.DataFrame()
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, auto_adjust=True)
        if df.empty:
            log.warning(f"No data for {ticker}")
        return df
    except Exception as e:
        log.error(f"Failed to fetch {ticker}: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Fractal swing detection
# ---------------------------------------------------------------------------

def detect_fractals(highs: pd.Series, lows: pd.Series, order: int = 2) -> dict:
    """
    Detect fractal swing highs and lows using Williams fractals.
    order=2 means we need 2 bars on each side (5-bar pattern).
    Returns dict with lists of (index, price) for swing highs and swing lows.
    """
    swing_highs = []
    swing_lows = []

    for i in range(order, len(highs) - order):
        # Swing high: center bar high is higher than all surrounding bars
        is_high = True
        for j in range(1, order + 1):
            if highs.iloc[i] <= highs.iloc[i - j] or highs.iloc[i] <= highs.iloc[i + j]:
                is_high = False
                break
        if is_high:
            swing_highs.append((highs.index[i], float(highs.iloc[i])))

        # Swing low: center bar low is lower than all surrounding bars
        is_low = True
        for j in range(1, order + 1):
            if lows.iloc[i] >= lows.iloc[i - j] or lows.iloc[i] >= lows.iloc[i + j]:
                is_low = False
                break
        if is_low:
            swing_lows.append((lows.index[i], float(lows.iloc[i])))

    return {"swing_highs": swing_highs, "swing_lows": swing_lows}


def find_support_resistance(
    df: pd.DataFrame, n_levels: int = 3, lookback: int = 60
) -> dict:
    """
    Find key support and resistance levels from fractal swings.
    Uses clustering to merge nearby levels (within 0.5% of each other).
    Returns the N most recent/relevant levels above and below current price.
    """
    if df.empty or len(df) < 10:
        return {"support": [], "resistance": []}

    subset = df.tail(lookback)
    current_price = float(subset["Close"].iloc[-1])

    # Detect fractals at multiple orders for robustness
    all_highs = []
    all_lows = []
    for order in [2, 3, 5]:
        fractals = detect_fractals(subset["High"], subset["Low"], order=order)
        all_highs.extend([p for _, p in fractals["swing_highs"]])
        all_lows.extend([p for _, p in fractals["swing_lows"]])

    # Cluster nearby levels (within 0.5%)
    def cluster_levels(levels, threshold_pct=0.005):
        if not levels:
            return []
        levels = sorted(levels)
        clusters = [[levels[0]]]
        for lvl in levels[1:]:
            if abs(lvl - clusters[-1][-1]) / clusters[-1][-1] < threshold_pct:
                clusters[-1].append(lvl)
            else:
                clusters.append([lvl])
        # Return mean of each cluster, weighted by count (more touches = stronger)
        return [(np.mean(c), len(c)) for c in clusters]

    high_clusters = cluster_levels(all_highs)
    low_clusters = cluster_levels(all_lows)

    # Separate into support (below price) and resistance (above price)
    all_levels = high_clusters + low_clusters
    support = sorted(
        [(lvl, count) for lvl, count in all_levels if lvl < current_price],
        key=lambda x: x[0],
        reverse=True,
    )[:n_levels]
    resistance = sorted(
        [(lvl, count) for lvl, count in all_levels if lvl >= current_price],
        key=lambda x: x[0],
    )[:n_levels]

    return {
        "support": support,  # List of (price, touch_count), nearest first
        "resistance": resistance,  # List of (price, touch_count), nearest first
        "current_price": current_price,
    }


# ---------------------------------------------------------------------------
# Range position calculations
# ---------------------------------------------------------------------------

def compute_range_position(price: float, low: float, high: float) -> float:
    """Where price is in the range [low, high]. Returns 0.0 (at low) to 1.0 (at high)."""
    if high <= low:
        return 0.5
    return max(0.0, min(1.0, (price - low) / (high - low)))


def compute_multi_tf_positions(df: pd.DataFrame) -> dict:
    """
    Compute where current price sits relative to highs/lows across
    multiple lookback windows. Uses SLIDING windows only.
    """
    if df.empty or len(df) < 5:
        return {}

    current_price = float(df["Close"].iloc[-1])
    positions = {}

    windows = {
        "5d": 5,
        "10d": 10,
        "20d": 20,
        "60d": 60,   # ~3 months
        "120d": 120,  # ~6 months
        "252d": 252,  # ~52 weeks
    }

    for label, n in windows.items():
        if len(df) < n:
            continue
        window = df.tail(n)
        hi = float(window["High"].max())
        lo = float(window["Low"].min())
        pos = compute_range_position(current_price, lo, hi)
        positions[label] = {
            "high": hi,
            "low": lo,
            "position": pos,  # 0=at low, 1=at high
            "pct_from_high": (hi - current_price) / hi * 100 if hi > 0 else 0,
            "pct_from_low": (current_price - lo) / lo * 100 if lo > 0 else 0,
        }

    return positions


def compute_pivot_points(df: pd.DataFrame) -> dict:
    """
    Classic floor pivot points from prior day's OHLC.
    PP = (H + L + C) / 3
    R1 = 2*PP - L,  S1 = 2*PP - H
    R2 = PP + (H-L), S2 = PP - (H-L)
    """
    if len(df) < 2:
        return {}

    prev = df.iloc[-2]
    h, l, c = float(prev["High"]), float(prev["Low"]), float(prev["Close"])
    pp = (h + l + c) / 3.0
    r1 = 2 * pp - l
    s1 = 2 * pp - h
    r2 = pp + (h - l)
    s2 = pp - (h - l)
    r3 = h + 2 * (pp - l)
    s3 = l - 2 * (h - pp)

    current = float(df["Close"].iloc[-1])

    return {
        "pp": pp,
        "r1": r1, "r2": r2, "r3": r3,
        "s1": s1, "s2": s2, "s3": s3,
        "current": current,
        "above_pp": current > pp,
    }


# ---------------------------------------------------------------------------
# Main scoring
# ---------------------------------------------------------------------------

def compute_price_structure_score(df: pd.DataFrame) -> dict:
    """
    Compute raw (direction-neutral) price structure score.

    Score meaning:
      0-20:  Price at/near multi-TF highs, against strong resistance
      20-40: Price in upper range, some resistance above
      40-60: Price mid-range, no strong structural bias
      60-80: Price in lower range, some support below
      80-100: Price at/near multi-TF lows, strong support nearby

    Higher score = closer to support = better for calls, worse for puts.
    """
    if df.empty or len(df) < 20:
        return {
            "score": 50.0,
            "components": {},
            "details": {"error": "Insufficient data"},
        }

    current_price = float(df["Close"].iloc[-1])
    components = {}
    details = {}

    # --- Component 1: Multi-timeframe range position (40% weight) ---
    tf_positions = compute_multi_tf_positions(df)
    details["tf_positions"] = {}

    if tf_positions:
        # Weight shorter timeframes more for entry timing
        tf_weights = {"5d": 0.10, "10d": 0.15, "20d": 0.25, "60d": 0.25, "120d": 0.15, "252d": 0.10}
        weighted_pos = 0.0
        total_weight = 0.0

        for tf, data in tf_positions.items():
            w = tf_weights.get(tf, 0.1)
            weighted_pos += data["position"] * w
            total_weight += w
            details["tf_positions"][tf] = round(data["position"], 3)

        if total_weight > 0:
            avg_position = weighted_pos / total_weight
        else:
            avg_position = 0.5

        # Invert: position=0 (at low) → score=100, position=1 (at high) → score=0
        components["tf_range"] = round((1.0 - avg_position) * 100, 1)
        details["avg_range_position"] = round(avg_position, 3)
    else:
        components["tf_range"] = 50.0

    # --- Component 2: Distance to nearest support/resistance (30% weight) ---
    sr = find_support_resistance(df, n_levels=3, lookback=min(120, len(df)))
    details["support_levels"] = [(round(p, 2), c) for p, c in sr.get("support", [])]
    details["resistance_levels"] = [(round(p, 2), c) for p, c in sr.get("resistance", [])]

    if sr.get("support") and sr.get("resistance"):
        nearest_support = sr["support"][0][0]
        nearest_resistance = sr["resistance"][0][0]
        support_touches = sr["support"][0][1]
        resistance_touches = sr["resistance"][0][1]

        dist_to_support_pct = (current_price - nearest_support) / current_price * 100
        dist_to_resistance_pct = (nearest_resistance - current_price) / current_price * 100

        details["dist_to_support_pct"] = round(dist_to_support_pct, 2)
        details["dist_to_resistance_pct"] = round(dist_to_resistance_pct, 2)

        # Score: closer to support = higher score
        total_dist = dist_to_support_pct + dist_to_resistance_pct
        if total_dist > 0:
            support_proximity = 1.0 - (dist_to_support_pct / total_dist)
        else:
            support_proximity = 0.5

        # Bonus for strong levels (more touches)
        support_strength = min(support_touches / 4.0, 1.0)  # cap at 4 touches
        resistance_strength = min(resistance_touches / 4.0, 1.0)

        # If near strong support → boost score; near strong resistance → lower score
        strength_adj = (support_strength - resistance_strength) * 10

        raw_sr_score = support_proximity * 100 + strength_adj
        components["sr_proximity"] = round(max(0, min(100, raw_sr_score)), 1)
    elif sr.get("support"):
        # Only support found, no resistance → price likely near highs
        components["sr_proximity"] = 30.0
    elif sr.get("resistance"):
        # Only resistance found, no support → price likely near lows
        components["sr_proximity"] = 70.0
    else:
        components["sr_proximity"] = 50.0

    # --- Component 3: Pivot point position (15% weight) ---
    pivots = compute_pivot_points(df)
    if pivots:
        pp = pivots["pp"]
        current = pivots["current"]
        details["pivot_point"] = round(pp, 2)
        details["above_pivot"] = pivots["above_pp"]

        # Score based on position relative to pivot levels
        if current <= pivots["s2"]:
            components["pivot"] = 90  # At S2 or below — deep support
        elif current <= pivots["s1"]:
            components["pivot"] = 75  # Between S1 and S2
        elif current <= pp:
            components["pivot"] = 60  # Between PP and S1
        elif current <= pivots["r1"]:
            components["pivot"] = 40  # Between PP and R1
        elif current <= pivots["r2"]:
            components["pivot"] = 25  # Between R1 and R2
        else:
            components["pivot"] = 10  # At R2 or above — deep resistance
    else:
        components["pivot"] = 50

    # --- Component 4: 52-week position context (15% weight) ---
    if "252d" in tf_positions:
        pos_52w = tf_positions["252d"]["position"]
        pct_from_high = tf_positions["252d"]["pct_from_high"]
        pct_from_low = tf_positions["252d"]["pct_from_low"]

        details["pct_from_52w_high"] = round(pct_from_high, 2)
        details["pct_from_52w_low"] = round(pct_from_low, 2)

        # Near 52w high = low score (resistance), near 52w low = high score (support)
        components["yearly_context"] = round((1.0 - pos_52w) * 100, 1)
    else:
        components["yearly_context"] = 50.0

    # --- Weighted total ---
    total = (
        components.get("tf_range", 50) * 0.40
        + components.get("sr_proximity", 50) * 0.30
        + components.get("pivot", 50) * 0.15
        + components.get("yearly_context", 50) * 0.15
    )

    return {
        "score": round(total, 1),
        "components": components,
        "details": details,
        "current_price": current_price,
    }


def get_directional_score(raw_score: float, direction: str) -> float:
    """
    Convert raw structure score to direction-adjusted score.

    Raw score: high = near support, low = near resistance.

    For CALLS:  near support (high raw) = GOOD timing → keep high score
    For PUTS:   near resistance (low raw) = GOOD timing → invert score

    Returns 0-100 where higher = better timing for the given direction.
    """
    if direction == "call":
        return raw_score  # High raw = near support = good for calls
    elif direction == "put":
        return 100.0 - raw_score  # Low raw = near resistance = good for puts
    else:
        return 50.0


def score_price_structure(
    ticker: str,
    direction: str = "call",
    df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Public API: compute price action structure score for a ticker+direction.
    This is meant to be called from timing_score.py.

    Returns dict compatible with timing_score.py's layer format:
      {"score": float, "components": dict, "details": dict}
    """
    if df is None:
        df = fetch_daily_data(ticker, period="1y")

    raw = compute_price_structure_score(df)
    directional = get_directional_score(raw["score"], direction)

    return {
        "score": round(directional, 1),
        "raw_score": raw["score"],
        "direction": direction,
        "components": raw["components"],
        "details": raw["details"],
        "current_price": raw.get("current_price"),
    }


# ---------------------------------------------------------------------------
# Backtest: does this score improve entry timing?
# ---------------------------------------------------------------------------

def backtest_structure_score(
    tickers: list,
    lookback_years: float = 1.0,
    hold_days: int = 10,
    n_permutations: int = 1000,
) -> dict:
    """
    Backtest the structure score on historical options-like entries.

    Method:
    - Walk through history with a SLIDING window.
    - At each day, compute structure score using ONLY prior data (no lookahead).
    - Measure forward N-day return.
    - Compare returns when score is favorable (>65) vs unfavorable (<35).
    - Permutation test: shuffle score-return pairs to get null distribution.

    For calls: favorable = high score (near support), forward return = positive
    For puts:  favorable = low score (near resistance), forward return = negative
    """
    log.info(f"Backtesting structure score on {tickers}, hold={hold_days}d")

    all_results = {}
    combined_favorable = []
    combined_unfavorable = []
    combined_neutral = []

    for ticker in tickers:
        log.info(f"  Processing {ticker}...")
        df = fetch_daily_data(ticker, period="2y")
        if df.empty or len(df) < 260 + hold_days:
            log.warning(f"  Skipping {ticker}: insufficient data ({len(df)} bars)")
            continue

        scores_and_returns = []

        # Walk through with sliding window, min 120 days of history for structure calc
        min_history = 120
        for i in range(min_history, len(df) - hold_days):
            # Use only data up to day i (no lookahead)
            hist = df.iloc[:i + 1].copy()
            future_close = float(df["Close"].iloc[i + hold_days])
            current_close = float(df["Close"].iloc[i])
            fwd_return = (future_close / current_close - 1) * 100

            # Compute structure score on historical data only
            raw = compute_price_structure_score(hist)
            raw_score = raw["score"]

            scores_and_returns.append({
                "date": str(df.index[i].date()),
                "raw_score": raw_score,
                "fwd_return": fwd_return,
            })

        if not scores_and_returns:
            continue

        sr_df = pd.DataFrame(scores_and_returns)

        # For calls: high score (near support) should predict positive returns
        favorable_call = sr_df[sr_df["raw_score"] >= 65]["fwd_return"]
        unfavorable_call = sr_df[sr_df["raw_score"] <= 35]["fwd_return"]
        neutral = sr_df[(sr_df["raw_score"] > 35) & (sr_df["raw_score"] < 65)]["fwd_return"]

        combined_favorable.extend(favorable_call.tolist())
        combined_unfavorable.extend(unfavorable_call.tolist())
        combined_neutral.extend(neutral.tolist())

        ticker_result = {
            "n_days": len(sr_df),
            "call_analysis": {
                "favorable_n": len(favorable_call),
                "favorable_mean_return": round(float(favorable_call.mean()), 3) if len(favorable_call) > 0 else None,
                "favorable_median_return": round(float(favorable_call.median()), 3) if len(favorable_call) > 0 else None,
                "favorable_win_rate": round(float((favorable_call > 0).mean()) * 100, 1) if len(favorable_call) > 0 else None,
                "unfavorable_n": len(unfavorable_call),
                "unfavorable_mean_return": round(float(unfavorable_call.mean()), 3) if len(unfavorable_call) > 0 else None,
                "unfavorable_median_return": round(float(unfavorable_call.median()), 3) if len(unfavorable_call) > 0 else None,
                "unfavorable_win_rate": round(float((unfavorable_call > 0).mean()) * 100, 1) if len(unfavorable_call) > 0 else None,
            },
            "put_analysis": {
                "favorable_n": len(unfavorable_call),  # Low score = good for puts
                "favorable_mean_return": round(float(-unfavorable_call.mean()), 3) if len(unfavorable_call) > 0 else None,
                "unfavorable_n": len(favorable_call),
                "unfavorable_mean_return": round(float(-favorable_call.mean()), 3) if len(favorable_call) > 0 else None,
            },
        }

        # Mean return across all entries
        ticker_result["baseline_mean_return"] = round(float(sr_df["fwd_return"].mean()), 3)
        ticker_result["baseline_win_rate"] = round(float((sr_df["fwd_return"] > 0).mean()) * 100, 1)

        all_results[ticker] = ticker_result

    # --- Aggregate analysis ---
    agg = {}
    if combined_favorable and combined_unfavorable:
        fav_arr = np.array(combined_favorable)
        unfav_arr = np.array(combined_unfavorable)

        fav_mean = float(fav_arr.mean())
        unfav_mean = float(unfav_arr.mean())
        edge = fav_mean - unfav_mean

        agg["call_favorable_mean"] = round(fav_mean, 3)
        agg["call_favorable_n"] = len(fav_arr)
        agg["call_favorable_wr"] = round(float((fav_arr > 0).mean()) * 100, 1)
        agg["call_unfavorable_mean"] = round(unfav_mean, 3)
        agg["call_unfavorable_n"] = len(unfav_arr)
        agg["call_unfavorable_wr"] = round(float((unfav_arr > 0).mean()) * 100, 1)
        agg["call_edge_pct"] = round(edge, 3)

        if combined_neutral:
            neut_arr = np.array(combined_neutral)
            agg["neutral_mean"] = round(float(neut_arr.mean()), 3)
            agg["neutral_n"] = len(neut_arr)

        # --- Permutation test ---
        log.info(f"  Running permutation test ({n_permutations} iterations)...")
        all_returns = np.concatenate([fav_arr, unfav_arr])
        n_fav = len(fav_arr)
        observed_edge = fav_mean - unfav_mean

        perm_edges = []
        rng = np.random.default_rng(42)
        for _ in range(n_permutations):
            shuffled = rng.permutation(all_returns)
            perm_fav = shuffled[:n_fav].mean()
            perm_unfav = shuffled[n_fav:].mean()
            perm_edges.append(perm_fav - perm_unfav)

        perm_edges = np.array(perm_edges)
        p_value = float((perm_edges >= observed_edge).mean())
        agg["permutation_p_value"] = round(p_value, 4)
        agg["permutation_significant"] = p_value < 0.05
        agg["observed_edge"] = round(observed_edge, 3)
        agg["perm_mean_edge"] = round(float(perm_edges.mean()), 4)
        agg["perm_std_edge"] = round(float(perm_edges.std()), 4)

        if perm_edges.std() > 0:
            agg["z_score"] = round((observed_edge - perm_edges.mean()) / perm_edges.std(), 2)
        else:
            agg["z_score"] = 0.0

        log.info(f"  Permutation test: edge={observed_edge:.3f}%, p={p_value:.4f}, "
                 f"z={agg.get('z_score', 0):.2f}")

    return {
        "generated_at": datetime.now().isoformat(),
        "hold_days": hold_days,
        "tickers": tickers,
        "per_ticker": all_results,
        "aggregate": agg,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Price Action Structure Score")
    parser.add_argument("--ticker", help="Single ticker to analyze")
    parser.add_argument("--dir", choices=["call", "put"], default="call", help="Direction")
    parser.add_argument("--backtest", action="store_true", help="Run backtest")
    parser.add_argument(
        "--tickers",
        default="XLF,XLE,XLU,XLC,XLP,XLK",
        help="Comma-sep tickers for backtest",
    )
    parser.add_argument("--hold-days", type=int, default=10, help="Forward hold days for backtest")
    parser.add_argument("--permutations", type=int, default=1000, help="N permutations")
    args = parser.parse_args()

    if args.backtest:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
        results = backtest_structure_score(
            tickers,
            hold_days=args.hold_days,
            n_permutations=args.permutations,
        )

        # Save results
        out_path = STATE / "price_structure_backtest.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        log.info(f"Backtest results saved to {out_path}")

        # Print summary
        print("\n" + "=" * 70)
        print("PRICE ACTION STRUCTURE — BACKTEST RESULTS")
        print(f"Hold period: {args.hold_days} days | Tickers: {', '.join(tickers)}")
        print("=" * 70)

        for ticker, res in results["per_ticker"].items():
            ca = res["call_analysis"]
            print(f"\n  {ticker} ({res['n_days']} days scored)")
            print(f"    Baseline: mean={res['baseline_mean_return']:+.3f}%, WR={res['baseline_win_rate']:.1f}%")
            if ca["favorable_mean_return"] is not None:
                print(f"    Calls - Favorable (score>=65): n={ca['favorable_n']}, "
                      f"mean={ca['favorable_mean_return']:+.3f}%, WR={ca['favorable_win_rate']:.1f}%")
            if ca["unfavorable_mean_return"] is not None:
                print(f"    Calls - Unfavorable (score<=35): n={ca['unfavorable_n']}, "
                      f"mean={ca['unfavorable_mean_return']:+.3f}%, WR={ca['unfavorable_win_rate']:.1f}%")

        agg = results.get("aggregate", {})
        if agg:
            print(f"\n  AGGREGATE:")
            print(f"    Favorable entries:   n={agg.get('call_favorable_n', 0)}, "
                  f"mean={agg.get('call_favorable_mean', 0):+.3f}%, "
                  f"WR={agg.get('call_favorable_wr', 0):.1f}%")
            print(f"    Unfavorable entries:  n={agg.get('call_unfavorable_n', 0)}, "
                  f"mean={agg.get('call_unfavorable_mean', 0):+.3f}%, "
                  f"WR={agg.get('call_unfavorable_wr', 0):.1f}%")
            print(f"    Edge (fav - unfav):   {agg.get('call_edge_pct', 0):+.3f}%")
            print(f"    Permutation p-value:  {agg.get('permutation_p_value', 'N/A')}")
            print(f"    Z-score:              {agg.get('z_score', 'N/A')}")
            sig = "YES" if agg.get("permutation_significant") else "NO"
            print(f"    Statistically significant (p<0.05): {sig}")

        print("\n" + "=" * 70)
        return results

    elif args.ticker:
        ticker = args.ticker.upper()
        result = score_price_structure(ticker, args.dir)

        print(f"\n{'=' * 60}")
        print(f"PRICE STRUCTURE: {ticker} ({args.dir.upper()})")
        print(f"{'=' * 60}")
        print(f"  Current price: ${result['current_price']:.2f}")
        print(f"  Raw score (direction-neutral): {result['raw_score']:.1f}")
        print(f"  Directional score ({args.dir}): {result['score']:.1f}")
        print(f"\n  Components:")
        for k, v in result["components"].items():
            print(f"    {k}: {v}")
        print(f"\n  Details:")
        for k, v in result["details"].items():
            if k == "tf_positions":
                print(f"    Range positions: {v}")
            elif k in ("support_levels", "resistance_levels"):
                if v:
                    levels_str = ", ".join(f"${p:.2f}(x{c})" for p, c in v)
                    print(f"    {k}: {levels_str}")
            else:
                print(f"    {k}: {v}")
        print(f"{'=' * 60}\n")
        return result

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
