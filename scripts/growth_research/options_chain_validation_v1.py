#!/usr/bin/env python3
"""
Options Chain Validation v1 — BS Approximation vs Real Market Prices
=====================================================================

Validates our ATR-based Black-Scholes approximation against real Dolt options
chain data (bid/ask/IV) for all 11 sector ETFs.

For each date in the backtest period where we have real chain data:
  1. Find the actual option chain prices for the spreads our model would trade
  2. Compare: BS-approximated spread price vs real market spread price
  3. Calculate: pricing error (%), fill probability (bid>0), actual vs modeled IV

Reports:
  - Mean/median pricing error by sector
  - Pricing error distribution (percentiles)
  - Cases where BS price was TOO GENEROUS vs TOO CONSERVATIVE
  - Adjusted Sharpe if using real prices instead of BS approximation
  - IV comparison: our ATR-based estimate vs actual market IV

Outputs to: output/growth_research/options_chain_validation_v1/
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

_builtin_print = print


def fprint(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()


# ── Standardized tools ──
sys.path.insert(0, "/home/jupiter/Lvl3Quant")
from research.tools.options_pricer import (
    bs_call_price,
    bs_put_price,
    estimate_iv,
    price_bull_call_spread,
    price_bear_put_spread,
    DEFAULT_HAIRCUT,
    COMMISSION_RT_SPREAD,
    RISK_FREE_RATE,
)

# ── Config ──
BASE = Path("/home/jupiter/Lvl3Quant")
CHAINS_DIR = BASE / "wheel_strategy_v1" / "data" / "cache" / "options_real" / "chains"
OUTPUT_DIR = BASE / "output" / "growth_research" / "options_chain_validation_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
DTE_TARGET = 14  # target DTE for spreads
DTE_TOLERANCE = 5  # accept DTE within +/- this many days of target
OTM_PCT = 2.0  # 2% OTM for K1
SPREAD_PCT = 3.0  # 3% spread width
HAIRCUT = 0.15  # 15% entry haircut used by BS model

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "options_chain_validation_v1"

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen(MLFLOW_URI, timeout=3)
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    MLFLOW_OK = True
    fprint(f"MLflow connected: {MLFLOW_URI}")
except Exception:
    fprint("MLflow unavailable — results saved to disk only")


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_chain(ticker: str) -> Optional[pd.DataFrame]:
    """Load per-ticker Dolt chain parquet."""
    path = CHAINS_DIR / f"{ticker}.parquet"
    if not path.exists():
        fprint(f"  {ticker}: chain parquet not found at {path}")
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df["expiration"] = pd.to_datetime(df["expiration"])
    for c in ["strike", "bid", "ask", "mid", "vol", "delta"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def download_price_data():
    """Download sector ETF + SPY/VIX price data via yfinance."""
    import yfinance as yf

    all_tickers = SECTORS + ["SPY", "^VIX"]
    fprint(f"Downloading {len(all_tickers)} tickers...")
    raw = yf.download(all_tickers, start="2019-01-01", progress=False, auto_adjust=True)
    mi = isinstance(raw.columns, pd.MultiIndex)

    close = raw["Close"] if mi else raw[["Close"]]
    high = raw["High"] if mi else raw[["High"]]
    low = raw["Low"] if mi else raw[["Low"]]

    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
        high.columns = high.columns.get_level_values(-1)
        low.columns = low.columns.get_level_values(-1)

    close = close.ffill()
    high = high.ffill()
    low = low.ffill()

    rename_map = {"^VIX": "VIX"}
    close = close.rename(columns=rename_map)
    high = high.rename(columns=rename_map)
    low = low.rename(columns=rename_map)

    fprint(f"Price data: {len(close)} days, {close.index[0].date()} to {close.index[-1].date()}")
    return close, high, low


def compute_atr_series(high, low, close, period=14):
    """Compute ATR series for all sectors."""
    atr_dict = {}
    for tk in SECTORS:
        if tk in high.columns and tk in low.columns and tk in close.columns:
            h = high[tk].dropna()
            l = low[tk].dropna()
            c = close[tk].dropna()
            common = h.index.intersection(l.index).intersection(c.index)
            if len(common) > period:
                tr1 = h.loc[common] - l.loc[common]
                tr2 = (h.loc[common] - c.loc[common].shift(1)).abs()
                tr3 = (l.loc[common] - c.loc[common].shift(1)).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr_dict[tk] = tr.ewm(alpha=1 / period, min_periods=period).mean()
    return atr_dict


# ═══════════════════════════════════════════════════════════════
# STRIKE MATCHING LOGIC
# ═══════════════════════════════════════════════════════════════

def find_nearest_strike(chain_day: pd.DataFrame, target_strike: float,
                        option_type: str, min_bid: float = 0.05) -> Optional[pd.Series]:
    """
    Find the nearest available LIQUID strike in the real chain for a given target.

    Args:
        chain_day: chain rows for one date + one expiration
        target_strike: desired strike price
        option_type: 'c' or 'p'
        min_bid: minimum bid to consider liquid (filter out penny bids)

    Returns:
        Row from chain with nearest strike, or None if nothing close.
    """
    subset = chain_day[chain_day["type"] == option_type].copy()
    if subset.empty:
        return None

    # Filter for liquidity: require bid > min_bid to exclude phantom quotes
    liquid = subset[subset["bid"] >= min_bid]
    if liquid.empty:
        # Fall back to all strikes if none are liquid
        liquid = subset

    liquid = liquid.copy()
    liquid["dist"] = (liquid["strike"] - target_strike).abs()
    best = liquid.loc[liquid["dist"].idxmin()]

    # Accept if within 3% of target (slightly wider tolerance)
    if best["dist"] / (target_strike + 1e-10) > 0.03:
        return None

    return best


def find_best_expiration(chain_on_date: pd.DataFrame, dte_target: int,
                         dte_tolerance: int) -> Optional[pd.Timestamp]:
    """Find the expiration closest to target DTE within tolerance."""
    if chain_on_date.empty:
        return None

    exps = chain_on_date[["expiration", "dte"]].drop_duplicates()
    exps["dte_dist"] = (exps["dte"] - dte_target).abs()
    valid = exps[exps["dte_dist"] <= dte_tolerance]

    if valid.empty:
        return None

    best = valid.loc[valid["dte_dist"].idxmin()]
    return best["expiration"]


# ═══════════════════════════════════════════════════════════════
# CORE COMPARISON ENGINE
# ═══════════════════════════════════════════════════════════════

def compare_spread_pricing(
    ticker: str,
    chain: pd.DataFrame,
    close: pd.Series,
    atr_series: pd.Series,
    vix_series: pd.Series,
    spread_type: str = "bull_call",
) -> list[dict]:
    """
    For each date in the chain, compare BS-approximated spread price
    vs real market spread price.

    Args:
        ticker: sector ETF ticker
        chain: full chain DataFrame for this ticker
        close: close price series
        atr_series: ATR series
        vix_series: VIX series
        spread_type: 'bull_call' or 'bear_put'

    Returns:
        List of comparison records.
    """
    records = []
    chain_dates = sorted(chain["date"].unique())

    for dt in chain_dates:
        if dt not in close.index or dt not in vix_series.index:
            continue
        if dt not in atr_series.index or pd.isna(atr_series.loc[dt]):
            continue

        S = float(close.loc[dt])
        vix = float(vix_series.loc[dt])
        atr = float(atr_series.loc[dt])

        if S <= 0 or atr <= 0:
            continue

        chain_day = chain[chain["date"] == dt]

        # Find best expiration near our DTE target
        best_exp = find_best_expiration(chain_day, DTE_TARGET, DTE_TOLERANCE)
        if best_exp is None:
            continue

        chain_exp = chain_day[chain_day["expiration"] == best_exp]
        actual_dte = int((best_exp - dt).days) if hasattr(best_exp, 'day') else DTE_TARGET

        # Determine strike structure
        if spread_type == "bull_call":
            # Bull call: K1=ATM (or slightly OTM), K2 = K1 + spread_width
            K1_target = round(S, 0)  # ATM
            K2_target = round(S * (1 + SPREAD_PCT / 100), 0)

            leg1 = find_nearest_strike(chain_exp, K1_target, "c")
            leg2 = find_nearest_strike(chain_exp, K2_target, "c")

            if leg1 is None or leg2 is None:
                continue

            K1_real = float(leg1["strike"])
            K2_real = float(leg2["strike"])
            if K2_real <= K1_real:
                continue

            # --- REAL MARKET PRICE (from bid/ask) ---
            # Buy K1 call at ask, sell K2 call at bid
            buy_ask = float(leg1["ask"]) if not pd.isna(leg1["ask"]) else None
            sell_bid = float(leg2["bid"]) if not pd.isna(leg2["bid"]) else None

            if buy_ask is None or sell_bid is None or buy_ask <= 0:
                continue

            real_spread_cost = buy_ask - sell_bid  # cost to enter at market
            real_spread_mid = float(leg1["mid"]) - float(leg2["mid"])  # mid-to-mid

            # Real IV from chain
            real_iv_k1 = float(leg1["vol"]) if not pd.isna(leg1.get("vol", np.nan)) else None
            real_iv_k2 = float(leg2["vol"]) if not pd.isna(leg2.get("vol", np.nan)) else None

            # --- BS APPROXIMATED PRICE ---
            bs_iv = estimate_iv(atr, S, vix)
            T = actual_dte / 365.0

            bs_fair = bs_call_price(S, K1_real, T, RISK_FREE_RATE, bs_iv) - \
                      bs_call_price(S, K2_real, T, RISK_FREE_RATE, bs_iv)
            bs_fair = max(bs_fair, 0.001)
            bs_with_haircut = bs_fair * (1 + HAIRCUT)

        elif spread_type == "bear_put":
            # Bear put: buy put at K2 (higher), sell put at K1 (lower)
            K2_target = round(S, 0)  # ATM
            K1_target = round(S * (1 - SPREAD_PCT / 100), 0)

            leg1 = find_nearest_strike(chain_exp, K1_target, "p")
            leg2 = find_nearest_strike(chain_exp, K2_target, "p")

            if leg1 is None or leg2 is None:
                continue

            K1_real = float(leg1["strike"])
            K2_real = float(leg2["strike"])
            if K2_real <= K1_real:
                continue

            # Buy K2 put at ask, sell K1 put at bid
            buy_ask = float(leg2["ask"]) if not pd.isna(leg2["ask"]) else None
            sell_bid = float(leg1["bid"]) if not pd.isna(leg1["bid"]) else None

            if buy_ask is None or sell_bid is None or buy_ask <= 0:
                continue

            real_spread_cost = buy_ask - sell_bid
            real_spread_mid = float(leg2["mid"]) - float(leg1["mid"])

            real_iv_k1 = float(leg1["vol"]) if not pd.isna(leg1.get("vol", np.nan)) else None
            real_iv_k2 = float(leg2["vol"]) if not pd.isna(leg2.get("vol", np.nan)) else None

            # BS approximation
            bs_iv = estimate_iv(atr, S, vix)
            T = actual_dte / 365.0
            bs_fair = bs_put_price(S, K2_real, T, RISK_FREE_RATE, bs_iv) - \
                      bs_put_price(S, K1_real, T, RISK_FREE_RATE, bs_iv)
            bs_fair = max(bs_fair, 0.001)
            bs_with_haircut = bs_fair * (1 + HAIRCUT)
        else:
            continue

        # --- COMPUTE COMPARISON METRICS ---
        spread_width = K2_real - K1_real

        # Bid-ask spread quality metrics
        leg1_bid = float(leg1["bid"]) if not pd.isna(leg1["bid"]) else 0
        leg2_bid = float(leg2["bid"]) if not pd.isna(leg2["bid"]) else 0
        leg1_ask = float(leg1["ask"]) if not pd.isna(leg1["ask"]) else 0
        leg2_ask = float(leg2["ask"]) if not pd.isna(leg2["ask"]) else 0

        # Effective real-world bid-ask haircut on the spread
        if real_spread_mid > 0:
            real_haircut_pct = (real_spread_cost - real_spread_mid) / real_spread_mid * 100
        else:
            real_haircut_pct = np.nan

        # Pricing error vs MARKET FILL (ask-bid, worst case)
        if real_spread_cost > 0:
            pricing_error_vs_market = (bs_with_haircut - real_spread_cost) / real_spread_cost * 100
        else:
            pricing_error_vs_market = np.nan

        # Pricing error vs MID (fair value, more realistic for limit orders)
        if real_spread_mid > 0:
            pricing_error_vs_mid = (bs_with_haircut - real_spread_mid) / real_spread_mid * 100
        else:
            pricing_error_vs_mid = np.nan

        # Fill probability: are both legs liquid (bid > 0.05)?
        fillable = leg1_bid > 0.05 and leg2_bid > 0.05

        # IV comparison
        iv_error_pct = None
        if real_iv_k1 is not None and real_iv_k1 > 0:
            iv_error_pct = (bs_iv - real_iv_k1) / real_iv_k1 * 100

        # Max profit comparison
        bs_max_profit = spread_width - bs_with_haircut
        real_max_profit_market = spread_width - real_spread_cost
        real_max_profit_mid = spread_width - real_spread_mid if real_spread_mid > 0 else np.nan

        records.append({
            "date": str(dt.date()) if hasattr(dt, 'date') else str(dt),
            "ticker": ticker,
            "spread_type": spread_type,
            "spot": round(S, 2),
            "K1": K1_real,
            "K2": K2_real,
            "dte": actual_dte,
            "vix": round(vix, 2),
            # BS model
            "bs_iv": round(bs_iv, 4),
            "bs_fair_value": round(bs_fair, 4),
            "bs_with_haircut": round(bs_with_haircut, 4),
            "bs_max_profit": round(bs_max_profit, 4),
            # Real market
            "real_spread_cost": round(real_spread_cost, 4),
            "real_spread_mid": round(real_spread_mid, 4),
            "real_iv_k1": round(real_iv_k1, 4) if real_iv_k1 else None,
            "real_iv_k2": round(real_iv_k2, 4) if real_iv_k2 else None,
            "real_max_profit_market": round(real_max_profit_market, 4),
            "real_max_profit_mid": round(float(real_max_profit_mid), 4) if not np.isnan(real_max_profit_mid) else None,
            "real_haircut_pct": round(float(real_haircut_pct), 2) if not np.isnan(real_haircut_pct) else None,
            # Comparison vs market fill (ask-bid, worst case)
            "pricing_error_pct": round(float(pricing_error_vs_market), 2) if not np.isnan(pricing_error_vs_market) else None,
            # Comparison vs mid (fair value, limit order scenario)
            "pricing_error_vs_mid_pct": round(float(pricing_error_vs_mid), 2) if not np.isnan(pricing_error_vs_mid) else None,
            "fillable": fillable,
            "iv_error_pct": round(iv_error_pct, 2) if iv_error_pct is not None else None,
            "bs_too_generous": bs_with_haircut < real_spread_cost,  # model says cheaper than reality
            "bs_too_conservative": bs_with_haircut > real_spread_cost,  # model overcharges
            "spread_width": round(spread_width, 2),
            # Leg-level bid-ask quality
            "leg1_bid": round(leg1_bid, 3),
            "leg1_ask": round(leg1_ask, 3),
            "leg2_bid": round(leg2_bid, 3),
            "leg2_ask": round(leg2_ask, 3),
        })

    return records


# ═══════════════════════════════════════════════════════════════
# SHARPE ADJUSTMENT ESTIMATION
# ═══════════════════════════════════════════════════════════════

def estimate_sharpe_impact(records_df: pd.DataFrame) -> dict:
    """
    Estimate how using real prices instead of BS approximation would change
    backtest performance.

    The key insight: if BS+haircut UNDERPAYS (too generous), then real entry costs
    are higher, reducing max_profit and inflating the backtest Sharpe.
    """
    valid = records_df.dropna(subset=["pricing_error_pct", "bs_max_profit", "real_max_profit_market"])
    if valid.empty:
        return {"error": "No valid comparisons"}

    # Average pricing error
    mean_err = valid["pricing_error_pct"].mean()
    median_err = valid["pricing_error_pct"].median()

    too_generous = (valid["bs_too_generous"]).sum()
    too_conservative = (valid["bs_too_conservative"]).sum()
    total = len(valid)

    # Profit impact: ratio of real max profit to BS max profit
    profit_ratios = valid["real_max_profit_market"] / (valid["bs_max_profit"] + 1e-10)
    avg_profit_ratio = profit_ratios.clip(-5, 5).mean()

    # Also compute vs mid for a more realistic estimate
    valid_mid = records_df.dropna(subset=["pricing_error_vs_mid_pct", "bs_max_profit", "real_max_profit_mid"])
    if not valid_mid.empty:
        mid_ratios = valid_mid["real_max_profit_mid"] / (valid_mid["bs_max_profit"] + 1e-10)
        avg_profit_ratio_mid = mid_ratios.clip(-5, 5).mean()
    else:
        avg_profit_ratio_mid = avg_profit_ratio

    # Crude Sharpe adjustment
    # If mean_err > 0, BS overcharges => real Sharpe should be HIGHER
    # If mean_err < 0, BS undercharges => real Sharpe should be LOWER
    # Approximate: Sharpe_adj = Sharpe_bs * avg_profit_ratio
    sharpe_multiplier = max(0.1, min(3.0, avg_profit_ratio))

    sharpe_multiplier_mid = max(0.1, min(3.0, avg_profit_ratio_mid))

    return {
        "mean_pricing_error_pct": round(mean_err, 2),
        "median_pricing_error_pct": round(median_err, 2),
        "too_generous_count": int(too_generous),
        "too_conservative_count": int(too_conservative),
        "total_comparisons": int(total),
        "pct_too_generous": round(too_generous / total * 100, 1),
        "pct_too_conservative": round(too_conservative / total * 100, 1),
        "avg_profit_ratio_market": round(avg_profit_ratio, 3),
        "avg_profit_ratio_mid": round(avg_profit_ratio_mid, 3),
        "sharpe_multiplier_market": round(sharpe_multiplier, 3),
        "sharpe_multiplier_mid": round(sharpe_multiplier_mid, 3),
        "interpretation": (
            f"BS+haircut is {'CONSERVATIVE (overcharges)' if mean_err > 0 else 'OPTIMISTIC (undercharges)'} "
            f"by {abs(mean_err):.1f}% vs market fill. "
            f"Sharpe adj: market_fill={sharpe_multiplier:.2f}x, limit_order={sharpe_multiplier_mid:.2f}x."
        ),
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    fprint("=" * 70)
    fprint("OPTIONS CHAIN VALIDATION v1")
    fprint("BS Approximation vs Real Market Prices (Dolt Chain Data)")
    fprint("=" * 70)
    fprint(f"Started: {datetime.now().isoformat()}")
    fprint(f"Sectors: {SECTORS}")
    fprint(f"DTE target: {DTE_TARGET} (±{DTE_TOLERANCE})")
    fprint(f"Spread: {SPREAD_PCT}% width, ATM entry")
    fprint(f"BS haircut: {HAIRCUT*100:.0f}%")
    fprint()

    # Check which sector ETFs have chain data
    available = {}
    for tk in SECTORS:
        chain = load_chain(tk)
        if chain is not None and len(chain) > 0:
            dates = chain["date"].unique()
            available[tk] = {
                "rows": len(chain),
                "dates": len(dates),
                "min_date": str(chain["date"].min().date()),
                "max_date": str(chain["date"].max().date()),
            }
            fprint(f"  {tk}: {len(chain):,} rows, {len(dates)} dates "
                   f"({chain['date'].min().date()} to {chain['date'].max().date()})")
        else:
            fprint(f"  {tk}: NO DATA")

    if not available:
        fprint("\nERROR: No sector ETF chain data found. Run daily_options_downloader.py first")
        fprint("  to materialize sector ETFs from Dolt.")
        return

    fprint(f"\n{len(available)}/{len(SECTORS)} sectors have chain data")

    # Download price data
    close, high, low = download_price_data()
    vix = close["VIX"] if "VIX" in close.columns else pd.Series(20.0, index=close.index)
    atr_dict = compute_atr_series(high, low, close)

    # --- Run comparison for each sector ---
    all_records = []
    sector_summaries = {}

    for tk in SECTORS:
        if tk not in available:
            continue

        fprint(f"\n{'─'*40}")
        fprint(f"Processing {tk}...")

        chain = load_chain(tk)
        if chain is None:
            continue

        if tk not in close.columns:
            fprint(f"  {tk}: not in price data, skipping")
            continue

        if tk not in atr_dict:
            fprint(f"  {tk}: no ATR data, skipping")
            continue

        # Compare bull call spreads
        bull_records = compare_spread_pricing(
            tk, chain, close[tk], atr_dict[tk], vix, spread_type="bull_call"
        )
        fprint(f"  Bull call spreads: {len(bull_records)} comparisons")

        # Compare bear put spreads
        bear_records = compare_spread_pricing(
            tk, chain, close[tk], atr_dict[tk], vix, spread_type="bear_put"
        )
        fprint(f"  Bear put spreads:  {len(bear_records)} comparisons")

        tk_records = bull_records + bear_records
        all_records.extend(tk_records)

        if tk_records:
            tk_df = pd.DataFrame(tk_records)
            valid = tk_df.dropna(subset=["pricing_error_pct"])
            if not valid.empty:
                sector_summaries[tk] = {
                    "n_comparisons": len(valid),
                    "mean_pricing_error_pct": round(valid["pricing_error_pct"].mean(), 2),
                    "median_pricing_error_pct": round(valid["pricing_error_pct"].median(), 2),
                    "std_pricing_error_pct": round(valid["pricing_error_pct"].std(), 2),
                    "pct_too_generous": round(valid["bs_too_generous"].mean() * 100, 1),
                    "pct_fillable": round(valid["fillable"].mean() * 100, 1),
                    "mean_iv_error_pct": round(valid["iv_error_pct"].dropna().mean(), 2) if valid["iv_error_pct"].dropna().any() else None,
                }
                fprint(f"  Mean pricing error: {sector_summaries[tk]['mean_pricing_error_pct']:+.1f}%")
                fprint(f"  BS too generous: {sector_summaries[tk]['pct_too_generous']:.0f}%")
                fprint(f"  Fillable: {sector_summaries[tk]['pct_fillable']:.0f}%")

    if not all_records:
        fprint("\nERROR: No valid comparisons produced. Check chain data quality.")
        return

    # --- Aggregate analysis ---
    fprint(f"\n{'═'*70}")
    fprint("AGGREGATE RESULTS")
    fprint(f"{'═'*70}")

    results_df = pd.DataFrame(all_records)
    valid_df = results_df.dropna(subset=["pricing_error_pct"])

    fprint(f"\nTotal comparisons: {len(valid_df)}")
    fprint(f"Date range: {valid_df['date'].min()} to {valid_df['date'].max()}")
    fprint(f"Sectors covered: {sorted(valid_df['ticker'].unique())}")

    # Overall pricing error distribution — vs MARKET FILL (ask-bid)
    pe = valid_df["pricing_error_pct"]
    fprint(f"\n--- COMPARISON 1: BS+haircut vs MARKET FILL (buy@ask, sell@bid) ---")
    fprint(f"  This is the WORST CASE: crossing the spread on both legs.")
    fprint(f"  Positive = BS overcharges (conservative), Negative = BS undercharges (optimistic)")
    fprint(f"  Mean:   {pe.mean():+.2f}%")
    fprint(f"  Median: {pe.median():+.2f}%")
    fprint(f"  Std:    {pe.std():.2f}%")
    fprint(f"  P5:     {pe.quantile(0.05):+.2f}%")
    fprint(f"  P25:    {pe.quantile(0.25):+.2f}%")
    fprint(f"  P75:    {pe.quantile(0.75):+.2f}%")
    fprint(f"  P95:    {pe.quantile(0.95):+.2f}%")

    # vs MID (fair value / limit order scenario)
    pe_mid = valid_df["pricing_error_vs_mid_pct"].dropna()
    # Clip extreme outliers (from near-zero mid values)
    pe_mid = pe_mid.clip(-500, 500)
    if not pe_mid.empty:
        fprint(f"\n--- COMPARISON 2: BS+haircut vs MID PRICE (limit order scenario) ---")
        fprint(f"  This is the REALISTIC case for patient fills with limit orders.")
        fprint(f"  Mean:   {pe_mid.mean():+.2f}%")
        fprint(f"  Median: {pe_mid.median():+.2f}%")
        fprint(f"  P25:    {pe_mid.quantile(0.25):+.2f}%")
        fprint(f"  P75:    {pe_mid.quantile(0.75):+.2f}%")

    # Real-world bid-ask haircut
    rh = valid_df["real_haircut_pct"].dropna().clip(-500, 500)
    if not rh.empty:
        fprint(f"\n--- REAL-WORLD BID-ASK HAIRCUT ---")
        fprint(f"  How much more market fills cost vs mid (BS assumes {HAIRCUT*100:.0f}%):")
        fprint(f"  Mean:   {rh.mean():+.1f}%")
        fprint(f"  Median: {rh.median():+.1f}%")
        fprint(f"  P25:    {rh.quantile(0.25):+.1f}%")
        fprint(f"  P75:    {rh.quantile(0.75):+.1f}%")

    # BS direction analysis
    too_gen = valid_df["bs_too_generous"].sum()
    too_cons = valid_df["bs_too_conservative"].sum()
    fprint(f"\nBS Direction (vs market fill):")
    fprint(f"  Too generous (underprice entry): {too_gen} ({too_gen/len(valid_df)*100:.1f}%)")
    fprint(f"  Too conservative (overprice entry): {too_cons} ({too_cons/len(valid_df)*100:.1f}%)")

    # Fill probability
    fill_pct = valid_df["fillable"].mean() * 100
    fprint(f"\nFill Probability: {fill_pct:.1f}% of spreads are fillable (both legs have bid>$0.05)")

    # IV comparison
    iv_valid = valid_df.dropna(subset=["iv_error_pct"])
    if not iv_valid.empty:
        ive = iv_valid["iv_error_pct"]
        fprint(f"\nIV Estimation Error (BS ATR-based IV vs real market IV):")
        fprint(f"  Mean:   {ive.mean():+.2f}%")
        fprint(f"  Median: {ive.median():+.2f}%")
        fprint(f"  Std:    {ive.std():.2f}%")

    # By spread type
    fprint(f"\nBy Spread Type:")
    for st in ["bull_call", "bear_put"]:
        sub = valid_df[valid_df["spread_type"] == st]
        if not sub.empty:
            fprint(f"  {st}: n={len(sub)}, mean_err={sub['pricing_error_pct'].mean():+.1f}%, "
                   f"median_err={sub['pricing_error_pct'].median():+.1f}%")

    # By sector
    fprint(f"\nBy Sector:")
    fprint(f"  {'Sector':<6} {'N':>5} {'Mean Err%':>10} {'Med Err%':>10} {'Too Gen%':>10} {'Fill%':>8} {'IV Err%':>10}")
    fprint(f"  {'─'*6} {'─'*5} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*10}")
    for tk in sorted(sector_summaries.keys()):
        s = sector_summaries[tk]
        iv_str = f"{s['mean_iv_error_pct']:+.1f}" if s['mean_iv_error_pct'] is not None else "N/A"
        fprint(f"  {tk:<6} {s['n_comparisons']:>5} {s['mean_pricing_error_pct']:>+10.1f} "
               f"{s['median_pricing_error_pct']:>+10.1f} {s['pct_too_generous']:>10.1f} "
               f"{s['pct_fillable']:>8.1f} {iv_str:>10}")

    # By VIX regime
    fprint(f"\nBy VIX Regime:")
    vix_bins = [(0, 15, "Low (<15)"), (15, 20, "Normal (15-20)"),
                (20, 25, "Elevated (20-25)"), (25, 35, "High (25-35)"),
                (35, 100, "Crisis (>35)")]
    for lo, hi, label in vix_bins:
        sub = valid_df[(valid_df["vix"] >= lo) & (valid_df["vix"] < hi)]
        if len(sub) > 5:
            fprint(f"  VIX {label}: n={len(sub)}, mean_err={sub['pricing_error_pct'].mean():+.1f}%, "
                   f"std={sub['pricing_error_pct'].std():.1f}%")

    # Sharpe impact estimation
    fprint(f"\n{'═'*70}")
    fprint("SHARPE IMPACT ESTIMATION")
    fprint(f"{'═'*70}")

    impact = estimate_sharpe_impact(valid_df)
    for k, v in impact.items():
        fprint(f"  {k}: {v}")

    # By spread type
    for st in ["bull_call", "bear_put"]:
        sub = valid_df[valid_df["spread_type"] == st]
        if len(sub) > 10:
            imp = estimate_sharpe_impact(sub)
            fprint(f"\n  {st} Sharpe impact:")
            fprint(f"    multiplier (market): {imp.get('sharpe_multiplier_market', 'N/A')}")
            fprint(f"    multiplier (mid/limit): {imp.get('sharpe_multiplier_mid', 'N/A')}")
            fprint(f"    interpretation: {imp['interpretation']}")

    # --- Save results ---
    results_df.to_csv(OUTPUT_DIR / "all_comparisons.csv", index=False)
    results_df.to_parquet(OUTPUT_DIR / "all_comparisons.parquet", index=False)

    summary = {
        "run_timestamp": datetime.now().isoformat(),
        "config": {
            "dte_target": DTE_TARGET,
            "dte_tolerance": DTE_TOLERANCE,
            "otm_pct": OTM_PCT,
            "spread_pct": SPREAD_PCT,
            "haircut": HAIRCUT,
        },
        "aggregate": {
            "total_comparisons": len(valid_df),
            "date_range": [str(valid_df["date"].min()), str(valid_df["date"].max())],
            "sectors_covered": sorted(valid_df["ticker"].unique().tolist()),
            "pricing_error": {
                "mean_pct": round(pe.mean(), 2),
                "median_pct": round(pe.median(), 2),
                "std_pct": round(pe.std(), 2),
                "p5": round(pe.quantile(0.05), 2),
                "p95": round(pe.quantile(0.95), 2),
            },
            "too_generous_pct": round(too_gen / len(valid_df) * 100, 1),
            "too_conservative_pct": round(too_cons / len(valid_df) * 100, 1),
            "fill_probability_pct": round(fill_pct, 1),
        },
        "sharpe_impact": impact,
        "by_sector": sector_summaries,
    }

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    fprint(f"\nResults saved to {OUTPUT_DIR}")

    # MLflow logging
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name="options_chain_validation_v1"):
                mlflow.log_param("dte_target", DTE_TARGET)
                mlflow.log_param("spread_pct", SPREAD_PCT)
                mlflow.log_param("haircut", HAIRCUT)
                mlflow.log_param("sectors", ",".join(SECTORS))

                mlflow.log_metric("total_comparisons", len(valid_df))
                mlflow.log_metric("mean_pricing_error_pct", pe.mean())
                mlflow.log_metric("median_pricing_error_pct", pe.median())
                mlflow.log_metric("std_pricing_error_pct", pe.std())
                mlflow.log_metric("pct_too_generous", too_gen / len(valid_df) * 100)
                mlflow.log_metric("fill_probability_pct", fill_pct)
                mlflow.log_metric("sharpe_multiplier_market", impact.get("sharpe_multiplier_market", 1.0))
                mlflow.log_metric("sharpe_multiplier_mid", impact.get("sharpe_multiplier_mid", 1.0))

                mlflow.log_artifact(str(OUTPUT_DIR / "summary.json"))
                mlflow.log_artifact(str(OUTPUT_DIR / "all_comparisons.csv"))

            fprint("MLflow run logged successfully")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    fprint(f"\nCompleted: {datetime.now().isoformat()}")
    fprint("=" * 70)


if __name__ == "__main__":
    main()
