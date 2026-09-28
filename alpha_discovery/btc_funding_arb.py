# -*- coding: utf-8 -*-
"""
BTC Perpetual Futures Funding Rate Arbitrage Analysis
======================================================
Downloads historical BTCUSDT funding rates from Binance Futures API,
calculates statistics, and simulates several strategy variations.

Strategy: Delta-neutral carry (long spot BTC + short BTCUSDT perpetual).
  - Collect funding payments when rate > 0 (longs pay shorts).
  - Reverse (short spot, long perp) when rate < 0 to collect negative funding.

Usage:
    python btc_funding_arb.py

Requirements:
    pip install requests pandas numpy

Author: Generated for Lvl3Quant alpha discovery
"""

import requests
import time
import json
import sys
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple, Optional

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

BINANCE_FUTURES_BASE = "https://fapi.binance.com"
SYMBOL = "BTCUSDT"
LOOKBACK_DAYS = 365          # 1 year of history
FUNDING_PERIODS_PER_DAY = 3  # Binance: every 8 hours

# -- Fee assumptions (Binance standard retail tier, no BNB discount) ----------
SPOT_MAKER_FEE   = 0.0010  # 0.10% spot maker (adds liquidity, limit order)
SPOT_TAKER_FEE   = 0.0010  # 0.10% spot taker (market order)
PERP_MAKER_FEE   = 0.0002  # 0.02% perp maker
PERP_TAKER_FEE   = 0.0005  # 0.05% perp taker
SLIPPAGE_BPS     = 5        # 5 bps one-way slippage on each leg (conservative)

# -- Margin / capital assumptions ---------------------------------------------
# For delta-neutral: $1 of notional requires ~$1 spot + ~$0.10 perp margin (10x)
# We model it as: total capital = notional (spot cash) + 10% margin on perp
PERP_MARGIN_RATIO   = 0.10  # 10% initial margin on perp side (10x leverage)
RISK_FREE_RATE_APY  = 0.053 # 5.3% - opportunity cost (US T-bill proxy, 2024 avg)

# -- Strategy thresholds -------------------------------------------------------
# Strategy 2: Only enter if |funding_rate| exceeds this per-period threshold
ENTRY_THRESHOLD_BPS  = 5.0   # 5 bps = 0.05% per 8h (~54% annualized)
EXIT_THRESHOLD_BPS   = 1.0   # Exit (close position) when rate < 1 bps
COOLDOWN_PERIODS     = 3     # Don't re-enter for N periods after forced exit

# -----------------------------------------------------------------------------
# DATA DOWNLOAD
# -----------------------------------------------------------------------------

def download_funding_rates(
    symbol: str = SYMBOL,
    lookback_days: int = LOOKBACK_DAYS,
) -> List[Dict]:
    """
    Download historical funding rates from Binance Futures API.

    Returns list of dicts sorted ascending by fundingTime:
        [{"symbol", "fundingTime" (ms), "fundingRate" (str), "markPrice" (str)}, ...]

    Handles pagination automatically (API limit = 1000 per call).
    """
    endpoint = f"{BINANCE_FUTURES_BASE}/fapi/v1/fundingRate"

    end_ms   = int(time.time() * 1000)
    start_ms = end_ms - int(lookback_days * 24 * 3600 * 1000)

    all_records: List[Dict] = []
    cursor_start = start_ms
    batch_size   = 1000

    print(f"Downloading BTCUSDT funding rates: "
          f"{datetime.utcfromtimestamp(start_ms/1000).date()} -> "
          f"{datetime.utcfromtimestamp(end_ms/1000).date()}")

    while cursor_start < end_ms:
        params = {
            "symbol":    symbol,
            "startTime": cursor_start,
            "endTime":   end_ms,
            "limit":     batch_size,
        }
        try:
            resp = requests.get(endpoint, params=params, timeout=15)
            resp.raise_for_status()
            batch = resp.json()
        except requests.exceptions.RequestException as exc:
            print(f"  [ERROR] API request failed: {exc}")
            print("  Using synthetic data for demonstration ...")
            return _generate_synthetic_data(lookback_days)

        if not batch:
            break

        all_records.extend(batch)
        last_time = batch[-1]["fundingTime"]

        print(f"  Fetched {len(batch):4d} records | "
              f"latest: {datetime.utcfromtimestamp(last_time/1000).strftime('%Y-%m-%d %H:%M')}")

        if len(batch) < batch_size:
            break  # last page
        cursor_start = last_time + 1  # advance past last record

        time.sleep(0.2)  # gentle rate limiting

    # Deduplicate and sort
    seen = set()
    unique = []
    for r in all_records:
        if r["fundingTime"] not in seen:
            seen.add(r["fundingTime"])
            unique.append(r)
    unique.sort(key=lambda x: x["fundingTime"])

    print(f"  Total records: {len(unique)}")
    return unique


def _generate_synthetic_data(lookback_days: int) -> List[Dict]:
    """
    Generate realistic synthetic BTC funding rate data when the API is
    unavailable (geo-block, rate limit, etc.).

    Calibrated to observed Binance BTCUSDT statistics:
      - Mean ~0.01%/period (long-run baseline = 10.95% APY)
      - Std  ~0.04%/period
      - Occasional spikes to +/-0.1%/period during volatile markets
      - Slight positive autocorrelation (AR(1) phi=0.30)
      - Bull markets (2024): mean drifts up to ~0.03%
    """
    import random
    import math
    random.seed(42)

    n_periods = lookback_days * FUNDING_PERIODS_PER_DAY
    now_ms    = int(time.time() * 1000)
    # Start of range
    start_ms  = now_ms - int(lookback_days * 24 * 3600 * 1000)
    period_ms = 8 * 3600 * 1000  # 8 hours in ms

    records = []
    prev_rate = 0.0001  # 1 bps starting point
    phi       = 0.30    # AR(1) coefficient

    for i in range(n_periods):
        ts  = start_ms + i * period_ms
        pct = i / n_periods  # fraction through the year

        # Slowly drifting mean (positive trend in bull market)
        mu_shift  = 0.00005 + 0.00005 * math.sin(pct * 2 * math.pi)  # seasonal
        residual  = random.gauss(0, 0.0003)

        # AR(1) with heavy-ish tail (occasional spike)
        spike = 0.0
        if random.random() < 0.01:   # 1% chance of spike
            spike = random.gauss(0, 0.002) * random.choice([-1, 1])

        rate = phi * prev_rate + (1 - phi) * mu_shift + residual + spike
        # Clamp to Binance exchange limits (+/-0.75%)
        rate = max(-0.0075, min(0.0075, rate))
        prev_rate = rate

        records.append({
            "symbol":      SYMBOL,
            "fundingTime": ts,
            "fundingRate": f"{rate:.8f}",
            "markPrice":   "50000.00",  # placeholder
        })

    print(f"  Generated {len(records)} synthetic records")
    return records


# -----------------------------------------------------------------------------
# STATISTICS
# -----------------------------------------------------------------------------

def compute_statistics(rates: List[float]) -> Dict:
    """
    Compute descriptive statistics on raw funding rate series (per-period values).
    """
    n = len(rates)
    if n == 0:
        return {}

    # Basic moments
    mean_rate = sum(rates) / n
    variance  = sum((r - mean_rate) ** 2 for r in rates) / (n - 1)
    std_rate  = variance ** 0.5

    sorted_r  = sorted(rates)
    median    = sorted_r[n // 2]
    p10       = sorted_r[int(n * 0.10)]
    p25       = sorted_r[int(n * 0.25)]
    p75       = sorted_r[int(n * 0.75)]
    p90       = sorted_r[int(n * 0.90)]
    p95       = sorted_r[int(n * 0.95)]
    p99       = sorted_r[int(n * 0.99)]
    min_r     = sorted_r[0]
    max_r     = sorted_r[-1]

    # Annualization (3 periods / day x 365 days)
    periods_per_year  = FUNDING_PERIODS_PER_DAY * 365
    ann_return_simple = mean_rate * periods_per_year
    ann_return_compnd = (1 + mean_rate) ** periods_per_year - 1
    ann_std           = std_rate * (periods_per_year ** 0.5)

    # Fraction of periods where long pays (rate > 0 -> short earns)
    pct_positive = sum(1 for r in rates if r > 0) / n
    pct_negative = sum(1 for r in rates if r < 0) / n

    # Auto-correlation lag-1
    if n > 1:
        mean_r = mean_rate
        cov    = sum((rates[i] - mean_r) * (rates[i-1] - mean_r) for i in range(1, n))
        var    = sum((r - mean_r) ** 2 for r in rates)
        acf1   = cov / var if var > 0 else 0.0
    else:
        acf1   = 0.0

    return {
        "n_periods":         n,
        "mean_per_period":   mean_rate,
        "median_per_period": median,
        "std_per_period":    std_rate,
        "min_rate":          min_r,
        "max_rate":          max_r,
        "p10": p10, "p25": p25, "p75": p75, "p90": p90, "p95": p95, "p99": p99,
        "ann_return_simple":  ann_return_simple,
        "ann_return_compound":ann_return_compnd,
        "ann_std":            ann_std,
        "pct_positive":       pct_positive,
        "pct_negative":       pct_negative,
        "acf_lag1":           acf1,
        "periods_per_year":   periods_per_year,
    }


# -----------------------------------------------------------------------------
# COST MODEL
# -----------------------------------------------------------------------------

def round_trip_cost_bps(
    spot_fee: float = SPOT_TAKER_FEE,
    perp_fee: float = PERP_TAKER_FEE,
    slippage: float = SLIPPAGE_BPS / 10_000,
    perp_margin_ratio: float = PERP_MARGIN_RATIO,
    holding_periods: int = 0,
) -> Dict:
    """
    Compute the round-trip cost to enter and exit a delta-neutral position.

    Per leg:
      - Entry: pay spot_fee + perp_fee + 2x slippage
      - Exit:  same
      - Total round-trip cost as a fraction of notional

    Capital efficiency:
      - Notional $1 BTC requires: $1 spot cash + $perp_margin_ratio on perp
      - Effective capital deployed = 1 + perp_margin_ratio
      - Opportunity cost = risk_free_rate x holding_periods / periods_per_year x capital
    """
    entry_cost = spot_fee + perp_fee + 2 * slippage
    exit_cost  = spot_fee + perp_fee + 2 * slippage
    total_cost = entry_cost + exit_cost

    capital_per_unit_notional = 1.0 + perp_margin_ratio
    opp_cost_per_period = (
        RISK_FREE_RATE_APY / (FUNDING_PERIODS_PER_DAY * 365)
        * capital_per_unit_notional
    )
    total_opp_cost = opp_cost_per_period * holding_periods

    return {
        "entry_cost_bps":    entry_cost * 10_000,
        "exit_cost_bps":     exit_cost  * 10_000,
        "round_trip_bps":    total_cost * 10_000,
        "round_trip_pct":    total_cost * 100,
        "capital_ratio":     capital_per_unit_notional,
        "opp_cost_per_period": opp_cost_per_period,
        "total_opp_cost_for_hold": total_opp_cost,
        "break_even_periods": (
            total_cost / opp_cost_per_period if opp_cost_per_period > 0 else 0
        ),
    }


# -----------------------------------------------------------------------------
# STRATEGY SIMULATIONS
# -----------------------------------------------------------------------------

def _drawdown_series(equity: List[float]) -> Tuple[float, int, int]:
    """
    Compute max drawdown, duration (in periods), and recovery index
    from an equity curve.

    Returns: (max_drawdown_pct, max_dd_start_idx, max_dd_end_idx)
    """
    if len(equity) < 2:
        return 0.0, 0, 0

    peak_val  = equity[0]
    peak_idx  = 0
    max_dd    = 0.0
    dd_start  = 0
    dd_end    = 0

    for i in range(1, len(equity)):
        if equity[i] > peak_val:
            peak_val = equity[i]
            peak_idx = i
        dd = (peak_val - equity[i]) / peak_val
        if dd > max_dd:
            max_dd   = dd
            dd_start = peak_idx
            dd_end   = i

    return max_dd, dd_start, dd_end


def simulate_strategy_1_always_on(
    rates: List[float],
    timestamps: List[int],
) -> Dict:
    """
    Strategy 1: Always-on delta-neutral carry.

    Logic:
      - Always hold: long spot + short perp (or reverse if rate negative).
      - Collect |rate| every period.
      - Subtract round-trip cost ONCE at start and end of entire holding period.
      - Subtract opportunity cost of capital each period.
      - Net P&L per period: |rate| - opp_cost_per_period.
    """
    n = len(rates)
    if n == 0:
        return {}

    # One-time round-trip cost at entry + exit (amortized over full holding period)
    rt_costs       = round_trip_cost_bps()
    entry_exit_hit = rt_costs["round_trip_pct"] / 100   # fraction of notional
    opp_cost_pp    = rt_costs["opp_cost_per_period"]

    # Equity curve (starts at 1.0 = one unit of notional)
    equity: List[float]     = [1.0]
    pnl_per_period: List[float] = []
    cumulative_pnl           = -entry_exit_hit  # immediate entry cost

    for i, rate in enumerate(rates):
        # Collect |rate| regardless of sign (we flip direction with the rate)
        gross   = abs(rate)
        net_pnl = gross - opp_cost_pp
        cumulative_pnl += net_pnl
        pnl_per_period.append(net_pnl)
        equity.append(1.0 + cumulative_pnl)

    # Final exit cost
    cumulative_pnl -= rt_costs["exit_cost_bps"] / 10_000
    equity[-1]     = 1.0 + cumulative_pnl

    max_dd, dd_start, dd_end = _drawdown_series(equity)

    periods_per_year = FUNDING_PERIODS_PER_DAY * 365
    total_return     = equity[-1] - 1.0
    days_held        = n / FUNDING_PERIODS_PER_DAY
    ann_return       = (1 + total_return) ** (365 / days_held) - 1

    # Sharpe: daily PnL scaled
    daily_pnl: List[float] = []
    for d in range(int(days_held)):
        chunk = pnl_per_period[d*3 : d*3+3]
        daily_pnl.append(sum(chunk))

    if len(daily_pnl) > 1:
        daily_mean  = sum(daily_pnl) / len(daily_pnl)
        daily_var   = sum((x - daily_mean)**2 for x in daily_pnl) / (len(daily_pnl)-1)
        daily_std   = daily_var ** 0.5
        daily_rf    = RISK_FREE_RATE_APY / 365
        sharpe      = (daily_mean - daily_rf) / daily_std * (252 ** 0.5) if daily_std > 0 else 0
    else:
        sharpe = 0.0

    # Max drawdown duration in days
    dd_duration_days = (dd_end - dd_start) / FUNDING_PERIODS_PER_DAY

    return {
        "strategy":          "1_always_on",
        "total_return_pct":  total_return * 100,
        "annualized_return_pct": ann_return * 100,
        "sharpe_ratio":      sharpe,
        "max_drawdown_pct":  max_dd * 100,
        "dd_duration_days":  dd_duration_days,
        "n_periods":         n,
        "days_held":         days_held,
        "equity_final":      equity[-1],
        "equity_curve":      equity,
        "pnl_per_period":    pnl_per_period,
    }


def simulate_strategy_2_threshold(
    rates: List[float],
    timestamps: List[int],
    entry_thresh: float = ENTRY_THRESHOLD_BPS / 10_000,
    exit_thresh:  float = EXIT_THRESHOLD_BPS  / 10_000,
    cooldown:     int   = COOLDOWN_PERIODS,
) -> Dict:
    """
    Strategy 2: Enter only when |funding_rate| >= entry_thresh.
                Exit when |funding_rate| < exit_thresh.

    Avoids deploying capital during low-funding regimes.
    Capital earns risk-free rate when not deployed.
    """
    n          = len(rates)
    rt_costs   = round_trip_cost_bps()
    opp_pp     = rt_costs["opp_cost_per_period"]
    rt_cost_f  = rt_costs["round_trip_pct"] / 100
    rf_pp      = RISK_FREE_RATE_APY / (FUNDING_PERIODS_PER_DAY * 365)

    equity          = [1.0]
    pnl_per_period  = []
    in_position     = False
    cooldown_count  = 0
    n_entries       = 0
    n_exits         = 0
    total_periods_in = 0

    cumulative_pnl = 0.0

    for i, rate in enumerate(rates):
        if cooldown_count > 0:
            cooldown_count -= 1

        if not in_position:
            if abs(rate) >= entry_thresh and cooldown_count == 0:
                # Enter position
                in_position     = True
                n_entries      += 1
                cumulative_pnl -= rt_costs["entry_cost_bps"] / 10_000
                gross           = abs(rate) - opp_pp
                cumulative_pnl += gross
                pnl_per_period.append(gross)
                total_periods_in += 1
            else:
                # Sit in cash / T-bills
                cumulative_pnl += rf_pp
                pnl_per_period.append(rf_pp)
        else:
            if abs(rate) < exit_thresh:
                # Exit position
                in_position     = False
                n_exits        += 1
                cooldown_count  = cooldown
                cumulative_pnl -= rt_costs["exit_cost_bps"] / 10_000
                cumulative_pnl += rf_pp  # cash for this period
                pnl_per_period.append(rf_pp)
            else:
                gross           = abs(rate) - opp_pp
                cumulative_pnl += gross
                pnl_per_period.append(gross)
                total_periods_in += 1

        equity.append(1.0 + cumulative_pnl)

    # Force close if still open at end
    if in_position:
        cumulative_pnl -= rt_costs["exit_cost_bps"] / 10_000
        n_exits += 1
        equity[-1] = 1.0 + cumulative_pnl

    max_dd, dd_start, dd_end = _drawdown_series(equity)

    days_held  = n / FUNDING_PERIODS_PER_DAY
    total_ret  = equity[-1] - 1.0
    ann_return = (1 + total_ret) ** (365 / days_held) - 1 if days_held > 0 else 0

    # Sharpe from daily buckets
    daily_pnl = []
    for d in range(int(days_held)):
        chunk = pnl_per_period[d*3 : d*3+3]
        daily_pnl.append(sum(chunk))
    if len(daily_pnl) > 1:
        dm = sum(daily_pnl) / len(daily_pnl)
        dv = sum((x-dm)**2 for x in daily_pnl) / (len(daily_pnl)-1)
        ds = dv ** 0.5
        rf_d = RISK_FREE_RATE_APY / 365
        sharpe = (dm - rf_d) / ds * (252**0.5) if ds > 0 else 0
    else:
        sharpe = 0.0

    pct_deployed = total_periods_in / n * 100 if n > 0 else 0

    return {
        "strategy":               "2_threshold",
        "entry_threshold_bps":    entry_thresh * 10_000,
        "exit_threshold_bps":     exit_thresh  * 10_000,
        "total_return_pct":       total_ret   * 100,
        "annualized_return_pct":  ann_return  * 100,
        "sharpe_ratio":           sharpe,
        "max_drawdown_pct":       max_dd * 100,
        "dd_duration_days":       (dd_end - dd_start) / FUNDING_PERIODS_PER_DAY,
        "n_entries":              n_entries,
        "n_exits":                n_exits,
        "pct_time_deployed":      pct_deployed,
        "equity_final":           equity[-1],
        "equity_curve":           equity,
        "pnl_per_period":         pnl_per_period,
    }


def simulate_strategy_3_cross_exchange(
    rates_exchange_a: List[float],
    rates_exchange_b: List[float],
    timestamps: List[int],
    min_spread_bps: float = 2.0,
) -> Dict:
    """
    Strategy 3: Cross-exchange arbitrage.

    Enter when |rate_A - rate_B| > round_trip_cost + min_spread.
    Long perp on low-rate exchange, short perp on high-rate exchange.
    This is pure funding arb - no spot leg needed.

    NOTE: In practice this requires simultaneous execution on two exchanges
    and is exposed to operational risk (API outages, different settlement times).

    We simulate using synthetic spread around real rates (exchange B rates
    are modelled as rate_A + noise + occasional divergence).
    """
    n        = len(rates_exchange_a)
    rt_costs = round_trip_cost_bps(
        spot_fee=0,        # No spot leg
        perp_fee=PERP_TAKER_FEE,
        slippage=SLIPPAGE_BPS / 10_000,
    )
    opp_pp   = rt_costs["opp_cost_per_period"] / 2  # halved (no spot margin needed)
    rt_cost  = rt_costs["round_trip_pct"] / 100 * 2  # two perp legs
    rf_pp    = RISK_FREE_RATE_APY / (FUNDING_PERIODS_PER_DAY * 365)
    threshold = min_spread_bps / 10_000 + PERP_TAKER_FEE * 4

    equity         = [1.0]
    pnl_per_period = []
    in_position    = False
    n_entries      = 0
    cumulative_pnl = 0.0

    for i in range(n):
        ra = rates_exchange_a[i]
        rb = rates_exchange_b[i] if i < len(rates_exchange_b) else ra
        spread = rb - ra  # positive: B is higher -> short B, long A

        if not in_position:
            if abs(spread) >= threshold:
                in_position     = True
                n_entries      += 1
                cumulative_pnl -= rt_cost / 2  # entry cost
                net = abs(spread) - opp_pp * 2
                cumulative_pnl += net
                pnl_per_period.append(net)
            else:
                cumulative_pnl += rf_pp
                pnl_per_period.append(rf_pp)
        else:
            if abs(spread) < threshold / 2:
                in_position    = False
                cumulative_pnl -= rt_cost / 2  # exit
                cumulative_pnl += rf_pp
                pnl_per_period.append(rf_pp)
            else:
                net = abs(spread) - opp_pp * 2
                cumulative_pnl += net
                pnl_per_period.append(net)

        equity.append(1.0 + cumulative_pnl)

    if in_position:
        cumulative_pnl -= rt_cost / 2
        equity[-1] = 1.0 + cumulative_pnl

    max_dd, dd_start, dd_end = _drawdown_series(equity)
    days_held  = n / FUNDING_PERIODS_PER_DAY
    total_ret  = equity[-1] - 1.0
    ann_return = (1 + total_ret) ** (365 / days_held) - 1 if days_held > 0 else 0

    daily_pnl = []
    for d in range(int(days_held)):
        chunk = pnl_per_period[d*3 : d*3+3]
        daily_pnl.append(sum(chunk))
    if len(daily_pnl) > 1:
        dm = sum(daily_pnl) / len(daily_pnl)
        dv = sum((x-dm)**2 for x in daily_pnl) / (len(daily_pnl)-1)
        ds = dv ** 0.5
        rf_d = RISK_FREE_RATE_APY / 365
        sharpe = (dm - rf_d) / ds * (252**0.5) if ds > 0 else 0
    else:
        sharpe = 0.0

    return {
        "strategy":              "3_cross_exchange",
        "min_spread_bps":        min_spread_bps,
        "total_return_pct":      total_ret   * 100,
        "annualized_return_pct": ann_return  * 100,
        "sharpe_ratio":          sharpe,
        "max_drawdown_pct":      max_dd * 100,
        "dd_duration_days":      (dd_end - dd_start) / FUNDING_PERIODS_PER_DAY,
        "n_entries":             n_entries,
        "equity_final":          equity[-1],
        "equity_curve":          equity,
    }


# -----------------------------------------------------------------------------
# SENSITIVITY ANALYSIS
# -----------------------------------------------------------------------------

def sensitivity_analysis(rates: List[float], timestamps: List[int]) -> List[Dict]:
    """
    Sweep entry threshold from 1 bps to 20 bps and show resulting metrics.
    Useful for finding optimal filter level.
    """
    results = []
    thresholds = [1, 2, 3, 5, 7, 10, 15, 20]
    for t_bps in thresholds:
        res = simulate_strategy_2_threshold(
            rates, timestamps,
            entry_thresh=t_bps / 10_000,
            exit_thresh=max(0.5, t_bps / 4) / 10_000,
        )
        results.append({
            "entry_thresh_bps":       t_bps,
            "annualized_return_pct":  res["annualized_return_pct"],
            "sharpe_ratio":           res["sharpe_ratio"],
            "max_drawdown_pct":       res["max_drawdown_pct"],
            "pct_time_deployed":      res["pct_time_deployed"],
            "n_entries":              res["n_entries"],
        })
    return results


# -----------------------------------------------------------------------------
# DRAWDOWN PERIOD ANALYSIS
# -----------------------------------------------------------------------------

def find_worst_drawdown_periods(
    equity: List[float],
    timestamps: List[int],
    top_n: int = 5,
) -> List[Dict]:
    """
    Identify the top-N worst drawdown periods (peak-to-trough segments).
    """
    periods = []
    peak    = equity[0]
    peak_i  = 0

    for i in range(1, len(equity)):
        if equity[i] > peak:
            peak   = equity[i]
            peak_i = i
        else:
            dd = (peak - equity[i]) / peak
            if dd > 0.001:  # filter trivial drawdowns
                periods.append({
                    "dd_pct":     dd * 100,
                    "start_idx":  peak_i,
                    "end_idx":    i,
                    "start_ts":   timestamps[min(peak_i, len(timestamps)-1)] if timestamps else 0,
                    "end_ts":     timestamps[min(i,      len(timestamps)-1)] if timestamps else 0,
                    "duration_days": (i - peak_i) / FUNDING_PERIODS_PER_DAY,
                })

    # Keep only unique non-overlapping worst periods
    periods.sort(key=lambda x: -x["dd_pct"])
    selected = []
    used_ranges = []
    for p in periods:
        overlap = any(
            not (p["end_idx"] < s or p["start_idx"] > e)
            for s, e in used_ranges
        )
        if not overlap:
            selected.append(p)
            used_ranges.append((p["start_idx"], p["end_idx"]))
        if len(selected) >= top_n:
            break

    return selected


# -----------------------------------------------------------------------------
# REPORT FORMATTING
# -----------------------------------------------------------------------------

def print_separator(char: str = "-", width: int = 65) -> None:
    print(char * width)


def print_header(title: str) -> None:
    print_separator("=")
    print(f"  {title}")
    print_separator("=")


def format_pct(val: float, decimals: int = 4) -> str:
    return f"{val:.{decimals}f}%"


def print_statistics(stats: Dict) -> None:
    print_header("FUNDING RATE STATISTICS")
    print(f"  Periods downloaded:      {stats['n_periods']:,}")
    print(f"  Periods per year:        {stats['periods_per_year']:,}  (3x daily)")
    print()
    print(f"  Mean rate / period:      {format_pct(stats['mean_per_period']*100, 5)}")
    print(f"  Median rate / period:    {format_pct(stats['median_per_period']*100, 5)}")
    print(f"  Std dev / period:        {format_pct(stats['std_per_period']*100, 5)}")
    print(f"  Min rate:                {format_pct(stats['min_rate']*100, 5)}")
    print(f"  Max rate:                {format_pct(stats['max_rate']*100, 5)}")
    print()
    print(f"  Percentiles (per-period rate):")
    print(f"    10th: {format_pct(stats['p10']*100, 5)}  |  "
          f"25th: {format_pct(stats['p25']*100, 5)}  |  "
          f"75th: {format_pct(stats['p75']*100, 5)}")
    print(f"    90th: {format_pct(stats['p90']*100, 5)}  |  "
          f"95th: {format_pct(stats['p95']*100, 5)}  |  "
          f"99th: {format_pct(stats['p99']*100, 5)}")
    print()
    print(f"  Annualized return (simple):    {format_pct(stats['ann_return_simple']*100, 2)}")
    print(f"  Annualized return (compound):  {format_pct(stats['ann_return_compound']*100, 2)}")
    print(f"  Annualized std dev:            {format_pct(stats['ann_std']*100, 2)}")
    print(f"  Annualized Sharpe (raw):       {stats['ann_return_compound'] / stats['ann_std']:.3f}"
          f"  (before costs, using zero risk-free)")
    print()
    print(f"  Pct of periods rate > 0:  {stats['pct_positive']*100:.1f}%  "
          f"(longs pay shorts)")
    print(f"  Pct of periods rate < 0:  {stats['pct_negative']*100:.1f}%  "
          f"(shorts pay longs)")
    print(f"  Auto-correlation (lag-1): {stats['acf_lag1']:.3f}")
    print()


def print_cost_analysis(costs: Dict) -> None:
    print_header("COST ANALYSIS - ROUND TRIP (RETAIL TAKER FEES)")
    print(f"  Spot taker fee:              {SPOT_TAKER_FEE*100:.2f}% each side")
    print(f"  Perp taker fee:              {PERP_TAKER_FEE*100:.3f}% each side")
    print(f"  Slippage (est):              {SLIPPAGE_BPS} bps each side")
    print()
    print(f"  Entry cost (bps):            {costs['entry_cost_bps']:.1f} bps")
    print(f"  Exit cost  (bps):            {costs['exit_cost_bps']:.1f} bps")
    print(f"  Total round-trip (bps):      {costs['round_trip_bps']:.1f} bps")
    print(f"  Total round-trip (%):        {costs['round_trip_pct']:.3f}%")
    print()
    print(f"  Capital ratio (1+margin):    {costs['capital_ratio']:.2f}x  "
          f"(spot $1 + perp {PERP_MARGIN_RATIO*100:.0f}% margin)")
    print(f"  Opportunity cost / period:   {costs['opp_cost_per_period']*10000:.3f} bps")
    print(f"  Break-even periods:          {costs['break_even_periods']:.0f}  "
          f"({costs['break_even_periods']/FUNDING_PERIODS_PER_DAY:.1f} days)")
    print()
    print(f"  NOTE: Using maker fees reduces RT cost to ~"
          f"{(SPOT_MAKER_FEE + PERP_MAKER_FEE)*2*10000:.0f} bps + slippage")
    print()


def print_strategy_result(res: Dict) -> None:
    strat_names = {
        "1_always_on":      "Strategy 1: Always-On Delta-Neutral Carry",
        "2_threshold":      "Strategy 2: Threshold-Filtered Entry",
        "3_cross_exchange": "Strategy 3: Cross-Exchange Arb (Simulated Spread)",
    }
    name = strat_names.get(res["strategy"], res["strategy"])
    print_header(name)

    if res["strategy"] == "2_threshold":
        print(f"  Entry threshold:      {res['entry_threshold_bps']:.1f} bps/period")
        print(f"  Exit threshold:       {res['exit_threshold_bps']:.1f} bps/period")
        print(f"  Time deployed:        {res['pct_time_deployed']:.1f}%")
        print(f"  Number of entries:    {res['n_entries']}")
        print()
    elif res["strategy"] == "3_cross_exchange":
        print(f"  Min spread required:  {res['min_spread_bps']:.1f} bps")
        print(f"  Number of entries:    {res['n_entries']}")
        print()

    print(f"  Total return:         {res['total_return_pct']:+.2f}%")
    print(f"  Annualized return:    {res['annualized_return_pct']:+.2f}%")
    print(f"  Sharpe ratio:         {res['sharpe_ratio']:.3f}")
    print(f"  Max drawdown:         {res['max_drawdown_pct']:.3f}%")
    print(f"  Max DD duration:      {res['dd_duration_days']:.1f} days")
    print(f"  Equity final:         {res['equity_final']:.5f}  (started at 1.0)")
    print()


def print_sensitivity(sensitivity: List[Dict]) -> None:
    print_header("SENSITIVITY ANALYSIS - Entry Threshold Sweep")
    header = (
        f"  {'Threshold':>12}  {'Ann Ret':>8}  {'Sharpe':>7}  "
        f"{'Max DD':>7}  {'Deployed':>9}  {'Entries':>8}"
    )
    print(header)
    print_separator("-", 68)
    for row in sensitivity:
        print(
            f"  {row['entry_thresh_bps']:>9.0f} bps"
            f"  {row['annualized_return_pct']:>+7.2f}%"
            f"  {row['sharpe_ratio']:>7.3f}"
            f"  {row['max_drawdown_pct']:>6.3f}%"
            f"  {row['pct_time_deployed']:>8.1f}%"
            f"  {row['n_entries']:>8,}"
        )
    print()


def print_drawdown_periods(periods: List[Dict], timestamps: List[int]) -> None:
    print_header("WORST DRAWDOWN PERIODS (Strategy 1: Always-On)")
    if not periods:
        print("  No significant drawdowns found.")
        return
    for i, p in enumerate(periods, 1):
        start_dt = (
            datetime.utcfromtimestamp(p["start_ts"] / 1000).strftime("%Y-%m-%d")
            if p["start_ts"] else "N/A"
        )
        end_dt = (
            datetime.utcfromtimestamp(p["end_ts"] / 1000).strftime("%Y-%m-%d")
            if p["end_ts"] else "N/A"
        )
        print(f"  #{i}  Drawdown: {p['dd_pct']:.3f}%  |  "
              f"{start_dt} -> {end_dt}  |  "
              f"Duration: {p['duration_days']:.1f} days")
    print()


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 65)
    print("  BTC Perpetual Futures Funding Rate Arbitrage Analysis")
    print(f"  Run date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 65 + "\n")

    # -- 1. Download data ------------------------------------------------------
    raw_records = download_funding_rates(SYMBOL, LOOKBACK_DAYS)

    if not raw_records:
        print("ERROR: No funding rate data available. Exiting.")
        sys.exit(1)

    rates      = [float(r["fundingRate"]) for r in raw_records]
    timestamps = [r["fundingTime"] for r in raw_records]
    dates_str  = [
        datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M")
        for ts in timestamps
    ]

    # Summarise date range
    if timestamps:
        t_start = datetime.utcfromtimestamp(timestamps[0]  / 1000).strftime("%Y-%m-%d")
        t_end   = datetime.utcfromtimestamp(timestamps[-1] / 1000).strftime("%Y-%m-%d")
        print(f"\n  Date range: {t_start} -> {t_end}  ({len(rates):,} periods)\n")

    # -- 2. Statistics ---------------------------------------------------------
    stats = compute_statistics(rates)
    print_statistics(stats)

    # -- 3. Cost analysis ------------------------------------------------------
    costs = round_trip_cost_bps(holding_periods=len(rates))
    print_cost_analysis(costs)

    # -- 4. Strategy simulations -----------------------------------------------
    print_separator()
    print("  RUNNING STRATEGY SIMULATIONS ...")
    print_separator()
    print()

    res1 = simulate_strategy_1_always_on(rates, timestamps)
    res2 = simulate_strategy_2_threshold(rates, timestamps)

    # Cross-exchange: simulate B as A with added noise + occasional divergence
    import random
    random.seed(99)
    rates_b = []
    for r in rates:
        noise = random.gauss(0, 0.0001)
        spike = random.gauss(0, 0.0005) if random.random() < 0.05 else 0.0
        rates_b.append(r + noise + spike)
    res3 = simulate_strategy_3_cross_exchange(rates, rates_b, timestamps)

    print_strategy_result(res1)
    print_strategy_result(res2)
    print_strategy_result(res3)

    # -- 5. Threshold sweep ----------------------------------------------------
    sensitivity = sensitivity_analysis(rates, timestamps)
    print_sensitivity(sensitivity)

    # -- 6. Worst drawdown periods ---------------------------------------------
    dd_periods = find_worst_drawdown_periods(
        res1["equity_curve"], timestamps, top_n=5
    )
    print_drawdown_periods(dd_periods, timestamps)

    # -- 7. Summary table ------------------------------------------------------
    print_header("STRATEGY COMPARISON SUMMARY")
    print(f"  {'Strategy':<35}  {'Ann Return':>10}  {'Sharpe':>8}  {'Max DD':>8}")
    print_separator("-", 68)
    for res in [res1, res2, res3]:
        label = {
            "1_always_on":       "Always-On Carry",
            "2_threshold":       f"Threshold ({ENTRY_THRESHOLD_BPS:.0f} bps)",
            "3_cross_exchange":  "Cross-Exchange Arb",
        }.get(res["strategy"], res["strategy"])
        print(
            f"  {label:<35}  "
            f"{res['annualized_return_pct']:>+9.2f}%  "
            f"{res['sharpe_ratio']:>8.3f}  "
            f"{res['max_drawdown_pct']:>7.3f}%"
        )
    print()

    # -- 8. Key observations ---------------------------------------------------
    print_header("KEY OBSERVATIONS & PRACTICAL NOTES")
    mean_bps = stats["mean_per_period"] * 10_000
    ann_raw  = stats["ann_return_compound"] * 100
    rt_bps   = costs["round_trip_bps"]

    observations = [
        f"Raw mean funding: {mean_bps:.2f} bps/period "
        f"({stats['ann_return_compound']*100:.1f}% annualized, gross)",
        f"Round-trip transaction cost: {rt_bps:.0f} bps "
        f"(~{rt_bps/(mean_bps*3):.1f}x one day of avg funding)",
        f"Positive funding {stats['pct_positive']*100:.0f}% of periods - "
        f"{'strong' if stats['pct_positive'] > 0.7 else 'moderate'} long bias in market",
        f"Autocorrelation lag-1: {stats['acf_lag1']:.3f} - "
        f"{'persistence helps threshold strategy' if stats['acf_lag1'] > 0.2 else 'limited persistence'}",
        "Critical risk: liquidation cascade -> perp basis widens -> funding spikes negative",
        "Capital efficiency: use cross-margin + BNB discount to reduce fees by ~25%",
        "Optimal execution: use limit orders on perp side (maker fees ~0.02% vs 0.05%)",
        f"Opportunity cost dominates at low rates: break-even at ~{costs['break_even_periods']:.0f} periods "
        f"({costs['break_even_periods']/3:.0f} days) per round-trip",
        "Strategy 2 benefit: avoids negative-funding periods by staying out of market",
        "Cross-exchange arb requires sub-second execution and same settlement window",
    ]

    for i, obs in enumerate(observations, 1):
        print(f"  {i:2d}. {obs}")
    print()

    # -- 9. Save results JSON --------------------------------------------------
    import os
    _script_path = globals().get("__file__", __import__("sys").argv[0] if __import__("sys").argv else ".")
    output_dir  = os.path.dirname(os.path.abspath(_script_path))
    output_path = os.path.join(output_dir, "data", "btc_funding_arb_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    save_data = {
        "run_date":   datetime.utcnow().isoformat(),
        "symbol":     SYMBOL,
        "n_periods":  len(rates),
        "date_range": {"start": dates_str[0] if dates_str else "", "end": dates_str[-1] if dates_str else ""},
        "statistics": {k: v for k, v in stats.items()
                       if k not in ("periods_per_year",)},
        "costs":      {k: v for k, v in costs.items()},
        "strategies": {
            "always_on": {
                k: v for k, v in res1.items()
                if k not in ("equity_curve", "pnl_per_period")
            },
            "threshold": {
                k: v for k, v in res2.items()
                if k not in ("equity_curve", "pnl_per_period")
            },
            "cross_exchange": {
                k: v for k, v in res3.items()
                if k not in ("equity_curve",)
            },
        },
        "sensitivity": sensitivity,
    }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"  Results saved to: {output_path}")
    print()

    print_separator("=")
    print("  Analysis complete.")
    print_separator("=")
    print()


if __name__ == "__main__":
    main()
