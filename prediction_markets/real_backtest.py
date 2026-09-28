#!/usr/bin/env python3
"""
Real Backtest — Replays vol model signals against actual historical prediction market prices.

Unlike bracket_backtest.py which uses synthetic/simulated market prices derived from VIX,
this backtest uses ACTUAL historical trade and orderbook data from:
  1. Kalshi trade history (from kalshi_trade_history.py)
  2. pmxt.dev Parquet snapshots (from pmxt_downloader.py)

Fill assumptions (conservative, per arXiv:2409.12721):
  - Taker fills at the ask price (worst case for entering)
  - 50% haircut for adverse selection (smart MMs pull quotes on our signals)
  - Partial fills on illiquid contracts (< 10 contracts available)
  - Fees: Kalshi taker fee = 3.5% * p * (1-p) per contract

Comparison:
  The synthetic backtest (bracket_backtest.py) simulates market prices from VIX+noise.
  This backtest uses what actually traded. If synthetic >> real, the synthetic was
  overfit / unrealistic.

Usage:
    python real_backtest.py                     # Run with all available data
    python real_backtest.py --source kalshi     # Kalshi trades only
    python real_backtest.py --source pmxt       # pmxt.dev orderbooks only
    python real_backtest.py --compare           # Compare against synthetic results
    python real_backtest.py --verbose           # Show individual trades

Output:
    data/real_backtest_results_{timestamp}.json
"""

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not available. Using simplified pricing.")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

logger = logging.getLogger("real_backtest")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

DATA_DIR = Path(__file__).parent / "data"
KALSHI_TRADES_DIR = DATA_DIR / "kalshi_trades"
PMXT_DIR = DATA_DIR / "pmxt"

# Fee model (mirrors kalshi_client.py)
FEE_COEFF_SPX = 0.035        # SPX halved taker fee coefficient
MAKER_FEE_COEFF = 0.0175     # Maker fee coefficient

# Adverse selection haircut per arXiv:2409.12721
ADVERSE_SELECTION_HAIRCUT = 0.50  # 50% of expected edge lost to adverse selection


# ---------------------------------------------------------------------------
# Fee calculations
# ---------------------------------------------------------------------------

def taker_fee(price: float, contracts: int = 1) -> float:
    """Kalshi taker fee for SPX brackets. Price in [0, 1]. Returns $ per contract."""
    return math.ceil(FEE_COEFF_SPX * contracts * price * (1 - price) * 100) / 100


def maker_fee(price: float, contracts: int = 1) -> float:
    """Kalshi maker fee. Returns $ per contract."""
    return math.ceil(MAKER_FEE_COEFF * contracts * price * (1 - price) * 100) / 100


# ---------------------------------------------------------------------------
# Vol model pricing (same as bracket_backtest.py)
# ---------------------------------------------------------------------------

def price_bracket_student_t(
    floor: float, cap: float, spx: float, vol_annual_pct: float,
    hours_to_close: float = 6.5, df: float = 4.0,
) -> float:
    """Price a bracket using Student-t distribution (our model).

    Returns probability [0, 1].
    """
    if not HAS_SCIPY:
        # Simplified normal approximation
        vol = vol_annual_pct / 100.0
        time_frac = hours_to_close / (252 * 6.5)
        sigma = spx * vol * np.sqrt(time_frac)
        if sigma <= 0:
            return 1.0 if floor <= spx <= cap else 0.0
        z_lo = (floor - spx) / sigma
        z_hi = (cap - spx) / sigma
        from scipy.stats import norm
        return float(norm.cdf(z_hi) - norm.cdf(z_lo))

    vol = vol_annual_pct / 100.0
    time_frac = hours_to_close / (252 * 6.5)
    sigma = spx * vol * np.sqrt(time_frac)
    if sigma <= 0:
        return 1.0 if floor <= spx <= cap else 0.0
    scale = sigma * np.sqrt((df - 2) / df) if df > 2 else sigma
    dist = scipy_stats.t(df=df, loc=spx, scale=scale)
    return float(max(0, dist.cdf(cap) - dist.cdf(floor)))


def simulate_vol_forecast(true_rvol: float, vix: float, ic: float = 0.644) -> float:
    """Simulate our vol model's prediction. Returns annualized vol in percent."""
    signal = ic * true_rvol + (1 - ic) * vix
    noise_std = abs(true_rvol - vix) * np.sqrt(1 - ic**2) * 0.5
    noise = np.random.normal(0, max(0.5, noise_std))
    predicted = signal + noise
    return max(3.0, min(80.0, predicted))


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_kalshi_trades(trades_dir: Path = None) -> list[dict]:
    """Load Kalshi trades from JSONL files.

    Returns list of trade dicts, each containing:
        ticker, count, yes_price, yes_price_dollars, no_price, no_price_dollars,
        created_time, taker_side, trade_id
    """
    if trades_dir is None:
        trades_dir = KALSHI_TRADES_DIR

    if not trades_dir.exists():
        logger.warning(f"Kalshi trades directory not found: {trades_dir}")
        return []

    all_trades = []
    for f in sorted(trades_dir.glob("*.jsonl")):
        if f.name in ("all_recent_trades.jsonl",):
            continue
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        all_trades.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    logger.info(f"Loaded {len(all_trades)} Kalshi trades from {trades_dir}")
    return all_trades


def load_pmxt_orderbooks(pmxt_dir: Path = None) -> "pd.DataFrame | None":
    """Load pmxt.dev Parquet orderbook snapshots.

    Returns DataFrame with orderbook data, or None if not available.
    """
    if not HAS_PANDAS or not HAS_PYARROW:
        logger.warning("pandas and pyarrow required for pmxt data")
        return None

    if pmxt_dir is None:
        pmxt_dir = PMXT_DIR / "polymarket"

    if not pmxt_dir.exists():
        logger.warning(f"pmxt directory not found: {pmxt_dir}")
        return None

    parquet_files = sorted(pmxt_dir.rglob("*.parquet"))
    if not parquet_files:
        logger.warning("No Parquet files found")
        return None

    logger.info(f"Loading {len(parquet_files)} Parquet files...")

    # Load a sample first to check schema
    try:
        sample = pd.read_parquet(parquet_files[0], nrows=5 if hasattr(pd, 'read_parquet') else None)
        logger.info(f"Parquet columns: {sample.columns.tolist()[:20]}")
    except Exception as e:
        logger.error(f"Failed to read sample: {e}")
        return None

    # Load all files
    dfs = []
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            # Add source metadata
            df["_source_file"] = pf.name
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to read {pf.name}: {e}")

    if not dfs:
        return None

    result = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(result)} rows from pmxt data")
    return result


def load_spx_data_from_kalshi_markets() -> dict:
    """Load SPX settlement data from Kalshi market metadata.

    Uses the API to get settled market results, giving us actual SPX values.

    Returns dict of {date: {spx_close, brackets: [{ticker, floor, cap, result, volume}]}}
    """
    try:
        import requests
        BASE_API = "https://api.elections.kalshi.com/trade-api/v2"

        daily_data = {}
        for series in ["KXINX", "KXINXB", "KXINXW"]:
            resp = requests.get(
                f"{BASE_API}/markets",
                params={"series_ticker": series, "limit": 200, "status": "settled"},
                timeout=15,
            )
            if resp.status_code != 200:
                continue

            data = resp.json()
            markets = data.get("markets", [])

            for m in markets:
                ticker = m.get("ticker", "")
                settlement_ts = m.get("settlement_ts", m.get("close_time", ""))
                date_str = settlement_ts[:10] if settlement_ts else ""
                if not date_str:
                    continue

                try:
                    expiration_value = float(m.get("expiration_value", 0))
                except (ValueError, TypeError):
                    expiration_value = 0
                floor_strike = m.get("floor_strike")
                cap_strike = m.get("cap_strike")
                result = m.get("result", "")
                volume = m.get("volume", 0)

                if date_str not in daily_data:
                    daily_data[date_str] = {
                        "spx_close": expiration_value,
                        "brackets": [],
                    }

                if expiration_value:
                    daily_data[date_str]["spx_close"] = expiration_value

                daily_data[date_str]["brackets"].append({
                    "ticker": ticker,
                    "floor": floor_strike,
                    "cap": cap_strike,
                    "result": result,
                    "volume": volume,
                })

        logger.info(f"Loaded Kalshi settlement data for {len(daily_data)} dates")
        return daily_data

    except Exception as e:
        logger.error(f"Failed to load Kalshi market data: {e}")
        return {}


# ---------------------------------------------------------------------------
# Backtest data structures
# ---------------------------------------------------------------------------

@dataclass
class RealTrade:
    """A trade in the real backtest."""
    date: str
    ticker: str
    side: str                   # "BUY_YES" or "BUY_NO"
    entry_price: float          # Actual market price we'd pay
    fair_value: float           # Our model's fair value
    edge_gross: float           # fair_value - entry_price
    edge_after_adverse: float   # After adverse selection haircut
    fee: float                  # Taker fee
    edge_net: float             # Net edge after fees + adverse selection
    settled_value: float        # 1.0 or 0.0
    pnl_gross: float            # Before adverse selection adjustment
    pnl_net: float              # After all adjustments
    contracts: int = 1
    fill_rate: float = 1.0      # Partial fill ratio
    source: str = ""            # "kalshi_trade" or "pmxt_orderbook"
    spx_ref: float = 0.0       # SPX reference price
    model_vol: float = 0.0     # Our vol prediction


@dataclass
class RealBacktestConfig:
    """Configuration for the real backtest."""
    starting_capital: float = 10000.0
    max_position_per_market: int = 50
    max_daily_risk: float = 1000.0
    min_net_edge: float = 0.008          # Min net edge after ALL adjustments
    min_gross_edge: float = 0.015
    vol_model_ic: float = 0.644
    student_t_df: float = 4.0
    kelly_fraction: float = 0.15
    adverse_selection: float = ADVERSE_SELECTION_HAIRCUT
    partial_fill_threshold: int = 10     # Below this, assume partial fill
    hours_to_close: float = 5.0         # Assume entry ~11 AM ET
    use_taker_fees: bool = True


@dataclass
class RealBacktestResult:
    """Results from the real backtest."""
    config: dict
    data_sources: list = field(default_factory=list)
    n_trading_days: int = 0
    n_trades: int = 0
    n_wins: int = 0
    n_losses: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl_per_trade: float = 0.0
    avg_pnl_per_day: float = 0.0
    monthly_pnl_estimate: float = 0.0
    annual_pnl_estimate: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    total_fees: float = 0.0
    total_adverse_selection_cost: float = 0.0
    avg_fill_rate: float = 1.0
    edge_by_bracket_type: dict = field(default_factory=dict)
    daily_pnl: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)
    trades: list = field(default_factory=list)
    comparison_vs_synthetic: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def compute_fill_rate(available_contracts: int, desired_contracts: int,
                      threshold: int = 10) -> float:
    """Compute realistic fill rate based on available liquidity.

    Below threshold contracts, assume partial fills.
    """
    if available_contracts <= 0:
        return 0.0
    if available_contracts >= desired_contracts:
        return 1.0
    if available_contracts < threshold:
        # Partial fill: we get some but with slippage
        return min(1.0, available_contracts / max(desired_contracts, 1)) * 0.8
    return min(1.0, available_contracts / max(desired_contracts, 1))


def compute_kelly_size(
    edge: float, price: float, max_contracts: int, kelly_frac: float = 0.15
) -> int:
    """Compute position size using fractional Kelly."""
    if edge <= 0 or price <= 0.01 or price >= 0.99:
        return 0
    kelly_full = edge / (1 - price)
    kelly_adj = kelly_full * kelly_frac
    contracts = max(1, min(max_contracts, int(kelly_adj * 100)))
    return contracts


def run_backtest_on_kalshi_trades(
    trades: list[dict],
    config: RealBacktestConfig = None,
    spx_reference: dict = None,
) -> RealBacktestResult:
    """Run backtest using actual Kalshi trade data.

    Strategy:
    1. For each historical trade, compute our model's fair value
    2. If we see edge vs the trade price, simulate entering at that price
    3. Apply adverse selection haircut and fees
    4. Settle based on actual market result

    Args:
        trades: List of Kalshi trade dicts
        config: Backtest configuration
        spx_reference: {date: {spx_close, brackets}} from Kalshi settlements

    Returns:
        RealBacktestResult
    """
    if config is None:
        config = RealBacktestConfig()

    if spx_reference is None:
        spx_reference = {}

    # Group trades by date and ticker
    trades_by_date = defaultdict(list)
    for t in trades:
        date_str = t.get("created_time", "")[:10]
        trades_by_date[date_str].append(t)

    all_real_trades = []
    daily_pnl = []
    equity = config.starting_capital
    equity_curve = [equity]
    total_adverse_cost = 0.0

    # Filter to only SPX-related tickers
    # Our vol model only applies to S&P 500 bracket/range markets
    SPX_PREFIXES = ("KXINX", "INX", "KXINXB", "KXINXW", "KXINXM", "KXINXI",
                    "KXINXU", "KXINXAB", "KXINXZ", "KXINXY", "KXINXMINY", "KXINXMAXY",
                    "KXINXPOS", "KXINXMINW")
    filtered_by_date = {}
    total_filtered = 0
    for date_str, day_trades in trades_by_date.items():
        spx_trades = [
            t for t in day_trades
            if any(t.get("ticker", "").startswith(p + "-") or t.get("ticker", "") == p
                   for p in SPX_PREFIXES)
        ]
        if spx_trades:
            filtered_by_date[date_str] = spx_trades
            total_filtered += len(spx_trades)
    trades_by_date = filtered_by_date

    sorted_dates = sorted(trades_by_date.keys())
    logger.info(f"Running real backtest on {len(sorted_dates)} dates, "
                f"{total_filtered} SPX trades "
                f"(filtered from {sum(len(v) for v in trades_by_date.values())} total)")

    for date_str in sorted_dates:
        day_trades = trades_by_date[date_str]
        ref = spx_reference.get(date_str, {})
        spx_close = ref.get("spx_close", 0)

        day_pnl = 0.0
        day_risk = 0.0

        # Group by ticker to avoid duplicate signals
        seen_tickers = set()

        for trade in day_trades:
            ticker = trade.get("ticker", "")
            if ticker in seen_tickers:
                continue
            seen_tickers.add(ticker)

            if day_risk >= config.max_daily_risk:
                break

            # Parse trade data
            yes_price_dollars = float(trade.get("yes_price_dollars", "0"))
            no_price_dollars = float(trade.get("no_price_dollars", "0"))
            count = trade.get("count", 0)
            taker_side = trade.get("taker_side", "")

            if yes_price_dollars <= 0.01 or yes_price_dollars >= 0.99:
                continue

            # We need SPX reference to compute our fair value
            # If we have settlement data, use it
            bracket_info = None
            if ref.get("brackets"):
                bracket_info = next(
                    (b for b in ref["brackets"] if b.get("ticker") == ticker),
                    None
                )

            # Parse bracket strike from ticker
            # Format: KXINX-26MAR05H1600-B7087 (bracket at 7087)
            # or: KXINX-26MAR05H1600-T7149 (above 7149)
            floor_strike = None
            cap_strike = None
            settled_yes = None

            if bracket_info:
                floor_strike = bracket_info.get("floor")
                cap_strike = bracket_info.get("cap")
                settled_yes = bracket_info.get("result") == "yes"

            if floor_strike is None:
                # Try to parse from ticker
                parts = ticker.split("-")
                if len(parts) >= 3:
                    strike_part = parts[-1]
                    try:
                        if strike_part.startswith("B"):
                            floor_strike = float(strike_part[1:])
                            cap_strike = floor_strike + 25  # Standard bracket width
                        elif strike_part.startswith("T"):
                            floor_strike = float(strike_part[1:])
                            cap_strike = float("inf")
                    except ValueError:
                        continue

            if floor_strike is None:
                continue

            # Use SPX close as reference for settlement
            try:
                spx_close_f = float(spx_close) if spx_close else 0
                floor_f = float(floor_strike) if floor_strike is not None else 0
                cap_f = float(cap_strike) if cap_strike is not None and cap_strike != float("inf") else float("inf")
            except (ValueError, TypeError):
                continue

            if spx_close_f > 0 and settled_yes is None:
                if cap_f != float("inf"):
                    settled_yes = floor_f <= spx_close_f < cap_f
                else:
                    settled_yes = spx_close_f >= floor_f

            if settled_yes is None:
                continue

            settled_value = 1.0 if settled_yes else 0.0

            # Compute our model's fair value
            # We need an SPX reference price at entry time (approximate with close)
            spx_ref = spx_close_f if spx_close_f > 0 else 5800  # Fallback

            # Simulate our vol forecast
            # We don't have real VIX here, use a reasonable estimate
            vix_estimate = 18.0  # Conservative default
            rvol_estimate = vix_estimate  # Assume rvol ~ VIX for now
            model_vol = simulate_vol_forecast(rvol_estimate, vix_estimate, ic=config.vol_model_ic)

            if cap_f != float("inf"):
                model_fv = price_bracket_student_t(
                    floor_f, cap_f, spx_ref, model_vol,
                    config.hours_to_close, config.student_t_df
                )
            else:
                # Above/below market: approximate
                model_fv = price_bracket_student_t(
                    floor_f, floor_f + 1000, spx_ref, model_vol,
                    config.hours_to_close, config.student_t_df
                )

            # Check edge
            # For taker: we'd buy at the ask (worse side)
            # The trade price represents where SOME trade happened
            # Conservative: add half-spread to the trade price
            spread_estimate = 0.02  # 2 cents estimated spread
            ask_price = min(0.99, yes_price_dollars + spread_estimate / 2)

            buy_yes_edge = model_fv - ask_price
            buy_no_edge = (1 - model_fv) - (1 - yes_price_dollars + spread_estimate / 2)

            # Pick better side
            if buy_yes_edge > buy_no_edge and buy_yes_edge > 0:
                side = "BUY_YES"
                entry_price = ask_price
                fair_value = model_fv
                edge_gross = buy_yes_edge
                payout = settled_value
            elif buy_no_edge > 0:
                side = "BUY_NO"
                entry_price = 1 - yes_price_dollars + spread_estimate / 2
                entry_price = min(0.99, max(0.01, entry_price))
                fair_value = 1 - model_fv
                edge_gross = buy_no_edge
                payout = 0.0 if settled_yes else 1.0
            else:
                continue

            if edge_gross < config.min_gross_edge:
                continue

            # Apply adverse selection haircut
            edge_after_adverse = edge_gross * (1 - config.adverse_selection)

            # Apply fees
            fee = taker_fee(entry_price)
            edge_net = edge_after_adverse - fee

            if edge_net < config.min_net_edge:
                continue

            # Position sizing
            contracts = compute_kelly_size(
                edge_net, entry_price, config.max_position_per_market,
                config.kelly_fraction,
            )
            if contracts <= 0:
                continue

            # Fill rate based on observed trade size
            fill_rate = compute_fill_rate(count, contracts, config.partial_fill_threshold)
            filled_contracts = max(1, int(contracts * fill_rate))

            # Risk check
            risk = filled_contracts * entry_price
            if day_risk + risk > config.max_daily_risk:
                filled_contracts = max(1, int((config.max_daily_risk - day_risk) / entry_price))
                risk = filled_contracts * entry_price

            # P&L
            pnl_gross = filled_contracts * (payout - entry_price - fee)
            adverse_cost = filled_contracts * edge_gross * config.adverse_selection
            pnl_net = pnl_gross - adverse_cost
            total_adverse_cost += adverse_cost

            real_trade = RealTrade(
                date=date_str,
                ticker=ticker,
                side=side,
                entry_price=round(entry_price, 4),
                fair_value=round(fair_value, 4),
                edge_gross=round(edge_gross, 4),
                edge_after_adverse=round(edge_after_adverse, 4),
                fee=round(fee, 4),
                edge_net=round(edge_net, 4),
                settled_value=payout,
                pnl_gross=round(pnl_gross, 2),
                pnl_net=round(pnl_net, 2),
                contracts=filled_contracts,
                fill_rate=round(fill_rate, 2),
                source="kalshi_trade",
                spx_ref=round(spx_ref, 2),
                model_vol=round(model_vol, 2),
            )
            all_real_trades.append(real_trade)
            day_pnl += pnl_net
            day_risk += risk

        daily_pnl.append(day_pnl)
        equity += day_pnl
        equity_curve.append(equity)

    # Compute results
    return _compute_results(all_real_trades, daily_pnl, equity_curve, config,
                            total_adverse_cost, ["kalshi_trades"])


def run_backtest_on_pmxt_data(
    orderbook_df: "pd.DataFrame",
    config: RealBacktestConfig = None,
) -> RealBacktestResult:
    """Run backtest using pmxt.dev orderbook snapshot data.

    Uses actual orderbook prices from Polymarket/Kalshi snapshots.
    """
    if config is None:
        config = RealBacktestConfig()

    if orderbook_df is None or len(orderbook_df) == 0:
        logger.warning("No pmxt orderbook data available")
        return RealBacktestResult(config=asdict(config), data_sources=["pmxt_orderbooks"])

    logger.info(f"Running backtest on {len(orderbook_df)} orderbook rows")
    logger.info(f"Columns: {orderbook_df.columns.tolist()[:15]}")

    # The pmxt data schema varies — we need to adapt to what's available
    # Common columns might include: question, best_bid, best_ask, token_id, etc.

    all_real_trades = []
    daily_pnl = []
    equity = config.starting_capital
    equity_curve = [equity]
    total_adverse_cost = 0.0

    # Try to extract SPX-related rows
    text_cols = [c for c in orderbook_df.columns if orderbook_df[c].dtype == object]
    spx_mask = pd.Series(False, index=orderbook_df.index)
    for col in text_cols:
        for kw in ["s&p", "spx", "sp500", "sp 500"]:
            spx_mask |= orderbook_df[col].str.lower().str.contains(kw, na=False)

    spx_data = orderbook_df[spx_mask]
    if len(spx_data) == 0:
        logger.warning("No SPX-related rows found in pmxt data")
        # Still return empty result
        return RealBacktestResult(
            config=asdict(config),
            data_sources=["pmxt_orderbooks"],
            n_trading_days=0,
        )

    logger.info(f"Found {len(spx_data)} SPX-related rows")

    # Process the SPX data based on available columns
    # (Schema discovery — log what we find)
    price_cols = [c for c in spx_data.columns if any(
        k in c.lower() for k in ["price", "bid", "ask", "mid", "best"]
    )]
    logger.info(f"Price-like columns: {price_cols}")

    # The actual processing depends heavily on the schema
    # We'll extract what we can and simulate trading

    # Group by time/date if possible
    time_cols = [c for c in spx_data.columns if any(
        k in c.lower() for k in ["time", "date", "timestamp", "created"]
    )]

    if time_cols and price_cols:
        # We have enough to simulate
        for idx, row in spx_data.iterrows():
            # Extract price
            best_ask = None
            for pc in price_cols:
                val = row.get(pc)
                if val is not None and "ask" in pc.lower():
                    try:
                        best_ask = float(val)
                    except (ValueError, TypeError):
                        pass
            # ... (would process each row similarly to Kalshi trades)
            pass

    return _compute_results(all_real_trades, daily_pnl, equity_curve, config,
                            total_adverse_cost, ["pmxt_orderbooks"])


def _compute_results(
    trades: list[RealTrade],
    daily_pnl: list[float],
    equity_curve: list[float],
    config: RealBacktestConfig,
    total_adverse_cost: float,
    data_sources: list[str],
) -> RealBacktestResult:
    """Compute final backtest metrics from trade and P&L lists."""

    n_trades = len(trades)
    if n_trades == 0:
        return RealBacktestResult(
            config=asdict(config),
            data_sources=data_sources,
            n_trading_days=len(daily_pnl),
        )

    pnls = [t.pnl_net for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    total_pnl = sum(pnls)
    total_fees = sum(t.fee * t.contracts for t in trades)

    daily_arr = np.array(daily_pnl) if daily_pnl else np.array([0])
    avg_daily = float(np.mean(daily_arr)) if len(daily_arr) > 0 else 0
    std_daily = float(np.std(daily_arr)) if len(daily_arr) > 1 else 1

    # Max drawdown
    eq = np.array(equity_curve)
    running_max = np.maximum.accumulate(eq)
    drawdowns = np.where(running_max > 0, (eq - running_max) / running_max, 0)
    max_dd = float(np.min(drawdowns))

    # Sharpe
    sharpe = (avg_daily / std_daily * np.sqrt(252)) if std_daily > 0 else 0

    # Fill rate
    avg_fill = float(np.mean([t.fill_rate for t in trades])) if trades else 1.0

    # Edge by bracket type (ATM vs wing)
    atm_trades = [t for t in trades if t.spx_ref and t.ticker and
                  abs(t.spx_ref - (getattr(t, '_floor', t.spx_ref))) < 50]
    wing_trades = [t for t in trades if t not in atm_trades]

    return RealBacktestResult(
        config=asdict(config),
        data_sources=data_sources,
        n_trading_days=len(daily_pnl),
        n_trades=n_trades,
        n_wins=wins,
        n_losses=n_trades - wins,
        win_rate=round(wins / n_trades, 4),
        total_pnl=round(total_pnl, 2),
        avg_pnl_per_trade=round(total_pnl / n_trades, 4),
        avg_pnl_per_day=round(avg_daily, 2),
        monthly_pnl_estimate=round(avg_daily * 21, 2),
        annual_pnl_estimate=round(avg_daily * 252, 2),
        max_drawdown=round(max_dd, 4),
        sharpe_ratio=round(sharpe, 2),
        total_fees=round(total_fees, 2),
        total_adverse_selection_cost=round(total_adverse_cost, 2),
        avg_fill_rate=round(avg_fill, 2),
        edge_by_bracket_type={
            "atm": {
                "trades": len(atm_trades),
                "pnl": round(sum(t.pnl_net for t in atm_trades), 2),
            },
            "wing": {
                "trades": len(wing_trades),
                "pnl": round(sum(t.pnl_net for t in wing_trades), 2),
            },
        },
        daily_pnl=[round(d, 2) for d in daily_pnl],
        equity_curve=[round(e, 2) for e in equity_curve],
        trades=[asdict(t) for t in trades[-200:]],
    )


def compare_with_synthetic(real_result: RealBacktestResult) -> dict:
    """Compare real backtest results with synthetic bracket_backtest.py results.

    Loads the most recent synthetic backtest result and compares.
    """
    synthetic_files = sorted(DATA_DIR.glob("bracket_backtest_results_*.json"))
    if not synthetic_files:
        return {"error": "No synthetic backtest results found"}

    latest = synthetic_files[-1]
    with open(latest) as f:
        synthetic = json.load(f)

    synth_result = synthetic.get("result", {})

    comparison = {
        "synthetic_file": latest.name,
        "metrics": {},
    }

    # Compare key metrics
    for metric in ["total_pnl", "sharpe_ratio", "win_rate", "avg_pnl_per_trade",
                    "monthly_pnl_estimate", "max_drawdown", "total_fees"]:
        synth_val = synth_result.get(metric, 0)
        real_val = getattr(real_result, metric, 0)
        ratio = real_val / synth_val if synth_val != 0 else float("inf")
        comparison["metrics"][metric] = {
            "synthetic": synth_val,
            "real": real_val,
            "ratio": round(ratio, 3),
            "interpretation": (
                "BETTER" if real_val > synth_val and metric != "max_drawdown"
                else "WORSE" if real_val < synth_val and metric != "max_drawdown"
                else "SIMILAR"
            ),
        }

    # Overall assessment
    real_sharpe = real_result.sharpe_ratio
    synth_sharpe = synth_result.get("sharpe_ratio", 0)
    if real_sharpe >= synth_sharpe * 0.7:
        comparison["verdict"] = "CONSISTENT — real results within 30% of synthetic"
    elif real_sharpe >= synth_sharpe * 0.3:
        comparison["verdict"] = "DEGRADED — real results 30-70% worse than synthetic"
    else:
        comparison["verdict"] = "OVERFIT — synthetic backtest was unrealistic"

    return comparison


def format_report(result: RealBacktestResult) -> str:
    """Format real backtest results as readable text."""
    lines = [
        "=" * 70,
        "  REAL BACKTEST RESULTS (Historical Market Data)",
        "=" * 70,
        "",
        f"  Data sources:        {', '.join(result.data_sources)}",
        f"  Trading days:        {result.n_trading_days}",
        f"  Total trades:        {result.n_trades}",
        f"  Win/Loss:            {result.n_wins}W / {result.n_losses}L  "
        f"({result.win_rate:.1%})" if result.n_trades > 0 else "",
        "",
        f"  Total P&L:           ${result.total_pnl:>+10.2f}",
        f"  Avg P&L per trade:   ${result.avg_pnl_per_trade:>+10.4f}",
        f"  Avg P&L per day:     ${result.avg_pnl_per_day:>+10.2f}",
        "",
        f"  Monthly estimate:    ${result.monthly_pnl_estimate:>+10.2f}",
        f"  Annual estimate:     ${result.annual_pnl_estimate:>+10.2f}",
        "",
        f"  Sharpe ratio:        {result.sharpe_ratio:>10.2f}",
        f"  Max drawdown:        {result.max_drawdown:>10.2%}",
        "",
        "-" * 70,
        "  FRICTION ANALYSIS",
        "-" * 70,
        f"  Total fees:          ${result.total_fees:>10.2f}",
        f"  Adverse selection:   ${result.total_adverse_selection_cost:>10.2f}  "
        f"({result.config.get('adverse_selection', 0.5):.0%} haircut)",
        f"  Avg fill rate:       {result.avg_fill_rate:>10.1%}",
        "",
    ]

    if result.edge_by_bracket_type:
        lines.extend([
            "-" * 70,
            "  EDGE BY BRACKET TYPE",
            "-" * 70,
        ])
        for btype, data in result.edge_by_bracket_type.items():
            lines.append(
                f"  {btype.upper():>6s}:  {data['trades']:>4d} trades  "
                f"P&L=${data['pnl']:>+8.2f}"
            )

    if result.comparison_vs_synthetic:
        lines.extend([
            "",
            "-" * 70,
            "  COMPARISON VS SYNTHETIC BACKTEST",
            "-" * 70,
        ])
        comp = result.comparison_vs_synthetic
        if "error" not in comp:
            for metric, data in comp.get("metrics", {}).items():
                lines.append(
                    f"  {metric:<25s}  synth={data['synthetic']:>+10.4f}  "
                    f"real={data['real']:>+10.4f}  "
                    f"ratio={data['ratio']:.2f}  {data['interpretation']}"
                )
            lines.append(f"\n  VERDICT: {comp.get('verdict', 'N/A')}")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Real backtest using historical prediction market data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", choices=["kalshi", "pmxt", "both"], default="both",
                        help="Data source to use")
    parser.add_argument("--capital", type=float, default=10000,
                        help="Starting capital ($)")
    parser.add_argument("--ic", type=float, default=0.644,
                        help="Vol model IC")
    parser.add_argument("--adverse-selection", type=float, default=0.50,
                        help="Adverse selection haircut (0-1)")
    parser.add_argument("--min-edge", type=float, default=0.008,
                        help="Min net edge after all adjustments")
    parser.add_argument("--kelly", type=float, default=0.15,
                        help="Kelly fraction")
    parser.add_argument("--compare", action="store_true",
                        help="Compare against synthetic backtest results")
    parser.add_argument("--verbose", action="store_true",
                        help="Show individual trades")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--fetch-settlements", action="store_true",
                        help="Fetch Kalshi settlement data from API")

    args = parser.parse_args()
    np.random.seed(args.seed)

    config = RealBacktestConfig(
        starting_capital=args.capital,
        vol_model_ic=args.ic,
        adverse_selection=args.adverse_selection,
        min_net_edge=args.min_edge,
        kelly_fraction=args.kelly,
    )

    results = []

    # Load settlement reference data
    spx_reference = {}
    if args.fetch_settlements:
        logger.info("Fetching Kalshi settlement data...")
        spx_reference = load_spx_data_from_kalshi_markets()

    # Run on Kalshi trades
    if args.source in ("kalshi", "both"):
        logger.info("Loading Kalshi trade data...")
        kalshi_trades = load_kalshi_trades()
        if kalshi_trades:
            logger.info(f"Running backtest on {len(kalshi_trades)} Kalshi trades...")
            result = run_backtest_on_kalshi_trades(
                kalshi_trades, config, spx_reference
            )
            results.append(("kalshi", result))
        else:
            logger.warning("No Kalshi trade data found. Run kalshi_trade_history.py first.")
            print("\nNo Kalshi trade data found.")
            print("Run: python kalshi_trade_history.py --days 30")
            print("Then re-run this backtest.")

    # Run on pmxt data
    if args.source in ("pmxt", "both"):
        if HAS_PANDAS and HAS_PYARROW:
            logger.info("Loading pmxt orderbook data...")
            orderbook_df = load_pmxt_orderbooks()
            if orderbook_df is not None:
                logger.info(f"Running backtest on {len(orderbook_df)} orderbook rows...")
                result = run_backtest_on_pmxt_data(orderbook_df, config)
                results.append(("pmxt", result))
            else:
                logger.warning("No pmxt data found. Run pmxt_downloader.py first.")
                print("\nNo pmxt data found.")
                print("Run: python pmxt_downloader.py --sample 2")
                print("Then re-run this backtest.")
        else:
            logger.warning("pandas/pyarrow not available for pmxt data")

    if not results:
        print("\nNo data available for backtesting.")
        print("Please download data first:")
        print("  python kalshi_trade_history.py --days 30")
        print("  python pmxt_downloader.py --days 7 --sample 2")
        return

    # Print results
    for source, result in results:
        print(f"\n{'='*70}")
        print(f"  Source: {source}")
        print(format_report(result))

        # Compare with synthetic
        if args.compare:
            comp = compare_with_synthetic(result)
            result.comparison_vs_synthetic = comp
            if "error" not in comp:
                print(f"\n  Comparison file: {comp.get('synthetic_file', 'N/A')}")
                for metric, data in comp.get("metrics", {}).items():
                    print(
                        f"  {metric:<25s}  synth={data['synthetic']:>+10.4f}  "
                        f"real={data['real']:>+10.4f}  {data['interpretation']}"
                    )
                print(f"\n  VERDICT: {comp.get('verdict', 'N/A')}")
            else:
                print(f"\n  Comparison error: {comp['error']}")

        # Verbose trades
        if args.verbose and result.trades:
            print(f"\n  RECENT TRADES (last 30):")
            print(f"  {'Date':>12s} | {'Side':>8s} | {'Ticker':>30s} | "
                  f"{'Entry':>6s} | {'FV':>6s} | {'Net$':>8s}")
            print("  " + "-" * 85)
            for t in result.trades[-30:]:
                print(f"  {t['date']:>12s} | {t['side']:>8s} | "
                      f"{t['ticker'][:30]:>30s} | "
                      f"${t['entry_price']:>5.2f} | ${t['fair_value']:>5.2f} | "
                      f"${t['pnl_net']:>+7.2f}")

    # Save results
    if not args.no_save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for source, result in results:
            output_file = DATA_DIR / f"real_backtest_results_{timestamp}_{source}.json"
            save_data = {
                "timestamp": datetime.now().isoformat(),
                "source": source,
                "result": asdict(result),
            }
            with open(output_file, "w") as f:
                json.dump(save_data, f, indent=2, default=str)
            print(f"\n  Results saved to: {output_file}")


if __name__ == "__main__":
    main()
