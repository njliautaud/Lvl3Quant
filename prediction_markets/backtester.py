"""
Prediction Markets Strategy Backtester

Comprehensive backtesting framework for testing trading strategies against
historical Polymarket and Kalshi data collected by data_collector.py.

Design principles:
  1. P&L in dollars, not abstract units.
  2. Taker fills at ask price (conservative; no mid-fill assumption).
  3. Capital locked until resolution (opportunity cost tracked).
  4. Settlement delay: 24-48h for Polymarket, same-day for Kalshi.
  5. Position sizing via half-Kelly criterion.
  6. Monte Carlo bootstrapping to quantify real edge vs. luck.

Usage:
    python backtester.py --strategy longshot_fade
    python backtester.py --strategy all --monte-carlo
    python backtester.py --show-results data/backtest_results_longshot_fade_2026-03-03.json
"""

import json
import logging
import math
import random
import statistics
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger("backtester")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Fee constants (mirror kalshi_client.py / polymarket_client.py)
# ---------------------------------------------------------------------------

POLYMARKET_TAKER_FEE = 0.02        # 2% taker fee on entry price * (1 - entry price)
KALSHI_FEE_COEFF_SPX = 0.035      # SPX/NDX halved fee
KALSHI_FEE_COEFF_STD = 0.07       # Standard Kalshi markets

# Polymarket settlement delay (hours). Capital is locked this long after entry.
POLYMARKET_SETTLEMENT_HOURS = 36   # 24-48h; use 36h as conservative midpoint
KALSHI_SETTLEMENT_HOURS = 1        # Near-instant after resolution

# Annual risk-free rate for opportunity cost calculation
ANNUAL_RISK_FREE = 0.045           # 4.5%


# ---------------------------------------------------------------------------
# Data Types
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """A single completed trade (entry + exit)."""
    trade_id: str
    platform: str                      # "polymarket" or "kalshi"
    strategy: str
    question: str
    market_slug: str
    side: str                          # "YES" or "NO"
    entry_price: float                 # Price paid per contract (0-1)
    contracts: int
    entry_timestamp: str
    exit_timestamp: Optional[str]
    resolution: Optional[str]          # "YES" or "NO"
    payout: float                      # Per-contract payout at resolution
    gross_pnl: float                   # (payout - entry_price) * contracts
    fee: float                         # Total fee paid
    opportunity_cost: float            # Interest foregone while capital locked
    net_pnl: float                     # gross_pnl - fee - opportunity_cost
    capital_at_risk: float             # entry_price * contracts
    settlement_hours: float            # How long capital was locked


@dataclass
class BacktestResult:
    """Full result of a strategy backtest run."""
    strategy: str
    platform: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float

    # Aggregate metrics
    total_pnl: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_pnl_per_trade: float
    avg_gross_edge: float              # Avg raw edge before costs
    avg_net_edge: float                # Avg net edge after costs

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    calmar_ratio: float

    # Capital efficiency
    avg_capital_locked: float
    total_opportunity_cost: float
    total_fees: float

    # Time series
    daily_pnl_series: list             # [{"date": ..., "pnl": ...}]
    equity_curve: list                 # [{"date": ..., "equity": ...}]

    # Trade log
    trades_log: list                   # List of Trade dicts

    # Breakdown by sub-group
    breakdown_by_strategy: dict
    breakdown_by_platform: dict

    # Monte Carlo summary (populated by monte_carlo_edge)
    monte_carlo: Optional[dict] = None


# ---------------------------------------------------------------------------
# Fee Calculators
# ---------------------------------------------------------------------------

def polymarket_fee(contracts: int, entry_price: float) -> float:
    """
    Polymarket taker fee in dollars.
    Fee = 2% * entry_price * (1 - entry_price) * contracts
    (fee is a fraction of the maximum possible variance, not raw price)
    """
    return POLYMARKET_TAKER_FEE * entry_price * (1 - entry_price) * contracts


def kalshi_fee(contracts: int, entry_price: float, is_spx: bool = True) -> float:
    """
    Kalshi taker fee in dollars.
    Uses ceil to nearest cent (per Kalshi fee schedule).
    """
    coeff = KALSHI_FEE_COEFF_SPX if is_spx else KALSHI_FEE_COEFF_STD
    fee_raw = coeff * contracts * entry_price * (1 - entry_price)
    return math.ceil(fee_raw * 100) / 100  # Round up to nearest cent


def opportunity_cost(
    capital: float, hours_locked: float, annual_rate: float = ANNUAL_RISK_FREE
) -> float:
    """
    Opportunity cost in dollars for capital locked while waiting for resolution.
    Uses simple interest (not compound) for sub-annual periods.
    """
    return capital * annual_rate * (hours_locked / (365.25 * 24))


# ---------------------------------------------------------------------------
# Position Sizing: Half-Kelly
# ---------------------------------------------------------------------------

def kelly_position(
    edge: float,
    win_prob: float,
    capital: float,
    max_fraction: float = 0.25,
) -> float:
    """
    Full Kelly = edge / (win_prob * (1/win_prob - 1)) = edge / (1 - win_prob).
    We use half-Kelly (or less) to account for model uncertainty.

    Args:
        edge: Net edge in dollar terms per contract (0-1 scale).
        win_prob: Estimated probability of winning.
        capital: Available capital.
        max_fraction: Maximum fraction of capital to risk (hard cap).

    Returns:
        Dollar amount to stake (capped at max_fraction * capital).
    """
    if win_prob <= 0 or win_prob >= 1 or edge <= 0:
        return 0.0

    odds = (1 - win_prob) / win_prob    # b in standard Kelly formula
    full_kelly = edge / odds             # fraction of capital to bet

    half_kelly = full_kelly * 0.5        # apply half-Kelly shrinkage
    capped = min(half_kelly, max_fraction)
    return capped * capital


# ---------------------------------------------------------------------------
# Built-in Strategy Functions
# ---------------------------------------------------------------------------

def strategy_longshot_fade(market: dict, **kwargs) -> Optional[dict]:
    """
    Fade longshot contracts (<8 cents YES price) in non-financial markets.

    Academic finding: contracts priced <$0.08 win only ~2% of the time
    (true prob ~40% of their implied price). Buying NO on these is a
    systematic positive-EV trade.

    Only applies to NON-financial markets to avoid real tail risk.

    Returns:
        Trade dict or None if no signal.
    """
    yes_price = market.get("yes_price")
    volume = market.get("volume", 0)
    question = (market.get("question", "") or "").lower()

    if yes_price is None or volume < 200_000:
        return None

    if yes_price > 0.08:
        return None  # Not a longshot

    # Skip financial markets — tail events are real
    financial_kw = [
        "spx", "s&p", "fed", "interest rate", "gdp", "cpi",
        "recession", "stock market", "bitcoin", "btc",
        "treasury", "yield", "inflation",
    ]
    if any(kw in question for kw in financial_kw):
        return None

    # Longshot overpricing: estimated true prob ~40% of implied
    actual_prob_estimate = yes_price * 0.40
    no_price = 1.0 - yes_price
    no_fair_value = 1.0 - actual_prob_estimate

    raw_edge = no_fair_value - no_price      # Edge on BUY_NO
    if raw_edge <= 0.005:
        return None

    return {
        "side": "NO",
        "entry_price": no_price,
        "fair_value": no_fair_value,
        "raw_edge": raw_edge,
        "strategy": "longshot_fade",
        "note": f"Longshot fade: implied {yes_price:.3f} vs est. true {actual_prob_estimate:.3f}",
    }


def strategy_momentum(market: dict, history: list = None, **kwargs) -> Optional[dict]:
    """
    Buy markets that are moving from extremes (0-20% or 80-100%) toward 50%.

    Rationale: extremes often represent stale prices. As new information
    arrives, these markets revert toward their terminal distribution (50%
    unless one outcome is near-certain). Trend followers create momentum.

    Requires at least 3 prior snapshots to detect trend direction.

    Args:
        market: Current market snapshot.
        history: List of prior snapshots for this market, oldest-first.

    Returns:
        Trade dict or None.
    """
    yes_price = market.get("yes_price")
    volume = market.get("volume", 0)

    if yes_price is None or volume < 1_000_000:
        return None

    # Only trade markets in the extreme zones
    in_low_zone = yes_price < 0.20
    in_high_zone = yes_price > 0.80
    if not (in_low_zone or in_high_zone):
        return None

    if not history or len(history) < 3:
        return None

    # Get recent price trajectory
    recent_prices = [
        h.get("yes_price") for h in history[-3:]
        if h.get("yes_price") is not None
    ]
    if len(recent_prices) < 3:
        return None

    price_trend = recent_prices[-1] - recent_prices[0]  # Positive = moving up

    # Low zone + moving UP toward 50% → buy YES
    if in_low_zone and price_trend > 0.02:
        raw_edge = 0.50 - yes_price   # Conservative: target 50%
        if raw_edge < 0.05:
            return None
        return {
            "side": "YES",
            "entry_price": yes_price,
            "fair_value": min(yes_price + raw_edge, 0.50),
            "raw_edge": raw_edge,
            "strategy": "momentum",
            "note": f"Momentum: low zone {yes_price:.3f}, trend {price_trend:+.3f}",
        }

    # High zone + moving DOWN toward 50% → buy NO
    if in_high_zone and price_trend < -0.02:
        no_price = 1.0 - yes_price
        raw_edge = 0.50 - no_price
        if raw_edge < 0.05:
            return None
        return {
            "side": "NO",
            "entry_price": no_price,
            "fair_value": min(no_price + raw_edge, 0.50),
            "raw_edge": raw_edge,
            "strategy": "momentum",
            "note": f"Momentum: high zone {yes_price:.3f}, trend {price_trend:+.3f}",
        }

    return None


def strategy_mean_reversion(market: dict, history: list = None, **kwargs) -> Optional[dict]:
    """
    Fade rapid price moves: if YES price moved >10 cents in the last snapshot
    interval, bet on partial reversal.

    Rationale: prediction markets are illiquid. Large moves often overshoot
    due to a single large order or news overreaction. Reversion to fair value
    provides a short-term edge.

    Args:
        market: Current market snapshot.
        history: Prior snapshots, oldest-first.

    Returns:
        Trade dict or None.
    """
    yes_price = market.get("yes_price")
    volume = market.get("volume", 0)

    if yes_price is None or volume < 500_000:
        return None

    if not history or len(history) < 2:
        return None

    prev_price = history[-1].get("yes_price")
    if prev_price is None:
        return None

    move = yes_price - prev_price  # Positive = price moved up

    # Require at least 10-cent move to trigger
    if abs(move) < 0.10:
        return None

    # Only trade markets in the tradeable zone (10-90%)
    if not (0.10 <= yes_price <= 0.90):
        return None

    # Fade the move: if price spiked UP, buy NO; if crashed DOWN, buy YES
    reversion_target = prev_price + move * 0.40  # Expect 40% reversion
    raw_edge = abs(yes_price - reversion_target)

    if raw_edge < 0.04:
        return None

    if move > 0:
        # Price jumped up — fade with BUY_NO
        no_price = 1.0 - yes_price
        return {
            "side": "NO",
            "entry_price": no_price,
            "fair_value": 1.0 - reversion_target,
            "raw_edge": raw_edge,
            "strategy": "mean_reversion",
            "note": f"Mean reversion: spike {prev_price:.3f}→{yes_price:.3f}, target {reversion_target:.3f}",
        }
    else:
        # Price crashed down — fade with BUY_YES
        return {
            "side": "YES",
            "entry_price": yes_price,
            "fair_value": reversion_target,
            "raw_edge": raw_edge,
            "strategy": "mean_reversion",
            "note": f"Mean reversion: crash {prev_price:.3f}→{yes_price:.3f}, target {reversion_target:.3f}",
        }


def strategy_vol_bracket(
    bracket: dict,
    vol_prediction: float = 0.18,
    hours_to_close: float = 4.0,
    spx_price: float = None,
    **kwargs,
) -> Optional[dict]:
    """
    Trade Kalshi SPX brackets using our vol model (IC=0.644 at 30min).

    Calculates fair value using a Student-t distribution (fat tails, df=5)
    and looks for contracts where the market price deviates meaningfully.

    Args:
        bracket: Kalshi bracket snapshot dict.
        vol_prediction: Annualized vol forecast (e.g. 0.18 = 18%).
        hours_to_close: Hours until 4pm ET market close.
        spx_price: Current SPX index level.

    Returns:
        Trade dict or None.
    """
    try:
        from scipy import stats as scipy_stats
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False

    floor = bracket.get("floor")
    cap = bracket.get("cap")
    yes_ask = bracket.get("yes_ask")
    no_ask = bracket.get("no_ask")

    # Fallback SPX price from snapshot
    if spx_price is None:
        spx_price = bracket.get("spx_at_snapshot")

    if any(v is None for v in [floor, cap, yes_ask, no_ask, spx_price]):
        return None

    if hours_to_close <= 0:
        hours_to_close = 0.01

    # Fair value via Student-t distribution (matches empirical SPX kurtosis ~4-6)
    time_fraction = hours_to_close / (252 * 6.5)
    sigma_full = spx_price * vol_prediction * math.sqrt(time_fraction)

    if sigma_full <= 0:
        return None

    df = 5.0  # Degrees of freedom for fat tails
    if HAS_SCIPY:
        scale = sigma_full * math.sqrt((df - 2) / df)
        dist = scipy_stats.t(df=df, loc=spx_price, scale=scale)

        if floor is None or floor <= 0:
            fv = float(dist.cdf(cap))
        elif cap is None or cap >= spx_price * 2:
            fv = float(1.0 - dist.cdf(floor))
        else:
            fv = float(dist.cdf(cap) - dist.cdf(floor))
    else:
        # Gaussian approximation if scipy unavailable
        def _norm_cdf(x, mu, s):
            z = (x - mu) / s
            return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))

        if floor is None or floor <= 0:
            fv = _norm_cdf(cap, spx_price, sigma_full)
        elif cap is None or cap >= spx_price * 2:
            fv = 1.0 - _norm_cdf(floor, spx_price, sigma_full)
        else:
            fv = _norm_cdf(cap, spx_price, sigma_full) - _norm_cdf(floor, spx_price, sigma_full)

    fv = max(0.0, min(1.0, fv))

    # Calculate edges on both sides
    buy_yes_edge = fv - yes_ask
    buy_no_edge = (1.0 - fv) - no_ask

    if buy_yes_edge <= 0 and buy_no_edge <= 0:
        return None

    if buy_yes_edge >= buy_no_edge:
        raw_edge = buy_yes_edge
        side = "YES"
        entry_price = yes_ask
    else:
        raw_edge = buy_no_edge
        side = "NO"
        entry_price = no_ask

    if raw_edge < 0.01:  # Below 1-cent threshold
        return None

    return {
        "side": side,
        "entry_price": entry_price,
        "fair_value": fv if side == "YES" else (1.0 - fv),
        "raw_edge": raw_edge,
        "strategy": "vol_bracket",
        "note": (
            f"Vol bracket: SPX={spx_price:.0f}, vol={vol_prediction:.3f}, "
            f"FV={fv:.4f}, side={side}, edge={raw_edge:.4f}"
        ),
    }


def strategy_consensus_divergence(
    market: dict,
    consensus: float,
    min_divergence: float = 0.12,
    **kwargs,
) -> Optional[dict]:
    """
    Trade when the market price diverges from an expert consensus estimate
    by more than min_divergence.

    Rationale: Expert consensus (e.g., Bloomberg economist surveys for CPI,
    Reuters polls for NFP) reflects the base rate from domain experts.
    Large divergences from consensus are exploitable until new data arrives.

    Args:
        market: Market snapshot dict.
        consensus: Expert consensus YES probability (0-1).
        min_divergence: Minimum absolute divergence to trade (default: 12%).

    Returns:
        Trade dict or None.
    """
    yes_price = market.get("yes_price")
    volume = market.get("volume", 0)

    if yes_price is None or volume < 250_000:
        return None

    divergence = consensus - yes_price  # Positive = market is too bearish vs consensus

    if abs(divergence) < min_divergence:
        return None

    if divergence > 0:
        # Market is too cheap vs consensus → buy YES
        raw_edge = divergence
        return {
            "side": "YES",
            "entry_price": yes_price,
            "fair_value": consensus,
            "raw_edge": raw_edge,
            "strategy": "consensus_divergence",
            "note": (
                f"Consensus divergence: market {yes_price:.3f} vs "
                f"consensus {consensus:.3f} (gap {divergence:+.3f})"
            ),
        }
    else:
        # Market is too expensive vs consensus → buy NO
        no_price = 1.0 - yes_price
        no_consensus = 1.0 - consensus
        raw_edge = abs(divergence)
        return {
            "side": "NO",
            "entry_price": no_price,
            "fair_value": no_consensus,
            "raw_edge": raw_edge,
            "strategy": "consensus_divergence",
            "note": (
                f"Consensus divergence: market {yes_price:.3f} vs "
                f"consensus {consensus:.3f} (gap {divergence:+.3f})"
            ),
        }


# ---------------------------------------------------------------------------
# StrategyBacktester
# ---------------------------------------------------------------------------

class StrategyBacktester:
    """
    Simulate prediction market strategies against historical data.

    Primary method: run_backtest(strategy_fn, data, initial_capital)
    Supports Monte Carlo bootstrapping to quantify real edge vs. luck.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)

    # -----------------------------------------------------------------------
    # Core backtest engine
    # -----------------------------------------------------------------------

    def run_backtest(
        self,
        strategy_fn: Callable,
        data: list,
        initial_capital: float = 10_000.0,
        platform: str = "polymarket",
        max_position_fraction: float = 0.25,
        strategy_kwargs: dict = None,
    ) -> BacktestResult:
        """
        Simulate a strategy over historical market snapshots.

        Args:
            strategy_fn: Strategy function signature:
                f(market: dict, history: list, **kwargs) -> Optional[dict]
                Returns trade signal dict or None.
            data: List of market snapshot dicts, sorted by timestamp ascending.
                  Each dict must have: timestamp, question, market_slug,
                  yes_price, volume, resolved, resolution.
            initial_capital: Starting capital in dollars.
            platform: "polymarket" or "kalshi" — affects fee calculation.
            max_position_fraction: Max fraction of capital per trade (half-Kelly cap).
            strategy_kwargs: Extra keyword arguments passed to strategy_fn.

        Returns:
            BacktestResult with full metrics and trade log.
        """
        if strategy_kwargs is None:
            strategy_kwargs = {}

        strategy_name = getattr(strategy_fn, "__name__", "custom")
        logger.info(
            f"Running backtest: strategy={strategy_name}, "
            f"data={len(data)} records, capital=${initial_capital:,.0f}"
        )

        capital = initial_capital
        trades: list[Trade] = []
        equity_by_date: dict = {}
        pnl_by_date: dict = {}

        # Group data by market slug for history lookups
        market_history: dict = {}  # slug -> [snapshots...]
        data_sorted = sorted(data, key=lambda r: r.get("timestamp", ""))

        # Only process unresolved snapshots (resolved=False) as entry signals;
        # resolutions are used for exit P&L
        resolution_map: dict = {}  # market_slug -> resolution ("YES"/"NO")

        # Build resolution map first
        for record in data_sorted:
            slug = record.get("market_slug", record.get("ticker", ""))
            if record.get("resolved") and record.get("resolution") in ("YES", "NO"):
                resolution_map[slug] = record["resolution"]

        trade_id = 0
        for record in data_sorted:
            slug = record.get("market_slug", record.get("ticker", ""))
            ts = record.get("timestamp", "")
            date_str = ts[:10] if ts else "unknown"

            # Skip already-resolved records as entry signals
            if record.get("resolved"):
                continue

            # Get history for this market
            history = market_history.get(slug, [])

            # Ask strategy for a signal
            try:
                signal = strategy_fn(record, history=history, **strategy_kwargs)
            except Exception as e:
                logger.warning(f"Strategy error on {slug}: {e}")
                signal = None

            # Update history AFTER calling strategy (don't leak future data)
            if slug not in market_history:
                market_history[slug] = []
            market_history[slug].append(record)

            if signal is None:
                continue

            side = signal["side"]            # "YES" or "NO"
            entry_price = signal["entry_price"]
            raw_edge = signal.get("raw_edge", 0.0)
            fair_value = signal.get("fair_value", entry_price + raw_edge)

            # Check resolution available (needed to compute P&L)
            resolution = resolution_map.get(slug)
            if resolution is None:
                continue  # Can't compute P&L without resolution

            # Determine payout
            win = (side == "YES" and resolution == "YES") or \
                  (side == "NO" and resolution == "NO")
            payout_per_contract = 1.0 if win else 0.0

            # Estimate win probability from fair value
            win_prob = fair_value if side == "YES" else (1.0 - fair_value)
            win_prob = max(0.01, min(0.99, win_prob))

            # Size position via half-Kelly (capped at max_position_fraction)
            stake_dollars = kelly_position(
                edge=raw_edge,
                win_prob=win_prob,
                capital=capital,
                max_fraction=max_position_fraction,
            )
            if stake_dollars < 0.01:
                continue  # Position too small to bother

            contracts = max(1, int(stake_dollars / entry_price))
            capital_at_risk = contracts * entry_price

            # Make sure we have capital
            if capital_at_risk > capital:
                contracts = max(1, int(capital / entry_price))
                capital_at_risk = contracts * entry_price

            if capital_at_risk > capital or capital <= 0:
                continue

            # Calculate fees
            if platform == "kalshi":
                fee = kalshi_fee(contracts, entry_price, is_spx=True)
            else:
                fee = polymarket_fee(contracts, entry_price)

            # Settlement delay & opportunity cost
            settle_hours = (
                POLYMARKET_SETTLEMENT_HOURS
                if platform == "polymarket"
                else KALSHI_SETTLEMENT_HOURS
            )
            opp_cost = opportunity_cost(capital_at_risk, settle_hours)

            # Compute P&L
            gross_pnl = (payout_per_contract - entry_price) * contracts
            net_pnl = gross_pnl - fee - opp_cost

            # Update capital
            capital += net_pnl

            # Determine exit timestamp (approximate: use end_date if available)
            end_date = record.get("end_date")
            exit_ts = f"{end_date}T18:00:00Z" if end_date else None

            trade_id += 1
            trade = Trade(
                trade_id=f"{strategy_name}_{trade_id:05d}",
                platform=platform,
                strategy=signal.get("strategy", strategy_name),
                question=record.get("question", ""),
                market_slug=slug,
                side=side,
                entry_price=round(entry_price, 6),
                contracts=contracts,
                entry_timestamp=ts,
                exit_timestamp=exit_ts,
                resolution=resolution,
                payout=round(payout_per_contract, 4),
                gross_pnl=round(gross_pnl, 4),
                fee=round(fee, 4),
                opportunity_cost=round(opp_cost, 6),
                net_pnl=round(net_pnl, 4),
                capital_at_risk=round(capital_at_risk, 4),
                settlement_hours=settle_hours,
            )
            trades.append(trade)

            # Track by date
            pnl_by_date[date_str] = pnl_by_date.get(date_str, 0.0) + net_pnl
            equity_by_date[date_str] = capital

        logger.info(
            f"Backtest complete: {len(trades)} trades, "
            f"final capital=${capital:,.2f}"
        )

        return self._build_result(
            strategy=strategy_name,
            platform=platform,
            trades=trades,
            initial_capital=initial_capital,
            final_capital=capital,
            pnl_by_date=pnl_by_date,
            equity_by_date=equity_by_date,
        )

    # -----------------------------------------------------------------------
    # Result construction
    # -----------------------------------------------------------------------

    def _build_result(
        self,
        strategy: str,
        platform: str,
        trades: list,
        initial_capital: float,
        final_capital: float,
        pnl_by_date: dict,
        equity_by_date: dict,
    ) -> BacktestResult:
        """Compute all metrics from raw trade list."""
        if not trades:
            return BacktestResult(
                strategy=strategy,
                platform=platform,
                start_date="N/A",
                end_date="N/A",
                initial_capital=initial_capital,
                final_capital=initial_capital,
                total_pnl=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                avg_pnl_per_trade=0.0,
                avg_gross_edge=0.0,
                avg_net_edge=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0,
                max_drawdown_pct=0.0,
                calmar_ratio=0.0,
                avg_capital_locked=0.0,
                total_opportunity_cost=0.0,
                total_fees=0.0,
                daily_pnl_series=[],
                equity_curve=[],
                trades_log=[],
                breakdown_by_strategy={},
                breakdown_by_platform={},
            )

        winning = [t for t in trades if t.net_pnl > 0]
        losing = [t for t in trades if t.net_pnl <= 0]
        net_pnls = [t.net_pnl for t in trades]
        gross_edges = [
            (t.payout - t.entry_price) / max(t.entry_price, 0.01)
            for t in trades
        ]

        # Dates
        timestamps = [t.entry_timestamp for t in trades if t.entry_timestamp]
        start_date = min(timestamps)[:10] if timestamps else "N/A"
        end_date = max(timestamps)[:10] if timestamps else "N/A"

        # Risk-adjusted metrics
        daily_pnls = [v for v in pnl_by_date.values()]
        sharpe = self._sharpe(daily_pnls)
        sortino = self._sortino(daily_pnls)
        max_dd, max_dd_pct = self._max_drawdown(equity_by_date, initial_capital)
        calmar = (
            (final_capital - initial_capital) / abs(max_dd)
            if max_dd != 0 else 0.0
        )

        # Cost breakdown
        total_fees = sum(t.fee for t in trades)
        total_opp_cost = sum(t.opportunity_cost for t in trades)
        avg_capital_locked = (
            sum(t.capital_at_risk for t in trades) / len(trades)
        )

        # Breakdowns
        def _breakdown(field_fn):
            groups: dict = {}
            for t in trades:
                key = field_fn(t)
                if key not in groups:
                    groups[key] = {"trades": 0, "total_pnl": 0.0, "wins": 0}
                groups[key]["trades"] += 1
                groups[key]["total_pnl"] += t.net_pnl
                groups[key]["wins"] += int(t.net_pnl > 0)
            for k, v in groups.items():
                v["total_pnl"] = round(v["total_pnl"], 4)
                v["win_rate"] = round(v["wins"] / v["trades"], 3)
            return groups

        # Build time series
        sorted_dates = sorted(pnl_by_date.keys())
        daily_series = [
            {"date": d, "pnl": round(pnl_by_date[d], 4)}
            for d in sorted_dates
        ]
        equity_series = [
            {"date": d, "equity": round(equity_by_date.get(d, initial_capital), 2)}
            for d in sorted_dates
        ]

        return BacktestResult(
            strategy=strategy,
            platform=platform,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_capital=round(final_capital, 2),
            total_pnl=round(final_capital - initial_capital, 2),
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=round(len(winning) / len(trades), 4) if trades else 0.0,
            avg_pnl_per_trade=round(
                sum(net_pnls) / len(net_pnls) if net_pnls else 0.0, 4
            ),
            avg_gross_edge=round(
                sum(gross_edges) / len(gross_edges) if gross_edges else 0.0, 4
            ),
            avg_net_edge=round(
                sum(net_pnls) / len(net_pnls) if net_pnls else 0.0, 4
            ),
            sharpe_ratio=round(sharpe, 3),
            sortino_ratio=round(sortino, 3),
            max_drawdown=round(max_dd, 2),
            max_drawdown_pct=round(max_dd_pct, 4),
            calmar_ratio=round(calmar, 3),
            avg_capital_locked=round(avg_capital_locked, 2),
            total_opportunity_cost=round(total_opp_cost, 4),
            total_fees=round(total_fees, 4),
            daily_pnl_series=daily_series,
            equity_curve=equity_series,
            trades_log=[asdict(t) for t in trades],
            breakdown_by_strategy=_breakdown(lambda t: t.strategy),
            breakdown_by_platform=_breakdown(lambda t: t.platform),
        )

    # -----------------------------------------------------------------------
    # Statistical helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _sharpe(daily_pnls: list, periods_per_year: float = 252.0) -> float:
        """Annualised Sharpe ratio from daily P&L series."""
        if len(daily_pnls) < 2:
            return 0.0
        mu = statistics.mean(daily_pnls)
        sigma = statistics.stdev(daily_pnls)
        if sigma == 0:
            return 0.0
        return mu / sigma * math.sqrt(periods_per_year)

    @staticmethod
    def _sortino(
        daily_pnls: list,
        periods_per_year: float = 252.0,
        target: float = 0.0,
    ) -> float:
        """Annualised Sortino ratio (downside deviation denominator)."""
        if len(daily_pnls) < 2:
            return 0.0
        mu = statistics.mean(daily_pnls)
        downside = [min(0.0, p - target) for p in daily_pnls]
        downside_var = sum(d ** 2 for d in downside) / len(downside)
        downside_std = math.sqrt(downside_var)
        if downside_std == 0:
            return 0.0
        return mu / downside_std * math.sqrt(periods_per_year)

    @staticmethod
    def _max_drawdown(equity_by_date: dict, initial_capital: float) -> tuple:
        """
        Returns (max_drawdown_dollars, max_drawdown_fraction).
        Drawdown measured from running peak.
        """
        if not equity_by_date:
            return 0.0, 0.0

        sorted_equities = [
            equity_by_date[d]
            for d in sorted(equity_by_date.keys())
        ]
        sorted_equities = [initial_capital] + sorted_equities

        peak = sorted_equities[0]
        max_dd = 0.0
        max_dd_pct = 0.0

        for eq in sorted_equities[1:]:
            if eq > peak:
                peak = eq
            dd = peak - eq
            dd_pct = dd / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd_pct

        return max_dd, max_dd_pct

    # -----------------------------------------------------------------------
    # Monte Carlo
    # -----------------------------------------------------------------------

    def monte_carlo_edge(
        self,
        strategy_fn: Callable,
        data: list,
        n_simulations: int = 10_000,
        initial_capital: float = 10_000.0,
        platform: str = "polymarket",
        strategy_kwargs: dict = None,
        confidence_levels: tuple = (0.05, 0.25, 0.50, 0.75, 0.95),
    ) -> dict:
        """
        Bootstrap Monte Carlo to distinguish real edge from luck.

        Runs the backtest once to get the trade history, then resamples
        trade outcomes with replacement to build a distribution of P&L,
        Sharpe, and win rate under the null hypothesis that each trade is
        an iid draw from the empirical distribution.

        Args:
            strategy_fn: Strategy function.
            data: Historical snapshot data.
            n_simulations: Number of bootstrap iterations.
            initial_capital: Starting capital.
            platform: "polymarket" or "kalshi".
            strategy_kwargs: Extra kwargs for strategy_fn.
            confidence_levels: Percentiles to report.

        Returns:
            dict with 'base_result' and 'monte_carlo' confidence intervals.
        """
        # Run base backtest
        logger.info(f"Running Monte Carlo ({n_simulations} simulations)...")
        base = self.run_backtest(
            strategy_fn, data, initial_capital, platform,
            strategy_kwargs=strategy_kwargs,
        )

        if base.total_trades == 0:
            return {
                "base_result": asdict(base),
                "monte_carlo": {
                    "status": "NO_TRADES",
                    "message": "No trades generated — cannot run Monte Carlo.",
                },
            }

        trade_pnls = [t["net_pnl"] for t in base.trades_log]
        n_trades = len(trade_pnls)

        # Bootstrap: resample trade outcomes
        sim_totals = []
        sim_sharpes = []
        sim_win_rates = []

        for _ in range(n_simulations):
            sample = random.choices(trade_pnls, k=n_trades)
            sim_total = sum(sample)
            sim_wins = sum(1 for p in sample if p > 0)
            sim_win_rate = sim_wins / n_trades

            # Daily P&L distribution (assume uniform across trades)
            # For Sharpe: use trade-level P&L as proxy for daily (conservative)
            if len(sample) >= 2:
                mu = statistics.mean(sample)
                sigma = statistics.stdev(sample)
                if sigma > 0:
                    sharpe = mu / sigma * math.sqrt(n_trades)
                else:
                    sharpe = 0.0
            else:
                sharpe = 0.0

            sim_totals.append(sim_total)
            sim_sharpes.append(sharpe)
            sim_win_rates.append(sim_win_rate)

        def _percentiles(values, levels):
            sorted_v = sorted(values)
            result = {}
            for lvl in levels:
                idx = int(lvl * len(sorted_v))
                idx = min(idx, len(sorted_v) - 1)
                result[f"p{int(lvl*100)}"] = round(sorted_v[idx], 4)
            return result

        # Probability that observed total P&L > 0 is due to real edge
        prob_positive = sum(1 for v in sim_totals if v > 0) / n_simulations
        prob_positive_pct = round(prob_positive * 100, 1)

        mc_result = {
            "n_simulations": n_simulations,
            "n_trades": n_trades,
            "base_total_pnl": base.total_pnl,
            "base_sharpe": base.sharpe_ratio,
            "base_win_rate": base.win_rate,
            "probability_positive": prob_positive_pct,
            "verdict": (
                "REAL_EDGE" if prob_positive_pct >= 80
                else "MARGINAL" if prob_positive_pct >= 60
                else "LUCK"
            ),
            "total_pnl_distribution": _percentiles(sim_totals, confidence_levels),
            "sharpe_distribution": _percentiles(sim_sharpes, confidence_levels),
            "win_rate_distribution": _percentiles(sim_win_rates, confidence_levels),
            "mean_sim_pnl": round(statistics.mean(sim_totals), 4),
            "std_sim_pnl": round(statistics.stdev(sim_totals), 4) if len(sim_totals) > 1 else 0.0,
        }

        logger.info(
            f"Monte Carlo done: {prob_positive_pct}% of simulations positive "
            f"→ verdict: {mc_result['verdict']}"
        )

        return {
            "base_result": asdict(base),
            "monte_carlo": mc_result,
        }

    # -----------------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------------

    def generate_report(self, result: BacktestResult) -> str:
        """
        Format a BacktestResult as a human-readable text report.

        Returns:
            Multi-line string report.
        """
        lines = [
            "=" * 70,
            f"BACKTEST REPORT — {result.strategy.upper().replace('_', ' ')}",
            f"Platform: {result.platform.upper()}  |  "
            f"Period: {result.start_date} → {result.end_date}",
            "=" * 70,
            "",
            "--- PERFORMANCE SUMMARY ---",
            f"  Initial capital:    ${result.initial_capital:>12,.2f}",
            f"  Final capital:      ${result.final_capital:>12,.2f}",
            f"  Total P&L:          ${result.total_pnl:>+12,.2f}",
            f"  Return:             {(result.total_pnl / result.initial_capital * 100):>+11.1f}%",
            "",
            "--- TRADE STATISTICS ---",
            f"  Total trades:       {result.total_trades:>12,}",
            f"  Winning trades:     {result.winning_trades:>12,}",
            f"  Losing trades:      {result.losing_trades:>12,}",
            f"  Win rate:           {result.win_rate:>12.1%}",
            f"  Avg P&L/trade:      ${result.avg_pnl_per_trade:>+11.4f}",
            f"  Avg gross edge:     {result.avg_gross_edge:>12.4f}",
            f"  Avg net edge:       {result.avg_net_edge:>+12.4f}",
            "",
            "--- RISK METRICS ---",
            f"  Sharpe ratio:       {result.sharpe_ratio:>12.3f}",
            f"  Sortino ratio:      {result.sortino_ratio:>12.3f}",
            f"  Max drawdown:       ${result.max_drawdown:>12,.2f}  ({result.max_drawdown_pct:.1%})",
            f"  Calmar ratio:       {result.calmar_ratio:>12.3f}",
            "",
            "--- COST BREAKDOWN ---",
            f"  Total fees paid:    ${result.total_fees:>12,.4f}",
            f"  Total opp. cost:    ${result.total_opportunity_cost:>12,.4f}",
            f"  Avg capital locked: ${result.avg_capital_locked:>12,.2f}",
        ]

        if result.breakdown_by_strategy:
            lines += ["", "--- BREAKDOWN BY STRATEGY ---"]
            for strat, data in sorted(
                result.breakdown_by_strategy.items(),
                key=lambda x: x[1]["total_pnl"],
                reverse=True,
            ):
                lines.append(
                    f"  {strat:<25}  trades={data['trades']:>4}  "
                    f"pnl=${data['total_pnl']:>+9.2f}  "
                    f"win={data['win_rate']:.1%}"
                )

        if result.monte_carlo:
            mc = result.monte_carlo
            lines += [
                "",
                "--- MONTE CARLO ANALYSIS ---",
                f"  Simulations:        {mc.get('n_simulations', 'N/A'):>12,}",
                f"  Prob. positive:     {mc.get('probability_positive', 0):>11.1f}%",
                f"  Verdict:            {mc.get('verdict', 'N/A'):>12}",
                f"  P50 sim P&L:        ${mc.get('total_pnl_distribution', {}).get('p50', 0):>+11.4f}",
                f"  P5–P95 P&L range:   "
                f"${mc.get('total_pnl_distribution', {}).get('p5', 0):>+.2f} "
                f"→ ${mc.get('total_pnl_distribution', {}).get('p95', 0):>+.2f}",
            ]

        if result.trades_log:
            lines += ["", "--- LAST 5 TRADES ---"]
            for t in result.trades_log[-5:]:
                lines.append(
                    f"  {t['entry_timestamp'][:10]}  "
                    f"{t['side']:<3}  "
                    f"{t['question'][:40]:<40}  "
                    f"P&L=${t['net_pnl']:>+8.4f}"
                )

        lines.append("=" * 70)
        return "\n".join(lines)

    def save_result(self, result: BacktestResult, strategy_override: str = None) -> Path:
        """
        Save BacktestResult to `data/backtest_results_{strategy}_{date}.json`.

        Returns:
            Path of the written file.
        """
        strategy_slug = (strategy_override or result.strategy).lower().replace(" ", "_")
        today = datetime.now().strftime("%Y-%m-%d")
        out_path = DATA_DIR / f"backtest_results_{strategy_slug}_{today}.json"

        _dict = asdict(result)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(_dict, f, indent=2, default=str)

        logger.info(f"Backtest result saved → {out_path}")
        return out_path


# ---------------------------------------------------------------------------
# Convenience: load data from JSONL files
# ---------------------------------------------------------------------------

def load_polymarket_data(date_str: str = None, days_back: int = 30) -> list:
    """
    Load Polymarket snapshots + resolutions from data/ directory.

    Merges snapshot records with resolution info to build a unified
    dataset suitable for run_backtest().

    Args:
        date_str: Specific 'YYYY-MM-DD' snapshot file to load, or None for all.
        days_back: If date_str is None, load files this many days back.

    Returns:
        List of enriched market dicts sorted by timestamp.
    """
    resolution_map: dict = {}
    resolved_path = DATA_DIR / "polymarket_resolved.jsonl"
    if resolved_path.exists():
        with open(resolved_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                slug = rec.get("market_slug") or rec.get("condition_id")
                if slug and rec.get("resolution") in ("YES", "NO"):
                    resolution_map[slug] = rec["resolution"]

    records = []
    if date_str:
        snap_files = [DATA_DIR / f"polymarket_snapshots_{date_str}.jsonl"]
    else:
        cutoff = datetime.now() - timedelta(days=days_back)
        snap_files = sorted(DATA_DIR.glob("polymarket_snapshots_*.jsonl"))
        snap_files = [
            f for f in snap_files
            if f.stem.split("_")[-1] >= cutoff.strftime("%Y-%m-%d")
        ]

    for snap_file in snap_files:
        if not snap_file.exists():
            continue
        with open(snap_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                slug = rec.get("market_slug") or rec.get("condition_id")
                res = resolution_map.get(slug)
                rec["resolved"] = res is not None
                rec["resolution"] = res
                records.append(rec)

    records.sort(key=lambda r: r.get("timestamp", ""))
    logger.info(f"Loaded {len(records)} Polymarket records ({len(resolution_map)} resolutions available)")
    return records


def load_kalshi_data(days_back: int = 30) -> list:
    """
    Load Kalshi bracket snapshots + resolutions.

    Returns:
        List of enriched bracket dicts sorted by timestamp.
    """
    resolution_map: dict = {}
    resolved_path = DATA_DIR / "kalshi_resolved.jsonl"
    if resolved_path.exists():
        with open(resolved_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ticker = rec.get("ticker")
                if ticker and rec.get("resolution") in ("YES", "NO"):
                    resolution_map[ticker] = rec["resolution"]

    cutoff = datetime.now() - timedelta(days=days_back)
    bracket_files = sorted(DATA_DIR.glob("kalshi_brackets_*.jsonl"))
    bracket_files = [
        f for f in bracket_files
        if f.stem.split("_")[-1] >= cutoff.strftime("%Y-%m-%d")
    ]

    records = []
    for bf in bracket_files:
        with open(bf, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ticker = rec.get("ticker")
                res = resolution_map.get(ticker)
                rec["resolved"] = res is not None
                rec["resolution"] = res
                # Adapt bracket fields to match generic snapshot format
                rec.setdefault("question", f"SPX {rec.get('subtitle', rec.get('ticker', ''))}")
                rec.setdefault("market_slug", rec.get("ticker", ""))
                # Use yes_ask as the "yes_price" for strategy functions
                rec.setdefault("yes_price", rec.get("yes_ask"))
                rec.setdefault("volume", rec.get("volume", 0))
                records.append(rec)

    records.sort(key=lambda r: r.get("timestamp", ""))
    logger.info(f"Loaded {len(records)} Kalshi records ({len(resolution_map)} resolutions available)")
    return records


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

STRATEGY_MAP = {
    "longshot_fade": strategy_longshot_fade,
    "momentum": strategy_momentum,
    "mean_reversion": strategy_mean_reversion,
    "vol_bracket": strategy_vol_bracket,
    "consensus_divergence": strategy_consensus_divergence,
}


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Prediction Market Strategy Backtester"
    )
    parser.add_argument(
        "--strategy",
        default="longshot_fade",
        choices=list(STRATEGY_MAP.keys()) + ["all"],
        help="Strategy to backtest (default: longshot_fade)",
    )
    parser.add_argument(
        "--platform",
        default="polymarket",
        choices=["polymarket", "kalshi"],
        help="Data platform (default: polymarket)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10_000.0,
        help="Initial capital in dollars (default: 10000)",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=30,
        help="Days of historical data to load (default: 30)",
    )
    parser.add_argument(
        "--monte-carlo",
        action="store_true",
        help="Run Monte Carlo simulation after backtest",
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=10_000,
        help="Number of Monte Carlo simulations (default: 10000)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save result to data/ directory",
    )
    parser.add_argument(
        "--show-results",
        type=str,
        default=None,
        help="Load and display a saved backtest JSON file",
    )
    args = parser.parse_args()

    # Show saved results
    if args.show_results:
        result_path = Path(args.show_results)
        if not result_path.exists():
            print(f"File not found: {result_path}")
            return
        with open(result_path, encoding="utf-8") as f:
            result_dict = json.load(f)
        # Reconstruct BacktestResult from dict
        result = BacktestResult(**{
            k: v for k, v in result_dict.items()
            if k in BacktestResult.__dataclass_fields__
        })
        bt = StrategyBacktester()
        print(bt.generate_report(result))
        return

    # Load data
    if args.platform == "kalshi":
        data = load_kalshi_data(days_back=args.days_back)
    else:
        data = load_polymarket_data(days_back=args.days_back)

    if not data:
        print(
            f"No historical data found in {DATA_DIR}. "
            "Run data_collector.py first to collect snapshots."
        )
        return

    bt = StrategyBacktester()

    strategies_to_run = (
        list(STRATEGY_MAP.keys())
        if args.strategy == "all"
        else [args.strategy]
    )

    for strat_name in strategies_to_run:
        strat_fn = STRATEGY_MAP[strat_name]
        print(f"\nRunning: {strat_name} on {args.platform} data...")

        if args.monte_carlo:
            mc_output = bt.monte_carlo_edge(
                strat_fn,
                data,
                n_simulations=args.n_sims,
                initial_capital=args.capital,
                platform=args.platform,
            )
            # Reconstruct result for report
            result_dict = mc_output["base_result"]
            result = BacktestResult(**{
                k: v for k, v in result_dict.items()
                if k in BacktestResult.__dataclass_fields__
            })
            result.monte_carlo = mc_output["monte_carlo"]
        else:
            result = bt.run_backtest(
                strat_fn,
                data,
                initial_capital=args.capital,
                platform=args.platform,
            )

        print(bt.generate_report(result))

        if args.save:
            saved_path = bt.save_result(result, strategy_override=strat_name)
            print(f"Result saved: {saved_path}")


if __name__ == "__main__":
    main()
