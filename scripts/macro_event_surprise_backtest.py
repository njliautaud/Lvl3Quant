#!/usr/bin/env python3
"""
Macro Event Surprise Sector Rotation Backtest
----------------------------------------------
Trades sector ETFs based on macroeconomic data surprise direction/magnitude.
Detects CPI, NFP, and FOMC surprises from market reactions, then rotates
into beneficiary sectors.

6 Variants: A-F (see docstrings below)
OOT: Jan 2022 - Jul 2026
Initial Capital: $645
"""

import json
import datetime as dt
import warnings
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Hardcoded Macro Event Dates (2022-2026) ──────────────────────────────

CPI_DATES = [
    # 2022
    "2022-01-12", "2022-02-10", "2022-03-10", "2022-04-12", "2022-05-11",
    "2022-06-10", "2022-07-13", "2022-08-10", "2022-09-13", "2022-10-13",
    "2022-11-10", "2022-12-13",
    # 2023
    "2023-01-12", "2023-02-14", "2023-03-14", "2023-04-12", "2023-05-10",
    "2023-06-13", "2023-07-12", "2023-08-10", "2023-09-13", "2023-10-12",
    "2023-11-14", "2023-12-12",
    # 2024
    "2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10", "2024-05-15",
    "2024-06-12", "2024-07-11", "2024-08-14", "2024-09-11", "2024-10-10",
    "2024-11-13", "2024-12-11",
    # 2025
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10", "2025-05-13",
    "2025-06-11", "2025-07-15", "2025-08-12", "2025-09-10", "2025-10-14",
    "2025-11-12", "2025-12-10",
    # 2026
    "2026-01-13", "2026-02-11", "2026-03-11", "2026-04-14", "2026-05-12",
    "2026-06-10", "2026-07-14",
]

# NFP = first Friday of each month, 2022-2026
def _first_fridays(start_year=2022, end_year=2026, end_month=7):
    dates = []
    for y in range(start_year, end_year + 1):
        max_m = end_month if y == end_year else 12
        for m in range(1, max_m + 1):
            d = dt.date(y, m, 1)
            # find first Friday
            while d.weekday() != 4:
                d += dt.timedelta(days=1)
            dates.append(d.strftime("%Y-%m-%d"))
    return dates

NFP_DATES = _first_fridays()

FOMC_DATES = [
    # 2022 (8 meetings)
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-17",
    # 2026
    "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
    "2026-07-29",
]

# ── Configuration ────────────────────────────────────────────────────────

UNIVERSE = ["XLK", "XLY", "XLP", "XLF", "XLV", "XLE", "XLI", "XLU", "XLC", "XLRE", "QQQ", "TLT"]
BENCHMARK = "SPY"
VIX_TICKER = "^VIX"

INITIAL_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
MAX_CONCURRENT = 2

OOT_START = "2022-01-01"
OOT_END = "2026-07-29"

# 5-gate thresholds
SHARPE_THRESHOLD = 0.5
PERM_P_THRESHOLD = 0.05
REGIME_GAP_THRESHOLD = 0.5
MAX_DD_THRESHOLD = -0.50
MIN_TRADES = 20
N_PERMUTATIONS = 1000

RESULTS_PATH = Path("/home/jupiter/Lvl3Quant/data/macro_event_surprise_results.json")

# ── Data Download ────────────────────────────────────────────────────────

def download_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Download OHLCV for universe + SPY + VIX via yfinance."""
    import yfinance as yf

    tickers = UNIVERSE + [BENCHMARK, VIX_TICKER]
    print(f"Downloading data for {len(tickers)} tickers...")

    data = yf.download(tickers, start="2021-06-01", end=OOT_END, auto_adjust=True, progress=False)

    # Get close prices
    close = data["Close"].copy()
    # Rename ^VIX
    if "^VIX" in close.columns:
        close.rename(columns={"^VIX": "VIX"}, inplace=True)

    # Get daily returns
    returns = close.pct_change()

    # Filter to OOT period
    close = close.loc[OOT_START:]
    returns = returns.loc[OOT_START:]

    print(f"Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} trading days")
    return close, returns


# ── Event Detection ──────────────────────────────────────────────────────

@dataclass
class MacroEvent:
    date: str
    event_type: str   # CPI, NFP, FOMC
    direction: str    # hot/cool, strong/weak, hawkish/dovish
    magnitude: float  # absolute return used for detection


def detect_events(returns: pd.DataFrame) -> List[MacroEvent]:
    """Detect macro surprise direction from market reactions."""
    events = []
    trading_dates = set(returns.index.strftime("%Y-%m-%d"))

    # CPI detection
    for d in CPI_DATES:
        if d not in trading_dates:
            # Try next trading day
            dd = pd.Timestamp(d)
            for offset in range(1, 4):
                nd = (dd + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
                if nd in trading_dates:
                    d = nd
                    break
            else:
                continue

        ts = pd.Timestamp(d)
        if ts not in returns.index:
            continue

        spy_ret = returns.loc[ts, "SPY"] if "SPY" in returns.columns else 0
        tlt_ret = returns.loc[ts, "TLT"] if "TLT" in returns.columns else 0

        if pd.isna(spy_ret) or pd.isna(tlt_ret):
            continue

        # Hot CPI: SPY down >0.5% AND TLT down >0.5%
        if spy_ret < -0.005 and tlt_ret < -0.005:
            events.append(MacroEvent(d, "CPI", "hot", abs(spy_ret) + abs(tlt_ret)))
        # Cool CPI: SPY up AND TLT up
        elif spy_ret > 0.005 and tlt_ret > 0.005:
            events.append(MacroEvent(d, "CPI", "cool", abs(spy_ret) + abs(tlt_ret)))

    # NFP detection
    for d in NFP_DATES:
        if d not in trading_dates:
            dd = pd.Timestamp(d)
            for offset in range(1, 4):
                nd = (dd + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
                if nd in trading_dates:
                    d = nd
                    break
            else:
                continue

        ts = pd.Timestamp(d)
        if ts not in returns.index:
            continue

        xlf_ret = returns.loc[ts, "XLF"] if "XLF" in returns.columns else 0
        tlt_ret = returns.loc[ts, "TLT"] if "TLT" in returns.columns else 0

        if pd.isna(xlf_ret) or pd.isna(tlt_ret):
            continue

        # Strong jobs: XLF up >0.5% AND TLT down
        if xlf_ret > 0.005 and tlt_ret < 0:
            events.append(MacroEvent(d, "NFP", "strong", abs(xlf_ret) + abs(tlt_ret)))
        # Weak jobs: XLF down AND TLT up
        elif xlf_ret < -0.005 and tlt_ret > 0:
            events.append(MacroEvent(d, "NFP", "weak", abs(xlf_ret) + abs(tlt_ret)))

    # FOMC detection
    for d in FOMC_DATES:
        if d not in trading_dates:
            dd = pd.Timestamp(d)
            for offset in range(1, 4):
                nd = (dd + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
                if nd in trading_dates:
                    d = nd
                    break
            else:
                continue

        ts = pd.Timestamp(d)
        if ts not in returns.index:
            continue

        spy_ret = returns.loc[ts, "SPY"] if "SPY" in returns.columns else 0

        if pd.isna(spy_ret):
            continue

        if abs(spy_ret) > 0.005:
            direction = "dovish" if spy_ret > 0 else "hawkish"
            events.append(MacroEvent(d, "FOMC", direction, abs(spy_ret)))

    events.sort(key=lambda e: e.date)
    print(f"Detected {len(events)} macro events: "
          f"CPI={sum(1 for e in events if e.event_type=='CPI')}, "
          f"NFP={sum(1 for e in events if e.event_type=='NFP')}, "
          f"FOMC={sum(1 for e in events if e.event_type=='FOMC')}")
    return events


# ── Sector Mapping ───────────────────────────────────────────────────────

# Maps (event_type, direction) -> list of (ticker, weight)
# weight > 0 = long, weight < 0 = short
SECTOR_MAP = {
    ("CPI", "hot"):      [("XLE", 1.0), ("XLY", -1.0)],
    ("CPI", "cool"):     [("XLY", 1.0), ("QQQ", 1.0)],
    ("NFP", "strong"):   [("XLF", 1.0), ("XLI", 1.0)],
    ("NFP", "weak"):     [("XLU", 1.0), ("TLT", 1.0)],
    ("FOMC", "hawkish"): [("XLE", 1.0), ("QQQ", -1.0)],
    ("FOMC", "dovish"):  [("QQQ", 1.0), ("XLY", 1.0)],
}


# ── Position / Trade Tracking ───────────────────────────────────────────

@dataclass
class Position:
    ticker: str
    entry_date: str
    entry_price: float
    direction: int  # +1 long, -1 short
    shares: float
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    event_type: str = ""
    event_direction: str = ""


# ── Backtest Engine ──────────────────────────────────────────────────────

def run_backtest(
    events: List[MacroEvent],
    close: pd.DataFrame,
    returns: pd.DataFrame,
    hold_days: int = 5,
    vix_filter: Optional[float] = None,
    contrarian: bool = False,
    combined_signal: bool = False,
    options_mode: bool = False,
    variant_name: str = "A",
) -> Dict:
    """
    Core backtest engine.

    Args:
        events: list of detected macro events
        close: close price DataFrame
        returns: returns DataFrame
        hold_days: number of trading days to hold
        vix_filter: if set, only trade when VIX < this value
        contrarian: if True, reverse the sector mapping (fade reaction)
        combined_signal: if True, require 2+ events in same direction within 30 days
        options_mode: if True, simulate call/put options with BS pricing
        variant_name: label for this variant
    """

    capital = INITIAL_CAPITAL
    equity_curve = []
    positions: List[Position] = []
    active_positions: List[Position] = []
    trade_log = []

    trading_days = close.index.tolist()
    date_to_idx = {d: i for i, d in enumerate(trading_days)}

    # Pre-filter events for combined signal mode
    if combined_signal:
        events = _filter_combined_events(events)

    for event in events:
        event_ts = pd.Timestamp(event.date)
        if event_ts not in date_to_idx:
            continue

        idx = date_to_idx[event_ts]

        # Entry is next trading day after event
        entry_idx = idx + 1
        if entry_idx >= len(trading_days):
            continue
        entry_date = trading_days[entry_idx]

        # Exit date
        exit_idx = min(entry_idx + hold_days, len(trading_days) - 1)
        exit_date = trading_days[exit_idx]

        # VIX filter
        if vix_filter is not None and "VIX" in close.columns:
            vix_val = close.loc[entry_date, "VIX"]
            if pd.notna(vix_val) and vix_val >= vix_filter:
                continue

        # Get sector mapping
        key = (event.event_type, event.direction)
        if key not in SECTOR_MAP:
            continue

        sector_trades = SECTOR_MAP[key]

        if contrarian:
            # Reverse directions
            sector_trades = [(t, -w) for t, w in sector_trades]

        # Check max concurrent positions
        # Count active positions at entry_date
        active_at_entry = sum(
            1 for p in positions
            if p.exit_date is not None
            and pd.Timestamp(p.entry_date) <= entry_date <= pd.Timestamp(p.exit_date)
        )

        available_slots = MAX_CONCURRENT - active_at_entry
        if available_slots <= 0:
            continue

        # Limit trades to available slots
        sector_trades = sector_trades[:available_slots]

        for ticker, weight in sector_trades:
            if ticker not in close.columns:
                continue

            entry_price = close.loc[entry_date, ticker]
            exit_price = close.loc[exit_date, ticker]

            if pd.isna(entry_price) or pd.isna(exit_price):
                continue

            direction = 1 if weight > 0 else -1

            if options_mode:
                # Simulate option trade
                pnl = _simulate_option_trade(
                    entry_price, exit_price, direction, hold_days, capital
                )
            else:
                # Apply slippage
                if direction == 1:
                    adj_entry = entry_price * (1 + SLIPPAGE_PCT)
                    adj_exit = exit_price * (1 - SLIPPAGE_PCT)
                else:
                    adj_entry = entry_price * (1 - SLIPPAGE_PCT)
                    adj_exit = exit_price * (1 + SLIPPAGE_PCT)

                # Size: equal weight allocation of current capital
                alloc = capital / MAX_CONCURRENT
                shares = alloc / adj_entry

                pnl = direction * shares * (adj_exit - adj_entry)

            capital += pnl

            pos = Position(
                ticker=ticker,
                entry_date=str(entry_date.date()),
                entry_price=float(entry_price),
                direction=direction,
                shares=float(shares) if not options_mode else 0,
                exit_date=str(exit_date.date()),
                exit_price=float(exit_price),
                pnl=float(pnl),
                event_type=event.event_type,
                event_direction=event.direction,
            )
            positions.append(pos)

            trade_log.append({
                "entry_date": pos.entry_date,
                "exit_date": pos.exit_date,
                "ticker": ticker,
                "direction": "LONG" if direction == 1 else "SHORT",
                "pnl": round(pnl, 2),
                "event": f"{event.event_type}_{event.direction}",
            })

        equity_curve.append({"date": str(entry_date.date()), "equity": round(capital, 2)})

    # Build daily equity curve from positions
    daily_equity = _build_daily_equity(positions, close, INITIAL_CAPITAL)

    # Compute metrics
    metrics = _compute_metrics(daily_equity, positions, close, variant_name)

    return {
        "variant": variant_name,
        "config": {
            "hold_days": hold_days,
            "vix_filter": vix_filter,
            "contrarian": contrarian,
            "combined_signal": combined_signal,
            "options_mode": options_mode,
        },
        "metrics": metrics,
        "n_trades": len(positions),
        "final_equity": round(capital, 2),
        "trade_log": trade_log[:50],  # first 50 for brevity
    }


def _filter_combined_events(events: List[MacroEvent]) -> List[MacroEvent]:
    """Keep only events where 2+ events point same direction within 30 days."""
    # Map directions to bullish/bearish
    direction_map = {
        ("CPI", "cool"): "bullish",
        ("CPI", "hot"): "bearish",
        ("NFP", "strong"): "bullish",
        ("NFP", "weak"): "bearish",
        ("FOMC", "dovish"): "bullish",
        ("FOMC", "hawkish"): "bearish",
    }

    filtered = []
    for i, ev in enumerate(events):
        ev_dir = direction_map.get((ev.event_type, ev.direction))
        if ev_dir is None:
            continue

        ev_date = pd.Timestamp(ev.date)

        # Count same-direction events within prior 30 days
        count = 0
        for j, other in enumerate(events):
            if i == j:
                continue
            other_dir = direction_map.get((other.event_type, other.direction))
            other_date = pd.Timestamp(other.date)
            if other_dir == ev_dir and 0 < (ev_date - other_date).days <= 30:
                count += 1

        if count >= 1:  # 2+ total (this event + 1 prior)
            filtered.append(ev)

    print(f"Combined signal filter: {len(events)} -> {len(filtered)} events")
    return filtered


def _simulate_option_trade(
    entry_price: float, exit_price: float, direction: int,
    hold_days: int, capital: float,
) -> float:
    """
    Simulate option trade with BS pricing, 30% haircut, $0.65/contract, 5% spread.
    """
    # Simple BS-like option pricing
    sigma = 0.30  # assumed IV
    T = hold_days / 252

    # ATM option delta ~ 0.5
    # Option price ~ entry_price * sigma * sqrt(T) * 0.4 (simplified BS ATM)
    option_price = entry_price * sigma * np.sqrt(T) * 0.4

    # 30% haircut on theoretical price
    option_price *= 0.70

    # 5% bid-ask spread cost
    spread_cost = option_price * 0.05

    # Commission
    commission_per_contract = 0.65

    # Size: allocate capital / MAX_CONCURRENT
    alloc = capital / MAX_CONCURRENT
    n_contracts = max(1, int(alloc / (option_price * 100)))

    # Price move
    price_move = (exit_price - entry_price) / entry_price

    if direction == 1:  # call
        # Approximate P&L: delta * price_move * entry_price * 100 * n_contracts - costs
        delta = 0.50
        option_pnl = delta * price_move * entry_price * 100 * n_contracts
    else:  # put
        delta = 0.50
        option_pnl = delta * (-price_move) * entry_price * 100 * n_contracts

    # Theta decay (simplified)
    theta_cost = option_price * 100 * n_contracts * (hold_days / (T * 252)) * 0.3

    # Total costs
    total_cost = (spread_cost * 100 * n_contracts * 2  # entry + exit spread
                  + commission_per_contract * n_contracts * 2  # entry + exit
                  + theta_cost)

    pnl = option_pnl - total_cost

    # Cap loss at premium paid
    max_loss = -(option_price * 100 * n_contracts + total_cost)
    pnl = max(pnl, max_loss)

    return float(pnl)


def _build_daily_equity(
    positions: List[Position], close: pd.DataFrame, initial_capital: float
) -> pd.Series:
    """Build daily equity curve from trade P&L."""
    if not positions:
        idx = close.loc[OOT_START:OOT_END].index
        return pd.Series(initial_capital, index=idx)

    idx = close.loc[OOT_START:OOT_END].index
    equity = pd.Series(initial_capital, index=idx, dtype=float)

    # Add P&L on exit dates
    for pos in positions:
        if pos.pnl is not None and pos.exit_date is not None:
            exit_ts = pd.Timestamp(pos.exit_date)
            if exit_ts in equity.index:
                # Spread the P&L across exit date
                equity.loc[exit_ts:] += pos.pnl

    return equity


def _compute_metrics(
    equity: pd.Series, positions: List[Position],
    close: pd.DataFrame, variant_name: str,
) -> Dict:
    """Compute performance metrics including 5-gate validation."""

    daily_returns = equity.pct_change().dropna()
    daily_returns = daily_returns.replace([np.inf, -np.inf], 0).fillna(0)

    n_trades = len(positions)

    if n_trades == 0 or len(daily_returns) == 0:
        return {
            "sharpe": 0, "sortino": 0, "profit_factor": 0,
            "win_rate": 0, "max_dd": 0, "total_return_pct": 0,
            "n_trades": 0, "cagr": 0,
            "gates": {"sharpe": False, "perm_p": False, "regime_gap": False,
                      "max_dd": False, "min_trades": False, "passed": False},
        }

    # Basic metrics
    mean_ret = daily_returns.mean()
    std_ret = daily_returns.std()
    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0

    # Sortino
    downside = daily_returns[daily_returns < 0]
    downside_std = downside.std() if len(downside) > 0 else 1e-10
    sortino = (mean_ret / downside_std * np.sqrt(252)) if downside_std > 0 else 0

    # Win rate
    pnls = [p.pnl for p in positions if p.pnl is not None]
    wins = sum(1 for p in pnls if p > 0)
    win_rate = wins / len(pnls) if pnls else 0

    # Profit factor
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (99.0 if gross_profit > 0 else 0)

    # Max drawdown
    peak = equity.expanding().max()
    dd = (equity - peak) / peak
    max_dd = dd.min()

    # Total return
    total_return_pct = (equity.iloc[-1] / equity.iloc[0] - 1) * 100

    # CAGR
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = ((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) * 100 if years > 0 else 0

    # Regime analysis
    spy_close = close["SPY"] if "SPY" in close.columns else None
    regime_sharpes = _regime_analysis(positions, close, spy_close)

    bull_sharpe = regime_sharpes.get("bull_sharpe", 0)
    bear_sharpe = regime_sharpes.get("bear_sharpe", 0)
    max_regime = max(abs(bull_sharpe), abs(bear_sharpe))
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_regime if max_regime > 0 else 0

    # Permutation test
    perm_p = _permutation_test(positions, close, n_permutations=N_PERMUTATIONS)

    # 5-gate validation
    gates = {
        "sharpe": sharpe > SHARPE_THRESHOLD,
        "perm_p": perm_p < PERM_P_THRESHOLD,
        "regime_gap": regime_gap < REGIME_GAP_THRESHOLD,
        "max_dd": max_dd > MAX_DD_THRESHOLD,
        "min_trades": n_trades >= MIN_TRADES,
    }
    gates["passed"] = all(gates.values())

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(profit_factor, 3),
        "win_rate": round(win_rate, 3),
        "max_dd": round(max_dd, 4),
        "total_return_pct": round(total_return_pct, 2),
        "n_trades": n_trades,
        "cagr": round(cagr, 2),
        "avg_pnl": round(np.mean(pnls), 2) if pnls else 0,
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "regime_gap": round(regime_gap, 3),
        "perm_p_value": round(perm_p, 4),
        "gates": gates,
    }


def _regime_analysis(
    positions: List[Position], close: pd.DataFrame,
    spy_close: Optional[pd.Series],
) -> Dict:
    """Split trades into bull/bear regime based on SPY>200SMA."""
    if spy_close is None or len(positions) == 0:
        return {"bull_sharpe": 0, "bear_sharpe": 0}

    sma200 = spy_close.rolling(200).mean()

    bull_pnls = []
    bear_pnls = []

    for pos in positions:
        if pos.pnl is None:
            continue
        entry_ts = pd.Timestamp(pos.entry_date)
        if entry_ts in sma200.index and pd.notna(sma200.loc[entry_ts]):
            spy_val = spy_close.loc[entry_ts] if entry_ts in spy_close.index else None
            sma_val = sma200.loc[entry_ts]
            if spy_val is not None and pd.notna(spy_val):
                if spy_val > sma_val:
                    bull_pnls.append(pos.pnl)
                else:
                    bear_pnls.append(pos.pnl)

    def _sharpe_from_pnls(pnls):
        if len(pnls) < 3:
            return 0
        arr = np.array(pnls)
        mean = arr.mean()
        std = arr.std()
        if std == 0:
            return 0
        # Annualize: assume ~1 trade/week on average
        return (mean / std) * np.sqrt(52)

    return {
        "bull_sharpe": _sharpe_from_pnls(bull_pnls),
        "bear_sharpe": _sharpe_from_pnls(bear_pnls),
        "bull_trades": len(bull_pnls),
        "bear_trades": len(bear_pnls),
    }


def _permutation_test(
    positions: List[Position], close: pd.DataFrame,
    n_permutations: int = 1000,
) -> float:
    """
    Permutation test: shuffle event-to-sector mapping.
    On each permutation, for every trade, randomly pick sector ETFs
    instead of the prescribed ones.
    """
    if len(positions) == 0:
        return 1.0

    actual_total_pnl = sum(p.pnl for p in positions if p.pnl is not None)

    available_tickers = [t for t in UNIVERSE if t in close.columns]
    if not available_tickers:
        return 1.0

    rng = np.random.RandomState(42)
    count_better = 0

    for _ in range(n_permutations):
        perm_pnl = 0
        for pos in positions:
            if pos.pnl is None:
                continue
            # Random sector instead of prescribed one
            rand_ticker = rng.choice(available_tickers)
            entry_ts = pd.Timestamp(pos.entry_date)
            exit_ts = pd.Timestamp(pos.exit_date)

            if entry_ts not in close.index or exit_ts not in close.index:
                continue

            entry_p = close.loc[entry_ts, rand_ticker]
            exit_p = close.loc[exit_ts, rand_ticker]

            if pd.isna(entry_p) or pd.isna(exit_p) or entry_p == 0:
                continue

            ret = (exit_p - entry_p) / entry_p
            # Apply same direction as original trade
            pnl = pos.direction * ret * abs(pos.shares * pos.entry_price if pos.shares else INITIAL_CAPITAL / MAX_CONCURRENT)

            # Apply slippage
            pnl -= abs(pnl) * SLIPPAGE_PCT * 2

            perm_pnl += pnl

        if perm_pnl >= actual_total_pnl:
            count_better += 1

    return count_better / n_permutations


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("MACRO EVENT SURPRISE SECTOR ROTATION BACKTEST")
    print("=" * 70)

    # Download data
    close, returns = download_data()

    # Detect events
    events = detect_events(returns)

    if len(events) == 0:
        print("ERROR: No macro events detected. Check data.")
        sys.exit(1)

    # Print event summary
    print("\nEvent breakdown:")
    for etype in ["CPI", "NFP", "FOMC"]:
        type_events = [e for e in events if e.event_type == etype]
        directions = {}
        for e in type_events:
            directions[e.direction] = directions.get(e.direction, 0) + 1
        print(f"  {etype}: {len(type_events)} total — {directions}")

    # Run 6 variants
    results = {}

    print("\n" + "=" * 70)
    print("VARIANT A: Single event, hold 5 days, equity only")
    print("=" * 70)
    results["A"] = run_backtest(events, close, returns, hold_days=5, variant_name="A")
    _print_summary(results["A"])

    print("\n" + "=" * 70)
    print("VARIANT B: Single event, hold 20 days")
    print("=" * 70)
    results["B"] = run_backtest(events, close, returns, hold_days=20, variant_name="B")
    _print_summary(results["B"])

    print("\n" + "=" * 70)
    print("VARIANT C: Event + VIX<25 filter, hold 10 days")
    print("=" * 70)
    results["C"] = run_backtest(events, close, returns, hold_days=10, vix_filter=25, variant_name="C")
    _print_summary(results["C"])

    print("\n" + "=" * 70)
    print("VARIANT D: Contrarian (fade reaction), hold 5 days")
    print("=" * 70)
    results["D"] = run_backtest(events, close, returns, hold_days=5, contrarian=True, variant_name="D")
    _print_summary(results["D"])

    print("\n" + "=" * 70)
    print("VARIANT E: Combined signal (2+ events same direction), hold 10 days")
    print("=" * 70)
    results["E"] = run_backtest(events, close, returns, hold_days=10, combined_signal=True, variant_name="E")
    _print_summary(results["E"])

    print("\n" + "=" * 70)
    print("VARIANT F: Options (BS pricing, 30% haircut, $0.65/contract)")
    print("=" * 70)
    results["F"] = run_backtest(events, close, returns, hold_days=10, options_mode=True, variant_name="F")
    _print_summary(results["F"])

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Var':<5} {'Trades':<7} {'Return%':<10} {'Sharpe':<8} {'Sortino':<9} "
          f"{'PF':<7} {'WR':<7} {'MaxDD':<8} {'PermP':<7} {'Pass?':<6}")
    print("-" * 80)
    for v in ["A", "B", "C", "D", "E", "F"]:
        m = results[v]["metrics"]
        g = m["gates"]
        passed = "YES" if g["passed"] else "NO"
        print(f"{v:<5} {m['n_trades']:<7} {m['total_return_pct']:<10} {m['sharpe']:<8} "
              f"{m['sortino']:<9} {m['profit_factor']:<7} {m['win_rate']:<7} "
              f"{m['max_dd']:<8} {m['perm_p_value']:<7} {passed:<6}")

    # Gate detail
    print("\n5-GATE VALIDATION DETAIL:")
    for v in ["A", "B", "C", "D", "E", "F"]:
        m = results[v]["metrics"]
        g = m["gates"]
        flags = []
        if not g["sharpe"]: flags.append(f"Sharpe={m['sharpe']}<{SHARPE_THRESHOLD}")
        if not g["perm_p"]: flags.append(f"PermP={m['perm_p_value']}>{PERM_P_THRESHOLD}")
        if not g["regime_gap"]: flags.append(f"RegimeGap={m['regime_gap']}>{REGIME_GAP_THRESHOLD}")
        if not g["max_dd"]: flags.append(f"MaxDD={m['max_dd']}<{MAX_DD_THRESHOLD}")
        if not g["min_trades"]: flags.append(f"Trades={m['n_trades']}<{MIN_TRADES}")
        status = "PASS" if g["passed"] else f"FAIL: {', '.join(flags)}"
        print(f"  Variant {v}: {status}")

    # Save results
    output = {
        "strategy": "macro_event_surprise_sector_rotation",
        "run_date": str(dt.datetime.now()),
        "oot_period": f"{OOT_START} to {OOT_END}",
        "initial_capital": INITIAL_CAPITAL,
        "n_events_detected": len(events),
        "event_breakdown": {
            "CPI": sum(1 for e in events if e.event_type == "CPI"),
            "NFP": sum(1 for e in events if e.event_type == "NFP"),
            "FOMC": sum(1 for e in events if e.event_type == "FOMC"),
        },
        "variants": {v: results[v] for v in ["A", "B", "C", "D", "E", "F"]},
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {RESULTS_PATH}")

    # Final verdict
    passed = [v for v in results if results[v]["metrics"]["gates"]["passed"]]
    if passed:
        print(f"\nVERDICT: {len(passed)} variant(s) passed all 5 gates: {passed}")
    else:
        print("\nVERDICT: No variants passed all 5 gates.")


def _print_summary(result: Dict):
    m = result["metrics"]
    print(f"  Trades: {m['n_trades']}  |  Return: {m['total_return_pct']}%  |  "
          f"Sharpe: {m['sharpe']}  |  Sortino: {m['sortino']}")
    print(f"  PF: {m['profit_factor']}  |  WR: {m['win_rate']}  |  "
          f"MaxDD: {m['max_dd']}  |  AvgPnL: ${m['avg_pnl']}")
    print(f"  Bull Sharpe: {m['bull_sharpe']}  |  Bear Sharpe: {m['bear_sharpe']}  |  "
          f"Regime Gap: {m['regime_gap']}")
    print(f"  Perm test p-value: {m['perm_p_value']}")
    g = m["gates"]
    print(f"  5-Gate: {'PASS' if g['passed'] else 'FAIL'} "
          f"[Sharpe:{g['sharpe']} PermP:{g['perm_p']} Regime:{g['regime_gap']} "
          f"DD:{g['max_dd']} Trades:{g['min_trades']}]")


if __name__ == "__main__":
    main()
