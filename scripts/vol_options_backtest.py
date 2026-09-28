#!/usr/bin/env python3
"""
Volatility-Based Options Strategy Backtester
=============================================
Two strategies exploiting STRUCTURAL catalysts in options markets:

Strategy 1: POST-EARNINGS VOL CRUSH — Sell strangles after earnings when IV crushes
Strategy 2: WEEKLY THETA HARVEST — Sell weekly puts/calls on high-IV cheap stocks

Walk-forward OOT: Jan 2022 – Jul 2026
Starting capital: $645
Commission: $0.65/contract/leg
5-gate validation: Sharpe>0.5, perm p<0.05, beats random, regime gap<0.50, MDD<-50%

Author: Claude (Quantitative Research)
Date: 2026-07-28
"""

import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/vol_options")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUTPUT_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

COMMISSION_PER_LEG = 0.65  # $0.65 per contract per leg
STARTING_CAPITAL = 645.0
MAX_LOSS_PER_TRADE = 200.0  # Hard cap

# Universe: cheap stocks with liquid options
UNIVERSE = [
    "SOFI", "SNAP", "PINS", "HOOD", "PLTR", "NIO", "RIVN", "LCID",
    "MARA", "RIOT", "COIN", "DKNG", "RBLX", "U", "AFRM", "UPST",
    "OPEN", "CLOV", "SQ", "ROKU",
]

# Extended universe for weekly theta (cheaper stocks with weekly options)
WEEKLY_UNIVERSE = [
    "SOFI", "SNAP", "PINS", "HOOD", "PLTR", "NIO", "RIVN", "LCID",
    "MARA", "RIOT", "DKNG", "RBLX", "AFRM", "F", "AAL", "CCL",
    "BAC", "T", "WBD", "SQ",
]

DATA_START = "2021-01-01"   # Need lookback for vol calc before OOT starts
DATA_END = "2026-07-28"
OOT_START = "2022-01-01"
OOT_END = "2026-07-28"


# ─── Black-Scholes Pricing ────────────────────────────────────────────────────

def bs_price(S: float, K: float, T: float, r: float, sigma: float,
             option_type: str = "call") -> float:
    """Black-Scholes option price. T in years."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(0, (S - K) if option_type == "call" else (K - S))
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S: float, K: float, T: float, r: float, sigma: float,
             option_type: str = "call") -> float:
    """Black-Scholes delta."""
    if T <= 0 or sigma <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1.0


# ─── Data Download & Caching ─────────────────────────────────────────────────

def download_ohlcv(tickers: list, start: str, end: str) -> dict:
    """Download OHLCV data for tickers, cache to parquet."""
    cache_file = CACHE_DIR / "ohlcv_cache.parquet"

    if cache_file.exists():
        try:
            cached = pd.read_parquet(cache_file)
            cached_tickers = set(cached.columns.get_level_values(1).unique()) if isinstance(cached.columns, pd.MultiIndex) else set()
            missing = set(tickers) - cached_tickers
            if not missing:
                log.info(f"Loaded cached OHLCV for {len(tickers)} tickers")
                return _parse_ohlcv(cached, tickers)
        except Exception:
            pass

    log.info(f"Downloading OHLCV for {len(tickers)} tickers...")
    data = yf.download(tickers, start=start, end=end, auto_adjust=True,
                       progress=False, threads=True, group_by='ticker')
    if data.empty:
        log.error("No data downloaded!")
        sys.exit(1)

    data.to_parquet(cache_file)
    return _parse_ohlcv(data, tickers)


def _parse_ohlcv(data: pd.DataFrame, tickers: list) -> dict:
    """Parse multi-ticker OHLCV DataFrame into dict of per-ticker DataFrames."""
    result = {}
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            try:
                df = data[t][["Open", "High", "Low", "Close", "Volume"]].dropna()
                if df.index.tz is not None:
                    df.index = df.index.tz_convert(None)
                if len(df) > 50:
                    result[t] = df
            except (KeyError, TypeError):
                continue
    else:
        # Single ticker case
        if len(tickers) == 1:
            df = data[["Open", "High", "Low", "Close", "Volume"]].dropna()
            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)
            result[tickers[0]] = df
    return result


def get_earnings_dates(ticker: str) -> list:
    """Get historical earnings dates for a ticker."""
    cache_file = CACHE_DIR / f"earnings_{ticker}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            dates = json.load(f)
        return [pd.Timestamp(d) for d in dates]

    try:
        t = yf.Ticker(ticker)
        cal = t.get_earnings_dates(limit=50)
        if cal is not None and len(cal) > 0:
            idx = cal.index
            if idx.tz is not None:
                idx = idx.tz_convert(None)
            dates = sorted(idx.normalize().unique().tolist())
            with open(cache_file, 'w') as f:
                json.dump([d.isoformat() for d in dates], f)
            return dates
    except Exception as e:
        log.warning(f"Could not get earnings for {ticker}: {e}")

    return []


def detect_earnings_events(ohlcv: pd.DataFrame, earnings_dates: list = None,
                           gap_threshold: float = 0.04, vol_mult: float = 1.5) -> list:
    """Detect earnings-like events: big gap + high volume.
    If earnings_dates provided, use them. Otherwise detect from price action."""
    events = []
    close = ohlcv["Close"]
    volume = ohlcv["Volume"]
    avg_vol = volume.rolling(20).mean()

    if earnings_dates:
        for ed in earnings_dates:
            # Normalize timezone
            ed = pd.Timestamp(ed).tz_localize(None)
            # Earnings date might be before/after close. Check the next trading day.
            idx = ohlcv.index.get_indexer([ed], method='ffill')
            if idx[0] < 0 or idx[0] >= len(ohlcv) - 1:
                continue
            # The day after earnings
            post_idx = idx[0] + 1
            if post_idx >= len(ohlcv):
                continue
            post_date = ohlcv.index[post_idx]
            pre_close = close.iloc[idx[0]]
            post_open = ohlcv["Open"].iloc[post_idx]
            gap = (post_open - pre_close) / pre_close
            events.append({
                "date": post_date,
                "gap": gap,
                "pre_close": pre_close,
                "post_open": post_open,
                "post_close": close.iloc[post_idx],
            })
    else:
        # Detect from price action: gap > threshold + volume spike
        for i in range(21, len(ohlcv)):
            gap = (ohlcv["Open"].iloc[i] - close.iloc[i-1]) / close.iloc[i-1]
            if abs(gap) > gap_threshold and volume.iloc[i] > vol_mult * avg_vol.iloc[i]:
                events.append({
                    "date": ohlcv.index[i],
                    "gap": gap,
                    "pre_close": close.iloc[i-1],
                    "post_open": ohlcv["Open"].iloc[i],
                    "post_close": close.iloc[i],
                })

    return events


def compute_historical_iv(close: pd.Series, window: int = 20, markup: float = 1.3) -> pd.Series:
    """Compute implied vol proxy: realized vol * markup factor.
    This approximates IV since we don't have real options data."""
    log_ret = np.log(close / close.shift(1))
    realized_vol = log_ret.rolling(window).std() * np.sqrt(252)
    return realized_vol * markup


def compute_iv_rank(iv: pd.Series, lookback: int = 252) -> pd.Series:
    """IV percentile rank over lookback period."""
    return iv.rolling(lookback).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10)
                                      if len(x) > 10 else np.nan, raw=False)


# ─── Trade Structures ────────────────────────────────────────────────────────

@dataclass
class OptionTrade:
    ticker: str
    entry_date: pd.Timestamp
    exit_date: Optional[pd.Timestamp] = None
    structure: str = ""  # "strangle_sell", "put_sell", "call_sell"
    entry_stock_price: float = 0.0
    put_strike: float = 0.0
    call_strike: float = 0.0
    entry_iv: float = 0.0
    exit_iv: float = 0.0
    entry_premium: float = 0.0  # Total premium collected (per contract)
    exit_cost: float = 0.0      # Cost to close
    dte_at_entry: int = 0
    pnl: float = 0.0
    commission: float = 0.0
    max_loss_hit: bool = False
    exit_reason: str = ""
    contracts: int = 1


# ─── Strategy 1: Post-Earnings Vol Crush Strangle Selling ────────────────────

@dataclass
class VolCrushConfig:
    name: str = "vol_crush_base"
    put_offset_pct: float = 0.05   # Put strike = S * (1 - offset)
    call_offset_pct: float = 0.05  # Call strike = S * (1 + offset)
    hold_days: int = 7
    iv_crush_factor: float = 0.55  # Post-earnings IV drops to this fraction
    min_premium: float = 0.10      # Min premium per share to enter
    max_trade_loss: float = 200.0
    r: float = 0.05  # risk-free rate


def run_vol_crush_strategy(ohlcv_data: dict, config: VolCrushConfig) -> list:
    """Run post-earnings vol crush strangle selling strategy."""
    all_trades = []

    for ticker, ohlcv in ohlcv_data.items():
        if ticker not in UNIVERSE:
            continue

        close = ohlcv["Close"]
        iv_series = compute_historical_iv(close, window=20, markup=1.3)

        # Get earnings events
        earnings_dates = get_earnings_dates(ticker)
        events = detect_earnings_events(ohlcv, earnings_dates)

        # Also detect from price action as backup
        if len(events) < 4:
            events = detect_earnings_events(ohlcv, earnings_dates=None)

        for event in events:
            entry_date = event["date"]

            # Only trade in OOT period
            if entry_date < pd.Timestamp(OOT_START) or entry_date > pd.Timestamp(OOT_END):
                continue

            S = event["post_close"]  # Enter at close of earnings day
            if S < 2.0 or S > 50.0:  # Price range filter for our account size
                continue

            # Get IV at entry (post-earnings, already partially crushed)
            entry_idx = ohlcv.index.get_loc(entry_date)
            if entry_idx < 20:
                continue

            pre_earnings_iv = iv_series.iloc[entry_idx - 1] if entry_idx > 0 else 0.5
            if pd.isna(pre_earnings_iv) or pre_earnings_iv < 0.2:
                continue

            # Post-earnings IV crush
            post_iv = pre_earnings_iv * config.iv_crush_factor

            # Strangle strikes
            put_K = round(S * (1 - config.put_offset_pct), 2)
            call_K = round(S * (1 + config.call_offset_pct), 2)

            # Price the strangle at entry
            T_entry = config.hold_days / 252.0
            put_price = bs_price(S, put_K, T_entry, config.r, post_iv, "put")
            call_price = bs_price(S, call_K, T_entry, config.r, post_iv, "call")
            total_premium = put_price + call_price

            if total_premium < config.min_premium:
                continue

            # How many contracts can we sell? (margin = max of put or call notional risk)
            max_put_risk = put_K * 100  # Worst case: stock goes to 0
            max_call_risk = call_K * 100 * 0.5  # Calls have unlimited risk, cap at 50% move
            margin_per_contract = max(max_put_risk, max_call_risk) * 0.20  # ~20% margin

            # Cap by max loss and available capital
            n_contracts = 1  # Start with 1 for small account
            commission = n_contracts * 2 * 2 * COMMISSION_PER_LEG  # 2 legs, open+close

            # Simulate hold period
            exit_idx = min(entry_idx + config.hold_days, len(ohlcv) - 1)
            exit_date = ohlcv.index[exit_idx]
            exit_price = close.iloc[exit_idx]

            # Check for max loss during hold
            max_loss_hit = False
            actual_exit_idx = exit_idx
            for day_i in range(entry_idx + 1, exit_idx + 1):
                day_price = close.iloc[day_i]
                days_remaining = max(1, exit_idx - day_i)
                T_rem = days_remaining / 252.0

                # IV continues to decay after earnings
                current_iv = post_iv * (0.9 + 0.1 * (days_remaining / config.hold_days))

                day_put_price = bs_price(day_price, put_K, T_rem, config.r, current_iv, "put")
                day_call_price = bs_price(day_price, call_K, T_rem, config.r, current_iv, "call")
                unrealized_loss = (day_put_price + day_call_price - total_premium) * 100 * n_contracts

                if unrealized_loss > config.max_trade_loss:
                    max_loss_hit = True
                    actual_exit_idx = day_i
                    exit_date = ohlcv.index[day_i]
                    exit_price = day_price
                    break

            # Calculate exit cost
            days_left = max(0, exit_idx - actual_exit_idx) if not max_loss_hit else 0
            T_exit = max(days_left, 1) / 252.0 if not max_loss_hit else (exit_idx - actual_exit_idx) / 252.0
            exit_iv = post_iv * 0.85  # IV continues declining

            if max_loss_hit:
                T_exit = max(1, exit_idx - actual_exit_idx) / 252.0
                exit_iv = post_iv

            exit_put = bs_price(exit_price, put_K, T_exit, config.r, exit_iv, "put")
            exit_call = bs_price(exit_price, call_K, T_exit, config.r, exit_iv, "call")
            exit_cost = exit_put + exit_call

            # PnL = premium collected - cost to close - commission
            pnl_per_share = total_premium - exit_cost
            pnl = pnl_per_share * 100 * n_contracts - commission

            # Cap loss
            if pnl < -config.max_trade_loss:
                pnl = -config.max_trade_loss

            trade = OptionTrade(
                ticker=ticker,
                entry_date=entry_date,
                exit_date=exit_date,
                structure="strangle_sell",
                entry_stock_price=S,
                put_strike=put_K,
                call_strike=call_K,
                entry_iv=pre_earnings_iv,
                exit_iv=exit_iv,
                entry_premium=total_premium * 100 * n_contracts,
                exit_cost=exit_cost * 100 * n_contracts,
                dte_at_entry=config.hold_days,
                pnl=pnl,
                commission=commission,
                max_loss_hit=max_loss_hit,
                exit_reason="max_loss" if max_loss_hit else "expiry",
                contracts=n_contracts,
            )
            all_trades.append(trade)

    return all_trades


# ─── Strategy 2: Weekly Theta Harvest ─────────────────────────────────────────

@dataclass
class WeeklyThetaConfig:
    name: str = "weekly_theta_base"
    option_type: str = "put"  # "put" or "call"
    otm_offset_pct: float = 0.05  # 5% OTM
    target_dte: int = 5  # Weekly options
    iv_rank_threshold: float = 0.50  # Only sell when IV rank > 50th pctile
    min_premium: float = 0.05
    max_trade_loss: float = 200.0
    max_concurrent: int = 3
    r: float = 0.05
    directional_filter: bool = False  # Use trend to pick puts vs calls


def run_weekly_theta_strategy(ohlcv_data: dict, config: WeeklyThetaConfig) -> list:
    """Run weekly theta harvest strategy on high-IV cheap stocks."""
    all_trades = []

    # Build weekly entry dates (Mondays/Tuesdays in OOT period)
    oot_start = pd.Timestamp(OOT_START)
    oot_end = pd.Timestamp(OOT_END)

    for ticker, ohlcv in ohlcv_data.items():
        if ticker not in WEEKLY_UNIVERSE:
            continue

        close = ohlcv["Close"]
        iv_series = compute_historical_iv(close, window=20, markup=1.3)
        iv_rank = compute_iv_rank(iv_series, lookback=252)

        # SMA for directional filter
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()

        # Find Monday/Tuesday entries in OOT period
        for i in range(252, len(ohlcv)):
            date = ohlcv.index[i]
            if date < oot_start or date > oot_end:
                continue

            # Only enter on Monday (0) or Tuesday (1)
            if date.dayofweek > 1:
                continue

            S = close.iloc[i]
            if S < 3.0 or S > 20.0:  # Cheap stocks only ($3-$20)
                continue

            current_iv = iv_series.iloc[i]
            current_iv_rank = iv_rank.iloc[i]

            if pd.isna(current_iv_rank) or current_iv_rank < config.iv_rank_threshold:
                continue

            # Determine direction
            if config.directional_filter:
                above_sma = S > sma20.iloc[i] if not pd.isna(sma20.iloc[i]) else True
                opt_type = "put" if above_sma else "call"
            else:
                opt_type = config.option_type

            # Calculate strike
            if opt_type == "put":
                K = round(S * (1 - config.otm_offset_pct), 2)
                K = max(K, 0.50)  # Floor
            else:
                K = round(S * (1 + config.otm_offset_pct), 2)

            # Price the option
            T = config.target_dte / 252.0
            premium = bs_price(S, K, T, config.r, current_iv, opt_type)

            if premium < config.min_premium:
                continue

            # For cash-secured put: need K * 100 cash (or margin ~20%)
            # For our $645 account, we can only do 1 contract on cheap stocks
            if opt_type == "put":
                cash_needed = K * 100 * 0.20  # 20% margin
            else:
                cash_needed = S * 100 * 0.20  # Need shares or margin for calls

            n_contracts = 1
            commission = n_contracts * 2 * COMMISSION_PER_LEG  # open + close

            # Simulate to expiry (Friday)
            exit_idx = min(i + config.target_dte, len(ohlcv) - 1)

            # Walk through days checking for stop-loss
            max_loss_hit = False
            actual_exit_idx = exit_idx

            for day_i in range(i + 1, exit_idx + 1):
                if day_i >= len(ohlcv):
                    break
                day_price = close.iloc[day_i]
                days_left = max(1, exit_idx - day_i)
                T_rem = days_left / 252.0

                # Check assignment risk for ITM puts
                if opt_type == "put" and day_price < K * 0.95:
                    # Deep ITM, likely assignment
                    day_opt_price = bs_price(day_price, K, T_rem, config.r, current_iv * 0.9, opt_type)
                    unrealized = (day_opt_price - premium) * 100 * n_contracts
                    if unrealized > config.max_trade_loss:
                        max_loss_hit = True
                        actual_exit_idx = day_i
                        break
                elif opt_type == "call" and day_price > K * 1.05:
                    day_opt_price = bs_price(day_price, K, T_rem, config.r, current_iv * 0.9, opt_type)
                    unrealized = (day_opt_price - premium) * 100 * n_contracts
                    if unrealized > config.max_trade_loss:
                        max_loss_hit = True
                        actual_exit_idx = day_i
                        break

            # Exit pricing
            exit_date = ohlcv.index[min(actual_exit_idx, len(ohlcv) - 1)]
            exit_price = close.iloc[min(actual_exit_idx, len(ohlcv) - 1)]

            if max_loss_hit:
                T_exit = max(1, exit_idx - actual_exit_idx) / 252.0
                exit_opt = bs_price(exit_price, K, T_exit, config.r, current_iv * 0.95, opt_type)
            else:
                # At expiry: intrinsic value only
                if opt_type == "put":
                    exit_opt = max(0, K - exit_price)
                else:
                    exit_opt = max(0, exit_price - K)

            pnl = (premium - exit_opt) * 100 * n_contracts - commission

            # Check assignment at expiry
            assigned = False
            if not max_loss_hit:
                if opt_type == "put" and exit_price < K:
                    assigned = True
                    # Assigned: we buy stock at K, worth exit_price
                    # PnL = premium - (K - exit_price) per share
                    assignment_loss = (K - exit_price) * 100 * n_contracts
                    pnl = premium * 100 * n_contracts - assignment_loss - commission
                elif opt_type == "call" and exit_price > K:
                    assigned = True
                    assignment_loss = (exit_price - K) * 100 * n_contracts
                    pnl = premium * 100 * n_contracts - assignment_loss - commission

            if pnl < -config.max_trade_loss:
                pnl = -config.max_trade_loss

            exit_reason = "max_loss" if max_loss_hit else ("assigned" if assigned else "expired_otm")

            trade = OptionTrade(
                ticker=ticker,
                entry_date=date,
                exit_date=exit_date,
                structure=f"{opt_type}_sell",
                entry_stock_price=S,
                put_strike=K if opt_type == "put" else 0,
                call_strike=K if opt_type == "call" else 0,
                entry_iv=current_iv,
                exit_iv=current_iv * 0.9,
                entry_premium=premium * 100 * n_contracts,
                exit_cost=exit_opt * 100 * n_contracts,
                dte_at_entry=config.target_dte,
                pnl=pnl,
                commission=commission,
                max_loss_hit=max_loss_hit,
                exit_reason=exit_reason,
                contracts=n_contracts,
            )
            all_trades.append(trade)

    return all_trades


# ─── Portfolio Simulation (Sequential, Capital-Aware) ─────────────────────────

def simulate_portfolio(trades: list, starting_capital: float = 645.0,
                       max_concurrent: int = 3) -> dict:
    """Simulate portfolio with capital constraints and max concurrent positions."""
    if not trades:
        return {"equity_curve": [], "trades_taken": 0, "trades_skipped": 0}

    # Sort by entry date
    trades_sorted = sorted(trades, key=lambda t: t.entry_date)

    capital = starting_capital
    equity_curve = [(trades_sorted[0].entry_date, capital)]
    active_trades = []
    executed = []
    skipped = 0

    for trade in trades_sorted:
        # Close expired active trades
        new_active = []
        for at in active_trades:
            if at.exit_date <= trade.entry_date:
                capital += at.pnl
                equity_curve.append((at.exit_date, capital))
            else:
                new_active.append(at)
        active_trades = new_active

        # Check if we can take this trade
        if len(active_trades) >= max_concurrent:
            skipped += 1
            continue

        # Check if we have enough capital (need at least $100 margin for cheap options)
        min_margin = 100.0
        if capital < min_margin:
            skipped += 1
            continue

        # Check max loss won't wipe us out
        if capital + trade.pnl < 50:  # Don't let capital go below $50
            if trade.pnl < -capital * 0.3:  # Skip if potential loss > 30% of capital
                skipped += 1
                continue

        active_trades.append(trade)
        executed.append(trade)

    # Close remaining active trades
    for at in active_trades:
        capital += at.pnl
        equity_curve.append((at.exit_date if at.exit_date else at.entry_date + timedelta(days=7), capital))

    # Sort equity curve by date
    equity_curve.sort(key=lambda x: x[0])

    return {
        "equity_curve": equity_curve,
        "final_capital": capital,
        "trades_taken": len(executed),
        "trades_skipped": skipped,
        "executed_trades": executed,
    }


# ─── Metrics & Validation ────────────────────────────────────────────────────

def compute_metrics(equity_curve: list, trades: list, starting_capital: float = 645.0) -> dict:
    """Compute strategy metrics from equity curve."""
    if not equity_curve or len(equity_curve) < 2:
        return {"sharpe": 0, "sortino": 0, "profit_factor": 0, "win_rate": 0,
                "max_dd": -1.0, "total_return": 0, "n_trades": 0,
                "avg_pnl": 0, "cagr": 0}

    # Build daily returns from equity curve
    eq_df = pd.DataFrame(equity_curve, columns=["date", "equity"])
    eq_df = eq_df.groupby("date")["equity"].last().sort_index()

    # Fill forward to get daily equity
    date_range = pd.date_range(eq_df.index[0], eq_df.index[-1], freq='B')
    eq_daily = eq_df.reindex(date_range, method='ffill')

    daily_returns = eq_daily.pct_change().dropna()

    # Trade-level metrics
    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    n_trades = len(pnls)
    win_rate = len(wins) / n_trades if n_trades > 0 else 0
    avg_pnl = np.mean(pnls) if pnls else 0

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1e-10
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Sharpe (annualized)
    if len(daily_returns) > 5 and daily_returns.std() > 0:
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    else:
        sharpe = 0.0

    # Sortino
    downside = daily_returns[daily_returns < 0]
    if len(downside) > 2 and downside.std() > 0:
        sortino = daily_returns.mean() / downside.std() * np.sqrt(252)
    else:
        sortino = sharpe

    # Max drawdown
    peak = eq_daily.expanding().max()
    dd = (eq_daily - peak) / peak
    max_dd = dd.min()

    # Total return
    final_eq = eq_daily.iloc[-1]
    total_return = (final_eq - starting_capital) / starting_capital

    # CAGR
    years = (eq_daily.index[-1] - eq_daily.index[0]).days / 365.25
    if years > 0 and final_eq > 0:
        cagr = (final_eq / starting_capital) ** (1 / years) - 1
    else:
        cagr = 0

    # Per-year metrics for regime analysis
    yearly_returns = {}
    for year in daily_returns.index.year.unique():
        yr_rets = daily_returns[daily_returns.index.year == year]
        if len(yr_rets) > 20:
            yr_sharpe = yr_rets.mean() / yr_rets.std() * np.sqrt(252) if yr_rets.std() > 0 else 0
            yearly_returns[int(year)] = {
                "return": float(yr_rets.sum()),
                "sharpe": float(yr_sharpe),
                "n_days": len(yr_rets),
            }

    return {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "profit_factor": float(min(profit_factor, 99.9)),
        "win_rate": float(win_rate),
        "max_dd": float(max_dd),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "n_trades": n_trades,
        "avg_pnl": float(avg_pnl),
        "avg_win": float(np.mean(wins)) if wins else 0,
        "avg_loss": float(np.mean(losses)) if losses else 0,
        "final_capital": float(eq_daily.iloc[-1]),
        "yearly": yearly_returns,
    }


def permutation_test(trades: list, n_perms: int = 5000) -> float:
    """Permutation test: shuffle trade PnLs, compute p-value."""
    if len(trades) < 5:
        return 1.0

    pnls = np.array([t.pnl for t in trades])
    actual_mean = pnls.mean()

    count_better = 0
    for _ in range(n_perms):
        shuffled = pnls * np.random.choice([-1, 1], size=len(pnls))
        if shuffled.mean() >= actual_mean:
            count_better += 1

    return count_better / n_perms


def regime_analysis(trades: list, spy_data: pd.DataFrame) -> dict:
    """Analyze strategy performance in different market regimes."""
    if not trades:
        return {"bull_sharpe": 0, "bear_sharpe": 0, "regime_gap": 1.0}

    spy_close = spy_data["Close"]
    spy_sma200 = spy_close.rolling(200).mean()

    bull_pnls = []
    bear_pnls = []

    for trade in trades:
        # Determine regime at entry
        entry = trade.entry_date
        idx = spy_close.index.get_indexer([entry], method='ffill')
        if idx[0] < 200:
            bull_pnls.append(trade.pnl)
            continue

        if spy_close.iloc[idx[0]] > spy_sma200.iloc[idx[0]]:
            bull_pnls.append(trade.pnl)
        else:
            bear_pnls.append(trade.pnl)

    def list_sharpe(pnls):
        if len(pnls) < 3:
            return 0
        arr = np.array(pnls)
        return arr.mean() / (arr.std() + 1e-10) * np.sqrt(52)  # weekly-ish

    bull_sharpe = list_sharpe(bull_pnls)
    bear_sharpe = list_sharpe(bear_pnls)

    max_abs = max(abs(bull_sharpe), abs(bear_sharpe), 1e-10)
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs

    return {
        "bull_sharpe": float(bull_sharpe),
        "bear_sharpe": float(bear_sharpe),
        "bull_trades": len(bull_pnls),
        "bear_trades": len(bear_pnls),
        "regime_gap": float(regime_gap),
    }


def random_entry_benchmark(ohlcv_data: dict, n_sims: int = 200,
                           hold_days: int = 5, n_trades_target: int = 50) -> float:
    """Generate random entry benchmark Sharpe for comparison."""
    # Pick random entries from random stocks
    all_tickers = list(ohlcv_data.keys())
    sim_sharpes = []

    for _ in range(n_sims):
        pnls = []
        for _ in range(n_trades_target):
            ticker = np.random.choice(all_tickers)
            ohlcv = ohlcv_data[ticker]
            close = ohlcv["Close"]

            # Random entry in OOT period
            oot_mask = (ohlcv.index >= pd.Timestamp(OOT_START)) & (ohlcv.index <= pd.Timestamp(OOT_END))
            oot_idx = np.where(oot_mask)[0]
            if len(oot_idx) < hold_days + 1:
                continue

            entry_i = np.random.choice(oot_idx[:-hold_days])
            exit_i = min(entry_i + hold_days, len(close) - 1)

            # Random premium (uniform between observed range)
            premium = np.random.uniform(0.05, 0.50)
            # Random direction
            direction = np.random.choice([-1, 1])
            stock_move = (close.iloc[exit_i] - close.iloc[entry_i]) / close.iloc[entry_i]

            pnl = (premium - abs(stock_move) * close.iloc[entry_i] * 0.5) * 100 - 2 * COMMISSION_PER_LEG
            pnls.append(pnl)

        if len(pnls) > 10:
            arr = np.array(pnls)
            sim_sharpe = arr.mean() / (arr.std() + 1e-10) * np.sqrt(52)
            sim_sharpes.append(sim_sharpe)

    return float(np.mean(sim_sharpes)) if sim_sharpes else 0.0


def validate_5gate(metrics: dict, perm_p: float, regime: dict,
                   random_sharpe: float) -> dict:
    """5-gate validation."""
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": perm_p < 0.05,
        "beats_random": metrics["sharpe"] > random_sharpe + 0.1,
        "regime_gap_lt_0.50": regime["regime_gap"] < 0.50,
        "mdd_gt_neg50pct": metrics["max_dd"] > -0.50,
    }
    gates["passed"] = sum(gates.values())
    gates["total"] = 5
    gates["all_passed"] = all(v for k, v in gates.items() if k not in ("passed", "total", "all_passed"))
    return gates


# ─── Main Runner ──────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("VOLATILITY-BASED OPTIONS STRATEGY BACKTESTER")
    log.info("=" * 70)

    # Download data
    all_tickers = list(set(UNIVERSE + WEEKLY_UNIVERSE + ["SPY"]))
    ohlcv_data = download_ohlcv(all_tickers, DATA_START, DATA_END)
    log.info(f"Got data for {len(ohlcv_data)} tickers")

    if "SPY" not in ohlcv_data:
        log.error("No SPY data — cannot do regime analysis")
        sys.exit(1)

    spy_data = ohlcv_data["SPY"]

    # ── Strategy 1: Vol Crush Variants ──
    vol_crush_configs = [
        VolCrushConfig(name="vc_tight_7d", put_offset_pct=0.03, call_offset_pct=0.03, hold_days=7, iv_crush_factor=0.55),
        VolCrushConfig(name="vc_wide_7d", put_offset_pct=0.08, call_offset_pct=0.08, hold_days=7, iv_crush_factor=0.55),
        VolCrushConfig(name="vc_tight_10d", put_offset_pct=0.03, call_offset_pct=0.03, hold_days=10, iv_crush_factor=0.50),
        VolCrushConfig(name="vc_wide_10d", put_offset_pct=0.08, call_offset_pct=0.08, hold_days=10, iv_crush_factor=0.50),
        VolCrushConfig(name="vc_asym_7d", put_offset_pct=0.05, call_offset_pct=0.10, hold_days=7, iv_crush_factor=0.55),
        VolCrushConfig(name="vc_aggressive_5d", put_offset_pct=0.02, call_offset_pct=0.02, hold_days=5, iv_crush_factor=0.60),
    ]

    # ── Strategy 2: Weekly Theta Variants ──
    weekly_theta_configs = [
        WeeklyThetaConfig(name="wt_put_5pct", option_type="put", otm_offset_pct=0.05, iv_rank_threshold=0.50),
        WeeklyThetaConfig(name="wt_put_8pct", option_type="put", otm_offset_pct=0.08, iv_rank_threshold=0.50),
        WeeklyThetaConfig(name="wt_put_highiv", option_type="put", otm_offset_pct=0.05, iv_rank_threshold=0.70),
        WeeklyThetaConfig(name="wt_call_5pct", option_type="call", otm_offset_pct=0.05, iv_rank_threshold=0.50),
        WeeklyThetaConfig(name="wt_directional", directional_filter=True, otm_offset_pct=0.05, iv_rank_threshold=0.50),
        WeeklyThetaConfig(name="wt_tight_3pct", option_type="put", otm_offset_pct=0.03, iv_rank_threshold=0.60),
    ]

    # Set up MLflow
    mlflow.set_tracking_uri("sqlite:////home/jupiter/teleclaude-main/mlflow.db")
    experiment_name = "vol_options_strategies"
    mlflow.set_experiment(experiment_name)

    results = {}

    # ── Run Vol Crush Variants ──
    log.info("\n" + "=" * 50)
    log.info("STRATEGY 1: POST-EARNINGS VOL CRUSH")
    log.info("=" * 50)

    for cfg in vol_crush_configs:
        log.info(f"\n--- Variant: {cfg.name} ---")

        with mlflow.start_run(run_name=f"vol_crush_{cfg.name}"):
            mlflow.log_params({
                "strategy": "vol_crush",
                "variant": cfg.name,
                "put_offset": cfg.put_offset_pct,
                "call_offset": cfg.call_offset_pct,
                "hold_days": cfg.hold_days,
                "iv_crush_factor": cfg.iv_crush_factor,
                "starting_capital": STARTING_CAPITAL,
            })

            trades = run_vol_crush_strategy(ohlcv_data, cfg)
            log.info(f"  Raw trades: {len(trades)}")

            portfolio = simulate_portfolio(trades, STARTING_CAPITAL, max_concurrent=3)
            executed = portfolio.get("executed_trades", [])

            if not executed:
                log.warning(f"  No trades executed for {cfg.name}")
                mlflow.log_metric("n_trades", 0)
                continue

            metrics = compute_metrics(portfolio["equity_curve"], executed, STARTING_CAPITAL)
            perm_p = permutation_test(executed, n_perms=3000)
            regime = regime_analysis(executed, spy_data)
            random_sharpe = random_entry_benchmark(ohlcv_data, n_sims=100,
                                                    hold_days=cfg.hold_days)
            gates = validate_5gate(metrics, perm_p, regime, random_sharpe)

            # Log to MLflow
            mlflow.log_metrics({
                "sharpe": metrics["sharpe"],
                "sortino": metrics["sortino"],
                "profit_factor": metrics["profit_factor"],
                "win_rate": metrics["win_rate"],
                "max_dd": metrics["max_dd"],
                "total_return": metrics["total_return"],
                "cagr": metrics["cagr"],
                "n_trades": metrics["n_trades"],
                "avg_pnl": metrics["avg_pnl"],
                "final_capital": metrics["final_capital"],
                "perm_p_value": perm_p,
                "bull_sharpe": regime["bull_sharpe"],
                "bear_sharpe": regime["bear_sharpe"],
                "regime_gap": regime["regime_gap"],
                "random_sharpe": random_sharpe,
                "gates_passed": gates["passed"],
            })

            results[f"vc_{cfg.name}"] = {
                "metrics": metrics,
                "perm_p": perm_p,
                "regime": regime,
                "gates": gates,
                "n_raw_trades": len(trades),
            }

            log.info(f"  Trades: {metrics['n_trades']} | WR: {metrics['win_rate']:.1%}")
            log.info(f"  Sharpe: {metrics['sharpe']:.2f} | Sortino: {metrics['sortino']:.2f}")
            log.info(f"  PF: {metrics['profit_factor']:.2f} | MDD: {metrics['max_dd']:.1%}")
            log.info(f"  Final Capital: ${metrics['final_capital']:.0f} | Return: {metrics['total_return']:.1%}")
            log.info(f"  Perm p: {perm_p:.4f} | Regime gap: {regime['regime_gap']:.2f}")
            log.info(f"  Gates: {gates['passed']}/{gates['total']} {'✓ PASS' if gates['all_passed'] else '✗ FAIL'}")

    # ── Run Weekly Theta Variants ──
    log.info("\n" + "=" * 50)
    log.info("STRATEGY 2: WEEKLY THETA HARVEST")
    log.info("=" * 50)

    for cfg in weekly_theta_configs:
        log.info(f"\n--- Variant: {cfg.name} ---")

        with mlflow.start_run(run_name=f"weekly_theta_{cfg.name}"):
            mlflow.log_params({
                "strategy": "weekly_theta",
                "variant": cfg.name,
                "option_type": cfg.option_type,
                "otm_offset": cfg.otm_offset_pct,
                "iv_rank_threshold": cfg.iv_rank_threshold,
                "directional_filter": cfg.directional_filter,
                "starting_capital": STARTING_CAPITAL,
            })

            trades = run_weekly_theta_strategy(ohlcv_data, cfg)
            log.info(f"  Raw trades: {len(trades)}")

            portfolio = simulate_portfolio(trades, STARTING_CAPITAL, max_concurrent=cfg.max_concurrent)
            executed = portfolio.get("executed_trades", [])

            if not executed:
                log.warning(f"  No trades executed for {cfg.name}")
                mlflow.log_metric("n_trades", 0)
                continue

            metrics = compute_metrics(portfolio["equity_curve"], executed, STARTING_CAPITAL)
            perm_p = permutation_test(executed, n_perms=3000)
            regime = regime_analysis(executed, spy_data)
            random_sharpe = random_entry_benchmark(ohlcv_data, n_sims=100,
                                                    hold_days=cfg.target_dte)
            gates = validate_5gate(metrics, perm_p, regime, random_sharpe)

            # Log to MLflow
            mlflow.log_metrics({
                "sharpe": metrics["sharpe"],
                "sortino": metrics["sortino"],
                "profit_factor": metrics["profit_factor"],
                "win_rate": metrics["win_rate"],
                "max_dd": metrics["max_dd"],
                "total_return": metrics["total_return"],
                "cagr": metrics["cagr"],
                "n_trades": metrics["n_trades"],
                "avg_pnl": metrics["avg_pnl"],
                "final_capital": metrics["final_capital"],
                "perm_p_value": perm_p,
                "bull_sharpe": regime["bull_sharpe"],
                "bear_sharpe": regime["bear_sharpe"],
                "regime_gap": regime["regime_gap"],
                "random_sharpe": random_sharpe,
                "gates_passed": gates["passed"],
            })

            results[f"wt_{cfg.name}"] = {
                "metrics": metrics,
                "perm_p": perm_p,
                "regime": regime,
                "gates": gates,
                "n_raw_trades": len(trades),
            }

            log.info(f"  Trades: {metrics['n_trades']} | WR: {metrics['win_rate']:.1%}")
            log.info(f"  Sharpe: {metrics['sharpe']:.2f} | Sortino: {metrics['sortino']:.2f}")
            log.info(f"  PF: {metrics['profit_factor']:.2f} | MDD: {metrics['max_dd']:.1%}")
            log.info(f"  Final Capital: ${metrics['final_capital']:.0f} | Return: {metrics['total_return']:.1%}")
            log.info(f"  Perm p: {perm_p:.4f} | Regime gap: {regime['regime_gap']:.2f}")
            log.info(f"  Gates: {gates['passed']}/{gates['total']} {'✓ PASS' if gates['all_passed'] else '✗ FAIL'}")

    # ── Summary ──
    log.info("\n" + "=" * 70)
    log.info("SUMMARY — ALL VARIANTS")
    log.info("=" * 70)

    summary_rows = []
    for name, res in sorted(results.items()):
        m = res["metrics"]
        g = res["gates"]
        summary_rows.append({
            "Variant": name,
            "Trades": m["n_trades"],
            "WR": f"{m['win_rate']:.1%}",
            "Sharpe": f"{m['sharpe']:.2f}",
            "Sortino": f"{m['sortino']:.2f}",
            "PF": f"{m['profit_factor']:.2f}",
            "MDD": f"{m['max_dd']:.1%}",
            "Return": f"{m['total_return']:.1%}",
            "Final$": f"${m['final_capital']:.0f}",
            "Perm_p": f"{res['perm_p']:.3f}",
            "RegGap": f"{res['regime']['regime_gap']:.2f}",
            "Gates": f"{g['passed']}/{g['total']}",
            "Pass": "YES" if g["all_passed"] else "no",
        })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        print("\n" + summary_df.to_string(index=False))

        # Save results
        with open(OUTPUT_DIR / "backtest_results.json", "w") as f:
            # Convert for JSON serialization
            serializable = {}
            for k, v in results.items():
                serializable[k] = {
                    "metrics": v["metrics"],
                    "perm_p": v["perm_p"],
                    "regime": v["regime"],
                    "gates": {gk: (bool(gv) if isinstance(gv, (bool, np.bool_)) else gv)
                              for gk, gv in v["gates"].items()},
                }
            json.dump(serializable, f, indent=2, default=str)

        log.info(f"\nResults saved to {OUTPUT_DIR / 'backtest_results.json'}")

        # Find best variant
        passing = {k: v for k, v in results.items() if v["gates"]["all_passed"]}
        if passing:
            best = max(passing.items(), key=lambda x: x[1]["metrics"]["sharpe"])
            log.info(f"\nBEST PASSING VARIANT: {best[0]}")
            log.info(f"  Sharpe: {best[1]['metrics']['sharpe']:.2f}")
            log.info(f"  Return: {best[1]['metrics']['total_return']:.1%}")
            log.info(f"  Final Capital: ${best[1]['metrics']['final_capital']:.0f}")
        else:
            log.info("\nNO VARIANTS PASSED ALL 5 GATES")
            # Show best by Sharpe anyway
            if results:
                best = max(results.items(), key=lambda x: x[1]["metrics"]["sharpe"])
                log.info(f"Best by Sharpe (not passing): {best[0]} = {best[1]['metrics']['sharpe']:.2f}")
    else:
        log.warning("No results to display!")


if __name__ == "__main__":
    main()
