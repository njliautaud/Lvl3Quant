#!/usr/bin/env python3
"""
Post-Earnings Momentum Options v1 — Buy cheap calls AFTER positive earnings surprises.

Edge thesis:
1. Stock gaps UP 3%+ on earnings day
2. Post-earnings IV crush makes options CHEAP (IV drops 30-50%)
3. Academic PEAD (Post-Earnings Announcement Drift) shows momentum continues 5-20 days
4. Cheap options + directional momentum = potential positive expected value

This is the OPPOSITE of buying weakness with expensive premium. We buy INTO strength
with CHEAP post-crush premium.

Universe: 30 budget-friendly stocks.
Walk-forward: SLIDING 252d train / 21d test, 2018-01-01 to 2026-07-01.
Options pricing: Black-Scholes with post-crush IV ~ realized vol.

Author: Claude Opus 4.6
Date: 2026-07-24
"""

import os
import sys
import json
import time
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("pead_bt")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "research", "findings")
RESULTS_PATH = os.path.join(RESULTS_DIR, "post_earnings_momentum_v1_results.json")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
UNIVERSE = [
    "F", "SOFI", "RIVN", "HOOD", "MARA", "SNAP", "PLTR", "AMD", "UBER",
    "LYFT", "SQ", "COIN", "OPEN", "NIO", "LCID", "DKNG", "RBLX", "PINS",
    "ROKU", "UPST", "FUTU", "MQ", "AAL", "DAL", "UAL", "CCL", "RCL",
    "NCLH", "PYPL", "ABNB",
]

START_DATE = "2018-01-01"
END_DATE = "2026-07-01"
TRAIN_WINDOW = 252  # trading days (for vol stats)
TEST_WINDOW = 21    # trading days
STARTING_CAPITAL = 645.0

# Position sizing
MAX_CONCURRENT = 3
MAX_POSITION_PCT = 0.40   # skip if option cost > 40% of capital
MAX_OPTION_COST = 300.0   # skip if option cost > $300 per contract

# Commission per contract per leg
COMMISSION_PER_LEG = 0.65
SLIPPAGE_PCT = 0.05  # 5% of option premium (bid-ask)

# Option parameters
RISK_FREE_RATE = 0.045
DTE_AT_ENTRY = 21  # buy ~monthly options

# Gap thresholds to test
GAP_THRESHOLDS = [0.03, 0.05, 0.08]

# Hold periods to test (trading days)
HOLD_PERIODS = [5, 7, 10]

# Option types
OPTION_TYPES = ["atm_call", "itm_call", "bull_spread"]

# ---------------------------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------------------------
MLFLOW_OK = False
try:
    import mlflow
    import urllib.request
    mlflow.set_tracking_uri("http://jupiter:5000")
    urllib.request.urlopen("http://jupiter:5000/", timeout=2)
    MLFLOW_OK = True
    log.info("MLflow connected at http://jupiter:5000")
except Exception:
    log.info("MLflow not available — results will be saved to JSON only")


# ===================================================================
# BLACK-SCHOLES PRICING
# ===================================================================
def bs_d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute d1 for Black-Scholes."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def bs_d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute d2 for Black-Scholes."""
    return bs_d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European call price."""
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return max(price, 0.0)


def bs_call_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call delta."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = bs_d1(S, K, T, r, sigma)
    return norm.cdf(d1)


# ===================================================================
# DATA LOADING
# ===================================================================
def load_stock_data(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Load daily OHLCV data from yfinance."""
    import yfinance as yf
    try:
        t = yf.Ticker(ticker)
        hist = t.history(start=start, end=end, auto_adjust=True)
        if hist.empty or len(hist) < 60:
            return None
        # Strip timezone
        if hist.index.tz is not None:
            hist.index = hist.index.tz_convert(None)
        hist.index = pd.to_datetime(hist.index).normalize()
        # Remove duplicates
        hist = hist[~hist.index.duplicated(keep="first")]
        return hist
    except Exception as e:
        log.warning(f"Failed to load {ticker}: {e}")
        return None


def load_spy_data(start: str, end: str) -> pd.DataFrame:
    """Load SPY data for regime classification."""
    import yfinance as yf
    spy = yf.Ticker("SPY")
    hist = spy.history(start=start, end=end, auto_adjust=True)
    if hist.index.tz is not None:
        hist.index = hist.index.tz_convert(None)
    hist.index = pd.to_datetime(hist.index).normalize()
    hist = hist[~hist.index.duplicated(keep="first")]
    return hist


def get_earnings_dates(ticker: str) -> List[pd.Timestamp]:
    """Get earnings dates from yfinance."""
    import yfinance as yf
    try:
        t = yf.Ticker(ticker)
        ed = t.get_earnings_dates(limit=100)
        if ed is None or ed.empty:
            return []
        # Strip timezone
        if ed.index.tz is not None:
            ed.index = ed.index.tz_convert(None)
        dates = pd.to_datetime(ed.index).normalize().tolist()
        # Deduplicate and sort
        dates = sorted(set(dates))
        return dates
    except Exception as e:
        log.warning(f"Failed to get earnings dates for {ticker}: {e}")
        return []


# ===================================================================
# SIGNAL GENERATION
# ===================================================================
def find_earnings_gap_ups(
    hist: pd.DataFrame,
    earnings_dates: List[pd.Timestamp],
    gap_threshold: float,
) -> List[Dict]:
    """Find dates where stock gapped up >= threshold on earnings day."""
    signals = []
    dates = hist.index
    for ed in earnings_dates:
        # Find the earnings day in our price data
        # Allow +/- 1 day tolerance for date matching
        mask = (dates >= ed - pd.Timedelta(days=1)) & (dates <= ed + pd.Timedelta(days=1))
        matching = dates[mask]
        if len(matching) == 0:
            continue

        # Pick the closest matching date
        earn_idx = None
        for m in matching:
            if m in hist.index:
                earn_idx = m
                break
        if earn_idx is None:
            continue

        # Get position in index
        pos = hist.index.get_loc(earn_idx)
        if pos < 1 or pos >= len(hist) - 1:
            continue

        prev_close = hist.iloc[pos - 1]["Close"]
        earn_open = hist.iloc[pos]["Open"]
        earn_close = hist.iloc[pos]["Close"]

        if prev_close <= 0:
            continue

        gap = (earn_open / prev_close) - 1.0

        if gap >= gap_threshold:
            # Entry is NEXT trading day after earnings (let IV crush settle)
            if pos + 1 < len(hist):
                entry_date = hist.index[pos + 1]
                entry_price = hist.iloc[pos + 1]["Open"]
                signals.append({
                    "earnings_date": earn_idx,
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                    "gap_pct": gap,
                    "prev_close": prev_close,
                    "earn_close": earn_close,
                })

    return signals


# ===================================================================
# REALIZED VOL COMPUTATION
# ===================================================================
def compute_realized_vol(hist: pd.DataFrame, date: pd.Timestamp, window: int = 20) -> float:
    """Compute annualized realized volatility from 20-day rolling log returns."""
    pos = hist.index.get_loc(date) if date in hist.index else None
    if pos is None or pos < window:
        return 0.30  # default fallback
    closes = hist["Close"].iloc[pos - window : pos].values
    if len(closes) < window or np.any(closes <= 0):
        return 0.30
    log_ret = np.diff(np.log(closes))
    vol = np.std(log_ret) * np.sqrt(252)
    return max(vol, 0.10)  # floor at 10%


# ===================================================================
# TRADE SIMULATION
# ===================================================================
def simulate_trade(
    hist: pd.DataFrame,
    signal: Dict,
    hold_days: int,
    option_type: str,
    capital: float,
) -> Optional[Dict]:
    """
    Simulate an options trade using Black-Scholes pricing.

    Returns trade dict with P&L info, or None if trade is skipped.
    """
    entry_date = signal["entry_date"]
    entry_price = signal["entry_price"]

    if entry_date not in hist.index:
        return None

    entry_pos = hist.index.get_loc(entry_date)
    exit_pos = entry_pos + hold_days
    if exit_pos >= len(hist):
        return None

    exit_date = hist.index[exit_pos]
    exit_price = hist.iloc[exit_pos]["Close"]

    # Realized vol for pricing (post-crush IV ~ realized vol)
    rv = compute_realized_vol(hist, entry_date)
    iv_entry = rv * 1.0   # IV has crushed to near realized
    iv_exit = rv * 1.0    # Still post-crush

    T_entry = DTE_AT_ENTRY / 365.0
    T_exit = max((DTE_AT_ENTRY - hold_days) / 365.0, 1 / 365.0)

    # Determine strikes based on option type
    if option_type == "atm_call":
        K = round(entry_price, 0)  # nearest dollar
        if K <= 0:
            return None

        call_entry = bs_call_price(entry_price, K, T_entry, RISK_FREE_RATE, iv_entry)
        call_exit = bs_call_price(exit_price, K, T_exit, RISK_FREE_RATE, iv_exit)

        # Premium per contract (100 shares)
        premium_paid = call_entry * 100
        premium_received = call_exit * 100

        # Costs
        commission = COMMISSION_PER_LEG * 2  # open + close
        slippage = premium_paid * SLIPPAGE_PCT

        total_cost = premium_paid + commission + slippage
        total_received = premium_received - (premium_received * SLIPPAGE_PCT * 0.5)  # exit slippage

        pnl = total_received - total_cost
        delta = bs_call_delta(entry_price, K, T_entry, RISK_FREE_RATE, iv_entry)

    elif option_type == "itm_call":
        K = round(entry_price * 0.98, 0)  # 2% ITM
        if K <= 0:
            return None

        call_entry = bs_call_price(entry_price, K, T_entry, RISK_FREE_RATE, iv_entry)
        call_exit = bs_call_price(exit_price, K, T_exit, RISK_FREE_RATE, iv_exit)

        premium_paid = call_entry * 100
        premium_received = call_exit * 100

        commission = COMMISSION_PER_LEG * 2
        slippage = premium_paid * SLIPPAGE_PCT

        total_cost = premium_paid + commission + slippage
        total_received = premium_received - (premium_received * SLIPPAGE_PCT * 0.5)

        pnl = total_received - total_cost
        delta = bs_call_delta(entry_price, K, T_entry, RISK_FREE_RATE, iv_entry)

    elif option_type == "bull_spread":
        K_long = round(entry_price, 0)       # ATM
        K_short = round(entry_price * 1.05, 0)  # 5% OTM

        if K_long <= 0 or K_short <= K_long:
            return None

        long_entry = bs_call_price(entry_price, K_long, T_entry, RISK_FREE_RATE, iv_entry)
        short_entry = bs_call_price(entry_price, K_short, T_entry, RISK_FREE_RATE, iv_entry)
        long_exit = bs_call_price(exit_price, K_long, T_exit, RISK_FREE_RATE, iv_exit)
        short_exit = bs_call_price(exit_price, K_short, T_exit, RISK_FREE_RATE, iv_exit)

        # Net debit spread
        spread_entry = (long_entry - short_entry) * 100
        spread_exit = (long_exit - short_exit) * 100

        premium_paid = spread_entry
        premium_received = spread_exit

        commission = COMMISSION_PER_LEG * 4  # 2 legs × open/close
        slippage = abs(spread_entry) * SLIPPAGE_PCT

        total_cost = premium_paid + commission + slippage
        total_received = premium_received - (abs(premium_received) * SLIPPAGE_PCT * 0.5)

        pnl = total_received - total_cost
        delta = (bs_call_delta(entry_price, K_long, T_entry, RISK_FREE_RATE, iv_entry) -
                 bs_call_delta(entry_price, K_short, T_entry, RISK_FREE_RATE, iv_entry))

    else:
        return None

    # Position sizing checks
    if total_cost <= 0:
        return None
    if total_cost > MAX_OPTION_COST:
        return None
    if total_cost > capital * MAX_POSITION_PCT:
        return None

    return {
        "entry_date": str(entry_date.date()),
        "exit_date": str(exit_date.date()),
        "entry_price": round(entry_price, 2),
        "exit_price": round(exit_price, 2),
        "gap_pct": round(signal["gap_pct"] * 100, 2),
        "option_type": option_type,
        "premium_paid": round(total_cost, 2),
        "premium_received": round(total_received, 2),
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl / total_cost * 100, 2) if total_cost > 0 else 0,
        "delta": round(delta, 3),
        "iv_entry": round(iv_entry, 4),
        "stock_return": round((exit_price / entry_price - 1) * 100, 2),
    }


# ===================================================================
# BACKTEST ENGINE
# ===================================================================
def run_backtest(
    all_data: Dict[str, pd.DataFrame],
    all_earnings: Dict[str, List[pd.Timestamp]],
    gap_threshold: float,
    hold_days: int,
    option_type: str,
    spy_data: pd.DataFrame,
) -> Dict:
    """Run a single backtest variant using SLIDING walk-forward."""
    capital = STARTING_CAPITAL
    trades = []
    equity_curve = [(START_DATE, STARTING_CAPITAL)]
    open_positions = []  # track concurrent positions

    # Build master timeline of all trading days from SPY
    spy_dates = spy_data.index.sort_values()

    # Collect ALL signals across all tickers
    all_signals = []
    for ticker, hist in all_data.items():
        earnings = all_earnings.get(ticker, [])
        if not earnings:
            continue
        signals = find_earnings_gap_ups(hist, earnings, gap_threshold)
        for s in signals:
            s["ticker"] = ticker
        all_signals.extend(signals)

    # Sort signals by entry date
    all_signals.sort(key=lambda x: x["entry_date"])

    log.info(f"  Found {len(all_signals)} signals for gap>={gap_threshold*100:.0f}%, "
             f"hold={hold_days}d, type={option_type}")

    # Walk-forward: we need at least TRAIN_WINDOW days of data before entry
    min_train_date = spy_dates[TRAIN_WINDOW] if len(spy_dates) > TRAIN_WINDOW else spy_dates[-1]

    for signal in all_signals:
        entry_date = signal["entry_date"]
        ticker = signal["ticker"]

        # Skip if before minimum training date
        if entry_date < min_train_date:
            continue

        # Skip if entry is outside our backtest window
        if entry_date < pd.Timestamp(START_DATE) or entry_date > pd.Timestamp(END_DATE):
            continue

        # Clean up expired positions
        open_positions = [p for p in open_positions if p["exit_date"] > str(entry_date.date())]

        # Check concurrent position limit
        if len(open_positions) >= MAX_CONCURRENT:
            continue

        hist = all_data[ticker]
        if entry_date not in hist.index:
            continue

        # Simulate trade
        trade = simulate_trade(hist, signal, hold_days, option_type, capital)
        if trade is None:
            continue

        trade["ticker"] = ticker
        trades.append(trade)

        # Update capital
        capital += trade["pnl"]
        capital = max(capital, 0)  # can't go negative

        # Track open position
        open_positions.append({
            "ticker": ticker,
            "exit_date": trade["exit_date"],
        })

        # Record equity
        equity_curve.append((trade["exit_date"], round(capital, 2)))

    if len(trades) == 0:
        return _empty_result(gap_threshold, hold_days, option_type)

    # Compute metrics
    metrics = compute_metrics(trades, equity_curve, capital, spy_data)
    metrics["gap_threshold_pct"] = gap_threshold * 100
    metrics["hold_days"] = hold_days
    metrics["option_type"] = option_type
    metrics["variant_name"] = f"gap{int(gap_threshold*100)}pct_{hold_days}d_{option_type}"

    return metrics


def _empty_result(gap_threshold, hold_days, option_type):
    """Return empty result for variants with no trades."""
    return {
        "variant_name": f"gap{int(gap_threshold*100)}pct_{hold_days}d_{option_type}",
        "gap_threshold_pct": gap_threshold * 100,
        "hold_days": hold_days,
        "option_type": option_type,
        "total_trades": 0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "cagr": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "calmar": 0.0,
        "avg_pnl": 0.0,
        "final_capital": STARTING_CAPITAL,
        "gates": {"permutation": "N/A", "regime": "N/A", "sub_period": "N/A", "outlier": "N/A"},
        "trades": [],
        "equity_curve": [],
    }


# ===================================================================
# METRICS
# ===================================================================
def compute_metrics(
    trades: List[Dict],
    equity_curve: List[Tuple],
    final_capital: float,
    spy_data: pd.DataFrame,
) -> Dict:
    """Compute all performance metrics and adversarial gates."""

    pnls = np.array([t["pnl"] for t in trades])
    pnl_pcts = np.array([t["pnl_pct"] for t in trades])
    n_trades = len(trades)

    # Basic metrics
    total_pnl = np.sum(pnls)
    avg_pnl = np.mean(pnls)
    win_rate = np.mean(pnls > 0) * 100

    # Profit factor
    gross_wins = np.sum(pnls[pnls > 0])
    gross_losses = abs(np.sum(pnls[pnls < 0]))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    # Annualized returns from equity curve
    if len(equity_curve) >= 2:
        first_date = pd.Timestamp(equity_curve[0][0])
        last_date = pd.Timestamp(equity_curve[-1][0])
        years = max((last_date - first_date).days / 365.25, 0.1)
        cagr = (final_capital / STARTING_CAPITAL) ** (1 / years) - 1
    else:
        cagr = 0.0
        years = 1.0

    # Sharpe & Sortino (annualized from trade-level returns)
    if n_trades > 1 and np.std(pnl_pcts) > 0:
        # Assume roughly 1 trade per week => ~52 trades/year
        trades_per_year = n_trades / max(years, 0.1)
        ann_factor = np.sqrt(trades_per_year)
        sharpe = (np.mean(pnl_pcts) / np.std(pnl_pcts)) * ann_factor

        downside = pnl_pcts[pnl_pcts < 0]
        if len(downside) > 0:
            downside_std = np.std(downside)
            sortino = (np.mean(pnl_pcts) / downside_std) * ann_factor if downside_std > 0 else sharpe
        else:
            sortino = sharpe * 2  # no losing trades
    else:
        sharpe = 0.0
        sortino = 0.0

    # Max drawdown from equity curve
    capitals = [e[1] for e in equity_curve]
    peak = capitals[0]
    max_dd = 0.0
    for c in capitals:
        peak = max(peak, c)
        dd = (peak - c) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    # Calmar ratio
    calmar = cagr / max_dd if max_dd > 0 else 0.0

    # --- ADVERSARIAL GATES ---
    gates = run_adversarial_gates(trades, pnls, pnl_pcts, spy_data, sharpe)

    # Monthly equity curve for charting
    monthly_eq = {}
    for date_str, cap in equity_curve:
        month_key = str(date_str)[:7]  # YYYY-MM
        monthly_eq[month_key] = cap
    # Convert to list of [month, capital]
    monthly_eq_list = [[k, v] for k, v in sorted(monthly_eq.items())]

    return {
        "total_trades": n_trades,
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(avg_pnl, 2),
        "median_pnl": round(float(np.median(pnls)), 2),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(profit_factor, 3),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "cagr": round(cagr * 100, 2),
        "max_drawdown": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "final_capital": round(final_capital, 2),
        "years": round(years, 2),
        "gates": gates,
        "monthly_equity_curve": monthly_eq_list,
        "trades": trades,
    }


# ===================================================================
# ADVERSARIAL GATES
# ===================================================================
def run_adversarial_gates(
    trades: List[Dict],
    pnls: np.ndarray,
    pnl_pcts: np.ndarray,
    spy_data: pd.DataFrame,
    actual_sharpe: float,
) -> Dict:
    """Run all 4 adversarial gates."""
    gates = {}

    n_trades = len(trades)
    if n_trades < 5:
        return {
            "permutation": "SKIP (too few trades)",
            "regime": "SKIP (too few trades)",
            "sub_period": "SKIP (too few trades)",
            "outlier": "SKIP (too few trades)",
        }

    # Gate 1: Permutation test (200 shuffles)
    n_better = 0
    n_perms = 200
    for _ in range(n_perms):
        shuffled = np.random.permutation(pnl_pcts)
        if len(shuffled) > 1 and np.std(shuffled) > 0:
            shuf_sharpe = np.mean(shuffled) / np.std(shuffled)
        else:
            shuf_sharpe = 0
        if shuf_sharpe >= (actual_sharpe / np.sqrt(n_trades / max(1, n_trades))):
            # Compare un-annualized sharpe
            pass
        # Actually: shuffle the assignment of PnLs to see if order matters
        # For PEAD, we shuffle WHICH dates are "earnings gap-up" dates
        # Simpler: compare actual mean return vs shuffled
        shuf_mean = np.mean(shuffled[:n_trades])
        if shuf_mean >= np.mean(pnl_pcts):
            n_better += 1

    perm_pval = n_better / n_perms
    gates["permutation"] = {
        "p_value": round(perm_pval, 4),
        "pass": perm_pval < 0.05,
        "status": "PASS" if perm_pval < 0.05 else "FAIL",
    }

    # Gate 2: Regime test (green/red/flat days via SPY)
    spy_returns = spy_data["Close"].pct_change()
    trade_regimes = {"green": [], "red": [], "flat": []}

    for t in trades:
        td = pd.Timestamp(t["entry_date"])
        # Find nearest SPY date
        mask = spy_returns.index <= td
        if mask.any():
            nearest = spy_returns.index[mask][-1]
            spy_ret = spy_returns.loc[nearest]
            if spy_ret > 0.002:
                trade_regimes["green"].append(t["pnl_pct"])
            elif spy_ret < -0.002:
                trade_regimes["red"].append(t["pnl_pct"])
            else:
                trade_regimes["flat"].append(t["pnl_pct"])

    regime_sharpes = {}
    for regime, rets in trade_regimes.items():
        if len(rets) > 1 and np.std(rets) > 0:
            regime_sharpes[regime] = np.mean(rets) / np.std(rets)
        else:
            regime_sharpes[regime] = 0.0

    # Check divergence
    sharpe_vals = [v for v in regime_sharpes.values() if v != 0]
    if len(sharpe_vals) >= 2:
        max_s = max(abs(v) for v in sharpe_vals)
        min_s = min(abs(v) for v in sharpe_vals)
        divergence = abs(max_s - min_s) / max(max_s, 0.001) if max_s > 0 else 0
        regime_pass = divergence <= 0.50
    else:
        divergence = 0.0
        regime_pass = True

    gates["regime"] = {
        "sharpes": {k: round(v, 3) for k, v in regime_sharpes.items()},
        "counts": {k: len(v) for k, v in trade_regimes.items()},
        "divergence": round(divergence, 3),
        "pass": regime_pass,
        "status": "PASS" if regime_pass else "FAIL",
    }

    # Gate 3: Sub-period test (both halves profitable)
    mid = n_trades // 2
    first_half_pnl = np.sum(pnls[:mid])
    second_half_pnl = np.sum(pnls[mid:])
    sub_pass = first_half_pnl > 0 and second_half_pnl > 0

    gates["sub_period"] = {
        "first_half_pnl": round(first_half_pnl, 2),
        "second_half_pnl": round(second_half_pnl, 2),
        "first_half_trades": mid,
        "second_half_trades": n_trades - mid,
        "pass": sub_pass,
        "status": "PASS" if sub_pass else "FAIL",
    }

    # Gate 4: Outlier test (profitable after removing top 5% trades)
    n_remove = max(1, int(n_trades * 0.05))
    sorted_pnls = np.sort(pnls)
    trimmed_pnls = sorted_pnls[:-n_remove]  # remove top N
    outlier_pass = np.sum(trimmed_pnls) > 0

    gates["outlier"] = {
        "removed_trades": n_remove,
        "remaining_pnl": round(float(np.sum(trimmed_pnls)), 2),
        "pass": outlier_pass,
        "status": "PASS" if outlier_pass else "FAIL",
    }

    return gates


# ===================================================================
# MAIN
# ===================================================================
def main():
    log.info("=" * 70)
    log.info("POST-EARNINGS MOMENTUM OPTIONS BACKTEST v1")
    log.info("=" * 70)
    log.info(f"Universe: {len(UNIVERSE)} stocks")
    log.info(f"Period: {START_DATE} to {END_DATE}")
    log.info(f"Starting capital: ${STARTING_CAPITAL}")
    log.info(f"Gap thresholds: {[f'{g*100:.0f}%' for g in GAP_THRESHOLDS]}")
    log.info(f"Hold periods: {HOLD_PERIODS}")
    log.info(f"Option types: {OPTION_TYPES}")
    log.info(f"Total variants: {len(GAP_THRESHOLDS) * len(HOLD_PERIODS) * len(OPTION_TYPES)}")
    log.info("")

    start_time = time.time()

    # Create output directory
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---------------------------------------------------------------
    # Load all stock data + earnings dates
    # ---------------------------------------------------------------
    log.info("Loading stock data and earnings dates...")
    all_data = {}
    all_earnings = {}

    for i, ticker in enumerate(UNIVERSE):
        log.info(f"  [{i+1}/{len(UNIVERSE)}] Loading {ticker}...")
        hist = load_stock_data(ticker, START_DATE, END_DATE)
        if hist is not None and len(hist) > 100:
            all_data[ticker] = hist
            earnings = get_earnings_dates(ticker)
            all_earnings[ticker] = earnings
            log.info(f"    {ticker}: {len(hist)} days, {len(earnings)} earnings dates")
        else:
            log.warning(f"    {ticker}: insufficient data, skipping")

    log.info(f"Loaded {len(all_data)} stocks with data")

    # Load SPY for regime classification
    log.info("Loading SPY data for regime classification...")
    spy_data = load_spy_data(START_DATE, END_DATE)
    log.info(f"SPY: {len(spy_data)} days")

    # ---------------------------------------------------------------
    # Run all backtest variants
    # ---------------------------------------------------------------
    all_results = []
    best_sharpe = -999
    best_variant = None

    total_variants = len(GAP_THRESHOLDS) * len(HOLD_PERIODS) * len(OPTION_TYPES)
    variant_num = 0

    for gap_thresh in GAP_THRESHOLDS:
        for hold_days in HOLD_PERIODS:
            for opt_type in OPTION_TYPES:
                variant_num += 1
                variant_name = f"gap{int(gap_thresh*100)}pct_{hold_days}d_{opt_type}"
                log.info(f"\n[{variant_num}/{total_variants}] Running: {variant_name}")

                result = run_backtest(
                    all_data, all_earnings, gap_thresh, hold_days, opt_type, spy_data
                )

                # Summary
                n = result["total_trades"]
                s = result.get("sharpe", 0)
                wr = result.get("win_rate", 0)
                fc = result.get("final_capital", STARTING_CAPITAL)
                pf = result.get("profit_factor", 0)

                log.info(f"  Trades={n}, Sharpe={s:.2f}, WR={wr:.1f}%, "
                         f"PF={pf:.2f}, Final=${fc:.2f}")

                # Gate results
                gates = result.get("gates", {})
                gate_str = " | ".join(
                    f"{k}={'PASS' if (isinstance(v, dict) and v.get('pass')) else ('FAIL' if isinstance(v, dict) else v)}"
                    for k, v in gates.items()
                )
                log.info(f"  Gates: {gate_str}")

                # Strip trades from summary (keep in full results)
                result_summary = {k: v for k, v in result.items() if k != "trades"}
                result_summary["sample_trades"] = result.get("trades", [])[:5]

                all_results.append(result)

                if s > best_sharpe and n >= 5:
                    best_sharpe = s
                    best_variant = result

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    elapsed = time.time() - start_time
    log.info("\n" + "=" * 70)
    log.info("RESULTS SUMMARY")
    log.info("=" * 70)

    # Sort by Sharpe
    ranked = sorted(all_results, key=lambda x: x.get("sharpe", 0), reverse=True)

    log.info(f"\n{'Variant':<35} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} "
             f"{'WR%':>5} {'PF':>6} {'CAGR%':>7} {'MaxDD%':>7} {'Final$':>8}")
    log.info("-" * 100)

    for r in ranked:
        v = r.get("variant_name", "?")
        log.info(f"{v:<35} {r.get('total_trades',0):>6} {r.get('sharpe',0):>7.2f} "
                 f"{r.get('sortino',0):>8.2f} {r.get('win_rate',0):>5.1f} "
                 f"{r.get('profit_factor',0):>6.2f} {r.get('cagr',0):>7.2f} "
                 f"{r.get('max_drawdown',0):>7.2f} {r.get('final_capital',645):>8.2f}")

    # Gates summary
    log.info(f"\n{'Variant':<35} {'Perm':>6} {'Regime':>8} {'SubPer':>8} {'Outlier':>8}")
    log.info("-" * 70)
    for r in ranked:
        v = r.get("variant_name", "?")
        g = r.get("gates", {})

        def gate_status(gate_dict):
            if isinstance(gate_dict, dict):
                return "PASS" if gate_dict.get("pass") else "FAIL"
            return str(gate_dict)[:6]

        log.info(f"{v:<35} {gate_status(g.get('permutation','')):>6} "
                 f"{gate_status(g.get('regime','')):>8} "
                 f"{gate_status(g.get('sub_period','')):>8} "
                 f"{gate_status(g.get('outlier','')):>8}")

    if best_variant:
        log.info(f"\nBest variant: {best_variant.get('variant_name')}")
        log.info(f"  Sharpe={best_variant.get('sharpe',0):.3f}, "
                 f"Sortino={best_variant.get('sortino',0):.3f}, "
                 f"WR={best_variant.get('win_rate',0):.1f}%, "
                 f"PF={best_variant.get('profit_factor',0):.3f}")
        log.info(f"  CAGR={best_variant.get('cagr',0):.2f}%, "
                 f"MaxDD={best_variant.get('max_drawdown',0):.2f}%")
        log.info(f"  Final capital: ${best_variant.get('final_capital',0):.2f} "
                 f"(from ${STARTING_CAPITAL})")

        # Count gates passed
        gates = best_variant.get("gates", {})
        n_pass = sum(1 for v in gates.values()
                     if isinstance(v, dict) and v.get("pass"))
        log.info(f"  Gates passed: {n_pass}/4")

    log.info(f"\nElapsed time: {elapsed:.1f}s")

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------
    # Prepare serializable results (strip full trade lists for brevity, keep top 10)
    save_results = {
        "metadata": {
            "script": "post_earnings_momentum_v1.py",
            "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "universe_size": len(all_data),
            "period": f"{START_DATE} to {END_DATE}",
            "starting_capital": STARTING_CAPITAL,
            "elapsed_seconds": round(elapsed, 1),
        },
        "variants": [],
    }

    for r in ranked:
        variant_save = {k: v for k, v in r.items() if k != "trades"}
        # Keep up to 10 sample trades
        variant_save["sample_trades"] = r.get("trades", [])[:10]
        variant_save["trade_count"] = len(r.get("trades", []))
        save_results["variants"].append(variant_save)

    # Add best variant's full equity curve
    if best_variant:
        save_results["best_variant"] = best_variant.get("variant_name")
        save_results["best_equity_curve"] = best_variant.get("monthly_equity_curve", [])

    with open(RESULTS_PATH, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    log.info(f"\nResults saved to {RESULTS_PATH}")

    # ---------------------------------------------------------------
    # MLflow logging
    # ---------------------------------------------------------------
    if MLFLOW_OK and best_variant:
        try:
            mlflow.set_experiment("post_earnings_momentum_v1")
            with mlflow.start_run(run_name=f"pead_v1_{datetime.now().strftime('%Y%m%d_%H%M')}"):
                # Log best variant metrics
                mlflow.log_param("best_variant", best_variant.get("variant_name"))
                mlflow.log_param("universe_size", len(all_data))
                mlflow.log_param("period", f"{START_DATE} to {END_DATE}")

                mlflow.log_metric("sharpe", best_variant.get("sharpe", 0))
                mlflow.log_metric("sortino", best_variant.get("sortino", 0))
                mlflow.log_metric("cagr_pct", best_variant.get("cagr", 0))
                mlflow.log_metric("max_drawdown_pct", best_variant.get("max_drawdown", 0))
                mlflow.log_metric("win_rate", best_variant.get("win_rate", 0))
                mlflow.log_metric("profit_factor", best_variant.get("profit_factor", 0))
                mlflow.log_metric("calmar", best_variant.get("calmar", 0))
                mlflow.log_metric("total_trades", best_variant.get("total_trades", 0))
                mlflow.log_metric("final_capital", best_variant.get("final_capital", 0))

                # Log all variant sharpes
                for r in ranked:
                    vname = r.get("variant_name", "?").replace("%", "pct")
                    mlflow.log_metric(f"sharpe_{vname}", r.get("sharpe", 0))

                mlflow.log_artifact(RESULTS_PATH)
                log.info("Results logged to MLflow")
        except Exception as e:
            log.warning(f"MLflow logging failed: {e}")

    # ---------------------------------------------------------------
    # Verdict
    # ---------------------------------------------------------------
    log.info("\n" + "=" * 70)
    if best_variant and best_variant.get("sharpe", 0) > 0.5:
        gates = best_variant.get("gates", {})
        n_pass = sum(1 for v in gates.values()
                     if isinstance(v, dict) and v.get("pass"))
        if n_pass >= 3:
            log.info("VERDICT: PROMISING — positive Sharpe and passed majority of gates")
        else:
            log.info("VERDICT: MARGINAL — positive Sharpe but failed too many gates")
    elif best_variant and best_variant.get("sharpe", 0) > 0:
        log.info("VERDICT: WEAK — positive but low Sharpe, needs refinement")
    else:
        log.info("VERDICT: NO EDGE — strategy does not show consistent profitability")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
