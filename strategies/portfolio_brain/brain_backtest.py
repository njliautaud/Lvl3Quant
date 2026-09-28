#!/usr/bin/env python3
"""
Unified Portfolio Brain — Historical Backtest with Adversarial Tests

Simulates the brain's logic (signal aggregation, Kelly sizing, sector concentration,
VIX gating, earnings avoidance) on the quality universe from 2022-01-01 to 2026-07-31.

Signal generation: mean reversion dip-buying on quality stocks
  - RSI(14) < 35
  - Price > 5% below 20-day SMA
  - Must be in quality universe

Adversarial tests:
  1. Random timing (1000 shuffles, p<0.05)
  2. Inverse signal (should lose money)
  3. No Kelly (equal sizing comparison)
  4. No VIX gate
  5. No sector concentration limit
  6. Sub-period stability (4 periods, all positive Sharpe)
  7. Regime gap (bull vs bear Sharpe gap < 0.50)
"""

import json
import os
import sys
import logging
import math
import warnings
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATE_DIR = os.path.join(BASE_DIR, "state")
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [BrainBT] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "brain_backtest.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (match unified_brain.py)
# ---------------------------------------------------------------------------
STARTING_CAPITAL = 750.0
MAX_POSITION_PCT = 0.20
MAX_PORTFOLIO_HEAT = 0.60
MIN_CONFIDENCE = 0.40
HOLD_DAYS = 10
QUARTER_KELLY = 0.25
MIN_TRADE_SIZE = 30.0

VIX_HIGH = 25
VIX_EXTREME = 35

TICKER_SECTOR = {
    "AAPL": "tech", "MSFT": "tech", "GOOGL": "tech", "AMZN": "tech",
    "NVDA": "tech", "META": "tech", "AMD": "tech", "CRM": "tech",
    "AVGO": "tech", "INTU": "tech", "TXN": "tech", "AMAT": "tech",
    "NFLX": "tech", "ACN": "tech",
    "BRK-B": "finance", "JPM": "finance", "V": "finance", "MA": "finance",
    "LLY": "health", "UNH": "health", "JNJ": "health", "ABBV": "health",
    "MRK": "health", "TMO": "health", "ISRG": "health",
    "PG": "consumer", "HD": "consumer", "COST": "consumer", "MCD": "consumer",
    "PEP": "consumer", "KO": "consumer", "LOW": "consumer",
    "LIN": "industrial",
}

START_DATE = "2022-01-01"
END_DATE = "2026-07-31"

# ---------------------------------------------------------------------------
# Data Download
# ---------------------------------------------------------------------------

def download_data(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """Download daily OHLCV for all tickers + ^VIX."""
    all_tickers = list(set(tickers + ["^VIX"]))
    log.info(f"Downloading data for {len(all_tickers)} symbols...")

    # Download in bulk
    data = yf.download(all_tickers, start=START_DATE, end=END_DATE,
                       auto_adjust=True, progress=False, group_by="ticker")

    result = {}
    for t in all_tickers:
        try:
            if len(all_tickers) == 1:
                df = data.copy()
            else:
                df = data[t].copy()
            df = df.dropna(subset=["Close"])
            if len(df) > 50:
                result[t] = df
                log.info(f"  {t}: {len(df)} days")
            else:
                log.warning(f"  {t}: only {len(df)} days, skipping")
        except Exception as e:
            log.warning(f"  {t}: failed - {e}")

    return result


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def get_earnings_dates(ticker: str) -> set:
    """Get approximate earnings dates for a ticker (quarterly)."""
    # Use yfinance earnings calendar if available, else approximate quarterly
    try:
        t = yf.Ticker(ticker)
        cal = t.earnings_dates
        if cal is not None and len(cal) > 0:
            return set(cal.index.date)
    except:
        pass
    return set()


# ---------------------------------------------------------------------------
# Signal Generation
# ---------------------------------------------------------------------------

def generate_signals(
    price_data: Dict[str, pd.DataFrame],
    tickers: List[str],
    vix_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate mean-reversion dip-buy signals.
    Signal fires when:
      - RSI(14) < 35
      - Price > 5% below 20-SMA

    Confidence = composite of how oversold + how far below SMA.
    Multiple signals on same day for same ticker count as confirming sources.
    """
    signals = []

    for ticker in tickers:
        if ticker not in price_data:
            continue
        df = price_data[ticker].copy()
        close = df["Close"].squeeze() if isinstance(df["Close"], pd.DataFrame) else df["Close"]

        # Compute indicators
        rsi = compute_rsi(close, 14)
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        pct_below_sma20 = (close - sma20) / sma20

        # Bollinger bands
        bb_std = close.rolling(20).std()
        bb_lower = sma20 - 2 * bb_std

        for i in range(50, len(df)):
            date = df.index[i]
            r = rsi.iloc[i]
            pct = pct_below_sma20.iloc[i]
            price = close.iloc[i]
            bb_low = bb_lower.iloc[i]

            # Core signal: RSI < 35 AND > 5% below 20-SMA
            if r < 35 and pct < -0.05:
                # Count confirming sub-signals
                n_sources = 1  # base: RSI oversold + below SMA
                reasons = ["RSI_oversold_below_SMA"]

                # Additional confirmations
                if price < bb_low:
                    n_sources += 1
                    reasons.append("below_bollinger")

                if r < 25:
                    n_sources += 1
                    reasons.append("deeply_oversold")

                if pct < -0.10:
                    n_sources += 1
                    reasons.append("deeply_below_SMA")

                # Volume spike check
                vol = df["Volume"].squeeze() if isinstance(df["Volume"], pd.DataFrame) else df["Volume"]
                avg_vol = vol.iloc[max(0,i-20):i].mean()
                if avg_vol > 0 and vol.iloc[i] > 1.5 * avg_vol:
                    n_sources += 1
                    reasons.append("volume_spike")

                # Trend: 50-SMA still upward (quality dip, not collapse)
                sma50_val = sma50.iloc[i]
                sma50_prev = sma50.iloc[i-20] if i >= 70 else sma50_val
                if sma50_val > sma50_prev:
                    n_sources += 1
                    reasons.append("uptrend_intact")

                # Confidence based on RSI depth and SMA gap
                rsi_score = max(0, (35 - r) / 35)  # 0 to 1
                sma_score = min(1.0, abs(pct) / 0.15)  # 0 to 1
                confidence = 0.3 + 0.4 * rsi_score + 0.3 * sma_score
                confidence = min(0.95, confidence)

                signals.append({
                    "date": date,
                    "ticker": ticker,
                    "direction": "bull",
                    "confidence": confidence,
                    "n_sources": n_sources,
                    "reasons": reasons,
                    "entry_price": price,
                    "rsi": r,
                    "pct_below_sma": pct,
                })

    sig_df = pd.DataFrame(signals)
    if len(sig_df) > 0:
        sig_df["date"] = pd.to_datetime(sig_df["date"])
    log.info(f"Generated {len(sig_df)} raw signals across {len(tickers)} tickers")
    return sig_df


# ---------------------------------------------------------------------------
# Kelly Sizing (matches unified_brain.py)
# ---------------------------------------------------------------------------

def kelly_size(confidence: float, account_equity: float) -> float:
    """Quarter-Kelly position sizing."""
    if confidence < MIN_CONFIDENCE:
        return 0.0
    win_prob = confidence
    odds = 2.0
    kelly_f = (odds * win_prob - (1 - win_prob)) / odds
    kelly_f = max(0, kelly_f)
    quarter_kelly = kelly_f * QUARTER_KELLY
    position_pct = min(quarter_kelly, MAX_POSITION_PCT)
    size = round(position_pct * account_equity, 2)
    if size < MIN_TRADE_SIZE:
        return 0.0
    return size


# ---------------------------------------------------------------------------
# Backtest Engine
# ---------------------------------------------------------------------------

def run_backtest(
    signals: pd.DataFrame,
    price_data: Dict[str, pd.DataFrame],
    vix_data: pd.DataFrame,
    apply_kelly: bool = True,
    apply_vix_gate: bool = True,
    apply_sector_limit: bool = True,
    apply_earnings_avoid: bool = True,
    inverse: bool = False,
    shuffle_dates: bool = False,
    rng: Optional[np.random.RandomState] = None,
    cached_earnings: Optional[Dict[str, set]] = None,
) -> Dict[str, Any]:
    """
    Run the backtest with configurable features for adversarial testing.

    Returns dict with trades, equity curve, and metrics.
    """
    if len(signals) == 0:
        return {"trades": [], "equity_curve": [], "metrics": _empty_metrics()}

    sig = signals.copy()

    # Shuffle dates if requested (adversarial test)
    if shuffle_dates and rng is not None:
        dates = sig["date"].values.copy()
        rng.shuffle(dates)
        sig["date"] = dates

    # Sort by date
    sig = sig.sort_values("date").reset_index(drop=True)

    # Get VIX as a series indexed by date
    vix_close = vix_data["Close"].squeeze() if isinstance(vix_data["Close"], pd.DataFrame) else vix_data["Close"]

    # Use cached earnings dates or fetch
    earnings_dates = cached_earnings or {}
    if apply_earnings_avoid and not cached_earnings:
        for ticker in sig["ticker"].unique():
            earnings_dates[ticker] = get_earnings_dates(ticker)

    # Run through signals
    equity = STARTING_CAPITAL
    trades = []
    open_positions = []  # list of (ticker, entry_date, entry_price, size, exit_date)
    equity_curve = []

    # Get unique trading dates
    all_dates = sorted(sig["date"].unique())

    for date in all_dates:
        date_ts = pd.Timestamp(date)

        # Close expired positions
        new_open = []
        for pos in open_positions:
            if date_ts >= pos["exit_date"]:
                # Get exit price
                ticker = pos["ticker"]
                if ticker in price_data:
                    pdf = price_data[ticker]
                    close_col = pdf["Close"].squeeze() if isinstance(pdf["Close"], pd.DataFrame) else pdf["Close"]
                    # Find closest available date
                    mask = pdf.index >= pos["exit_date"]
                    if mask.any():
                        exit_price = close_col[mask].iloc[0]
                    else:
                        exit_price = close_col.iloc[-1]

                    if inverse:
                        # Inverse: sell when brain says buy -> short
                        ret_pct = (pos["entry_price"] - exit_price) / pos["entry_price"]
                    else:
                        ret_pct = (exit_price - pos["entry_price"]) / pos["entry_price"]

                    pnl = pos["size"] * ret_pct
                    equity += pnl

                    trades.append({
                        "ticker": ticker,
                        "entry_date": str(pos["entry_date"].date()),
                        "exit_date": str(date_ts.date()),
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_price,
                        "size": pos["size"],
                        "return_pct": ret_pct,
                        "pnl": pnl,
                        "direction": "short" if inverse else "long",
                    })
            else:
                new_open.append(pos)
        open_positions = new_open

        equity_curve.append({"date": str(date_ts.date()), "equity": equity})

        # Get day's signals
        day_signals = sig[sig["date"] == date]
        if len(day_signals) == 0:
            continue

        # VIX gating
        vix_level = 16.0  # default
        try:
            vix_mask = vix_close.index <= date_ts
            if vix_mask.any():
                vix_level = float(vix_close[vix_mask].iloc[-1])
        except:
            pass

        if apply_vix_gate:
            if vix_level > VIX_EXTREME:
                continue  # cash only
            vix_multiplier = 0.5 if vix_level > VIX_HIGH else 1.0
        else:
            vix_multiplier = 1.0

        # Process signals for the day
        day_trades = []
        for _, row in day_signals.iterrows():
            ticker = row["ticker"]
            confidence = row["confidence"]

            # Earnings avoidance
            if apply_earnings_avoid and ticker in earnings_dates:
                ed = earnings_dates[ticker]
                current_date = date_ts.date() if hasattr(date_ts, 'date') else date_ts
                try:
                    current_date = pd.Timestamp(current_date).date()
                except:
                    pass
                skip = False
                for earn_date in ed:
                    try:
                        days_until = (earn_date - current_date).days
                        if 0 <= days_until <= 2:
                            skip = True
                            break
                    except:
                        pass
                if skip:
                    continue

            # Kelly sizing
            if apply_kelly:
                size = kelly_size(confidence, equity)
            else:
                # Equal sizing: fixed fraction
                size = min(equity * 0.05, equity * MAX_POSITION_PCT)
                if size < MIN_TRADE_SIZE:
                    size = 0.0

            if size <= 0:
                continue

            # VIX adjustment
            size *= vix_multiplier

            if size < MIN_TRADE_SIZE:
                continue

            day_trades.append({
                "ticker": ticker,
                "confidence": confidence,
                "n_sources": row["n_sources"],
                "size": size,
                "entry_price": row["entry_price"],
            })

        # Sector concentration
        if apply_sector_limit and len(day_trades) > 0:
            sector_counts = defaultdict(int)
            for t in day_trades:
                sector = TICKER_SECTOR.get(t["ticker"], "other")
                sector_counts[sector] += 1
            concentrated = {s for s, c in sector_counts.items() if c >= 3}
            if concentrated:
                for t in day_trades:
                    if TICKER_SECTOR.get(t["ticker"], "other") in concentrated:
                        t["size"] *= 0.5

        # Portfolio heat cap
        current_risk = sum(p["size"] for p in open_positions)
        for t in day_trades:
            if current_risk + t["size"] > equity * MAX_PORTFOLIO_HEAT:
                remaining = max(0, equity * MAX_PORTFOLIO_HEAT - current_risk)
                if remaining < MIN_TRADE_SIZE:
                    break
                t["size"] = min(t["size"], remaining)

            exit_date = date_ts + pd.Timedelta(days=HOLD_DAYS)
            open_positions.append({
                "ticker": t["ticker"],
                "entry_date": date_ts,
                "entry_price": t["entry_price"],
                "size": t["size"],
                "exit_date": exit_date,
            })
            current_risk += t["size"]

    # Close any remaining positions at last available price
    for pos in open_positions:
        ticker = pos["ticker"]
        if ticker in price_data:
            pdf = price_data[ticker]
            close_col = pdf["Close"].squeeze() if isinstance(pdf["Close"], pd.DataFrame) else pdf["Close"]
            exit_price = close_col.iloc[-1]
            if inverse:
                ret_pct = (pos["entry_price"] - exit_price) / pos["entry_price"]
            else:
                ret_pct = (exit_price - pos["entry_price"]) / pos["entry_price"]
            pnl = pos["size"] * ret_pct
            equity += pnl
            trades.append({
                "ticker": ticker,
                "entry_date": str(pos["entry_date"].date()),
                "exit_date": "final",
                "entry_price": pos["entry_price"],
                "exit_price": exit_price,
                "size": pos["size"],
                "return_pct": ret_pct,
                "pnl": pnl,
                "direction": "short" if inverse else "long",
            })

    equity_curve.append({"date": "final", "equity": equity})

    metrics = compute_metrics(trades, equity_curve)
    return {"trades": trades, "equity_curve": equity_curve, "metrics": metrics}


def _empty_metrics() -> Dict:
    return {
        "total_return_pct": 0, "total_return_dollar": 0, "sharpe": 0,
        "sortino": 0, "max_drawdown_pct": 0, "win_rate": 0,
        "profit_factor": 0, "n_trades": 0, "avg_pnl": 0,
    }


def compute_metrics(trades: List[Dict], equity_curve: List[Dict]) -> Dict:
    """Compute performance metrics from trades and equity curve."""
    if not trades:
        return _empty_metrics()

    pnls = [t["pnl"] for t in trades]
    returns = [t["return_pct"] for t in trades]
    n_trades = len(trades)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    total_return_pct = total_pnl / STARTING_CAPITAL * 100

    # Win rate
    win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0

    # Profit factor
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1e-9
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Sharpe (annualized, assuming ~252 trading days, avg hold = 10 days)
    if len(returns) > 1:
        ret_arr = np.array(returns)
        trades_per_year = 252 / HOLD_DAYS
        sharpe = (ret_arr.mean() / ret_arr.std()) * np.sqrt(trades_per_year) if ret_arr.std() > 0 else 0
    else:
        sharpe = 0

    # Sortino
    if len(returns) > 1:
        ret_arr = np.array(returns)
        downside = ret_arr[ret_arr < 0]
        downside_std = downside.std() if len(downside) > 1 else 1e-9
        trades_per_year = 252 / HOLD_DAYS
        sortino = (ret_arr.mean() / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0
    else:
        sortino = 0

    # Max drawdown from equity curve
    equities = [e["equity"] for e in equity_curve if isinstance(e["equity"], (int, float))]
    if equities:
        peak = equities[0]
        max_dd = 0
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
    else:
        max_dd = 0

    return {
        "total_return_pct": round(total_return_pct, 2),
        "total_return_dollar": round(total_pnl, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(profit_factor, 3),
        "n_trades": n_trades,
        "avg_pnl": round(total_pnl / n_trades, 2) if n_trades > 0 else 0,
        "final_equity": round(STARTING_CAPITAL + total_pnl, 2),
    }


# ---------------------------------------------------------------------------
# Adversarial Tests
# ---------------------------------------------------------------------------

def adversarial_random_timing(signals, price_data, vix_data, base_sharpe, cached_earnings=None, n_shuffles=1000):
    """Shuffle signal dates 1000x, compare Sharpe. p<0.05 required."""
    log.info(f"Adversarial: random timing test ({n_shuffles} shuffles)...")
    shuffle_sharpes = []
    for i in range(n_shuffles):
        rng = np.random.RandomState(i)
        result = run_backtest(signals, price_data, vix_data, shuffle_dates=True, rng=rng, cached_earnings=cached_earnings)
        shuffle_sharpes.append(result["metrics"]["sharpe"])
        if (i + 1) % 200 == 0:
            log.info(f"  Completed {i+1}/{n_shuffles} shuffles")

    shuffle_sharpes = np.array(shuffle_sharpes)
    p_value = np.mean(shuffle_sharpes >= base_sharpe)

    return {
        "test": "random_timing",
        "base_sharpe": base_sharpe,
        "shuffle_mean_sharpe": round(float(np.mean(shuffle_sharpes)), 3),
        "shuffle_std_sharpe": round(float(np.std(shuffle_sharpes)), 3),
        "p_value": round(float(p_value), 4),
        "pass": p_value < 0.05,
        "description": f"Signal timing matters: p={p_value:.4f} (need <0.05)",
    }


def adversarial_inverse(signals, price_data, vix_data, base_sharpe, cached_earnings=None):
    """Inverse signal: sell when brain says buy. Should lose money."""
    log.info("Adversarial: inverse signal test...")
    result = run_backtest(signals, price_data, vix_data, inverse=True, cached_earnings=cached_earnings)
    inv_sharpe = result["metrics"]["sharpe"]
    inv_return = result["metrics"]["total_return_pct"]

    return {
        "test": "inverse_signal",
        "base_sharpe": base_sharpe,
        "inverse_sharpe": inv_sharpe,
        "inverse_return_pct": inv_return,
        "pass": inv_return < 0,
        "description": f"Inverse return: {inv_return:.1f}% (need negative)",
    }


def adversarial_no_kelly(signals, price_data, vix_data, base_result, cached_earnings=None):
    """Compare Kelly vs equal sizing."""
    log.info("Adversarial: no-Kelly (equal sizing) test...")
    result = run_backtest(signals, price_data, vix_data, apply_kelly=False, cached_earnings=cached_earnings)
    eq_metrics = result["metrics"]

    return {
        "test": "no_kelly",
        "kelly_sharpe": base_result["sharpe"],
        "equal_sharpe": eq_metrics["sharpe"],
        "kelly_sortino": base_result["sortino"],
        "equal_sortino": eq_metrics["sortino"],
        "kelly_dd": base_result["max_drawdown_pct"],
        "equal_dd": eq_metrics["max_drawdown_pct"],
        "pass": base_result["sharpe"] >= eq_metrics["sharpe"],
        "description": f"Kelly Sharpe {base_result['sharpe']:.3f} vs Equal {eq_metrics['sharpe']:.3f}",
    }


def adversarial_no_vix_gate(signals, price_data, vix_data, base_result, cached_earnings=None):
    """Compare with vs without VIX gating."""
    log.info("Adversarial: no VIX gate test...")
    result = run_backtest(signals, price_data, vix_data, apply_vix_gate=False, cached_earnings=cached_earnings)
    no_vix = result["metrics"]

    return {
        "test": "no_vix_gate",
        "with_vix_dd": base_result["max_drawdown_pct"],
        "without_vix_dd": no_vix["max_drawdown_pct"],
        "with_vix_sharpe": base_result["sharpe"],
        "without_vix_sharpe": no_vix["sharpe"],
        "pass": base_result["max_drawdown_pct"] <= no_vix["max_drawdown_pct"],
        "description": f"VIX gate DD {base_result['max_drawdown_pct']:.1f}% vs no gate {no_vix['max_drawdown_pct']:.1f}%",
    }


def adversarial_no_sector_limit(signals, price_data, vix_data, base_result, cached_earnings=None):
    """Compare with vs without sector concentration limit."""
    log.info("Adversarial: no sector limit test...")
    result = run_backtest(signals, price_data, vix_data, apply_sector_limit=False, cached_earnings=cached_earnings)
    no_sec = result["metrics"]

    return {
        "test": "no_sector_limit",
        "with_limit_dd": base_result["max_drawdown_pct"],
        "without_limit_dd": no_sec["max_drawdown_pct"],
        "with_limit_sharpe": base_result["sharpe"],
        "without_limit_sharpe": no_sec["sharpe"],
        "pass": base_result["max_drawdown_pct"] <= no_sec["max_drawdown_pct"],
        "description": f"Sector limit DD {base_result['max_drawdown_pct']:.1f}% vs no limit {no_sec['max_drawdown_pct']:.1f}%",
    }


def adversarial_subperiod_stability(trades: List[Dict]) -> Dict:
    """4 equal sub-periods, all must have positive Sharpe."""
    log.info("Adversarial: sub-period stability test...")
    if not trades:
        return {"test": "subperiod_stability", "pass": False, "description": "No trades"}

    # Parse dates
    trade_df = pd.DataFrame(trades)
    trade_df["entry_dt"] = pd.to_datetime(trade_df["entry_date"])
    trade_df = trade_df.sort_values("entry_dt")

    # Split into 4 equal periods
    n = len(trade_df)
    quarter = n // 4
    periods = []
    for i in range(4):
        start_idx = i * quarter
        end_idx = (i + 1) * quarter if i < 3 else n
        period_trades = trade_df.iloc[start_idx:end_idx]
        returns = period_trades["return_pct"].values
        if len(returns) > 1 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252 / HOLD_DAYS)
        else:
            sharpe = 0
        date_range = f"{period_trades['entry_date'].iloc[0]} to {period_trades['entry_date'].iloc[-1]}"
        periods.append({
            "period": i + 1,
            "date_range": date_range,
            "n_trades": len(period_trades),
            "sharpe": round(sharpe, 3),
            "win_rate": round((period_trades["pnl"] > 0).mean() * 100, 1),
            "total_pnl": round(period_trades["pnl"].sum(), 2),
        })

    all_positive = all(p["sharpe"] > 0 for p in periods)

    return {
        "test": "subperiod_stability",
        "periods": periods,
        "all_positive_sharpe": all_positive,
        "pass": all_positive,
        "description": f"Sub-period Sharpes: {[p['sharpe'] for p in periods]} — {'ALL positive' if all_positive else 'NOT all positive'}",
    }


def adversarial_regime_gap(trades: List[Dict], price_data: Dict) -> Dict:
    """Bull vs bear Sharpe, gap < 0.50 required."""
    log.info("Adversarial: regime gap test...")
    if not trades:
        return {"test": "regime_gap", "pass": False, "description": "No trades"}

    # Use SPY as market proxy for regime classification
    spy_data = price_data.get("SPY")
    if spy_data is None:
        spy_data = price_data.get("AAPL")
    if spy_data is None:
        # Download SPY
        spy_data = yf.download("SPY", start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

    spy_close = spy_data["Close"].squeeze() if isinstance(spy_data["Close"], pd.DataFrame) else spy_data["Close"]
    spy_sma200 = spy_close.rolling(200).mean()

    trade_df = pd.DataFrame(trades)
    trade_df["entry_dt"] = pd.to_datetime(trade_df["entry_date"])

    bull_returns = []
    bear_returns = []

    for _, row in trade_df.iterrows():
        dt = row["entry_dt"]
        try:
            mask = spy_close.index <= dt
            if mask.any():
                price = float(spy_close[mask].iloc[-1])
                sma = float(spy_sma200[mask].iloc[-1])
                if not np.isnan(sma):
                    if price > sma:
                        bull_returns.append(row["return_pct"])
                    else:
                        bear_returns.append(row["return_pct"])
        except:
            pass

    bull_arr = np.array(bull_returns) if bull_returns else np.array([0])
    bear_arr = np.array(bear_returns) if bear_returns else np.array([0])

    bull_sharpe = (bull_arr.mean() / bull_arr.std() * np.sqrt(252/HOLD_DAYS)) if len(bull_arr) > 1 and bull_arr.std() > 0 else 0
    bear_sharpe = (bear_arr.mean() / bear_arr.std() * np.sqrt(252/HOLD_DAYS)) if len(bear_arr) > 1 and bear_arr.std() > 0 else 0

    max_abs = max(abs(bull_sharpe), abs(bear_sharpe), 1e-9)
    gap = abs(bull_sharpe - bear_sharpe) / max_abs

    return {
        "test": "regime_gap",
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "n_bull_trades": len(bull_returns),
        "n_bear_trades": len(bear_returns),
        "gap_ratio": round(gap, 3),
        "pass": gap < 0.50,
        "description": f"Bull Sharpe {bull_sharpe:.3f} vs Bear {bear_sharpe:.3f}, gap ratio {gap:.3f} (need <0.50)",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=" * 70)
    log.info("UNIFIED PORTFOLIO BRAIN — HISTORICAL BACKTEST")
    log.info(f"Period: {START_DATE} to {END_DATE}")
    log.info(f"Starting capital: ${STARTING_CAPITAL}")
    log.info("=" * 70)

    # Load quality universe
    qu_path = os.path.join(DATA_DIR, "quality_universe.json")
    with open(qu_path) as f:
        qu = json.load(f)
    tickers = qu["tickers"]
    log.info(f"Quality universe: {len(tickers)} tickers")

    # Download data
    price_data = download_data(tickers)
    vix_data = price_data.get("^VIX")
    if vix_data is None:
        log.error("Failed to download VIX data")
        return

    # Also download SPY for regime classification
    if "SPY" not in price_data:
        spy = yf.download("SPY", start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
        if len(spy) > 0:
            price_data["SPY"] = spy

    # Generate signals
    signals = generate_signals(price_data, tickers, vix_data)
    if len(signals) == 0:
        log.error("No signals generated")
        return

    log.info(f"\nTotal signals: {len(signals)}")
    log.info(f"Date range: {signals['date'].min()} to {signals['date'].max()}")
    log.info(f"Tickers with signals: {signals['ticker'].nunique()}")

    # =====================================================================
    # BASE BACKTEST (all features enabled)
    # =====================================================================
    # Pre-cache earnings dates (avoid repeated API calls in adversarial tests)
    log.info("Pre-caching earnings dates...")
    cached_earnings = {}
    for ticker in tickers:
        cached_earnings[ticker] = get_earnings_dates(ticker)
    log.info(f"Cached earnings for {len(cached_earnings)} tickers")

    log.info("\n" + "=" * 50)
    log.info("RUNNING BASE BACKTEST (all features enabled)")
    log.info("=" * 50)

    base_result = run_backtest(signals, price_data, vix_data, cached_earnings=cached_earnings)
    base_metrics = base_result["metrics"]

    log.info(f"\n--- BASE RESULTS ---")
    for k, v in base_metrics.items():
        log.info(f"  {k}: {v}")

    # =====================================================================
    # ADVERSARIAL TESTS
    # =====================================================================
    log.info("\n" + "=" * 50)
    log.info("RUNNING ADVERSARIAL TESTS")
    log.info("=" * 50)

    adversarial_results = []

    # 1. Random timing
    rt = adversarial_random_timing(signals, price_data, vix_data, base_metrics["sharpe"], cached_earnings=cached_earnings)
    adversarial_results.append(rt)
    log.info(f"  [{'PASS' if rt['pass'] else 'FAIL'}] {rt['description']}")

    # 2. Inverse signal
    inv = adversarial_inverse(signals, price_data, vix_data, base_metrics["sharpe"], cached_earnings=cached_earnings)
    adversarial_results.append(inv)
    log.info(f"  [{'PASS' if inv['pass'] else 'FAIL'}] {inv['description']}")

    # 3. No Kelly
    nk = adversarial_no_kelly(signals, price_data, vix_data, base_metrics, cached_earnings=cached_earnings)
    adversarial_results.append(nk)
    log.info(f"  [{'PASS' if nk['pass'] else 'FAIL'}] {nk['description']}")

    # 4. No VIX gate
    nv = adversarial_no_vix_gate(signals, price_data, vix_data, base_metrics, cached_earnings=cached_earnings)
    adversarial_results.append(nv)
    log.info(f"  [{'PASS' if nv['pass'] else 'FAIL'}] {nv['description']}")

    # 5. No sector limit
    ns = adversarial_no_sector_limit(signals, price_data, vix_data, base_metrics, cached_earnings=cached_earnings)
    adversarial_results.append(ns)
    log.info(f"  [{'PASS' if ns['pass'] else 'FAIL'}] {ns['description']}")

    # 6. Sub-period stability
    sp = adversarial_subperiod_stability(base_result["trades"])
    adversarial_results.append(sp)
    log.info(f"  [{'PASS' if sp['pass'] else 'FAIL'}] {sp['description']}")

    # 7. Regime gap
    rg = adversarial_regime_gap(base_result["trades"], price_data)
    adversarial_results.append(rg)
    log.info(f"  [{'PASS' if rg['pass'] else 'FAIL'}] {rg['description']}")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    n_pass = sum(1 for a in adversarial_results if a["pass"])
    n_total = len(adversarial_results)

    log.info("\n" + "=" * 50)
    log.info(f"ADVERSARIAL SUMMARY: {n_pass}/{n_total} tests passed")
    log.info("=" * 50)

    # Build output
    output = {
        "generated_at": datetime.now().isoformat(),
        "backtest_period": f"{START_DATE} to {END_DATE}",
        "starting_capital": STARTING_CAPITAL,
        "quality_universe_size": len(tickers),
        "total_signals_generated": len(signals),
        "base_metrics": base_metrics,
        "adversarial_tests": adversarial_results,
        "adversarial_summary": {
            "passed": n_pass,
            "total": n_total,
            "all_pass": n_pass == n_total,
        },
        "trade_summary": {
            "n_trades": len(base_result["trades"]),
            "unique_tickers": len(set(t["ticker"] for t in base_result["trades"])),
            "avg_trades_per_year": round(len(base_result["trades"]) / 4.5, 1),
        },
    }

    # Save results
    output_path = os.path.join(STATE_DIR, "brain_backtest_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"\nResults saved to {output_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"BRAIN BACKTEST RESULTS ({START_DATE} to {END_DATE})")
    print(f"{'='*60}")
    print(f"Starting Capital:  ${STARTING_CAPITAL:.0f}")
    print(f"Final Equity:      ${base_metrics.get('final_equity', 0):.2f}")
    print(f"Total Return:      {base_metrics['total_return_pct']:.1f}%")
    print(f"Sharpe Ratio:      {base_metrics['sharpe']:.3f}")
    print(f"Sortino Ratio:     {base_metrics['sortino']:.3f}")
    print(f"Max Drawdown:      {base_metrics['max_drawdown_pct']:.1f}%")
    print(f"Win Rate:          {base_metrics['win_rate']:.1f}%")
    print(f"Profit Factor:     {base_metrics['profit_factor']:.3f}")
    print(f"Number of Trades:  {base_metrics['n_trades']}")
    print(f"Avg P&L/Trade:     ${base_metrics['avg_pnl']:.2f}")
    print(f"\nAdversarial Tests: {n_pass}/{n_total} passed")
    for a in adversarial_results:
        status = "PASS" if a["pass"] else "FAIL"
        print(f"  [{status}] {a['test']}: {a['description']}")

    return output


if __name__ == "__main__":
    main()
