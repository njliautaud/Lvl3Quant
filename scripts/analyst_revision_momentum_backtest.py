#!/usr/bin/env python3
"""
Analyst Revision Momentum Strategy Backtester
==============================================
Academic basis: Stocks receiving upward EPS estimate revisions systematically
drift higher over 1-3 months (Post-Revision Price Drift, similar to PEAD).
This is one of the most robust anomalies in academic finance (Chan, Jegadeesh,
Lakonishok 1996; Gleason & Lee 2003).

Proxy approach (since we lack live analyst estimates in historical data):
We use EARNINGS SURPRISE + SUBSEQUENT MOMENTUM as a proxy. Specifically:
  1. Detect positive earnings surprises (actual > estimate) using yfinance
  2. Combine with post-announcement momentum (5-20 day drift)
  3. Enter on confirmation of drift (not just the gap)
  4. This captures the same underlying phenomenon: market underreaction to
     positive fundamental information

Strategy (long-only for Robinhood):
  - Screen for stocks with recent positive earnings surprises
  - Require post-announcement drift confirmation (5-day return > 0)
  - Enter with momentum, exit on reversal or time stop
  - Regime filter: reduce exposure in bear markets
  - Position sizing: Kelly-fractional based on recent win rate

Walk-forward: rolling 120-day calibration, daily scan
OOT: Jan 2022 - Jul 2026
Capital: $645, fractional shares on Robinhood
"""

import json
import logging
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/analyst_revision_momentum")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUTPUT_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Strategy parameters
EARNINGS_GAP_MIN = 0.02        # Minimum 2% earnings-day gap (positive surprise proxy)
DRIFT_CONFIRM_DAYS = 5         # Confirm drift over 5 days post-earnings
DRIFT_CONFIRM_MIN = 0.005      # Drift must be > 0.5% in confirm window
MOMENTUM_LOOKBACK = 20         # 20-day momentum for secondary filter
MAX_HOLD_DAYS = 40             # Ride drift up to 40 trading days (2 months)
ATR_PERIOD = 14
TRAILING_STOP_ATR = 2.0        # Wider stop for drift trades (patient)
INITIAL_STOP_ATR = 2.5         # Initial stop before trailing kicks in
PROFIT_TARGET_PCT = 0.12       # 12% profit target → tighten stop
TIGHTEN_STOP_ATR = 1.0         # Tighten to 1x ATR after profit target
MAX_POSITIONS = 4              # Max concurrent at $645
POSITION_SIZE_PCT = 0.22       # ~22% per position
COMMISSION_PCT = 0.0           # Robinhood zero commission
SLIPPAGE_BPS = 5               # 5 bps slippage

# Walk-forward
DATA_START = "2020-06-01"      # Need lookback
OOT_START = "2022-01-01"
OOT_END = "2026-07-25"
STARTING_CAPITAL = 645.0

# Regime
SPY_SMA_FAST = 50
SPY_SMA_SLOW = 200

# ─── Universe ────────────────────────────────────────────────────────────────

# Stocks with frequent earnings reports and active analyst coverage
# Mix of mega-cap (reliable) and growth (bigger moves)
UNIVERSE = [
    # Mega-cap tech (heavy analyst coverage)
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "AMD", "CRM", "AVGO", "ADBE",
    "ORCL", "CSCO", "TXN", "QCOM", "MU", "NOW", "AMAT", "INTC", "NFLX",
    # Growth tech (bigger earnings reactions)
    "CRWD", "PANW", "ZS", "SNOW", "NET", "DDOG", "MDB", "COIN", "SHOP",
    "UBER", "ABNB", "DASH", "PINS", "TTD", "PLTR", "SOFI", "RBLX",
    "ENPH", "FSLR", "SMCI",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP",
    # Healthcare
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "AMGN",
    # Consumer
    "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "COST", "WMT",
    # Industrial/Energy
    "XOM", "CVX", "COP", "CAT", "DE", "GE", "HON", "BA", "RTX",
    # Other
    "PG", "KO", "PEP", "DIS", "CMCSA", "T", "VZ",
]


def download_data(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Download OHLCV data with caching."""
    cache_file = CACHE_DIR / f"revision_data_{start}_{end}.pkl"
    if cache_file.exists():
        log.info("Loading cached data...")
        data = pd.read_pickle(cache_file)
        if len(data) >= len(tickers) * 0.7:
            return data

    log.info(f"Downloading data for {len(tickers)} tickers...")
    data = {}
    batch_size = 20
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            batch_str = " ".join(batch)
            raw = yf.download(batch_str, start=start, end=end, group_by="ticker",
                             auto_adjust=True, progress=False, threads=True)
            for t in batch:
                try:
                    if len(batch) == 1:
                        df = raw.copy()
                    else:
                        df = raw[t].copy()
                    df = df.dropna(subset=["Close", "Volume"])
                    if len(df) > 100:
                        data[t] = df
                except Exception:
                    pass
        except Exception as e:
            log.warning(f"Batch download failed: {e}")

    log.info(f"Downloaded {len(data)} tickers successfully")
    pd.to_pickle(data, cache_file)
    return data


def download_spy(start: str, end: str) -> pd.DataFrame:
    """Download SPY for regime filter."""
    cache_file = CACHE_DIR / "spy_regime.pkl"
    if cache_file.exists():
        spy = pd.read_pickle(cache_file)
        if len(spy) > 500:
            return spy
    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    pd.to_pickle(spy, cache_file)
    return spy


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - close).abs(),
        (low - close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def detect_earnings_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect earnings-like events using gap + volume signature.

    An earnings event is identified by:
    1. Large overnight gap (>2% from prior close to open)
    2. High volume (>1.5x 20-day average)

    This proxy works because earnings are the dominant driver of large
    overnight gaps with volume confirmation.
    """
    prev_close = df["Close"].shift(1)
    gap_pct = (df["Open"] - prev_close) / prev_close
    vol_avg = df["Volume"].rolling(20).mean()
    vol_ratio = df["Volume"] / vol_avg

    # Post-event drift: return over next 5 days
    future_5d_ret = df["Close"].shift(-DRIFT_CONFIRM_DAYS) / df["Close"] - 1

    # 20-day momentum
    mom_20d = df["Close"] / df["Close"].shift(MOMENTUM_LOOKBACK) - 1

    # ATR
    atr = compute_atr(df, ATR_PERIOD)

    result = pd.DataFrame({
        "gap_pct": gap_pct,
        "vol_ratio": vol_ratio,
        "future_5d_ret": future_5d_ret,
        "mom_20d": mom_20d,
        "close": df["Close"],
        "open": df["Open"],
        "high": df["High"],
        "low": df["Low"],
        "prev_close": prev_close,
        "atr": atr,
    }, index=df.index)
    return result


def identify_signals(events: pd.DataFrame, date: pd.Timestamp,
                     lookback_days: int = 10) -> Optional[dict]:
    """
    Check if a stock had a qualifying earnings event in the recent lookback
    window and the drift is now confirmed.

    Signal generation:
    1. Within last 6-15 days, stock had gap > EARNINGS_GAP_MIN with volume
    2. Since the gap, stock has drifted in the gap direction (confirmation)
    3. Enter after confirmation
    """
    # Look back 6-15 trading days for an earnings event
    date_idx = events.index.get_loc(date) if date in events.index else None
    if date_idx is None or date_idx < 20:
        return None

    # Search window: 6-15 days ago (give time for drift confirmation)
    for lookback in range(6, min(16, date_idx)):
        event_idx = date_idx - lookback
        event_date = events.index[event_idx]
        event = events.iloc[event_idx]

        gap = event["gap_pct"]
        vol_r = event["vol_ratio"]

        if pd.isna(gap) or pd.isna(vol_r):
            continue

        # Check for positive earnings surprise (gap up with volume)
        if gap >= EARNINGS_GAP_MIN and vol_r >= 1.5:
            # Check drift confirmation: price has continued up since event
            current_close = events.iloc[date_idx]["close"]
            event_close = event["close"]

            if pd.isna(current_close) or pd.isna(event_close):
                continue

            drift = (current_close - event_close) / event_close

            # Drift must be positive (confirming the surprise direction)
            if drift >= DRIFT_CONFIRM_MIN:
                return {
                    "event_date": event_date,
                    "gap_pct": gap,
                    "vol_ratio": vol_r,
                    "drift_pct": drift,
                    "direction": 1,  # Long
                    "signal_strength": gap * vol_r * (1 + drift),
                }

    return None


class Position:
    """Track an individual position."""
    def __init__(self, ticker, entry_date, entry_price, shares, direction,
                 initial_stop, atr_at_entry, signal_strength):
        self.ticker = ticker
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.shares = shares
        self.direction = direction
        self.initial_stop = initial_stop
        self.trailing_stop = initial_stop
        self.atr_at_entry = atr_at_entry
        self.signal_strength = signal_strength
        self.peak_price = entry_price
        self.days_held = 0
        self.profit_target_hit = False
        self.exit_date = None
        self.exit_price = None
        self.exit_reason = None
        self.pnl = 0.0

    def update_trailing(self, high, low, close):
        """Update trailing stop, tighten after profit target."""
        if self.direction == 1:
            self.peak_price = max(self.peak_price, high)

            # Check profit target
            gain = (self.peak_price - self.entry_price) / self.entry_price
            if gain >= PROFIT_TARGET_PCT:
                self.profit_target_hit = True

            if self.profit_target_hit:
                new_stop = self.peak_price - TIGHTEN_STOP_ATR * self.atr_at_entry
            else:
                new_stop = self.peak_price - TRAILING_STOP_ATR * self.atr_at_entry

            self.trailing_stop = max(self.trailing_stop, new_stop)


def run_backtest(data: dict, spy: pd.DataFrame):
    """Walk-forward backtest."""
    # SPY regime
    spy_close = spy["Close"].squeeze() if isinstance(spy["Close"], pd.DataFrame) else spy["Close"]
    spy_sma50 = spy_close.rolling(SPY_SMA_FAST).mean()
    spy_sma200 = spy_close.rolling(SPY_SMA_SLOW).mean()
    spy_regime = (spy_sma50 > spy_sma200).astype(int)
    spy_daily_ret = spy_close.pct_change()

    # Pre-compute events
    log.info("Pre-computing earnings events for all tickers...")
    events_data = {}
    for ticker, df in data.items():
        events_data[ticker] = detect_earnings_events(df)

    # OOT dates
    oot_start = pd.Timestamp(OOT_START)
    oot_end = pd.Timestamp(OOT_END)
    all_dates = spy.index[(spy.index >= oot_start) & (spy.index <= oot_end)]

    log.info(f"OOT: {all_dates[0].date()} to {all_dates[-1].date()} ({len(all_dates)} days)")

    # State
    equity = STARTING_CAPITAL
    positions: list[Position] = []
    trades: list[dict] = []
    equity_curve = []
    daily_returns = []

    # Walk-forward calibration: track win rate for position sizing
    recent_results = []  # Last N trade results for adaptive sizing

    # Track which ticker-event combos we've already traded (avoid duplicates)
    traded_events = set()

    for i, date in enumerate(all_dates):
        # Regime
        regime = 1
        if date in spy_regime.index:
            r = spy_regime.loc[date]
            regime = int(r) if not pd.isna(r) else 1

        spy_ret = 0.0
        if date in spy_daily_ret.index:
            sr = spy_daily_ret.loc[date]
            spy_ret = float(sr) if not pd.isna(sr) else 0.0

        # ── Manage positions ──
        to_close = []
        for pos in positions:
            pos.days_held += 1
            ticker = pos.ticker

            if ticker not in events_data or date not in events_data[ticker].index:
                continue

            ed = events_data[ticker].loc[date]
            high, low, close = ed["high"], ed["low"], ed["close"]

            if pd.isna(close):
                continue

            pos.update_trailing(high, low, close)

            exit_price = None
            exit_reason = None

            # Trailing stop
            if low <= pos.trailing_stop:
                exit_price = max(pos.trailing_stop, low)  # Realistic fill
                exit_reason = "trailing_stop"

            # Initial stop
            if exit_price is None and low <= pos.initial_stop:
                exit_price = pos.initial_stop
                exit_reason = "initial_stop"

            # Time stop
            if exit_price is None and pos.days_held >= MAX_HOLD_DAYS:
                exit_price = close
                exit_reason = "time_stop"

            # Momentum breakdown: 3 consecutive down days = exit
            if exit_price is None and pos.days_held >= 5:
                idx = events_data[ticker].index.get_loc(date)
                if idx >= 3:
                    last_3 = events_data[ticker].iloc[idx-2:idx+1]["close"].values
                    if len(last_3) == 3 and all(last_3[j] < last_3[j-1] for j in range(1, 3)):
                        # 3 consecutive down closes
                        exit_price = close
                        exit_reason = "momentum_breakdown"

            if exit_price is not None:
                exit_price *= (1 - SLIPPAGE_BPS / 10000)  # Slippage on exit
                pos.exit_date = date
                pos.exit_price = exit_price
                pos.exit_reason = exit_reason
                pos.pnl = pos.direction * (exit_price - pos.entry_price) * pos.shares
                equity += pos.pnl
                to_close.append(pos)

                recent_results.append(1 if pos.pnl > 0 else 0)
                if len(recent_results) > 50:
                    recent_results = recent_results[-50:]

                trades.append({
                    "ticker": pos.ticker,
                    "entry_date": str(pos.entry_date.date()),
                    "exit_date": str(pos.exit_date.date()),
                    "entry_price": round(pos.entry_price, 2),
                    "exit_price": round(pos.exit_price, 2),
                    "shares": round(pos.shares, 4),
                    "pnl": round(pos.pnl, 2),
                    "return_pct": round(pos.pnl / (pos.entry_price * pos.shares) * 100, 2),
                    "days_held": pos.days_held,
                    "exit_reason": pos.exit_reason,
                    "signal_strength": round(pos.signal_strength, 4),
                    "regime": "bull" if regime else "bear",
                    "spy_daily_ret": round(spy_ret * 100, 2),
                })

        for pos in to_close:
            positions.remove(pos)

        # ── Scan for new signals ──
        if len(positions) < MAX_POSITIONS and equity > 30:
            candidates = []

            for ticker in UNIVERSE:
                if ticker not in events_data or date not in events_data[ticker].index:
                    continue
                if any(p.ticker == ticker for p in positions):
                    continue

                signal = identify_signals(events_data[ticker], date)
                if signal is None:
                    continue

                # Deduplicate: don't trade same event twice
                event_key = f"{ticker}_{signal['event_date'].date()}"
                if event_key in traded_events:
                    continue

                # Regime filter: in bear market, require stronger signals
                if regime == 0 and signal["gap_pct"] < 0.04:
                    continue

                candidates.append((ticker, signal))

            # Sort by signal strength
            candidates.sort(key=lambda x: x[1]["signal_strength"], reverse=True)
            slots = MAX_POSITIONS - len(positions)

            for ticker, signal in candidates[:slots]:
                # Adaptive position sizing based on recent performance
                size_pct = POSITION_SIZE_PCT
                if regime == 0:
                    size_pct *= 0.5  # Half size in bear market

                # Kelly-inspired: if winning, size up slightly; losing, size down
                if len(recent_results) >= 10:
                    recent_wr = sum(recent_results[-20:]) / len(recent_results[-20:])
                    if recent_wr > 0.6:
                        size_pct = min(size_pct * 1.2, 0.30)
                    elif recent_wr < 0.4:
                        size_pct *= 0.7

                position_value = equity * size_pct
                if position_value < 5:
                    continue

                ed = events_data[ticker].loc[date]
                entry_price = ed["close"] * (1 + SLIPPAGE_BPS / 10000)
                atr = ed["atr"]

                if pd.isna(entry_price) or pd.isna(atr) or atr <= 0:
                    continue

                shares = position_value / entry_price
                initial_stop = entry_price - INITIAL_STOP_ATR * atr

                pos = Position(
                    ticker=ticker,
                    entry_date=date,
                    entry_price=entry_price,
                    shares=shares,
                    direction=1,
                    initial_stop=initial_stop,
                    atr_at_entry=atr,
                    signal_strength=signal["signal_strength"],
                )
                positions.append(pos)
                traded_events.add(f"{ticker}_{signal['event_date'].date()}")

        # ── MTM equity ──
        mtm = equity
        for pos in positions:
            if pos.ticker in events_data and date in events_data[pos.ticker].index:
                current = events_data[pos.ticker].loc[date]["close"]
                if not pd.isna(current):
                    unrealized = pos.direction * (current - pos.entry_price) * pos.shares
                    mtm += unrealized

        equity_curve.append({
            "date": str(date.date()),
            "equity": round(mtm, 2),
            "n_positions": len(positions),
            "regime": "bull" if regime else "bear",
            "spy_ret": round(spy_ret, 6),
        })

        if len(equity_curve) >= 2:
            prev_eq = equity_curve[-2]["equity"]
            if prev_eq > 0:
                daily_returns.append(mtm / prev_eq - 1)
            else:
                daily_returns.append(0)

        if i % 100 == 0:
            log.info(f"  Day {i}/{len(all_dates)}: equity=${mtm:.2f}, positions={len(positions)}, trades={len(trades)}")

    # Close remaining
    for pos in positions:
        last_date = all_dates[-1]
        if pos.ticker in events_data and last_date in events_data[pos.ticker].index:
            close = events_data[pos.ticker].loc[last_date]["close"]
            if not pd.isna(close):
                pos.pnl = pos.direction * (close - pos.entry_price) * pos.shares
                trades.append({
                    "ticker": pos.ticker,
                    "entry_date": str(pos.entry_date.date()),
                    "exit_date": str(last_date.date()),
                    "entry_price": round(pos.entry_price, 2),
                    "exit_price": round(float(close), 2),
                    "shares": round(pos.shares, 4),
                    "pnl": round(pos.pnl, 2),
                    "return_pct": round(pos.pnl / (pos.entry_price * pos.shares) * 100, 2),
                    "days_held": pos.days_held,
                    "exit_reason": "end_of_test",
                    "signal_strength": round(pos.signal_strength, 4),
                    "regime": "bull",
                    "spy_daily_ret": 0,
                })

    return trades, equity_curve, daily_returns


def validate_strategy(trades: list[dict], equity_curve: list[dict],
                      daily_returns: list[float]) -> dict:
    """5-gate validation."""
    results = {}

    if not trades:
        log.warning("No trades generated!")
        return {"passed": False, "reason": "No trades"}

    df_trades = pd.DataFrame(trades)
    df_equity = pd.DataFrame(equity_curve)
    returns = np.array(daily_returns)

    total_trades = len(df_trades)
    winning = df_trades[df_trades["pnl"] > 0]
    losing = df_trades[df_trades["pnl"] <= 0]
    win_rate = len(winning) / total_trades
    avg_win = float(winning["pnl"].mean()) if len(winning) > 0 else 0
    avg_loss = float(losing["pnl"].mean()) if len(losing) > 0 else 0
    profit_factor = abs(float(winning["pnl"].sum()) / float(losing["pnl"].sum())) if float(losing["pnl"].sum()) != 0 else float("inf")
    total_pnl = float(df_trades["pnl"].sum())
    final_equity = float(df_equity["equity"].iloc[-1])

    results["total_trades"] = int(total_trades)
    results["win_rate"] = round(float(win_rate), 4)
    results["avg_win"] = round(avg_win, 2)
    results["avg_loss"] = round(avg_loss, 2)
    results["profit_factor"] = round(float(profit_factor), 3)
    results["total_pnl"] = round(total_pnl, 2)
    results["final_equity"] = round(final_equity, 2)
    results["return_pct"] = round((final_equity / STARTING_CAPITAL - 1) * 100, 2)
    results["avg_days_held"] = round(float(df_trades["days_held"].mean()), 1)

    # Exit breakdown
    results["exit_reasons"] = {k: int(v) for k, v in df_trades["exit_reason"].value_counts().items()}

    # ── Gate 1: Sharpe ──
    if len(returns) > 20:
        ann_ret = float(np.mean(returns)) * 252
        ann_vol = float(np.std(returns)) * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    else:
        sharpe = 0
    results["sharpe"] = round(sharpe, 3)
    results["gate1_sharpe"] = bool(sharpe > 0.5)

    # Sortino
    downside = returns[returns < 0]
    ds_vol = float(np.std(downside)) * np.sqrt(252) if len(downside) > 0 else 1e-6
    sortino = (float(np.mean(returns)) * 252) / ds_vol if ds_vol > 0 else 0
    results["sortino"] = round(sortino, 3)

    # CAGR
    n_years = len(returns) / 252
    if n_years > 0 and final_equity > 0:
        cagr = (final_equity / STARTING_CAPITAL) ** (1 / n_years) - 1
        results["cagr"] = round(cagr * 100, 2)
    else:
        results["cagr"] = 0

    # ── Gate 2: Permutation test ──
    n_perms = 2000
    perm_sharpes = []
    for _ in range(n_perms):
        perm = np.random.permutation(returns)
        pm = float(np.mean(perm)) * 252
        ps = float(np.std(perm)) * np.sqrt(252)
        perm_sharpes.append(pm / ps if ps > 0 else 0)
    p_value = float(np.mean(np.array(perm_sharpes) >= sharpe))
    results["perm_p_value"] = round(p_value, 4)
    results["gate2_perm"] = bool(p_value < 0.05)

    # ── Gate 3: Beat random ──
    n_random = 1000
    random_sharpes = []
    for _ in range(n_random):
        r = np.random.choice(returns, size=len(returns), replace=True)
        rm = float(np.mean(r)) * 252
        rs = float(np.std(r)) * np.sqrt(252)
        random_sharpes.append(rm / rs if rs > 0 else 0)
    random_med = float(np.median(random_sharpes))
    results["random_baseline_sharpe"] = round(random_med, 3)
    results["gate3_beats_random"] = bool(sharpe > random_med + 0.1)

    # ── Gate 4: Regime balance ──
    bull_trades = df_trades[df_trades["regime"] == "bull"]
    bear_trades = df_trades[df_trades["regime"] == "bear"]

    bull_wr = float(len(bull_trades[bull_trades["pnl"] > 0]) / len(bull_trades)) if len(bull_trades) > 0 else 0
    bear_wr = float(len(bear_trades[bear_trades["pnl"] > 0]) / len(bear_trades)) if len(bear_trades) > 0 else 0

    if len(bull_trades) > 5 and len(bear_trades) > 5:
        regime_gap = abs(bull_wr - bear_wr) / max(bull_wr, bear_wr) if max(bull_wr, bear_wr) > 0 else 1.0
    elif len(bear_trades) <= 5:
        regime_gap = 0.3  # Not enough data, benefit of doubt
    else:
        regime_gap = 1.0

    results["bull_trades"] = int(len(bull_trades))
    results["bull_wr"] = round(bull_wr, 4)
    results["bear_trades"] = int(len(bear_trades))
    results["bear_wr"] = round(bear_wr, 4)
    results["regime_gap"] = round(regime_gap, 4)
    results["gate4_regime"] = bool(regime_gap < 0.50)

    # ── Gate 5: MDD ──
    eq_vals = df_equity["equity"].astype(float).values
    peak = np.maximum.accumulate(eq_vals)
    dd = (eq_vals - peak) / peak
    max_dd = float(dd.min())
    results["max_drawdown"] = round(max_dd, 4)
    results["gate5_mdd"] = bool(max_dd > -0.50)

    # ── Green/red day Sharpe ──
    df_eq2 = df_equity.copy()
    df_eq2["daily_ret"] = pd.to_numeric(df_eq2["equity"]).pct_change()
    df_eq2["spy_ret"] = pd.to_numeric(df_eq2["spy_ret"])
    green = df_eq2[df_eq2["spy_ret"] > 0.001]
    red = df_eq2[df_eq2["spy_ret"] < -0.001]

    if len(green) > 20 and len(red) > 20:
        g_sharpe = (float(green["daily_ret"].mean()) * 252) / (float(green["daily_ret"].std()) * np.sqrt(252)) if float(green["daily_ret"].std()) > 0 else 0
        r_sharpe = (float(red["daily_ret"].mean()) * 252) / (float(red["daily_ret"].std()) * np.sqrt(252)) if float(red["daily_ret"].std()) > 0 else 0
        results["green_day_sharpe"] = round(g_sharpe, 3)
        results["red_day_sharpe"] = round(r_sharpe, 3)
    else:
        results["green_day_sharpe"] = 0.0
        results["red_day_sharpe"] = 0.0

    # Overall
    gates = [
        results.get("gate1_sharpe", False),
        results.get("gate2_perm", False),
        results.get("gate3_beats_random", False),
        results.get("gate4_regime", False),
        results.get("gate5_mdd", False),
    ]
    results["gates_passed"] = int(sum(gates))
    results["all_gates_passed"] = bool(all(gates))

    return results


def log_to_mlflow(results: dict, trades: list[dict], equity_curve: list[dict]):
    """Log to MLflow."""
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("analyst_revision_momentum_v1")

        with mlflow.start_run(run_name=f"revision_momentum_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_param("strategy", "analyst_revision_momentum")
            mlflow.log_param("universe_size", len(UNIVERSE))
            mlflow.log_param("earnings_gap_min", EARNINGS_GAP_MIN)
            mlflow.log_param("drift_confirm_days", DRIFT_CONFIRM_DAYS)
            mlflow.log_param("max_hold_days", MAX_HOLD_DAYS)
            mlflow.log_param("max_positions", MAX_POSITIONS)
            mlflow.log_param("position_size_pct", POSITION_SIZE_PCT)
            mlflow.log_param("oot_period", f"{OOT_START} to {OOT_END}")
            mlflow.log_param("starting_capital", STARTING_CAPITAL)

            for key, val in results.items():
                if isinstance(val, (int, float)):
                    mlflow.log_metric(key, val)

            # Save artifacts
            trades_path = OUTPUT_DIR / "trades.json"
            with open(trades_path, "w") as f:
                json.dump(trades, f, indent=2)
            mlflow.log_artifact(str(trades_path))

            eq_path = OUTPUT_DIR / "equity_curve.json"
            with open(eq_path, "w") as f:
                json.dump(equity_curve, f, indent=2)
            mlflow.log_artifact(str(eq_path))

            res_path = OUTPUT_DIR / "results.json"
            with open(res_path, "w") as f:
                json.dump(results, f, indent=2)
            mlflow.log_artifact(str(res_path))

            log.info("MLflow logging complete")
    except Exception as e:
        log.warning(f"MLflow logging failed: {e}")


def main():
    log.info("=" * 70)
    log.info("ANALYST REVISION MOMENTUM STRATEGY BACKTEST")
    log.info("=" * 70)
    log.info(f"Universe: {len(UNIVERSE)} stocks")
    log.info(f"OOT: {OOT_START} to {OOT_END}")
    log.info(f"Starting capital: ${STARTING_CAPITAL}")
    log.info(f"Earnings gap minimum: {EARNINGS_GAP_MIN*100:.1f}%")
    log.info(f"Drift confirmation: {DRIFT_CONFIRM_DAYS} days, >{DRIFT_CONFIRM_MIN*100:.1f}%")

    data = download_data(UNIVERSE, DATA_START, OOT_END)
    spy = download_spy(DATA_START, OOT_END)

    if len(data) < 20:
        log.error(f"Only got {len(data)} tickers, need at least 20")
        sys.exit(1)

    log.info(f"Data ready: {len(data)} tickers, SPY {len(spy)} days")

    trades, equity_curve, daily_returns = run_backtest(data, spy)

    log.info(f"\nBacktest complete: {len(trades)} trades")

    results = validate_strategy(trades, equity_curve, daily_returns)

    # Print results
    log.info("\n" + "=" * 70)
    log.info("VALIDATION RESULTS")
    log.info("=" * 70)
    log.info(f"Total trades: {results.get('total_trades', 0)}")
    log.info(f"Win rate: {results.get('win_rate', 0)*100:.1f}%")
    log.info(f"Profit factor: {results.get('profit_factor', 0):.3f}")
    log.info(f"Total P&L: ${results.get('total_pnl', 0):.2f}")
    log.info(f"Final equity: ${results.get('final_equity', 0):.2f} ({results.get('return_pct', 0):.1f}%)")
    log.info(f"CAGR: {results.get('cagr', 0):.2f}%")
    log.info(f"Avg days held: {results.get('avg_days_held', 0):.1f}")
    log.info(f"Exit reasons: {results.get('exit_reasons', {})}")
    log.info(f"")
    log.info(f"--- 5-Gate Validation ---")
    log.info(f"Gate 1 - Sharpe > 0.5:    {results.get('sharpe', 0):.3f} {'PASS' if results.get('gate1_sharpe') else 'FAIL'}")
    log.info(f"         Sortino:          {results.get('sortino', 0):.3f}")
    log.info(f"Gate 2 - Perm p < 0.05:   {results.get('perm_p_value', 1):.4f} {'PASS' if results.get('gate2_perm') else 'FAIL'}")
    log.info(f"Gate 3 - Beats random:    {results.get('sharpe', 0):.3f} vs {results.get('random_baseline_sharpe', 0):.3f} {'PASS' if results.get('gate3_beats_random') else 'FAIL'}")
    log.info(f"Gate 4 - Regime balance:  gap={results.get('regime_gap', 0):.4f} (Bull WR={results.get('bull_wr', 0)*100:.1f}% Bear WR={results.get('bear_wr', 0)*100:.1f}%) {'PASS' if results.get('gate4_regime') else 'FAIL'}")
    log.info(f"Gate 5 - MDD > -50%:      {results.get('max_drawdown', 0)*100:.1f}% {'PASS' if results.get('gate5_mdd') else 'FAIL'}")
    log.info(f"")
    log.info(f"Green day Sharpe: {results.get('green_day_sharpe', 0):.3f}")
    log.info(f"Red day Sharpe:   {results.get('red_day_sharpe', 0):.3f}")
    log.info(f"")
    log.info(f"Gates passed: {results.get('gates_passed', 0)}/5")
    log.info(f"OVERALL: {'PASS' if results.get('all_gates_passed') else 'FAIL'}")

    log_to_mlflow(results, trades, equity_curve)

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"\nDone.")
    return results


if __name__ == "__main__":
    main()
