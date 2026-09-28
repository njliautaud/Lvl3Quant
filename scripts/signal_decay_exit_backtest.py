#!/usr/bin/env python3
"""
Signal Decay Exit Backtest
==========================
Compare signal-aware exit strategies vs fixed price-based exits
for sector ETF options trading.

Exit variants tested:
  A) Signal Count Decay — exit when confirming signals drop below 3
  B) Key Signal Flip — exit when the strongest entry signal flips
  C) Momentum Reversal — exit when 5d momentum flips from entry direction
  D) RSI Mean-Revert Complete — exit when RSI crosses 50 from oversold/overbought
  E) Hybrid — first of: signal count < 3 OR key signal flip OR fixed TP/SL
  F) Baseline — fixed exits only (+30% TP, -25% SL, 5d time stop, trailing 15%/50%)
"""

import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
SECTOR_ETFS = ["XLE", "XLU", "XLP", "XLF", "XLK", "XLC", "XLV", "XLY", "XLRE", "XLB", "XLI"]
MARKET_TICKERS = ["^VIX", "SPY", "TLT", "HYG"]
ALL_TICKERS = SECTOR_ETFS + MARKET_TICKERS

LOOKBACK = 252  # sliding window (HC #0)
COST_RT = 0.001  # 0.10% round-trip for options spread proxy

# Fixed exit params (baseline)
TP_PCT = 0.30
SL_PCT = -0.25
TIME_STOP_DAYS = 5
TRAILING_TRIGGER_PCT = 0.15
TRAILING_STOP_PCT = 0.50  # trail 50% of peak gain once triggered

N_PERMUTATIONS = 1000
REGIME_GAP_THRESHOLD = 0.50

# ── Data Download ───────────────────────────────────────────────────────────
def download_data() -> pd.DataFrame:
    """Download 2 years of daily data for all tickers."""
    end = datetime.now()
    start = end - timedelta(days=2 * 365 + 60)  # extra buffer for indicator warmup

    print(f"Downloading data from {start.date()} to {end.date()}...")
    data = yf.download(ALL_TICKERS, start=start, end=end, auto_adjust=True, progress=False)

    # Handle multi-level columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
        high = data["High"]
        low = data["Low"]
        volume = data["Volume"]
    else:
        close = data[["Close"]].copy()
        high = data[["High"]].copy()
        low = data[["Low"]].copy()
        volume = data[["Volume"]].copy()

    print(f"  Downloaded {len(close)} trading days, {len(close.columns)} tickers")
    return close, high, low, volume


# ── Signal Computation ──────────────────────────────────────────────────────
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_signals(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame,
                    volume: pd.DataFrame) -> dict:
    """
    For each sector ETF and each day, compute signal values.
    Returns dict of DataFrames keyed by signal name.
    """
    signals = {}

    # VIX data (handle ^VIX column name)
    vix_col = "^VIX" if "^VIX" in close.columns else "VIX"
    vix = close[vix_col] if vix_col in close.columns else pd.Series(dtype=float)

    spy = close["SPY"] if "SPY" in close.columns else pd.Series(dtype=float)

    for ticker in SECTOR_ETFS:
        if ticker not in close.columns:
            print(f"  WARNING: {ticker} not in data, skipping")
            continue

        px = close[ticker].dropna()
        if len(px) < LOOKBACK:
            continue

        # RSI(14)
        rsi = compute_rsi(px, 14)

        # 20-day SMA and deviation
        sma20 = px.rolling(20).mean()
        pct_from_sma = (px - sma20) / sma20

        # 5-day momentum (simple return)
        mom5 = px.pct_change(5)

        # 21-day momentum
        mom21 = px.pct_change(21)

        # VIX elevated (> 20)
        vix_elevated = (vix > 20).astype(float).reindex(px.index, method="ffill")

        # Sector vs SPY relative performance (5d)
        if len(spy) > 0:
            spy_aligned = spy.reindex(px.index, method="ffill")
            sector_ret_5d = px.pct_change(5)
            spy_ret_5d = spy_aligned.pct_change(5)
            rel_perf = sector_ret_5d - spy_ret_5d
        else:
            rel_perf = pd.Series(0, index=px.index)

        # Store per-ticker signals
        signals[ticker] = pd.DataFrame({
            "close": px,
            "rsi": rsi,
            "sma20": sma20,
            "pct_from_sma": pct_from_sma,
            "mom5": mom5,
            "mom21": mom21,
            "vix_elevated": vix_elevated,
            "rel_perf_5d": rel_perf,
        })

    return signals


def count_bullish_signals(row) -> int:
    """Count how many signals confirm a bullish thesis."""
    count = 0
    if row["rsi"] < 35: count += 1
    if row["pct_from_sma"] < -0.05: count += 1
    if row["mom5"] < 0: count += 1  # oversold bounce setup
    if row["mom21"] < 0: count += 1  # longer-term oversold
    if row["vix_elevated"] > 0.5: count += 1  # fear = opportunity
    if row["rel_perf_5d"] < -0.02: count += 1  # underperforming = catch-up
    return count


def count_bearish_signals(row) -> int:
    """Count how many signals confirm a bearish thesis."""
    count = 0
    if row["rsi"] > 65: count += 1
    if row["pct_from_sma"] > 0.05: count += 1
    if row["mom5"] > 0: count += 1
    if row["mom21"] > 0: count += 1
    if row["vix_elevated"] < 0.5: count += 1  # complacency
    if row["rel_perf_5d"] > 0.02: count += 1  # overperforming = revert
    return count


def get_strongest_signal(row, direction: str) -> str:
    """Identify which signal has the most extreme z-score at entry."""
    if direction == "long":
        scores = {
            "rsi": (30 - row["rsi"]) / 10 if row["rsi"] < 50 else 0,
            "sma_dev": abs(row["pct_from_sma"]) / 0.05 if row["pct_from_sma"] < 0 else 0,
            "mom5": abs(row["mom5"]) / 0.03 if row["mom5"] < 0 else 0,
            "mom21": abs(row["mom21"]) / 0.05 if row["mom21"] < 0 else 0,
            "rel_perf": abs(row["rel_perf_5d"]) / 0.02 if row["rel_perf_5d"] < -0.02 else 0,
        }
    else:  # short
        scores = {
            "rsi": (row["rsi"] - 70) / 10 if row["rsi"] > 50 else 0,
            "sma_dev": abs(row["pct_from_sma"]) / 0.05 if row["pct_from_sma"] > 0 else 0,
            "mom5": abs(row["mom5"]) / 0.03 if row["mom5"] > 0 else 0,
            "mom21": abs(row["mom21"]) / 0.05 if row["mom21"] > 0 else 0,
            "rel_perf": abs(row["rel_perf_5d"]) / 0.02 if row["rel_perf_5d"] > 0.02 else 0,
        }
    return max(scores, key=scores.get) if any(v > 0 for v in scores.values()) else "rsi"


def key_signal_flipped(entry_row, current_row, direction: str, key_signal: str) -> bool:
    """Check if the key signal has flipped to neutral/opposite."""
    if key_signal == "rsi":
        if direction == "long":
            return current_row["rsi"] > 50
        else:
            return current_row["rsi"] < 50
    elif key_signal == "sma_dev":
        if direction == "long":
            return current_row["pct_from_sma"] > 0
        else:
            return current_row["pct_from_sma"] < 0
    elif key_signal == "mom5":
        if direction == "long":
            return current_row["mom5"] > 0
        else:
            return current_row["mom5"] < 0
    elif key_signal == "mom21":
        if direction == "long":
            return current_row["mom21"] > 0
        else:
            return current_row["mom21"] < 0
    elif key_signal == "rel_perf":
        if direction == "long":
            return current_row["rel_perf_5d"] > 0
        else:
            return current_row["rel_perf_5d"] < 0
    return False


# ── Trade Simulation ────────────────────────────────────────────────────────
@dataclass
class Trade:
    ticker: str
    direction: str  # "long" or "short"
    entry_date: pd.Timestamp
    entry_price: float
    entry_signals: int
    key_signal: str
    entry_rsi: float
    entry_mom5_sign: float  # +1 or -1
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    pnl_pct: float = 0.0
    hold_days: int = 0
    mfe_pct: float = 0.0  # max favorable excursion
    mae_pct: float = 0.0  # max adverse excursion


def simulate_exits(signals_dict: dict, close: pd.DataFrame, spy_close: pd.Series,
                   variant: str) -> list:
    """
    Simulate entries and exits for a given variant.
    Returns list of Trade objects.
    """
    trades = []

    for ticker in SECTOR_ETFS:
        if ticker not in signals_dict:
            continue

        df = signals_dict[ticker].copy()
        df = df.dropna()

        if len(df) < LOOKBACK + 50:
            continue

        # Only trade within the last 2 years (skip warmup)
        trade_start_idx = LOOKBACK
        in_trade = False
        current_trade = None

        for i in range(trade_start_idx, len(df)):
            row = df.iloc[i]
            date = df.index[i]

            if not in_trade:
                # Check entry conditions
                bull_count = count_bullish_signals(row)
                bear_count = count_bearish_signals(row)

                direction = None
                signal_count = 0

                if row["rsi"] < 35 and row["pct_from_sma"] < -0.05 and bull_count >= 3:
                    direction = "long"
                    signal_count = bull_count
                elif row["rsi"] > 65 and row["pct_from_sma"] > 0.05 and bear_count >= 3:
                    direction = "short"
                    signal_count = bear_count

                if direction:
                    key_sig = get_strongest_signal(row, direction)
                    current_trade = Trade(
                        ticker=ticker,
                        direction=direction,
                        entry_date=date,
                        entry_price=row["close"],
                        entry_signals=signal_count,
                        key_signal=key_sig,
                        entry_rsi=row["rsi"],
                        entry_mom5_sign=1.0 if row["mom5"] > 0 else -1.0,
                    )
                    in_trade = True
                    peak_pnl = 0.0

            else:
                # In a trade — check exits
                px = row["close"]
                entry_px = current_trade.entry_price
                days_held = (date - current_trade.entry_date).days

                if current_trade.direction == "long":
                    pnl_pct = (px - entry_px) / entry_px
                else:
                    pnl_pct = (entry_px - px) / entry_px

                # Track MFE / MAE
                if pnl_pct > current_trade.mfe_pct:
                    current_trade.mfe_pct = pnl_pct
                if pnl_pct < current_trade.mae_pct:
                    current_trade.mae_pct = pnl_pct

                peak_pnl = max(peak_pnl, pnl_pct)
                exit_reason = None

                # ── Variant-specific exit logic ──
                if variant == "A":  # Signal Count Decay
                    if current_trade.direction == "long":
                        sig_count = count_bullish_signals(row)
                    else:
                        sig_count = count_bearish_signals(row)
                    if sig_count < 3:
                        exit_reason = "signal_count_decay"
                    # Still apply hard SL
                    elif pnl_pct <= SL_PCT:
                        exit_reason = "stop_loss"

                elif variant == "B":  # Key Signal Flip
                    if key_signal_flipped(df.iloc[i], row, current_trade.direction,
                                          current_trade.key_signal):
                        exit_reason = "key_signal_flip"
                    elif pnl_pct <= SL_PCT:
                        exit_reason = "stop_loss"

                elif variant == "C":  # Momentum Reversal
                    current_mom5_sign = 1.0 if row["mom5"] > 0 else -1.0
                    if current_trade.direction == "long" and current_mom5_sign > 0 and current_trade.entry_mom5_sign < 0:
                        # Was negative (oversold), now positive — momentum reversed from entry thesis
                        # Actually for a long entry on oversold, mom flipping positive means the bounce happened
                        exit_reason = "momentum_reversal"
                    elif current_trade.direction == "long" and row["mom5"] > 0.03:
                        # Strong positive momentum = mean reversion played out
                        exit_reason = "momentum_reversal"
                    elif current_trade.direction == "short" and row["mom5"] < -0.03:
                        exit_reason = "momentum_reversal"
                    elif pnl_pct <= SL_PCT:
                        exit_reason = "stop_loss"

                elif variant == "D":  # RSI Mean-Revert Complete
                    if current_trade.direction == "long" and row["rsi"] > 50:
                        exit_reason = "rsi_mean_revert"
                    elif current_trade.direction == "short" and row["rsi"] < 50:
                        exit_reason = "rsi_mean_revert"
                    elif pnl_pct <= SL_PCT:
                        exit_reason = "stop_loss"

                elif variant == "E":  # Hybrid
                    if current_trade.direction == "long":
                        sig_count = count_bullish_signals(row)
                    else:
                        sig_count = count_bearish_signals(row)

                    if sig_count < 3:
                        exit_reason = "signal_count_decay"
                    elif key_signal_flipped(df.iloc[i], row, current_trade.direction,
                                            current_trade.key_signal):
                        exit_reason = "key_signal_flip"
                    elif pnl_pct >= TP_PCT:
                        exit_reason = "take_profit"
                    elif pnl_pct <= SL_PCT:
                        exit_reason = "stop_loss"

                elif variant == "F":  # Baseline (fixed exits)
                    if pnl_pct >= TP_PCT:
                        exit_reason = "take_profit"
                    elif pnl_pct <= SL_PCT:
                        exit_reason = "stop_loss"
                    elif days_held >= TIME_STOP_DAYS:
                        exit_reason = "time_stop"
                    elif peak_pnl >= TRAILING_TRIGGER_PCT:
                        trail_level = peak_pnl * (1 - TRAILING_STOP_PCT)
                        if pnl_pct <= trail_level:
                            exit_reason = "trailing_stop"

                # Close trade if exit triggered
                if exit_reason:
                    current_trade.exit_date = date
                    current_trade.exit_price = px
                    current_trade.exit_reason = exit_reason
                    current_trade.pnl_pct = pnl_pct - COST_RT
                    current_trade.hold_days = days_held
                    trades.append(current_trade)
                    in_trade = False
                    current_trade = None
                    peak_pnl = 0.0

        # Force-close any open trade at end
        if in_trade and current_trade:
            last_row = df.iloc[-1]
            px = last_row["close"]
            if current_trade.direction == "long":
                pnl_pct = (px - current_trade.entry_price) / current_trade.entry_price
            else:
                pnl_pct = (current_trade.entry_price - px) / current_trade.entry_price
            current_trade.exit_date = df.index[-1]
            current_trade.exit_price = px
            current_trade.exit_reason = "end_of_data"
            current_trade.pnl_pct = pnl_pct - COST_RT
            current_trade.hold_days = (df.index[-1] - current_trade.entry_date).days
            trades.append(current_trade)

    return trades


# ── Metrics ─────────────────────────────────────────────────────────────────
def compute_metrics(trades: list, spy_close: pd.Series) -> dict:
    """Compute performance metrics for a list of trades."""
    if not trades:
        return {
            "n_trades": 0, "sharpe": 0, "sortino": 0, "win_rate": 0,
            "profit_factor": 0, "avg_win": 0, "avg_loss": 0, "mdd": 0,
            "avg_hold_days": 0, "edge_capture_pct": 0,
            "sharpe_green": 0, "sharpe_red": 0, "regime_gap": 0,
            "exit_reasons": {},
        }

    pnls = np.array([t.pnl_pct for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    n = len(pnls)
    wr = len(wins) / n if n > 0 else 0
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    pf = (wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else float("inf")

    # Sharpe / Sortino (annualized, assuming ~50 trades/year as proxy)
    mean_ret = pnls.mean()
    std_ret = pnls.std() if len(pnls) > 1 else 1e-9
    downside = pnls[pnls < 0]
    downside_std = downside.std() if len(downside) > 1 else 1e-9

    trades_per_year = max(n / 2, 1)  # 2 years of data
    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0
    sortino = (mean_ret / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0

    # Max drawdown (sequential)
    cumulative = np.cumsum(pnls)
    peak = np.maximum.accumulate(cumulative)
    dd = cumulative - peak
    mdd = dd.min() if len(dd) > 0 else 0

    # Average hold time
    hold_days = np.array([t.hold_days for t in trades])
    avg_hold = hold_days.mean()

    # Edge capture efficiency: pnl / mfe
    mfes = np.array([t.mfe_pct for t in trades])
    edge_captures = []
    for t in trades:
        if t.mfe_pct > 0:
            edge_captures.append(t.pnl_pct / t.mfe_pct)
        elif t.pnl_pct <= 0:
            edge_captures.append(0)
    edge_capture = np.mean(edge_captures) if edge_captures else 0

    # Regime stratification: green vs red SPY days
    spy_daily_ret = spy_close.pct_change()
    green_pnls = []
    red_pnls = []
    for t in trades:
        if t.entry_date in spy_daily_ret.index:
            spy_ret = spy_daily_ret.loc[t.entry_date]
            if spy_ret > 0:
                green_pnls.append(t.pnl_pct)
            else:
                red_pnls.append(t.pnl_pct)

    green_arr = np.array(green_pnls) if green_pnls else np.array([0])
    red_arr = np.array(red_pnls) if red_pnls else np.array([0])

    green_std = green_arr.std() if len(green_arr) > 1 else 1e-9
    red_std = red_arr.std() if len(red_arr) > 1 else 1e-9

    sharpe_green = (green_arr.mean() / green_std) * np.sqrt(max(len(green_arr)/2, 1)) if green_std > 0 else 0
    sharpe_red = (red_arr.mean() / red_std) * np.sqrt(max(len(red_arr)/2, 1)) if red_std > 0 else 0

    max_sharpe = max(abs(sharpe_green), abs(sharpe_red), 1e-9)
    regime_gap = abs(sharpe_green - sharpe_red) / max_sharpe

    # Exit reason breakdown
    reasons = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

    return {
        "n_trades": n,
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "win_rate": round(wr * 100, 1),
        "profit_factor": round(min(pf, 99.9), 2),
        "avg_win_pct": round(avg_win * 100, 2),
        "avg_loss_pct": round(avg_loss * 100, 2),
        "mdd_pct": round(mdd * 100, 2),
        "avg_hold_days": round(avg_hold, 1),
        "edge_capture_pct": round(edge_capture * 100, 1),
        "sharpe_green": round(sharpe_green, 3),
        "sharpe_red": round(sharpe_red, 3),
        "regime_gap": round(regime_gap, 3),
        "regime_ok": regime_gap <= REGIME_GAP_THRESHOLD,
        "exit_reasons": reasons,
    }


def compute_per_sector(trades: list) -> dict:
    """Per-sector breakdown."""
    by_sector = {}
    for t in trades:
        by_sector.setdefault(t.ticker, []).append(t)

    result = {}
    for ticker, sector_trades in by_sector.items():
        pnls = [t.pnl_pct for t in sector_trades]
        result[ticker] = {
            "n_trades": len(pnls),
            "avg_pnl_pct": round(np.mean(pnls) * 100, 2),
            "win_rate": round(sum(1 for p in pnls if p > 0) / len(pnls) * 100, 1),
        }
    return result


# ── Permutation Test ────────────────────────────────────────────────────────
def permutation_test(trades_variant: list, trades_baseline: list, n_perms: int = 1000) -> float:
    """
    Test if variant Sharpe is significantly different from baseline.
    Shuffle the assignment of trades between variant and baseline.
    Returns p-value.
    """
    if not trades_variant or not trades_baseline:
        return 1.0

    pnls_v = np.array([t.pnl_pct for t in trades_variant])
    pnls_b = np.array([t.pnl_pct for t in trades_baseline])

    observed_diff = pnls_v.mean() - pnls_b.mean()

    # Pool all PnLs
    pooled = np.concatenate([pnls_v, pnls_b])
    n_v = len(pnls_v)

    count_extreme = 0
    rng = np.random.RandomState(42)
    for _ in range(n_perms):
        perm = rng.permutation(pooled)
        perm_v = perm[:n_v]
        perm_b = perm[n_v:]
        perm_diff = perm_v.mean() - perm_b.mean()
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1

    return count_extreme / n_perms


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("SIGNAL DECAY EXIT BACKTEST")
    print("=" * 70)

    # 1. Download data
    close, high, low, volume = download_data()

    # SPY for regime classification
    spy_close = close["SPY"] if "SPY" in close.columns else pd.Series(dtype=float)

    # 2. Compute signals
    print("\nComputing signals for all sector ETFs...")
    signals_dict = compute_signals(close, high, low, volume)
    print(f"  Signals computed for {len(signals_dict)} tickers")

    # 3. Run all variants
    variants = {
        "A": "Signal Count Decay",
        "B": "Key Signal Flip",
        "C": "Momentum Reversal",
        "D": "RSI Mean-Revert Complete",
        "E": "Hybrid (signals + fixed TP/SL)",
        "F": "Baseline (fixed exits only)",
    }

    all_results = {}
    all_trades = {}

    for code, name in variants.items():
        print(f"\n  Simulating variant {code}: {name}...")
        trades = simulate_exits(signals_dict, close, spy_close, code)
        metrics = compute_metrics(trades, spy_close)
        sector_breakdown = compute_per_sector(trades)
        metrics["sector_breakdown"] = sector_breakdown

        all_results[code] = {"name": name, **metrics}
        all_trades[code] = trades
        print(f"    {metrics['n_trades']} trades | Sharpe {metrics['sharpe']:.3f} | "
              f"WR {metrics['win_rate']:.1f}% | PF {metrics['profit_factor']:.2f} | "
              f"Avg hold {metrics['avg_hold_days']:.1f}d | Edge capture {metrics['edge_capture_pct']:.1f}%")

    # 4. Permutation tests vs baseline
    print("\n\nPermutation tests (vs baseline F)...")
    baseline_trades = all_trades["F"]
    for code in ["A", "B", "C", "D", "E"]:
        if all_trades[code]:
            p_val = permutation_test(all_trades[code], baseline_trades, N_PERMUTATIONS)
            all_results[code]["p_value_vs_baseline"] = round(p_val, 4)
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
            print(f"  {code} vs F: p={p_val:.4f} {sig}")
        else:
            all_results[code]["p_value_vs_baseline"] = 1.0

    # 5. Print comparison table
    print("\n" + "=" * 110)
    print(f"{'Variant':<40} {'N':>5} {'Sharpe':>7} {'Sortino':>8} {'WR%':>6} {'PF':>6} "
          f"{'AvgHold':>7} {'EdgeCap%':>9} {'MDD%':>7} {'RegGap':>7} {'p-val':>6}")
    print("-" * 110)

    baseline_sharpe = all_results["F"]["sharpe"]
    for code in ["F", "A", "B", "C", "D", "E"]:
        r = all_results[code]
        marker = ""
        if code != "F":
            if r["sharpe"] > baseline_sharpe:
                marker = " <-- BEATS BASELINE"
            p = r.get("p_value_vs_baseline", 1.0)
            if p < 0.05:
                marker += " (significant)"

        print(f"  {code}: {r['name']:<36} {r['n_trades']:>5} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} "
              f"{r['win_rate']:>5.1f}% {r['profit_factor']:>6.2f} {r['avg_hold_days']:>6.1f}d "
              f"{r['edge_capture_pct']:>8.1f}% {r['mdd_pct']:>6.2f}% {r['regime_gap']:>6.3f}"
              f"{marker}")

    # Regime check
    print("\n  Regime robustness check (gap < 0.50 = pass):")
    for code in ["F", "A", "B", "C", "D", "E"]:
        r = all_results[code]
        status = "PASS" if r["regime_ok"] else "FAIL"
        print(f"    {code}: Green Sharpe={r['sharpe_green']:.3f}  Red Sharpe={r['sharpe_red']:.3f}  "
              f"Gap={r['regime_gap']:.3f}  [{status}]")

    # Exit reason breakdown
    print("\n  Exit reason breakdown:")
    for code in ["A", "B", "C", "D", "E", "F"]:
        r = all_results[code]
        reasons_str = ", ".join(f"{k}: {v}" for k, v in sorted(r["exit_reasons"].items()))
        print(f"    {code}: {reasons_str}")

    # 6. Key finding
    print("\n" + "=" * 70)
    print("KEY FINDINGS:")
    best_code = max(all_results.keys(), key=lambda c: all_results[c]["sharpe"])
    best = all_results[best_code]
    print(f"  Best variant: {best_code} ({best['name']}) — Sharpe {best['sharpe']:.3f}")
    if best_code != "F":
        print(f"  Signal-aware exit BEATS baseline by "
              f"{best['sharpe'] - baseline_sharpe:.3f} Sharpe points")
        print(f"  Edge capture: {best['edge_capture_pct']:.1f}% vs baseline "
              f"{all_results['F']['edge_capture_pct']:.1f}%")
    else:
        print(f"  Baseline fixed exits remain best — signal-aware exits did NOT improve performance")

    best_ec = max(all_results.keys(), key=lambda c: all_results[c]["edge_capture_pct"])
    if best_ec != best_code:
        print(f"  Highest edge capture: {best_ec} ({all_results[best_ec]['name']}) — "
              f"{all_results[best_ec]['edge_capture_pct']:.1f}%")

    # 7. Save results
    output_path = Path("/home/jupiter/Lvl3Quant/output/signal_decay_exit_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Clean up for JSON serialization
    json_results = {}
    for code, r in all_results.items():
        clean = {k: v for k, v in r.items()}
        json_results[code] = clean

    json_output = {
        "run_date": datetime.now().isoformat(),
        "config": {
            "sector_etfs": SECTOR_ETFS,
            "lookback": LOOKBACK,
            "cost_rt": COST_RT,
            "tp_pct": TP_PCT,
            "sl_pct": SL_PCT,
            "time_stop_days": TIME_STOP_DAYS,
            "trailing_trigger_pct": TRAILING_TRIGGER_PCT,
            "trailing_stop_pct": TRAILING_STOP_PCT,
            "n_permutations": N_PERMUTATIONS,
        },
        "results": json_results,
        "best_variant": best_code,
        "baseline_beaten": best_code != "F",
    }

    with open(output_path, "w") as f:
        json.dump(json_output, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
