#!/usr/bin/env python3
"""
Multi-Strategy Portfolio Allocation Backtest v2
Tests 6 variants of combining 3 validated mean-reversion strategies on quality stocks.

Strategies (all 10-day hold unless noted):
1. Multi-TF Dual Signal L: >5% below 20d high, RSI<35, green after 2+ red, weekly RSI declining 2+ weeks
2. Dual Signal D: >5% below 20d high, RSI<35, green after 2+ red
3. Quality MR-A: >5% below 20d high, RSI<35

Variants A-F test different allocation/sizing approaches.
"""

import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]
START = "2021-06-01"  # extra lookback for indicators
TRADE_START = "2022-01-01"
END = "2026-07-31"
CAPITAL = 669.0
SLIPPAGE_BPS = 2
HOLD_DAYS = 10
RSI_PERIOD = 14
HIGH_LOOKBACK = 20
DIP_THRESHOLD = 0.05  # >5% below 20d high
RSI_THRESHOLD = 35

# ── Helpers ──────────────────────────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_weekly_rsi(daily_close: pd.Series, period: int = 14) -> pd.Series:
    """Compute weekly RSI from daily closes, resampled to Friday."""
    weekly = daily_close.resample("W-FRI").last().dropna()
    wrsi = compute_rsi(weekly, period)
    # Forward-fill back to daily index
    return wrsi.reindex(daily_close.index, method="ffill")


def weekly_rsi_declining_2_weeks(daily_close: pd.Series, period: int = 14) -> pd.Series:
    """True when weekly RSI has been declining for 2+ consecutive weeks."""
    weekly = daily_close.resample("W-FRI").last().dropna()
    wrsi = compute_rsi(weekly, period)
    declining = (wrsi < wrsi.shift(1)) & (wrsi.shift(1) < wrsi.shift(2))
    declining_daily = declining.reindex(daily_close.index, method="ffill").fillna(False)
    return declining_daily


def consecutive_red_days(daily_close: pd.Series, daily_open: pd.Series) -> pd.Series:
    """Count consecutive red days ending yesterday."""
    is_red = daily_close < daily_open
    # Count consecutive reds backward from yesterday
    counts = pd.Series(0, index=daily_close.index, dtype=int)
    for i in range(1, len(counts)):
        if is_red.iloc[i - 1]:
            counts.iloc[i] = counts.iloc[i - 1] + 1
        else:
            counts.iloc[i] = 0
    return counts


def is_green_today(daily_close: pd.Series, daily_open: pd.Series) -> pd.Series:
    return daily_close >= daily_open


# ── Data Download ────────────────────────────────────────────────────────────

print("Downloading data...")
data = {}
for ticker in UNIVERSE:
    try:
        df = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=True)
        if df is not None and len(df) > 100:
            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            data[ticker] = df
            print(f"  {ticker}: {len(df)} rows")
        else:
            print(f"  {ticker}: insufficient data, skipping")
    except Exception as e:
        print(f"  {ticker}: download failed: {e}")

print(f"\nLoaded {len(data)} tickers")

# ── Precompute Signals ───────────────────────────────────────────────────────

print("\nComputing signals...")

signals = {}  # ticker -> DataFrame with signal columns

for ticker, df in data.items():
    s = pd.DataFrame(index=df.index)
    s["close"] = df["Close"]
    s["open"] = df["Open"]
    s["high"] = df["High"]
    s["low"] = df["Low"]

    # Core indicators
    s["rsi14"] = compute_rsi(s["close"], RSI_PERIOD)
    s["high_20d"] = s["close"].rolling(HIGH_LOOKBACK).max()
    s["pct_below_high"] = (s["high_20d"] - s["close"]) / s["high_20d"]

    # Conditions
    s["dip_5pct"] = s["pct_below_high"] > DIP_THRESHOLD
    s["rsi_below_35"] = s["rsi14"] < RSI_THRESHOLD
    s["green_today"] = is_green_today(s["close"], s["open"])
    s["consec_red"] = consecutive_red_days(s["close"], s["open"])
    s["recovery"] = s["green_today"] & (s["consec_red"] >= 2)
    s["weekly_rsi_declining"] = weekly_rsi_declining_2_weeks(s["close"], RSI_PERIOD)

    # Strategy signals
    # Quality MR-A: dip + RSI<35
    s["sig_mra"] = s["dip_5pct"] & s["rsi_below_35"]
    # Dual Signal D: MR-A + recovery (green after 2+ red)
    s["sig_dsd"] = s["sig_mra"] & s["recovery"]
    # Multi-TF Dual Signal L: DSD + weekly RSI declining
    s["sig_mtfl"] = s["sig_dsd"] & s["weekly_rsi_declining"]

    # Forward return for exit (10-day hold)
    s["fwd_ret_10d"] = s["close"].shift(-HOLD_DAYS) / s["close"] - 1

    signals[ticker] = s

# Filter to trade period
trade_start_dt = pd.Timestamp(TRADE_START)
for ticker in signals:
    signals[ticker] = signals[ticker].loc[trade_start_dt:]


# ── Backtest Engine ──────────────────────────────────────────────────────────

def run_backtest(variant: str, signals: dict, capital: float = CAPITAL) -> dict:
    """
    Run a specific variant backtest.
    Returns dict with equity curve, trades, and stats.
    """
    all_dates = sorted(set().union(*(s.index for s in signals.values())))
    all_dates = [d for d in all_dates if d >= trade_start_dt]

    equity = capital
    equity_curve = []
    trades = []
    positions = []  # list of {ticker, entry_date, entry_price, size_dollars, exit_date, exit_price, hold_days}

    for date in all_dates:
        # Close expired positions
        new_positions = []
        for pos in positions:
            if variant == "F":
                target_hold = 5
            else:
                target_hold = HOLD_DAYS

            days_held = len([d for d in all_dates if pos["entry_date"] < d <= date])

            if days_held >= target_hold:
                # Exit
                ticker_sig = signals.get(pos["ticker"])
                if ticker_sig is not None and date in ticker_sig.index:
                    exit_price = ticker_sig.loc[date, "close"]
                else:
                    # Use last known price
                    exit_price = pos["entry_price"]

                slippage = exit_price * SLIPPAGE_BPS / 10000
                exit_price_adj = exit_price - slippage  # selling
                shares = pos["shares"]
                pnl = (exit_price_adj - pos["entry_price_adj"]) * shares
                equity += pnl + pos["size_dollars"]  # return capital + pnl

                trades.append({
                    "ticker": pos["ticker"],
                    "entry_date": str(pos["entry_date"].date()),
                    "exit_date": str(date.date()),
                    "entry_price": round(pos["entry_price"], 2),
                    "exit_price": round(exit_price, 2),
                    "shares": round(shares, 4),
                    "pnl": round(pnl, 2),
                    "ret": round(pnl / pos["size_dollars"], 4),
                    "hold_days": days_held,
                })
            else:
                new_positions.append(pos)
        positions = new_positions

        # Determine max concurrent and current open
        if variant == "C":
            max_concurrent = 2
        else:
            max_concurrent = 3

        if len(positions) >= max_concurrent:
            equity_curve.append({"date": str(date.date()), "equity": round(equity, 2)})
            continue

        # Collect candidate signals for today
        candidates = []
        for ticker, s in signals.items():
            if date not in s.index:
                continue
            row = s.loc[date]
            if pd.isna(row.get("close")) or pd.isna(row.get("rsi14")):
                continue

            # Skip if already in position
            if any(p["ticker"] == ticker for p in positions):
                continue

            sig_mtfl = bool(row.get("sig_mtfl", False))
            sig_dsd = bool(row.get("sig_dsd", False))
            sig_mra = bool(row.get("sig_mra", False))

            if variant == "F":
                # Weekly rotation: Monday only, any signal
                if date.weekday() != 0:
                    continue
                if not (sig_mtfl or sig_dsd or sig_mra):
                    continue
                # Score by depth of dip
                pct_below = float(row.get("pct_below_high", 0))
                n_signals = int(sig_mtfl) + int(sig_dsd) + int(sig_mra)
                candidates.append({
                    "ticker": ticker,
                    "priority": -pct_below,  # lower = deeper dip = better
                    "size": 200.0,
                    "n_signals": n_signals,
                    "close": float(row["close"]),
                })
            elif variant == "A":
                # Priority allocation
                if sig_mtfl:
                    candidates.append({"ticker": ticker, "priority": 0, "size": 200.0, "close": float(row["close"])})
                elif sig_dsd:
                    candidates.append({"ticker": ticker, "priority": 1, "size": 150.0, "close": float(row["close"])})
                elif sig_mra:
                    candidates.append({"ticker": ticker, "priority": 2, "size": 100.0, "close": float(row["close"])})
            elif variant == "B":
                # Equal allocation, priority ordering
                if sig_mtfl:
                    candidates.append({"ticker": ticker, "priority": 0, "size": 200.0, "close": float(row["close"])})
                elif sig_dsd:
                    candidates.append({"ticker": ticker, "priority": 1, "size": 200.0, "close": float(row["close"])})
                elif sig_mra:
                    candidates.append({"ticker": ticker, "priority": 2, "size": 200.0, "close": float(row["close"])})
            elif variant == "C":
                # Concentration: only Multi-TF L
                if sig_mtfl:
                    candidates.append({"ticker": ticker, "priority": 0, "size": 300.0, "close": float(row["close"])})
            elif variant == "D":
                # Diversified: only Quality MR-A (broadest)
                if sig_mra:
                    candidates.append({"ticker": ticker, "priority": 0, "size": 200.0, "close": float(row["close"])})
            elif variant == "E":
                # Tiered by conviction
                n_agree = int(sig_mtfl) + int(sig_dsd) + int(sig_mra)
                if n_agree == 0:
                    continue
                if n_agree == 3:
                    size = 250.0
                elif n_agree == 2:
                    size = 200.0
                else:
                    size = 150.0
                candidates.append({
                    "ticker": ticker,
                    "priority": -n_agree,  # more agreement = higher priority
                    "size": size,
                    "close": float(row["close"]),
                })

        # Sort by priority (lower = better)
        candidates.sort(key=lambda x: x["priority"])

        # Enter positions
        for cand in candidates:
            if len(positions) >= max_concurrent:
                break
            size = min(cand["size"], equity * 0.95)  # don't use more than 95% of remaining equity
            if size < 10:
                continue

            entry_price = cand["close"]
            slippage = entry_price * SLIPPAGE_BPS / 10000
            entry_price_adj = entry_price + slippage  # buying
            shares = size / entry_price_adj

            equity -= size  # allocate capital

            positions.append({
                "ticker": cand["ticker"],
                "entry_date": date,
                "entry_price": entry_price,
                "entry_price_adj": entry_price_adj,
                "shares": shares,
                "size_dollars": size,
            })

        equity_curve.append({"date": str(date.date()), "equity": round(equity, 2)})

    # Close any remaining positions at last known price
    for pos in positions:
        last_date = all_dates[-1]
        ticker_sig = signals.get(pos["ticker"])
        if ticker_sig is not None and last_date in ticker_sig.index:
            exit_price = float(ticker_sig.loc[last_date, "close"])
        else:
            exit_price = pos["entry_price"]
        slippage = exit_price * SLIPPAGE_BPS / 10000
        exit_price_adj = exit_price - slippage
        shares = pos["shares"]
        pnl = (exit_price_adj - pos["entry_price_adj"]) * shares
        equity += pnl + pos["size_dollars"]
        days_held = len([d for d in all_dates if pos["entry_date"] < d <= last_date])
        trades.append({
            "ticker": pos["ticker"],
            "entry_date": str(pos["entry_date"].date()),
            "exit_date": str(last_date.date()),
            "entry_price": round(pos["entry_price"], 2),
            "exit_price": round(exit_price, 2),
            "shares": round(shares, 4),
            "pnl": round(pnl, 2),
            "ret": round(pnl / pos["size_dollars"], 4),
            "hold_days": days_held,
        })

    # Build daily equity series including position mark-to-market
    # Rebuild properly with MTM
    eq_series = _build_mtm_equity(variant, signals, capital, all_dates)

    return {"trades": trades, "equity_curve": eq_series, "final_equity": round(equity, 2)}


def _build_mtm_equity(variant: str, signals: dict, capital: float, all_dates: list) -> list:
    """Rebuild equity curve with daily mark-to-market of open positions."""
    equity_cash = capital
    positions = []
    curve = []

    for date in all_dates:
        # Close expired positions
        new_positions = []
        for pos in positions:
            target_hold = 5 if variant == "F" else HOLD_DAYS
            days_held = len([d for d in all_dates if pos["entry_date"] < d <= date])

            if days_held >= target_hold:
                ticker_sig = signals.get(pos["ticker"])
                if ticker_sig is not None and date in ticker_sig.index:
                    exit_price = float(ticker_sig.loc[date, "close"])
                else:
                    exit_price = pos["entry_price"]
                slippage = exit_price * SLIPPAGE_BPS / 10000
                exit_price_adj = exit_price - slippage
                pnl = (exit_price_adj - pos["entry_price_adj"]) * pos["shares"]
                equity_cash += pnl + pos["size_dollars"]
            else:
                new_positions.append(pos)
        positions = new_positions

        max_concurrent = 2 if variant == "C" else 3

        # Open new positions (same logic as run_backtest)
        if len(positions) < max_concurrent:
            candidates = []
            for ticker, s in signals.items():
                if date not in s.index:
                    continue
                row = s.loc[date]
                if pd.isna(row.get("close")) or pd.isna(row.get("rsi14")):
                    continue
                if any(p["ticker"] == ticker for p in positions):
                    continue

                sig_mtfl = bool(row.get("sig_mtfl", False))
                sig_dsd = bool(row.get("sig_dsd", False))
                sig_mra = bool(row.get("sig_mra", False))

                if variant == "F":
                    if date.weekday() != 0:
                        continue
                    if not (sig_mtfl or sig_dsd or sig_mra):
                        continue
                    pct_below = float(row.get("pct_below_high", 0))
                    candidates.append({"ticker": ticker, "priority": -pct_below, "size": 200.0, "close": float(row["close"])})
                elif variant == "A":
                    if sig_mtfl:
                        candidates.append({"ticker": ticker, "priority": 0, "size": 200.0, "close": float(row["close"])})
                    elif sig_dsd:
                        candidates.append({"ticker": ticker, "priority": 1, "size": 150.0, "close": float(row["close"])})
                    elif sig_mra:
                        candidates.append({"ticker": ticker, "priority": 2, "size": 100.0, "close": float(row["close"])})
                elif variant == "B":
                    if sig_mtfl:
                        candidates.append({"ticker": ticker, "priority": 0, "size": 200.0, "close": float(row["close"])})
                    elif sig_dsd:
                        candidates.append({"ticker": ticker, "priority": 1, "size": 200.0, "close": float(row["close"])})
                    elif sig_mra:
                        candidates.append({"ticker": ticker, "priority": 2, "size": 200.0, "close": float(row["close"])})
                elif variant == "C":
                    if sig_mtfl:
                        candidates.append({"ticker": ticker, "priority": 0, "size": 300.0, "close": float(row["close"])})
                elif variant == "D":
                    if sig_mra:
                        candidates.append({"ticker": ticker, "priority": 0, "size": 200.0, "close": float(row["close"])})
                elif variant == "E":
                    n_agree = int(sig_mtfl) + int(sig_dsd) + int(sig_mra)
                    if n_agree == 0:
                        continue
                    size = {3: 250.0, 2: 200.0, 1: 150.0}[n_agree]
                    candidates.append({"ticker": ticker, "priority": -n_agree, "size": size, "close": float(row["close"])})

            candidates.sort(key=lambda x: x["priority"])
            for cand in candidates:
                if len(positions) >= max_concurrent:
                    break
                size = min(cand["size"], equity_cash * 0.95)
                if size < 10:
                    continue
                entry_price = cand["close"]
                slippage = entry_price * SLIPPAGE_BPS / 10000
                entry_price_adj = entry_price + slippage
                shares = size / entry_price_adj
                equity_cash -= size
                positions.append({
                    "ticker": cand["ticker"],
                    "entry_date": date,
                    "entry_price": entry_price,
                    "entry_price_adj": entry_price_adj,
                    "shares": shares,
                    "size_dollars": size,
                })

        # Mark-to-market
        mtm = 0.0
        for pos in positions:
            ticker_sig = signals.get(pos["ticker"])
            if ticker_sig is not None and date in ticker_sig.index:
                current_price = float(ticker_sig.loc[date, "close"])
            else:
                current_price = pos["entry_price"]
            mtm += (current_price - pos["entry_price_adj"]) * pos["shares"]

        total_invested = sum(p["size_dollars"] for p in positions)
        total_equity = equity_cash + total_invested + mtm
        curve.append({"date": str(date.date()), "equity": round(total_equity, 2)})

    return curve


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(result: dict, capital: float = CAPITAL) -> dict:
    trades = result["trades"]
    curve = result["equity_curve"]

    if not trades:
        return {"n_trades": 0, "sharpe": 0, "sortino": 0, "pf": 0, "wr": 0, "max_dd": 0, "total_ret": 0}

    rets = [t["ret"] for t in trades]
    pnls = [t["pnl"] for t in trades]
    n = len(trades)
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]

    avg_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1) if n > 1 else 1e-9
    downside = np.std([r for r in rets if r < 0], ddof=1) if any(r < 0 for r in rets) else 1e-9

    # Annualize: assume ~25 trades/year as rough basis, use daily equity for Sharpe
    eq_values = [c["equity"] for c in curve]
    eq_series = pd.Series(eq_values)
    daily_rets = eq_series.pct_change().dropna()

    if len(daily_rets) > 1 and daily_rets.std() > 0:
        sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252)
    else:
        sharpe = 0.0

    neg_daily = daily_rets[daily_rets < 0]
    if len(neg_daily) > 0 and neg_daily.std() > 0:
        sortino = daily_rets.mean() / neg_daily.std() * np.sqrt(252)
    else:
        sortino = 0.0

    gross_win = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    wr = len(wins) / n

    # Max drawdown from equity curve
    eq_arr = np.array(eq_values)
    running_max = np.maximum.accumulate(eq_arr)
    drawdowns = (eq_arr - running_max) / running_max
    max_dd = float(np.min(drawdowns))

    total_ret = (eq_values[-1] - capital) / capital

    # CAGR
    dates = [c["date"] for c in curve]
    years = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days / 365.25
    if years > 0 and eq_values[-1] > 0:
        cagr = (eq_values[-1] / capital) ** (1 / years) - 1
    else:
        cagr = 0.0

    # Average trade stats
    avg_pnl = np.mean(pnls)
    avg_win = np.mean([p for p in pnls if p > 0]) if wins else 0
    avg_loss = np.mean([p for p in pnls if p <= 0]) if losses else 0

    return {
        "n_trades": n,
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(pf, 3),
        "win_rate": round(wr, 3),
        "max_drawdown": round(max_dd, 4),
        "total_return": round(total_ret, 4),
        "cagr": round(cagr, 4),
        "final_equity": round(eq_values[-1], 2),
        "avg_pnl_per_trade": round(avg_pnl, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "total_pnl": round(sum(pnls), 2),
    }


# ── Regime Analysis ─────────────────────────────────────────────────────────

def regime_analysis(result: dict, spy_data: pd.DataFrame) -> dict:
    """Split trades by SPY regime (green/red months)."""
    trades = result["trades"]
    if not trades:
        return {"gap": 1.0, "bull_sharpe": 0, "bear_sharpe": 0}

    # Monthly SPY returns
    spy_monthly = spy_data["Close"].resample("ME").last().pct_change()

    bull_rets = []
    bear_rets = []

    for t in trades:
        entry = pd.Timestamp(t["entry_date"])
        month_key = entry.to_period("M").to_timestamp()
        # Find closest month
        closest = spy_monthly.index[spy_monthly.index <= entry]
        if len(closest) == 0:
            bull_rets.append(t["ret"])
            continue
        monthly_ret = spy_monthly.loc[closest[-1]] if closest[-1] in spy_monthly.index else 0

        if monthly_ret >= 0:
            bull_rets.append(t["ret"])
        else:
            bear_rets.append(t["ret"])

    bull_sharpe = np.mean(bull_rets) / np.std(bull_rets, ddof=1) * np.sqrt(252/10) if len(bull_rets) > 1 and np.std(bull_rets) > 0 else 0
    bear_sharpe = np.mean(bear_rets) / np.std(bear_rets, ddof=1) * np.sqrt(252/10) if len(bear_rets) > 1 and np.std(bear_rets) > 0 else 0

    if max(abs(bull_sharpe), abs(bear_sharpe)) > 0:
        gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe))
    else:
        gap = 1.0

    return {
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "regime_gap": round(gap, 3),
        "n_bull_trades": len(bull_rets),
        "n_bear_trades": len(bear_rets),
    }


# ── Permutation Test ─────────────────────────────────────────────────────────

def permutation_test(trades: list, n_perms: int = 1000) -> float:
    """Shuffle trade returns, compute p-value of observed mean."""
    if not trades:
        return 1.0
    rets = np.array([t["ret"] for t in trades])
    observed = np.mean(rets)
    count = 0
    for _ in range(n_perms):
        shuffled = rets.copy()
        np.random.shuffle(shuffled)
        # Randomly flip signs
        signs = np.random.choice([-1, 1], size=len(shuffled))
        if np.mean(shuffled * signs) >= observed:
            count += 1
    return count / n_perms


# ── 5-Gate Validation ────────────────────────────────────────────────────────

def validate_5gate(metrics: dict, regime: dict, perm_p: float) -> dict:
    gates = {
        "G1_sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "G2_perm_p_lt_0.05": perm_p < 0.05,
        "G3_regime_gap_lt_0.5": regime["regime_gap"] < 0.5,
        "G4_max_dd_gt_neg50pct": metrics["max_drawdown"] > -0.50,
        "G5_min_20_trades": metrics["n_trades"] >= 20,
    }
    gates["all_passed"] = all(gates.values())
    gates["gates_passed"] = sum(v for k, v in gates.items() if k.startswith("G"))
    gates["perm_p_value"] = round(perm_p, 4)
    return gates


# ── Main ─────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("MULTI-STRATEGY PORTFOLIO BACKTEST v2")
print("=" * 70)

# Download SPY for regime analysis
print("\nDownloading SPY for regime analysis...")
spy = yf.download("SPY", start=START, end=END, progress=False, auto_adjust=True)
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)

variants = {
    "A": "Priority Allocation (L=$200, D=$150, MRA=$100, max 3)",
    "B": "Equal Allocation ($200 each, priority L>D>MRA, max 3)",
    "C": "Concentration (L only, $300, max 2)",
    "D": "Diversified (MRA only, $200, max 3)",
    "E": "Tiered by Conviction (3-agree=$250, 2=$200, 1=$150, max 3)",
    "F": "Weekly Rotation (Monday only, deepest dip, 5-day hold, $200, max 3)",
}

results_all = {}
np.random.seed(42)

for var_key, var_desc in variants.items():
    print(f"\n{'─' * 60}")
    print(f"Variant {var_key}: {var_desc}")
    print(f"{'─' * 60}")

    result = run_backtest(var_key, signals, CAPITAL)
    metrics = compute_metrics(result, CAPITAL)
    regime = regime_analysis(result, spy)
    perm_p = permutation_test(result["trades"], n_perms=1000)
    gates = validate_5gate(metrics, regime, perm_p)

    print(f"  Trades: {metrics['n_trades']}")
    print(f"  Sharpe: {metrics['sharpe']:.3f}  |  Sortino: {metrics['sortino']:.3f}")
    print(f"  PF: {metrics['profit_factor']:.2f}  |  WR: {metrics['win_rate']:.1%}")
    print(f"  Max DD: {metrics['max_drawdown']:.2%}")
    print(f"  Total Return: {metrics['total_return']:.2%}  |  CAGR: {metrics['cagr']:.2%}")
    print(f"  Final Equity: ${metrics['final_equity']:.2f} (from ${CAPITAL})")
    print(f"  Avg PnL/trade: ${metrics['avg_pnl_per_trade']:.2f}")
    print(f"  Regime: bull={regime['bull_sharpe']:.2f}, bear={regime['bear_sharpe']:.2f}, gap={regime['regime_gap']:.3f}")
    print(f"  Perm p-value: {perm_p:.4f}")
    print(f"  Gates: {gates['gates_passed']}/5 passed | {'PASS' if gates['all_passed'] else 'FAIL'}")
    for g, v in gates.items():
        if g.startswith("G"):
            status = "✓" if v else "✗"
            print(f"    {status} {g}: {v}")

    results_all[var_key] = {
        "description": var_desc,
        "metrics": metrics,
        "regime": regime,
        "validation": gates,
        "sample_trades": result["trades"][:10],  # first 10 trades for inspection
        "equity_curve_summary": {
            "start": result["equity_curve"][0] if result["equity_curve"] else None,
            "end": result["equity_curve"][-1] if result["equity_curve"] else None,
            "length": len(result["equity_curve"]),
        },
    }

# ── Summary Ranking ──────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("RANKING SUMMARY")
print("=" * 70)
print(f"{'Var':>4} {'Sharpe':>7} {'Sort':>7} {'PF':>6} {'WR':>6} {'MaxDD':>8} {'CAGR':>7} {'#Tr':>5} {'Gap':>6} {'Gates':>6}")
print("-" * 70)

ranked = sorted(results_all.items(), key=lambda x: x[1]["metrics"]["sharpe"], reverse=True)
for var_key, res in ranked:
    m = res["metrics"]
    r = res["regime"]
    g = res["validation"]
    status = "PASS" if g["all_passed"] else "FAIL"
    print(f"  {var_key:>2} {m['sharpe']:>7.3f} {m['sortino']:>7.3f} {m['profit_factor']:>6.2f} {m['win_rate']:>5.1%} {m['max_drawdown']:>7.2%} {m['cagr']:>6.2%} {m['n_trades']:>5} {r['regime_gap']:>6.3f} {status:>6}")

# Best variant
best_key = ranked[0][0]
best = ranked[0][1]
print(f"\nBEST: Variant {best_key} — {best['description']}")
print(f"  Sharpe {best['metrics']['sharpe']:.3f}, "
      f"Sortino {best['metrics']['sortino']:.3f}, "
      f"CAGR {best['metrics']['cagr']:.2%}, "
      f"PF {best['metrics']['profit_factor']:.2f}, "
      f"WR {best['metrics']['win_rate']:.1%}")

# ── Save Results ─────────────────────────────────────────────────────────────

output = {
    "backtest": "Multi-Strategy Portfolio Allocation v2",
    "universe": UNIVERSE,
    "period": f"{TRADE_START} to {END}",
    "capital": CAPITAL,
    "slippage_bps": SLIPPAGE_BPS,
    "generated": datetime.now().isoformat(),
    "strategies": {
        "Multi-TF Dual Signal L": ">5% below 20d high + RSI<35 + green after 2+ red + weekly RSI declining 2+ weeks",
        "Dual Signal D": ">5% below 20d high + RSI<35 + green after 2+ red",
        "Quality MR-A": ">5% below 20d high + RSI<35",
    },
    "variants": results_all,
    "ranking": [{"variant": k, "sharpe": v["metrics"]["sharpe"], "passed": v["validation"]["all_passed"]} for k, v in ranked],
    "best_variant": best_key,
}

output_path = Path("/home/jupiter/Lvl3Quant/data/multi_strategy_portfolio_v2_results.json")
with open(output_path, "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to {output_path}")
print("Done.")
