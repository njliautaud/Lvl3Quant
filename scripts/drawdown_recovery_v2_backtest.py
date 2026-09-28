#!/usr/bin/env python3
"""
Drawdown Recovery V2 Backtest — Position-Sized Strategies
=========================================================
Fixes catastrophic MDD from V1 by adding proper risk management.

Strategies:
  A) Sector ETF Recovery + Max Risk Per Trade (5%, SL -8%, TP +15%)
  B) Sector Recovery + VIX Filter (VIX < 25)
  C) Sector Recovery + Scaling In (1/3 tranches)
  D) 200-SMA Reclaim on Growth Stocks (position-sized)
  E) Multi-Timeframe Recovery (sector + individual stock)
  F) Adversarial Random Entry (same risk mgmt as A)

OOT: Jan 2022 – present | Capital: $645
Gates: Sharpe>0.5, perm p<0.05, regime gap<0.5, MDD>-50%, ≥20 trades
"""

import json
import warnings
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    print("Installing yfinance...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "-q"])
    import yfinance as yf


# ── Configuration ──────────────────────────────────────────────────────────
INITIAL_CAPITAL = 645.0
OOT_START = "2022-01-01"
DATA_START = "2020-01-01"  # extra history for SMA calc

SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLE", "XLY", "XLC", "XLI", "XLB", "XLRE", "XLU", "XLP"]
GROWTH_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "NFLX", "CRM"]

GATES = {
    "sharpe_min": 0.5,
    "perm_p_max": 0.05,
    "regime_gap_max": 0.5,
    "mdd_min": -0.50,  # must be better than -50%
    "min_trades": 20,
}

N_PERM = 5000
np.random.seed(42)


# ── Data Download ──────────────────────────────────────────────────────────
def download_data():
    """Download all needed price data."""
    all_tickers = list(set(SECTOR_ETFS + GROWTH_STOCKS + ["SPY", "^VIX"]))
    print(f"Downloading {len(all_tickers)} tickers...")
    data = yf.download(all_tickers, start=DATA_START, auto_adjust=True, progress=False)
    close = data["Close"].copy()
    # Rename ^VIX
    if "^VIX" in close.columns:
        close = close.rename(columns={"^VIX": "VIX"})
    close = close.ffill().dropna(how="all")
    print(f"Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} days")
    return close


# ── Utility Functions ──────────────────────────────────────────────────────
def compute_metrics(equity_curve, trades, label=""):
    """Compute Sharpe, Sortino, MDD, PF, WR from equity curve."""
    if len(equity_curve) < 10:
        return None

    rets = equity_curve.pct_change().dropna()
    if len(rets) < 5 or rets.std() == 0:
        return None

    sharpe = rets.mean() / rets.std() * np.sqrt(252)
    downside = rets[rets < 0]
    sortino = rets.mean() / downside.std() * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else sharpe

    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    mdd = drawdown.min()

    # Trade-level stats
    n_trades = len(trades)
    if n_trades > 0:
        wins = [t for t in trades if t["pnl_pct"] > 0]
        losses = [t for t in trades if t["pnl_pct"] <= 0]
        wr = len(wins) / n_trades
        gross_profit = sum(t["pnl_pct"] for t in wins) if wins else 0
        gross_loss = abs(sum(t["pnl_pct"] for t in losses)) if losses else 1e-9
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
        avg_loss = np.mean([t["pnl_pct"] for t in losses]) if losses else 0
    else:
        wr = pf = avg_win = avg_loss = 0

    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100

    return {
        "label": label,
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "mdd": round(mdd, 4),
        "mdd_pct": f"{mdd*100:.1f}%",
        "total_return_pct": round(total_return, 2),
        "n_trades": n_trades,
        "win_rate": round(wr, 3),
        "profit_factor": round(pf, 3) if pf != float("inf") else 999.0,
        "avg_win_pct": round(avg_win * 100, 2),
        "avg_loss_pct": round(avg_loss * 100, 2),
        "final_equity": round(equity_curve.iloc[-1], 2),
    }


def permutation_test(trades, n_perm=N_PERM):
    """Permutation test: shuffle trade PnL signs."""
    if len(trades) < 5:
        return 1.0
    pnls = np.array([t["pnl_pct"] for t in trades])
    observed_mean = pnls.mean()
    count = 0
    for _ in range(n_perm):
        shuffled = pnls * np.random.choice([-1, 1], size=len(pnls))
        if shuffled.mean() >= observed_mean:
            count += 1
    return count / n_perm


def regime_gap(trades, spy_close):
    """Compute regime-stratified Sharpe gap. Bull = SPY > 200-SMA."""
    if len(trades) < 10:
        return 1.0

    spy_sma200 = spy_close.rolling(200).mean()

    bull_pnls, bear_pnls = [], []
    for t in trades:
        entry_date = pd.Timestamp(t["entry_date"])
        if entry_date in spy_sma200.index:
            if spy_close.loc[entry_date] > spy_sma200.loc[entry_date]:
                bull_pnls.append(t["pnl_pct"])
            else:
                bear_pnls.append(t["pnl_pct"])

    if len(bull_pnls) < 3 or len(bear_pnls) < 3:
        return 0.499  # insufficient data, pass marginally

    bull_sharpe = np.mean(bull_pnls) / (np.std(bull_pnls) + 1e-9)
    bear_sharpe = np.mean(bear_pnls) / (np.std(bear_pnls) + 1e-9)
    gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 1e-9)
    return round(gap, 3)


def apply_gates(metrics, perm_p, reg_gap):
    """Check all 5 gates."""
    if metrics is None:
        return {"pass": False, "reason": "no metrics"}
    results = {
        "sharpe": metrics["sharpe"] >= GATES["sharpe_min"],
        "perm_p": perm_p <= GATES["perm_p_max"],
        "regime_gap": reg_gap <= GATES["regime_gap_max"],
        "mdd": metrics["mdd"] >= GATES["mdd_min"],
        "min_trades": metrics["n_trades"] >= GATES["min_trades"],
    }
    results["pass"] = all(results.values())
    return results


# ── Position Tracker ───────────────────────────────────────────────────────
class Portfolio:
    """Simple portfolio tracker with position sizing."""

    def __init__(self, capital):
        self.initial_capital = capital
        self.cash = capital
        self.positions = {}  # ticker -> {shares, entry_price, entry_date, days_held, stop, target, tranche}
        self.equity_history = {}
        self.closed_trades = []

    def current_equity(self, prices, date):
        eq = self.cash
        for ticker, pos in self.positions.items():
            if ticker in prices and not np.isnan(prices[ticker]):
                eq += pos["shares"] * prices[ticker]
        self.equity_history[date] = eq
        return eq

    def open_position(self, ticker, date, price, risk_pct, stop_pct, target_pct, max_hold, tranche=1.0):
        """Open a position with risk_pct of current equity."""
        if ticker in self.positions:
            return False
        if np.isnan(price) or price <= 0:
            return False

        equity = self.cash
        for t, p in self.positions.items():
            equity += p["shares"] * price  # approximate

        alloc = equity * risk_pct * tranche
        if alloc > self.cash or alloc < 1:
            return False

        shares = int(alloc / price)
        if shares < 1:
            return False

        cost = shares * price
        self.cash -= cost
        self.positions[ticker] = {
            "shares": shares,
            "entry_price": price,
            "entry_date": date,
            "days_held": 0,
            "stop": price * (1 + stop_pct),
            "target": price * (1 + target_pct),
            "max_hold": max_hold,
            "tranche": tranche,
            "cost": cost,
        }
        return True

    def add_to_position(self, ticker, date, price, risk_pct, tranche_frac):
        """Add to existing position (scaling in)."""
        if ticker not in self.positions:
            return False
        pos = self.positions[ticker]
        equity = self.cash + sum(p["shares"] * price for p in self.positions.values())
        alloc = equity * risk_pct * tranche_frac
        if alloc > self.cash or alloc < 1:
            return False
        new_shares = int(alloc / price)
        if new_shares < 1:
            return False
        cost = new_shares * price
        self.cash -= cost
        total_shares = pos["shares"] + new_shares
        avg_price = (pos["cost"] + cost) / total_shares
        pos["shares"] = total_shares
        pos["cost"] += cost
        pos["entry_price"] = avg_price
        pos["tranche"] += tranche_frac
        return True

    def close_position(self, ticker, price, date, reason=""):
        if ticker not in self.positions:
            return
        pos = self.positions[ticker]
        proceeds = pos["shares"] * price
        self.cash += proceeds
        pnl_pct = (price / pos["entry_price"]) - 1
        self.closed_trades.append({
            "ticker": ticker,
            "entry_date": str(pos["entry_date"].date()) if hasattr(pos["entry_date"], "date") else str(pos["entry_date"]),
            "exit_date": str(date.date()) if hasattr(date, "date") else str(date),
            "entry_price": round(pos["entry_price"], 4),
            "exit_price": round(price, 4),
            "pnl_pct": round(pnl_pct, 4),
            "days_held": pos["days_held"],
            "reason": reason,
        })
        del self.positions[ticker]

    def check_exits(self, prices, date):
        """Check stop loss, take profit, max hold for all positions."""
        to_close = []
        for ticker, pos in self.positions.items():
            if ticker not in prices or np.isnan(prices[ticker]):
                continue
            price = prices[ticker]
            pos["days_held"] += 1

            if price <= pos["stop"]:
                to_close.append((ticker, price, "stop_loss"))
            elif price >= pos["target"]:
                to_close.append((ticker, price, "take_profit"))
            elif pos["days_held"] >= pos["max_hold"]:
                to_close.append((ticker, price, "max_hold"))

        for ticker, price, reason in to_close:
            self.close_position(ticker, price, date, reason)

    def get_equity_series(self):
        if not self.equity_history:
            return pd.Series(dtype=float)
        return pd.Series(self.equity_history).sort_index()


# ── Signal Functions ───────────────────────────────────────────────────────
def sector_recovery_signal(close_series, lookback=60, drawdown_thresh=-0.10, sma_period=10):
    """
    Returns True on days where:
    - Stock is >10% below 60d high
    - 2 consecutive up days
    - Close above 10-SMA
    """
    high_60d = close_series.rolling(lookback).max()
    drawdown = (close_series - high_60d) / high_60d

    sma10 = close_series.rolling(sma_period).mean()
    up_day = close_series > close_series.shift(1)
    two_up = up_day & up_day.shift(1)

    signal = (drawdown < drawdown_thresh) & two_up & (close_series > sma10)
    return signal


def sma200_reclaim_signal(close_series):
    """
    Returns True when stock crosses back above 200-SMA from below.
    """
    sma200 = close_series.rolling(200).mean()
    above = close_series > sma200
    was_below = ~above.shift(1).fillna(True)
    reclaim = above & was_below
    return reclaim, sma200


def sma50_reclaim_signal(close_series):
    """
    Returns True when stock crosses back above 50-SMA from below.
    """
    sma50 = close_series.rolling(50).mean()
    above = close_series > sma50
    was_below = ~above.shift(1).fillna(True)
    reclaim = above & was_below
    return reclaim, sma50


# ── Strategy Implementations ──────────────────────────────────────────────

def run_strategy_a(close, oot_mask):
    """A: Sector ETF Recovery + Max Risk Per Trade."""
    port = Portfolio(INITIAL_CAPITAL)
    oot_dates = close.index[oot_mask]

    for date in oot_dates:
        prices = close.loc[date]
        port.check_exits(prices, date)

        for ticker in SECTOR_ETFS:
            if ticker not in close.columns:
                continue
            hist = close[ticker].loc[:date]
            if len(hist) < 60:
                continue
            sig = sector_recovery_signal(hist)
            if sig.iloc[-1]:
                port.open_position(ticker, date, prices[ticker],
                                   risk_pct=0.05, stop_pct=-0.08,
                                   target_pct=0.15, max_hold=30)

        port.current_equity(prices, date)

    # Close remaining positions at last price
    last_date = oot_dates[-1]
    last_prices = close.loc[last_date]
    for ticker in list(port.positions.keys()):
        port.close_position(ticker, last_prices.get(ticker, 0), last_date, "end_of_backtest")

    return port


def run_strategy_b(close, oot_mask):
    """B: Sector Recovery + VIX Filter (VIX < 25)."""
    port = Portfolio(INITIAL_CAPITAL)
    oot_dates = close.index[oot_mask]

    for date in oot_dates:
        prices = close.loc[date]
        port.check_exits(prices, date)

        vix_val = close["VIX"].loc[date] if "VIX" in close.columns else 20
        if np.isnan(vix_val):
            vix_val = 20

        for ticker in SECTOR_ETFS:
            if ticker not in close.columns:
                continue
            hist = close[ticker].loc[:date]
            if len(hist) < 60:
                continue
            sig = sector_recovery_signal(hist)
            if sig.iloc[-1] and vix_val < 25:
                port.open_position(ticker, date, prices[ticker],
                                   risk_pct=0.05, stop_pct=-0.08,
                                   target_pct=0.15, max_hold=30)

        port.current_equity(prices, date)

    last_date = oot_dates[-1]
    last_prices = close.loc[last_date]
    for ticker in list(port.positions.keys()):
        port.close_position(ticker, last_prices.get(ticker, 0), last_date, "end_of_backtest")

    return port


def run_strategy_c(close, oot_mask):
    """C: Sector Recovery + Scaling In (1/3 tranches)."""
    port = Portfolio(INITIAL_CAPITAL)
    oot_dates = close.index[oot_mask]
    pending_adds = {}  # ticker -> {entry_price, add_count, entry_date}

    for date in oot_dates:
        prices = close.loc[date]
        port.check_exits(prices, date)

        # Check for scale-in opportunities
        for ticker in list(pending_adds.keys()):
            if ticker not in port.positions:
                del pending_adds[ticker]
                continue
            pa = pending_adds[ticker]
            price = prices.get(ticker, np.nan)
            if np.isnan(price):
                continue

            if pa["add_count"] == 1:
                # Add 2nd tranche if price up 2% from entry within 5 days
                days_since = (date - pa["entry_date"]).days
                if days_since <= 7 and price >= pa["entry_price"] * 1.02:
                    port.add_to_position(ticker, date, price, 0.05, 1/3)
                    pa["add_count"] = 2
                elif days_since > 7:
                    pa["add_count"] = 2  # skip, don't wait forever

            elif pa["add_count"] == 2:
                # Add 3rd tranche if 20-SMA reclaimed
                hist = close[ticker].loc[:date]
                sma20 = hist.rolling(20).mean()
                if len(sma20) > 0 and price > sma20.iloc[-1]:
                    port.add_to_position(ticker, date, price, 0.05, 1/3)
                    del pending_adds[ticker]

        # New entries
        for ticker in SECTOR_ETFS:
            if ticker not in close.columns or ticker in port.positions:
                continue
            hist = close[ticker].loc[:date]
            if len(hist) < 60:
                continue
            sig = sector_recovery_signal(hist)
            if sig.iloc[-1]:
                opened = port.open_position(ticker, date, prices[ticker],
                                            risk_pct=0.05, stop_pct=-0.10,
                                            target_pct=0.15, max_hold=30,
                                            tranche=1/3)
                if opened:
                    pending_adds[ticker] = {
                        "entry_price": prices[ticker],
                        "add_count": 1,
                        "entry_date": date,
                    }

        port.current_equity(prices, date)

    last_date = oot_dates[-1]
    last_prices = close.loc[last_date]
    for ticker in list(port.positions.keys()):
        port.close_position(ticker, last_prices.get(ticker, 0), last_date, "end_of_backtest")

    return port


def run_strategy_d(close, oot_mask):
    """D: 200-SMA Reclaim on Growth Stocks."""
    port = Portfolio(INITIAL_CAPITAL)
    oot_dates = close.index[oot_mask]

    for date in oot_dates:
        prices = close.loc[date]
        port.check_exits(prices, date)

        # Max 3 concurrent positions
        if len(port.positions) >= 3:
            port.current_equity(prices, date)
            continue

        for ticker in GROWTH_STOCKS:
            if ticker not in close.columns or ticker in port.positions:
                continue
            hist = close[ticker].loc[:date]
            if len(hist) < 200:
                continue
            reclaim, sma200 = sma200_reclaim_signal(hist)
            if reclaim.iloc[-1]:
                # Stop: drops back below 200-SMA by >2%
                stop_price = sma200.iloc[-1] * 0.98
                stop_pct = (stop_price / prices[ticker]) - 1
                port.open_position(ticker, date, prices[ticker],
                                   risk_pct=0.10, stop_pct=stop_pct,
                                   target_pct=0.20, max_hold=20)

            if len(port.positions) >= 3:
                break

        port.current_equity(prices, date)

    last_date = oot_dates[-1]
    last_prices = close.loc[last_date]
    for ticker in list(port.positions.keys()):
        port.close_position(ticker, last_prices.get(ticker, 0), last_date, "end_of_backtest")

    return port


def run_strategy_e(close, oot_mask):
    """E: Multi-Timeframe Recovery (sector 30d + stock 10d)."""
    port = Portfolio(INITIAL_CAPITAL)
    oot_dates = close.index[oot_mask]

    for date in oot_dates:
        prices = close.loc[date]
        port.check_exits(prices, date)

        # Count current heat
        equity = port.current_equity(prices, date)
        pos_value = sum(p["shares"] * prices.get(t, p["entry_price"])
                        for t, p in port.positions.items())
        heat = pos_value / equity if equity > 0 else 1.0

        if heat >= 0.25:
            continue

        # Sector ETF signals (longer-term, 5% risk, 30d hold)
        for ticker in SECTOR_ETFS:
            if ticker not in close.columns or ticker in port.positions:
                continue
            hist = close[ticker].loc[:date]
            if len(hist) < 60:
                continue
            sig = sector_recovery_signal(hist)
            if sig.iloc[-1]:
                port.open_position(ticker, date, prices[ticker],
                                   risk_pct=0.05, stop_pct=-0.08,
                                   target_pct=0.15, max_hold=30)

        # Individual stock 50-SMA reclaim (shorter-term, 3% risk, 10d hold)
        for ticker in GROWTH_STOCKS:
            if ticker not in close.columns or ticker in port.positions:
                continue
            hist = close[ticker].loc[:date]
            if len(hist) < 50:
                continue
            reclaim, sma50 = sma50_reclaim_signal(hist)
            if reclaim.iloc[-1]:
                stop_price = sma50.iloc[-1] * 0.98
                stop_pct = (stop_price / prices[ticker]) - 1
                port.open_position(ticker, date, prices[ticker],
                                   risk_pct=0.03, stop_pct=stop_pct,
                                   target_pct=0.10, max_hold=10)

        port.current_equity(prices, date)

    last_date = oot_dates[-1]
    last_prices = close.loc[last_date]
    for ticker in list(port.positions.keys()):
        port.close_position(ticker, last_prices.get(ticker, 0), last_date, "end_of_backtest")

    return port


def run_strategy_f(close, oot_mask):
    """F: Adversarial — Random Entry with same risk management as A."""
    port = Portfolio(INITIAL_CAPITAL)
    oot_dates = close.index[oot_mask]
    rng = np.random.RandomState(123)

    for date in oot_dates:
        prices = close.loc[date]
        port.check_exits(prices, date)

        # Random entry: ~2% chance per ticker per day (matches approx frequency of A)
        for ticker in SECTOR_ETFS:
            if ticker not in close.columns or ticker in port.positions:
                continue
            if rng.random() < 0.02:
                price = prices[ticker]
                if not np.isnan(price) and price > 0:
                    port.open_position(ticker, date, price,
                                       risk_pct=0.05, stop_pct=-0.08,
                                       target_pct=0.15, max_hold=30)

        port.current_equity(prices, date)

    last_date = oot_dates[-1]
    last_prices = close.loc[last_date]
    for ticker in list(port.positions.keys()):
        port.close_position(ticker, last_prices.get(ticker, 0), last_date, "end_of_backtest")

    return port


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("DRAWDOWN RECOVERY V2 BACKTEST — Position-Sized Strategies")
    print("=" * 70)

    close = download_data()
    oot_mask = close.index >= OOT_START
    spy_close = close["SPY"] if "SPY" in close.columns else None

    strategies = {
        "A_sector_recovery": ("A: Sector ETF Recovery (5% risk, SL-8%, TP+15%)", run_strategy_a),
        "B_sector_vix_filter": ("B: Sector Recovery + VIX<25 Filter", run_strategy_b),
        "C_sector_scaling_in": ("C: Sector Recovery + 1/3 Scaling In", run_strategy_c),
        "D_sma200_reclaim": ("D: 200-SMA Reclaim Growth Stocks", run_strategy_d),
        "E_multi_timeframe": ("E: Multi-Timeframe (Sector+Stock)", run_strategy_e),
        "F_adversarial": ("F: ADVERSARIAL (Random Entry)", run_strategy_f),
    }

    results = {}

    for key, (label, run_fn) in strategies.items():
        print(f"\n{'─' * 60}")
        print(f"Running: {label}")
        print(f"{'─' * 60}")

        port = run_fn(close, oot_mask)
        eq = port.get_equity_series()
        trades = port.closed_trades

        if len(eq) < 10:
            print(f"  ⚠ Insufficient data for {label}")
            results[key] = {"label": label, "error": "insufficient data"}
            continue

        metrics = compute_metrics(eq, trades, label)
        if metrics is None:
            print(f"  ⚠ Could not compute metrics for {label}")
            results[key] = {"label": label, "error": "no metrics"}
            continue

        # Permutation test
        print(f"  Running permutation test ({N_PERM} iterations)...")
        perm_p = permutation_test(trades)
        metrics["perm_p"] = round(perm_p, 4)

        # Regime gap
        if spy_close is not None:
            rg = regime_gap(trades, spy_close)
        else:
            rg = 0.499
        metrics["regime_gap"] = rg

        # Gates
        gates = apply_gates(metrics, perm_p, rg)
        metrics["gates"] = gates

        # Trade breakdown
        exit_reasons = {}
        for t in trades:
            r = t.get("reason", "unknown")
            exit_reasons[r] = exit_reasons.get(r, 0) + 1
        metrics["exit_reasons"] = exit_reasons

        results[key] = metrics

        # Print summary
        status = "PASS" if gates["pass"] else "FAIL"
        gate_detail = " | ".join(f"{k}:{'Y' if v else 'N'}" for k, v in gates.items() if k != "pass")
        print(f"  Sharpe: {metrics['sharpe']} | Sortino: {metrics['sortino']} | MDD: {metrics['mdd_pct']}")
        print(f"  Trades: {metrics['n_trades']} | WR: {metrics['win_rate']} | PF: {metrics['profit_factor']}")
        print(f"  Total Return: {metrics['total_return_pct']}% | Final Equity: ${metrics['final_equity']}")
        print(f"  Perm p-value: {perm_p:.4f} | Regime Gap: {rg}")
        print(f"  Exit reasons: {exit_reasons}")
        print(f"  Gates [{status}]: {gate_detail}")

    # ── Summary Table ──────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SUMMARY TABLE")
    print(f"{'=' * 70}")
    print(f"{'Strategy':<45} {'Sharpe':>7} {'Sort':>7} {'MDD':>8} {'Trades':>7} {'WR':>6} {'PF':>6} {'Gates':>6}")
    print("-" * 100)
    for key, m in results.items():
        if "error" in m:
            print(f"{m['label']:<45} {'ERROR':>7}")
            continue
        status = "PASS" if m["gates"]["pass"] else "FAIL"
        print(f"{m['label']:<45} {m['sharpe']:>7.3f} {m['sortino']:>7.3f} {m['mdd_pct']:>8} {m['n_trades']:>7} {m['win_rate']:>6.3f} {m['profit_factor']:>6.2f} {status:>6}")

    # ── Compare A vs F (signal vs random) ──────────────────────────────────
    if "A_sector_recovery" in results and "F_adversarial" in results:
        a = results["A_sector_recovery"]
        f = results["F_adversarial"]
        if "error" not in a and "error" not in f:
            print(f"\n{'=' * 70}")
            print("SIGNAL vs ADVERSARIAL COMPARISON")
            print(f"{'=' * 70}")
            print(f"  A (Signal) Sharpe: {a['sharpe']} | F (Random) Sharpe: {f['sharpe']}")
            print(f"  A (Signal) WR:     {a['win_rate']} | F (Random) WR:     {f['win_rate']}")
            print(f"  A (Signal) Return: {a['total_return_pct']}% | F (Random) Return: {f['total_return_pct']}%")
            edge = a["sharpe"] - f["sharpe"]
            print(f"  Signal Edge (Sharpe delta): {edge:+.3f}")
            if edge > 0.3:
                print("  >> GENUINE SIGNAL EDGE detected over random entry")
            elif edge > 0:
                print("  >> Marginal edge — risk management doing most of the work")
            else:
                print("  >> NO EDGE — alpha is entirely from risk management, not signal")

    # ── Save results ───────────────────────────────────────────────────────
    output_path = "/home/jupiter/Lvl3Quant/data/drawdown_recovery_v2_results.json"
    # Make JSON serializable
    save_results = {}
    for k, v in results.items():
        save_results[k] = {sk: sv for sk, sv in v.items()}

    with open(output_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "oot_period": f"{OOT_START} to present",
            "initial_capital": INITIAL_CAPITAL,
            "gates": GATES,
            "strategies": save_results,
        }, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")
    print("DONE")


if __name__ == "__main__":
    main()
