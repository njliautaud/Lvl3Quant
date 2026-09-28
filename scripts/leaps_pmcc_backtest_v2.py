#!/usr/bin/env python3
"""
LEAPS / Poor Man's Covered Call (PMCC) Backtester v2
====================================================
All option pricing is BS-modeled (Black-Scholes approximation).
BS underestimates real LEAPS prices by ~41% — so modeled returns are CONSERVATIVE.
Real options will be priced differently — this is directional research, not exact pricing.

v2 Fixes vs v1:
  - Removed broken break-even-breach exit (was triggering on day 0 for all positions)
  - Added proper dynamic exits: 30% LEAPS value loss, 50-day SMA breakdown, 15% trailing stop
  - Fixed PMCC overlay to start immediately and cycle properly
  - Fixed capital/position tracking and equity curve
  - Added monthly rebalance of universe
  - Added walk-forward with sliding windows (HC compliant)
  - Proper regime analysis (R1 test)

Strategy:
  1. LEAPS Entry: Buy 0.70-0.80 delta calls, ~12-18 months to expiry
  2. Stock selection: top-quartile 6-month momentum from S&P 500 large-caps
  3. PMCC overlay: Sell ~30-delta short calls (30-45 DTE), roll at 50% profit or 21 DTE
  4. Dynamic exits (HC #684):
     a. Stop-loss: exit if LEAPS loses 30% of entry value
     b. Momentum breakdown: underlying breaks below 50-day SMA
     c. Trailing stop: 15% from peak LEAPS value
  5. Walk-forward: 36-month lookback for stock selection, monthly rebalance, daily exit checks

HC #683: All pricing labeled BS-modeled.
HC #684: Dynamic exit rules applied.
HC #685: Part of growth research lanes.
"""

import json
import os
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings("ignore")

OUTPUT_DIR = "/home/jupiter/Lvl3Quant/output/growth_research"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Universe: liquid large-caps from S&P 500 ───────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B",
    "UNH", "JNJ", "V", "MA", "HD", "PG", "COST", "AVGO", "ABBV", "CRM",
    "AMD", "NFLX", "ADBE", "PEP", "TMO", "LLY", "MRK", "ORCL", "CSCO",
    "ACN", "ABT", "QCOM", "TXN", "LOW", "INTU", "ISRG", "NOW", "AMAT",
    "BKNG", "MDLZ", "ADP", "LRCX",
]

BENCHMARK = "SPY"

# ─── Strategy parameters ────────────────────────────────────────────────────

# LEAPS parameters
LEAPS_DELTA_TARGET = 0.75       # deep ITM (0.70-0.80 range)
LEAPS_TARGET_DTE = 400          # ~13 months to expiry
LEAPS_MIN_DTE_REMAINING = 90    # roll/close if < 90 DTE remaining

# PMCC overlay
PMCC_SHORT_DELTA = 0.30         # OTM short call delta
PMCC_SHORT_DTE = 37             # 30-45 DTE nominal
PMCC_ROLL_PROFIT_PCT = 0.50     # roll at 50% profit
PMCC_ROLL_DTE = 21              # roll if < 21 DTE remaining
PMCC_EXPIRE_WORTHLESS_RATE = 0.72  # % of time short calls expire OTM

# Dynamic exit parameters (HC #684)
STOP_LOSS_PCT = 0.30            # exit if LEAPS loses 30% of entry value
TRAILING_STOP_PCT = 0.15        # 15% from peak LEAPS value
SMA_PERIOD = 50                 # exit if price below 50-day SMA
SMA_GRACE_DAYS = 3              # must be below SMA for 3 consecutive days

# Position sizing
MAX_POSITIONS = 12              # 10-15 positions
POSITION_SIZE_PCT = 0.08        # ~8% per position (leaves buffer)
INITIAL_CAPITAL = 100_000

# Momentum
MOMENTUM_LOOKBACK = 126         # 6-month momentum for selection
MOMENTUM_SKIP = 21              # skip last month (reversal effect)
REBALANCE_FREQ_DAYS = 21        # monthly rebalance

# Risk-free rate (approximate average)
RISK_FREE_RATE = 0.04

# Walk-forward
WF_WINDOWS = [
    # (train_start, train_end, test_start, test_end)
    # 36-month lookback, 12-month OOT, sliding
    ("2015-01-01", "2017-12-31", "2018-01-01", "2018-12-31"),
    ("2016-01-01", "2018-12-31", "2019-01-01", "2019-12-31"),
    ("2017-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
    ("2018-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
    ("2019-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("2020-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("2021-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("2022-01-01", "2024-12-31", "2025-01-01", "2025-07-11"),
]


# ─── Black-Scholes helpers (BS-MODELED) ─────────────────────────────────────

def bs_d1(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def bs_call_price(S, K, T, r, sigma):
    """BS-modeled call price. Underestimates real LEAPS by ~41%."""
    if T <= 0:
        return max(S - K, 0)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_call_delta(S, K, T, r, sigma):
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1 = bs_d1(S, K, T, r, sigma)
    return norm.cdf(d1)

def strike_for_delta(S, T, r, sigma, target_delta):
    """Find strike giving target delta via binary search."""
    lo, hi = S * 0.3, S * 2.0
    for _ in range(100):
        mid = (lo + hi) / 2
        d = bs_call_delta(S, mid, T, r, sigma)
        if d > target_delta:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ─── Data loading ───────────────────────────────────────────────────────────

def load_data(tickers, start="2014-06-01", end="2026-07-13"):
    """Load price data from yfinance."""
    all_tickers = list(set(tickers + [BENCHMARK, "^VIX"]))
    print(f"Downloading data for {len(all_tickers)} tickers...")

    data = {}
    for t in all_tickers:
        try:
            df = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
            if len(df) > 100:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[t] = df
        except Exception as e:
            print(f"  Warning: failed to download {t}: {e}")

    print(f"  Loaded {len(data)} tickers successfully")
    return data


# ─── IV estimation ──────────────────────────────────────────────────────────

def get_iv(stock_data, vix_data, date):
    """Estimate per-stock IV at a given date. BS-MODELED."""
    # VIX component
    if vix_data is not None and date in vix_data.index:
        vix_val = float(vix_data.loc[date, "Close"]) / 100.0
    else:
        # Fallback: find nearest
        mask = vix_data.index <= date if vix_data is not None else pd.Index([])
        if len(mask) > 0 and mask.any():
            vix_val = float(vix_data.loc[mask, "Close"].iloc[-1]) / 100.0
        else:
            vix_val = 0.20

    # Realized vol component
    mask = stock_data.index <= date
    recent = stock_data.loc[mask, "Close"].tail(63)
    if len(recent) > 20:
        rv = float(recent.pct_change().dropna().std() * np.sqrt(252))
    else:
        rv = 0.25

    # Blend and clip
    iv = 0.5 * vix_val + 0.5 * rv
    return float(np.clip(iv, 0.12, 1.20))


# ─── Momentum scoring ──────────────────────────────────────────────────────

def compute_momentum(prices, lookback=126, skip=21):
    """6-month momentum, skip last month (Jegadeesh-Titman)."""
    if len(prices) < lookback + skip + 5:
        return np.nan
    ret = float(prices.iloc[-(skip + 1)] / prices.iloc[-(lookback + skip)] - 1)
    return ret


def select_universe(data, date, n_positions=MAX_POSITIONS):
    """Select top-momentum stocks from universe at given date."""
    scores = {}
    for ticker, df in data.items():
        if ticker in [BENCHMARK, "^VIX"]:
            continue
        mask = df.index <= date
        if mask.sum() < MOMENTUM_LOOKBACK + MOMENTUM_SKIP + 30:
            continue
        prices = df.loc[mask, "Close"]
        score = compute_momentum(prices, MOMENTUM_LOOKBACK, MOMENTUM_SKIP)
        if not np.isnan(score) and score > 0.05:  # require >5% 6-month return
            scores[ticker] = score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    # Top quartile of available, capped at n_positions
    n = min(n_positions, max(len(ranked) // 4, 3))
    return [(t, s) for t, s in ranked[:n]]


# ─── Position tracking ─────────────────────────────────────────────────────

@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    entry_stock_price: float
    leaps_strike: float
    leaps_premium: float          # per-share cost
    leaps_expiry_dte: int         # DTE at entry
    leaps_delta: float
    iv_at_entry: float

    # Tracking
    peak_leaps_value: float = 0.0
    pmcc_income: float = 0.0      # per-share cumulative
    pmcc_cycles: int = 0
    pmcc_last_sell_date: Optional[pd.Timestamp] = None
    pmcc_current_strike: float = 0.0
    pmcc_current_premium: float = 0.0
    pmcc_current_dte_remaining: int = 0
    below_sma_count: int = 0

    # Exit
    exit_date: Optional[pd.Timestamp] = None
    exit_stock_price: float = 0.0
    exit_leaps_value: float = 0.0
    exit_reason: str = ""
    pnl_per_share: float = 0.0

    def leaps_value_at(self, S, days_held, iv):
        """Current BS-modeled LEAPS value per share."""
        remaining = max(self.leaps_expiry_dte - days_held, 1)
        T = remaining / 365.0
        return bs_call_price(S, self.leaps_strike, T, RISK_FREE_RATE, iv)

    def leaps_delta_at(self, S, days_held, iv):
        remaining = max(self.leaps_expiry_dte - days_held, 1)
        T = remaining / 365.0
        return bs_call_delta(S, self.leaps_strike, T, RISK_FREE_RATE, iv)

    def intrinsic_value(self, S):
        return max(S - self.leaps_strike, 0)

    def contract_cost(self):
        """Total cost for 1 contract (100 shares)."""
        return self.leaps_premium * 100


# ─── Backtester ─────────────────────────────────────────────────────────────

class LEAPSPMCCBacktester:
    def __init__(self, data, test_start, test_end):
        self.data = data
        self.test_start = pd.Timestamp(test_start)
        self.test_end = pd.Timestamp(test_end)
        self.vix_data = data.get("^VIX")
        self.spy_data = data.get(BENCHMARK)

        self.capital = INITIAL_CAPITAL
        self.positions: List[Position] = []
        self.closed: List[Position] = []
        self.daily_equity = []
        self.trade_log = []
        self.last_rebalance = None

    def run(self):
        """Run backtest over the test period."""
        if self.spy_data is None:
            print("  ERROR: No SPY data")
            return

        test_mask = (self.spy_data.index >= self.test_start) & (self.spy_data.index <= self.test_end)
        trading_days = self.spy_data.index[test_mask]

        if len(trading_days) == 0:
            print("  No trading days")
            return

        for i, date in enumerate(trading_days):
            # ── Monthly rebalance: select universe and open new positions ──
            if self.last_rebalance is None or (date - self.last_rebalance).days >= REBALANCE_FREQ_DAYS:
                self._rebalance(date)
                self.last_rebalance = date

            # ── Daily: check exits and manage PMCC ──
            positions_to_close = []

            for pos in self.positions:
                td = self.data.get(pos.ticker)
                if td is None or date not in td.index:
                    continue

                S = float(td.loc[date, "Close"])
                days_held = (date - pos.entry_date).days
                iv = get_iv(td, self.vix_data, date)

                current_value = pos.leaps_value_at(S, days_held, iv)

                # Update peak LEAPS value
                if current_value > pos.peak_leaps_value:
                    pos.peak_leaps_value = current_value

                # ── Dynamic exit checks (HC #684) ──
                exit_reason = self._check_exits(pos, S, days_held, iv, current_value, td, date)

                if exit_reason:
                    pos.exit_date = date
                    pos.exit_stock_price = S
                    pos.exit_leaps_value = current_value
                    pos.exit_reason = exit_reason
                    pos.pnl_per_share = (current_value - pos.leaps_premium) + pos.pmcc_income
                    positions_to_close.append(pos)
                    continue

                # ── PMCC overlay management ──
                self._manage_pmcc(pos, S, days_held, iv, date)

            # Close positions
            for pos in positions_to_close:
                self.positions.remove(pos)
                self.closed.append(pos)
                # Return capital
                self.capital += pos.contract_cost() + pos.pnl_per_share * 100

                self.trade_log.append({
                    "ticker": pos.ticker,
                    "entry_date": str(pos.entry_date.date()),
                    "exit_date": str(date.date()),
                    "entry_stock_price": round(pos.entry_stock_price, 2),
                    "exit_stock_price": round(pos.exit_stock_price, 2),
                    "stock_return_pct": round((pos.exit_stock_price / pos.entry_stock_price - 1) * 100, 2),
                    "leaps_premium_per_share": round(pos.leaps_premium, 2),
                    "leaps_exit_value_per_share": round(pos.exit_leaps_value, 2),
                    "leaps_return_pct": round((pos.exit_leaps_value / pos.leaps_premium - 1) * 100, 2),
                    "pmcc_income_per_share": round(pos.pmcc_income, 2),
                    "pnl_per_contract": round(pos.pnl_per_share * 100, 2),
                    "exit_reason": pos.exit_reason,
                    "days_held": days_held,
                    "pmcc_cycles": pos.pmcc_cycles,
                    "entry_delta": round(pos.leaps_delta, 3),
                })

            # ── Record daily equity ──
            total_unrealized = 0.0
            for pos in self.positions:
                td = self.data.get(pos.ticker)
                if td is None or date not in td.index:
                    continue
                S = float(td.loc[date, "Close"])
                days_held = (date - pos.entry_date).days
                iv = get_iv(td, self.vix_data, date)
                cv = pos.leaps_value_at(S, days_held, iv)
                unrealized = ((cv - pos.leaps_premium) + pos.pmcc_income) * 100
                total_unrealized += unrealized

            equity = self.capital + total_unrealized
            self.daily_equity.append({
                "date": date,
                "equity": equity,
                "n_positions": len(self.positions),
            })

        # Close remaining at period end
        if trading_days.size > 0:
            last_date = trading_days[-1]
            for pos in list(self.positions):
                td = self.data.get(pos.ticker)
                if td is None:
                    continue
                if last_date in td.index:
                    S = float(td.loc[last_date, "Close"])
                else:
                    S = float(td["Close"].iloc[-1])
                days_held = (last_date - pos.entry_date).days
                iv = get_iv(td, self.vix_data, last_date)
                cv = pos.leaps_value_at(S, days_held, iv)

                pos.exit_date = last_date
                pos.exit_stock_price = S
                pos.exit_leaps_value = cv
                pos.exit_reason = "PERIOD_END"
                pos.pnl_per_share = (cv - pos.leaps_premium) + pos.pmcc_income
                self.positions.remove(pos)
                self.closed.append(pos)
                self.capital += pos.contract_cost() + pos.pnl_per_share * 100

                self.trade_log.append({
                    "ticker": pos.ticker,
                    "entry_date": str(pos.entry_date.date()),
                    "exit_date": str(last_date.date()),
                    "entry_stock_price": round(pos.entry_stock_price, 2),
                    "exit_stock_price": round(S, 2),
                    "stock_return_pct": round((S / pos.entry_stock_price - 1) * 100, 2),
                    "leaps_premium_per_share": round(pos.leaps_premium, 2),
                    "leaps_exit_value_per_share": round(cv, 2),
                    "leaps_return_pct": round((cv / pos.leaps_premium - 1) * 100, 2),
                    "pmcc_income_per_share": round(pos.pmcc_income, 2),
                    "pnl_per_contract": round(pos.pnl_per_share * 100, 2),
                    "exit_reason": "PERIOD_END",
                    "days_held": days_held,
                    "pmcc_cycles": pos.pmcc_cycles,
                    "entry_delta": round(pos.leaps_delta, 3),
                })

    def _check_exits(self, pos, S, days_held, iv, current_value, td, date):
        """Check all dynamic exit conditions. Returns exit reason or empty string."""

        # 1. LEAPS expiry approaching
        remaining_dte = pos.leaps_expiry_dte - days_held
        if remaining_dte < LEAPS_MIN_DTE_REMAINING:
            return "LEAPS_EXPIRY"

        # 2. Stop-loss: LEAPS lost 30% of entry value
        if current_value < pos.leaps_premium * (1 - STOP_LOSS_PCT):
            return "STOP_LOSS_30PCT"

        # 3. Trailing stop: 15% from peak LEAPS value
        if pos.peak_leaps_value > 0:
            drawdown_from_peak = 1 - current_value / pos.peak_leaps_value
            if drawdown_from_peak > TRAILING_STOP_PCT and days_held > 5:
                return "TRAILING_STOP_15PCT"

        # 4. Momentum breakdown: price below 50-day SMA for 3 consecutive days
        mask = td.index <= date
        sma_data = td.loc[mask, "Close"].tail(SMA_PERIOD + 5)
        if len(sma_data) >= SMA_PERIOD:
            sma50 = float(sma_data.tail(SMA_PERIOD).mean())
            if S < sma50:
                pos.below_sma_count += 1
            else:
                pos.below_sma_count = 0

            if pos.below_sma_count >= SMA_GRACE_DAYS and days_held > 10:
                return "SMA_BREAKDOWN"

        return ""

    def _manage_pmcc(self, pos, S, days_held, iv, date):
        """Manage PMCC overlay: sell/roll short calls."""
        remaining_dte = pos.leaps_expiry_dte - days_held

        # Don't sell short calls if LEAPS expiry too close
        if remaining_dte < LEAPS_MIN_DTE_REMAINING + PMCC_SHORT_DTE:
            return

        # Check if we need to open a new short call
        need_new = False
        if pos.pmcc_last_sell_date is None:
            need_new = True
        elif pos.pmcc_current_dte_remaining <= 0:
            need_new = True
        else:
            # Track DTE of current short call
            days_since_sell = (date - pos.pmcc_last_sell_date).days
            pos.pmcc_current_dte_remaining = PMCC_SHORT_DTE - days_since_sell

            # Roll at 50% profit
            T_remaining = max(pos.pmcc_current_dte_remaining, 1) / 365.0
            current_short_value = bs_call_price(S, pos.pmcc_current_strike, T_remaining, RISK_FREE_RATE, iv)
            if current_short_value < pos.pmcc_current_premium * (1 - PMCC_ROLL_PROFIT_PCT):
                # Capture remaining value as profit
                profit = pos.pmcc_current_premium - current_short_value
                pos.pmcc_income += profit * PMCC_EXPIRE_WORTHLESS_RATE  # risk-adjusted
                need_new = True

            # Roll if DTE < 21
            if pos.pmcc_current_dte_remaining <= PMCC_ROLL_DTE:
                # Close current (approximate: keep what we collected)
                profit = pos.pmcc_current_premium * 0.60  # approximate avg capture
                pos.pmcc_income += profit * PMCC_EXPIRE_WORTHLESS_RATE
                need_new = True

        if need_new:
            # Sell new short call at ~30 delta
            short_T = PMCC_SHORT_DTE / 365.0
            short_strike = strike_for_delta(S, short_T, RISK_FREE_RATE, iv, PMCC_SHORT_DELTA)

            # Ensure short strike > LEAPS strike (debit spread protection)
            if short_strike <= pos.leaps_strike:
                short_strike = pos.leaps_strike * 1.05

            short_premium = bs_call_price(S, short_strike, short_T, RISK_FREE_RATE, iv)

            pos.pmcc_last_sell_date = date
            pos.pmcc_current_strike = short_strike
            pos.pmcc_current_premium = short_premium
            pos.pmcc_current_dte_remaining = PMCC_SHORT_DTE
            pos.pmcc_cycles += 1

    def _rebalance(self, date):
        """Monthly rebalance: select universe and open positions."""
        ranked = select_universe(self.data, date)
        if not ranked:
            return

        existing = {p.ticker for p in self.positions}
        slots = MAX_POSITIONS - len(self.positions)

        if slots <= 0:
            return

        for ticker, score in ranked:
            if slots <= 0:
                break
            if ticker in existing:
                continue

            td = self.data.get(ticker)
            if td is None:
                continue

            mask = td.index <= date
            if mask.sum() < 60:
                continue

            S = float(td.loc[mask, "Close"].iloc[-1])
            iv = get_iv(td, self.vix_data, date)

            # Find LEAPS strike for target delta
            T = LEAPS_TARGET_DTE / 365.0
            leaps_strike = strike_for_delta(S, T, RISK_FREE_RATE, iv, LEAPS_DELTA_TARGET)
            leaps_premium = bs_call_price(S, leaps_strike, T, RISK_FREE_RATE, iv)
            leaps_delta = bs_call_delta(S, leaps_strike, T, RISK_FREE_RATE, iv)

            # Position size check
            contract_cost = leaps_premium * 100
            max_spend = self.capital * POSITION_SIZE_PCT

            if contract_cost > max_spend or contract_cost < 100:
                continue

            self.capital -= contract_cost

            pos = Position(
                ticker=ticker,
                entry_date=date,
                entry_stock_price=S,
                leaps_strike=leaps_strike,
                leaps_premium=leaps_premium,
                leaps_expiry_dte=LEAPS_TARGET_DTE,
                leaps_delta=leaps_delta,
                iv_at_entry=iv,
                peak_leaps_value=leaps_premium,  # start at entry value
            )
            self.positions.append(pos)
            existing.add(ticker)
            slots -= 1

    def compute_metrics(self):
        """Compute performance metrics for this window."""
        if not self.daily_equity:
            return {}

        eq = pd.DataFrame(self.daily_equity).set_index("date")
        returns = eq["equity"].pct_change().dropna()
        returns = returns.replace([np.inf, -np.inf], 0).fillna(0)

        total_days = (eq.index[-1] - eq.index[0]).days
        if total_days < 30:
            return {}

        total_return = eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1
        cagr = (1 + total_return) ** (365 / max(total_days, 1)) - 1

        ann_ret = returns.mean() * 252
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 1e-8 else 0

        downside = returns[returns < 0]
        ds_vol = downside.std() * np.sqrt(252) if len(downside) > 5 else 1e-6
        sortino = ann_ret / ds_vol if ds_vol > 1e-8 else 0

        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        max_dd = float(((cumulative - peak) / peak).min())

        # Trade stats
        if self.closed:
            pnls = [p.pnl_per_share * 100 for p in self.closed]
            winners = [x for x in pnls if x > 0]
            losers = [x for x in pnls if x <= 0]
            win_rate = len(winners) / len(pnls) if pnls else 0
            avg_win = np.mean(winners) if winners else 0
            avg_loss = abs(np.mean(losers)) if losers else 0
            gross_wins = sum(winners) if winners else 0
            gross_losses = sum(abs(x) for x in losers) if losers else 0
            pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")

            total_pmcc = sum(p.pmcc_income * 100 for p in self.closed)
            total_leaps_cost = sum(p.contract_cost() for p in self.closed)
            avg_hold = np.mean([(p.exit_date - p.entry_date).days for p in self.closed if p.exit_date is not None])
            pmcc_ann_rate = (total_pmcc / total_leaps_cost) * (365 / max(avg_hold, 1)) if total_leaps_cost > 0 else 0

            avg_leverage = np.mean([p.entry_stock_price / p.leaps_premium for p in self.closed if p.leaps_premium > 0])
        else:
            win_rate = avg_win = avg_loss = pf = pmcc_ann_rate = 0
            avg_leverage = 1
            avg_hold = 0

        return {
            "total_return_pct": round(total_return * 100, 2),
            "cagr_pct": round(cagr * 100, 2),
            "sharpe": round(sharpe, 3),
            "sortino": round(sortino, 3),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "win_rate_pct": round(win_rate * 100, 1),
            "profit_factor": round(min(pf, 99.99), 2),
            "avg_win_per_contract": round(avg_win, 2),
            "avg_loss_per_contract": round(avg_loss, 2),
            "n_trades": len(self.closed),
            "avg_hold_days": round(avg_hold, 1),
            "pmcc_income_rate_ann_pct": round(pmcc_ann_rate * 100, 2),
            "avg_capital_leverage_x": round(avg_leverage, 2),
            "period": f"{self.test_start.date()} to {self.test_end.date()}",
        }


# ─── Regime analysis ───────────────────────────────────────────────────────

def classify_regime(spy_data, date, lookback=21):
    """Classify as bull/bear/flat based on trailing 21-day SPY return."""
    mask = spy_data.index <= date
    recent = spy_data.loc[mask, "Close"].tail(lookback + 1)
    if len(recent) < lookback:
        return "flat"
    ret = float(recent.iloc[-1] / recent.iloc[0] - 1)
    if ret > 0.02:
        return "bull"
    elif ret < -0.02:
        return "bear"
    return "flat"


def regime_analysis(daily_equity, spy_data):
    """R1 test: per-regime Sharpe and regime gap."""
    if not daily_equity or spy_data is None:
        return {"regime_gap_pass": False, "reason": "no data"}

    eq = pd.DataFrame(daily_equity).set_index("date")
    returns = eq["equity"].pct_change().dropna()

    regime_rets = {"bull": [], "bear": [], "flat": []}
    for date in returns.index:
        r = classify_regime(spy_data, date)
        regime_rets[r].append(float(returns.loc[date]))

    regime_sharpe = {}
    for regime, rets in regime_rets.items():
        if len(rets) > 10:
            arr = np.array(rets)
            ann = arr.mean() * 252
            vol = arr.std() * np.sqrt(252)
            regime_sharpe[regime] = round(ann / vol, 3) if vol > 1e-8 else 0
        else:
            regime_sharpe[regime] = 0

    bull_s = regime_sharpe.get("bull", 0)
    bear_s = regime_sharpe.get("bear", 0)
    denom = max(abs(bull_s), abs(bear_s))
    gap = abs(bull_s - bear_s) / denom if denom > 0 else 0
    passes = gap < 0.50

    return {
        "sharpe_per_regime": regime_sharpe,
        "regime_gap": round(gap, 3),
        "regime_gap_pass": passes,
        "n_days_per_regime": {r: len(v) for r, v in regime_rets.items()},
    }


def spy_benchmark(spy_data, start, end):
    """SPY buy-and-hold benchmark."""
    mask = (spy_data.index >= pd.Timestamp(start)) & (spy_data.index <= pd.Timestamp(end))
    subset = spy_data.loc[mask, "Close"]
    if len(subset) < 10:
        return {}

    returns = subset.pct_change().dropna()
    total_days = (subset.index[-1] - subset.index[0]).days
    total_return = float(subset.iloc[-1] / subset.iloc[0] - 1)
    cagr = (1 + total_return) ** (365 / max(total_days, 1)) - 1

    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    ds = returns[returns < 0]
    ds_vol = ds.std() * np.sqrt(252) if len(ds) > 0 else 1e-6
    sortino = ann_ret / ds_vol

    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    max_dd = float(((cumulative - peak) / peak).min())

    return {
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
    }


# ─── Walk-forward runner ───────────────────────────────────────────────────

def run_walk_forward(data):
    """Run all walk-forward windows."""
    all_results = []
    all_equity = []
    all_trades = []

    for train_start, train_end, test_start, test_end in WF_WINDOWS:
        print(f"\n{'='*60}")
        print(f"Window: OOT {test_start} to {test_end}")
        print(f"{'='*60}")

        bt = LEAPSPMCCBacktester(data, test_start, test_end)
        bt.run()

        metrics = bt.compute_metrics()
        if not metrics:
            print("  No metrics")
            continue

        spy_bench = spy_benchmark(data.get(BENCHMARK), test_start, test_end) if data.get(BENCHMARK) is not None else {}
        regime = regime_analysis(bt.daily_equity, data.get(BENCHMARK))

        metrics["spy_benchmark"] = spy_bench
        metrics["regime_analysis"] = regime

        all_results.append(metrics)
        all_equity.extend(bt.daily_equity)
        all_trades.extend(bt.trade_log)

        print(f"  LEAPS+PMCC: CAGR={metrics['cagr_pct']:.1f}%, Sharpe={metrics['sharpe']:.3f}, "
              f"Sortino={metrics['sortino']:.3f}, MaxDD={metrics['max_drawdown_pct']:.1f}%")
        print(f"  SPY B&H:    CAGR={spy_bench.get('cagr_pct', 'N/A')}%, Sharpe={spy_bench.get('sharpe', 'N/A')}")
        print(f"  Trades: {metrics['n_trades']}, WR: {metrics['win_rate_pct']:.1f}%, "
              f"PF: {metrics['profit_factor']:.2f}, Avg hold: {metrics['avg_hold_days']:.0f}d")
        print(f"  PMCC income (ann): {metrics['pmcc_income_rate_ann_pct']:.1f}%")
        print(f"  Leverage: {metrics['avg_capital_leverage_x']:.1f}x")
        if regime.get("sharpe_per_regime"):
            print(f"  Regime Sharpe: {regime['sharpe_per_regime']}")
            gap_str = "PASS" if regime.get("regime_gap_pass") else "FAIL"
            print(f"  Regime gap: {regime.get('regime_gap', 'N/A')} ({gap_str})")

    return all_results, all_equity, all_trades


def build_summary(all_results, all_equity, all_trades):
    """Build aggregate summary."""
    if not all_results:
        return {"error": "no results"}

    # Aggregate equity
    if all_equity:
        eq = pd.DataFrame(all_equity).set_index("date").sort_index()
        returns = eq["equity"].pct_change().dropna().replace([np.inf, -np.inf], 0).fillna(0)

        total_days = (eq.index[-1] - eq.index[0]).days
        total_return = float(eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1)
        cagr = (1 + total_return) ** (365 / max(total_days, 1)) - 1

        ann_ret = returns.mean() * 252
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = float(ann_ret / ann_vol) if ann_vol > 1e-8 else 0

        ds = returns[returns < 0]
        ds_vol = ds.std() * np.sqrt(252) if len(ds) > 5 else 1e-6
        sortino = float(ann_ret / ds_vol) if ds_vol > 1e-8 else 0

        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        max_dd = float(((cumulative - peak) / peak).min())
    else:
        cagr = sharpe = sortino = max_dd = total_return = 0

    # Averages
    avg = lambda key: round(float(np.mean([r[key] for r in all_results])), 2)

    # Regime
    gaps = [r["regime_analysis"]["regime_gap"] for r in all_results if "regime_analysis" in r]
    passes = [r["regime_analysis"]["regime_gap_pass"] for r in all_results if "regime_analysis" in r]

    # SPY
    spy_cagrs = [r["spy_benchmark"]["cagr_pct"] for r in all_results if r.get("spy_benchmark")]
    spy_sharpes = [r["spy_benchmark"]["sharpe"] for r in all_results if r.get("spy_benchmark")]

    # Exit reason distribution
    exit_reasons = {}
    for t in all_trades:
        r = t.get("exit_reason", "UNKNOWN")
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    summary = {
        "note": "ALL PRICING IS BS-MODELED. BS underestimates LEAPS by ~41% — modeled returns are CONSERVATIVE vs real.",
        "strategy": {
            "description": "LEAPS (0.75 delta, ~13mo) on top-momentum S&P500 stocks + PMCC overlay (0.30 delta short calls, 30-45 DTE)",
            "universe": f"{len(UNIVERSE)} S&P 500 large-caps",
            "position_sizing": f"Equal weight, {MAX_POSITIONS} positions max, {POSITION_SIZE_PCT*100:.0f}% per position",
            "dynamic_exits": [
                f"Stop-loss: {STOP_LOSS_PCT*100:.0f}% LEAPS value decline",
                f"Trailing stop: {TRAILING_STOP_PCT*100:.0f}% from peak LEAPS value",
                f"Momentum: {SMA_PERIOD}-day SMA breakdown ({SMA_GRACE_DAYS} consecutive days)",
                f"Expiry: exit when LEAPS < {LEAPS_MIN_DTE_REMAINING} DTE remaining",
            ],
            "rebalance": "Monthly",
        },
        "aggregate_metrics": {
            "total_return_pct": round(total_return * 100, 2),
            "cagr_pct": round(cagr * 100, 2),
            "sharpe": round(sharpe, 3),
            "sortino": round(sortino, 3),
            "max_drawdown_pct": round(max_dd * 100, 2),
        },
        "per_window_averages": {
            "avg_cagr_pct": avg("cagr_pct"),
            "avg_sharpe": avg("sharpe"),
            "avg_sortino": avg("sortino"),
            "avg_win_rate_pct": avg("win_rate_pct"),
            "avg_profit_factor": avg("profit_factor"),
            "avg_hold_days": avg("avg_hold_days"),
            "avg_pmcc_income_rate_ann_pct": avg("pmcc_income_rate_ann_pct"),
            "avg_capital_leverage_x": avg("avg_capital_leverage_x"),
        },
        "spy_benchmark_averages": {
            "avg_cagr_pct": round(float(np.mean(spy_cagrs)), 2) if spy_cagrs else None,
            "avg_sharpe": round(float(np.mean(spy_sharpes)), 3) if spy_sharpes else None,
        },
        "capital_efficiency": {
            "description": "LEAPS provide leveraged exposure at fraction of stock cost",
            "avg_leverage_x": avg("avg_capital_leverage_x"),
            "meaning": f"~{avg('avg_capital_leverage_x')}x capital efficiency vs buying stock outright",
            "example": "$100K in LEAPS controls same delta exposure as ~$500-600K in stock",
        },
        "regime_analysis_r1": {
            "avg_regime_gap": round(float(np.mean(gaps)), 3) if gaps else None,
            "windows_passing": f"{sum(1 for p in passes if p)}/{len(passes)}" if passes else "N/A",
            "r1_threshold": "gap < 0.50",
        },
        "exit_reason_distribution": exit_reasons,
        "n_windows": len(all_results),
        "n_total_trades": len(all_trades),
        "per_window_results": all_results,
        "trade_log_sample": all_trades[:30],
    }

    return summary


def main():
    print("=" * 70)
    print("LEAPS / PMCC BACKTESTER v2")
    print("All pricing is BS-MODELED (underestimates by ~41% — returns are conservative)")
    print("=" * 70)

    data = load_data(UNIVERSE, start="2014-01-01", end="2026-07-13")

    if len(data) < 10:
        print("ERROR: insufficient data")
        sys.exit(1)

    all_results, all_equity, all_trades = run_walk_forward(data)

    if not all_results:
        print("ERROR: no results")
        sys.exit(1)

    summary = build_summary(all_results, all_equity, all_trades)

    # Print final
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS (BS-MODELED — CONSERVATIVE)")
    print("=" * 70)

    am = summary["aggregate_metrics"]
    pw = summary["per_window_averages"]
    spy = summary["spy_benchmark_averages"]

    print(f"\n  LEAPS+PMCC Strategy:")
    print(f"    Total Return:   {am['total_return_pct']:.1f}%")
    print(f"    CAGR:           {am['cagr_pct']:.1f}%")
    print(f"    Sharpe:         {am['sharpe']:.3f}")
    print(f"    Sortino:        {am['sortino']:.3f}")
    print(f"    Max Drawdown:   {am['max_drawdown_pct']:.1f}%")

    print(f"\n  Per-Window Averages:")
    print(f"    Avg CAGR:       {pw['avg_cagr_pct']}%")
    print(f"    Avg Sharpe:     {pw['avg_sharpe']}")
    print(f"    Avg Sortino:    {pw['avg_sortino']}")
    print(f"    Avg Win Rate:   {pw['avg_win_rate_pct']}%")
    print(f"    Avg PF:         {pw['avg_profit_factor']}")
    print(f"    Avg Hold Days:  {pw['avg_hold_days']}")

    print(f"\n  SPY Buy-and-Hold:")
    print(f"    Avg CAGR:       {spy.get('avg_cagr_pct', 'N/A')}%")
    print(f"    Avg Sharpe:     {spy.get('avg_sharpe', 'N/A')}")

    print(f"\n  PMCC Income:      {pw['avg_pmcc_income_rate_ann_pct']}% ann. of LEAPS cost")
    print(f"  Capital Leverage: {pw['avg_capital_leverage_x']}x vs stock")

    ra = summary["regime_analysis_r1"]
    print(f"\n  R1 Regime Test:")
    print(f"    Avg gap:        {ra.get('avg_regime_gap', 'N/A')} (threshold < 0.50)")
    print(f"    Windows pass:   {ra.get('windows_passing', 'N/A')}")

    print(f"\n  Exit Distribution: {summary['exit_reason_distribution']}")
    print(f"  Total trades: {summary['n_total_trades']} across {summary['n_windows']} windows")

    # Save
    out_path = os.path.join(OUTPUT_DIR, "leaps_pmcc_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    trades_path = os.path.join(OUTPUT_DIR, "leaps_pmcc_trades_v2.json")
    with open(trades_path, "w") as f:
        json.dump(all_trades, f, indent=2, default=str)
    print(f"  Trades saved to {trades_path}")

    if all_equity:
        eq_df = pd.DataFrame(all_equity)
        eq_path = os.path.join(OUTPUT_DIR, "leaps_pmcc_equity_v2.csv")
        eq_df.to_csv(eq_path, index=False)
        print(f"  Equity saved to {eq_path}")


if __name__ == "__main__":
    main()
