#!/usr/bin/env python3
"""
LEAPS / Poor Man's Covered Call (PMCC) Backtester
=================================================
All option pricing is BS-modeled (Black-Scholes approximation).
Real options will be priced differently — this is directional research, not exact pricing.

Strategy:
  1. LEAPS Entry: Buy 0.70-0.80 delta calls, 9-18 months to expiry
  2. PMCC overlay: Sell 0.25-0.30 delta calls, 30-45 DTE against each LEAPS
  3. Dynamic exits: break-even breach, 20d momentum flip, trailing stop at 2x ATR

Walk-forward: 2018-2026, 2-year train (momentum ranking), 1-year OOT

HC #683: All pricing labeled BS-modeled.
HC #684: Dynamic exit rules applied.
"""

import json
import os
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings("ignore")

OUTPUT_DIR = "/home/jupiter/Lvl3Quant/output/leaps_pmcc"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Universe: liquid large/mid caps from S&P 500 + NASDAQ-100 ──────────────
UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B",
    "UNH", "JNJ", "V", "MA", "HD", "PG", "COST", "AVGO", "ABBV", "CRM",
    "AMD", "NFLX", "ADBE", "PEP", "TMO", "LLY", "MRK", "ORCL", "CSCO",
    "ACN", "ABT", "QCOM",
]

# Benchmark
BENCHMARK = "SPY"

# Strategy parameters
LEAPS_DELTA_TARGET = 0.75          # target delta for LEAPS call (0.70-0.80 range)
LEAPS_MIN_DTE = 270                # ~9 months minimum
LEAPS_MAX_DTE = 540                # ~18 months maximum
LEAPS_TARGET_DTE = 365             # 12 months nominal

PMCC_SHORT_DELTA = 0.27            # target delta for short call (0.25-0.30)
PMCC_SHORT_DTE = 37                # 30-45 DTE nominal
PMCC_EXPIRE_WORTHLESS_RATE = 0.75  # 75% of OTM short calls expire worthless

# Exit parameters (HC #684)
TRAILING_STOP_ATR_MULT = 2.0
MOMENTUM_LOOKBACK = 20             # 20-day momentum
RISK_FREE_RATE = 0.045             # approximate average over 2018-2026

# Position sizing
MAX_POSITIONS = 10                 # max concurrent LEAPS positions
POSITION_SIZE_PCT = 0.10           # 10% of portfolio per position
INITIAL_CAPITAL = 100_000

# Walk-forward
WF_TRAIN_YEARS = 2
WF_OOT_YEARS = 1


# ─── Black-Scholes helpers (BS-MODELED) ─────────────────────────────────────

def bs_d1(S, K, T, r, sigma):
    """BS d1 parameter."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def bs_d2(S, K, T, r, sigma):
    return bs_d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def bs_call_price(S, K, T, r, sigma):
    """BS-modeled call price."""
    if T <= 0:
        return max(S - K, 0)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_call_delta(S, K, T, r, sigma):
    """BS-modeled call delta."""
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1 = bs_d1(S, K, T, r, sigma)
    return norm.cdf(d1)


def bs_call_theta(S, K, T, r, sigma):
    """BS-modeled call theta (per day)."""
    if T <= 0:
        return 0.0
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    term1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
    return (term1 + term2) / 252  # per trading day


def strike_for_delta(S, T, r, sigma, target_delta, call=True):
    """Find strike that gives target delta (BS-modeled). Binary search."""
    lo, hi = S * 0.3, S * 2.0
    for _ in range(100):
        mid = (lo + hi) / 2
        d = bs_call_delta(S, mid, T, r, sigma)
        if call:
            if d > target_delta:
                lo = mid
            else:
                hi = mid
        else:
            if d < target_delta:
                lo = mid
            else:
                hi = mid
    return (lo + hi) / 2


# ─── Data loading ───────────────────────────────────────────────────────────

def load_data(tickers, start="2016-01-01", end="2026-07-11"):
    """Load price data from yfinance."""
    all_tickers = list(set(tickers + [BENCHMARK, "^VIX"]))
    print(f"Downloading data for {len(all_tickers)} tickers...")

    data = {}
    for t in all_tickers:
        try:
            df = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
            if len(df) > 100:
                # Handle multi-level columns from yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[t] = df
        except Exception as e:
            print(f"  Warning: failed to download {t}: {e}")

    print(f"  Loaded {len(data)} tickers successfully")
    return data


# ─── IV estimation (BS-modeled) ─────────────────────────────────────────────

def estimate_iv(stock_data, vix_data):
    """
    Estimate per-stock IV using VIX as base + stock beta scaling.
    BS-MODELED: Real IV surfaces are much more complex.
    """
    # Align dates
    common_idx = stock_data.index.intersection(vix_data.index)
    if len(common_idx) < 60:
        return pd.Series(0.30, index=stock_data.index)  # fallback

    stock_rets = stock_data["Close"].pct_change()

    # Use VIX / 100 as base IV, scale by realized vol ratio
    vix_iv = vix_data["Close"].reindex(stock_data.index, method="ffill") / 100.0

    # Rolling 60-day realized vol (annualized)
    realized_vol = stock_rets.rolling(60).std() * np.sqrt(252)

    # SPY realized vol for beta-like scaling
    spy_vol = realized_vol.rolling(20).mean()  # smoothed

    # IV estimate: blend of VIX-scaled and realized vol
    # This is approximate — real IV has skew, term structure, etc.
    iv = 0.5 * vix_iv + 0.5 * realized_vol
    iv = iv.clip(0.10, 1.50)  # cap at reasonable range
    iv = iv.fillna(0.30)

    return iv


# ─── Momentum scoring ───────────────────────────────────────────────────────

def compute_momentum_score(prices, lookback=252):
    """12-month momentum, skip last month (Jegadeesh-Titman style)."""
    if len(prices) < lookback + 21:
        return np.nan
    ret_12m = prices.iloc[-22] / prices.iloc[-lookback] - 1  # skip last 21 days
    return ret_12m


def rank_universe(data, date, lookback=252):
    """Rank universe by momentum at given date. Return top stocks."""
    scores = {}
    for ticker, df in data.items():
        if ticker in [BENCHMARK, "^VIX"]:
            continue
        mask = df.index <= date
        if mask.sum() < lookback + 30:
            continue
        subset = df.loc[mask, "Close"]
        score = compute_momentum_score(subset, lookback)
        if not np.isnan(score) and score > 0:  # only positive momentum
            scores[ticker] = score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:MAX_POSITIONS]


# ─── Position tracking ──────────────────────────────────────────────────────

@dataclass
class LEAPSPosition:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float         # stock price at entry
    leaps_strike: float        # LEAPS call strike (deep ITM)
    leaps_premium: float       # BS-modeled premium paid
    leaps_expiry_dte: int      # days to expiry at entry
    leaps_delta: float         # initial delta
    iv_at_entry: float

    # PMCC overlay tracking
    pmcc_premium_collected: float = 0.0
    pmcc_cycles: int = 0

    # Trailing stop
    peak_price: float = 0.0
    atr_at_entry: float = 0.0

    # Status
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    pnl: float = 0.0

    # Note: leaps_premium is PER-SHARE cost. Multiply by 100 for per-contract.

    def break_even_price(self):
        """Stock price at which LEAPS breaks even (per-share)."""
        return self.leaps_strike + self.leaps_premium

    def current_leaps_value(self, S, days_held, iv):
        """BS-modeled current LEAPS value (per-share)."""
        remaining_dte = max(self.leaps_expiry_dte - days_held, 0)
        T = remaining_dte / 365.0
        return bs_call_price(S, self.leaps_strike, T, RISK_FREE_RATE, iv)

    def current_delta(self, S, days_held, iv):
        """BS-modeled current delta."""
        remaining_dte = max(self.leaps_expiry_dte - days_held, 0)
        T = remaining_dte / 365.0
        return bs_call_delta(S, self.leaps_strike, T, RISK_FREE_RATE, iv)


# ─── Backtester ──────────────────────────────────────────────────────────────

class LEAPSPMCCBacktester:
    def __init__(self, data, train_start, train_end, test_start, test_end):
        self.data = data
        self.train_start = pd.Timestamp(train_start)
        self.train_end = pd.Timestamp(train_end)
        self.test_start = pd.Timestamp(test_start)
        self.test_end = pd.Timestamp(test_end)

        self.capital = INITIAL_CAPITAL
        self.positions: list[LEAPSPosition] = []
        self.closed_positions: list[LEAPSPosition] = []
        self.daily_equity = []
        self.trade_log = []

    def run(self):
        """Run the backtest over the OOT period."""
        # Rank universe using training period momentum
        ranked = rank_universe(self.data, self.train_end)
        target_tickers = [t for t, _ in ranked]

        if not target_tickers:
            print("  No tickers passed momentum filter")
            return

        print(f"  Top momentum stocks: {target_tickers}")

        # Get trading days in test period
        spy_data = self.data.get(BENCHMARK)
        if spy_data is None:
            print("  No SPY data")
            return

        test_mask = (spy_data.index >= self.test_start) & (spy_data.index <= self.test_end)
        trading_days = spy_data.index[test_mask]

        if len(trading_days) == 0:
            print("  No trading days in test period")
            return

        # Initialize positions on first day
        self._enter_positions(target_tickers, trading_days[0])

        # Daily simulation
        pmcc_next_cycle = {}  # ticker -> next date to sell short call

        for i, date in enumerate(trading_days):
            day_pnl = 0.0
            positions_to_close = []

            for pos in self.positions:
                ticker_data = self.data.get(pos.ticker)
                if ticker_data is None or date not in ticker_data.index:
                    continue

                S = ticker_data.loc[date, "Close"]
                days_held = (date - pos.entry_date).days

                # Update peak price for trailing stop
                if S > pos.peak_price:
                    pos.peak_price = S

                # Estimate current IV
                vix_data = self.data.get("^VIX")
                if vix_data is not None and date in vix_data.index:
                    current_iv = vix_data.loc[date, "Close"] / 100.0
                    # Scale by stock's historical vol ratio
                    stock_rets = ticker_data["Close"].pct_change().loc[:date].tail(60)
                    stock_rv = stock_rets.std() * np.sqrt(252) if len(stock_rets) > 20 else 0.30
                    current_iv = 0.5 * current_iv + 0.5 * stock_rv
                    current_iv = np.clip(current_iv, 0.10, 1.50)
                else:
                    current_iv = pos.iv_at_entry

                # ── Exit checks (HC #684) ──
                exit_reason = ""

                # 1. LEAPS expiry approaching (< 60 DTE remaining) → roll or exit
                remaining_dte = pos.leaps_expiry_dte - days_held
                if remaining_dte < 60:
                    exit_reason = "LEAPS_EXPIRY_APPROACHING"

                # 2. Break-even breach
                if S < pos.break_even_price():
                    exit_reason = "BREAK_EVEN_BREACH"

                # 3. 20-day momentum negative
                recent = ticker_data["Close"].loc[:date].tail(MOMENTUM_LOOKBACK + 1)
                if len(recent) >= MOMENTUM_LOOKBACK:
                    mom_20d = recent.iloc[-1] / recent.iloc[0] - 1
                    if mom_20d < -0.05:  # meaningful negative momentum
                        exit_reason = "MOMENTUM_NEGATIVE"

                # 4. Trailing stop at 2x ATR
                if pos.atr_at_entry > 0:
                    trail_stop = pos.peak_price - TRAILING_STOP_ATR_MULT * pos.atr_at_entry
                    if S < trail_stop:
                        exit_reason = "TRAILING_STOP"

                if exit_reason:
                    # Close position — all values per-share, multiply by 100 for contract P&L
                    leaps_value = pos.current_leaps_value(S, days_held, current_iv)
                    per_share_pnl = (leaps_value - pos.leaps_premium) + pos.pmcc_premium_collected
                    pos.pnl = per_share_pnl * 100  # per-contract P&L
                    pos.exit_date = date
                    pos.exit_price = S
                    pos.exit_reason = exit_reason
                    positions_to_close.append(pos)

                    self.trade_log.append({
                        "ticker": pos.ticker,
                        "entry_date": str(pos.entry_date.date()),
                        "exit_date": str(date.date()),
                        "entry_price": round(pos.entry_price, 2),
                        "exit_price": round(S, 2),
                        "leaps_premium_per_share": round(pos.leaps_premium, 2),
                        "leaps_exit_value_per_share": round(leaps_value, 2),
                        "pmcc_income_per_share": round(pos.pmcc_premium_collected, 2),
                        "total_pnl_per_contract": round(pos.pnl, 2),
                        "exit_reason": exit_reason,
                        "days_held": days_held,
                        "pmcc_cycles": pos.pmcc_cycles,
                    })
                    continue

                # ── PMCC overlay: sell short calls every ~37 days ──
                if pos.ticker not in pmcc_next_cycle:
                    pmcc_next_cycle[pos.ticker] = pos.entry_date + timedelta(days=7)

                if date >= pmcc_next_cycle.get(pos.ticker, date):
                    # Sell OTM call at ~0.27 delta
                    short_T = PMCC_SHORT_DTE / 365.0
                    short_strike = strike_for_delta(S, short_T, RISK_FREE_RATE, current_iv, PMCC_SHORT_DELTA)
                    short_premium = bs_call_price(S, short_strike, short_T, RISK_FREE_RATE, current_iv)

                    # Model: 75% expire worthless (keep full premium), 25% ITM (lose some)
                    # When ITM, average loss ≈ premium (net zero on those)
                    expected_income = short_premium * PMCC_EXPIRE_WORTHLESS_RATE

                    # Scale by number of contracts (1 LEAPS = 1 short call overlay)
                    pos.pmcc_premium_collected += expected_income
                    pos.pmcc_cycles += 1

                    pmcc_next_cycle[pos.ticker] = date + timedelta(days=PMCC_SHORT_DTE)

                # Track unrealized P&L for equity curve (per-share * 100)
                leaps_value = pos.current_leaps_value(S, days_held, current_iv)
                unrealized = ((leaps_value - pos.leaps_premium) + pos.pmcc_premium_collected) * 100
                day_pnl += unrealized

            # Remove closed positions
            for pos in positions_to_close:
                self.positions.remove(pos)
                self.closed_positions.append(pos)
                # Return original cost + P&L to capital
                self.capital += pos.leaps_premium * 100 + pos.pnl

            # Re-enter if we have capacity (quarterly rebalance check)
            if len(self.positions) < MAX_POSITIONS // 2 and i % 63 == 0 and i > 0:
                # Re-rank and enter new positions
                existing_tickers = {p.ticker for p in self.positions}
                new_ranked = rank_universe(self.data, date)
                new_targets = [t for t, _ in new_ranked if t not in existing_tickers]
                slots = MAX_POSITIONS - len(self.positions)
                self._enter_positions(new_targets[:slots], date)

            # Record daily equity
            total_invested = sum(p.leaps_premium for p in self.positions)
            total_unrealized = day_pnl
            equity = self.capital + total_unrealized

            self.daily_equity.append({
                "date": date,
                "equity": equity,
                "n_positions": len(self.positions),
                "capital_deployed": total_invested,
            })

        # Close remaining positions at end
        for pos in list(self.positions):
            ticker_data = self.data.get(pos.ticker)
            if ticker_data is None:
                continue
            last_date = trading_days[-1]
            if last_date in ticker_data.index:
                S = ticker_data.loc[last_date, "Close"]
            else:
                S = ticker_data["Close"].iloc[-1]
            days_held = (last_date - pos.entry_date).days
            leaps_value = pos.current_leaps_value(S, days_held, pos.iv_at_entry)
            pos.pnl = (leaps_value - pos.leaps_premium) + pos.pmcc_premium_collected
            pos.exit_date = last_date
            pos.exit_price = S
            pos.exit_reason = "PERIOD_END"
            self.closed_positions.append(pos)
            self.trade_log.append({
                "ticker": pos.ticker,
                "entry_date": str(pos.entry_date.date()),
                "exit_date": str(last_date.date()),
                "entry_price": round(pos.entry_price, 2),
                "exit_price": round(S, 2),
                "leaps_premium": round(pos.leaps_premium, 2),
                "leaps_exit_value": round(leaps_value, 2),
                "pmcc_income": round(pos.pmcc_premium_collected, 2),
                "total_pnl": round(pos.pnl, 2),
                "exit_reason": "PERIOD_END",
                "days_held": days_held,
                "pmcc_cycles": pos.pmcc_cycles,
            })

    def _enter_positions(self, tickers, date):
        """Enter LEAPS positions for given tickers on date."""
        for ticker in tickers:
            if len(self.positions) >= MAX_POSITIONS:
                break

            ticker_data = self.data.get(ticker)
            if ticker_data is None:
                continue

            mask = ticker_data.index <= date
            if mask.sum() < 60:
                continue

            S = ticker_data.loc[mask, "Close"].iloc[-1]

            # Estimate IV
            vix_data = self.data.get("^VIX")
            if vix_data is not None and date in vix_data.index:
                vix_val = vix_data.loc[date, "Close"] / 100.0
            else:
                vix_val = 0.25

            stock_rets = ticker_data["Close"].pct_change().loc[:date].tail(60)
            stock_rv = stock_rets.std() * np.sqrt(252) if len(stock_rets) > 20 else 0.30
            iv = 0.5 * vix_val + 0.5 * stock_rv
            iv = np.clip(iv, 0.10, 1.50)

            # Calculate ATR for trailing stop
            recent = ticker_data.loc[mask].tail(20)
            if len(recent) >= 2:
                high_low = recent["High"] - recent["Low"]
                atr = high_low.mean()
            else:
                atr = S * 0.02

            # Find LEAPS strike for target delta
            T = LEAPS_TARGET_DTE / 365.0
            leaps_strike = strike_for_delta(S, T, RISK_FREE_RATE, iv, LEAPS_DELTA_TARGET)
            leaps_premium = bs_call_price(S, leaps_strike, T, RISK_FREE_RATE, iv)
            leaps_delta = bs_call_delta(S, leaps_strike, T, RISK_FREE_RATE, iv)

            # Position size check: 1 contract = 100 shares
            contract_cost = leaps_premium * 100
            max_spend = self.capital * POSITION_SIZE_PCT
            if contract_cost > max_spend:
                continue  # too expensive

            # Deduct per-contract cost from capital
            self.capital -= contract_cost

            pos = LEAPSPosition(
                ticker=ticker,
                entry_date=date,
                entry_price=S,
                leaps_strike=leaps_strike,
                leaps_premium=leaps_premium,  # PER-SHARE premium
                leaps_expiry_dte=LEAPS_TARGET_DTE,
                leaps_delta=leaps_delta,
                iv_at_entry=iv,
                peak_price=S,
                atr_at_entry=atr,
            )
            self.positions.append(pos)

    def compute_metrics(self):
        """Compute performance metrics."""
        if not self.daily_equity:
            return {}

        eq = pd.DataFrame(self.daily_equity)
        eq.set_index("date", inplace=True)

        returns = eq["equity"].pct_change().dropna()
        returns = returns.replace([np.inf, -np.inf], 0).fillna(0)

        # Basic metrics
        total_days = (eq.index[-1] - eq.index[0]).days
        total_return = eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1
        cagr = (1 + total_return) ** (365 / max(total_days, 1)) - 1

        # Risk metrics
        ann_ret = returns.mean() * 252
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

        downside = returns[returns < 0]
        downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-6
        sortino = ann_ret / downside_vol if downside_vol > 0 else 0

        # Max drawdown
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_dd = drawdown.min()

        # Trade stats
        if self.closed_positions:
            pnls = [p.pnl for p in self.closed_positions]
            winners = [p for p in pnls if p > 0]
            win_rate = len(winners) / len(pnls) if pnls else 0
            avg_win = np.mean(winners) if winners else 0
            losers = [p for p in pnls if p <= 0]
            avg_loss = abs(np.mean(losers)) if losers else 1e-6
            profit_factor = (sum(winners) / sum(abs(l) for l in losers)) if losers and sum(abs(l) for l in losers) > 0 else float("inf")

            total_pmcc_income = sum(p.pmcc_premium_collected for p in self.closed_positions)
            total_leaps_cost = sum(p.leaps_premium for p in self.closed_positions)
            pmcc_income_rate = total_pmcc_income / total_leaps_cost if total_leaps_cost > 0 else 0
            # Annualize
            avg_hold_days = np.mean([p.exit_date - p.entry_date for p in self.closed_positions if p.exit_date]).days if self.closed_positions else 365
            pmcc_income_rate_ann = pmcc_income_rate * (365 / max(avg_hold_days, 1))
        else:
            win_rate = 0
            profit_factor = 0
            pmcc_income_rate_ann = 0
            avg_win = 0
            avg_loss = 0

        # Capital efficiency: compare returns to buying stock outright
        # LEAPS cost ~15-25% of stock price, so leverage = stock_price / leaps_premium
        avg_leverage = np.mean([p.entry_price * 100 / p.leaps_premium for p in self.closed_positions]) if self.closed_positions else 1

        return {
            "total_return_pct": round(total_return * 100, 2),
            "cagr_pct": round(cagr * 100, 2),
            "sharpe": round(sharpe, 3),
            "sortino": round(sortino, 3),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "win_rate_pct": round(win_rate * 100, 1),
            "profit_factor": round(profit_factor, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "n_trades": len(self.closed_positions),
            "pmcc_income_rate_annualized_pct": round(pmcc_income_rate_ann * 100, 2),
            "avg_capital_leverage": round(avg_leverage, 2),
            "period": f"{self.test_start.date()} to {self.test_end.date()}",
        }


# ─── Regime analysis ────────────────────────────────────────────────────────

def classify_regime(spy_data, date, lookback=21):
    """Classify month as bull/bear/flat based on SPY return."""
    mask = spy_data.index <= date
    recent = spy_data.loc[mask, "Close"].tail(lookback + 1)
    if len(recent) < lookback:
        return "flat"
    ret = recent.iloc[-1] / recent.iloc[0] - 1
    if ret > 0.02:
        return "bull"
    elif ret < -0.02:
        return "bear"
    else:
        return "flat"


def regime_analysis(daily_equity, spy_data):
    """Compute per-regime Sharpe. Return dict + pass/fail on regime gap."""
    if not daily_equity:
        return {"pass": False, "reason": "no data"}

    eq = pd.DataFrame(daily_equity).set_index("date")
    returns = eq["equity"].pct_change().dropna()

    regime_returns = {"bull": [], "bear": [], "flat": []}

    for date in returns.index:
        regime = classify_regime(spy_data, date)
        regime_returns[regime].append(returns.loc[date])

    regime_sharpe = {}
    for regime, rets in regime_returns.items():
        if len(rets) > 10:
            arr = np.array(rets)
            ann_ret = arr.mean() * 252
            ann_vol = arr.std() * np.sqrt(252)
            regime_sharpe[regime] = round(ann_ret / ann_vol, 3) if ann_vol > 0 else 0
        else:
            regime_sharpe[regime] = None

    # Regime gap test: |Sharpe_bull - Sharpe_bear| / max(|Sharpe_bull|, |Sharpe_bear|) < 0.50
    bull_s = regime_sharpe.get("bull")
    bear_s = regime_sharpe.get("bear")

    if bull_s is not None and bear_s is not None and max(abs(bull_s), abs(bear_s)) > 0:
        gap = abs(bull_s - bear_s) / max(abs(bull_s), abs(bear_s))
        regime_pass = gap < 0.50
    else:
        gap = None
        regime_pass = None

    return {
        "sharpe_per_regime": regime_sharpe,
        "regime_gap": round(gap, 3) if gap is not None else None,
        "regime_gap_pass": regime_pass,
        "n_days_per_regime": {r: len(v) for r, v in regime_returns.items()},
    }


# ─── SPY benchmark ──────────────────────────────────────────────────────────

def spy_benchmark(spy_data, start, end):
    """Compute SPY buy-and-hold metrics for comparison."""
    mask = (spy_data.index >= pd.Timestamp(start)) & (spy_data.index <= pd.Timestamp(end))
    subset = spy_data.loc[mask, "Close"]
    if len(subset) < 10:
        return {}

    returns = subset.pct_change().dropna()
    total_days = (subset.index[-1] - subset.index[0]).days
    total_return = subset.iloc[-1] / subset.iloc[0] - 1
    cagr = (1 + total_return) ** (365 / max(total_days, 1)) - 1

    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(252)
    sortino = ann_ret / downside_vol if downside_vol > 0 else 0

    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    max_dd = ((cumulative - peak) / peak).min()

    return {
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
    }


# ─── Walk-forward orchestrator ──────────────────────────────────────────────

def run_walk_forward(data):
    """Walk-forward: 2-year train, 1-year OOT, rolling from 2018-2026."""

    # Define windows
    windows = [
        ("2016-01-01", "2017-12-31", "2018-01-01", "2018-12-31"),
        ("2017-01-01", "2018-12-31", "2019-01-01", "2019-12-31"),
        ("2018-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
        ("2019-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
        ("2020-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
        ("2021-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
        ("2022-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
        ("2023-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    ]

    all_results = []
    all_equity = []
    all_trades = []
    all_regime = []

    for train_start, train_end, test_start, test_end in windows:
        print(f"\n{'='*60}")
        print(f"Window: Train {train_start} to {train_end} | OOT {test_start} to {test_end}")
        print(f"{'='*60}")

        bt = LEAPSPMCCBacktester(data, train_start, train_end, test_start, test_end)
        bt.run()

        metrics = bt.compute_metrics()
        if not metrics:
            print("  No metrics (insufficient data)")
            continue

        # Regime analysis
        spy_data = data.get(BENCHMARK)
        regime = regime_analysis(bt.daily_equity, spy_data) if spy_data is not None else {}

        # SPY benchmark for same period
        spy_bench = spy_benchmark(spy_data, test_start, test_end) if spy_data is not None else {}

        metrics["spy_benchmark"] = spy_bench
        metrics["regime_analysis"] = regime

        all_results.append(metrics)
        all_equity.extend(bt.daily_equity)
        all_trades.extend(bt.trade_log)
        all_regime.append(regime)

        # Print summary
        print(f"\n  LEAPS+PMCC: CAGR={metrics['cagr_pct']:.1f}%, Sharpe={metrics['sharpe']:.3f}, "
              f"Sortino={metrics['sortino']:.3f}, MaxDD={metrics['max_drawdown_pct']:.1f}%")
        print(f"  SPY B&H:    CAGR={spy_bench.get('cagr_pct', 'N/A')}%, "
              f"Sharpe={spy_bench.get('sharpe', 'N/A')}, MaxDD={spy_bench.get('max_drawdown_pct', 'N/A')}%")
        print(f"  Trades: {metrics['n_trades']}, WR: {metrics['win_rate_pct']:.1f}%, "
              f"PF: {metrics['profit_factor']:.2f}")
        print(f"  PMCC income rate (ann): {metrics['pmcc_income_rate_annualized_pct']:.1f}%")
        print(f"  Capital leverage: {metrics['avg_capital_leverage']:.1f}x")
        if regime.get("sharpe_per_regime"):
            print(f"  Regime Sharpe: {regime['sharpe_per_regime']}")
            print(f"  Regime gap: {regime.get('regime_gap', 'N/A')} "
                  f"({'PASS' if regime.get('regime_gap_pass') else 'FAIL'})")

    return all_results, all_equity, all_trades


# ─── Aggregate summary ──────────────────────────────────────────────────────

def compute_aggregate(all_results, all_equity, all_trades):
    """Compute aggregate metrics across all walk-forward windows."""
    if not all_results:
        return {}

    # Aggregate equity curve
    if all_equity:
        eq = pd.DataFrame(all_equity)
        eq.set_index("date", inplace=True)
        eq = eq.sort_index()

        returns = eq["equity"].pct_change().dropna()
        returns = returns.replace([np.inf, -np.inf], 0).fillna(0)

        total_days = (eq.index[-1] - eq.index[0]).days
        total_return = eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1
        cagr = (1 + total_return) ** (365 / max(total_days, 1)) - 1

        ann_ret = returns.mean() * 252
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

        downside = returns[returns < 0]
        downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-6
        sortino = ann_ret / downside_vol

        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        max_dd = ((cumulative - peak) / peak).min()
    else:
        cagr = sharpe = sortino = max_dd = 0

    # Per-window averages
    avg_sharpe = np.mean([r["sharpe"] for r in all_results])
    avg_sortino = np.mean([r["sortino"] for r in all_results])
    avg_cagr = np.mean([r["cagr_pct"] for r in all_results])
    avg_wr = np.mean([r["win_rate_pct"] for r in all_results])
    avg_pf = np.mean([r["profit_factor"] for r in all_results if r["profit_factor"] < 100])
    avg_pmcc = np.mean([r["pmcc_income_rate_annualized_pct"] for r in all_results])
    avg_leverage = np.mean([r["avg_capital_leverage"] for r in all_results if r["avg_capital_leverage"] > 0])

    # SPY averages
    spy_sharpes = [r["spy_benchmark"].get("sharpe", 0) for r in all_results if r.get("spy_benchmark")]
    spy_cagrs = [r["spy_benchmark"].get("cagr_pct", 0) for r in all_results if r.get("spy_benchmark")]

    # Regime results
    regime_gaps = [r["regime_analysis"].get("regime_gap") for r in all_results
                   if r.get("regime_analysis", {}).get("regime_gap") is not None]
    regime_passes = [r["regime_analysis"].get("regime_gap_pass") for r in all_results
                     if r.get("regime_analysis", {}).get("regime_gap_pass") is not None]

    agg = {
        "note": "ALL PRICING IS BS-MODELED. Real options will be priced differently. Directional research only.",
        "aggregate_metrics": {
            "cagr_pct": round(cagr * 100, 2),
            "sharpe": round(sharpe, 3),
            "sortino": round(sortino, 3),
            "max_drawdown_pct": round(max_dd * 100, 2),
        },
        "per_window_averages": {
            "avg_cagr_pct": round(avg_cagr, 2),
            "avg_sharpe": round(avg_sharpe, 3),
            "avg_sortino": round(avg_sortino, 3),
            "avg_win_rate_pct": round(avg_wr, 1),
            "avg_profit_factor": round(avg_pf, 2),
            "avg_pmcc_income_rate_ann_pct": round(avg_pmcc, 2),
            "avg_capital_leverage_x": round(avg_leverage, 2),
        },
        "spy_benchmark_averages": {
            "avg_cagr_pct": round(np.mean(spy_cagrs), 2) if spy_cagrs else None,
            "avg_sharpe": round(np.mean(spy_sharpes), 3) if spy_sharpes else None,
        },
        "capital_efficiency": {
            "description": "LEAPS provide leveraged stock exposure at ~15-25% of stock cost",
            "avg_leverage": round(avg_leverage, 2),
            "meaning": f"~{round(avg_leverage, 1)}x capital efficiency vs buying stock outright",
        },
        "regime_analysis": {
            "avg_regime_gap": round(np.mean(regime_gaps), 3) if regime_gaps else None,
            "windows_passing_regime_test": f"{sum(1 for p in regime_passes if p)}/{len(regime_passes)}" if regime_passes else "N/A",
        },
        "n_windows": len(all_results),
        "n_total_trades": len(all_trades),
        "per_window_results": all_results,
        "trade_log_sample": all_trades[:20],  # first 20 trades
    }

    return agg


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("LEAPS / PMCC BACKTESTER")
    print("All pricing is BS-MODELED (Black-Scholes approximation)")
    print("Real options will be priced differently — directional research only")
    print("=" * 70)

    # Load data
    data = load_data(UNIVERSE, start="2015-06-01", end="2026-07-11")

    if len(data) < 5:
        print("ERROR: insufficient data loaded")
        sys.exit(1)

    # Run walk-forward
    all_results, all_equity, all_trades = run_walk_forward(data)

    if not all_results:
        print("ERROR: no results produced")
        sys.exit(1)

    # Compute aggregate
    agg = compute_aggregate(all_results, all_equity, all_trades)

    # Print final summary
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS (BS-MODELED)")
    print("=" * 70)

    am = agg["aggregate_metrics"]
    pw = agg["per_window_averages"]
    spy = agg["spy_benchmark_averages"]

    print(f"\n  LEAPS+PMCC Strategy:")
    print(f"    CAGR:           {am['cagr_pct']:.2f}%")
    print(f"    Sharpe:         {am['sharpe']:.3f}")
    print(f"    Sortino:        {am['sortino']:.3f}")
    print(f"    Max Drawdown:   {am['max_drawdown_pct']:.2f}%")

    print(f"\n  Per-Window Averages:")
    print(f"    Avg CAGR:       {pw['avg_cagr_pct']:.2f}%")
    print(f"    Avg Sharpe:     {pw['avg_sharpe']:.3f}")
    print(f"    Avg Sortino:    {pw['avg_sortino']:.3f}")
    print(f"    Avg Win Rate:   {pw['avg_win_rate_pct']:.1f}%")
    print(f"    Avg PF:         {pw['avg_profit_factor']:.2f}")

    print(f"\n  SPY Buy-and-Hold Benchmark:")
    print(f"    Avg CAGR:       {spy.get('avg_cagr_pct', 'N/A')}%")
    print(f"    Avg Sharpe:     {spy.get('avg_sharpe', 'N/A')}")

    print(f"\n  PMCC Income:")
    print(f"    Ann income rate: {pw['avg_pmcc_income_rate_ann_pct']:.2f}% of LEAPS cost")

    print(f"\n  Capital Efficiency:")
    ce = agg["capital_efficiency"]
    print(f"    Avg leverage:   {ce['avg_leverage']:.1f}x vs stock ownership")
    print(f"    {ce['meaning']}")

    ra = agg["regime_analysis"]
    print(f"\n  Regime Analysis:")
    print(f"    Avg regime gap: {ra.get('avg_regime_gap', 'N/A')}")
    print(f"    Windows passing: {ra.get('windows_passing_regime_test', 'N/A')}")

    print(f"\n  Total trades: {agg['n_total_trades']} across {agg['n_windows']} windows")

    # Save results
    summary_path = os.path.join(OUTPUT_DIR, "leaps_pmcc_summary.json")
    with open(summary_path, "w") as f:
        # Convert non-serializable types
        json.dump(agg, f, indent=2, default=str)
    print(f"\n  Results saved to {summary_path}")

    trades_path = os.path.join(OUTPUT_DIR, "leaps_pmcc_trades.json")
    with open(trades_path, "w") as f:
        json.dump(all_trades, f, indent=2, default=str)
    print(f"  Trade log saved to {trades_path}")

    # Save equity curve
    if all_equity:
        eq_df = pd.DataFrame(all_equity)
        eq_path = os.path.join(OUTPUT_DIR, "leaps_pmcc_equity.csv")
        eq_df.to_csv(eq_path, index=False)
        print(f"  Equity curve saved to {eq_path}")

    print("\n  NOTE: All pricing is BS-modeled. Real options pricing will differ.")
    print("  This is directional research for evaluating LEAPS/PMCC as a growth strategy.")


if __name__ == "__main__":
    main()
