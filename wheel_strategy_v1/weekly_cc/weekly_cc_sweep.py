#!/usr/bin/env python3
"""
Weekly Covered Call Strategy Parameter Sweep.

Backtests a pure covered call strategy: buy 100 shares, sell weekly CCs,
and systematically test combinations of:
  - CC delta (0.15 to 0.50)
  - DTE (3, 5, 7, 10 days)
  - Early close threshold (50%, 65%, 75%, or none)
  - Roll strategy (roll_down, roll_out, let_expire)

When shares are called away, the strategy immediately rebuys and sells a
new CC — keeping continuous exposure.

Uses yfinance for historical stock data and Black-Scholes with IV-calibrated
estimates for option pricing.  If Alpaca API keys are present in .env, real
greeks/IV can be used for the *current* snapshot, but backtests always use
the BS model (no historical options tape needed).

Usage:
    python -m trading_agents.wheel_strategy.weekly_cc_sweep --tickers AAPL,TSLA,NVDA --days 365
    python -m trading_agents.wheel_strategy.weekly_cc_sweep --days 180 --capital 50000
    python -m trading_agents.wheel_strategy.weekly_cc_sweep --tickers AAPL --days 365 --workers 8
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

# ── Black-Scholes (self-contained so the script runs standalone on Jupiter) ──

_SQRT2PI = math.sqrt(2.0 * math.pi)


def _norm_cdf(x):
    """Standard normal CDF (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x):
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / _SQRT2PI


def bs_price(S, K, T, r, sigma, opt_type="call"):
    """Black-Scholes option price."""
    if T <= 0 or sigma <= 0:
        if opt_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if opt_type == "call":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def bs_delta(S, K, T, r, sigma, opt_type="call"):
    """Black-Scholes delta."""
    if T <= 0 or sigma <= 0:
        if opt_type == "call":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    if opt_type == "call":
        return _norm_cdf(d1)
    return _norm_cdf(d1) - 1.0


def find_strike_for_delta(S, T, r, sigma, target_delta, opt_type="call"):
    """Find the strike that yields approximately *target_delta* for a call."""
    if T <= 0 or sigma <= 0:
        return round(S)
    # For calls, delta decreases as strike increases.
    low_k = S * 0.70
    high_k = S * 1.50
    best_k = S
    best_diff = float("inf")
    # Coarse sweep
    for k in np.linspace(low_k, high_k, 300):
        d = bs_delta(S, k, T, r, sigma, opt_type)
        diff = abs(d - target_delta)
        if diff < best_diff:
            best_diff = diff
            best_k = k
    # Round to nearest 0.50 (common strike increment for most liquid names)
    best_k = round(best_k * 2) / 2.0
    return best_k


def estimate_iv(prices, idx, lookback=30, iv_premium=1.20):
    """Estimate IV from trailing realized vol * premium factor."""
    start = max(0, idx - lookback)
    window = prices[start: idx + 1]
    if len(window) < 5:
        return 0.25
    rets = np.diff(np.log(window))
    hv = float(np.std(rets)) * math.sqrt(252)
    return max(hv * iv_premium, 0.05)


# ── Roll strategies ──────────────────────────────────────────────────────────

ROLL_STRATEGIES = ["roll_down", "roll_out", "let_expire"]

# roll_down  — if ITM near expiry, buy back and sell a lower-strike CC at
#              2x the original premium (or the best available).
# roll_out   — if ITM near expiry, buy back current CC and sell same-strike
#              CC at the next expiry for a net credit.
# let_expire — do nothing; let shares be called away, then rebuy + sell new CC.


# ── Core backtest engine ─────────────────────────────────────────────────────

class WeeklyCCBacktester:
    """
    Backtest a pure covered-call strategy on one ticker.

    Lifecycle:
        1.  Buy 100 shares at market.
        2.  Sell 1 CC at *target_delta*, *target_dte* DTE.
        3.  Each trading day, check:
            a.  Early close: if CC has decayed >= *early_close_pct*, buy back
                for remaining value and immediately sell a new CC.
            b.  Roll check (only within *roll_dte_threshold* days of expiry):
                - roll_down: if ITM, buy back + sell lower strike for 2x prem.
                - roll_out:  if ITM, buy back + sell same strike next expiry.
                - let_expire: do nothing.
        4.  At expiry:
            - OTM: CC expires worthless, sell a new CC.
            - ITM: shares called away at strike, immediately rebuy at market
              and sell new CC.
        5.  Repeat until end of data.
    """

    def __init__(
        self,
        ticker,
        prices,          # np array of daily close prices
        dates,           # corresponding date strings
        capital=100_000,
        cc_delta=0.30,
        target_dte=7,
        early_close_pct=None,   # None = hold to expiry
        roll_strategy="let_expire",
        risk_free_rate=0.05,
        iv_premium=1.20,
        commission_per_contract=1.00,
        bid_ask_haircut=0.10,
        roll_dte_threshold=2,
    ):
        self.ticker = ticker
        self.prices = prices
        self.dates = dates
        self.capital = float(capital)
        self.initial_capital = float(capital)
        self.cc_delta = cc_delta
        self.target_dte = target_dte
        self.early_close_pct = early_close_pct
        self.roll_strategy = roll_strategy
        self.r = risk_free_rate
        self.iv_premium = iv_premium
        self.commission = commission_per_contract
        self.haircut = bid_ask_haircut
        self.roll_dte_threshold = roll_dte_threshold

        # State
        self.shares = 0
        self.cost_basis = 0.0
        self.cc_position = None  # dict or None

        # Counters
        self.total_premium = 0.0
        self.total_commissions = 0.0
        self.expirations_otm = 0
        self.calls_away = 0
        self.early_closes = 0
        self.rolls = 0
        self.num_ccs_sold = 0
        self.stock_appreciation = 0.0

        # Equity curve
        self.equity_curve = []

    # ── helpers ───────────────────────────────────────────────────────────

    def _iv(self, idx):
        return estimate_iv(self.prices, idx, iv_premium=self.iv_premium)

    def _find_expiry_idx(self, start_idx, dte):
        """Map calendar DTE to trading-day index."""
        trading_days = max(1, int(dte * 252 / 365))
        return min(start_idx + trading_days, len(self.prices) - 1)

    def _sell_cc(self, price, iv, idx):
        """Sell a covered call.  Returns the position dict."""
        T = self.target_dte / 365.0
        strike = find_strike_for_delta(price, T, self.r, iv, self.cc_delta, "call")
        prem_mid = bs_price(price, strike, T, self.r, iv, "call")
        prem = prem_mid * (1.0 - self.haircut)
        prem = max(prem, 0.01)

        self.capital += prem * 100.0
        self.total_premium += prem * 100.0
        self.total_commissions += self.commission
        self.capital -= self.commission
        self.num_ccs_sold += 1

        pos = {
            "strike": strike,
            "premium": prem,
            "entry_price": price,
            "entry_idx": idx,
            "expiry_idx": self._find_expiry_idx(idx, self.target_dte),
            "iv": iv,
        }
        self.cc_position = pos
        return pos

    def _buy_back_cc(self, price, iv, idx):
        """Buy back the open CC at current theoretical value.  Returns cost."""
        pos = self.cc_position
        if pos is None:
            return 0.0
        remaining_days = max(pos["expiry_idx"] - idx, 0)
        T_rem = max(remaining_days / 252.0, 1e-6)
        val_mid = bs_price(price, pos["strike"], T_rem, self.r, iv, "call")
        cost = val_mid * (1.0 + self.haircut)  # pay the ask
        self.capital -= cost * 100.0
        self.total_commissions += self.commission
        self.capital -= self.commission
        self.cc_position = None
        return cost

    def _buy_shares(self, price, idx):
        """Buy 100 shares at market."""
        cost = price * 100.0
        self.capital -= cost
        self.shares = 100
        self.cost_basis = price

    def _sell_shares(self, price):
        """Sell 100 shares at market (called away or explicit)."""
        proceeds = price * 100.0
        self.capital += proceeds
        gain = (price - self.cost_basis) * 100.0
        self.stock_appreciation += gain
        self.shares = 0
        self.cost_basis = 0.0
        return gain

    def _portfolio_value(self, price):
        return self.capital + self.shares * price

    # ── main loop ─────────────────────────────────────────────────────────

    def run(self):
        warmup = 45
        if len(self.prices) <= warmup:
            return None

        start_idx = warmup
        total_days = len(self.prices)
        start_price = float(self.prices[start_idx])

        # Buy shares to start
        self._buy_shares(start_price, start_idx)
        iv0 = self._iv(start_idx)
        self._sell_cc(start_price, iv0, start_idx)

        # Buy-and-hold benchmark
        bnh_shares = self.initial_capital / start_price

        for idx in range(start_idx + 1, total_days):
            price = float(self.prices[idx])
            iv = self._iv(idx)
            pos = self.cc_position

            # ── Early close check ─────────────────────────────────────
            if pos is not None and self.early_close_pct is not None:
                remaining_days = max(pos["expiry_idx"] - idx, 0)
                T_rem = max(remaining_days / 252.0, 1e-6)
                current_val = bs_price(price, pos["strike"], T_rem, self.r, iv, "call")
                prem_captured = pos["premium"] - current_val
                if pos["premium"] > 0 and prem_captured / pos["premium"] >= self.early_close_pct:
                    # Buy back cheap, sell new CC
                    self._buy_back_cc(price, iv, idx)
                    self._sell_cc(price, iv, idx)
                    self.early_closes += 1
                    pos = self.cc_position  # refresh

            # ── Expiry / roll check ───────────────────────────────────
            if pos is not None and idx >= pos["expiry_idx"]:
                expiry_price = price

                if expiry_price >= pos["strike"]:
                    # ITM — shares called away (or roll)
                    if self.roll_strategy == "roll_out" and idx < total_days - 1:
                        # Roll out: buy back, sell same strike next expiry
                        buyback_cost = self._buy_back_cc(price, iv, idx)
                        # Sell new CC at same strike
                        T_new = self.target_dte / 365.0
                        new_prem_mid = bs_price(price, pos["strike"], T_new, self.r, iv, "call")
                        new_prem = new_prem_mid * (1.0 - self.haircut)
                        new_prem = max(new_prem, 0.01)
                        net_credit = new_prem - buyback_cost
                        # Only roll if net credit
                        if net_credit > 0:
                            self.capital += new_prem * 100.0
                            self.total_premium += new_prem * 100.0
                            self.total_commissions += self.commission
                            self.capital -= self.commission
                            self.num_ccs_sold += 1
                            self.cc_position = {
                                "strike": pos["strike"],
                                "premium": new_prem,
                                "entry_price": price,
                                "entry_idx": idx,
                                "expiry_idx": self._find_expiry_idx(idx, self.target_dte),
                                "iv": iv,
                            }
                            self.rolls += 1
                        else:
                            # Can't roll for credit — let it go
                            self._sell_shares(pos["strike"])
                            self.calls_away += 1
                            # Immediately rebuy + sell new CC
                            self._buy_shares(price, idx)
                            self._sell_cc(price, iv, idx)

                    elif self.roll_strategy == "roll_down" and idx < total_days - 1:
                        # Roll down: buy back, sell lower strike (at 2x prem target)
                        buyback_cost = self._buy_back_cc(price, iv, idx)
                        # Try to find a lower strike that gives 2x original prem
                        T_new = self.target_dte / 365.0
                        # Use a slightly higher delta to get more premium
                        new_delta = min(self.cc_delta + 0.10, 0.60)
                        new_strike = find_strike_for_delta(price, T_new, self.r, iv, new_delta, "call")
                        new_prem_mid = bs_price(price, new_strike, T_new, self.r, iv, "call")
                        new_prem = new_prem_mid * (1.0 - self.haircut)
                        new_prem = max(new_prem, 0.01)
                        net_credit = new_prem - buyback_cost
                        if net_credit > 0:
                            self.capital += new_prem * 100.0
                            self.total_premium += new_prem * 100.0
                            self.total_commissions += self.commission
                            self.capital -= self.commission
                            self.num_ccs_sold += 1
                            self.cc_position = {
                                "strike": new_strike,
                                "premium": new_prem,
                                "entry_price": price,
                                "entry_idx": idx,
                                "expiry_idx": self._find_expiry_idx(idx, self.target_dte),
                                "iv": iv,
                            }
                            self.rolls += 1
                        else:
                            self._sell_shares(pos["strike"])
                            self.calls_away += 1
                            self._buy_shares(price, idx)
                            self._sell_cc(price, iv, idx)

                    else:
                        # let_expire (default)
                        self._sell_shares(pos["strike"])
                        self.calls_away += 1
                        # Immediately rebuy + sell new CC
                        if idx < total_days - 1:
                            self._buy_shares(price, idx)
                            self._sell_cc(price, iv, idx)

                else:
                    # OTM — CC expires worthless, sell new CC
                    self.cc_position = None
                    self.expirations_otm += 1
                    if idx < total_days - 1:
                        self._sell_cc(price, iv, idx)

            # Record equity
            pv = self._portfolio_value(price)
            self.equity_curve.append(pv)

        # ── Finalize ──────────────────────────────────────────────────
        final_price = float(self.prices[-1])
        final_value = self._portfolio_value(final_price)
        bnh_final = bnh_shares * final_price

        # Metrics
        eq = np.array(self.equity_curve, dtype=np.float64)
        if len(eq) > 1:
            daily_ret = np.diff(eq) / eq[:-1]
            daily_ret = daily_ret[np.isfinite(daily_ret)]
            if len(daily_ret) > 1 and np.std(daily_ret) > 0:
                sharpe = float(np.mean(daily_ret) / np.std(daily_ret) * np.sqrt(252))
            else:
                sharpe = 0.0
            peak = np.maximum.accumulate(eq)
            dd = (eq - peak) / np.where(peak > 0, peak, 1.0)
            max_dd = float(np.min(dd))
            # Weekly P&L for win rate
            weekly_pnl = []
            step = 5  # ~1 trading week
            for w in range(0, len(eq) - step, step):
                weekly_pnl.append(eq[w + step] - eq[w])
            win_weeks = sum(1 for p in weekly_pnl if p > 0)
            total_weeks = max(len(weekly_pnl), 1)
            win_rate = win_weeks / total_weeks * 100.0
        else:
            sharpe = 0.0
            max_dd = 0.0
            win_rate = 0.0
            total_weeks = 0

        total_return = (final_value - self.initial_capital) / self.initial_capital
        bnh_return = (bnh_final - self.initial_capital) / self.initial_capital
        trading_days = total_days - warmup
        years = trading_days / 252.0
        premium_yield_weekly = (
            (self.total_premium / self.initial_capital) / max(total_weeks, 1) * 100.0
        )

        total_cycles = self.expirations_otm + self.calls_away
        assignment_freq = self.calls_away / max(total_cycles, 1) * 100.0

        return {
            "ticker": self.ticker,
            "cc_delta": self.cc_delta,
            "target_dte": self.target_dte,
            "early_close_pct": self.early_close_pct,
            "roll_strategy": self.roll_strategy,
            "total_return_pct": round(total_return * 100, 2),
            "bnh_return_pct": round(bnh_return * 100, 2),
            "cc_vs_bnh_pct": round((total_return - bnh_return) * 100, 2),
            "total_premium": round(self.total_premium, 2),
            "premium_yield_weekly_pct": round(premium_yield_weekly, 3),
            "stock_appreciation": round(self.stock_appreciation, 2),
            "total_commissions": round(self.total_commissions, 2),
            "sharpe": round(sharpe, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "win_rate_pct": round(win_rate, 1),
            "total_cc_cycles": total_cycles,
            "expirations_otm": self.expirations_otm,
            "calls_away": self.calls_away,
            "early_closes": self.early_closes,
            "rolls": self.rolls,
            "num_ccs_sold": self.num_ccs_sold,
            "assignment_freq_pct": round(assignment_freq, 1),
            "final_value": round(final_value, 2),
            "trading_days": trading_days,
            "total_weeks": total_weeks,
        }


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_stock_data(ticker, days=365):
    """Fetch historical daily closes via yfinance. Returns (prices, dates)."""
    if not HAS_YFINANCE:
        raise ImportError("yfinance required.  pip install yfinance")
    import datetime as dt
    end = dt.date.today()
    start = end - dt.timedelta(days=int(days * 1.5))  # extra for IV warmup
    df = yf.download(ticker, start=str(start), end=str(end), progress=False,
                     auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    # Flatten MultiIndex if present
    if hasattr(df.columns, 'levels') and len(df.columns.levels) > 1:
        df.columns = df.columns.get_level_values(0)
    prices = df["Close"].values.flatten().astype(np.float64)
    dates = [str(d.date()) if hasattr(d, 'date') else str(d)[:10] for d in df.index]
    return prices, dates


# ── Sweep runner (single ticker, one config) ─────────────────────────────────

def _run_single_config(args):
    """Worker function for parallel execution.  Receives a tuple."""
    ticker, prices, dates, capital, cc_delta, dte, ec_pct, roll_strat = args
    try:
        bt = WeeklyCCBacktester(
            ticker=ticker,
            prices=prices,
            dates=dates,
            capital=capital,
            cc_delta=cc_delta,
            target_dte=dte,
            early_close_pct=ec_pct,
            roll_strategy=roll_strat,
        )
        result = bt.run()
        return result
    except Exception as e:
        return {
            "ticker": ticker,
            "cc_delta": cc_delta,
            "target_dte": dte,
            "early_close_pct": ec_pct,
            "roll_strategy": roll_strat,
            "error": str(e),
        }


# ── Main sweep ────────────────────────────────────────────────────────────────

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "AMZN", "GOOGL",
    "PLTR", "HOOD", "SOFI", "COIN", "MARA", "NIO", "F",
    "BAC", "DAL", "AAL", "C", "JPM",
]

CC_DELTAS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
DTES = [3, 5, 7, 10]
EARLY_CLOSE_PCTS = [0.50, 0.65, 0.75, None]


def run_sweep(tickers=None, days=365, capital=100_000, workers=None):
    """
    Run full parameter sweep across tickers x deltas x DTEs x early_close x roll.

    Returns list of result dicts sorted by Sharpe descending.
    """
    tickers = tickers or DEFAULT_TICKERS
    configs = list(product(CC_DELTAS, DTES, EARLY_CLOSE_PCTS, ROLL_STRATEGIES))
    total_configs = len(configs)
    total_runs = len(tickers) * total_configs

    print("=" * 90)
    print("WEEKLY COVERED CALL PARAMETER SWEEP")
    print("=" * 90)
    print(f"Tickers:      {len(tickers)}  ({', '.join(tickers[:8])}{'...' if len(tickers) > 8 else ''})")
    print(f"CC deltas:    {CC_DELTAS}")
    print(f"DTEs:         {DTES}")
    print(f"Early close:  {EARLY_CLOSE_PCTS}")
    print(f"Roll strats:  {ROLL_STRATEGIES}")
    print(f"Configs/tick: {total_configs}")
    print(f"Total runs:   {total_runs}")
    print(f"Capital:      ${capital:,.0f}")
    print(f"Lookback:     {days} days")
    print(f"Workers:      {workers or 'sequential'}")
    print("=" * 90)
    print()

    # ── Fetch data for all tickers ────────────────────────────────────
    ticker_data = {}
    for t in tickers:
        try:
            prices, dates = fetch_stock_data(t, days)
            ticker_data[t] = (prices, dates)
            print(f"  {t:6s}: {len(prices)} trading days fetched  "
                  f"(${prices[-1]:.2f} last close)")
        except Exception as e:
            print(f"  {t:6s}: FAILED — {e}")

    if not ticker_data:
        print("\nNo data fetched. Exiting.")
        return []

    print(f"\nData ready for {len(ticker_data)} tickers.  Starting sweep...\n")

    # ── Build work items ──────────────────────────────────────────────
    work = []
    for t, (prices, dates) in ticker_data.items():
        for cc_delta, dte, ec_pct, roll_strat in configs:
            work.append((t, prices, dates, capital, cc_delta, dte, ec_pct, roll_strat))

    # ── Execute ───────────────────────────────────────────────────────
    results = []
    t0 = time.time()

    if workers and workers > 1:
        # Parallel execution
        completed = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_single_config, w): w for w in work}
            for fut in as_completed(futures):
                r = fut.result()
                if r is not None:
                    results.append(r)
                completed += 1
                if completed % 200 == 0 or completed == len(work):
                    elapsed = time.time() - t0
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (len(work) - completed) / rate if rate > 0 else 0
                    print(f"  Progress: {completed}/{len(work)}  "
                          f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")
    else:
        # Sequential (simpler, no pickle issues)
        for i, w in enumerate(work):
            r = _run_single_config(w)
            if r is not None:
                results.append(r)
            if (i + 1) % 100 == 0 or i + 1 == len(work):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(work) - i - 1) / rate if rate > 0 else 0
                print(f"  Progress: {i + 1}/{len(work)}  "
                      f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    elapsed_total = time.time() - t0
    errors = [r for r in results if "error" in r]
    valid = [r for r in results if "error" not in r]

    print(f"\nSweep complete: {len(valid)} valid, {len(errors)} errors "
          f"in {elapsed_total:.1f}s")

    # ── Aggregate by config (average across tickers) ──────────────────
    config_agg = {}
    for r in valid:
        key = (r["cc_delta"], r["target_dte"], r["early_close_pct"], r["roll_strategy"])
        if key not in config_agg:
            config_agg[key] = []
        config_agg[key].append(r)

    aggregated = []
    for key, runs in config_agg.items():
        cc_delta, dte, ec_pct, roll_strat = key
        n = len(runs)
        avg = lambda field: sum(r[field] for r in runs) / n
        aggregated.append({
            "cc_delta": cc_delta,
            "target_dte": dte,
            "early_close_pct": ec_pct,
            "roll_strategy": roll_strat,
            "n_tickers": n,
            "avg_total_return_pct": round(avg("total_return_pct"), 2),
            "avg_bnh_return_pct": round(avg("bnh_return_pct"), 2),
            "avg_cc_vs_bnh_pct": round(avg("cc_vs_bnh_pct"), 2),
            "avg_premium_yield_weekly_pct": round(avg("premium_yield_weekly_pct"), 3),
            "avg_sharpe": round(avg("sharpe"), 2),
            "avg_max_drawdown_pct": round(avg("max_drawdown_pct"), 2),
            "avg_win_rate_pct": round(avg("win_rate_pct"), 1),
            "avg_assignment_freq_pct": round(avg("assignment_freq_pct"), 1),
            "avg_total_premium": round(avg("total_premium"), 2),
            "avg_early_closes": round(avg("early_closes"), 1),
            "avg_rolls": round(avg("rolls"), 1),
        })

    # Sort by Sharpe descending
    aggregated.sort(key=lambda x: x["avg_sharpe"], reverse=True)

    return {
        "sweep_params": {
            "tickers": list(ticker_data.keys()),
            "cc_deltas": CC_DELTAS,
            "dtes": DTES,
            "early_close_pcts": EARLY_CLOSE_PCTS,
            "roll_strategies": ROLL_STRATEGIES,
            "capital": capital,
            "days": days,
            "total_configs": total_configs,
            "total_runs": len(work),
        },
        "aggregated": aggregated,
        "per_ticker": valid,
        "errors": errors,
        "elapsed_seconds": round(elapsed_total, 1),
    }


# ── Summary formatting ────────────────────────────────────────────────────────

def format_summary(sweep_result, top_n=30):
    """Print a ranked summary table of the best configs."""
    agg = sweep_result["aggregated"]
    params = sweep_result["sweep_params"]

    lines = [
        "",
        "=" * 120,
        "WEEKLY COVERED CALL SWEEP — RESULTS (sorted by avg Sharpe)",
        "=" * 120,
        f"Tickers: {', '.join(params['tickers'])}",
        f"Period: {params['days']} days | Capital: ${params['capital']:,.0f} | "
        f"Total configs: {params['total_configs']}",
        "",
        f"{'Rank':>4s} {'Delta':>6s} {'DTE':>4s} {'EClose':>7s} {'Roll':>12s} "
        f"{'Return%':>8s} {'B&H%':>7s} {'vs B&H':>7s} {'Sharpe':>7s} "
        f"{'MaxDD%':>7s} {'WinR%':>6s} {'Assign%':>8s} {'WkYld%':>7s} "
        f"{'Premium':>9s} {'#Tick':>5s}",
        "-" * 120,
    ]

    for i, row in enumerate(agg[:top_n], 1):
        ec_str = f"{row['early_close_pct']*100:.0f}%" if row['early_close_pct'] else "none"
        lines.append(
            f"{i:>4d} "
            f"{row['cc_delta']:>6.2f} "
            f"{row['target_dte']:>4d} "
            f"{ec_str:>7s} "
            f"{row['roll_strategy']:>12s} "
            f"{row['avg_total_return_pct']:>+7.1f}% "
            f"{row['avg_bnh_return_pct']:>+6.1f}% "
            f"{row['avg_cc_vs_bnh_pct']:>+6.1f}% "
            f"{row['avg_sharpe']:>7.2f} "
            f"{row['avg_max_drawdown_pct']:>6.1f}% "
            f"{row['avg_win_rate_pct']:>5.1f}% "
            f"{row['avg_assignment_freq_pct']:>7.1f}% "
            f"{row['avg_premium_yield_weekly_pct']:>6.3f}% "
            f"${row['avg_total_premium']:>7,.0f} "
            f"{row['n_tickers']:>5d}"
        )

    lines.append("-" * 120)

    # Bottom 5 (worst)
    if len(agg) > top_n:
        lines.append("")
        lines.append("WORST 5:")
        for row in agg[-5:]:
            ec_str = f"{row['early_close_pct']*100:.0f}%" if row['early_close_pct'] else "none"
            lines.append(
                f"     "
                f"{row['cc_delta']:>6.2f} "
                f"{row['target_dte']:>4d} "
                f"{ec_str:>7s} "
                f"{row['roll_strategy']:>12s} "
                f"{row['avg_total_return_pct']:>+7.1f}% "
                f"{row['avg_bnh_return_pct']:>+6.1f}% "
                f"{row['avg_cc_vs_bnh_pct']:>+6.1f}% "
                f"{row['avg_sharpe']:>7.2f} "
                f"{row['avg_max_drawdown_pct']:>6.1f}% "
                f"{row['avg_win_rate_pct']:>5.1f}% "
                f"{row['avg_assignment_freq_pct']:>7.1f}% "
                f"{row['avg_premium_yield_weekly_pct']:>6.3f}% "
                f"${row['avg_total_premium']:>7,.0f} "
                f"{row['n_tickers']:>5d}"
            )

    # Best per dimension
    lines.append("")
    lines.append("=" * 80)
    lines.append("BEST CONFIG PER DIMENSION")
    lines.append("=" * 80)

    # Best by delta
    for delta in CC_DELTAS:
        subset = [r for r in agg if r["cc_delta"] == delta]
        if subset:
            best = subset[0]  # already sorted by Sharpe
            ec_str = f"{best['early_close_pct']*100:.0f}%" if best['early_close_pct'] else "none"
            lines.append(
                f"  Delta {delta:.2f}: Sharpe={best['avg_sharpe']:+.2f}  "
                f"Ret={best['avg_total_return_pct']:+.1f}%  "
                f"DTE={best['target_dte']}  EC={ec_str}  Roll={best['roll_strategy']}"
            )

    lines.append("")
    for dte in DTES:
        subset = [r for r in agg if r["target_dte"] == dte]
        if subset:
            best = subset[0]
            ec_str = f"{best['early_close_pct']*100:.0f}%" if best['early_close_pct'] else "none"
            lines.append(
                f"  DTE {dte:2d}:    Sharpe={best['avg_sharpe']:+.2f}  "
                f"Ret={best['avg_total_return_pct']:+.1f}%  "
                f"Delta={best['cc_delta']:.2f}  EC={ec_str}  Roll={best['roll_strategy']}"
            )

    lines.append("")
    for roll in ROLL_STRATEGIES:
        subset = [r for r in agg if r["roll_strategy"] == roll]
        if subset:
            best = subset[0]
            ec_str = f"{best['early_close_pct']*100:.0f}%" if best['early_close_pct'] else "none"
            lines.append(
                f"  Roll={roll:12s}: Sharpe={best['avg_sharpe']:+.2f}  "
                f"Ret={best['avg_total_return_pct']:+.1f}%  "
                f"Delta={best['cc_delta']:.2f}  DTE={best['target_dte']}  EC={ec_str}"
            )

    # Per-ticker best
    lines.append("")
    lines.append("=" * 80)
    lines.append("BEST CONFIG PER TICKER (top Sharpe)")
    lines.append("=" * 80)

    per_ticker = sweep_result.get("per_ticker", [])
    ticker_best = {}
    for r in per_ticker:
        t = r["ticker"]
        if t not in ticker_best or r["sharpe"] > ticker_best[t]["sharpe"]:
            ticker_best[t] = r

    for t in sorted(ticker_best.keys()):
        r = ticker_best[t]
        ec_str = f"{r['early_close_pct']*100:.0f}%" if r['early_close_pct'] else "none"
        lines.append(
            f"  {t:6s}: Sharpe={r['sharpe']:+.2f}  "
            f"Ret={r['total_return_pct']:+.1f}%  "
            f"Delta={r['cc_delta']:.2f}  DTE={r['target_dte']}  "
            f"EC={ec_str}  Roll={r['roll_strategy']}  "
            f"Prem=${r['total_premium']:,.0f}  Assign={r['assignment_freq_pct']:.0f}%"
        )

    lines.append("")
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Weekly Covered Call Parameter Sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m trading_agents.wheel_strategy.weekly_cc_sweep --tickers AAPL,TSLA,NVDA --days 365
  python -m trading_agents.wheel_strategy.weekly_cc_sweep --days 180 --capital 50000
  python -m trading_agents.wheel_strategy.weekly_cc_sweep --tickers AAPL --days 730 --workers 8
  python -m trading_agents.wheel_strategy.weekly_cc_sweep --top 50
        """,
    )
    parser.add_argument("--tickers", type=str, default=None,
                        help="Comma-separated tickers (default: 20-stock universe)")
    parser.add_argument("--days", type=int, default=365,
                        help="Lookback period in calendar days (default: 365)")
    parser.add_argument("--capital", type=float, default=100_000,
                        help="Starting capital per ticker (default: $100,000)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers (default: sequential)")
    parser.add_argument("--top", type=int, default=30,
                        help="Number of top configs to display (default: 30)")
    parser.add_argument("--json-only", action="store_true",
                        help="Output raw JSON, skip summary table")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: auto-generated in data/)")
    args = parser.parse_args()

    tickers = args.tickers.split(",") if args.tickers else None

    result = run_sweep(
        tickers=tickers,
        days=args.days,
        capital=args.capital,
        workers=args.workers,
    )

    if not result:
        print("Sweep returned no results.")
        sys.exit(1)

    # Print summary
    if not args.json_only:
        print(format_summary(result, top_n=args.top))

    # Save JSON
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        out_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = data_dir / f"weekly_cc_sweep_{ts}.json"

    # Strip per-ticker detail for the saved file to keep size reasonable
    save_data = {
        "sweep_params": result["sweep_params"],
        "aggregated": result["aggregated"],
        "per_ticker_best": {},
        "errors": result["errors"],
        "elapsed_seconds": result["elapsed_seconds"],
    }
    # Keep only best config per ticker
    per_ticker = result.get("per_ticker", [])
    for r in per_ticker:
        t = r.get("ticker", "?")
        if t not in save_data["per_ticker_best"] or r.get("sharpe", 0) > save_data["per_ticker_best"][t].get("sharpe", 0):
            save_data["per_ticker_best"][t] = r

    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\nResults saved to: {out_path}")
    print(f"Aggregated configs: {len(result['aggregated'])}")
    print(f"Valid per-ticker runs: {len(result.get('per_ticker', []))}")


if __name__ == "__main__":
    main()
