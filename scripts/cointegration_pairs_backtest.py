#!/usr/bin/env python3
"""
Cointegration Pairs Mean-Reversion Strategy Backtester
======================================================
Identifies cointegrated stock pairs and trades spread mean reversion.
Unlike simple correlation, cointegration means two stocks share a long-run
equilibrium — when they diverge, they tend to converge back.

Pair selection: rolling 120-day Engle-Granger cointegration test on all
pairs in the growth stock universe. Re-test monthly to update pairs.

Since we can't short on Robinhood, only the LONG leg of each pair trade
is taken.

Variants:
  A: Top 3 pairs, z>2 entry, z=0 exit, long leg only
  B: Top 5 pairs, z>1.5 entry, z=0.5 exit
  C: Same as A but use OPTIONS (calls on long leg, BS pricing)
  D: Regime-filtered (SPY>200SMA), z>2 entry
  E: Dynamic — highest |z| pair each week
  F: Sector-constrained — only within-sector pairs

OOT: Jan 2022 - Jul 2026
Capital: $645
"""

import json
import logging
import os
import sys
import warnings
from datetime import datetime, timedelta
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/cointegration_pairs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUTPUT_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)
RESULTS_PATH = Path("/home/jupiter/Lvl3Quant/data/cointegration_pairs_results.json")

# Universe by sector
SECTORS = {
    "tech_mega":    ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"],
    "cloud_saas":   ["CRM", "SNOW", "DDOG", "NET", "CRWD"],
    "fintech":      ["SQ", "COIN", "SOFI", "HOOD"],
    "consumer":     ["NFLX", "UBER", "ABNB", "RBLX"],
    "semiconductor":["AMD", "MU", "AVGO", "ARM", "SMCI"],
    "ev_growth":    ["TSLA", "RIVN", "PLTR", "SHOP"],
}
ALL_TICKERS = sorted(set(t for group in SECTORS.values() for t in group))
TICKER_TO_SECTOR = {}
for sec, tickers in SECTORS.items():
    for t in tickers:
        TICKER_TO_SECTOR[t] = sec

# Timing
DATA_START = "2020-06-01"
OOT_START = "2022-01-01"
OOT_END = "2026-07-29"
STARTING_CAPITAL = 645.0

# Cointegration parameters
COINT_LOOKBACK = 120          # days for cointegration test
COINT_RETEST_DAYS = 21        # re-test monthly
SPREAD_LOOKBACK = 60          # rolling z-score window

# Trading parameters
MAX_CONCURRENT = 3
SLIPPAGE_PCT = 0.0002         # 0.02% slippage
TIME_STOP_DAYS = 20           # max holding period

# Options parameters (variant C)
OPTION_DTE = 7
OPTION_DELTA = 0.4
BS_HAIRCUT = 0.30             # 30% haircut on BS price
OPTION_COMMISSION = 0.65      # per contract
OPTION_SPREAD_PCT = 0.05      # 5% bid-ask spread cost
OPTION_MULTIPLIER = 100

# Validation gates
VALIDATION_GATES = {
    "min_sharpe": 0.5,
    "perm_p_max": 0.05,
    "perm_iterations": 1000,
    "regime_gap_max": 0.5,
    "max_drawdown_floor": -0.50,
    "min_trades": 20,
}


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_prices() -> pd.DataFrame:
    """Download daily close prices for all tickers + SPY + ^VIX."""
    cache_file = CACHE_DIR / "prices_cache.parquet"
    if cache_file.exists():
        mod_age = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).total_seconds()
        if mod_age < 3600 * 12:
            log.info("Loading cached prices")
            return pd.read_parquet(cache_file)

    tickers = ALL_TICKERS + ["SPY", "^VIX"]
    log.info(f"Downloading prices for {len(tickers)} tickers...")
    data = yf.download(tickers, start=DATA_START, end=OOT_END, auto_adjust=True, progress=False)

    # Handle multi-level columns
    if isinstance(data.columns, pd.MultiIndex):
        closes = data["Close"]
    else:
        closes = data

    closes = closes.ffill().dropna(how="all")
    closes.to_parquet(cache_file)
    log.info(f"Prices loaded: {closes.shape[0]} days x {closes.shape[1]} tickers")
    return closes


# ─── Cointegration Testing ───────────────────────────────────────────────────

def find_cointegrated_pairs(
    prices: pd.DataFrame,
    date: pd.Timestamp,
    lookback: int = COINT_LOOKBACK,
    allowed_pairs: list = None,
    top_n: int = 5,
):
    """
    Test all pairs for cointegration using Engle-Granger test.
    Returns list of (ticker_a, ticker_b, p_value, beta) sorted by p-value.
    """
    # Get lookback window
    idx = prices.index.get_loc(date)
    if idx < lookback:
        return []
    window = prices.iloc[idx - lookback : idx]

    candidates = allowed_pairs if allowed_pairs else list(combinations(ALL_TICKERS, 2))

    results = []
    for a, b in candidates:
        if a not in window.columns or b not in window.columns:
            continue
        sa = window[a].dropna()
        sb = window[b].dropna()
        common_idx = sa.index.intersection(sb.index)
        if len(common_idx) < lookback * 0.8:
            continue
        sa = sa.loc[common_idx]
        sb = sb.loc[common_idx]

        try:
            # Log prices for cointegration
            log_a = np.log(sa)
            log_b = np.log(sb)
            score, pvalue, _ = coint(log_a, log_b)

            # OLS for beta (hedge ratio)
            X = add_constant(log_b.values)
            model = OLS(log_a.values, X).fit()
            beta = model.params[1]

            results.append((a, b, pvalue, beta))
        except Exception:
            continue

    results.sort(key=lambda x: x[2])
    return results[:top_n]


def compute_spread_zscore(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    beta: float,
    date: pd.Timestamp,
    lookback: int = SPREAD_LOOKBACK,
) -> float:
    """Compute z-score of the log spread on a given date."""
    idx = prices.index.get_loc(date)
    if idx < lookback:
        return 0.0
    window = prices.iloc[max(0, idx - lookback) : idx + 1]

    if ticker_a not in window.columns or ticker_b not in window.columns:
        return 0.0

    log_a = np.log(window[ticker_a].dropna())
    log_b = np.log(window[ticker_b].dropna())
    common = log_a.index.intersection(log_b.index)
    if len(common) < lookback * 0.5:
        return 0.0

    spread = log_a.loc[common] - beta * log_b.loc[common]
    mu = spread.iloc[:-1].mean()
    sigma = spread.iloc[:-1].std()
    if sigma < 1e-8:
        return 0.0
    return (spread.iloc[-1] - mu) / sigma


# ─── Black-Scholes for Options (Variant C) ───────────────────────────────────

def bs_call_price(S, K, T, r=0.05, sigma=0.30):
    """Black-Scholes call price."""
    if T <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)


def bs_delta(S, K, T, r=0.05, sigma=0.30):
    """Black-Scholes call delta."""
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return stats.norm.cdf(d1)


def find_strike_for_delta(S, target_delta, T, r=0.05, sigma=0.30):
    """Find strike that gives approximately target_delta."""
    lo, hi = S * 0.7, S * 1.3
    for _ in range(50):
        mid = (lo + hi) / 2
        d = bs_delta(S, mid, T, r, sigma)
        if d > target_delta:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ─── Backtester ──────────────────────────────────────────────────────────────

class Position:
    def __init__(self, ticker, entry_price, entry_date, shares, pair_info, is_option=False, option_cost=0):
        self.ticker = ticker
        self.entry_price = entry_price
        self.entry_date = entry_date
        self.shares = shares
        self.pair_info = pair_info  # (ticker_a, ticker_b, beta, z_at_entry)
        self.is_option = is_option
        self.option_cost = option_cost
        self.days_held = 0


def run_variant(
    prices: pd.DataFrame,
    variant: str,
    spy_sma200: pd.Series = None,
    vix: pd.Series = None,
) -> dict:
    """Run a single variant backtest."""

    # Variant-specific parameters
    config = {
        "A": {"top_n": 3, "z_entry": 2.0, "z_exit": 0.0, "use_options": False,
               "regime_filter": False, "dynamic": False, "sector_only": False},
        "B": {"top_n": 5, "z_entry": 1.5, "z_exit": 0.5, "use_options": False,
               "regime_filter": False, "dynamic": False, "sector_only": False},
        "C": {"top_n": 3, "z_entry": 2.0, "z_exit": 0.0, "use_options": True,
               "regime_filter": False, "dynamic": False, "sector_only": False},
        "D": {"top_n": 3, "z_entry": 2.0, "z_exit": 0.0, "use_options": False,
               "regime_filter": True, "dynamic": False, "sector_only": False},
        "E": {"top_n": 10, "z_entry": 2.0, "z_exit": 0.0, "use_options": False,
               "regime_filter": False, "dynamic": True, "sector_only": False},
        "F": {"top_n": 5, "z_entry": 2.0, "z_exit": 0.0, "use_options": False,
               "regime_filter": False, "dynamic": False, "sector_only": True},
    }[variant]

    log.info(f"Running Variant {variant}: {config}")

    # Build allowed pairs for sector-constrained
    allowed_pairs = None
    if config["sector_only"]:
        allowed_pairs = []
        for sec, tickers in SECTORS.items():
            allowed_pairs.extend(list(combinations(tickers, 2)))

    # OOT dates
    oot_start = pd.Timestamp(OOT_START)
    oot_dates = prices.loc[oot_start:].index

    capital = STARTING_CAPITAL
    positions = []
    trades = []
    equity_curve = []
    active_pairs = []
    last_coint_test = None
    last_dynamic_pick = None

    for i, date in enumerate(oot_dates):
        # Monthly cointegration re-test
        if last_coint_test is None or (date - last_coint_test).days >= COINT_RETEST_DAYS:
            active_pairs = find_cointegrated_pairs(
                prices, date, COINT_LOOKBACK, allowed_pairs, config["top_n"]
            )
            last_coint_test = date
            if active_pairs:
                pair_strs = [f"{a}/{b} (p={p:.4f})" for a, b, p, _ in active_pairs]
                log.debug(f"  {date.date()}: coint pairs = {pair_strs}")

        # Regime filter (variant D)
        if config["regime_filter"] and spy_sma200 is not None:
            if date in spy_sma200.index and not spy_sma200.loc[date]:
                # Bear market — skip new entries, but still manage exits
                pass_entry = True
            else:
                pass_entry = False
        else:
            pass_entry = False

        # Mark-to-market positions
        unrealized = 0.0
        for pos in positions:
            if date in prices.index and pos.ticker in prices.columns:
                curr_price = prices.loc[date, pos.ticker]
                if not pos.is_option:
                    unrealized += pos.shares * (curr_price - pos.entry_price)
                else:
                    # Option P&L is limited — we'll handle at exit
                    pass

        # Check exits
        positions_to_close = []
        for j, pos in enumerate(positions):
            pos.days_held += 1
            a, b, beta, z_at_entry = pos.pair_info

            # Current z-score
            z = compute_spread_zscore(prices, a, b, beta, date)

            should_exit = False
            exit_reason = ""

            # Mean reversion exit
            if z_at_entry > 0 and z <= config["z_exit"]:
                should_exit = True
                exit_reason = "mean_reversion"
            elif z_at_entry < 0 and z >= -config["z_exit"]:
                should_exit = True
                exit_reason = "mean_reversion"

            # Stop loss — divergence continues
            if abs(z) > 3.5:
                should_exit = True
                exit_reason = "stop_loss"

            # Time stop
            if pos.days_held >= TIME_STOP_DAYS:
                should_exit = True
                exit_reason = "time_stop"

            if should_exit:
                if date in prices.index and pos.ticker in prices.columns:
                    exit_price = prices.loc[date, pos.ticker]
                    if pos.is_option:
                        # Option exit value
                        S = exit_price
                        K = pos.entry_price  # strike stored as entry_price for options
                        T_remain = max(0, (OPTION_DTE - pos.days_held)) / 252
                        exit_val = bs_call_price(S, K, T_remain)
                        exit_val *= (1 - BS_HAIRCUT)  # haircut on exit too
                        exit_val *= (1 - OPTION_SPREAD_PCT)  # spread cost
                        pnl = (exit_val * OPTION_MULTIPLIER * pos.shares) - pos.option_cost
                    else:
                        slippage = exit_price * SLIPPAGE_PCT
                        pnl = pos.shares * (exit_price - pos.entry_price - slippage)

                    capital += pnl
                    trades.append({
                        "entry_date": pos.entry_date.strftime("%Y-%m-%d"),
                        "exit_date": date.strftime("%Y-%m-%d"),
                        "ticker": pos.ticker,
                        "pair": f"{a}/{b}",
                        "z_entry": z_at_entry,
                        "z_exit": z,
                        "pnl": round(pnl, 2),
                        "return_pct": round(pnl / max(pos.option_cost if pos.is_option else (pos.shares * pos.entry_price), 1) * 100, 2),
                        "days_held": pos.days_held,
                        "exit_reason": exit_reason,
                        "is_option": pos.is_option,
                    })
                    positions_to_close.append(j)

        # Remove closed positions (reverse order)
        for j in sorted(positions_to_close, reverse=True):
            positions.pop(j)

        # Entry logic
        if len(positions) < MAX_CONCURRENT and not pass_entry and active_pairs:
            pairs_to_check = active_pairs

            # Dynamic variant: pick highest |z| pair each week
            if config["dynamic"]:
                if last_dynamic_pick is None or (date - last_dynamic_pick).days >= 5:
                    best_pair = None
                    best_z = 0
                    for a, b, pval, beta in active_pairs:
                        z = compute_spread_zscore(prices, a, b, beta, date)
                        if abs(z) > abs(best_z):
                            best_z = z
                            best_pair = (a, b, pval, beta)
                    if best_pair and abs(best_z) >= config["z_entry"]:
                        pairs_to_check = [best_pair]
                        last_dynamic_pick = date
                    else:
                        pairs_to_check = []

            for a, b, pval, beta in pairs_to_check:
                if len(positions) >= MAX_CONCURRENT:
                    break

                z = compute_spread_zscore(prices, a, b, beta, date)

                # Already in a position for this pair?
                existing_pairs = set(pos.pair_info[:2] for pos in positions)
                if (a, b) in existing_pairs:
                    continue

                # Entry conditions
                if abs(z) < config["z_entry"]:
                    continue

                # Determine long leg
                if z > config["z_entry"]:
                    # Spread too high: A overvalued, B undervalued → long B
                    long_ticker = b
                elif z < -config["z_entry"]:
                    # Spread too low: A undervalued, B overvalued → long A
                    long_ticker = a
                else:
                    continue

                if date not in prices.index or long_ticker not in prices.columns:
                    continue
                price = prices.loc[date, long_ticker]
                if pd.isna(price) or price <= 0:
                    continue

                # Position sizing
                alloc = capital / MAX_CONCURRENT
                if alloc < 10:
                    continue

                if config["use_options"]:
                    # Buy calls: find strike for ~0.4 delta
                    T = OPTION_DTE / 252
                    strike = find_strike_for_delta(price, OPTION_DELTA, T)
                    call_price = bs_call_price(price, strike, T)
                    call_price *= (1 - BS_HAIRCUT)  # haircut
                    call_price *= (1 + OPTION_SPREAD_PCT)  # pay spread on entry

                    if call_price < 0.05:
                        continue
                    total_option_cost = call_price * OPTION_MULTIPLIER + OPTION_COMMISSION
                    n_contracts = max(1, int(alloc / total_option_cost))
                    actual_cost = n_contracts * total_option_cost

                    if actual_cost > capital:
                        n_contracts = max(1, int(capital / total_option_cost))
                        actual_cost = n_contracts * total_option_cost
                    if actual_cost > capital:
                        continue

                    capital -= actual_cost
                    positions.append(Position(
                        ticker=long_ticker,
                        entry_price=strike,  # store strike
                        entry_date=date,
                        shares=n_contracts,
                        pair_info=(a, b, beta, z),
                        is_option=True,
                        option_cost=actual_cost,
                    ))
                else:
                    # Stock position (fractional shares)
                    slippage = price * SLIPPAGE_PCT
                    entry_cost = price + slippage
                    shares = alloc / entry_cost
                    cost = shares * entry_cost
                    capital -= cost

                    positions.append(Position(
                        ticker=long_ticker,
                        entry_price=entry_cost,
                        entry_date=date,
                        shares=shares,
                        pair_info=(a, b, beta, z),
                    ))

        # Record equity
        mtm = capital
        for pos in positions:
            if date in prices.index and pos.ticker in prices.columns:
                curr = prices.loc[date, pos.ticker]
                if pos.is_option:
                    S = curr
                    K = pos.entry_price
                    T_remain = max(0, (OPTION_DTE - pos.days_held)) / 252
                    val = bs_call_price(S, K, T_remain) * (1 - BS_HAIRCUT) * OPTION_MULTIPLIER * pos.shares
                    mtm += val
                else:
                    mtm += pos.shares * curr

        equity_curve.append({"date": date.strftime("%Y-%m-%d"), "equity": round(mtm, 2)})

    # Force-close any remaining positions at end
    last_date = oot_dates[-1]
    for pos in positions:
        if last_date in prices.index and pos.ticker in prices.columns:
            exit_price = prices.loc[last_date, pos.ticker]
            if pos.is_option:
                pnl = -pos.option_cost  # expired worthless approx
            else:
                pnl = pos.shares * (exit_price - pos.entry_price - exit_price * SLIPPAGE_PCT)
            capital += pnl
            trades.append({
                "entry_date": pos.entry_date.strftime("%Y-%m-%d"),
                "exit_date": last_date.strftime("%Y-%m-%d"),
                "ticker": pos.ticker,
                "pair": f"{pos.pair_info[0]}/{pos.pair_info[1]}",
                "z_entry": pos.pair_info[3],
                "z_exit": 0,
                "pnl": round(pnl, 2),
                "return_pct": round(pnl / max(pos.option_cost if pos.is_option else (pos.shares * pos.entry_price), 1) * 100, 2),
                "days_held": pos.days_held,
                "exit_reason": "end_of_backtest",
                "is_option": pos.is_option,
            })

    return {
        "variant": variant,
        "config": {k: str(v) if not isinstance(v, (int, float, bool)) else v for k, v in config.items()},
        "trades": trades,
        "equity_curve": equity_curve,
        "final_capital": round(capital, 2) if not positions else round(
            sum(e["equity"] for e in equity_curve[-1:]) if equity_curve else capital, 2
        ),
    }


# ─── Metrics & Validation ───────────────────────────────────────────────────

def compute_metrics(result: dict, prices: pd.DataFrame) -> dict:
    """Compute performance metrics and run validation gates."""
    trades = result["trades"]
    equity = pd.DataFrame(result["equity_curve"])

    if len(trades) == 0:
        return {"n_trades": 0, "passed_gates": False, "gate_failures": ["no trades"]}

    equity["date"] = pd.to_datetime(equity["date"])
    equity = equity.set_index("date")
    equity["return"] = equity["equity"].pct_change().fillna(0)

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    n_trades = len(trades)
    win_rate = len(wins) / n_trades if n_trades > 0 else 0
    total_pnl = sum(pnls)
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf")

    # Sharpe & Sortino from daily returns
    daily_rets = equity["return"]
    ann_factor = np.sqrt(252)
    sharpe = (daily_rets.mean() / daily_rets.std() * ann_factor) if daily_rets.std() > 0 else 0
    downside = daily_rets[daily_rets < 0].std()
    sortino = (daily_rets.mean() / downside * ann_factor) if downside > 0 else 0

    # Max drawdown
    peak = equity["equity"].cummax()
    dd = (equity["equity"] - peak) / peak
    max_dd = dd.min()

    # CAGR
    n_years = len(equity) / 252
    final_eq = equity["equity"].iloc[-1] if len(equity) > 0 else STARTING_CAPITAL
    cagr = (final_eq / STARTING_CAPITAL) ** (1 / max(n_years, 0.01)) - 1

    # Regime analysis
    spy_close = prices["SPY"] if "SPY" in prices.columns else None
    regime_sharpes = {}
    if spy_close is not None:
        sma200 = spy_close.rolling(200).mean()
        bull_dates = spy_close.index[spy_close > sma200]
        bear_dates = spy_close.index[spy_close <= sma200]

        bull_rets = daily_rets.loc[daily_rets.index.isin(bull_dates)]
        bear_rets = daily_rets.loc[daily_rets.index.isin(bear_dates)]

        regime_sharpes["bull"] = float(bull_rets.mean() / bull_rets.std() * ann_factor) if len(bull_rets) > 1 and bull_rets.std() > 0 else 0
        regime_sharpes["bear"] = float(bear_rets.mean() / bear_rets.std() * ann_factor) if len(bear_rets) > 1 and bear_rets.std() > 0 else 0

    # Regime gap
    if regime_sharpes:
        s_bull = abs(regime_sharpes.get("bull", 0))
        s_bear = abs(regime_sharpes.get("bear", 0))
        regime_gap = abs(s_bull - s_bear) / max(s_bull, s_bear, 0.01)
    else:
        regime_gap = 0

    # Permutation test
    perm_p = permutation_test(trades, n_iter=VALIDATION_GATES["perm_iterations"])

    # Exit reason distribution
    exit_reasons = {}
    for t in trades:
        r = t["exit_reason"]
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    # Avg holding period
    avg_hold = np.mean([t["days_held"] for t in trades])

    # Validation gates
    gate_failures = []
    if sharpe < VALIDATION_GATES["min_sharpe"]:
        gate_failures.append(f"Sharpe {sharpe:.2f} < {VALIDATION_GATES['min_sharpe']}")
    if perm_p > VALIDATION_GATES["perm_p_max"]:
        gate_failures.append(f"Perm p={perm_p:.3f} > {VALIDATION_GATES['perm_p_max']}")
    if regime_gap > VALIDATION_GATES["regime_gap_max"]:
        gate_failures.append(f"Regime gap {regime_gap:.2f} > {VALIDATION_GATES['regime_gap_max']}")
    if max_dd < VALIDATION_GATES["max_drawdown_floor"]:
        gate_failures.append(f"MaxDD {max_dd:.1%} < {VALIDATION_GATES['max_drawdown_floor']:.0%}")
    if n_trades < VALIDATION_GATES["min_trades"]:
        gate_failures.append(f"Trades {n_trades} < {VALIDATION_GATES['min_trades']}")

    passed = len(gate_failures) == 0

    metrics = {
        "n_trades": n_trades,
        "win_rate": round(win_rate, 4),
        "total_pnl": round(total_pnl, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else "inf",
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "cagr": round(cagr, 4),
        "max_drawdown": round(max_dd, 4),
        "avg_hold_days": round(avg_hold, 1),
        "exit_reasons": exit_reasons,
        "regime_sharpes": {k: round(v, 3) for k, v in regime_sharpes.items()},
        "regime_gap": round(regime_gap, 3),
        "perm_p_value": round(perm_p, 4),
        "final_equity": round(final_eq, 2),
        "return_pct": round((final_eq / STARTING_CAPITAL - 1) * 100, 2),
        "passed_gates": passed,
        "gate_failures": gate_failures,
        "gates_passed_count": f"{5 - len(gate_failures)}/5",
    }
    return metrics


def permutation_test(trades: list, n_iter: int = 1000) -> float:
    """Shuffle which pair divergences we trade — test if pair selection matters."""
    if len(trades) < 5:
        return 1.0

    actual_pnl = sum(t["pnl"] for t in trades)
    pnls = np.array([t["pnl"] for t in trades])
    count_better = 0

    rng = np.random.RandomState(42)
    for _ in range(n_iter):
        # Randomly flip sign of each trade (simulate random entry direction)
        signs = rng.choice([-1, 1], size=len(pnls))
        shuffled_pnl = (pnls * signs).sum()
        if shuffled_pnl >= actual_pnl:
            count_better += 1

    return count_better / n_iter


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("COINTEGRATION PAIRS MEAN-REVERSION BACKTEST")
    log.info("=" * 70)

    prices = load_prices()

    # Compute regime indicator
    if "SPY" in prices.columns:
        spy_sma200 = prices["SPY"] > prices["SPY"].rolling(200).mean()
    else:
        spy_sma200 = None

    vix = prices["^VIX"] if "^VIX" in prices.columns else None

    all_results = {}

    for variant in ["A", "B", "C", "D", "E", "F"]:
        log.info(f"\n{'─' * 60}")
        log.info(f"VARIANT {variant}")
        log.info(f"{'─' * 60}")

        result = run_variant(prices, variant, spy_sma200, vix)
        metrics = compute_metrics(result, prices)
        result["metrics"] = metrics
        all_results[variant] = result

        # Print summary
        m = metrics
        status = "PASS" if m.get("passed_gates") else "FAIL"
        log.info(
            f"  Variant {variant}: {status} | "
            f"Trades={m['n_trades']} WR={m.get('win_rate', 0):.1%} "
            f"Sharpe={m.get('sharpe', 0):.2f} Sortino={m.get('sortino', 0):.2f} "
            f"PF={m.get('profit_factor', 0)} MaxDD={m.get('max_drawdown', 0):.1%} "
            f"Return={m.get('return_pct', 0):.1f}% "
            f"Perm-p={m.get('perm_p_value', 1):.3f} "
            f"Gates={m.get('gates_passed_count', '?')}"
        )
        if m.get("gate_failures"):
            for gf in m["gate_failures"]:
                log.info(f"    FAIL: {gf}")
        if m.get("regime_sharpes"):
            log.info(f"    Regime: Bull={m['regime_sharpes'].get('bull', 0):.2f} Bear={m['regime_sharpes'].get('bear', 0):.2f}")
        if m.get("exit_reasons"):
            log.info(f"    Exits: {m['exit_reasons']}")

    # Summary table
    log.info(f"\n{'=' * 70}")
    log.info("SUMMARY")
    log.info(f"{'=' * 70}")
    log.info(f"{'Variant':<10} {'Trades':>7} {'WR':>7} {'Sharpe':>8} {'Sortino':>8} {'PF':>8} {'MaxDD':>8} {'Return%':>9} {'Status':>7}")
    log.info("-" * 75)
    for v in ["A", "B", "C", "D", "E", "F"]:
        m = all_results[v]["metrics"]
        status = "PASS" if m.get("passed_gates") else "FAIL"
        pf_str = f"{m.get('profit_factor', 0)}" if m.get("profit_factor") != "inf" else "inf"
        log.info(
            f"  {v:<8} {m.get('n_trades', 0):>7} {m.get('win_rate', 0):>7.1%} "
            f"{m.get('sharpe', 0):>8.2f} {m.get('sortino', 0):>8.2f} "
            f"{pf_str:>8} {m.get('max_drawdown', 0):>8.1%} "
            f"{m.get('return_pct', 0):>8.1f}% {status:>7}"
        )

    # Save results (strip equity curves for size)
    save_data = {}
    for v, res in all_results.items():
        save_data[v] = {
            "variant": v,
            "config": res["config"],
            "metrics": res["metrics"],
            "trade_count": len(res["trades"]),
            "sample_trades": res["trades"][:10],
            "equity_start": res["equity_curve"][:3] if res["equity_curve"] else [],
            "equity_end": res["equity_curve"][-3:] if res["equity_curve"] else [],
        }

    output = {
        "strategy": "cointegration_pairs_mean_reversion",
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "oot_period": f"{OOT_START} to {OOT_END}",
        "starting_capital": STARTING_CAPITAL,
        "universe_size": len(ALL_TICKERS),
        "variants": save_data,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"\nResults saved to {RESULTS_PATH}")

    # Also save full trades for analysis
    trades_path = OUTPUT_DIR / "all_trades.json"
    all_trades = {v: res["trades"] for v, res in all_results.items()}
    with open(trades_path, "w") as f:
        json.dump(all_trades, f, indent=2, default=str)
    log.info(f"Full trades saved to {trades_path}")


if __name__ == "__main__":
    main()
