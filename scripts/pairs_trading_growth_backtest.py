#!/usr/bin/env python3
"""
Pairs Trading on Growth Stocks — Walk-Forward OOT Backtest
===========================================================
Tests 6 variants across 8 stock pairs (growth/tech).
Walk-forward: 60-day rolling train, OOT Jan 2022 – Jul 2026.
5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades.

Account context: $669 Robinhood, $200/leg max, max 3 concurrent pairs.
"""

import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────
PAIRS = [
    ("GOOGL", "META", "Ad-driven tech"),
    ("MSFT", "AAPL", "Mega-cap tech"),
    ("AMD", "NVDA", "Semiconductors"),
    ("UBER", "LYFT", "Rideshare"),
    ("SNAP", "PINS", "Social media"),
    ("COIN", "HOOD", "Fintech/trading"),
    ("DDOG", "NET", "Cloud infra"),
    ("CRM", "SNOW", "Enterprise SaaS"),
]

OOT_START = "2022-01-01"
OOT_END = "2026-07-30"
LOOKBACK_DAYS = 120  # extra data before OOT for rolling calcs
LEG_SIZE = 200.0  # $ per leg
MAX_CONCURRENT = 3
SHARE_SLIPPAGE = 0.0002  # 0.02%
PUT_COMMISSION = 0.65  # per contract each way
PUT_PREMIUM_PCT = 0.03  # 3% of stock price for 30-day ATM
PUT_BIDASK_SPREAD = 0.10  # 10% of premium
N_PERMUTATIONS = 1000


def download_data():
    """Download all required stock data + VIX."""
    tickers = list(set(t for p in PAIRS for t in (p[0], p[1])))
    tickers.append("^VIX")

    start = (pd.Timestamp(OOT_START) - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    print(f"Downloading {len(tickers)} tickers from {start} to {OOT_END}...")

    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=OOT_END, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 50:
                data[ticker] = df["Close"].copy()
                print(f"  {ticker}: {len(df)} days")
            else:
                print(f"  {ticker}: SKIP (only {len(df)} days)")
        except Exception as e:
            print(f"  {ticker}: FAILED ({e})")

    prices = pd.DataFrame(data)
    prices.index = pd.to_datetime(prices.index)
    # Remove timezone info if present
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices = prices.ffill().dropna(how="all")
    return prices


def compute_spread_zscore(s1, s2, window=60):
    """Classic log-price spread z-score."""
    spread = np.log(s1) - np.log(s2)
    roll_mean = spread.rolling(window).mean()
    roll_std = spread.rolling(window).std()
    zscore = (spread - roll_mean) / roll_std.replace(0, np.nan)
    return spread, zscore


def compute_relative_return(s1, s2, window=20):
    """Relative return divergence."""
    ret1 = s1.pct_change(window)
    ret2 = s2.pct_change(window)
    divergence = ret1 - ret2
    return divergence


def compute_percentile_rank(spread, window=90):
    """Rolling percentile rank of spread."""
    def pctile_rank(x):
        if len(x) < 10:
            return np.nan
        return stats.percentileofscore(x[:-1], x.iloc[-1]) / 100.0

    rank = spread.rolling(window).apply(pctile_rank, raw=False)
    return rank


def calculate_trade_costs(price_long, price_short, use_puts=True):
    """
    Calculate round-trip costs for a pair trade.
    Long leg: shares (0.02% slippage each way).
    Short leg: put options ($0.65/contract each way, premium=3%, bid-ask=10%).
    """
    # Long leg costs (shares)
    shares_cost = LEG_SIZE * SHARE_SLIPPAGE * 2  # entry + exit slippage

    if use_puts:
        # Short leg costs (puts)
        premium_per_share = price_short * PUT_PREMIUM_PCT
        n_contracts = max(1, int(LEG_SIZE / (premium_per_share * 100)))
        premium_total = n_contracts * premium_per_share * 100
        commission_rt = PUT_COMMISSION * 2 * n_contracts  # entry + exit
        bidask_cost = premium_total * PUT_BIDASK_SPREAD  # bid-ask spread cost
        put_cost = commission_rt + bidask_cost
    else:
        put_cost = 0

    return shares_cost + put_cost


def run_backtest_variant_A(prices, s1_name, s2_name, oot_mask):
    """Variant A: Classic z-score, enter at |z|>2, exit at z~0."""
    s1, s2 = prices[s1_name], prices[s2_name]
    spread, zscore = compute_spread_zscore(s1, s2, window=60)

    trades = []
    position = 0  # 0=flat, 1=long spread, -1=short spread
    entry_date = None
    entry_prices = None

    for i in range(len(prices)):
        if not oot_mask.iloc[i]:
            continue
        date = prices.index[i]
        z = zscore.iloc[i]
        if np.isnan(z):
            continue

        if position == 0:
            if z > 2.0:
                # Spread too high: short spread (short s1, long s2)
                position = -1
                entry_date = date
                entry_prices = (s1.iloc[i], s2.iloc[i])
            elif z < -2.0:
                # Spread too low: long spread (long s1, short s2)
                position = 1
                entry_date = date
                entry_prices = (s1.iloc[i], s2.iloc[i])
        else:
            # Exit when z crosses 0 (mean reversion) or after 30 days
            days_held = (date - entry_date).days
            exit_signal = (position == 1 and z >= 0) or (position == -1 and z <= 0)
            timeout = days_held >= 30

            if exit_signal or timeout:
                exit_prices = (s1.iloc[i], s2.iloc[i])
                # Calculate P&L
                if position == 1:
                    # Long s1, short s2
                    long_ret = (exit_prices[0] / entry_prices[0]) - 1
                    short_ret = 1 - (exit_prices[1] / entry_prices[1])
                else:
                    # Short s1, long s2
                    long_ret = (exit_prices[1] / entry_prices[1]) - 1
                    short_ret = 1 - (exit_prices[0] / entry_prices[0])

                long_pnl = LEG_SIZE * long_ret
                short_pnl = LEG_SIZE * short_ret
                gross_pnl = long_pnl + short_pnl

                # Costs
                if position == 1:
                    cost = calculate_trade_costs(entry_prices[0], entry_prices[1], use_puts=True)
                else:
                    cost = calculate_trade_costs(entry_prices[1], entry_prices[0], use_puts=True)

                net_pnl = gross_pnl - cost

                trades.append({
                    "entry_date": str(entry_date.date()),
                    "exit_date": str(date.date()),
                    "days_held": days_held,
                    "direction": "long_spread" if position == 1 else "short_spread",
                    "entry_z": float(zscore.loc[entry_date]) if entry_date in zscore.index else 0,
                    "exit_z": float(z),
                    "gross_pnl": round(gross_pnl, 2),
                    "costs": round(cost, 2),
                    "net_pnl": round(net_pnl, 2),
                    "exit_reason": "reversion" if exit_signal else "timeout",
                })
                position = 0
                entry_date = None
                entry_prices = None

    return trades


def run_backtest_variant_B(prices, s1_name, s2_name, oot_mask):
    """Variant B: Relative return divergence > 10%."""
    s1, s2 = prices[s1_name], prices[s2_name]
    divergence = compute_relative_return(s1, s2, window=20)

    trades = []
    position = 0
    entry_date = None
    entry_prices = None

    for i in range(len(prices)):
        if not oot_mask.iloc[i]:
            continue
        date = prices.index[i]
        div = divergence.iloc[i]
        if np.isnan(div):
            continue

        if position == 0:
            if div > 0.10:
                # s1 outperformed s2: long s2, short s1
                position = -1
                entry_date = date
                entry_prices = (s1.iloc[i], s2.iloc[i])
                entry_div = div
            elif div < -0.10:
                # s2 outperformed s1: long s1, short s2
                position = 1
                entry_date = date
                entry_prices = (s1.iloc[i], s2.iloc[i])
                entry_div = div
        else:
            days_held = (date - entry_date).days
            exit_signal = abs(div) < 0.02  # revert to within 2%
            timeout = days_held >= 30

            if exit_signal or timeout:
                exit_prices = (s1.iloc[i], s2.iloc[i])
                if position == 1:
                    long_ret = (exit_prices[0] / entry_prices[0]) - 1
                    short_ret = 1 - (exit_prices[1] / entry_prices[1])
                else:
                    long_ret = (exit_prices[1] / entry_prices[1]) - 1
                    short_ret = 1 - (exit_prices[0] / entry_prices[0])

                long_pnl = LEG_SIZE * long_ret
                short_pnl = LEG_SIZE * short_ret
                gross_pnl = long_pnl + short_pnl

                if position == 1:
                    cost = calculate_trade_costs(entry_prices[0], entry_prices[1], use_puts=True)
                else:
                    cost = calculate_trade_costs(entry_prices[1], entry_prices[0], use_puts=True)

                net_pnl = gross_pnl - cost

                trades.append({
                    "entry_date": str(entry_date.date()),
                    "exit_date": str(date.date()),
                    "days_held": days_held,
                    "direction": "long_spread" if position == 1 else "short_spread",
                    "entry_div": round(float(entry_div), 4),
                    "exit_div": round(float(div), 4),
                    "gross_pnl": round(gross_pnl, 2),
                    "costs": round(cost, 2),
                    "net_pnl": round(net_pnl, 2),
                    "exit_reason": "reversion" if exit_signal else "timeout",
                })
                position = 0

    return trades


def run_backtest_variant_C(prices, s1_name, s2_name, oot_mask):
    """Variant C: Long-only underperformer (no short leg)."""
    s1, s2 = prices[s1_name], prices[s2_name]
    spread, zscore = compute_spread_zscore(s1, s2, window=60)

    trades = []
    position = 0
    entry_date = None
    entry_price = None
    long_ticker = None

    for i in range(len(prices)):
        if not oot_mask.iloc[i]:
            continue
        date = prices.index[i]
        z = zscore.iloc[i]
        if np.isnan(z):
            continue

        if position == 0:
            if z > 2.0:
                # s1 overperformed, buy s2 (underperformer)
                position = 1
                entry_date = date
                entry_price = s2.iloc[i]
                long_ticker = s2_name
            elif z < -2.0:
                # s2 overperformed, buy s1 (underperformer)
                position = 1
                entry_date = date
                entry_price = s1.iloc[i]
                long_ticker = s1_name
        else:
            days_held = (date - entry_date).days
            exit_signal = abs(z) < 0.5
            timeout = days_held >= 30

            if exit_signal or timeout:
                if long_ticker == s1_name:
                    exit_price = s1.iloc[i]
                else:
                    exit_price = s2.iloc[i]

                ret = (exit_price / entry_price) - 1
                gross_pnl = LEG_SIZE * ret
                cost = LEG_SIZE * SHARE_SLIPPAGE * 2  # shares only
                net_pnl = gross_pnl - cost

                trades.append({
                    "entry_date": str(entry_date.date()),
                    "exit_date": str(date.date()),
                    "days_held": days_held,
                    "long_ticker": long_ticker,
                    "gross_pnl": round(gross_pnl, 2),
                    "costs": round(cost, 2),
                    "net_pnl": round(net_pnl, 2),
                    "exit_reason": "reversion" if exit_signal else "timeout",
                })
                position = 0

    return trades


def run_backtest_variant_E(prices, s1_name, s2_name, oot_mask):
    """Variant E: Adaptive threshold using 90-day percentile rank."""
    s1, s2 = prices[s1_name], prices[s2_name]
    spread = np.log(s1) - np.log(s2)
    pct_rank = compute_percentile_rank(spread, window=90)

    trades = []
    position = 0
    entry_date = None
    entry_prices = None

    for i in range(len(prices)):
        if not oot_mask.iloc[i]:
            continue
        date = prices.index[i]
        rank = pct_rank.iloc[i]
        if np.isnan(rank):
            continue

        if position == 0:
            if rank > 0.95:
                # Spread at 95th pctile: short spread
                position = -1
                entry_date = date
                entry_prices = (s1.iloc[i], s2.iloc[i])
            elif rank < 0.05:
                # Spread at 5th pctile: long spread
                position = 1
                entry_date = date
                entry_prices = (s1.iloc[i], s2.iloc[i])
        else:
            days_held = (date - entry_date).days
            exit_signal = 0.4 < rank < 0.6  # near 50th pctile
            timeout = days_held >= 30

            if exit_signal or timeout:
                exit_prices = (s1.iloc[i], s2.iloc[i])
                if position == 1:
                    long_ret = (exit_prices[0] / entry_prices[0]) - 1
                    short_ret = 1 - (exit_prices[1] / entry_prices[1])
                else:
                    long_ret = (exit_prices[1] / entry_prices[1]) - 1
                    short_ret = 1 - (exit_prices[0] / entry_prices[0])

                long_pnl = LEG_SIZE * long_ret
                short_pnl = LEG_SIZE * short_ret
                gross_pnl = long_pnl + short_pnl

                if position == 1:
                    cost = calculate_trade_costs(entry_prices[0], entry_prices[1], use_puts=True)
                else:
                    cost = calculate_trade_costs(entry_prices[1], entry_prices[0], use_puts=True)

                net_pnl = gross_pnl - cost

                trades.append({
                    "entry_date": str(entry_date.date()),
                    "exit_date": str(date.date()),
                    "days_held": days_held,
                    "direction": "long_spread" if position == 1 else "short_spread",
                    "gross_pnl": round(gross_pnl, 2),
                    "costs": round(cost, 2),
                    "net_pnl": round(net_pnl, 2),
                    "exit_reason": "reversion" if exit_signal else "timeout",
                })
                position = 0

    return trades


def run_backtest_variant_F(prices, s1_name, s2_name, oot_mask, vix):
    """Variant F: Regime-aware — only enter when VIX < 25."""
    s1, s2 = prices[s1_name], prices[s2_name]
    spread, zscore = compute_spread_zscore(s1, s2, window=60)

    trades = []
    position = 0
    entry_date = None
    entry_prices = None

    for i in range(len(prices)):
        if not oot_mask.iloc[i]:
            continue
        date = prices.index[i]
        z = zscore.iloc[i]
        v = vix.iloc[i] if i < len(vix) else 30
        if np.isnan(z) or np.isnan(v):
            continue

        if position == 0:
            if v >= 25:
                continue  # skip high-vol regime
            if z > 2.0:
                position = -1
                entry_date = date
                entry_prices = (s1.iloc[i], s2.iloc[i])
            elif z < -2.0:
                position = 1
                entry_date = date
                entry_prices = (s1.iloc[i], s2.iloc[i])
        else:
            days_held = (date - entry_date).days
            exit_signal = (position == 1 and z >= 0) or (position == -1 and z <= 0)
            timeout = days_held >= 30

            if exit_signal or timeout:
                exit_prices = (s1.iloc[i], s2.iloc[i])
                if position == 1:
                    long_ret = (exit_prices[0] / entry_prices[0]) - 1
                    short_ret = 1 - (exit_prices[1] / entry_prices[1])
                else:
                    long_ret = (exit_prices[1] / entry_prices[1]) - 1
                    short_ret = 1 - (exit_prices[0] / entry_prices[0])

                long_pnl = LEG_SIZE * long_ret
                short_pnl = LEG_SIZE * short_ret
                gross_pnl = long_pnl + short_pnl

                if position == 1:
                    cost = calculate_trade_costs(entry_prices[0], entry_prices[1], use_puts=True)
                else:
                    cost = calculate_trade_costs(entry_prices[1], entry_prices[0], use_puts=True)

                net_pnl = gross_pnl - cost

                trades.append({
                    "entry_date": str(entry_date.date()),
                    "exit_date": str(date.date()),
                    "days_held": days_held,
                    "direction": "long_spread" if position == 1 else "short_spread",
                    "entry_vix": round(float(v), 1),
                    "gross_pnl": round(gross_pnl, 2),
                    "costs": round(cost, 2),
                    "net_pnl": round(net_pnl, 2),
                    "exit_reason": "reversion" if exit_signal else "timeout",
                })
                position = 0

    return trades


def run_variant_D(all_pair_trades, prices, oot_mask):
    """
    Variant D: All-pairs portfolio. Combine trades from all 8 pairs (variant A signals).
    Enforce max 3 concurrent positions. Equal weight.
    """
    # Collect all potential trades with their dates
    all_trades_flat = []
    for pair_key, trades in all_pair_trades.items():
        for t in trades:
            t_copy = dict(t)
            t_copy["pair"] = pair_key
            all_trades_flat.append(t_copy)

    # Sort by entry date
    all_trades_flat.sort(key=lambda x: x["entry_date"])

    # Simulate with concurrency limit
    active = []
    accepted = []

    for t in all_trades_flat:
        entry = t["entry_date"]
        # Close any active positions that ended before this entry
        active = [a for a in active if a["exit_date"] > entry]

        if len(active) < MAX_CONCURRENT:
            active.append(t)
            accepted.append(t)

    return accepted


def compute_metrics(trades):
    """Compute strategy metrics from trade list."""
    if not trades or len(trades) == 0:
        return {
            "n_trades": 0, "sharpe": 0, "sortino": 0, "profit_factor": 0,
            "win_rate": 0, "total_pnl": 0, "avg_pnl": 0, "max_dd_pct": 0,
            "avg_days_held": 0, "perm_p_value": 1.0, "regime_gap": 0,
        }

    pnls = np.array([t["net_pnl"] for t in trades])
    n = len(pnls)
    total = pnls.sum()
    avg = pnls.mean()
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    wr = len(wins) / n if n > 0 else 0

    # Sharpe (annualized assuming ~12 trades/year baseline)
    if pnls.std() > 0:
        sharpe = (pnls.mean() / pnls.std()) * np.sqrt(min(n, 252))
    else:
        sharpe = 0

    # Sortino
    downside = pnls[pnls < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = (pnls.mean() / downside.std()) * np.sqrt(min(n, 252))
    else:
        sortino = sharpe * 1.5 if sharpe > 0 else 0

    # Profit factor
    gross_wins = wins.sum() if len(wins) > 0 else 0
    gross_losses = abs(losses.sum()) if len(losses) > 0 else 0.01
    pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    # Max drawdown
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    # As pct of account
    account = 669.0
    max_dd_pct = (dd.min() / account * 100) if account > 0 else 0

    # Average days held
    days_held = [t.get("days_held", 0) for t in trades]
    avg_days = np.mean(days_held) if days_held else 0

    # Permutation test
    obs_mean = pnls.mean()
    perm_count = 0
    for _ in range(N_PERMUTATIONS):
        shuffled = pnls.copy()
        np.random.shuffle(shuffled)
        signs = np.random.choice([-1, 1], size=len(shuffled))
        perm_pnls = shuffled * signs
        if perm_pnls.mean() >= obs_mean:
            perm_count += 1
    perm_p = perm_count / N_PERMUTATIONS

    # Regime analysis — split by year halves as proxy
    entry_dates = [t.get("entry_date", "2022-01-01") for t in trades]
    pnl_by_year = {}
    for date_str, pnl in zip(entry_dates, pnls):
        year = date_str[:4]
        pnl_by_year.setdefault(year, []).append(pnl)

    year_sharpes = {}
    for yr, yr_pnls in pnl_by_year.items():
        arr = np.array(yr_pnls)
        if len(arr) > 1 and arr.std() > 0:
            year_sharpes[yr] = arr.mean() / arr.std() * np.sqrt(len(arr))
        else:
            year_sharpes[yr] = 0

    if len(year_sharpes) >= 2:
        sharpes_list = list(year_sharpes.values())
        max_s = max(abs(s) for s in sharpes_list) if sharpes_list else 1
        if max_s > 0:
            regime_gap = (max(sharpes_list) - min(sharpes_list)) / max_s
        else:
            regime_gap = 0
    else:
        regime_gap = 0

    return {
        "n_trades": n,
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(pf, 3),
        "win_rate": round(wr, 4),
        "total_pnl": round(total, 2),
        "avg_pnl": round(avg, 2),
        "max_dd_pct": round(max_dd_pct, 2),
        "avg_days_held": round(avg_days, 1),
        "perm_p_value": round(perm_p, 4),
        "regime_gap": round(regime_gap, 3),
        "year_sharpes": {k: round(v, 3) for k, v in year_sharpes.items()},
    }


def five_gate_check(metrics):
    """Apply 5-gate validation."""
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": metrics["perm_p_value"] < 0.05,
        "regime_gap_lt_0.5": metrics["regime_gap"] < 0.5,
        "max_dd_gt_neg50": metrics["max_dd_pct"] > -50,
        "min_20_trades": metrics["n_trades"] >= 20,
    }
    gates["passed_all"] = all(gates.values())
    gates["gates_passed"] = sum(1 for v in list(gates.values())[:-2] if v)
    return gates


def compute_pair_correlation(prices, s1_name, s2_name, oot_mask):
    """Compute rolling and overall correlation for a pair."""
    s1 = prices[s1_name][oot_mask]
    s2 = prices[s2_name][oot_mask]
    ret1 = s1.pct_change().dropna()
    ret2 = s2.pct_change().dropna()
    common = ret1.index.intersection(ret2.index)
    if len(common) < 20:
        return 0, 0
    overall_corr = ret1.loc[common].corr(ret2.loc[common])
    # Rolling 60-day correlation stability
    roll_corr = ret1.loc[common].rolling(60).corr(ret2.loc[common]).dropna()
    corr_stability = 1 - roll_corr.std() if len(roll_corr) > 0 else 0
    return round(overall_corr, 4), round(corr_stability, 4)


def main():
    np.random.seed(42)

    print("=" * 70)
    print("PAIRS TRADING GROWTH STOCKS — WALK-FORWARD BACKTEST")
    print("=" * 70)

    # Download data
    prices = download_data()
    oot_mask = pd.Series(prices.index >= pd.Timestamp(OOT_START), index=prices.index)

    # VIX for variant F
    vix = prices.get("^VIX", pd.Series(dtype=float))
    if vix.empty:
        vix = pd.Series(20.0, index=prices.index)  # fallback

    results = {
        "metadata": {
            "strategy": "Pairs Trading on Growth Stocks",
            "oot_period": f"{OOT_START} to {OOT_END}",
            "account_size": 669,
            "leg_size": LEG_SIZE,
            "max_concurrent_pairs": MAX_CONCURRENT,
            "pairs_tested": len(PAIRS),
            "variants_tested": 6,
            "run_timestamp": datetime.now().isoformat(),
            "share_slippage": SHARE_SLIPPAGE,
            "put_commission_per_contract": PUT_COMMISSION,
            "put_premium_pct": PUT_PREMIUM_PCT,
            "put_bidask_spread": PUT_BIDASK_SPREAD,
        },
        "pair_correlations": {},
        "variants": {},
    }

    # ── Compute pair correlations ──────────────────────────────────────────
    print("\n── Pair Correlations ──")
    for s1, s2, label in PAIRS:
        if s1 in prices.columns and s2 in prices.columns:
            corr, stability = compute_pair_correlation(prices, s1, s2, oot_mask)
            results["pair_correlations"][f"{s1}/{s2}"] = {
                "label": label,
                "correlation": corr,
                "stability": stability,
            }
            print(f"  {s1}/{s2} ({label}): corr={corr:.3f}, stability={stability:.3f}")

    # ── Run each variant per pair ──────────────────────────────────────────
    variant_configs = {
        "A_classic_zscore": ("Classic z-score (|z|>2, exit at 0)", run_backtest_variant_A),
        "B_relative_return": ("Relative return divergence >10%", run_backtest_variant_B),
        "C_long_only": ("Long-only underperformer", run_backtest_variant_C),
        "E_adaptive_threshold": ("Adaptive 90-day percentile", run_backtest_variant_E),
        "F_regime_aware": ("Regime-aware (VIX<25)", None),  # special handler
    }

    # Store variant A trades per pair for variant D
    variant_a_pair_trades = {}

    for var_key, (var_label, var_func) in variant_configs.items():
        print(f"\n{'=' * 70}")
        print(f"VARIANT {var_key}: {var_label}")
        print(f"{'=' * 70}")

        all_variant_trades = []
        pair_results = {}

        for s1, s2, label in PAIRS:
            if s1 not in prices.columns or s2 not in prices.columns:
                print(f"  SKIP {s1}/{s2} — missing data")
                continue

            if var_key == "F_regime_aware":
                trades = run_backtest_variant_F(prices, s1, s2, oot_mask, vix)
            else:
                trades = var_func(prices, s1, s2, oot_mask)

            if var_key == "A_classic_zscore":
                variant_a_pair_trades[f"{s1}/{s2}"] = trades

            metrics = compute_metrics(trades)
            gates = five_gate_check(metrics)

            pair_results[f"{s1}/{s2}"] = {
                "label": label,
                "metrics": metrics,
                "gates": gates,
                "sample_trades": trades[:5] if trades else [],
            }

            status = "PASS" if gates["passed_all"] else f"FAIL ({gates['gates_passed']}/5)"
            print(f"  {s1}/{s2}: {len(trades)} trades, Sharpe={metrics['sharpe']:.2f}, "
                  f"WR={metrics['win_rate']:.1%}, PF={metrics['profit_factor']:.2f}, "
                  f"MDD={metrics['max_dd_pct']:.1f}%, P&L=${metrics['total_pnl']:.0f} → {status}")

            all_variant_trades.extend(trades)

        # Aggregate variant metrics
        agg_metrics = compute_metrics(all_variant_trades)
        agg_gates = five_gate_check(agg_metrics)

        results["variants"][var_key] = {
            "description": var_label,
            "pair_results": pair_results,
            "aggregate": {
                "metrics": agg_metrics,
                "gates": agg_gates,
            },
        }

        print(f"\n  AGGREGATE: {agg_metrics['n_trades']} trades, Sharpe={agg_metrics['sharpe']:.2f}, "
              f"WR={agg_metrics['win_rate']:.1%}, PF={agg_metrics['profit_factor']:.2f}, "
              f"MDD={agg_metrics['max_dd_pct']:.1f}%, P&L=${agg_metrics['total_pnl']:.0f}")
        status = "PASS" if agg_gates["passed_all"] else f"FAIL ({agg_gates['gates_passed']}/5)"
        print(f"  5-GATE: {status}")

    # ── Variant D: All-pairs portfolio ─────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("VARIANT D: All-pairs portfolio (8 pairs, max 3 concurrent)")
    print(f"{'=' * 70}")

    portfolio_trades = run_variant_D(variant_a_pair_trades, prices, oot_mask)
    port_metrics = compute_metrics(portfolio_trades)
    port_gates = five_gate_check(port_metrics)

    results["variants"]["D_all_pairs_portfolio"] = {
        "description": "All 8 pairs traded simultaneously, max 3 concurrent, variant A signals",
        "aggregate": {
            "metrics": port_metrics,
            "gates": port_gates,
        },
        "sample_trades": portfolio_trades[:10] if portfolio_trades else [],
    }

    print(f"  Portfolio: {port_metrics['n_trades']} trades, Sharpe={port_metrics['sharpe']:.2f}, "
          f"WR={port_metrics['win_rate']:.1%}, PF={port_metrics['profit_factor']:.2f}, "
          f"MDD={port_metrics['max_dd_pct']:.1f}%, P&L=${port_metrics['total_pnl']:.0f}")
    status = "PASS" if port_gates["passed_all"] else f"FAIL ({port_gates['gates_passed']}/5)"
    print(f"  5-GATE: {status}")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY — 5-GATE RESULTS")
    print(f"{'=' * 70}")

    summary_rows = []
    for var_key, var_data in results["variants"].items():
        agg = var_data["aggregate"]
        m = agg["metrics"]
        g = agg["gates"]
        passed = "PASS" if g["passed_all"] else "FAIL"
        summary_rows.append({
            "variant": var_key,
            "trades": m["n_trades"],
            "sharpe": m["sharpe"],
            "sortino": m["sortino"],
            "pf": m["profit_factor"],
            "wr": m["win_rate"],
            "pnl": m["total_pnl"],
            "mdd%": m["max_dd_pct"],
            "perm_p": m["perm_p_value"],
            "regime_gap": m["regime_gap"],
            "result": passed,
            "gates": f"{g['gates_passed']}/5",
        })

    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))

    # Find best variant
    passing = [r for r in summary_rows if r["result"] == "PASS"]
    if passing:
        best = max(passing, key=lambda x: x["sharpe"])
        results["recommendation"] = {
            "best_variant": best["variant"],
            "reason": f"Highest Sharpe ({best['sharpe']}) among passing variants",
            "deployable": True,
        }
        print(f"\nBEST: {best['variant']} — Sharpe {best['sharpe']}, {best['trades']} trades, "
              f"P&L ${best['pnl']:.0f}")
    else:
        # Find best failing
        best_fail = max(summary_rows, key=lambda x: x["gates"]) if summary_rows else None
        results["recommendation"] = {
            "best_variant": best_fail["variant"] if best_fail else "none",
            "reason": "No variant passed all 5 gates",
            "deployable": False,
        }
        print(f"\nNO VARIANT PASSED ALL 5 GATES.")
        if best_fail:
            print(f"Closest: {best_fail['variant']} ({best_fail['gates']} gates)")

    # ── Best individual pairs ──────────────────────────────────────────────
    print(f"\n── Best Individual Pairs (Variant A) ──")
    best_pairs = []
    var_a = results["variants"].get("A_classic_zscore", {}).get("pair_results", {})
    for pair_key, pdata in var_a.items():
        m = pdata["metrics"]
        g = pdata["gates"]
        best_pairs.append((pair_key, m["sharpe"], m["n_trades"], m["win_rate"],
                           m["total_pnl"], g["passed_all"], g["gates_passed"]))

    best_pairs.sort(key=lambda x: x[1], reverse=True)
    for bp in best_pairs:
        status = "PASS" if bp[5] else f"FAIL({bp[6]}/5)"
        print(f"  {bp[0]}: Sharpe={bp[1]:.2f}, {bp[2]} trades, WR={bp[3]:.1%}, "
              f"P&L=${bp[4]:.0f} → {status}")

    results["best_pairs_ranked"] = [
        {"pair": bp[0], "sharpe": bp[1], "trades": bp[2], "win_rate": bp[3],
         "pnl": bp[4], "passed": bp[5], "gates": f"{bp[6]}/5"}
        for bp in best_pairs
    ]

    # ── Save results ───────────────────────────────────────────────────────
    output_path = "/home/jupiter/Lvl3Quant/data/pairs_trading_growth_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
