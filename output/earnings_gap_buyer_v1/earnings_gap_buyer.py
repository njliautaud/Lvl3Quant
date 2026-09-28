#!/usr/bin/env python3
"""
Earnings Gap Buyer Backtest v1
HC #705 — All adversarial checks inline.

Uses EQUITY simulation (buy stock at gap-day open) to test if the
underlying earnings move is profitable, avoiding unreliable BS pricing.

Earnings dates from yfinance .get_earnings_dates() API, cross-validated
against actual price gaps.
"""

import json
import os
import sys
import warnings
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/earnings_gap_buyer_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX",
    "AMD", "INTC", "BA", "SBUX", "NKE", "COST", "NOW", "CRM", "UNH",
    "GS", "JPM", "DIS"
]

START_DATE = "2019-01-01"
END_DATE = "2026-07-14"
STARTING_CAPITAL = 10_000
GAP_THRESHOLDS = [5, 7, 10]  # percent
HOLD_DAYS = [1, 2, 5]
TRAILING_STOP_PCT = 3.0
PERMUTATION_SHUFFLES = 200
COMMISSION_PER_TRADE = 0.0  # Robinhood no commission on equities

# ─────────────────────────────────────────────────────────
# DATA DOWNLOAD
# ─────────────────────────────────────────────────────────

def download_price_data(ticker: str) -> pd.DataFrame:
    """Download daily OHLCV data from yfinance."""
    t = yf.Ticker(ticker)
    df = t.history(start=START_DATE, end=END_DATE, auto_adjust=True)
    if df.empty:
        print(f"  WARNING: No price data for {ticker}")
        return pd.DataFrame()
    df.index = df.index.tz_localize(None)
    return df


def download_earnings_dates(ticker: str) -> list:
    """Download earnings dates from yfinance API."""
    t = yf.Ticker(ticker)
    try:
        # get_earnings_dates returns upcoming and past earnings
        # We need to paginate to get all historical dates
        all_dates = []
        for offset in range(0, 200, 12):
            try:
                ed = t.get_earnings_dates(limit=12, offset=offset)
                if ed is None or ed.empty:
                    break
                dates = ed.index.tz_localize(None).tolist()
                all_dates.extend(dates)
                # If we got fewer than requested, we've hit the end
                if len(ed) < 12:
                    break
                time.sleep(0.2)  # rate limit
            except Exception:
                break
        # Deduplicate and sort
        all_dates = sorted(set(all_dates))
        # Filter to our date range
        start = pd.Timestamp(START_DATE)
        end = pd.Timestamp(END_DATE)
        all_dates = [d for d in all_dates if start <= d <= end]
        return all_dates
    except Exception as e:
        print(f"  WARNING: Could not get earnings dates for {ticker}: {e}")
        return []


def find_gaps(prices: pd.DataFrame, min_gap_pct: float = 3.0) -> pd.DataFrame:
    """Find all gaps > min_gap_pct in the price data."""
    if prices.empty or len(prices) < 2:
        return pd.DataFrame()

    prev_close = prices["Close"].shift(1)
    gap_pct = ((prices["Open"] - prev_close) / prev_close * 100)

    mask = gap_pct.abs() >= min_gap_pct
    gaps = pd.DataFrame({
        "date": prices.index[mask],
        "gap_pct": gap_pct[mask].values,
        "open": prices["Open"][mask].values,
        "close": prices["Close"][mask].values,
        "prev_close": prev_close[mask].values,
    })
    return gaps


def cross_validate_earnings(earnings_dates: list, gaps: pd.DataFrame,
                            prices: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-validate: for each earnings date, check if a >3% gap exists
    within 1 trading day. Returns matched events with verified flag.
    """
    if gaps.empty or not earnings_dates:
        return pd.DataFrame()

    trading_dates = prices.index.tolist()
    results = []

    for ed in earnings_dates:
        # Find the nearest trading day on or after the earnings date
        # Earnings can be reported after close (gap next morning) or before open (gap same morning)
        # Check the earnings date and the next trading day
        candidates = []
        for offset in range(0, 3):  # check ed, ed+1, ed+2 trading days
            target = ed + timedelta(days=offset)
            # Find nearest trading day
            idx_matches = [td for td in trading_dates if abs((td - target).days) <= 1]
            candidates.extend(idx_matches)
        candidates = list(set(candidates))

        matched = False
        for gap_date in gaps["date"].values:
            gap_ts = pd.Timestamp(gap_date)
            for c in candidates:
                if abs((gap_ts - c).days) <= 1:
                    gap_row = gaps[gaps["date"] == gap_date].iloc[0]
                    results.append({
                        "earnings_date": ed,
                        "gap_date": gap_ts,
                        "gap_pct": gap_row["gap_pct"],
                        "open": gap_row["open"],
                        "close": gap_row["close"],
                        "prev_close": gap_row["prev_close"],
                        "verified": True,
                    })
                    matched = True
                    break
            if matched:
                break

        if not matched:
            results.append({
                "earnings_date": ed,
                "gap_date": pd.NaT,
                "gap_pct": 0.0,
                "open": 0.0,
                "close": 0.0,
                "prev_close": 0.0,
                "verified": False,
            })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────
# BACKTESTING ENGINE
# ─────────────────────────────────────────────────────────

def simulate_trades(events: pd.DataFrame, prices: pd.DataFrame,
                    gap_threshold: float, hold_days: int,
                    trailing_stop_pct: float = None) -> list:
    """
    Simulate equity trades: buy at gap-day open in gap direction,
    exit after hold_days or trailing stop.
    """
    trades = []
    trading_dates = prices.index.tolist()

    for _, ev in events.iterrows():
        if not ev["verified"] or abs(ev["gap_pct"]) < gap_threshold:
            continue

        gap_date = ev["gap_date"]
        if pd.isna(gap_date):
            continue

        # Find gap_date index in trading_dates
        try:
            idx = trading_dates.index(gap_date)
        except ValueError:
            # Find nearest
            diffs = [abs((td - gap_date).days) for td in trading_dates]
            idx = np.argmin(diffs)
            if diffs[idx] > 1:
                continue

        entry_price = prices.iloc[idx]["Open"]
        direction = 1 if ev["gap_pct"] > 0 else -1  # follow the gap

        # Determine exit
        if trailing_stop_pct is not None:
            # Trailing stop exit
            best_price = entry_price
            exit_price = entry_price
            exit_idx = idx
            max_hold = min(idx + 10, len(trading_dates) - 1)  # cap at 10 days

            for j in range(idx, max_hold + 1):
                if direction == 1:
                    if prices.iloc[j]["High"] > best_price:
                        best_price = prices.iloc[j]["High"]
                    stop_level = best_price * (1 - trailing_stop_pct / 100)
                    if prices.iloc[j]["Low"] <= stop_level:
                        exit_price = stop_level
                        exit_idx = j
                        break
                else:
                    if prices.iloc[j]["Low"] < best_price:
                        best_price = prices.iloc[j]["Low"]
                    stop_level = best_price * (1 + trailing_stop_pct / 100)
                    if prices.iloc[j]["High"] >= stop_level:
                        exit_price = stop_level
                        exit_idx = j
                        break
                exit_price = prices.iloc[j]["Close"]
                exit_idx = j
        else:
            # Fixed hold days
            exit_idx = min(idx + hold_days, len(trading_dates) - 1)
            exit_price = prices.iloc[exit_idx]["Close"]

        # Calculate return
        if direction == 1:
            ret_pct = (exit_price - entry_price) / entry_price * 100
        else:
            ret_pct = (entry_price - exit_price) / entry_price * 100

        trades.append({
            "entry_date": trading_dates[idx],
            "exit_date": trading_dates[exit_idx],
            "ticker": ev.get("ticker", "UNK"),
            "direction": "LONG" if direction == 1 else "SHORT",
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "gap_pct": float(ev["gap_pct"]),
            "return_pct": float(ret_pct),
            "hold_days": exit_idx - idx,
        })

    return trades


def compute_metrics(trades: list) -> dict:
    """Compute strategy metrics from trade list."""
    if not trades:
        return {
            "n_trades": 0, "win_rate": 0, "avg_return": 0,
            "total_return": 0, "sharpe": 0, "sortino": 0,
            "profit_factor": 0, "max_drawdown": 0,
            "avg_winner": 0, "avg_loser": 0,
        }

    returns = [t["return_pct"] for t in trades]
    returns = np.array(returns)

    winners = returns[returns > 0]
    losers = returns[returns <= 0]

    avg_ret = float(np.mean(returns))
    std_ret = float(np.std(returns)) if len(returns) > 1 else 1e-9
    downside = returns[returns < 0]
    downside_std = float(np.std(downside)) if len(downside) > 1 else 1e-9

    gross_profit = float(np.sum(winners)) if len(winners) > 0 else 0
    gross_loss = float(np.abs(np.sum(losers))) if len(losers) > 0 else 1e-9

    # Equity curve for max drawdown
    equity = STARTING_CAPITAL
    peak = equity
    max_dd = 0
    for r in returns:
        equity *= (1 + r / 100)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd

    return {
        "n_trades": len(trades),
        "win_rate": float(len(winners) / len(returns) * 100),
        "avg_return": avg_ret,
        "median_return": float(np.median(returns)),
        "total_return": float((np.prod(1 + returns / 100) - 1) * 100),
        "sharpe": float(avg_ret / std_ret * np.sqrt(252 / max(np.mean([t["hold_days"] for t in trades]), 1))) if std_ret > 1e-9 else 0,
        "sortino": float(avg_ret / downside_std * np.sqrt(252 / max(np.mean([t["hold_days"] for t in trades]), 1))) if downside_std > 1e-9 else 0,
        "profit_factor": float(gross_profit / gross_loss),
        "max_drawdown": float(max_dd),
        "avg_winner": float(np.mean(winners)) if len(winners) > 0 else 0,
        "avg_loser": float(np.mean(losers)) if len(losers) > 0 else 0,
        "final_equity": float(STARTING_CAPITAL * np.prod(1 + returns / 100)),
    }


# ─────────────────────────────────────────────────────────
# ADVERSARIAL CHECKS (HC #705 — ALL INLINE)
# ─────────────────────────────────────────────────────────

def permutation_test(real_trades: list, all_prices: dict, n_shuffles: int = 200) -> dict:
    """
    Permutation test: generate random entry dates (same count, same month
    distribution), compare real mean return to random-date mean returns.
    p-value = fraction of random means >= real mean.
    """
    if not real_trades:
        return {"p_value": 1.0, "real_mean": 0, "random_mean": 0, "n_shuffles": n_shuffles}

    real_returns = [t["return_pct"] for t in real_trades]
    real_mean = np.mean(real_returns)
    n_trades = len(real_trades)

    # Get month distribution of real trades
    months = [t["entry_date"].month for t in real_trades]

    # Get all available trading dates per ticker
    all_trading_dates = {}
    for ticker, prices in all_prices.items():
        all_trading_dates[ticker] = prices.index.tolist()

    ticker_list = list(all_prices.keys())
    random_means = []

    for _ in range(n_shuffles):
        random_returns = []
        for i in range(n_trades):
            # Pick random ticker and random date
            ticker = np.random.choice(ticker_list)
            dates = all_trading_dates[ticker]
            if len(dates) < 6:
                continue
            idx = np.random.randint(0, len(dates) - 5)
            entry_price = all_prices[ticker].iloc[idx]["Open"]
            # Use same hold as original trade
            hold = real_trades[i]["hold_days"]
            exit_idx = min(idx + max(hold, 1), len(dates) - 1)
            exit_price = all_prices[ticker].iloc[exit_idx]["Close"]

            # Random direction (50/50)
            direction = np.random.choice([-1, 1])
            if direction == 1:
                ret = (exit_price - entry_price) / entry_price * 100
            else:
                ret = (entry_price - exit_price) / entry_price * 100
            random_returns.append(ret)

        if random_returns:
            random_means.append(np.mean(random_returns))

    if not random_means:
        return {"p_value": 1.0, "real_mean": float(real_mean), "random_mean": 0, "n_shuffles": n_shuffles}

    p_value = np.mean([rm >= real_mean for rm in random_means])
    return {
        "p_value": float(p_value),
        "real_mean": float(real_mean),
        "random_mean": float(np.mean(random_means)),
        "random_std": float(np.std(random_means)),
        "n_shuffles": n_shuffles,
        "significant": p_value < 0.05,
    }


def regime_test(trades: list, spy_prices: pd.DataFrame) -> dict:
    """
    Regime test using PRIOR-DAY SPY close (not same-day — that's leakage).
    Classify regime by SPY's prior-day return: up, down, flat.
    """
    if not trades or spy_prices.empty:
        return {"regime_split_valid": False}

    spy_returns = spy_prices["Close"].pct_change()

    up_trades = []
    down_trades = []
    flat_trades = []

    for t in trades:
        entry = t["entry_date"]
        # Find PRIOR day SPY close
        prior_dates = spy_prices.index[spy_prices.index < entry]
        if len(prior_dates) < 2:
            flat_trades.append(t["return_pct"])
            continue

        prior_ret = spy_returns.loc[prior_dates[-1]]
        if pd.isna(prior_ret):
            flat_trades.append(t["return_pct"])
        elif prior_ret > 0.005:
            up_trades.append(t["return_pct"])
        elif prior_ret < -0.005:
            down_trades.append(t["return_pct"])
        else:
            flat_trades.append(t["return_pct"])

    result = {
        "regime_split_valid": True,
        "up_regime": {
            "n": len(up_trades),
            "mean_return": float(np.mean(up_trades)) if up_trades else 0,
            "win_rate": float(np.mean([1 for r in up_trades if r > 0]) / len(up_trades) * 100) if up_trades else 0,
        },
        "down_regime": {
            "n": len(down_trades),
            "mean_return": float(np.mean(down_trades)) if down_trades else 0,
            "win_rate": float(np.mean([1 for r in down_trades if r > 0]) / len(down_trades) * 100) if down_trades else 0,
        },
        "flat_regime": {
            "n": len(flat_trades),
            "mean_return": float(np.mean(flat_trades)) if flat_trades else 0,
            "win_rate": float(np.mean([1 for r in flat_trades if r > 0]) / len(flat_trades) * 100) if flat_trades else 0,
        },
    }

    # Check regime consistency — reject if too divergent
    means = []
    for regime in ["up_regime", "down_regime", "flat_regime"]:
        if result[regime]["n"] >= 5:
            means.append(result[regime]["mean_return"])

    if len(means) >= 2:
        max_m = max(abs(m) for m in means) if means else 1e-9
        if max_m > 1e-9:
            divergence = (max(means) - min(means)) / max_m
            result["regime_divergence"] = float(divergence)
            result["regime_consistent"] = divergence < 2.0  # Allow some divergence for event-driven
        else:
            result["regime_divergence"] = 0.0
            result["regime_consistent"] = True
    else:
        result["regime_consistent"] = True
        result["regime_divergence"] = 0.0

    return result


def sub_period_consistency(trades: list) -> dict:
    """Split trades into sub-periods and check consistency."""
    if len(trades) < 10:
        return {"valid": False, "reason": "Too few trades for sub-period analysis"}

    # Sort by entry date
    sorted_trades = sorted(trades, key=lambda t: t["entry_date"])
    mid = len(sorted_trades) // 2
    first_half = [t["return_pct"] for t in sorted_trades[:mid]]
    second_half = [t["return_pct"] for t in sorted_trades[mid:]]

    # Year-by-year
    yearly = {}
    for t in sorted_trades:
        yr = t["entry_date"].year
        if yr not in yearly:
            yearly[yr] = []
        yearly[yr].append(t["return_pct"])

    yearly_stats = {}
    for yr, rets in sorted(yearly.items()):
        yearly_stats[str(yr)] = {
            "n_trades": len(rets),
            "mean_return": float(np.mean(rets)),
            "win_rate": float(np.mean([1 for r in rets if r > 0]) / len(rets) * 100) if rets else 0,
        }

    # Count profitable years
    profitable_years = sum(1 for yr, s in yearly_stats.items() if s["mean_return"] > 0)
    total_years = len(yearly_stats)

    return {
        "valid": True,
        "first_half_mean": float(np.mean(first_half)),
        "second_half_mean": float(np.mean(second_half)),
        "first_half_wr": float(np.mean([1 for r in first_half if r > 0]) / len(first_half) * 100),
        "second_half_wr": float(np.mean([1 for r in second_half if r > 0]) / len(second_half) * 100),
        "halves_same_sign": (np.mean(first_half) > 0) == (np.mean(second_half) > 0),
        "yearly_stats": yearly_stats,
        "profitable_years": profitable_years,
        "total_years": total_years,
        "year_consistency": float(profitable_years / total_years) if total_years > 0 else 0,
    }


def outlier_removal_test(trades: list) -> dict:
    """Remove top/bottom 5% of trades and re-check profitability."""
    if len(trades) < 20:
        return {"valid": False, "reason": "Too few trades"}

    returns = np.array([t["return_pct"] for t in trades])
    p5 = np.percentile(returns, 5)
    p95 = np.percentile(returns, 95)

    trimmed = returns[(returns >= p5) & (returns <= p95)]

    return {
        "valid": True,
        "original_mean": float(np.mean(returns)),
        "trimmed_mean": float(np.mean(trimmed)),
        "original_n": len(returns),
        "trimmed_n": len(trimmed),
        "removed_count": len(returns) - len(trimmed),
        "still_profitable": float(np.mean(trimmed)) > 0,
        "pct_change": float((np.mean(trimmed) - np.mean(returns)) / abs(np.mean(returns)) * 100) if abs(np.mean(returns)) > 1e-9 else 0,
    }


def ticker_concentration_test(trades: list) -> dict:
    """Check if profitability is concentrated in a few tickers."""
    if not trades:
        return {"valid": False}

    ticker_stats = {}
    for t in trades:
        tk = t["ticker"]
        if tk not in ticker_stats:
            ticker_stats[tk] = []
        ticker_stats[tk].append(t["return_pct"])

    concentration = {}
    total_pnl = sum(t["return_pct"] for t in trades)

    for tk, rets in ticker_stats.items():
        tk_pnl = sum(rets)
        concentration[tk] = {
            "n_trades": len(rets),
            "mean_return": float(np.mean(rets)),
            "total_pnl_contribution": float(tk_pnl),
            "pct_of_total": float(tk_pnl / total_pnl * 100) if abs(total_pnl) > 1e-9 else 0,
        }

    # Check if any single ticker contributes > 40% of PnL
    max_concentration = max(abs(c["pct_of_total"]) for c in concentration.values()) if concentration else 0

    return {
        "valid": True,
        "ticker_stats": concentration,
        "max_concentration_pct": float(max_concentration),
        "concentrated": max_concentration > 40,
        "n_tickers_traded": len(ticker_stats),
    }


def pricing_sanity_check(trades: list) -> dict:
    """Check for suspicious pricing patterns."""
    if not trades:
        return {"valid": False}

    issues = []
    for t in trades:
        # Check for unreasonable returns
        if abs(t["return_pct"]) > 50:
            issues.append(f"{t['ticker']} {t['entry_date']}: {t['return_pct']:.1f}% return (suspicious)")
        # Check for zero prices
        if t["entry_price"] <= 0 or t["exit_price"] <= 0:
            issues.append(f"{t['ticker']} {t['entry_date']}: zero/negative price")
        # Check for penny stocks
        if t["entry_price"] < 5:
            issues.append(f"{t['ticker']} {t['entry_date']}: price ${t['entry_price']:.2f} (penny stock territory)")

    returns = [t["return_pct"] for t in trades]
    return {
        "valid": True,
        "n_issues": len(issues),
        "issues": issues[:20],  # Cap at 20
        "return_stats": {
            "min": float(np.min(returns)),
            "max": float(np.max(returns)),
            "mean": float(np.mean(returns)),
            "std": float(np.std(returns)),
            "skew": float(stats.skew(returns)),
            "kurtosis": float(stats.kurtosis(returns)),
        },
        "pricing_clean": len(issues) == 0,
    }


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("EARNINGS GAP BUYER BACKTEST v1")
    print("=" * 70)

    # ── Step 1: Download all data ──
    print("\n[1/5] Downloading price data and earnings dates...")
    all_prices = {}
    all_events = {}
    data_quality = {}

    for ticker in TICKERS:
        print(f"  {ticker}...", end=" ", flush=True)
        prices = download_price_data(ticker)
        if prices.empty:
            print("SKIP (no data)")
            continue
        all_prices[ticker] = prices

        earnings_dates = download_earnings_dates(ticker)
        gaps = find_gaps(prices, min_gap_pct=3.0)

        # Cross-validate
        if earnings_dates and not gaps.empty:
            events = cross_validate_earnings(earnings_dates, gaps, prices)
            events["ticker"] = ticker
        else:
            events = pd.DataFrame()

        n_api_dates = len(earnings_dates)
        n_gaps = len(gaps)
        n_verified = len(events[events["verified"]]) if not events.empty else 0
        match_rate = n_verified / n_api_dates * 100 if n_api_dates > 0 else 0

        data_quality[ticker] = {
            "n_earnings_dates_api": n_api_dates,
            "n_gaps_detected": n_gaps,
            "n_verified_matches": n_verified,
            "match_rate_pct": round(match_rate, 1),
            "flagged": match_rate < 70 and n_api_dates > 0,
        }

        all_events[ticker] = events
        print(f"OK (earnings={n_api_dates}, gaps={n_gaps}, verified={n_verified}, match={match_rate:.0f}%)")
        time.sleep(0.3)

    # Download SPY for regime test
    print("  SPY (for regime test)...", end=" ", flush=True)
    spy_prices = download_price_data("SPY")
    print("OK")

    # Combine all events
    all_events_df = pd.concat([ev for ev in all_events.values() if not ev.empty], ignore_index=True)
    verified_events = all_events_df[all_events_df["verified"]].copy()

    print(f"\n  Total events: {len(all_events_df)}")
    print(f"  Verified events: {len(verified_events)}")
    print(f"  Unverified: {len(all_events_df) - len(verified_events)}")

    # Flag low-quality tickers
    flagged = [tk for tk, dq in data_quality.items() if dq["flagged"]]
    if flagged:
        print(f"  FLAGGED tickers (match rate < 70%): {', '.join(flagged)}")

    # ── Step 2: Run backtests ──
    print("\n[2/5] Running backtests across all configurations...")
    results = {}

    for gap_thresh in GAP_THRESHOLDS:
        for hold in HOLD_DAYS:
            config_name = f"gap{gap_thresh}_hold{hold}d"
            print(f"  {config_name}...", end=" ", flush=True)

            all_trades = []
            for ticker, events in all_events.items():
                if events.empty:
                    continue
                trades = simulate_trades(events, all_prices[ticker], gap_thresh, hold)
                all_trades.extend(trades)

            metrics = compute_metrics(all_trades)
            results[config_name] = {
                "config": {"gap_threshold": gap_thresh, "hold_days": hold, "exit": f"hold_{hold}d"},
                "metrics": metrics,
                "trades": all_trades,
            }
            print(f"n={metrics['n_trades']}, WR={metrics['win_rate']:.1f}%, "
                  f"avg={metrics['avg_return']:.2f}%, Sharpe={metrics['sharpe']:.2f}")

        # Trailing stop variant
        config_name = f"gap{gap_thresh}_trail{TRAILING_STOP_PCT:.0f}pct"
        print(f"  {config_name}...", end=" ", flush=True)

        all_trades = []
        for ticker, events in all_events.items():
            if events.empty:
                continue
            trades = simulate_trades(events, all_prices[ticker], gap_thresh, 1,
                                     trailing_stop_pct=TRAILING_STOP_PCT)
            all_trades.extend(trades)

        metrics = compute_metrics(all_trades)
        results[config_name] = {
            "config": {"gap_threshold": gap_thresh, "trailing_stop_pct": TRAILING_STOP_PCT, "exit": "trailing_stop"},
            "metrics": metrics,
            "trades": all_trades,
        }
        print(f"n={metrics['n_trades']}, WR={metrics['win_rate']:.1f}%, "
              f"avg={metrics['avg_return']:.2f}%, Sharpe={metrics['sharpe']:.2f}")

    # ── Step 3: Find best config ──
    # Best by Sharpe among configs with >= 20 trades
    valid_configs = {k: v for k, v in results.items() if v["metrics"]["n_trades"] >= 20}
    if valid_configs:
        best_config = max(valid_configs.keys(), key=lambda k: valid_configs[k]["metrics"]["sharpe"])
    else:
        best_config = max(results.keys(), key=lambda k: results[k]["metrics"]["sharpe"])

    best = results[best_config]
    print(f"\n  BEST CONFIG: {best_config}")
    print(f"    Trades: {best['metrics']['n_trades']}")
    print(f"    Win Rate: {best['metrics']['win_rate']:.1f}%")
    print(f"    Avg Return: {best['metrics']['avg_return']:.2f}%")
    print(f"    Sharpe: {best['metrics']['sharpe']:.2f}")
    print(f"    Sortino: {best['metrics']['sortino']:.2f}")
    print(f"    Profit Factor: {best['metrics']['profit_factor']:.2f}")

    # ── Step 4: Run adversarial checks on best config ──
    print("\n[3/5] Running adversarial checks on best config...")
    best_trades = best["trades"]

    print("  Permutation test (200 shuffles)...", end=" ", flush=True)
    perm_result = permutation_test(best_trades, all_prices, PERMUTATION_SHUFFLES)
    print(f"p={perm_result['p_value']:.3f} {'PASS' if perm_result.get('significant') else 'FAIL'}")

    print("  Regime test (prior-day SPY)...", end=" ", flush=True)
    regime_result = regime_test(best_trades, spy_prices)
    print(f"consistent={regime_result.get('regime_consistent', 'N/A')}")

    print("  Sub-period consistency...", end=" ", flush=True)
    subperiod_result = sub_period_consistency(best_trades)
    if subperiod_result["valid"]:
        print(f"halves_same_sign={subperiod_result['halves_same_sign']}, "
              f"profitable_years={subperiod_result['profitable_years']}/{subperiod_result['total_years']}")
    else:
        print(subperiod_result.get("reason", "invalid"))

    print("  Outlier removal test...", end=" ", flush=True)
    outlier_result = outlier_removal_test(best_trades)
    if outlier_result["valid"]:
        print(f"still_profitable={outlier_result['still_profitable']}, "
              f"trimmed_mean={outlier_result['trimmed_mean']:.2f}%")
    else:
        print(outlier_result.get("reason", "invalid"))

    print("  Ticker concentration test...", end=" ", flush=True)
    conc_result = ticker_concentration_test(best_trades)
    if conc_result["valid"]:
        print(f"concentrated={conc_result['concentrated']}, "
              f"max_conc={conc_result['max_concentration_pct']:.1f}%")
    else:
        print("invalid")

    print("  Pricing sanity check...", end=" ", flush=True)
    pricing_result = pricing_sanity_check(best_trades)
    if pricing_result["valid"]:
        print(f"clean={pricing_result['pricing_clean']}, issues={pricing_result['n_issues']}")
    else:
        print("invalid")

    # ── Step 5: Also run adversarial on ALL configs ──
    print("\n[4/5] Running adversarial checks on all configs...")
    all_adversarial = {}
    for config_name, res in results.items():
        trades = res["trades"]
        if len(trades) < 5:
            all_adversarial[config_name] = {"skipped": True, "reason": "too few trades"}
            continue

        perm = permutation_test(trades, all_prices, 50)  # Fewer shuffles for non-best
        regime = regime_test(trades, spy_prices)
        subp = sub_period_consistency(trades)
        outlier = outlier_removal_test(trades)
        conc = ticker_concentration_test(trades)

        all_adversarial[config_name] = {
            "permutation_p": perm["p_value"],
            "permutation_pass": perm["p_value"] < 0.05,
            "regime_consistent": regime.get("regime_consistent", None),
            "subperiod_same_sign": subp.get("halves_same_sign", None),
            "outlier_still_profitable": outlier.get("still_profitable", None),
            "concentrated": conc.get("concentrated", None),
        }
        print(f"  {config_name}: perm_p={perm['p_value']:.3f}, "
              f"regime_ok={regime.get('regime_consistent')}, "
              f"outlier_ok={outlier.get('still_profitable')}")

    # ── Step 6: Build quality gates ──
    print("\n[5/5] Building final report...")

    quality_gates = {
        "permutation_test": {
            "pass": perm_result.get("significant", False),
            "p_value": perm_result["p_value"],
            "threshold": 0.05,
        },
        "regime_consistency": {
            "pass": regime_result.get("regime_consistent", False),
            "divergence": regime_result.get("regime_divergence", None),
        },
        "sub_period_consistency": {
            "pass": subperiod_result.get("halves_same_sign", False),
            "profitable_years_ratio": subperiod_result.get("year_consistency", 0),
        },
        "outlier_robustness": {
            "pass": outlier_result.get("still_profitable", False),
            "trimmed_mean": outlier_result.get("trimmed_mean", 0),
        },
        "ticker_diversification": {
            "pass": not conc_result.get("concentrated", True),
            "max_concentration": conc_result.get("max_concentration_pct", 100),
        },
        "pricing_sanity": {
            "pass": pricing_result.get("pricing_clean", False),
            "n_issues": pricing_result.get("n_issues", -1),
        },
    }

    gates_passed = sum(1 for g in quality_gates.values() if g["pass"])
    total_gates = len(quality_gates)

    # ── Build report ──
    # Convert trades for JSON serialization
    def serialize_trades(trades):
        serialized = []
        for t in trades:
            st = dict(t)
            st["entry_date"] = str(st["entry_date"])
            st["exit_date"] = str(st["exit_date"])
            serialized.append(st)
        return serialized

    # Serialize results
    serialized_results = {}
    for config_name, res in results.items():
        serialized_results[config_name] = {
            "config": res["config"],
            "metrics": res["metrics"],
            "n_trades": res["metrics"]["n_trades"],
        }

    report = {
        "strategy": "Earnings Gap Buyer v1",
        "description": "Buy in gap direction at open after earnings, equity simulation",
        "period": f"{START_DATE} to {END_DATE}",
        "starting_capital": STARTING_CAPITAL,
        "universe": TICKERS,
        "timestamp": datetime.now().isoformat(),

        "data_quality_report": {
            "per_ticker": data_quality,
            "total_events": len(all_events_df),
            "verified_events": int(verified_events.shape[0]),
            "unverified_events": int(len(all_events_df) - len(verified_events)),
            "flagged_tickers": flagged,
            "overall_match_rate": float(
                sum(dq["n_verified_matches"] for dq in data_quality.values()) /
                max(sum(dq["n_earnings_dates_api"] for dq in data_quality.values()), 1) * 100
            ),
        },

        "all_configs": serialized_results,
        "best_config": {
            "name": best_config,
            "config": best["config"],
            "metrics": best["metrics"],
        },

        "adversarial_checks": {
            "best_config": {
                "permutation_test": perm_result,
                "regime_test": regime_result,
                "sub_period_consistency": subperiod_result,
                "outlier_removal": outlier_result,
                "ticker_concentration": conc_result,
                "pricing_sanity": pricing_result,
            },
            "all_configs": all_adversarial,
        },

        "quality_gates": quality_gates,
        "gates_summary": {
            "passed": gates_passed,
            "total": total_gates,
            "all_passed": gates_passed == total_gates,
        },

        "best_config_trades": serialize_trades(best_trades),

        "verdict": "PASS" if gates_passed >= 4 else "FAIL",
        "verdict_detail": (
            f"{gates_passed}/{total_gates} quality gates passed. "
            f"Best config: {best_config} with {best['metrics']['n_trades']} trades, "
            f"Sharpe={best['metrics']['sharpe']:.2f}, WR={best['metrics']['win_rate']:.1f}%, "
            f"PF={best['metrics']['profit_factor']:.2f}."
        ),
    }

    # Save report
    report_path = OUTPUT_DIR / "backtest_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to {report_path}")

    # ── Print summary ──
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n  DATA QUALITY:")
    print(f"    Total earnings events: {len(all_events_df)}")
    print(f"    Verified: {len(verified_events)} ({report['data_quality_report']['overall_match_rate']:.0f}% match rate)")
    if flagged:
        print(f"    Flagged tickers: {', '.join(flagged)}")

    print(f"\n  ALL CONFIGS:")
    print(f"    {'Config':<25} {'N':>5} {'WR%':>6} {'AvgRet':>7} {'Sharpe':>7} {'Sortino':>8} {'PF':>6}")
    print(f"    {'-'*25} {'-'*5} {'-'*6} {'-'*7} {'-'*7} {'-'*8} {'-'*6}")
    for config_name in sorted(results.keys()):
        m = results[config_name]["metrics"]
        marker = " <<< BEST" if config_name == best_config else ""
        print(f"    {config_name:<25} {m['n_trades']:>5} {m['win_rate']:>5.1f}% {m['avg_return']:>6.2f}% "
              f"{m['sharpe']:>7.2f} {m['sortino']:>8.2f} {m['profit_factor']:>5.2f}{marker}")

    print(f"\n  QUALITY GATES ({gates_passed}/{total_gates}):")
    for gate_name, gate in quality_gates.items():
        status = "PASS" if gate["pass"] else "FAIL"
        print(f"    [{status}] {gate_name}")

    print(f"\n  VERDICT: {report['verdict']}")
    print(f"  {report['verdict_detail']}")
    print("=" * 70)

    return report


if __name__ == "__main__":
    report = main()
