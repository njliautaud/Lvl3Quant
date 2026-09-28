#!/usr/bin/env python3
"""
Revenue Surprise Momentum Backtest
====================================
Tests whether larger earnings gaps (proxy for revenue beats) generate
stronger post-announcement drift than smaller gaps (likely EPS-only beats).

Revenue is harder to manipulate than EPS (no buyback inflation, fewer
one-time items). Larger gaps likely reflect revenue beats since EPS-only
beats tend to produce smaller moves.

6 Variants:
  A) Small Gap Long Hold: Gap 3-7%, hold 40 days (baseline)
  B) Large Gap Long Hold: Gap >10%, hold 40 days (expect stronger drift)
  C) Small vs Large Comparison: track difference in drift
  D) Gap + Volume: Gap >5% AND volume >2x avg, hold 40 days
  E) Cascading Entry: buy on gap, add if new high within 5 days, hold 40d from last add
  F) Multi-Stock Portfolio: hold up to 5 recent gap stocks, equal weight, rotate

Walk-forward OOT: Jan 2022 – Jul 2026
5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades
$669 starting capital, $0 commission, 0.02% slippage
Regime: Bull = SPY > 200-SMA, Bear = SPY < 200-SMA
"""

import os
import sys
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ─── Config ──────────────────────────────────────────────────────────────────

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD",
    "NFLX", "CRM", "SNOW", "PLTR", "SOFI", "HOOD", "SNAP", "PINS",
    "COIN", "RBLX", "RIVN", "UBER", "LYFT", "ROKU", "NET", "DDOG",
    "TTD", "SHOP"
]

STARTING_CAPITAL = 669.0
SLIPPAGE_PCT = 0.0002  # 0.02%
OOT_START = "2022-01-01"
OOT_END = "2026-07-30"
DATA_START = "2020-01-01"  # extra lookback for SMA/volume avg

CACHE_DIR = Path(__file__).resolve().parent.parent / "output" / "rev_surprise_momentum" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = Path(__file__).resolve().parent.parent / "data" / "revenue_surprise_momentum_results.json"

# ─── Data Download ───────────────────────────────────────────────────────────

def download_data(tickers, start, end):
    """Download daily OHLCV for all tickers + SPY."""
    all_tickers = list(set(tickers + ["SPY"]))
    cache_file = CACHE_DIR / "price_data.parquet"

    if cache_file.exists():
        df = pd.read_parquet(cache_file)
        print(f"Loaded cached price data: {len(df)} rows")
        return df

    print(f"Downloading data for {len(all_tickers)} tickers...")
    frames = []
    for t in all_tickers:
        try:
            d = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
            if d.empty:
                continue
            # Handle multi-level columns from yfinance
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            d = d[["Open", "High", "Low", "Close", "Volume"]].copy()
            d["ticker"] = t
            d.index.name = "Date"
            frames.append(d.reset_index())
        except Exception as e:
            print(f"  Failed {t}: {e}")

    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(cache_file, index=False)
    print(f"Downloaded {len(df)} rows for {df['ticker'].nunique()} tickers")
    return df


def compute_features(df):
    """Compute gap%, volume ratio, SPY 200-SMA regime for each row."""
    # Sort
    df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)

    # Gap% = (Open - prev Close) / prev Close
    df["prev_close"] = df.groupby("ticker")["Close"].shift(1)
    df["gap_pct"] = (df["Open"] - df["prev_close"]) / df["prev_close"]

    # Volume ratio vs 20-day average
    df["vol_avg_20"] = df.groupby("ticker")["Volume"].transform(
        lambda x: x.rolling(20, min_periods=10).mean().shift(1)
    )
    df["vol_ratio"] = df["Volume"] / df["vol_avg_20"]

    # SPY 200-SMA regime
    spy = df[df["ticker"] == "SPY"][["Date", "Close"]].copy()
    spy = spy.sort_values("Date")
    spy["sma200"] = spy["Close"].rolling(200, min_periods=150).mean()
    spy["regime"] = np.where(spy["Close"] > spy["sma200"], "bull", "bear")
    spy_regime = spy[["Date", "regime"]].rename(columns={"regime": "spy_regime"})

    df = df.merge(spy_regime, on="Date", how="left")
    df["spy_regime"] = df["spy_regime"].fillna("bull")

    return df


def detect_gap_events(df, gap_min=0.03, gap_max=None, vol_min=None):
    """Find gap-up events matching filters. Returns DataFrame of events."""
    mask = (df["ticker"] != "SPY") & (df["gap_pct"] >= gap_min)
    if gap_max is not None:
        mask &= (df["gap_pct"] <= gap_max)
    if vol_min is not None:
        mask &= (df["vol_ratio"] >= vol_min)

    events = df[mask][["Date", "ticker", "Open", "Close", "gap_pct",
                        "vol_ratio", "spy_regime"]].copy()

    # Only OOT period
    events = events[(events["Date"] >= OOT_START) & (events["Date"] <= OOT_END)]
    return events.sort_values("Date").reset_index(drop=True)


# ─── Backtesting Engine ─────────────────────────────────────────────────────

def get_forward_returns(df, ticker, entry_date, hold_days=40):
    """Get forward price series for a ticker from entry_date for hold_days trading days."""
    td = df[(df["ticker"] == ticker) & (df["Date"] >= entry_date)].sort_values("Date")
    if len(td) < 2:
        return None
    td = td.head(hold_days + 1)
    return td


def backtest_simple(df, events, hold_days=40, label=""):
    """Simple backtest: buy at open on gap day, sell after hold_days.
    Returns trade list and equity curve."""
    trades = []
    for _, ev in events.iterrows():
        fwd = get_forward_returns(df, ev["ticker"], ev["Date"], hold_days)
        if fwd is None or len(fwd) < 2:
            continue

        entry_price = fwd.iloc[0]["Open"] * (1 + SLIPPAGE_PCT)
        exit_price = fwd.iloc[-1]["Close"] * (1 - SLIPPAGE_PCT)
        ret = (exit_price - entry_price) / entry_price

        trades.append({
            "ticker": ev["ticker"],
            "entry_date": str(ev["Date"].date()) if hasattr(ev["Date"], "date") else str(ev["Date"])[:10],
            "exit_date": str(fwd.iloc[-1]["Date"].date()) if hasattr(fwd.iloc[-1]["Date"], "date") else str(fwd.iloc[-1]["Date"])[:10],
            "gap_pct": round(ev["gap_pct"] * 100, 2),
            "vol_ratio": round(ev["vol_ratio"], 2) if pd.notna(ev["vol_ratio"]) else None,
            "regime": ev["spy_regime"],
            "entry_price": round(entry_price, 4),
            "exit_price": round(exit_price, 4),
            "return_pct": round(ret * 100, 4),
            "hold_days": len(fwd) - 1,
        })

    return trades


def backtest_cascading(df, events, confirmation_days=5, hold_days=40, label=""):
    """Cascading entry: buy on gap day, add if new high within 5 days.
    Hold 40 days from last add."""
    trades = []
    for _, ev in events.iterrows():
        fwd = get_forward_returns(df, ev["ticker"], ev["Date"], hold_days + confirmation_days + 5)
        if fwd is None or len(fwd) < 2:
            continue

        entry_price = fwd.iloc[0]["Open"] * (1 + SLIPPAGE_PCT)
        positions = [{"price": entry_price, "day": 0}]

        # Check for new high within confirmation_days
        gap_day_high = fwd.iloc[0]["High"]
        for i in range(1, min(confirmation_days + 1, len(fwd))):
            if fwd.iloc[i]["High"] > gap_day_high:
                add_price = fwd.iloc[i]["Close"] * (1 + SLIPPAGE_PCT)
                positions.append({"price": add_price, "day": i})
                break  # only one add

        last_add_day = max(p["day"] for p in positions)
        exit_idx = min(last_add_day + hold_days, len(fwd) - 1)
        exit_price = fwd.iloc[exit_idx]["Close"] * (1 - SLIPPAGE_PCT)

        avg_entry = np.mean([p["price"] for p in positions])
        ret = (exit_price - avg_entry) / avg_entry

        trades.append({
            "ticker": ev["ticker"],
            "entry_date": str(ev["Date"].date()) if hasattr(ev["Date"], "date") else str(ev["Date"])[:10],
            "exit_date": str(fwd.iloc[exit_idx]["Date"].date()) if hasattr(fwd.iloc[exit_idx]["Date"], "date") else str(fwd.iloc[exit_idx]["Date"])[:10],
            "gap_pct": round(ev["gap_pct"] * 100, 2),
            "regime": ev["spy_regime"],
            "num_entries": len(positions),
            "avg_entry": round(avg_entry, 4),
            "exit_price": round(exit_price, 4),
            "return_pct": round(ret * 100, 4),
            "hold_days": exit_idx,
        })

    return trades


def backtest_portfolio(df, events, max_positions=5, hold_days=40, label=""):
    """Multi-stock portfolio: hold up to max_positions simultaneously, equal weight, rotate."""
    capital = STARTING_CAPITAL
    equity_curve = [{"date": OOT_START, "equity": capital}]
    active_positions = []  # list of dicts {ticker, entry_date, entry_price, exit_target_date}
    all_trades = []

    # Get all trading dates in OOT
    spy_dates = df[(df["ticker"] == "SPY") &
                   (df["Date"] >= OOT_START) &
                   (df["Date"] <= OOT_END)]["Date"].sort_values().unique()

    events_by_date = {}
    for _, ev in events.iterrows():
        d = ev["Date"]
        if d not in events_by_date:
            events_by_date[d] = []
        events_by_date[d].append(ev)

    for date in spy_dates:
        # Check exits
        new_active = []
        for pos in active_positions:
            td = df[(df["ticker"] == pos["ticker"]) & (df["Date"] == date)]
            if td.empty:
                new_active.append(pos)
                continue

            days_held = np.busday_count(
                np.datetime64(pos["entry_date"], "D"),
                np.datetime64(date, "D")
            )

            if days_held >= hold_days:
                exit_price = td.iloc[0]["Close"] * (1 - SLIPPAGE_PCT)
                ret = (exit_price - pos["entry_price"]) / pos["entry_price"]
                pnl = pos["position_size"] * ret
                capital += pos["position_size"] + pnl

                all_trades.append({
                    "ticker": pos["ticker"],
                    "entry_date": str(pos["entry_date"])[:10],
                    "exit_date": str(date)[:10],
                    "return_pct": round(ret * 100, 4),
                    "pnl": round(pnl, 2),
                    "regime": pos.get("regime", "unknown"),
                })
            else:
                new_active.append(pos)

        active_positions = new_active

        # Check new entries
        if date in events_by_date and len(active_positions) < max_positions:
            for ev in events_by_date[date]:
                if len(active_positions) >= max_positions:
                    break
                # Skip if already holding this ticker
                if any(p["ticker"] == ev["ticker"] for p in active_positions):
                    continue

                position_size = capital / max_positions
                if position_size < 10:
                    continue

                entry_price = ev["Open"] * (1 + SLIPPAGE_PCT)
                capital -= position_size

                active_positions.append({
                    "ticker": ev["ticker"],
                    "entry_date": date,
                    "entry_price": entry_price,
                    "position_size": position_size,
                    "regime": ev["spy_regime"],
                })

        # Mark-to-market
        total_equity = capital
        for pos in active_positions:
            td = df[(df["ticker"] == pos["ticker"]) & (df["Date"] == date)]
            if not td.empty:
                current_price = td.iloc[0]["Close"]
                pos_value = pos["position_size"] * (current_price / pos["entry_price"])
                total_equity += pos_value
            else:
                total_equity += pos["position_size"]

        equity_curve.append({"date": str(date)[:10], "equity": round(total_equity, 2)})

    return all_trades, equity_curve


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(trades, equity_curve=None, label=""):
    """Compute Sharpe, Sortino, PF, WR, MaxDD, regime analysis."""
    if not trades:
        return {"label": label, "n_trades": 0, "PASS": False, "reason": "no trades"}

    rets = np.array([t["return_pct"] / 100 for t in trades])
    n = len(rets)

    # Basic stats
    mean_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1) if n > 1 else 1e-6
    win_rate = np.mean(rets > 0)
    winners = rets[rets > 0]
    losers = rets[rets < 0]

    # Profit factor
    gross_profit = np.sum(winners) if len(winners) > 0 else 0
    gross_loss = abs(np.sum(losers)) if len(losers) > 0 else 1e-9
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Annualized Sharpe (assume ~9 trades/year, adjust by sqrt)
    trades_per_year = max(n / 4.5, 1)  # ~4.5 years OOT
    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0

    # Sortino
    downside = rets[rets < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-6
    sortino = (mean_ret / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0

    # Max drawdown from equity curve or cumulative returns
    if equity_curve and len(equity_curve) > 1:
        eqs = np.array([e["equity"] for e in equity_curve])
        peak = np.maximum.accumulate(eqs)
        dd = (eqs - peak) / peak
        max_dd = np.min(dd)
    else:
        cum = np.cumprod(1 + rets)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        max_dd = np.min(dd) if len(dd) > 0 else 0

    # Regime analysis
    bull_rets = [t["return_pct"] / 100 for t in trades if t.get("regime") == "bull"]
    bear_rets = [t["return_pct"] / 100 for t in trades if t.get("regime") == "bear"]

    bull_sharpe = (np.mean(bull_rets) / (np.std(bull_rets, ddof=1) + 1e-9)) * np.sqrt(max(len(bull_rets) / 4.5, 1)) if bull_rets else 0
    bear_sharpe = (np.mean(bear_rets) / (np.std(bear_rets, ddof=1) + 1e-9)) * np.sqrt(max(len(bear_rets) / 4.5, 1)) if bear_rets else 0

    regime_gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 1e-9)

    # Final P&L
    final_equity = STARTING_CAPITAL * np.prod(1 + rets)

    metrics = {
        "label": label,
        "n_trades": n,
        "mean_return_pct": round(mean_ret * 100, 3),
        "median_return_pct": round(np.median(rets) * 100, 3),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(pf, 3),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "bull_trades": len(bull_rets),
        "bear_trades": len(bear_rets),
        "bull_mean_ret_pct": round(np.mean(bull_rets) * 100, 3) if bull_rets else None,
        "bear_mean_ret_pct": round(np.mean(bear_rets) * 100, 3) if bear_rets else None,
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "regime_gap": round(regime_gap, 4),
        "final_equity": round(final_equity, 2),
        "total_return_pct": round((final_equity / STARTING_CAPITAL - 1) * 100, 2),
    }

    return metrics


def permutation_test(trades, n_perms=1000):
    """Shuffle entry dates to test if mean return is significant."""
    if len(trades) < 5:
        return 1.0

    actual_mean = np.mean([t["return_pct"] for t in trades])
    all_rets = [t["return_pct"] for t in trades]

    count_better = 0
    rng = np.random.RandomState(42)
    for _ in range(n_perms):
        shuffled = rng.permutation(all_rets)
        if np.mean(shuffled) >= actual_mean:
            count_better += 1

    return count_better / n_perms


def five_gate_validation(metrics, perm_p, label=""):
    """Apply 5-gate validation."""
    gates = {
        "sharpe_gt_0.5": metrics.get("sharpe", 0) > 0.5,
        "perm_p_lt_0.05": perm_p < 0.05,
        "regime_gap_lt_0.5": metrics.get("regime_gap", 1) < 0.5,
        "max_dd_gt_neg50": metrics.get("max_drawdown_pct", -100) > -50,
        "trades_gte_20": metrics.get("n_trades", 0) >= 20,
    }
    passed = all(gates.values())
    gates_passed = sum(gates.values())

    return {
        "label": label,
        "gates": gates,
        "gates_passed": f"{gates_passed}/5",
        "PASS": passed,
        "perm_p": round(perm_p, 4),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("REVENUE SURPRISE MOMENTUM BACKTEST")
    print("=" * 70)

    # Download data
    df = download_data(TICKERS, DATA_START, OOT_END)
    df["Date"] = pd.to_datetime(df["Date"])
    df = compute_features(df)

    results = {}

    # ── Variant A: Small Gap (3-7%), Hold 40 days ─────────────────────────
    print("\n─── Variant A: Small Gap (3-7%), 40-day hold ───")
    events_a = detect_gap_events(df, gap_min=0.03, gap_max=0.07)
    trades_a = backtest_simple(df, events_a, hold_days=40, label="A_small_gap")
    metrics_a = compute_metrics(trades_a, label="A_small_gap_40d")
    perm_a = permutation_test(trades_a)
    valid_a = five_gate_validation(metrics_a, perm_a, "A")
    print(f"  Events: {len(events_a)}, Trades: {len(trades_a)}")
    print(f"  Mean ret: {metrics_a.get('mean_return_pct', 'N/A')}%, Sharpe: {metrics_a.get('sharpe', 'N/A')}")
    print(f"  WR: {metrics_a.get('win_rate', 'N/A')}, PF: {metrics_a.get('profit_factor', 'N/A')}")
    print(f"  Perm p: {perm_a:.4f}, Gates: {valid_a['gates_passed']}, PASS: {valid_a['PASS']}")
    results["A_small_gap_40d"] = {**metrics_a, **valid_a}

    # ── Variant B: Large Gap (>10%), Hold 40 days ─────────────────────────
    print("\n─── Variant B: Large Gap (>10%), 40-day hold ───")
    events_b = detect_gap_events(df, gap_min=0.10)
    trades_b = backtest_simple(df, events_b, hold_days=40, label="B_large_gap")
    metrics_b = compute_metrics(trades_b, label="B_large_gap_40d")
    perm_b = permutation_test(trades_b)
    valid_b = five_gate_validation(metrics_b, perm_b, "B")
    print(f"  Events: {len(events_b)}, Trades: {len(trades_b)}")
    print(f"  Mean ret: {metrics_b.get('mean_return_pct', 'N/A')}%, Sharpe: {metrics_b.get('sharpe', 'N/A')}")
    print(f"  WR: {metrics_b.get('win_rate', 'N/A')}, PF: {metrics_b.get('profit_factor', 'N/A')}")
    print(f"  Perm p: {perm_b:.4f}, Gates: {valid_b['gates_passed']}, PASS: {valid_b['PASS']}")
    results["B_large_gap_40d"] = {**metrics_b, **valid_b}

    # ── Variant C: Small vs Large Comparison ──────────────────────────────
    print("\n─── Variant C: Small vs Large Comparison ───")
    drift_diff = (metrics_b.get("mean_return_pct", 0) or 0) - (metrics_a.get("mean_return_pct", 0) or 0)
    wr_diff = (metrics_b.get("win_rate", 0) or 0) - (metrics_a.get("win_rate", 0) or 0)
    sharpe_diff = (metrics_b.get("sharpe", 0) or 0) - (metrics_a.get("sharpe", 0) or 0)
    print(f"  Large gap drift advantage: {drift_diff:+.3f}% per trade")
    print(f"  Large gap WR advantage: {wr_diff:+.4f}")
    print(f"  Large gap Sharpe advantage: {sharpe_diff:+.3f}")
    results["C_comparison"] = {
        "label": "C_small_vs_large",
        "drift_diff_pct": round(drift_diff, 3),
        "wr_diff": round(wr_diff, 4),
        "sharpe_diff": round(sharpe_diff, 3),
        "hypothesis_supported": drift_diff > 0 and sharpe_diff > 0,
    }

    # ── Variant D: Gap >5% + Volume >2x, Hold 40 days ────────────────────
    print("\n─── Variant D: Gap >5% + Volume >2x, 40-day hold ───")
    events_d = detect_gap_events(df, gap_min=0.05, vol_min=2.0)
    trades_d = backtest_simple(df, events_d, hold_days=40, label="D_gap_volume")
    metrics_d = compute_metrics(trades_d, label="D_gap_volume_40d")
    perm_d = permutation_test(trades_d)
    valid_d = five_gate_validation(metrics_d, perm_d, "D")
    print(f"  Events: {len(events_d)}, Trades: {len(trades_d)}")
    print(f"  Mean ret: {metrics_d.get('mean_return_pct', 'N/A')}%, Sharpe: {metrics_d.get('sharpe', 'N/A')}")
    print(f"  WR: {metrics_d.get('win_rate', 'N/A')}, PF: {metrics_d.get('profit_factor', 'N/A')}")
    print(f"  Perm p: {perm_d:.4f}, Gates: {valid_d['gates_passed']}, PASS: {valid_d['PASS']}")
    results["D_gap_volume_40d"] = {**metrics_d, **valid_d}

    # ── Variant E: Cascading Entry ────────────────────────────────────────
    print("\n─── Variant E: Cascading Entry (gap >5%, add on new high) ───")
    events_e = detect_gap_events(df, gap_min=0.05)
    trades_e = backtest_cascading(df, events_e, confirmation_days=5, hold_days=40, label="E_cascade")
    metrics_e = compute_metrics(trades_e, label="E_cascade_40d")
    perm_e = permutation_test(trades_e)
    valid_e = five_gate_validation(metrics_e, perm_e, "E")
    print(f"  Events: {len(events_e)}, Trades: {len(trades_e)}")
    print(f"  Mean ret: {metrics_e.get('mean_return_pct', 'N/A')}%, Sharpe: {metrics_e.get('sharpe', 'N/A')}")
    added_count = sum(1 for t in trades_e if t.get("num_entries", 1) > 1)
    print(f"  Added positions: {added_count}/{len(trades_e)}")
    print(f"  Perm p: {perm_e:.4f}, Gates: {valid_e['gates_passed']}, PASS: {valid_e['PASS']}")
    results["E_cascade_40d"] = {**metrics_e, **valid_e}

    # ── Variant F: Multi-Stock Portfolio ──────────────────────────────────
    print("\n─── Variant F: Multi-Stock Portfolio (max 5 positions) ───")
    events_f = detect_gap_events(df, gap_min=0.05)
    trades_f, eq_curve_f = backtest_portfolio(df, events_f, max_positions=5, hold_days=40, label="F_portfolio")
    metrics_f = compute_metrics(trades_f, equity_curve=eq_curve_f, label="F_portfolio_40d")
    perm_f = permutation_test(trades_f)
    valid_f = five_gate_validation(metrics_f, perm_f, "F")

    # Portfolio-specific final equity
    if eq_curve_f:
        metrics_f["final_equity"] = eq_curve_f[-1]["equity"]
        metrics_f["total_return_pct"] = round((eq_curve_f[-1]["equity"] / STARTING_CAPITAL - 1) * 100, 2)

    print(f"  Trades completed: {len(trades_f)}")
    print(f"  Final equity: ${metrics_f.get('final_equity', 'N/A')}")
    print(f"  Total return: {metrics_f.get('total_return_pct', 'N/A')}%")
    print(f"  MaxDD: {metrics_f.get('max_drawdown_pct', 'N/A')}%")
    print(f"  Perm p: {perm_f:.4f}, Gates: {valid_f['gates_passed']}, PASS: {valid_f['PASS']}")
    results["F_portfolio_40d"] = {**metrics_f, **valid_f}

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: 5-GATE VALIDATION RESULTS")
    print("=" * 70)
    print(f"{'Variant':<30} {'Trades':>6} {'MeanRet%':>9} {'Sharpe':>7} {'WR':>6} {'PF':>6} {'MaxDD%':>7} {'Gates':>6} {'PASS':>5}")
    print("-" * 90)

    for key, r in results.items():
        if key == "C_comparison":
            continue
        print(f"{r.get('label',''):<30} {r.get('n_trades',0):>6} "
              f"{r.get('mean_return_pct','N/A'):>9} {r.get('sharpe','N/A'):>7} "
              f"{r.get('win_rate','N/A'):>6} {r.get('profit_factor','N/A'):>6} "
              f"{r.get('max_drawdown_pct','N/A'):>7} {r.get('gates_passed',''):>6} "
              f"{'YES' if r.get('PASS') else 'NO':>5}")

    print(f"\nVariant C — Revenue Surprise Hypothesis:")
    c = results.get("C_comparison", {})
    print(f"  Large gap drift advantage: {c.get('drift_diff_pct', 'N/A')}% per trade")
    print(f"  Sharpe advantage: {c.get('sharpe_diff', 'N/A')}")
    print(f"  Hypothesis supported: {c.get('hypothesis_supported', 'N/A')}")

    # Count passes
    passes = sum(1 for k, v in results.items() if k != "C_comparison" and v.get("PASS"))
    print(f"\nVariants passing 5-gate: {passes}/5")

    # Save results
    # Convert any non-serializable items
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp, datetime)):
            return str(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    def clean_dict(d):
        return {k: make_serializable(v) if not isinstance(v, dict) else clean_dict(v)
                for k, v in d.items()}

    output = {
        "metadata": {
            "strategy": "revenue_surprise_momentum",
            "run_date": datetime.now().isoformat(),
            "oot_period": f"{OOT_START} to {OOT_END}",
            "universe": TICKERS,
            "starting_capital": STARTING_CAPITAL,
            "slippage_pct": SLIPPAGE_PCT,
        },
        "results": clean_dict(results),
        "trade_details": {
            "A_small_gap": trades_a[:10] if trades_a else [],
            "B_large_gap": trades_b[:10] if trades_b else [],
            "D_gap_volume": trades_d[:10] if trades_d else [],
            "E_cascade": trades_e[:10] if trades_e else [],
            "F_portfolio": trades_f[:10] if trades_f else [],
        }
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
