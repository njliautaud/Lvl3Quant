#!/usr/bin/env python3
"""
Multi-Timeframe Relaxed Sweep Backtest
---------------------------------------
Sweeps 12 variants of weekly filter strictness on a daily mean-reversion signal
to find the optimal Sharpe vs trade-count balance.

Base daily signal: Stock drops >5% from 20d high + RSI(14)<35 + first green after 3+ consecutive red days.
Weekly filters: Various combinations of weekly RSI and distance from 10-week high.

OOT: Jan 2022 – Jul 2026 | Capital: $645 | Max per trade: $200 | Max concurrent: 3
Slippage: 0.02% each way
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

# ── Configuration ──────────────────────────────────────────────────────────────

UNIVERSE = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]

START_DATE = "2021-06-01"  # extra lookback for indicators
OOT_START = "2022-01-01"
OOT_END = "2026-07-31"
CAPITAL = 645.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
SLIPPAGE_PCT = 0.0002  # 0.02% each way
PERM_ITERATIONS = 1000

VARIANTS = {
    "A": {"desc": "No weekly filter (baseline)", "weekly_rsi": None, "weekly_drop": None, "weekly_neg_ret": False, "weekly_rsi_declining": False, "hold_days": 10},
    "B": {"desc": "Weekly RSI < 50", "weekly_rsi": 50, "weekly_drop": None, "weekly_neg_ret": False, "weekly_rsi_declining": False, "hold_days": 10},
    "C": {"desc": "Weekly RSI < 45", "weekly_rsi": 45, "weekly_drop": None, "weekly_neg_ret": False, "weekly_rsi_declining": False, "hold_days": 10},
    "D": {"desc": "Weekly RSI < 40, hold 15d", "weekly_rsi": 40, "weekly_drop": None, "weekly_neg_ret": False, "weekly_rsi_declining": False, "hold_days": 15},
    "E": {"desc": "Weekly RSI < 45 + >5% below 10wk high", "weekly_rsi": 45, "weekly_drop": 0.05, "weekly_neg_ret": False, "weekly_rsi_declining": False, "hold_days": 10},
    "F": {"desc": "Weekly RSI < 40 + >5% below 10wk high, hold 15d", "weekly_rsi": 40, "weekly_drop": 0.05, "weekly_neg_ret": False, "weekly_rsi_declining": False, "hold_days": 15},
    "G": {"desc": "Weekly RSI < 45 + >7% below 10wk high, hold 15d", "weekly_rsi": 45, "weekly_drop": 0.07, "weekly_neg_ret": False, "weekly_rsi_declining": False, "hold_days": 15},
    "H": {"desc": "Weekly RSI < 40 + >7% below 10wk high, hold 15d", "weekly_rsi": 40, "weekly_drop": 0.07, "weekly_neg_ret": False, "weekly_rsi_declining": False, "hold_days": 15},
    "I": {"desc": "Weekly RSI < 50 + >3% below 10wk high", "weekly_rsi": 50, "weekly_drop": 0.03, "weekly_neg_ret": False, "weekly_rsi_declining": False, "hold_days": 10},
    "J": {"desc": "Weekly RSI < 45 + negative weekly return", "weekly_rsi": 45, "weekly_drop": None, "weekly_neg_ret": True, "weekly_rsi_declining": False, "hold_days": 10},
    "K": {"desc": "Weekly RSI < 40 + negative weekly return", "weekly_rsi": 40, "weekly_drop": None, "weekly_neg_ret": True, "weekly_rsi_declining": False, "hold_days": 10},
    "L": {"desc": "Weekly RSI declining 2+ weeks", "weekly_rsi": None, "weekly_drop": None, "weekly_neg_ret": False, "weekly_rsi_declining": True, "hold_days": 10},
}


# ── Indicator Functions ────────────────────────────────────────────────────────

def compute_rsi(series, period=14):
    """Compute RSI on a price series."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def prepare_daily_data(df):
    """Add daily indicators to a stock dataframe."""
    df = df.copy()
    df["rsi_14"] = compute_rsi(df["Close"], 14)
    df["high_20d"] = df["Close"].rolling(20).max()
    df["pct_from_20d_high"] = (df["Close"] - df["high_20d"]) / df["high_20d"]
    df["daily_return"] = df["Close"].pct_change()
    df["green"] = df["Close"] > df["Open"]

    # Count consecutive red days before each bar
    red = ~df["green"]
    consec_red = []
    count = 0
    for r in red:
        if r:
            count += 1
        else:
            consec_red.append(count)
            count = 0
            continue
        consec_red.append(0)  # placeholder for red days
    # We need: on a green day, how many consecutive red days preceded it
    consec_before = []
    count = 0
    for i, g in enumerate(df["green"].values):
        if not g:
            count += 1
            consec_before.append(0)
        else:
            consec_before.append(count)
            count = 0
    df["consec_red_before"] = consec_before
    return df


def prepare_weekly_data(df):
    """Resample daily close to weekly and compute weekly indicators."""
    weekly = df["Close"].resample("W-FRI").last().dropna()
    w = pd.DataFrame({"close": weekly})
    w["rsi_14"] = compute_rsi(w["close"], 14)
    w["high_10w"] = w["close"].rolling(10).max()
    w["pct_from_10w_high"] = (w["close"] - w["high_10w"]) / w["high_10w"]
    w["weekly_return"] = w["close"].pct_change()
    w["rsi_declining"] = (w["rsi_14"] < w["rsi_14"].shift(1)) & (w["rsi_14"].shift(1) < w["rsi_14"].shift(2))
    return w


def get_weekly_values_for_date(weekly_df, date):
    """Get the most recent weekly values as of a given date."""
    mask = weekly_df.index <= date
    if mask.sum() == 0:
        return None
    return weekly_df.loc[mask].iloc[-1]


# ── Signal Detection ──────────────────────────────────────────────────────────

def detect_daily_signal(row):
    """Check if daily base signal fires: >5% below 20d high + RSI<35 + first green after 3+ red."""
    if pd.isna(row["pct_from_20d_high"]) or pd.isna(row["rsi_14"]):
        return False
    return (
        row["pct_from_20d_high"] < -0.05 and
        row["rsi_14"] < 35 and
        row["green"] and
        row["consec_red_before"] >= 3
    )


def check_weekly_filter(weekly_row, variant_cfg):
    """Check if weekly filter passes for a given variant."""
    if weekly_row is None:
        return False

    # No weekly filter
    if (variant_cfg["weekly_rsi"] is None and
        variant_cfg["weekly_drop"] is None and
        not variant_cfg["weekly_neg_ret"] and
        not variant_cfg["weekly_rsi_declining"]):
        return True

    # Weekly RSI filter
    if variant_cfg["weekly_rsi"] is not None:
        if pd.isna(weekly_row["rsi_14"]) or weekly_row["rsi_14"] >= variant_cfg["weekly_rsi"]:
            return False

    # Weekly drop from 10-week high
    if variant_cfg["weekly_drop"] is not None:
        if pd.isna(weekly_row["pct_from_10w_high"]) or weekly_row["pct_from_10w_high"] > -variant_cfg["weekly_drop"]:
            return False

    # Negative weekly return
    if variant_cfg["weekly_neg_ret"]:
        if pd.isna(weekly_row["weekly_return"]) or weekly_row["weekly_return"] >= 0:
            return False

    # RSI declining 2+ weeks
    if variant_cfg["weekly_rsi_declining"]:
        if pd.isna(weekly_row.get("rsi_declining", np.nan)) or not weekly_row["rsi_declining"]:
            return False

    return True


# ── Backtest Engine ────────────────────────────────────────────────────────────

def run_backtest(all_daily, all_weekly, spy_daily, variant_key, variant_cfg):
    """Run backtest for a single variant. Returns trade list and equity curve."""
    hold_days = variant_cfg["hold_days"]
    trades = []
    active_positions = []  # list of dicts: {ticker, entry_date, entry_price, shares, exit_date_idx}
    equity = CAPITAL
    equity_curve = []

    # Get all trading dates in OOT period
    oot_dates = spy_daily.loc[OOT_START:OOT_END].index

    for date in oot_dates:
        # Check for exits
        new_active = []
        for pos in active_positions:
            ticker = pos["ticker"]
            if ticker not in all_daily:
                new_active.append(pos)
                continue
            tdf = all_daily[ticker]

            # Find exit date: hold_days trading days after entry
            entry_loc = tdf.index.get_loc(pos["entry_date"]) if pos["entry_date"] in tdf.index else None
            if entry_loc is None:
                new_active.append(pos)
                continue

            exit_loc = entry_loc + hold_days
            if exit_loc < len(tdf) and tdf.index[exit_loc] <= date:
                # Exit
                exit_price = tdf.iloc[exit_loc]["Close"]
                exit_price_after_slip = exit_price * (1 - SLIPPAGE_PCT)
                pnl = (exit_price_after_slip - pos["entry_price"]) * pos["shares"]
                equity += pnl
                trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                    "exit_date": tdf.index[exit_loc].strftime("%Y-%m-%d"),
                    "entry_price": pos["entry_price"],
                    "exit_price": exit_price_after_slip,
                    "shares": pos["shares"],
                    "pnl": pnl,
                    "return_pct": (exit_price_after_slip / pos["entry_price"] - 1) * 100,
                    "hold_days": hold_days,
                })
            else:
                new_active.append(pos)
        active_positions = new_active

        # Check for new entries (if slots available)
        if len(active_positions) < MAX_CONCURRENT:
            for ticker in UNIVERSE:
                if len(active_positions) >= MAX_CONCURRENT:
                    break
                if ticker not in all_daily or ticker not in all_weekly:
                    continue
                # Skip if already holding this ticker
                if any(p["ticker"] == ticker for p in active_positions):
                    continue

                tdf = all_daily[ticker]
                if date not in tdf.index:
                    continue
                row = tdf.loc[date]

                # Check daily signal
                if not detect_daily_signal(row):
                    continue

                # Check weekly filter
                weekly_row = get_weekly_values_for_date(all_weekly[ticker], date)
                if not check_weekly_filter(weekly_row, variant_cfg):
                    continue

                # Enter trade next day
                entry_loc = tdf.index.get_loc(date) + 1
                if entry_loc >= len(tdf):
                    continue

                entry_price = tdf.iloc[entry_loc]["Open"] * (1 + SLIPPAGE_PCT)
                position_size = min(MAX_PER_TRADE, equity / MAX_CONCURRENT)
                if position_size <= 0:
                    continue
                shares = position_size / entry_price

                active_positions.append({
                    "ticker": ticker,
                    "entry_date": tdf.index[entry_loc],
                    "entry_price": entry_price,
                    "shares": shares,
                })

        equity_curve.append({"date": date.strftime("%Y-%m-%d"), "equity": equity})

    # Force-close any remaining positions at last available price
    for pos in active_positions:
        ticker = pos["ticker"]
        if ticker in all_daily:
            tdf = all_daily[ticker]
            last_price = tdf.iloc[-1]["Close"] * (1 - SLIPPAGE_PCT)
            pnl = (last_price - pos["entry_price"]) * pos["shares"]
            equity += pnl
            trades.append({
                "ticker": ticker,
                "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                "exit_date": tdf.index[-1].strftime("%Y-%m-%d"),
                "entry_price": pos["entry_price"],
                "exit_price": last_price,
                "shares": pos["shares"],
                "pnl": pnl,
                "return_pct": (last_price / pos["entry_price"] - 1) * 100,
                "hold_days": "forced",
            })

    return trades, equity_curve, equity


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(trades, equity_curve):
    """Compute performance metrics from trade list."""
    if not trades:
        return {
            "sharpe": 0, "sortino": 0, "win_rate": 0, "profit_factor": 0,
            "max_drawdown_pct": 0, "total_return_pct": 0, "num_trades": 0,
            "avg_return_pct": 0, "avg_win_pct": 0, "avg_loss_pct": 0,
        }

    returns = [t["return_pct"] / 100 for t in trades]
    returns = np.array(returns)

    num_trades = len(trades)
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    win_rate = len(wins) / num_trades if num_trades > 0 else 0

    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_return = returns.mean()
    std_return = returns.std()
    sharpe = (avg_return / std_return) * np.sqrt(252 / 10) if std_return > 0 else 0  # annualized approx

    downside = returns[returns < 0]
    downside_std = downside.std() if len(downside) > 1 else 0
    sortino = (avg_return / downside_std) * np.sqrt(252 / 10) if downside_std > 0 else 0

    # Max drawdown from equity curve
    equities = [e["equity"] for e in equity_curve]
    peak = equities[0]
    max_dd = 0
    for e in equities:
        peak = max(peak, e)
        dd = (e - peak) / peak
        max_dd = min(max_dd, dd)

    total_return = (equities[-1] / CAPITAL - 1) * 100 if equities else 0

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "win_rate": round(win_rate * 100, 1),
        "profit_factor": round(profit_factor, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "total_return_pct": round(total_return, 2),
        "num_trades": num_trades,
        "avg_return_pct": round(avg_return * 100, 3),
        "avg_win_pct": round(wins.mean() * 100, 3) if len(wins) > 0 else 0,
        "avg_loss_pct": round(losses.mean() * 100, 3) if len(losses) > 0 else 0,
    }


def regime_analysis(trades, spy_daily):
    """Split trades by bull/bear regime (SPY > 200-SMA)."""
    spy_sma200 = spy_daily["Close"].rolling(200).mean()

    bull_returns = []
    bear_returns = []

    for t in trades:
        entry_date = pd.Timestamp(t["entry_date"])
        # Find nearest SPY date
        mask = spy_daily.index <= entry_date
        if mask.sum() == 0:
            continue
        nearest = spy_daily.index[mask][-1]

        if spy_daily.loc[nearest, "Close"] > spy_sma200.loc[nearest]:
            bull_returns.append(t["return_pct"] / 100)
        else:
            bear_returns.append(t["return_pct"] / 100)

    def regime_sharpe(rets):
        rets = np.array(rets)
        if len(rets) < 2:
            return 0
        return (rets.mean() / rets.std()) * np.sqrt(252 / 10) if rets.std() > 0 else 0

    bull_sharpe = regime_sharpe(bull_returns)
    bear_sharpe = regime_sharpe(bear_returns)

    max_abs = max(abs(bull_sharpe), abs(bear_sharpe))
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs if max_abs > 0 else 0

    return {
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "bull_trades": len(bull_returns),
        "bear_trades": len(bear_returns),
        "regime_gap": round(regime_gap, 3),
    }


def permutation_test(trades, n_iter=1000):
    """Permutation test: shuffle trade returns, compute fraction with Sharpe >= observed."""
    if len(trades) < 5:
        return 1.0

    returns = np.array([t["return_pct"] / 100 for t in trades])
    observed_sharpe = (returns.mean() / returns.std()) * np.sqrt(252 / 10) if returns.std() > 0 else 0

    rng = np.random.default_rng(42)
    count_ge = 0
    for _ in range(n_iter):
        shuffled = rng.choice(returns, size=len(returns), replace=True)
        # Randomly flip signs to break temporal structure
        signs = rng.choice([-1, 1], size=len(returns))
        shuffled = returns * signs
        s = (shuffled.mean() / shuffled.std()) * np.sqrt(252 / 10) if shuffled.std() > 0 else 0
        if s >= observed_sharpe:
            count_ge += 1

    return round(count_ge / n_iter, 4)


def five_gate_validation(metrics, regime, perm_p):
    """5-gate validation check."""
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": perm_p < 0.05,
        "regime_gap_lt_0.5": regime["regime_gap"] < 0.5,
        "mdd_gt_neg50": metrics["max_drawdown_pct"] > -50,
        "trades_gte_20": metrics["num_trades"] >= 20,
    }
    return gates, all(gates.values())


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("MULTI-TIMEFRAME RELAXED SWEEP BACKTEST")
    print(f"Universe: {len(UNIVERSE)} stocks | OOT: {OOT_START} to {OOT_END}")
    print(f"Capital: ${CAPITAL} | Max/trade: ${MAX_PER_TRADE} | Max concurrent: {MAX_CONCURRENT}")
    print("=" * 80)

    # Download data
    print("\nDownloading price data...")
    all_tickers = UNIVERSE + ["SPY"]
    data = yf.download(all_tickers, start=START_DATE, end=OOT_END, auto_adjust=True, progress=True)

    # Prepare per-stock dataframes
    all_daily = {}
    all_weekly = {}

    for ticker in UNIVERSE:
        try:
            df = pd.DataFrame({
                "Open": data["Open"][ticker],
                "Close": data["Close"][ticker],
            }).dropna()
            df = prepare_daily_data(df)
            all_daily[ticker] = df
            all_weekly[ticker] = prepare_weekly_data(df)
        except Exception as e:
            print(f"  Warning: Could not process {ticker}: {e}")

    # SPY for regime analysis
    spy_daily = pd.DataFrame({
        "Open": data["Open"]["SPY"],
        "Close": data["Close"]["SPY"],
    }).dropna()

    print(f"\nProcessed {len(all_daily)} stocks successfully.")

    # Run all variants
    results = {}

    for vkey in sorted(VARIANTS.keys()):
        vcfg = VARIANTS[vkey]
        print(f"\n{'─' * 60}")
        print(f"Variant {vkey}: {vcfg['desc']}")
        print(f"{'─' * 60}")

        trades, equity_curve, final_equity = run_backtest(
            all_daily, all_weekly, spy_daily, vkey, vcfg
        )

        metrics = compute_metrics(trades, equity_curve)
        regime = regime_analysis(trades, spy_daily)
        perm_p = permutation_test(trades, PERM_ITERATIONS)
        gates, passes_all = five_gate_validation(metrics, regime, perm_p)

        results[vkey] = {
            "description": vcfg["desc"],
            "hold_days": vcfg["hold_days"],
            "metrics": metrics,
            "regime": regime,
            "perm_p_value": perm_p,
            "five_gates": gates,
            "passes_all_gates": passes_all,
            "final_equity": round(final_equity, 2),
        }

        print(f"  Trades: {metrics['num_trades']} | Sharpe: {metrics['sharpe']} | "
              f"Sortino: {metrics['sortino']} | WR: {metrics['win_rate']}%")
        print(f"  PF: {metrics['profit_factor']} | MDD: {metrics['max_drawdown_pct']}% | "
              f"Total Return: {metrics['total_return_pct']}%")
        print(f"  Bull Sharpe: {regime['bull_sharpe']} ({regime['bull_trades']}t) | "
              f"Bear Sharpe: {regime['bear_sharpe']} ({regime['bear_trades']}t) | "
              f"Gap: {regime['regime_gap']}")
        print(f"  Perm p-value: {perm_p} | 5-Gate: {'PASS' if passes_all else 'FAIL'} "
              f"{[k for k,v in gates.items() if not v]}")

    # ── Summary Table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("SUMMARY TABLE: Sharpe vs Trade-Count Tradeoff")
    print("=" * 100)
    print(f"{'Var':>3} {'Description':<45} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} "
          f"{'WR%':>5} {'PF':>6} {'MDD%':>7} {'Ret%':>7} {'5G':>4}")
    print("-" * 100)

    # Sort by Sharpe descending
    sorted_variants = sorted(results.keys(), key=lambda k: results[k]["metrics"]["sharpe"], reverse=True)

    for vkey in sorted_variants:
        r = results[vkey]
        m = r["metrics"]
        tag = "PASS" if r["passes_all_gates"] else "FAIL"
        print(f"  {vkey:>1} {r['description']:<45} {m['num_trades']:>6} {m['sharpe']:>7.3f} "
              f"{m['sortino']:>8.3f} {m['win_rate']:>5.1f} {m['profit_factor']:>6.2f} "
              f"{m['max_drawdown_pct']:>7.2f} {m['total_return_pct']:>7.2f} {tag:>4}")

    print("-" * 100)

    # Find best variant with 60+ trades
    candidates_60 = {k: v for k, v in results.items() if v["metrics"]["num_trades"] >= 60}
    candidates_50 = {k: v for k, v in results.items() if v["metrics"]["num_trades"] >= 50}

    if candidates_60:
        best_60 = max(candidates_60, key=lambda k: candidates_60[k]["metrics"]["sharpe"])
        r = results[best_60]
        print(f"\nBEST with 60+ trades: Variant {best_60} — "
              f"Sharpe {r['metrics']['sharpe']}, {r['metrics']['num_trades']} trades, "
              f"WR {r['metrics']['win_rate']}%, PF {r['metrics']['profit_factor']}")
    else:
        print("\nNo variant reached 60+ trades.")

    if candidates_50:
        best_50 = max(candidates_50, key=lambda k: candidates_50[k]["metrics"]["sharpe"])
        r = results[best_50]
        print(f"BEST with 50+ trades: Variant {best_50} — "
              f"Sharpe {r['metrics']['sharpe']}, {r['metrics']['num_trades']} trades, "
              f"WR {r['metrics']['win_rate']}%, PF {r['metrics']['profit_factor']}")

    # Overall best
    best_overall = max(results, key=lambda k: results[k]["metrics"]["sharpe"])
    r = results[best_overall]
    print(f"BEST overall Sharpe: Variant {best_overall} — "
          f"Sharpe {r['metrics']['sharpe']}, {r['metrics']['num_trades']} trades")

    # Passing variants
    passing = [k for k, v in results.items() if v["passes_all_gates"]]
    print(f"\nVariants passing all 5 gates: {passing if passing else 'None'}")

    # ── Save Results ───────────────────────────────────────────────────────────
    output_path = Path("/home/jupiter/Lvl3Quant/data/multi_timeframe_relaxed_sweep_results.json")

    output = {
        "run_timestamp": datetime.now().isoformat(),
        "config": {
            "universe": UNIVERSE,
            "oot_start": OOT_START,
            "oot_end": OOT_END,
            "capital": CAPITAL,
            "max_per_trade": MAX_PER_TRADE,
            "max_concurrent": MAX_CONCURRENT,
            "slippage_pct": SLIPPAGE_PCT,
            "perm_iterations": PERM_ITERATIONS,
        },
        "variants": results,
        "summary": {
            "best_overall_sharpe": best_overall,
            "best_60_plus_trades": best_60 if candidates_60 else None,
            "best_50_plus_trades": best_50 if candidates_50 else None,
            "passing_all_gates": passing,
        },
    }

    output_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
