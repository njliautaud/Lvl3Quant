#!/usr/bin/env python3
"""
Holding Period Optimization Backtest for Dual Signal D
Tests 8 exit variants (A-H) on a quality universe with 5-gate validation.
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import sys
import os

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]
OOT_START = "2022-01-01"
OOT_END = "2026-07-31"
# Fetch extra history for indicator warm-up
FETCH_START = "2021-06-01"

STARTING_CAPITAL = 645.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
SLIPPAGE_PCT = 0.0002  # 0.02% each way

PERM_ITERATIONS = 1000
TRADING_DAYS_PER_YEAR = 252

OUTPUT_PATH = "/home/jupiter/Lvl3Quant/data/holding_period_optimization_results.json"


# ── Indicator helpers ──────────────────────────────────────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_sma(series, period):
    return series.rolling(period).mean()


# ── Data download ──────────────────────────────────────────────────────────
def download_data():
    print("Downloading price data...")
    tickers = UNIVERSE + ["SPY"]
    data = {}
    for tick in tickers:
        try:
            df = yf.download(tick, start=FETCH_START, end=OOT_END, progress=False, auto_adjust=True)
            if len(df) > 50:
                df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                df.columns = ["Open", "High", "Low", "Close", "Volume"]
                data[tick] = df
                print(f"  {tick}: {len(df)} bars")
            else:
                print(f"  {tick}: SKIPPED (only {len(df)} bars)")
        except Exception as e:
            print(f"  {tick}: ERROR {e}")
    return data


# ── Entry signal detection ─────────────────────────────────────────────────
def find_entries(df):
    """
    Entry: stock drops >5% from 20-day high + RSI(14)<35 + first green after 3+ consecutive red days.
    Returns list of entry dates (index positions within OOT).
    """
    close = df["Close"]
    rsi = compute_rsi(close, 14)
    high20 = close.rolling(20).max()
    drawdown = (close - high20) / high20

    # Red day: close < open
    red = (close < df["Open"]).astype(int)

    entries = []
    oot_mask = df.index >= pd.Timestamp(OOT_START)

    for i in range(25, len(df)):
        if not oot_mask[i]:
            continue
        # Must be a green day (close >= open)
        if close.iloc[i] < df["Open"].iloc[i]:
            continue
        # Check 3+ consecutive red days before today
        consec_red = 0
        for j in range(i - 1, max(i - 30, 0) - 1, -1):
            if red.iloc[j] == 1:
                consec_red += 1
            else:
                break
        if consec_red < 3:
            continue
        # Drawdown > 5% from 20-day high
        if drawdown.iloc[i] > -0.05:
            continue
        # RSI < 35
        if rsi.iloc[i] >= 35:
            continue

        entries.append(i)

    return entries


# ── Exit logic per variant ─────────────────────────────────────────────────
def simulate_exit(df, entry_idx, variant):
    """
    Returns (exit_idx, exit_price_adjusted) for the given variant.
    Entry is at close of entry_idx day. Exit at close of exit day.
    """
    close = df["Close"]
    n = len(df)
    entry_price = close.iloc[entry_idx]

    if variant == "A":  # Hold 5 days
        exit_idx = min(entry_idx + 5, n - 1)
        return exit_idx, close.iloc[exit_idx]

    elif variant == "B":  # Hold 10 days (baseline)
        exit_idx = min(entry_idx + 10, n - 1)
        return exit_idx, close.iloc[exit_idx]

    elif variant == "C":  # Hold 15 days
        exit_idx = min(entry_idx + 15, n - 1)
        return exit_idx, close.iloc[exit_idx]

    elif variant == "D":  # Hold 20 days
        exit_idx = min(entry_idx + 20, n - 1)
        return exit_idx, close.iloc[exit_idx]

    elif variant == "E":  # RSI exit: RSI > 60 or max 20 days
        rsi = compute_rsi(close, 14)
        for d in range(1, 21):
            idx = entry_idx + d
            if idx >= n:
                return n - 1, close.iloc[n - 1]
            if rsi.iloc[idx] > 60:
                return idx, close.iloc[idx]
        exit_idx = min(entry_idx + 20, n - 1)
        return exit_idx, close.iloc[exit_idx]

    elif variant == "F":  # Profit target: +3% or max 15 days
        for d in range(1, 16):
            idx = entry_idx + d
            if idx >= n:
                return n - 1, close.iloc[n - 1]
            ret = (close.iloc[idx] - entry_price) / entry_price
            if ret >= 0.03:
                return idx, close.iloc[idx]
        exit_idx = min(entry_idx + 15, n - 1)
        return exit_idx, close.iloc[exit_idx]

    elif variant == "G":  # SMA(5) recross or max 20 days
        sma5 = compute_sma(close, 5)
        for d in range(1, 21):
            idx = entry_idx + d
            if idx >= n:
                return n - 1, close.iloc[n - 1]
            if close.iloc[idx] > sma5.iloc[idx]:
                return idx, close.iloc[idx]
        exit_idx = min(entry_idx + 20, n - 1)
        return exit_idx, close.iloc[exit_idx]

    elif variant == "H":  # Trailing: hold min 3 days, exit on first red day. Max 20.
        for d in range(1, 21):
            idx = entry_idx + d
            if idx >= n:
                return n - 1, close.iloc[n - 1]
            if d >= 3 and close.iloc[idx] < close.iloc[idx - 1]:
                return idx, close.iloc[idx]
        exit_idx = min(entry_idx + 20, n - 1)
        return exit_idx, close.iloc[exit_idx]


# ── Portfolio-level backtest ───────────────────────────────────────────────
def run_backtest(all_data, spy_data, variant):
    """
    Run full portfolio backtest for a given exit variant.
    Returns dict of metrics + list of trades.
    """
    # Collect all potential entries across stocks
    raw_signals = []
    for ticker, df in all_data.items():
        if ticker == "SPY":
            continue
        entries = find_entries(df)
        for entry_idx in entries:
            entry_date = df.index[entry_idx]
            raw_signals.append((entry_date, ticker, entry_idx))

    # Sort by date
    raw_signals.sort(key=lambda x: x[0])

    # Simulate with capital constraints
    capital = STARTING_CAPITAL
    equity_curve = [STARTING_CAPITAL]
    equity_dates = [pd.Timestamp(OOT_START)]
    trades = []
    active_trades = []  # list of (ticker, entry_date, entry_price, shares, exit_idx, exit_date)
    peak_equity = STARTING_CAPITAL

    # SPY SMA(200) for regime
    spy_close = spy_data["Close"]
    spy_sma200 = compute_sma(spy_close, 200)

    for entry_date, ticker, entry_idx in raw_signals:
        df = all_data[ticker]

        # Close any active trades that have exited by this date
        still_active = []
        for at in active_trades:
            if at["exit_date"] <= entry_date:
                # Realize the trade
                pnl = (at["exit_price"] * (1 - SLIPPAGE_PCT) - at["entry_price"] * (1 + SLIPPAGE_PCT)) * at["shares"]
                capital += at["position_cost"] + pnl
                trade_ret = pnl / at["position_cost"] if at["position_cost"] > 0 else 0
                hold_days = (at["exit_date"] - at["entry_date"]).days

                # Determine regime at entry
                regime = "unknown"
                if at["entry_date"] in spy_close.index:
                    spy_idx = spy_close.index.get_loc(at["entry_date"])
                elif len(spy_close.index) > 0:
                    spy_idx = spy_close.index.searchsorted(at["entry_date"]) - 1
                else:
                    spy_idx = -1
                if 0 <= spy_idx < len(spy_close):
                    if not np.isnan(spy_sma200.iloc[spy_idx]):
                        regime = "bull" if spy_close.iloc[spy_idx] > spy_sma200.iloc[spy_idx] else "bear"

                trades.append({
                    "ticker": at["ticker"],
                    "entry_date": str(at["entry_date"].date()),
                    "exit_date": str(at["exit_date"].date()),
                    "entry_price": round(float(at["entry_price"]), 4),
                    "exit_price": round(float(at["exit_price"]), 4),
                    "shares": round(float(at["shares"]), 4),
                    "pnl": round(float(pnl), 2),
                    "return_pct": round(float(trade_ret * 100), 4),
                    "hold_days": hold_days,
                    "regime": regime,
                })

                equity_curve.append(capital)
                equity_dates.append(at["exit_date"])
            else:
                still_active.append(at)
        active_trades = still_active

        # Check if we can open a new trade
        if len(active_trades) >= MAX_CONCURRENT:
            continue
        available = capital - sum(at["position_cost"] for at in active_trades)
        position_size = min(MAX_PER_TRADE, available)
        if position_size < 10:  # minimum viable position
            continue

        entry_price = df["Close"].iloc[entry_idx]
        entry_cost = entry_price * (1 + SLIPPAGE_PCT)
        shares = position_size / entry_cost

        exit_idx, exit_price = simulate_exit(df, entry_idx, variant)
        exit_date = df.index[exit_idx]

        active_trades.append({
            "ticker": ticker,
            "entry_date": entry_date,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "shares": shares,
            "position_cost": position_size,
            "exit_date": exit_date,
            "exit_idx": exit_idx,
        })

    # Close remaining active trades
    for at in active_trades:
        pnl = (at["exit_price"] * (1 - SLIPPAGE_PCT) - at["entry_price"] * (1 + SLIPPAGE_PCT)) * at["shares"]
        capital += at["position_cost"] + pnl
        trade_ret = pnl / at["position_cost"] if at["position_cost"] > 0 else 0
        hold_days = (at["exit_date"] - at["entry_date"]).days

        regime = "unknown"
        if at["entry_date"] in spy_close.index:
            spy_idx = spy_close.index.get_loc(at["entry_date"])
        elif len(spy_close.index) > 0:
            spy_idx = spy_close.index.searchsorted(at["entry_date"]) - 1
        else:
            spy_idx = -1
        if 0 <= spy_idx < len(spy_close):
            if not np.isnan(spy_sma200.iloc[spy_idx]):
                regime = "bull" if spy_close.iloc[spy_idx] > spy_sma200.iloc[spy_idx] else "bear"

        trades.append({
            "ticker": at["ticker"],
            "entry_date": str(at["entry_date"].date()),
            "exit_date": str(at["exit_date"].date()),
            "entry_price": round(float(at["entry_price"]), 4),
            "exit_price": round(float(at["exit_price"]), 4),
            "shares": round(float(at["shares"]), 4),
            "pnl": round(float(pnl), 2),
            "return_pct": round(float(trade_ret * 100), 4),
            "hold_days": hold_days,
            "regime": regime,
        })
        equity_curve.append(capital)
        equity_dates.append(at["exit_date"])

    return trades, equity_curve, equity_dates


# ── Metrics computation ────────────────────────────────────────────────────
def compute_metrics(trades, equity_curve):
    if len(trades) == 0:
        return {
            "num_trades": 0, "sharpe": 0, "sortino": 0, "win_rate": 0,
            "profit_factor": 0, "max_drawdown_pct": 0, "total_return_pct": 0,
            "avg_hold_days": 0, "avg_return_pct": 0,
        }

    returns = [t["return_pct"] / 100 for t in trades]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]

    win_rate = len(wins) / len(returns) if returns else 0
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.0001
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999

    avg_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1) if len(returns) > 1 else 0.0001

    # Annualize: assume average ~1 trade per week
    avg_hold = np.mean([t["hold_days"] for t in trades])
    trades_per_year = TRADING_DAYS_PER_YEAR / max(avg_hold, 1)
    sharpe = (avg_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0

    downside_returns = [r for r in returns if r < 0]
    downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else 0.0001
    sortino = (avg_ret / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0

    # Max drawdown from equity curve
    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = float(np.min(dd)) * 100

    total_return = (equity_curve[-1] / equity_curve[0] - 1) * 100

    return {
        "num_trades": len(trades),
        "sharpe": round(float(sharpe), 4),
        "sortino": round(float(sortino), 4),
        "win_rate": round(float(win_rate * 100), 2),
        "profit_factor": round(float(profit_factor), 4),
        "max_drawdown_pct": round(float(max_dd), 2),
        "total_return_pct": round(float(total_return), 2),
        "avg_hold_days": round(float(avg_hold), 2),
        "avg_return_pct": round(float(avg_ret * 100), 4),
    }


def regime_analysis(trades):
    bull_trades = [t for t in trades if t["regime"] == "bull"]
    bear_trades = [t for t in trades if t["regime"] == "bear"]

    def regime_sharpe(trade_list):
        if len(trade_list) < 2:
            return 0
        rets = [t["return_pct"] / 100 for t in trade_list]
        avg_hold = np.mean([t["hold_days"] for t in trade_list])
        tpy = TRADING_DAYS_PER_YEAR / max(avg_hold, 1)
        std = np.std(rets, ddof=1)
        if std == 0:
            return 0
        return float((np.mean(rets) / std) * np.sqrt(tpy))

    bull_sharpe = regime_sharpe(bull_trades)
    bear_sharpe = regime_sharpe(bear_trades)
    max_abs = max(abs(bull_sharpe), abs(bear_sharpe), 0.0001)
    gap = abs(bull_sharpe - bear_sharpe) / max_abs

    return {
        "bull_trades": len(bull_trades),
        "bear_trades": len(bear_trades),
        "bull_sharpe": round(bull_sharpe, 4),
        "bear_sharpe": round(bear_sharpe, 4),
        "regime_gap": round(float(gap), 4),
    }


def permutation_test(trades, observed_sharpe, n_iter=1000):
    if len(trades) < 5:
        return 1.0
    returns = [t["return_pct"] / 100 for t in trades]
    avg_hold = np.mean([t["hold_days"] for t in trades])
    tpy = TRADING_DAYS_PER_YEAR / max(avg_hold, 1)

    count_ge = 0
    for _ in range(n_iter):
        shuffled = np.random.choice(returns, size=len(returns), replace=True)
        # Randomly flip signs to break any structure
        signs = np.random.choice([-1, 1], size=len(returns))
        shuffled = shuffled * signs
        std = np.std(shuffled, ddof=1)
        if std > 0:
            perm_sharpe = (np.mean(shuffled) / std) * np.sqrt(tpy)
        else:
            perm_sharpe = 0
        if perm_sharpe >= observed_sharpe:
            count_ge += 1
    return count_ge / n_iter


# ── 5-Gate validation ──────────────────────────────────────────────────────
def five_gate_check(metrics, regime, perm_p):
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": perm_p < 0.05,
        "regime_gap_lt_0.5": regime["regime_gap"] < 0.5,
        "mdd_gt_neg50": metrics["max_drawdown_pct"] > -50,
        "trades_gte_20": metrics["num_trades"] >= 20,
    }
    gates["all_passed"] = all(gates.values())
    return gates


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    np.random.seed(42)

    all_data = download_data()
    if "SPY" not in all_data:
        print("ERROR: SPY data required for regime analysis")
        sys.exit(1)

    spy_data = all_data["SPY"]
    stock_data = {k: v for k, v in all_data.items() if k != "SPY"}

    variants = {
        "A": "Hold 5 days",
        "B": "Hold 10 days (BASELINE)",
        "C": "Hold 15 days",
        "D": "Hold 20 days",
        "E": "RSI Exit (RSI>60 or max 20d)",
        "F": "Profit Target (+3% or max 15d)",
        "G": "SMA(5) Recross (or max 20d)",
        "H": "Trailing Exit (min 3d, exit on red, max 20d)",
    }

    results = {}
    print("\n" + "=" * 80)
    print("HOLDING PERIOD OPTIMIZATION BACKTEST — Dual Signal D")
    print(f"Universe: {len(stock_data)} stocks | OOT: {OOT_START} to {OOT_END}")
    print(f"Capital: ${STARTING_CAPITAL} | Max/trade: ${MAX_PER_TRADE} | Max concurrent: {MAX_CONCURRENT}")
    print("=" * 80)

    for var_code, var_desc in variants.items():
        print(f"\n--- Variant {var_code}: {var_desc} ---")
        trades, equity_curve, equity_dates = run_backtest(stock_data, spy_data, var_code)
        metrics = compute_metrics(trades, equity_curve)
        regime = regime_analysis(trades)

        print(f"  Trades: {metrics['num_trades']} | Avg hold: {metrics['avg_hold_days']}d")
        print(f"  Sharpe: {metrics['sharpe']} | Sortino: {metrics['sortino']}")
        print(f"  WR: {metrics['win_rate']}% | PF: {metrics['profit_factor']}")
        print(f"  Total Return: {metrics['total_return_pct']}% | MDD: {metrics['max_drawdown_pct']}%")
        print(f"  Regime — Bull Sharpe: {regime['bull_sharpe']} | Bear Sharpe: {regime['bear_sharpe']} | Gap: {regime['regime_gap']}")

        # Permutation test
        print(f"  Running permutation test ({PERM_ITERATIONS} iterations)...", end=" ", flush=True)
        perm_p = permutation_test(trades, metrics["sharpe"], PERM_ITERATIONS)
        print(f"p = {perm_p:.4f}")

        gates = five_gate_check(metrics, regime, perm_p)
        gate_status = "PASS" if gates["all_passed"] else "FAIL"
        failed_gates = [k for k, v in gates.items() if not v and k != "all_passed"]
        if failed_gates:
            print(f"  5-Gate: {gate_status} (failed: {', '.join(failed_gates)})")
        else:
            print(f"  5-Gate: {gate_status}")

        results[var_code] = {
            "description": var_desc,
            "metrics": metrics,
            "regime": regime,
            "perm_p_value": round(perm_p, 4),
            "five_gate": gates,
            "final_equity": round(float(equity_curve[-1]), 2),
            "trade_log": trades,
        }

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    header = f"{'Var':>4} {'Description':<40} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'WR%':>6} {'PF':>6} {'MDD%':>7} {'Return%':>8} {'AvgHold':>7} {'Gate':>5}"
    print(header)
    print("-" * len(header))
    for var_code in sorted(results.keys()):
        r = results[var_code]
        m = r["metrics"]
        gate = "PASS" if r["five_gate"]["all_passed"] else "FAIL"
        print(
            f"{var_code:>4} {r['description']:<40} {m['num_trades']:>6} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
            f"{m['win_rate']:>6.1f} {m['profit_factor']:>6.2f} {m['max_drawdown_pct']:>7.2f} "
            f"{m['total_return_pct']:>8.2f} {m['avg_hold_days']:>7.1f} {gate:>5}"
        )

    # ── Best variant ───────────────────────────────────────────────────────
    passing = {k: v for k, v in results.items() if v["five_gate"]["all_passed"]}
    if passing:
        best = max(passing.keys(), key=lambda k: passing[k]["metrics"]["sharpe"])
        print(f"\nBEST PASSING VARIANT: {best} — {results[best]['description']}")
        print(f"  Sharpe: {results[best]['metrics']['sharpe']} | Return: {results[best]['metrics']['total_return_pct']}%")
    else:
        best_overall = max(results.keys(), key=lambda k: results[k]["metrics"]["sharpe"])
        print(f"\nNO VARIANT PASSED ALL 5 GATES. Best Sharpe: {best_overall} — {results[best_overall]['description']}")

    # ── Save results ───────────────────────────────────────────────────────
    # Remove trade_log for JSON (can be large); keep a summary
    output = {
        "metadata": {
            "strategy": "Dual Signal D — Holding Period Optimization",
            "universe_size": len(stock_data),
            "oot_period": f"{OOT_START} to {OOT_END}",
            "starting_capital": STARTING_CAPITAL,
            "max_per_trade": MAX_PER_TRADE,
            "max_concurrent": MAX_CONCURRENT,
            "slippage_pct": SLIPPAGE_PCT,
            "run_timestamp": datetime.now().isoformat(),
        },
        "variants": {},
    }
    for k, v in results.items():
        output["variants"][k] = {
            "description": v["description"],
            "metrics": v["metrics"],
            "regime": v["regime"],
            "perm_p_value": v["perm_p_value"],
            "five_gate": v["five_gate"],
            "final_equity": v["final_equity"],
            "num_trades_logged": len(v["trade_log"]),
            "trade_log": v["trade_log"],
        }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
