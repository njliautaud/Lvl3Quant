#!/usr/bin/env python3
"""
Dividend Capture on Quality Stocks Backtest
============================================
Buy quality dividend-paying stocks around ex-dividend dates, capture the dividend,
sell after. Tests whether dividend income + quality stock recovery makes this profitable.

Variants:
  A: Buy 5 days before ex-date, sell on ex-date
  B: Buy 5 days before ex-date, hold 5 days after ex-date
  C: Buy 5 days before ex-date, hold 10 days after ex-date
  D: Buy on ex-date (buy the dip), hold 10 days
  E: Buy 3 days before ONLY if RSI(14) < 50, hold 5 days after
  F: Buy 5 days before, sell when price recovers to pre-ex level or after 10 days

Universe: AAPL, MSFT, JPM, JNJ, PG, KO, PEP, HD, COST, UNH, ABBV, MRK, V, MA, WMT
Capital: $645, max $200/trade, max 3 concurrent, slippage 2bps
Period: 2022-01-01 to 2026-07-31

5-Gate Validation:
  1. Sharpe > 0.5
  2. Permutation test: 1000 random entry shuffles, p-value < 0.05
  3. Regime gap < 0.5 (bull/bear using SPY vs 200-SMA)
  4. Max drawdown > -50%
  5. At least 20 trades
"""

import json
import warnings
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta

warnings.filterwarnings("ignore")

# ─── Configuration ───────────────────────────────────────────────────────────
UNIVERSE = ["AAPL", "MSFT", "JPM", "JNJ", "PG", "KO", "PEP", "HD",
            "COST", "UNH", "ABBV", "MRK", "V", "MA", "WMT"]
START = "2022-01-01"
END = "2026-07-31"
CAPITAL = 645.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
SLIPPAGE_BPS = 2  # 2 basis points each way
N_PERMUTATIONS = 1000


def download_data():
    """Download price and dividend data for all tickers + SPY."""
    tickers = UNIVERSE + ["SPY"]
    print(f"Downloading data for {len(tickers)} tickers...")
    price_data = {}
    dividend_data = {}

    for t in tickers:
        try:
            tk = yf.Ticker(t)
            hist = tk.history(start=START, end=END, auto_adjust=False)
            if hist.empty:
                print(f"  WARNING: No data for {t}")
                continue
            # Store adjusted close for returns, close for levels
            price_data[t] = hist[["Close", "Adj Close", "Open", "High", "Low"]].copy()
            price_data[t].index = price_data[t].index.tz_localize(None)

            # Get dividends (strip timezone to match price index)
            divs = hist["Dividends"]
            divs = divs[divs > 0]
            divs.index = divs.index.tz_localize(None)
            dividend_data[t] = divs
            print(f"  {t}: {len(hist)} bars, {len(divs)} dividends")
        except Exception as e:
            print(f"  ERROR downloading {t}: {e}")

    return price_data, dividend_data


def compute_rsi(series, period=14):
    """Compute RSI(period) for a price series."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def get_spy_regime(spy_prices):
    """Classify each day as bull (Close > SMA200) or bear."""
    close = spy_prices["Close"]
    sma200 = close.rolling(200).mean()
    regime = pd.Series("bull", index=close.index)
    regime[close < sma200] = "bear"
    return regime


def run_variant(variant_name, price_data, dividend_data, spy_regime):
    """
    Run a single variant of the dividend capture strategy.
    Returns list of trade dicts and daily equity series.
    """
    trades = []
    all_dates = price_data["SPY"].index.sort_values()

    for ticker in UNIVERSE:
        if ticker not in price_data or ticker not in dividend_data:
            continue
        prices = price_data[ticker]
        divs = dividend_data[ticker]
        close = prices["Close"]
        rsi_series = compute_rsi(close, 14) if variant_name == "E" else None

        for ex_date, div_amount in divs.items():
            ex_date = pd.Timestamp(ex_date)
            if ex_date not in close.index:
                continue

            # Determine entry/exit dates based on variant
            entry_date = None
            exit_date = None
            trade_filter = True

            if variant_name == "A":
                # Buy 5 days before, sell on ex-date
                idx = close.index.get_loc(ex_date)
                entry_idx = max(0, idx - 5)
                entry_date = close.index[entry_idx]
                exit_date = ex_date

            elif variant_name == "B":
                # Buy 5 days before, hold 5 days after
                idx = close.index.get_loc(ex_date)
                entry_idx = max(0, idx - 5)
                exit_idx = min(len(close) - 1, idx + 5)
                entry_date = close.index[entry_idx]
                exit_date = close.index[exit_idx]

            elif variant_name == "C":
                # Buy 5 days before, hold 10 days after
                idx = close.index.get_loc(ex_date)
                entry_idx = max(0, idx - 5)
                exit_idx = min(len(close) - 1, idx + 10)
                entry_date = close.index[entry_idx]
                exit_date = close.index[exit_idx]

            elif variant_name == "D":
                # Buy on ex-date, hold 10 days
                idx = close.index.get_loc(ex_date)
                exit_idx = min(len(close) - 1, idx + 10)
                entry_date = ex_date
                exit_date = close.index[exit_idx]

            elif variant_name == "E":
                # Buy 3 days before if RSI(14) < 50, hold 5 days after
                idx = close.index.get_loc(ex_date)
                entry_idx = max(0, idx - 3)
                entry_date = close.index[entry_idx]
                exit_idx = min(len(close) - 1, idx + 5)
                exit_date = close.index[exit_idx]
                if rsi_series is not None and entry_date in rsi_series.index:
                    rsi_val = rsi_series.loc[entry_date]
                    if pd.isna(rsi_val) or rsi_val >= 50:
                        trade_filter = False

            elif variant_name == "F":
                # Buy 5 days before, sell when price recovers to pre-ex level or 10 days
                idx = close.index.get_loc(ex_date)
                entry_idx = max(0, idx - 5)
                entry_date = close.index[entry_idx]
                entry_price = close.iloc[entry_idx]
                # Look for recovery or max 10 days after ex-date
                exit_date = None
                for offset in range(1, 11):
                    check_idx = min(len(close) - 1, idx + offset)
                    if close.iloc[check_idx] >= entry_price:
                        exit_date = close.index[check_idx]
                        break
                if exit_date is None:
                    exit_idx = min(len(close) - 1, idx + 10)
                    exit_date = close.index[exit_idx]

            if not trade_filter:
                continue
            if entry_date is None or exit_date is None:
                continue
            if entry_date >= exit_date:
                continue
            if entry_date not in close.index or exit_date not in close.index:
                continue

            entry_price = close.loc[entry_date]
            exit_price = close.loc[exit_date]

            if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
                continue

            # Position sizing: max $200/trade, buy whole shares
            shares = int(MAX_PER_TRADE // entry_price)
            if shares < 1:
                continue

            # Slippage
            entry_cost = entry_price * (1 + SLIPPAGE_BPS / 10000)
            exit_proceeds = exit_price * (1 - SLIPPAGE_BPS / 10000)

            # Did we hold through the ex-date? If so, capture dividend
            div_captured = 0.0
            if entry_date < ex_date <= exit_date:
                div_captured = div_amount * shares
            elif variant_name == "D":
                # Variant D buys on ex-date, doesn't capture that dividend
                # (must own before ex-date to get the dividend)
                div_captured = 0.0

            price_pnl = (exit_proceeds - entry_cost) * shares
            total_pnl = price_pnl + div_captured
            invested = entry_cost * shares

            # Regime at entry
            regime = "unknown"
            if entry_date in spy_regime.index:
                regime = spy_regime.loc[entry_date]

            trades.append({
                "ticker": ticker,
                "entry_date": str(entry_date.date()),
                "exit_date": str(exit_date.date()),
                "ex_date": str(ex_date.date()),
                "entry_price": round(float(entry_price), 2),
                "exit_price": round(float(exit_price), 2),
                "shares": shares,
                "div_per_share": round(float(div_amount), 4),
                "div_captured": round(float(div_captured), 2),
                "price_pnl": round(float(price_pnl), 2),
                "total_pnl": round(float(total_pnl), 2),
                "return_pct": round(float(total_pnl / invested * 100), 4) if invested > 0 else 0,
                "regime": regime,
                "invested": round(float(invested), 2),
            })

    # Sort trades by entry date
    trades.sort(key=lambda t: t["entry_date"])

    # Build equity curve respecting max concurrent positions
    equity_curve = build_equity_curve(trades, all_dates)

    return trades, equity_curve


def build_equity_curve(trades, all_dates):
    """
    Build a daily equity curve from trades, respecting max concurrent positions.
    Returns a pandas Series of daily portfolio value.
    """
    if not trades:
        return pd.Series(CAPITAL, index=all_dates)

    # Filter trades to respect max concurrent
    active_trades = []
    accepted_trades = []

    for t in trades:
        entry = pd.Timestamp(t["entry_date"])
        exit_d = pd.Timestamp(t["exit_date"])

        # Count how many active trades overlap with this entry
        concurrent = sum(1 for at in accepted_trades
                        if pd.Timestamp(at["entry_date"]) <= entry <= pd.Timestamp(at["exit_date"]))
        if concurrent < MAX_CONCURRENT:
            accepted_trades.append(t)

    # Now build daily P&L from accepted trades
    daily_pnl = pd.Series(0.0, index=all_dates)
    for t in accepted_trades:
        entry = pd.Timestamp(t["entry_date"])
        exit_d = pd.Timestamp(t["exit_date"])
        # Spread total PnL evenly across holding days for equity curve
        mask = (all_dates >= entry) & (all_dates <= exit_d)
        n_days = mask.sum()
        if n_days > 0:
            daily_pnl[mask] += t["total_pnl"] / n_days

    equity = CAPITAL + daily_pnl.cumsum()
    return equity


def compute_metrics(trades, equity_curve, spy_regime):
    """Compute all performance metrics for a variant."""
    if not trades or len(trades) < 2:
        return None

    returns_pct = [t["return_pct"] / 100 for t in trades]
    total_pnl = sum(t["total_pnl"] for t in trades)
    total_div = sum(t["div_captured"] for t in trades)
    total_invested_sum = sum(t["invested"] for t in trades)

    # Win rate
    wins = [r for r in returns_pct if r > 0]
    losses = [r for r in returns_pct if r <= 0]
    win_rate = len(wins) / len(returns_pct) if returns_pct else 0

    # Profit factor
    gross_profit = sum(t["total_pnl"] for t in trades if t["total_pnl"] > 0)
    gross_loss = abs(sum(t["total_pnl"] for t in trades if t["total_pnl"] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Total return
    total_return = total_pnl / CAPITAL

    # Annualized from daily equity returns
    daily_rets = equity_curve.pct_change().dropna()
    daily_rets = daily_rets.replace([np.inf, -np.inf], 0).fillna(0)

    ann_factor = np.sqrt(252)
    mean_daily = daily_rets.mean()
    std_daily = daily_rets.std()

    sharpe = (mean_daily / std_daily * ann_factor) if std_daily > 0 else 0

    # Sortino
    downside = daily_rets[daily_rets < 0]
    downside_std = downside.std() if len(downside) > 0 else 0
    sortino = (mean_daily / downside_std * ann_factor) if downside_std > 0 else 0

    # Max drawdown
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    max_drawdown = drawdown.min()

    # Average dividend yield captured per trade
    avg_div_yield = np.mean([t["div_captured"] / t["invested"] * 100
                            for t in trades if t["invested"] > 0])

    # Regime analysis
    bull_trades = [t for t in trades if t["regime"] == "bull"]
    bear_trades = [t for t in trades if t["regime"] == "bear"]

    def regime_sharpe(regime_trades):
        if len(regime_trades) < 3:
            return 0.0
        rets = [t["return_pct"] / 100 for t in regime_trades]
        m = np.mean(rets)
        s = np.std(rets, ddof=1)
        # Annualize: assume ~4 trades per year per stock, rough
        return (m / s) * np.sqrt(len(regime_trades)) if s > 0 else 0.0

    bull_sharpe = regime_sharpe(bull_trades)
    bear_sharpe = regime_sharpe(bear_trades)
    regime_gap = (abs(bull_sharpe - bear_sharpe) /
                  max(abs(bull_sharpe), abs(bear_sharpe), 1e-6))

    return {
        "num_trades": len(trades),
        "total_pnl": round(total_pnl, 2),
        "total_return": round(total_return * 100, 2),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "win_rate": round(win_rate * 100, 2),
        "profit_factor": round(min(profit_factor, 99.99), 4),
        "max_drawdown": round(max_drawdown * 100, 2),
        "total_div_captured": round(total_div, 2),
        "avg_div_yield_captured": round(avg_div_yield, 4),
        "bull_sharpe": round(bull_sharpe, 4),
        "bear_sharpe": round(bear_sharpe, 4),
        "regime_gap": round(regime_gap, 4),
        "bull_trades": len(bull_trades),
        "bear_trades": len(bear_trades),
    }


def permutation_test(trades, equity_curve, observed_sharpe, n_perms=N_PERMUTATIONS):
    """
    Shuffle trade entry dates randomly, recompute Sharpe each time.
    p-value = fraction of permuted Sharpes >= observed.
    """
    if not trades or len(trades) < 5:
        return 1.0

    returns_pct = np.array([t["return_pct"] / 100 for t in trades])
    count_better = 0

    for _ in range(n_perms):
        shuffled = np.random.permutation(returns_pct)
        m = shuffled.mean()
        s = shuffled.std(ddof=1)
        perm_sharpe = (m / s * np.sqrt(len(shuffled))) if s > 0 else 0
        if perm_sharpe >= observed_sharpe:
            count_better += 1

    return count_better / n_perms


def five_gate_check(metrics, perm_p):
    """Apply the 5-gate validation."""
    if metrics is None:
        return {"pass": False, "reason": "No metrics"}

    gates = {}
    gates["sharpe_gt_0.5"] = metrics["sharpe"] > 0.5
    gates["perm_p_lt_0.05"] = perm_p < 0.05
    gates["regime_gap_lt_0.5"] = metrics["regime_gap"] < 0.5
    gates["max_dd_gt_neg50"] = metrics["max_drawdown"] > -50.0
    gates["min_20_trades"] = metrics["num_trades"] >= 20

    all_pass = all(gates.values())
    return {"pass": all_pass, "gates": gates}


def main():
    np.random.seed(42)

    # Download data
    price_data, dividend_data = download_data()

    if "SPY" not in price_data:
        print("ERROR: Could not download SPY data. Aborting.")
        sys.exit(1)

    spy_regime = get_spy_regime(price_data["SPY"])

    variants = ["A", "B", "C", "D", "E", "F"]
    results = {}

    for v in variants:
        print(f"\n{'='*60}")
        print(f"Running Variant {v}...")
        print(f"{'='*60}")

        trades, equity = run_variant(v, price_data, dividend_data, spy_regime)
        print(f"  Total trades generated: {len(trades)}")

        if not trades:
            results[v] = {"error": "No trades generated"}
            continue

        metrics = compute_metrics(trades, equity, spy_regime)
        if metrics is None:
            results[v] = {"error": "Could not compute metrics"}
            continue

        print(f"  Sharpe: {metrics['sharpe']:.4f}")
        print(f"  Sortino: {metrics['sortino']:.4f}")
        print(f"  Win Rate: {metrics['win_rate']:.1f}%")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"  Max DD: {metrics['max_drawdown']:.2f}%")
        print(f"  Total Return: {metrics['total_return']:.2f}%")
        print(f"  Dividends Captured: ${metrics['total_div_captured']:.2f}")
        print(f"  Avg Div Yield/Trade: {metrics['avg_div_yield_captured']:.4f}%")

        # Compute trade-level Sharpe for permutation test
        rets = np.array([t["return_pct"] / 100 for t in trades])
        trade_sharpe = (rets.mean() / rets.std(ddof=1) * np.sqrt(len(rets))) if rets.std(ddof=1) > 0 else 0

        print(f"  Running permutation test ({N_PERMUTATIONS} shuffles)...")
        perm_p = permutation_test(trades, equity, trade_sharpe)
        print(f"  Permutation p-value: {perm_p:.4f}")

        gate_result = five_gate_check(metrics, perm_p)
        print(f"  5-Gate: {'PASS' if gate_result['pass'] else 'FAIL'}")
        if not gate_result["pass"]:
            for g, v_pass in gate_result["gates"].items():
                if not v_pass:
                    print(f"    FAILED: {g}")

        metrics["permutation_p"] = round(perm_p, 4)
        metrics["five_gate"] = gate_result

        # Sample trades for output
        sample_trades = trades[:5] if len(trades) > 5 else trades

        results[v] = {
            "metrics": metrics,
            "sample_trades": sample_trades,
        }

    # ─── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("DIVIDEND CAPTURE BACKTEST — SUMMARY")
    print(f"{'='*60}")
    print(f"{'Variant':<10} {'Sharpe':>8} {'Sortino':>8} {'WR%':>6} {'PF':>6} "
          f"{'MaxDD%':>8} {'Return%':>9} {'Divs$':>7} {'Trades':>7} {'5-Gate':>7}")
    print("-" * 90)

    for v in variants:
        if "error" in results.get(v, {}):
            print(f"  {v:<8} {'ERROR':>8}")
            continue
        m = results[v]["metrics"]
        gp = "PASS" if m["five_gate"]["pass"] else "FAIL"
        print(f"  {v:<8} {m['sharpe']:>8.4f} {m['sortino']:>8.4f} {m['win_rate']:>5.1f}% "
              f"{m['profit_factor']:>6.2f} {m['max_drawdown']:>7.2f}% {m['total_return']:>8.2f}% "
              f"{m['total_div_captured']:>7.2f} {m['num_trades']:>7d} {gp:>7}")

    # Find best variant
    valid = {v: results[v] for v in variants if "metrics" in results.get(v, {})}
    if valid:
        best = max(valid.keys(), key=lambda v: valid[v]["metrics"]["sharpe"])
        bm = valid[best]["metrics"]
        print(f"\nBest variant: {best} (Sharpe={bm['sharpe']:.4f})")

    # Save results
    output = {
        "strategy": "Dividend Capture on Quality Stocks",
        "universe": UNIVERSE,
        "period": f"{START} to {END}",
        "capital": CAPITAL,
        "max_per_trade": MAX_PER_TRADE,
        "max_concurrent": MAX_CONCURRENT,
        "slippage_bps": SLIPPAGE_BPS,
        "variants": {},
    }

    for v in variants:
        if "error" in results.get(v, {}):
            output["variants"][v] = results[v]
        else:
            output["variants"][v] = results[v]["metrics"]
            output["variants"][v]["sample_trades"] = results[v]["sample_trades"]

    out_path = "/home/jupiter/Lvl3Quant/data/dividend_capture_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
