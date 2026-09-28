#!/usr/bin/env python3
"""
Multi-TF L + Multi-Asset Portfolio Backtest
============================================
Combines Multi-TF Variant L (Dual Signal D + weekly RSI declining 2+ weeks)
with multi-asset diversification across US stocks, international ADRs, and sector ETFs.

Signal Logic (Multi-TF L):
- Stocks/ADRs: 5% dip from 20d high + RSI<35 + first green after 3+ red + weekly RSI declining 2+ weeks. Hold 10d.
- ETFs: 3% dip + RSI<35 + first green after 2+ red + weekly RSI declining 2+ weeks. Hold 10d.

OOT: Jan 2022 - Jul 2026, Starting Capital: $645
"""

import json
import warnings
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ─── Asset Universes ───────────────────────────────────────────────────────────
US_QUALITY = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META"
]
INTL_ADRS = ["TSM", "ASML", "NVO", "SAP", "AZN", "NVS", "SHOP", "TM", "SONY", "BHP"]
SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLC", "XLY", "XLP", "XLU", "XLRE"]

ALL_TICKERS = US_QUALITY + INTL_ADRS + SECTOR_ETFS + ["SPY", "UUP"]

OOT_START = "2022-01-01"
OOT_END = "2026-07-31"
HOLD_DAYS = 10
STARTING_CAPITAL = 645.0

SLIPPAGE_STOCK = 0.0002  # 0.02%
SLIPPAGE_ETF = 0.0001    # 0.01%


def get_universe(ticker):
    if ticker in US_QUALITY:
        return "US"
    elif ticker in INTL_ADRS:
        return "Intl"
    elif ticker in SECTOR_ETFS:
        return "ETF"
    return None


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_weekly_rsi(daily_close, rsi_period=14):
    """Resample daily close to weekly, compute RSI, track weeks of decline."""
    weekly = daily_close.resample("W-FRI").last().dropna()
    weekly_rsi = compute_rsi(weekly, rsi_period)

    # Track consecutive weeks of RSI decline
    rsi_declining = weekly_rsi.diff() < 0
    decline_weeks = pd.Series(0, index=weekly_rsi.index, dtype=int)
    for i in range(1, len(decline_weeks)):
        if rsi_declining.iloc[i]:
            decline_weeks.iloc[i] = decline_weeks.iloc[i-1] + 1
        else:
            decline_weeks.iloc[i] = 0

    # Forward-fill to daily index
    decline_weeks_daily = decline_weeks.reindex(daily_close.index, method="ffill")
    return decline_weeks_daily


def generate_signals(daily_data, ticker):
    """Generate Multi-TF L signals for a single ticker."""
    universe = get_universe(ticker)
    if universe is None:
        return pd.Series(False, index=daily_data.index)

    close = daily_data["Close"]
    if len(close) < 30:
        return pd.Series(False, index=daily_data.index)

    # Parameters based on universe
    if universe == "ETF":
        dip_pct = 0.03
        min_red = 2
    else:
        dip_pct = 0.05
        min_red = 3

    # Daily RSI
    rsi = compute_rsi(close, 14)

    # 20-day rolling high
    high_20d = close.rolling(20).max()

    # Dip from 20d high
    dip = (close - high_20d) / high_20d
    dip_condition = dip <= -dip_pct

    # RSI < 35
    rsi_condition = rsi < 35

    # Daily candle color: green = close > open (or close > prev close)
    daily_green = close > close.shift(1)
    daily_red = close <= close.shift(1)

    # Count consecutive red days before current day
    consec_red = pd.Series(0, index=close.index, dtype=int)
    for i in range(1, len(consec_red)):
        if daily_red.iloc[i-1]:
            consec_red.iloc[i] = consec_red.iloc[i-1] + 1
        else:
            consec_red.iloc[i] = 0

    # First green after N+ red
    first_green = daily_green & (consec_red >= min_red)

    # Weekly RSI declining 2+ weeks
    weekly_decline = compute_weekly_rsi(close)
    weekly_decline_condition = weekly_decline >= 2

    signal = dip_condition & rsi_condition & first_green & weekly_decline_condition
    return signal


def download_data():
    """Download all ticker data."""
    print(f"Downloading data for {len(ALL_TICKERS)} tickers...")
    # Download with extra buffer for indicator warmup
    start = (pd.Timestamp(OOT_START) - pd.DateOffset(months=6)).strftime("%Y-%m-%d")

    data = {}
    failed = []
    for ticker in ALL_TICKERS:
        try:
            df = yf.download(ticker, start=start, end=OOT_END, progress=False, auto_adjust=True)
            if df is not None and len(df) > 50:
                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[ticker] = df
            else:
                failed.append(ticker)
        except Exception as e:
            failed.append(ticker)
            print(f"  Failed {ticker}: {e}")

    if failed:
        print(f"  Failed to download: {failed}")
    print(f"  Successfully downloaded {len(data)} tickers")
    return data


def run_backtest(variant, data, spy_data, uup_data=None):
    """
    Run backtest for a specific variant.
    Returns trades list and equity curve.
    """
    # Variant configs
    configs = {
        "A": {"universes": ["US"], "capital": {"US": 645}, "max_per_univ": {"US": 3}, "max_total": 3},
        "B": {"universes": ["US", "Intl", "ETF"], "capital": {"US": 215, "Intl": 215, "ETF": 215},
               "max_per_univ": {"US": 2, "Intl": 2, "ETF": 2}, "max_total": 6},
        "C": {"universes": ["US", "Intl", "ETF"], "capital": {"US": 322.5, "Intl": 193.5, "ETF": 129},
               "max_per_univ": {"US": 6, "Intl": 6, "ETF": 6}, "max_total": 6},
        "D": {"universes": ["US", "Intl"], "capital": {"US": 322.5, "Intl": 322.5},
               "max_per_univ": {"US": 3, "Intl": 3}, "max_total": 6},
        "E": {"universes": ["US", "Intl", "ETF"], "capital": {"US": 215, "Intl": 215, "ETF": 215},
               "max_per_univ": {"US": 3, "Intl": 3, "ETF": 3}, "max_total": 3},
        "F": {"universes": ["US", "Intl", "ETF"], "capital": {"US": 215, "Intl": 215, "ETF": 215},
               "max_per_univ": {"US": 6, "Intl": 6, "ETF": 6}, "max_total": 6},
    }

    cfg = configs[variant]

    # Max trade size
    max_trade = {
        "A": {"US": 200},
        "B": {"US": 107.5, "Intl": 107.5, "ETF": 107.5},
        "C": {"US": 53.75, "Intl": 32.25, "ETF": 21.5},
        "D": {"US": 107.5, "Intl": 107.5},
        "E": {"US": 71.67, "Intl": 71.67, "ETF": 71.67},
        "F": {"US": 107.5, "Intl": 107.5, "ETF": 107.5},
    }

    # Build signal calendar
    all_signals = []
    tickers_for_variant = []
    for univ in cfg["universes"]:
        if univ == "US":
            tickers_for_variant.extend(US_QUALITY)
        elif univ == "Intl":
            tickers_for_variant.extend(INTL_ADRS)
        elif univ == "ETF":
            tickers_for_variant.extend(SECTOR_ETFS)

    for ticker in tickers_for_variant:
        if ticker not in data:
            continue
        df = data[ticker]
        signals = generate_signals(df, ticker)
        sig_dates = signals[signals].index
        sig_dates = sig_dates[(sig_dates >= OOT_START) & (sig_dates <= OOT_END)]
        for d in sig_dates:
            rsi_val = compute_rsi(df["Close"], 14).loc[:d].iloc[-1] if d in df.index else 50
            all_signals.append({"ticker": ticker, "date": d, "rsi": rsi_val})

    all_signals.sort(key=lambda x: (x["date"], x["rsi"]))  # sort by date, then lowest RSI first

    # Simulate
    trades = []
    open_positions = []  # list of {ticker, universe, entry_date, entry_price, shares, exit_date}
    equity = STARTING_CAPITAL
    equity_curve = []

    # Get all trading dates in OOT
    all_dates = spy_data.index[(spy_data.index >= OOT_START) & (spy_data.index <= OOT_END)]

    # UUP condition for variant F
    uup_sma20 = None
    if variant == "F" and uup_data is not None:
        uup_sma20 = uup_data["Close"].rolling(20).mean()

    for date in all_dates:
        # Close positions that have reached hold period
        still_open = []
        for pos in open_positions:
            if (date - pos["entry_date"]).days >= HOLD_DAYS:
                # Exit
                ticker = pos["ticker"]
                if ticker in data and date in data[ticker].index:
                    exit_price = data[ticker]["Close"].loc[date]
                else:
                    # Find nearest prior date
                    avail = data[ticker].index[data[ticker].index <= date]
                    if len(avail) == 0:
                        still_open.append(pos)
                        continue
                    exit_price = data[ticker]["Close"].iloc[-1]

                univ = get_universe(ticker)
                slip = SLIPPAGE_ETF if univ == "ETF" else SLIPPAGE_STOCK
                exit_price_adj = exit_price * (1 - slip)

                pnl = (exit_price_adj - pos["entry_price"]) * pos["shares"]
                ret = (exit_price_adj / pos["entry_price"]) - 1
                equity += pnl
                trades.append({
                    "ticker": ticker,
                    "universe": univ,
                    "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                    "exit_date": date.strftime("%Y-%m-%d"),
                    "entry_price": round(pos["entry_price"], 2),
                    "exit_price": round(exit_price_adj, 2),
                    "shares": round(pos["shares"], 4),
                    "pnl": round(pnl, 2),
                    "return": round(ret, 4),
                })
            else:
                still_open.append(pos)
        open_positions = still_open

        # Count open by universe
        open_by_univ = {}
        for pos in open_positions:
            u = get_universe(pos["ticker"])
            open_by_univ[u] = open_by_univ.get(u, 0) + 1
        total_open = len(open_positions)
        open_tickers = {pos["ticker"] for pos in open_positions}

        # Check for new signals on this date
        day_signals = [s for s in all_signals if s["date"] == date]

        # Variant E: prioritize deepest oversold (lowest RSI) across universes
        # Already sorted by RSI ascending

        for sig in day_signals:
            ticker = sig["ticker"]
            univ = get_universe(ticker)
            if univ not in cfg["universes"]:
                continue
            if ticker in open_tickers:
                continue
            if total_open >= cfg["max_total"]:
                break
            if open_by_univ.get(univ, 0) >= cfg["max_per_univ"].get(univ, 99):
                continue

            # Variant F: reduce international when USD strengthening
            if variant == "F" and univ == "Intl" and uup_sma20 is not None:
                if date in uup_data.index and date in uup_sma20.index:
                    if uup_data["Close"].loc[date] > uup_sma20.loc[date]:
                        continue  # Skip intl when USD strong

            # Enter position
            if date not in data[ticker].index:
                continue
            entry_price = data[ticker]["Close"].loc[date]
            slip = SLIPPAGE_ETF if univ == "ETF" else SLIPPAGE_STOCK
            entry_price_adj = entry_price * (1 + slip)

            trade_cap = max_trade[variant].get(univ, 200)
            shares = trade_cap / entry_price_adj
            if shares * entry_price_adj > equity * 0.95:  # Don't use more than 95% of remaining
                shares = (equity * 0.3) / entry_price_adj  # Scale down
            if shares <= 0:
                continue

            open_positions.append({
                "ticker": ticker,
                "universe": univ,
                "entry_date": date,
                "entry_price": entry_price_adj,
                "shares": shares,
            })
            open_tickers.add(ticker)
            open_by_univ[univ] = open_by_univ.get(univ, 0) + 1
            total_open += 1

        # Record equity (mark to market)
        mtm = equity
        for pos in open_positions:
            ticker = pos["ticker"]
            if ticker in data and date in data[ticker].index:
                current = data[ticker]["Close"].loc[date]
                mtm += (current - pos["entry_price"]) * pos["shares"]
        equity_curve.append({"date": date.strftime("%Y-%m-%d"), "equity": round(mtm, 2)})

    # Force close remaining positions at last available date
    for pos in open_positions:
        ticker = pos["ticker"]
        if ticker in data and len(data[ticker]) > 0:
            exit_price = data[ticker]["Close"].iloc[-1]
            univ = get_universe(ticker)
            slip = SLIPPAGE_ETF if univ == "ETF" else SLIPPAGE_STOCK
            exit_price_adj = exit_price * (1 - slip)
            pnl = (exit_price_adj - pos["entry_price"]) * pos["shares"]
            ret = (exit_price_adj / pos["entry_price"]) - 1
            trades.append({
                "ticker": ticker,
                "universe": univ,
                "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                "exit_date": "2026-07-31",
                "entry_price": round(pos["entry_price"], 2),
                "exit_price": round(exit_price_adj, 2),
                "shares": round(pos["shares"], 4),
                "pnl": round(pnl, 2),
                "return": round(ret, 4),
            })

    return trades, equity_curve


def compute_metrics(trades, equity_curve):
    """Compute performance metrics from trades."""
    if not trades:
        return {
            "sharpe": 0, "sortino": 0, "win_rate": 0, "profit_factor": 0,
            "max_drawdown": 0, "total_return": 0, "num_trades": 0,
            "avg_return": 0, "avg_winner": 0, "avg_loser": 0,
        }

    returns = [t["return"] for t in trades]
    pnls = [t["pnl"] for t in trades]

    winners = [r for r in returns if r > 0]
    losers = [r for r in returns if r <= 0]

    win_rate = len(winners) / len(returns) if returns else 0

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Sharpe and Sortino (annualized, assuming ~25 trades/year avg)
    avg_ret = np.mean(returns)
    std_ret = np.std(returns) if len(returns) > 1 else 1
    downside = [r for r in returns if r < 0]
    downside_std = np.std(downside) if len(downside) > 1 else std_ret

    trades_per_year = len(returns) / 4.5  # ~4.5 years of OOT
    annualization = np.sqrt(max(trades_per_year, 1))

    sharpe = (avg_ret / std_ret) * annualization if std_ret > 0 else 0
    sortino = (avg_ret / downside_std) * annualization if downside_std > 0 else 0

    # Max drawdown from equity curve
    equities = [e["equity"] for e in equity_curve]
    if equities:
        peak = equities[0]
        max_dd = 0
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (eq - peak) / peak
            if dd < max_dd:
                max_dd = dd
    else:
        max_dd = 0

    total_return = (equities[-1] / equities[0] - 1) if equities and equities[0] > 0 else 0

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 3),
        "max_drawdown": round(max_dd, 4),
        "total_return": round(total_return, 4),
        "num_trades": len(trades),
        "avg_return": round(avg_ret, 4),
        "avg_winner": round(np.mean(winners), 4) if winners else 0,
        "avg_loser": round(np.mean(losers), 4) if losers else 0,
    }


def regime_analysis(trades, spy_data):
    """Split trades into bull (SPY > 200-SMA) and bear regimes."""
    spy_sma200 = spy_data["Close"].rolling(200).mean()

    bull_trades = []
    bear_trades = []

    for t in trades:
        entry = pd.Timestamp(t["entry_date"])
        if entry in spy_data.index and entry in spy_sma200.index:
            if spy_data["Close"].loc[entry] > spy_sma200.loc[entry]:
                bull_trades.append(t)
            else:
                bear_trades.append(t)
        else:
            # Find nearest
            idx = spy_data.index[spy_data.index <= entry]
            if len(idx) > 0:
                nearest = idx[-1]
                if spy_data["Close"].loc[nearest] > spy_sma200.loc[nearest]:
                    bull_trades.append(t)
                else:
                    bear_trades.append(t)

    bull_returns = [t["return"] for t in bull_trades]
    bear_returns = [t["return"] for t in bear_trades]

    def calc_sharpe(rets):
        if len(rets) < 2:
            return 0
        avg = np.mean(rets)
        std = np.std(rets)
        tpy = len(rets) / 4.5
        ann = np.sqrt(max(tpy, 1))
        return round((avg / std) * ann, 3) if std > 0 else 0

    bull_sharpe = calc_sharpe(bull_returns)
    bear_sharpe = calc_sharpe(bear_returns)
    gap = abs(bull_sharpe - bear_sharpe)

    return {
        "bull_trades": len(bull_trades),
        "bear_trades": len(bear_trades),
        "bull_sharpe": bull_sharpe,
        "bear_sharpe": bear_sharpe,
        "regime_gap": round(gap, 3),
        "bull_wr": round(len([r for r in bull_returns if r > 0]) / len(bull_returns), 4) if bull_returns else 0,
        "bear_wr": round(len([r for r in bear_returns if r > 0]) / len(bear_returns), 4) if bear_returns else 0,
    }


def universe_breakdown(trades):
    """Break down trades by universe."""
    breakdown = {}
    for univ in ["US", "Intl", "ETF"]:
        univ_trades = [t for t in trades if t["universe"] == univ]
        if univ_trades:
            rets = [t["return"] for t in univ_trades]
            breakdown[univ] = {
                "trade_count": len(univ_trades),
                "avg_return": round(np.mean(rets), 4),
                "win_rate": round(len([r for r in rets if r > 0]) / len(rets), 4),
                "total_pnl": round(sum(t["pnl"] for t in univ_trades), 2),
            }
    return breakdown


def permutation_test(trades, n_iter=1000):
    """Permutation test: shuffle trade returns, compute fraction with Sharpe >= observed."""
    if len(trades) < 5:
        return 1.0

    returns = np.array([t["return"] for t in trades])
    observed_sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0

    rng = np.random.RandomState(42)
    count = 0
    for _ in range(n_iter):
        # Shuffle signs randomly
        signs = rng.choice([-1, 1], size=len(returns))
        shuffled = returns * signs
        s = np.mean(shuffled) / np.std(shuffled) if np.std(shuffled) > 0 else 0
        if s >= observed_sharpe:
            count += 1

    return round(count / n_iter, 4)


def five_gate_validation(metrics, perm_p, regime):
    """5-gate validation."""
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": perm_p < 0.05,
        "regime_gap_lt_0.5": regime["regime_gap"] < 0.5,
        "mdd_gt_neg50": metrics["max_drawdown"] > -0.50,
        "trades_gte_20": metrics["num_trades"] >= 20,
    }
    gates["passed"] = all(gates.values())
    gates["gates_passed"] = sum(1 for v in list(gates.values())[:-1] if v)
    return gates


def main():
    print("=" * 70)
    print("Multi-TF L + Multi-Asset Portfolio Backtest")
    print("=" * 70)

    data = download_data()

    spy_data = data.get("SPY")
    uup_data = data.get("UUP")

    if spy_data is None:
        print("ERROR: Could not download SPY data")
        sys.exit(1)

    results = {}
    variants = ["A", "B", "C", "D", "E", "F"]

    for v in variants:
        print(f"\n{'─'*50}")
        print(f"Running Variant {v}...")
        trades, equity_curve = run_backtest(v, data, spy_data, uup_data)

        metrics = compute_metrics(trades, equity_curve)
        regime = regime_analysis(trades, spy_data)
        breakdown = universe_breakdown(trades)
        perm_p = permutation_test(trades, 1000)
        gates = five_gate_validation(metrics, perm_p, regime)

        results[f"Variant_{v}"] = {
            "metrics": metrics,
            "regime": regime,
            "universe_breakdown": breakdown,
            "permutation_p": perm_p,
            "five_gate": gates,
        }

        print(f"  Trades: {metrics['num_trades']}, Sharpe: {metrics['sharpe']}, "
              f"Sortino: {metrics['sortino']}, WR: {metrics['win_rate']:.1%}, "
              f"PF: {metrics['profit_factor']}, MDD: {metrics['max_drawdown']:.1%}, "
              f"Return: {metrics['total_return']:.1%}")
        print(f"  Regime: Bull Sharpe={regime['bull_sharpe']}, Bear Sharpe={regime['bear_sharpe']}, "
              f"Gap={regime['regime_gap']}")
        print(f"  Perm p={perm_p}, Gates: {gates['gates_passed']}/5 {'PASS' if gates['passed'] else 'FAIL'}")
        if breakdown:
            for u, b in breakdown.items():
                print(f"    {u}: {b['trade_count']} trades, WR={b['win_rate']:.1%}, "
                      f"Avg={b['avg_return']:.2%}, PnL=${b['total_pnl']:.2f}")

    # Save results — convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    results = convert(results)
    output_path = "/home/jupiter/Lvl3Quant/data/multi_tf_L_multi_asset_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary table
    print(f"\n{'='*90}")
    print(f"{'Variant':<10} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'WR':>7} {'PF':>7} {'MDD':>8} {'Return':>8} {'Gates':>6}")
    print(f"{'─'*90}")
    for v in variants:
        m = results[f"Variant_{v}"]["metrics"]
        g = results[f"Variant_{v}"]["five_gate"]
        status = "PASS" if g["passed"] else "FAIL"
        print(f"  {v:<8} {m['num_trades']:>6} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{m['win_rate']:>6.1%} {m['profit_factor']:>7.3f} {m['max_drawdown']:>7.1%} "
              f"{m['total_return']:>7.1%} {g['gates_passed']}/5 {status}")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
