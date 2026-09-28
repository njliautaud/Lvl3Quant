#!/usr/bin/env python3
"""
Scaled/Pyramided Mean Reversion Entry Backtest on Quality Stocks
6 variants (A-F) with 5-gate validation.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from collections import defaultdict

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]
CAPITAL = 669.0
SLIPPAGE_BPS = 2
START = "2022-01-01"
END = "2026-07-31"
DOWNLOAD_START = "2021-06-01"
PERMUTATION_N = 1000
RSI_PERIOD = 14


# ── Data Download ───────────────────────────────────────────────────────────
def download_data():
    tickers = UNIVERSE + ["SPY"]
    print(f"Downloading {len(tickers)} tickers...")
    raw = yf.download(tickers, start=DOWNLOAD_START, end=END, auto_adjust=True, progress=False)
    close = raw["Close"]
    if hasattr(close.columns, 'get_level_values'):
        close.columns = [str(c) for c in close.columns]
    close = close.ffill()
    return close


def compute_rsi(series, period=RSI_PERIOD):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_indicators(close_df):
    indicators = {}
    for ticker in UNIVERSE:
        if ticker not in close_df.columns:
            continue
        c = close_df[ticker].dropna()
        high20 = c.rolling(20).max()
        sma20 = c.rolling(20).mean()
        rsi = compute_rsi(c)
        log_ret = np.log(c / c.shift(1))
        vol20 = log_ret.rolling(20).std() * np.sqrt(252) * 100
        drawdown_pct = (c - high20) / high20 * 100
        indicators[ticker] = pd.DataFrame({
            "close": c,
            "high20": high20,
            "sma20": sma20,
            "rsi": rsi,
            "vol20": vol20,
            "dd_pct": drawdown_pct,
        })
    return indicators


def apply_slippage(price, direction="buy"):
    slip = price * SLIPPAGE_BPS / 10000
    return price + slip if direction == "buy" else price - slip


# ── Position / Trade tracking ───────────────────────────────────────────────
class Position:
    def __init__(self, ticker, entry_date, entry_price, dollars, variant_meta=None):
        self.ticker = ticker
        self.entries = [(entry_date, entry_price, dollars)]
        self.total_dollars = dollars
        self.shares = dollars / entry_price
        self.avg_price = entry_price
        self.last_entry_date = entry_date
        self.peak_price = entry_price
        self.prev_close = entry_price  # for daily P&L tracking
        self.meta = variant_meta or {}

    def add_tranche(self, date, price, dollars):
        self.entries.append((date, price, dollars))
        new_shares = dollars / price
        self.total_dollars += dollars
        self.shares += new_shares
        self.avg_price = self.total_dollars / self.shares
        self.last_entry_date = date

    def day_pnl(self, current_price):
        """Return day-over-day P&L and update prev_close."""
        pnl = self.shares * (current_price - self.prev_close)
        self.prev_close = current_price
        return pnl

    def close(self, exit_date, exit_price):
        sell_price = apply_slippage(exit_price, "sell")
        pnl = self.shares * (sell_price - self.avg_price)
        ret = (sell_price - self.avg_price) / self.avg_price
        entry_d = self.entries[0][0]
        return {
            "ticker": self.ticker,
            "entry_date": str(entry_d.date()) if hasattr(entry_d, 'date') else str(entry_d),
            "exit_date": str(exit_date.date()) if hasattr(exit_date, 'date') else str(exit_date),
            "n_tranches": len(self.entries),
            "total_invested": round(float(self.total_dollars), 2),
            "shares": round(float(self.shares), 4),
            "avg_price": round(float(self.avg_price), 2),
            "exit_price": round(float(sell_price), 2),
            "pnl": round(float(pnl), 2),
            "return_pct": round(float(ret * 100), 4),
            "hold_days": int((exit_date - entry_d).days) if hasattr(exit_date, 'date') else 0,
        }


def get_daily_pnl(positions, indicators, date):
    """Compute day-over-day P&L for all positions."""
    total = 0.0
    for pos in positions:
        t = pos.ticker
        if t in indicators and date in indicators[t].index:
            cur = indicators[t].loc[date, "close"]
            total += pos.day_pnl(cur)
    return total


# ── Variant Strategies ──────────────────────────────────────────────────────

def run_variant_a(indicators, close_df, dates):
    """Tranche Entry (3 levels)"""
    positions = []
    trades = []
    daily_pnl = []

    for date in dates:
        # Exits: 10 days from LAST tranche entry
        closed = []
        for pos in positions:
            if (date - pos.last_entry_date).days >= 10:
                t = pos.ticker
                if t in indicators and date in indicators[t].index:
                    trades.append(pos.close(date, indicators[t].loc[date, "close"]))
                    closed.append(pos)
        for c in closed:
            positions.remove(c)

        # Add tranches to existing positions
        active_tickers = {p.ticker for p in positions}
        for pos in positions:
            t = pos.ticker
            if t not in indicators or date not in indicators[t].index:
                continue
            row = indicators[t].loc[date]
            dd = row["dd_pct"]
            n = len(pos.entries)
            if n == 1 and dd <= -10 and pos.total_dollars < 200:
                buy_px = apply_slippage(row["close"], "buy")
                pos.add_tranche(date, buy_px, 67)
            elif n == 2 and dd <= -15 and pos.total_dollars < 200:
                buy_px = apply_slippage(row["close"], "buy")
                amt = min(67, 201 - pos.total_dollars)
                if amt > 0:
                    pos.add_tranche(date, buy_px, amt)

        # New positions
        for ticker in UNIVERSE:
            if ticker in active_tickers or len(positions) >= 3:
                continue
            if ticker not in indicators or date not in indicators[ticker].index:
                continue
            row = indicators[ticker].loc[date]
            if row["dd_pct"] <= -5 and row["rsi"] < 40 and not np.isnan(row["rsi"]):
                buy_px = apply_slippage(row["close"], "buy")
                positions.append(Position(ticker, date, buy_px, 67))
                active_tickers.add(ticker)

        daily_pnl.append(get_daily_pnl(positions, indicators, date))

    # Close remaining
    last_date = dates[-1]
    for pos in positions:
        t = pos.ticker
        if t in indicators and last_date in indicators[t].index:
            trades.append(pos.close(last_date, indicators[t].loc[last_date, "close"]))
    return trades, daily_pnl


def run_variant_b(indicators, close_df, dates):
    """DCA Dip: buy $50/day for up to 4 days while below -5%"""
    positions = []
    trades = []
    daily_pnl = []

    for date in dates:
        closed = []
        for pos in positions:
            if (date - pos.last_entry_date).days >= 10:
                t = pos.ticker
                if t in indicators and date in indicators[t].index:
                    trades.append(pos.close(date, indicators[t].loc[date, "close"]))
                    closed.append(pos)
        for c in closed:
            positions.remove(c)

        # Add DCA buys to existing positions
        for pos in positions:
            t = pos.ticker
            if t not in indicators or date not in indicators[t].index:
                continue
            row = indicators[t].loc[date]
            if len(pos.entries) < 4 and row["dd_pct"] <= -5 and pos.total_dollars < 200:
                buy_px = apply_slippage(row["close"], "buy")
                pos.add_tranche(date, buy_px, 50)

        active_tickers = {p.ticker for p in positions}
        for ticker in UNIVERSE:
            if ticker in active_tickers or len(positions) >= 3:
                continue
            if ticker not in indicators or date not in indicators[ticker].index:
                continue
            row = indicators[ticker].loc[date]
            if row["dd_pct"] <= -5 and row["rsi"] < 40 and not np.isnan(row["rsi"]):
                buy_px = apply_slippage(row["close"], "buy")
                positions.append(Position(ticker, date, buy_px, 50))
                active_tickers.add(ticker)

        daily_pnl.append(get_daily_pnl(positions, indicators, date))

    last_date = dates[-1]
    for pos in positions:
        t = pos.ticker
        if t in indicators and last_date in indicators[t].index:
            trades.append(pos.close(last_date, indicators[t].loc[last_date, "close"]))
    return trades, daily_pnl


def run_variant_c(indicators, close_df, dates):
    """Volatility-Scaled Entry"""
    positions = []
    trades = []
    daily_pnl = []

    for date in dates:
        closed = []
        for pos in positions:
            if (date - pos.last_entry_date).days >= 10:
                t = pos.ticker
                if t in indicators and date in indicators[t].index:
                    trades.append(pos.close(date, indicators[t].loc[date, "close"]))
                    closed.append(pos)
        for c in closed:
            positions.remove(c)

        active_tickers = {p.ticker for p in positions}
        for ticker in UNIVERSE:
            if ticker in active_tickers or len(positions) >= 3:
                continue
            if ticker not in indicators or date not in indicators[ticker].index:
                continue
            row = indicators[ticker].loc[date]
            if row["dd_pct"] <= -5 and row["rsi"] < 40 and not np.isnan(row["rsi"]) and not np.isnan(row["vol20"]):
                vol = row["vol20"]
                if vol > 40:
                    dollars = 100
                elif vol > 20:
                    dollars = 150
                else:
                    dollars = 200
                buy_px = apply_slippage(row["close"], "buy")
                positions.append(Position(ticker, date, buy_px, dollars))
                active_tickers.add(ticker)

        daily_pnl.append(get_daily_pnl(positions, indicators, date))

    last_date = dates[-1]
    for pos in positions:
        t = pos.ticker
        if t in indicators and last_date in indicators[t].index:
            trades.append(pos.close(last_date, indicators[t].loc[last_date, "close"]))
    return trades, daily_pnl


def run_variant_d(indicators, close_df, dates):
    """Waterfall Entry: deepest level determines sizing and hold period"""
    positions = []
    trades = []
    daily_pnl = []

    for date in dates:
        closed = []
        for pos in positions:
            hold_days = pos.meta.get("hold_days", 10)
            if (date - pos.last_entry_date).days >= hold_days:
                t = pos.ticker
                if t in indicators and date in indicators[t].index:
                    trades.append(pos.close(date, indicators[t].loc[date, "close"]))
                    closed.append(pos)
        for c in closed:
            positions.remove(c)

        active_tickers = {p.ticker for p in positions}
        for ticker in UNIVERSE:
            if ticker in active_tickers or len(positions) >= 3:
                continue
            if ticker not in indicators or date not in indicators[ticker].index:
                continue
            row = indicators[ticker].loc[date]
            rsi = row["rsi"]
            dd = row["dd_pct"]
            if np.isnan(rsi) or rsi >= 45:
                continue

            if dd <= -15:
                dollars, hold = 200, 15
            elif dd <= -10:
                dollars, hold = 200, 10
            elif dd <= -5:
                dollars, hold = 100, 5
            else:
                continue

            buy_px = apply_slippage(row["close"], "buy")
            positions.append(Position(ticker, date, buy_px, dollars, {"hold_days": hold}))
            active_tickers.add(ticker)

        daily_pnl.append(get_daily_pnl(positions, indicators, date))

    last_date = dates[-1]
    for pos in positions:
        t = pos.ticker
        if t in indicators and last_date in indicators[t].index:
            trades.append(pos.close(last_date, indicators[t].loc[last_date, "close"]))
    return trades, daily_pnl


def run_variant_e(indicators, close_df, dates):
    """Mean Reversion with Trailing Stop"""
    positions = []
    trades = []
    daily_pnl = []

    for date in dates:
        closed = []
        for pos in positions:
            t = pos.ticker
            if t not in indicators or date not in indicators[t].index:
                continue
            cur_px = indicators[t].loc[date, "close"]
            sma20 = indicators[t].loc[date, "sma20"]
            pos.peak_price = max(pos.peak_price, cur_px)
            days_held = (date - pos.last_entry_date).days

            trailing_stop_hit = cur_px < pos.peak_price * 0.97
            target_hit = not np.isnan(sma20) and cur_px >= sma20
            time_stop = days_held >= 10

            if trailing_stop_hit or target_hit or time_stop:
                trades.append(pos.close(date, cur_px))
                closed.append(pos)

        for c in closed:
            positions.remove(c)

        active_tickers = {p.ticker for p in positions}
        for ticker in UNIVERSE:
            if ticker in active_tickers or len(positions) >= 3:
                continue
            if ticker not in indicators or date not in indicators[ticker].index:
                continue
            row = indicators[ticker].loc[date]
            if row["dd_pct"] <= -7 and row["rsi"] < 35 and not np.isnan(row["rsi"]):
                buy_px = apply_slippage(row["close"], "buy")
                pos = Position(ticker, date, buy_px, 200)
                pos.peak_price = buy_px
                positions.append(pos)
                active_tickers.add(ticker)

        daily_pnl.append(get_daily_pnl(positions, indicators, date))

    last_date = dates[-1]
    for pos in positions:
        t = pos.ticker
        if t in indicators and last_date in indicators[t].index:
            trades.append(pos.close(last_date, indicators[t].loc[last_date, "close"]))
    return trades, daily_pnl


def run_variant_f(indicators, close_df, dates):
    """Accumulation Zone: buy, wait 3 days, add if still in zone, exit on SMA cross or 15d"""
    positions = []
    trades = []
    daily_pnl = []

    for date in dates:
        closed = []
        for pos in positions:
            t = pos.ticker
            if t not in indicators or date not in indicators[t].index:
                continue
            row = indicators[t].loc[date]
            days_held = (date - pos.entries[0][0]).days

            above_sma = not np.isnan(row["sma20"]) and row["close"] > row["sma20"]
            if above_sma or days_held >= 15:
                trades.append(pos.close(date, row["close"]))
                closed.append(pos)
            else:
                # Add second tranche after 3 days if still in zone
                if len(pos.entries) < 2 and days_held >= 3 and row["dd_pct"] <= -5:
                    buy_px = apply_slippage(row["close"], "buy")
                    pos.add_tranche(date, buy_px, 100)

        for c in closed:
            positions.remove(c)

        active_tickers = {p.ticker for p in positions}
        for ticker in UNIVERSE:
            if ticker in active_tickers or len(positions) >= 3:
                continue
            if ticker not in indicators or date not in indicators[ticker].index:
                continue
            row = indicators[ticker].loc[date]
            if row["dd_pct"] <= -5 and row["rsi"] < 40 and not np.isnan(row["rsi"]):
                buy_px = apply_slippage(row["close"], "buy")
                positions.append(Position(ticker, date, buy_px, 100))
                active_tickers.add(ticker)

        daily_pnl.append(get_daily_pnl(positions, indicators, date))

    last_date = dates[-1]
    for pos in positions:
        t = pos.ticker
        if t in indicators and last_date in indicators[t].index:
            trades.append(pos.close(last_date, indicators[t].loc[last_date, "close"]))
    return trades, daily_pnl


# ── 5-Gate Validation ───────────────────────────────────────────────────────

def compute_sharpe(daily_pnl_series, capital):
    if len(daily_pnl_series) < 2:
        return 0.0
    rets = np.array(daily_pnl_series) / capital
    std = np.std(rets)
    if std == 0:
        return 0.0
    return float(np.mean(rets) / std * np.sqrt(252))


def compute_max_drawdown(daily_pnl_series, capital):
    equity = capital + np.cumsum(daily_pnl_series)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(np.min(dd)) * 100


def permutation_test(trades, daily_pnl, capital, n_perms=PERMUTATION_N):
    """Shuffle trade entry dates (randomize timing), recompute avg return, get p-value."""
    if len(trades) < 5:
        return 1.0
    actual_mean_ret = np.mean([t["return_pct"] for t in trades])
    all_returns = [t["return_pct"] for t in trades]
    count_better = 0
    rng = np.random.default_rng(42)
    for _ in range(n_perms):
        # Shuffle the assignment of returns to trade slots
        shuffled = rng.permutation(all_returns)
        if np.mean(shuffled) >= actual_mean_ret:
            count_better += 1
    # Shuffling returns and comparing means of same set = always same mean
    # Instead: bootstrap random subsets of same size from a null distribution
    # Null: random entry points -> what return would you get?
    # Approximate: shuffle the sign of each return (sign test)
    count_better = 0
    for _ in range(n_perms):
        signs = rng.choice([-1, 1], size=len(all_returns))
        shuffled_rets = np.array(all_returns) * signs
        if np.mean(shuffled_rets) >= actual_mean_ret:
            count_better += 1
    return count_better / n_perms


def regime_gap(trades, spy_close, dates):
    if len(trades) < 10:
        return 1.0

    spy_ret20 = spy_close.pct_change(20)
    bull_rets = []
    bear_rets = []
    for t in trades:
        entry_date = pd.Timestamp(t["entry_date"])
        if entry_date in spy_ret20.index:
            regime_val = spy_ret20.loc[entry_date]
            if np.isnan(regime_val):
                continue
            if regime_val >= 0:
                bull_rets.append(t["return_pct"])
            else:
                bear_rets.append(t["return_pct"])

    if len(bull_rets) < 3 or len(bear_rets) < 3:
        return 0.0

    bull_sharpe = np.mean(bull_rets) / max(np.std(bull_rets), 1e-8)
    bear_sharpe = np.mean(bear_rets) / max(np.std(bear_rets), 1e-8)
    gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 1e-8)
    return float(gap)


def validate_5gate(trades, daily_pnl, capital, spy_close, dates, variant_name):
    sharpe = compute_sharpe(daily_pnl, capital)
    max_dd = compute_max_drawdown(daily_pnl, capital)
    n_trades = len(trades)
    perm_p = permutation_test(trades, daily_pnl, capital) if n_trades >= 5 else 1.0
    rgap = regime_gap(trades, spy_close, dates) if n_trades >= 10 else 1.0

    gates = {
        "sharpe": {"value": round(sharpe, 3), "threshold": ">0.5", "pass": bool(sharpe > 0.5)},
        "permutation_p": {"value": round(perm_p, 4), "threshold": "<0.05", "pass": bool(perm_p < 0.05)},
        "regime_gap": {"value": round(float(rgap), 3), "threshold": "<0.5", "pass": bool(rgap < 0.5)},
        "max_drawdown_pct": {"value": round(max_dd, 2), "threshold": ">-50%", "pass": bool(max_dd > -50)},
        "n_trades": {"value": int(n_trades), "threshold": ">=20", "pass": bool(n_trades >= 20)},
    }
    all_pass = all(g["pass"] for g in gates.values())
    return gates, all_pass


# ── Analytics ───────────────────────────────────────────────────────────────

def summarize_trades(trades, daily_pnl, capital):
    if not trades:
        return {"n_trades": 0}
    rets = [t["return_pct"] for t in trades]
    pnls = [t["pnl"] for t in trades]
    winners = [r for r in rets if r > 0]
    losers = [r for r in rets if r <= 0]

    total_pnl = sum(pnls)
    win_rate = len(winners) / len(rets) * 100 if rets else 0
    avg_win = float(np.mean([p for p in pnls if p > 0])) if winners else 0
    avg_loss = float(np.mean([p for p in pnls if p <= 0])) if losers else 0
    gross_loss = sum(p for p in pnls if p < 0)
    profit_factor = abs(sum(p for p in pnls if p > 0) / min(gross_loss, -0.01)) if gross_loss != 0 else 999

    sharpe = compute_sharpe(daily_pnl, capital)
    max_dd = compute_max_drawdown(daily_pnl, capital)

    daily_rets = np.array(daily_pnl) / capital
    downside = daily_rets[daily_rets < 0]
    sortino = float(np.mean(daily_rets) / np.std(downside) * np.sqrt(252)) if len(downside) > 0 and np.std(downside) > 0 else 0

    avg_hold = float(np.mean([t.get("hold_days", 0) for t in trades]))
    avg_tranches = float(np.mean([t.get("n_tranches", 1) for t in trades]))
    total_return = total_pnl / capital * 100

    return {
        "n_trades": int(len(trades)),
        "total_pnl": round(float(total_pnl), 2),
        "total_return_pct": round(float(total_return), 2),
        "win_rate_pct": round(float(win_rate), 1),
        "avg_return_pct": round(float(np.mean(rets)), 3),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(float(profit_factor), 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "avg_hold_days": round(avg_hold, 1),
        "avg_tranches": round(avg_tranches, 1),
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    close_df = download_data()
    indicators = compute_indicators(close_df)

    all_dates = close_df.loc[START:END].index
    dates = list(all_dates)
    spy_close = close_df["SPY"].loc[START:END] if "SPY" in close_df.columns else pd.Series()

    print(f"Backtest period: {dates[0].date()} to {dates[-1].date()} ({len(dates)} trading days)")
    print(f"Universe: {len(UNIVERSE)} stocks, Capital: ${CAPITAL}")
    print()

    variants = {
        "A_Tranche_Entry": run_variant_a,
        "B_DCA_Dip": run_variant_b,
        "C_Vol_Scaled": run_variant_c,
        "D_Waterfall": run_variant_d,
        "E_Trailing_Stop": run_variant_e,
        "F_Accumulation_Zone": run_variant_f,
    }

    results = {}
    for name, func in variants.items():
        print(f"Running {name}...")
        trades, daily_pnl = func(indicators, close_df, dates)
        summary = summarize_trades(trades, daily_pnl, CAPITAL)
        gates, all_pass = validate_5gate(trades, daily_pnl, CAPITAL, spy_close, dates, name)
        results[name] = {
            "summary": summary,
            "gates": gates,
            "all_gates_pass": bool(all_pass),
            "trades": trades,
        }
        status = "PASS" if all_pass else "FAIL"
        print(f"  {name}: {summary['n_trades']} trades, Sharpe={summary.get('sharpe', 0)}, "
              f"WR={summary.get('win_rate_pct', 0)}%, PF={summary.get('profit_factor', 0)}, "
              f"Total={summary.get('total_return_pct', 0)}%, [{status}]")
        for gname, gval in gates.items():
            pflag = "+" if gval["pass"] else "x"
            print(f"    {pflag} {gname}: {gval['value']} (need {gval['threshold']})")
        print()

    # Summary comparison
    print("=" * 80)
    print("VARIANT COMPARISON")
    print("=" * 80)
    header = f"{'Variant':<25} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'WR%':>6} {'PF':>6} {'Return%':>8} {'MaxDD%':>7} {'Gates':>6}"
    print(header)
    print("-" * 80)
    for name, r in results.items():
        s = r["summary"]
        g = "PASS" if r["all_gates_pass"] else "FAIL"
        print(f"{name:<25} {s.get('n_trades',0):>6} {s.get('sharpe',0):>7.2f} {s.get('sortino',0):>8.2f} "
              f"{s.get('win_rate_pct',0):>5.1f}% {s.get('profit_factor',0):>5.1f} "
              f"{s.get('total_return_pct',0):>7.1f}% {s.get('max_drawdown_pct',0):>6.1f}% {g:>6}")

    # Save results
    output = {}
    for name, r in results.items():
        output[name] = {
            "summary": r["summary"],
            "gates": r["gates"],
            "all_gates_pass": r["all_gates_pass"],
            "sample_trades": r["trades"][:10],
            "total_trades_list_length": len(r["trades"]),
        }

    output_path = "/home/jupiter/Lvl3Quant/data/scaled_entry_mr_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Best variant
    passing = {k: v for k, v in results.items() if v["all_gates_pass"]}
    if passing:
        best = max(passing, key=lambda k: passing[k]["summary"]["sharpe"])
        print(f"\nBEST PASSING VARIANT: {best} (Sharpe={passing[best]['summary']['sharpe']})")
    else:
        print("\nNo variants passed all 5 gates.")
        best = max(results, key=lambda k: results[k]["summary"].get("sharpe", 0))
        print(f"Best by Sharpe (but failed gates): {best} (Sharpe={results[best]['summary'].get('sharpe', 0)})")


if __name__ == "__main__":
    main()
