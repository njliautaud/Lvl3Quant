#!/usr/bin/env python3
"""
Consecutive Down Day Patterns on Quality Stocks — Backtest
Tests whether consecutive red day count/pattern predicts mean reversion quality.
"""

import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]
CAPITAL = 669.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
SLIPPAGE_BPS = 2
BT_START = "2022-01-01"
BT_END = "2026-07-31"
DL_START = "2021-06-01"  # extra lookback
HOLD_DAYS = 10
N_PERMS = 1000

OUTPUT_PATH = Path("/home/jupiter/Lvl3Quant/data/consecutive_dip_pattern_results.json")


# ── Data Download ───────────────────────────────────────────────────────────
def download_data():
    print("Downloading price data …")
    tickers = UNIVERSE + ["SPY"]
    data = yf.download(tickers, start=DL_START, end=BT_END, auto_adjust=True, progress=False)
    close = data["Close"]
    volume = data["Volume"]
    # Ensure columns are simple strings
    close.columns = [str(c) for c in close.columns]
    volume.columns = [str(c) for c in volume.columns]
    return close, volume


# ── Feature helpers ─────────────────────────────────────────────────────────
def consecutive_red_days(close_series):
    """Return series of consecutive red day count ending on each day."""
    ret = close_series.pct_change()
    red = (ret < 0).astype(int)
    streak = pd.Series(0, index=close_series.index, dtype=int)
    for i in range(len(red)):
        if red.iloc[i] == 1:
            streak.iloc[i] = (streak.iloc[i - 1] + 1) if i > 0 else 1
        else:
            streak.iloc[i] = 0
    return streak


def pct_below_high(close_series, window=20):
    """Percent below rolling high."""
    rh = close_series.rolling(window).max()
    return (close_series - rh) / rh


def weekly_rsi(close_series, period=14):
    """RSI on weekly closes."""
    weekly = close_series.resample("W-FRI").last().dropna()
    delta = weekly.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    return rsi.reindex(close_series.index, method="ffill")


# ── Signal generators ──────────────────────────────────────────────────────
def signals_A(close, volume, ticker):
    """5+ consecutive red days AND >5% below 20d high."""
    c = close[ticker].dropna()
    streak = consecutive_red_days(c)
    pb = pct_below_high(c, 20)
    mask = (streak >= 5) & (pb < -0.05)
    return c.index[mask].tolist()


def signals_B(close, volume, ticker):
    """3+ red days with each day's loss larger than previous."""
    c = close[ticker].dropna()
    ret = c.pct_change()
    pb = pct_below_high(c, 20)
    dates = []
    for i in range(3, len(c)):
        # need 3+ consecutive red with deepening
        reds = []
        for j in range(i, max(i - 10, -1), -1):
            r = ret.iloc[j]
            if r < 0:
                reds.append(r)
            else:
                break
        if len(reds) < 3:
            continue
        reds = reds[::-1]  # chronological
        deepening = all(reds[k] < reds[k - 1] for k in range(1, len(reds)))
        if deepening and pb.iloc[i] < -0.05:
            dates.append(c.index[i])
    return dates


def signals_C(close, volume, ticker):
    """4+ red days AND volume < 20d avg (exhaustion)."""
    c = close[ticker].dropna()
    v = volume[ticker].reindex(c.index)
    streak = consecutive_red_days(c)
    v_avg = v.rolling(20).mean()
    pb = pct_below_high(c, 20)
    mask = (streak >= 4) & (v < v_avg) & (pb < -0.05)
    return c.index[mask].tolist()


def signals_D(close, volume, ticker):
    """3+ red days, position size proportional to streak length.
    Returns list of (date, size) tuples."""
    c = close[ticker].dropna()
    streak = consecutive_red_days(c)
    results = []
    for i in range(len(c)):
        s = streak.iloc[i]
        if s >= 3:
            if s == 3:
                sz = 100.0
            elif s == 4:
                sz = 150.0
            else:
                sz = 200.0
            results.append((c.index[i], sz))
    return results


def signals_E(close, volume, ticker):
    """4+ red days >5% below high, then buy on first green day."""
    c = close[ticker].dropna()
    ret = c.pct_change()
    streak = consecutive_red_days(c)
    pb = pct_below_high(c, 20)
    dates = []
    for i in range(1, len(c)):
        if streak.iloc[i - 1] >= 4 and pb.iloc[i - 1] < -0.05 and ret.iloc[i] > 0:
            dates.append(c.index[i])
    return dates


def signals_F(close, volume, ticker):
    """2+ consecutive red weeks AND >7% below 20d high AND weekly RSI<40. Hold 15d."""
    c = close[ticker].dropna()
    weekly_c = c.resample("W-FRI").last().dropna()
    weekly_ret = weekly_c.pct_change()
    weekly_red = (weekly_ret < 0).astype(int)
    w_streak = pd.Series(0, index=weekly_c.index, dtype=int)
    for i in range(len(weekly_red)):
        if weekly_red.iloc[i] == 1:
            w_streak.iloc[i] = (w_streak.iloc[i - 1] + 1) if i > 0 else 1
        else:
            w_streak.iloc[i] = 0

    rsi = weekly_rsi(c, 14)
    pb = pct_below_high(c, 20)

    # map weekly streak back to daily: on the last day of each week
    w_streak_daily = w_streak.reindex(c.index, method="ffill")

    dates = []
    for i in range(len(c)):
        dt = c.index[i]
        if (w_streak_daily.iloc[i] >= 2 and pb.iloc[i] < -0.07
                and rsi.iloc[i] < 40 if not np.isnan(rsi.iloc[i]) else False):
            # only trigger on Fridays (end of week) to avoid repeated signals
            if dt.weekday() == 4:
                dates.append(dt)
    return dates


# ── Backtester ──────────────────────────────────────────────────────────────
def run_backtest(close, volume, signal_func, variant_name, hold=HOLD_DAYS,
                 max_trade=MAX_PER_TRADE, proportional=False):
    """Run backtest for one variant. Returns trades list and equity curve."""
    bt_dates = close.index[(close.index >= BT_START) & (close.index <= BT_END)]
    trades = []

    for ticker in UNIVERSE:
        if proportional:
            raw_signals = signal_func(close, volume, ticker)
            signal_dates = [(d, sz) for d, sz in raw_signals
                            if BT_START <= str(d.date()) <= BT_END]
        else:
            raw = signal_func(close, volume, ticker)
            signal_dates = [(d, max_trade) for d in raw
                            if BT_START <= str(d.date()) <= BT_END]

        c = close[ticker]
        for entry_date, size in signal_dates:
            idx = close.index.get_loc(entry_date)
            exit_idx = min(idx + hold, len(close) - 1)
            exit_date = close.index[exit_idx]
            entry_price = c.iloc[idx]
            exit_price = c.iloc[exit_idx]
            if pd.isna(entry_price) or pd.isna(exit_price):
                continue
            # slippage
            entry_adj = entry_price * (1 + SLIPPAGE_BPS / 10000)
            exit_adj = exit_price * (1 - SLIPPAGE_BPS / 10000)
            shares = int(size / entry_adj) if entry_adj > 0 else 0
            if shares == 0:
                continue
            pnl = shares * (exit_adj - entry_adj)
            ret = (exit_adj - entry_adj) / entry_adj
            trades.append({
                "ticker": ticker,
                "entry_date": str(entry_date.date()),
                "exit_date": str(exit_date.date()),
                "entry_price": round(float(entry_adj), 4),
                "exit_price": round(float(exit_adj), 4),
                "shares": shares,
                "pnl": round(float(pnl), 4),
                "return": round(float(ret), 6),
                "size": round(float(size), 2),
            })

    # enforce max concurrent: sort by entry date, skip if too many open
    trades.sort(key=lambda t: t["entry_date"])
    filtered = []
    for t in trades:
        open_count = sum(
            1 for ft in filtered
            if ft["exit_date"] > t["entry_date"]
        )
        if open_count < MAX_CONCURRENT:
            filtered.append(t)

    # build daily equity curve
    equity = pd.Series(0.0, index=bt_dates)
    for t in filtered:
        ed = pd.Timestamp(t["entry_date"])
        xd = pd.Timestamp(t["exit_date"])
        mask = (bt_dates >= ed) & (bt_dates <= xd)
        n_days = mask.sum()
        if n_days > 0:
            daily_pnl = t["pnl"] / n_days
            equity.loc[mask] += daily_pnl

    cum_equity = equity.cumsum()
    return filtered, cum_equity


# ── Metrics ─────────────────────────────────────────────────────────────────
def compute_metrics(trades, equity_curve):
    if not trades:
        return {"n_trades": 0, "sharpe": 0, "sortino": 0, "pf": 0, "wr": 0,
                "total_pnl": 0, "max_dd_pct": 0, "avg_return": 0}

    returns = [t["return"] for t in trades]
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    wr = len(wins) / len(pnls) if pnls else 0
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")

    # annualized sharpe from trade returns
    avg_r = np.mean(returns)
    std_r = np.std(returns, ddof=1) if len(returns) > 1 else 1e-9
    trades_per_year = len(trades) / 4.5  # ~4.5 year period
    sharpe = (avg_r / std_r) * np.sqrt(trades_per_year) if std_r > 0 else 0

    # sortino
    downside = [r for r in returns if r < 0]
    down_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = (avg_r / down_std) * np.sqrt(trades_per_year) if down_std > 0 else 0

    # max drawdown from equity curve
    cum = equity_curve
    running_max = cum.cummax()
    dd = cum - running_max
    max_dd = dd.min()
    max_dd_pct = (max_dd / (CAPITAL + running_max.max())) * 100 if running_max.max() > 0 else 0

    return {
        "n_trades": len(trades),
        "total_pnl": round(float(total_pnl), 2),
        "avg_return": round(float(avg_r * 100), 4),
        "wr": round(float(wr * 100), 2),
        "pf": round(float(pf), 3),
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "max_dd_pct": round(float(max_dd_pct), 2),
    }


# ── 5-Gate Validation ───────────────────────────────────────────────────────
def permutation_test(trades, close, n_perms=N_PERMS):
    """Shuffle entry dates among valid trading days, compute p-value."""
    if len(trades) < 5:
        return 1.0
    actual_mean = np.mean([t["return"] for t in trades])
    bt_dates = close.index[(close.index >= BT_START) & (close.index <= BT_END)]
    count_better = 0
    for _ in range(n_perms):
        perm_returns = []
        for t in trades:
            rand_idx = np.random.randint(0, max(1, len(bt_dates) - HOLD_DAYS - 1))
            entry_p = close[t["ticker"]].iloc[rand_idx] if rand_idx < len(close) else np.nan
            exit_idx = min(rand_idx + HOLD_DAYS, len(close) - 1)
            exit_p = close[t["ticker"]].iloc[exit_idx] if exit_idx < len(close) else np.nan
            if pd.notna(entry_p) and pd.notna(exit_p) and entry_p > 0:
                perm_returns.append((exit_p - entry_p) / entry_p)
        if perm_returns and np.mean(perm_returns) >= actual_mean:
            count_better += 1
    return count_better / n_perms


def regime_gap(trades, spy_close):
    """Split trades by SPY 20d return regime, compute Sharpe gap."""
    spy_ret20 = spy_close.pct_change(20)
    bull_returns, bear_returns = [], []
    for t in trades:
        dt = pd.Timestamp(t["entry_date"])
        if dt in spy_ret20.index:
            regime_val = spy_ret20.loc[:dt].iloc[-1] if dt <= spy_ret20.index[-1] else 0
        else:
            nearest = spy_ret20.index[spy_ret20.index.get_indexer([dt], method="ffill")[0]]
            regime_val = spy_ret20.loc[nearest]
        if regime_val >= 0:
            bull_returns.append(t["return"])
        else:
            bear_returns.append(t["return"])

    def _sharpe(rets):
        if len(rets) < 2:
            return 0
        return np.mean(rets) / (np.std(rets, ddof=1) + 1e-9)

    s_bull = _sharpe(bull_returns)
    s_bear = _sharpe(bear_returns)
    denom = max(abs(s_bull), abs(s_bear), 1e-9)
    gap = abs(s_bull - s_bear) / denom
    return round(float(gap), 4), len(bull_returns), len(bear_returns)


def validate_5gate(metrics, trades, close, spy_close, variant_name):
    """Run 5-gate validation. Returns dict with pass/fail per gate."""
    gates = {}

    # Gate 1: Sharpe > 0.5
    gates["sharpe_gt_0.5"] = metrics["sharpe"] > 0.5

    # Gate 2: Permutation test p < 0.05
    p_val = permutation_test(trades, close)
    gates["perm_p_value"] = round(p_val, 4)
    gates["perm_pass"] = p_val < 0.05

    # Gate 3: Regime gap < 0.5
    gap, n_bull, n_bear = regime_gap(trades, spy_close)
    gates["regime_gap"] = gap
    gates["regime_bull_trades"] = n_bull
    gates["regime_bear_trades"] = n_bear
    gates["regime_pass"] = gap < 0.5

    # Gate 4: Max DD > -50%
    gates["max_dd_pass"] = metrics["max_dd_pct"] > -50

    # Gate 5: At least 20 trades
    gates["min_trades_pass"] = metrics["n_trades"] >= 20

    gates["all_pass"] = all([
        gates["sharpe_gt_0.5"],
        gates["perm_pass"],
        gates["regime_pass"],
        gates["max_dd_pass"],
        gates["min_trades_pass"],
    ])

    return gates


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    close, volume = download_data()
    spy_close = close["SPY"]

    variants = {
        "A_5plus_red_days": (signals_A, HOLD_DAYS, False),
        "B_deepening_losses": (signals_B, HOLD_DAYS, False),
        "C_volume_exhaustion": (signals_C, HOLD_DAYS, False),
        "D_proportional_sizing": (signals_D, HOLD_DAYS, True),
        "E_reversal_confirmation": (signals_E, HOLD_DAYS, False),
        "F_weekly_red_pattern": (signals_F, 15, False),
    }

    results = {}
    for name, (sig_func, hold, proportional) in variants.items():
        print(f"\n{'='*60}")
        print(f"  Variant {name}")
        print(f"{'='*60}")

        trades, equity = run_backtest(close, volume, sig_func, name,
                                       hold=hold, proportional=proportional)
        metrics = compute_metrics(trades, equity)
        gates = validate_5gate(metrics, trades, close, spy_close, name)

        print(f"  Trades: {metrics['n_trades']}")
        print(f"  Total PnL: ${metrics['total_pnl']:.2f}")
        print(f"  Win Rate: {metrics['wr']:.1f}%")
        print(f"  Profit Factor: {metrics['pf']:.3f}")
        print(f"  Sharpe: {metrics['sharpe']:.3f}")
        print(f"  Sortino: {metrics['sortino']:.3f}")
        print(f"  Max DD: {metrics['max_dd_pct']:.2f}%")
        print(f"  Avg Return: {metrics['avg_return']:.4f}%")
        print(f"  --- 5-Gate ---")
        print(f"  Sharpe>0.5: {'PASS' if gates['sharpe_gt_0.5'] else 'FAIL'}")
        print(f"  Perm p={gates['perm_p_value']:.4f}: {'PASS' if gates['perm_pass'] else 'FAIL'}")
        print(f"  Regime gap={gates['regime_gap']:.4f}: {'PASS' if gates['regime_pass'] else 'FAIL'}")
        print(f"  MaxDD>-50%: {'PASS' if gates['max_dd_pass'] else 'FAIL'}")
        print(f"  Trades>=20: {'PASS' if gates['min_trades_pass'] else 'FAIL'}")
        print(f"  ALL GATES: {'PASS' if gates['all_pass'] else 'FAIL'}")

        # top tickers
        if trades:
            ticker_pnl = {}
            for t in trades:
                ticker_pnl[t["ticker"]] = ticker_pnl.get(t["ticker"], 0) + t["pnl"]
            top = sorted(ticker_pnl.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"  Top tickers: {', '.join(f'{tk}: ${pn:.2f}' for tk, pn in top)}")

        results[name] = {
            "metrics": metrics,
            "gates": gates,
            "sample_trades": trades[:10] if trades else [],
            "all_trades_count": len(trades),
        }

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    passing = [n for n, r in results.items() if r["gates"]["all_pass"]]
    print(f"  Variants passing all 5 gates: {passing if passing else 'NONE'}")
    for name, r in results.items():
        m = r["metrics"]
        g = r["gates"]
        status = "PASS" if g["all_pass"] else "FAIL"
        print(f"  {name}: {status} | {m['n_trades']} trades | "
              f"Sharpe {m['sharpe']:.3f} | WR {m['wr']:.1f}% | "
              f"PF {m['pf']:.3f} | PnL ${m['total_pnl']:.2f}")

    # Save
    output = {
        "strategy": "consecutive_dip_pattern",
        "universe": UNIVERSE,
        "period": f"{BT_START} to {BT_END}",
        "capital": CAPITAL,
        "run_date": datetime.now().isoformat(),
        "variants": results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
