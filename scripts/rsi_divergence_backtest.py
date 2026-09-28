#!/usr/bin/env python3
"""
RSI Divergence on Quality Stocks — Backtest
============================================
6 variants (A-F) testing bullish RSI divergence as a buy signal.

Divergence: price makes a lower low, RSI makes a higher low.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from itertools import product

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]
START = "2022-01-01"
END = "2026-07-31"
CAPITAL = 645.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
SLIPPAGE_BPS = 2
N_PERMUTATIONS = 1000
RSI_PERIOD = 14
LOCAL_LOW_WINDOW = 5  # days each side for local low detection


# ── Helpers ─────────────────────────────────────────────────────────────────
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_macd_hist(close: pd.Series) -> pd.Series:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line - signal_line


def find_local_lows(prices: pd.Series, window: int = LOCAL_LOW_WINDOW) -> pd.Series:
    """Return boolean series where price is lower than surrounding `window` days each side."""
    is_low = pd.Series(False, index=prices.index)
    for i in range(window, len(prices) - window):
        chunk = prices.iloc[i - window : i + window + 1]
        if prices.iloc[i] == chunk.min() and (prices.iloc[i] < chunk.drop(prices.index[i])).all():
            is_low.iloc[i] = True
    return is_low


def find_period_lows(prices: pd.Series, lookback: int) -> pd.Series:
    """Return boolean series where price is at a `lookback`-day low."""
    rolling_min = prices.rolling(lookback, min_periods=lookback).min()
    return prices <= rolling_min


def compute_weekly_rsi(daily_close: pd.Series, period: int = 14) -> pd.Series:
    """Resample to weekly, compute RSI, forward-fill back to daily index."""
    weekly = daily_close.resample("W-FRI").last().dropna()
    wrsi = compute_rsi(weekly, period)
    return wrsi.reindex(daily_close.index, method="ffill")


def download_data() -> dict[str, pd.DataFrame]:
    """Download OHLCV for universe + SPY."""
    tickers = UNIVERSE + ["SPY"]
    data = {}
    raw = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False, group_by="ticker")
    for t in tickers:
        try:
            df = raw[t].dropna()
            if len(df) > 100:
                data[t] = df
        except Exception:
            pass
    # Flatten columns if MultiIndex
    for t in data:
        if isinstance(data[t].columns, pd.MultiIndex):
            data[t].columns = data[t].columns.get_level_values(-1)
    return data


# ── Signal generators ──────────────────────────────────────────────────────
def generate_signals_A(df: pd.DataFrame) -> list[pd.Timestamp]:
    """Variant A: price 20-day low + RSI higher than at prior 20-day low."""
    close = df["Close"].squeeze()
    rsi = compute_rsi(close, RSI_PERIOD)
    is_20d_low = find_period_lows(close, 20)
    signals = []
    prev_low_idx = None
    for i in range(20, len(close)):
        if is_20d_low.iloc[i]:
            if prev_low_idx is not None:
                if close.iloc[i] < close.iloc[prev_low_idx] and rsi.iloc[i] > rsi.iloc[prev_low_idx]:
                    signals.append(close.index[i])
            prev_low_idx = i
    return signals


def generate_signals_B(df: pd.DataFrame) -> list[pd.Timestamp]:
    """Variant B: price 40-day low + RSI higher than at prior 40-day low."""
    close = df["Close"].squeeze()
    rsi = compute_rsi(close, RSI_PERIOD)
    is_40d_low = find_period_lows(close, 40)
    signals = []
    prev_low_idx = None
    for i in range(40, len(close)):
        if is_40d_low.iloc[i]:
            if prev_low_idx is not None:
                if close.iloc[i] < close.iloc[prev_low_idx] and rsi.iloc[i] > rsi.iloc[prev_low_idx]:
                    signals.append(close.index[i])
            prev_low_idx = i
    return signals


def generate_signals_C(df: pd.DataFrame) -> list[pd.Timestamp]:
    """Variant C: same as A + declining volume (selling exhaustion)."""
    close = df["Close"].squeeze()
    volume = df["Volume"].squeeze()
    rsi = compute_rsi(close, RSI_PERIOD)
    is_20d_low = find_period_lows(close, 20)
    signals = []
    prev_low_idx = None
    for i in range(20, len(close)):
        if is_20d_low.iloc[i]:
            if prev_low_idx is not None:
                price_lower = close.iloc[i] < close.iloc[prev_low_idx]
                rsi_higher = rsi.iloc[i] > rsi.iloc[prev_low_idx]
                # Volume declining: avg volume over last 5 days < avg over 5 days around prior low
                vol_now = volume.iloc[max(0, i - 4) : i + 1].mean()
                j = prev_low_idx
                vol_prev = volume.iloc[max(0, j - 2) : j + 3].mean()
                vol_declining = vol_now < vol_prev
                if price_lower and rsi_higher and vol_declining:
                    signals.append(close.index[i])
            prev_low_idx = i
    return signals


def generate_signals_D(df: pd.DataFrame) -> list[pd.Timestamp]:
    """Variant D: RSI divergence + first green day after 2+ red days."""
    close = df["Close"].squeeze()
    rsi = compute_rsi(close, RSI_PERIOD)
    is_20d_low = find_period_lows(close, 20)
    signals = []
    prev_low_idx = None
    for i in range(20, len(close)):
        if is_20d_low.iloc[i]:
            if prev_low_idx is not None:
                price_lower = close.iloc[i] < close.iloc[prev_low_idx]
                rsi_higher = rsi.iloc[i] > rsi.iloc[prev_low_idx]
                if price_lower and rsi_higher:
                    # Look ahead up to 5 days for first green after 2+ red
                    for k in range(i, min(i + 6, len(close))):
                        if k >= 3:
                            red_count = sum(
                                1 for m in range(k - 2, k)
                                if close.iloc[m] < close.iloc[m - 1]
                            )
                            if red_count >= 2 and close.iloc[k] > close.iloc[k - 1]:
                                signals.append(close.index[k])
                                break
            prev_low_idx = i
    return signals


def generate_signals_E(df: pd.DataFrame) -> list[pd.Timestamp]:
    """Variant E: daily RSI divergence + weekly RSI also diverging."""
    close = df["Close"].squeeze()
    rsi_daily = compute_rsi(close, RSI_PERIOD)
    rsi_weekly = compute_weekly_rsi(close, RSI_PERIOD)
    is_20d_low = find_period_lows(close, 20)
    signals = []
    prev_low_idx = None
    for i in range(20, len(close)):
        if is_20d_low.iloc[i]:
            if prev_low_idx is not None:
                price_lower = close.iloc[i] < close.iloc[prev_low_idx]
                daily_rsi_higher = rsi_daily.iloc[i] > rsi_daily.iloc[prev_low_idx]
                weekly_rsi_higher = (
                    not pd.isna(rsi_weekly.iloc[i])
                    and not pd.isna(rsi_weekly.iloc[prev_low_idx])
                    and rsi_weekly.iloc[i] > rsi_weekly.iloc[prev_low_idx]
                )
                if price_lower and daily_rsi_higher and weekly_rsi_higher:
                    signals.append(close.index[i])
            prev_low_idx = i
    return signals


def generate_signals_F(df: pd.DataFrame) -> list[pd.Timestamp]:
    """Variant F: triple divergence — price/RSI/MACD histogram all diverging."""
    close = df["Close"].squeeze()
    rsi = compute_rsi(close, RSI_PERIOD)
    macd_h = compute_macd_hist(close)
    is_20d_low = find_period_lows(close, 20)
    signals = []
    prev_low_idx = None
    for i in range(26, len(close)):  # need 26 for MACD
        if is_20d_low.iloc[i]:
            if prev_low_idx is not None and prev_low_idx >= 26:
                price_lower = close.iloc[i] < close.iloc[prev_low_idx]
                rsi_higher = rsi.iloc[i] > rsi.iloc[prev_low_idx]
                macd_higher = macd_h.iloc[i] > macd_h.iloc[prev_low_idx]
                if price_lower and rsi_higher and macd_higher:
                    signals.append(close.index[i])
            prev_low_idx = i
    return signals


VARIANTS = {
    "A": {"fn": generate_signals_A, "hold": 10, "desc": "Classic 20d divergence"},
    "B": {"fn": generate_signals_B, "hold": 10, "desc": "Stronger 40d divergence"},
    "C": {"fn": generate_signals_C, "hold": 10, "desc": "Divergence + volume decline"},
    "D": {"fn": generate_signals_D, "hold": 10, "desc": "Divergence + first green after reds"},
    "E": {"fn": generate_signals_E, "hold": 15, "desc": "Multi-timeframe divergence"},
    "F": {"fn": generate_signals_F, "hold": 10, "desc": "Triple divergence (RSI+MACD)"},
}


# ── Backtest engine ────────────────────────────────────────────────────────
def backtest(
    all_signals: list[tuple[str, pd.Timestamp]],
    data: dict[str, pd.DataFrame],
    hold_days: int,
    capital: float = CAPITAL,
    max_per_trade: float = MAX_PER_TRADE,
    max_concurrent: int = MAX_CONCURRENT,
    slippage_bps: float = SLIPPAGE_BPS,
) -> dict:
    """Run backtest with position limits and slippage."""
    # Sort signals by date
    all_signals.sort(key=lambda x: x[1])
    trades = []
    open_positions = []  # list of (ticker, entry_date, exit_date, entry_price)

    for ticker, entry_date in all_signals:
        # Check concurrent limit
        open_positions = [p for p in open_positions if p[2] > entry_date]
        if len(open_positions) >= max_concurrent:
            continue

        df = data[ticker]
        close = df["Close"].squeeze()
        if entry_date not in close.index:
            continue

        idx = close.index.get_loc(entry_date)
        exit_idx = min(idx + hold_days, len(close) - 1)
        if exit_idx <= idx:
            continue

        entry_price = float(close.iloc[idx])
        exit_price = float(close.iloc[exit_idx])
        exit_date = close.index[exit_idx]

        # Position sizing
        shares = int(max_per_trade // entry_price)
        if shares < 1:
            continue

        # Apply slippage
        entry_cost = entry_price * (1 + slippage_bps / 10000)
        exit_proceeds = exit_price * (1 - slippage_bps / 10000)

        pnl = (exit_proceeds - entry_cost) * shares
        ret = (exit_proceeds - entry_cost) / entry_cost

        trades.append({
            "ticker": ticker,
            "entry_date": str(entry_date.date()),
            "exit_date": str(exit_date.date()),
            "entry_price": round(entry_cost, 2),
            "exit_price": round(exit_proceeds, 2),
            "shares": shares,
            "pnl": round(pnl, 2),
            "return": round(ret, 4),
        })

        open_positions.append((ticker, entry_date, exit_date, entry_price))

    return _compute_stats(trades, capital)


def _compute_stats(trades: list[dict], capital: float) -> dict:
    if not trades:
        return {"n_trades": 0, "total_pnl": 0, "sharpe": 0, "sortino": 0,
                "profit_factor": 0, "win_rate": 0, "max_drawdown": 0,
                "trades": [], "daily_returns": []}

    returns = np.array([t["return"] for t in trades])
    pnls = np.array([t["pnl"] for t in trades])
    total_pnl = float(pnls.sum())
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    # Sharpe (annualized, assume ~25 trades/year avg)
    avg_ret = returns.mean()
    std_ret = returns.std() if len(returns) > 1 else 1e-9
    trades_per_year = max(len(trades) / 4.5, 1)  # ~4.5 years of data
    sharpe = (avg_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0

    # Sortino
    downside = returns[returns < 0]
    downside_std = downside.std() if len(downside) > 1 else 1e-9
    sortino = (avg_ret / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0

    # Profit factor
    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 1e-9
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    # Win rate
    win_rate = float(len(wins) / len(trades))

    # Max drawdown (equity curve)
    equity = capital + np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd = float(drawdown.min())

    return {
        "n_trades": len(trades),
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_pnl / capital * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(profit_factor, 3),
        "win_rate": round(win_rate, 3),
        "max_drawdown": round(max_dd, 4),
        "avg_return": round(float(avg_ret), 4),
        "avg_pnl": round(float(pnls.mean()), 2),
        "trades": trades,
        "daily_returns": returns.tolist(),
    }


# ── Validation gates ───────────────────────────────────────────────────────
def permutation_test(returns: list[float], observed_sharpe: float, n_perms: int = N_PERMUTATIONS) -> float:
    """Shuffle returns, compute Sharpe each time, return p-value."""
    if len(returns) < 5:
        return 1.0
    rng = np.random.default_rng(42)
    arr = np.array(returns)
    count_better = 0
    for _ in range(n_perms):
        shuffled = rng.permutation(arr)
        s = shuffled.mean() / (shuffled.std() + 1e-9)
        if s >= observed_sharpe / max(np.sqrt(len(arr) / 4.5), 1):
            count_better += 1
    return count_better / n_perms


def regime_test(
    trades: list[dict],
    spy_data: pd.DataFrame,
) -> float:
    """Compute regime gap: |Sharpe_bull - Sharpe_bear| / max."""
    spy_close = spy_data["Close"].squeeze()
    sma200 = spy_close.rolling(200).mean()

    bull_rets, bear_rets = [], []
    for t in trades:
        ed = pd.Timestamp(t["entry_date"])
        if ed in spy_close.index and ed in sma200.index:
            if spy_close.loc[ed] > sma200.loc[ed]:
                bull_rets.append(t["return"])
            else:
                bear_rets.append(t["return"])

    if not bull_rets or not bear_rets:
        return 1.0  # can't assess, fail

    s_bull = np.mean(bull_rets) / (np.std(bull_rets) + 1e-9)
    s_bear = np.mean(bear_rets) / (np.std(bear_rets) + 1e-9)
    denom = max(abs(s_bull), abs(s_bear), 1e-9)
    return abs(s_bull - s_bear) / denom


def validate(stats: dict, spy_data: pd.DataFrame) -> dict:
    """Run 5-gate validation."""
    gates = {}

    # Gate 1: Sharpe > 0.5
    gates["sharpe_gt_0.5"] = {
        "pass": stats["sharpe"] > 0.5,
        "value": stats["sharpe"],
        "threshold": 0.5,
    }

    # Gate 2: Permutation test p < 0.05
    if stats["n_trades"] >= 5:
        p_val = permutation_test(stats["daily_returns"], stats["sharpe"])
    else:
        p_val = 1.0
    gates["permutation_p_lt_0.05"] = {
        "pass": p_val < 0.05,
        "value": round(p_val, 4),
        "threshold": 0.05,
    }

    # Gate 3: Regime gap < 0.5
    if stats["n_trades"] >= 5:
        rgap = regime_test(stats["trades"], spy_data)
    else:
        rgap = 1.0
    gates["regime_gap_lt_0.5"] = {
        "pass": rgap < 0.5,
        "value": round(rgap, 4),
        "threshold": 0.5,
    }

    # Gate 4: Max drawdown > -50%
    gates["max_dd_gt_neg50pct"] = {
        "pass": stats["max_drawdown"] > -0.50,
        "value": stats["max_drawdown"],
        "threshold": -0.50,
    }

    # Gate 5: At least 20 trades
    gates["min_20_trades"] = {
        "pass": stats["n_trades"] >= 20,
        "value": stats["n_trades"],
        "threshold": 20,
    }

    passed = sum(1 for g in gates.values() if g["pass"])
    return {"gates_passed": f"{passed}/5", "all_passed": passed == 5, "gates": gates}


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("Downloading data...")
    data = download_data()
    print(f"  Got {len(data)} tickers, {len(data.get('SPY', []))} SPY bars")

    spy_data = data.get("SPY")
    if spy_data is None:
        print("ERROR: could not download SPY data")
        return

    results = {}

    for var_name, var_cfg in VARIANTS.items():
        print(f"\n{'='*60}")
        print(f"Variant {var_name}: {var_cfg['desc']}")
        print(f"{'='*60}")

        # Generate signals across universe
        all_signals = []
        for ticker in UNIVERSE:
            if ticker not in data:
                continue
            sigs = var_cfg["fn"](data[ticker])
            for s in sigs:
                all_signals.append((ticker, s))

        print(f"  Raw signals: {len(all_signals)}")

        # Backtest
        stats = backtest(all_signals, data, hold_days=var_cfg["hold"])
        print(f"  Trades taken: {stats['n_trades']}")
        print(f"  Total PnL:    ${stats['total_pnl']:.2f} ({stats.get('total_return_pct', 0):.1f}%)")
        print(f"  Sharpe:       {stats['sharpe']:.3f}")
        print(f"  Sortino:      {stats['sortino']:.3f}")
        print(f"  PF:           {stats['profit_factor']:.3f}")
        print(f"  Win Rate:     {stats['win_rate']:.1%}")
        print(f"  Max DD:       {stats['max_drawdown']:.2%}")

        # Validate
        validation = validate(stats, spy_data)
        print(f"\n  Validation: {validation['gates_passed']} gates passed")
        for gname, gval in validation["gates"].items():
            status = "PASS" if gval["pass"] else "FAIL"
            print(f"    [{status}] {gname}: {gval['value']} (threshold: {gval['threshold']})")

        # Store (without individual trades for JSON size)
        results[f"variant_{var_name}"] = {
            "description": var_cfg["desc"],
            "hold_days": var_cfg["hold"],
            "n_signals_raw": len(all_signals),
            "n_trades": stats["n_trades"],
            "total_pnl": stats["total_pnl"],
            "total_return_pct": stats.get("total_return_pct", 0),
            "sharpe": stats["sharpe"],
            "sortino": stats["sortino"],
            "profit_factor": stats["profit_factor"],
            "win_rate": stats["win_rate"],
            "max_drawdown": stats["max_drawdown"],
            "avg_return": stats["avg_return"],
            "avg_pnl": stats["avg_pnl"],
            "validation": validation,
            "sample_trades": stats["trades"][:10],  # first 10 for inspection
        }

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Var':<4} {'Trades':>6} {'PnL':>10} {'Ret%':>7} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR':>6} {'MaxDD':>8} {'Gates':>6}")
    print("-" * 70)

    best_var = None
    best_sharpe = -999

    for var_name in ["A", "B", "C", "D", "E", "F"]:
        key = f"variant_{var_name}"
        if key not in results:
            continue
        r = results[key]
        v = r["validation"]
        marker = " *" if v["all_passed"] else ""
        print(
            f"  {var_name:<3} {r['n_trades']:>6} {r['total_pnl']:>10.2f} {r['total_return_pct']:>6.1f}% "
            f"{r['sharpe']:>7.3f} {r['sortino']:>8.3f} {r['profit_factor']:>6.2f} "
            f"{r['win_rate']:>5.1%} {r['max_drawdown']:>7.2%} {v['gates_passed']:>6}{marker}"
        )
        if r["sharpe"] > best_sharpe:
            best_sharpe = r["sharpe"]
            best_var = var_name

    passed_all = [k for k, v in results.items() if v["validation"]["all_passed"]]
    print(f"\nBest Sharpe: Variant {best_var} ({best_sharpe:.3f})")
    if passed_all:
        print(f"Passed ALL 5 gates: {', '.join(passed_all)}")
    else:
        print("No variant passed all 5 gates.")

    # ── Save ────────────────────────────────────────────────────────────
    output = {
        "strategy": "RSI Divergence on Quality Stocks",
        "run_date": datetime.now().isoformat(),
        "period": f"{START} to {END}",
        "universe": UNIVERSE,
        "capital": CAPITAL,
        "max_per_trade": MAX_PER_TRADE,
        "max_concurrent": MAX_CONCURRENT,
        "slippage_bps": SLIPPAGE_BPS,
        "variants": results,
        "best_variant": best_var,
        "any_passed_all_gates": len(passed_all) > 0,
    }

    out_path = "/home/jupiter/Lvl3Quant/data/rsi_divergence_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
