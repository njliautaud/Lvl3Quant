#!/usr/bin/env python3
"""
Volatility Clustering Backtest
------------------------------
Exploits vol clustering: high-vol days follow high-vol days, low-vol follows low-vol.
After vol spikes on quality stocks, buy the dip (vol mean-reverts -> price recovers).
After prolonged low vol, prepare for breakout.

Universe: 20 quality large-cap stocks
OOT: Jan 2022 - Jul 2026
Starting capital: $645
"""

import json
import datetime as dt
import warnings
import sys
import os

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    print("Installing yfinance...")
    os.system(f"{sys.executable} -m pip install yfinance -q")
    import yfinance as yf

# ─── CONFIG ───────────────────────────────────────────────────────────────────

UNIVERSE = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]
BENCHMARK = "SPY"

START = "2021-01-01"       # extra lookback for indicators
OOT_START = "2022-01-01"
OOT_END = "2026-07-31"

INITIAL_CAPITAL = 645.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
SLIPPAGE_PCT = 0.0002     # 0.02% each way

OUTPUT_PATH = "/home/jupiter/Lvl3Quant/data/volatility_clustering_results.json"


# ─── DATA ─────────────────────────────────────────────────────────────────────

def download_data():
    """Download OHLCV for universe + SPY."""
    tickers = UNIVERSE + [BENCHMARK]
    print(f"Downloading {len(tickers)} tickers...")
    data = yf.download(tickers, start=START, end=OOT_END, auto_adjust=True, progress=False)
    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    # Drop any ticker with insufficient data
    min_rows = 252
    valid = close.columns[close.count() >= min_rows]
    close = close[valid]
    high = high[valid]
    low = low[valid]
    print(f"  Got {len(close)} rows, {len(close.columns)} tickers with sufficient data")
    return close, high, low


# ─── INDICATORS ───────────────────────────────────────────────────────────────

def realized_vol(close, window):
    """Annualized realized vol from log returns."""
    lr = np.log(close / close.shift(1))
    return lr.rolling(window).std() * np.sqrt(252)

def rsi(close, period=14):
    """Standard RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)

def bollinger_band_width(close, window=20, num_std=2):
    """Bollinger Band width = (upper - lower) / middle."""
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return (upper - lower) / sma

def sma(series, window):
    return series.rolling(window).mean()


# ─── SIGNAL GENERATORS ───────────────────────────────────────────────────────

def generate_signals_A(close):
    """Vol spike + reversal confirmation. 5d vol > 2x 60d vol AND close > prev close. Hold 5d."""
    vol5 = realized_vol(close, 5)
    vol60 = realized_vol(close, 60)
    ratio = vol5 / vol60
    reversal = close > close.shift(1)
    signals = (ratio > 2.0) & reversal
    return signals, 5

def generate_signals_B(close):
    """Vol spike + oversold. 5d vol > 2.5x 60d vol AND RSI < 35. Hold 10d."""
    vol5 = realized_vol(close, 5)
    vol60 = realized_vol(close, 60)
    ratio = vol5 / vol60
    r = rsi(close, 14)
    signals = (ratio > 2.5) & (r < 35)
    return signals, 10

def generate_signals_C(close):
    """Vol compression + near highs. 10d vol < 0.5x 60d vol AND within 3% of 20d high. Hold 15d."""
    vol10 = realized_vol(close, 10)
    vol60 = realized_vol(close, 60)
    ratio = vol10 / vol60
    high20 = close.rolling(20).max()
    near_high = close >= high20 * 0.97
    signals = (ratio < 0.5) & near_high
    return signals, 15

def generate_signals_D(close):
    """Large down day. |return| > 2 std AND return < 0. Hold 5d."""
    ret = close.pct_change()
    roll_std = ret.rolling(60).std()
    roll_mean = ret.rolling(60).mean()
    z = (ret - roll_mean) / roll_std
    signals = (z < -2.0)  # large down day
    return signals, 5

def generate_signals_E(close):
    """BB width contraction + lower band touch. Hold 10d."""
    bbw = bollinger_band_width(close, 20, 2)
    # Rolling percentile of BBW over 252 days
    bbw_pctile = bbw.rolling(252).apply(lambda x: (x.iloc[-1] <= x).sum() / len(x) * 100 if len(x) == 252 else np.nan, raw=False)
    # Lower band
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    lower_band = sma20 - 2 * std20
    touches_lower = close <= lower_band * 1.005  # within 0.5% of lower band
    signals = (bbw_pctile <= 10) & touches_lower
    return signals, 10

def generate_signals_F(close):
    """Combine B + D: vol spike via either measure + oversold. Hold 10d."""
    # B conditions
    vol5 = realized_vol(close, 5)
    vol60 = realized_vol(close, 60)
    ratio = vol5 / vol60
    r = rsi(close, 14)
    b_signal = (ratio > 2.5) & (r < 35)

    # D conditions
    ret = close.pct_change()
    roll_std = ret.rolling(60).std()
    roll_mean = ret.rolling(60).mean()
    z = (ret - roll_mean) / roll_std
    d_signal = (z < -2.0)

    # Either B or D
    signals = b_signal | d_signal
    return signals, 10


# ─── BACKTESTER ───────────────────────────────────────────────────────────────

def backtest_variant(close, signals, hold_days, variant_name):
    """
    Run backtest with position sizing, max concurrent, slippage.
    Returns dict of metrics + trade list.
    """
    # Filter to OOT period
    mask = (close.index >= pd.Timestamp(OOT_START)) & (close.index <= pd.Timestamp(OOT_END))
    oot_dates = close.index[mask]

    tickers = [t for t in close.columns if t != BENCHMARK]

    trades = []
    capital = INITIAL_CAPITAL
    equity_curve = []
    open_positions = []  # list of dicts: {ticker, entry_date, entry_price, shares, exit_date_idx}

    date_list = list(oot_dates)

    for i, date in enumerate(date_list):
        # Close expired positions
        newly_closed = []
        still_open = []
        for pos in open_positions:
            if i >= pos["exit_idx"]:
                # Close position
                exit_date = date_list[min(pos["exit_idx"], len(date_list) - 1)]
                if exit_date in close.index and pos["ticker"] in close.columns:
                    exit_price_raw = close.loc[exit_date, pos["ticker"]]
                    if pd.notna(exit_price_raw):
                        exit_price = exit_price_raw * (1 - SLIPPAGE_PCT)  # selling
                        pnl = (exit_price - pos["entry_price"]) * pos["shares"]
                        capital += pos["shares"] * exit_price
                        trades.append({
                            "ticker": pos["ticker"],
                            "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                            "exit_date": exit_date.strftime("%Y-%m-%d"),
                            "entry_price": round(pos["entry_price"], 2),
                            "exit_price": round(exit_price, 2),
                            "shares": pos["shares"],
                            "pnl": round(pnl, 2),
                            "return_pct": round((exit_price / pos["entry_price"] - 1) * 100, 2),
                        })
                        newly_closed.append(pos)
                    else:
                        still_open.append(pos)
                else:
                    still_open.append(pos)
            else:
                still_open.append(pos)
        open_positions = still_open

        # Open new positions if signals fire
        if len(open_positions) < MAX_CONCURRENT:
            for ticker in tickers:
                if len(open_positions) >= MAX_CONCURRENT:
                    break
                if ticker not in signals.columns or ticker not in close.columns:
                    continue
                if date not in signals.index:
                    continue
                if pd.notna(signals.loc[date, ticker]) and signals.loc[date, ticker]:
                    # Check not already in this ticker
                    if any(p["ticker"] == ticker for p in open_positions):
                        continue
                    entry_price_raw = close.loc[date, ticker]
                    if pd.isna(entry_price_raw) or entry_price_raw <= 0:
                        continue
                    entry_price = entry_price_raw * (1 + SLIPPAGE_PCT)  # buying
                    position_size = min(MAX_PER_TRADE, capital * 0.95)
                    if position_size < 10:
                        continue
                    shares = position_size / entry_price
                    if shares * entry_price > capital:
                        continue
                    capital -= shares * entry_price
                    exit_idx = i + hold_days
                    open_positions.append({
                        "ticker": ticker,
                        "entry_date": date,
                        "entry_price": entry_price,
                        "shares": shares,
                        "exit_idx": exit_idx,
                    })

        # Mark-to-market
        mtm = capital
        for pos in open_positions:
            if pos["ticker"] in close.columns and date in close.index:
                cur = close.loc[date, pos["ticker"]]
                if pd.notna(cur):
                    mtm += pos["shares"] * cur
        equity_curve.append({"date": date.strftime("%Y-%m-%d"), "equity": round(mtm, 2)})

    # Force close remaining
    last_date = date_list[-1]
    for pos in open_positions:
        if pos["ticker"] in close.columns and last_date in close.index:
            exit_price_raw = close.loc[last_date, pos["ticker"]]
            if pd.notna(exit_price_raw):
                exit_price = exit_price_raw * (1 - SLIPPAGE_PCT)
                pnl = (exit_price - pos["entry_price"]) * pos["shares"]
                capital += pos["shares"] * exit_price
                trades.append({
                    "ticker": pos["ticker"],
                    "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                    "exit_date": last_date.strftime("%Y-%m-%d"),
                    "entry_price": round(pos["entry_price"], 2),
                    "exit_price": round(exit_price, 2),
                    "shares": pos["shares"],
                    "pnl": round(pnl, 2),
                    "return_pct": round((exit_price / pos["entry_price"] - 1) * 100, 2),
                })

    return trades, equity_curve


# ─── METRICS ──────────────────────────────────────────────────────────────────

def compute_metrics(trades, equity_curve):
    """Compute performance metrics from trade list and equity curve."""
    if not trades:
        return {"n_trades": 0, "sharpe": 0, "sortino": 0, "pf": 0, "wr": 0, "maxdd": 0, "total_return": 0}

    returns = [t["return_pct"] / 100 for t in trades]
    pnls = [t["pnl"] for t in trades]

    n = len(trades)
    wr = sum(1 for r in returns if r > 0) / n * 100

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1) if n > 1 else 1e-9

    # Annualized Sharpe (assume avg ~50 trades/yr for daily strategies)
    trades_per_year = max(n / 4.5, 1)  # 4.5 year OOT
    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0

    # Sortino
    downside = [r for r in returns if r < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = (mean_ret / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0

    # Max drawdown from equity curve
    equities = [e["equity"] for e in equity_curve]
    peak = equities[0]
    max_dd = 0
    for eq in equities:
        peak = max(peak, eq)
        dd = (eq - peak) / peak
        max_dd = min(max_dd, dd)

    total_return = (equities[-1] / equities[0] - 1) * 100 if equities else 0
    total_pnl = sum(pnls)

    return {
        "n_trades": n,
        "win_rate": round(wr, 1),
        "profit_factor": round(pf, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_dd_pct": round(max_dd * 100, 1),
        "total_return_pct": round(total_return, 1),
        "total_pnl": round(total_pnl, 2),
        "final_equity": round(equities[-1], 2) if equities else INITIAL_CAPITAL,
        "avg_return_pct": round(mean_ret * 100, 2),
        "avg_hold_days": round(np.mean([(pd.Timestamp(t["exit_date"]) - pd.Timestamp(t["entry_date"])).days for t in trades]), 1),
    }


# ─── PERMUTATION TEST ────────────────────────────────────────────────────────

def permutation_test(close, signals, hold_days, actual_pnl, n_perms=1000):
    """Shuffle entry dates to get null distribution of total PnL."""
    # Get all signal dates/tickers
    mask = (signals.index >= pd.Timestamp(OOT_START)) & (signals.index <= pd.Timestamp(OOT_END))
    sig_oot = signals[mask]

    entries = []
    for date in sig_oot.index:
        for ticker in sig_oot.columns:
            if ticker == BENCHMARK:
                continue
            if pd.notna(sig_oot.loc[date, ticker]) and sig_oot.loc[date, ticker]:
                entries.append((date, ticker))

    if len(entries) < 5:
        return 1.0  # Not enough entries

    # Get all valid trade dates
    oot_dates = list(close.index[(close.index >= pd.Timestamp(OOT_START)) & (close.index <= pd.Timestamp(OOT_END))])

    null_pnls = []
    rng = np.random.RandomState(42)

    for _ in range(n_perms):
        pnl = 0
        n_selected = min(len(entries), 50)  # cap for speed
        sample_idx = rng.choice(len(entries), size=n_selected, replace=False) if len(entries) > n_selected else range(len(entries))

        for idx in sample_idx:
            _, ticker = entries[idx]
            # Random entry date
            rand_idx = rng.randint(0, max(1, len(oot_dates) - hold_days - 1))
            entry_date = oot_dates[rand_idx]
            exit_idx = min(rand_idx + hold_days, len(oot_dates) - 1)
            exit_date = oot_dates[exit_idx]

            if ticker in close.columns:
                ep = close.loc[entry_date, ticker] if entry_date in close.index else np.nan
                xp = close.loc[exit_date, ticker] if exit_date in close.index else np.nan
                if pd.notna(ep) and pd.notna(xp) and ep > 0:
                    ret = (xp / ep - 1)
                    trade_size = min(MAX_PER_TRADE, INITIAL_CAPITAL * 0.3)
                    pnl += ret * trade_size
        null_pnls.append(pnl)

    # p-value: fraction of null >= actual
    p = np.mean([n >= actual_pnl for n in null_pnls])
    return round(p, 4)


# ─── REGIME ANALYSIS ─────────────────────────────────────────────────────────

def regime_analysis(trades, close):
    """Split trades into bull/bear based on SPY > 200-SMA."""
    if BENCHMARK not in close.columns:
        return 0.0, {}, {}

    spy = close[BENCHMARK]
    spy_sma200 = spy.rolling(200).mean()

    bull_trades = []
    bear_trades = []

    for t in trades:
        entry = pd.Timestamp(t["entry_date"])
        if entry in spy.index:
            is_bull = spy.loc[entry] > spy_sma200.loc[entry] if pd.notna(spy_sma200.loc[entry]) else True
            if is_bull:
                bull_trades.append(t)
            else:
                bear_trades.append(t)

    def regime_sharpe(trade_list):
        if len(trade_list) < 2:
            return 0.0
        rets = [t["return_pct"] / 100 for t in trade_list]
        m = np.mean(rets)
        s = np.std(rets, ddof=1)
        return m / s * np.sqrt(len(trade_list)) if s > 0 else 0.0

    bull_sharpe = regime_sharpe(bull_trades)
    bear_sharpe = regime_sharpe(bear_trades)

    max_s = max(abs(bull_sharpe), abs(bear_sharpe))
    gap = abs(bull_sharpe - bear_sharpe) / max_s if max_s > 0 else 0.0

    bull_metrics = {"n": len(bull_trades), "sharpe": round(bull_sharpe, 2),
                    "wr": round(sum(1 for t in bull_trades if t["return_pct"] > 0) / max(len(bull_trades), 1) * 100, 1)}
    bear_metrics = {"n": len(bear_trades), "sharpe": round(bear_sharpe, 2),
                    "wr": round(sum(1 for t in bear_trades if t["return_pct"] > 0) / max(len(bear_trades), 1) * 100, 1)}

    return round(gap, 3), bull_metrics, bear_metrics


# ─── 5-GATE VALIDATION ───────────────────────────────────────────────────────

def validate_5gate(metrics, perm_p, regime_gap):
    """Apply 5-gate validation."""
    gates = {
        "G1_sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "G2_perm_p_lt_0.05": perm_p < 0.05,
        "G3_regime_gap_lt_0.5": regime_gap < 0.5,
        "G4_maxdd_gt_neg50": metrics["max_dd_pct"] > -50,
        "G5_min_20_trades": metrics["n_trades"] >= 20,
    }
    gates["PASS_ALL"] = all(gates.values())
    return gates


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    close, high, low = download_data()

    variants = {
        "A": ("Vol spike + reversal (5d hold)", generate_signals_A),
        "B": ("Vol spike + oversold (10d hold)", generate_signals_B),
        "C": ("Vol compression + breakout (15d hold)", generate_signals_C),
        "D": ("Large down day mean-reversion (5d hold)", generate_signals_D),
        "E": ("BB contraction + lower touch (10d hold)", generate_signals_E),
        "F": ("Combined B+D vol spike (10d hold)", generate_signals_F),
    }

    results = {}

    for name, (desc, gen_func) in variants.items():
        print(f"\n{'='*70}")
        print(f"Variant {name}: {desc}")
        print(f"{'='*70}")

        signals, hold_days = gen_func(close)
        trades, equity_curve = backtest_variant(close, signals, hold_days, name)
        metrics = compute_metrics(trades, equity_curve)

        print(f"  Trades: {metrics['n_trades']}")
        print(f"  Win Rate: {metrics['win_rate']}%")
        print(f"  Sharpe: {metrics['sharpe']}")
        print(f"  Sortino: {metrics['sortino']}")
        print(f"  PF: {metrics['profit_factor']}")
        print(f"  MaxDD: {metrics['max_dd_pct']}%")
        print(f"  Total Return: {metrics['total_return_pct']}%")
        print(f"  Final Equity: ${metrics['final_equity']}")

        # Permutation test
        total_pnl = metrics.get("total_pnl", 0)
        perm_p = permutation_test(close, signals, hold_days, total_pnl, n_perms=1000)
        print(f"  Perm test p-value: {perm_p}")

        # Regime analysis
        regime_gap, bull, bear = regime_analysis(trades, close)
        print(f"  Regime gap: {regime_gap}")
        print(f"    Bull: {bull}")
        print(f"    Bear: {bear}")

        # 5-gate
        gates = validate_5gate(metrics, perm_p, regime_gap)
        print(f"  5-Gate: {'PASS' if gates['PASS_ALL'] else 'FAIL'}")
        for g, v in gates.items():
            if g != "PASS_ALL":
                print(f"    {g}: {'PASS' if v else 'FAIL'}")

        results[f"Variant_{name}"] = {
            "description": desc,
            "metrics": metrics,
            "permutation_p_value": perm_p,
            "regime_gap": regime_gap,
            "bull_regime": bull,
            "bear_regime": bear,
            "gates": gates,
            "sample_trades": trades[:10] if trades else [],
        }

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Variant':<12} {'Trades':>6} {'WR%':>6} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'MaxDD%':>7} {'Return%':>8} {'5-Gate':>7}")
    print("-" * 75)
    for name in ["A", "B", "C", "D", "E", "F"]:
        r = results[f"Variant_{name}"]
        m = r["metrics"]
        g = "PASS" if r["gates"]["PASS_ALL"] else "FAIL"
        print(f"  {name:<10} {m['n_trades']:>6} {m['win_rate']:>6.1f} {m['sharpe']:>7.2f} {m['sortino']:>8.2f} {m['profit_factor']:>6.2f} {m['max_dd_pct']:>7.1f} {m['total_return_pct']:>8.1f} {g:>7}")

    # Save
    output = {
        "strategy": "Volatility Clustering",
        "universe": UNIVERSE,
        "oot_period": f"{OOT_START} to {OOT_END}",
        "initial_capital": INITIAL_CAPITAL,
        "slippage_pct": SLIPPAGE_PCT,
        "max_per_trade": MAX_PER_TRADE,
        "max_concurrent": MAX_CONCURRENT,
        "run_timestamp": dt.datetime.now().isoformat(),
        "variants": results,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
