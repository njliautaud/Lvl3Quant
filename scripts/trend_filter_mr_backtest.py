#!/usr/bin/env python3
"""
Trend-Filtered Mean Reversion on Quality Stocks
Tests 6 trend/regime filters (A-F) layered on top of a base quality MR signal.

Base signal: Quality stock >5% below 20-day high AND RSI(14) < 35.
Each variant adds an additional filter. Hold 10 days for all.

Capital $669, max $200/trade, max 3 concurrent, slippage 2bps.
Period: 2022-01-01 to 2026-07-31.
"""

import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Universe & Config ──────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]

SECTOR_ETF_MAP = {
    "AAPL": "XLK", "MSFT": "XLK", "AVGO": "XLK", "GOOGL": "XLK", "META": "XLK", "AMZN": "XLK",
    "JPM": "XLF", "V": "XLF", "MA": "XLF",
    "UNH": "XLV", "LLY": "XLV", "ABBV": "XLV", "MRK": "XLV", "JNJ": "XLV",
    "PG": "XLP", "KO": "XLP", "PEP": "XLP", "WMT": "XLP", "COST": "XLP",
    "HD": "XLY",
}
SECTOR_ETFS = list(set(SECTOR_ETF_MAP.values()))

CAPITAL = 669.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
HOLD_DAYS = 10
SLIPPAGE_BPS = 2
START = "2021-06-01"   # extra lookback for 200-SMA
END = "2026-07-31"
TRADE_START = "2022-01-01"  # actual trading starts here


# ── Indicator helpers ──────────────────────────────────────────────
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / period, min_periods=period).mean()


# ── Data download ──────────────────────────────────────────────────
def download_data():
    tickers = UNIVERSE + SECTOR_ETFS + ["SPY"]
    tickers = list(set(tickers))
    print(f"Downloading {len(tickers)} tickers...")
    raw = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)
    # yfinance returns MultiIndex columns (Price, Ticker)
    close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw["Adj Close"]
    high = raw["High"]
    low = raw["Low"]
    return close, high, low


# ── Pre-compute indicators ────────────────────────────────────────
def compute_indicators(close, high, low):
    indicators = {}
    for sym in UNIVERSE:
        if sym not in close.columns:
            continue
        c = close[sym].dropna()
        h = high[sym].dropna()
        lo = low[sym].dropna()
        # Align
        idx = c.index.intersection(h.index).intersection(lo.index)
        c, h, lo = c.loc[idx], h.loc[idx], lo.loc[idx]

        indicators[sym] = {
            "close": c,
            "high20": c.rolling(20).max(),
            "rsi14": rsi(c, 14),
            "sma200": c.rolling(200).mean(),
            "sma50": c.rolling(50).mean(),
            "sma20": c.rolling(20).mean(),
            "sma100": c.rolling(100).mean(),
            "adx14": adx(h, lo, c, 14),
        }
    return indicators


# ── Base signal ────────────────────────────────────────────────────
def base_signal(indicators, sym, date):
    """Stock >5% below 20-day high AND RSI(14) < 35."""
    ind = indicators.get(sym)
    if ind is None or date not in ind["close"].index:
        return False
    price = ind["close"].loc[date]
    h20 = ind["high20"].loc[date]
    r14 = ind["rsi14"].loc[date]
    if pd.isna(price) or pd.isna(h20) or pd.isna(r14):
        return False
    return (price < h20 * 0.95) and (r14 < 35)


# ── Variant filters ───────────────────────────────────────────────
def filter_a(indicators, sym, date, close, **_):
    """200-SMA: price above 200-day SMA."""
    ind = indicators[sym]
    if date not in ind["close"].index:
        return False
    return ind["close"].loc[date] > ind["sma200"].loc[date]


def filter_b(indicators, sym, date, **_):
    """50-SMA slope positive (SMA today > SMA 10 days ago)."""
    ind = indicators[sym]
    sma50 = ind["sma50"]
    if date not in sma50.index:
        return False
    loc = sma50.index.get_loc(date)
    if loc < 10:
        return False
    return sma50.iloc[loc] > sma50.iloc[loc - 10]


def filter_c(indicators, sym, date, **_):
    """ADX(14) < 25 (range-bound)."""
    ind = indicators[sym]
    if date not in ind["adx14"].index:
        return False
    val = ind["adx14"].loc[date]
    return (not pd.isna(val)) and val < 25


def filter_d(indicators, sym, date, close, **_):
    """Sector ETF outperforming SPY over 20 days."""
    etf = SECTOR_ETF_MAP.get(sym)
    if etf is None or etf not in close.columns or "SPY" not in close.columns:
        return False
    etf_c = close[etf]
    spy_c = close["SPY"]
    if date not in etf_c.index or date not in spy_c.index:
        return False
    loc_etf = etf_c.index.get_loc(date)
    loc_spy = spy_c.index.get_loc(date)
    if loc_etf < 20 or loc_spy < 20:
        return False
    etf_ret = etf_c.iloc[loc_etf] / etf_c.iloc[loc_etf - 20] - 1
    spy_ret = spy_c.iloc[loc_spy] / spy_c.iloc[loc_spy - 20] - 1
    return etf_ret > spy_ret


def filter_e(indicators, sym, date, **_):
    """Breadth: >60% of 20 quality stocks above their 50-SMA."""
    count_above = 0
    count_total = 0
    for s in UNIVERSE:
        ind = indicators.get(s)
        if ind is None or date not in ind["close"].index:
            continue
        c_val = ind["close"].loc[date]
        sma_val = ind["sma50"].loc[date]
        if pd.isna(c_val) or pd.isna(sma_val):
            continue
        count_total += 1
        if c_val > sma_val:
            count_above += 1
    if count_total == 0:
        return False
    return (count_above / count_total) > 0.60


def filter_f(indicators, sym, date, **_):
    """Dual TF: price below 20-SMA but above 100-SMA."""
    ind = indicators[sym]
    if date not in ind["close"].index:
        return False
    price = ind["close"].loc[date]
    s20 = ind["sma20"].loc[date]
    s100 = ind["sma100"].loc[date]
    if pd.isna(price) or pd.isna(s20) or pd.isna(s100):
        return False
    return (price < s20) and (price > s100)


VARIANTS = {
    "A_200SMA": filter_a,
    "B_50SMA_Slope": filter_b,
    "C_ADX_Low": filter_c,
    "D_Sector_Outperf": filter_d,
    "E_Breadth": filter_e,
    "F_Dual_TF": filter_f,
}


# ── Backtest engine ───────────────────────────────────────────────
def run_backtest(variant_name, variant_filter, indicators, close, spy_close):
    dates = close.index
    trade_dates = dates[dates >= TRADE_START]

    positions = []  # list of active: {sym, entry_date, entry_price, shares, exit_idx}
    trades = []     # completed trades
    equity = CAPITAL
    equity_curve = []

    for i, date in enumerate(trade_dates):
        # Close expired positions
        newly_closed = []
        still_open = []
        for pos in positions:
            bars_held = len(trade_dates[trade_dates > pos["entry_date"]])
            entry_idx = list(trade_dates).index(pos["entry_date"])
            bars_since = i - entry_idx
            if bars_since >= HOLD_DAYS:
                # Exit
                if date in close.index and pos["sym"] in close.columns:
                    exit_price = close.loc[date, pos["sym"]]
                    if pd.isna(exit_price):
                        still_open.append(pos)
                        continue
                    slip = exit_price * SLIPPAGE_BPS / 10000
                    exit_price_adj = exit_price - slip  # selling
                    pnl = (exit_price_adj - pos["entry_price"]) * pos["shares"]
                    ret = exit_price_adj / pos["entry_price"] - 1
                    equity += pos["shares"] * exit_price_adj
                    trades.append({
                        "sym": pos["sym"],
                        "entry_date": str(pos["entry_date"].date()),
                        "exit_date": str(date.date()),
                        "entry_price": round(pos["entry_price"], 2),
                        "exit_price": round(exit_price_adj, 2),
                        "shares": pos["shares"],
                        "pnl": round(pnl, 2),
                        "return": round(ret, 4),
                    })
                    newly_closed.append(pos)
                else:
                    still_open.append(pos)
            else:
                still_open.append(pos)
        positions = still_open

        # Mark-to-market for equity curve
        mtm = equity
        for pos in positions:
            if date in close.index and pos["sym"] in close.columns:
                cur = close.loc[date, pos["sym"]]
                if not pd.isna(cur):
                    mtm += pos["shares"] * cur
        equity_curve.append({"date": str(date.date()), "equity": round(mtm, 2)})

        # Check for new entries
        if len(positions) >= MAX_CONCURRENT:
            continue

        candidates = []
        for sym in UNIVERSE:
            if any(p["sym"] == sym for p in positions):
                continue
            if not base_signal(indicators, sym, date):
                continue
            if not variant_filter(indicators, sym, date, close=close):
                continue
            # Score by RSI (lower = more oversold = better)
            r = indicators[sym]["rsi14"].loc[date]
            candidates.append((sym, r))

        # Sort by RSI ascending, take up to available slots
        candidates.sort(key=lambda x: x[1])
        slots = MAX_CONCURRENT - len(positions)
        for sym, _ in candidates[:slots]:
            price = close.loc[date, sym]
            if pd.isna(price) or price <= 0:
                continue
            slip = price * SLIPPAGE_BPS / 10000
            entry_price = price + slip  # buying
            shares = int(MAX_PER_TRADE / entry_price)
            if shares < 1:
                continue
            cost = shares * entry_price
            if cost > equity:
                continue
            equity -= cost
            positions.append({
                "sym": sym,
                "entry_date": date,
                "entry_price": entry_price,
                "shares": shares,
            })

    # Force-close remaining positions at last date
    last_date = trade_dates[-1]
    for pos in positions:
        if pos["sym"] in close.columns and last_date in close.index:
            exit_price = close.loc[last_date, pos["sym"]]
            if pd.isna(exit_price):
                continue
            slip = exit_price * SLIPPAGE_BPS / 10000
            exit_price_adj = exit_price - slip
            pnl = (exit_price_adj - pos["entry_price"]) * pos["shares"]
            ret = exit_price_adj / pos["entry_price"] - 1
            equity += pos["shares"] * exit_price_adj
            trades.append({
                "sym": pos["sym"],
                "entry_date": str(pos["entry_date"].date()),
                "exit_date": str(last_date.date()),
                "entry_price": round(pos["entry_price"], 2),
                "exit_price": round(exit_price_adj, 2),
                "shares": pos["shares"],
                "pnl": round(pnl, 2),
                "return": round(ret, 4),
            })

    return trades, equity_curve


# ── Analytics ──────────────────────────────────────────────────────
def compute_metrics(trades, equity_curve, spy_close):
    if len(trades) == 0:
        return {"n_trades": 0, "sharpe": 0, "total_return": 0}

    returns = np.array([t["return"] for t in trades])
    pnls = np.array([t["pnl"] for t in trades])
    n = len(returns)
    win_rate = np.mean(returns > 0)
    avg_ret = np.mean(returns)
    total_pnl = np.sum(pnls)
    total_ret = total_pnl / CAPITAL

    # Annualized Sharpe from trade returns (assume ~25 trades/yr as rough scaling)
    if np.std(returns) > 0:
        trades_per_year = max(n / 4.5, 1)  # ~4.5 years of data
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(trades_per_year)
    else:
        sharpe = 0

    # Sortino
    downside = returns[returns < 0]
    if len(downside) > 0 and np.std(downside) > 0:
        trades_per_year = max(n / 4.5, 1)
        sortino = (np.mean(returns) / np.std(downside)) * np.sqrt(trades_per_year)
    else:
        sortino = float("inf") if avg_ret > 0 else 0

    # Profit Factor
    gross_profit = np.sum(pnls[pnls > 0]) if np.any(pnls > 0) else 0
    gross_loss = -np.sum(pnls[pnls < 0]) if np.any(pnls < 0) else 0.01
    profit_factor = gross_profit / gross_loss

    # Max drawdown from equity curve
    equities = np.array([e["equity"] for e in equity_curve])
    peak = np.maximum.accumulate(equities)
    dd = (equities - peak) / peak
    max_dd = np.min(dd)

    # Regime analysis: classify by SPY monthly return
    trade_df = pd.DataFrame(trades)
    trade_df["entry_date"] = pd.to_datetime(trade_df["entry_date"])
    trade_df["month"] = trade_df["entry_date"].dt.to_period("M")

    # SPY monthly returns
    spy_monthly = spy_close.resample("ME").last().pct_change()

    up_rets, down_rets = [], []
    for _, row in trade_df.iterrows():
        m = row["entry_date"].to_period("M")
        # Find matching SPY month
        spy_month_dates = [d for d in spy_monthly.index if d.to_period("M") == m]
        if spy_month_dates:
            spy_ret = spy_monthly.loc[spy_month_dates[0]]
            if not pd.isna(spy_ret):
                if spy_ret >= 0:
                    up_rets.append(row["return"])
                else:
                    down_rets.append(row["return"])

    up_sharpe = np.mean(up_rets) / np.std(up_rets) if len(up_rets) > 2 and np.std(up_rets) > 0 else 0
    down_sharpe = np.mean(down_rets) / np.std(down_rets) if len(down_rets) > 2 and np.std(down_rets) > 0 else 0
    regime_gap = abs(up_sharpe - down_sharpe) / max(abs(up_sharpe), abs(down_sharpe), 0.01)

    # Best/worst trades
    best = max(trades, key=lambda t: t["return"])
    worst = min(trades, key=lambda t: t["return"])

    # Avg hold days
    hold_days_list = []
    for t in trades:
        d1 = pd.Timestamp(t["entry_date"])
        d2 = pd.Timestamp(t["exit_date"])
        hold_days_list.append((d2 - d1).days)

    return {
        "n_trades": n,
        "win_rate": round(win_rate, 3),
        "avg_return": round(avg_ret, 4),
        "total_pnl": round(total_pnl, 2),
        "total_return": round(total_ret, 4),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(profit_factor, 3),
        "max_drawdown": round(max_dd, 4),
        "avg_hold_days": round(np.mean(hold_days_list), 1),
        "regime_gap": round(regime_gap, 3),
        "up_market_sharpe": round(up_sharpe, 3),
        "down_market_sharpe": round(down_sharpe, 3),
        "best_trade": f"{best['sym']} {best['entry_date']} +{best['return']:.1%}",
        "worst_trade": f"{worst['sym']} {worst['entry_date']} {worst['return']:.1%}",
    }


# ── Permutation test ──────────────────────────────────────────────
def permutation_test(trades, equity_curve, n_perms=1000):
    """Shuffle trade returns, recompute Sharpe. p = fraction >= actual."""
    if len(trades) < 5:
        return 1.0
    returns = np.array([t["return"] for t in trades])
    actual_sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0

    rng = np.random.default_rng(42)
    count_ge = 0
    for _ in range(n_perms):
        shuffled = rng.permutation(returns)
        s = np.mean(shuffled) / np.std(shuffled) if np.std(shuffled) > 0 else 0
        if s >= actual_sharpe:
            count_ge += 1
    return count_ge / n_perms


# ── 5-Gate Validation ─────────────────────────────────────────────
def validate_5gate(metrics, perm_p):
    gates = {}
    gates["G1_Sharpe_gt_0.5"] = {"pass": metrics["sharpe"] > 0.5, "value": metrics["sharpe"]}
    gates["G2_Perm_p_lt_0.05"] = {"pass": perm_p < 0.05, "value": round(perm_p, 4)}
    gates["G3_Regime_gap_lt_0.5"] = {"pass": metrics["regime_gap"] < 0.5, "value": metrics["regime_gap"]}
    gates["G4_MaxDD_gt_neg50"] = {"pass": metrics["max_drawdown"] > -0.50, "value": metrics["max_drawdown"]}
    gates["G5_Trades_ge_20"] = {"pass": metrics["n_trades"] >= 20, "value": metrics["n_trades"]}
    gates["all_pass"] = all(g["pass"] for g in gates.values() if isinstance(g, dict))
    return gates


# ── Main ───────────────────────────────────────────────────────────
def main():
    close, high, low = download_data()
    print(f"Data shape: {close.shape}, date range: {close.index[0].date()} to {close.index[-1].date()}")

    indicators = compute_indicators(close, high, low)
    spy_close = close["SPY"].dropna()

    results = {}
    for name, filt in VARIANTS.items():
        print(f"\n{'='*60}")
        print(f"Running variant {name}...")
        trades, eq_curve = run_backtest(name, filt, indicators, close, spy_close)
        print(f"  Trades: {len(trades)}")

        if len(trades) == 0:
            results[name] = {"n_trades": 0, "gates": {"all_pass": False}}
            continue

        metrics = compute_metrics(trades, eq_curve, spy_close)
        perm_p = permutation_test(trades, eq_curve)
        gates = validate_5gate(metrics, perm_p)

        results[name] = {
            "metrics": metrics,
            "gates": gates,
            "sample_trades": trades[:5],
            "final_equity": eq_curve[-1]["equity"] if eq_curve else CAPITAL,
        }

        # Print summary
        print(f"  Win Rate: {metrics['win_rate']:.1%} | Sharpe: {metrics['sharpe']:.2f} | "
              f"Sortino: {metrics['sortino']:.2f} | PF: {metrics['profit_factor']:.2f}")
        print(f"  Total PnL: ${metrics['total_pnl']:.2f} | Return: {metrics['total_return']:.1%} | "
              f"MaxDD: {metrics['max_drawdown']:.1%}")
        print(f"  Regime gap: {metrics['regime_gap']:.3f} | Perm p: {perm_p:.4f}")
        g_str = " | ".join(f"{'✓' if v['pass'] else '✗'} {k}" for k, v in gates.items() if isinstance(v, dict))
        print(f"  Gates: {g_str}")
        print(f"  ALL PASS: {'YES ✓' if gates['all_pass'] else 'NO ✗'}")

    # ── Summary table ──────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Variant':<20} {'Trades':>6} {'WR':>6} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} "
          f"{'PnL':>8} {'MaxDD':>7} {'RGap':>6} {'Pass':>5}")
    print("-" * 80)
    for name, r in results.items():
        m = r.get("metrics")
        if m is None or m.get("n_trades", 0) == 0:
            print(f"{name:<20} {'0':>6} {'--':>6} {'--':>7} {'--':>8} {'--':>6} {'--':>8} {'--':>7} {'--':>6} {'NO':>5}")
            continue
        passed = "YES" if r["gates"]["all_pass"] else "NO"
        print(f"{name:<20} {m['n_trades']:>6} {m['win_rate']:>5.0%} {m['sharpe']:>7.2f} "
              f"{m['sortino']:>8.2f} {m['profit_factor']:>6.2f} {m['total_pnl']:>7.0f} "
              f"{m['max_drawdown']:>6.1%} {m['regime_gap']:>6.3f} {passed:>5}")

    # ── Save results ───────────────────────────────────────────────
    output = {
        "strategy": "Trend-Filtered Mean Reversion on Quality Stocks",
        "run_date": str(dt.datetime.now()),
        "period": f"{TRADE_START} to {END}",
        "capital": CAPITAL,
        "max_per_trade": MAX_PER_TRADE,
        "max_concurrent": MAX_CONCURRENT,
        "hold_days": HOLD_DAYS,
        "slippage_bps": SLIPPAGE_BPS,
        "variants": {},
    }
    for name, r in results.items():
        # Convert any non-serializable values
        variant_data = {}
        if r.get("metrics") is None or r.get("metrics", {}).get("n_trades", 0) == 0:
            variant_data = {"n_trades": 0, "gates": {"all_pass": False}}
        else:
            variant_data = {
                "metrics": r["metrics"],
                "gates": r["gates"],
                "final_equity": r["final_equity"],
                "sample_trades": r["sample_trades"],
            }
        output["variants"][name] = variant_data

    out_path = Path("/home/jupiter/Lvl3Quant/data/trend_filter_mr_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
