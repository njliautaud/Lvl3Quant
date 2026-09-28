#!/usr/bin/env python3
"""
Regime-Switching Quality Stock Strategies Backtest
===================================================
Variants A-F with 5-gate validation.

A: ADX regime switch (ranging vs trending)
B: Volatility regime (tercile-based)
C: Trend strength switch (SPY vs 50-SMA)
D: Correlation regime (macro vs stock-picking)
E: Mean reversion baseline
F: Aggressive MR timing (VIX filter)
"""

import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Parameters ──────────────────────────────────────────────────────────────
CAPITAL = 669.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
SLIPPAGE_BPS = 2
HOLD_DAYS = 10
START = "2022-01-01"
END = "2026-07-31"

UNIVERSE = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]

RESULTS_PATH = Path("/home/jupiter/Lvl3Quant/data/momentum_regime_switch_results.json")


# ── Data Download ───────────────────────────────────────────────────────────
def download_data():
    tickers = UNIVERSE + ["SPY", "^VIX"]
    print(f"Downloading {len(tickers)} tickers...")
    raw = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)
    # Handle multi-level columns from yf.download
    close = raw["Close"].copy()
    volume = raw["Volume"].copy()
    high = raw["High"].copy()
    low = raw["Low"].copy()
    # Rename ^VIX -> VIX
    for df in [close, volume, high, low]:
        if "^VIX" in df.columns:
            df.rename(columns={"^VIX": "VIX"}, inplace=True)
    return close, volume, high, low


# ── Technical Indicators ────────────────────────────────────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_adx(high, low, close, period=14):
    """ADX for a single series (SPY)."""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    # Where +DM > -DM, keep +DM; else 0 (and vice versa)
    plus_dm[plus_dm <= minus_dm] = 0
    minus_dm[minus_dm <= plus_dm] = 0

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    adx = dx.rolling(period).mean()
    return adx


def compute_realized_vol(close, window=20):
    return close.pct_change().rolling(window).std() * np.sqrt(252)


# ── Position Tracker ────────────────────────────────────────────────────────
class Backtest:
    def __init__(self, capital, max_per_trade, max_concurrent, slippage_bps, hold_days):
        self.capital = capital
        self.initial_capital = capital
        self.max_per_trade = max_per_trade
        self.max_concurrent = max_concurrent
        self.slippage_bps = slippage_bps
        self.hold_days = hold_days
        self.positions = []  # list of {ticker, entry_price, shares, entry_date, entry_idx}
        self.trades = []     # completed trades
        self.equity_curve = []

    def open_position(self, ticker, price, date, idx):
        if len(self.positions) >= self.max_concurrent:
            return False
        entry_price = price * (1 + self.slippage_bps / 10000)
        shares = min(self.max_per_trade, self.capital) / entry_price
        if shares * entry_price < 1:  # min $1
            return False
        cost = shares * entry_price
        self.capital -= cost
        self.positions.append({
            "ticker": ticker,
            "entry_price": entry_price,
            "shares": shares,
            "entry_date": date,
            "entry_idx": idx,
        })
        return True

    def check_exits(self, close_prices, date, idx):
        remaining = []
        for pos in self.positions:
            days_held = idx - pos["entry_idx"]
            if days_held >= self.hold_days:
                exit_price = close_prices.get(pos["ticker"], pos["entry_price"])
                exit_price *= (1 - self.slippage_bps / 10000)
                proceeds = pos["shares"] * exit_price
                self.capital += proceeds
                pnl = proceeds - pos["shares"] * pos["entry_price"]
                ret = pnl / (pos["shares"] * pos["entry_price"])
                self.trades.append({
                    "ticker": pos["ticker"],
                    "entry_date": str(pos["entry_date"]),
                    "exit_date": str(date),
                    "entry_price": round(pos["entry_price"], 4),
                    "exit_price": round(exit_price, 4),
                    "pnl": round(pnl, 4),
                    "return": round(ret, 6),
                })
            else:
                remaining.append(pos)
        self.positions = remaining

    def mark_to_market(self, close_prices):
        mtm = self.capital
        for pos in self.positions:
            price = close_prices.get(pos["ticker"], pos["entry_price"])
            mtm += pos["shares"] * price
        return mtm

    def record_equity(self, date, close_prices):
        self.equity_curve.append({
            "date": str(date),
            "equity": round(self.mark_to_market(close_prices), 2),
        })

    def get_metrics(self):
        if not self.trades:
            return self._empty_metrics()
        returns = np.array([t["return"] for t in self.trades])
        n = len(returns)
        win_rate = np.mean(returns > 0)
        avg_ret = np.mean(returns)
        total_pnl = sum(t["pnl"] for t in self.trades)
        gross_profit = sum(t["pnl"] for t in self.trades if t["pnl"] > 0)
        gross_loss = abs(sum(t["pnl"] for t in self.trades if t["pnl"] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Equity-based metrics
        eq = np.array([e["equity"] for e in self.equity_curve])
        if len(eq) < 2:
            return self._empty_metrics()
        daily_rets = np.diff(eq) / eq[:-1]
        sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252) if np.std(daily_rets) > 0 else 0
        downside = daily_rets[daily_rets < 0]
        sortino = np.mean(daily_rets) / np.std(downside) * np.sqrt(252) if len(downside) > 0 and np.std(downside) > 0 else 0
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        max_dd = dd.min()
        total_return = (eq[-1] / eq[0]) - 1

        return {
            "n_trades": n,
            "win_rate": round(win_rate, 4),
            "avg_return": round(avg_ret, 6),
            "total_pnl": round(total_pnl, 2),
            "total_return": round(total_return, 4),
            "profit_factor": round(profit_factor, 3),
            "sharpe": round(sharpe, 4),
            "sortino": round(sortino, 4),
            "max_drawdown": round(max_dd, 4),
            "final_equity": round(eq[-1], 2),
        }

    def _empty_metrics(self):
        return {
            "n_trades": 0, "win_rate": 0, "avg_return": 0, "total_pnl": 0,
            "total_return": 0, "profit_factor": 0, "sharpe": 0, "sortino": 0,
            "max_drawdown": 0, "final_equity": self.initial_capital,
        }


# ── Signal Generators ──────────────────────────────────────────────────────
def is_quality_dip(ticker, idx, close, high_df, rsi_df, pct_below=0.05, rsi_thresh=40):
    """Stock is >pct_below% below its 20-day high AND RSI < rsi_thresh."""
    if idx < 20:
        return False
    high_20 = high_df[ticker].iloc[max(0, idx-20):idx+1].max()
    cur = close[ticker].iloc[idx]
    rsi_val = rsi_df[ticker].iloc[idx]
    if pd.isna(cur) or pd.isna(rsi_val) or pd.isna(high_20):
        return False
    return (cur < high_20 * (1 - pct_below)) and (rsi_val < rsi_thresh)


def is_breakout(ticker, idx, close, volume_df, lookback=20, vol_mult=1.2):
    """New 20-day high with volume > vol_mult × average."""
    if idx < lookback:
        return False
    cur_close = close[ticker].iloc[idx]
    prev_high = close[ticker].iloc[idx-lookback:idx].max()
    cur_vol = volume_df[ticker].iloc[idx]
    avg_vol = volume_df[ticker].iloc[idx-lookback:idx].mean()
    if pd.isna(cur_close) or pd.isna(prev_high) or pd.isna(cur_vol) or pd.isna(avg_vol):
        return False
    return (cur_close > prev_high) and (cur_vol > vol_mult * avg_vol)


# ── Strategy Runners ───────────────────────────────────────────────────────
def run_variant_a(close, volume, high_df, low_df):
    """ADX regime switch."""
    print("  Running Variant A: ADX regime switch...")
    bt = Backtest(CAPITAL, MAX_PER_TRADE, MAX_CONCURRENT, SLIPPAGE_BPS, HOLD_DAYS)
    rsi_df = pd.DataFrame({t: compute_rsi(close[t]) for t in UNIVERSE})
    adx = compute_adx(high_df["SPY"], low_df["SPY"], close["SPY"], 14)
    dates = close.index

    for idx in range(50, len(dates)):
        date = dates[idx]
        prices = {t: close[t].iloc[idx] for t in UNIVERSE if not pd.isna(close[t].iloc[idx])}
        bt.check_exits(prices, date, idx)

        adx_val = adx.iloc[idx]
        if pd.isna(adx_val):
            bt.record_equity(date, prices)
            continue

        signals = []
        if adx_val < 20:  # ranging -> MR dip buys
            for t in UNIVERSE:
                if is_quality_dip(t, idx, close, high_df, rsi_df):
                    signals.append(t)
        elif adx_val > 30:  # trending -> breakouts
            for t in UNIVERSE:
                if is_breakout(t, idx, close, volume):
                    signals.append(t)

        for t in signals:
            if t in prices:
                bt.open_position(t, prices[t], date, idx)

        bt.record_equity(date, prices)
    # Force close remaining
    _force_close(bt, close, dates)
    return bt


def run_variant_b(close, volume, high_df, low_df):
    """Volatility regime."""
    print("  Running Variant B: Volatility regime...")
    bt = Backtest(CAPITAL, MAX_PER_TRADE, MAX_CONCURRENT, SLIPPAGE_BPS, HOLD_DAYS)
    rsi_df = pd.DataFrame({t: compute_rsi(close[t]) for t in UNIVERSE})
    spy_vol = compute_realized_vol(close["SPY"], 20)
    # Compute expanding tercile boundaries
    dates = close.index

    for idx in range(60, len(dates)):
        date = dates[idx]
        prices = {t: close[t].iloc[idx] for t in UNIVERSE if not pd.isna(close[t].iloc[idx])}
        bt.check_exits(prices, date, idx)

        vol_val = spy_vol.iloc[idx]
        if pd.isna(vol_val):
            bt.record_equity(date, prices)
            continue

        # Use expanding window for tercile thresholds (avoid lookahead)
        vol_hist = spy_vol.iloc[20:idx+1].dropna()
        lo_thresh = vol_hist.quantile(0.33)
        hi_thresh = vol_hist.quantile(0.67)

        signals = []
        if vol_val <= lo_thresh or vol_val >= hi_thresh:
            # Low vol (ranging) or high vol (panic) -> MR dip buys
            for t in UNIVERSE:
                if is_quality_dip(t, idx, close, high_df, rsi_df):
                    signals.append(t)
        # Middle tercile -> sit out

        for t in signals:
            if t in prices:
                bt.open_position(t, prices[t], date, idx)

        bt.record_equity(date, prices)
    _force_close(bt, close, dates)
    return bt


def run_variant_c(close, volume, high_df, low_df):
    """Trend strength switch (SPY vs 50-SMA)."""
    print("  Running Variant C: Trend strength switch...")
    bt = Backtest(CAPITAL, MAX_PER_TRADE, MAX_CONCURRENT, SLIPPAGE_BPS, HOLD_DAYS)
    rsi_df = pd.DataFrame({t: compute_rsi(close[t]) for t in UNIVERSE})
    spy_sma50 = close["SPY"].rolling(50).mean()
    dates = close.index

    for idx in range(60, len(dates)):
        date = dates[idx]
        prices = {t: close[t].iloc[idx] for t in UNIVERSE if not pd.isna(close[t].iloc[idx])}
        bt.check_exits(prices, date, idx)

        spy_price = close["SPY"].iloc[idx]
        sma_val = spy_sma50.iloc[idx]
        if pd.isna(spy_price) or pd.isna(sma_val):
            bt.record_equity(date, prices)
            continue

        pct_from_sma = (spy_price - sma_val) / sma_val

        signals = []
        if pct_from_sma > 0.02:
            # Strong uptrend -> momentum breakouts
            for t in UNIVERSE:
                if is_breakout(t, idx, close, volume):
                    signals.append(t)
        elif pct_from_sma > -0.02:
            # Flat -> standard MR dip buys
            for t in UNIVERSE:
                if is_quality_dip(t, idx, close, high_df, rsi_df):
                    signals.append(t)
        else:
            # Downtrend -> deep MR only (RSI < 30)
            for t in UNIVERSE:
                if is_quality_dip(t, idx, close, high_df, rsi_df, rsi_thresh=30):
                    signals.append(t)

        for t in signals:
            if t in prices:
                bt.open_position(t, prices[t], date, idx)

        bt.record_equity(date, prices)
    _force_close(bt, close, dates)
    return bt


def run_variant_d(close, volume, high_df, low_df):
    """Correlation regime."""
    print("  Running Variant D: Correlation regime...")
    bt = Backtest(CAPITAL, MAX_PER_TRADE, MAX_CONCURRENT, SLIPPAGE_BPS, HOLD_DAYS)
    rsi_df = pd.DataFrame({t: compute_rsi(close[t]) for t in UNIVERSE})
    # Pre-compute rolling returns for correlation
    rets = close[UNIVERSE].pct_change()
    dates = close.index

    for idx in range(80, len(dates)):
        date = dates[idx]
        prices = {t: close[t].iloc[idx] for t in UNIVERSE if not pd.isna(close[t].iloc[idx])}
        bt.check_exits(prices, date, idx)

        # Compute avg pairwise correlation over trailing 60 days
        window_rets = rets.iloc[max(0, idx-60):idx+1]
        corr_matrix = window_rets.corr()
        # Average of upper triangle (exclude diagonal)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        avg_corr = corr_matrix.values[mask].mean()

        if pd.isna(avg_corr):
            bt.record_equity(date, prices)
            continue

        signals = []
        if avg_corr > 0.6:
            # Macro-driven -> buy SPY dips
            spy_rsi = compute_rsi(close["SPY"]).iloc[idx]
            spy_high20 = close["SPY"].iloc[max(0, idx-20):idx+1].max()
            spy_price = close["SPY"].iloc[idx]
            if not pd.isna(spy_rsi) and spy_rsi < 40 and spy_price < spy_high20 * 0.95:
                signals.append("SPY")
        elif avg_corr < 0.4:
            # Stock-picking -> individual quality dips
            for t in UNIVERSE:
                if is_quality_dip(t, idx, close, high_df, rsi_df):
                    signals.append(t)

        for t in signals:
            p = close[t].iloc[idx] if t in close.columns else None
            if p is not None and not pd.isna(p):
                bt.open_position(t, p, date, idx)

        bt.record_equity(date, prices)
    _force_close(bt, close, dates)
    return bt


def run_variant_e(close, volume, high_df, low_df):
    """Baseline: always MR dip buys."""
    print("  Running Variant E: Mean reversion baseline...")
    bt = Backtest(CAPITAL, MAX_PER_TRADE, MAX_CONCURRENT, SLIPPAGE_BPS, HOLD_DAYS)
    rsi_df = pd.DataFrame({t: compute_rsi(close[t]) for t in UNIVERSE})
    dates = close.index

    for idx in range(30, len(dates)):
        date = dates[idx]
        prices = {t: close[t].iloc[idx] for t in UNIVERSE if not pd.isna(close[t].iloc[idx])}
        bt.check_exits(prices, date, idx)

        signals = []
        for t in UNIVERSE:
            if is_quality_dip(t, idx, close, high_df, rsi_df):
                signals.append(t)

        for t in signals:
            if t in prices:
                bt.open_position(t, prices[t], date, idx)

        bt.record_equity(date, prices)
    _force_close(bt, close, dates)
    return bt


def run_variant_f(close, volume, high_df, low_df):
    """Aggressive MR timing: VIX filter."""
    print("  Running Variant F: Aggressive MR + VIX filter...")
    bt = Backtest(CAPITAL, MAX_PER_TRADE, MAX_CONCURRENT, SLIPPAGE_BPS, HOLD_DAYS)
    rsi_df = pd.DataFrame({t: compute_rsi(close[t]) for t in UNIVERSE})
    dates = close.index

    for idx in range(30, len(dates)):
        date = dates[idx]
        prices = {t: close[t].iloc[idx] for t in UNIVERSE if not pd.isna(close[t].iloc[idx])}
        bt.check_exits(prices, date, idx)

        vix_val = close["VIX"].iloc[idx] if "VIX" in close.columns else np.nan
        if pd.isna(vix_val) or vix_val < 20:
            # Skip when VIX < 20 (complacency)
            bt.record_equity(date, prices)
            continue

        signals = []
        for t in UNIVERSE:
            if is_quality_dip(t, idx, close, high_df, rsi_df):
                signals.append(t)

        for t in signals:
            if t in prices:
                bt.open_position(t, prices[t], date, idx)

        bt.record_equity(date, prices)
    _force_close(bt, close, dates)
    return bt


def _force_close(bt, close, dates):
    """Force-close any remaining positions at last available prices."""
    last_idx = len(dates) - 1
    last_date = dates[last_idx]
    prices = {t: close[t].iloc[last_idx] for t in close.columns if not pd.isna(close[t].iloc[last_idx])}
    # Override hold check by setting entry_idx far back
    for pos in bt.positions:
        pos["entry_idx"] = -9999
    bt.check_exits(prices, last_date, last_idx)


# ── Validation ──────────────────────────────────────────────────────────────
def permutation_test(returns, n_perms=1000):
    """Shuffle trade returns, compute fraction of shuffled Sharpes >= actual."""
    if len(returns) < 5:
        return 1.0
    actual_sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
    count = 0
    for _ in range(n_perms):
        shuffled = np.random.permutation(returns)
        s = np.mean(shuffled) / np.std(shuffled) if np.std(shuffled) > 0 else 0
        if s >= actual_sharpe:
            count += 1
    return count / n_perms


def compute_regime_gap(bt, close):
    """Compute |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|)."""
    if not bt.trades:
        return 1.0
    spy_rets = close["SPY"].pct_change()
    green_trades = []
    red_trades = []
    for t in bt.trades:
        entry_date = pd.Timestamp(t["entry_date"])
        exit_date = pd.Timestamp(t["exit_date"])
        # SPY return over trade period
        mask = (spy_rets.index >= entry_date) & (spy_rets.index <= exit_date)
        spy_period_ret = spy_rets[mask].sum()
        if spy_period_ret >= 0:
            green_trades.append(t["return"])
        else:
            red_trades.append(t["return"])

    def _sharpe(rets_list):
        if len(rets_list) < 2:
            return 0
        arr = np.array(rets_list)
        return np.mean(arr) / np.std(arr) if np.std(arr) > 0 else 0

    sg = _sharpe(green_trades)
    sr = _sharpe(red_trades)
    denom = max(abs(sg), abs(sr))
    if denom == 0:
        return 0
    return abs(sg - sr) / denom


def validate(bt, close, variant_name):
    """5-gate validation."""
    metrics = bt.get_metrics()
    returns = np.array([t["return"] for t in bt.trades]) if bt.trades else np.array([])

    perm_p = permutation_test(returns) if len(returns) >= 5 else 1.0
    regime_gap = compute_regime_gap(bt, close)

    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": perm_p < 0.05,
        "regime_gap_lt_0.5": regime_gap < 0.5,
        "max_dd_gt_neg50pct": metrics["max_drawdown"] > -0.50,
        "min_20_trades": metrics["n_trades"] >= 20,
    }
    passed = sum(gates.values())

    return {
        "variant": variant_name,
        "metrics": metrics,
        "validation": {
            "gates_passed": f"{passed}/5",
            "all_passed": passed == 5,
            "details": {k: bool(v) for k, v in gates.items()},
            "permutation_p": round(perm_p, 4),
            "regime_gap": round(regime_gap, 4),
        },
        "n_sample_trades": bt.trades[:5] if bt.trades else [],
    }


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    np.random.seed(42)
    close, volume, high_df, low_df = download_data()
    print(f"Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} days\n")

    runners = {
        "A_ADX_Regime_Switch": run_variant_a,
        "B_Volatility_Regime": run_variant_b,
        "C_Trend_Strength_Switch": run_variant_c,
        "D_Correlation_Regime": run_variant_d,
        "E_MR_Baseline": run_variant_e,
        "F_Aggressive_MR_VIX": run_variant_f,
    }

    results = {}
    for name, runner in runners.items():
        bt = runner(close, volume, high_df, low_df)
        result = validate(bt, close, name)
        results[name] = result
        m = result["metrics"]
        v = result["validation"]
        print(f"  {name}: {m['n_trades']} trades | Sharpe {m['sharpe']:.3f} | "
              f"Sortino {m['sortino']:.3f} | WR {m['win_rate']:.1%} | "
              f"PF {m['profit_factor']:.2f} | MaxDD {m['max_drawdown']:.1%} | "
              f"Total {m['total_return']:.1%} | Gates {v['gates_passed']} "
              f"{'PASS' if v['all_passed'] else 'FAIL'}")
        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Variant':<30} {'Trades':>6} {'Sharpe':>8} {'Sortino':>8} {'WR':>6} "
          f"{'PF':>6} {'MaxDD':>8} {'Return':>8} {'Gates':>7}")
    print("-" * 80)
    for name, r in results.items():
        m = r["metrics"]
        v = r["validation"]
        tag = "PASS" if v["all_passed"] else "FAIL"
        print(f"{name:<30} {m['n_trades']:>6} {m['sharpe']:>8.3f} {m['sortino']:>8.3f} "
              f"{m['win_rate']:>5.1%} {m['profit_factor']:>6.2f} {m['max_drawdown']:>7.1%} "
              f"{m['total_return']:>7.1%} {v['gates_passed']:>5} {tag}")

    # Save
    output = {
        "metadata": {
            "strategy": "Regime-Switching Quality Stock Strategies",
            "period": f"{START} to {END}",
            "capital": CAPITAL,
            "max_per_trade": MAX_PER_TRADE,
            "max_concurrent": MAX_CONCURRENT,
            "slippage_bps": SLIPPAGE_BPS,
            "hold_days": HOLD_DAYS,
            "universe": UNIVERSE,
            "run_date": str(dt.datetime.now()),
        },
        "results": results,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
