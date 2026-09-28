#!/usr/bin/env python3
"""
Volatility Regime-Based Entry Backtest on Quality Stocks
=========================================================
6 variants (A-F) using VIX/vol regime signals to time entries into
dipped quality stocks. 5-gate validation framework.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]
START = "2021-06-01"  # extra lookback for indicators
END = "2026-07-31"
TRADE_START = "2022-01-01"
CAPITAL = 669.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
HOLD_DAYS = 10
SLIPPAGE_BPS = 2
N_PERM = 1000
OUTPUT_PATH = Path("/home/jupiter/Lvl3Quant/data/vol_regime_entry_results.json")


# ── DATA DOWNLOAD ───────────────────────────────────────────────────────────
def download_data():
    """Download all required price data."""
    print("Downloading stock data...")
    tickers = UNIVERSE + ["SPY"]
    stock_data = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)

    print("Downloading VIX data...")
    vix = yf.download("^VIX", start=START, end=END, auto_adjust=True, progress=False)

    print("Downloading VIX3M / VIXM data...")
    # Try ^VIX3M first, fall back to VIXM ETF
    vix3m = yf.download("^VIX3M", start=START, end=END, auto_adjust=True, progress=False)
    if vix3m.empty or len(vix3m) < 100:
        print("  ^VIX3M unavailable, trying VIXM ETF as proxy...")
        vix3m = yf.download("VIXM", start=START, end=END, auto_adjust=True, progress=False)
        vix3m_is_etf = True
    else:
        vix3m_is_etf = False

    return stock_data, vix, vix3m, vix3m_is_etf


def build_features(stock_data, vix, vix3m, vix3m_is_etf):
    """Build all features needed for the 6 variants."""
    # Extract close prices - handle MultiIndex
    close = stock_data["Close"].copy()
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)

    spy_close = close["SPY"] if "SPY" in close.columns else None

    # VIX close
    vix_close = vix["Close"].copy()
    if isinstance(vix_close, pd.DataFrame):
        vix_close = vix_close.iloc[:, 0]
    vix_close.name = "VIX"

    # VIX3M close
    vix3m_close = vix3m["Close"].copy()
    if isinstance(vix3m_close, pd.DataFrame):
        vix3m_close = vix3m_close.iloc[:, 0]
    vix3m_close.name = "VIX3M"

    # Per-stock features
    high_20d = close.rolling(20).max()
    dip_pct = (close - high_20d) / high_20d  # negative when below high

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # Realized vol (10d and 60d) for each stock
    log_ret = np.log(close / close.shift(1))
    rvol_10d = log_ret.rolling(10).std() * np.sqrt(252)
    rvol_60d = log_ret.rolling(60).std() * np.sqrt(252)

    # SPY realized vol for variant F
    if spy_close is not None:
        spy_logret = np.log(spy_close / spy_close.shift(1))
        spy_rvol_20d = spy_logret.rolling(20).std() * np.sqrt(252)
    else:
        spy_rvol_20d = None

    # VIX features
    vix_pct_change_5d = vix_close.pct_change(5) * 100  # % change over 5 days
    vix_daily_change = vix_close.diff()

    # Avg RSI across universe (for variant D)
    stock_rsi = rsi[[t for t in UNIVERSE if t in rsi.columns]]
    avg_rsi = stock_rsi.mean(axis=1)

    return {
        "close": close,
        "high_20d": high_20d,
        "dip_pct": dip_pct,
        "rsi": rsi,
        "rvol_10d": rvol_10d,
        "rvol_60d": rvol_60d,
        "vix_close": vix_close,
        "vix3m_close": vix3m_close,
        "vix3m_is_etf": vix3m_is_etf,
        "vix_pct_change_5d": vix_pct_change_5d,
        "vix_daily_change": vix_daily_change,
        "avg_rsi": avg_rsi,
        "spy_rvol_20d": spy_rvol_20d,
        "spy_close": spy_close,
    }


# ── SIGNAL GENERATORS ──────────────────────────────────────────────────────
def get_dipped_stocks(feat, date, min_dip=-0.05, rsi_filter=None):
    """Get stocks dipped >5% below 20-day high, optionally RSI-filtered."""
    candidates = []
    for ticker in UNIVERSE:
        if ticker not in feat["close"].columns:
            continue
        try:
            dip = feat["dip_pct"].loc[date, ticker]
            r = feat["rsi"].loc[date, ticker]
            price = feat["close"].loc[date, ticker]
        except (KeyError, TypeError):
            continue
        if pd.isna(dip) or pd.isna(price):
            continue
        if dip > min_dip:  # not dipped enough
            continue
        if rsi_filter is not None and (pd.isna(r) or r >= rsi_filter):
            continue
        candidates.append((ticker, dip, r, price))
    return candidates


def variant_A_signals(feat, dates):
    """VIX spike entry: buy 3 most-dipped when VIX spikes >30% in 5d."""
    signals = []
    for date in dates:
        try:
            vix_chg = feat["vix_pct_change_5d"].loc[date]
        except KeyError:
            continue
        if pd.isna(vix_chg) or vix_chg <= 30:
            continue
        candidates = get_dipped_stocks(feat, date, min_dip=-0.05)
        if not candidates:
            continue
        # Sort by dip (most negative first) → top 3
        candidates.sort(key=lambda x: x[1])
        for ticker, dip, r, price in candidates[:3]:
            signals.append({"date": date, "ticker": ticker, "price": price, "dip": dip})
    return signals


def variant_B_signals(feat, dates):
    """VIX term structure inversion: VIX > VIX3M → panic. Buy dipped+RSI<40."""
    signals = []
    for date in dates:
        try:
            v = feat["vix_close"].loc[date]
            v3m = feat["vix3m_close"].loc[date]
        except KeyError:
            continue
        if pd.isna(v) or pd.isna(v3m):
            continue
        # For VIXM ETF, inversion means VIX is relatively high vs medium-term
        # We can't directly compare VIX level to VIXM price, so use ratio change
        if feat["vix3m_is_etf"]:
            # VIXM goes UP when medium-term vol rises. If VIX spikes but VIXM doesn't
            # proportionally, that's short-term panic. Use VIX > 25 as proxy.
            if v <= 25:
                continue
        else:
            if v <= v3m:
                continue

        candidates = get_dipped_stocks(feat, date, min_dip=-0.05, rsi_filter=40)
        if not candidates:
            continue
        candidates.sort(key=lambda x: x[1])
        for ticker, dip, r, price in candidates[:3]:
            signals.append({"date": date, "ticker": ticker, "price": price, "dip": dip})
    return signals


def variant_C_signals(feat, dates):
    """Realized vol compression after dip: 10d rvol < 60d rvol AND dipped >5%."""
    signals = []
    for date in dates:
        for ticker in UNIVERSE:
            if ticker not in feat["close"].columns:
                continue
            try:
                dip = feat["dip_pct"].loc[date, ticker]
                rv10 = feat["rvol_10d"].loc[date, ticker]
                rv60 = feat["rvol_60d"].loc[date, ticker]
                price = feat["close"].loc[date, ticker]
            except (KeyError, TypeError):
                continue
            if pd.isna(dip) or pd.isna(rv10) or pd.isna(rv60) or pd.isna(price):
                continue
            if dip > -0.05:
                continue
            if rv10 >= rv60:
                continue
            signals.append({"date": date, "ticker": ticker, "price": price, "dip": dip})
    return signals


def variant_D_signals(feat, dates):
    """Cross-stock vol signal: avg RSI of all 20 < 40 → buy 3 most oversold."""
    signals = []
    for date in dates:
        try:
            avg_r = feat["avg_rsi"].loc[date]
        except KeyError:
            continue
        if pd.isna(avg_r) or avg_r >= 40:
            continue
        # Get all stocks with valid RSI, pick 3 most oversold
        candidates = []
        for ticker in UNIVERSE:
            if ticker not in feat["rsi"].columns:
                continue
            try:
                r = feat["rsi"].loc[date, ticker]
                price = feat["close"].loc[date, ticker]
                dip = feat["dip_pct"].loc[date, ticker]
            except (KeyError, TypeError):
                continue
            if pd.isna(r) or pd.isna(price):
                continue
            candidates.append((ticker, dip if not pd.isna(dip) else 0, r, price))
        if not candidates:
            continue
        # Sort by RSI ascending (most oversold)
        candidates.sort(key=lambda x: x[2])
        for ticker, dip, r, price in candidates[:3]:
            signals.append({"date": date, "ticker": ticker, "price": price, "rsi": r})
    return signals


def variant_E_signals(feat, dates):
    """VIX mean reversion + dip: VIX>25, VIX dropping today, stock dipped+RSI<40."""
    signals = []
    for date in dates:
        try:
            v = feat["vix_close"].loc[date]
            vchg = feat["vix_daily_change"].loc[date]
        except KeyError:
            continue
        if pd.isna(v) or pd.isna(vchg):
            continue
        if v <= 25 or vchg >= 0:
            continue
        candidates = get_dipped_stocks(feat, date, min_dip=-0.05, rsi_filter=40)
        if not candidates:
            continue
        candidates.sort(key=lambda x: x[1])
        for ticker, dip, r, price in candidates[:3]:
            signals.append({"date": date, "ticker": ticker, "price": price, "dip": dip})
    return signals


def variant_F_signals(feat, dates):
    """Implied-realized vol gap: VIX > SPY 20d rvol + 5pts → buy dipped stocks."""
    signals = []
    if feat["spy_rvol_20d"] is None:
        return signals
    for date in dates:
        try:
            v = feat["vix_close"].loc[date]
            spy_rv = feat["spy_rvol_20d"].loc[date]
        except KeyError:
            continue
        if pd.isna(v) or pd.isna(spy_rv):
            continue
        # VIX is annualized %, spy_rvol_20d is annualized decimal → convert
        spy_rv_pct = spy_rv * 100
        if v <= spy_rv_pct + 5:
            continue
        candidates = get_dipped_stocks(feat, date, min_dip=-0.05)
        if not candidates:
            continue
        candidates.sort(key=lambda x: x[1])
        for ticker, dip, r, price in candidates[:3]:
            signals.append({"date": date, "ticker": ticker, "price": price, "dip": dip})
    return signals


# ── BACKTEST ENGINE ─────────────────────────────────────────────────────────
def run_backtest(signals, feat, variant_name):
    """
    Execute trades from signals with position limits and sizing.
    Returns trade list and equity curve.
    """
    close = feat["close"]
    trades = []
    open_positions = []  # (ticker, entry_date, entry_price, shares, exit_date_idx)
    equity = CAPITAL
    equity_curve = []
    trade_dates = close.index[close.index >= TRADE_START]

    # Build signal lookup: date → list of signals
    sig_by_date = {}
    for s in signals:
        d = s["date"]
        if d not in sig_by_date:
            sig_by_date[d] = []
        sig_by_date[d].append(s)

    for i, date in enumerate(trade_dates):
        # Close expired positions
        still_open = []
        for pos in open_positions:
            ticker, entry_date, entry_price, shares, exit_idx = pos
            if i >= exit_idx:
                # Find exit price
                exit_date_actual = trade_dates[min(exit_idx, len(trade_dates) - 1)]
                try:
                    exit_price = close.loc[exit_date_actual, ticker]
                except (KeyError, TypeError):
                    exit_price = entry_price  # fallback
                if pd.isna(exit_price):
                    exit_price = entry_price

                # Apply slippage on both sides
                adj_entry = entry_price * (1 + SLIPPAGE_BPS / 10000)
                adj_exit = exit_price * (1 - SLIPPAGE_BPS / 10000)
                pnl = (adj_exit - adj_entry) * shares
                ret = (adj_exit / adj_entry) - 1
                equity += pnl
                trades.append({
                    "ticker": ticker,
                    "entry_date": str(entry_date.date()) if hasattr(entry_date, 'date') else str(entry_date),
                    "exit_date": str(exit_date_actual.date()) if hasattr(exit_date_actual, 'date') else str(exit_date_actual),
                    "entry_price": round(float(entry_price), 2),
                    "exit_price": round(float(exit_price), 2),
                    "shares": int(shares),
                    "pnl": round(float(pnl), 2),
                    "return": round(float(ret), 4),
                })
            else:
                still_open.append(pos)
        open_positions = still_open

        # Open new positions from signals
        if date in sig_by_date:
            for sig in sig_by_date[date]:
                if len(open_positions) >= MAX_CONCURRENT:
                    break
                ticker = sig["ticker"]
                price = sig["price"]
                # Check not already holding this ticker
                if any(p[0] == ticker for p in open_positions):
                    continue
                # Size: min(MAX_PER_TRADE, available equity / remaining slots)
                available = equity - sum(p[2] * p[3] for p in open_positions)  # rough
                size = min(MAX_PER_TRADE, max(available, 0))
                if size < 10 or price <= 0:
                    continue
                shares = max(int(size / price), 1)
                if shares * price > MAX_PER_TRADE:
                    shares = max(int(MAX_PER_TRADE / price), 1)
                exit_idx = i + HOLD_DAYS
                open_positions.append((ticker, date, price, shares, exit_idx))

        # Track equity (mark-to-market)
        mtm = equity
        for pos in open_positions:
            ticker, entry_date, entry_price, shares, exit_idx = pos
            try:
                cur_price = close.loc[date, ticker]
            except (KeyError, TypeError):
                cur_price = entry_price
            if pd.isna(cur_price):
                cur_price = entry_price
            mtm += (cur_price - entry_price) * shares
        equity_curve.append({"date": str(date.date()) if hasattr(date, 'date') else str(date), "equity": round(float(mtm), 2)})

    # Force-close any remaining open positions at last available price
    last_date = trade_dates[-1]
    for pos in open_positions:
        ticker, entry_date, entry_price, shares, exit_idx = pos
        try:
            exit_price = close.loc[last_date, ticker]
        except (KeyError, TypeError):
            exit_price = entry_price
        if pd.isna(exit_price):
            exit_price = entry_price
        adj_entry = entry_price * (1 + SLIPPAGE_BPS / 10000)
        adj_exit = exit_price * (1 - SLIPPAGE_BPS / 10000)
        pnl = (adj_exit - adj_entry) * shares
        ret = (adj_exit / adj_entry) - 1
        equity += pnl
        trades.append({
            "ticker": ticker,
            "entry_date": str(entry_date.date()) if hasattr(entry_date, 'date') else str(entry_date),
            "exit_date": str(last_date.date()) if hasattr(last_date, 'date') else str(last_date),
            "entry_price": round(float(entry_price), 2),
            "exit_price": round(float(exit_price), 2),
            "shares": int(shares),
            "pnl": round(float(pnl), 2),
            "return": round(float(ret), 4),
        })

    return trades, equity_curve


# ── METRICS & VALIDATION ───────────────────────────────────────────────────
def compute_metrics(trades, equity_curve):
    """Compute performance metrics from trade list."""
    if not trades:
        return {
            "total_trades": 0, "sharpe": 0, "sortino": 0, "profit_factor": 0,
            "win_rate": 0, "total_pnl": 0, "max_drawdown": 0, "avg_return": 0,
            "final_equity": CAPITAL,
        }

    returns = [t["return"] for t in trades]
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    total_pnl = sum(pnls)
    win_rate = len(wins) / len(pnls) if pnls else 0
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1e-9
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1) if len(returns) > 1 else 1e-9
    # Annualize: ~25 trades/year assumption for Sharpe scaling
    trades_per_year = max(len(trades) / 4.5, 1)  # 4.5 years of data
    sharpe = (avg_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0

    downside = [r for r in returns if r < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = (avg_ret / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0

    # Max drawdown from equity curve
    if equity_curve:
        eq = [e["equity"] for e in equity_curve]
        peak = eq[0]
        max_dd = 0
        for e in eq:
            if e > peak:
                peak = e
            dd = (e - peak) / peak if peak > 0 else 0
            if dd < max_dd:
                max_dd = dd
    else:
        max_dd = 0

    final_eq = equity_curve[-1]["equity"] if equity_curve else CAPITAL

    return {
        "total_trades": len(trades),
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "profit_factor": round(float(profit_factor), 3),
        "win_rate": round(float(win_rate), 4),
        "total_pnl": round(float(total_pnl), 2),
        "max_drawdown": round(float(max_dd), 4),
        "avg_return": round(float(avg_ret), 5),
        "final_equity": round(float(final_eq), 2),
    }


def permutation_test(trades, feat, n_perm=N_PERM):
    """
    Shuffle entry dates randomly, re-run P&L, compare to actual.
    Returns p-value.
    """
    if len(trades) < 5:
        return 1.0

    actual_pnl = sum(t["pnl"] for t in trades)
    trade_dates_available = feat["close"].index[feat["close"].index >= TRADE_START]
    tickers_used = list(set(t["ticker"] for t in trades))
    close = feat["close"]
    n_trades = len(trades)
    count_better = 0

    rng = np.random.default_rng(42)
    for _ in range(n_perm):
        perm_pnl = 0
        random_dates = rng.choice(trade_dates_available, size=n_trades, replace=True)
        for j, rd in enumerate(random_dates):
            ticker = tickers_used[j % len(tickers_used)]
            try:
                entry_price = close.loc[rd, ticker]
            except (KeyError, TypeError):
                continue
            if pd.isna(entry_price) or entry_price <= 0:
                continue
            # Find exit price HOLD_DAYS later
            idx = trade_dates_available.get_loc(rd)
            exit_idx = min(idx + HOLD_DAYS, len(trade_dates_available) - 1)
            exit_date = trade_dates_available[exit_idx]
            try:
                exit_price = close.loc[exit_date, ticker]
            except (KeyError, TypeError):
                continue
            if pd.isna(exit_price):
                continue
            shares = max(int(MAX_PER_TRADE / entry_price), 1)
            adj_entry = entry_price * (1 + SLIPPAGE_BPS / 10000)
            adj_exit = exit_price * (1 - SLIPPAGE_BPS / 10000)
            perm_pnl += (adj_exit - adj_entry) * shares
        if perm_pnl >= actual_pnl:
            count_better += 1

    return round(count_better / n_perm, 4)


def regime_gap(trades, feat):
    """
    Compute |Sharpe_up - Sharpe_down| / max(|Sharpe_up|, |Sharpe_down|).
    Up/down regime based on SPY 20-day return.
    """
    if len(trades) < 10:
        return 1.0

    spy = feat["spy_close"]
    if spy is None:
        return 0.0

    spy_20d_ret = spy.pct_change(20)

    up_returns = []
    down_returns = []
    for t in trades:
        entry_date = pd.Timestamp(t["entry_date"])
        try:
            regime = spy_20d_ret.loc[entry_date]
        except KeyError:
            # Find nearest
            idx = spy_20d_ret.index.get_indexer([entry_date], method="nearest")[0]
            regime = spy_20d_ret.iloc[idx]
        if pd.isna(regime):
            continue
        if regime >= 0:
            up_returns.append(t["return"])
        else:
            down_returns.append(t["return"])

    if not up_returns or not down_returns:
        return 1.0  # can't compute, fail

    sharpe_up = np.mean(up_returns) / (np.std(up_returns, ddof=1) + 1e-9)
    sharpe_down = np.mean(down_returns) / (np.std(down_returns, ddof=1) + 1e-9)
    denom = max(abs(sharpe_up), abs(sharpe_down), 1e-9)
    gap = abs(sharpe_up - sharpe_down) / denom
    return round(float(gap), 4)


def validate_5gates(metrics, perm_p, reg_gap):
    """Apply 5-gate validation."""
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_test_p_lt_0.05": perm_p < 0.05,
        "regime_gap_lt_0.5": reg_gap < 0.5,
        "max_dd_gt_neg50pct": metrics["max_drawdown"] > -0.50,
        "min_20_trades": metrics["total_trades"] >= 20,
    }
    gates["all_passed"] = all(gates.values())
    return gates


# ── MAIN ────────────────────────────────────────────────────────────────────
def main():
    stock_data, vix, vix3m, vix3m_is_etf = download_data()
    feat = build_features(stock_data, vix, vix3m, vix3m_is_etf)

    trade_dates = feat["close"].index[feat["close"].index >= TRADE_START]

    variants = {
        "A_vix_spike": variant_A_signals,
        "B_vix_term_inversion": variant_B_signals,
        "C_rvol_compression": variant_C_signals,
        "D_cross_stock_rsi": variant_D_signals,
        "E_vix_mean_reversion": variant_E_signals,
        "F_iv_rv_gap": variant_F_signals,
    }

    results = {}
    for name, signal_fn in variants.items():
        print(f"\n{'='*60}")
        print(f"  Variant {name}")
        print(f"{'='*60}")

        signals = signal_fn(feat, trade_dates)
        print(f"  Raw signals generated: {len(signals)}")

        trades, equity_curve = run_backtest(signals, feat, name)
        metrics = compute_metrics(trades, equity_curve)

        print(f"  Trades executed: {metrics['total_trades']}")
        print(f"  Total PnL: ${metrics['total_pnl']:.2f}")
        print(f"  Win Rate: {metrics['win_rate']:.1%}")
        print(f"  Sharpe: {metrics['sharpe']:.3f}")
        print(f"  Sortino: {metrics['sortino']:.3f}")
        print(f"  Profit Factor: {metrics['profit_factor']:.3f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"  Final Equity: ${metrics['final_equity']:.2f}")

        # Permutation test
        print(f"  Running permutation test ({N_PERM} shuffles)...")
        perm_p = permutation_test(trades, feat) if metrics["total_trades"] >= 5 else 1.0
        print(f"  Permutation p-value: {perm_p:.4f}")

        # Regime gap
        reg_gap = regime_gap(trades, feat) if metrics["total_trades"] >= 10 else 1.0
        print(f"  Regime gap: {reg_gap:.4f}")

        # 5-gate validation
        gates = validate_5gates(metrics, perm_p, reg_gap)
        print(f"  5-Gate Results:")
        for g, v in gates.items():
            status = "PASS" if v else "FAIL"
            print(f"    {g}: {status}")

        results[name] = {
            "metrics": metrics,
            "permutation_p_value": perm_p,
            "regime_gap": reg_gap,
            "gates": gates,
            "sample_trades": trades[:5] if trades else [],
            "trade_count_by_ticker": {},
        }

        # Trade distribution by ticker
        if trades:
            from collections import Counter
            ticker_counts = Counter(t["ticker"] for t in trades)
            results[name]["trade_count_by_ticker"] = dict(ticker_counts)

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    passing = []
    for name, r in results.items():
        passed = r["gates"]["all_passed"]
        status = "ALL GATES PASSED" if passed else "FAILED"
        print(f"  {name}: {status} | Sharpe={r['metrics']['sharpe']:.3f} | "
              f"PnL=${r['metrics']['total_pnl']:.2f} | Trades={r['metrics']['total_trades']} | "
              f"WR={r['metrics']['win_rate']:.1%}")
        if passed:
            passing.append(name)

    if passing:
        print(f"\n  Variants passing all 5 gates: {', '.join(passing)}")
    else:
        print(f"\n  No variant passed all 5 gates.")

    # Save results
    output = {
        "strategy": "Volatility Regime-Based Entry on Quality Stocks",
        "period": f"{TRADE_START} to {END}",
        "capital": CAPITAL,
        "max_per_trade": MAX_PER_TRADE,
        "max_concurrent": MAX_CONCURRENT,
        "hold_days": HOLD_DAYS,
        "slippage_bps": SLIPPAGE_BPS,
        "universe": UNIVERSE,
        "run_timestamp": datetime.now().isoformat(),
        "variants": results,
        "passing_variants": passing,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
