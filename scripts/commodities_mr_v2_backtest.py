#!/usr/bin/env python3
"""
Commodities Mean Reversion v2 Backtest
Dual Signal D logic (dip-buy + recovery confirmation) on commodity ETFs.
5-Gate Validation with permutation tests.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# ─── Configuration ───────────────────────────────────────────────────────────
START = "2022-01-01"
END = "2026-07-31"
CAPITAL = 645.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
HOLD_DAYS = 10
SLIPPAGE_BPS = 2
DIP_PCT = 0.05        # 5% from 20-day high
RSI_THRESH = 35
RSI_PERIOD = 14
HIGH_LOOKBACK = 20
RED_DAYS_MIN = 3
PERM_N = 1000

VARIANTS = {
    "A_all_commodities":   ["GLD", "SLV", "USO", "UNG", "DBA", "CPER", "PDBC", "WEAT"],
    "B_precious_metals":   ["GLD", "SLV", "PPLT"],
    "C_energy":            ["USO", "UNG", "XLE"],
    "D_broad_basket":      ["DJP", "GSG", "PDBC", "DBC"],
    "E_agricultural":      ["DBA", "WEAT", "CORN", "SOYB"],
    "F_industrial_metals": ["CPER", "DBB", "COPX"],
}


# ─── Helpers ─────────────────────────────────────────────────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def download_data(tickers):
    """Download daily OHLCV for all tickers + SPY."""
    all_tickers = list(set(tickers + ["SPY"]))
    print(f"  Downloading: {all_tickers}")
    data = {}
    for t in all_tickers:
        try:
            df = yf.download(t, start=START, end=END, auto_adjust=True, progress=False)
            if df is not None and len(df) > 50:
                # Flatten multi-index columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[t] = df
                print(f"    {t}: {len(df)} bars")
            else:
                print(f"    {t}: insufficient data, skipping")
        except Exception as e:
            print(f"    {t}: download failed ({e})")
    return data


def generate_signals(price_df):
    """Generate entry signals using Dual Signal D logic."""
    close = price_df["Close"].copy()
    rsi = compute_rsi(close, RSI_PERIOD)
    high_20 = close.rolling(HIGH_LOOKBACK).max()
    dip = (close <= high_20 * (1 - DIP_PCT))

    # Consecutive red days: close < prev close
    red = (close < close.shift(1)).astype(int)
    consec_red = red.copy() * 0
    for i in range(1, len(red)):
        if red.iloc[i] == 1:
            consec_red.iloc[i] = consec_red.iloc[i-1] + 1
        else:
            consec_red.iloc[i] = 0

    # Green day after 3+ red
    green_day = close > close.shift(1)
    prev_consec_red = consec_red.shift(1)
    recovery = green_day & (prev_consec_red >= RED_DAYS_MIN)

    signals = dip & (rsi < RSI_THRESH) & recovery
    return signals


def run_backtest(tickers, data):
    """Run the mean reversion backtest on given tickers."""
    spy_close = data.get("SPY")
    if spy_close is None:
        return None
    spy_c = spy_close["Close"]
    spy_sma200 = spy_c.rolling(200).mean()

    trades = []
    active_positions = []  # list of (ticker, entry_date, entry_price, shares, exit_date_target)

    # Build a unified date index
    all_dates = set()
    valid_tickers = []
    for t in tickers:
        if t in data:
            valid_tickers.append(t)
            all_dates.update(data[t].index.tolist())
    if not valid_tickers:
        return None

    all_dates = sorted(all_dates)

    # Pre-compute signals per ticker
    ticker_signals = {}
    for t in valid_tickers:
        ticker_signals[t] = generate_signals(data[t])

    # Track equity curve
    cash = CAPITAL
    equity_curve = []

    for date in all_dates:
        # Check exits
        new_active = []
        for pos in active_positions:
            t, entry_date, entry_price, shares, exit_target = pos
            if date >= exit_target:
                # Exit
                if date in data[t].index:
                    exit_price = data[t].loc[date, "Close"]
                else:
                    # Find nearest date
                    idx = data[t].index
                    future = idx[idx >= exit_target]
                    if len(future) == 0:
                        new_active.append(pos)
                        continue
                    exit_date_actual = future[0]
                    exit_price = data[t].loc[exit_date_actual, "Close"]
                    date_for_exit = exit_date_actual

                slip = exit_price * SLIPPAGE_BPS / 10000
                net_exit = exit_price - slip
                pnl = (net_exit - entry_price) * shares
                ret = (net_exit / entry_price) - 1
                cash += shares * net_exit

                # Determine regime at entry
                regime = "unknown"
                if entry_date in spy_c.index and entry_date in spy_sma200.index:
                    if pd.notna(spy_sma200.loc[entry_date]):
                        regime = "bull" if spy_c.loc[entry_date] > spy_sma200.loc[entry_date] else "bear"

                trades.append({
                    "ticker": t,
                    "entry_date": str(entry_date.date()) if hasattr(entry_date, 'date') else str(entry_date),
                    "exit_date": str(date.date()) if hasattr(date, 'date') else str(date),
                    "entry_price": float(entry_price),
                    "exit_price": float(net_exit),
                    "shares": float(shares),
                    "pnl": float(pnl),
                    "return": float(ret),
                    "regime": regime,
                })
            else:
                new_active.append(pos)
        active_positions = new_active

        # Check entries
        if len(active_positions) < MAX_CONCURRENT:
            for t in valid_tickers:
                if len(active_positions) >= MAX_CONCURRENT:
                    break
                if date not in data[t].index:
                    continue
                if date not in ticker_signals[t].index:
                    continue
                if not ticker_signals[t].loc[date]:
                    continue
                # Already have position in this ticker?
                if any(p[0] == t for p in active_positions):
                    continue

                entry_price = data[t].loc[date, "Close"]
                slip = entry_price * SLIPPAGE_BPS / 10000
                entry_cost = entry_price + slip
                shares = min(MAX_PER_TRADE, cash) / entry_cost
                if shares * entry_cost < 10:  # min trade size
                    continue

                cost = shares * entry_cost
                cash -= cost

                # Target exit date
                future_dates = data[t].index[data[t].index > date]
                if len(future_dates) >= HOLD_DAYS:
                    exit_target = future_dates[HOLD_DAYS - 1]
                elif len(future_dates) > 0:
                    exit_target = future_dates[-1]
                else:
                    continue

                active_positions.append((t, date, entry_cost, shares, exit_target))

        # Equity
        port_val = cash
        for pos in active_positions:
            t, _, _, shares, _ = pos
            if date in data[t].index:
                port_val += shares * data[t].loc[date, "Close"]
        equity_curve.append((date, port_val))

    # Force-close remaining positions at last available price
    for pos in active_positions:
        t, entry_date, entry_price, shares, _ = pos
        if t in data and len(data[t]) > 0:
            last_date = data[t].index[-1]
            exit_price = data[t].loc[last_date, "Close"]
            slip = exit_price * SLIPPAGE_BPS / 10000
            net_exit = exit_price - slip
            pnl = (net_exit - entry_price) * shares
            ret = (net_exit / entry_price) - 1
            regime = "unknown"
            if entry_date in spy_c.index and entry_date in spy_sma200.index:
                if pd.notna(spy_sma200.loc[entry_date]):
                    regime = "bull" if spy_c.loc[entry_date] > spy_sma200.loc[entry_date] else "bear"
            trades.append({
                "ticker": t,
                "entry_date": str(entry_date.date()) if hasattr(entry_date, 'date') else str(entry_date),
                "exit_date": str(last_date.date()) if hasattr(last_date, 'date') else str(last_date),
                "entry_price": float(entry_price),
                "exit_price": float(net_exit),
                "shares": float(shares),
                "pnl": float(pnl),
                "return": float(ret),
                "regime": regime,
            })

    return trades, equity_curve


def compute_metrics(trades, equity_curve):
    """Compute performance metrics from trades."""
    if not trades:
        return None

    returns = np.array([t["return"] for t in trades])
    pnls = np.array([t["pnl"] for t in trades])
    n = len(trades)

    total_pnl = float(np.sum(pnls))
    total_return = total_pnl / CAPITAL

    # Win rate
    wins = np.sum(returns > 0)
    win_rate = float(wins / n) if n > 0 else 0

    # Profit factor
    gross_profit = float(np.sum(pnls[pnls > 0])) if np.any(pnls > 0) else 0
    gross_loss = float(np.abs(np.sum(pnls[pnls < 0]))) if np.any(pnls < 0) else 0.001
    profit_factor = gross_profit / gross_loss

    # Annualized Sharpe (assuming ~25 trades/year avg hold 10 days)
    if len(returns) > 1 and np.std(returns) > 0:
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        # Annualize: each trade is ~10 trading days, ~25 trades/year
        trades_per_year = 252 / HOLD_DAYS
        sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    # Sortino
    downside = returns[returns < 0]
    if len(downside) > 1:
        downside_std = np.std(downside)
        if downside_std > 0:
            trades_per_year = 252 / HOLD_DAYS
            sortino = (np.mean(returns) / downside_std) * np.sqrt(trades_per_year)
        else:
            sortino = float('inf')
    else:
        sortino = float('inf')

    # Max drawdown from equity curve
    if equity_curve:
        eq = np.array([e[1] for e in equity_curve])
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        max_dd = float(np.min(dd))
    else:
        max_dd = 0.0

    # Regime breakdown
    bull_rets = [t["return"] for t in trades if t["regime"] == "bull"]
    bear_rets = [t["return"] for t in trades if t["regime"] == "bear"]

    def regime_sharpe(rets):
        if len(rets) < 2:
            return 0.0
        r = np.array(rets)
        if np.std(r) == 0:
            return 0.0
        trades_per_year = 252 / HOLD_DAYS
        return float((np.mean(r) / np.std(r)) * np.sqrt(trades_per_year))

    bull_sharpe = regime_sharpe(bull_rets)
    bear_sharpe = regime_sharpe(bear_rets)

    max_abs = max(abs(bull_sharpe), abs(bear_sharpe), 0.001)
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs

    # Per-ticker breakdown
    ticker_stats = {}
    for t in set(tr["ticker"] for tr in trades):
        t_trades = [tr for tr in trades if tr["ticker"] == t]
        t_rets = np.array([tr["return"] for tr in t_trades])
        t_pnls = np.array([tr["pnl"] for tr in t_trades])
        t_wins = np.sum(t_rets > 0)
        ticker_stats[t] = {
            "num_trades": len(t_trades),
            "win_rate": float(t_wins / len(t_trades)) if len(t_trades) > 0 else 0,
            "total_pnl": float(np.sum(t_pnls)),
            "avg_return": float(np.mean(t_rets)),
        }

    return {
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "win_rate": round(win_rate, 3),
        "profit_factor": round(profit_factor, 3),
        "max_drawdown": round(max_dd, 3),
        "total_return": round(total_return, 3),
        "total_pnl": round(total_pnl, 2),
        "num_trades": n,
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "bull_trades": len(bull_rets),
        "bear_trades": len(bear_rets),
        "regime_gap": round(regime_gap, 3),
        "per_ticker": ticker_stats,
    }


def permutation_test(trades, observed_sharpe, n_perms=PERM_N):
    """Shuffle entry dates randomly, recompute Sharpe each time, get p-value."""
    if len(trades) < 5:
        return 1.0

    returns = np.array([t["return"] for t in trades])
    n = len(returns)
    trades_per_year = 252 / HOLD_DAYS

    count_better = 0
    for _ in range(n_perms):
        # Shuffle returns to break any temporal structure
        perm_rets = np.random.permutation(returns)
        # Randomly flip signs to simulate random entries
        signs = np.random.choice([-1, 1], size=n)
        perm_rets = perm_rets * signs
        std = np.std(perm_rets)
        if std > 0:
            perm_sharpe = (np.mean(perm_rets) / std) * np.sqrt(trades_per_year)
        else:
            perm_sharpe = 0
        if perm_sharpe >= observed_sharpe:
            count_better += 1

    return float(count_better / n_perms)


def five_gate_check(metrics, perm_p):
    """Apply 5-gate validation."""
    gates = {
        "gate1_sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "gate2_perm_p_lt_0.05": perm_p < 0.05,
        "gate3_regime_gap_lt_0.5": metrics["regime_gap"] < 0.5,
        "gate4_maxdd_gt_neg50": metrics["max_drawdown"] > -0.50,
        "gate5_min_20_trades": metrics["num_trades"] >= 20,
    }
    gates["all_pass"] = all(gates.values())
    return gates


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("COMMODITIES MEAN REVERSION v2 BACKTEST")
    print(f"Period: {START} to {END}")
    print(f"Capital: ${CAPITAL}, Max/trade: ${MAX_PER_TRADE}, Max concurrent: {MAX_CONCURRENT}")
    print("=" * 70)

    # Collect all unique tickers
    all_tickers = list(set(t for v in VARIANTS.values() for t in v))
    print(f"\nDownloading data for {len(all_tickers)} unique tickers + SPY...")
    data = download_data(all_tickers)

    results = {}

    for variant_name, tickers in VARIANTS.items():
        print(f"\n{'─' * 60}")
        print(f"VARIANT: {variant_name}")
        print(f"Tickers: {tickers}")

        available = [t for t in tickers if t in data]
        if not available:
            print(f"  No data available, skipping")
            results[variant_name] = {"error": "no data available"}
            continue

        print(f"  Available: {available}")

        result = run_backtest(available, data)
        if result is None:
            print(f"  Backtest returned no results")
            results[variant_name] = {"error": "backtest failed"}
            continue

        trades, equity_curve = result

        if not trades:
            print(f"  No trades generated")
            results[variant_name] = {"error": "no trades", "num_trades": 0}
            continue

        metrics = compute_metrics(trades, equity_curve)
        if metrics is None:
            results[variant_name] = {"error": "metrics computation failed"}
            continue

        # Permutation test
        print(f"  Running permutation test ({PERM_N} iterations)...")
        perm_p = permutation_test(trades, metrics["sharpe"])
        metrics["permutation_p"] = round(perm_p, 4)

        # 5-gate validation
        gates = five_gate_check(metrics, perm_p)
        metrics["five_gate"] = gates

        results[variant_name] = metrics

        # Print summary
        status = "PASS" if gates["all_pass"] else "FAIL"
        print(f"\n  [{status}] {variant_name}")
        print(f"  Trades: {metrics['num_trades']} | Sharpe: {metrics['sharpe']} | Sortino: {metrics['sortino']}")
        print(f"  WR: {metrics['win_rate']:.1%} | PF: {metrics['profit_factor']} | MaxDD: {metrics['max_drawdown']:.1%}")
        print(f"  Total Return: {metrics['total_return']:.1%} (${metrics['total_pnl']:.2f})")
        print(f"  Bull Sharpe: {metrics['bull_sharpe']} ({metrics['bull_trades']} trades)")
        print(f"  Bear Sharpe: {metrics['bear_sharpe']} ({metrics['bear_trades']} trades)")
        print(f"  Regime Gap: {metrics['regime_gap']:.3f}")
        print(f"  Permutation p-value: {perm_p:.4f}")
        print(f"  Gates: {gates}")

    # Save results
    output_path = Path("/home/jupiter/Lvl3Quant/data/commodities_mr_v2_results.json")
    output = {
        "metadata": {
            "strategy": "Commodities Mean Reversion v2 (Dual Signal D)",
            "period": f"{START} to {END}",
            "capital": CAPITAL,
            "max_per_trade": MAX_PER_TRADE,
            "max_concurrent": MAX_CONCURRENT,
            "hold_days": HOLD_DAYS,
            "slippage_bps": SLIPPAGE_BPS,
            "entry_conditions": {
                "dip_pct": DIP_PCT,
                "rsi_threshold": RSI_THRESH,
                "rsi_period": RSI_PERIOD,
                "high_lookback": HIGH_LOOKBACK,
                "min_red_days": RED_DAYS_MIN,
            },
            "permutation_iterations": PERM_N,
            "run_timestamp": datetime.now().isoformat(),
        },
        "variants": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"Results saved to {output_path}")

    # Summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'Variant':<25} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} {'PF':>6} {'MaxDD':>7} {'Perm-p':>7} {'Pass':>5}")
    print("-" * 85)
    for v, m in results.items():
        if "error" in m:
            print(f"{v:<25} {'ERROR':>6} — {m.get('error','')}")
            continue
        status = "YES" if m["five_gate"]["all_pass"] else "NO"
        print(f"{v:<25} {m['num_trades']:>6} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['win_rate']:>5.1%} {m['profit_factor']:>6.2f} {m['max_drawdown']:>6.1%} {m['permutation_p']:>7.4f} {status:>5}")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    np.random.seed(42)
    main()
