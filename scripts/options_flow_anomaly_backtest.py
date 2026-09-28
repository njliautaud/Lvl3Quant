#!/usr/bin/env python3
"""
Options Flow Anomaly Backtest
Tests 6 variants using volume/vol/RSI proxies for options flow signals.
OOT: Jan 2022 - Jul 2026 | Starting Capital: $645
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
UNIVERSE = ["AAPL","MSFT","GOOGL","META","AMZN","NVDA","TSLA","AMD","NFLX","CRM",
            "AVGO","ADBE","COST","SPY","QQQ"]
START = "2021-06-01"   # need lookback before OOT
OOT_START = "2022-01-01"
OOT_END = "2026-07-31"
CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02% each way
MAX_POS_VALUE = 200.0
MAX_CONCURRENT = 3
PERMUTATION_ITERS = 1000
RSI_PERIOD = 14

np.random.seed(42)


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def download_data():
    """Download price/volume data for universe + SPY for regime."""
    tickers = list(set(UNIVERSE + ["SPY"]))
    print(f"Downloading data for {len(tickers)} tickers...")
    data = yf.download(tickers, start=START, end=OOT_END, auto_adjust=True, progress=False)
    # Handle multi-level columns from yfinance
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
    volume = data["Volume"] if isinstance(data.columns, pd.MultiIndex) else data[["Volume"]]
    high = data["High"] if isinstance(data.columns, pd.MultiIndex) else data[["High"]]
    low = data["Low"] if isinstance(data.columns, pd.MultiIndex) else data[["Low"]]
    opn = data["Open"] if isinstance(data.columns, pd.MultiIndex) else data[["Open"]]
    return close, volume, high, low, opn


def compute_features(close, volume):
    """Compute all features needed for signal generation."""
    features = {}
    for ticker in UNIVERSE:
        if ticker not in close.columns:
            continue
        c = close[ticker].dropna()
        v = volume[ticker].dropna()
        idx = c.index.intersection(v.index)
        c, v = c.loc[idx], v.loc[idx]

        df = pd.DataFrame(index=idx)
        df["close"] = c
        df["volume"] = v
        df["ret"] = c.pct_change()
        df["vol_avg_20"] = v.rolling(20).mean()
        df["vol_avg_5"] = v.rolling(5).mean()
        df["vol_ratio"] = v / df["vol_avg_20"]
        df["vol_ratio_5d"] = df["vol_avg_5"] / df["vol_avg_20"]
        df["rsi"] = compute_rsi(c, RSI_PERIOD)
        df["sma_20"] = c.rolling(20).mean()
        df["realized_vol_20"] = df["ret"].rolling(20).std()
        df["realized_vol_60"] = df["ret"].rolling(60).std()
        df["vol_compression"] = df["realized_vol_20"] / df["realized_vol_60"].replace(0, np.nan)
        df["high_52w"] = c.rolling(252).max()
        df["pct_from_52w_high"] = (c - df["high_52w"]) / df["high_52w"]
        df["pct_from_sma20"] = (c - df["sma_20"]) / df["sma_20"]
        features[ticker] = df.dropna()
    return features


def get_spy_regime(close):
    """Bull = SPY > 200-SMA, Bear = SPY < 200-SMA."""
    spy = close["SPY"].dropna()
    sma200 = spy.rolling(200).mean()
    regime = (spy > sma200).astype(int)  # 1=bull, 0=bear
    return regime


def generate_signals(features, variant):
    """Generate entry signals for a given variant. Returns list of (date, ticker)."""
    signals = []
    for ticker, df in features.items():
        oot_mask = (df.index >= OOT_START) & (df.index <= OOT_END)
        df_oot = df[oot_mask]
        for i, (date, row) in enumerate(df_oot.iterrows()):
            if variant == "A":
                # Volume > 2x 20-day avg AND RSI < 30
                if row["vol_ratio"] > 2.0 and row["rsi"] < 30:
                    signals.append((date, ticker))
            elif variant == "B":
                # Vol compression: 20-day vol < 0.6 * 60-day vol
                if row["vol_compression"] < 0.6:
                    signals.append((date, ticker))
            elif variant == "C":
                # Drop > 3% on volume > 3x average
                if row["ret"] < -0.03 and row["vol_ratio"] > 3.0:
                    signals.append((date, ticker))
            elif variant == "D":
                # 5-day avg volume > 2x 20-day avg AND >5% below SMA20
                if row["vol_ratio_5d"] > 2.0 and row["pct_from_sma20"] < -0.05:
                    signals.append((date, ticker))
            elif variant == "E":
                # Monthly rebalance handled separately
                pass
            elif variant == "F":
                # A conditions + within 15% of 52-week high
                if (row["vol_ratio"] > 2.0 and row["rsi"] < 30 and
                        row["pct_from_52w_high"] > -0.15):
                    signals.append((date, ticker))
    return signals


def generate_signals_variant_e(features):
    """Monthly rebalance: buy 3 stocks with highest vol-to-avg ratio that also have RSI < 40."""
    signals = []
    # Get all dates in OOT
    all_dates = set()
    for df in features.values():
        all_dates.update(df.index)
    all_dates = sorted([d for d in all_dates if OOT_START <= str(d) <= OOT_END])

    # Monthly rebalance dates (first trading day of each month)
    rebal_dates = []
    current_month = None
    for d in all_dates:
        ym = (d.year, d.month)
        if ym != current_month:
            rebal_dates.append(d)
            current_month = ym

    for rebal_date in rebal_dates:
        candidates = []
        for ticker, df in features.items():
            if rebal_date in df.index:
                row = df.loc[rebal_date]
                if row["rsi"] < 40 and row["vol_ratio"] > 1.0:
                    candidates.append((ticker, row["vol_ratio"]))
        # Pick top 3 by vol ratio
        candidates.sort(key=lambda x: x[1], reverse=True)
        for ticker, _ in candidates[:3]:
            signals.append((rebal_date, ticker))
    return signals


def simulate_trades(signals, close, hold_days, spy_regime, capital=CAPITAL):
    """Simulate trades with position sizing, slippage, and max concurrent positions."""
    if not signals:
        return [], capital

    trades = []
    equity = capital
    open_positions = []  # list of (exit_date, ticker, shares, entry_price)

    # Sort signals by date
    signals = sorted(signals, key=lambda x: x[0])

    for entry_date, ticker in signals:
        # Close expired positions
        new_open = []
        for pos in open_positions:
            if entry_date >= pos[0]:
                # Position expired, close it
                exit_date_actual = pos[0]
                t = pos[1]
                shares = pos[2]
                ep = pos[3]
                regime_at_entry = pos[4]
                if exit_date_actual in close.index and t in close.columns:
                    exit_price = close.loc[exit_date_actual, t]
                    if pd.notna(exit_price):
                        exit_price_slip = exit_price * (1 - SLIPPAGE_PCT)
                        pnl = (exit_price_slip - ep) * shares
                        equity += pnl
                        ret = (exit_price_slip / ep) - 1
                        trades.append({
                            "entry_date": str(pos[5]),
                            "exit_date": str(exit_date_actual),
                            "ticker": t,
                            "entry_price": round(ep, 4),
                            "exit_price": round(exit_price_slip, 4),
                            "shares": shares,
                            "pnl": round(pnl, 4),
                            "return": round(ret, 6),
                            "regime": "bull" if regime_at_entry == 1 else "bear"
                        })
                    else:
                        new_open.append(pos)
                        continue
                else:
                    # Can't close yet, keep
                    new_open.append(pos)
                    continue
            else:
                new_open.append(pos)
        open_positions = new_open

        # Check concurrent position limit
        if len(open_positions) >= MAX_CONCURRENT:
            continue

        # Check if already holding this ticker
        if any(p[1] == ticker for p in open_positions):
            continue

        # Entry
        if entry_date not in close.index or ticker not in close.columns:
            continue
        entry_price = close.loc[entry_date, ticker]
        if pd.isna(entry_price) or entry_price <= 0:
            continue

        entry_price_slip = entry_price * (1 + SLIPPAGE_PCT)
        pos_size = min(MAX_POS_VALUE, equity * 0.33)
        if pos_size <= 0:
            continue
        shares = int(pos_size / entry_price_slip)
        if shares < 1:
            continue

        # Determine exit date
        future_dates = close.index[close.index > entry_date]
        if len(future_dates) < hold_days:
            exit_date = future_dates[-1] if len(future_dates) > 0 else entry_date
        else:
            exit_date = future_dates[hold_days - 1]

        # Regime
        regime_val = 1  # default bull
        if entry_date in spy_regime.index:
            regime_val = spy_regime.loc[entry_date]

        open_positions.append((exit_date, ticker, shares, entry_price_slip, regime_val, entry_date))

    # Close any remaining open positions at last available date
    last_date = close.index[-1]
    for pos in open_positions:
        t = pos[1]
        shares = pos[2]
        ep = pos[3]
        regime_val = pos[4]
        exit_date = min(pos[0], last_date)
        if exit_date in close.index and t in close.columns:
            exit_price = close.loc[exit_date, t]
            if pd.notna(exit_price):
                exit_price_slip = exit_price * (1 - SLIPPAGE_PCT)
                pnl = (exit_price_slip - ep) * shares
                equity += pnl
                ret = (exit_price_slip / ep) - 1
                trades.append({
                    "entry_date": str(pos[5]),
                    "exit_date": str(exit_date),
                    "ticker": t,
                    "entry_price": round(ep, 4),
                    "exit_price": round(exit_price_slip, 4),
                    "shares": shares,
                    "pnl": round(pnl, 4),
                    "return": round(ret, 6),
                    "regime": "bull" if regime_val == 1 else "bear"
                })

    final_equity = equity
    return trades, final_equity


def compute_metrics(trades, capital=CAPITAL):
    """Compute Sharpe, Sortino, MaxDD, WR, PF, etc."""
    if not trades:
        return {
            "n_trades": 0, "sharpe": 0, "sortino": 0, "pf": 0, "wr": 0,
            "max_dd_pct": 0, "total_return_pct": 0, "final_equity": capital,
            "avg_return": 0, "sharpe_bull": 0, "sharpe_bear": 0, "regime_gap": 0,
            "n_bull": 0, "n_bear": 0
        }

    returns = np.array([t["return"] for t in trades])
    pnls = np.array([t["pnl"] for t in trades])
    n = len(trades)
    wr = np.mean(returns > 0)
    gross_profit = pnls[pnls > 0].sum() if (pnls > 0).any() else 0
    gross_loss = abs(pnls[pnls < 0].sum()) if (pnls < 0).any() else 1e-9
    pf = gross_profit / gross_loss

    # Equity curve for MaxDD
    equity_curve = [capital]
    for pnl in pnls:
        equity_curve.append(equity_curve[-1] + pnl)
    equity_curve = np.array(equity_curve)
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak
    max_dd = dd.min()

    # Annualize: assume avg hold ~10 days, so ~25 trades/year
    avg_ret = returns.mean()
    std_ret = returns.std() if returns.std() > 0 else 1e-9
    downside = returns[returns < 0]
    downside_std = downside.std() if len(downside) > 1 and downside.std() > 0 else 1e-9

    # Sharpe/Sortino annualized (scale by sqrt of trades per year)
    trades_per_year = max(n / 4.5, 1)  # ~4.5 year OOT
    sharpe = (avg_ret / std_ret) * np.sqrt(trades_per_year)
    sortino = (avg_ret / downside_std) * np.sqrt(trades_per_year)

    total_return = (equity_curve[-1] / capital - 1) * 100

    # Regime analysis
    bull_rets = [t["return"] for t in trades if t["regime"] == "bull"]
    bear_rets = [t["return"] for t in trades if t["regime"] == "bear"]

    def regime_sharpe(rets):
        if len(rets) < 2:
            return 0
        r = np.array(rets)
        s = r.std()
        if s == 0:
            return 0
        return r.mean() / s * np.sqrt(len(r))

    sharpe_bull = regime_sharpe(bull_rets)
    sharpe_bear = regime_sharpe(bear_rets)
    denom = max(abs(sharpe_bull), abs(sharpe_bear), 1e-9)
    regime_gap = abs(sharpe_bull - sharpe_bear) / denom

    return {
        "n_trades": n,
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "pf": round(pf, 4),
        "wr": round(wr, 4),
        "max_dd_pct": round(max_dd * 100, 2),
        "total_return_pct": round(total_return, 2),
        "final_equity": round(equity_curve[-1], 2),
        "avg_return": round(avg_ret * 100, 4),
        "sharpe_bull": round(sharpe_bull, 4),
        "sharpe_bear": round(sharpe_bear, 4),
        "regime_gap": round(regime_gap, 4),
        "n_bull": len(bull_rets),
        "n_bear": len(bear_rets)
    }


def permutation_test(trades, close, hold_days, spy_regime, features, variant, n_iters=PERMUTATION_ITERS):
    """Shuffle entry dates to test if strategy Sharpe is significantly better than random."""
    if len(trades) < 5:
        return 1.0

    actual_sharpe = compute_metrics(trades)["sharpe"]

    # Get all valid OOT dates
    oot_dates = close.index[(close.index >= OOT_START) & (close.index <= OOT_END)]
    tickers_with_data = [t for t in UNIVERSE if t in close.columns]

    count_better = 0
    for _ in range(n_iters):
        # Random signals: same number of trades, random dates and tickers
        n_signals = len(trades)
        rand_dates = np.random.choice(oot_dates, size=n_signals, replace=True)
        rand_tickers = np.random.choice(tickers_with_data, size=n_signals, replace=True)
        rand_signals = list(zip(rand_dates, rand_tickers))
        rand_trades, _ = simulate_trades(rand_signals, close, hold_days, spy_regime)
        if len(rand_trades) > 0:
            rand_sharpe = compute_metrics(rand_trades)["sharpe"]
            if rand_sharpe >= actual_sharpe:
                count_better += 1

    p_value = (count_better + 1) / (n_iters + 1)
    return round(p_value, 4)


def run_variant(variant, features, close, spy_regime, hold_days):
    """Run a single variant end-to-end."""
    print(f"\n{'='*60}")
    print(f"  VARIANT {variant} (hold={hold_days}d)")
    print(f"{'='*60}")

    if variant == "E":
        signals = generate_signals_variant_e(features)
        hold_days = 21  # ~1 month
    else:
        signals = generate_signals(features, variant)

    print(f"  Raw signals: {len(signals)}")

    trades, final_eq = simulate_trades(signals, close, hold_days, spy_regime)
    metrics = compute_metrics(trades)
    metrics["final_equity"] = round(final_eq, 2)

    print(f"  Trades: {metrics['n_trades']}")
    print(f"  Sharpe: {metrics['sharpe']}")
    print(f"  Sortino: {metrics['sortino']}")
    print(f"  WR: {metrics['wr']:.1%}")
    print(f"  PF: {metrics['pf']:.2f}")
    print(f"  MaxDD: {metrics['max_dd_pct']:.1f}%")
    print(f"  Total Return: {metrics['total_return_pct']:.1f}%")
    print(f"  Final Equity: ${metrics['final_equity']:.2f}")
    print(f"  Bull/Bear trades: {metrics['n_bull']}/{metrics['n_bear']}")
    print(f"  Sharpe Bull/Bear: {metrics['sharpe_bull']}/{metrics['sharpe_bear']}")
    print(f"  Regime Gap: {metrics['regime_gap']:.3f}")

    # Permutation test
    print(f"  Running permutation test ({PERMUTATION_ITERS} iters)...")
    p_val = permutation_test(trades, close, hold_days, spy_regime, features, variant)
    metrics["perm_p_value"] = p_val
    print(f"  Permutation p-value: {p_val}")

    # 5-Gate validation
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": p_val < 0.05,
        "regime_gap_lt_0.5": metrics["regime_gap"] < 0.5,
        "maxdd_gt_neg50": metrics["max_dd_pct"] > -50,
        "min_20_trades": metrics["n_trades"] >= 20
    }
    metrics["gates"] = gates
    metrics["gates_passed"] = sum(gates.values())
    metrics["all_gates_passed"] = all(gates.values())

    print(f"  Gates: {metrics['gates_passed']}/5 passed")
    for g, v in gates.items():
        status = "PASS" if v else "FAIL"
        print(f"    {status}: {g}")

    return metrics


def main():
    print("Options Flow Anomaly Backtest")
    print(f"OOT: {OOT_START} to {OOT_END} | Capital: ${CAPITAL}")
    print("=" * 60)

    close, volume, high, low, opn = download_data()
    features = compute_features(close, volume)
    spy_regime = get_spy_regime(close)

    print(f"\nData loaded. {len(features)} tickers with features.")
    for t, df in features.items():
        oot_rows = ((df.index >= OOT_START) & (df.index <= OOT_END)).sum()
        print(f"  {t}: {len(df)} total days, {oot_rows} in OOT")

    variants = {
        "A": 10, "B": 15, "C": 5, "D": 10, "E": 21, "F": 10
    }

    results = {}
    for v, hold in variants.items():
        results[v] = run_variant(v, features, close, spy_regime, hold)

    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    header = f"{'Var':>3} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} {'PF':>6} {'MaxDD':>7} {'Return':>8} {'Final$':>8} {'RegGap':>7} {'Perm-p':>7} {'Gates':>5} {'Pass':>4}"
    print(header)
    print("-" * 100)

    for v in ["A","B","C","D","E","F"]:
        m = results[v]
        passed = "YES" if m["all_gates_passed"] else "NO"
        print(f"  {v} {m['n_trades']:>6} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{m['wr']:>5.1%} {m['pf']:>6.2f} {m['max_dd_pct']:>6.1f}% "
              f"{m['total_return_pct']:>7.1f}% ${m['final_equity']:>7.2f} "
              f"{m['regime_gap']:>7.3f} {m['perm_p_value']:>7.4f} "
              f"{m['gates_passed']:>3}/5 {passed:>4}")

    # Save results
    output = {
        "metadata": {
            "strategy": "Options Flow Anomaly",
            "oot_period": f"{OOT_START} to {OOT_END}",
            "starting_capital": CAPITAL,
            "slippage_pct": SLIPPAGE_PCT,
            "max_position_value": MAX_POS_VALUE,
            "max_concurrent_positions": MAX_CONCURRENT,
            "universe": UNIVERSE,
            "permutation_iterations": PERMUTATION_ITERS,
            "run_timestamp": datetime.now().isoformat()
        },
        "variants": {}
    }
    for v in ["A","B","C","D","E","F"]:
        output["variants"][v] = results[v]

    out_path = "/home/jupiter/Lvl3Quant/data/options_flow_anomaly_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
