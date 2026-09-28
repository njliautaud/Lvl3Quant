#!/usr/bin/env python3
"""
Stock Split Effect Backtest
===========================
Event-driven strategy exploiting post-split outperformance.
Uses yfinance split data. Walk-forward OOT: Jan 2022 - Jul 2026.
5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades.

6 Variants:
A) Post-Split Drift (20d hold)
B) Post-Split Extended (60d hold)
C) Split Sector Effect (buy sector ETF 20d)
D) Pre-Split Run (buy 10d before, sell on split)
E) Split + Momentum (post-split if positive 20d momentum, hold 40d)
F) Small Split Bias (inverse price weighting, hold 30d)
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

# ─── Configuration ───────────────────────────────────────────────────────────
INITIAL_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
COMMISSION = 0.0
MAX_CONCURRENT = 3

OOT_START = pd.Timestamp("2022-01-01")
OOT_END = pd.Timestamp("2026-07-29")
DATA_START = pd.Timestamp("2020-01-01")  # look back for more splits

PERM_ITERATIONS = 1000

UNIVERSE = [
    # Large/mega caps
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "NFLX", "CRM",
    "SHOP", "ABNB", "DDOG", "SQ", "NET", "MELI", "UBER", "COIN",
    # Mid-caps
    "PANW", "LULU", "DECK", "GWW", "BRO", "MOH", "MPWR", "PCAR", "ODFL", "CMG",
    "AVGO", "KLAC", "MCHP", "ANET",
    # ETFs
    "QQQ", "SPY", "XLK",
]

# Sector ETF mapping for variant C
SECTOR_ETF_MAP = {
    "AAPL": "XLK", "MSFT": "XLK", "GOOGL": "XLK", "AMZN": "XLY", "META": "XLK",
    "NVDA": "XLK", "TSLA": "XLY", "AMD": "XLK", "NFLX": "XLY", "CRM": "XLK",
    "SHOP": "XLK", "ABNB": "XLY", "DDOG": "XLK", "SQ": "XLF", "NET": "XLK",
    "MELI": "XLY", "UBER": "XLI", "COIN": "XLF",
    "PANW": "XLK", "LULU": "XLY", "DECK": "XLY", "GWW": "XLI", "BRO": "XLF",
    "MOH": "XLV", "MPWR": "XLK", "PCAR": "XLI", "ODFL": "XLI", "CMG": "XLY",
    "AVGO": "XLK", "KLAC": "XLK", "MCHP": "XLK", "ANET": "XLK",
    "QQQ": "XLK", "SPY": "SPY", "XLK": "XLK",
}

# Additional sector ETFs we might need
EXTRA_ETFS = ["XLY", "XLF", "XLV", "XLI"]


def download_data():
    """Download price and split data for all tickers."""
    all_tickers = list(set(UNIVERSE + EXTRA_ETFS + ["SPY"]))
    print(f"Downloading data for {len(all_tickers)} tickers...")

    prices = {}
    splits_data = []

    for ticker_str in all_tickers:
        try:
            ticker = yf.Ticker(ticker_str)
            # Get price history
            hist = ticker.history(start=DATA_START.strftime("%Y-%m-%d"),
                                  end=(OOT_END + timedelta(days=5)).strftime("%Y-%m-%d"),
                                  auto_adjust=True)
            if len(hist) < 50:
                print(f"  {ticker_str}: insufficient data ({len(hist)} rows)")
                continue

            prices[ticker_str] = hist["Close"]

            # Get splits
            sp = ticker.splits
            if sp is not None and len(sp) > 0:
                for date, ratio in sp.items():
                    if ratio > 1:  # actual forward split
                        split_date = pd.Timestamp(date).tz_localize(None)
                        if split_date >= DATA_START:
                            splits_data.append({
                                "ticker": ticker_str,
                                "date": split_date,
                                "ratio": ratio,
                            })
                            print(f"  {ticker_str}: split {ratio}:1 on {split_date.date()}")
        except Exception as e:
            print(f"  {ticker_str}: error - {e}")

    price_df = pd.DataFrame(prices)
    price_df.index = pd.to_datetime(price_df.index).tz_localize(None)
    price_df = price_df.sort_index()

    splits_df = pd.DataFrame(splits_data)
    if len(splits_df) > 0:
        splits_df = splits_df.sort_values("date").reset_index(drop=True)

    print(f"\nTotal splits found: {len(splits_df)}")
    if len(splits_df) > 0:
        oot_splits = splits_df[splits_df["date"] >= OOT_START]
        print(f"OOT splits (2022+): {len(oot_splits)}")
        print(splits_df.to_string(index=False))

    return price_df, splits_df


def get_spy_regime(price_df):
    """Bull = SPY > 200-SMA, Bear = SPY < 200-SMA."""
    if "SPY" not in price_df.columns:
        return pd.Series(True, index=price_df.index)  # default bull
    spy = price_df["SPY"].dropna()
    sma200 = spy.rolling(200).mean()
    regime = spy > sma200  # True = bull, False = bear
    return regime.reindex(price_df.index, method="ffill")


def find_trading_day(price_df, target_date, ticker, direction="forward"):
    """Find nearest trading day on or after/before target_date with valid price."""
    if ticker not in price_df.columns:
        return None
    valid_dates = price_df[ticker].dropna().index
    if direction == "forward":
        candidates = valid_dates[valid_dates >= target_date]
    else:
        candidates = valid_dates[valid_dates <= target_date]
    if len(candidates) == 0:
        return None
    return candidates[0] if direction == "forward" else candidates[-1]


def offset_trading_days(price_df, date, ticker, n_days):
    """Get date n trading days after (positive) or before (negative) given date."""
    if ticker not in price_df.columns:
        return None
    valid_dates = price_df[ticker].dropna().index
    idx = valid_dates.get_indexer([date], method="nearest")[0]
    target_idx = idx + n_days
    if target_idx < 0 or target_idx >= len(valid_dates):
        return None
    return valid_dates[target_idx]


def run_backtest(trades, price_df, initial_capital=INITIAL_CAPITAL, max_concurrent=MAX_CONCURRENT):
    """
    Run backtest from a list of trade signals.
    Each trade: {"ticker": str, "entry_date": Timestamp, "exit_date": Timestamp, "weight": float}
    Returns equity curve, trade results.
    """
    if len(trades) == 0:
        return pd.Series(dtype=float), []

    # Sort by entry date
    trades = sorted(trades, key=lambda x: x["entry_date"])

    # Build daily equity curve
    all_dates = price_df.index
    min_date = min(t["entry_date"] for t in trades)
    max_date = max(t["exit_date"] for t in trades)
    dates = all_dates[(all_dates >= min_date) & (all_dates <= max_date)]

    equity = initial_capital
    active_positions = []
    trade_results = []
    equity_series = {}

    for date in dates:
        # Close positions that hit exit date
        still_active = []
        for pos in active_positions:
            if date >= pos["exit_date"]:
                # Close position
                if pos["ticker"] in price_df.columns and not pd.isna(price_df.loc[date, pos["ticker"]] if date in price_df.index else np.nan):
                    exit_price = price_df.loc[date, pos["ticker"]]
                    exit_price *= (1 - SLIPPAGE_PCT)  # slippage on sell
                    pnl = pos["shares"] * (exit_price - pos["entry_price"])
                    equity += pos["capital_used"] + pnl
                    trade_results.append({
                        "ticker": pos["ticker"],
                        "entry_date": pos["entry_date"].isoformat(),
                        "exit_date": date.isoformat(),
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_price,
                        "shares": pos["shares"],
                        "pnl": pnl,
                        "return_pct": pnl / pos["capital_used"] if pos["capital_used"] > 0 else 0,
                        "regime": pos.get("regime", "unknown"),
                    })
                else:
                    # Can't close, keep for next day
                    still_active.append(pos)
                    continue
            else:
                still_active.append(pos)
        active_positions = still_active

        # Open new positions
        for trade in trades:
            if trade["entry_date"] == date and len(active_positions) < max_concurrent:
                ticker = trade["ticker"]
                if ticker not in price_df.columns:
                    continue
                price_val = price_df.loc[date, ticker] if date in price_df.index else np.nan
                if pd.isna(price_val):
                    continue
                entry_price = price_val * (1 + SLIPPAGE_PCT)  # slippage on buy
                weight = trade.get("weight", 1.0)
                capital_per_trade = equity * weight / max(1, min(max_concurrent, max_concurrent - len(active_positions)))
                capital_per_trade = min(capital_per_trade, equity * 0.95)  # keep 5% cash buffer
                if capital_per_trade < 10:
                    continue
                shares = capital_per_trade / entry_price
                active_positions.append({
                    "ticker": ticker,
                    "entry_date": date,
                    "exit_date": trade["exit_date"],
                    "entry_price": entry_price,
                    "shares": shares,
                    "capital_used": capital_per_trade,
                    "regime": trade.get("regime", "unknown"),
                })
                equity -= capital_per_trade

        # Mark-to-market
        mtm = equity
        for pos in active_positions:
            if pos["ticker"] in price_df.columns:
                current = price_df.loc[date, pos["ticker"]] if date in price_df.index else pos["entry_price"]
                if not pd.isna(current):
                    mtm += pos["shares"] * current
                else:
                    mtm += pos["capital_used"]
            else:
                mtm += pos["capital_used"]
        equity_series[date] = mtm

    eq = pd.Series(equity_series).sort_index()
    return eq, trade_results


def compute_metrics(equity_curve, trade_results, initial_capital=INITIAL_CAPITAL):
    """Compute Sharpe, Sortino, PF, WR, MaxDD, etc."""
    if len(equity_curve) < 2 or len(trade_results) == 0:
        return {
            "sharpe": 0, "sortino": 0, "profit_factor": 0, "win_rate": 0,
            "max_drawdown_pct": -100, "total_trades": len(trade_results),
            "final_equity": initial_capital, "total_return_pct": 0,
        }

    # Daily returns
    daily_rets = equity_curve.pct_change().dropna()
    if len(daily_rets) == 0:
        daily_rets = pd.Series([0.0])

    # Sharpe (annualized)
    mean_r = daily_rets.mean()
    std_r = daily_rets.std()
    sharpe = (mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0

    # Sortino
    downside = daily_rets[daily_rets < 0]
    down_std = downside.std() if len(downside) > 0 else 1e-10
    sortino = (mean_r / down_std * np.sqrt(252)) if down_std > 0 else 0

    # Win rate
    pnls = [t["pnl"] for t in trade_results]
    winners = sum(1 for p in pnls if p > 0)
    win_rate = winners / len(pnls) if len(pnls) > 0 else 0

    # Profit factor
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999 if gross_profit > 0 else 0)

    # Max drawdown
    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak
    max_dd = dd.min()

    final_eq = equity_curve.iloc[-1]
    total_return = (final_eq - initial_capital) / initial_capital * 100

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(profit_factor, 3),
        "win_rate": round(win_rate, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "total_trades": len(trade_results),
        "final_equity": round(final_eq, 2),
        "total_return_pct": round(total_return, 2),
    }


def compute_regime_gap(trade_results):
    """Compute regime gap: |Sharpe_bull - Sharpe_bear| / max(|Sharpe_bull|, |Sharpe_bear|)."""
    bull_rets = [t["return_pct"] for t in trade_results if t.get("regime") == "bull"]
    bear_rets = [t["return_pct"] for t in trade_results if t.get("regime") == "bear"]

    def sharpe_from_rets(rets):
        if len(rets) < 2:
            return 0
        return np.mean(rets) / np.std(rets) * np.sqrt(len(rets)) if np.std(rets) > 0 else 0

    s_bull = sharpe_from_rets(bull_rets)
    s_bear = sharpe_from_rets(bear_rets)
    denom = max(abs(s_bull), abs(s_bear))
    gap = abs(s_bull - s_bear) / denom if denom > 0 else 0
    return round(gap, 3), len(bull_rets), len(bear_rets)


def permutation_test(real_sharpe, equity_fn, n_iter=PERM_ITERATIONS):
    """
    Permutation test: shuffle entry dates, recompute Sharpe.
    equity_fn(shuffled=True) -> equity_curve, trade_results
    Returns p-value.
    """
    count_better = 0
    for _ in range(n_iter):
        eq, trs = equity_fn(shuffled=True)
        if len(eq) < 2 or len(trs) == 0:
            continue
        m = compute_metrics(eq, trs)
        if m["sharpe"] >= real_sharpe:
            count_better += 1
    p_val = (count_better + 1) / (n_iter + 1)
    return round(p_val, 4)


def generate_trades_variant_a(splits_df, price_df, regime, oot_only=True):
    """Variant A: Post-Split Drift, buy on split date, hold 20 trading days."""
    trades = []
    for _, row in splits_df.iterrows():
        if oot_only and row["date"] < OOT_START:
            continue
        ticker = row["ticker"]
        entry = find_trading_day(price_df, row["date"], ticker, "forward")
        if entry is None:
            continue
        exit_date = offset_trading_days(price_df, entry, ticker, 20)
        if exit_date is None or exit_date > OOT_END:
            continue
        r = "bull" if (entry in regime.index and regime.loc[entry]) else "bear"
        trades.append({"ticker": ticker, "entry_date": entry, "exit_date": exit_date,
                        "weight": 1.0, "regime": r})
    return trades


def generate_trades_variant_b(splits_df, price_df, regime, oot_only=True):
    """Variant B: Post-Split Extended, hold 60 trading days."""
    trades = []
    for _, row in splits_df.iterrows():
        if oot_only and row["date"] < OOT_START:
            continue
        ticker = row["ticker"]
        entry = find_trading_day(price_df, row["date"], ticker, "forward")
        if entry is None:
            continue
        exit_date = offset_trading_days(price_df, entry, ticker, 60)
        if exit_date is None or exit_date > OOT_END:
            continue
        r = "bull" if (entry in regime.index and regime.loc[entry]) else "bear"
        trades.append({"ticker": ticker, "entry_date": entry, "exit_date": exit_date,
                        "weight": 1.0, "regime": r})
    return trades


def generate_trades_variant_c(splits_df, price_df, regime, oot_only=True):
    """Variant C: Sector contagion — buy sector ETF on split date, hold 20 days."""
    trades = []
    for _, row in splits_df.iterrows():
        if oot_only and row["date"] < OOT_START:
            continue
        ticker = row["ticker"]
        etf = SECTOR_ETF_MAP.get(ticker, "SPY")
        entry = find_trading_day(price_df, row["date"], etf, "forward")
        if entry is None:
            continue
        exit_date = offset_trading_days(price_df, entry, etf, 20)
        if exit_date is None or exit_date > OOT_END:
            continue
        r = "bull" if (entry in regime.index and regime.loc[entry]) else "bear"
        trades.append({"ticker": etf, "entry_date": entry, "exit_date": exit_date,
                        "weight": 1.0, "regime": r})
    return trades


def generate_trades_variant_d(splits_df, price_df, regime, oot_only=True):
    """Variant D: Pre-Split Run — buy 10 days before split, sell on split date."""
    trades = []
    for _, row in splits_df.iterrows():
        if oot_only and row["date"] < OOT_START:
            continue
        ticker = row["ticker"]
        split_day = find_trading_day(price_df, row["date"], ticker, "forward")
        if split_day is None:
            continue
        entry = offset_trading_days(price_df, split_day, ticker, -10)
        if entry is None or entry < OOT_START:
            continue
        r = "bull" if (entry in regime.index and regime.loc[entry]) else "bear"
        trades.append({"ticker": ticker, "entry_date": entry, "exit_date": split_day,
                        "weight": 1.0, "regime": r})
    return trades


def generate_trades_variant_e(splits_df, price_df, regime, oot_only=True):
    """Variant E: Split + Momentum — only if positive 20d momentum pre-split. Hold 40d."""
    trades = []
    for _, row in splits_df.iterrows():
        if oot_only and row["date"] < OOT_START:
            continue
        ticker = row["ticker"]
        entry = find_trading_day(price_df, row["date"], ticker, "forward")
        if entry is None:
            continue
        # Check 20d momentum
        lookback_start = offset_trading_days(price_df, entry, ticker, -20)
        if lookback_start is None:
            continue
        if ticker not in price_df.columns:
            continue
        p_start = price_df.loc[lookback_start, ticker] if lookback_start in price_df.index else np.nan
        p_entry = price_df.loc[entry, ticker] if entry in price_df.index else np.nan
        if pd.isna(p_start) or pd.isna(p_entry) or p_start <= 0:
            continue
        momentum = (p_entry - p_start) / p_start
        if momentum <= 0:
            continue  # skip negative momentum
        exit_date = offset_trading_days(price_df, entry, ticker, 40)
        if exit_date is None or exit_date > OOT_END:
            continue
        r = "bull" if (entry in regime.index and regime.loc[entry]) else "bear"
        trades.append({"ticker": ticker, "entry_date": entry, "exit_date": exit_date,
                        "weight": 1.0, "regime": r})
    return trades


def generate_trades_variant_f(splits_df, price_df, regime, oot_only=True):
    """Variant F: Small Split Bias — weight inversely to price. Hold 30d."""
    trades = []
    for _, row in splits_df.iterrows():
        if oot_only and row["date"] < OOT_START:
            continue
        ticker = row["ticker"]
        entry = find_trading_day(price_df, row["date"], ticker, "forward")
        if entry is None:
            continue
        exit_date = offset_trading_days(price_df, entry, ticker, 30)
        if exit_date is None or exit_date > OOT_END:
            continue
        # Weight inversely to price (lower price = larger weight)
        price_val = price_df.loc[entry, ticker] if entry in price_df.index else np.nan
        if pd.isna(price_val) or price_val <= 0:
            continue
        # Normalize: $50 stock gets weight 1.0, $500 stock gets 0.1, etc.
        weight = min(2.0, max(0.1, 50.0 / price_val))
        r = "bull" if (entry in regime.index and regime.loc[entry]) else "bear"
        trades.append({"ticker": ticker, "entry_date": entry, "exit_date": exit_date,
                        "weight": weight, "regime": r})
    return trades


VARIANT_GENERATORS = {
    "A_PostSplitDrift_20d": generate_trades_variant_a,
    "B_PostSplitExtended_60d": generate_trades_variant_b,
    "C_SectorContagion_20d": generate_trades_variant_c,
    "D_PreSplitRun_10d": generate_trades_variant_d,
    "E_SplitMomentum_40d": generate_trades_variant_e,
    "F_SmallSplitBias_30d": generate_trades_variant_f,
}


def generate_shuffled_trades(gen_fn, splits_df, price_df, regime):
    """Generate trades with randomized split dates for permutation test."""
    # Create shuffled splits by assigning random dates from OOT period
    trading_days = price_df.index[(price_df.index >= OOT_START) & (price_df.index <= OOT_END)]
    if len(trading_days) == 0:
        return pd.Series(dtype=float), []

    shuffled_splits = splits_df.copy()
    oot_mask = shuffled_splits["date"] >= OOT_START
    n_oot = oot_mask.sum()
    if n_oot == 0:
        return pd.Series(dtype=float), []

    random_dates = np.random.choice(trading_days, size=n_oot, replace=True)
    shuffled_splits.loc[oot_mask, "date"] = random_dates

    trades = gen_fn(shuffled_splits, price_df, regime, oot_only=True)
    if len(trades) == 0:
        return pd.Series(dtype=float), []
    return run_backtest(trades, price_df)


def five_gate_check(metrics, perm_p, regime_gap):
    """5-gate validation."""
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": perm_p < 0.05,
        "regime_gap_lt_0.5": regime_gap < 0.5,
        "maxdd_gt_neg50": metrics["max_drawdown_pct"] > -50,
        "trades_gte_20": metrics["total_trades"] >= 20,
    }
    gates["all_pass"] = all(gates.values())
    return gates


def main():
    print("=" * 70)
    print("STOCK SPLIT EFFECT BACKTEST")
    print("=" * 70)

    # Download data
    price_df, splits_df = download_data()

    if len(splits_df) == 0:
        print("\nNO SPLITS FOUND. Strategy is untestable.")
        results = {"error": "No stock splits found in universe/period", "variants": {}}
        Path("/home/jupiter/Lvl3Quant/data/stock_split_effect_results.json").write_text(
            json.dumps(results, indent=2))
        return

    # Check OOT split count
    oot_splits = splits_df[splits_df["date"] >= OOT_START]
    all_splits_count = len(splits_df)
    oot_splits_count = len(oot_splits)

    print(f"\nTotal splits found: {all_splits_count}")
    print(f"OOT splits (2022+): {oot_splits_count}")

    insufficient = oot_splits_count < 10
    if insufficient:
        print(f"\nWARNING: Only {oot_splits_count} OOT splits found (<10). "
              "Results will be statistically unreliable.")

    # Compute regime
    regime = get_spy_regime(price_df)

    # Run all variants
    results = {
        "metadata": {
            "strategy": "Stock Split Effect",
            "universe_size": len(UNIVERSE),
            "total_splits_found": all_splits_count,
            "oot_splits_found": oot_splits_count,
            "oot_period": f"{OOT_START.date()} to {OOT_END.date()}",
            "initial_capital": INITIAL_CAPITAL,
            "slippage_pct": SLIPPAGE_PCT,
            "max_concurrent_positions": MAX_CONCURRENT,
            "insufficient_sample": insufficient,
            "run_timestamp": datetime.now().isoformat(),
        },
        "splits_found": [],
        "variants": {},
    }

    # Record splits
    for _, row in splits_df.iterrows():
        results["splits_found"].append({
            "ticker": row["ticker"],
            "date": row["date"].isoformat(),
            "ratio": float(row["ratio"]),
            "in_oot": bool(row["date"] >= OOT_START),
        })

    print("\n" + "=" * 70)
    print("RUNNING VARIANTS")
    print("=" * 70)

    for variant_name, gen_fn in VARIANT_GENERATORS.items():
        print(f"\n--- {variant_name} ---")

        # Generate trades
        trades = gen_fn(splits_df, price_df, regime, oot_only=True)
        print(f"  Trades generated: {len(trades)}")

        if len(trades) == 0:
            results["variants"][variant_name] = {
                "metrics": {"total_trades": 0, "note": "No valid trades generated"},
                "gates": {"all_pass": False},
            }
            continue

        # Run backtest
        eq, trade_results = run_backtest(trades, price_df)

        if len(eq) < 2:
            results["variants"][variant_name] = {
                "metrics": {"total_trades": 0, "note": "Insufficient equity curve data"},
                "gates": {"all_pass": False},
            }
            continue

        metrics = compute_metrics(eq, trade_results)
        print(f"  Sharpe: {metrics['sharpe']}, Sortino: {metrics['sortino']}, "
              f"PF: {metrics['profit_factor']}, WR: {metrics['win_rate']}, "
              f"MaxDD: {metrics['max_drawdown_pct']}%, Trades: {metrics['total_trades']}, "
              f"Final: ${metrics['final_equity']}")

        # Regime gap
        rgap, n_bull, n_bear = compute_regime_gap(trade_results)
        print(f"  Regime gap: {rgap} (bull trades: {n_bull}, bear trades: {n_bear})")

        # Permutation test
        print(f"  Running {PERM_ITERATIONS} permutations...")

        def equity_fn(shuffled=False):
            if shuffled:
                return generate_shuffled_trades(gen_fn, splits_df, price_df, regime)
            return eq, trade_results

        perm_p = permutation_test(metrics["sharpe"], equity_fn, PERM_ITERATIONS)
        print(f"  Permutation p-value: {perm_p}")

        # 5-gate check
        gates = five_gate_check(metrics, perm_p, rgap)
        print(f"  Gates: {gates}")

        results["variants"][variant_name] = {
            "metrics": metrics,
            "regime_gap": rgap,
            "regime_bull_trades": n_bull,
            "regime_bear_trades": n_bear,
            "permutation_p_value": perm_p,
            "gates": gates,
            "trade_details": trade_results[:20],  # first 20 for inspection
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Variant':<30} {'Sharpe':>7} {'Sort':>7} {'PF':>7} {'WR':>6} "
          f"{'MaxDD':>7} {'Trades':>6} {'Final$':>8} {'p-val':>7} {'RGap':>6} {'Pass':>5}")
    print("-" * 105)

    for vname, vdata in results["variants"].items():
        m = vdata.get("metrics", {})
        if m.get("total_trades", 0) == 0:
            print(f"{vname:<30} {'--':>7} {'--':>7} {'--':>7} {'--':>6} "
                  f"{'--':>7} {'0':>6} {'--':>8} {'--':>7} {'--':>6} {'SKIP':>5}")
            continue
        gates = vdata.get("gates", {})
        print(f"{vname:<30} {m['sharpe']:>7.3f} {m['sortino']:>7.3f} {m['profit_factor']:>7.3f} "
              f"{m['win_rate']:>6.3f} {m['max_drawdown_pct']:>6.1f}% {m['total_trades']:>6} "
              f"{m['final_equity']:>8.2f} {vdata.get('permutation_p_value', 'N/A'):>7} "
              f"{vdata.get('regime_gap', 'N/A'):>6} {'YES' if gates.get('all_pass') else 'NO':>5}")

    if insufficient:
        print(f"\n*** INSUFFICIENT SAMPLE: Only {oot_splits_count} OOT splits. "
              "Strategy is statistically unreliable with this universe/period. ***")

    # Save
    out_path = Path("/home/jupiter/Lvl3Quant/data/stock_split_effect_results.json")
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
