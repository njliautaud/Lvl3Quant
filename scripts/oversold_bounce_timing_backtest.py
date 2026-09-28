#!/usr/bin/env python3
"""
Oversold Bounce Timing Backtest
Tests whether buying QQQ/SPY at extreme oversold levels has predictive value
with proper timing, confirmation signals, and position sizing.

Variants:
  A: RSI<25 simple
  B: RSI<25 + Volume capitulation
  C: RSI<20 + VIX spike
  D: Progressive entry (scale in as RSI drops)
  E: RSI<25 + Bollinger Band breakdown
  F: Adversarial random buy (control)
"""

import json
import warnings
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Config ──────────────────────────────────────────────────────────────
OOT_START = "2022-01-01"
OOT_END = "2026-07-29"
FETCH_START = "2020-06-01"  # need lookback for indicators
STARTING_CAPITAL = 645.0
PERM_ITERATIONS = 1000

RESULTS_PATH = "/home/jupiter/Lvl3Quant/data/oversold_bounce_timing_results.json"


# ── Data Download ───────────────────────────────────────────────────────
def download_data():
    print("Downloading SPY, QQQ, ^VIX ...")
    tickers = ["SPY", "QQQ", "^VIX"]
    raw = yf.download(tickers, start=FETCH_START, end=OOT_END, auto_adjust=True)

    # Handle multi-level columns from yfinance
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
        volume = raw["Volume"].copy()
    else:
        # single ticker fallback
        close = raw[["Close"]].copy()
        volume = raw[["Volume"]].copy()

    # Flatten column names if needed
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = [c[0] if isinstance(c, tuple) else c for c in close.columns]
    if isinstance(volume.columns, pd.MultiIndex):
        volume.columns = [c[0] if isinstance(c, tuple) else c for c in volume.columns]

    # Rename VIX
    if "^VIX" in close.columns:
        close = close.rename(columns={"^VIX": "VIX"})
    if "^VIX" in volume.columns:
        volume = volume.rename(columns={"^VIX": "VIX"})

    close = close.dropna(subset=["SPY", "QQQ"])
    print(f"  Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} rows")
    return close, volume


# ── Indicators ──────────────────────────────────────────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_bollinger(series, window=20, num_std=2):
    mid = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def build_features(close, volume):
    """Build all indicators needed for the variants."""
    df = pd.DataFrame(index=close.index)
    for sym in ["SPY", "QQQ"]:
        df[f"{sym}_close"] = close[sym]
        df[f"{sym}_rsi"] = compute_rsi(close[sym])
        df[f"{sym}_vol"] = volume[sym]
        df[f"{sym}_vol_avg20"] = volume[sym].rolling(20).mean()
        df[f"{sym}_vol_ratio"] = volume[sym] / df[f"{sym}_vol_avg20"]
        mid, upper, lower = compute_bollinger(close[sym])
        df[f"{sym}_bb_mid"] = mid
        df[f"{sym}_bb_upper"] = upper
        df[f"{sym}_bb_lower"] = lower
        df[f"{sym}_ret"] = close[sym].pct_change()

    df["VIX"] = close["VIX"] if "VIX" in close.columns else np.nan
    df["SPY_sma200"] = close["SPY"].rolling(200).mean()
    df["bull_regime"] = (close["SPY"] > df["SPY_sma200"]).astype(int)
    df = df.dropna()
    return df


# ── Backtest Engine ─────────────────────────────────────────────────────
def run_backtest(df, signal_func, variant_name, symbols=("SPY", "QQQ")):
    """
    Generic backtest engine.
    signal_func(df, sym, i) -> (action, weight)
      action: 'buy', 'sell', 'hold'
      weight: fraction of capital to deploy (0-1)

    Returns per-symbol results dict.
    """
    oot_mask = df.index >= OOT_START
    df_oot = df[oot_mask].copy()

    results = {}
    for sym in symbols:
        capital = STARTING_CAPITAL
        position_value = 0.0
        shares = 0.0
        entry_price = 0.0
        daily_equity = []
        trades = []
        trade_open = False

        for i in range(len(df_oot)):
            idx = df_oot.index[i]
            price = df_oot[f"{sym}_close"].iloc[i]
            action, weight = signal_func(df_oot, sym, i)

            if action == "buy" and not trade_open and capital > 0:
                invest = capital * weight
                shares = invest / price
                entry_price = price
                capital -= invest
                trade_open = True
            elif action == "sell" and trade_open:
                exit_value = shares * price
                pnl = exit_value - (shares * entry_price)
                trades.append({
                    "entry_date": str(trades[-1]["entry_date"]) if trades else str(idx.date()),
                    "exit_date": str(idx.date()),
                    "pnl": pnl,
                    "ret": (price - entry_price) / entry_price,
                    "regime": "bull" if df_oot["bull_regime"].iloc[i] == 1 else "bear"
                })
                capital += exit_value
                shares = 0.0
                trade_open = False

            # Track equity
            equity = capital + (shares * price if trade_open else 0)
            daily_equity.append({"date": str(idx.date()), "equity": equity,
                                 "regime": "bull" if df_oot["bull_regime"].iloc[i] == 1 else "bear"})

        # Close open position at end
        if trade_open:
            price = df_oot[f"{sym}_close"].iloc[-1]
            exit_value = shares * price
            pnl = exit_value - (shares * entry_price)
            trades.append({
                "entry_date": "open",
                "exit_date": str(df_oot.index[-1].date()),
                "pnl": pnl,
                "ret": (price - entry_price) / entry_price,
                "regime": "bull" if df_oot["bull_regime"].iloc[-1] == 1 else "bear"
            })
            capital += exit_value

        results[sym] = compute_metrics(daily_equity, trades, variant_name, sym)
    return results


def run_backtest_progressive(df, symbols=("SPY", "QQQ")):
    """Special backtest for variant D (progressive entry)."""
    oot_mask = df.index >= OOT_START
    df_oot = df[oot_mask].copy()

    results = {}
    for sym in symbols:
        capital = STARTING_CAPITAL
        tranches = [
            {"rsi_thresh": 30, "fired": False},
            {"rsi_thresh": 25, "fired": False},
            {"rsi_thresh": 20, "fired": False},
            {"rsi_thresh": 15, "fired": False},
        ]
        total_shares = 0.0
        total_cost = 0.0
        daily_equity = []
        trades = []
        in_position = False

        for i in range(len(df_oot)):
            price = df_oot[f"{sym}_close"].iloc[i]
            rsi = df_oot[f"{sym}_rsi"].iloc[i]
            regime = "bull" if df_oot["bull_regime"].iloc[i] == 1 else "bear"

            # Buy tranches
            for t in tranches:
                if not t["fired"] and rsi < t["rsi_thresh"] and capital > 0:
                    invest = STARTING_CAPITAL * 0.25  # 25% of original capital
                    invest = min(invest, capital)
                    if invest > 0:
                        new_shares = invest / price
                        total_shares += new_shares
                        total_cost += invest
                        capital -= invest
                        t["fired"] = True
                        in_position = True

            # Sell all when RSI > 50
            if in_position and rsi > 50:
                exit_value = total_shares * price
                avg_entry = total_cost / total_shares if total_shares > 0 else price
                pnl = exit_value - total_cost
                trades.append({
                    "entry_date": "progressive",
                    "exit_date": str(df_oot.index[i].date()),
                    "pnl": pnl,
                    "ret": (price - avg_entry) / avg_entry,
                    "regime": regime
                })
                capital += exit_value
                total_shares = 0.0
                total_cost = 0.0
                in_position = False
                for t in tranches:
                    t["fired"] = False

            equity = capital + (total_shares * price if in_position else 0)
            daily_equity.append({"date": str(df_oot.index[i].date()), "equity": equity, "regime": regime})

        # Close open
        if in_position:
            price = df_oot[f"{sym}_close"].iloc[-1]
            exit_value = total_shares * price
            avg_entry = total_cost / total_shares if total_shares > 0 else price
            pnl = exit_value - total_cost
            trades.append({
                "entry_date": "progressive",
                "exit_date": str(df_oot.index[-1].date()),
                "pnl": pnl,
                "ret": (price - avg_entry) / avg_entry,
                "regime": "bull" if df_oot["bull_regime"].iloc[-1] == 1 else "bear"
            })
            capital += exit_value

        results[sym] = compute_metrics(daily_equity, trades, "D_progressive", sym)
    return results


# ── Metrics ─────────────────────────────────────────────────────────────
def compute_metrics(daily_equity, trades, variant_name, sym):
    eq = pd.DataFrame(daily_equity)
    eq["equity"] = eq["equity"].astype(float)
    eq["ret"] = eq["equity"].pct_change().fillna(0)

    n_trades = len(trades)
    if n_trades == 0:
        return {
            "variant": variant_name, "symbol": sym, "n_trades": 0,
            "sharpe": 0, "sortino": 0, "pf": 0, "wr": 0,
            "max_dd_pct": 0, "total_ret_pct": 0,
            "sharpe_bull": 0, "sharpe_bear": 0, "regime_gap": 0,
            "perm_p": 1.0, "pass_5gate": False,
            "final_equity": STARTING_CAPITAL
        }

    # Sharpe (annualized)
    daily_rets = eq["ret"].values
    sharpe = (np.mean(daily_rets) / (np.std(daily_rets) + 1e-9)) * np.sqrt(252)

    # Sortino
    downside = daily_rets[daily_rets < 0]
    downside_std = np.std(downside) if len(downside) > 0 else 1e-9
    sortino = (np.mean(daily_rets) / (downside_std + 1e-9)) * np.sqrt(252)

    # Profit Factor
    trade_rets = [t["ret"] for t in trades]
    gains = sum(r for r in trade_rets if r > 0)
    losses = abs(sum(r for r in trade_rets if r < 0))
    pf = gains / (losses + 1e-9)

    # Win Rate
    wr = sum(1 for r in trade_rets if r > 0) / n_trades

    # Max Drawdown
    eq_series = eq["equity"].values
    peak = np.maximum.accumulate(eq_series)
    dd = (eq_series - peak) / (peak + 1e-9)
    max_dd = np.min(dd) * 100

    # Total return
    total_ret = (eq_series[-1] / STARTING_CAPITAL - 1) * 100

    # Regime-stratified Sharpe
    bull_mask = eq["regime"] == "bull"
    bear_mask = eq["regime"] == "bear"
    bull_rets = eq.loc[bull_mask, "ret"].values
    bear_rets = eq.loc[bear_mask, "ret"].values

    sharpe_bull = (np.mean(bull_rets) / (np.std(bull_rets) + 1e-9)) * np.sqrt(252) if len(bull_rets) > 20 else 0
    sharpe_bear = (np.mean(bear_rets) / (np.std(bear_rets) + 1e-9)) * np.sqrt(252) if len(bear_rets) > 20 else 0

    regime_gap = abs(sharpe_bull - sharpe_bear) / (max(abs(sharpe_bull), abs(sharpe_bear)) + 1e-9)

    return {
        "variant": variant_name, "symbol": sym, "n_trades": n_trades,
        "sharpe": round(sharpe, 3), "sortino": round(sortino, 3),
        "pf": round(pf, 3), "wr": round(wr, 3),
        "max_dd_pct": round(max_dd, 2), "total_ret_pct": round(total_ret, 2),
        "sharpe_bull": round(sharpe_bull, 3), "sharpe_bear": round(sharpe_bear, 3),
        "regime_gap": round(regime_gap, 3),
        "final_equity": round(eq_series[-1], 2),
        "perm_p": None,  # filled later
        "pass_5gate": None  # filled later
    }


# ── Permutation Test ────────────────────────────────────────────────────
def permutation_test(df, actual_sharpe, sym, hold_days, n_iter=PERM_ITERATIONS):
    """Random-entry permutation test."""
    oot_mask = df.index >= OOT_START
    df_oot = df[oot_mask].copy()
    prices = df_oot[f"{sym}_close"].values
    n = len(prices)

    random_sharpes = []
    for _ in range(n_iter):
        capital = STARTING_CAPITAL
        equity_curve = np.full(n, STARTING_CAPITAL, dtype=float)
        in_trade = False
        exit_day = 0

        for i in range(n):
            if not in_trade and np.random.random() < 0.02:  # ~2% chance per day
                shares = capital / prices[i]
                entry_price = prices[i]
                capital = 0
                in_trade = True
                exit_day = min(i + hold_days, n - 1)
            elif in_trade and i >= exit_day:
                capital = shares * prices[i]
                in_trade = False

            equity_curve[i] = capital + (shares * prices[i] if in_trade else 0)

        rets = np.diff(equity_curve) / (equity_curve[:-1] + 1e-9)
        s = (np.mean(rets) / (np.std(rets) + 1e-9)) * np.sqrt(252)
        random_sharpes.append(s)

    p_value = np.mean([1 for s in random_sharpes if s >= actual_sharpe]) / n_iter
    return round(p_value, 4)


# ── 5-Gate Check ────────────────────────────────────────────────────────
def check_5gate(m):
    gates = {
        "sharpe_gt_0.5": m["sharpe"] > 0.5,
        "perm_p_lt_0.05": m["perm_p"] < 0.05 if m["perm_p"] is not None else False,
        "regime_gap_lt_0.5": m["regime_gap"] < 0.5,
        "mdd_gt_neg50": m["max_dd_pct"] > -50,
        "trades_gte_20": m["n_trades"] >= 20,
    }
    m["gates"] = gates
    m["pass_5gate"] = all(gates.values())
    return m


# ── Signal Functions ────────────────────────────────────────────────────
def signal_A(df, sym, i):
    """RSI<25 simple. Hold until RSI>50."""
    rsi = df[f"{sym}_rsi"].iloc[i]
    if rsi < 25:
        return ("buy", 1.0)
    elif rsi > 50:
        return ("sell", 0)
    return ("hold", 0)


def signal_B(df, sym, i):
    """RSI<25 + volume > 1.5x avg. Hold 10 days."""
    rsi = df[f"{sym}_rsi"].iloc[i]
    vol_ratio = df[f"{sym}_vol_ratio"].iloc[i]
    # Track hold counter via a simple approach: use date-based
    if rsi < 25 and vol_ratio > 1.5:
        return ("buy", 1.0)
    return ("hold", 0)


def signal_E(df, sym, i):
    """RSI<25 + price < lower BB. Hold until price > mid BB."""
    rsi = df[f"{sym}_rsi"].iloc[i]
    price = df[f"{sym}_close"].iloc[i]
    bb_lower = df[f"{sym}_bb_lower"].iloc[i]
    bb_mid = df[f"{sym}_bb_mid"].iloc[i]
    if rsi < 25 and price < bb_lower:
        return ("buy", 1.0)
    elif price > bb_mid:
        return ("sell", 0)
    return ("hold", 0)


# ── Specialized Backtest for Hold-N-Days Variants ───────────────────────
def run_backtest_hold_n(df, entry_func, hold_days, variant_name, symbols=("SPY", "QQQ")):
    """For variants that hold a fixed number of days."""
    oot_mask = df.index >= OOT_START
    df_oot = df[oot_mask].copy()

    results = {}
    for sym in symbols:
        capital = STARTING_CAPITAL
        shares = 0.0
        entry_price = 0.0
        daily_equity = []
        trades = []
        trade_open = False
        exit_day = 0

        for i in range(len(df_oot)):
            price = df_oot[f"{sym}_close"].iloc[i]
            regime = "bull" if df_oot["bull_regime"].iloc[i] == 1 else "bear"

            # Check exit first
            if trade_open and i >= exit_day:
                exit_value = shares * price
                pnl = exit_value - (shares * entry_price)
                trades.append({
                    "entry_date": entry_date_str,
                    "exit_date": str(df_oot.index[i].date()),
                    "pnl": pnl,
                    "ret": (price - entry_price) / entry_price,
                    "regime": regime
                })
                capital += exit_value
                shares = 0.0
                trade_open = False

            # Check entry
            if not trade_open and entry_func(df_oot, sym, i):
                shares = capital / price
                entry_price = price
                entry_date_str = str(df_oot.index[i].date())
                capital = 0
                trade_open = True
                exit_day = i + hold_days

            equity = capital + (shares * price if trade_open else 0)
            daily_equity.append({"date": str(df_oot.index[i].date()), "equity": equity, "regime": regime})

        # Close open
        if trade_open:
            price = df_oot[f"{sym}_close"].iloc[-1]
            exit_value = shares * price
            pnl = exit_value - (shares * entry_price)
            trades.append({
                "entry_date": entry_date_str,
                "exit_date": str(df_oot.index[-1].date()),
                "pnl": pnl,
                "ret": (price - entry_price) / entry_price,
                "regime": "bull" if df_oot["bull_regime"].iloc[-1] == 1 else "bear"
            })
            capital += exit_value

        results[sym] = compute_metrics(daily_equity, trades, variant_name, sym)
    return results


def entry_B(df, sym, i):
    rsi = df[f"{sym}_rsi"].iloc[i]
    vol_ratio = df[f"{sym}_vol_ratio"].iloc[i]
    return rsi < 25 and vol_ratio > 1.5


def entry_C(df, sym, i):
    rsi = df[f"{sym}_rsi"].iloc[i]
    vix = df["VIX"].iloc[i]
    return rsi < 20 and vix > 25


# ── Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("OVERSOLD BOUNCE TIMING BACKTEST")
    print(f"OOT: {OOT_START} to {OOT_END} | Capital: ${STARTING_CAPITAL}")
    print("=" * 70)

    close, volume = download_data()
    df = build_features(close, volume)
    print(f"Features built: {len(df)} rows, {len(df.columns)} columns")
    print(f"OOT rows: {(df.index >= OOT_START).sum()}")
    print()

    all_results = {}

    # ── Variant A: RSI<25 Simple ────────────────────────────────────────
    print("[A] RSI<25 Simple (hold until RSI>50) ...")
    res_A = run_backtest(df, signal_A, "A_rsi25_simple")
    all_results["A_rsi25_simple"] = res_A
    for sym, m in res_A.items():
        print(f"    {sym}: {m['n_trades']} trades, Sharpe={m['sharpe']}, WR={m['wr']}, MDD={m['max_dd_pct']}%")

    # ── Variant B: RSI<25 + Volume ──────────────────────────────────────
    print("[B] RSI<25 + Volume>1.5x (hold 10d) ...")
    res_B = run_backtest_hold_n(df, entry_B, 10, "B_rsi25_volume")
    all_results["B_rsi25_volume"] = res_B
    for sym, m in res_B.items():
        print(f"    {sym}: {m['n_trades']} trades, Sharpe={m['sharpe']}, WR={m['wr']}, MDD={m['max_dd_pct']}%")

    # ── Variant C: RSI<20 + VIX>25 ─────────────────────────────────────
    print("[C] RSI<20 + VIX>25 (hold 15d) ...")
    res_C = run_backtest_hold_n(df, entry_C, 15, "C_rsi20_vix")
    all_results["C_rsi20_vix"] = res_C
    for sym, m in res_C.items():
        print(f"    {sym}: {m['n_trades']} trades, Sharpe={m['sharpe']}, WR={m['wr']}, MDD={m['max_dd_pct']}%")

    # ── Variant D: Progressive Entry ────────────────────────────────────
    print("[D] Progressive Entry (scale in at RSI 30/25/20/15, sell at RSI>50) ...")
    res_D = run_backtest_progressive(df)
    all_results["D_progressive"] = res_D
    for sym, m in res_D.items():
        print(f"    {sym}: {m['n_trades']} trades, Sharpe={m['sharpe']}, WR={m['wr']}, MDD={m['max_dd_pct']}%")

    # ── Variant E: RSI + Bollinger ──────────────────────────────────────
    print("[E] RSI<25 + Below Lower BB (hold until above mid BB) ...")
    res_E = run_backtest(df, signal_E, "E_rsi_bollinger")
    all_results["E_rsi_bollinger"] = res_E
    for sym, m in res_E.items():
        print(f"    {sym}: {m['n_trades']} trades, Sharpe={m['sharpe']}, WR={m['wr']}, MDD={m['max_dd_pct']}%")

    # ── Determine best variant hold period for adversarial ──────────────
    # Use average of ~10-15 days as typical hold for random
    best_hold = 12

    # ── Variant F: Adversarial Random ───────────────────────────────────
    print("[F] Adversarial Random Buy (control, hold 12d) ...")
    def entry_random(df, sym, i):
        return np.random.random() < 0.02
    np.random.seed(42)
    res_F = run_backtest_hold_n(df, entry_random, best_hold, "F_random")
    all_results["F_random"] = res_F
    for sym, m in res_F.items():
        print(f"    {sym}: {m['n_trades']} trades, Sharpe={m['sharpe']}, WR={m['wr']}, MDD={m['max_dd_pct']}%")

    # ── Permutation Tests ───────────────────────────────────────────────
    print()
    print("Running permutation tests (1000 iterations each) ...")
    hold_map = {
        "A_rsi25_simple": 15,
        "B_rsi25_volume": 10,
        "C_rsi20_vix": 15,
        "D_progressive": 15,
        "E_rsi_bollinger": 15,
        "F_random": 12,
    }

    for vname, vres in all_results.items():
        for sym, m in vres.items():
            if m["n_trades"] > 0:
                print(f"  Perm test: {vname}/{sym} (actual Sharpe={m['sharpe']}) ...")
                m["perm_p"] = permutation_test(df, m["sharpe"], sym, hold_map[vname])
                print(f"    p-value = {m['perm_p']}")
            else:
                m["perm_p"] = 1.0

    # ── 5-Gate Check ────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("5-GATE RESULTS")
    print("=" * 70)
    print(f"{'Variant':<22} {'Sym':<5} {'#Tr':>4} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} "
          f"{'WR':>5} {'MDD%':>7} {'Ret%':>8} {'S_bull':>7} {'S_bear':>7} {'RGap':>6} "
          f"{'perm_p':>7} {'PASS':>5}")
    print("-" * 120)

    for vname, vres in all_results.items():
        for sym, m in vres.items():
            m = check_5gate(m)
            vres[sym] = m
            flag = "YES" if m["pass_5gate"] else "NO"
            print(f"{m['variant']:<22} {sym:<5} {m['n_trades']:>4} {m['sharpe']:>7.3f} "
                  f"{m['sortino']:>8.3f} {m['pf']:>6.2f} {m['wr']:>5.2f} {m['max_dd_pct']:>7.2f} "
                  f"{m['total_ret_pct']:>8.2f} {m['sharpe_bull']:>7.3f} {m['sharpe_bear']:>7.3f} "
                  f"{m['regime_gap']:>6.3f} {m['perm_p']:>7.4f} {flag:>5}")

    # ── Summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passers = []
    for vname, vres in all_results.items():
        for sym, m in vres.items():
            if m["pass_5gate"]:
                passers.append(f"  {m['variant']}/{sym}: Sharpe={m['sharpe']}, Sortino={m['sortino']}, "
                              f"WR={m['wr']}, PF={m['pf']}, {m['n_trades']} trades")

    if passers:
        print("PASSED 5-gate:")
        for p in passers:
            print(p)
    else:
        print("NO variant passed all 5 gates.")

    # Check if any beat random
    print()
    for sym in ["SPY", "QQQ"]:
        random_sharpe = all_results["F_random"][sym]["sharpe"]
        print(f"Random baseline ({sym}): Sharpe={random_sharpe}")
        for vname, vres in all_results.items():
            if vname != "F_random" and sym in vres:
                diff = vres[sym]["sharpe"] - random_sharpe
                better = "BETTER" if diff > 0 else "WORSE"
                print(f"  {vname}: {better} by {diff:+.3f} Sharpe")

    # ── Save Results ────────────────────────────────────────────────────
    # Convert for JSON serialization
    save_data = {
        "metadata": {
            "run_date": datetime.now().isoformat(),
            "oot_start": OOT_START,
            "oot_end": OOT_END,
            "starting_capital": STARTING_CAPITAL,
            "perm_iterations": PERM_ITERATIONS,
        },
        "results": {}
    }
    for vname, vres in all_results.items():
        save_data["results"][vname] = {}
        for sym, m in vres.items():
            # Convert numpy types
            clean = {}
            for k, v in m.items():
                if isinstance(v, (np.integer,)):
                    clean[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    clean[k] = float(v)
                elif isinstance(v, dict):
                    clean[k] = {kk: bool(vv) if isinstance(vv, (np.bool_,)) else vv for kk, vv in v.items()}
                elif isinstance(v, (np.bool_,)):
                    clean[k] = bool(v)
                else:
                    clean[k] = v
            save_data["results"][vname][sym] = clean

    Path(RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
