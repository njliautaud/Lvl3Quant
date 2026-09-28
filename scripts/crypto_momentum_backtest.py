#!/usr/bin/env python3
"""
Crypto Momentum Backtest — 6 Variants on BTC/ETH via Robinhood
Walk-forward OOT: Jan 2022 – Jul 2026
5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades
Cost: 0.3% spread (crypto), $669 initial capital, fractional allowed.
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

# ── Constants ──────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 669.0
CRYPTO_SPREAD_PCT = 0.003  # 0.3% round-trip spread cost
OOT_START = "2022-01-01"
OOT_END = "2026-07-30"
DATA_START = "2020-06-01"  # extra history for SMA warm-up
N_PERM = 1000
SEED = 42

# ── Data Download ──────────────────────────────────────────────────────────────
def download_data():
    """Download BTC-USD, ETH-USD, SPY daily data."""
    tickers = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SPY": "SPY"}
    data = {}
    for name, ticker in tickers.items():
        df = yf.download(ticker, start=DATA_START, end=OOT_END, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        data[name] = df
        print(f"  {name}: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
    return data


# ── Technical Indicators ───────────────────────────────────────────────────────
def add_indicators(df):
    """Add SMA, RSI, rolling high/low, average range."""
    c = df["Close"]
    df["SMA20"] = c.rolling(20).mean()
    df["SMA50"] = c.rolling(50).mean()
    df["SMA200"] = c.rolling(200).mean()

    # RSI(14)
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI14"] = 100 - (100 / (1 + rs))

    # 30-day rolling high/low
    df["High30"] = df["High"].rolling(30).max()
    df["Low30"] = df["Low"].rolling(30).min()

    # Average daily range (20-day)
    df["DailyRange"] = df["High"] - df["Low"]
    df["AvgRange20"] = df["DailyRange"].rolling(20).mean()

    # Volume average
    df["AvgVol20"] = df["Volume"].rolling(20).mean()

    # 20-day momentum (return)
    df["Mom20"] = c.pct_change(20)

    return df


# ── Backtest Engine ────────────────────────────────────────────────────────────

def run_backtest_v2(signals_df, cost_pct=CRYPTO_SPREAD_PCT, initial_cap=INITIAL_CAPITAL):
    """
    Backtest engine. signals_df needs: Close, Signal (1=long, 0=flat).
    Returns trades list, daily equity DataFrame.
    """
    df = signals_df.dropna(subset=["Close", "Signal"]).copy()

    equity = initial_cap
    position = 0.0
    entry_price = 0.0
    entry_date = None
    in_trade = False
    trades = []
    daily_equity = []

    for i in range(len(df)):
        date = df.index[i]
        price = float(df.iloc[i]["Close"])
        sig = int(df.iloc[i]["Signal"])

        if sig == 1 and not in_trade:
            # Enter long
            cost = equity * cost_pct / 2
            position = (equity - cost) / price
            entry_price = price
            entry_date = date
            in_trade = True

        elif sig == 0 and in_trade:
            # Exit long
            gross = position * price
            cost = gross * cost_pct / 2
            net = gross - cost
            pnl = net - (position * entry_price)
            ret_pct = (price / entry_price - 1) * 100 - cost_pct * 100
            trades.append({
                "entry_date": str(entry_date.date()) if hasattr(entry_date, 'date') else str(entry_date),
                "exit_date": str(date.date()) if hasattr(date, 'date') else str(date),
                "entry_price": round(entry_price, 2),
                "exit_price": round(price, 2),
                "pnl": round(pnl, 2),
                "return_pct": round(ret_pct, 2),
            })
            equity = net
            position = 0.0
            in_trade = False

        # Daily equity
        if in_trade:
            daily_equity.append({"date": date, "equity": position * price})
        else:
            daily_equity.append({"date": date, "equity": equity})

    # Force close open position
    if in_trade and position > 0:
        price = float(df.iloc[-1]["Close"])
        date = df.index[-1]
        gross = position * price
        cost = gross * cost_pct / 2
        net = gross - cost
        pnl = net - (position * entry_price)
        ret_pct = (price / entry_price - 1) * 100 - cost_pct * 100
        trades.append({
            "entry_date": str(entry_date.date()) if hasattr(entry_date, 'date') else str(entry_date),
            "exit_date": str(date.date()) if hasattr(date, 'date') else str(date),
            "entry_price": round(entry_price, 2),
            "exit_price": round(price, 2),
            "pnl": round(pnl, 2),
            "return_pct": round(ret_pct, 2),
        })
        equity = net

    eq_df = pd.DataFrame(daily_equity)
    if len(eq_df) > 0:
        eq_df.set_index("date", inplace=True)

    return trades, eq_df


# ── Strategy Signal Generators ─────────────────────────────────────────────────

def strategy_a_trend_following(btc):
    """BTC Trend Following: long when price > SMA20 AND SMA20 > SMA50."""
    df = btc.copy()
    df["Signal"] = 0
    mask = (df["Close"] > df["SMA20"]) & (df["SMA20"] > df["SMA50"])
    df.loc[mask, "Signal"] = 1
    return df[df.index >= OOT_START][["Close", "Signal"]]


def strategy_b_mean_reversion(btc):
    """BTC Mean Reversion: buy RSI<30, sell RSI>70."""
    df = btc.copy()
    df["Signal"] = np.nan
    df.loc[df["RSI14"] < 30, "Signal"] = 1
    df.loc[df["RSI14"] > 70, "Signal"] = 0
    df["Signal"] = df["Signal"].ffill().fillna(0).astype(int)
    return df[df.index >= OOT_START][["Close", "Signal"]]


def strategy_c_breakout(btc):
    """BTC Breakout: buy new 30-day high on above-avg volume. Sell on 30-day low or after 20 days."""
    df = btc.copy()
    signals = []
    in_trade = False
    days_held = 0

    for i in range(len(df)):
        if pd.isna(df.iloc[i]["High30"]) or pd.isna(df.iloc[i]["AvgVol20"]):
            signals.append(0)
            continue

        price = df.iloc[i]["Close"]
        high30 = df.iloc[i]["High30"]
        low30 = df.iloc[i]["Low30"]
        vol = df.iloc[i]["Volume"]
        avg_vol = df.iloc[i]["AvgVol20"]

        if not in_trade:
            if price >= high30 and vol > avg_vol:
                in_trade = True
                days_held = 0
                signals.append(1)
            else:
                signals.append(0)
        else:
            days_held += 1
            if price <= low30 or days_held >= 20:
                in_trade = False
                signals.append(0)
            else:
                signals.append(1)

    df["Signal"] = signals
    return df[df.index >= OOT_START][["Close", "Signal"]]


def strategy_d_dual_momentum(btc, eth):
    """Dual Crypto Momentum: buy whichever of BTC/ETH has better 20d momentum. Rebalance weekly.
    On rebalance, force exit old asset (signal=0) then enter new asset next day.
    We track returns per-segment to avoid price-switching bugs."""
    common = btc.index.intersection(eth.index)
    b = btc.loc[common].copy()
    e = eth.loc[common].copy()

    df = pd.DataFrame(index=common)
    df["BTC_Close"] = b["Close"]
    df["ETH_Close"] = e["Close"]
    df["BTC_Mom20"] = b["Mom20"]
    df["ETH_Mom20"] = e["Mom20"]

    # Instead of feeding mixed prices into a single backtest, compute returns directly.
    # Weekly rebalance: pick best momentum asset, compute weekly return segments.
    oot_df = df[df.index >= OOT_START].copy()

    equity = INITIAL_CAPITAL
    trades = []
    daily_equity = []
    current_asset = None
    segment_start_price = None
    segment_start_equity = None
    segment_entry_date = None

    last_rebal_week = -1
    last_rebal_year = -1

    for i in range(len(oot_df)):
        row = oot_df.iloc[i]
        date = row.name
        w = date.isocalendar()[1]
        y = date.year
        btc_mom = row["BTC_Mom20"]
        eth_mom = row["ETH_Mom20"]

        need_rebal = (w != last_rebal_week or y != last_rebal_year)

        if pd.isna(btc_mom) or pd.isna(eth_mom):
            daily_equity.append({"date": date, "equity": equity})
            continue

        if need_rebal:
            last_rebal_week = w
            last_rebal_year = y

            # Close current position
            if current_asset is not None and segment_start_price is not None:
                if current_asset == "BTC":
                    exit_price = float(row["BTC_Close"])
                else:
                    exit_price = float(row["ETH_Close"])
                ret = exit_price / segment_start_price - 1
                cost = CRYPTO_SPREAD_PCT  # exit + entry cost
                net_ret = ret - cost
                pnl = segment_start_equity * net_ret
                equity = segment_start_equity * (1 + net_ret)
                trades.append({
                    "entry_date": str(segment_entry_date.date()) if hasattr(segment_entry_date, 'date') else str(segment_entry_date),
                    "exit_date": str(date.date()) if hasattr(date, 'date') else str(date),
                    "entry_price": round(segment_start_price, 2),
                    "exit_price": round(exit_price, 2),
                    "pnl": round(pnl, 2),
                    "return_pct": round(net_ret * 100, 2),
                })

            # Pick new asset
            if btc_mom > eth_mom and btc_mom > 0:
                new_asset = "BTC"
            elif eth_mom > btc_mom and eth_mom > 0:
                new_asset = "ETH"
            else:
                new_asset = None

            if new_asset is not None:
                current_asset = new_asset
                segment_start_price = float(row["BTC_Close"] if new_asset == "BTC" else row["ETH_Close"])
                segment_start_equity = equity
                segment_entry_date = date
            else:
                current_asset = None
                segment_start_price = None

        # Daily equity mark-to-market
        if current_asset is not None and segment_start_price is not None:
            if current_asset == "BTC":
                cur_price = float(row["BTC_Close"])
            else:
                cur_price = float(row["ETH_Close"])
            mtm_ret = cur_price / segment_start_price - 1
            daily_equity.append({"date": date, "equity": segment_start_equity * (1 + mtm_ret)})
        else:
            daily_equity.append({"date": date, "equity": equity})

    # Close final position
    if current_asset is not None and segment_start_price is not None:
        row = oot_df.iloc[-1]
        date = row.name
        if current_asset == "BTC":
            exit_price = float(row["BTC_Close"])
        else:
            exit_price = float(row["ETH_Close"])
        ret = exit_price / segment_start_price - 1
        net_ret = ret - CRYPTO_SPREAD_PCT
        pnl = segment_start_equity * net_ret
        equity = segment_start_equity * (1 + net_ret)
        trades.append({
            "entry_date": str(segment_entry_date.date()) if hasattr(segment_entry_date, 'date') else str(segment_entry_date),
            "exit_date": str(date.date()) if hasattr(date, 'date') else str(date),
            "entry_price": round(segment_start_price, 2),
            "exit_price": round(exit_price, 2),
            "pnl": round(pnl, 2),
            "return_pct": round(net_ret * 100, 2),
        })

    eq_df = pd.DataFrame(daily_equity)
    if len(eq_df) > 0:
        eq_df.set_index("date", inplace=True)

    # Return a dummy signals df for compatibility — but we already have trades + equity
    return trades, eq_df


def strategy_e_btc_spy_hedge(btc, spy):
    """50% BTC (trend following) + 50% SPY buy-and-hold. Combined equity curve."""
    common = btc.index.intersection(spy.index)
    b = btc.loc[common].copy()
    s = spy.loc[common].copy()

    oot_mask = common >= OOT_START
    b_oot = b[oot_mask]
    s_oot = s[oot_mask]

    # BTC portion: trend following (same as A)
    btc_signal = ((b_oot["Close"] > b_oot["SMA20"]) & (b_oot["SMA20"] > b_oot["SMA50"])).astype(int)

    # SPY: buy and hold
    spy_signal = pd.Series(1, index=b_oot.index)

    btc_half = pd.DataFrame({"Close": b_oot["Close"], "Signal": btc_signal})
    spy_half = pd.DataFrame({"Close": s_oot["Close"], "Signal": spy_signal})

    btc_trades, btc_eq = run_backtest_v2(btc_half, cost_pct=CRYPTO_SPREAD_PCT, initial_cap=INITIAL_CAPITAL / 2)
    _, spy_eq = run_backtest_v2(spy_half, cost_pct=0.001, initial_cap=INITIAL_CAPITAL / 2)

    common_eq = btc_eq.index.intersection(spy_eq.index)
    combined_eq = pd.DataFrame(index=common_eq)
    combined_eq["equity"] = btc_eq.loc[common_eq, "equity"].values + spy_eq.loc[common_eq, "equity"].values

    return btc_trades, combined_eq


def strategy_f_vol_breakout(btc):
    """BTC Vol Breakout: buy when daily range > 2x avg range. Hold 5 days."""
    df = btc.copy()
    signals = []
    hold_counter = 0

    for i in range(len(df)):
        if pd.isna(df.iloc[i]["AvgRange20"]):
            signals.append(0)
            continue

        daily_range = df.iloc[i]["DailyRange"]
        avg_range = df.iloc[i]["AvgRange20"]

        if hold_counter > 0:
            signals.append(1)
            hold_counter -= 1
        elif daily_range > 2 * avg_range:
            signals.append(1)
            hold_counter = 4
        else:
            signals.append(0)

    df["Signal"] = signals
    return df[df.index >= OOT_START][["Close", "Signal"]]


# ── Metrics Calculation ────────────────────────────────────────────────────────

def calc_metrics(trades, equity_df, initial_cap=INITIAL_CAPITAL):
    """Calculate Sharpe, Sortino, MaxDD, PF, WR from trades and equity curve."""
    if len(trades) < 2:
        return {
            "n_trades": len(trades),
            "sharpe": 0, "sortino": 0, "profit_factor": 0,
            "win_rate": 0, "max_dd_pct": -100, "total_return_pct": 0,
            "avg_trade_pct": 0, "final_equity": initial_cap,
        }

    rets = [t["return_pct"] / 100 for t in trades]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]

    win_rate = len(wins) / len(rets) if rets else 0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else 999

    # Daily returns from equity curve
    if len(equity_df) > 1:
        eq_vals = equity_df["equity"].values
        daily_rets = np.diff(eq_vals) / eq_vals[:-1]
        daily_rets = daily_rets[np.isfinite(daily_rets)]

        if len(daily_rets) > 1 and np.std(daily_rets) > 0:
            sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(365)
            downside = daily_rets[daily_rets < 0]
            downside_std = np.std(downside) if len(downside) > 0 else 0.0001
            sortino = np.mean(daily_rets) / downside_std * np.sqrt(365)
        else:
            sharpe = 0
            sortino = 0

        peak = np.maximum.accumulate(eq_vals)
        dd = (eq_vals - peak) / peak
        max_dd = float(np.min(dd)) * 100
    else:
        sharpe = sortino = 0
        max_dd = 0

    final_eq = float(equity_df["equity"].iloc[-1]) if len(equity_df) > 0 else initial_cap
    total_ret = (final_eq / initial_cap - 1) * 100

    return {
        "n_trades": len(trades),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(profit_factor, 3),
        "win_rate": round(win_rate * 100, 1),
        "max_dd_pct": round(max_dd, 1),
        "total_return_pct": round(total_ret, 1),
        "avg_trade_pct": round(np.mean(rets) * 100, 2),
        "final_equity": round(final_eq, 2),
    }


# ── Permutation Test ───────────────────────────────────────────────────────────

def permutation_test(trades, n_perm=N_PERM, seed=SEED):
    """Shuffle trade returns with sign flip, compute p-value for observed mean return."""
    if len(trades) < 5:
        return 1.0

    rets = np.array([t["return_pct"] for t in trades])
    observed = np.mean(rets)

    rng = np.random.RandomState(seed)
    count_ge = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=len(rets))
        perm_mean = np.mean(rets * signs)
        if perm_mean >= observed:
            count_ge += 1

    return round(count_ge / n_perm, 4)


# ── Regime Analysis ────────────────────────────────────────────────────────────

def regime_analysis(trades, btc_df, spy_df):
    """Classify each trade by crypto regime (BTC vs 200-SMA) and equity regime (SPY vs 200-SMA)."""
    if len(trades) < 5:
        return {"crypto_bull_sharpe": 0, "crypto_bear_sharpe": 0, "regime_gap": 1.0,
                "equity_bull_sharpe": 0, "equity_bear_sharpe": 0,
                "crypto_bull_trades": 0, "crypto_bear_trades": 0}

    crypto_bull_rets = []
    crypto_bear_rets = []
    equity_bull_rets = []
    equity_bear_rets = []

    for t in trades:
        entry_date = pd.Timestamp(t["entry_date"])
        ret = t["return_pct"] / 100

        if entry_date in btc_df.index and not pd.isna(btc_df.loc[entry_date, "SMA200"]):
            if btc_df.loc[entry_date, "Close"] > btc_df.loc[entry_date, "SMA200"]:
                crypto_bull_rets.append(ret)
            else:
                crypto_bear_rets.append(ret)

        if entry_date in spy_df.index and not pd.isna(spy_df.loc[entry_date, "SMA200"]):
            if spy_df.loc[entry_date, "Close"] > spy_df.loc[entry_date, "SMA200"]:
                equity_bull_rets.append(ret)
            else:
                equity_bear_rets.append(ret)

    def _sharpe(rets_list):
        if len(rets_list) < 3:
            return 0.0
        r = np.array(rets_list)
        if np.std(r) == 0:
            return 0.0
        return float(np.mean(r) / np.std(r) * np.sqrt(len(r)))

    cb = _sharpe(crypto_bull_rets)
    cbr = _sharpe(crypto_bear_rets)
    eb = _sharpe(equity_bull_rets)
    ebr = _sharpe(equity_bear_rets)

    max_regime = max(abs(cb), abs(cbr)) if max(abs(cb), abs(cbr)) > 0 else 1
    regime_gap = abs(cb - cbr) / max_regime

    return {
        "crypto_bull_sharpe": round(cb, 3),
        "crypto_bear_sharpe": round(cbr, 3),
        "crypto_bull_trades": len(crypto_bull_rets),
        "crypto_bear_trades": len(crypto_bear_rets),
        "regime_gap": round(regime_gap, 3),
        "equity_bull_sharpe": round(eb, 3),
        "equity_bear_sharpe": round(ebr, 3),
    }


# ── 5-Gate Validation ──────────────────────────────────────────────────────────

def five_gate_validation(metrics, perm_p, regime_gap):
    """Apply 5-gate filter."""
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": perm_p < 0.05,
        "regime_gap_lt_0.5": regime_gap < 0.5,
        "max_dd_gt_neg50": metrics["max_dd_pct"] > -50,
        "trades_gte_20": metrics["n_trades"] >= 20,
    }
    gates["all_passed"] = all(gates.values())
    return gates


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("CRYPTO MOMENTUM BACKTEST — 6 Variants")
    print(f"OOT Period: {OOT_START} to {OOT_END}")
    print(f"Initial Capital: ${INITIAL_CAPITAL}")
    print(f"Crypto Spread Cost: {CRYPTO_SPREAD_PCT*100}%")
    print("=" * 70)

    print("\n[1/4] Downloading price data...")
    data = download_data()

    print("\n[2/4] Computing indicators...")
    for name in data:
        data[name] = add_indicators(data[name])

    btc = data["BTC"]
    eth = data["ETH"]
    spy = data["SPY"]

    print("\n[3/4] Running strategies...")
    strategies = {
        "A_BTC_Trend_Following": ("trend", lambda: strategy_a_trend_following(btc)),
        "B_BTC_Mean_Reversion": ("mean_rev", lambda: strategy_b_mean_reversion(btc)),
        "C_BTC_Breakout": ("breakout", lambda: strategy_c_breakout(btc)),
        "D_Dual_Crypto_Momentum": ("dual_mom", None),
        "E_BTC_SPY_Hedge": ("hedge", None),
        "F_BTC_Vol_Breakout": ("vol_brk", lambda: strategy_f_vol_breakout(btc)),
    }

    results = {}

    for name, (stype, gen_fn) in strategies.items():
        print(f"\n  --- {name} ---")

        if name in ("E_BTC_SPY_Hedge", "D_Dual_Crypto_Momentum"):
            if name == "E_BTC_SPY_Hedge":
                trades, eq_df = strategy_e_btc_spy_hedge(btc, spy)
            else:
                trades, eq_df = strategy_d_dual_momentum(btc, eth)
        else:
            signals = gen_fn()
            trades, eq_df = run_backtest_v2(signals)

        metrics = calc_metrics(trades, eq_df)
        perm_p = permutation_test(trades)
        regime = regime_analysis(trades, btc, spy)
        gates = five_gate_validation(metrics, perm_p, regime["regime_gap"])

        results[name] = {
            "metrics": metrics,
            "permutation_p": perm_p,
            "regime": regime,
            "gates": gates,
            "sample_trades": trades[:5] if trades else [],
        }

        print(f"    Trades: {metrics['n_trades']}, Sharpe: {metrics['sharpe']}, "
              f"Sortino: {metrics['sortino']}, WR: {metrics['win_rate']}%")
        print(f"    PF: {metrics['profit_factor']}, MaxDD: {metrics['max_dd_pct']}%, "
              f"Total Return: {metrics['total_return_pct']}%")
        print(f"    Final Equity: ${metrics['final_equity']}")
        print(f"    Perm p-value: {perm_p}")
        print(f"    Regime gap: {regime['regime_gap']} "
              f"(Bull: {regime['crypto_bull_sharpe']}, Bear: {regime['crypto_bear_sharpe']})")
        print(f"    5-Gate: {'PASS' if gates['all_passed'] else 'FAIL'} — {gates}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY — 5-GATE VALIDATION")
    print("=" * 70)

    passed = []
    failed = []
    for name, r in results.items():
        status = "PASS" if r["gates"]["all_passed"] else "FAIL"
        failed_gates = [g for g, v in r["gates"].items() if not v and g != "all_passed"]
        if r["gates"]["all_passed"]:
            passed.append(name)
        else:
            failed.append((name, failed_gates))
        print(f"  {status} | {name:30s} | Sharpe={r['metrics']['sharpe']:6.3f} | "
              f"WR={r['metrics']['win_rate']:5.1f}% | DD={r['metrics']['max_dd_pct']:6.1f}% | "
              f"Return={r['metrics']['total_return_pct']:7.1f}% | "
              f"p={r['permutation_p']:.4f} | RGap={r['regime']['regime_gap']:.3f}")

    print(f"\n  PASSED: {len(passed)}/{len(results)}")
    if passed:
        print(f"  Winners: {', '.join(passed)}")
    if failed:
        print(f"  Failed:")
        for name, fg in failed:
            print(f"    {name}: failed [{', '.join(fg)}]")

    # Buy & Hold benchmark
    btc_oot = btc[btc.index >= OOT_START]
    bnh_ret = (btc_oot["Close"].iloc[-1] / btc_oot["Close"].iloc[0] - 1) * 100
    print(f"\n  BTC Buy & Hold Return (OOT): {bnh_ret:.1f}%")
    print(f"  BTC Buy & Hold Final Equity: ${INITIAL_CAPITAL * (1 + bnh_ret/100):.2f}")

    # Save results
    print("\n[4/4] Saving results...")
    output = {
        "metadata": {
            "run_date": datetime.now().isoformat(),
            "oot_period": f"{OOT_START} to {OOT_END}",
            "initial_capital": INITIAL_CAPITAL,
            "crypto_spread_pct": CRYPTO_SPREAD_PCT,
            "n_permutations": N_PERM,
            "btc_buy_hold_return_pct": round(bnh_ret, 1),
        },
        "variants": {},
    }

    for name, r in results.items():
        output["variants"][name] = {
            "metrics": r["metrics"],
            "permutation_p": r["permutation_p"],
            "regime": r["regime"],
            "five_gate": r["gates"],
            "sample_trades": r["sample_trades"],
        }

    output_path = Path("/home/jupiter/Lvl3Quant/data/crypto_momentum_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved to {output_path}")

    return results


if __name__ == "__main__":
    results = main()
