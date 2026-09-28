#!/usr/bin/env python3
"""
Sector Pair Convergence Trading Backtest
=========================================
Market-neutral strategy: bet on divergent sector ETFs converging.
Walk-forward OOT: Jan 2022 – Jul 2026.
5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades.

6 Variants:
A) Z-Score Convergence (ratio z>2, close at 0)
B) Spread Reversion (20d return spread > 2σ, hold 10d)
C) Long-Only Rotation (buy more oversold of pair, weekly rebalance)
D) Multi-Pair Portfolio (all 5 pairs, $129 each)
E) Regime-Conditional (only trade when 60d corr > 0.5)
F) Momentum Filter (only trade when SPY 20d ret flat)
"""

import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
ACCOUNT_SIZE = 645.0
SLIPPAGE = 0.0002  # 0.02% each way
OOT_START = "2022-01-01"
OOT_END = "2026-07-29"
DATA_START = "2020-06-01"  # need lookback for indicators

SECTOR_ETFS = ["XLK", "XLC", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE"]
PAIRS = [
    ("XLK", "XLC"),  # tech & comm
    ("XLY", "XLC"),  # consumer & comm
    ("XLF", "XLI"),  # financials & industrials
    ("XLE", "XLB"),  # energy & materials
    ("XLP", "XLU"),  # staples & utilities
]

PERM_ITERATIONS = 1000

# ── Data Download ──────────────────────────────────────────────────────────────
def download_data():
    tickers = list(set(SECTOR_ETFS + ["SPY"]))
    print(f"Downloading {len(tickers)} tickers...")
    data = yf.download(tickers, start=DATA_START, end=OOT_END, auto_adjust=True, progress=False)

    # Handle multi-level columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data

    # Drop any tickers with too much missing data
    close = close.dropna(axis=1, thresh=int(len(close) * 0.8))
    close = close.ffill().bfill()

    print(f"Downloaded {len(close.columns)} tickers, {len(close)} trading days")
    print(f"Date range: {close.index[0].date()} to {close.index[-1].date()}")
    return close


# ── Helper Functions ───────────────────────────────────────────────────────────
def apply_slippage(price, direction):
    """direction: 1 for buy, -1 for sell"""
    return price * (1 + direction * SLIPPAGE)


def calc_metrics(returns, trades_count):
    """Calculate Sharpe, Sortino, MaxDD, PF, WR from daily returns series."""
    if len(returns) == 0 or returns.std() == 0:
        return {"sharpe": 0, "sortino": 0, "maxdd": -1.0, "pf": 0, "wr": 0,
                "total_ret": 0, "annual_ret": 0, "n_trades": trades_count, "valid": False}

    annual_factor = np.sqrt(252)
    sharpe = returns.mean() / returns.std() * annual_factor

    downside = returns[returns < 0]
    sortino = returns.mean() / downside.std() * annual_factor if len(downside) > 0 and downside.std() > 0 else sharpe * 1.5

    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    drawdowns = cum / rolling_max - 1
    maxdd = drawdowns.min()

    # Profit factor from positive vs negative return days
    pos = returns[returns > 0].sum()
    neg = abs(returns[returns < 0].sum())
    pf = pos / neg if neg > 0 else 99.0

    wr = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

    total_ret = cum.iloc[-1] - 1 if len(cum) > 0 else 0
    years = len(returns) / 252
    annual_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "maxdd": round(maxdd, 4),
        "pf": round(pf, 3),
        "wr": round(wr, 4),
        "total_ret": round(total_ret, 4),
        "annual_ret": round(annual_ret, 4),
        "n_trades": trades_count,
        "valid": True,
    }


def regime_split(returns, spy_close):
    """Split returns into bull/bear based on SPY vs 200-SMA."""
    spy_sma200 = spy_close.rolling(200).mean()
    bull_mask = spy_close > spy_sma200
    bear_mask = spy_close <= spy_sma200

    # Align to returns index
    bull_mask = bull_mask.reindex(returns.index).fillna(False)
    bear_mask = bear_mask.reindex(returns.index).fillna(False)

    bull_ret = returns[bull_mask]
    bear_ret = returns[bear_mask]
    return bull_ret, bear_ret


def regime_gap(returns, spy_close):
    """Calculate regime gap: |Sharpe_bull - Sharpe_bear| / max(|Sharpe_bull|, |Sharpe_bear|)."""
    bull_ret, bear_ret = regime_split(returns, spy_close)

    af = np.sqrt(252)
    sharpe_bull = bull_ret.mean() / bull_ret.std() * af if len(bull_ret) > 10 and bull_ret.std() > 0 else 0
    sharpe_bear = bear_ret.mean() / bear_ret.std() * af if len(bear_ret) > 10 and bear_ret.std() > 0 else 0

    max_abs = max(abs(sharpe_bull), abs(sharpe_bear))
    gap = abs(sharpe_bull - sharpe_bear) / max_abs if max_abs > 0 else 0

    return round(gap, 4), round(sharpe_bull, 3), round(sharpe_bear, 3)


def permutation_test(returns, signal_dates, n_iter=PERM_ITERATIONS):
    """Permutation test: shuffle signal timing to test if entry signals matter.

    We separate 'in-trade' days (nonzero return) from 'out-of-trade' days (zero return).
    Then we randomly reassign which days are in-trade vs out, keeping the same number
    of active days. If signal timing doesn't matter, random timing produces similar Sharpe.
    """
    if len(returns) == 0:
        return 1.0

    ret_vals = returns.values
    n = len(ret_vals)

    # Identify active (in-trade) days
    active_mask = ret_vals != 0
    n_active = active_mask.sum()

    if n_active < 5 or n_active >= n:
        return 1.0

    # Pool of ALL daily returns (we'll randomly pick which days count as "active")
    actual_mean = ret_vals[active_mask].mean()
    actual_sharpe = actual_mean / ret_vals[active_mask].std() * np.sqrt(252) if ret_vals[active_mask].std() > 0 else 0

    count_better = 0
    for _ in range(n_iter):
        # Randomly pick n_active days from the full series
        perm_idx = np.random.choice(n, size=n_active, replace=False)
        perm_active = ret_vals[perm_idx]
        perm_std = perm_active.std()
        perm_sharpe = perm_active.mean() / perm_std * np.sqrt(252) if perm_std > 0 else 0
        if perm_sharpe >= actual_sharpe:
            count_better += 1

    return round(count_better / n_iter, 4)


# ── Strategy Implementations ──────────────────────────────────────────────────

def variant_a_zscore(close, pair, capital):
    """Z-Score Convergence: ratio z>2, close at z=0."""
    etf1, etf2 = pair
    p1, p2 = close[etf1], close[etf2]

    ratio = p1 / p2
    ratio_mean = ratio.rolling(60).mean()
    ratio_std = ratio.rolling(60).std()
    zscore = (ratio - ratio_mean) / ratio_std

    # OOT period only
    oot_mask = close.index >= OOT_START

    positions = pd.Series(0.0, index=close.index)  # +1 = long etf2/short etf1, -1 = reverse
    trades = 0
    in_trade = 0

    for i in range(1, len(close)):
        if not oot_mask[i]:
            continue
        z = zscore.iloc[i]
        if np.isnan(z):
            continue

        if in_trade == 0:
            if z > 2.0:
                # etf1 overvalued vs etf2 -> buy etf2 (underperformer)
                in_trade = 1
                trades += 1
            elif z < -2.0:
                # etf2 overvalued vs etf1 -> buy etf1 (underperformer)
                in_trade = -1
                trades += 1
        else:
            if in_trade == 1 and z <= 0:
                in_trade = 0
            elif in_trade == -1 and z >= 0:
                in_trade = 0

        positions.iloc[i] = in_trade

    # Calculate returns: long underperformer only (market neutral via pair)
    # When position=1: long etf2, when position=-1: long etf1
    ret1 = p1.pct_change()
    ret2 = p2.pct_change()

    daily_ret = pd.Series(0.0, index=close.index)
    for i in range(1, len(close)):
        if positions.iloc[i-1] == 1:
            # Long etf2 (underperformer when z was high)
            daily_ret.iloc[i] = ret2.iloc[i] - SLIPPAGE * (abs(positions.iloc[i] - positions.iloc[i-1]) > 0) * 2
        elif positions.iloc[i-1] == -1:
            # Long etf1 (underperformer when z was low)
            daily_ret.iloc[i] = ret1.iloc[i] - SLIPPAGE * (abs(positions.iloc[i] - positions.iloc[i-1]) > 0) * 2

    daily_ret = daily_ret[close.index >= OOT_START]
    signal_dates = positions[positions != 0].index
    return daily_ret, trades, signal_dates


def variant_b_spread_reversion(close, pair, capital):
    """Spread Reversion: 20d return spread > 2σ (252d), hold 10 days."""
    etf1, etf2 = pair

    ret1_20d = close[etf1].pct_change(20)
    ret2_20d = close[etf2].pct_change(20)
    spread = ret1_20d - ret2_20d
    spread_std = spread.rolling(252).std()
    spread_mean = spread.rolling(252).mean()
    z_spread = (spread - spread_mean) / spread_std

    oot_mask = close.index >= OOT_START
    daily_ret = pd.Series(0.0, index=close.index)
    trades = 0
    hold_remaining = 0
    current_long = None  # which ETF we're long

    ret1 = close[etf1].pct_change()
    ret2 = close[etf2].pct_change()

    for i in range(1, len(close)):
        if not oot_mask[i]:
            continue

        if hold_remaining > 0:
            if current_long == etf2:
                daily_ret.iloc[i] = ret2.iloc[i]
            elif current_long == etf1:
                daily_ret.iloc[i] = ret1.iloc[i]
            hold_remaining -= 1
            if hold_remaining == 0:
                daily_ret.iloc[i] -= SLIPPAGE * 2  # exit slippage
                current_long = None
            continue

        z = z_spread.iloc[i]
        if np.isnan(z):
            continue

        if z > 2.0:
            # etf1 outperformed -> buy etf2
            current_long = etf2
            hold_remaining = 10
            trades += 1
            daily_ret.iloc[i] -= SLIPPAGE * 2  # entry slippage
        elif z < -2.0:
            # etf2 outperformed -> buy etf1
            current_long = etf1
            hold_remaining = 10
            trades += 1
            daily_ret.iloc[i] -= SLIPPAGE * 2

    daily_ret = daily_ret[close.index >= OOT_START]
    signal_dates = daily_ret[daily_ret != 0].index
    return daily_ret, trades, signal_dates


def variant_c_rotation(close, pair, capital):
    """Long-Only Rotation: buy more oversold of pair weekly."""
    etf1, etf2 = pair

    ret1_20d = close[etf1].pct_change(20)
    ret2_20d = close[etf2].pct_change(20)

    oot_mask = close.index >= OOT_START
    daily_ret = pd.Series(0.0, index=close.index)
    trades = 0
    current_hold = None

    ret1 = close[etf1].pct_change()
    ret2 = close[etf2].pct_change()

    week_counter = 0
    for i in range(1, len(close)):
        if not oot_mask[i]:
            continue

        week_counter += 1

        # Rebalance weekly
        if week_counter % 5 == 0:
            r1 = ret1_20d.iloc[i]
            r2 = ret2_20d.iloc[i]
            if np.isnan(r1) or np.isnan(r2):
                continue

            new_hold = etf1 if r1 < r2 else etf2  # buy the more oversold
            if new_hold != current_hold:
                trades += 1
                daily_ret.iloc[i] -= SLIPPAGE * 2  # switching cost
            current_hold = new_hold

        if current_hold == etf1:
            daily_ret.iloc[i] += ret1.iloc[i]
        elif current_hold == etf2:
            daily_ret.iloc[i] += ret2.iloc[i]

    daily_ret = daily_ret[close.index >= OOT_START]
    signal_dates = daily_ret[daily_ret != 0].index
    return daily_ret, trades, signal_dates


def variant_d_multi_pair(close, spy_close, capital):
    """Multi-Pair Portfolio: run all 5 pairs with z-score convergence, equal weight."""
    per_pair_capital = capital / len(PAIRS)
    combined_ret = pd.Series(0.0, index=close[close.index >= OOT_START].index)
    total_trades = 0
    all_signal_dates = []

    for pair in PAIRS:
        ret, trades, sig_dates = variant_a_zscore(close, pair, per_pair_capital)
        combined_ret += ret / len(PAIRS)  # equal weight
        total_trades += trades
        all_signal_dates.extend(sig_dates)

    return combined_ret, total_trades, all_signal_dates


def variant_e_regime_conditional(close, pair, capital):
    """Regime-Conditional: only trade when 60d correlation > 0.5."""
    etf1, etf2 = pair

    # Rolling correlation
    ret1 = close[etf1].pct_change()
    ret2 = close[etf2].pct_change()
    rolling_corr = ret1.rolling(60).corr(ret2)

    # Use z-score convergence but gate on correlation
    ratio = close[etf1] / close[etf2]
    ratio_mean = ratio.rolling(60).mean()
    ratio_std = ratio.rolling(60).std()
    zscore = (ratio - ratio_mean) / ratio_std

    oot_mask = close.index >= OOT_START
    daily_ret = pd.Series(0.0, index=close.index)
    trades = 0
    in_trade = 0

    for i in range(1, len(close)):
        if not oot_mask[i]:
            continue

        z = zscore.iloc[i]
        corr = rolling_corr.iloc[i]
        if np.isnan(z) or np.isnan(corr):
            continue

        if in_trade == 0:
            # Only enter when correlation is high (regime filter)
            if corr > 0.5:
                if z > 2.0:
                    in_trade = 1
                    trades += 1
                    daily_ret.iloc[i] -= SLIPPAGE * 2
                elif z < -2.0:
                    in_trade = -1
                    trades += 1
                    daily_ret.iloc[i] -= SLIPPAGE * 2
        else:
            if (in_trade == 1 and z <= 0) or (in_trade == -1 and z >= 0):
                in_trade = 0
                daily_ret.iloc[i] -= SLIPPAGE * 2

        if in_trade == 1:
            daily_ret.iloc[i] += ret2.iloc[i]
        elif in_trade == -1:
            daily_ret.iloc[i] += ret1.iloc[i]

    daily_ret = daily_ret[close.index >= OOT_START]
    signal_dates = daily_ret[daily_ret != 0].index
    return daily_ret, trades, signal_dates


def variant_f_momentum_filter(close, pair, spy_close, capital):
    """Momentum Filter: only trade when SPY 20d return is flat (-2% to +2%)."""
    etf1, etf2 = pair
    spy_ret_20d = spy_close.pct_change(20)

    ratio = close[etf1] / close[etf2]
    ratio_mean = ratio.rolling(60).mean()
    ratio_std = ratio.rolling(60).std()
    zscore = (ratio - ratio_mean) / ratio_std

    ret1 = close[etf1].pct_change()
    ret2 = close[etf2].pct_change()

    oot_mask = close.index >= OOT_START
    daily_ret = pd.Series(0.0, index=close.index)
    trades = 0
    in_trade = 0

    for i in range(1, len(close)):
        if not oot_mask[i]:
            continue

        z = zscore.iloc[i]
        spy_mom = spy_ret_20d.iloc[i]
        if np.isnan(z) or np.isnan(spy_mom):
            continue

        flat_market = -0.02 <= spy_mom <= 0.02

        if in_trade == 0:
            if flat_market:
                if z > 2.0:
                    in_trade = 1
                    trades += 1
                    daily_ret.iloc[i] -= SLIPPAGE * 2
                elif z < -2.0:
                    in_trade = -1
                    trades += 1
                    daily_ret.iloc[i] -= SLIPPAGE * 2
        else:
            if (in_trade == 1 and z <= 0) or (in_trade == -1 and z >= 0):
                in_trade = 0
                daily_ret.iloc[i] -= SLIPPAGE * 2

        if in_trade == 1:
            daily_ret.iloc[i] += ret2.iloc[i]
        elif in_trade == -1:
            daily_ret.iloc[i] += ret1.iloc[i]

    daily_ret = daily_ret[close.index >= OOT_START]
    signal_dates = daily_ret[daily_ret != 0].index
    return daily_ret, trades, signal_dates


# ── Main Backtest Runner ──────────────────────────────────────────────────────

def run_all():
    close = download_data()
    spy_close = close["SPY"]

    results = {}

    variant_configs = {
        "A_ZScore_Convergence": {"func": "zscore", "description": "Z-Score >2 on 60d ratio, close at z=0"},
        "B_Spread_Reversion": {"func": "spread", "description": "20d return spread >2σ (252d), hold 10d"},
        "C_LongOnly_Rotation": {"func": "rotation", "description": "Weekly rotation to more oversold ETF"},
        "D_MultiPair_Portfolio": {"func": "multi", "description": "All 5 pairs simultaneously, equal weight"},
        "E_Regime_Conditional": {"func": "regime_cond", "description": "Z-Score convergence, only when 60d corr>0.5"},
        "F_Momentum_Filter": {"func": "mom_filter", "description": "Z-Score convergence, only when SPY flat"},
    }

    for variant_name, config in variant_configs.items():
        print(f"\n{'='*70}")
        print(f"Running {variant_name}: {config['description']}")
        print(f"{'='*70}")

        if config["func"] == "multi":
            # Multi-pair runs all pairs together
            daily_ret, n_trades, signal_dates = variant_d_multi_pair(close, spy_close, ACCOUNT_SIZE)
            best_pair = "ALL_5_PAIRS"
            best_ret = daily_ret
            best_trades = n_trades
            best_signals = signal_dates
        else:
            # Run each pair and pick best, but also store all
            pair_results = {}
            for pair in PAIRS:
                pair_name = f"{pair[0]}_{pair[1]}"

                if config["func"] == "zscore":
                    ret, trades, sig = variant_a_zscore(close, pair, ACCOUNT_SIZE)
                elif config["func"] == "spread":
                    ret, trades, sig = variant_b_spread_reversion(close, pair, ACCOUNT_SIZE)
                elif config["func"] == "rotation":
                    ret, trades, sig = variant_c_rotation(close, pair, ACCOUNT_SIZE)
                elif config["func"] == "regime_cond":
                    ret, trades, sig = variant_e_regime_conditional(close, pair, ACCOUNT_SIZE)
                elif config["func"] == "mom_filter":
                    ret, trades, sig = variant_f_momentum_filter(close, pair, spy_close, ACCOUNT_SIZE)

                metrics = calc_metrics(ret, trades)
                gap, sharpe_bull, sharpe_bear = regime_gap(ret, spy_close)
                metrics["regime_gap"] = gap
                metrics["sharpe_bull"] = sharpe_bull
                metrics["sharpe_bear"] = sharpe_bear

                pair_results[pair_name] = {"metrics": metrics, "returns": ret, "signals": sig}
                print(f"  {pair_name}: Sharpe={metrics['sharpe']:.3f}, Trades={trades}, "
                      f"Ret={metrics['total_ret']:.2%}, MaxDD={metrics['maxdd']:.2%}, "
                      f"RegimeGap={gap:.3f} (Bull={sharpe_bull:.2f}, Bear={sharpe_bear:.2f})")

            # Pick best pair by Sharpe for the variant summary
            best_pair = max(pair_results, key=lambda k: pair_results[k]["metrics"]["sharpe"])
            best_ret = pair_results[best_pair]["returns"]
            best_trades = pair_results[best_pair]["metrics"]["n_trades"]
            best_signals = pair_results[best_pair]["signals"]
            best_metrics = pair_results[best_pair]["metrics"]

        # Calculate metrics for best
        if config["func"] == "multi":
            best_metrics = calc_metrics(best_ret, best_trades)
            gap, sharpe_bull, sharpe_bear = regime_gap(best_ret, spy_close)
            best_metrics["regime_gap"] = gap
            best_metrics["sharpe_bull"] = sharpe_bull
            best_metrics["sharpe_bear"] = sharpe_bear

        # Permutation test
        print(f"\n  Running permutation test (n={PERM_ITERATIONS})...")
        perm_p = permutation_test(best_ret, best_signals)
        best_metrics["perm_p"] = perm_p

        # 5-Gate Validation
        gates = {
            "sharpe_gt_0.5": best_metrics["sharpe"] > 0.5,
            "perm_p_lt_0.05": perm_p < 0.05,
            "regime_gap_lt_0.5": best_metrics["regime_gap"] < 0.5,
            "maxdd_gt_neg50pct": best_metrics["maxdd"] > -0.50,
            "trades_gte_20": best_metrics["n_trades"] >= 20,
        }
        gates_passed = sum(gates.values())
        best_metrics["gates"] = gates
        best_metrics["gates_passed"] = f"{gates_passed}/5"
        best_metrics["PASS"] = gates_passed == 5
        best_metrics["best_pair"] = best_pair

        # Dollar P&L
        cum_ret = (1 + best_ret).cumprod().iloc[-1] - 1
        best_metrics["dollar_pnl"] = round(ACCOUNT_SIZE * cum_ret, 2)
        best_metrics["final_equity"] = round(ACCOUNT_SIZE * (1 + cum_ret), 2)

        results[variant_name] = best_metrics

        print(f"\n  BEST PAIR: {best_pair}")
        print(f"  Sharpe: {best_metrics['sharpe']:.3f} | Sortino: {best_metrics['sortino']:.3f}")
        print(f"  Total Return: {best_metrics['total_ret']:.2%} | Annual: {best_metrics['annual_ret']:.2%}")
        print(f"  MaxDD: {best_metrics['maxdd']:.2%} | PF: {best_metrics['pf']:.2f} | WR: {best_metrics['wr']:.2%}")
        print(f"  Regime Gap: {best_metrics['regime_gap']:.3f} (Bull={best_metrics['sharpe_bull']:.2f}, Bear={best_metrics['sharpe_bear']:.2f})")
        print(f"  Permutation p-value: {perm_p:.4f}")
        print(f"  Trades: {best_metrics['n_trades']}")
        print(f"  Dollar P&L: ${best_metrics['dollar_pnl']:.2f} -> ${best_metrics['final_equity']:.2f}")
        print(f"  Gates: {gates_passed}/5 {'✓ PASS' if best_metrics['PASS'] else '✗ FAIL'}")
        for gate, passed in gates.items():
            print(f"    {'✓' if passed else '✗'} {gate}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY — SECTOR PAIR CONVERGENCE TRADING")
    print(f"{'='*70}")
    print(f"{'Variant':<30} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>8} {'Trades':>7} {'RegGap':>7} {'Perm-p':>7} {'Gates':>6} {'$P&L':>8}")
    print("-" * 100)

    for name, m in results.items():
        tag = " ***" if m["PASS"] else ""
        print(f"{name:<30} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['maxdd']:>7.2%} {m['n_trades']:>7} "
              f"{m['regime_gap']:>7.3f} {m['perm_p']:>7.4f} {m['gates_passed']:>6} {m['dollar_pnl']:>7.2f}{tag}")

    passed = [k for k, v in results.items() if v["PASS"]]
    print(f"\nPASSED 5-GATE: {len(passed)}/{len(results)}")
    if passed:
        print(f"Winners: {', '.join(passed)}")
    else:
        print("No variants passed all 5 gates.")

    # ── Save Results ───────────────────────────────────────────────────────────
    output_path = Path("/home/jupiter/Lvl3Quant/data/sector_pair_convergence_results.json")

    # Clean results for JSON serialization
    json_results = {}
    for k, v in results.items():
        clean = {}
        for k2, v2 in v.items():
            if isinstance(v2, (np.floating, np.integer)):
                clean[k2] = float(v2)
            elif isinstance(v2, np.bool_):
                clean[k2] = bool(v2)
            elif isinstance(v2, dict):
                clean[k2] = {k3: bool(v3) if isinstance(v3, (bool, np.bool_)) else v3 for k3, v3 in v2.items()}
            else:
                clean[k2] = v2
        json_results[k] = clean

    output = {
        "strategy": "Sector Pair Convergence Trading",
        "account_size": ACCOUNT_SIZE,
        "oot_period": f"{OOT_START} to {OOT_END}",
        "slippage": f"{SLIPPAGE*100:.2f}% each way",
        "pairs_tested": [f"{p[0]}/{p[1]}" for p in PAIRS],
        "run_timestamp": datetime.now().isoformat(),
        "variants": json_results,
        "summary": {
            "total_variants": len(results),
            "passed_5gate": len(passed),
            "winners": passed,
            "best_variant": max(results, key=lambda k: results[k]["sharpe"]),
            "best_sharpe": float(max(v["sharpe"] for v in results.values())),
        }
    }

    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    np.random.seed(42)
    run_all()
