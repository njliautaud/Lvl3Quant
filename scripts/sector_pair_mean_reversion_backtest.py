#!/usr/bin/env python3
"""
Sector Pair Mean Reversion Backtest
====================================
Trades the spread between related sector ETFs.
When spread z-score exceeds threshold, buy the laggard expecting reversion.

Pairs: XLK/XLC, XLF/XLV, XLE/XLI, XLY/XLP
Walk-forward OOT: Jan 2022 – Jul 2026
Account: $645, single position, $0 commission, 0.02% slippage.

6 Variants:
  A) Single Best Pair (XLK/XLC)
  B) All Pairs Combined
  C) Aggressive Threshold (z=1.5)
  D) Options Version (calls on underperformer)
  E) Regime-Filtered (SPY > 200-SMA only)
  F) XLY/XLP Only (Risk-On/Off)
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────
CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
COMMISSION = 0.0

OOT_START = "2022-01-01"
OOT_END = "2026-07-30"
DATA_START = "2021-06-01"  # extra for warmup

PAIRS = [
    ("XLK", "XLC", "Tech vs Comms"),
    ("XLF", "XLV", "Financials vs Healthcare"),
    ("XLE", "XLI", "Energy vs Industrials"),
    ("XLY", "XLP", "Discretionary vs Staples"),
]

RETURN_WINDOW = 20
ZSCORE_LOOKBACK = 60


# ── Data Download ──────────────────────────────────────────────────────────
def download_data():
    tickers = list(set(t for p in PAIRS for t in p[:2]) | {"SPY"})
    print(f"Downloading {tickers} ...")
    data = yf.download(tickers, start=DATA_START, end=OOT_END, auto_adjust=True, progress=False)
    close = data["Close"]
    close = close.ffill().dropna()
    print(f"  Data shape: {close.shape}, range: {close.index[0].date()} to {close.index[-1].date()}")
    return close


# ── Spread Calculation ─────────────────────────────────────────────────────
def calc_spread_zscore(close, etf_a, etf_b, ret_window=RETURN_WINDOW, z_lookback=ZSCORE_LOOKBACK):
    """Return spread z-score: positive = A outperforming B."""
    ret_a = close[etf_a].pct_change(ret_window)
    ret_b = close[etf_b].pct_change(ret_window)
    spread = ret_a - ret_b
    roll_mean = spread.rolling(z_lookback).mean()
    roll_std = spread.rolling(z_lookback).std()
    zscore = (spread - roll_mean) / roll_std.replace(0, np.nan)
    return zscore


# ── Single-Pair Backtester ─────────────────────────────────────────────────
def backtest_pair(close, etf_a, etf_b, zscore, entry_z=2.0, exit_z=0.0,
                  oot_start=OOT_START, capital=CAPITAL, regime_filter=False,
                  spy_sma=None):
    """
    When z > entry_z (A outperforming): buy B (the laggard), expecting reversion.
    When z < -entry_z (B outperforming): buy A (the laggard), expecting reversion.
    Exit when |z| crosses exit_z toward zero.
    """
    oot_mask = zscore.index >= pd.Timestamp(oot_start)
    dates = zscore.index[oot_mask]

    equity = capital
    position = None  # None or dict with {ticker, shares, entry_price, entry_date}
    trades = []
    equity_curve = []

    for dt in dates:
        z = zscore.loc[dt]
        if np.isnan(z):
            equity_curve.append(equity)
            continue

        # Regime filter
        if regime_filter and spy_sma is not None:
            if close["SPY"].loc[dt] < spy_sma.loc[dt]:
                # Bear market — skip new entries, but still manage exits
                if position is not None:
                    # Still exit if z crosses
                    cur_price = close[position["ticker"]].loc[dt]
                    if (position["side"] == "buy_b" and z <= exit_z) or \
                       (position["side"] == "buy_a" and z >= -exit_z):
                        exit_price = cur_price * (1 - SLIPPAGE_PCT)
                        pnl = (exit_price - position["entry_price"]) * position["shares"]
                        equity += pnl
                        trades.append({
                            "entry_date": position["entry_date"].isoformat(),
                            "exit_date": dt.isoformat(),
                            "ticker": position["ticker"],
                            "side": position["side"],
                            "entry_price": round(position["entry_price"], 4),
                            "exit_price": round(exit_price, 4),
                            "shares": position["shares"],
                            "pnl": round(pnl, 2),
                            "hold_days": (dt - position["entry_date"]).days,
                        })
                        position = None
                equity_curve.append(equity)
                continue

        # Exit logic
        if position is not None:
            cur_price = close[position["ticker"]].loc[dt]
            should_exit = False
            if position["side"] == "buy_b" and z <= exit_z:
                should_exit = True
            elif position["side"] == "buy_a" and z >= -exit_z:
                should_exit = True

            if should_exit:
                exit_price = cur_price * (1 - SLIPPAGE_PCT)
                pnl = (exit_price - position["entry_price"]) * position["shares"]
                equity += pnl
                trades.append({
                    "entry_date": position["entry_date"].isoformat(),
                    "exit_date": dt.isoformat(),
                    "ticker": position["ticker"],
                    "side": position["side"],
                    "entry_price": round(position["entry_price"], 4),
                    "exit_price": round(exit_price, 4),
                    "shares": position["shares"],
                    "pnl": round(pnl, 2),
                    "hold_days": (dt - position["entry_date"]).days,
                })
                position = None

        # Entry logic (only if flat)
        if position is None:
            if z > entry_z:
                # A outperforming → buy B (laggard)
                ticker = etf_b
                entry_price = close[ticker].loc[dt] * (1 + SLIPPAGE_PCT)
                shares = int(equity // entry_price)
                if shares > 0:
                    position = {
                        "ticker": ticker,
                        "shares": shares,
                        "entry_price": entry_price,
                        "entry_date": dt,
                        "side": "buy_b",
                    }
            elif z < -entry_z:
                # B outperforming → buy A (laggard)
                ticker = etf_a
                entry_price = close[ticker].loc[dt] * (1 + SLIPPAGE_PCT)
                shares = int(equity // entry_price)
                if shares > 0:
                    position = {
                        "ticker": ticker,
                        "shares": shares,
                        "entry_price": entry_price,
                        "entry_date": dt,
                        "side": "buy_a",
                    }

        equity_curve.append(equity)

    # Close any open position at end
    if position is not None:
        final_price = close[position["ticker"]].iloc[-1] * (1 - SLIPPAGE_PCT)
        pnl = (final_price - position["entry_price"]) * position["shares"]
        equity += pnl
        trades.append({
            "entry_date": position["entry_date"].isoformat(),
            "exit_date": dates[-1].isoformat(),
            "ticker": position["ticker"],
            "side": position["side"],
            "entry_price": round(position["entry_price"], 4),
            "exit_price": round(final_price, 4),
            "shares": position["shares"],
            "pnl": round(pnl, 2),
            "hold_days": (dates[-1] - position["entry_date"]).days,
        })
        equity_curve[-1] = equity

    return trades, equity_curve, dates


# ── Options Variant ────────────────────────────────────────────────────────
def backtest_pair_options(close, etf_a, etf_b, zscore, entry_z=2.0,
                          oot_start=OOT_START, capital=CAPITAL):
    """
    Buy calls on underperformer when z exceeds threshold.
    Hold for 15 trading days (2-3 weeks).
    Model: 3% premium cost, 5% spread cost on premium.
    Payoff: max(0, pct_move * leverage - premium_cost) where leverage ~ 5x for ATM calls.
    """
    oot_mask = zscore.index >= pd.Timestamp(oot_start)
    dates = zscore.index[oot_mask]

    equity = capital
    position = None
    trades = []
    equity_curve = []
    HOLD_DAYS = 15
    PREMIUM_PCT = 0.03
    SPREAD_COST_PCT = 0.05
    LEVERAGE = 5.0

    for i, dt in enumerate(dates):
        z = zscore.loc[dt]
        if np.isnan(z):
            equity_curve.append(equity)
            continue

        # Exit: hold period expired
        if position is not None:
            days_held = (dt - position["entry_date"]).days
            if days_held >= HOLD_DAYS:
                cur_price = close[position["ticker"]].loc[dt]
                pct_move = (cur_price - position["entry_price"]) / position["entry_price"]
                # Call payoff: leveraged upside, floored at -premium
                raw_payoff = pct_move * LEVERAGE
                net_payoff = max(-1.0, raw_payoff) - PREMIUM_PCT - (PREMIUM_PCT * SPREAD_COST_PCT)
                pnl = position["notional"] * net_payoff
                equity += pnl
                trades.append({
                    "entry_date": position["entry_date"].isoformat(),
                    "exit_date": dt.isoformat(),
                    "ticker": position["ticker"],
                    "side": position["side"],
                    "entry_price": round(position["entry_price"], 4),
                    "exit_price": round(cur_price, 4),
                    "notional": round(position["notional"], 2),
                    "pnl": round(pnl, 2),
                    "hold_days": days_held,
                    "pct_move": round(pct_move * 100, 2),
                })
                position = None

        # Entry
        if position is None:
            if z > entry_z:
                ticker = etf_b
                entry_price = close[ticker].loc[dt]
                notional = equity * 0.5  # risk 50% of capital on options
                position = {
                    "ticker": ticker,
                    "entry_price": entry_price,
                    "entry_date": dt,
                    "side": "call_on_b",
                    "notional": notional,
                }
            elif z < -entry_z:
                ticker = etf_a
                entry_price = close[ticker].loc[dt]
                notional = equity * 0.5
                position = {
                    "ticker": ticker,
                    "entry_price": entry_price,
                    "entry_date": dt,
                    "side": "call_on_a",
                    "notional": notional,
                }

        equity_curve.append(equity)

    # Close open
    if position is not None:
        cur_price = close[position["ticker"]].iloc[-1]
        pct_move = (cur_price - position["entry_price"]) / position["entry_price"]
        raw_payoff = pct_move * LEVERAGE
        net_payoff = max(-1.0, raw_payoff) - PREMIUM_PCT - (PREMIUM_PCT * SPREAD_COST_PCT)
        pnl = position["notional"] * net_payoff
        equity += pnl
        trades.append({
            "entry_date": position["entry_date"].isoformat(),
            "exit_date": dates[-1].isoformat(),
            "ticker": position["ticker"],
            "side": position["side"],
            "entry_price": round(position["entry_price"], 4),
            "exit_price": round(cur_price, 4),
            "notional": round(position["notional"], 2),
            "pnl": round(pnl, 2),
            "hold_days": (dates[-1] - position["entry_date"]).days,
        })
        equity_curve[-1] = equity

    return trades, equity_curve, dates


# ── Multi-Pair Combined ───────────────────────────────────────────────────
def backtest_multi_pair(close, pairs_data, entry_z=2.0, exit_z=0.0,
                        oot_start=OOT_START, capital=CAPITAL, regime_filter=False,
                        spy_sma=None):
    """Trade all pairs, rotating into whichever fires first. Single position at a time."""
    # Collect all z-scores
    all_dates = None
    for etf_a, etf_b, zscore in pairs_data:
        idx = zscore.index[zscore.index >= pd.Timestamp(oot_start)]
        if all_dates is None:
            all_dates = idx
        else:
            all_dates = all_dates.intersection(idx)

    equity = capital
    position = None
    trades = []
    equity_curve = []

    for dt in all_dates:
        # Regime filter
        if regime_filter and spy_sma is not None:
            if close["SPY"].loc[dt] < spy_sma.loc[dt]:
                if position is not None:
                    etf_a, etf_b, zscore = pairs_data[position["pair_idx"]]
                    z = zscore.loc[dt]
                    if (position["side"] == "buy_b" and z <= exit_z) or \
                       (position["side"] == "buy_a" and z >= -exit_z):
                        cur_price = close[position["ticker"]].loc[dt] * (1 - SLIPPAGE_PCT)
                        pnl = (cur_price - position["entry_price"]) * position["shares"]
                        equity += pnl
                        trades.append({
                            "entry_date": position["entry_date"].isoformat(),
                            "exit_date": dt.isoformat(),
                            "ticker": position["ticker"],
                            "pair": f"{etf_a}/{etf_b}",
                            "side": position["side"],
                            "pnl": round(pnl, 2),
                            "hold_days": (dt - position["entry_date"]).days,
                        })
                        position = None
                equity_curve.append(equity)
                continue

        # Exit check
        if position is not None:
            etf_a, etf_b, zscore = pairs_data[position["pair_idx"]]
            z = zscore.loc[dt]
            should_exit = False
            if position["side"] == "buy_b" and z <= exit_z:
                should_exit = True
            elif position["side"] == "buy_a" and z >= -exit_z:
                should_exit = True

            if should_exit:
                cur_price = close[position["ticker"]].loc[dt] * (1 - SLIPPAGE_PCT)
                pnl = (cur_price - position["entry_price"]) * position["shares"]
                equity += pnl
                trades.append({
                    "entry_date": position["entry_date"].isoformat(),
                    "exit_date": dt.isoformat(),
                    "ticker": position["ticker"],
                    "pair": f"{etf_a}/{etf_b}",
                    "side": position["side"],
                    "pnl": round(pnl, 2),
                    "hold_days": (dt - position["entry_date"]).days,
                })
                position = None

        # Entry: pick strongest z-score signal across all pairs
        if position is None:
            best_z = 0
            best_idx = -1
            best_side = None
            for pi, (etf_a, etf_b, zscore) in enumerate(pairs_data):
                z = zscore.loc[dt]
                if np.isnan(z):
                    continue
                if abs(z) > abs(best_z) and abs(z) > entry_z:
                    best_z = z
                    best_idx = pi
                    best_side = "buy_b" if z > 0 else "buy_a"

            if best_idx >= 0:
                etf_a, etf_b, _ = pairs_data[best_idx]
                ticker = etf_b if best_side == "buy_b" else etf_a
                entry_price = close[ticker].loc[dt] * (1 + SLIPPAGE_PCT)
                shares = int(equity // entry_price)
                if shares > 0:
                    position = {
                        "ticker": ticker,
                        "shares": shares,
                        "entry_price": entry_price,
                        "entry_date": dt,
                        "side": best_side,
                        "pair_idx": best_idx,
                    }

        equity_curve.append(equity)

    # Close open
    if position is not None:
        final_price = close[position["ticker"]].iloc[-1] * (1 - SLIPPAGE_PCT)
        pnl = (final_price - position["entry_price"]) * position["shares"]
        equity += pnl
        etf_a, etf_b, _ = pairs_data[position["pair_idx"]]
        trades.append({
            "entry_date": position["entry_date"].isoformat(),
            "exit_date": all_dates[-1].isoformat(),
            "ticker": position["ticker"],
            "pair": f"{etf_a}/{etf_b}",
            "side": position["side"],
            "pnl": round(pnl, 2),
            "hold_days": (all_dates[-1] - position["entry_date"]).days,
        })
        equity_curve[-1] = equity

    return trades, equity_curve, all_dates


# ── Metrics Calculation ───────────────────────────────────────────────────
def calc_metrics(trades, equity_curve, capital=CAPITAL):
    if not trades or len(equity_curve) < 2:
        return {
            "total_return_pct": 0, "sharpe": 0, "sortino": 0, "profit_factor": 0,
            "win_rate": 0, "num_trades": 0, "max_drawdown_pct": 0,
            "avg_hold_days": 0, "final_equity": capital,
        }

    eq = np.array(equity_curve, dtype=float)
    daily_ret = np.diff(eq) / eq[:-1]
    daily_ret = daily_ret[np.isfinite(daily_ret)]

    # Sharpe (annualized)
    if len(daily_ret) > 1 and np.std(daily_ret) > 0:
        sharpe = (np.mean(daily_ret) / np.std(daily_ret)) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Sortino
    downside = daily_ret[daily_ret < 0]
    if len(downside) > 0 and np.std(downside) > 0:
        sortino = (np.mean(daily_ret) / np.std(downside)) * np.sqrt(252)
    else:
        sortino = 0.0

    # Profit factor
    pnls = [t["pnl"] for t in trades]
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

    # Win rate
    wins = sum(1 for p in pnls if p > 0)
    win_rate = wins / len(pnls) if pnls else 0.0

    # Max drawdown
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = float(np.min(dd)) * 100

    # Avg hold days
    hold_days = [t.get("hold_days", 0) for t in trades]
    avg_hold = np.mean(hold_days) if hold_days else 0

    total_ret = (eq[-1] - capital) / capital * 100

    return {
        "total_return_pct": round(total_ret, 2),
        "final_equity": round(float(eq[-1]), 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(profit_factor, 3),
        "win_rate": round(win_rate * 100, 1),
        "num_trades": len(trades),
        "max_drawdown_pct": round(max_dd, 2),
        "avg_hold_days": round(avg_hold, 1),
    }


# ── Regime Analysis ───────────────────────────────────────────────────────
def regime_analysis(trades, close):
    spy_sma200 = close["SPY"].rolling(200).mean()
    bull_trades = []
    bear_trades = []

    for t in trades:
        entry_dt = pd.Timestamp(t["entry_date"])
        if entry_dt in spy_sma200.index:
            spy_val = close["SPY"].loc[entry_dt]
            sma_val = spy_sma200.loc[entry_dt]
            if not np.isnan(sma_val):
                if spy_val >= sma_val:
                    bull_trades.append(t)
                else:
                    bear_trades.append(t)

    def _regime_sharpe(rtrades):
        pnls = [t["pnl"] for t in rtrades]
        if len(pnls) < 2:
            return 0.0
        arr = np.array(pnls)
        if np.std(arr) == 0:
            return 0.0
        return float(np.mean(arr) / np.std(arr) * np.sqrt(len(arr)))

    bull_sharpe = _regime_sharpe(bull_trades)
    bear_sharpe = _regime_sharpe(bear_trades)
    max_sharpe = max(abs(bull_sharpe), abs(bear_sharpe))
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_sharpe if max_sharpe > 0 else 0.0

    return {
        "bull_trades": len(bull_trades),
        "bear_trades": len(bear_trades),
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "regime_gap": round(regime_gap, 3),
    }


# ── Permutation Test ──────────────────────────────────────────────────────
def permutation_test(trades, n_perms=1000):
    if len(trades) < 5:
        return {"perm_p_value": 1.0, "actual_mean_pnl": 0.0, "perm_mean_pnl": 0.0}

    pnls = np.array([t["pnl"] for t in trades])
    actual_mean = np.mean(pnls)

    rng = np.random.RandomState(42)
    count_better = 0
    perm_means = []
    for _ in range(n_perms):
        shuffled = pnls.copy()
        rng.shuffle(shuffled)
        # Shuffle signs to test if direction matters
        signs = rng.choice([-1, 1], size=len(pnls))
        perm_pnl = shuffled * signs
        pm = np.mean(perm_pnl)
        perm_means.append(pm)
        if pm >= actual_mean:
            count_better += 1

    p_value = count_better / n_perms

    return {
        "perm_p_value": round(p_value, 4),
        "actual_mean_pnl": round(float(actual_mean), 4),
        "perm_mean_pnl": round(float(np.mean(perm_means)), 4),
    }


# ── 5-Gate Validation ─────────────────────────────────────────────────────
def validate_5gate(metrics, regime, perm):
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": perm["perm_p_value"] < 0.05,
        "regime_gap_lt_0.5": regime["regime_gap"] < 0.5,
        "max_dd_gt_neg50": metrics["max_drawdown_pct"] > -50,
        "num_trades_gte_20": metrics["num_trades"] >= 20,
    }
    gates["all_passed"] = all(gates.values())
    return gates


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("SECTOR PAIR MEAN REVERSION BACKTEST")
    print(f"OOT: {OOT_START} to {OOT_END} | Capital: ${CAPITAL}")
    print("=" * 70)

    close = download_data()
    spy_sma200 = close["SPY"].rolling(200).mean()

    # Precompute z-scores for all pairs
    zscores = {}
    for etf_a, etf_b, label in PAIRS:
        zscores[(etf_a, etf_b)] = calc_spread_zscore(close, etf_a, etf_b)

    pairs_data = [(a, b, zscores[(a, b)]) for a, b, _ in PAIRS]

    results = {}

    # ── Variant A: Single Best Pair (XLK/XLC) ──────────────────────────
    print("\n[A] Single Best Pair: XLK/XLC, z=2.0")
    trades_a, eq_a, dates_a = backtest_pair(close, "XLK", "XLC", zscores[("XLK", "XLC")],
                                             entry_z=2.0, exit_z=0.0)
    m_a = calc_metrics(trades_a, eq_a)
    r_a = regime_analysis(trades_a, close)
    p_a = permutation_test(trades_a)
    g_a = validate_5gate(m_a, r_a, p_a)
    results["A_XLK_XLC_z2"] = {"metrics": m_a, "regime": r_a, "permutation": p_a, "gates": g_a,
                                 "description": "Single pair XLK/XLC, z-entry=2.0, z-exit=0"}
    print(f"  Trades: {m_a['num_trades']}, Return: {m_a['total_return_pct']}%, "
          f"Sharpe: {m_a['sharpe']}, WR: {m_a['win_rate']}%, MaxDD: {m_a['max_drawdown_pct']}%")
    print(f"  Gates: {'PASS' if g_a['all_passed'] else 'FAIL'} — {g_a}")

    # ── Variant B: All Pairs Combined ──────────────────────────────────
    print("\n[B] All Pairs Combined, z=2.0")
    trades_b, eq_b, dates_b = backtest_multi_pair(close, pairs_data, entry_z=2.0, exit_z=0.0)
    m_b = calc_metrics(trades_b, eq_b)
    r_b = regime_analysis(trades_b, close)
    p_b = permutation_test(trades_b)
    g_b = validate_5gate(m_b, r_b, p_b)
    results["B_all_pairs_z2"] = {"metrics": m_b, "regime": r_b, "permutation": p_b, "gates": g_b,
                                   "description": "All 4 pairs combined, strongest z-score wins, z=2.0"}
    print(f"  Trades: {m_b['num_trades']}, Return: {m_b['total_return_pct']}%, "
          f"Sharpe: {m_b['sharpe']}, WR: {m_b['win_rate']}%, MaxDD: {m_b['max_drawdown_pct']}%")
    print(f"  Gates: {'PASS' if g_b['all_passed'] else 'FAIL'} — {g_b}")

    # ── Variant C: Aggressive Threshold (z=1.5) ───────────────────────
    print("\n[C] All Pairs Aggressive, z=1.5")
    trades_c, eq_c, dates_c = backtest_multi_pair(close, pairs_data, entry_z=1.5, exit_z=0.0)
    m_c = calc_metrics(trades_c, eq_c)
    r_c = regime_analysis(trades_c, close)
    p_c = permutation_test(trades_c)
    g_c = validate_5gate(m_c, r_c, p_c)
    results["C_all_pairs_z1.5"] = {"metrics": m_c, "regime": r_c, "permutation": p_c, "gates": g_c,
                                     "description": "All 4 pairs, aggressive entry z=1.5"}
    print(f"  Trades: {m_c['num_trades']}, Return: {m_c['total_return_pct']}%, "
          f"Sharpe: {m_c['sharpe']}, WR: {m_c['win_rate']}%, MaxDD: {m_c['max_drawdown_pct']}%")
    print(f"  Gates: {'PASS' if g_c['all_passed'] else 'FAIL'} — {g_c}")

    # ── Variant D: Options Version ─────────────────────────────────────
    print("\n[D] Options (calls on underperformer), z=2.0, XLK/XLC")
    trades_d, eq_d, dates_d = backtest_pair_options(close, "XLK", "XLC", zscores[("XLK", "XLC")],
                                                     entry_z=2.0)
    m_d = calc_metrics(trades_d, eq_d)
    r_d = regime_analysis(trades_d, close)
    p_d = permutation_test(trades_d)
    g_d = validate_5gate(m_d, r_d, p_d)
    results["D_options_XLK_XLC"] = {"metrics": m_d, "regime": r_d, "permutation": p_d, "gates": g_d,
                                      "description": "Options: buy calls on underperformer, 3% premium, 5% spread, 5x leverage"}
    print(f"  Trades: {m_d['num_trades']}, Return: {m_d['total_return_pct']}%, "
          f"Sharpe: {m_d['sharpe']}, WR: {m_d['win_rate']}%, MaxDD: {m_d['max_drawdown_pct']}%")
    print(f"  Gates: {'PASS' if g_d['all_passed'] else 'FAIL'} — {g_d}")

    # ── Variant E: Regime-Filtered (bull only) ─────────────────────────
    print("\n[E] All Pairs Regime-Filtered (SPY > 200-SMA), z=2.0")
    trades_e, eq_e, dates_e = backtest_multi_pair(close, pairs_data, entry_z=2.0, exit_z=0.0,
                                                    regime_filter=True, spy_sma=spy_sma200)
    m_e = calc_metrics(trades_e, eq_e)
    r_e = regime_analysis(trades_e, close)
    p_e = permutation_test(trades_e)
    g_e = validate_5gate(m_e, r_e, p_e)
    results["E_regime_filtered_z2"] = {"metrics": m_e, "regime": r_e, "permutation": p_e, "gates": g_e,
                                         "description": "All pairs, bull-only filter (SPY>200-SMA), z=2.0"}
    print(f"  Trades: {m_e['num_trades']}, Return: {m_e['total_return_pct']}%, "
          f"Sharpe: {m_e['sharpe']}, WR: {m_e['win_rate']}%, MaxDD: {m_e['max_drawdown_pct']}%")
    print(f"  Gates: {'PASS' if g_e['all_passed'] else 'FAIL'} — {g_e}")

    # ── Variant F: XLY/XLP Only ────────────────────────────────────────
    print("\n[F] XLY/XLP Only (Risk-On/Off), z=2.0")
    trades_f, eq_f, dates_f = backtest_pair(close, "XLY", "XLP", zscores[("XLY", "XLP")],
                                             entry_z=2.0, exit_z=0.0)
    m_f = calc_metrics(trades_f, eq_f)
    r_f = regime_analysis(trades_f, close)
    p_f = permutation_test(trades_f)
    g_f = validate_5gate(m_f, r_f, p_f)
    results["F_XLY_XLP_z2"] = {"metrics": m_f, "regime": r_f, "permutation": p_f, "gates": g_f,
                                 "description": "XLY/XLP only, risk-on vs risk-off pair, z=2.0"}
    print(f"  Trades: {m_f['num_trades']}, Return: {m_f['total_return_pct']}%, "
          f"Sharpe: {m_f['sharpe']}, WR: {m_f['win_rate']}%, MaxDD: {m_f['max_drawdown_pct']}%")
    print(f"  Gates: {'PASS' if g_f['all_passed'] else 'FAIL'} — {g_f}")

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Variant':<30} {'Trades':>6} {'Return%':>8} {'Sharpe':>7} {'Sortino':>8} "
          f"{'PF':>6} {'WR%':>5} {'MaxDD%':>7} {'5-Gate':>7}")
    print("-" * 90)

    for key, val in results.items():
        m = val["metrics"]
        g = val["gates"]
        label = key.replace("_", " ")
        print(f"{label:<30} {m['num_trades']:>6} {m['total_return_pct']:>8.1f} {m['sharpe']:>7.3f} "
              f"{m['sortino']:>8.3f} {m['profit_factor']:>6.2f} {m['win_rate']:>5.1f} "
              f"{m['max_drawdown_pct']:>7.1f} {'PASS' if g['all_passed'] else 'FAIL':>7}")

    # ── Save Results ───────────────────────────────────────────────────
    output = {
        "strategy": "Sector Pair Mean Reversion",
        "run_date": datetime.now().isoformat(),
        "oot_period": f"{OOT_START} to {OOT_END}",
        "capital": CAPITAL,
        "pairs": [f"{a}/{b}" for a, b, _ in PAIRS],
        "parameters": {
            "return_window": RETURN_WINDOW,
            "zscore_lookback": ZSCORE_LOOKBACK,
            "slippage_pct": SLIPPAGE_PCT,
            "commission": COMMISSION,
        },
        "variants": results,
    }

    out_path = Path("/home/jupiter/Lvl3Quant/data/sector_pair_mean_reversion_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
