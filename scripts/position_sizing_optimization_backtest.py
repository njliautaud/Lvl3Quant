#!/usr/bin/env python3
"""
Position Sizing Optimization Backtest for Dual Signal D
========================================================
Tests 6 position sizing variants against the baseline fixed $200/trade strategy.
Strategy: Quality stocks, dual signal (>5% drop from 20d high + RSI<35 + first green after 3+ red days), hold 10 days.
OOT: Jan 2022 - Jul 2026. Starting capital: $645. Slippage: 0.02% each way.

Variants:
  A: Fixed $200/trade (baseline)
  B: Half-Kelly fraction (capped 33% equity)
  C: Volatility-adjusted ($200 * target_vol/stock_vol)
  D: Confidence-scaled (oversold depth -> $100-$300)
  E: Equity curve scaling ($250 above 20-trade MA, $100 below)
  F: Progressive (increase $20 per win, decrease $20 per loss, $100-$300)
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────
TICKERS = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP",
    "HD", "COST", "UNH", "LLY", "V", "MA", "ABBV", "MRK",
    "WMT", "AMZN", "GOOGL", "META"
]

START_DATE = "2021-11-01"  # extra lookback for 20-day high calc
OOT_START = "2022-01-01"
OOT_END = "2026-07-31"
STARTING_CAPITAL = 645.0
HOLD_DAYS = 10
SLIPPAGE_PCT = 0.0002  # 0.02% each way
RSI_PERIOD = 14
HIGH_LOOKBACK = 20
RED_DAYS_REQUIRED = 3
DROP_THRESHOLD = 0.05  # 5% from 20-day high
RSI_THRESHOLD = 35
N_PERMUTATIONS = 1000


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def download_data():
    """Download price data for all tickers."""
    print("Downloading price data...")
    all_data = {}
    for ticker in TICKERS:
        try:
            df = yf.download(ticker, start=START_DATE, end=OOT_END, progress=False, auto_adjust=True)
            if df is not None and len(df) > 50:
                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                all_data[ticker] = df
        except Exception as e:
            print(f"  Warning: Failed to download {ticker}: {e}")
    print(f"  Downloaded {len(all_data)}/{len(TICKERS)} tickers")
    return all_data


def generate_signals(all_data: dict) -> list:
    """
    Generate Dual Signal D entry signals.
    Conditions:
      1. Stock drops >5% from 20-day high
      2. RSI(14) < 35
      3. First green day after 3+ consecutive red days
    """
    signals = []
    for ticker, df in all_data.items():
        df = df.copy()
        df["rsi"] = compute_rsi(df["Close"], RSI_PERIOD)
        df["high_20d"] = df["High"].rolling(HIGH_LOOKBACK).max()
        df["drop_from_high"] = (df["high_20d"] - df["Close"]) / df["high_20d"]
        df["daily_return"] = df["Close"].pct_change()

        # Count consecutive red days
        is_red = (df["daily_return"] < 0).astype(int)
        consec_red = is_red.copy() * 0
        for i in range(1, len(consec_red)):
            if is_red.iloc[i-1] == 1:
                consec_red.iloc[i] = consec_red.iloc[i-1] + 1
            else:
                consec_red.iloc[i] = 0
        # Shift to get yesterday's consecutive red count
        df["prev_consec_red"] = consec_red.shift(1)
        df["is_green"] = df["daily_return"] > 0

        oot_mask = df.index >= OOT_START

        for i in range(1, len(df)):
            if not oot_mask[i]:
                continue
            row = df.iloc[i]
            if (pd.notna(row["drop_from_high"]) and row["drop_from_high"] > DROP_THRESHOLD
                    and pd.notna(row["rsi"]) and row["rsi"] < RSI_THRESHOLD
                    and row["is_green"]
                    and pd.notna(row["prev_consec_red"]) and row["prev_consec_red"] >= RED_DAYS_REQUIRED):

                entry_date = df.index[i]
                entry_price = float(row["Close"])
                rsi_val = float(row["rsi"])
                drop_val = float(row["drop_from_high"])

                # Calculate exit (hold 10 trading days)
                future_idx = i + HOLD_DAYS
                if future_idx < len(df):
                    exit_price = float(df["Close"].iloc[future_idx])
                    exit_date = df.index[future_idx]
                else:
                    # Use last available price
                    exit_price = float(df["Close"].iloc[-1])
                    exit_date = df.index[-1]

                # Annualized volatility of the stock (trailing 20 days)
                if i >= 20:
                    stock_vol = float(df["daily_return"].iloc[i-20:i].std() * np.sqrt(252))
                else:
                    stock_vol = 0.20  # default

                signals.append({
                    "ticker": ticker,
                    "entry_date": str(entry_date.date()) if hasattr(entry_date, 'date') else str(entry_date)[:10],
                    "exit_date": str(exit_date.date()) if hasattr(exit_date, 'date') else str(exit_date)[:10],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "rsi": rsi_val,
                    "drop_from_high": drop_val,
                    "stock_vol": max(stock_vol, 0.05),  # floor at 5%
                })

    # Sort by entry date
    signals.sort(key=lambda x: x["entry_date"])
    print(f"  Generated {len(signals)} signals across {len(set(s['ticker'] for s in signals))} tickers")
    return signals


# ── Position Sizing Functions ──────────────────────────────────────────────

def size_fixed(signal, equity, history, prog_state):
    """A: Fixed $200/trade"""
    return min(200.0, equity * 0.95)


def size_half_kelly(signal, equity, history, prog_state):
    """B: Half-Kelly, capped at 33% of equity"""
    if len(history) < 5:
        return min(200.0, equity * 0.33)
    wins = [t for t in history if t["pnl"] > 0]
    losses = [t for t in history if t["pnl"] <= 0]
    if not losses or not wins:
        return min(200.0, equity * 0.33)
    win_rate = len(wins) / len(history)
    avg_win = np.mean([t["pnl_pct"] for t in wins])
    avg_loss = abs(np.mean([t["pnl_pct"] for t in losses]))
    if avg_loss == 0:
        return min(200.0, equity * 0.33)
    payoff_ratio = avg_win / avg_loss
    kelly = win_rate - (1 - win_rate) / payoff_ratio
    half_kelly = max(kelly * 0.5, 0.02)  # floor at 2%
    size = equity * min(half_kelly, 0.33)
    return max(min(size, equity * 0.95), 50.0)


def size_vol_adjusted(signal, equity, history, prog_state):
    """C: Volatility-adjusted. $200 * (target_vol / stock_vol)"""
    target_vol = 0.15
    stock_vol = signal["stock_vol"]
    ratio = target_vol / stock_vol
    size = 200.0 * ratio
    size = max(min(size, 400.0), 50.0)  # cap range
    return min(size, equity * 0.95)


def size_confidence_scaled(signal, equity, history, prog_state):
    """D: Confidence-scaled based on oversold depth. Range $100-$300."""
    # Normalize drop (5%-20% range -> 0-1) and RSI (35-10 range -> 0-1)
    drop_score = min((signal["drop_from_high"] - 0.05) / 0.15, 1.0)
    rsi_score = min((RSI_THRESHOLD - signal["rsi"]) / 25.0, 1.0)
    confidence = (drop_score + rsi_score) / 2.0
    size = 100.0 + confidence * 200.0
    return min(size, equity * 0.95)


def size_equity_curve(signal, equity, history, prog_state):
    """E: Equity curve scaling. Above 20-trade MA -> $250, below -> $100."""
    if len(history) < 20:
        return min(200.0, equity * 0.95)
    # Compute equity at each trade end
    running_equity = STARTING_CAPITAL
    equity_points = []
    for t in history:
        running_equity += t["pnl"]
        equity_points.append(running_equity)
    ma_20 = np.mean(equity_points[-20:])
    current_eq = equity_points[-1]
    size = 250.0 if current_eq >= ma_20 else 100.0
    return min(size, equity * 0.95)


def size_progressive(signal, equity, history, prog_state):
    """F: Progressive. Start $100, +$20/win, -$20/loss. Range $100-$300."""
    current = prog_state.get("current_size", 100.0)
    if history:
        last = history[-1]
        if last["pnl"] > 0:
            current = min(current + 20.0, 300.0)
        else:
            current = max(current - 20.0, 100.0)
    prog_state["current_size"] = current
    return min(current, equity * 0.95)


SIZING_VARIANTS = {
    "A_fixed_200": size_fixed,
    "B_half_kelly": size_half_kelly,
    "C_vol_adjusted": size_vol_adjusted,
    "D_confidence_scaled": size_confidence_scaled,
    "E_equity_curve": size_equity_curve,
    "F_progressive": size_progressive,
}


# ── Backtest Engine ────────────────────────────────────────────────────────

def run_backtest(signals: list, sizing_fn, label: str = "") -> dict:
    """Run backtest with given position sizing function."""
    equity = STARTING_CAPITAL
    history = []
    prog_state = {}
    peak_equity = equity
    max_dd = 0.0
    equity_curve = [equity]

    for sig in signals:
        if equity < 50:
            break  # blown up

        position_size = sizing_fn(sig, equity, history, prog_state)
        shares = position_size / sig["entry_price"]
        if shares < 0.001:
            continue

        # Apply slippage
        effective_entry = sig["entry_price"] * (1 + SLIPPAGE_PCT)
        effective_exit = sig["exit_price"] * (1 - SLIPPAGE_PCT)

        pnl = shares * (effective_exit - effective_entry)
        pnl_pct = pnl / equity  # equity-relative return (sizing matters)

        equity += pnl
        peak_equity = max(peak_equity, equity)
        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
        max_dd = max(max_dd, dd)

        history.append({
            "ticker": sig["ticker"],
            "entry_date": sig["entry_date"],
            "exit_date": sig["exit_date"],
            "entry_price": sig["entry_price"],
            "exit_price": sig["exit_price"],
            "position_size": position_size,
            "shares": shares,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "equity_after": equity,
        })
        equity_curve.append(equity)

    return _compute_metrics(history, equity_curve, label)


def _compute_metrics(history: list, equity_curve: list, label: str) -> dict:
    """Compute performance metrics from trade history."""
    if not history:
        return {"label": label, "n_trades": 0, "total_return_pct": 0}

    pnls = [t["pnl"] for t in history]
    pnl_pcts = [t["pnl_pct"] for t in history]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_return = (equity_curve[-1] - STARTING_CAPITAL)
    total_return_pct = total_return / STARTING_CAPITAL * 100

    # Sharpe (annualized, assuming ~25 trades/year approximation)
    if len(pnl_pcts) > 1 and np.std(pnl_pcts) > 0:
        trades_per_year = max(len(pnl_pcts) / 4.5, 1)  # ~4.5 year OOT
        sharpe = (np.mean(pnl_pcts) / np.std(pnl_pcts)) * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    # Sortino
    downside = [p for p in pnl_pcts if p < 0]
    if downside and np.std(downside) > 0:
        sortino = (np.mean(pnl_pcts) / np.std(downside)) * np.sqrt(max(len(pnl_pcts) / 4.5, 1))
    else:
        sortino = float("inf") if np.mean(pnl_pcts) > 0 else 0.0

    # Max drawdown from equity curve
    peak = equity_curve[0]
    max_dd = 0
    for eq in equity_curve:
        peak = max(peak, eq)
        dd = (peak - eq) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    # Win rate, profit factor
    win_rate = len(wins) / len(pnls) * 100 if pnls else 0
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max consecutive losers
    max_consec_loss = 0
    current_streak = 0
    for p in pnls:
        if p <= 0:
            current_streak += 1
            max_consec_loss = max(max_consec_loss, current_streak)
        else:
            current_streak = 0

    # Largest single loss
    largest_loss = min(pnls) if pnls else 0

    # Regime analysis (bull vs bear based on year)
    bear_dates = set()  # 2022 is bear market
    bull_pnls = []
    bear_pnls = []
    for t in history:
        year = int(t["entry_date"][:4])
        if year == 2022:
            bear_pnls.append(t["pnl_pct"])
        else:
            bull_pnls.append(t["pnl_pct"])

    bull_sharpe = 0.0
    bear_sharpe = 0.0
    if len(bull_pnls) > 1 and np.std(bull_pnls) > 0:
        bull_sharpe = np.mean(bull_pnls) / np.std(bull_pnls) * np.sqrt(len(bull_pnls))
    if len(bear_pnls) > 1 and np.std(bear_pnls) > 0:
        bear_sharpe = np.mean(bear_pnls) / np.std(bear_pnls) * np.sqrt(len(bear_pnls))

    regime_gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 0.01)

    # Average position size
    avg_pos_size = np.mean([t["position_size"] for t in history])

    return {
        "label": label,
        "n_trades": len(pnls),
        "total_return_pct": round(total_return_pct, 2),
        "total_return_dollars": round(total_return, 2),
        "final_equity": round(equity_curve[-1], 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(profit_factor, 3),
        "max_consecutive_losers": max_consec_loss,
        "largest_single_loss": round(largest_loss, 2),
        "avg_position_size": round(avg_pos_size, 2),
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "regime_gap": round(regime_gap, 3),
        "n_bull_trades": len(bull_pnls),
        "n_bear_trades": len(bear_pnls),
        "equity_curve": [round(e, 2) for e in equity_curve],
        "pnl_series": [round(p, 4) for p in pnl_pcts],
    }


# ── 5-Gate Validation ──────────────────────────────────────────────────────

def validate_5_gates(metrics: dict) -> dict:
    """
    5-Gate validation:
      G1: Sharpe > 0.5
      G2: Profit Factor > 1.0
      G3: Max DD < 30%
      G4: Win Rate > 40%
      G5: Regime gap < 0.70 (bull vs bear Sharpe ratio divergence)
    """
    gates = {
        "G1_sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "G2_pf_gt_1.0": metrics["profit_factor"] > 1.0,
        "G3_maxdd_lt_30": metrics["max_drawdown_pct"] < 30.0,
        "G4_winrate_gt_40": metrics["win_rate_pct"] > 40.0,
        "G5_regime_gap_lt_0.70": metrics["regime_gap"] < 0.70,
    }
    gates["all_passed"] = all(gates.values())
    gates["gates_passed"] = sum(v for k, v in gates.items() if k.startswith("G"))
    return gates


# ── Permutation Test ───────────────────────────────────────────────────────

def permutation_test(pnl_series: list, n_perms: int = 1000) -> dict:
    """Test if mean return is significantly different from random."""
    if len(pnl_series) < 5:
        return {"p_value": 1.0, "significant_5pct": False, "observed_mean": 0.0}

    observed_mean = np.mean(pnl_series)
    pnl_arr = np.array(pnl_series)
    count_ge = 0

    rng = np.random.RandomState(42)
    for _ in range(n_perms):
        # Randomly flip signs
        signs = rng.choice([-1, 1], size=len(pnl_arr))
        perm_mean = np.mean(pnl_arr * signs)
        if perm_mean >= observed_mean:
            count_ge += 1

    p_value = count_ge / n_perms
    return {
        "p_value": round(p_value, 4),
        "significant_5pct": p_value < 0.05,
        "significant_10pct": p_value < 0.10,
        "observed_mean_pct": round(observed_mean * 100, 4),
        "n_permutations": n_perms,
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("POSITION SIZING OPTIMIZATION BACKTEST — Dual Signal D")
    print("=" * 80)
    print(f"OOT Period: {OOT_START} to {OOT_END}")
    print(f"Starting Capital: ${STARTING_CAPITAL}")
    print(f"Hold Period: {HOLD_DAYS} trading days")
    print(f"Slippage: {SLIPPAGE_PCT*100:.2f}% each way")
    print(f"Tickers: {len(TICKERS)} quality stocks")
    print()

    # Download data
    all_data = download_data()

    # Generate signals
    signals = generate_signals(all_data)
    if not signals:
        print("ERROR: No signals generated. Exiting.")
        return

    print(f"\n{'='*80}")
    print("RUNNING 6 POSITION SIZING VARIANTS")
    print(f"{'='*80}\n")

    results = {}
    for label, sizing_fn in SIZING_VARIANTS.items():
        print(f"  Running {label}...")
        metrics = run_backtest(signals, sizing_fn, label)
        gates = validate_5_gates(metrics)
        perm = permutation_test(metrics.get("pnl_series", []), N_PERMUTATIONS)

        results[label] = {
            "metrics": metrics,
            "gates": gates,
            "permutation": perm,
        }

    # ── Print Comparison Table ─────────────────────────────────────────────
    baseline = results["A_fixed_200"]["metrics"]

    print(f"\n{'='*80}")
    print("COMPARISON TABLE vs BASELINE (A: Fixed $200)")
    print(f"{'='*80}")

    header = f"{'Variant':<22} {'Trades':>6} {'Return%':>8} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>7} {'WR%':>6} {'PF':>6} {'AvgSz':>7} {'Gates':>5} {'p-val':>6}"
    print(header)
    print("-" * len(header))

    for label, data in results.items():
        m = data["metrics"]
        g = data["gates"]
        p = data["permutation"]
        marker = " *" if label == "A_fixed_200" else ""
        sort_str = f"{m['sortino']:.2f}" if m['sortino'] != float('inf') else "Inf"
        print(f"{label:<22} {m['n_trades']:>6} {m['total_return_pct']:>7.1f}% {m['sharpe']:>7.3f} {sort_str:>8} {m['max_drawdown_pct']:>6.1f}% {m['win_rate_pct']:>5.1f}% {m['profit_factor']:>6.2f} ${m['avg_position_size']:>5.0f} {g['gates_passed']:>3}/5 {p['p_value']:>6.3f}{marker}")

    # Delta vs baseline
    print(f"\n{'='*80}")
    print("DELTA vs BASELINE (A)")
    print(f"{'='*80}")
    print(f"{'Variant':<22} {'dReturn%':>9} {'dSharpe':>8} {'dMaxDD':>7} {'dWR%':>6} {'dPF':>7}")
    print("-" * 60)

    for label, data in results.items():
        if label == "A_fixed_200":
            continue
        m = data["metrics"]
        dr = m["total_return_pct"] - baseline["total_return_pct"]
        ds = m["sharpe"] - baseline["sharpe"]
        dd = m["max_drawdown_pct"] - baseline["max_drawdown_pct"]
        dw = m["win_rate_pct"] - baseline["win_rate_pct"]
        dp = m["profit_factor"] - baseline["profit_factor"]
        print(f"{label:<22} {dr:>+8.1f}% {ds:>+8.3f} {dd:>+6.1f}% {dw:>+5.1f}% {dp:>+7.2f}")

    # Regime breakdown
    print(f"\n{'='*80}")
    print("REGIME BREAKDOWN (Bull=2023-2026 vs Bear=2022)")
    print(f"{'='*80}")
    print(f"{'Variant':<22} {'Bull#':>5} {'Bull Sharpe':>12} {'Bear#':>5} {'Bear Sharpe':>12} {'Gap':>6}")
    print("-" * 65)
    for label, data in results.items():
        m = data["metrics"]
        print(f"{label:<22} {m['n_bull_trades']:>5} {m['bull_sharpe']:>12.3f} {m['n_bear_trades']:>5} {m['bear_sharpe']:>12.3f} {m['regime_gap']:>6.3f}")

    # Risk metrics
    print(f"\n{'='*80}")
    print("RISK METRICS")
    print(f"{'='*80}")
    print(f"{'Variant':<22} {'MaxConsecLoss':>14} {'LargestLoss$':>13} {'MaxDD%':>7} {'Final$':>8}")
    print("-" * 66)
    for label, data in results.items():
        m = data["metrics"]
        print(f"{label:<22} {m['max_consecutive_losers']:>14} {m['largest_single_loss']:>12.2f} {m['max_drawdown_pct']:>6.1f}% ${m['final_equity']:>7.2f}")

    # Gate details
    print(f"\n{'='*80}")
    print("5-GATE VALIDATION DETAILS")
    print(f"{'='*80}")
    for label, data in results.items():
        g = data["gates"]
        status = "PASS" if g["all_passed"] else "FAIL"
        fails = [k for k, v in g.items() if k.startswith("G") and not v]
        fail_str = f" (failed: {', '.join(fails)})" if fails else ""
        print(f"  {label:<22} {g['gates_passed']}/5 gates — {status}{fail_str}")

    # Permutation test
    print(f"\n{'='*80}")
    print(f"PERMUTATION TEST ({N_PERMUTATIONS} iterations)")
    print(f"{'='*80}")
    for label, data in results.items():
        p = data["permutation"]
        sig = "YES" if p["significant_5pct"] else ("marginal" if p.get("significant_10pct") else "NO")
        print(f"  {label:<22} p={p['p_value']:.4f}  significant@5%: {sig}  mean_return: {p['observed_mean_pct']:.3f}%")

    # Best variant recommendation
    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print(f"{'='*80}")

    # Rank by Sharpe among those passing all gates
    passing = [(label, data) for label, data in results.items() if data["gates"]["all_passed"]]
    if passing:
        passing.sort(key=lambda x: x[1]["metrics"]["sharpe"], reverse=True)
        best_label, best_data = passing[0]
        bm = best_data["metrics"]
        print(f"  Best variant (by Sharpe, all gates passed): {best_label}")
        print(f"    Sharpe: {bm['sharpe']:.3f}  |  Sortino: {bm['sortino']:.3f}  |  Return: {bm['total_return_pct']:.1f}%  |  MaxDD: {bm['max_drawdown_pct']:.1f}%")
        if best_label != "A_fixed_200":
            improvement = bm["sharpe"] - baseline["sharpe"]
            print(f"    Sharpe improvement over baseline: {improvement:+.3f}")
        else:
            print(f"    Baseline remains best — no sizing variant improved risk-adjusted returns.")
    else:
        print("  WARNING: No variant passed all 5 gates.")
        # Show best by Sharpe anyway
        all_sorted = sorted(results.items(), key=lambda x: x[1]["metrics"]["sharpe"], reverse=True)
        best_label, best_data = all_sorted[0]
        print(f"  Best by Sharpe (gates not all passed): {best_label} — Sharpe {best_data['metrics']['sharpe']:.3f}")

    # ── Save results ───────────────────────────────────────────────────────
    output_path = Path("/home/jupiter/Lvl3Quant/data/position_sizing_results.json")

    # Prepare serializable output
    output = {
        "metadata": {
            "strategy": "Dual Signal D",
            "oot_period": f"{OOT_START} to {OOT_END}",
            "starting_capital": STARTING_CAPITAL,
            "hold_days": HOLD_DAYS,
            "slippage_pct": SLIPPAGE_PCT,
            "n_tickers": len(TICKERS),
            "tickers": TICKERS,
            "n_signals": len(signals),
            "n_permutations": N_PERMUTATIONS,
            "run_timestamp": datetime.now().isoformat(),
        },
        "variants": {},
    }

    for label, data in results.items():
        # Remove equity_curve and pnl_series from saved metrics to keep file manageable
        metrics_save = {k: v for k, v in data["metrics"].items() if k not in ("equity_curve", "pnl_series")}
        output["variants"][label] = {
            "metrics": metrics_save,
            "gates": data["gates"],
            "permutation": data["permutation"],
        }

    # Add recommendation
    if passing:
        output["recommendation"] = {
            "best_variant": passing[0][0],
            "sharpe": passing[0][1]["metrics"]["sharpe"],
            "all_gates_passed": True,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Custom encoder to handle numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
