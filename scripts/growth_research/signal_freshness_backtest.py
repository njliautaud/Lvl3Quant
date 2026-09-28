#!/usr/bin/env python3
"""
Signal Freshness Backtest — Burst vs Persistent Signals on Sector ETFs
======================================================================
Tests whether "sudden burst" signals (0-1→3+ indicators in one day) outperform
"persistent" signals (3+ indicators for 2+ consecutive days).

Uses SLIDING windows (HC #0), realistic costs, and 5-gate validation.
"""

import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ─── Configuration ───────────────────────────────────────────────────────────

SECTOR_ETFS = ["XLE", "XLU", "XLF", "XLP", "XLK", "XLC", "XLV", "XLB", "XLRE", "XLY", "XLI"]
REGIME_PROXY = "SPY"
START_DATE = "2020-01-01"
END_DATE = "2026-08-15"
HOLDING_PERIODS = [1, 3, 5, 10]
MIN_TRADES = 30
COST_PCT = 0.0038  # ~0.38% RT commission on ~$1250 option contract
PERM_SHUFFLES = 1000
SHARPE_GATE = 0.5
REGIME_GAP_GATE = 0.50
PERM_P_GATE = 0.05
MDD_GATE = 0.50  # 50%

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Data Loading ────────────────────────────────────────────────────────────

def download_data():
    """Download sector ETF + SPY data via yfinance."""
    tickers = SECTOR_ETFS + [REGIME_PROXY]
    print(f"Downloading data for {len(tickers)} tickers: {', '.join(tickers)}")

    data = yf.download(tickers, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

    # Handle multi-level columns from yfinance
    close = data["Close"] if "Close" in data.columns.get_level_values(0) else data[("Close",)]
    volume = data["Volume"] if "Volume" in data.columns.get_level_values(0) else data[("Volume",)]
    high = data["High"] if "High" in data.columns.get_level_values(0) else data[("High",)]
    low = data["Low"] if "Low" in data.columns.get_level_values(0) else data[("Low",)]

    print(f"  Data range: {close.index[0].strftime('%Y-%m-%d')} to {close.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Trading days: {len(close)}")

    return close, volume, high, low


# ─── Signal Generation ───────────────────────────────────────────────────────

def compute_rsi(series, period=14):
    """Standard RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series, fast=12, slow=26, signal=9):
    """MACD crossover detection."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def generate_indicator_signals(close, volume):
    """
    Generate 6 indicator signals for each ticker.
    Each indicator = 1 'source'. Returns DataFrame of source counts per ticker per day.
    """
    results = {}

    for ticker in SECTOR_ETFS:
        if ticker not in close.columns:
            continue

        px = close[ticker].dropna()
        vol = volume[ticker].dropna()

        # Align
        common_idx = px.index.intersection(vol.index)
        px = px.loc[common_idx]
        vol = vol.loc[common_idx]

        signals = pd.DataFrame(index=px.index)

        # 1. RSI < 30 (oversold → bullish signal)
        rsi = compute_rsi(px, 14)
        signals["rsi_oversold"] = (rsi < 30).astype(int)

        # 2. MACD bullish crossover (MACD crosses above signal line today)
        macd_line, signal_line = compute_macd(px)
        macd_cross = ((macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1)))
        # Keep signal active for 3 days after crossover
        signals["macd_cross"] = macd_cross.rolling(3, min_periods=1).max().fillna(0).astype(int)

        # 3. 21d momentum turning positive (negative yesterday, positive today)
        mom_21 = px.pct_change(21)
        mom_turn = ((mom_21 > 0) & (mom_21.shift(1) <= 0))
        signals["mom_21_turn"] = mom_turn.rolling(3, min_periods=1).max().fillna(0).astype(int)

        # 4. 5d momentum positive
        mom_5 = px.pct_change(5)
        signals["mom_5_pos"] = (mom_5 > 0).astype(int)

        # 5. Price > 200 SMA
        sma_200 = px.rolling(200, min_periods=200).mean()
        signals["above_sma200"] = (px > sma_200).astype(int)

        # 6. Volume surge (> 1.5x 20d average)
        vol_avg = vol.rolling(20, min_periods=20).mean()
        signals["vol_surge"] = (vol > 1.5 * vol_avg).astype(int)

        # Total source count
        results[ticker] = signals.sum(axis=1)

    source_counts = pd.DataFrame(results)
    return source_counts


def classify_signals(source_counts):
    """
    Classify each ticker-day as burst, persistent, or neither.

    Burst: 0-1 sources yesterday, 3+ sources today
    Persistent: 3+ sources for 2+ consecutive days
    """
    burst_signals = []
    persistent_signals = []

    for ticker in source_counts.columns:
        counts = source_counts[ticker].dropna()
        prev_counts = counts.shift(1)

        for i in range(1, len(counts)):
            date = counts.index[i]
            today = counts.iloc[i]
            yesterday = prev_counts.iloc[i]

            if np.isnan(yesterday):
                continue

            # Burst: 0-1 → 3+
            if yesterday <= 1 and today >= 3:
                burst_signals.append({
                    "date": date,
                    "ticker": ticker,
                    "sources_today": int(today),
                    "sources_yesterday": int(yesterday),
                    "type": "burst"
                })

            # Persistent: 3+ for 2+ days (today and yesterday both 3+)
            if today >= 3 and yesterday >= 3:
                persistent_signals.append({
                    "date": date,
                    "ticker": ticker,
                    "sources_today": int(today),
                    "sources_yesterday": int(yesterday),
                    "type": "persistent"
                })

    return burst_signals, persistent_signals


def classify_staleness(source_counts):
    """
    Track staleness decay: how does the signal perform on day 1, 2, 3, 5 of
    consecutive 3+ source activation?
    """
    staleness_signals = {d: [] for d in [1, 2, 3, 5]}

    for ticker in source_counts.columns:
        counts = source_counts[ticker].dropna()

        # Find runs of 3+ sources
        active = (counts >= 3).astype(int)

        # Track consecutive days
        streak = 0
        for i in range(len(active)):
            if active.iloc[i] == 1:
                streak += 1
                if streak in staleness_signals:
                    staleness_signals[streak].append({
                        "date": counts.index[i],
                        "ticker": ticker,
                        "streak_day": streak,
                        "sources": int(counts.iloc[i])
                    })
            else:
                streak = 0

    return staleness_signals


# ─── Forward Returns ─────────────────────────────────────────────────────────

def compute_forward_returns(signals, close, holding_periods=HOLDING_PERIODS):
    """
    Compute forward returns and MFE for each signal.
    Uses SLIDING approach — each signal is evaluated independently.
    """
    enriched = []

    for sig in signals:
        ticker = sig["ticker"]
        date = sig["date"]

        if ticker not in close.columns:
            continue

        px = close[ticker]

        try:
            loc = px.index.get_loc(date)
        except KeyError:
            continue

        entry_price = px.iloc[loc]
        if np.isnan(entry_price) or entry_price <= 0:
            continue

        sig_enriched = dict(sig)
        sig_enriched["entry_price"] = float(entry_price)

        max_hp = max(holding_periods)
        if loc + max_hp >= len(px):
            continue

        for hp in holding_periods:
            exit_price = px.iloc[loc + hp]
            if np.isnan(exit_price):
                sig_enriched[f"ret_{hp}d"] = np.nan
                sig_enriched[f"ret_{hp}d_raw"] = np.nan
                sig_enriched[f"mfe_{hp}d"] = np.nan
                continue
            raw_ret = (exit_price / entry_price) - 1.0
            net_ret = raw_ret - COST_PCT  # deduct RT cost
            sig_enriched[f"ret_{hp}d"] = float(net_ret)
            sig_enriched[f"ret_{hp}d_raw"] = float(raw_ret)

            # MFE within holding period
            future_prices = px.iloc[loc + 1 : loc + hp + 1]
            if len(future_prices) > 0:
                mfe = (future_prices.max() / entry_price) - 1.0
                sig_enriched[f"mfe_{hp}d"] = float(mfe)
            else:
                sig_enriched[f"mfe_{hp}d"] = 0.0

        enriched.append(sig_enriched)

    return enriched


# ─── Statistics ──────────────────────────────────────────────────────────────

def compute_stats(trades, hp):
    """Compute Sharpe, win rate, mean return, MFE for a given holding period."""
    if len(trades) < MIN_TRADES:
        return None

    rets = np.array([t[f"ret_{hp}d"] for t in trades if not np.isnan(t.get(f"ret_{hp}d", np.nan))])
    raw_rets = np.array([t[f"ret_{hp}d_raw"] for t in trades if not np.isnan(t.get(f"ret_{hp}d_raw", np.nan))])
    mfes = np.array([t[f"mfe_{hp}d"] for t in trades if not np.isnan(t.get(f"mfe_{hp}d", np.nan))])

    # Filter NaN/Inf
    valid = np.isfinite(rets)
    rets = rets[valid]
    raw_rets = raw_rets[valid[:len(raw_rets)]] if len(raw_rets) == len(valid) else raw_rets[np.isfinite(raw_rets)]
    mfes = mfes[np.isfinite(mfes)]

    if len(rets) < MIN_TRADES:
        return None

    mean_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1) if len(rets) > 1 else 1e-9

    # Annualize: assume 252/hp trades per year
    trades_per_year = 252 / hp
    sharpe = (mean_ret / max(std_ret, 1e-9)) * np.sqrt(trades_per_year)

    # Sortino
    downside = rets[rets < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = (mean_ret / max(downside_std, 1e-9)) * np.sqrt(trades_per_year)

    # Win rate
    wr = np.mean(rets > 0)

    # Profit factor
    gains = rets[rets > 0].sum()
    losses = abs(rets[rets < 0].sum())
    pf = gains / max(losses, 1e-9)

    # Max drawdown (sequential equity curve)
    equity = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    mdd = abs(dd.min())

    return {
        "n_trades": len(trades),
        "mean_ret_pct": float(mean_ret * 100),
        "mean_ret_raw_pct": float(np.mean(raw_rets) * 100),
        "std_ret_pct": float(std_ret * 100),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "win_rate": float(wr),
        "profit_factor": float(pf),
        "max_drawdown": float(mdd),
        "mean_mfe_pct": float(np.mean(mfes) * 100),
        "median_mfe_pct": float(np.median(mfes) * 100),
    }


# ─── Validation Gates ────────────────────────────────────────────────────────

def permutation_test(trades, hp, n_shuffles=PERM_SHUFFLES):
    """Permutation test: shuffle returns, compute Sharpe, get p-value."""
    if len(trades) < MIN_TRADES:
        return 1.0

    rets = np.array([t[f"ret_{hp}d"] for t in trades if not np.isnan(t.get(f"ret_{hp}d", np.nan))])
    rets = rets[np.isfinite(rets)]
    if len(rets) < MIN_TRADES:
        return 1.0
    trades_per_year = 252 / hp

    observed_sharpe = (np.mean(rets) / max(np.std(rets, ddof=1), 1e-9)) * np.sqrt(trades_per_year)

    count_better = 0
    rng = np.random.RandomState(42)

    for _ in range(n_shuffles):
        shuffled = rng.permutation(rets)
        shuf_sharpe = (np.mean(shuffled) / max(np.std(shuffled, ddof=1), 1e-9)) * np.sqrt(trades_per_year)
        if shuf_sharpe >= observed_sharpe:
            count_better += 1

    return count_better / n_shuffles


def regime_stratification(trades, spy_returns, hp):
    """
    Split trades into bull/bear regimes based on SPY trailing 21d return.
    Check regime gap.
    """
    if len(trades) < MIN_TRADES:
        return None

    bull_trades = []
    bear_trades = []

    spy_mom = spy_returns.rolling(21).sum()

    for t in trades:
        date = t["date"]
        if date in spy_mom.index:
            regime = "bull" if spy_mom.loc[date] > 0 else "bear"
        else:
            # Find nearest
            nearest = spy_mom.index[spy_mom.index.get_indexer([date], method="nearest")[0]]
            regime = "bull" if spy_mom.loc[nearest] > 0 else "bear"

        if regime == "bull":
            bull_trades.append(t)
        else:
            bear_trades.append(t)

    result = {"bull_n": len(bull_trades), "bear_n": len(bear_trades)}

    trades_per_year = 252 / hp

    for label, subset in [("bull", bull_trades), ("bear", bear_trades)]:
        if len(subset) >= 10:
            rets = np.array([t[f"ret_{hp}d"] for t in subset])
            sharpe = (np.mean(rets) / max(np.std(rets, ddof=1), 1e-9)) * np.sqrt(trades_per_year)
            result[f"{label}_sharpe"] = float(sharpe)
            result[f"{label}_wr"] = float(np.mean(rets > 0))
        else:
            result[f"{label}_sharpe"] = None
            result[f"{label}_wr"] = None

    # Regime gap
    if result["bull_sharpe"] is not None and result["bear_sharpe"] is not None:
        gap = abs(result["bull_sharpe"] - result["bear_sharpe"]) / max(
            abs(result["bull_sharpe"]), abs(result["bear_sharpe"]), 1e-9
        )
        result["regime_gap"] = float(gap)
    else:
        result["regime_gap"] = None

    return result


def apply_5_gates(stats, perm_p, regime_result):
    """Apply the 5-gate validation. Returns dict of gate results."""
    gates = {}

    if stats is None:
        return {"all_passed": False, "reason": "insufficient trades"}

    # Gate 1: Sharpe > 0.5
    gates["sharpe_gate"] = stats["sharpe"] > SHARPE_GATE
    gates["sharpe_value"] = stats["sharpe"]

    # Gate 2: Regime gap < 0.50
    if regime_result and regime_result.get("regime_gap") is not None:
        gates["regime_gate"] = regime_result["regime_gap"] < REGIME_GAP_GATE
        gates["regime_gap"] = regime_result["regime_gap"]
    else:
        gates["regime_gate"] = False
        gates["regime_gap"] = None

    # Gate 3: Permutation p < 0.05
    gates["perm_gate"] = perm_p < PERM_P_GATE
    gates["perm_p"] = perm_p

    # Gate 4: Min 30 trades
    gates["min_trades_gate"] = stats["n_trades"] >= MIN_TRADES
    gates["n_trades"] = stats["n_trades"]

    # Gate 5: MDD < 50%
    gates["mdd_gate"] = stats["max_drawdown"] < MDD_GATE
    gates["mdd"] = stats["max_drawdown"]

    gates["all_passed"] = all([
        gates["sharpe_gate"], gates["regime_gate"], gates["perm_gate"],
        gates["min_trades_gate"], gates["mdd_gate"]
    ])

    return gates


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("SIGNAL FRESHNESS BACKTEST — Burst vs Persistent on Sector ETFs")
    print("=" * 80)
    print(f"  Tickers: {', '.join(SECTOR_ETFS)}")
    print(f"  Period: {START_DATE} to {END_DATE}")
    print(f"  Cost: {COST_PCT*100:.2f}% RT")
    print(f"  Holding periods: {HOLDING_PERIODS}")
    print(f"  Validation: 5-gate (Sharpe>{SHARPE_GATE}, regime gap<{REGIME_GAP_GATE}, "
          f"perm p<{PERM_P_GATE}, min {MIN_TRADES} trades, MDD<{MDD_GATE*100:.0f}%)")
    print()

    # Download data
    close, volume, high, low = download_data()

    # SPY returns for regime
    spy_returns = close[REGIME_PROXY].pct_change().dropna()

    # Generate indicator signals
    print("\nGenerating multi-indicator signals...")
    source_counts = generate_indicator_signals(close, volume)
    print(f"  Source count matrix: {source_counts.shape[0]} days x {source_counts.shape[1]} tickers")

    # Classify signals
    print("\nClassifying burst vs persistent signals...")
    burst_signals, persistent_signals = classify_signals(source_counts)
    print(f"  Burst signals (0-1→3+): {len(burst_signals)}")
    print(f"  Persistent signals (3+ for 2+ days): {len(persistent_signals)}")

    # Staleness decay
    print("\nClassifying staleness decay...")
    staleness_signals = classify_staleness(source_counts)
    for day, sigs in staleness_signals.items():
        print(f"  Day {day} of streak: {len(sigs)} signals")

    # Compute forward returns
    print("\nComputing forward returns...")
    burst_trades = compute_forward_returns(burst_signals, close)
    persistent_trades = compute_forward_returns(persistent_signals, close)
    staleness_trades = {d: compute_forward_returns(sigs, close) for d, sigs in staleness_signals.items()}

    print(f"  Burst trades with returns: {len(burst_trades)}")
    print(f"  Persistent trades with returns: {len(persistent_trades)}")

    # ─── Results ─────────────────────────────────────────────────────────

    all_results = {
        "metadata": {
            "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tickers": SECTOR_ETFS,
            "period": f"{START_DATE} to {END_DATE}",
            "cost_pct": COST_PCT,
            "sliding_window": True,
            "perm_shuffles": PERM_SHUFFLES,
        },
        "burst": {},
        "persistent": {},
        "staleness_decay": {},
        "comparison": {},
    }

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    for label, trades in [("BURST (0-1→3+)", burst_trades), ("PERSISTENT (3+ for 2+ days)", persistent_trades)]:
        key = "burst" if "BURST" in label else "persistent"
        print(f"\n{'─' * 60}")
        print(f"  {label}  |  N = {len(trades)}")
        print(f"{'─' * 60}")

        for hp in HOLDING_PERIODS:
            stats = compute_stats(trades, hp)
            if stats is None:
                print(f"  {hp}d: INSUFFICIENT DATA (< {MIN_TRADES} trades)")
                all_results[key][f"{hp}d"] = {"status": "insufficient_data"}
                continue

            perm_p = permutation_test(trades, hp)
            regime = regime_stratification(trades, spy_returns, hp)
            gates = apply_5_gates(stats, perm_p, regime)

            passed = "PASS" if gates["all_passed"] else "FAIL"
            gate_details = []
            for g in ["sharpe_gate", "regime_gate", "perm_gate", "min_trades_gate", "mdd_gate"]:
                gate_details.append("+" if gates.get(g) else "-")

            print(f"\n  {hp}d hold | {passed} [{'/'.join(gate_details)}]")
            print(f"    N={stats['n_trades']}  Sharpe={stats['sharpe']:.2f}  "
                  f"Sortino={stats['sortino']:.2f}  WR={stats['win_rate']:.1%}  "
                  f"PF={stats['profit_factor']:.2f}")
            print(f"    Mean ret={stats['mean_ret_pct']:.3f}%  (raw={stats['mean_ret_raw_pct']:.3f}%)  "
                  f"MDD={stats['max_drawdown']:.1%}")
            print(f"    MFE mean={stats['mean_mfe_pct']:.3f}%  median={stats['median_mfe_pct']:.3f}%")
            print(f"    Perm p={perm_p:.3f}  ", end="")

            if regime:
                print(f"Regime gap={regime.get('regime_gap', 'N/A'):.3f}" if regime.get('regime_gap') is not None else "Regime gap=N/A", end="")
                if regime.get("bull_sharpe") is not None:
                    print(f"  Bull Sharpe={regime['bull_sharpe']:.2f} (n={regime['bull_n']})", end="")
                if regime.get("bear_sharpe") is not None:
                    print(f"  Bear Sharpe={regime['bear_sharpe']:.2f} (n={regime['bear_n']})", end="")
            print()

            all_results[key][f"{hp}d"] = {
                "stats": stats,
                "perm_p": perm_p,
                "regime": regime,
                "gates": gates,
            }

    # Staleness decay
    print(f"\n{'─' * 60}")
    print(f"  STALENESS DECAY (edge by streak day)")
    print(f"{'─' * 60}")

    # Use 5d holding period as reference
    ref_hp = 5
    print(f"\n  Reference holding period: {ref_hp}d")
    print(f"  {'Day':>5} | {'N':>5} | {'Sharpe':>8} | {'WR':>6} | {'Mean Ret':>10} | {'MFE':>8}")
    print(f"  {'─'*5}-+-{'─'*5}-+-{'─'*8}-+-{'─'*6}-+-{'─'*10}-+-{'─'*8}")

    for day in sorted(staleness_trades.keys()):
        trades = staleness_trades[day]
        stats = compute_stats(trades, ref_hp)

        if stats:
            print(f"  {day:>5} | {stats['n_trades']:>5} | {stats['sharpe']:>8.2f} | "
                  f"{stats['win_rate']:>5.1%} | {stats['mean_ret_pct']:>9.3f}% | "
                  f"{stats['mean_mfe_pct']:>7.3f}%")
            all_results["staleness_decay"][f"day_{day}_{ref_hp}d"] = stats
        else:
            print(f"  {day:>5} | {'<30':>5} |      N/A |    N/A |        N/A |      N/A")

    # Comparison summary
    print(f"\n{'=' * 80}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 80}")

    for hp in HOLDING_PERIODS:
        burst_stats = compute_stats(burst_trades, hp)
        persist_stats = compute_stats(persistent_trades, hp)

        if burst_stats and persist_stats:
            sharpe_diff = burst_stats["sharpe"] - persist_stats["sharpe"]
            wr_diff = burst_stats["win_rate"] - persist_stats["win_rate"]
            ret_diff = burst_stats["mean_ret_pct"] - persist_stats["mean_ret_pct"]

            print(f"\n  {hp}d: Burst Sharpe={burst_stats['sharpe']:.2f} vs Persistent={persist_stats['sharpe']:.2f}  "
                  f"Δ={sharpe_diff:+.2f}")
            print(f"      Burst WR={burst_stats['win_rate']:.1%} vs Persistent={persist_stats['win_rate']:.1%}  "
                  f"Δ={wr_diff:+.1%}")
            print(f"      Burst Ret={burst_stats['mean_ret_pct']:.3f}% vs Persistent={persist_stats['mean_ret_pct']:.3f}%  "
                  f"Δ={ret_diff:+.3f}%")

            all_results["comparison"][f"{hp}d"] = {
                "sharpe_diff": float(sharpe_diff),
                "wr_diff": float(wr_diff),
                "ret_diff": float(ret_diff),
                "burst_sharpe": burst_stats["sharpe"],
                "persistent_sharpe": persist_stats["sharpe"],
                "verdict": "burst_wins" if sharpe_diff > 0 else "persistent_wins"
            }

    # Ticker breakdown
    print(f"\n{'─' * 60}")
    print(f"  TICKER BREAKDOWN (5d hold, burst signals)")
    print(f"{'─' * 60}")

    ticker_breakdown = {}
    for ticker in SECTOR_ETFS:
        ticker_burst = [t for t in burst_trades if t["ticker"] == ticker]
        if len(ticker_burst) >= 10:
            rets = np.array([t["ret_5d"] for t in ticker_burst])
            sharpe = (np.mean(rets) / max(np.std(rets, ddof=1), 1e-9)) * np.sqrt(252/5)
            print(f"  {ticker:>5}: N={len(ticker_burst):>4}  Sharpe={sharpe:>6.2f}  "
                  f"WR={np.mean(rets>0):>5.1%}  Mean={np.mean(rets)*100:>7.3f}%")
            ticker_breakdown[ticker] = {
                "n": len(ticker_burst),
                "sharpe": float(sharpe),
                "wr": float(np.mean(rets > 0)),
                "mean_ret_pct": float(np.mean(rets) * 100)
            }

    all_results["ticker_breakdown_burst_5d"] = ticker_breakdown

    # Key finding
    print(f"\n{'=' * 80}")
    print("KEY FINDING")
    print(f"{'=' * 80}")

    best_hp = None
    best_sharpe_diff = -999
    for hp in HOLDING_PERIODS:
        if f"{hp}d" in all_results["comparison"]:
            diff = all_results["comparison"][f"{hp}d"]["sharpe_diff"]
            if diff > best_sharpe_diff:
                best_sharpe_diff = diff
                best_hp = hp

    if best_hp and best_sharpe_diff > 0:
        comp = all_results["comparison"][f"{best_hp}d"]
        print(f"\n  Burst signals outperform persistent at {best_hp}d hold:")
        print(f"    Burst Sharpe {comp['burst_sharpe']:.2f} vs Persistent {comp['persistent_sharpe']:.2f}")
        print(f"    Sharpe advantage: {best_sharpe_diff:+.2f}")

        # Check if it passes 5-gate
        burst_key = f"{best_hp}d"
        if burst_key in all_results["burst"]:
            gates = all_results["burst"][burst_key].get("gates", {})
            if gates.get("all_passed"):
                print(f"    STATUS: PASSES all 5 validation gates")
            else:
                failed = [g for g in ["sharpe_gate","regime_gate","perm_gate","min_trades_gate","mdd_gate"]
                          if not gates.get(g)]
                print(f"    STATUS: FAILS gates: {', '.join(failed)}")
    elif best_hp:
        print(f"\n  Persistent signals outperform burst at all holding periods.")
        print(f"  Best HP={best_hp}d, Sharpe diff={best_sharpe_diff:+.2f}")
    else:
        print("\n  Insufficient data for comparison.")

    print()

    # Save results
    output_path = OUTPUT_DIR / "signal_freshness_backtest_results.json"

    # Convert dates to strings for JSON serialization
    def json_safe(obj):
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.strftime("%Y-%m-%d")
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=json_safe)

    print(f"Results saved to {output_path}")

    return all_results


if __name__ == "__main__":
    results = main()
