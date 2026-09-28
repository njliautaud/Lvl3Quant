#!/usr/bin/env python3
"""
VIX Term Structure Trading Strategies Backtest
===============================================
Tests 6 VIX-based strategy variants against a 5-gate validation framework.
Walk-forward OOT: Jan 2022 - Jul 2026.
Account size: $645 (Robinhood, $0 commission on stocks/ETFs).

Gates:
  1. Sharpe > 0.5
  2. Permutation test p < 0.05 (1000 shuffles)
  3. Regime gap < 0.5
  4. MaxDD > -50%
  5. >= 20 trades
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

# ─── Config ───────────────────────────────────────────────────────────────────
ACCOUNT_SIZE = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02% slippage on stocks/ETFs
OOT_START = "2022-01-01"
OOT_END = "2026-07-28"
N_PERMUTATIONS = 1000
RANDOM_SEED = 42

# ─── Data Download ────────────────────────────────────────────────────────────
def download_data():
    """Download all needed tickers with a safety margin before OOT for indicator warmup."""
    warmup_start = "2020-06-01"  # ~1.5 yr warmup for 252-day percentile
    tickers = ["^VIX", "^VIX9D", "SPY", "QQQ", "TQQQ", "TLT", "SVXY", "UVXY", "VIXY"]

    data = {}
    for t in tickers:
        try:
            df = yf.download(t, start=warmup_start, end=OOT_END, progress=False, auto_adjust=True)
            if df is not None and len(df) > 50:
                # Flatten multi-level columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[t] = df
                print(f"  {t}: {len(df)} rows ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")
            else:
                print(f"  {t}: SKIPPED (insufficient data)")
        except Exception as e:
            print(f"  {t}: FAILED ({e})")

    return data


def build_features(data):
    """Build VIX-based features and combine into a single DataFrame."""
    vix = data["^VIX"]["Close"].copy()
    spy = data["SPY"]["Close"].copy()

    # Align all series to common dates
    idx = vix.index.intersection(spy.index)

    features = pd.DataFrame(index=idx)
    features["vix"] = vix.reindex(idx)
    features["spy"] = spy.reindex(idx)
    features["spy_ret"] = features["spy"].pct_change()

    # QQQ
    if "QQQ" in data:
        features["qqq"] = data["QQQ"]["Close"].reindex(idx)

    # TQQQ
    if "TQQQ" in data:
        features["tqqq"] = data["TQQQ"]["Close"].reindex(idx)
        features["tqqq_ret"] = features["tqqq"].pct_change()

    # TLT
    if "TLT" in data:
        features["tlt"] = data["TLT"]["Close"].reindex(idx)
        features["tlt_ret"] = features["tlt"].pct_change()

    # SVXY
    if "SVXY" in data:
        features["svxy"] = data["SVXY"]["Close"].reindex(idx)
        features["svxy_ret"] = features["svxy"].pct_change()

    # UVXY
    if "UVXY" in data:
        features["uvxy"] = data["UVXY"]["Close"].reindex(idx)

    # VIX9D
    if "^VIX9D" in data:
        vix9d = data["^VIX9D"]["Close"].reindex(idx)
        features["vix9d"] = vix9d
        features["vix_vix9d_ratio"] = features["vix"] / vix9d

    # VIX features
    features["vix_ma20"] = features["vix"].rolling(20).mean()
    features["vix_pct_from_ma20"] = (features["vix"] - features["vix_ma20"]) / features["vix_ma20"]
    features["vix_percentile_252"] = features["vix"].rolling(252).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    features["vix_roc_5d"] = features["vix"].pct_change(5)
    features["vix_roc_1d"] = features["vix"].pct_change(1)

    # SPY trend
    features["spy_sma50"] = features["spy"].rolling(50).mean()
    features["spy_sma200"] = features["spy"].rolling(200).mean()
    features["spy_above_200sma"] = (features["spy"] > features["spy_sma200"]).astype(int)

    # Regime: Bull = SPY above 200-SMA, Bear = below
    features["regime"] = features["spy_above_200sma"].map({1: "bull", 0: "bear"})

    features.dropna(subset=["vix_percentile_252", "spy_sma200"], inplace=True)

    return features


# ─── Strategy Engine ──────────────────────────────────────────────────────────
def simulate_strategy(features, signals_func, strategy_name, trade_instrument="spy"):
    """
    Generic strategy simulator.

    signals_func(features) -> pd.Series of positions:
        1 = long, 0 = cash, -1 = short (if applicable)

    Returns dict of trades and daily returns.
    """
    oot_mask = features.index >= OOT_START
    oot = features[oot_mask].copy()

    if len(oot) < 30:
        return None

    # Generate signals
    positions = signals_func(features).reindex(oot.index).fillna(0)

    # Determine which return series to use
    if trade_instrument == "spy":
        oot_ret = oot["spy_ret"]
    elif trade_instrument == "tqqq" and "tqqq_ret" in oot.columns:
        oot_ret = oot["tqqq_ret"]
    elif trade_instrument == "tlt" and "tlt_ret" in oot.columns:
        oot_ret = oot["tlt_ret"]
    elif trade_instrument == "svxy" and "svxy_ret" in oot.columns:
        oot_ret = oot["svxy_ret"]
    elif trade_instrument == "dynamic":
        # Handled in the signals function — positions encode instrument choice
        # We'll need per-day returns based on position type
        oot_ret = oot["spy_ret"]  # default, overridden below
    else:
        oot_ret = oot["spy_ret"]

    # Apply slippage on position changes
    pos_changes = positions.diff().abs().fillna(0)
    slippage_cost = pos_changes * SLIPPAGE_PCT

    # Strategy returns
    strat_ret = positions.shift(1) * oot_ret - slippage_cost
    strat_ret = strat_ret.dropna()

    # Extract trades (position changes)
    trades = []
    current_pos = 0
    entry_date = None
    entry_price = None

    for date, pos in positions.items():
        pos_val = int(pos)
        if pos_val != current_pos:
            # Close previous trade
            if current_pos != 0 and entry_date is not None:
                exit_price = oot.loc[date, "spy"] if trade_instrument in ["spy", "dynamic"] else oot.loc[date, trade_instrument] if trade_instrument in oot.columns else oot.loc[date, "spy"]
                trade_ret = (exit_price / entry_price - 1) * current_pos - SLIPPAGE_PCT
                trades.append({
                    "entry_date": str(entry_date.date()),
                    "exit_date": str(date.date()),
                    "direction": "long" if current_pos > 0 else "short",
                    "return": float(trade_ret),
                    "regime": oot.loc[entry_date, "regime"] if entry_date in oot.index else "unknown"
                })

            # Open new trade
            if pos_val != 0:
                entry_date = date
                entry_price = oot.loc[date, "spy"] if trade_instrument in ["spy", "dynamic"] else oot.loc[date, trade_instrument] if trade_instrument in oot.columns else oot.loc[date, "spy"]

            current_pos = pos_val

    # Close last trade
    if current_pos != 0 and entry_date is not None:
        last_date = oot.index[-1]
        exit_price = oot.loc[last_date, "spy"] if trade_instrument in ["spy", "dynamic"] else oot.loc[last_date, trade_instrument] if trade_instrument in oot.columns else oot.loc[last_date, "spy"]
        trade_ret = (exit_price / entry_price - 1) * current_pos - SLIPPAGE_PCT
        trades.append({
            "entry_date": str(entry_date.date()),
            "exit_date": str(last_date.date()),
            "direction": "long" if current_pos > 0 else "short",
            "return": float(trade_ret),
            "regime": oot.loc[entry_date, "regime"] if entry_date in oot.index else "unknown"
        })

    return {
        "strategy": strategy_name,
        "daily_returns": strat_ret,
        "positions": positions,
        "trades": trades,
        "n_trades": len(trades)
    }


# ─── Strategy Variants ───────────────────────────────────────────────────────

def strategy_A_vix_mean_reversion(features):
    """Buy SPY when VIX > 80th percentile (1yr), sell when < 20th."""
    pos = pd.Series(0, index=features.index, dtype=float)
    in_trade = False
    for i in range(len(features)):
        pct = features["vix_percentile_252"].iloc[i]
        if not in_trade and pct > 0.80:
            in_trade = True
        elif in_trade and pct < 0.20:
            in_trade = False
        pos.iloc[i] = 1.0 if in_trade else 0.0
    return pos


def strategy_B_vix_momentum(features):
    """Buy SPY puts (short SPY) when VIX 5d ROC > 20%; buy calls (long SPY) when ROC < -15%."""
    pos = pd.Series(0, index=features.index, dtype=float)
    in_trade = False
    direction = 0
    hold_counter = 0
    max_hold = 10  # hold for up to 10 trading days

    for i in range(len(features)):
        roc = features["vix_roc_5d"].iloc[i]

        if in_trade:
            hold_counter += 1
            if hold_counter >= max_hold:
                in_trade = False
                direction = 0
                hold_counter = 0
            pos.iloc[i] = float(direction)
        else:
            if roc > 0.20:
                in_trade = True
                direction = -1  # short SPY (proxy for buying puts)
                hold_counter = 0
                pos.iloc[i] = -1.0
            elif roc < -0.15:
                in_trade = True
                direction = 1  # long SPY (proxy for buying calls)
                hold_counter = 0
                pos.iloc[i] = 1.0
            else:
                pos.iloc[i] = 0.0
    return pos


def strategy_C_contango_proxy(features):
    """When VIX < 20d MA by >10%, buy SPY (ride calm market). When VIX > 20d MA by >20%, cash."""
    pos = pd.Series(0, index=features.index, dtype=float)
    state = 0  # 0=cash, 1=long

    for i in range(len(features)):
        pct_from_ma = features["vix_pct_from_ma20"].iloc[i]

        if pct_from_ma < -0.10:
            state = 1  # calm, go long
        elif pct_from_ma > 0.20:
            state = 0  # stressed, go cash
        # else: maintain current state

        pos.iloc[i] = float(state)
    return pos


def strategy_D_vix_spike_fade(features):
    """When VIX spikes >30% in 5 days, buy SPY expecting mean reversion within 10 days."""
    pos = pd.Series(0, index=features.index, dtype=float)
    hold_counter = 0
    in_trade = False

    for i in range(len(features)):
        roc = features["vix_roc_5d"].iloc[i]

        if in_trade:
            hold_counter += 1
            if hold_counter >= 10:
                in_trade = False
                hold_counter = 0
                pos.iloc[i] = 0.0
            else:
                pos.iloc[i] = 1.0
        else:
            if roc > 0.30:
                in_trade = True
                hold_counter = 0
                pos.iloc[i] = 1.0
            else:
                pos.iloc[i] = 0.0
    return pos


def strategy_E_regime_switch(features):
    """
    VIX < 15: aggressive (TQQQ proxy via 3x SPY return)
    VIX 15-25: moderate (SPY)
    VIX > 25: defensive (cash — or TLT if available)

    Returns position in SPY-equivalent terms.
    """
    pos = pd.Series(0, index=features.index, dtype=float)

    for i in range(len(features)):
        v = features["vix"].iloc[i]
        if v < 15:
            pos.iloc[i] = 3.0   # 3x = TQQQ-like leverage
        elif v <= 25:
            pos.iloc[i] = 1.0   # SPY
        else:
            pos.iloc[i] = 0.0   # cash/defensive
    return pos


def strategy_F_combined(features):
    """
    Combined: VIX percentile + SPY trend + VIX direction.
    Long SPY only when: VIX percentile < 60% AND SPY above 50-SMA AND VIX falling (1d ROC < 0).
    Short SPY when: VIX percentile > 80% AND SPY below 50-SMA AND VIX rising (1d ROC > 0).
    """
    pos = pd.Series(0, index=features.index, dtype=float)

    for i in range(len(features)):
        pct = features["vix_percentile_252"].iloc[i]
        spy_above_50 = features["spy"].iloc[i] > features["spy_sma50"].iloc[i]
        vix_falling = features["vix_roc_1d"].iloc[i] < 0
        vix_rising = features["vix_roc_1d"].iloc[i] > 0

        long_signals = int(pct < 0.60) + int(spy_above_50) + int(vix_falling)
        short_signals = int(pct > 0.80) + int(not spy_above_50) + int(vix_rising)

        if long_signals == 3:
            pos.iloc[i] = 1.0
        elif short_signals == 3:
            pos.iloc[i] = -1.0
        else:
            pos.iloc[i] = 0.0

    return pos


# ─── Metrics & Validation ────────────────────────────────────────────────────

def compute_metrics(daily_returns, trades, positions):
    """Compute strategy performance metrics."""
    dr = daily_returns.dropna()
    if len(dr) < 20:
        return None

    # Annualized Sharpe
    mean_ret = dr.mean()
    std_ret = dr.std()
    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0

    # Sortino
    downside = dr[dr < 0].std()
    sortino = (mean_ret / downside * np.sqrt(252)) if downside > 0 else 0.0

    # Max drawdown
    cum_ret = (1 + dr).cumprod()
    rolling_max = cum_ret.cummax()
    drawdown = (cum_ret - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # Win rate from trades
    if len(trades) > 0:
        wins = sum(1 for t in trades if t["return"] > 0)
        win_rate = wins / len(trades)
        avg_win = np.mean([t["return"] for t in trades if t["return"] > 0]) if wins > 0 else 0
        losses = [t["return"] for t in trades if t["return"] <= 0]
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        profit_factor = abs(avg_win * wins / (avg_loss * len(losses))) if avg_loss != 0 and len(losses) > 0 else float('inf')
    else:
        win_rate = 0.0
        profit_factor = 0.0

    # Total return
    total_return = cum_ret.iloc[-1] - 1 if len(cum_ret) > 0 else 0

    # CAGR
    n_years = len(dr) / 252
    cagr = (cum_ret.iloc[-1]) ** (1 / n_years) - 1 if n_years > 0 and cum_ret.iloc[-1] > 0 else 0

    # Time in market
    time_in_market = (positions.shift(1).reindex(dr.index).fillna(0) != 0).mean()

    return {
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "max_dd": round(float(max_dd), 4),
        "max_dd_pct": round(float(max_dd * 100), 2),
        "total_return": round(float(total_return), 4),
        "total_return_pct": round(float(total_return * 100), 2),
        "cagr": round(float(cagr), 4),
        "cagr_pct": round(float(cagr * 100), 2),
        "win_rate": round(float(win_rate), 3),
        "profit_factor": round(float(profit_factor), 3),
        "n_trades": len(trades),
        "n_days": len(dr),
        "time_in_market": round(float(time_in_market), 3),
        "final_equity": round(float(ACCOUNT_SIZE * cum_ret.iloc[-1]), 2),
    }


def regime_sharpe(daily_returns, features, positions):
    """Compute Sharpe in bull vs bear regimes."""
    oot_mask = features.index >= OOT_START
    oot = features[oot_mask]
    regime = oot["regime"].reindex(daily_returns.index)

    results = {}
    for r in ["bull", "bear"]:
        mask = regime == r
        dr = daily_returns[mask]
        if len(dr) > 10 and dr.std() > 0:
            results[r] = float(dr.mean() / dr.std() * np.sqrt(252))
        else:
            results[r] = 0.0

    # Regime gap
    bull_s = results.get("bull", 0)
    bear_s = results.get("bear", 0)
    max_abs = max(abs(bull_s), abs(bear_s))
    regime_gap = abs(bull_s - bear_s) / max_abs if max_abs > 0 else 0.0

    results["gap"] = float(regime_gap)
    return results


def permutation_test(daily_returns, positions, features, n_perms=N_PERMUTATIONS):
    """Shuffle entry dates to test if strategy Sharpe is significant."""
    np.random.seed(RANDOM_SEED)

    oot_mask = features.index >= OOT_START
    oot_ret = features[oot_mask]["spy_ret"].reindex(daily_returns.index)

    actual_sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0

    count_better = 0
    pos_vals = positions.reindex(daily_returns.index).shift(1).fillna(0).values
    ret_vals = oot_ret.fillna(0).values

    for _ in range(n_perms):
        # Shuffle positions (block shuffle to preserve autocorrelation)
        shuffled_pos = pos_vals.copy()
        # Random circular shift
        shift = np.random.randint(1, len(shuffled_pos))
        shuffled_pos = np.roll(shuffled_pos, shift)

        perm_ret = shuffled_pos * ret_vals
        perm_std = perm_ret.std()
        if perm_std > 0:
            perm_sharpe = perm_ret.mean() / perm_std * np.sqrt(252)
        else:
            perm_sharpe = 0

        if perm_sharpe >= actual_sharpe:
            count_better += 1

    p_value = count_better / n_perms
    return float(p_value)


def validate_gates(metrics, regime_info, p_value):
    """Check all 5 validation gates."""
    gates = {
        "sharpe_gt_0.5": {
            "value": metrics["sharpe"],
            "threshold": 0.5,
            "pass": metrics["sharpe"] > 0.5
        },
        "perm_test_p_lt_0.05": {
            "value": round(p_value, 4),
            "threshold": 0.05,
            "pass": p_value < 0.05
        },
        "regime_gap_lt_0.5": {
            "value": round(regime_info["gap"], 4),
            "threshold": 0.5,
            "pass": regime_info["gap"] < 0.5
        },
        "max_dd_gt_neg50": {
            "value": metrics["max_dd_pct"],
            "threshold": -50.0,
            "pass": metrics["max_dd"] > -0.50
        },
        "n_trades_gte_20": {
            "value": metrics["n_trades"],
            "threshold": 20,
            "pass": metrics["n_trades"] >= 20
        }
    }

    all_pass = all(g["pass"] for g in gates.values())
    n_pass = sum(1 for g in gates.values() if g["pass"])

    return gates, all_pass, n_pass


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("VIX TERM STRUCTURE TRADING STRATEGIES BACKTEST")
    print(f"OOT Period: {OOT_START} to {OOT_END}")
    print(f"Account Size: ${ACCOUNT_SIZE:.0f}")
    print(f"Permutation Test: {N_PERMUTATIONS} iterations")
    print("=" * 80)

    print("\n[1/4] Downloading data...")
    data = download_data()

    required = ["^VIX", "SPY"]
    for t in required:
        if t not in data:
            print(f"FATAL: Required ticker {t} not available. Aborting.")
            sys.exit(1)

    print(f"\n[2/4] Building features...")
    features = build_features(data)
    print(f"  Feature matrix: {len(features)} rows, {len(features.columns)} columns")
    print(f"  Date range: {features.index[0].strftime('%Y-%m-%d')} to {features.index[-1].strftime('%Y-%m-%d')}")
    oot_count = (features.index >= OOT_START).sum()
    print(f"  OOT days: {oot_count}")

    # Strategy definitions
    strategies = [
        ("A) VIX Mean Reversion", strategy_A_vix_mean_reversion, "spy"),
        ("B) VIX Momentum", strategy_B_vix_momentum, "spy"),
        ("C) Contango Proxy", strategy_C_contango_proxy, "spy"),
        ("D) VIX Spike Fade", strategy_D_vix_spike_fade, "spy"),
        ("E) Regime Switch (3x/1x/cash)", strategy_E_regime_switch, "spy"),
        ("F) Combined (VIX+Trend+Direction)", strategy_F_combined, "spy"),
    ]

    print(f"\n[3/4] Running {len(strategies)} strategy variants...")
    results = {}

    for name, func, instrument in strategies:
        print(f"\n  --- {name} ---")
        sim = simulate_strategy(features, func, name, instrument)

        if sim is None or sim["n_trades"] == 0:
            print(f"    SKIPPED: No trades generated")
            results[name] = {"status": "no_trades"}
            continue

        metrics = compute_metrics(sim["daily_returns"], sim["trades"], sim["positions"])
        if metrics is None:
            print(f"    SKIPPED: Insufficient data")
            results[name] = {"status": "insufficient_data"}
            continue

        regime_info = regime_sharpe(sim["daily_returns"], features, sim["positions"])

        print(f"    Computing permutation test ({N_PERMUTATIONS} iterations)...")
        p_value = permutation_test(sim["daily_returns"], sim["positions"], features)

        gates, all_pass, n_pass = validate_gates(metrics, regime_info, p_value)

        print(f"    Sharpe: {metrics['sharpe']:.3f} | Sortino: {metrics['sortino']:.3f}")
        print(f"    Total Return: {metrics['total_return_pct']:.1f}% | MaxDD: {metrics['max_dd_pct']:.1f}%")
        print(f"    Win Rate: {metrics['win_rate']:.1%} | Trades: {metrics['n_trades']}")
        print(f"    Regime Sharpe — Bull: {regime_info['bull']:.3f}, Bear: {regime_info['bear']:.3f}, Gap: {regime_info['gap']:.3f}")
        print(f"    Permutation p-value: {p_value:.4f}")
        print(f"    Gates: {n_pass}/5 pass {'✓ ALL PASS' if all_pass else '✗ FAIL'}")

        results[name] = {
            "status": "complete",
            "metrics": metrics,
            "regime": {k: round(v, 4) for k, v in regime_info.items()},
            "perm_p_value": round(p_value, 4),
            "gates": {k: {"value": v["value"], "threshold": v["threshold"], "pass": v["pass"]} for k, v in gates.items()},
            "all_gates_pass": all_pass,
            "n_gates_pass": n_pass,
            "trades_sample": sim["trades"][:5] if len(sim["trades"]) > 5 else sim["trades"],
        }

    # ─── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"{'Strategy':<40} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>7} {'WR':>6} {'#Tr':>5} {'Gates':>6} {'Result':>8}")
    print("-" * 80)

    for name, res in results.items():
        if res["status"] != "complete":
            print(f"{name:<40} {'N/A':>7} {'N/A':>8} {'N/A':>7} {'N/A':>6} {'N/A':>5} {'N/A':>6} {'SKIP':>8}")
            continue
        m = res["metrics"]
        status = "PASS" if res["all_gates_pass"] else "FAIL"
        print(f"{name:<40} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['max_dd_pct']:>6.1f}% {m['win_rate']:>5.1%} {m['n_trades']:>5d} {res['n_gates_pass']:>2}/5   {status:>6}")

    print("-" * 80)

    # Gate details
    print("\nGATE DETAILS:")
    for name, res in results.items():
        if res["status"] != "complete":
            continue
        print(f"\n  {name}:")
        for gate_name, gate_info in res["gates"].items():
            symbol = "PASS" if gate_info["pass"] else "FAIL"
            print(f"    [{symbol}] {gate_name}: {gate_info['value']} (threshold: {gate_info['threshold']})")

    # Final equity
    print("\nFINAL EQUITY ($645 starting):")
    for name, res in results.items():
        if res["status"] != "complete":
            continue
        m = res["metrics"]
        pnl = m["final_equity"] - ACCOUNT_SIZE
        print(f"  {name}: ${m['final_equity']:.2f} ({'+' if pnl >= 0 else ''}{pnl:.2f})")

    # Passers
    passers = [name for name, res in results.items() if res.get("all_gates_pass")]
    print(f"\n{'=' * 80}")
    if passers:
        print(f"STRATEGIES PASSING ALL 5 GATES: {len(passers)}")
        for p in passers:
            print(f"  -> {p}")
    else:
        print("NO STRATEGY PASSED ALL 5 GATES")
        # Find best
        best = max(
            [(name, res) for name, res in results.items() if res["status"] == "complete"],
            key=lambda x: x[1]["n_gates_pass"],
            default=None
        )
        if best:
            print(f"  Best: {best[0]} ({best[1]['n_gates_pass']}/5 gates)")
    print("=" * 80)

    # ─── Save results ─────────────────────────────────────────────────────────
    print(f"\n[4/4] Saving results...")
    output = {
        "run_timestamp": datetime.now().isoformat(),
        "config": {
            "account_size": ACCOUNT_SIZE,
            "oot_start": OOT_START,
            "oot_end": OOT_END,
            "slippage_pct": SLIPPAGE_PCT,
            "n_permutations": N_PERMUTATIONS,
        },
        "strategies": {},
    }

    for name, res in results.items():
        # Convert any remaining non-serializable values
        output["strategies"][name] = res

    output_path = Path("/home/jupiter/Lvl3Quant/data/vix_term_structure_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"  Saved to {output_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
