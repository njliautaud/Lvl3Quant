#!/usr/bin/env python3
"""
Adversarial Validation for Pairs Trading Strategies
====================================================
Tests whether MSFT/AAPL and SNAP/PINS pairs trading edge is real or spurious.

5 adversarial tests:
1. Inverse test (trade opposite direction)
2. Random pair test (50 random growth stock pairs)
3. Shuffled entry test (1000 permutations)
4. Time stability (3 sub-periods)
5. Convergence analysis (z-score return to 0)
"""

import json
import warnings
import sys
from datetime import datetime, timedelta
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────

OOT_START = "2022-01-01"
OOT_END = "2026-07-25"
INITIAL_CAPITAL = 645.0
ZSCORE_LOOKBACK = 60  # trading days for rolling z-score
ZSCORE_ENTRY = 2.0
ZSCORE_EXIT = 0.0
MAX_HOLD_DAYS = 30
N_RANDOM_PAIRS = 50
N_SHUFFLE_ITERS = 1000
SEED = 42

# Growth stocks pool for random pair test
GROWTH_STOCKS = [
    "MSFT", "AAPL", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX",
    "CRM", "ADBE", "PYPL", "SQ", "SHOP", "SNAP", "PINS", "UBER",
    "LYFT", "ZM", "DKNG", "RBLX", "PLTR", "SOFI", "HOOD", "COIN",
    "AMD", "INTC", "QCOM", "MU", "AVGO", "NOW", "SNOW", "PANW",
    "CRWD", "ZS", "NET", "DDOG", "MDB", "TTD", "ROKU", "SPOT"
]

np.random.seed(SEED)


# ── Data download ───────────────────────────────────────────────────────────

def download_data(tickers, start, end):
    """Download adjusted close prices for a list of tickers."""
    # Add SPY for regime classification
    all_tickers = list(set(tickers + ["SPY"]))
    print(f"  Downloading {len(all_tickers)} tickers...")
    data = yf.download(all_tickers, start=start, end=end, auto_adjust=True, progress=False)

    # Handle multi-level columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data

    return close


def get_regime(spy_prices):
    """Bull = SPY > 200-SMA, Bear = SPY < 200-SMA."""
    sma200 = spy_prices.rolling(200).mean()
    regime = pd.Series("bull", index=spy_prices.index)
    regime[spy_prices < sma200] = "bear"
    return regime


# ── Core pairs trading backtest ─────────────────────────────────────────────

def pairs_backtest(prices_a, prices_b, capital=INITIAL_CAPITAL, inverse=False):
    """
    Z-score mean reversion pairs backtest (long-only on both legs).

    When z-score(ratio) > ZSCORE_ENTRY: A is expensive vs B → buy B (underperformer).
    When z-score(ratio) < -ZSCORE_ENTRY: B is expensive vs A → buy A (underperformer).
    Exit when z-score crosses 0 or max hold exceeded.

    If inverse=True, trade the OPPOSITE direction (buy the outperformer).

    Returns: trades DataFrame, equity curve.
    """
    ratio = prices_a / prices_b
    ratio_clean = ratio.dropna()

    if len(ratio_clean) < ZSCORE_LOOKBACK + 10:
        return pd.DataFrame(), pd.Series(dtype=float)

    rolling_mean = ratio_clean.rolling(ZSCORE_LOOKBACK).mean()
    rolling_std = ratio_clean.rolling(ZSCORE_LOOKBACK).std()
    zscore = (ratio_clean - rolling_mean) / rolling_std

    # Align all series
    valid_idx = zscore.dropna().index
    zscore = zscore.loc[valid_idx]
    pa = prices_a.reindex(valid_idx).ffill()
    pb = prices_b.reindex(valid_idx).ffill()

    trades = []
    equity = capital
    equity_curve = []
    position = None  # None or dict with entry info

    for i, date in enumerate(valid_idx):
        z = zscore.iloc[i]

        if position is None:
            # Entry logic
            if abs(z) >= ZSCORE_ENTRY:
                # z > 2: A expensive, B cheap → buy B (unless inverse)
                # z < -2: B expensive, A cheap → buy A (unless inverse)
                if z > ZSCORE_ENTRY:
                    buy_ticker = "B" if not inverse else "A"
                elif z < -ZSCORE_ENTRY:
                    buy_ticker = "A" if not inverse else "B"
                else:
                    continue

                entry_price = pa.iloc[i] if buy_ticker == "A" else pb.iloc[i]
                shares = int(equity / entry_price) if entry_price > 0 else 0
                if shares == 0:
                    continue

                position = {
                    "buy": buy_ticker,
                    "entry_date": date,
                    "entry_price": entry_price,
                    "entry_z": z,
                    "shares": shares,
                    "entry_idx": i,
                }
        else:
            # Exit logic: z crosses 0 or max hold
            days_held = i - position["entry_idx"]
            z_crossed = (position["entry_z"] > 0 and z <= ZSCORE_EXIT) or \
                        (position["entry_z"] < 0 and z >= ZSCORE_EXIT)

            if z_crossed or days_held >= MAX_HOLD_DAYS:
                exit_price = pa.iloc[i] if position["buy"] == "A" else pb.iloc[i]
                pnl = (exit_price - position["entry_price"]) * position["shares"]
                ret = (exit_price / position["entry_price"]) - 1.0
                equity += pnl

                trades.append({
                    "entry_date": position["entry_date"],
                    "exit_date": date,
                    "buy": position["buy"],
                    "entry_price": position["entry_price"],
                    "exit_price": exit_price,
                    "shares": position["shares"],
                    "pnl": pnl,
                    "return": ret,
                    "days_held": days_held,
                    "entry_z": position["entry_z"],
                    "exit_z": z,
                    "converged": z_crossed,
                })
                position = None

        equity_curve.append({"date": date, "equity": equity})

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    eq_series = pd.DataFrame(equity_curve).set_index("date")["equity"] if equity_curve else pd.Series(dtype=float)

    return trades_df, eq_series


# ── Metrics computation ────────────────────────────────────────────────────

def compute_metrics(trades_df, equity_curve, capital=INITIAL_CAPITAL):
    """Compute Sharpe, Sortino, PF, WR, MDD."""
    if trades_df.empty or len(trades_df) < 2:
        return {
            "sharpe": 0.0, "sortino": 0.0, "pf": 0.0, "wr": 0.0,
            "mdd": 0.0, "n_trades": len(trades_df) if not trades_df.empty else 0,
            "total_return": 0.0,
        }

    returns = trades_df["return"].values
    n_trades = len(returns)

    # Annualize assuming ~252/max_hold trades per year
    avg_days = trades_df["days_held"].mean() if "days_held" in trades_df else 10
    trades_per_year = 252 / max(avg_days, 1)

    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1) if len(returns) > 1 else 1e-9

    # Sharpe (annualized)
    sharpe = (mean_ret / max(std_ret, 1e-9)) * np.sqrt(trades_per_year)

    # Sortino
    downside = returns[returns < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = (mean_ret / max(downside_std, 1e-9)) * np.sqrt(trades_per_year)

    # Profit Factor
    gross_profit = trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()
    gross_loss = abs(trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum())
    pf = gross_profit / max(gross_loss, 1e-9)

    # Win Rate
    wr = (returns > 0).mean()

    # Max Drawdown from equity curve
    if len(equity_curve) > 0:
        peak = equity_curve.cummax()
        dd = (equity_curve - peak) / peak
        mdd = dd.min()
    else:
        mdd = 0.0

    total_ret = (equity_curve.iloc[-1] / capital - 1.0) if len(equity_curve) > 0 else 0.0

    return {
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "pf": round(float(pf), 3),
        "wr": round(float(wr), 3),
        "mdd": round(float(mdd), 3),
        "n_trades": int(n_trades),
        "total_return": round(float(total_ret), 4),
    }


def regime_sharpe(trades_df, regime_series):
    """Compute Sharpe in bull vs bear regimes. Returns regime gap."""
    if trades_df.empty or "entry_date" not in trades_df.columns:
        return 0.0, 0.0, 1.0

    trades_df = trades_df.copy()
    # Map each trade to regime at entry
    trades_df["regime"] = trades_df["entry_date"].map(
        lambda d: regime_series.get(d, "bull") if hasattr(regime_series, 'get')
        else regime_series.reindex([d], method="ffill").iloc[0] if d in regime_series.index
        else "bull"
    )

    bull_trades = trades_df[trades_df["regime"] == "bull"]["return"]
    bear_trades = trades_df[trades_df["regime"] == "bear"]["return"]

    bull_sharpe = (bull_trades.mean() / max(bull_trades.std(), 1e-9)) if len(bull_trades) > 1 else 0.0
    bear_sharpe = (bear_trades.mean() / max(bear_trades.std(), 1e-9)) if len(bear_trades) > 1 else 0.0

    max_abs = max(abs(bull_sharpe), abs(bear_sharpe), 1e-9)
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs

    return float(bull_sharpe), float(bear_sharpe), float(regime_gap)


# ── Test 1: Inverse Direction ───────────────────────────────────────────────

def test_inverse(prices_a, prices_b, pair_name, regime_series):
    """Trade opposite direction. If inverse also profitable, signal is just beta."""
    print(f"\n[TEST 1] Inverse direction test for {pair_name}")

    # Normal
    trades_n, eq_n = pairs_backtest(prices_a, prices_b, inverse=False)
    metrics_n = compute_metrics(trades_n, eq_n)

    # Inverse
    trades_i, eq_i = pairs_backtest(prices_a, prices_b, inverse=True)
    metrics_i = compute_metrics(trades_i, eq_i)

    print(f"  Normal:  Sharpe={metrics_n['sharpe']}, WR={metrics_n['wr']}, PF={metrics_n['pf']}, N={metrics_n['n_trades']}")
    print(f"  Inverse: Sharpe={metrics_i['sharpe']}, WR={metrics_i['wr']}, PF={metrics_i['pf']}, N={metrics_i['n_trades']}")

    # If inverse Sharpe > 0, the signal may just be beta
    is_beta = metrics_i["sharpe"] > 0.3  # inverse also positive = suspicious
    verdict = "FAIL (likely beta)" if is_beta else "PASS (directional edge)"
    print(f"  Verdict: {verdict}")

    return {
        "pair": pair_name,
        "normal": metrics_n,
        "inverse": metrics_i,
        "inverse_sharpe_positive": metrics_i["sharpe"] > 0,
        "likely_beta": is_beta,
        "verdict": verdict,
    }


# ── Test 2: Random Pair Test ───────────────────────────────────────────────

def test_random_pairs(all_prices, regime_series):
    """Apply same z-score strategy to 50 random pairs. If they also work, edge is generic."""
    print(f"\n[TEST 2] Random pair test ({N_RANDOM_PAIRS} pairs)")

    available = [t for t in GROWTH_STOCKS if t in all_prices.columns and all_prices[t].notna().sum() > ZSCORE_LOOKBACK + 50]
    print(f"  {len(available)} stocks with sufficient data")

    if len(available) < 10:
        print("  ERROR: Not enough stocks with data")
        return {"error": "insufficient_data", "n_available": len(available)}

    all_combos = list(combinations(available, 2))
    np.random.shuffle(all_combos)
    selected_pairs = all_combos[:N_RANDOM_PAIRS]

    results = []
    for i, (a, b) in enumerate(selected_pairs):
        trades, eq = pairs_backtest(all_prices[a], all_prices[b])
        m = compute_metrics(trades, eq)
        m["pair"] = f"{a}/{b}"
        results.append(m)
        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{N_RANDOM_PAIRS} done")

    results_df = pd.DataFrame(results)
    sharpes = results_df["sharpe"].values

    pct_positive_sharpe = (sharpes > 0).mean()
    pct_above_05 = (sharpes > 0.5).mean()
    median_sharpe = np.median(sharpes)
    mean_sharpe = np.mean(sharpes)

    print(f"  Random pairs: median Sharpe={median_sharpe:.3f}, mean={mean_sharpe:.3f}")
    print(f"  {pct_positive_sharpe*100:.1f}% positive Sharpe, {pct_above_05*100:.1f}% above 0.5")

    # If >40% of random pairs also have Sharpe>0.5, the edge is generic
    is_generic = pct_above_05 > 0.40
    verdict = "FAIL (generic mean-reversion)" if is_generic else "PASS (pair-specific edge)"
    print(f"  Verdict: {verdict}")

    return {
        "n_pairs_tested": len(results),
        "median_sharpe": round(float(median_sharpe), 3),
        "mean_sharpe": round(float(mean_sharpe), 3),
        "pct_positive_sharpe": round(float(pct_positive_sharpe), 3),
        "pct_above_05": round(float(pct_above_05), 3),
        "is_generic_edge": is_generic,
        "verdict": verdict,
        "top5": sorted(results, key=lambda x: x["sharpe"], reverse=True)[:5],
        "bottom5": sorted(results, key=lambda x: x["sharpe"])[:5],
    }


# ── Test 3: Shuffled Entry Test ────────────────────────────────────────────

def test_shuffled_entries(prices_a, prices_b, pair_name):
    """Keep trade dates, randomize which stock to buy. 1000 iterations."""
    print(f"\n[TEST 3] Shuffled entry test for {pair_name} ({N_SHUFFLE_ITERS} iterations)")

    # Get the actual trades first
    trades_real, eq_real = pairs_backtest(prices_a, prices_b)
    if trades_real.empty:
        return {"error": "no_trades", "pair": pair_name}

    real_sharpe = compute_metrics(trades_real, eq_real)["sharpe"]

    # For each shuffle iteration, randomize the buy direction
    ratio = prices_a / prices_b
    ratio_clean = ratio.dropna()
    rolling_mean = ratio_clean.rolling(ZSCORE_LOOKBACK).mean()
    rolling_std = ratio_clean.rolling(ZSCORE_LOOKBACK).std()
    zscore = ((ratio_clean - rolling_mean) / rolling_std).dropna()

    valid_idx = zscore.index
    pa = prices_a.reindex(valid_idx).ffill()
    pb = prices_b.reindex(valid_idx).ffill()

    shuffle_sharpes = []

    for iteration in range(N_SHUFFLE_ITERS):
        trades = []
        equity = INITIAL_CAPITAL
        position = None

        for i in range(len(valid_idx)):
            z = zscore.iloc[i]
            date = valid_idx[i]

            if position is None:
                if abs(z) >= ZSCORE_ENTRY:
                    # Random assignment instead of z-score directed
                    buy_ticker = "A" if np.random.random() > 0.5 else "B"
                    entry_price = pa.iloc[i] if buy_ticker == "A" else pb.iloc[i]
                    shares = int(equity / entry_price) if entry_price > 0 else 0
                    if shares == 0:
                        continue
                    position = {
                        "buy": buy_ticker, "entry_price": entry_price,
                        "entry_z": z, "shares": shares, "entry_idx": i,
                    }
            else:
                days_held = i - position["entry_idx"]
                z_crossed = (position["entry_z"] > 0 and z <= ZSCORE_EXIT) or \
                            (position["entry_z"] < 0 and z >= ZSCORE_EXIT)

                if z_crossed or days_held >= MAX_HOLD_DAYS:
                    exit_price = pa.iloc[i] if position["buy"] == "A" else pb.iloc[i]
                    pnl = (exit_price - position["entry_price"]) * position["shares"]
                    ret = (exit_price / position["entry_price"]) - 1.0
                    equity += pnl
                    trades.append({"return": ret, "pnl": pnl, "days_held": days_held})
                    position = None

        if len(trades) >= 2:
            rets = np.array([t["return"] for t in trades])
            avg_days = np.mean([t["days_held"] for t in trades])
            tpy = 252 / max(avg_days, 1)
            s = (np.mean(rets) / max(np.std(rets, ddof=1), 1e-9)) * np.sqrt(tpy)
            shuffle_sharpes.append(s)

    shuffle_sharpes = np.array(shuffle_sharpes)
    p_value = (shuffle_sharpes >= real_sharpe).mean()

    print(f"  Real Sharpe: {real_sharpe:.3f}")
    print(f"  Shuffled: mean={np.mean(shuffle_sharpes):.3f}, median={np.median(shuffle_sharpes):.3f}")
    print(f"  p-value: {p_value:.4f}")

    verdict = "PASS (p<0.05)" if p_value < 0.05 else "FAIL (p>=0.05, could be random)"
    print(f"  Verdict: {verdict}")

    return {
        "pair": pair_name,
        "real_sharpe": round(float(real_sharpe), 3),
        "shuffle_mean_sharpe": round(float(np.mean(shuffle_sharpes)), 3),
        "shuffle_median_sharpe": round(float(np.median(shuffle_sharpes)), 3),
        "shuffle_std": round(float(np.std(shuffle_sharpes)), 3),
        "p_value": round(float(p_value), 4),
        "n_iterations": N_SHUFFLE_ITERS,
        "verdict": verdict,
    }


# ── Test 4: Time Stability ─────────────────────────────────────────────────

def test_time_stability(prices_a, prices_b, pair_name, regime_series):
    """Split OOT into 3 sub-periods. Each must have positive Sharpe."""
    print(f"\n[TEST 4] Time stability test for {pair_name}")

    ratio = prices_a / prices_b
    valid_dates = ratio.dropna().index
    n = len(valid_dates)
    split1 = valid_dates[n // 3]
    split2 = valid_dates[2 * n // 3]

    periods = [
        ("Period 1", valid_dates[0], split1),
        ("Period 2", split1, split2),
        ("Period 3", split2, valid_dates[-1]),
    ]

    results = []
    all_positive = True

    for name, start, end in periods:
        pa_sub = prices_a.loc[start:end]
        pb_sub = prices_b.loc[start:end]

        trades, eq = pairs_backtest(pa_sub, pb_sub)
        m = compute_metrics(trades, eq)

        bull_s, bear_s, gap = regime_sharpe(trades, regime_series)

        print(f"  {name} ({str(start.date())} to {str(end.date())}): "
              f"Sharpe={m['sharpe']}, WR={m['wr']}, N={m['n_trades']}, regime_gap={gap:.3f}")

        if m["sharpe"] <= 0:
            all_positive = False

        results.append({
            "period": name,
            "start": str(start.date()),
            "end": str(end.date()),
            "metrics": m,
            "bull_sharpe": round(bull_s, 3),
            "bear_sharpe": round(bear_s, 3),
            "regime_gap": round(gap, 3),
        })

    verdict = "PASS (all sub-periods positive)" if all_positive else "FAIL (not stable across time)"
    print(f"  Verdict: {verdict}")

    return {
        "pair": pair_name,
        "periods": results,
        "all_positive_sharpe": all_positive,
        "verdict": verdict,
    }


# ── Test 5: Convergence Analysis ───────────────────────────────────────────

def test_convergence(prices_a, prices_b, pair_name):
    """What % of trades actually converge (z-score returns to 0)?"""
    print(f"\n[TEST 5] Convergence analysis for {pair_name}")

    trades, eq = pairs_backtest(prices_a, prices_b)

    if trades.empty:
        return {"error": "no_trades", "pair": pair_name}

    n_converged = trades["converged"].sum()
    convergence_rate = n_converged / len(trades)

    # Separate metrics for converged vs non-converged trades
    converged_trades = trades[trades["converged"]]
    timeout_trades = trades[~trades["converged"]]

    conv_wr = (converged_trades["return"] > 0).mean() if len(converged_trades) > 0 else 0
    timeout_wr = (timeout_trades["return"] > 0).mean() if len(timeout_trades) > 0 else 0

    conv_avg_ret = converged_trades["return"].mean() if len(converged_trades) > 0 else 0
    timeout_avg_ret = timeout_trades["return"].mean() if len(timeout_trades) > 0 else 0

    conv_avg_days = converged_trades["days_held"].mean() if len(converged_trades) > 0 else 0
    timeout_avg_days = timeout_trades["days_held"].mean() if len(timeout_trades) > 0 else 0

    print(f"  Total trades: {len(trades)}")
    print(f"  Converged: {n_converged} ({convergence_rate*100:.1f}%)")
    print(f"    Converged WR={conv_wr:.3f}, avg return={conv_avg_ret:.4f}, avg days={conv_avg_days:.1f}")
    print(f"    Timeout  WR={timeout_wr:.3f}, avg return={timeout_avg_ret:.4f}, avg days={timeout_avg_days:.1f}")

    # If convergence <30% and timeout trades are profitable, it's directional drift
    drift_profits = convergence_rate < 0.30 and timeout_avg_ret > 0

    if convergence_rate < 0.30:
        verdict = "FAIL (low convergence — profits likely from drift)"
    elif drift_profits:
        verdict = "WARNING (profits from non-converging trades)"
    else:
        verdict = "PASS (genuine mean reversion)"

    print(f"  Verdict: {verdict}")

    return {
        "pair": pair_name,
        "n_trades": int(len(trades)),
        "n_converged": int(n_converged),
        "convergence_rate": round(float(convergence_rate), 3),
        "converged_wr": round(float(conv_wr), 3),
        "converged_avg_return": round(float(conv_avg_ret), 4),
        "converged_avg_days": round(float(conv_avg_days), 1),
        "timeout_wr": round(float(timeout_wr), 3),
        "timeout_avg_return": round(float(timeout_avg_ret), 4),
        "timeout_avg_days": round(float(timeout_avg_days), 1),
        "drift_profits": drift_profits,
        "verdict": verdict,
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("ADVERSARIAL VALIDATION: Pairs Trading Strategies")
    print("=" * 70)
    print(f"OOT: {OOT_START} to {OOT_END} | Capital: ${INITIAL_CAPITAL}")
    print(f"Z-score: lookback={ZSCORE_LOOKBACK}, entry={ZSCORE_ENTRY}, exit={ZSCORE_EXIT}")

    # Download all data
    print("\n[DATA] Downloading price data...")
    all_tickers = list(set(GROWTH_STOCKS + ["SPY"]))
    all_prices = download_data(all_tickers, OOT_START, OOT_END)

    # Verify we have our target pairs
    for t in ["MSFT", "AAPL", "SNAP", "PINS", "SPY"]:
        if t not in all_prices.columns or all_prices[t].notna().sum() < 100:
            print(f"  ERROR: {t} data insufficient")
            sys.exit(1)

    regime = get_regime(all_prices["SPY"])

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "oot_start": OOT_START, "oot_end": OOT_END,
            "initial_capital": INITIAL_CAPITAL,
            "zscore_lookback": ZSCORE_LOOKBACK,
            "zscore_entry": ZSCORE_ENTRY,
            "zscore_exit": ZSCORE_EXIT,
            "max_hold_days": MAX_HOLD_DAYS,
        },
        "pairs": {},
    }

    # ── MSFT/AAPL ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PAIR 1: MSFT/AAPL")
    print("=" * 70)

    msft_aapl = {}
    msft_aapl["test1_inverse"] = test_inverse(all_prices["MSFT"], all_prices["AAPL"], "MSFT/AAPL", regime)
    msft_aapl["test3_shuffled"] = test_shuffled_entries(all_prices["MSFT"], all_prices["AAPL"], "MSFT/AAPL")
    msft_aapl["test4_time_stability"] = test_time_stability(all_prices["MSFT"], all_prices["AAPL"], "MSFT/AAPL", regime)
    msft_aapl["test5_convergence"] = test_convergence(all_prices["MSFT"], all_prices["AAPL"], "MSFT/AAPL")

    results["pairs"]["MSFT_AAPL"] = msft_aapl

    # ── SNAP/PINS ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PAIR 2: SNAP/PINS")
    print("=" * 70)

    snap_pins = {}
    snap_pins["test1_inverse"] = test_inverse(all_prices["SNAP"], all_prices["PINS"], "SNAP/PINS", regime)
    snap_pins["test3_shuffled"] = test_shuffled_entries(all_prices["SNAP"], all_prices["PINS"], "SNAP/PINS")
    snap_pins["test4_time_stability"] = test_time_stability(all_prices["SNAP"], all_prices["PINS"], "SNAP/PINS", regime)
    snap_pins["test5_convergence"] = test_convergence(all_prices["SNAP"], all_prices["PINS"], "SNAP/PINS")

    results["pairs"]["SNAP_PINS"] = snap_pins

    # ── Test 2: Random Pairs (shared across both) ──────────────────────────
    results["test2_random_pairs"] = test_random_pairs(all_prices, regime)

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    summary = {"MSFT_AAPL": {}, "SNAP_PINS": {}}

    for pair_key, pair_results in [("MSFT_AAPL", msft_aapl), ("SNAP_PINS", snap_pins)]:
        tests_passed = 0
        tests_total = 0

        for test_name, test_result in pair_results.items():
            if "verdict" in test_result:
                tests_total += 1
                if "PASS" in test_result["verdict"]:
                    tests_passed += 1
                print(f"  {pair_key} {test_name}: {test_result['verdict']}")

        # Add random pair test
        if "verdict" in results["test2_random_pairs"]:
            tests_total += 1
            if "PASS" in results["test2_random_pairs"]["verdict"]:
                tests_passed += 1
            print(f"  {pair_key} test2_random_pairs: {results['test2_random_pairs']['verdict']}")

        summary[pair_key] = {
            "tests_passed": tests_passed,
            "tests_total": tests_total,
            "overall_verdict": f"{tests_passed}/{tests_total} adversarial tests passed",
        }

        # Final judgment
        if tests_passed >= 4:
            summary[pair_key]["conclusion"] = "LIKELY REAL EDGE"
        elif tests_passed >= 3:
            summary[pair_key]["conclusion"] = "MARGINAL — needs more data"
        else:
            summary[pair_key]["conclusion"] = "LIKELY SPURIOUS — do not trade"

        print(f"\n  {pair_key}: {summary[pair_key]['overall_verdict']} → {summary[pair_key]['conclusion']}")

    results["summary"] = summary

    # Save results
    output_path = Path("/home/jupiter/Lvl3Quant/data/pairs_adversarial_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
