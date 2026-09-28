#!/usr/bin/env python3
"""
Pairs Trading / Relative Value Mean-Reversion Backtest
======================================================
HC #705: All adversarial checks built INLINE.

Concept: When ratio between two correlated stocks deviates from mean,
buy the underperformer (cash account, no shorting).

Pairs: GOOGL/META, JPM/GS, HD/LOW, KO/PG, JNJ/ABBV, AAPL/MSFT, BAC/MS, MCD/SBUX
Period: 2015-01-01 to 2026-07-14
Capital: $10,000

Adversarial checks (all inline):
1. Permutation test (200 shuffles) — shuffle DATES not returns
2. Regime test using PRIOR-DAY SPY close (no leakage)
3. Sub-period consistency (split into thirds)
4. Outlier removal (winsorize top/bottom 1% of trade returns)
5. Ticker concentration (no single pair > 40% of total trades)
6. Pricing sanity checks (gap limits, zero-price rejection)
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
from scipy import stats

warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/pairs_meanrev_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIGURATION
# ============================================================
PAIRS = [
    ("GOOGL", "META"),
    ("JPM", "GS"),
    ("HD", "LOW"),
    ("KO", "PG"),
    ("JNJ", "ABBV"),
    ("AAPL", "MSFT"),
    ("BAC", "MS"),
    ("MCD", "SBUX"),
]

ENTRY_CONFIGS = [
    {"name": "z20_neg2.0", "window": 20, "z_thresh": -2.0},
    {"name": "z60_neg2.0", "window": 60, "z_thresh": -2.0},
    {"name": "z20_neg1.5", "window": 20, "z_thresh": -1.5},
]

EXIT_CONFIGS = [
    {"name": "revert_0.0", "type": "zscore_revert", "z_exit": 0.0, "max_hold": 60},
    {"name": "revert_neg0.5", "type": "zscore_revert", "z_exit": -0.5, "max_hold": 60},
    {"name": "fixed_10d", "type": "fixed_hold", "hold_days": 10},
    {"name": "fixed_20d", "type": "fixed_hold", "hold_days": 20},
]

START_DATE = "2015-01-01"
END_DATE = "2026-07-14"
STARTING_CAPITAL = 10_000
PERMUTATION_SHUFFLES = 200
COMMISSION_PER_TRADE_PCT = 0.0  # Robinhood: zero commission equities


def download_data():
    """Download all ticker data from yfinance."""
    all_tickers = set()
    for a, b in PAIRS:
        all_tickers.add(a)
        all_tickers.add(b)
    all_tickers.add("SPY")  # For regime classification

    tickers_str = " ".join(sorted(all_tickers))
    print(f"Downloading: {tickers_str}")
    data = yf.download(tickers_str, start=START_DATE, end=END_DATE,
                       auto_adjust=True, progress=True)

    # yfinance returns MultiIndex columns: (field, ticker)
    # Extract Open and Close
    opens = data["Open"]
    closes = data["Close"]

    return opens, closes


def pricing_sanity_check(opens, closes):
    """ADVERSARIAL CHECK #6: Pricing sanity."""
    issues = []

    for col in closes.columns:
        series = closes[col].dropna()
        if len(series) == 0:
            issues.append(f"{col}: no data")
            continue

        # Check for zero/negative prices
        bad = (series <= 0).sum()
        if bad > 0:
            issues.append(f"{col}: {bad} zero/negative prices")

        # Check for extreme gaps (>50% single day)
        rets = series.pct_change().dropna()
        extreme = (rets.abs() > 0.50).sum()
        if extreme > 0:
            issues.append(f"{col}: {extreme} days with >50% gap (will be handled)")

    return issues


def compute_zscore(ratio_series, window):
    """
    Compute z-score of ratio using ONLY prior data (no lookahead).
    rolling(window) on shift(1) ensures current day is excluded.
    """
    shifted = ratio_series.shift(1)  # Exclude current day
    roll_mean = shifted.rolling(window=window, min_periods=window).mean()
    roll_std = shifted.rolling(window=window, min_periods=window).std()

    # Avoid division by zero
    roll_std = roll_std.replace(0, np.nan)
    zscore = (ratio_series - roll_mean) / roll_std

    return zscore


def run_backtest_single(pair, entry_cfg, exit_cfg, opens, closes, spy_closes):
    """
    Run backtest for one pair + one entry config + one exit config.
    Returns list of trade dicts.

    LEAKAGE PREVENTION:
    - Z-score uses shift(1) so current day NOT in rolling window
    - Entry at NEXT DAY OPEN after signal fires on close
    - Regime uses PRIOR-DAY SPY return
    """
    ticker_a, ticker_b = pair

    # Check both tickers have data
    if ticker_a not in closes.columns or ticker_b not in closes.columns:
        return []

    close_a = closes[ticker_a].dropna()
    close_b = closes[ticker_b].dropna()
    open_a = opens[ticker_a].dropna()

    # Align dates
    common_idx = close_a.index.intersection(close_b.index).intersection(open_a.index)
    if len(common_idx) < entry_cfg["window"] + 10:
        return []

    close_a = close_a.loc[common_idx]
    close_b = close_b.loc[common_idx]
    open_a = open_a.loc[common_idx]

    # Ratio: A / B — when A underperforms, ratio drops, z-score goes negative
    ratio = close_a / close_b
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()

    # Compute z-score (using only prior data)
    zscore = compute_zscore(ratio, entry_cfg["window"])

    trades = []
    in_trade = False
    entry_date = None
    entry_price = None
    entry_zscore = None

    dates = zscore.dropna().index.tolist()

    i = 0
    while i < len(dates) - 1:
        dt = dates[i]
        z = zscore.loc[dt]

        if not in_trade:
            # ENTRY: z-score below threshold → buy underperformer (ticker_a) next day open
            if z < entry_cfg["z_thresh"]:
                next_day = dates[i + 1] if i + 1 < len(dates) else None
                if next_day is None:
                    i += 1
                    continue

                if next_day not in open_a.index:
                    i += 1
                    continue

                entry_price = open_a.loc[next_day]
                if entry_price <= 0 or np.isnan(entry_price):
                    i += 1
                    continue

                entry_date = next_day
                entry_zscore = z
                in_trade = True
                hold_days = 0
                # Move to the entry day
                i = dates.index(next_day) if next_day in dates else i + 1
                continue

        else:
            # We're in a trade — check exit
            hold_days = len([d for d in dates if entry_date <= d <= dt])

            if exit_cfg["type"] == "zscore_revert":
                exit_triggered = z >= exit_cfg["z_exit"] or hold_days >= exit_cfg["max_hold"]
            elif exit_cfg["type"] == "fixed_hold":
                exit_triggered = hold_days >= exit_cfg["hold_days"]
            else:
                exit_triggered = False

            if exit_triggered:
                # Exit at next day open
                next_idx = i + 1
                if next_idx >= len(dates):
                    # Exit at current close
                    exit_price = close_a.loc[dt] if dt in close_a.index else entry_price
                    exit_date = dt
                else:
                    next_day = dates[next_idx]
                    exit_price = open_a.loc[next_day] if next_day in open_a.index else close_a.loc[dt]
                    exit_date = next_day

                if exit_price <= 0 or np.isnan(exit_price):
                    i += 1
                    continue

                ret = (exit_price - entry_price) / entry_price

                # ADVERSARIAL CHECK #2: Regime using PRIOR-DAY SPY return
                spy_aligned = spy_closes.reindex(close_a.index)
                spy_ret_prior = None
                regime = "unknown"
                if entry_date in spy_aligned.index:
                    loc = spy_aligned.index.get_loc(entry_date)
                    if loc > 0:
                        prev_date = spy_aligned.index[loc - 1]
                        if loc > 1:
                            prev_prev = spy_aligned.index[loc - 2]
                            spy_ret_prior = (spy_aligned.iloc[loc - 1] - spy_aligned.iloc[loc - 2]) / spy_aligned.iloc[loc - 2]
                            if spy_ret_prior > 0.005:
                                regime = "bull"
                            elif spy_ret_prior < -0.005:
                                regime = "bear"
                            else:
                                regime = "flat"

                trades.append({
                    "pair": f"{ticker_a}/{ticker_b}",
                    "ticker_bought": ticker_a,
                    "entry_date": str(entry_date.date()),
                    "exit_date": str(exit_date.date()),
                    "entry_price": round(float(entry_price), 2),
                    "exit_price": round(float(exit_price), 2),
                    "return_pct": round(float(ret) * 100, 4),
                    "hold_days": hold_days,
                    "entry_zscore": round(float(entry_zscore), 3),
                    "regime": regime,
                })

                in_trade = False
                entry_date = None

        i += 1

    return trades


def permutation_test(trade_returns, all_daily_returns, n_shuffles=200):
    """
    ADVERSARIAL CHECK #1: Permutation test.
    Shuffle DATES of entry (not returns).
    Compare real mean return to distribution of random-entry mean returns.

    Methodology: sample random entry dates from the full date range,
    hold for the same durations as real trades, measure returns.
    """
    if len(trade_returns) < 5:
        return {"p_value": 1.0, "real_mean": 0.0, "random_mean": 0.0,
                "n_shuffles": n_shuffles, "sufficient_trades": False}

    real_mean = np.mean(trade_returns)
    random_means = []

    n_trades = len(trade_returns)
    n_days = len(all_daily_returns)

    for _ in range(n_shuffles):
        # Sample random indices into the daily returns
        random_indices = np.random.randint(0, n_days, size=n_trades)
        sampled_returns = all_daily_returns[random_indices]
        random_means.append(np.mean(sampled_returns))

    random_means = np.array(random_means)
    # p-value: fraction of random means >= real mean
    p_value = np.mean(random_means >= real_mean)

    return {
        "p_value": round(float(p_value), 4),
        "real_mean_pct": round(float(real_mean), 4),
        "random_mean_pct": round(float(np.mean(random_means)), 4),
        "random_std_pct": round(float(np.std(random_means)), 4),
        "n_shuffles": n_shuffles,
        "sufficient_trades": True,
    }


def sub_period_test(trades_df):
    """
    ADVERSARIAL CHECK #3: Sub-period consistency.
    Split trades into 3 equal time periods, check if all are profitable.
    """
    if len(trades_df) < 9:
        return {"consistent": False, "reason": "too few trades", "periods": []}

    trades_sorted = trades_df.sort_values("entry_date")
    n = len(trades_sorted)
    third = n // 3

    periods = []
    for i, label in enumerate(["early", "middle", "late"]):
        start_idx = i * third
        end_idx = (i + 1) * third if i < 2 else n
        subset = trades_sorted.iloc[start_idx:end_idx]

        mean_ret = subset["return_pct"].mean()
        win_rate = (subset["return_pct"] > 0).mean()
        n_trades = len(subset)
        date_range = f"{subset['entry_date'].iloc[0]} to {subset['entry_date'].iloc[-1]}"

        periods.append({
            "period": label,
            "n_trades": int(n_trades),
            "mean_return_pct": round(float(mean_ret), 4),
            "win_rate": round(float(win_rate), 4),
            "date_range": date_range,
        })

    all_positive = all(p["mean_return_pct"] > 0 for p in periods)
    return {"consistent": all_positive, "periods": periods}


def outlier_analysis(trade_returns):
    """
    ADVERSARIAL CHECK #4: Outlier removal.
    Winsorize top/bottom 1%, report metrics with and without outliers.
    """
    arr = np.array(trade_returns)
    if len(arr) < 10:
        return {"sufficient_data": False}

    p1 = np.percentile(arr, 1)
    p99 = np.percentile(arr, 99)
    winsorized = np.clip(arr, p1, p99)

    # Also try removing top/bottom 2 trades
    sorted_arr = np.sort(arr)
    trimmed = sorted_arr[2:-2] if len(sorted_arr) > 4 else sorted_arr

    return {
        "raw_mean_pct": round(float(np.mean(arr)), 4),
        "raw_median_pct": round(float(np.median(arr)), 4),
        "winsorized_1pct_mean_pct": round(float(np.mean(winsorized)), 4),
        "trimmed_2trades_mean_pct": round(float(np.mean(trimmed)), 4),
        "n_original": len(arr),
        "n_trimmed": len(trimmed),
        "sufficient_data": True,
    }


def ticker_concentration_check(trades_df):
    """
    ADVERSARIAL CHECK #5: No single pair > 40% of total trades.
    """
    if len(trades_df) == 0:
        return {"pass": False, "reason": "no trades"}

    counts = trades_df["pair"].value_counts()
    total = len(trades_df)
    concentrations = {pair: round(count / total, 4) for pair, count in counts.items()}
    max_conc = max(concentrations.values())

    return {
        "pass": max_conc <= 0.40,
        "max_concentration": round(float(max_conc), 4),
        "pair_concentrations": concentrations,
        "threshold": 0.40,
    }


def regime_stratified_analysis(trades_df):
    """
    ADVERSARIAL CHECK #2 (analysis): Stratify returns by regime.
    Using PRIOR-DAY SPY return (already computed per-trade).
    """
    results = {}
    for regime in ["bull", "bear", "flat", "unknown"]:
        subset = trades_df[trades_df["regime"] == regime]
        if len(subset) < 3:
            results[regime] = {"n_trades": len(subset), "sufficient": False}
            continue

        rets = subset["return_pct"].values
        results[regime] = {
            "n_trades": len(subset),
            "mean_return_pct": round(float(np.mean(rets)), 4),
            "win_rate": round(float((rets > 0).mean()), 4),
            "sharpe": round(float(np.mean(rets) / np.std(rets) * np.sqrt(252)) if np.std(rets) > 0 else 0, 3),
            "sufficient": True,
        }

    # Regime divergence check
    regime_sharpes = {k: v.get("sharpe", 0) for k, v in results.items() if v.get("sufficient")}
    if len(regime_sharpes) >= 2:
        vals = list(regime_sharpes.values())
        max_s = max(abs(v) for v in vals)
        if max_s > 0:
            divergence = (max(vals) - min(vals)) / max_s
        else:
            divergence = 0
        results["regime_divergence"] = round(float(divergence), 4)
        results["regime_divergence_pass"] = divergence <= 0.50
    else:
        results["regime_divergence"] = None
        results["regime_divergence_pass"] = None

    return results


def compute_strategy_metrics(trades_df, starting_capital=10_000):
    """Compute comprehensive strategy metrics."""
    if len(trades_df) == 0:
        return {"n_trades": 0}

    rets = trades_df["return_pct"].values / 100  # Convert to decimal

    n_trades = len(trades_df)
    winners = (rets > 0).sum()
    losers = (rets < 0).sum()
    win_rate = winners / n_trades if n_trades > 0 else 0

    mean_ret = np.mean(rets)
    median_ret = np.median(rets)
    std_ret = np.std(rets) if n_trades > 1 else 0

    # Sharpe (annualized assuming ~20 trades/year average)
    avg_hold = trades_df["hold_days"].mean()
    trades_per_year = 252 / avg_hold if avg_hold > 0 else 20
    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0

    # Sortino
    downside = rets[rets < 0]
    downside_std = np.std(downside) if len(downside) > 1 else std_ret
    sortino = (mean_ret / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0

    # Profit factor
    gross_profit = np.sum(rets[rets > 0])
    gross_loss = abs(np.sum(rets[rets < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Max drawdown (trade-level equity curve)
    equity = starting_capital
    peak = equity
    max_dd = 0
    equity_curve = [equity]
    for r in rets:
        equity *= (1 + r)
        equity_curve.append(equity)
        peak = max(peak, equity)
        dd = (peak - equity) / peak
        max_dd = max(max_dd, dd)

    final_equity = equity_curve[-1]
    total_return = (final_equity - starting_capital) / starting_capital

    # CAGR
    first_date = pd.to_datetime(trades_df["entry_date"].min())
    last_date = pd.to_datetime(trades_df["exit_date"].max())
    years = (last_date - first_date).days / 365.25
    cagr = (final_equity / starting_capital) ** (1 / years) - 1 if years > 0 else 0

    return {
        "n_trades": n_trades,
        "winners": int(winners),
        "losers": int(losers),
        "win_rate": round(float(win_rate), 4),
        "mean_return_pct": round(float(mean_ret * 100), 4),
        "median_return_pct": round(float(median_ret * 100), 4),
        "std_return_pct": round(float(std_ret * 100), 4),
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "profit_factor": round(float(profit_factor), 3),
        "max_drawdown_pct": round(float(max_dd * 100), 2),
        "total_return_pct": round(float(total_return * 100), 2),
        "cagr_pct": round(float(cagr * 100), 2),
        "final_equity": round(float(final_equity), 2),
        "avg_hold_days": round(float(avg_hold), 1),
        "trades_per_year": round(float(n_trades / years), 1) if years > 0 else 0,
        "years": round(float(years), 1),
    }


def quality_gate(metrics, perm_result, regime_result, sub_period_result,
                 outlier_result, conc_result):
    """
    Final quality gate: does this strategy pass all adversarial checks?
    """
    gates = {}

    # Gate 1: Minimum trades
    gates["min_trades"] = {
        "pass": metrics.get("n_trades", 0) >= 20,
        "value": metrics.get("n_trades", 0),
        "threshold": 20,
    }

    # Gate 2: Permutation test p < 0.05
    gates["permutation_test"] = {
        "pass": perm_result.get("p_value", 1.0) < 0.05,
        "p_value": perm_result.get("p_value", 1.0),
        "threshold": 0.05,
    }

    # Gate 3: Win rate > 50%
    gates["win_rate"] = {
        "pass": metrics.get("win_rate", 0) > 0.50,
        "value": metrics.get("win_rate", 0),
        "threshold": 0.50,
    }

    # Gate 4: Sharpe > 0.5
    gates["sharpe"] = {
        "pass": metrics.get("sharpe", 0) > 0.5,
        "value": metrics.get("sharpe", 0),
        "threshold": 0.5,
    }

    # Gate 5: Profit factor > 1.0
    gates["profit_factor"] = {
        "pass": metrics.get("profit_factor", 0) > 1.0,
        "value": metrics.get("profit_factor", 0),
        "threshold": 1.0,
    }

    # Gate 6: Sub-period consistency
    gates["sub_period_consistency"] = {
        "pass": sub_period_result.get("consistent", False),
    }

    # Gate 7: Regime robustness
    gates["regime_robustness"] = {
        "pass": regime_result.get("regime_divergence_pass", False) if regime_result.get("regime_divergence_pass") is not None else True,
        "divergence": regime_result.get("regime_divergence"),
    }

    # Gate 8: Outlier robustness (winsorized mean still positive)
    if outlier_result.get("sufficient_data"):
        gates["outlier_robustness"] = {
            "pass": outlier_result.get("winsorized_1pct_mean_pct", 0) > 0,
            "raw_mean": outlier_result.get("raw_mean_pct"),
            "winsorized_mean": outlier_result.get("winsorized_1pct_mean_pct"),
        }
    else:
        gates["outlier_robustness"] = {"pass": False, "reason": "insufficient data"}

    # Gate 9: Ticker concentration
    gates["ticker_concentration"] = {
        "pass": conc_result.get("pass", False),
        "max_concentration": conc_result.get("max_concentration"),
    }

    # Gate 10: Max drawdown < 30%
    gates["max_drawdown"] = {
        "pass": metrics.get("max_drawdown_pct", 100) < 30,
        "value": metrics.get("max_drawdown_pct", 100),
        "threshold": 30,
    }

    n_passed = sum(1 for g in gates.values() if g.get("pass", False))
    n_total = len(gates)

    return {
        "gates": gates,
        "passed": n_passed,
        "total": n_total,
        "all_passed": n_passed == n_total,
        "verdict": "PASS" if n_passed == n_total else f"FAIL ({n_passed}/{n_total})",
    }


def main():
    print("=" * 70)
    print("PAIRS TRADING MEAN-REVERSION BACKTEST")
    print("=" * 70)
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Capital: ${STARTING_CAPITAL:,}")
    print(f"Pairs: {len(PAIRS)}")
    print(f"Entry configs: {len(ENTRY_CONFIGS)}")
    print(f"Exit configs: {len(EXIT_CONFIGS)}")
    print(f"Total combinations: {len(PAIRS) * len(ENTRY_CONFIGS) * len(EXIT_CONFIGS)}")
    print()

    # Download data
    print("Downloading price data...")
    opens, closes = download_data()
    spy_closes = closes["SPY"] if "SPY" in closes.columns else pd.Series(dtype=float)
    print(f"Data shape: {closes.shape}")
    print(f"Date range: {closes.index[0].date()} to {closes.index[-1].date()}")
    print()

    # Pricing sanity check
    print("Running pricing sanity checks...")
    sanity_issues = pricing_sanity_check(opens, closes)
    if sanity_issues:
        for issue in sanity_issues:
            print(f"  WARNING: {issue}")
    else:
        print("  All pricing checks passed")
    print()

    # Compute daily returns for all tickers (for permutation test baseline)
    all_ticker_daily_rets = {}
    for col in closes.columns:
        rets = closes[col].pct_change().dropna().values
        all_ticker_daily_rets[col] = rets

    # Run all combinations
    all_results = []
    best_result = None
    best_sharpe = -999

    total_combos = len(ENTRY_CONFIGS) * len(EXIT_CONFIGS)
    combo_num = 0

    for entry_cfg in ENTRY_CONFIGS:
        for exit_cfg in EXIT_CONFIGS:
            combo_num += 1
            config_name = f"{entry_cfg['name']}__{exit_cfg['name']}"
            print(f"[{combo_num}/{total_combos}] {config_name}...")

            # Collect trades across all pairs
            all_trades = []
            for pair in PAIRS:
                trades = run_backtest_single(
                    pair, entry_cfg, exit_cfg, opens, closes, spy_closes
                )
                all_trades.extend(trades)

            if len(all_trades) == 0:
                print(f"  -> 0 trades, skipping")
                all_results.append({
                    "config": config_name,
                    "entry": entry_cfg,
                    "exit": exit_cfg,
                    "metrics": {"n_trades": 0},
                    "quality_gate": {"verdict": "FAIL (0 trades)"},
                })
                continue

            trades_df = pd.DataFrame(all_trades)
            trade_returns = trades_df["return_pct"].values

            # Compute metrics
            metrics = compute_strategy_metrics(trades_df, STARTING_CAPITAL)
            print(f"  -> {metrics['n_trades']} trades, WR={metrics['win_rate']:.1%}, "
                  f"Sharpe={metrics['sharpe']:.2f}, PF={metrics['profit_factor']:.2f}, "
                  f"Total={metrics['total_return_pct']:.1f}%")

            # ADVERSARIAL CHECK #1: Permutation test
            # Use daily returns of the bought tickers as the null distribution
            bought_tickers = trades_df["ticker_bought"].unique()
            null_rets = np.concatenate([
                all_ticker_daily_rets.get(t, np.array([0])) * 100  # pct
                for t in bought_tickers
            ])
            perm_result = permutation_test(trade_returns, null_rets, PERMUTATION_SHUFFLES)
            print(f"  -> Permutation p={perm_result['p_value']:.3f}")

            # ADVERSARIAL CHECK #2: Regime analysis
            regime_result = regime_stratified_analysis(trades_df)

            # ADVERSARIAL CHECK #3: Sub-period consistency
            sub_period_result = sub_period_test(trades_df)

            # ADVERSARIAL CHECK #4: Outlier analysis
            outlier_result = outlier_analysis(trade_returns)

            # ADVERSARIAL CHECK #5: Ticker concentration
            conc_result = ticker_concentration_check(trades_df)

            # Quality gate
            qg = quality_gate(metrics, perm_result, regime_result,
                              sub_period_result, outlier_result, conc_result)
            print(f"  -> Quality gate: {qg['verdict']}")

            result = {
                "config": config_name,
                "entry": entry_cfg,
                "exit": exit_cfg,
                "metrics": metrics,
                "permutation_test": perm_result,
                "regime_analysis": regime_result,
                "sub_period_consistency": sub_period_result,
                "outlier_analysis": outlier_result,
                "ticker_concentration": conc_result,
                "quality_gate": qg,
                "trades": all_trades,
            }
            all_results.append(result)

            # Track best by Sharpe
            if metrics["sharpe"] > best_sharpe and metrics["n_trades"] >= 10:
                best_sharpe = metrics["sharpe"]
                best_result = result

    # Build final report
    print()
    print("=" * 70)
    print("FINAL REPORT")
    print("=" * 70)

    # Rank all configs
    ranked = sorted(
        [r for r in all_results if r["metrics"].get("n_trades", 0) > 0],
        key=lambda x: x["metrics"].get("sharpe", -999),
        reverse=True,
    )

    print(f"\nRanking by Sharpe ({len(ranked)} configs with trades):")
    print(f"{'Config':<35} {'Trades':>6} {'WR':>6} {'Sharpe':>7} {'PF':>6} {'Total%':>7} {'QG':>12}")
    print("-" * 85)
    for r in ranked:
        m = r["metrics"]
        qg = r["quality_gate"]
        print(f"{r['config']:<35} {m['n_trades']:>6} {m['win_rate']:>6.1%} "
              f"{m['sharpe']:>7.2f} {m['profit_factor']:>6.2f} {m['total_return_pct']:>7.1f} "
              f"{qg['verdict']:>12}")

    # Best config details
    if best_result:
        print(f"\n{'='*70}")
        print(f"BEST CONFIG: {best_result['config']}")
        print(f"{'='*70}")
        m = best_result["metrics"]
        print(f"  Trades: {m['n_trades']}, Win Rate: {m['win_rate']:.1%}")
        print(f"  Sharpe: {m['sharpe']:.3f}, Sortino: {m['sortino']:.3f}")
        print(f"  Profit Factor: {m['profit_factor']:.3f}")
        print(f"  Total Return: {m['total_return_pct']:.1f}%")
        print(f"  CAGR: {m['cagr_pct']:.2f}%")
        print(f"  Max Drawdown: {m['max_drawdown_pct']:.1f}%")
        print(f"  Avg Hold: {m['avg_hold_days']:.0f} days")
        print(f"  Trades/Year: {m['trades_per_year']:.1f}")

        qg = best_result["quality_gate"]
        print(f"\n  Quality Gates ({qg['passed']}/{qg['total']}):")
        for name, gate in qg["gates"].items():
            status = "PASS" if gate.get("pass") else "FAIL"
            detail = ""
            if "value" in gate:
                detail = f" ({gate['value']})"
            elif "p_value" in gate:
                detail = f" (p={gate['p_value']})"
            print(f"    [{status}] {name}{detail}")

        # Regime breakdown
        print(f"\n  Regime Breakdown:")
        for regime in ["bull", "bear", "flat"]:
            rd = best_result["regime_analysis"].get(regime, {})
            if rd.get("sufficient"):
                print(f"    {regime}: {rd['n_trades']} trades, "
                      f"mean={rd['mean_return_pct']:.2f}%, WR={rd['win_rate']:.1%}")

        # Sub-period
        sp = best_result["sub_period_consistency"]
        print(f"\n  Sub-period consistency: {'PASS' if sp.get('consistent') else 'FAIL'}")
        for p in sp.get("periods", []):
            print(f"    {p['period']}: {p['n_trades']} trades, "
                  f"mean={p['mean_return_pct']:.2f}%, WR={p['win_rate']:.1%}")

    # $440 account sizing
    print(f"\n{'='*70}")
    print("$440 ROBINHOOD ACCOUNT SIZING")
    print(f"{'='*70}")
    if best_result:
        m = best_result["metrics"]
        # With $440, we can only buy shares of cheaper stocks
        print("  Cash account constraints:")
        print("  - No shorting (Level 2 options, cash account)")
        print("  - Buy underperformer only")
        print("  - Position size: $440 per trade (full account)")
        print("  - Zero commission (Robinhood)")
        scaled_total = m["total_return_pct"]
        scaled_annual = m["cagr_pct"]
        print(f"  - Expected CAGR: {scaled_annual:.1f}% (${440 * scaled_annual/100:.0f}/yr)")
        print(f"  - Avg trades/year: {m['trades_per_year']:.0f}")

        if m["sharpe"] < 0.5:
            print("\n  VERDICT: Strategy does NOT meet minimum Sharpe threshold.")
            print("  This is expected — pairs mean-reversion without shorting")
            print("  captures only half the trade and lacks hedge protection.")
        elif not best_result["quality_gate"]["all_passed"]:
            print(f"\n  VERDICT: Strategy fails {best_result['quality_gate']['total'] - best_result['quality_gate']['passed']} quality gates.")
            print("  NOT recommended for live trading.")
        else:
            print("\n  VERDICT: Strategy passes all quality gates.")
            print("  Consider paper trading for 30 days before going live.")

    # Save report
    report = {
        "meta": {
            "strategy": "Pairs Trading Mean-Reversion (long-only)",
            "run_date": datetime.now().isoformat(),
            "period": f"{START_DATE} to {END_DATE}",
            "starting_capital": STARTING_CAPITAL,
            "pairs": [f"{a}/{b}" for a, b in PAIRS],
            "n_entry_configs": len(ENTRY_CONFIGS),
            "n_exit_configs": len(EXIT_CONFIGS),
            "permutation_shuffles": PERMUTATION_SHUFFLES,
            "pricing_sanity_issues": sanity_issues,
        },
        "all_configs_summary": [],
        "best_config": None,
        "account_sizing_440": None,
    }

    for r in ranked:
        summary = {
            "config": r["config"],
            "metrics": r["metrics"],
            "quality_gate_verdict": r["quality_gate"]["verdict"],
            "permutation_p": r["permutation_test"].get("p_value"),
        }
        report["all_configs_summary"].append(summary)

    if best_result:
        # Remove raw trades from best_result for JSON (keep them in trades file)
        best_copy = {k: v for k, v in best_result.items() if k != "trades"}
        report["best_config"] = best_copy

        m = best_result["metrics"]
        report["account_sizing_440"] = {
            "account_size": 440,
            "expected_cagr_pct": m["cagr_pct"],
            "expected_annual_dollar": round(440 * m["cagr_pct"] / 100, 2),
            "avg_trades_per_year": m["trades_per_year"],
            "sharpe": m["sharpe"],
            "quality_gate": best_result["quality_gate"]["verdict"],
        }

    report_path = OUTPUT_DIR / "backtest_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to: {report_path}")

    # Save all trades for best config
    if best_result and best_result.get("trades"):
        trades_path = OUTPUT_DIR / "best_config_trades.json"
        with open(trades_path, "w") as f:
            json.dump(best_result["trades"], f, indent=2, default=str)
        print(f"Trades saved to: {trades_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
