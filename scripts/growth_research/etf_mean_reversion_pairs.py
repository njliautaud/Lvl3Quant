#!/usr/bin/env python3
"""
ETF Mean-Reversion Pairs Study
================================
Walk-forward sliding window cointegration-based pairs trading on 8 ETF pairs.
Tests for real mean-reversion edge after costs, regime robustness, and
correlation to existing portfolio strategies.

Output: /home/jupiter/Lvl3Quant/output/growth_research/etf_mean_reversion/
"""

import json
import warnings
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/etf_mean_reversion")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Pair definitions ──────────────────────────────────────────────────────────
PAIRS = [
    ("GLD", "GDX", "Gold vs Gold Miners"),
    ("XLE", "USO", "Energy Stocks vs Crude Oil"),
    ("EEM", "EFA", "Emerging vs Developed ex-US"),
    ("TLT", "IEF", "Long vs Intermediate Bonds"),
    ("XLK", "XLF", "Tech vs Financials"),
    ("SPY", "IWM", "Large vs Small Cap"),
    ("HYG", "LQD", "High Yield vs Investment Grade"),
    ("GLD", "TLT", "Gold vs Treasuries"),
]

# Walk-forward params
TRAIN_DAYS = 252
TEST_DAYS = 63
SLIDE_DAYS = 21
ZSCORE_LOOKBACK = 60
ENTRY_Z = 2.0
EXIT_Z = 0.0
STOP_Z = 3.5

# Cost assumption: ~5 bps round-trip per leg (ETF commission-free, but spread + slippage)
COST_PER_TRADE_BPS = 10  # 10 bps total for the pair (5 per leg)

N_PERMUTATIONS = 100
REGIME_GAP_THRESHOLD = 0.50


def download_data(tickers: list[str], start: str = "2004-01-01") -> pd.DataFrame:
    """Download adjusted close prices for all tickers."""
    all_tickers = list(set(tickers))
    print(f"Downloading {len(all_tickers)} tickers from {start}...")
    data = yf.download(all_tickers, start=start, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data
    prices = prices.dropna(how="all")
    print(f"  Got {len(prices)} trading days, {prices.columns.tolist()}")
    return prices


def test_cointegration(y: np.ndarray, x: np.ndarray) -> dict:
    """Engle-Granger cointegration test. Returns ADF stat, p-value, hedge ratio."""
    x_const = add_constant(x)
    model = OLS(y, x_const).fit()
    residuals = model.resid
    adf_result = adfuller(residuals, maxlag=10, autolag="AIC")
    return {
        "adf_stat": adf_result[0],
        "adf_pvalue": adf_result[1],
        "hedge_ratio": model.params[1],
        "intercept": model.params[0],
        "residuals": residuals,
    }


def run_pair_walkforward(prices_a: pd.Series, prices_b: pd.Series, spy_returns: pd.Series) -> dict:
    """
    Walk-forward sliding-window pairs trading for one pair.
    Returns trade list and daily returns.
    """
    # Align
    common = prices_a.dropna().index.intersection(prices_b.dropna().index)
    common = common.intersection(spy_returns.dropna().index)
    pa = prices_a.loc[common]
    pb = prices_b.loc[common]
    spy_ret = spy_returns.loc[common]

    log_a = np.log(pa)
    log_b = np.log(pb)

    n = len(common)
    if n < TRAIN_DAYS + TEST_DAYS:
        return {"error": "Insufficient data", "n_days": n}

    all_trades = []
    daily_returns = pd.Series(0.0, index=common)

    # Walk-forward windows
    window_start = 0
    n_windows = 0
    n_coint_windows = 0

    while window_start + TRAIN_DAYS + TEST_DAYS <= n:
        train_end = window_start + TRAIN_DAYS
        test_end = min(train_end + TEST_DAYS, n)

        # Train: estimate cointegration
        train_log_a = log_a.iloc[window_start:train_end].values
        train_log_b = log_b.iloc[window_start:train_end].values

        coint = test_cointegration(train_log_a, train_log_b)
        n_windows += 1

        is_cointegrated = coint["adf_pvalue"] < 0.05
        if is_cointegrated:
            n_coint_windows += 1

        hedge_ratio = coint["hedge_ratio"]
        intercept = coint["intercept"]

        # Compute spread on test period
        test_idx = common[train_end:test_end]
        test_log_a = log_a.iloc[train_end:test_end]
        test_log_b = log_b.iloc[train_end:test_end]
        spread = test_log_a.values - hedge_ratio * test_log_b.values - intercept

        # Rolling z-score using trailing data (include some training tail for warm-up)
        lookback_start = max(0, train_end - ZSCORE_LOOKBACK)
        full_log_a = log_a.iloc[lookback_start:test_end]
        full_log_b = log_b.iloc[lookback_start:test_end]
        full_spread = full_log_a.values - hedge_ratio * full_log_b.values - intercept
        full_spread_series = pd.Series(full_spread, index=common[lookback_start:test_end])

        rolling_mean = full_spread_series.rolling(ZSCORE_LOOKBACK).mean()
        rolling_std = full_spread_series.rolling(ZSCORE_LOOKBACK).std()
        zscore = (full_spread_series - rolling_mean) / rolling_std

        # Only trade on test period
        test_zscore = zscore.loc[test_idx]

        # Simulate trades
        position = 0  # +1 = long spread, -1 = short spread
        entry_date = None
        entry_z = None

        for i, (date, z) in enumerate(test_zscore.items()):
            if np.isnan(z):
                continue

            if position == 0:
                # Entry signals (only trade if cointegrated in training)
                if is_cointegrated:
                    if z > ENTRY_Z:
                        position = -1  # short spread (short A, long B)
                        entry_date = date
                        entry_z = z
                    elif z < -ENTRY_Z:
                        position = 1  # long spread (long A, short B)
                        entry_date = date
                        entry_z = z
            else:
                # Exit: mean reversion or stop
                exit_signal = False
                stop_hit = False

                if position == -1:
                    if z <= EXIT_Z:
                        exit_signal = True
                    elif z > STOP_Z:
                        exit_signal = True
                        stop_hit = True
                elif position == 1:
                    if z >= EXIT_Z:
                        exit_signal = True
                    elif z < -STOP_Z:
                        exit_signal = True
                        stop_hit = True

                if exit_signal:
                    # Compute P&L
                    entry_idx_pos = common.get_loc(entry_date)
                    exit_idx_pos = common.get_loc(date)
                    ret_a = (pa.iloc[exit_idx_pos] / pa.iloc[entry_idx_pos]) - 1
                    ret_b = (pb.iloc[exit_idx_pos] / pb.iloc[entry_idx_pos]) - 1

                    if position == 1:
                        # Long A, short B (dollar neutral)
                        trade_ret = ret_a - hedge_ratio * ret_b
                    else:
                        # Short A, long B
                        trade_ret = -ret_a + hedge_ratio * ret_b

                    # Subtract costs
                    trade_ret -= COST_PER_TRADE_BPS / 10000

                    duration = (date - entry_date).days

                    all_trades.append({
                        "entry_date": str(entry_date.date()) if hasattr(entry_date, 'date') else str(entry_date),
                        "exit_date": str(date.date()) if hasattr(date, 'date') else str(date),
                        "direction": "long_spread" if position == 1 else "short_spread",
                        "entry_z": round(float(entry_z), 3),
                        "exit_z": round(float(z), 3),
                        "return_bps": round(trade_ret * 10000, 1),
                        "duration_days": duration,
                        "stop_hit": stop_hit,
                        "cointegrated": is_cointegrated,
                    })

                    # Distribute return across holding days for daily return series
                    if duration > 0:
                        daily_ret = trade_ret / duration
                        for d in range(entry_idx_pos, exit_idx_pos):
                            if d < len(daily_returns):
                                daily_returns.iloc[d] += daily_ret

                    position = 0
                    entry_date = None
                    entry_z = None

        window_start += SLIDE_DAYS

    return {
        "trades": all_trades,
        "daily_returns": daily_returns,
        "n_windows": n_windows,
        "n_coint_windows": n_coint_windows,
        "coint_rate": n_coint_windows / max(n_windows, 1),
    }


def compute_metrics(daily_rets: pd.Series, trades: list) -> dict:
    """Compute strategy performance metrics."""
    if len(trades) == 0:
        return {
            "sharpe": 0, "sortino": 0, "max_dd": 0, "profit_factor": 0,
            "win_rate": 0, "n_trades": 0, "avg_duration": 0, "total_return_pct": 0,
            "annual_return_pct": 0, "avg_return_bps": 0,
        }

    # From daily returns
    dr = daily_rets[daily_rets != 0] if daily_rets.any() else daily_rets
    ann_factor = np.sqrt(252)

    mean_daily = daily_rets.mean()
    std_daily = daily_rets.std()
    sharpe = (mean_daily / std_daily * ann_factor) if std_daily > 0 else 0

    downside = daily_rets[daily_rets < 0].std()
    sortino = (mean_daily / downside * ann_factor) if downside > 0 else 0

    cum = (1 + daily_rets).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # From trades
    returns_bps = [t["return_bps"] for t in trades]
    wins = [r for r in returns_bps if r > 0]
    losses = [r for r in returns_bps if r <= 0]

    win_rate = len(wins) / len(trades) if trades else 0
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1e-9
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_duration = np.mean([t["duration_days"] for t in trades])
    total_ret = (cum.iloc[-1] - 1) * 100 if len(cum) > 0 else 0
    n_years = len(daily_rets) / 252
    annual_ret = (((1 + total_ret / 100) ** (1 / max(n_years, 0.1))) - 1) * 100

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd_pct": round(max_dd * 100, 2),
        "profit_factor": round(profit_factor, 3),
        "win_rate": round(win_rate * 100, 1),
        "n_trades": len(trades),
        "avg_duration_days": round(avg_duration, 1),
        "total_return_pct": round(total_ret, 2),
        "annual_return_pct": round(annual_ret, 2),
        "avg_return_bps": round(np.mean(returns_bps), 1),
        "median_return_bps": round(np.median(returns_bps), 1),
    }


def permutation_test(daily_rets: pd.Series, n_perms: int = N_PERMUTATIONS) -> float:
    """Permutation test: shuffle daily returns, compute fraction with higher Sharpe."""
    actual_sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if daily_rets.std() > 0 else 0
    count_higher = 0
    rets = daily_rets.values.copy()
    for _ in range(n_perms):
        np.random.shuffle(rets)
        shuffled = pd.Series(rets)
        s = shuffled.mean() / shuffled.std() * np.sqrt(252) if shuffled.std() > 0 else 0
        if s >= actual_sharpe:
            count_higher += 1
    return count_higher / n_perms


def regime_test(daily_rets: pd.Series, spy_daily_rets: pd.Series) -> dict:
    """R1 regime test: green vs red day performance."""
    common = daily_rets.index.intersection(spy_daily_rets.index)
    dr = daily_rets.loc[common]
    sr = spy_daily_rets.loc[common]

    green_mask = sr > 0
    red_mask = sr < 0

    green_rets = dr[green_mask]
    red_rets = dr[red_mask]

    ann = np.sqrt(252)
    sharpe_green = (green_rets.mean() / green_rets.std() * ann) if green_rets.std() > 0 else 0
    sharpe_red = (red_rets.mean() / red_rets.std() * ann) if red_rets.std() > 0 else 0

    gap = abs(sharpe_green - sharpe_red) / max(abs(sharpe_green), abs(sharpe_red), 1e-9)

    return {
        "sharpe_green": round(sharpe_green, 3),
        "sharpe_red": round(sharpe_red, 3),
        "regime_gap": round(gap, 3),
        "passes_r1": gap < REGIME_GAP_THRESHOLD,
    }


def compute_correlations(daily_rets: pd.Series, spy_rets: pd.Series) -> dict:
    """Compute correlation to SPY and proxy for UPRO/trend-following."""
    common = daily_rets.index.intersection(spy_rets.index)
    dr = daily_rets.loc[common]
    sr = spy_rets.loc[common]

    corr_spy = dr.corr(sr)
    # UPRO proxy = 3x SPY daily
    upro_proxy = sr * 3
    corr_upro = dr.corr(upro_proxy)

    # Trend-following proxy: sign of 200d SMA of SPY * SPY returns
    spy_cum = (1 + sr).cumprod()
    sma200 = spy_cum.rolling(200).mean()
    trend_signal = (spy_cum > sma200).astype(float) * 2 - 1  # +1 or -1
    trend_returns = trend_signal.shift(1) * sr
    trend_returns = trend_returns.dropna()
    common2 = dr.index.intersection(trend_returns.index)
    corr_trend = dr.loc[common2].corr(trend_returns.loc[common2])

    return {
        "corr_spy": round(corr_spy, 3),
        "corr_upro_proxy": round(corr_upro, 3),
        "corr_trend_proxy": round(corr_trend, 3),
    }


def main():
    print("=" * 80)
    print("ETF MEAN-REVERSION PAIRS STUDY")
    print(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    # Collect all tickers
    all_tickers = set(["SPY"])
    for a, b, _ in PAIRS:
        all_tickers.add(a)
        all_tickers.add(b)

    prices = download_data(list(all_tickers))
    spy_returns = prices["SPY"].pct_change().dropna()

    results = {}
    all_trade_details = []

    for etf_a, etf_b, pair_name in PAIRS:
        print(f"\n{'─' * 60}")
        print(f"PAIR: {etf_a}/{etf_b} — {pair_name}")
        print(f"{'─' * 60}")

        if etf_a not in prices.columns or etf_b not in prices.columns:
            print(f"  SKIP: Missing data for {etf_a} or {etf_b}")
            results[f"{etf_a}/{etf_b}"] = {"error": "Missing data"}
            continue

        pa = prices[etf_a].dropna()
        pb = prices[etf_b].dropna()

        # Data overlap
        common = pa.index.intersection(pb.index)
        print(f"  Data: {common[0].date()} to {common[-1].date()} ({len(common)} days)")

        # Full-sample cointegration test
        log_a = np.log(pa.loc[common]).values
        log_b = np.log(pb.loc[common]).values
        full_coint = test_cointegration(log_a, log_b)
        print(f"  Full-sample ADF: stat={full_coint['adf_stat']:.3f}, p={full_coint['adf_pvalue']:.4f}, "
              f"hedge={full_coint['hedge_ratio']:.3f}")
        print(f"  Cointegrated (full sample): {'YES' if full_coint['adf_pvalue'] < 0.05 else 'NO'}")

        # Walk-forward
        wf = run_pair_walkforward(pa, pb, spy_returns)
        if "error" in wf:
            print(f"  ERROR: {wf['error']}")
            results[f"{etf_a}/{etf_b}"] = wf
            continue

        trades = wf["trades"]
        daily_rets = wf["daily_returns"]

        print(f"  WF windows: {wf['n_windows']}, cointegrated: {wf['n_coint_windows']} "
              f"({wf['coint_rate']:.0%})")
        print(f"  Trades: {len(trades)}")

        # Metrics
        metrics = compute_metrics(daily_rets, trades)
        print(f"  Sharpe: {metrics['sharpe']:.3f} | Sortino: {metrics['sortino']:.3f} | "
              f"MaxDD: {metrics['max_dd_pct']:.1f}%")
        print(f"  WR: {metrics['win_rate']:.1f}% | PF: {metrics['profit_factor']:.2f} | "
              f"Avg dur: {metrics['avg_duration_days']:.0f}d")
        print(f"  Avg ret: {metrics['avg_return_bps']:.1f} bps | Med ret: {metrics['median_return_bps']:.1f} bps")
        print(f"  Total return: {metrics['total_return_pct']:.2f}% | Annual: {metrics['annual_return_pct']:.2f}%")

        # Permutation test
        if len(trades) >= 10:
            pval = permutation_test(daily_rets)
            print(f"  Permutation p-value: {pval:.3f} ({'SIGNIFICANT' if pval < 0.05 else 'NOT significant'})")
        else:
            pval = 1.0
            print(f"  Permutation test: SKIPPED (too few trades)")

        # Regime test
        regime = regime_test(daily_rets, spy_returns)
        print(f"  Regime: Sharpe_green={regime['sharpe_green']:.3f}, Sharpe_red={regime['sharpe_red']:.3f}, "
              f"gap={regime['regime_gap']:.3f} ({'PASS' if regime['passes_r1'] else 'FAIL'})")

        # Correlations
        corrs = compute_correlations(daily_rets, spy_returns)
        print(f"  Corr to SPY: {corrs['corr_spy']:.3f} | UPRO proxy: {corrs['corr_upro_proxy']:.3f} | "
              f"Trend proxy: {corrs['corr_trend_proxy']:.3f}")

        # Verdict
        is_viable = (
            metrics["sharpe"] > 0.3
            and metrics["n_trades"] >= 20
            and pval < 0.10
            and regime["passes_r1"]
            and abs(corrs["corr_spy"]) < 0.30
        )
        verdict = "VIABLE CANDIDATE" if is_viable else "NOT VIABLE"
        print(f"  >>> VERDICT: {verdict}")

        pair_result = {
            "pair": f"{etf_a}/{etf_b}",
            "name": pair_name,
            "full_sample_coint_pvalue": round(full_coint["adf_pvalue"], 4),
            "wf_coint_rate": round(wf["coint_rate"], 3),
            **metrics,
            "perm_pvalue": round(pval, 3),
            **regime,
            **corrs,
            "verdict": verdict,
        }
        results[f"{etf_a}/{etf_b}"] = pair_result

        # Tag trades with pair name
        for t in trades:
            t["pair"] = f"{etf_a}/{etf_b}"
        all_trade_details.extend(trades)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    viable = []
    print(f"\n{'Pair':<12} {'Sharpe':>7} {'Sortino':>8} {'WR%':>6} {'PF':>6} {'#Tr':>5} "
          f"{'Perm-p':>7} {'R1':>5} {'rSPY':>6} {'Verdict':<16}")
    print("-" * 85)
    for key, r in results.items():
        if "error" in r:
            print(f"{key:<12} {'ERROR':>7}")
            continue
        print(f"{key:<12} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} {r['win_rate']:>5.1f}% "
              f"{r['profit_factor']:>6.2f} {r['n_trades']:>5} {r['perm_pvalue']:>7.3f} "
              f"{'PASS' if r['passes_r1'] else 'FAIL':>5} {r['corr_spy']:>6.3f} {r['verdict']:<16}")
        if r["verdict"] == "VIABLE CANDIDATE":
            viable.append(key)

    print(f"\nViable pairs: {len(viable)} / {len(PAIRS)}")
    if viable:
        print(f"  {', '.join(viable)}")
    else:
        print("  None — no pair passes all filters (Sharpe>0.3, perm p<0.10, R1 pass, |rSPY|<0.30)")

    # ── Portfolio impact assessment ───────────────────────────────────────────
    print("\n" + "─" * 60)
    print("PORTFOLIO IMPACT ASSESSMENT")
    print("─" * 60)

    if viable:
        # Equal-weight combo of viable pairs
        combo_rets = pd.Series(0.0, index=prices.index)
        for key in viable:
            pair_result = results[key]
            etf_a, etf_b = key.split("/")
            wf = run_pair_walkforward(prices[etf_a], prices[etf_b], spy_returns)
            combo_rets += wf["daily_returns"] / len(viable)

        combo_metrics = compute_metrics(combo_rets, [t for t in all_trade_details if t["pair"] in viable])
        combo_corrs = compute_correlations(combo_rets, spy_returns)
        print(f"  Combined viable pairs: Sharpe={combo_metrics['sharpe']:.3f}, "
              f"corr_SPY={combo_corrs['corr_spy']:.3f}")
        print(f"  Potential diversification benefit: "
              f"{'YES' if abs(combo_corrs['corr_spy']) < 0.20 else 'MARGINAL' if abs(combo_corrs['corr_spy']) < 0.35 else 'NO'}")
    else:
        print("  No viable pairs to combine. Mean-reversion pairs on these ETFs")
        print("  do not show reliable edge after costs in walk-forward testing.")
        print("  This is common — ETF pairs lack the microstructure edge of")
        print("  single-name stat arb. The cointegration relationships are")
        print("  too unstable over time for reliable trading.")

    print("\n" + "─" * 60)
    print("OVERALL VERDICT")
    print("─" * 60)
    if len(viable) >= 2:
        print("  RECOMMEND: Add viable pairs as small allocation (5-10% of portfolio).")
        print("  Low correlation to existing strategies provides diversification.")
    elif len(viable) == 1:
        print("  MARGINAL: One viable pair found. Insufficient for standalone allocation.")
        print("  Consider paper-trading for 6 months before live allocation.")
    else:
        print("  NOT RECOMMENDED: No pairs show reliable mean-reversion edge.")
        print("  ETF pairs are too correlated to macro factors and cointegration")
        print("  relationships break down too frequently for profitable trading.")
        print("  Research budget better spent on other strategies.")

    # ── Save outputs ──────────────────────────────────────────────────────────
    # JSON summary
    summary = {
        "run_date": datetime.now().isoformat(),
        "params": {
            "train_days": TRAIN_DAYS,
            "test_days": TEST_DAYS,
            "slide_days": SLIDE_DAYS,
            "zscore_lookback": ZSCORE_LOOKBACK,
            "entry_z": ENTRY_Z,
            "exit_z": EXIT_Z,
            "stop_z": STOP_Z,
            "cost_bps": COST_PER_TRADE_BPS,
            "n_permutations": N_PERMUTATIONS,
        },
        "pair_results": results,
        "viable_pairs": viable,
        "n_viable": len(viable),
    }

    json_path = OUTPUT_DIR / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {json_path}")

    # CSV trade details
    if all_trade_details:
        trades_df = pd.DataFrame(all_trade_details)
        csv_path = OUTPUT_DIR / "trade_details.csv"
        trades_df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

    # CSV pair metrics
    metrics_rows = []
    for key, r in results.items():
        if "error" not in r:
            metrics_rows.append(r)
    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        csv_path = OUTPUT_DIR / "pair_metrics.csv"
        metrics_df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
