"""
ETF Pairs Trading — Walk-Forward Validated Statistical Arbitrage
================================================================
HC #0:   Sliding walk-forward ONLY (24mo train, 6mo OOT, sliding)
HC #428: R1 regime-agnostic validation, R2 MFE-within-horizon
HC #705: Adversarial checks (permutation, regime, sub-period, outlier, coint stability)

Universe: 10 economically-linked ETF pairs
Method:   Log-ratio z-score mean reversion with Engle-Granger cointegration gate
Costs:    5 bps per side per leg = 20 bps total RT
Data:     yfinance
"""

import os
import sys
import json
import warnings
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from statsmodels.tsa.stattools import coint

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = OUTPUT_DIR / "etf_pairs_trading_wf_results.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─── ETF PAIRS (with economic rationale) ────────────────────────────────────
ETF_PAIRS = [
    ("XLK", "XLY",  "Tech vs Consumer Discretionary (growth-sensitive)"),
    ("XLE", "XOP",  "Energy sector vs Oil Explorers"),
    ("XLF", "KRE",  "Financials vs Regional Banks"),
    ("GLD", "GDX",  "Gold vs Gold Miners"),
    ("TLT", "IEF",  "Long vs Intermediate Treasury"),
    ("SPY", "IWM",  "Large Cap vs Small Cap"),
    ("QQQ", "XLK",  "Nasdaq vs Tech Sector"),
    ("EEM", "EFA",  "Emerging vs Developed International"),
    ("XLU", "XLP",  "Utilities vs Staples (defensive)"),
    ("HYG", "LQD",  "High Yield vs Investment Grade Credit"),
]

# Walk-forward params (HC #0: sliding only)
# 24 months training ≈ 504 trading days, 6 months OOT ≈ 126 trading days
TRAIN_DAYS = 504
TEST_DAYS = 126
DATA_START = "2006-01-01"  # need history before first train window
DATA_END = "2026-07-16"

# Parameter grid to optimize on training window
LOOKBACK_OPTIONS = [20, 60, 120]
Z_ENTRY_OPTIONS = [1.5, 2.0, 2.5]
Z_EXIT_OPTIONS = [0.0, 0.25, 0.5]
Z_STOP = 4.0  # fixed stop-loss

# Cointegration gate
COINT_PVALUE_THRESHOLD = 0.05

# Transaction costs: 5 bps per side per leg
COST_PER_SIDE = 0.0005
TOTAL_RT_COST = 4 * COST_PER_SIDE  # 2 legs x 2 sides = 20 bps

# Adversarial
N_PERMS = 200
N_SUBPERIODS = 4
OUTLIER_REMOVE_PCT = 5  # remove top N trades by absolute PnL


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
def download_data():
    """Download adjusted close prices for all tickers."""
    all_tickers = list(set([t for pair in ETF_PAIRS for t in pair[:2]] + ["SPY"]))
    log.info(f"Downloading {len(all_tickers)} tickers...")

    data = yf.download(all_tickers, start=DATA_START, end=DATA_END,
                       auto_adjust=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data

    prices = prices.dropna(how="all").ffill()
    log.info(f"Data: {len(prices)} days, {prices.index[0].date()} to {prices.index[-1].date()}")
    return prices


# ─────────────────────────────────────────────────────────────────────────────
# COINTEGRATION TEST
# ─────────────────────────────────────────────────────────────────────────────
def test_cointegration(prices_a, prices_b):
    """Engle-Granger cointegration test. Returns (is_cointegrated, p_value)."""
    try:
        # Drop NaN
        mask = prices_a.notna() & prices_b.notna()
        a, b = prices_a[mask].values, prices_b[mask].values
        if len(a) < 50:
            return False, 1.0
        _, p_value, _ = coint(a, b)
        return p_value < COINT_PVALUE_THRESHOLD, float(p_value)
    except Exception:
        return False, 1.0


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-FOLD BACKTEST
# ─────────────────────────────────────────────────────────────────────────────
def run_fold_backtest(prices_a, prices_b, lookback, z_entry, z_exit, z_stop):
    """
    Run pairs trading on a single test window.
    prices_a, prices_b are the TEST window prices only.
    lookback data should already be prepended.

    Returns list of trades.
    """
    # Compute log ratio and z-score using rolling lookback
    log_ratio = np.log(prices_a / prices_b)
    roll_mean = log_ratio.rolling(window=lookback).mean()
    roll_std = log_ratio.rolling(window=lookback).std()
    z_score = (log_ratio - roll_mean) / roll_std.replace(0, np.nan)

    trades = []
    position = 0  # 0=flat, 1=long_spread, -1=short_spread
    entry_date = entry_z = entry_pa = entry_pb = None

    for i in range(lookback, len(z_score)):
        z = z_score.iloc[i]
        pa = prices_a.iloc[i]
        pb = prices_b.iloc[i]
        date = prices_a.index[i]

        if np.isnan(z):
            continue

        if position == 0:
            if z > z_entry:
                # Spread too high: short A, long B
                position = -1
                entry_date, entry_z, entry_pa, entry_pb = date, z, pa, pb
            elif z < -z_entry:
                # Spread too low: long A, short B
                position = 1
                entry_date, entry_z, entry_pa, entry_pb = date, z, pa, pb
        else:
            should_exit = False
            exit_reason = ""

            if position == -1:
                if z <= z_exit:
                    should_exit, exit_reason = True, "mean_reversion"
                elif z > z_stop:
                    should_exit, exit_reason = True, "stop_loss"
            else:
                if z >= -z_exit:
                    should_exit, exit_reason = True, "mean_reversion"
                elif z < -z_stop:
                    should_exit, exit_reason = True, "stop_loss"

            if should_exit:
                ret_a = (pa - entry_pa) / entry_pa
                ret_b = (pb - entry_pb) / entry_pb

                if position == -1:
                    gross_pnl = -ret_a + ret_b
                else:
                    gross_pnl = ret_a - ret_b

                net_pnl = gross_pnl - TOTAL_RT_COST
                hold_days = (date - entry_date).days

                trades.append({
                    "entry_date": str(entry_date.date()),
                    "exit_date": str(date.date()),
                    "direction": "short_spread" if position == -1 else "long_spread",
                    "entry_z": float(entry_z),
                    "exit_z": float(z),
                    "exit_reason": exit_reason,
                    "gross_pnl": float(gross_pnl),
                    "net_pnl": float(net_pnl),
                    "hold_days": int(hold_days),
                })
                position = 0

    # Force close at window end
    if position != 0:
        pa = prices_a.iloc[-1]
        pb = prices_b.iloc[-1]
        date = prices_a.index[-1]
        ret_a = (pa - entry_pa) / entry_pa
        ret_b = (pb - entry_pb) / entry_pb
        gross_pnl = (-ret_a + ret_b) if position == -1 else (ret_a - ret_b)
        net_pnl = gross_pnl - TOTAL_RT_COST
        hold_days = (date - entry_date).days
        trades.append({
            "entry_date": str(entry_date.date()),
            "exit_date": str(date.date()),
            "direction": "short_spread" if position == -1 else "long_spread",
            "entry_z": float(entry_z),
            "exit_z": float(z_score.iloc[-1]) if not np.isnan(z_score.iloc[-1]) else 0.0,
            "exit_reason": "window_end",
            "gross_pnl": float(gross_pnl),
            "net_pnl": float(net_pnl),
            "hold_days": int(hold_days),
        })

    return trades


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(trades, label=""):
    """Compute risk-adjusted metrics from trade list."""
    if not trades or len(trades) < 3:
        return None

    pnls = np.array([t["net_pnl"] for t in trades])
    n = len(pnls)
    win_rate = float(np.mean(pnls > 0))
    avg_pnl = float(np.mean(pnls))
    total_pnl = float(np.sum(pnls))

    # Profit factor
    wins = np.sum(pnls[pnls > 0])
    losses = np.abs(np.sum(pnls[pnls < 0]))
    pf = float(wins / losses) if losses > 0 else 99.9

    # Time span for annualization
    first = pd.Timestamp(trades[0]["entry_date"])
    last = pd.Timestamp(trades[-1]["exit_date"])
    years = max((last - first).days / 365.25, 0.25)
    trades_per_year = n / years

    # CAGR from cumulative return
    cum_return = np.sum(pnls)  # as fraction of capital
    cagr = float((1 + cum_return) ** (1 / years) - 1) if cum_return > -1 else -1.0

    # Sharpe (annualized from per-trade)
    if np.std(pnls) > 1e-10:
        sharpe = float((np.mean(pnls) / np.std(pnls)) * np.sqrt(trades_per_year))
    else:
        sharpe = 0.0

    # Sortino
    downside = pnls[pnls < 0]
    if len(downside) > 0 and np.std(downside) > 1e-10:
        sortino = float((np.mean(pnls) / np.std(downside)) * np.sqrt(trades_per_year))
    else:
        sortino = 99.9 if avg_pnl > 0 else 0.0

    # Max drawdown on cumulative PnL
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_dd = float(np.min(dd))

    # Hold days
    hold_days = [t["hold_days"] for t in trades]

    return {
        "n_trades": n,
        "trades_per_year": round(trades_per_year, 1),
        "win_rate": round(win_rate, 4),
        "avg_pnl_bps": round(avg_pnl * 10000, 2),
        "total_pnl_pct": round(total_pnl * 100, 3),
        "cagr_pct": round(cagr * 100, 3),
        "profit_factor": round(min(pf, 99.9), 3),
        "sharpe": round(sharpe, 3),
        "sortino": round(min(sortino, 99.9), 3),
        "max_dd_pct": round(max_dd * 100, 3),
        "avg_hold_days": round(np.mean(hold_days), 1),
        "median_hold_days": round(np.median(hold_days), 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PARAM OPTIMIZATION ON TRAINING WINDOW
# ─────────────────────────────────────────────────────────────────────────────
def optimize_params(prices_a, prices_b):
    """
    Grid search over lookback, z_entry, z_exit on training data.
    Returns best params by Sharpe ratio.
    """
    best_sharpe = -999
    best_params = (60, 2.0, 0.0)

    for lookback in LOOKBACK_OPTIONS:
        for z_entry in Z_ENTRY_OPTIONS:
            for z_exit in Z_EXIT_OPTIONS:
                trades = run_fold_backtest(prices_a, prices_b, lookback, z_entry, z_exit, Z_STOP)
                metrics = compute_metrics(trades)
                if metrics and metrics["sharpe"] > best_sharpe and metrics["n_trades"] >= 5:
                    best_sharpe = metrics["sharpe"]
                    best_params = (lookback, z_entry, z_exit)

    return best_params


# ─────────────────────────────────────────────────────────────────────────────
# ADVERSARIAL CHECKS (HC #705)
# ─────────────────────────────────────────────────────────────────────────────
def permutation_test(trades, n_perms=N_PERMS):
    """
    Shuffle trade direction labels (which leg is long/short).
    Real strategy Sharpe should beat >95% of shuffles.
    """
    if not trades or len(trades) < 10:
        return {"pass": False, "reason": "insufficient_trades", "percentile": 0}

    real_pnls = np.array([t["net_pnl"] for t in trades])
    real_sharpe = np.mean(real_pnls) / np.std(real_pnls) if np.std(real_pnls) > 0 else 0

    perm_sharpes = []
    for _ in range(n_perms):
        # Randomly flip direction of each trade
        signs = np.random.choice([-1, 1], size=len(real_pnls))
        # Flip gross PnL, re-apply cost
        gross_pnls = np.array([t["gross_pnl"] for t in trades])
        perm_pnl = gross_pnls * signs - TOTAL_RT_COST
        s = np.std(perm_pnl)
        perm_sharpes.append(np.mean(perm_pnl) / s if s > 0 else 0)

    percentile = float(np.mean(real_sharpe > np.array(perm_sharpes)) * 100)
    return {
        "pass": percentile >= 95,
        "percentile": round(percentile, 1),
        "real_sharpe": round(real_sharpe, 4),
    }


def regime_test(trades, spy_returns):
    """
    R1 regime test: Sharpe should be similar on green vs red days.
    Classify each trade by SPY regime during the trade.
    """
    if not trades or len(trades) < 10:
        return {"pass": False, "reason": "insufficient_trades"}

    green_pnls = []
    red_pnls = []
    flat_pnls = []

    for t in trades:
        entry = pd.Timestamp(t["entry_date"])
        exit_dt = pd.Timestamp(t["exit_date"])

        # Get SPY return during this trade's holding period
        mask = (spy_returns.index >= entry) & (spy_returns.index <= exit_dt)
        spy_ret = spy_returns[mask].sum()

        if spy_ret > 0.002:
            green_pnls.append(t["net_pnl"])
        elif spy_ret < -0.002:
            red_pnls.append(t["net_pnl"])
        else:
            flat_pnls.append(t["net_pnl"])

    def safe_sharpe(pnls):
        if len(pnls) < 3:
            return 0.0
        arr = np.array(pnls)
        return float(np.mean(arr) / np.std(arr)) if np.std(arr) > 0 else 0.0

    sharpe_green = safe_sharpe(green_pnls)
    sharpe_red = safe_sharpe(red_pnls)
    sharpe_flat = safe_sharpe(flat_pnls)

    # R1 gap check
    denom = max(abs(sharpe_green), abs(sharpe_red), 0.001)
    regime_gap = abs(sharpe_green - sharpe_red) / denom

    return {
        "pass": regime_gap <= 0.50,
        "regime_gap": round(regime_gap, 4),
        "sharpe_green": round(sharpe_green, 4),
        "sharpe_red": round(sharpe_red, 4),
        "sharpe_flat": round(sharpe_flat, 4),
        "n_green": len(green_pnls),
        "n_red": len(red_pnls),
        "n_flat": len(flat_pnls),
    }


def subperiod_test(trades, n_periods=N_SUBPERIODS):
    """Check consistency across sub-periods."""
    if not trades or len(trades) < n_periods * 3:
        return {"pass": False, "reason": "insufficient_trades"}

    chunk_size = len(trades) // n_periods
    period_results = []

    for i in range(n_periods):
        start = i * chunk_size
        end = start + chunk_size if i < n_periods - 1 else len(trades)
        chunk = trades[start:end]
        pnls = np.array([t["net_pnl"] for t in chunk])
        wr = float(np.mean(pnls > 0))
        avg = float(np.mean(pnls))
        period_results.append({
            "period": i + 1,
            "n_trades": len(chunk),
            "win_rate": round(wr, 4),
            "avg_pnl_bps": round(avg * 10000, 2),
            "positive": avg > 0,
        })

    n_positive = sum(1 for p in period_results if p["positive"])
    # Pass if majority of sub-periods are positive
    return {
        "pass": n_positive >= n_periods * 0.5,
        "n_positive": n_positive,
        "n_periods": n_periods,
        "periods": period_results,
    }


def outlier_removal_test(trades, remove_pct=OUTLIER_REMOVE_PCT):
    """Remove top N% trades by absolute PnL, check if strategy still works."""
    if not trades or len(trades) < 20:
        return {"pass": False, "reason": "insufficient_trades"}

    pnls = np.array([t["net_pnl"] for t in trades])
    abs_pnls = np.abs(pnls)
    threshold = np.percentile(abs_pnls, 100 - remove_pct)

    filtered = [t for t, ap in zip(trades, abs_pnls) if ap <= threshold]
    metrics_full = compute_metrics(trades)
    metrics_filtered = compute_metrics(filtered)

    if not metrics_full or not metrics_filtered:
        return {"pass": False, "reason": "insufficient_after_removal"}

    return {
        "pass": metrics_filtered["sharpe"] > 0 and metrics_filtered["win_rate"] > 0.45,
        "full_sharpe": metrics_full["sharpe"],
        "filtered_sharpe": metrics_filtered["sharpe"],
        "full_wr": metrics_full["win_rate"],
        "filtered_wr": metrics_filtered["win_rate"],
        "trades_removed": len(trades) - len(filtered),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SPY CORRELATION
# ─────────────────────────────────────────────────────────────────────────────
def compute_spy_correlation(trades, spy_returns):
    """Compute correlation between strategy returns and SPY returns."""
    if not trades or len(trades) < 10:
        return 0.0

    # Build daily strategy return series
    strat_daily = pd.Series(0.0, index=spy_returns.index)
    for t in trades:
        entry = pd.Timestamp(t["entry_date"])
        exit_dt = pd.Timestamp(t["exit_date"])
        hold_days_count = max((exit_dt - entry).days, 1)
        daily_pnl = t["net_pnl"] / hold_days_count

        mask = (strat_daily.index >= entry) & (strat_daily.index <= exit_dt)
        strat_daily[mask] += daily_pnl

    # Only compute on days with data
    aligned = pd.DataFrame({"strat": strat_daily, "spy": spy_returns}).dropna()
    if len(aligned) < 30:
        return 0.0

    corr = float(aligned["strat"].corr(aligned["spy"]))
    return round(corr, 4)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN WALK-FORWARD PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_pair_walkforward(pair_name, prices_a, prices_b, spy_returns):
    """
    Full walk-forward for a single pair.
    24mo train, 6mo OOT, sliding.
    On each fold:
      1. Test cointegration on training window
      2. If cointegrated: optimize params on training, test on OOT
      3. Collect OOT trades across all folds
    """
    n = len(prices_a)
    min_start = TRAIN_DAYS  # first possible test start

    folds = []
    all_oot_trades = []
    coint_results = []

    fold_idx = 0
    test_start = min_start

    while test_start + TEST_DAYS <= n:
        fold_idx += 1
        train_start = test_start - TRAIN_DAYS
        train_end = test_start
        test_end = min(test_start + TEST_DAYS, n)

        train_a = prices_a.iloc[train_start:train_end]
        train_b = prices_b.iloc[train_start:train_end]
        test_a = prices_a.iloc[test_start:test_end]
        test_b = prices_b.iloc[test_start:test_end]

        # Step 1: Cointegration gate
        is_coint, p_val = test_cointegration(train_a, train_b)
        coint_results.append({"fold": fold_idx, "cointegrated": is_coint, "p_value": p_val})

        fold_info = {
            "fold": fold_idx,
            "train_start": str(prices_a.index[train_start].date()),
            "train_end": str(prices_a.index[train_end - 1].date()),
            "test_start": str(prices_a.index[test_start].date()),
            "test_end": str(prices_a.index[test_end - 1].date()),
            "cointegrated": is_coint,
            "coint_pvalue": round(p_val, 6),
            "n_trades": 0,
            "params": None,
        }

        if is_coint:
            # Step 2: Optimize params on training window
            # For optimization, we need lookback + data
            best_lookback, best_z_entry, best_z_exit = optimize_params(train_a, train_b)

            # Step 3: Run on OOT with prepended lookback data for z-score computation
            # Need lookback days before test window for z-score rolling computation
            prepend_start = max(0, test_start - best_lookback)
            full_a = prices_a.iloc[prepend_start:test_end]
            full_b = prices_b.iloc[prepend_start:test_end]

            oot_trades = run_fold_backtest(full_a, full_b, best_lookback, best_z_entry, best_z_exit, Z_STOP)

            # Only keep trades that start within the actual test window
            test_start_date = prices_a.index[test_start]
            oot_trades = [t for t in oot_trades if pd.Timestamp(t["entry_date"]) >= test_start_date]

            fold_info["n_trades"] = len(oot_trades)
            fold_info["params"] = {
                "lookback": best_lookback,
                "z_entry": best_z_entry,
                "z_exit": best_z_exit,
            }

            all_oot_trades.extend(oot_trades)

        folds.append(fold_info)

        # Slide by TEST_DAYS
        test_start += TEST_DAYS

    # Compute cointegration stability
    n_coint = sum(1 for c in coint_results if c["cointegrated"])
    coint_stability = n_coint / len(coint_results) if coint_results else 0

    # Overall OOT metrics
    oot_metrics = compute_metrics(all_oot_trades)

    # SPY correlation
    spy_corr = compute_spy_correlation(all_oot_trades, spy_returns)

    # Adversarial checks
    adv = {}
    if all_oot_trades and len(all_oot_trades) >= 10:
        adv["permutation"] = permutation_test(all_oot_trades)
        adv["regime"] = regime_test(all_oot_trades, spy_returns)
        adv["subperiod"] = subperiod_test(all_oot_trades)
        adv["outlier_removal"] = outlier_removal_test(all_oot_trades)

    result = {
        "pair": pair_name,
        "n_folds": len(folds),
        "cointegration_stability": round(coint_stability, 4),
        "n_folds_cointegrated": n_coint,
        "total_oot_trades": len(all_oot_trades),
        "oot_metrics": oot_metrics,
        "spy_correlation": spy_corr,
        "adversarial": adv,
        "folds": folds,
    }

    # Gate checks
    gates_passed = True
    gate_reasons = []

    if not oot_metrics:
        gates_passed = False
        gate_reasons.append("insufficient_trades")
    else:
        if oot_metrics["sharpe"] < 0.3:
            gates_passed = False
            gate_reasons.append(f"sharpe={oot_metrics['sharpe']}<0.3")
        if oot_metrics["win_rate"] < 0.45:
            gates_passed = False
            gate_reasons.append(f"wr={oot_metrics['win_rate']}<0.45")
        if oot_metrics["profit_factor"] < 1.0:
            gates_passed = False
            gate_reasons.append(f"pf={oot_metrics['profit_factor']}<1.0")

    if coint_stability < 0.3:
        gates_passed = False
        gate_reasons.append(f"coint_stability={coint_stability}<0.3")

    if adv.get("permutation", {}).get("pass") is False:
        gates_passed = False
        gate_reasons.append("permutation_test_fail")
    if adv.get("regime", {}).get("pass") is False:
        gates_passed = False
        gate_reasons.append(f"regime_gap={adv.get('regime', {}).get('regime_gap', 'N/A')}>0.50")
    if adv.get("subperiod", {}).get("pass") is False:
        gates_passed = False
        gate_reasons.append("subperiod_inconsistent")
    if adv.get("outlier_removal", {}).get("pass") is False:
        gates_passed = False
        gate_reasons.append("fragile_to_outlier_removal")

    result["gates_passed"] = gates_passed
    result["gate_failures"] = gate_reasons

    return result


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO COMBINATION
# ─────────────────────────────────────────────────────────────────────────────
def combine_portfolio(pair_results):
    """Combine all passing pairs into equal-weight portfolio."""
    passing = [r for r in pair_results if r["gates_passed"]]

    if not passing:
        return None

    # Collect all trades from passing pairs, tag with pair name
    all_trades = []
    for r in passing:
        pair_name = r["pair"]
        for fold in r["folds"]:
            pass  # trades are aggregated in oot_metrics already

    # Reconstruct from fold-level (we need the actual trades)
    # Since we stored all_oot_trades in oot_metrics, we need to re-aggregate
    # For portfolio metrics, we combine the per-pair metrics
    combined_metrics = {
        "n_pairs": len(passing),
        "pairs": [r["pair"] for r in passing],
        "avg_sharpe": round(np.mean([r["oot_metrics"]["sharpe"] for r in passing]), 3),
        "avg_sortino": round(np.mean([r["oot_metrics"]["sortino"] for r in passing]), 3),
        "avg_win_rate": round(np.mean([r["oot_metrics"]["win_rate"] for r in passing]), 4),
        "avg_pf": round(np.mean([r["oot_metrics"]["profit_factor"] for r in passing]), 3),
        "avg_spy_corr": round(np.mean([r["spy_correlation"] for r in passing]), 4),
        "total_trades": sum(r["total_oot_trades"] for r in passing),
        "avg_coint_stability": round(np.mean([r["cointegration_stability"] for r in passing]), 4),
    }

    return combined_metrics


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 70)
    log.info("ETF PAIRS TRADING — WALK-FORWARD VALIDATION")
    log.info("=" * 70)
    log.info(f"Train: {TRAIN_DAYS}d (~24mo), Test: {TEST_DAYS}d (~6mo), Sliding")
    log.info(f"Cointegration gate: p < {COINT_PVALUE_THRESHOLD}")
    log.info(f"Costs: {TOTAL_RT_COST*10000:.0f} bps round-trip")
    log.info(f"Pairs: {len(ETF_PAIRS)}")

    # Download data
    prices = download_data()

    # SPY returns for regime classification
    spy_prices = prices["SPY"].dropna()
    spy_returns = spy_prices.pct_change().dropna()

    # Run each pair
    pair_results = []
    for ticker_a, ticker_b, rationale in ETF_PAIRS:
        pair_name = f"{ticker_a}/{ticker_b}"
        log.info(f"\n{'─'*50}")
        log.info(f"PAIR: {pair_name} ({rationale})")

        if ticker_a not in prices.columns or ticker_b not in prices.columns:
            log.warning(f"  Missing data for {pair_name}, skipping")
            continue

        pa = prices[ticker_a].dropna()
        pb = prices[ticker_b].dropna()

        # Align dates
        common = pa.index.intersection(pb.index)
        if len(common) < TRAIN_DAYS + TEST_DAYS:
            log.warning(f"  Insufficient data for {pair_name}: {len(common)} days")
            continue

        pa = pa.loc[common]
        pb = pb.loc[common]

        result = run_pair_walkforward(pair_name, pa, pb, spy_returns)
        pair_results.append(result)

        # Print summary
        log.info(f"  Folds: {result['n_folds']}, Cointegrated: {result['n_folds_cointegrated']}/{result['n_folds']} ({result['cointegration_stability']*100:.0f}%)")
        log.info(f"  OOT Trades: {result['total_oot_trades']}")

        if result["oot_metrics"]:
            m = result["oot_metrics"]
            log.info(f"  Sharpe: {m['sharpe']:.3f} | Sortino: {m['sortino']:.3f} | WR: {m['win_rate']:.1%} | PF: {m['profit_factor']:.2f}")
            log.info(f"  CAGR: {m['cagr_pct']:.2f}% | MaxDD: {m['max_dd_pct']:.2f}% | AvgHold: {m['avg_hold_days']:.0f}d")
            log.info(f"  SPY Corr: {result['spy_correlation']}")
        else:
            log.info(f"  No trades / insufficient data")

        if result["adversarial"]:
            adv = result["adversarial"]
            if "permutation" in adv:
                log.info(f"  Perm test: {'PASS' if adv['permutation']['pass'] else 'FAIL'} (p{adv['permutation']['percentile']:.0f})")
            if "regime" in adv:
                log.info(f"  Regime: {'PASS' if adv['regime']['pass'] else 'FAIL'} (gap={adv['regime']['regime_gap']:.3f})")
            if "subperiod" in adv:
                log.info(f"  Subperiod: {'PASS' if adv['subperiod']['pass'] else 'FAIL'} ({adv['subperiod']['n_positive']}/{adv['subperiod']['n_periods']} positive)")
            if "outlier_removal" in adv:
                log.info(f"  Outlier: {'PASS' if adv['outlier_removal']['pass'] else 'FAIL'}")

        status = "PASS ALL GATES" if result["gates_passed"] else f"FAIL: {', '.join(result['gate_failures'])}"
        log.info(f"  >>> {status}")

    # Portfolio summary
    log.info(f"\n{'='*70}")
    log.info("PORTFOLIO SUMMARY")
    log.info(f"{'='*70}")

    passing = [r for r in pair_results if r["gates_passed"]]
    failing = [r for r in pair_results if not r["gates_passed"]]

    log.info(f"\nPassing pairs ({len(passing)}/{len(pair_results)}):")
    for r in passing:
        m = r["oot_metrics"]
        log.info(f"  {r['pair']:12s} | Sharpe {m['sharpe']:+.3f} | Sortino {m['sortino']:+.3f} | WR {m['win_rate']:.1%} | PF {m['profit_factor']:.2f} | SPY-r {r['spy_correlation']:+.3f} | Coint {r['cointegration_stability']*100:.0f}%")

    log.info(f"\nFailing pairs ({len(failing)}/{len(pair_results)}):")
    for r in failing:
        log.info(f"  {r['pair']:12s} | Reasons: {', '.join(r['gate_failures'])}")

    portfolio = combine_portfolio(pair_results)
    if portfolio:
        log.info(f"\nCombined Portfolio ({portfolio['n_pairs']} pairs):")
        log.info(f"  Avg Sharpe: {portfolio['avg_sharpe']:.3f}")
        log.info(f"  Avg Sortino: {portfolio['avg_sortino']:.3f}")
        log.info(f"  Avg WR: {portfolio['avg_win_rate']:.1%}")
        log.info(f"  Avg PF: {portfolio['avg_pf']:.2f}")
        log.info(f"  Avg SPY Corr: {portfolio['avg_spy_corr']:.4f}")
        log.info(f"  Total Trades: {portfolio['total_trades']}")
        log.info(f"  Avg Coint Stability: {portfolio['avg_coint_stability']*100:.0f}%")

    # Save results
    output = {
        "run_timestamp": datetime.now().isoformat(),
        "config": {
            "train_days": TRAIN_DAYS,
            "test_days": TEST_DAYS,
            "coint_threshold": COINT_PVALUE_THRESHOLD,
            "cost_bps_rt": TOTAL_RT_COST * 10000,
            "lookback_options": LOOKBACK_OPTIONS,
            "z_entry_options": Z_ENTRY_OPTIONS,
            "z_exit_options": Z_EXIT_OPTIONS,
            "z_stop": Z_STOP,
            "n_permutations": N_PERMS,
        },
        "pair_results": pair_results,
        "portfolio": portfolio,
        "summary": {
            "total_pairs": len(pair_results),
            "passing_pairs": len(passing),
            "failing_pairs": len(failing),
            "passing_names": [r["pair"] for r in passing],
            "failing_names": [r["pair"] for r in failing],
        },
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"\nResults saved to {RESULTS_FILE}")

    return output


if __name__ == "__main__":
    main()
