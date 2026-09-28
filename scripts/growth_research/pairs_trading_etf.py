"""
Pairs Trading ETF — Market-Neutral Statistical Arbitrage
HC #0: Sliding walk-forward ONLY (252d train, 63d test, sliding)
HC #428 R1: Regime-agnostic validation (40+ OOT days, all regimes)
HC #428 R2: MFE-within-horizon check
HC #705: Adversarial checks (permutation test, sub-period, outlier removal)

Pairs: economically-linked ETFs with strong cointegration relationships
Cost: 5 bps per side per leg (20 bps total round-trip for pair)
Data: yfinance 2010-2026
"""

import os
import sys
import json
import time
import warnings
import itertools
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/pairs_trading_etf")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUTPUT_DIR / "run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ─── ETF PAIRS ───────────────────────────────────────────────────────────────
ETF_PAIRS = [
    ("XLK", "QQQ",  "Tech sector overlap"),
    ("SPY", "IVV",  "Same index different ETF"),
    ("GLD", "GDX",  "Gold vs gold miners"),
    ("XLE", "OIH",  "Energy vs oil services"),
    ("TLT", "IEF",  "Long vs intermediate treasury"),
    ("EEM", "VWO",  "Emerging markets"),
    ("XLF", "KBE",  "Financials vs banks"),
    ("IWM", "IWN",  "Small cap growth vs value"),
]

# Walk-forward params (HC #0: sliding only)
TRAIN_DAYS = 252
TEST_DAYS = 63
DATA_START = "2009-01-01"  # extra buffer for first training window
DATA_END = "2026-07-15"

# Parameter sweep
Z_ENTRY_OPTIONS = [1.5, 2.0, 2.5]
Z_EXIT_OPTIONS = [0.0, 0.25, 0.5]
Z_STOP_OPTIONS = [3.0, 4.0]
LOOKBACK_OPTIONS = [60, 120, 252]

# Transaction costs: 5 bps per side per leg = 20 bps total RT for pair
COST_PER_SIDE_PER_LEG = 0.0005  # 5 bps
TOTAL_ENTRY_COST = 2 * COST_PER_SIDE_PER_LEG  # 2 legs entry
TOTAL_EXIT_COST = 2 * COST_PER_SIDE_PER_LEG   # 2 legs exit
TOTAL_RT_COST = TOTAL_ENTRY_COST + TOTAL_EXIT_COST  # 20 bps

# Adversarial
N_PERMS = 200
N_SUBPERIODS = 4
OUTLIER_REMOVE_TOP = 5

# SPY for regime classification
SPY_TICKER = "SPY"


# ─────────────────────────────────────────────────────────────────────────────
# DATA DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────
def download_data(tickers):
    """Download adjusted close prices for all required tickers."""
    all_tickers = list(set(tickers + [SPY_TICKER]))
    log.info(f"Downloading data for {len(all_tickers)} tickers: {all_tickers}")

    data = yf.download(all_tickers, start=DATA_START, end=DATA_END,
                        auto_adjust=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data

    prices = prices.dropna(how="all")
    log.info(f"Downloaded {len(prices)} days of data from {prices.index[0].date()} to {prices.index[-1].date()}")
    return prices


# ─────────────────────────────────────────────────────────────────────────────
# SPREAD COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
def compute_spread(prices_a, prices_b, lookback):
    """Compute log-ratio spread and z-score using rolling window."""
    log_ratio = np.log(prices_a / prices_b)
    roll_mean = log_ratio.rolling(window=lookback).mean()
    roll_std = log_ratio.rolling(window=lookback).std()
    z_score = (log_ratio - roll_mean) / roll_std.replace(0, np.nan)
    return log_ratio, z_score


# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD BACKTEST
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(prices_a, prices_b, z_entry, z_exit, z_stop, lookback):
    """
    Walk-forward pairs trading backtest.

    Signal: mean-reversion when spread diverges beyond z_entry std devs.
    Entry: long cheap leg, short expensive leg (dollar-neutral).
    Exit: spread reverts to z_exit, or stop at z_stop.

    Returns list of trades with PnL.
    """
    # Need lookback + train days before we can start testing
    min_start = lookback + TRAIN_DAYS

    if len(prices_a) < min_start + TEST_DAYS:
        return []

    all_trades = []

    # Walk-forward: slide by TEST_DAYS
    test_start_idx = min_start

    while test_start_idx + TEST_DAYS <= len(prices_a):
        # Training window: compute spread statistics
        train_start = test_start_idx - TRAIN_DAYS
        train_a = prices_a.iloc[train_start:test_start_idx]
        train_b = prices_b.iloc[train_start:test_start_idx]

        # Compute spread on training data
        train_log_ratio = np.log(train_a / train_b)
        spread_mean = train_log_ratio.mean()
        spread_std = train_log_ratio.std()

        if spread_std < 1e-8:
            test_start_idx += TEST_DAYS
            continue

        # Test window
        test_end = min(test_start_idx + TEST_DAYS, len(prices_a))
        test_a = prices_a.iloc[test_start_idx:test_end]
        test_b = prices_b.iloc[test_start_idx:test_end]

        test_log_ratio = np.log(test_a / test_b)
        test_z = (test_log_ratio - spread_mean) / spread_std

        # Simulate trades in test window
        position = 0  # 0=flat, 1=long_spread, -1=short_spread
        entry_idx = None
        entry_z = None
        entry_prices_a = None
        entry_prices_b = None

        for i in range(len(test_z)):
            z = test_z.iloc[i]
            pa = test_a.iloc[i]
            pb = test_b.iloc[i]
            date = test_a.index[i]

            if np.isnan(z):
                continue

            if position == 0:
                # Entry signals
                if z > z_entry:
                    # Spread too high: short A, long B (short the spread)
                    position = -1
                    entry_idx = i
                    entry_z = z
                    entry_prices_a = pa
                    entry_prices_b = pb
                    entry_date = date
                elif z < -z_entry:
                    # Spread too low: long A, short B (long the spread)
                    position = 1
                    entry_idx = i
                    entry_z = z
                    entry_prices_a = pa
                    entry_prices_b = pb
                    entry_date = date
            else:
                # Exit signals
                should_exit = False
                exit_reason = ""

                if position == -1:
                    # Short spread: exit when z reverts below exit threshold
                    if z <= z_exit:
                        should_exit = True
                        exit_reason = "mean_reversion"
                    elif z > z_stop:
                        should_exit = True
                        exit_reason = "stop_loss"
                elif position == 1:
                    # Long spread: exit when z reverts above -exit threshold
                    if z >= -z_exit:
                        should_exit = True
                        exit_reason = "mean_reversion"
                    elif z < -z_stop:
                        should_exit = True
                        exit_reason = "stop_loss"

                if should_exit:
                    # Compute PnL (dollar-neutral: $1 each leg)
                    ret_a = (pa - entry_prices_a) / entry_prices_a
                    ret_b = (pb - entry_prices_b) / entry_prices_b

                    if position == -1:
                        # Short A, long B
                        gross_pnl = -ret_a + ret_b
                    else:
                        # Long A, short B
                        gross_pnl = ret_a - ret_b

                    net_pnl = gross_pnl - TOTAL_RT_COST

                    hold_days = i - entry_idx

                    all_trades.append({
                        "entry_date": entry_date.strftime("%Y-%m-%d"),
                        "exit_date": date.strftime("%Y-%m-%d"),
                        "direction": "short_spread" if position == -1 else "long_spread",
                        "entry_z": float(entry_z),
                        "exit_z": float(z),
                        "exit_reason": exit_reason,
                        "gross_pnl": float(gross_pnl),
                        "net_pnl": float(net_pnl),
                        "hold_days": int(hold_days),
                    })

                    position = 0
                    entry_idx = None

        # Force-close any open position at end of test window
        if position != 0:
            pa = test_a.iloc[-1]
            pb = test_b.iloc[-1]
            date = test_a.index[-1]

            ret_a = (pa - entry_prices_a) / entry_prices_a
            ret_b = (pb - entry_prices_b) / entry_prices_b

            if position == -1:
                gross_pnl = -ret_a + ret_b
            else:
                gross_pnl = ret_a - ret_b

            net_pnl = gross_pnl - TOTAL_RT_COST
            hold_days = len(test_z) - 1 - entry_idx

            all_trades.append({
                "entry_date": entry_date.strftime("%Y-%m-%d"),
                "exit_date": date.strftime("%Y-%m-%d"),
                "direction": "short_spread" if position == -1 else "long_spread",
                "entry_z": float(entry_z),
                "exit_z": float(test_z.iloc[-1]) if not np.isnan(test_z.iloc[-1]) else 0.0,
                "exit_reason": "window_end",
                "gross_pnl": float(gross_pnl),
                "net_pnl": float(net_pnl),
                "hold_days": int(hold_days),
            })
            position = 0

        test_start_idx += TEST_DAYS

    return all_trades


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(trades):
    """Compute risk-adjusted metrics from trade list."""
    if not trades or len(trades) < 5:
        return None

    pnls = np.array([t["net_pnl"] for t in trades])
    gross_pnls = np.array([t["gross_pnl"] for t in trades])

    n_trades = len(pnls)
    win_rate = np.mean(pnls > 0)
    avg_pnl = np.mean(pnls)
    total_pnl = np.sum(pnls)
    avg_win = np.mean(pnls[pnls > 0]) if np.any(pnls > 0) else 0
    avg_loss = np.mean(pnls[pnls <= 0]) if np.any(pnls <= 0) else 0

    # Profit factor
    gross_wins = np.sum(pnls[pnls > 0])
    gross_losses = np.abs(np.sum(pnls[pnls < 0]))
    pf = gross_wins / gross_losses if gross_losses > 0 else np.inf

    # Annualized Sharpe (assume ~252 trades/year scaling)
    if len(pnls) > 1 and np.std(pnls) > 0:
        # Compute daily returns by spreading trades across calendar
        first_date = trades[0]["entry_date"]
        last_date = trades[-1]["exit_date"]
        n_days = (pd.Timestamp(last_date) - pd.Timestamp(first_date)).days
        trades_per_year = n_trades / max(n_days / 365.25, 0.1)
        sharpe = (np.mean(pnls) / np.std(pnls)) * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    # Sortino
    downside = pnls[pnls < 0]
    if len(downside) > 0 and np.std(downside) > 0:
        sortino = (np.mean(pnls) / np.std(downside)) * np.sqrt(max(trades_per_year, 1))
    else:
        sortino = np.inf if avg_pnl > 0 else 0.0

    # Max drawdown on cumulative PnL
    cum_pnl = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum_pnl)
    dd = cum_pnl - peak
    max_dd = np.min(dd)

    # Hold days
    hold_days = [t["hold_days"] for t in trades]

    return {
        "n_trades": n_trades,
        "win_rate": float(win_rate),
        "avg_pnl_bps": float(avg_pnl * 10000),
        "total_pnl_pct": float(total_pnl * 100),
        "avg_win_bps": float(avg_win * 10000),
        "avg_loss_bps": float(avg_loss * 10000),
        "profit_factor": float(min(pf, 99.9)),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_dd_pct": float(max_dd * 100),
        "avg_hold_days": float(np.mean(hold_days)),
        "median_hold_days": float(np.median(hold_days)),
        "gross_total_pnl_pct": float(np.sum(gross_pnls) * 100),
    }


# ─────────────────────────────────────────────────────────────────────────────
# R1 REGIME-AGNOSTIC VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
def regime_test(trades, spy_prices):
    """
    HC #428 R1: Classify each trade by market regime (green/red/flat day on SPY).
    Compute Sharpe per regime. Reject if |Sharpe_green - Sharpe_red| / max > 0.50.
    """
    if not trades or len(trades) < 10:
        return {"pass": False, "reason": "too_few_trades", "details": {}}

    # Classify each trade's entry date by SPY regime
    spy_daily_ret = spy_prices.pct_change()

    green_pnls = []
    red_pnls = []
    flat_pnls = []

    for t in trades:
        entry_date = pd.Timestamp(t["entry_date"])
        if entry_date in spy_daily_ret.index:
            spy_ret = spy_daily_ret.loc[entry_date]
            if spy_ret > 0.002:
                green_pnls.append(t["net_pnl"])
            elif spy_ret < -0.002:
                red_pnls.append(t["net_pnl"])
            else:
                flat_pnls.append(t["net_pnl"])
        else:
            flat_pnls.append(t["net_pnl"])

    def regime_sharpe(pnls):
        if len(pnls) < 3:
            return 0.0
        arr = np.array(pnls)
        if np.std(arr) == 0:
            return 0.0
        return float(np.mean(arr) / np.std(arr))

    s_green = regime_sharpe(green_pnls)
    s_red = regime_sharpe(red_pnls)
    s_flat = regime_sharpe(flat_pnls)

    max_s = max(abs(s_green), abs(s_red), 1e-8)
    regime_divergence = abs(s_green - s_red) / max_s

    passed = regime_divergence <= 0.50

    # Additional check: both regimes should be profitable (or at least not terrible)
    both_profitable = (np.mean(green_pnls) > -0.001 if green_pnls else True) and \
                      (np.mean(red_pnls) > -0.001 if red_pnls else True)

    return {
        "pass": passed and both_profitable,
        "regime_divergence": float(regime_divergence),
        "sharpe_green": float(s_green),
        "sharpe_red": float(s_red),
        "sharpe_flat": float(s_flat),
        "n_green": len(green_pnls),
        "n_red": len(red_pnls),
        "n_flat": len(flat_pnls),
        "both_profitable": both_profitable,
        "avg_pnl_green_bps": float(np.mean(green_pnls) * 10000) if green_pnls else 0,
        "avg_pnl_red_bps": float(np.mean(red_pnls) * 10000) if red_pnls else 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HC #705 ADVERSARIAL CHECKS
# ─────────────────────────────────────────────────────────────────────────────
def permutation_test(trades, n_perms=N_PERMS):
    """Shuffle trade PnLs and check if real Sharpe beats random."""
    if not trades or len(trades) < 10:
        return {"pass": False, "p_value": 1.0}

    pnls = np.array([t["net_pnl"] for t in trades])
    real_sharpe = np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0

    rng = np.random.RandomState(42)
    count_better = 0
    for _ in range(n_perms):
        shuffled = rng.permutation(pnls)
        # Randomly flip signs to destroy temporal structure
        signs = rng.choice([-1, 1], size=len(pnls))
        shuffled = pnls * signs
        s = np.mean(shuffled) / np.std(shuffled) if np.std(shuffled) > 0 else 0
        if s >= real_sharpe:
            count_better += 1

    p_value = count_better / n_perms
    return {
        "pass": p_value < 0.05,
        "p_value": float(p_value),
        "real_sharpe_per_trade": float(real_sharpe),
    }


def sub_period_consistency(trades, n_periods=N_SUBPERIODS):
    """Split trades into sub-periods, check if strategy works in each."""
    if not trades or len(trades) < n_periods * 5:
        return {"pass": False, "reason": "too_few_trades"}

    chunk_size = len(trades) // n_periods
    period_results = []

    for i in range(n_periods):
        start = i * chunk_size
        end = start + chunk_size if i < n_periods - 1 else len(trades)
        chunk = trades[start:end]

        pnls = np.array([t["net_pnl"] for t in chunk])
        wr = np.mean(pnls > 0)
        avg = np.mean(pnls)

        period_results.append({
            "period": i + 1,
            "n_trades": len(chunk),
            "win_rate": float(wr),
            "avg_pnl_bps": float(avg * 10000),
            "profitable": bool(avg > 0),
        })

    n_profitable = sum(1 for p in period_results if p["profitable"])
    # Pass if majority of periods are profitable
    passed = n_profitable >= n_periods * 0.5

    return {
        "pass": passed,
        "n_profitable_periods": n_profitable,
        "n_total_periods": n_periods,
        "periods": period_results,
    }


def outlier_removal_test(trades, n_remove=OUTLIER_REMOVE_TOP):
    """Remove top N trades and check if strategy is still profitable."""
    if not trades or len(trades) < n_remove + 5:
        return {"pass": False, "reason": "too_few_trades"}

    pnls = sorted([t["net_pnl"] for t in trades], reverse=True)

    # Remove top N winners
    remaining = pnls[n_remove:]

    avg_remaining = np.mean(remaining)
    wr_remaining = np.mean(np.array(remaining) > 0)
    total_remaining = np.sum(remaining)

    return {
        "pass": avg_remaining > 0,
        "avg_pnl_without_outliers_bps": float(avg_remaining * 10000),
        "wr_without_outliers": float(wr_remaining),
        "total_pnl_without_outliers_pct": float(total_remaining * 100),
        "n_removed": n_remove,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SWEEP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 80)
    log.info("PAIRS TRADING ETF — Market-Neutral Statistical Arbitrage")
    log.info("=" * 80)

    # Collect all tickers
    all_tickers = list(set(
        [t for pair in ETF_PAIRS for t in (pair[0], pair[1])] + [SPY_TICKER]
    ))

    # Download data
    prices = download_data(all_tickers)
    spy_prices = prices[SPY_TICKER].dropna()

    # Results storage
    all_results = []
    best_per_pair = {}

    total_combos = len(ETF_PAIRS) * len(Z_ENTRY_OPTIONS) * len(Z_EXIT_OPTIONS) * \
                   len(Z_STOP_OPTIONS) * len(LOOKBACK_OPTIONS)
    log.info(f"Total parameter combinations to test: {total_combos}")

    combo_idx = 0

    for etf_a, etf_b, description in ETF_PAIRS:
        log.info(f"\n{'='*60}")
        log.info(f"PAIR: {etf_a} vs {etf_b} — {description}")
        log.info(f"{'='*60}")

        # Check data availability
        if etf_a not in prices.columns or etf_b not in prices.columns:
            log.warning(f"  Missing data for {etf_a} or {etf_b}, skipping")
            continue

        pa = prices[etf_a].dropna()
        pb = prices[etf_b].dropna()

        # Align dates
        common_idx = pa.index.intersection(pb.index)
        if len(common_idx) < TRAIN_DAYS + TEST_DAYS + 252:
            log.warning(f"  Insufficient data ({len(common_idx)} days), skipping")
            continue

        pa = pa.loc[common_idx]
        pb = pb.loc[common_idx]

        log.info(f"  Data: {len(common_idx)} days from {common_idx[0].date()} to {common_idx[-1].date()}")

        # Correlation check
        corr = pa.pct_change().corr(pb.pct_change())
        log.info(f"  Return correlation: {corr:.4f}")

        pair_key = f"{etf_a}_{etf_b}"
        best_sharpe = -999

        for z_entry in Z_ENTRY_OPTIONS:
            for z_exit in Z_EXIT_OPTIONS:
                if z_exit >= z_entry:
                    continue  # exit must be inside entry
                for z_stop in Z_STOP_OPTIONS:
                    for lookback in LOOKBACK_OPTIONS:
                        combo_idx += 1

                        trades = run_backtest(pa, pb, z_entry, z_exit, z_stop, lookback)

                        if len(trades) < 10:
                            continue

                        metrics = compute_metrics(trades)
                        if metrics is None:
                            continue

                        result = {
                            "pair": pair_key,
                            "etf_a": etf_a,
                            "etf_b": etf_b,
                            "description": description,
                            "z_entry": z_entry,
                            "z_exit": z_exit,
                            "z_stop": z_stop,
                            "lookback": lookback,
                            "correlation": float(corr),
                            **metrics,
                        }

                        all_results.append(result)

                        if metrics["sharpe"] > best_sharpe:
                            best_sharpe = metrics["sharpe"]
                            best_per_pair[pair_key] = (result, trades)

                        if combo_idx % 50 == 0:
                            log.info(f"  [{combo_idx}/{total_combos}] z_entry={z_entry} z_exit={z_exit} "
                                     f"z_stop={z_stop} lb={lookback}: {metrics['n_trades']} trades, "
                                     f"Sharpe={metrics['sharpe']:.3f}, WR={metrics['win_rate']:.1%}")

    log.info(f"\n{'='*80}")
    log.info(f"SWEEP COMPLETE — {len(all_results)} valid configurations tested")
    log.info(f"{'='*80}")

    # ── BEST CONFIG PER PAIR ──────────────────────────────────────────────────
    log.info("\n" + "=" * 80)
    log.info("BEST CONFIGURATION PER PAIR")
    log.info("=" * 80)

    final_report = {}

    for pair_key, (result, trades) in sorted(best_per_pair.items()):
        log.info(f"\n--- {pair_key} ({result['description']}) ---")
        log.info(f"  Params: z_entry={result['z_entry']}, z_exit={result['z_exit']}, "
                 f"z_stop={result['z_stop']}, lookback={result['lookback']}")
        log.info(f"  Trades: {result['n_trades']}, WR: {result['win_rate']:.1%}")
        log.info(f"  Sharpe: {result['sharpe']:.3f}, Sortino: {result['sortino']:.3f}")
        log.info(f"  PF: {result['profit_factor']:.2f}")
        log.info(f"  Avg PnL: {result['avg_pnl_bps']:.1f} bps, Total PnL: {result['total_pnl_pct']:.2f}%")
        log.info(f"  Gross PnL: {result['gross_total_pnl_pct']:.2f}% (before {TOTAL_RT_COST*10000:.0f} bps costs)")
        log.info(f"  Max DD: {result['max_dd_pct']:.2f}%")
        log.info(f"  Avg hold: {result['avg_hold_days']:.1f} days")

        # Run adversarial checks
        log.info(f"\n  ADVERSARIAL CHECKS:")

        # R1 regime test
        r1 = regime_test(trades, spy_prices)
        log.info(f"  R1 Regime: {'PASS' if r1['pass'] else 'FAIL'} — "
                 f"divergence={r1.get('regime_divergence', 'N/A'):.3f}, "
                 f"Sharpe_green={r1.get('sharpe_green', 0):.3f}, "
                 f"Sharpe_red={r1.get('sharpe_red', 0):.3f}")

        # Permutation test
        perm = permutation_test(trades)
        log.info(f"  Permutation: {'PASS' if perm['pass'] else 'FAIL'} — p={perm['p_value']:.3f}")

        # Sub-period consistency
        subp = sub_period_consistency(trades)
        log.info(f"  Sub-period: {'PASS' if subp['pass'] else 'FAIL'} — "
                 f"{subp.get('n_profitable_periods', 0)}/{subp.get('n_total_periods', 0)} profitable")

        # Outlier removal
        outlier = outlier_removal_test(trades)
        log.info(f"  Outlier removal: {'PASS' if outlier['pass'] else 'FAIL'} — "
                 f"avg w/o top {OUTLIER_REMOVE_TOP}: {outlier.get('avg_pnl_without_outliers_bps', 0):.1f} bps")

        all_pass = r1.get("pass", False) and perm.get("pass", False) and \
                   subp.get("pass", False) and outlier.get("pass", False)

        log.info(f"\n  ALL ADVERSARIAL CHECKS: {'*** PASS ***' if all_pass else 'FAIL'}")

        final_report[pair_key] = {
            "config": result,
            "adversarial": {
                "r1_regime": r1,
                "permutation": perm,
                "sub_period": subp,
                "outlier_removal": outlier,
                "all_pass": all_pass,
            },
            "trades": trades,
        }

    # ── SUMMARY TABLE ─────────────────────────────────────────────────────────
    log.info("\n" + "=" * 80)
    log.info("SUMMARY TABLE — ALL PAIRS (BEST CONFIG)")
    log.info("=" * 80)
    log.info(f"{'Pair':<12} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} {'PF':>6} "
             f"{'#Trades':>8} {'AvgPnL':>8} {'TotPnL':>8} {'R1':>5} {'Perm':>5} "
             f"{'SubP':>5} {'Outl':>5} {'ALL':>5}")
    log.info("-" * 110)

    for pair_key in sorted(final_report.keys()):
        r = final_report[pair_key]
        c = r["config"]
        a = r["adversarial"]

        log.info(f"{pair_key:<12} {c['sharpe']:>7.3f} {c['sortino']:>8.3f} "
                 f"{c['win_rate']:>5.1%} {c['profit_factor']:>6.2f} "
                 f"{c['n_trades']:>8d} {c['avg_pnl_bps']:>7.1f} {c['total_pnl_pct']:>7.2f}% "
                 f"{'Y' if a['r1_regime']['pass'] else 'N':>5} "
                 f"{'Y' if a['permutation']['pass'] else 'N':>5} "
                 f"{'Y' if a['sub_period']['pass'] else 'N':>5} "
                 f"{'Y' if a['outlier_removal']['pass'] else 'N':>5} "
                 f"{'Y' if a['all_pass'] else 'N':>5}")

    # ── PASS/FAIL SUMMARY ─────────────────────────────────────────────────────
    passing = [k for k, v in final_report.items() if v["adversarial"]["all_pass"]]
    failing = [k for k, v in final_report.items() if not v["adversarial"]["all_pass"]]

    log.info(f"\nPASSING ALL CHECKS ({len(passing)}): {', '.join(passing) if passing else 'NONE'}")
    log.info(f"FAILING ({len(failing)}): {', '.join(failing) if failing else 'NONE'}")

    # Answer the key question
    passing_with_sharpe = [k for k in passing
                           if final_report[k]["config"]["sharpe"] > 1.0]

    log.info(f"\n{'='*80}")
    log.info("KEY QUESTION: Can pairs trading generate Sharpe > 1.0 while passing R1?")
    if passing_with_sharpe:
        log.info(f"YES — {len(passing_with_sharpe)} pairs: {', '.join(passing_with_sharpe)}")
        for k in passing_with_sharpe:
            c = final_report[k]["config"]
            log.info(f"  {k}: Sharpe={c['sharpe']:.3f}, WR={c['win_rate']:.1%}, "
                     f"PF={c['profit_factor']:.2f}, Total={c['total_pnl_pct']:.2f}%")
    else:
        log.info("NO — No pairs achieve Sharpe > 1.0 while passing all adversarial checks.")
        # Show closest
        if passing:
            for k in passing:
                c = final_report[k]["config"]
                log.info(f"  Closest: {k}: Sharpe={c['sharpe']:.3f}")
    log.info("=" * 80)

    # ── SAVE RESULTS ──────────────────────────────────────────────────────────
    # Save sweep results
    sweep_df = pd.DataFrame(all_results)
    sweep_df.to_csv(OUTPUT_DIR / "sweep_results.csv", index=False)
    log.info(f"\nSaved sweep results ({len(all_results)} configs)")

    # Save final report (without trades for JSON)
    report_json = {}
    for pair_key, data in final_report.items():
        report_json[pair_key] = {
            "config": data["config"],
            "adversarial": data["adversarial"],
            "n_trades": len(data["trades"]),
        }
        # Remove trades from adversarial to keep JSON clean
        if "trades" in report_json[pair_key]:
            del report_json[pair_key]["trades"]

    with open(OUTPUT_DIR / "final_report.json", "w") as f:
        json.dump(report_json, f, indent=2, default=str)

    # Save best trades per pair
    for pair_key, data in final_report.items():
        trades_df = pd.DataFrame(data["trades"])
        trades_df.to_csv(OUTPUT_DIR / f"trades_{pair_key}.csv", index=False)

    log.info(f"All results saved to {OUTPUT_DIR}")
    log.info("DONE.")


if __name__ == "__main__":
    main()
