#!/usr/bin/env python3
"""
ETF Short-Term Reversal Strategy v2 — Survivorship-Bias-Free
=============================================================
Tests whether short-term mean-reversion edge survives when using ETFs
(no survivorship bias) instead of hand-picked large-cap stocks.

Key differences from v1 (individual stocks):
  - Universe: ~20 ETFs (sector SPDRs, broad, fixed income, commodities)
  - All ETFs existed since inception dates well before our test period
  - Walk-forward validation: 36-month train, 6-month OOT, sliding
  - Tests reversal (buy losers) vs momentum (buy winners) vs random baseline
  - HC #705 adversarial checks: permutation, sub-period, outlier, regime

Walk-forward approach:
  - Grid of (N_lookback, K_etfs, M_hold) optimized on train window
  - Best config applied OOT
  - At least 20 folds
"""

import json
import os
import sys
import warnings
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

# ── Universe (ETFs only — no survivorship bias) ─────────────────────────────
ETF_UNIVERSE = [
    # Sector SPDRs (all since 1998-12-16)
    "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",
    # Broad market
    "SPY", "QQQ", "IWM", "MDY", "EFA", "EEM",
    # Fixed income
    "TLT", "IEF", "HYG", "LQD",
    # Commodities
    "GLD", "SLV",
]

START = "2003-01-01"  # GLD inception 2004, but yfinance handles gracefully
END = "2026-07-10"
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_PERM = 200
np.random.seed(42)

# Walk-forward params
WF_TRAIN_MONTHS = 36
WF_OOT_MONTHS = 6

# Grid search space
LOOKBACK_GRID = [5, 10, 20]
K_GRID = [3, 5, 7]
HOLD_GRID = [5, 10, 20]


# ── Data Download ────────────────────────────────────────────────────────────
def download_data():
    """Download adjusted close prices for ETF universe + ^VIX."""
    tickers = ETF_UNIVERSE + ["^VIX"]
    print(f"Downloading {len(tickers)} tickers from {START} to {END}...")

    all_close = {}
    batch_size = 5
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        print(f"  Batch {i // batch_size + 1}: {batch}")
        try:
            data = yf.download(batch, start=START, end=END, auto_adjust=True,
                               progress=False, threads=False, timeout=30)
            if isinstance(data.columns, pd.MultiIndex):
                close = data["Close"]
            else:
                close = data
            if isinstance(close, pd.Series):
                all_close[batch[0]] = close
            else:
                for col in close.columns:
                    all_close[col] = close[col]
        except Exception as e:
            print(f"    WARNING: Failed to download {batch}: {e}")

    close_df = pd.DataFrame(all_close)
    close_df = close_df.ffill(limit=5)

    # Report data availability
    for etf in ETF_UNIVERSE:
        if etf in close_df.columns:
            first = close_df[etf].first_valid_index()
            last = close_df[etf].last_valid_index()
            pct_na = close_df[etf].isna().mean() * 100
            print(f"    {etf}: {first.date()} to {last.date()} ({pct_na:.1f}% missing)")
        else:
            print(f"    {etf}: NOT AVAILABLE")

    print(f"  Total: {len(close_df)} trading days, {close_df.shape[1]} tickers")
    return close_df


# ── Core Backtest Engine ────────────────────────────────────────────────────
def run_reversal_backtest(close_df, etf_cols, lookback, k_etfs, hold,
                          mode="reversal"):
    """
    Run reversal/momentum/random backtest on ETFs.

    mode: 'reversal' (buy losers), 'momentum' (buy winners), 'random'
    Returns: (dates_array, returns_array) or (None, None) if insufficient data.
    """
    spy = close_df["SPY"] if "SPY" in close_df.columns else None
    trailing_ret = close_df[etf_cols].pct_change(lookback)

    dates = close_df.index
    # Weekly rebalance: last trading day of each week
    week_groups = dates.to_series().dt.isocalendar()
    week_groups = week_groups[["year", "week"]].astype(str).agg("-".join, axis=1)
    rebal_dates = dates.to_series().groupby(week_groups).last()
    rebal_dates = rebal_dates.sort_values().values
    rebal_dates = [d for d in rebal_dates if d >= dates[lookback + 5]]

    period_returns = []
    period_dates = []

    for rebal_date in rebal_dates:
        rebal_idx = dates.get_loc(rebal_date)
        if rebal_idx + hold >= len(dates):
            break

        tr = trailing_ret.loc[rebal_date, etf_cols].dropna()
        if len(tr) < k_etfs:
            continue

        # Select ETFs based on mode
        if mode == "reversal":
            ranked = tr.sort_values()
            selected = ranked.head(k_etfs).index
        elif mode == "momentum":
            ranked = tr.sort_values(ascending=False)
            selected = ranked.head(k_etfs).index
        elif mode == "random":
            selected = tr.sample(min(k_etfs, len(tr))).index
        else:
            raise ValueError(f"Unknown mode: {mode}")

        entry_price = close_df.loc[rebal_date, selected]
        exit_date = dates[rebal_idx + hold]
        exit_price = close_df.loc[exit_date, selected]

        fwd_rets = (exit_price / entry_price - 1.0).dropna()
        if len(fwd_rets) == 0:
            continue

        basket_ret = fwd_rets.mean()
        period_returns.append(basket_ret)
        period_dates.append(rebal_date)

    if len(period_returns) < 10:
        return None, None

    return np.array(period_dates), np.array(period_returns)


def compute_metrics(returns, hold):
    """Compute performance metrics from period returns."""
    if returns is None or len(returns) < 10:
        return None

    periods_per_year = 252 / hold
    mean_ret = returns.mean()
    std_ret = returns.std()
    win_rate = (returns > 0).mean()

    ann_mean = mean_ret * periods_per_year
    ann_std = std_ret * np.sqrt(periods_per_year)
    sharpe = ann_mean / ann_std if ann_std > 0 else 0.0

    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(periods_per_year) if len(downside) > 1 else ann_std
    sortino = ann_mean / downside_std if downside_std > 0 else 0.0

    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    profit_factor = gains / losses if losses > 0 else 999.0

    cum = (1 + returns).cumprod()
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = dd.min()

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "mean_return_pct": mean_ret * 100,
        "max_dd_pct": max_dd * 100,
        "n_periods": len(returns),
        "ann_return_pct": ann_mean * 100,
    }


# ── Walk-Forward Validation ─────────────────────────────────────────────────
def walk_forward_validation(close_df, etf_cols):
    """
    36-month train, 6-month OOT, sliding window.
    Optimize (N, K, M) on train, apply best to OOT.
    """
    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION (36m train, 6m OOT, sliding)")
    print("=" * 80)

    dates = close_df.index
    start_date = dates[0]
    end_date = dates[-1]

    # Generate fold boundaries
    folds = []
    cursor = start_date + pd.DateOffset(months=WF_TRAIN_MONTHS)
    while cursor + pd.DateOffset(months=WF_OOT_MONTHS) <= end_date:
        train_start = cursor - pd.DateOffset(months=WF_TRAIN_MONTHS)
        train_end = cursor
        oot_start = cursor
        oot_end = cursor + pd.DateOffset(months=WF_OOT_MONTHS)
        folds.append({
            "train_start": train_start,
            "train_end": train_end,
            "oot_start": oot_start,
            "oot_end": oot_end,
        })
        cursor += pd.DateOffset(months=WF_OOT_MONTHS)

    print(f"  Generated {len(folds)} folds")
    if len(folds) < 20:
        print(f"  WARNING: Only {len(folds)} folds (target >=20)")

    oot_returns_all = []
    oot_dates_all = []
    fold_results = []

    for i, fold in enumerate(folds):
        # Extract train/OOT windows
        train_mask = (dates >= fold["train_start"]) & (dates < fold["train_end"])
        oot_mask = (dates >= fold["oot_start"]) & (dates < fold["oot_end"])

        train_df = close_df.loc[train_mask].copy()
        oot_df = close_df.loc[oot_mask].copy()

        if len(train_df) < 100 or len(oot_df) < 20:
            continue

        # Available ETFs in this period (need data in both train and OOT)
        avail = [c for c in etf_cols if c in train_df.columns
                 and train_df[c].notna().sum() > 60
                 and c in oot_df.columns
                 and oot_df[c].notna().sum() > 10]

        if len(avail) < 5:
            continue

        # Grid search on train data
        best_sharpe = -999
        best_params = None

        for lookback, k, hold in product(LOOKBACK_GRID, K_GRID, HOLD_GRID):
            if k > len(avail):
                continue
            _, rets = run_reversal_backtest(train_df, avail, lookback, k, hold,
                                            mode="reversal")
            if rets is None:
                continue
            metrics = compute_metrics(rets, hold)
            if metrics is None:
                continue
            if metrics["sharpe"] > best_sharpe:
                best_sharpe = metrics["sharpe"]
                best_params = (lookback, k, hold)

        if best_params is None:
            continue

        lookback, k, hold = best_params

        # Apply best params to OOT
        # Need enough context before OOT starts
        context_start = fold["oot_start"] - pd.DateOffset(days=lookback * 2 + 10)
        oot_extended_mask = (dates >= context_start) & (dates < fold["oot_end"])
        oot_ext_df = close_df.loc[oot_extended_mask].copy()

        oot_dates_raw, oot_rets = run_reversal_backtest(
            oot_ext_df, avail, lookback, k, hold, mode="reversal"
        )

        if oot_rets is None or len(oot_rets) == 0:
            continue

        # Filter to only OOT period dates
        oot_start_ts = pd.Timestamp(fold["oot_start"])
        in_oot = np.array([pd.Timestamp(d) >= oot_start_ts for d in oot_dates_raw])
        if in_oot.sum() == 0:
            continue

        oot_rets_filtered = oot_rets[in_oot]
        oot_dates_filtered = oot_dates_raw[in_oot]

        oot_metrics = compute_metrics(oot_rets_filtered, hold)

        fold_results.append({
            "fold": i + 1,
            "train_period": f"{fold['train_start'].date()} to {fold['train_end'].date()}",
            "oot_period": f"{fold['oot_start'].date()} to {fold['oot_end'].date()}",
            "best_params": {"lookback": lookback, "k": k, "hold": hold},
            "train_sharpe": round(best_sharpe, 3),
            "oot_sharpe": round(oot_metrics["sharpe"], 3) if oot_metrics else 0.0,
            "oot_wr": round(oot_metrics["win_rate"], 4) if oot_metrics else 0.0,
            "oot_mean_ret_pct": round(oot_metrics["mean_return_pct"], 4) if oot_metrics else 0.0,
            "n_oot_trades": len(oot_rets_filtered),
        })

        oot_returns_all.extend(oot_rets_filtered.tolist())
        oot_dates_all.extend(oot_dates_filtered.tolist())

        print(f"  Fold {i+1:2d}: Train Sharpe={best_sharpe:.3f}, "
              f"OOT Sharpe={oot_metrics['sharpe']:.3f}, "
              f"Params=({lookback},{k},{hold}), "
              f"OOT trades={len(oot_rets_filtered)}")

    if len(oot_returns_all) == 0:
        print("  NO valid OOT returns collected!")
        return None, None, fold_results

    oot_returns_all = np.array(oot_returns_all)
    oot_dates_all = np.array(oot_dates_all)

    return oot_dates_all, oot_returns_all, fold_results


# ── Fixed-Config Full-Period Backtest ────────────────────────────────────────
def run_fixed_configs(close_df, etf_cols):
    """Run a set of fixed configurations over the full period for comparison."""
    print("\n" + "=" * 80)
    print("FIXED-CONFIG FULL-PERIOD BACKTESTS")
    print("=" * 80)

    configs = [
        (5, 3, 5),   (5, 5, 5),   (5, 3, 10),  (5, 5, 10),
        (10, 3, 5),  (10, 5, 5),  (10, 3, 10),  (10, 5, 10),
        (20, 3, 5),  (20, 5, 5),  (20, 3, 10),  (20, 5, 10),
        (5, 3, 20),  (10, 3, 20), (20, 3, 20),
        (5, 7, 5),   (10, 7, 5),  (20, 7, 10),
    ]

    results = []
    for lookback, k, hold in configs:
        if k > len(etf_cols):
            continue

        # Reversal
        dates_r, rets_r = run_reversal_backtest(close_df, etf_cols, lookback, k, hold, "reversal")
        metrics_r = compute_metrics(rets_r, hold) if rets_r is not None else None

        # Momentum (for comparison)
        dates_m, rets_m = run_reversal_backtest(close_df, etf_cols, lookback, k, hold, "momentum")
        metrics_m = compute_metrics(rets_m, hold) if rets_m is not None else None

        if metrics_r:
            name = f"REV L={lookback} K={k} H={hold}"
            results.append({
                "name": name,
                "mode": "reversal",
                "lookback": lookback,
                "k": k,
                "hold": hold,
                "metrics": metrics_r,
                "returns": rets_r,
                "dates": dates_r,
            })
            print(f"  {name}: Sharpe={metrics_r['sharpe']:.3f}, "
                  f"WR={metrics_r['win_rate']:.1%}, PF={metrics_r['profit_factor']:.2f}")

        if metrics_m:
            name_m = f"MOM L={lookback} K={k} H={hold}"
            results.append({
                "name": name_m,
                "mode": "momentum",
                "lookback": lookback,
                "k": k,
                "hold": hold,
                "metrics": metrics_m,
                "returns": rets_m,
                "dates": dates_m,
            })

    return results


# ── HC #705 Adversarial Checks ──────────────────────────────────────────────
def adversarial_checks(close_df, etf_cols, dates_arr, returns, lookback, k, hold):
    """
    Run all adversarial checks on a given backtest result.
    Returns dict of check results.
    """
    checks = {}

    # 1. Permutation test (200 shuffles — vectorized: pre-compute forward returns, randomize selection)
    print("  Running permutation test (200 shuffles, vectorized)...")
    actual_mean = returns.mean()

    # Pre-compute all forward returns at each rebalance date
    trailing_ret = close_df[etf_cols].pct_change(lookback)
    dates = close_df.index
    week_groups = dates.to_series().dt.isocalendar()
    week_groups = week_groups[["year", "week"]].astype(str).agg("-".join, axis=1)
    rebal_dates = dates.to_series().groupby(week_groups).last().sort_values().values
    rebal_dates = [d for d in rebal_dates if d >= dates[lookback + 5]]

    # Collect forward return vectors for all valid periods
    fwd_ret_vectors = []
    for rebal_date in rebal_dates:
        rebal_idx = dates.get_loc(rebal_date)
        if rebal_idx + hold >= len(dates):
            break
        tr = trailing_ret.loc[rebal_date, etf_cols].dropna()
        if len(tr) < k:
            continue
        entry = close_df.loc[rebal_date, tr.index]
        exit_date = dates[rebal_idx + hold]
        exit_p = close_df.loc[exit_date, tr.index]
        fwd = (exit_p / entry - 1.0).dropna()
        if len(fwd) >= k:
            fwd_ret_vectors.append(fwd.values)

    perm_means = np.zeros(N_PERM)
    for p in range(N_PERM):
        period_rets = []
        for fwd_vec in fwd_ret_vectors:
            pick = np.random.choice(len(fwd_vec), size=min(k, len(fwd_vec)), replace=False)
            period_rets.append(fwd_vec[pick].mean())
        perm_means[p] = np.mean(period_rets) if period_rets else 0.0

    p_value = (perm_means >= actual_mean).mean()
    checks["permutation"] = {
        "actual_mean_pct": round(actual_mean * 100, 4),
        "perm_mean_pct": round(perm_means.mean() * 100, 4),
        "perm_std_pct": round(perm_means.std() * 100, 4),
        "p_value": round(float(p_value), 4),
        "pass": p_value < 0.05,
    }
    print(f"    Permutation: p={p_value:.4f} ({'PASS' if p_value < 0.05 else 'FAIL'})")

    # 2. Sub-period consistency
    mid = len(returns) // 2
    h1_mean = returns[:mid].mean()
    h2_mean = returns[mid:].mean()
    sub_pass = h1_mean > 0 and h2_mean > 0
    checks["subperiod"] = {
        "first_half_mean_pct": round(h1_mean * 100, 4),
        "second_half_mean_pct": round(h2_mean * 100, 4),
        "pass": sub_pass,
    }
    print(f"    Sub-period: H1={h1_mean*100:.4f}%, H2={h2_mean*100:.4f}% "
          f"({'PASS' if sub_pass else 'FAIL'})")

    # 3. Outlier removal (top 5% days removed)
    p5, p95 = np.percentile(returns, [5, 95])
    trimmed = returns[(returns >= p5) & (returns <= p95)]
    trimmed_mean = trimmed.mean() if len(trimmed) > 0 else 0.0
    outlier_pass = trimmed_mean > 0
    checks["outlier_removal"] = {
        "trimmed_mean_pct": round(trimmed_mean * 100, 4),
        "n_removed": len(returns) - len(trimmed),
        "pass": outlier_pass,
    }
    print(f"    Outlier removal: trimmed mean={trimmed_mean*100:.4f}% "
          f"({'PASS' if outlier_pass else 'FAIL'})")

    # 4. R1 regime test (SPY green/red/flat classification)
    spy = close_df["SPY"] if "SPY" in close_df.columns else None
    if spy is not None:
        spy_weekly_ret = spy.pct_change(5)
        green_rets, red_rets, flat_rets = [], [], []
        periods_per_year = 252 / hold

        for d, r in zip(dates_arr, returns):
            try:
                spy_r = spy_weekly_ret.loc[d]
                if pd.isna(spy_r):
                    continue
                if spy_r > 0.005:
                    green_rets.append(r)
                elif spy_r < -0.005:
                    red_rets.append(r)
                else:
                    flat_rets.append(r)
            except (KeyError, TypeError):
                continue

        green_rets = np.array(green_rets) if green_rets else np.array([0.0])
        red_rets = np.array(red_rets) if red_rets else np.array([0.0])
        flat_rets = np.array(flat_rets) if flat_rets else np.array([0.0])

        def _regime_sharpe(rets):
            if len(rets) < 5:
                return 0.0
            m = rets.mean() * periods_per_year
            s = rets.std() * np.sqrt(periods_per_year)
            return m / s if s > 0 else 0.0

        s_green = _regime_sharpe(green_rets)
        s_red = _regime_sharpe(red_rets)
        s_flat = _regime_sharpe(flat_rets)
        max_s = max(abs(s_green), abs(s_red))
        regime_gap = abs(s_green - s_red) / max_s if max_s > 0 else 0.0
        regime_pass = regime_gap <= 0.50

        checks["regime"] = {
            "sharpe_green": round(s_green, 3),
            "sharpe_red": round(s_red, 3),
            "sharpe_flat": round(s_flat, 3),
            "n_green": len(green_rets),
            "n_red": len(red_rets),
            "n_flat": len(flat_rets),
            "regime_gap": round(regime_gap, 3),
            "pass": regime_pass,
            "better_in_bear": s_red > s_green,
        }
        print(f"    Regime: Green Sharpe={s_green:.3f}, Red Sharpe={s_red:.3f}, "
              f"Gap={regime_gap:.3f} ({'PASS' if regime_pass else 'FAIL'})"
              f"{' [BETTER IN BEAR]' if s_red > s_green else ''}")
    else:
        checks["regime"] = {"pass": False, "error": "No SPY data"}

    # 5. Data integrity checks
    n_etfs_used = len(etf_cols)
    date_range_years = (pd.Timestamp(dates_arr[-1]) - pd.Timestamp(dates_arr[0])).days / 365.25
    integrity_pass = n_etfs_used >= 10 and date_range_years >= 5 and len(returns) >= 50
    checks["data_integrity"] = {
        "n_etfs": n_etfs_used,
        "date_range_years": round(date_range_years, 1),
        "n_periods": len(returns),
        "pass": integrity_pass,
    }

    # Overall
    all_pass = all(c.get("pass", False) for c in checks.values())
    checks["all_pass"] = all_pass

    return checks


# ── Correlation with SPY ─────────────────────────────────────────────────────
def compute_spy_correlation(close_df, dates_arr, returns, hold):
    """Compute correlation of strategy returns with SPY returns."""
    spy = close_df["SPY"] if "SPY" in close_df.columns else None
    if spy is None:
        return None

    spy_rets = []
    for d in dates_arr:
        try:
            idx = close_df.index.get_loc(d)
            if idx + hold < len(close_df):
                entry = spy.iloc[idx]
                exit_p = spy.iloc[idx + hold]
                spy_rets.append(exit_p / entry - 1.0)
            else:
                spy_rets.append(np.nan)
        except (KeyError, IndexError):
            spy_rets.append(np.nan)

    spy_rets = np.array(spy_rets)
    valid = ~np.isnan(spy_rets) & ~np.isnan(returns)
    if valid.sum() < 10:
        return None

    corr = np.corrcoef(returns[valid], spy_rets[valid])[0, 1]
    return round(float(corr), 4)


# ── VIX Correlation Check ───────────────────────────────────────────────────
def check_vix_correlation(close_df, dates_arr, returns):
    """Check if this is actually a VIX mean-reversion strategy in disguise."""
    vix = close_df.get("^VIX")
    if vix is None:
        return None

    vix_levels = []
    vix_changes = vix.pct_change(5)
    vix_chg_vals = []

    for d in dates_arr:
        try:
            vix_levels.append(vix.loc[d])
            vix_chg_vals.append(vix_changes.loc[d])
        except (KeyError, TypeError):
            vix_levels.append(np.nan)
            vix_chg_vals.append(np.nan)

    vix_levels = np.array(vix_levels)
    vix_chg_vals = np.array(vix_chg_vals)

    valid_level = ~np.isnan(vix_levels) & ~np.isnan(returns)
    valid_chg = ~np.isnan(vix_chg_vals) & ~np.isnan(returns)

    corr_level = np.corrcoef(returns[valid_level], vix_levels[valid_level])[0, 1] if valid_level.sum() > 10 else None
    corr_change = np.corrcoef(returns[valid_chg], vix_chg_vals[valid_chg])[0, 1] if valid_chg.sum() > 10 else None

    return {
        "corr_with_vix_level": round(float(corr_level), 4) if corr_level is not None else None,
        "corr_with_vix_5d_change": round(float(corr_change), 4) if corr_change is not None else None,
        "is_vix_proxy": abs(corr_level or 0) > 0.5 or abs(corr_change or 0) > 0.5,
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("ETF SHORT-TERM REVERSAL STRATEGY v2 — SURVIVORSHIP-BIAS-FREE")
    print("=" * 80)
    print(f"Universe: {len(ETF_UNIVERSE)} ETFs (sectors, broad, FI, commodities)")
    print(f"Period: {START} to {END}")
    print(f"Walk-forward: {WF_TRAIN_MONTHS}m train, {WF_OOT_MONTHS}m OOT, sliding")
    print(f"Permutation iterations: {N_PERM}")
    print()

    # Download data
    close_df = download_data()

    # Determine usable ETFs (need at least 60% data)
    etf_cols = [c for c in ETF_UNIVERSE if c in close_df.columns
                and close_df[c].notna().mean() > 0.6]
    print(f"\nUsable ETFs: {len(etf_cols)} — {etf_cols}")

    if len(etf_cols) < 5:
        print("ERROR: Too few ETFs with sufficient data.")
        return

    # ── Part 1: Fixed-config backtests ───────────────────────────────────
    fixed_results = run_fixed_configs(close_df, etf_cols)

    # ── Part 2: Walk-forward validation ──────────────────────────────────
    wf_dates, wf_returns, fold_results = walk_forward_validation(close_df, etf_cols)

    # ── Part 3: Best fixed config — full adversarial analysis ────────────
    print("\n" + "=" * 80)
    print("ADVERSARIAL ANALYSIS ON BEST FIXED CONFIGS")
    print("=" * 80)

    # Find best reversal configs by Sharpe
    reversal_results = [r for r in fixed_results if r["mode"] == "reversal"]
    reversal_results.sort(key=lambda x: x["metrics"]["sharpe"], reverse=True)

    adversarial_results = []
    for res in reversal_results[:5]:  # Top 5 configs
        print(f"\n  Analyzing: {res['name']} (Sharpe={res['metrics']['sharpe']:.3f})")
        checks = adversarial_checks(
            close_df, etf_cols, res["dates"], res["returns"],
            res["lookback"], res["k"], res["hold"]
        )

        spy_corr = compute_spy_correlation(
            close_df, res["dates"], res["returns"], res["hold"]
        )
        vix_info = check_vix_correlation(close_df, res["dates"], res["returns"])

        adversarial_results.append({
            "config": res["name"],
            "lookback": res["lookback"],
            "k": res["k"],
            "hold": res["hold"],
            "metrics": {k: round(v, 4) if isinstance(v, float) else v
                        for k, v in res["metrics"].items()},
            "checks": checks,
            "spy_correlation": spy_corr,
            "vix_analysis": vix_info,
        })

    # ── Part 4: Walk-forward adversarial (if we have data) ──────────────
    wf_adversarial = None
    wf_metrics = None
    if wf_returns is not None and len(wf_returns) > 20:
        print("\n" + "=" * 80)
        print("WALK-FORWARD OOT ADVERSARIAL ANALYSIS")
        print("=" * 80)

        # Use median hold from folds for annualization
        holds_used = [f["best_params"]["hold"] for f in fold_results if "best_params" in f]
        median_hold = int(np.median(holds_used)) if holds_used else 5

        wf_metrics = compute_metrics(wf_returns, median_hold)

        # For WF adversarial, use the most common params
        from collections import Counter
        param_counts = Counter(
            (f["best_params"]["lookback"], f["best_params"]["k"], f["best_params"]["hold"])
            for f in fold_results if "best_params" in f
        )
        most_common = param_counts.most_common(1)[0][0] if param_counts else (5, 3, 5)

        wf_adversarial = adversarial_checks(
            close_df, etf_cols, wf_dates, wf_returns,
            most_common[0], most_common[1], most_common[2]
        )

        spy_corr_wf = compute_spy_correlation(close_df, wf_dates, wf_returns, median_hold)
        vix_wf = check_vix_correlation(close_df, wf_dates, wf_returns)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY TABLE — REVERSAL vs MOMENTUM (Fixed Configs)")
    print("=" * 80)

    print(f"\n{'Config':<30} {'Mode':<10} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} "
          f"{'PF':>6} {'AnnRet%':>8} {'MaxDD%':>7} {'N':>5}")
    print("-" * 100)

    for r in sorted(fixed_results, key=lambda x: (-1 if x["mode"] == "reversal" else 1,
                                                    -x["metrics"]["sharpe"])):
        m = r["metrics"]
        print(f"{r['name']:<30} {r['mode']:<10} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{m['win_rate']:>5.1%} {m['profit_factor']:>6.2f} "
              f"{m['ann_return_pct']:>7.2f}% {m['max_dd_pct']:>6.2f}% {m['n_periods']:>5d}")

    # Reversal vs Momentum comparison
    print("\n" + "=" * 80)
    print("REVERSAL vs MOMENTUM HEAD-TO-HEAD")
    print("=" * 80)

    rev_configs = {(r["lookback"], r["k"], r["hold"]): r for r in fixed_results
                   if r["mode"] == "reversal"}
    mom_configs = {(r["lookback"], r["k"], r["hold"]): r for r in fixed_results
                   if r["mode"] == "momentum"}

    print(f"\n{'Params':<20} {'Rev Sharpe':>10} {'Mom Sharpe':>10} {'Rev Wins?':>10}")
    print("-" * 55)
    for params in sorted(rev_configs.keys()):
        if params in mom_configs:
            rev_s = rev_configs[params]["metrics"]["sharpe"]
            mom_s = mom_configs[params]["metrics"]["sharpe"]
            winner = "YES" if rev_s > mom_s else "NO"
            print(f"L={params[0]:2d} K={params[1]:1d} H={params[2]:2d}     "
                  f"{rev_s:>10.3f} {mom_s:>10.3f} {winner:>10}")

    # Walk-forward summary
    if wf_metrics:
        print("\n" + "=" * 80)
        print("WALK-FORWARD OOT RESULTS (combined across all folds)")
        print("=" * 80)
        print(f"  Sharpe:         {wf_metrics['sharpe']:.3f}")
        print(f"  Sortino:        {wf_metrics['sortino']:.3f}")
        print(f"  Win Rate:       {wf_metrics['win_rate']:.1%}")
        print(f"  Profit Factor:  {wf_metrics['profit_factor']:.2f}")
        print(f"  Ann Return:     {wf_metrics['ann_return_pct']:.2f}%")
        print(f"  Max DD:         {wf_metrics['max_dd_pct']:.2f}%")
        print(f"  N periods:      {wf_metrics['n_periods']}")
        if spy_corr_wf is not None:
            print(f"  SPY Correlation: {spy_corr_wf}")
        if vix_wf:
            print(f"  VIX Analysis:   level corr={vix_wf['corr_with_vix_level']}, "
                  f"change corr={vix_wf['corr_with_vix_5d_change']}, "
                  f"is VIX proxy={'YES' if vix_wf['is_vix_proxy'] else 'NO'}")

        if wf_adversarial:
            print(f"\n  Adversarial Checks:")
            for k, v in wf_adversarial.items():
                if isinstance(v, dict) and "pass" in v:
                    print(f"    {k}: {'PASS' if v['pass'] else 'FAIL'} — {v}")
                elif k == "all_pass":
                    print(f"    ALL PASS: {'YES' if v else 'NO'}")

    # Adversarial summary
    print("\n" + "=" * 80)
    print("ADVERSARIAL CHECK RESULTS — TOP 5 CONFIGS")
    print("=" * 80)
    for ar in adversarial_results:
        checks = ar["checks"]
        gates = []
        for k, v in checks.items():
            if isinstance(v, dict) and "pass" in v:
                gates.append(f"{k}:{'OK' if v['pass'] else 'FAIL'}")
        all_ok = checks.get("all_pass", False)
        print(f"  {ar['config']}: ALL_PASS={'YES' if all_ok else 'NO'} | "
              f"{' | '.join(gates)} | SPY_corr={ar['spy_correlation']}")
        if ar.get("vix_analysis"):
            va = ar["vix_analysis"]
            print(f"    VIX: level_corr={va['corr_with_vix_level']}, "
                  f"chg_corr={va['corr_with_vix_5d_change']}, "
                  f"proxy={'YES' if va['is_vix_proxy'] else 'NO'}")
        regime = checks.get("regime", {})
        if regime.get("better_in_bear"):
            print(f"    ** BETTER IN BEAR MARKETS (Red Sharpe={regime['sharpe_red']:.3f} "
                  f"> Green Sharpe={regime['sharpe_green']:.3f})")

    # Key questions
    print("\n" + "=" * 80)
    print("KEY QUESTIONS ANSWERED")
    print("=" * 80)

    any_pass = any(ar["checks"].get("all_pass", False) for ar in adversarial_results)
    print(f"\n1. Does mean-reversion survive without survivorship bias?")
    print(f"   {'YES' if any_pass else 'NO'} — "
          f"{sum(1 for ar in adversarial_results if ar['checks'].get('all_pass'))}/"
          f"{len(adversarial_results)} configs pass ALL gates")

    bear_better = any(ar["checks"].get("regime", {}).get("better_in_bear", False)
                      for ar in adversarial_results if ar["checks"].get("all_pass"))
    print(f"\n2. Does it still work better in bear markets?")
    print(f"   {'YES' if bear_better else 'NO / INCONCLUSIVE'}")

    vix_proxy = any(ar.get("vix_analysis", {}).get("is_vix_proxy", False)
                    for ar in adversarial_results)
    print(f"\n3. Is it a different edge from VIX mean-reversion?")
    print(f"   {'LIKELY SAME — high VIX correlation detected' if vix_proxy else 'YES — independent edge (low VIX correlation)'}")

    if adversarial_results:
        spy_corrs = [ar["spy_correlation"] for ar in adversarial_results
                     if ar["spy_correlation"] is not None]
        if spy_corrs:
            avg_corr = np.mean(spy_corrs)
            print(f"\n4. SPY Correlation: avg={avg_corr:.3f} "
                  f"({'LOW — good diversifier' if abs(avg_corr) < 0.3 else 'MODERATE' if abs(avg_corr) < 0.6 else 'HIGH — not diversifying'})")

    # ── Save results ─────────────────────────────────────────────────────
    output = {
        "metadata": {
            "script": "etf_reversal_v2.py",
            "run_date": datetime.now().isoformat(),
            "universe": ETF_UNIVERSE,
            "usable_etfs": etf_cols,
            "period": f"{START} to {END}",
            "n_permutations": N_PERM,
            "wf_train_months": WF_TRAIN_MONTHS,
            "wf_oot_months": WF_OOT_MONTHS,
        },
        "fixed_config_summary": [
            {
                "name": r["name"],
                "mode": r["mode"],
                "lookback": r["lookback"],
                "k": r["k"],
                "hold": r["hold"],
                "metrics": {k: round(v, 4) if isinstance(v, float) else v
                            for k, v in r["metrics"].items()},
            }
            for r in fixed_results
        ],
        "walkforward": {
            "n_folds": len(fold_results),
            "fold_details": fold_results,
            "oot_combined_metrics": {k: round(v, 4) if isinstance(v, float) else v
                                      for k, v in wf_metrics.items()} if wf_metrics else None,
            "oot_adversarial": wf_adversarial,
        },
        "adversarial_results": adversarial_results,
        "key_findings": {
            "survives_without_survivorship_bias": any_pass,
            "better_in_bear_markets": bear_better,
            "independent_from_vix": not vix_proxy,
            "avg_spy_correlation": round(float(np.mean(spy_corrs)), 4) if spy_corrs else None,
            "configs_passing_all_gates": [
                ar["config"] for ar in adversarial_results
                if ar["checks"].get("all_pass")
            ],
        },
    }

    out_path = OUTPUT_DIR / "etf_reversal_v2_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
