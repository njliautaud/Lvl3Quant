#!/usr/bin/env python3
"""
VIX Transition Zone (25-30) Research v1
========================================

Our knowledge base identified VIX 25-30 as the strategy's blind spot:
  - All model variants underperform there (Sharpe 1.6-1.8 vs 3.7+ elsewhere)
  - This is the "nervous but not crashing" zone

This script investigates WHAT happens in VIX 25-30 and tests 6 strategy
variants to find the best approach for this regime.

Analysis:
  Part 1 — Descriptive: duration, exit direction, sector dispersion, momentum efficacy
  Part 2 — Strategy variants (walk-forward LGBM, $645, 15% haircut entry+exit):
    A. Bull call spreads (standard approach)
    B. Put credit spreads (sell rich premium)
    C. Iron condors (range-bound bet on mean reversion)
    D. Reduced position size (half normal)
    E. Cash (skip the blind spot)
    F. Hedged spreads (bull call + protective put)
  Part 3 — Adversarial validation on each variant

HC #0: Sliding windows only. No expanding.
Uses standardized options_pricer and adversarial_validator tools.
MLflow logging to Jupiter (http://jupiter:5000).
"""
from __future__ import annotations

import sys
import json
import warnings
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add project root
sys.path.insert(0, "/home/jupiter/Lvl3Quant")

from research.tools.options_pricer import (
    price_bull_call_spread,
    price_bear_put_spread,
    exit_spread_value,
    spread_pnl,
    estimate_iv,
    bs_put_price,
    bs_call_price,
    compute_atr,
    COMMISSION_RT_SPREAD,
    DEFAULT_HAIRCUT,
)
from research.tools.adversarial_validator import validate_trades


def fprint(*args, **kwargs):
    print(*args, **kwargs, flush=True)


# ─── MLflow Setup ──────────────────────────────────────────────────

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen("http://jupiter:5000/", timeout=3)
    import mlflow
    mlflow.set_tracking_uri("http://jupiter:5000")
    MLFLOW_OK = True
    fprint("[MLflow] Connected to http://jupiter:5000")
except Exception:
    fprint("[MLflow] Not available, skipping logging")

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/vix_transition_zone_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Configuration ─────────────────────────────────────────────────

CONFIG = {
    "sector_etfs": [
        "XLE", "XLK", "XLF", "XLV", "XLI", "XLP", "XLU", "XLRE", "XLB", "XLC", "XLY",
    ],
    "extra_tickers": ["SPY", "TLT", "HYG", "GLD"],
    "start_date": "2009-01-01",
    "end_date": "2026-07-26",
    "initial_capital": 645.0,

    # VIX transition zone
    "vix_low": 25.0,
    "vix_high": 30.0,

    # Options parameters
    "target_dte": 30,
    "spread_width_pct": 0.03,
    "top_n": 3,
    "rebalance_freq_days": 10,
    "haircut": DEFAULT_HAIRCUT,
    "commission_rt": COMMISSION_RT_SPREAD,
    "max_trade_pct": 0.35,
    "max_total_pct": 0.80,

    # Iron condor parameters
    "ic_put_otm_pct": 0.04,
    "ic_call_otm_pct": 0.04,
    "ic_spread_width_pct": 0.02,

    # Walk-forward LGBM (6-month test windows for CPU efficiency)
    "train_months": 12,
    "test_months": 6,
    "lgbm_params": {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 24,
        "learning_rate": 0.1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "max_depth": 4,
        "verbosity": -1,
        "seed": 42,
        "n_jobs": 4,
    },
    "num_boost_round": 100,
    "early_stopping_rounds": 15,

    # Adversarial
    "n_perms": 500,
}


# ═══════════════════════════════════════════════════════════════════
# PART 0: DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════════════

def download_data(config):
    """Download all required tickers plus VIX."""
    import yfinance as yf

    all_tickers = list(set(config["sector_etfs"] + config["extra_tickers"]))
    fprint(f"\n{'='*70}")
    fprint(f"PART 0: DATA DOWNLOAD")
    fprint(f"{'='*70}")
    fprint(f"  Downloading {len(all_tickers)} tickers + VIX from {config['start_date']} to {config['end_date']}...")

    data = yf.download(
        all_tickers, start=config["start_date"], end=config["end_date"],
        auto_adjust=True, progress=False, group_by="ticker",
    )

    prices = {}
    for ticker in all_tickers:
        try:
            if len(all_tickers) > 1:
                df = data[ticker][["Close", "High", "Low", "Volume"]].dropna()
            else:
                df = data[["Close", "High", "Low", "Volume"]].dropna()
            df.columns = ["close", "high", "low", "volume"]
            if len(df) > 60:
                prices[ticker] = df
                fprint(f"    {ticker}: {len(df)} days ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")
        except Exception:
            pass

    # VIX
    vix_data = yf.download(
        "^VIX", start=config["start_date"], end=config["end_date"],
        auto_adjust=True, progress=False,
    )
    vix_series = vix_data["Close"].dropna()
    if hasattr(vix_series, "columns"):
        vix_series = vix_series.iloc[:, 0]
    fprint(f"    VIX: {len(vix_series)} days")

    return prices, vix_series


# ═══════════════════════════════════════════════════════════════════
# PART 1: DESCRIPTIVE ANALYSIS OF VIX 25-30 ZONE
# ═══════════════════════════════════════════════════════════════════

def analyze_transition_zone(prices, vix_series, config):
    """Comprehensive analysis of what happens in VIX 25-30."""
    fprint(f"\n{'='*70}")
    fprint(f"PART 1: DESCRIPTIVE ANALYSIS — VIX {config['vix_low']}-{config['vix_high']} ZONE")
    fprint(f"{'='*70}")

    vix_low, vix_high = config["vix_low"], config["vix_high"]

    # Align VIX with SPY dates
    spy_close = prices["SPY"]["close"]
    vix_aligned = vix_series.reindex(spy_close.index, method="ffill").dropna()

    # Boolean mask for transition zone
    in_zone = (vix_aligned >= vix_low) & (vix_aligned < vix_high)
    total_days = len(vix_aligned)
    zone_days = in_zone.sum()

    fprint(f"\n  Total trading days: {total_days}")
    fprint(f"  Days in VIX {vix_low}-{vix_high}: {zone_days} ({100*zone_days/total_days:.1f}%)")

    # ── 1a. Duration of VIX 25-30 episodes ──
    fprint(f"\n  --- 1a. Episode Duration ---")
    episodes = []
    in_episode = False
    start_date = None

    for date, is_in in in_zone.items():
        if is_in and not in_episode:
            in_episode = True
            start_date = date
        elif not is_in and in_episode:
            in_episode = False
            duration = (date - start_date).days
            episodes.append({
                "start": start_date,
                "end": date,
                "duration_calendar": duration,
                "duration_trading": in_zone.loc[start_date:date].sum(),
            })
    if in_episode:
        end_date = vix_aligned.index[-1]
        episodes.append({
            "start": start_date,
            "end": end_date,
            "duration_calendar": (end_date - start_date).days,
            "duration_trading": in_zone.loc[start_date:end_date].sum(),
        })

    episode_durations = [e["duration_trading"] for e in episodes]
    fprint(f"  Number of episodes: {len(episodes)}")
    if episodes:
        fprint(f"  Average duration: {np.mean(episode_durations):.1f} trading days")
        fprint(f"  Median duration:  {np.median(episode_durations):.1f} trading days")
        fprint(f"  Max duration:     {np.max(episode_durations)} trading days")
        fprint(f"  Min duration:     {np.min(episode_durations)} trading days")
        fprint(f"  Distribution:     <=3d: {sum(1 for d in episode_durations if d<=3)}, "
               f"4-10d: {sum(1 for d in episode_durations if 4<=d<=10)}, "
               f"11-20d: {sum(1 for d in episode_durations if 11<=d<=20)}, "
               f">20d: {sum(1 for d in episode_durations if d>20)}")

    # ── 1b. What happens AFTER VIX 25-30? ──
    fprint(f"\n  --- 1b. Post-Episode VIX Direction ---")
    up_count, down_count = 0, 0
    post_vix_changes = []
    post_spy_returns = []

    for ep in episodes:
        end_date = ep["end"]
        # Look at VIX 5 trading days after episode ends
        future_dates = vix_aligned.loc[end_date:].iloc[:6]
        if len(future_dates) >= 2:
            vix_at_exit = future_dates.iloc[0]
            vix_5d_later = future_dates.iloc[-1]
            change = vix_5d_later - vix_at_exit
            post_vix_changes.append(change)
            if vix_5d_later > vix_high:
                up_count += 1
            elif vix_5d_later < vix_low:
                down_count += 1
        # SPY return in 5 days after
        future_spy = spy_close.loc[end_date:].iloc[:6]
        if len(future_spy) >= 2:
            spy_ret = future_spy.iloc[-1] / future_spy.iloc[0] - 1
            post_spy_returns.append(spy_ret)

    if post_vix_changes:
        fprint(f"  After episode ends:")
        fprint(f"    VIX goes UP (>30, crash):    {up_count} times ({100*up_count/len(episodes):.0f}%)")
        fprint(f"    VIX goes DOWN (<25, recover): {down_count} times ({100*down_count/len(episodes):.0f}%)")
        fprint(f"    Stays in zone:               {len(episodes)-up_count-down_count} times")
        fprint(f"    Avg VIX change 5d after:     {np.mean(post_vix_changes):+.2f}")
        fprint(f"    Avg SPY return 5d after:     {100*np.mean(post_spy_returns):+.2f}%")
        fprint(f"    SPY 5d return positive:      {100*np.mean([r>0 for r in post_spy_returns]):.0f}%")

    # ── 1c. Sector dispersion during 25-30 ──
    fprint(f"\n  --- 1c. Sector Dispersion ---")
    sector_returns = {}
    for etf in config["sector_etfs"]:
        if etf in prices:
            sector_returns[etf] = prices[etf]["close"].pct_change()

    sector_ret_df = pd.DataFrame(sector_returns)
    # Cross-sectional dispersion (daily std of sector returns)
    daily_dispersion = sector_ret_df.std(axis=1)

    disp_in_zone = daily_dispersion[in_zone].mean()
    disp_below = daily_dispersion[(vix_aligned < vix_low)].mean()
    disp_above = daily_dispersion[(vix_aligned >= vix_high)].mean()

    fprint(f"  Cross-sectional sector dispersion (daily stdev of sector returns):")
    fprint(f"    VIX <{vix_low}:            {disp_below*100:.3f}%")
    fprint(f"    VIX {vix_low}-{vix_high}:          {disp_in_zone*100:.3f}%")
    fprint(f"    VIX >{vix_high}:           {disp_above*100:.3f}%")
    fprint(f"    Ratio (zone/below):   {disp_in_zone/disp_below:.2f}x")

    # ── 1d. Sector performance during 25-30 ──
    fprint(f"\n  --- 1d. Sector Performance During VIX {vix_low}-{vix_high} ---")
    sector_zone_perf = {}
    for etf in config["sector_etfs"]:
        if etf in sector_returns:
            zone_rets = sector_returns[etf][in_zone]
            avg_ret = zone_rets.mean() * 252  # annualized
            vol = zone_rets.std() * np.sqrt(252)
            sharpe = avg_ret / vol if vol > 0 else 0
            sector_zone_perf[etf] = {
                "ann_return": avg_ret,
                "ann_vol": vol,
                "sharpe": sharpe,
            }

    perf_sorted = sorted(sector_zone_perf.items(), key=lambda x: x[1]["sharpe"], reverse=True)
    fprint(f"  {'ETF':<6} {'Ann Ret':>8} {'Ann Vol':>8} {'Sharpe':>7}")
    fprint(f"  {'-'*32}")
    for etf, perf in perf_sorted:
        fprint(f"  {etf:<6} {perf['ann_return']*100:>7.1f}% {perf['ann_vol']*100:>7.1f}% {perf['sharpe']:>7.2f}")

    # ── 1e. Does momentum work during 25-30? ──
    fprint(f"\n  --- 1e. Momentum Efficacy During VIX {vix_low}-{vix_high} ---")
    # Rank sectors by 21d momentum, check if top ranks outperform
    mom_21d = {}
    for etf in config["sector_etfs"]:
        if etf in prices:
            mom_21d[etf] = prices[etf]["close"].pct_change(21)

    mom_df = pd.DataFrame(mom_21d)
    fwd_ret = sector_ret_df.shift(-5).rolling(5).sum()  # 5d forward return

    # Cross-sectional rank correlation (IC) between momentum and forward return
    zone_dates = vix_aligned[in_zone].index
    non_zone_dates = vix_aligned[~in_zone].index

    ic_zone = []
    ic_outside = []

    for date in zone_dates:
        if date in mom_df.index and date in fwd_ret.index:
            m = mom_df.loc[date].dropna()
            f = fwd_ret.loc[date].dropna()
            common = m.index.intersection(f.index)
            if len(common) >= 5:
                ic = m[common].corr(f[common])
                if not np.isnan(ic):
                    ic_zone.append(ic)

    for date in non_zone_dates.intersection(mom_df.index).intersection(fwd_ret.index):
        m = mom_df.loc[date].dropna()
        f = fwd_ret.loc[date].dropna()
        common = m.index.intersection(f.index)
        if len(common) >= 5:
            ic = m[common].corr(f[common])
            if not np.isnan(ic):
                ic_outside.append(ic)

    fprint(f"  Momentum IC (21d momentum -> 5d fwd return):")
    fprint(f"    In VIX {vix_low}-{vix_high}:   mean IC = {np.mean(ic_zone):.4f} (n={len(ic_zone)})")
    fprint(f"    Outside zone:  mean IC = {np.mean(ic_outside):.4f} (n={len(ic_outside)})")
    fprint(f"    Ratio:         {np.mean(ic_zone)/(np.mean(ic_outside)+1e-8):.2f}x")

    # TLT/HYG/GLD correlation analysis
    fprint(f"\n  --- Cross-Asset Behavior in VIX {vix_low}-{vix_high} ---")
    for ticker in ["TLT", "HYG", "GLD"]:
        if ticker in prices:
            t_ret = prices[ticker]["close"].pct_change()
            spy_ret_d = spy_close.pct_change()
            common_idx = t_ret.index.intersection(spy_ret_d.index).intersection(vix_aligned.index)
            zone_mask = in_zone.reindex(common_idx, fill_value=False)
            corr_zone = t_ret.reindex(common_idx)[zone_mask].corr(spy_ret_d.reindex(common_idx)[zone_mask])
            corr_all = t_ret.reindex(common_idx).corr(spy_ret_d.reindex(common_idx))
            avg_ret_zone = t_ret.reindex(common_idx)[zone_mask].mean() * 252
            fprint(f"  {ticker}: corr w/ SPY in zone={corr_zone:.3f} (overall={corr_all:.3f}), "
                   f"ann ret in zone={avg_ret_zone*100:.1f}%")

    descriptive_results = {
        "zone_days": int(zone_days),
        "pct_of_history": float(100 * zone_days / total_days),
        "n_episodes": len(episodes),
        "avg_episode_days": float(np.mean(episode_durations)) if episodes else 0,
        "median_episode_days": float(np.median(episode_durations)) if episodes else 0,
        "pct_vix_up_after": float(100 * up_count / len(episodes)) if episodes else 0,
        "pct_vix_down_after": float(100 * down_count / len(episodes)) if episodes else 0,
        "avg_spy_5d_after": float(np.mean(post_spy_returns)) if post_spy_returns else 0,
        "sector_dispersion_in_zone": float(disp_in_zone),
        "sector_dispersion_below": float(disp_below),
        "momentum_ic_zone": float(np.mean(ic_zone)) if ic_zone else 0,
        "momentum_ic_outside": float(np.mean(ic_outside)) if ic_outside else 0,
    }

    return descriptive_results, in_zone, episodes


# ═══════════════════════════════════════════════════════════════════
# PART 2: FEATURE ENGINEERING + WALK-FORWARD LGBM RANKING
# ═══════════════════════════════════════════════════════════════════

def build_features(prices, vix_series, config):
    """Build feature panel for sector ranking."""
    fprint(f"\n{'='*70}")
    fprint(f"PART 2: FEATURE ENGINEERING + WALK-FORWARD LGBM")
    fprint(f"{'='*70}")

    spy_close = prices["SPY"]["close"]
    records = []

    for ticker in config["sector_etfs"]:
        if ticker not in prices:
            continue
        df = prices[ticker].copy()
        close = df["close"]
        volume = df["volume"]

        for lb in [5, 10, 21, 63, 126]:
            df[f"mom_{lb}d"] = close.pct_change(lb)

        for lb in [21, 63]:
            spy_ret = spy_close.pct_change(lb)
            ticker_ret = close.pct_change(lb)
            df[f"rs_vs_spy_{lb}d"] = ticker_ret - spy_ret.reindex(ticker_ret.index)

        df["vol_21d"] = close.pct_change().rolling(21).std() * np.sqrt(252)
        df["vol_63d"] = close.pct_change().rolling(63).std() * np.sqrt(252)
        df["vol_ratio"] = df["vol_21d"] / (df["vol_63d"] + 1e-8)
        df["vol_sma_ratio"] = volume / volume.rolling(21).mean()

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14).mean()
        df["rsi_14"] = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-8)))

        df["sma_50_dist"] = close / close.rolling(50).mean() - 1
        df["sma_200_dist"] = close / close.rolling(200).mean() - 1
        df["high_52w_pct"] = close / close.rolling(252).max()

        # Skew and kurtosis
        lr = np.log(close / close.shift(1))
        df["skew_63d"] = lr.rolling(63).skew()
        df["kurt_63d"] = lr.rolling(63).kurt()

        # Forward return (target): 21d
        df["fwd_return_21d"] = close.pct_change(21).shift(-21)

        # ATR
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - close.shift(1)).abs()
        tr3 = (df["low"] - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr_14"] = tr.ewm(alpha=1 / 14, min_periods=14).mean()

        # VIX
        vix_aligned = vix_series.reindex(df.index, method="ffill")
        df["vix"] = vix_aligned

        # Vectorized panel construction (avoid iterrows)
        meta_cols = ["close", "high", "low", "volume", "fwd_return_21d", "atr_14", "vix"]
        feature_cols = [c for c in df.columns if c not in meta_cols]
        df_out = df[feature_cols + ["fwd_return_21d", "close", "atr_14", "vix"]].copy()
        df_out = df_out.rename(columns={"fwd_return_21d": "target"})
        df_out["date"] = df_out.index
        df_out["ticker"] = ticker
        records.append(df_out)

    panel = pd.concat(records, ignore_index=True).dropna(subset=["target"])
    fprint(f"  Feature panel: {len(panel)} rows, {len(panel.columns)} cols")
    return panel


def walk_forward_lgbm(panel, config):
    """Walk-forward LGBM ranking with sliding window. Returns predictions panel."""
    import lightgbm as lgb

    feature_cols = [c for c in panel.columns if c not in
                    ["date", "ticker", "target", "close", "atr_14", "vix"]]

    panel = panel.sort_values("date")
    dates = sorted(panel["date"].unique())

    train_months = config.get("train_months", 12)
    test_months = config.get("test_months", 1)

    predictions = []
    n_folds = 0
    total_steps = len(range(train_months * 21, len(dates), max(1, int(21 * test_months))))
    fprint(f"  Starting walk-forward: ~{total_steps} folds...")

    for step_num, i in enumerate(range(train_months * 21, len(dates), max(1, int(21 * test_months)))):
        if i >= len(dates):
            break

        # Sliding window: train on most recent train_months only
        train_start_idx = max(0, i - train_months * 21)
        train_dates = dates[train_start_idx:i]
        test_end_idx = min(len(dates), i + test_months * 21)
        test_dates = dates[i:test_end_idx]

        if len(train_dates) < 100 or len(test_dates) == 0:
            continue

        train_mask = panel["date"].isin(train_dates)
        test_mask = panel["date"].isin(test_dates)

        X_train = panel.loc[train_mask, feature_cols].values
        y_train = panel.loc[train_mask, "target"].values
        X_test = panel.loc[test_mask, feature_cols].values

        X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
        y_train = np.nan_to_num(y_train, nan=0, posinf=0, neginf=0)
        X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

        try:
            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(
                X_train[-len(X_train) // 5:],
                label=y_train[-len(y_train) // 5:],
            )
            model = lgb.train(
                config.get("lgbm_params", CONFIG["lgbm_params"]),
                dtrain,
                num_boost_round=config.get("num_boost_round", 300),
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(
                    config.get("early_stopping_rounds", 30),
                    verbose=False,
                )],
            )
            preds = model.predict(X_test, num_iteration=model.best_iteration)
            test_rows = panel.loc[test_mask].copy()
            test_rows["prediction"] = preds
            predictions.append(test_rows)
            n_folds += 1
            if n_folds % 10 == 0:
                fprint(f"    Fold {n_folds}/{total_steps} complete")
        except Exception as e:
            fprint(f"    Fold {n_folds} error: {e}")
            continue

    if not predictions:
        fprint("  ERROR: No predictions generated")
        return pd.DataFrame()

    pred_panel = pd.concat(predictions, ignore_index=True)
    fprint(f"  Walk-forward complete: {n_folds} folds, {len(pred_panel)} prediction rows")
    return pred_panel


# ═══════════════════════════════════════════════════════════════════
# PART 2b: STRATEGY VARIANTS — TRADE SIMULATION
# ═══════════════════════════════════════════════════════════════════

def price_put_credit_spread(S, K_short, K_long, dte, atr, vix, haircut=DEFAULT_HAIRCUT):
    """
    Price a put credit spread (sell put at K_short, buy put at K_long).
    K_short > K_long. You RECEIVE net credit.
    Returns (credit_received, max_loss) per share.
    """
    T = dte / 365.0
    sigma = estimate_iv(atr, S, vix)
    short_put = bs_put_price(S, K_short, T, sigma=sigma)
    long_put = bs_put_price(S, K_long, T, sigma=sigma)

    credit = short_put - long_put  # net credit
    credit = max(credit, 0.001)

    # Haircut: you receive LESS than fair value
    credit_received = credit * (1.0 - haircut)
    spread_width = K_short - K_long
    max_loss = spread_width - credit_received

    return float(credit_received), float(max_loss)


def price_iron_condor(S, put_short_K, put_long_K, call_short_K, call_long_K,
                      dte, atr, vix, haircut=DEFAULT_HAIRCUT):
    """
    Price an iron condor.
    Sell put at put_short_K, buy put at put_long_K (lower).
    Sell call at call_short_K, buy call at call_long_K (higher).
    Returns (credit_received, max_loss) per share.
    """
    T = dte / 365.0
    sigma = estimate_iv(atr, S, vix)

    put_credit = bs_put_price(S, put_short_K, T, sigma=sigma) - \
                 bs_put_price(S, put_long_K, T, sigma=sigma)
    call_credit = bs_call_price(S, call_short_K, T, sigma=sigma) - \
                  bs_call_price(S, call_long_K, T, sigma=sigma)

    total_credit = max(put_credit + call_credit, 0.001)
    credit_received = total_credit * (1.0 - haircut)

    put_spread_width = put_short_K - put_long_K
    call_spread_width = call_long_K - call_short_K
    max_loss_side = max(put_spread_width, call_spread_width) - credit_received

    return float(credit_received), float(max_loss_side)


def iron_condor_pnl_at_expiry(S_expiry, put_short_K, put_long_K,
                              call_short_K, call_long_K, credit_received):
    """PnL of iron condor at expiry."""
    # Put side
    put_short_val = max(put_short_K - S_expiry, 0)
    put_long_val = max(put_long_K - S_expiry, 0)
    put_pnl = -(put_short_val - put_long_val)

    # Call side
    call_short_val = max(S_expiry - call_short_K, 0)
    call_long_val = max(S_expiry - call_long_K, 0)
    call_pnl = -(call_short_val - call_long_val)

    return credit_received + put_pnl + call_pnl


def simulate_strategy_variant(variant_name, pred_panel, prices, vix_series, config):
    """
    Simulate a strategy variant within VIX 25-30 zone only.
    Returns list of trade dicts.
    """
    fprint(f"\n  --- Simulating: {variant_name} ---")

    vix_low, vix_high = config["vix_low"], config["vix_high"]
    spy_close = prices["SPY"]["close"]
    top_n = config["top_n"]
    target_dte = config["target_dte"]
    haircut = config["haircut"]
    commission = config["commission_rt"]
    capital = config["initial_capital"]
    max_trade_pct = config["max_trade_pct"]
    max_total_pct = config["max_total_pct"]

    # Only trade on rebalance dates within VIX 25-30
    dates = sorted(pred_panel["date"].unique())
    rebal_dates = dates[::config["rebalance_freq_days"]]

    trades = []
    equity = capital
    active_trades = []

    for date in rebal_dates:
        # Check VIX
        if date not in vix_series.index:
            continue
        vix_val = vix_series.loc[:date].iloc[-1] if date in vix_series.index else vix_series.asof(date)
        if np.isnan(vix_val):
            continue

        # Only trade in the VIX 25-30 zone
        if vix_val < vix_low or vix_val >= vix_high:
            continue

        # Check for expiring trades
        for trade in active_trades[:]:
            if date >= trade["expiry_date"]:
                # Settle at expiry
                ticker = trade["ticker"]
                if ticker in prices and date in prices[ticker]["close"].index:
                    S_exp = float(prices[ticker]["close"].loc[:date].iloc[-1])
                else:
                    S_exp = trade["entry_price"]

                pnl = _settle_trade(trade, S_exp, commission)
                trade["pnl"] = pnl
                trade["exit_date"] = date
                trade["exit_price"] = S_exp
                equity += pnl
                trades.append(trade)
                active_trades.remove(trade)

        if variant_name == "E_cash":
            # Strategy E: sit in cash, no trades
            continue

        # Get top-ranked sectors for this date
        day_preds = pred_panel[pred_panel["date"] == date].copy()
        if len(day_preds) == 0:
            continue

        day_preds = day_preds.sort_values("prediction", ascending=False)
        top_sectors = day_preds.head(top_n)

        # Capital allocation
        if variant_name == "D_reduced":
            effective_max_trade = max_trade_pct * 0.5  # Half position size
        else:
            effective_max_trade = max_trade_pct

        deployed = sum(t.get("capital_deployed", 0) for t in active_trades)
        available = equity * max_total_pct - deployed

        for _, row in top_sectors.iterrows():
            ticker = row["ticker"]
            S = float(row["close"])
            atr = float(row["atr_14"]) if not np.isnan(row["atr_14"]) else S * 0.02
            vix = float(row["vix"]) if not np.isnan(row["vix"]) else vix_val

            # Skip if already in this ticker
            if any(t["ticker"] == ticker for t in active_trades):
                continue

            trade_budget = min(equity * effective_max_trade, available)
            if trade_budget < 10:
                continue

            trade = _enter_trade(variant_name, ticker, S, atr, vix, date,
                                 target_dte, trade_budget, haircut, config)
            if trade is not None:
                active_trades.append(trade)
                available -= trade.get("capital_deployed", 0)

    # Settle any remaining active trades at last available date
    last_date = dates[-1] if dates else pd.Timestamp.now()
    for trade in active_trades:
        ticker = trade["ticker"]
        if ticker in prices:
            S_exp = float(prices[ticker]["close"].iloc[-1])
        else:
            S_exp = trade["entry_price"]
        pnl = _settle_trade(trade, S_exp, commission)
        trade["pnl"] = pnl
        trade["exit_date"] = last_date
        trade["exit_price"] = S_exp
        equity += pnl
        trades.append(trade)

    fprint(f"    Trades: {len(trades)}, Final equity: ${equity:.2f}")
    return trades


def _enter_trade(variant, ticker, S, atr, vix, date, target_dte, budget, haircut, config):
    """Enter a trade based on variant type. Returns trade dict or None."""
    expiry = date + pd.Timedelta(days=int(target_dte * 365 / 252))

    if variant in ("A_bull_call", "D_reduced"):
        # Bull call spread
        K1 = S  # ATM
        K2 = S * (1 + config["spread_width_pct"])
        entry_cost, max_profit = price_bull_call_spread(S, K1, K2, target_dte, atr, vix, haircut)
        cost_per_contract = entry_cost * 100 + COMMISSION_RT_SPREAD / 2
        if cost_per_contract <= 0:
            return None
        contracts = max(1, int(budget / cost_per_contract))
        capital_deployed = contracts * cost_per_contract

        return {
            "type": "bull_call",
            "ticker": ticker,
            "entry_date": date,
            "expiry_date": expiry,
            "entry_price": S,
            "K1": K1, "K2": K2,
            "entry_cost": entry_cost,
            "contracts": contracts,
            "capital_deployed": capital_deployed,
        }

    elif variant == "B_put_credit":
        # Put credit spread — sell premium
        K_short = S * (1 - config.get("ic_put_otm_pct", 0.03))
        K_long = K_short * (1 - config["spread_width_pct"])
        credit, max_loss = price_put_credit_spread(S, K_short, K_long, target_dte, atr, vix, haircut)
        # Capital required = max_loss * 100 (margin)
        margin_per = max_loss * 100 + COMMISSION_RT_SPREAD / 2
        if margin_per <= 0:
            return None
        contracts = max(1, int(budget / margin_per))
        capital_deployed = contracts * margin_per

        return {
            "type": "put_credit",
            "ticker": ticker,
            "entry_date": date,
            "expiry_date": expiry,
            "entry_price": S,
            "K_short": K_short, "K_long": K_long,
            "credit_received": credit,
            "contracts": contracts,
            "capital_deployed": capital_deployed,
        }

    elif variant == "C_iron_condor":
        # Iron condor — range-bound bet
        put_short_K = S * (1 - config["ic_put_otm_pct"])
        put_long_K = put_short_K * (1 - config["ic_spread_width_pct"])
        call_short_K = S * (1 + config["ic_call_otm_pct"])
        call_long_K = call_short_K * (1 + config["ic_spread_width_pct"])

        credit, max_loss = price_iron_condor(
            S, put_short_K, put_long_K, call_short_K, call_long_K,
            target_dte, atr, vix, haircut
        )
        margin_per = max_loss * 100 + COMMISSION_RT_SPREAD  # 4 legs each side
        if margin_per <= 0:
            return None
        contracts = max(1, int(budget / margin_per))
        capital_deployed = contracts * margin_per

        return {
            "type": "iron_condor",
            "ticker": ticker,
            "entry_date": date,
            "expiry_date": expiry,
            "entry_price": S,
            "put_short_K": put_short_K, "put_long_K": put_long_K,
            "call_short_K": call_short_K, "call_long_K": call_long_K,
            "credit_received": credit,
            "contracts": contracts,
            "capital_deployed": capital_deployed,
        }

    elif variant == "F_hedged":
        # Bull call spread + protective put
        K1 = S  # ATM call
        K2 = S * (1 + config["spread_width_pct"])
        entry_cost, max_profit = price_bull_call_spread(S, K1, K2, target_dte, atr, vix, haircut)

        # Add protective put (3% OTM)
        K_put = S * 0.97
        sigma = estimate_iv(atr, S, vix)
        put_price = bs_put_price(S, K_put, target_dte / 365.0, sigma=sigma)
        put_cost = put_price * (1 + haircut)

        total_cost = entry_cost + put_cost
        cost_per_contract = total_cost * 100 + COMMISSION_RT_SPREAD
        if cost_per_contract <= 0:
            return None
        contracts = max(1, int(budget / cost_per_contract))
        capital_deployed = contracts * cost_per_contract

        return {
            "type": "hedged",
            "ticker": ticker,
            "entry_date": date,
            "expiry_date": expiry,
            "entry_price": S,
            "K1": K1, "K2": K2,
            "K_put": K_put,
            "entry_cost_spread": entry_cost,
            "put_cost": put_cost,
            "total_entry_cost": total_cost,
            "contracts": contracts,
            "capital_deployed": capital_deployed,
        }

    return None


def _settle_trade(trade, S_exp, commission):
    """Settle a trade at expiry, return PnL."""
    contracts = trade["contracts"]
    ttype = trade["type"]

    if ttype == "bull_call":
        K1, K2 = trade["K1"], trade["K2"]
        intrinsic = max(S_exp - K1, 0) - max(S_exp - K2, 0)
        pnl = (intrinsic - trade["entry_cost"]) * contracts * 100 - commission * contracts
        return pnl

    elif ttype == "put_credit":
        K_short, K_long = trade["K_short"], trade["K_long"]
        # Put credit: profit if stock stays above K_short
        short_put_val = max(K_short - S_exp, 0)
        long_put_val = max(K_long - S_exp, 0)
        loss_at_exp = short_put_val - long_put_val  # what you owe
        pnl = (trade["credit_received"] - loss_at_exp) * contracts * 100 - commission * contracts
        return pnl

    elif ttype == "iron_condor":
        pnl_per_share = iron_condor_pnl_at_expiry(
            S_exp,
            trade["put_short_K"], trade["put_long_K"],
            trade["call_short_K"], trade["call_long_K"],
            trade["credit_received"],
        )
        pnl = pnl_per_share * contracts * 100 - commission * 2 * contracts  # 2x commission for IC
        return pnl

    elif ttype == "hedged":
        K1, K2 = trade["K1"], trade["K2"]
        K_put = trade["K_put"]
        # Bull call spread intrinsic
        spread_val = max(S_exp - K1, 0) - max(S_exp - K2, 0)
        # Protective put intrinsic
        put_val = max(K_put - S_exp, 0)
        total_exit = spread_val + put_val
        pnl = (total_exit - trade["total_entry_cost"]) * contracts * 100 - commission * contracts
        return pnl

    return 0.0


# ═══════════════════════════════════════════════════════════════════
# PART 3: METRICS AND VALIDATION
# ═══════════════════════════════════════════════════════════════════

def compute_strategy_metrics(trades, initial_capital, strategy_name):
    """Compute Sharpe, Sortino, MDD, WR etc from trade list."""
    if not trades:
        return {"name": strategy_name, "n_trades": 0, "sharpe": 0, "error": "no trades"}

    pnls = [t["pnl"] for t in trades]
    n_trades = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / n_trades if n_trades > 0 else 0

    # Build equity curve
    equity = [initial_capital]
    for p in pnls:
        equity.append(equity[-1] + p)
    equity = np.array(equity)

    # Returns (equity-based)
    returns = np.diff(equity) / equity[:-1]
    returns = returns[np.isfinite(returns)]

    if len(returns) < 2:
        return {"name": strategy_name, "n_trades": n_trades, "sharpe": 0, "error": "too few returns"}

    # Annualize: estimate trades per year from date range
    if trades[0].get("entry_date") and trades[-1].get("entry_date"):
        date_range = (trades[-1]["entry_date"] - trades[0]["entry_date"]).days
        years = max(date_range / 365.0, 0.5)
        trades_per_year = n_trades / years
    else:
        trades_per_year = 12
        years = n_trades / 12.0

    ann_factor = np.sqrt(trades_per_year) if trades_per_year > 0 else 1

    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    sharpe = (mean_ret / std_ret) * ann_factor if std_ret > 0 else 0

    downside = returns[returns < 0]
    downside_std = np.std(downside) if len(downside) > 1 else std_ret
    sortino = (mean_ret / downside_std) * ann_factor if downside_std > 0 else 0

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd = float(np.min(drawdown))

    # CAGR
    final = equity[-1]
    cagr = (final / initial_capital) ** (1.0 / max(years, 0.5)) - 1

    # Profit factor
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Average trade
    avg_trade = np.mean(pnls)

    return {
        "name": strategy_name,
        "n_trades": n_trades,
        "win_rate": float(wr),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "cagr": float(cagr),
        "max_dd": float(max_dd),
        "profit_factor": float(pf),
        "avg_trade_pnl": float(avg_trade),
        "total_pnl": float(sum(pnls)),
        "final_equity": float(equity[-1]),
        "years": float(years),
        "trades_per_year": float(trades_per_year),
    }


def run_adversarial(trades, initial_capital, spy_prices, strategy_name):
    """Run adversarial validation using standardized tool."""
    try:
        result = validate_trades(
            trades=trades,
            initial_capital=initial_capital,
            spy_prices=spy_prices,
            n_perms=CONFIG["n_perms"],
        )
        return result
    except Exception as e:
        fprint(f"    Adversarial validation error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    start_time = datetime.now()
    fprint(f"\n{'#'*70}")
    fprint(f"# VIX TRANSITION ZONE (25-30) RESEARCH v1")
    fprint(f"# Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint(f"{'#'*70}")

    # ── Download data ──
    prices, vix_series = download_data(CONFIG)

    # ── Part 1: Descriptive analysis ──
    descriptive, in_zone, episodes = analyze_transition_zone(prices, vix_series, CONFIG)

    # ── Part 2: Build features + LGBM ranking ──
    panel = build_features(prices, vix_series, CONFIG)
    pred_panel = walk_forward_lgbm(panel, CONFIG)

    if pred_panel.empty:
        fprint("FATAL: No predictions. Cannot simulate strategies.")
        return

    # ── Part 2b: Run 6 strategy variants ──
    variants = {
        "A_bull_call": "Bull Call Spreads (standard)",
        "B_put_credit": "Put Credit Spreads (sell premium)",
        "C_iron_condor": "Iron Condors (range-bound)",
        "D_reduced": "Reduced Position Size (50%)",
        "E_cash": "Cash (skip blind spot)",
        "F_hedged": "Hedged Spreads (call spread + put)",
    }

    fprint(f"\n{'='*70}")
    fprint(f"STRATEGY VARIANTS — VIX 25-30 ZONE ONLY")
    fprint(f"{'='*70}")

    all_trades = {}
    all_metrics = {}

    for var_key, var_name in variants.items():
        trades = simulate_strategy_variant(var_key, pred_panel, prices, vix_series, CONFIG)
        all_trades[var_key] = trades
        metrics = compute_strategy_metrics(trades, CONFIG["initial_capital"], var_name)
        all_metrics[var_key] = metrics

    # Cash variant
    all_trades["E_cash"] = []
    all_metrics["E_cash"] = {
        "name": "Cash (skip blind spot)",
        "n_trades": 0,
        "win_rate": 0,
        "sharpe": 0,
        "sortino": 0,
        "cagr": 0,
        "max_dd": 0,
        "profit_factor": 0,
        "avg_trade_pnl": 0,
        "total_pnl": 0,
        "final_equity": CONFIG["initial_capital"],
        "years": 0,
        "trades_per_year": 0,
    }

    # ── Print comparison table ──
    fprint(f"\n{'='*70}")
    fprint(f"STRATEGY COMPARISON — VIX 25-30 ZONE")
    fprint(f"{'='*70}")
    fprint(f"{'Variant':<35} {'Trades':>6} {'WR':>6} {'Sharpe':>7} {'Sortino':>8} {'MDD':>7} {'PF':>6} {'Total$':>9}")
    fprint(f"{'-'*87}")

    for var_key in variants:
        m = all_metrics[var_key]
        if m.get("error"):
            fprint(f"{m['name']:<35} {'N/A':>6} {'':>6} {'':>7} {'':>8} {'':>7} {'':>6} {m.get('error',''):>9}")
        else:
            fprint(f"{m['name']:<35} {m['n_trades']:>6} {m['win_rate']*100:>5.1f}% "
                   f"{m['sharpe']:>7.2f} {m['sortino']:>8.2f} {m['max_dd']*100:>6.1f}% "
                   f"{m['profit_factor']:>6.2f} {m['total_pnl']:>8.2f}")

    # ── Part 3: Adversarial validation ──
    fprint(f"\n{'='*70}")
    fprint(f"PART 3: ADVERSARIAL VALIDATION")
    fprint(f"{'='*70}")

    spy_close = prices["SPY"]["close"]
    adv_results = {}

    for var_key, var_name in variants.items():
        trades = all_trades[var_key]
        if len(trades) < 5:
            fprint(f"\n  {var_name}: Skipped (< 5 trades)")
            continue

        fprint(f"\n  --- {var_name} ---")
        result = run_adversarial(trades, CONFIG["initial_capital"], spy_close, var_name)
        if result is not None:
            adv_results[var_key] = result
            result.print_summary()

    # ── MLflow Logging ──
    if MLFLOW_OK:
        fprint(f"\n{'='*70}")
        fprint(f"LOGGING TO MLFLOW")
        fprint(f"{'='*70}")
        try:
            mlflow.set_experiment("vix-transition-zone-v1")
            with mlflow.start_run(run_name=f"vix_25_30_research_{datetime.now().strftime('%Y%m%d_%H%M')}"):
                # Log descriptive results
                for key, val in descriptive.items():
                    mlflow.log_metric(f"desc_{key}", val)

                # Log each variant
                for var_key in variants:
                    m = all_metrics[var_key]
                    if not m.get("error"):
                        prefix = f"v_{var_key}_"
                        mlflow.log_metric(f"{prefix}sharpe", m["sharpe"])
                        mlflow.log_metric(f"{prefix}sortino", m["sortino"])
                        mlflow.log_metric(f"{prefix}max_dd", m["max_dd"])
                        mlflow.log_metric(f"{prefix}win_rate", m["win_rate"])
                        mlflow.log_metric(f"{prefix}profit_factor", m["profit_factor"])
                        mlflow.log_metric(f"{prefix}n_trades", m["n_trades"])
                        mlflow.log_metric(f"{prefix}total_pnl", m["total_pnl"])
                        mlflow.log_metric(f"{prefix}cagr", m["cagr"])

                    # Adversarial gate results
                    if var_key in adv_results and adv_results[var_key] is not None:
                        ar = adv_results[var_key]
                        mlflow.log_metric(f"{prefix}adv_gates_passed",
                                          sum(1 for g in ar.gates if g.passed))
                        mlflow.log_metric(f"{prefix}adv_all_passed", int(ar.all_passed))

                # Log config
                mlflow.log_params({
                    "vix_low": CONFIG["vix_low"],
                    "vix_high": CONFIG["vix_high"],
                    "initial_capital": CONFIG["initial_capital"],
                    "target_dte": CONFIG["target_dte"],
                    "spread_width_pct": CONFIG["spread_width_pct"],
                    "top_n": CONFIG["top_n"],
                    "train_months": CONFIG["train_months"],
                    "n_perms": CONFIG["n_perms"],
                })

                fprint("  MLflow run logged successfully")
        except Exception as e:
            fprint(f"  MLflow logging error: {e}")

    # ── Save results ──
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {k: v for k, v in CONFIG.items() if not isinstance(v, dict)},
        "descriptive": descriptive,
        "strategy_metrics": {k: v for k, v in all_metrics.items()},
        "adversarial_summary": {},
    }

    for var_key, ar in adv_results.items():
        if ar is not None:
            results["adversarial_summary"][var_key] = {
                "all_passed": ar.all_passed,
                "gates_passed": sum(1 for g in ar.gates if g.passed),
                "total_gates": len(ar.gates),
                "sharpe": ar.sharpe,
                "sortino": ar.sortino,
                "max_dd": ar.max_dd,
                "win_rate": ar.win_rate,
            }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    fprint(f"\n  Results saved to {results_path}")

    # ── Final Summary ──
    elapsed = (datetime.now() - start_time).total_seconds()
    fprint(f"\n{'='*70}")
    fprint(f"RESEARCH COMPLETE — {elapsed/60:.1f} minutes")
    fprint(f"{'='*70}")

    # Find best variant
    valid_metrics = {k: v for k, v in all_metrics.items() if not v.get("error") and v["n_trades"] > 0}
    if valid_metrics:
        best_key = max(valid_metrics, key=lambda k: valid_metrics[k]["sharpe"])
        best = valid_metrics[best_key]
        fprint(f"\n  BEST VARIANT: {best['name']}")
        fprint(f"    Sharpe: {best['sharpe']:.2f}, Sortino: {best['sortino']:.2f}")
        fprint(f"    WR: {best['win_rate']*100:.1f}%, MDD: {best['max_dd']*100:.1f}%")
        fprint(f"    PF: {best['profit_factor']:.2f}, Total PnL: ${best['total_pnl']:.2f}")

        if best_key in adv_results and adv_results[best_key] is not None:
            ar = adv_results[best_key]
            fprint(f"    Adversarial: {sum(1 for g in ar.gates if g.passed)}/{len(ar.gates)} gates passed")
    else:
        fprint("\n  No valid variants found. VIX 25-30 may be a genuine no-trade zone.")

    fprint(f"\n  RECOMMENDATION:")
    if valid_metrics:
        best = valid_metrics[best_key]
        if best["sharpe"] >= 1.5 and best.get("profit_factor", 0) >= 1.3:
            fprint(f"    USE {best['name']} in VIX 25-30 zone.")
        elif best["sharpe"] >= 0.8:
            fprint(f"    CAUTIOUSLY use {best['name']} with reduced size in VIX 25-30.")
        else:
            fprint(f"    AVOID trading in VIX 25-30. The blind spot is real. Go to cash.")
    else:
        fprint(f"    AVOID trading in VIX 25-30. The blind spot is real. Go to cash.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        fprint(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
