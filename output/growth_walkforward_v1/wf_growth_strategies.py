#!/usr/bin/env python3
"""
Walk-Forward Predictive Tests for Growth Strategy Candidates
HC #697: NO CRYPTO, WALK-FORWARD MANDATORY
HC #0: SLIDING window only (never expanding)
HC #694: Commission-free (Robinhood)

Strategies tested:
1. Vol Harvesting (SVXY via VIX term structure)
2. TQQQ Trend Following (MA crossover signals)
3. Leveraged ETF Rotation (sector momentum)
4. Options Collar on QQQ (hedged beta test)
5. Factor Timing (value/momentum/quality rotation)

Each strategy uses SLIDING walk-forward:
- Train on lookback window
- Predict 1-5 day forward returns
- Slide forward, drop oldest
- Measure OOT: directional accuracy, IC, Sharpe, Sortino
- Compare vs buy-and-hold
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
import json
import os
import sys

OUTPUT_DIR = "/home/jupiter/Lvl3Quant/output/growth_walkforward_v1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Data Download ───────────────────────────────────────────────────────────

def download_data(tickers, start="2015-01-01", end=None):
    """Download daily OHLCV data for tickers."""
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    data = {}
    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
            if len(df) > 100:
                data[t] = df
                print(f"  {t}: {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")
            else:
                print(f"  {t}: insufficient data ({len(df)} days)")
        except Exception as e:
            print(f"  {t}: download failed - {e}")
    return data


# ─── Metrics ─────────────────────────────────────────────────────────────────

def calc_metrics(returns, name="Strategy"):
    """Calculate risk-adjusted metrics."""
    if len(returns) < 20:
        return {"name": name, "n_days": len(returns), "error": "insufficient data"}

    ann = 252
    mean_r = returns.mean()
    std_r = returns.std()
    downside = returns[returns < 0].std()

    sharpe = (mean_r / std_r * np.sqrt(ann)) if std_r > 0 else 0
    sortino = (mean_r / downside * np.sqrt(ann)) if downside > 0 else 0
    total_ret = (1 + returns).prod() - 1
    cagr = (1 + total_ret) ** (ann / len(returns)) - 1 if len(returns) > 0 else 0
    max_dd = (returns.cumsum() - returns.cumsum().cummax()).min()
    win_rate = (returns > 0).mean()
    pf = returns[returns > 0].sum() / abs(returns[returns < 0].sum()) if (returns < 0).any() else np.inf

    return {
        "name": name,
        "n_days": len(returns),
        "total_return": round(total_ret * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd_pct": round(max_dd * 100, 2),
        "win_rate": round(win_rate * 100, 1),
        "profit_factor": round(pf, 2) if pf != np.inf else "inf",
        "avg_daily_ret_bps": round(mean_r * 10000, 2),
    }


def information_coefficient(signal, forward_ret):
    """Rank IC between signal and forward returns."""
    mask = ~(np.isnan(signal) | np.isnan(forward_ret))
    if mask.sum() < 20:
        return np.nan
    ic, _ = stats.spearmanr(signal[mask], forward_ret[mask])
    return ic


# ─── Strategy 1: Vol Harvesting (SVXY Term Structure) ───────────────────────

def test_vol_harvesting(data):
    """
    Signal: VIX term structure ratio (VIX/VIX3M or VIX vs its own MA).
    When VIX is below long-term avg AND in contango → hold SVXY (short vol).
    When backwardation → go to cash.

    Walk-forward: 126-day train, 5-day test, sliding.
    """
    print("\n" + "="*70)
    print("STRATEGY 1: VOL HARVESTING (SVXY Term Structure Timing)")
    print("="*70)

    vix = data.get("^VIX")
    svxy = data.get("SVXY")

    if vix is None or svxy is None:
        return {"error": "Missing VIX or SVXY data"}

    # Align dates
    common = vix.index.intersection(svxy.index)
    vix_close = vix.loc[common, "Close"].squeeze()
    svxy_close = svxy.loc[common, "Close"].squeeze()
    svxy_ret = svxy_close.pct_change()

    # Signal features: VIX level relative to its moving averages
    train_window = 126  # 6 months
    test_window = 5

    signals = []
    actuals = []
    dates = []

    idx_list = common.tolist()

    for i in range(train_window, len(idx_list) - test_window, test_window):
        train_slice = slice(max(0, i - train_window), i)
        test_start = i
        test_end = min(i + test_window, len(idx_list))

        # Train: compute VIX percentile in training window
        vix_train = vix_close.iloc[train_slice]
        vix_median = vix_train.median()
        vix_p75 = vix_train.quantile(0.75)

        # Signal: VIX relative to training median + trend
        # Low VIX + declining = contango likely = hold SVXY
        # High VIX + rising = backwardation likely = cash
        for j in range(test_start, test_end):
            if j >= len(idx_list):
                break
            current_vix = vix_close.iloc[j]
            vix_ma20 = vix_close.iloc[max(0, j-20):j].mean()

            # Signal: negative = bearish vol (good for SVXY), positive = bullish vol
            sig = -(vix_median - current_vix) / vix_median  # below median = negative signal
            vix_trend = (current_vix - vix_ma20) / vix_ma20 if vix_ma20 > 0 else 0
            composite_signal = sig + vix_trend  # combined: low VIX + declining = most negative

            # Forward return
            if j + 1 < len(idx_list):
                fwd = svxy_ret.iloc[j + 1] if j + 1 < len(svxy_ret) else np.nan
                signals.append(composite_signal)
                actuals.append(fwd)
                dates.append(idx_list[j])

    signals = np.array(signals)
    actuals = np.array(actuals)

    # Strategy: hold SVXY when signal < 0 (low VIX, declining), cash otherwise
    positions = (signals < 0).astype(float)
    strategy_rets = pd.Series(positions * actuals, index=dates[:len(actuals)])
    bh_rets = pd.Series(actuals, index=dates[:len(actuals)])

    # Drop NaN
    strategy_rets = strategy_rets.dropna()
    bh_rets = bh_rets.dropna()

    ic = information_coefficient(signals, actuals)
    dir_acc = np.mean((np.sign(signals) == np.sign(actuals))[~np.isnan(actuals)])

    strat_metrics = calc_metrics(strategy_rets, "Vol Harvest (Timed)")
    bh_metrics = calc_metrics(bh_rets, "SVXY Buy & Hold")

    strat_metrics["IC"] = round(ic, 4) if not np.isnan(ic) else "N/A"
    strat_metrics["dir_accuracy"] = round(dir_acc * 100, 1)
    strat_metrics["signal_adds_value"] = strat_metrics["sharpe"] > bh_metrics["sharpe"]

    print(f"\n  OOT Signal IC: {ic:.4f}")
    print(f"  Directional Accuracy: {dir_acc*100:.1f}%")
    print(f"  Strategy Sharpe: {strat_metrics['sharpe']:.3f} vs B&H Sharpe: {bh_metrics['sharpe']:.3f}")
    print(f"  Strategy Sortino: {strat_metrics['sortino']:.3f} vs B&H Sortino: {bh_metrics['sortino']:.3f}")
    print(f"  Strategy MaxDD: {strat_metrics['max_dd_pct']:.1f}% vs B&H MaxDD: {bh_metrics['max_dd_pct']:.1f}%")
    print(f"  SIGNAL ADDS VALUE: {strat_metrics['signal_adds_value']}")

    return {"strategy": strat_metrics, "benchmark": bh_metrics}


# ─── Strategy 2: TQQQ Trend Following ───────────────────────────────────────

def test_tqqq_trend(data):
    """
    Signal: Moving average crossover on QQQ (not TQQQ - avoid leveraged decay in signal).
    Use QQQ 20/50 EMA crossover to predict forward TQQQ returns.
    Walk-forward: 252-day train, 5-day test, sliding.
    """
    print("\n" + "="*70)
    print("STRATEGY 2: TQQQ TREND FOLLOWING (MA Crossover on QQQ)")
    print("="*70)

    qqq = data.get("QQQ")
    tqqq = data.get("TQQQ")

    if qqq is None or tqqq is None:
        return {"error": "Missing QQQ or TQQQ data"}

    common = qqq.index.intersection(tqqq.index)
    qqq_close = qqq.loc[common, "Close"].squeeze()
    tqqq_close = tqqq.loc[common, "Close"].squeeze()
    tqqq_ret = tqqq_close.pct_change()

    train_window = 252
    test_window = 5

    signals = []
    actuals = []
    dates = []

    idx_list = common.tolist()

    for i in range(train_window, len(idx_list) - 1, test_window):
        # Compute EMAs on training + current data
        lookback_start = max(0, i - train_window - 50)  # extra for EMA warmup
        qqq_slice = qqq_close.iloc[lookback_start:i+1]

        ema20 = qqq_slice.ewm(span=20).mean()
        ema50 = qqq_slice.ewm(span=50).mean()
        ema200 = qqq_slice.ewm(span=200).mean() if len(qqq_slice) >= 200 else qqq_slice.ewm(span=len(qqq_slice)).mean()

        # Train: optimize threshold. What crossover strength predicts returns?
        train_start = max(0, i - train_window)

        for j in range(i, min(i + test_window, len(idx_list) - 1)):
            # Recompute for each day
            qqq_to_j = qqq_close.iloc[lookback_start:j+1]
            e20 = qqq_to_j.ewm(span=20).mean().iloc[-1]
            e50 = qqq_to_j.ewm(span=50).mean().iloc[-1]

            # Momentum signal
            price = qqq_close.iloc[j]
            mom_20d = (price / qqq_close.iloc[max(0, j-20)] - 1) if j >= 20 else 0

            # Signal: EMA crossover strength + momentum
            cross = (e20 - e50) / e50  # positive = bullish
            sig = cross + mom_20d * 0.5  # combined signal

            # Forward 1-day TQQQ return
            if j + 1 < len(idx_list):
                fwd = tqqq_ret.iloc[j + 1] if j + 1 < len(tqqq_ret) else np.nan
                signals.append(sig)
                actuals.append(fwd)
                dates.append(idx_list[j])

    signals = np.array(signals)
    actuals = np.array(actuals)

    # Strategy: hold TQQQ when signal > 0 (uptrend), cash otherwise
    positions = (signals > 0).astype(float)
    strategy_rets = pd.Series(positions * actuals, index=dates[:len(actuals)])
    bh_rets = pd.Series(actuals, index=dates[:len(actuals)])

    strategy_rets = strategy_rets.dropna()
    bh_rets = bh_rets.dropna()

    ic = information_coefficient(signals, actuals)
    dir_acc = np.mean((np.sign(signals) == np.sign(actuals))[~np.isnan(actuals)])

    strat_metrics = calc_metrics(strategy_rets, "TQQQ Trend (Timed)")
    bh_metrics = calc_metrics(bh_rets, "TQQQ Buy & Hold")

    strat_metrics["IC"] = round(ic, 4) if not np.isnan(ic) else "N/A"
    strat_metrics["dir_accuracy"] = round(dir_acc * 100, 1)
    strat_metrics["signal_adds_value"] = strat_metrics["sharpe"] > bh_metrics["sharpe"]

    # Also test: how much time in market?
    pct_invested = positions.mean() * 100

    print(f"\n  OOT Signal IC: {ic:.4f}")
    print(f"  Directional Accuracy: {dir_acc*100:.1f}%")
    print(f"  Time Invested: {pct_invested:.1f}%")
    print(f"  Strategy Sharpe: {strat_metrics['sharpe']:.3f} vs B&H Sharpe: {bh_metrics['sharpe']:.3f}")
    print(f"  Strategy Sortino: {strat_metrics['sortino']:.3f} vs B&H Sortino: {bh_metrics['sortino']:.3f}")
    print(f"  Strategy MaxDD: {strat_metrics['max_dd_pct']:.1f}% vs B&H MaxDD: {bh_metrics['max_dd_pct']:.1f}%")
    print(f"  SIGNAL ADDS VALUE: {strat_metrics['signal_adds_value']}")

    strat_metrics["pct_time_invested"] = round(pct_invested, 1)

    return {"strategy": strat_metrics, "benchmark": bh_metrics}


# ─── Strategy 3: Leveraged ETF Rotation ─────────────────────────────────────

def test_leveraged_rotation(data):
    """
    Signal: 20-day momentum ranking across sector leveraged ETFs.
    Each period, rank by trailing momentum, hold top 2, rebalance weekly.
    Walk-forward: 126-day train, 5-day test.
    """
    print("\n" + "="*70)
    print("STRATEGY 3: LEVERAGED ETF ROTATION (Momentum)")
    print("="*70)

    etf_tickers = ["TQQQ", "UPRO", "SOXL", "TNA", "TECL", "FAS"]
    available = {t: data[t] for t in etf_tickers if t in data}

    if len(available) < 3:
        return {"error": f"Need >=3 ETFs, got {len(available)}"}

    # Align all to common dates
    common_idx = None
    for t, df in available.items():
        if common_idx is None:
            common_idx = df.index
        else:
            common_idx = common_idx.intersection(df.index)

    closes = pd.DataFrame({t: df.loc[common_idx, "Close"].squeeze() for t, df in available.items()})
    rets = closes.pct_change()

    train_window = 126
    test_window = 5
    top_n = 2

    strategy_daily_rets = []
    bh_daily_rets = []  # equal-weight buy & hold
    signal_list = []
    actual_list = []
    date_list = []

    for i in range(train_window, len(common_idx) - test_window, test_window):
        # Train: compute optimal lookback momentum
        # We'll use 20-day momentum (simple, robust)

        # Current momentum (signal)
        mom_20 = closes.iloc[i] / closes.iloc[max(0, i-20)] - 1
        mom_60 = closes.iloc[i] / closes.iloc[max(0, i-60)] - 1

        # Combined momentum signal
        combined_mom = 0.6 * mom_20 + 0.4 * mom_60

        # Rank and pick top_n
        ranked = combined_mom.sort_values(ascending=False)
        top_picks = ranked.index[:top_n].tolist()

        # Equal weight in top picks for test period
        for j in range(i, min(i + test_window, len(common_idx) - 1)):
            day_ret = rets.iloc[j + 1][top_picks].mean()
            bh_ret = rets.iloc[j + 1].mean()  # equal weight all

            strategy_daily_rets.append(day_ret)
            bh_daily_rets.append(bh_ret)

            # For IC: signal = avg momentum of picks, actual = avg fwd return of picks
            signal_list.append(combined_mom[top_picks].mean())
            actual_list.append(day_ret)
            date_list.append(common_idx[j])

    strat_rets = pd.Series(strategy_daily_rets, index=date_list[:len(strategy_daily_rets)]).dropna()
    bh_rets = pd.Series(bh_daily_rets, index=date_list[:len(bh_daily_rets)]).dropna()

    signals = np.array(signal_list)
    actuals = np.array(actual_list)
    ic = information_coefficient(signals, actuals)
    dir_acc = np.mean((np.sign(signals) == np.sign(actuals))[~np.isnan(actuals)])

    strat_metrics = calc_metrics(strat_rets, "Lev ETF Rotation")
    bh_metrics = calc_metrics(bh_rets, "Equal-Weight B&H")

    strat_metrics["IC"] = round(ic, 4) if not np.isnan(ic) else "N/A"
    strat_metrics["dir_accuracy"] = round(dir_acc * 100, 1)
    strat_metrics["signal_adds_value"] = strat_metrics["sharpe"] > bh_metrics["sharpe"]

    print(f"\n  ETFs used: {list(available.keys())}")
    print(f"  OOT Signal IC: {ic:.4f}")
    print(f"  Directional Accuracy: {dir_acc*100:.1f}%")
    print(f"  Strategy Sharpe: {strat_metrics['sharpe']:.3f} vs EW B&H Sharpe: {bh_metrics['sharpe']:.3f}")
    print(f"  Strategy Sortino: {strat_metrics['sortino']:.3f} vs EW B&H Sortino: {bh_metrics['sortino']:.3f}")
    print(f"  Strategy MaxDD: {strat_metrics['max_dd_pct']:.1f}% vs EW B&H MaxDD: {bh_metrics['max_dd_pct']:.1f}%")
    print(f"  SIGNAL ADDS VALUE: {strat_metrics['signal_adds_value']}")

    return {"strategy": strat_metrics, "benchmark": bh_metrics}


# ─── Strategy 4: Options Collar Test (QQQ Hedged Beta) ──────────────────────

def test_collar_proxy(data):
    """
    Since we can't easily backtest options with daily data, we PROXY the collar:
    - Collar ≈ capped upside + floored downside
    - Simulate: QQQ returns capped at +X% and floored at -Y% per month
    - Test if TIMING the collar (tightening in high-vol, loosening in low-vol) adds value

    Signal: VIX level predicts whether to tighten or loosen collar.
    Walk-forward: 63-day train, 5-day test.
    """
    print("\n" + "="*70)
    print("STRATEGY 4: OPTIONS COLLAR ON QQQ (Hedged Beta Test)")
    print("="*70)

    qqq = data.get("QQQ")
    vix = data.get("^VIX")

    if qqq is None or vix is None:
        return {"error": "Missing QQQ or VIX data"}

    common = qqq.index.intersection(vix.index)
    qqq_close = qqq.loc[common, "Close"].squeeze()
    vix_close = vix.loc[common, "Close"].squeeze()
    qqq_ret = qqq_close.pct_change()

    train_window = 126
    test_window = 5

    collar_rets = []
    static_collar_rets = []
    bh_rets_list = []
    signal_list = []
    actual_list = []
    date_list = []

    for i in range(train_window, len(common) - 1, test_window):
        # Train: learn VIX percentile
        vix_train = vix_close.iloc[max(0, i-train_window):i]
        vix_p25 = vix_train.quantile(0.25)
        vix_p75 = vix_train.quantile(0.75)

        for j in range(i, min(i + test_window, len(common) - 1)):
            current_vix = vix_close.iloc[j]
            fwd_ret = qqq_ret.iloc[j + 1] if j + 1 < len(qqq_ret) else np.nan

            if np.isnan(fwd_ret):
                continue

            # Adaptive collar based on VIX regime
            if current_vix > vix_p75:
                # High vol: tight collar (protect more, give up upside)
                cap = 0.01   # 1% daily cap
                floor = -0.008  # -0.8% floor (put protection)
            elif current_vix < vix_p25:
                # Low vol: loose collar (more upside, less protection)
                cap = 0.03
                floor = -0.02
            else:
                # Normal: moderate collar
                cap = 0.02
                floor = -0.015

            collar_ret = np.clip(fwd_ret, floor, cap)
            static_collar_ret = np.clip(fwd_ret, -0.015, 0.02)  # static collar

            collar_rets.append(collar_ret)
            static_collar_rets.append(static_collar_ret)
            bh_rets_list.append(fwd_ret)

            # Signal: VIX-based collar tightness predicts risk-adjusted return?
            sig = -(current_vix - vix_train.median()) / vix_train.std()
            signal_list.append(sig)
            actual_list.append(fwd_ret)
            date_list.append(common[j])

    collar_s = pd.Series(collar_rets, index=date_list[:len(collar_rets)]).dropna()
    static_s = pd.Series(static_collar_rets, index=date_list[:len(static_collar_rets)]).dropna()
    bh_s = pd.Series(bh_rets_list, index=date_list[:len(bh_rets_list)]).dropna()

    signals = np.array(signal_list)
    actuals = np.array(actual_list)
    ic = information_coefficient(signals, actuals)

    adaptive_metrics = calc_metrics(collar_s, "Adaptive Collar")
    static_metrics = calc_metrics(static_s, "Static Collar")
    bh_metrics = calc_metrics(bh_s, "QQQ Buy & Hold")

    # The key question: does the ADAPTIVE collar beat STATIC collar?
    # If not, there's no predictive element — it's just hedged beta
    adaptive_metrics["IC"] = round(ic, 4) if not np.isnan(ic) else "N/A"
    adaptive_metrics["beats_static"] = adaptive_metrics["sharpe"] > static_metrics["sharpe"]
    adaptive_metrics["is_just_hedged_beta"] = not adaptive_metrics["beats_static"]

    print(f"\n  OOT Signal IC (VIX → QQQ returns): {ic:.4f}")
    print(f"  Adaptive Collar Sharpe: {adaptive_metrics['sharpe']:.3f}")
    print(f"  Static Collar Sharpe: {static_metrics['sharpe']:.3f}")
    print(f"  QQQ B&H Sharpe: {bh_metrics['sharpe']:.3f}")
    print(f"  Adaptive beats Static: {adaptive_metrics['beats_static']}")
    print(f"  IS JUST HEDGED BETA: {adaptive_metrics['is_just_hedged_beta']}")

    return {"adaptive": adaptive_metrics, "static": static_metrics, "benchmark": bh_metrics}


# ─── Strategy 5: Factor Timing ──────────────────────────────────────────────

def test_factor_timing(data):
    """
    Signal: Can we predict which factor (value, momentum, quality, size) will outperform?
    Use factor ETF proxies and test if recent momentum predicts forward factor returns.

    Walk-forward: 252-day train, 5-day test.
    """
    print("\n" + "="*70)
    print("STRATEGY 5: FACTOR TIMING (Value/Momentum/Quality Rotation)")
    print("="*70)

    # Factor ETF proxies
    factor_map = {
        "VLUE": "Value",
        "MTUM": "Momentum",
        "QUAL": "Quality",
        "SIZE": "Size",
    }

    available = {t: data[t] for t in factor_map if t in data}
    spy = data.get("SPY")

    if len(available) < 3 or spy is None:
        return {"error": f"Need >=3 factor ETFs + SPY, got {len(available)} ETFs"}

    common_idx = spy.index
    for t, df in available.items():
        common_idx = common_idx.intersection(df.index)

    closes = pd.DataFrame({t: df.loc[common_idx, "Close"].squeeze() for t, df in available.items()})
    spy_close = spy.loc[common_idx, "Close"].squeeze()

    # Excess returns vs SPY
    rets = closes.pct_change()
    spy_ret = spy_close.pct_change()
    excess_rets = rets.subtract(spy_ret, axis=0)

    train_window = 252
    test_window = 5
    top_n = 2

    strat_rets_list = []
    bh_rets_list = []
    signal_list = []
    actual_list = []
    date_list = []

    for i in range(train_window, len(common_idx) - test_window, test_window):
        # Signal: combined short + medium term momentum of excess returns
        mom_20 = excess_rets.iloc[max(0,i-20):i].sum()
        mom_60 = excess_rets.iloc[max(0,i-60):i].sum()

        # Mean reversion component (factors tend to mean-revert at longer horizons)
        mom_252 = excess_rets.iloc[max(0,i-252):i].sum()

        # Combined: short momentum + long mean-reversion
        signal = 0.5 * mom_20 + 0.3 * mom_60 - 0.2 * mom_252  # negative weight = mean reversion

        ranked = signal.sort_values(ascending=False)
        top_picks = ranked.index[:top_n].tolist()

        for j in range(i, min(i + test_window, len(common_idx) - 1)):
            # Hold top factor ETFs (absolute, not excess)
            day_ret = rets.iloc[j + 1][top_picks].mean()
            bh_ret = spy_ret.iloc[j + 1]  # SPY as benchmark
            ew_ret = rets.iloc[j + 1].mean()  # equal weight all factors

            strat_rets_list.append(day_ret)
            bh_rets_list.append(ew_ret)  # compare vs equal weight factors

            signal_list.append(signal[top_picks].mean())
            actual_list.append(excess_rets.iloc[j + 1][top_picks].mean() if j + 1 < len(excess_rets) else np.nan)
            date_list.append(common_idx[j])

    strat_s = pd.Series(strat_rets_list, index=date_list[:len(strat_rets_list)]).dropna()
    bh_s = pd.Series(bh_rets_list, index=date_list[:len(bh_rets_list)]).dropna()

    signals = np.array(signal_list)
    actuals = np.array(actual_list)
    ic = information_coefficient(signals, actuals)
    dir_acc = np.mean((np.sign(signals) == np.sign(actuals))[~np.isnan(actuals)]) if len(actuals) > 0 else 0

    strat_metrics = calc_metrics(strat_s, "Factor Timing (Top 2)")
    bh_metrics = calc_metrics(bh_s, "Equal-Weight Factors")

    strat_metrics["IC"] = round(ic, 4) if not np.isnan(ic) else "N/A"
    strat_metrics["dir_accuracy"] = round(dir_acc * 100, 1)
    strat_metrics["signal_adds_value"] = strat_metrics["sharpe"] > bh_metrics["sharpe"]

    print(f"\n  Factor ETFs used: {[f'{t} ({factor_map[t]})' for t in available]}")
    print(f"  OOT Signal IC (momentum → excess return): {ic:.4f}")
    print(f"  Directional Accuracy: {dir_acc*100:.1f}%")
    print(f"  Strategy Sharpe: {strat_metrics['sharpe']:.3f} vs EW Factors Sharpe: {bh_metrics['sharpe']:.3f}")
    print(f"  Strategy Sortino: {strat_metrics['sortino']:.3f} vs EW Factors Sortino: {bh_metrics['sortino']:.3f}")
    print(f"  SIGNAL ADDS VALUE: {strat_metrics['signal_adds_value']}")

    return {"strategy": strat_metrics, "benchmark": bh_metrics}


# ─── BONUS: Enhanced TQQQ with Volatility Filter ────────────────────────────

def test_tqqq_vol_filtered(data):
    """
    Enhancement of Strategy 2: Add VIX filter to trend signal.
    Only hold TQQQ when trend is UP AND VIX is below threshold.
    """
    print("\n" + "="*70)
    print("STRATEGY 2B: TQQQ TREND + VOL FILTER (Enhanced)")
    print("="*70)

    qqq = data.get("QQQ")
    tqqq = data.get("TQQQ")
    vix = data.get("^VIX")

    if qqq is None or tqqq is None or vix is None:
        return {"error": "Missing data"}

    common = qqq.index.intersection(tqqq.index).intersection(vix.index)
    qqq_close = qqq.loc[common, "Close"].squeeze()
    tqqq_close = tqqq.loc[common, "Close"].squeeze()
    vix_close = vix.loc[common, "Close"].squeeze()
    tqqq_ret = tqqq_close.pct_change()

    train_window = 252
    test_window = 5

    signals = []
    actuals = []
    dates = []

    idx_list = common.tolist()

    for i in range(train_window, len(idx_list) - 1, test_window):
        lookback_start = max(0, i - train_window - 50)

        # Train: compute VIX threshold from training data
        vix_train = vix_close.iloc[max(0, i-train_window):i]
        vix_threshold = vix_train.quantile(0.70)  # top 30% = danger zone

        for j in range(i, min(i + test_window, len(idx_list) - 1)):
            qqq_to_j = qqq_close.iloc[lookback_start:j+1]
            e20 = qqq_to_j.ewm(span=20).mean().iloc[-1]
            e50 = qqq_to_j.ewm(span=50).mean().iloc[-1]

            price = qqq_close.iloc[j]
            mom_20d = (price / qqq_close.iloc[max(0, j-20)] - 1) if j >= 20 else 0
            current_vix = vix_close.iloc[j]

            # Enhanced signal: trend + vol filter
            trend_sig = (e20 - e50) / e50 + mom_20d * 0.5
            vol_penalty = 1.0 if current_vix < vix_threshold else -0.5  # penalize high VIX

            sig = trend_sig * (1 if vol_penalty > 0 else 0.3)  # reduce signal in high vol

            if j + 1 < len(idx_list):
                fwd = tqqq_ret.iloc[j + 1] if j + 1 < len(tqqq_ret) else np.nan
                signals.append(sig)
                actuals.append(fwd)
                dates.append(idx_list[j])

    signals = np.array(signals)
    actuals = np.array(actuals)

    positions = (signals > 0).astype(float)
    strategy_rets = pd.Series(positions * actuals, index=dates[:len(actuals)]).dropna()
    bh_rets = pd.Series(actuals, index=dates[:len(actuals)]).dropna()

    ic = information_coefficient(signals, actuals)

    strat_metrics = calc_metrics(strategy_rets, "TQQQ Trend+Vol Filter")
    bh_metrics = calc_metrics(bh_rets, "TQQQ Buy & Hold")

    strat_metrics["IC"] = round(ic, 4) if not np.isnan(ic) else "N/A"
    strat_metrics["pct_time_invested"] = round(positions.mean() * 100, 1)
    strat_metrics["signal_adds_value"] = strat_metrics["sharpe"] > bh_metrics["sharpe"]

    print(f"\n  OOT Signal IC: {ic:.4f}")
    print(f"  Time Invested: {strat_metrics['pct_time_invested']:.1f}%")
    print(f"  Strategy Sharpe: {strat_metrics['sharpe']:.3f} vs B&H Sharpe: {bh_metrics['sharpe']:.3f}")
    print(f"  Strategy Sortino: {strat_metrics['sortino']:.3f} vs B&H Sortino: {bh_metrics['sortino']:.3f}")
    print(f"  Strategy MaxDD: {strat_metrics['max_dd_pct']:.1f}% vs B&H MaxDD: {bh_metrics['max_dd_pct']:.1f}%")
    print(f"  SIGNAL ADDS VALUE: {strat_metrics['signal_adds_value']}")

    return {"strategy": strat_metrics, "benchmark": bh_metrics}


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("WALK-FORWARD PREDICTIVE TESTS — GROWTH STRATEGY CANDIDATES")
    print("HC #697: No Crypto | HC #0: Sliding Window | HC #694: Commission-Free")
    print(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Download all needed data
    print("\n--- Downloading Data ---")
    all_tickers = [
        "^VIX", "SVXY",           # Vol harvesting
        "QQQ", "TQQQ",            # Trend following
        "UPRO", "SOXL", "TNA", "TECL", "FAS",  # Lev ETF rotation
        "SPY",                     # Benchmark
        "VLUE", "MTUM", "QUAL", "SIZE",  # Factor ETFs
    ]

    data = download_data(all_tickers, start="2015-01-01")

    # Run all strategies
    results = {}

    results["1_vol_harvesting"] = test_vol_harvesting(data)
    results["2_tqqq_trend"] = test_tqqq_trend(data)
    results["2b_tqqq_vol_filtered"] = test_tqqq_vol_filtered(data)
    results["3_lev_rotation"] = test_leveraged_rotation(data)
    results["4_collar_qqq"] = test_collar_proxy(data)
    results["5_factor_timing"] = test_factor_timing(data)

    # ─── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL SUMMARY — WALK-FORWARD PREDICTIVE EDGE ASSESSMENT")
    print("=" * 70)

    summary_rows = []

    for key, res in results.items():
        if "error" in res:
            print(f"  {key}: ERROR - {res['error']}")
            continue

        # Get strategy metrics
        if "strategy" in res:
            s = res["strategy"]
            b = res["benchmark"]
        elif "adaptive" in res:
            s = res["adaptive"]
            b = res["benchmark"]
        else:
            continue

        ic = s.get("IC", "N/A")
        sharpe = s.get("sharpe", 0)
        sortino = s.get("sortino", 0)
        b_sharpe = b.get("sharpe", 0)
        adds_value = s.get("signal_adds_value", s.get("beats_static", False))
        dir_acc = s.get("dir_accuracy", "N/A")

        verdict = "PASS" if adds_value else "FAIL"

        row = {
            "strategy": s["name"],
            "OOT_IC": ic,
            "dir_accuracy": dir_acc,
            "sharpe": sharpe,
            "sortino": sortino,
            "bh_sharpe": b_sharpe,
            "max_dd": s.get("max_dd_pct", "N/A"),
            "total_return": s.get("total_return", "N/A"),
            "verdict": verdict,
        }
        summary_rows.append(row)

        print(f"\n  {s['name']}:")
        print(f"    IC={ic} | DirAcc={dir_acc}% | Sharpe={sharpe} (vs B&H {b_sharpe})")
        print(f"    Sortino={sortino} | MaxDD={s.get('max_dd_pct','N/A')}% | Return={s.get('total_return','N/A')}%")
        print(f"    VERDICT: {'✓ PREDICTIVE EDGE' if adds_value else '✗ NO PREDICTIVE EDGE'}")

    # Save results
    output = {
        "run_date": datetime.now().isoformat(),
        "config": {
            "sliding_window": True,
            "no_crypto": True,
            "commission_free": True,
        },
        "detailed_results": {},
        "summary": summary_rows,
    }

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp,)):
            return obj.isoformat()
        return obj

    for key, res in results.items():
        clean = {}
        for k, v in res.items():
            if isinstance(v, dict):
                clean[k] = {kk: convert(vv) for kk, vv in v.items()}
            else:
                clean[k] = convert(v)
        output["detailed_results"][key] = clean

    with open(os.path.join(OUTPUT_DIR, "walkforward_results.json"), "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n\nResults saved to {OUTPUT_DIR}/walkforward_results.json")

    # Final recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    passed = [r for r in summary_rows if r["verdict"] == "PASS"]
    failed = [r for r in summary_rows if r["verdict"] == "FAIL"]

    if passed:
        print("\n  STRATEGIES WITH PREDICTIVE EDGE (proceed to deeper research):")
        for p in passed:
            print(f"    → {p['strategy']} (Sharpe={p['sharpe']}, IC={p['OOT_IC']})")

    if failed:
        print("\n  STRATEGIES WITHOUT PREDICTIVE EDGE (drop or rethink signal):")
        for f_ in failed:
            print(f"    ✗ {f_['strategy']} (Sharpe={f_['sharpe']} vs B&H {f_['bh_sharpe']})")

    return output


if __name__ == "__main__":
    results = main()
