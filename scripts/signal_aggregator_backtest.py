#!/usr/bin/env python3
"""
Multi-Signal Aggregator Backtest
================================
Combines 5 validated signals into a composite scoring system (0-5).
Tests 6 variants with full 5-gate validation framework.

Signals:
  1. REGIME: SPY > 200-SMA
  2. VIX CALM: VIX < 20 or declining from >25
  3. MOMENTUM: QQQ 20d return > 0
  4. VOLUME: Any sector ETF with 5+ days above-avg volume
  5. BREADTH: SPY outperforming RSP (proxy for breadth)

Variants A-F with different allocation rules.
OOT: Jan 2022 - Jul 2026. Capital: $645. Slippage: 0.02%.
"""

import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────
CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
OOT_START = "2022-01-01"
OOT_END = "2026-07-28"
N_PERMS = 1000
SEED = 42

TICKERS = ["SPY", "QQQ", "RSP", "TLT", "XLK", "XLC", "XLY", "XLE", "XLF", "XLV"]
VIX_TICKER = "^VIX"
SECTOR_ETFS = ["XLK", "XLC", "XLY", "XLE", "XLF", "XLV"]

# Validation gates
SHARPE_MIN = 0.5
PERM_P_MAX = 0.05
REGIME_GAP_MAX = 0.5
MDD_FLOOR = -0.50  # MDD must be > -50%
MIN_TRADES = 20


def download_data():
    """Download all required price data."""
    all_tickers = TICKERS + [VIX_TICKER]
    # Download with buffer for 200-SMA calculation
    start = (pd.Timestamp(OOT_START) - pd.DateOffset(days=300)).strftime("%Y-%m-%d")

    print(f"Downloading data for {len(all_tickers)} tickers from {start}...")
    data = yf.download(all_tickers, start=start, end=OOT_END, auto_adjust=True, progress=False)

    close = data["Close"].copy()
    volume = data["Volume"].copy()

    # Drop rows where SPY is NaN
    close = close.dropna(subset=["SPY"])
    volume = volume.loc[close.index]

    print(f"Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} trading days")
    return close, volume


def compute_signals(close, volume):
    """Compute all 5 signals daily. Returns DataFrame of scores."""
    idx = close.index
    signals = pd.DataFrame(index=idx, columns=[
        "regime", "vix_calm", "momentum", "volume_surge", "breadth"
    ], dtype=float)

    # Signal 1: REGIME - SPY > 200-SMA
    spy_sma200 = close["SPY"].rolling(200).mean()
    signals["regime"] = (close["SPY"] > spy_sma200).astype(float)

    # Signal 2: VIX CALM
    vix = close[VIX_TICKER] if VIX_TICKER in close.columns else close.get("^VIX")
    if vix is None:
        # Fallback
        signals["vix_calm"] = 0.0
    else:
        vix_below_20 = vix < 20
        # VIX declining from >25: was >25 in last 10 days AND current < 5-day ago
        vix_was_high = vix.rolling(10).max() > 25
        vix_declining = vix < vix.shift(5)
        vix_fade = vix_was_high & vix_declining
        signals["vix_calm"] = (vix_below_20 | vix_fade).astype(float)

    # Signal 3: MOMENTUM - QQQ 20d return > 0
    qqq_ret_20d = close["QQQ"].pct_change(20)
    signals["momentum"] = (qqq_ret_20d > 0).astype(float)

    # Signal 4: VOLUME - any sector ETF with 5+ consecutive days above avg volume
    vol_20d_avg = volume[SECTOR_ETFS].rolling(20).mean()
    above_avg = volume[SECTOR_ETFS] > vol_20d_avg

    # Count consecutive days above average for each sector (vectorized via numpy)
    any_streak = pd.DataFrame(index=idx, columns=SECTOR_ETFS, dtype=int)
    for etf in SECTOR_ETFS:
        arr = above_avg[etf].values.astype(int)
        result = np.zeros(len(arr), dtype=int)
        for i in range(len(arr)):
            if arr[i]:
                result[i] = result[i-1] + 1 if i > 0 else 1
        any_streak[etf] = result

    signals["volume_surge"] = (any_streak.max(axis=1) >= 5).astype(float)

    # Signal 5: BREADTH - SPY vs RSP (equal-weight)
    # If SPY outperforms RSP over 20d, large-caps leading = breadth concern
    # Actually: if RSP outperforms SPY, broader participation = good breadth
    # Proxy: RSP 20d return > SPY 20d return → broad participation → +1
    spy_ret_20d = close["SPY"].pct_change(20)
    rsp_ret_20d = close["RSP"].pct_change(20)
    # Alternative: both positive and RSP keeping up → healthy breadth
    signals["breadth"] = ((spy_ret_20d > 0) & (rsp_ret_20d > spy_ret_20d * 0.5)).astype(float)

    # Composite score
    signals["score"] = signals[["regime", "vix_calm", "momentum", "volume_surge", "breadth"]].sum(axis=1)

    return signals


def apply_slippage(returns, trades_mask):
    """Apply slippage cost on trade days."""
    adj = returns.copy()
    adj[trades_mask] -= SLIPPAGE_PCT
    return adj


def identify_regime(close):
    """Classify each day as bull/bear based on SPY 200-SMA."""
    sma200 = close["SPY"].rolling(200).mean()
    regime = pd.Series("bull", index=close.index)
    regime[close["SPY"] < sma200] = "bear"
    return regime


def calc_metrics(equity_curve, daily_returns, regime, trades_count):
    """Calculate performance metrics."""
    dr = daily_returns.dropna()
    if len(dr) < 10:
        return None

    ann_ret = dr.mean() * 252
    ann_vol = dr.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = dr[dr < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside > 0 else 0

    # Max drawdown
    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak
    mdd = dd.min()

    # Profit factor
    gains = dr[dr > 0].sum()
    losses = abs(dr[dr < 0].sum())
    pf = gains / losses if losses > 0 else float("inf")

    # Win rate
    wr = (dr[dr != 0] > 0).mean() if (dr != 0).any() else 0

    # Regime-stratified Sharpe
    bull_days = regime == "bull"
    bear_days = regime == "bear"

    bull_ret = dr[bull_days]
    bear_ret = dr[bear_days]

    bull_sharpe = (bull_ret.mean() * 252) / (bull_ret.std() * np.sqrt(252)) if len(bull_ret) > 10 and bull_ret.std() > 0 else 0
    bear_sharpe = (bear_ret.mean() * 252) / (bear_ret.std() * np.sqrt(252)) if len(bear_ret) > 10 and bear_ret.std() > 0 else 0

    max_sharpe = max(abs(bull_sharpe), abs(bear_sharpe))
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_sharpe if max_sharpe > 0 else 0

    total_return = (equity_curve.iloc[-1] / CAPITAL - 1) * 100

    return {
        "total_return_pct": round(total_return, 2),
        "ann_return_pct": round(ann_ret * 100, 2),
        "ann_vol_pct": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(mdd * 100, 2),
        "profit_factor": round(pf, 3),
        "win_rate": round(wr, 4),
        "trades": trades_count,
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "regime_gap": round(regime_gap, 3),
    }


def run_variant_a(signals, close, regime):
    """Threshold 3: Invest in QQQ when score >= 3, else cash."""
    oot = signals.loc[OOT_START:]
    qqq_ret = close["QQQ"].pct_change().loc[oot.index]

    invested = oot["score"] >= 3
    # LAG signal by 1 day to avoid look-ahead bias (trade on T+1 after signal)
    invested_lagged = invested.shift(1).fillna(False)
    # Trades = transitions
    trades_mask = invested_lagged != invested_lagged.shift(1)
    trades_count = trades_mask.sum()

    daily_ret = pd.Series(0.0, index=oot.index)
    daily_ret[invested_lagged] = qqq_ret[invested_lagged]
    daily_ret = apply_slippage(daily_ret, trades_mask & invested_lagged)

    equity = (1 + daily_ret).cumprod() * CAPITAL
    metrics = calc_metrics(equity, daily_ret, regime.loc[oot.index], trades_count)
    return metrics, daily_ret, equity, oot["score"]


def run_variant_b(signals, close, regime):
    """Threshold 4: Invest in QQQ when score >= 4, else cash."""
    oot = signals.loc[OOT_START:]
    qqq_ret = close["QQQ"].pct_change().loc[oot.index]

    invested = oot["score"] >= 4
    invested_lagged = invested.shift(1).fillna(False)
    trades_mask = invested_lagged != invested_lagged.shift(1)
    trades_count = trades_mask.sum()

    daily_ret = pd.Series(0.0, index=oot.index)
    daily_ret[invested_lagged] = qqq_ret[invested_lagged]
    daily_ret = apply_slippage(daily_ret, trades_mask & invested_lagged)

    equity = (1 + daily_ret).cumprod() * CAPITAL
    metrics = calc_metrics(equity, daily_ret, regime.loc[oot.index], trades_count)
    return metrics, daily_ret, equity, oot["score"]


def run_variant_c(signals, close, regime):
    """Proportional: Position size = score/5."""
    oot = signals.loc[OOT_START:]
    qqq_ret = close["QQQ"].pct_change().loc[oot.index]

    weight = (oot["score"] / 5.0).shift(1).fillna(0)  # Lag 1 day
    weight_change = weight.diff().abs() > 0.01
    trades_count = weight_change.sum()

    daily_ret = qqq_ret * weight
    daily_ret = apply_slippage(daily_ret, weight_change)

    equity = (1 + daily_ret).cumprod() * CAPITAL
    metrics = calc_metrics(equity, daily_ret, regime.loc[oot.index], trades_count)
    return metrics, daily_ret, equity, oot["score"]


def run_variant_d(signals, close, regime):
    """Contrarian layer: Score >= 3 → QQQ. Score 0 + SPY weekly drop >2% → contrarian buy SPY 5 days."""
    oot = signals.loc[OOT_START:]
    qqq_ret = close["QQQ"].pct_change().loc[oot.index]
    spy_ret = close["SPY"].pct_change().loc[oot.index]
    spy_weekly_ret = close["SPY"].pct_change(5).loc[oot.index]

    # Main signal (lagged 1 day)
    main_signal = (oot["score"] >= 3).shift(1).fillna(False)

    # Contrarian: score == 0 AND SPY dropped >2% over 5 days (lagged 1 day)
    contrarian_trigger = ((oot["score"] == 0) & (spy_weekly_ret < -0.02)).shift(1).fillna(False)

    # Hold contrarian for 5 days after trigger (vectorized)
    trigger_arr = contrarian_trigger.values.astype(bool)
    hold_arr = np.zeros(len(trigger_arr), dtype=bool)
    for shift in range(5):
        shifted = np.zeros_like(hold_arr)
        shifted[shift:] = trigger_arr[:len(trigger_arr)-shift] if shift > 0 else trigger_arr
        hold_arr |= shifted
    contrarian_hold = pd.Series(hold_arr, index=oot.index)

    daily_ret = pd.Series(0.0, index=oot.index)
    daily_ret[main_signal] = qqq_ret[main_signal]
    # Contrarian overrides only when main signal is off
    contrarian_only = contrarian_hold & ~main_signal
    daily_ret[contrarian_only] = spy_ret[contrarian_only]

    invested = main_signal | contrarian_only
    trades_mask = invested != invested.shift(1)
    trades_count = trades_mask.sum()
    daily_ret = apply_slippage(daily_ret, trades_mask & invested)

    equity = (1 + daily_ret).cumprod() * CAPITAL
    metrics = calc_metrics(equity, daily_ret, regime.loc[oot.index], trades_count)
    return metrics, daily_ret, equity, oot["score"]


def run_variant_e(signals, close, regime):
    """Sector selection: Score >= 3 → buy best 20d relative strength sector ETF."""
    oot = signals.loc[OOT_START:]

    # 20d returns for sector selection
    sector_ret_20d = close[SECTOR_ETFS].pct_change(20).loc[oot.index]
    sector_daily_ret = close[SECTOR_ETFS].pct_change().loc[oot.index]

    invested = (oot["score"] >= 3).shift(1).fillna(False)

    # Pick best sector each day (lagged 1 day)
    best_sector = sector_ret_20d.idxmax(axis=1).shift(1)

    # Vectorized: get return of best sector for each day
    best_sector_returns = pd.Series(0.0, index=oot.index)
    for sec in SECTOR_ETFS:
        mask = (best_sector == sec) & invested
        best_sector_returns[mask] = sector_daily_ret.loc[mask.values, sec].values
    daily_ret = best_sector_returns

    # Trades: transition in/out + sector switches
    prev_sector = best_sector.shift(1)
    sector_switch = (best_sector != prev_sector) & invested
    trades_mask = (invested != invested.shift(1)) | sector_switch
    trades_count = trades_mask.sum()
    daily_ret = apply_slippage(daily_ret, trades_mask & invested)

    equity = (1 + daily_ret).cumprod() * CAPITAL
    metrics = calc_metrics(equity, daily_ret, regime.loc[oot.index], trades_count)
    return metrics, daily_ret, equity, oot["score"]


def run_variant_f(signals, close, regime):
    """Risk parity: Score >= 3 → 60/40 QQQ/TLT. Score 1-2 → 30/70. Score 0 → 100% TLT."""
    oot = signals.loc[OOT_START:]
    qqq_ret = close["QQQ"].pct_change().loc[oot.index]
    tlt_ret = close["TLT"].pct_change().loc[oot.index]

    score = oot["score"].shift(1).fillna(0)  # Lag 1 day

    # Weights
    qqq_w = pd.Series(0.0, index=oot.index)
    tlt_w = pd.Series(1.0, index=oot.index)

    high = score >= 3
    mid = (score >= 1) & (score < 3)
    low = score < 1

    qqq_w[high] = 0.60
    tlt_w[high] = 0.40
    qqq_w[mid] = 0.30
    tlt_w[mid] = 0.70
    qqq_w[low] = 0.00
    tlt_w[low] = 1.00

    daily_ret = qqq_ret * qqq_w + tlt_ret * tlt_w

    # Trades = allocation changes
    alloc_change = (qqq_w.diff().abs() > 0.01)
    trades_count = alloc_change.sum()
    daily_ret = apply_slippage(daily_ret, alloc_change)

    equity = (1 + daily_ret).cumprod() * CAPITAL
    metrics = calc_metrics(equity, daily_ret, regime.loc[oot.index], trades_count)
    return metrics, daily_ret, equity, score


def permutation_test(daily_returns, scores, run_fn_from_scores, n_perms=N_PERMS):
    """
    Permutation test: shuffle daily scores across dates, recompute strategy returns.
    Returns p-value for the actual Sharpe being better than shuffled.
    """
    rng = np.random.RandomState(SEED)
    dr_std = daily_returns.std()
    actual_sharpe = daily_returns.mean() / dr_std * np.sqrt(252) if dr_std > 0 else 0

    score_vals = scores.values.copy()
    count_better = 0
    for i in range(n_perms):
        rng.shuffle(score_vals)
        shuffled = pd.Series(score_vals.copy(), index=scores.index)

        perm_ret = run_fn_from_scores(shuffled)
        ps = perm_ret.std()
        perm_sharpe = perm_ret.mean() / ps * np.sqrt(252) if ps > 0 else 0
        if perm_sharpe >= actual_sharpe:
            count_better += 1

    return count_better / n_perms


def validate_5gate(metrics, perm_p):
    """Apply 5-gate validation framework."""
    if metrics is None:
        return {"pass": False, "gates": {}, "reason": "No metrics"}

    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > SHARPE_MIN,
        "perm_p_lt_0.05": perm_p < PERM_P_MAX,
        "regime_gap_lt_0.5": metrics["regime_gap"] < REGIME_GAP_MAX,
        "mdd_gt_neg50": metrics["max_drawdown_pct"] > MDD_FLOOR * 100,
        "trades_gte_20": metrics["trades"] >= MIN_TRADES,
    }
    passed = sum(gates.values())
    return {
        "pass": passed >= 5,
        "gates_passed": passed,
        "gates": gates,
    }


def main():
    print("=" * 70)
    print("MULTI-SIGNAL AGGREGATOR BACKTEST")
    print("=" * 70)

    # Download data
    close, volume = download_data()

    # Compute signals
    print("\nComputing signals...")
    signals = compute_signals(close, volume)

    # Regime classification
    regime = identify_regime(close)

    # OOT signal stats
    oot_signals = signals.loc[OOT_START:]
    print(f"\nOOT period: {oot_signals.index[0].date()} to {oot_signals.index[-1].date()} ({len(oot_signals)} days)")
    print(f"Score distribution:")
    for s in range(6):
        pct = (oot_signals['score'] == s).mean() * 100
        print(f"  Score {s}: {pct:.1f}%")
    print(f"Mean score: {oot_signals['score'].mean():.2f}")

    # Regime distribution
    oot_regime = regime.loc[oot_signals.index]
    bull_pct = (oot_regime == "bull").mean() * 100
    print(f"Bull days: {bull_pct:.1f}%, Bear days: {100-bull_pct:.1f}%")

    # Run all variants
    variants = {
        "A_threshold3": run_variant_a,
        "B_threshold4": run_variant_b,
        "C_proportional": run_variant_c,
        "D_contrarian": run_variant_d,
        "E_sector_selection": run_variant_e,
        "F_risk_parity": run_variant_f,
    }

    # Buy-and-hold benchmarks
    oot_idx = oot_signals.index
    spy_bh_ret = close["SPY"].pct_change().loc[oot_idx]
    spy_bh_eq = (1 + spy_bh_ret).cumprod() * CAPITAL
    spy_metrics = calc_metrics(spy_bh_eq, spy_bh_ret, regime.loc[oot_idx], 1)

    qqq_bh_ret = close["QQQ"].pct_change().loc[oot_idx]
    qqq_bh_eq = (1 + qqq_bh_ret).cumprod() * CAPITAL
    qqq_metrics = calc_metrics(qqq_bh_eq, qqq_bh_ret, regime.loc[oot_idx], 1)

    print(f"\n{'='*70}")
    print(f"BENCHMARKS (OOT)")
    print(f"{'='*70}")
    print(f"SPY B&H: Return={spy_metrics['total_return_pct']:.1f}%, Sharpe={spy_metrics['sharpe']:.3f}, MDD={spy_metrics['max_drawdown_pct']:.1f}%, Sortino={spy_metrics['sortino']:.3f}")
    print(f"QQQ B&H: Return={qqq_metrics['total_return_pct']:.1f}%, Sharpe={qqq_metrics['sharpe']:.3f}, MDD={qqq_metrics['max_drawdown_pct']:.1f}%, Sortino={qqq_metrics['sortino']:.3f}")

    results = {
        "metadata": {
            "script": "signal_aggregator_backtest.py",
            "run_date": datetime.now().isoformat(),
            "oot_start": OOT_START,
            "oot_end": OOT_END,
            "capital": CAPITAL,
            "slippage_pct": SLIPPAGE_PCT,
            "n_permutations": N_PERMS,
            "oot_days": len(oot_signals),
        },
        "signal_stats": {
            "mean_score": round(oot_signals["score"].mean(), 2),
            "score_distribution": {str(s): round((oot_signals["score"] == s).mean(), 4) for s in range(6)},
            "bull_pct": round(bull_pct, 1),
            "bear_pct": round(100 - bull_pct, 1),
        },
        "benchmarks": {
            "SPY_buy_hold": spy_metrics,
            "QQQ_buy_hold": qqq_metrics,
        },
        "variants": {},
    }

    # Helper functions for permutation tests (create returns from shuffled scores)
    qqq_ret_oot = close["QQQ"].pct_change().loc[oot_idx]
    spy_ret_oot = close["SPY"].pct_change().loc[oot_idx]
    tlt_ret_oot = close["TLT"].pct_change().loc[oot_idx]
    sector_daily_ret_oot = close[SECTOR_ETFS].pct_change().loc[oot_idx]
    sector_ret_20d_oot = close[SECTOR_ETFS].pct_change(20).loc[oot_idx]
    spy_weekly_ret_oot = close["SPY"].pct_change(5).loc[oot_idx]

    def perm_a(shuffled_scores):
        invested = shuffled_scores >= 3
        r = pd.Series(0.0, index=oot_idx)
        r[invested] = qqq_ret_oot[invested]
        return r

    def perm_b(shuffled_scores):
        invested = shuffled_scores >= 4
        r = pd.Series(0.0, index=oot_idx)
        r[invested] = qqq_ret_oot[invested]
        return r

    def perm_c(shuffled_scores):
        w = shuffled_scores / 5.0
        return qqq_ret_oot * w

    def perm_d(shuffled_scores):
        main = shuffled_scores >= 3
        contrarian_trigger = (shuffled_scores == 0) & (spy_weekly_ret_oot < -0.02)
        # Vectorized 5-day hold
        trigger_arr = contrarian_trigger.values.astype(bool)
        hold_arr = np.zeros(len(trigger_arr), dtype=bool)
        for shift in range(5):
            shifted = np.zeros_like(hold_arr)
            if shift > 0:
                shifted[shift:] = trigger_arr[:len(trigger_arr)-shift]
            else:
                shifted[:] = trigger_arr
            hold_arr |= shifted
        contrarian_hold = pd.Series(hold_arr, index=oot_idx)
        r = pd.Series(0.0, index=oot_idx)
        r[main] = qqq_ret_oot[main]
        c_only = contrarian_hold & ~main
        r[c_only] = spy_ret_oot[c_only]
        return r

    # Pre-compute best sector returns for perm_e
    _best_sec = sector_ret_20d_oot.idxmax(axis=1)
    _best_sec_ret = pd.Series(0.0, index=oot_idx)
    for sec in SECTOR_ETFS:
        mask = _best_sec == sec
        _best_sec_ret[mask] = sector_daily_ret_oot.loc[mask, sec].values

    def perm_e(shuffled_scores):
        invested = shuffled_scores >= 3
        r = pd.Series(0.0, index=oot_idx)
        r[invested] = _best_sec_ret[invested]
        return r

    def perm_f(shuffled_scores):
        qw = pd.Series(0.0, index=oot_idx)
        tw = pd.Series(1.0, index=oot_idx)
        h = shuffled_scores >= 3
        m = (shuffled_scores >= 1) & (shuffled_scores < 3)
        qw[h] = 0.60; tw[h] = 0.40
        qw[m] = 0.30; tw[m] = 0.70
        return qqq_ret_oot * qw + tlt_ret_oot * tw

    perm_fns = {
        "A_threshold3": perm_a,
        "B_threshold4": perm_b,
        "C_proportional": perm_c,
        "D_contrarian": perm_d,
        "E_sector_selection": perm_e,
        "F_risk_parity": perm_f,
    }

    print(f"\n{'='*70}")
    print(f"VARIANT RESULTS")
    print(f"{'='*70}")

    for name, fn in variants.items():
        print(f"\n--- Variant {name} ---")
        metrics, daily_ret, equity, scores = fn(signals, close, regime)

        if metrics is None:
            print(f"  SKIPPED: insufficient data")
            results["variants"][name] = {"status": "skipped"}
            continue

        # Permutation test
        print(f"  Running {N_PERMS} permutations...")
        perm_p = permutation_test(daily_ret, scores, perm_fns[name], N_PERMS)

        # 5-gate validation
        validation = validate_5gate(metrics, perm_p)

        print(f"  Return: {metrics['total_return_pct']:+.1f}%")
        print(f"  Sharpe: {metrics['sharpe']:.3f}  Sortino: {metrics['sortino']:.3f}")
        print(f"  MDD: {metrics['max_drawdown_pct']:.1f}%  PF: {metrics['profit_factor']:.2f}  WR: {metrics['win_rate']:.1%}")
        print(f"  Trades: {metrics['trades']}")
        print(f"  Bull Sharpe: {metrics['bull_sharpe']:.3f}  Bear Sharpe: {metrics['bear_sharpe']:.3f}  Gap: {metrics['regime_gap']:.3f}")
        print(f"  Perm p-value: {perm_p:.4f}")
        print(f"  Validation: {validation['gates_passed']}/5 gates {'PASS' if validation['pass'] else 'FAIL'}")
        for gate, passed in validation["gates"].items():
            status = "PASS" if passed else "FAIL"
            print(f"    {gate}: {status}")

        results["variants"][name] = {
            "metrics": metrics,
            "perm_p_value": round(perm_p, 4),
            "validation": validation,
        }

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"{'Variant':<25} {'Sharpe':>8} {'Sortino':>8} {'Return%':>8} {'MDD%':>8} {'PermP':>8} {'Gates':>6}")
    print("-" * 70)

    for name, v in results["variants"].items():
        if v.get("status") == "skipped":
            continue
        m = v["metrics"]
        vd = v["validation"]
        print(f"{name:<25} {m['sharpe']:>8.3f} {m['sortino']:>8.3f} {m['total_return_pct']:>+7.1f}% {m['max_drawdown_pct']:>7.1f}% {v['perm_p_value']:>8.4f} {vd['gates_passed']:>2}/5 {'OK' if vd['pass'] else 'XX'}")

    print(f"\n{'SPY B&H':<25} {spy_metrics['sharpe']:>8.3f} {spy_metrics['sortino']:>8.3f} {spy_metrics['total_return_pct']:>+7.1f}% {spy_metrics['max_drawdown_pct']:>7.1f}%")
    print(f"{'QQQ B&H':<25} {qqq_metrics['sharpe']:>8.3f} {qqq_metrics['sortino']:>8.3f} {qqq_metrics['total_return_pct']:>+7.1f}% {qqq_metrics['max_drawdown_pct']:>7.1f}%")

    # Save results
    output_path = Path("/home/jupiter/Lvl3Quant/data/signal_aggregator_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    results = main()
