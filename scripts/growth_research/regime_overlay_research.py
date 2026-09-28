#!/usr/bin/env python3
"""
Regime Overlay Research: Can we hedge growth strategy drawdowns?
================================================================
HC #709: R1 failure is acceptable IF we can predict/hedge red-day drawdowns.

Tests 4 overlay approaches on walk-forward validated OOT returns:
  1. VIX-based regime filter
  2. Trend regime filter (SPY MA)
  3. Put hedge overlay (simulated)
  4. Dynamic leverage scaling

CRITICAL: Overlay parameters are ALSO walk-forward tested (sliding window).
HC #705: permutation test, sub-period consistency, outlier removal.
HC #0: Sliding windows only, never expanding.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from itertools import product
import json, warnings, time, sys
from datetime import datetime
warnings.filterwarnings("ignore")

_builtin_print = print  # save before any shadowing

def _flush_print(*args, **kwargs):
    """Print with immediate flush for non-interactive mode."""
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()

OOT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/walkforward_validation")
OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Walk-forward config for overlay optimization
OVERLAY_TRAIN_DAYS = 504   # ~2 years
OVERLAY_TEST_DAYS = 126    # ~6 months
OVERLAY_SLIDE_DAYS = 126   # slide by test size


# ── DATA LOADING ─────────────────────────────────────────────────────

def load_oot_returns():
    """Load OOT daily returns from walk-forward validation."""
    dm = pd.read_csv(OOT_DIR / "dm_oot_returns.csv", index_col=0, parse_dates=True)
    bo = pd.read_csv(OOT_DIR / "bo_oot_returns.csv", index_col=0, parse_dates=True)
    dm = dm.iloc[:, 0]  # single column
    bo = bo.iloc[:, 0]
    dm.name = "dual_momentum"
    bo.name = "breakout"
    _flush_print(f"Loaded DM OOT: {len(dm)} days ({dm.index[0].date()} to {dm.index[-1].date()})")
    _flush_print(f"Loaded BO OOT: {len(bo)} days ({bo.index[0].date()} to {bo.index[-1].date()})")
    return dm, bo


def download_market_data(start_date, end_date):
    """Download SPY and VIX daily data."""
    _flush_print(f"Downloading SPY, ^VIX from {start_date} to {end_date}...")
    spy = yf.download("SPY", start=start_date, end=end_date,
                       auto_adjust=True, progress=False)
    vix = yf.download("^VIX", start=start_date, end=end_date,
                       auto_adjust=True, progress=False)

    if isinstance(spy.columns, pd.MultiIndex):
        spy_close = spy["Close"].squeeze()
    else:
        spy_close = spy["Close"]

    if isinstance(vix.columns, pd.MultiIndex):
        vix_close = vix["Close"].squeeze()
    else:
        vix_close = vix["Close"]

    spy_ret = spy_close.pct_change()
    spy_ma50 = spy_close.rolling(50).mean()
    spy_ma200 = spy_close.rolling(200).mean()
    vix_slope10 = vix_close.diff(10) / vix_close.shift(10)  # 10-day pct change

    mkt = pd.DataFrame({
        "spy_close": spy_close,
        "spy_ret": spy_ret,
        "spy_ma50": spy_ma50,
        "spy_ma200": spy_ma200,
        "vix": vix_close,
        "vix_slope10": vix_slope10,
    })
    mkt = mkt.ffill()
    _flush_print(f"  Market data: {len(mkt)} days")
    return mkt


# ── METRICS ──────────────────────────────────────────────────────────

def calc_metrics(dr, label=""):
    """Compute Sharpe, CAGR, MaxDD, WR, Sortino from daily returns."""
    dr = dr.dropna()
    if len(dr) < 20 or dr.std() == 0:
        return {"label": label, "n_days": len(dr), "valid": False}

    ann_ret = dr.mean() * 252
    ann_vol = dr.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol
    years = len(dr) / 252
    cagr = (1 + dr).prod() ** (1 / max(years, 0.01)) - 1
    cum = (1 + dr).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    wr = (dr > 0).mean()
    downside = dr[dr < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside > 0 else 0.0

    return {
        "label": label, "n_days": int(len(dr)), "valid": True,
        "sharpe": round(float(sharpe), 3),
        "cagr": round(float(cagr * 100), 2),
        "max_dd": round(float(max_dd * 100), 2),
        "wr": round(float(wr * 100), 1),
        "sortino": round(float(sortino), 3),
        "ann_vol": round(float(ann_vol * 100), 1),
    }


def regime_test(dr, spy_ret):
    """R1 regime-agnostic test: stratify by green/red/flat SPY days."""
    common = dr.index.intersection(spy_ret.index)
    if len(common) < 30:
        return None
    dr_c = dr.loc[common]
    spy_c = spy_ret.loc[common]

    green = dr_c[spy_c > 0.001]
    red = dr_c[spy_c < -0.001]
    flat = dr_c[(spy_c >= -0.001) & (spy_c <= 0.001)]

    def _sharpe(s):
        if len(s) < 10 or s.std() == 0:
            return 0.0
        return float(s.mean() / s.std() * np.sqrt(252))

    s_green = _sharpe(green)
    s_red = _sharpe(red)
    denom = max(abs(s_green), abs(s_red), 0.001)
    regime_gap = abs(s_green - s_red) / denom

    return {
        "sharpe_green": round(s_green, 3), "n_green": int(len(green)),
        "sharpe_red": round(s_red, 3), "n_red": int(len(red)),
        "sharpe_flat": round(_sharpe(flat), 3), "n_flat": int(len(flat)),
        "regime_gap": round(float(regime_gap), 3),
        "pass_r1": regime_gap <= 0.50,
    }


def permutation_test(dr, n_shuffles=200):
    """Block-bootstrap sign-flip permutation test (HC #705)."""
    dr = dr.dropna().values
    if len(dr) < 30 or dr.std() == 0:
        return {"p_value": 1.0, "actual_sharpe": 0.0}
    actual = dr.mean() / dr.std() * np.sqrt(252)
    block_size = 5
    n_blocks = len(dr) // block_size
    null_sharpes = []
    for _ in range(n_shuffles):
        perm = dr.copy()
        for b in range(n_blocks):
            if np.random.random() < 0.5:
                s = b * block_size
                e = min(s + block_size, len(perm))
                perm[s:e] = -perm[s:e]
        std = perm.std()
        null_sharpes.append(perm.mean() / std * np.sqrt(252) if std > 0 else 0)
    p_value = (np.sum(np.array(null_sharpes) >= actual) + 1) / (n_shuffles + 1)
    return {"p_value": round(float(p_value), 4), "actual_sharpe": round(float(actual), 3)}


def subperiod_consistency(dr):
    """Split into halves, check both positive Sharpe."""
    dr = dr.dropna()
    if len(dr) < 60:
        return None
    mid = len(dr) // 2
    h1, h2 = dr.iloc[:mid], dr.iloc[mid:]
    s1 = float(h1.mean() / h1.std() * np.sqrt(252)) if h1.std() > 0 else 0
    s2 = float(h2.mean() / h2.std() * np.sqrt(252)) if h2.std() > 0 else 0
    return {
        "sharpe_h1": round(s1, 3), "sharpe_h2": round(s2, 3),
        "both_positive": s1 > 0 and s2 > 0,
    }


def outlier_removal_test(dr, pct=1):
    """Remove top/bottom pct% and recheck Sharpe."""
    dr = dr.dropna()
    if len(dr) < 60:
        return None
    lo = np.percentile(dr, pct)
    hi = np.percentile(dr, 100 - pct)
    trimmed = dr[(dr >= lo) & (dr <= hi)]
    if len(trimmed) < 30 or trimmed.std() == 0:
        return None
    orig = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0
    trim_s = float(trimmed.mean() / trimmed.std() * np.sqrt(252))
    return {
        "sharpe_original": round(orig, 3),
        "sharpe_trimmed_1pct": round(trim_s, 3),
        "survives": trim_s > 0,
    }


# ── OVERLAY APPROACHES ──────────────────────────────────────────────

def apply_vix_filter(strategy_ret, mkt, vix_threshold, vix_slope_threshold):
    """Approach 1: VIX-based regime filter (vectorized).
    When VIX > threshold OR VIX slope > slope_threshold → go to cash (0 return)."""
    common = strategy_ret.index.intersection(mkt.index)
    sr = strategy_ret.loc[common].copy()
    vix = mkt.loc[common, "vix"]
    vix_slope = mkt.loc[common, "vix_slope10"]

    cash_mask = (vix > vix_threshold) | (vix_slope > vix_slope_threshold)
    sr[cash_mask] = 0.0
    return sr


def apply_trend_filter(strategy_ret, mkt, use_ma50=True, use_ma200=True,
                        reduced_exposure=0.25):
    """Approach 2: Trend regime filter (vectorized).
    SPY < 200MA → cash. SPY < 50MA → reduced_exposure. SPY > both → full."""
    common = strategy_ret.index.intersection(mkt.index)
    sr = strategy_ret.loc[common].copy()
    spy = mkt.loc[common, "spy_close"].values
    ma50 = mkt.loc[common, "spy_ma50"].values
    ma200 = mkt.loc[common, "spy_ma200"].values

    valid = ~(np.isnan(ma50) | np.isnan(ma200))
    if use_ma200:
        below_200 = valid & (spy < ma200)
        sr.values[below_200] = 0.0
    if use_ma50:
        below_50_only = valid & (spy < ma50) & ~(valid & (spy < ma200)) if use_ma200 else valid & (spy < ma50)
        sr.values[below_50_only] *= reduced_exposure

    return sr


def apply_put_hedge(strategy_ret, mkt, put_otm_pct=0.05,
                     monthly_cost_pct=0.004, payoff_multiplier=1.0):
    """Approach 3: Simulated put hedge overlay (vectorized).
    Cost: monthly_cost_pct of portfolio spread daily.
    Payoff: when SPY drops > put_otm_pct in rolling 21d, receive proportional payoff.
    """
    common = strategy_ret.index.intersection(mkt.index)
    sr = strategy_ret.loc[common].copy()
    spy = mkt.loc[common, "spy_close"]

    daily_cost = monthly_cost_pct / 21.0
    spy_ret_21d = spy.pct_change(21)

    # Deduct daily cost
    sr = sr - daily_cost

    # Add put payoff where SPY crashed
    crash_mask = spy_ret_21d < -put_otm_pct
    intrinsic = (spy_ret_21d.abs() - put_otm_pct).clip(lower=0)
    sr = sr + (intrinsic * payoff_multiplier / 21.0 * crash_mask.astype(float))

    return sr


def apply_dynamic_leverage(strategy_ret, mkt, vix_low=20, vix_high=30,
                            lev_bull=1.5, lev_mixed=0.75, lev_bear=0.0):
    """Approach 4: Dynamic leverage scaling (vectorized).
    Bull (SPY > 50MA, VIX < vix_low): lev_bull
    Bear (SPY < 200MA, VIX > vix_high): lev_bear
    Mixed: lev_mixed
    """
    common = strategy_ret.index.intersection(mkt.index)
    sr = strategy_ret.loc[common].copy()
    spy = mkt.loc[common, "spy_close"].values
    ma50 = mkt.loc[common, "spy_ma50"].values
    ma200 = mkt.loc[common, "spy_ma200"].values
    vix = mkt.loc[common, "vix"].values

    valid = ~(np.isnan(ma50) | np.isnan(ma200) | np.isnan(vix))
    bull = valid & (spy > ma50) & (vix < vix_low)
    bear = valid & (spy < ma200) & (vix > vix_high)
    mixed = valid & ~bull & ~bear

    sr.values[bull] *= lev_bull
    sr.values[bear] *= lev_bear
    sr.values[mixed] *= lev_mixed

    return sr


# ── WALK-FORWARD OVERLAY OPTIMIZATION ───────────────────────────────

def generate_overlay_folds(dates):
    """Generate sliding window folds for overlay parameter optimization."""
    folds = []
    n = len(dates)
    start = 0
    while start + OVERLAY_TRAIN_DAYS + OVERLAY_TEST_DAYS <= n:
        train_end = start + OVERLAY_TRAIN_DAYS
        test_end = train_end + OVERLAY_TEST_DAYS
        folds.append((start, train_end, test_end))
        start += OVERLAY_SLIDE_DAYS
    # Use remaining data for last fold if enough
    if start + OVERLAY_TRAIN_DAYS < n:
        train_end = start + OVERLAY_TRAIN_DAYS
        folds.append((start, train_end, n))
    return folds


def wf_vix_overlay(strategy_ret, mkt):
    """Walk-forward optimize VIX overlay parameters."""
    common = strategy_ret.index.intersection(mkt.index)
    sr = strategy_ret.loc[common]
    mkt_c = mkt.loc[common]
    dates = sr.index

    vix_thresholds = [18, 20, 23, 25, 28, 30, 35]
    slope_thresholds = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

    folds = generate_overlay_folds(dates)
    _flush_print(f"    VIX overlay: {len(folds)} folds, {len(vix_thresholds)*len(slope_thresholds)} param combos/fold")
    oot_returns = []

    for fi, (fold_start, fold_train_end, fold_test_end) in enumerate(folds):
        train_dates = dates[fold_start:fold_train_end]
        test_dates = dates[fold_train_end:fold_test_end]

        if len(train_dates) < 100 or len(test_dates) < 20:
            continue

        best_sharpe = -999
        best_params = (25, 0.15)
        for vt, st in product(vix_thresholds, slope_thresholds):
            train_sr = sr.loc[train_dates]
            train_mkt = mkt_c.loc[train_dates]
            overlaid = apply_vix_filter(train_sr, train_mkt, vt, st)
            if overlaid.std() == 0 or len(overlaid) < 50:
                continue
            sharpe = float(overlaid.mean() / overlaid.std() * np.sqrt(252))
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = (vt, st)

        test_sr = sr.loc[test_dates]
        test_mkt = mkt_c.loc[test_dates]
        test_overlaid = apply_vix_filter(test_sr, test_mkt, best_params[0], best_params[1])
        oot_returns.append(test_overlaid)
        if (fi + 1) % 5 == 0:
            _flush_print(f"      fold {fi+1}/{len(folds)} done")

    _flush_print(f"    VIX overlay complete: {len(oot_returns)} OOT segments")
    if oot_returns:
        return pd.concat(oot_returns)
    return sr


def wf_trend_overlay(strategy_ret, mkt):
    """Walk-forward optimize trend overlay parameters."""
    common = strategy_ret.index.intersection(mkt.index)
    sr = strategy_ret.loc[common]
    mkt_c = mkt.loc[common]
    dates = sr.index

    exposure_levels = [0.0, 0.10, 0.25, 0.50]
    configs = [
        (True, True),   # use both MA50 and MA200
        (True, False),  # MA50 only
        (False, True),  # MA200 only
    ]

    folds = generate_overlay_folds(dates)
    oot_returns = []

    for fold_start, fold_train_end, fold_test_end in folds:
        train_dates = dates[fold_start:fold_train_end]
        test_dates = dates[fold_train_end:fold_test_end]

        if len(train_dates) < 100 or len(test_dates) < 20:
            continue

        best_sharpe = -999
        best_params = (True, True, 0.25)
        for (m50, m200), exp in product(configs, exposure_levels):
            train_sr = sr.loc[train_dates]
            train_mkt = mkt_c.loc[train_dates]
            overlaid = apply_trend_filter(train_sr, train_mkt, m50, m200, exp)
            if overlaid.std() == 0 or len(overlaid) < 50:
                continue
            sharpe = float(overlaid.mean() / overlaid.std() * np.sqrt(252))
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = (m50, m200, exp)

        test_sr = sr.loc[test_dates]
        test_mkt = mkt_c.loc[test_dates]
        test_overlaid = apply_trend_filter(test_sr, test_mkt,
                                            best_params[0], best_params[1], best_params[2])
        oot_returns.append(test_overlaid)

    if oot_returns:
        return pd.concat(oot_returns)
    return sr


def wf_put_hedge_overlay(strategy_ret, mkt):
    """Walk-forward optimize put hedge parameters."""
    common = strategy_ret.index.intersection(mkt.index)
    sr = strategy_ret.loc[common]
    mkt_c = mkt.loc[common]
    dates = sr.index

    otm_pcts = [0.03, 0.05, 0.07, 0.10]
    costs = [0.002, 0.003, 0.004, 0.005]
    multipliers = [0.5, 1.0, 1.5, 2.0]

    folds = generate_overlay_folds(dates)
    oot_returns = []

    for fold_start, fold_train_end, fold_test_end in folds:
        train_dates = dates[fold_start:fold_train_end]
        test_dates = dates[fold_train_end:fold_test_end]

        if len(train_dates) < 100 or len(test_dates) < 20:
            continue

        best_sharpe = -999
        best_params = (0.05, 0.004, 1.0)
        for otm, cost, mult in product(otm_pcts, costs, multipliers):
            train_sr = sr.loc[train_dates]
            train_mkt = mkt_c.loc[train_dates]
            overlaid = apply_put_hedge(train_sr, train_mkt, otm, cost, mult)
            if overlaid.std() == 0 or len(overlaid) < 50:
                continue
            sharpe = float(overlaid.mean() / overlaid.std() * np.sqrt(252))
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = (otm, cost, mult)

        test_sr = sr.loc[test_dates]
        test_mkt = mkt_c.loc[test_dates]
        test_overlaid = apply_put_hedge(test_sr, test_mkt,
                                         best_params[0], best_params[1], best_params[2])
        oot_returns.append(test_overlaid)

    if oot_returns:
        return pd.concat(oot_returns)
    return sr


def wf_dynamic_leverage_overlay(strategy_ret, mkt):
    """Walk-forward optimize dynamic leverage parameters."""
    common = strategy_ret.index.intersection(mkt.index)
    sr = strategy_ret.loc[common]
    mkt_c = mkt.loc[common]
    dates = sr.index

    # Reduced grid for speed — still covers the interesting space
    vix_lows = [18, 22, 25]
    vix_highs = [25, 30, 35]
    lev_bulls = [1.0, 1.5, 2.0]
    lev_mixeds = [0.5, 0.75, 1.0]
    lev_bears = [0.0, -0.25]

    folds = generate_overlay_folds(dates)
    oot_returns = []

    for fold_start, fold_train_end, fold_test_end in folds:
        train_dates = dates[fold_start:fold_train_end]
        test_dates = dates[fold_train_end:fold_test_end]

        if len(train_dates) < 100 or len(test_dates) < 20:
            continue

        best_sharpe = -999
        best_params = (20, 30, 1.5, 0.75, 0.0)
        for vl, vh, lb, lm, lbear in product(vix_lows, vix_highs, lev_bulls, lev_mixeds, lev_bears):
            if vl >= vh:
                continue  # nonsensical
            train_sr = sr.loc[train_dates]
            train_mkt = mkt_c.loc[train_dates]
            overlaid = apply_dynamic_leverage(train_sr, train_mkt, vl, vh, lb, lm, lbear)
            if overlaid.std() == 0 or len(overlaid) < 50:
                continue
            sharpe = float(overlaid.mean() / overlaid.std() * np.sqrt(252))
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = (vl, vh, lb, lm, lbear)

        test_sr = sr.loc[test_dates]
        test_mkt = mkt_c.loc[test_dates]
        test_overlaid = apply_dynamic_leverage(test_sr, test_mkt,
                                                best_params[0], best_params[1],
                                                best_params[2], best_params[3], best_params[4])
        oot_returns.append(test_overlaid)

    if oot_returns:
        return pd.concat(oot_returns)
    return sr


# ── FULL EVALUATION ─────────────────────────────────────────────────

def full_eval(dr, spy_ret, label):
    """Run all metrics + HC #705 checks on a return series."""
    m = calc_metrics(dr, label)
    r1 = regime_test(dr, spy_ret)
    perm = permutation_test(dr)
    sub = subperiod_consistency(dr)
    outlier = outlier_removal_test(dr)

    result = {**m}
    if r1:
        result["regime"] = r1
    result["permutation"] = perm
    if sub:
        result["subperiod"] = sub
    if outlier:
        result["outlier_robustness"] = outlier

    return result


# ── MAIN ─────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    np.random.seed(42)
    _flush_print("=" * 80)
    _flush_print("REGIME OVERLAY RESEARCH")
    _flush_print("HC #709: Can we hedge growth strategy R1 failures?")
    _flush_print("=" * 80)

    # Load data
    dm_ret, bo_ret = load_oot_returns()
    start_date = min(dm_ret.index[0], bo_ret.index[0]) - pd.Timedelta(days=300)
    end_date = max(dm_ret.index[-1], bo_ret.index[-1]) + pd.Timedelta(days=5)
    mkt = download_market_data(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    spy_ret = mkt["spy_ret"]

    results = {"generated": datetime.now().isoformat(), "strategies": {}}

    for strat_name, strat_ret in [("dual_momentum", dm_ret), ("breakout", bo_ret)]:
        _flush_print(f"\n{'='*80}")
        _flush_print(f"STRATEGY: {strat_name.upper()}")
        _flush_print(f"{'='*80}")

        strat_results = {}

        # Baseline (no overlay)
        _flush_print(f"\n--- Baseline (no overlay) ---")
        baseline = full_eval(strat_ret, spy_ret, f"{strat_name}_baseline")
        strat_results["baseline"] = baseline
        _print_summary(baseline)

        # Approach 1: VIX Filter (walk-forward)
        _flush_print(f"\n--- Approach 1: VIX Filter (walk-forward) ---")
        vix_oot = wf_vix_overlay(strat_ret, mkt)
        vix_eval = full_eval(vix_oot, spy_ret, f"{strat_name}_vix_filter")
        strat_results["vix_filter"] = vix_eval
        _print_summary(vix_eval)

        # Approach 2: Trend Filter (walk-forward)
        _flush_print(f"\n--- Approach 2: Trend Filter (walk-forward) ---")
        trend_oot = wf_trend_overlay(strat_ret, mkt)
        trend_eval = full_eval(trend_oot, spy_ret, f"{strat_name}_trend_filter")
        strat_results["trend_filter"] = trend_eval
        _print_summary(trend_eval)

        # Approach 3: Put Hedge (walk-forward)
        _flush_print(f"\n--- Approach 3: Put Hedge (walk-forward) ---")
        put_oot = wf_put_hedge_overlay(strat_ret, mkt)
        put_eval = full_eval(put_oot, spy_ret, f"{strat_name}_put_hedge")
        strat_results["put_hedge"] = put_eval
        _print_summary(put_eval)

        # Approach 4: Dynamic Leverage (walk-forward)
        _flush_print(f"\n--- Approach 4: Dynamic Leverage (walk-forward) ---")
        dynlev_oot = wf_dynamic_leverage_overlay(strat_ret, mkt)
        dynlev_eval = full_eval(dynlev_oot, spy_ret, f"{strat_name}_dynamic_leverage")
        strat_results["dynamic_leverage"] = dynlev_eval
        _print_summary(dynlev_eval)

        results["strategies"][strat_name] = strat_results

    # Summary comparison table
    _flush_print(f"\n\n{'='*80}")
    _flush_print("COMPARISON TABLE")
    _flush_print(f"{'='*80}")
    _print_comparison_table(results)

    # Identify R1 passers
    _flush_print(f"\n\n{'='*80}")
    _flush_print("R1 PASS/FAIL SUMMARY")
    _flush_print(f"{'='*80}")
    any_pass = False
    for strat_name, strat_data in results["strategies"].items():
        for overlay_name, overlay_data in strat_data.items():
            r1 = overlay_data.get("regime", {})
            gap = r1.get("regime_gap", "N/A")
            passed = r1.get("pass_r1", False)
            status = "PASS" if passed else "FAIL"
            _flush_print(f"  {strat_name:20s} + {overlay_name:20s}: R1 gap={gap:>6}  [{status}]")
            if passed:
                any_pass = True

    if any_pass:
        _flush_print("\n  ** At least one overlay passes R1! Worth pursuing. **")
    else:
        _flush_print("\n  ** No overlay passes R1 (gap <= 0.50). These strategies are")
        _flush_print("     fundamentally directional — hedging reduces returns but")
        _flush_print("     doesn't eliminate the regime dependency. **")

    elapsed = time.time() - t0
    results["elapsed_seconds"] = round(elapsed, 1)
    _flush_print(f"\nCompleted in {elapsed:.1f}s")

    # Save results
    out_path = OUT_DIR / "regime_overlay_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    _flush_print(f"\nResults saved to {out_path}")

    return results


def _print_summary(data):
    """Print concise summary of evaluation."""
    if not data.get("valid", False):
        _flush_print("  [INVALID - insufficient data]")
        return
    r1 = data.get("regime", {})
    perm = data.get("permutation", {})
    sub = data.get("subperiod", {})
    _flush_print(f"  Sharpe={data['sharpe']:.3f}  Sortino={data['sortino']:.3f}  "
          f"CAGR={data['cagr']:.1f}%  MaxDD={data['max_dd']:.1f}%  WR={data['wr']:.1f}%  "
          f"N={data['n_days']}")
    if r1:
        _flush_print(f"  R1: gap={r1['regime_gap']:.3f} ({'PASS' if r1.get('pass_r1') else 'FAIL'})  "
              f"Sharpe_green={r1['sharpe_green']:.3f}  Sharpe_red={r1['sharpe_red']:.3f}")
    if perm:
        _flush_print(f"  Permutation p={perm['p_value']:.4f}")
    if sub:
        _flush_print(f"  Sub-period: H1={sub['sharpe_h1']:.3f}  H2={sub['sharpe_h2']:.3f}  "
              f"{'PASS' if sub['both_positive'] else 'FAIL'}")


def _print_comparison_table(results):
    """Print side-by-side comparison."""
    header = f"{'Strategy':>15s} {'Overlay':>20s} {'Sharpe':>7s} {'Sortino':>8s} {'CAGR%':>7s} {'MaxDD%':>7s} {'WR%':>5s} {'R1 Gap':>7s} {'R1':>5s} {'Perm p':>7s}"
    _flush_print(header)
    _flush_print("-" * len(header))

    for strat_name, strat_data in results["strategies"].items():
        for overlay_name, data in strat_data.items():
            if not data.get("valid", False):
                continue
            r1 = data.get("regime", {})
            perm = data.get("permutation", {})
            gap = r1.get("regime_gap", -1)
            passed = "PASS" if r1.get("pass_r1", False) else "FAIL"
            p_val = perm.get("p_value", -1)
            _flush_print(f"{strat_name:>15s} {overlay_name:>20s} "
                  f"{data['sharpe']:>7.3f} {data['sortino']:>8.3f} "
                  f"{data['cagr']:>7.1f} {data['max_dd']:>7.1f} "
                  f"{data['wr']:>5.1f} {gap:>7.3f} {passed:>5s} {p_val:>7.4f}")


if __name__ == "__main__":
    main()
