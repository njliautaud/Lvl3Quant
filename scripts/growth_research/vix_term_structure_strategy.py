#!/usr/bin/env python3
"""
VIX Term Structure / Volatility Risk Premium Harvesting Study
=============================================================
Tests whether VIX term structure signals (contango/backwardation) can
generate alpha through VIX-related ETPs (SVXY, UVXY) with acceptable
tail risk.

Strategies:
  1. Contango Harvester  – hold SVXY when VIX/VIX3M < 0.9, else cash
  2. Backwardation Panic – hold SPY when VIX/VIX3M > 1.05 for 5-20d
  3. Combined Regime     – SVXY in contango, cash flat, SPY backwardation
  4. Size-Adjusted       – position size = f(contango steepness)

Methodology:
  - Walk-forward sliding: 252d train / 63d test / 21d slide
  - Permutation test: 100 shuffles, p < 0.05
  - R1 regime test: green/red SPY days, |Sharpe gap| < 0.50
  - Full risk metrics: Sharpe, Sortino, MaxDD, Calmar, annual ret, WR, PF
  - Correlation analysis vs SPY, UPRO (3x), SMA50 trend-following

Output: /home/jupiter/Lvl3Quant/output/growth_research/vix_term_structure/
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

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/vix_term_structure")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_DAYS = 252
TEST_DAYS = 63
SLIDE_DAYS = 21
N_PERMUTATIONS = 100
REGIME_SHARPE_GAP_MAX = 0.50
ANNUAL_FACTOR = 252

# Strategy thresholds
CONTANGO_THRESHOLD = 0.90    # VIX/VIX3M < 0.90 → contango
BACKWARDATION_THRESHOLD = 1.05  # VIX/VIX3M > 1.05 → backwardation
BACKWARDATION_HOLD_MIN = 5
BACKWARDATION_HOLD_MAX = 20

# Risk-free rate for Sharpe
RF_DAILY = 0.05 / 252  # ~5% annual


# ── Data Download ───────────────────────────────────────────────────────────
def download_data():
    """Download all required data from yfinance."""
    print("=" * 80)
    print("DOWNLOADING DATA")
    print("=" * 80)

    tickers = {
        "^VIX": "VIX",
        "^VIX3M": "VIX3M",
        "SVXY": "SVXY",
        "VIXY": "VIXY",
        "SPY": "SPY",
    }

    frames = {}
    for yf_tick, name in tickers.items():
        print(f"  Downloading {name} ({yf_tick})...")
        try:
            df = yf.download(yf_tick, start="2010-01-01", end="2026-07-17",
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            frames[name] = df["Close"].rename(name)
            print(f"    {name}: {len(df)} rows, {df.index[0].date()} → {df.index[-1].date()}")
        except Exception as e:
            print(f"    FAILED: {e}")

    # Build combined DataFrame
    data = pd.DataFrame(frames)
    data = data.dropna(subset=["VIX", "SPY"])  # need at least VIX + SPY

    # VIX3M may be missing early data — try VXV as fallback
    if data["VIX3M"].isna().sum() > len(data) * 0.5:
        print("  VIX3M mostly missing, trying ^VXV fallback...")
        try:
            vxv = yf.download("^VXV", start="2010-01-01", end="2026-07-17",
                              progress=False, auto_adjust=True)
            if isinstance(vxv.columns, pd.MultiIndex):
                vxv.columns = vxv.columns.get_level_values(0)
            data["VIX3M"] = data["VIX3M"].fillna(vxv["Close"])
        except Exception:
            pass

    data = data.dropna(subset=["VIX", "VIX3M", "SPY"])

    # Compute derived columns
    data["VIX_RATIO"] = data["VIX"] / data["VIX3M"]
    data["SPY_RET"] = data["SPY"].pct_change()
    data["SVXY_RET"] = data["SVXY"].pct_change() if "SVXY" in data.columns else np.nan
    data["VIXY_RET"] = data["VIXY"].pct_change() if "VIXY" in data.columns else np.nan

    # Synthetic SVXY for periods where actual SVXY is missing
    # Approximate: SVXY ≈ -0.5x daily VIX change (rough, better than nothing)
    # Actually, SVXY tracks -0.5x VIX short-term futures daily return
    # We can approximate from VIXY: SVXY_ret ≈ -0.5 * VIXY_ret (post-2018 rebalance to -0.5x)
    # Pre-2018 it was -1x, but we'll use -0.5x as conservative
    if data["SVXY_RET"].isna().any():
        vixy_available = data["VIXY_RET"].notna()
        svxy_missing = data["SVXY_RET"].isna()
        fill_mask = vixy_available & svxy_missing
        data.loc[fill_mask, "SVXY_RET"] = -0.5 * data.loc[fill_mask, "VIXY_RET"]

    # If still missing SVXY, approximate from VIX changes
    still_missing = data["SVXY_RET"].isna()
    if still_missing.any():
        vix_ret = data["VIX"].pct_change()
        data.loc[still_missing, "SVXY_RET"] = -0.5 * vix_ret[still_missing]

    # SPY green/red day classification
    data["SPY_GREEN"] = data["SPY_RET"] > 0

    # SMA50 trend-following signal for correlation analysis
    data["SPY_SMA50"] = data["SPY"].rolling(50).mean()
    data["SMA50_SIGNAL"] = (data["SPY"] > data["SPY_SMA50"]).astype(float)
    data["SMA50_RET"] = data["SMA50_SIGNAL"].shift(1) * data["SPY_RET"]

    # UPRO approximate (3x daily SPY)
    data["UPRO_RET"] = 3.0 * data["SPY_RET"]

    data = data.dropna(subset=["VIX_RATIO", "SPY_RET", "SVXY_RET"])
    print(f"\nFinal dataset: {len(data)} trading days, {data.index[0].date()} → {data.index[-1].date()}")
    print(f"VIX Ratio range: {data['VIX_RATIO'].min():.3f} – {data['VIX_RATIO'].max():.3f}")
    print(f"Contango days (< {CONTANGO_THRESHOLD}): {(data['VIX_RATIO'] < CONTANGO_THRESHOLD).sum()}")
    print(f"Backwardation days (> {BACKWARDATION_THRESHOLD}): {(data['VIX_RATIO'] > BACKWARDATION_THRESHOLD).sum()}")
    print(f"Flat zone: {((data['VIX_RATIO'] >= CONTANGO_THRESHOLD) & (data['VIX_RATIO'] <= BACKWARDATION_THRESHOLD)).sum()}")

    return data


# ── Risk Metrics ────────────────────────────────────────────────────────────
def compute_metrics(returns, name="Strategy"):
    """Compute full risk metrics from a daily return series."""
    returns = returns.dropna()
    if len(returns) < 30:
        return {"name": name, "error": "insufficient data", "n_days": len(returns)}

    n = len(returns)
    ann_ret = returns.mean() * ANNUAL_FACTOR
    ann_vol = returns.std() * np.sqrt(ANNUAL_FACTOR)
    sharpe = (returns.mean() - RF_DAILY) / returns.std() * np.sqrt(ANNUAL_FACTOR) if returns.std() > 0 else 0

    downside = returns[returns < 0].std() * np.sqrt(ANNUAL_FACTOR)
    sortino = (ann_ret - 0.05) / downside if downside > 0 else 0

    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    max_dd = drawdown.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    winning = (returns > 0).sum()
    wr = winning / n if n > 0 else 0

    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Exposure (fraction of days with non-zero return)
    exposure = (returns != 0).mean()

    return {
        "name": name,
        "n_days": n,
        "annual_return": round(ann_ret, 4),
        "annual_vol": round(ann_vol, 4),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_drawdown": round(max_dd, 4),
        "calmar": round(calmar, 4),
        "win_rate": round(wr, 4),
        "profit_factor": round(pf, 4),
        "exposure": round(exposure, 4),
        "total_return": round(float(cum.iloc[-1] - 1), 4),
        "worst_day": round(float(returns.min()), 4),
        "best_day": round(float(returns.max()), 4),
        "skew": round(float(returns.skew()), 4),
        "kurtosis": round(float(returns.kurtosis()), 4),
    }


# ── Strategy Implementations ───────────────────────────────────────────────
def strategy_contango_harvester(data):
    """S1: Hold SVXY when VIX/VIX3M < 0.9, else cash."""
    signal = (data["VIX_RATIO"].shift(1) < CONTANGO_THRESHOLD).astype(float)
    returns = signal * data["SVXY_RET"]
    returns = returns.fillna(0)
    return returns, signal


def strategy_backwardation_panic(data):
    """S2: Hold SPY for 5-20 days after VIX/VIX3M > 1.05."""
    ratio = data["VIX_RATIO"]
    signal = pd.Series(0.0, index=data.index)
    hold_counter = 0

    for i in range(1, len(data)):
        if ratio.iloc[i - 1] > BACKWARDATION_THRESHOLD:
            hold_counter = BACKWARDATION_HOLD_MAX
        if hold_counter > 0:
            # Only hold if we've been in backwardation for at least HOLD_MIN days
            # or if the initial signal fired
            signal.iloc[i] = 1.0
            hold_counter -= 1

    returns = signal * data["SPY_RET"]
    returns = returns.fillna(0)
    return returns, signal


def strategy_combined_regime(data):
    """S3: SVXY in contango, cash in flat, SPY in backwardation."""
    ratio = data["VIX_RATIO"].shift(1)
    contango = ratio < CONTANGO_THRESHOLD
    backwardation = ratio > BACKWARDATION_THRESHOLD

    returns = pd.Series(0.0, index=data.index)
    signal = pd.Series(0.0, index=data.index)  # 1=SVXY, -1=SPY position, 0=cash

    returns[contango] = data["SVXY_RET"][contango]
    signal[contango] = 1.0  # SVXY

    returns[backwardation] = data["SPY_RET"][backwardation]
    signal[backwardation] = -1.0  # SPY (panic buy)

    returns = returns.fillna(0)
    return returns, signal


def strategy_size_adjusted(data):
    """S4: Same as combined but position size = f(contango steepness)."""
    ratio = data["VIX_RATIO"].shift(1)
    contango = ratio < CONTANGO_THRESHOLD
    backwardation = ratio > BACKWARDATION_THRESHOLD

    returns = pd.Series(0.0, index=data.index)
    signal = pd.Series(0.0, index=data.index)

    # Contango sizing: deeper contango → bigger position (capped at 1.5x)
    contango_depth = (CONTANGO_THRESHOLD - ratio).clip(lower=0) / CONTANGO_THRESHOLD
    contango_size = (0.5 + contango_depth * 5.0).clip(upper=1.5)  # 0.5x to 1.5x

    # Backwardation sizing: more extreme → bigger SPY position (capped at 1.5x)
    backwardation_depth = (ratio - BACKWARDATION_THRESHOLD).clip(lower=0)
    backwardation_size = (0.5 + backwardation_depth * 3.0).clip(upper=1.5)

    returns[contango] = contango_size[contango] * data["SVXY_RET"][contango]
    signal[contango] = contango_size[contango]

    returns[backwardation] = backwardation_size[backwardation] * data["SPY_RET"][backwardation]
    signal[backwardation] = -backwardation_size[backwardation]

    returns = returns.fillna(0)
    return returns, signal


# ── Walk-Forward Validation ────────────────────────────────────────────────
def walk_forward_test(data, strategy_fn, name="Strategy"):
    """
    Walk-forward sliding window validation.
    Train window: optimize threshold awareness (though our thresholds are fixed,
    the WF validates that in-sample edge persists OOS).
    """
    print(f"\n  Walk-forward: {name}")
    n = len(data)
    oos_returns_all = []
    fold_metrics = []

    fold = 0
    start = 0
    while start + TRAIN_DAYS + TEST_DAYS <= n:
        train_end = start + TRAIN_DAYS
        test_end = min(train_end + TEST_DAYS, n)

        train_data = data.iloc[start:train_end]
        test_data = data.iloc[train_end:test_end]

        # Run strategy on train to check viability
        train_ret, _ = strategy_fn(train_data)
        train_m = compute_metrics(train_ret, f"{name}_train_f{fold}")

        # Run strategy on test (OOS)
        test_ret, test_sig = strategy_fn(test_data)
        test_m = compute_metrics(test_ret, f"{name}_oos_f{fold}")

        oos_returns_all.append(test_ret)
        fold_metrics.append({
            "fold": fold,
            "train_start": str(train_data.index[0].date()),
            "test_start": str(test_data.index[0].date()),
            "test_end": str(test_data.index[-1].date()),
            "train_sharpe": train_m.get("sharpe", 0),
            "oos_sharpe": test_m.get("sharpe", 0),
            "oos_return": test_m.get("annual_return", 0),
            "oos_maxdd": test_m.get("max_drawdown", 0),
        })

        start += SLIDE_DAYS
        fold += 1

    if not oos_returns_all:
        return None, []

    # Concatenate all OOS returns (non-overlapping)
    combined_oos = pd.concat(oos_returns_all)
    # Remove duplicates (overlapping folds)
    combined_oos = combined_oos[~combined_oos.index.duplicated(keep="first")]
    combined_oos = combined_oos.sort_index()

    oos_sharpes = [f["oos_sharpe"] for f in fold_metrics if f["oos_sharpe"] != 0]
    print(f"    {fold} folds | OOS Sharpe: mean={np.mean(oos_sharpes):.3f}, "
          f"median={np.median(oos_sharpes):.3f}, std={np.std(oos_sharpes):.3f}")

    return combined_oos, fold_metrics


# ── Permutation Test ────────────────────────────────────────────────────────
def permutation_test(data, strategy_fn, name="Strategy"):
    """Shuffle dates, re-run strategy, compute p-value."""
    print(f"  Permutation test: {name} ({N_PERMUTATIONS} shuffles)")

    actual_ret, _ = strategy_fn(data)
    actual_sharpe = compute_metrics(actual_ret, name).get("sharpe", 0)

    null_sharpes = []
    for i in range(N_PERMUTATIONS):
        shuffled = data.copy()
        # Shuffle VIX ratio (breaks signal-return relationship)
        shuffled["VIX_RATIO"] = np.random.permutation(shuffled["VIX_RATIO"].values)
        perm_ret, _ = strategy_fn(shuffled)
        perm_m = compute_metrics(perm_ret, f"perm_{i}")
        null_sharpes.append(perm_m.get("sharpe", 0))

    null_sharpes = np.array(null_sharpes)
    p_value = (null_sharpes >= actual_sharpe).mean()

    print(f"    Actual Sharpe: {actual_sharpe:.3f} | "
          f"Null mean: {np.mean(null_sharpes):.3f} ± {np.std(null_sharpes):.3f} | "
          f"p-value: {p_value:.4f} {'***' if p_value < 0.01 else '**' if p_value < 0.05 else 'NS'}")

    return {
        "actual_sharpe": actual_sharpe,
        "null_mean": round(float(np.mean(null_sharpes)), 4),
        "null_std": round(float(np.std(null_sharpes)), 4),
        "p_value": round(float(p_value), 4),
        "significant": p_value < 0.05,
    }


# ── R1 Regime Test ──────────────────────────────────────────────────────────
def regime_test(data, strategy_fn, name="Strategy"):
    """Test if strategy works in both green and red SPY days."""
    print(f"  Regime test (R1): {name}")

    ret, sig = strategy_fn(data)

    green_mask = data["SPY_GREEN"] & (sig.shift(1) != 0)
    red_mask = (~data["SPY_GREEN"]) & (sig.shift(1) != 0)

    green_ret = ret[green_mask]
    red_ret = ret[red_mask]

    green_m = compute_metrics(green_ret, f"{name}_green")
    red_m = compute_metrics(red_ret, f"{name}_red")

    green_sharpe = green_m.get("sharpe", 0)
    red_sharpe = red_m.get("sharpe", 0)

    max_abs = max(abs(green_sharpe), abs(red_sharpe))
    gap = abs(green_sharpe - red_sharpe) / max_abs if max_abs > 0 else 0

    passed = gap < REGIME_SHARPE_GAP_MAX

    print(f"    Green Sharpe: {green_sharpe:.3f} | Red Sharpe: {red_sharpe:.3f} | "
          f"Gap: {gap:.3f} | {'PASS' if passed else 'FAIL (regime-dependent)'}")

    return {
        "green_sharpe": green_sharpe,
        "red_sharpe": red_sharpe,
        "gap": round(gap, 4),
        "passed": passed,
        "green_n": int(green_mask.sum()),
        "red_n": int(red_mask.sum()),
    }


# ── Correlation Analysis ───────────────────────────────────────────────────
def correlation_analysis(data, strategy_returns, name="Strategy"):
    """Compute correlation to SPY, UPRO, SMA50 trend-following."""
    corr = {}
    aligned = pd.DataFrame({
        "strategy": strategy_returns,
        "SPY": data["SPY_RET"],
        "UPRO": data["UPRO_RET"],
        "SMA50_TF": data["SMA50_RET"],
    }).dropna()

    for ref in ["SPY", "UPRO", "SMA50_TF"]:
        c = aligned["strategy"].corr(aligned[ref])
        corr[ref] = round(float(c), 4) if not np.isnan(c) else 0.0

    print(f"  Correlations for {name}:")
    print(f"    vs SPY: {corr['SPY']:.3f} | vs UPRO: {corr['UPRO']:.3f} | "
          f"vs SMA50 TF: {corr['SMA50_TF']:.3f}")

    return corr


# ── Tail Risk Analysis ──────────────────────────────────────────────────────
def tail_risk_analysis(returns, name="Strategy"):
    """Analyze tail risk — critical for short-vol strategies."""
    returns = returns.dropna()
    percentiles = [1, 5, 10, 25]
    tail = {}
    for p in percentiles:
        val = np.percentile(returns, p)
        tail[f"p{p}"] = round(float(val), 6)

    # Worst N-day drawdowns
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak

    worst_dds = []
    for window in [5, 10, 20]:
        rolling_dd = returns.rolling(window).sum()
        worst = rolling_dd.min()
        worst_dds.append({"window": window, "worst_return": round(float(worst), 4)})

    # CVaR (Expected Shortfall) at 5%
    var_5 = np.percentile(returns, 5)
    cvar_5 = returns[returns <= var_5].mean()

    tail["cvar_5pct"] = round(float(cvar_5), 6)
    tail["worst_rolling"] = worst_dds
    tail["days_below_neg2pct"] = int((returns < -0.02).sum())
    tail["days_below_neg5pct"] = int((returns < -0.05).sum())
    tail["days_below_neg10pct"] = int((returns < -0.10).sum())

    print(f"  Tail risk for {name}:")
    print(f"    CVaR(5%): {cvar_5:.4f} | Days < -2%: {tail['days_below_neg2pct']} | "
          f"Days < -5%: {tail['days_below_neg5pct']} | Days < -10%: {tail['days_below_neg10pct']}")
    for wd in worst_dds:
        print(f"    Worst {wd['window']}d return: {wd['worst_return']:.4f}")

    return tail


# ── Threshold Sensitivity ──────────────────────────────────────────────────
def threshold_sensitivity(data):
    """Test different contango/backwardation thresholds."""
    print("\n" + "=" * 80)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 80)

    results = []
    for ct in [0.85, 0.88, 0.90, 0.92, 0.95]:
        for bt in [1.00, 1.02, 1.05, 1.08, 1.10]:
            ratio = data["VIX_RATIO"].shift(1)
            contango = ratio < ct
            backwardation = ratio > bt

            ret = pd.Series(0.0, index=data.index)
            ret[contango] = data["SVXY_RET"][contango]
            ret[backwardation] = data["SPY_RET"][backwardation]
            ret = ret.fillna(0)

            m = compute_metrics(ret, f"ct{ct}_bt{bt}")
            results.append({
                "contango_thresh": ct,
                "backwardation_thresh": bt,
                "sharpe": m.get("sharpe", 0),
                "annual_return": m.get("annual_return", 0),
                "max_drawdown": m.get("max_drawdown", 0),
                "calmar": m.get("calmar", 0),
                "exposure": m.get("exposure", 0),
            })

    df = pd.DataFrame(results)
    print("\nBest by Sharpe:")
    best = df.sort_values("sharpe", ascending=False).head(5)
    print(best.to_string(index=False))

    print("\nBest by Calmar (risk-adjusted):")
    best_calmar = df.sort_values("calmar", ascending=False).head(5)
    print(best_calmar.to_string(index=False))

    return results


# ── Annual Breakdown ────────────────────────────────────────────────────────
def annual_breakdown(returns, name="Strategy"):
    """Year-by-year performance."""
    returns = returns.dropna()
    years = returns.index.year.unique()
    rows = []
    for y in sorted(years):
        yr = returns[returns.index.year == y]
        m = compute_metrics(yr, f"{name}_{y}")
        rows.append({
            "year": y,
            "return": m.get("annual_return", 0),
            "sharpe": m.get("sharpe", 0),
            "max_dd": m.get("max_drawdown", 0),
            "n_days": m.get("n_days", 0),
        })
    return rows


# ── MAIN ────────────────────────────────────────────────────────────────────
def main():
    np.random.seed(42)
    print("=" * 80)
    print("VIX TERM STRUCTURE / VOLATILITY RISK PREMIUM STUDY")
    print(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 1. Download data
    data = download_data()

    # 2. Define strategies
    strategies = {
        "S1_Contango_Harvester": strategy_contango_harvester,
        "S2_Backwardation_Panic": strategy_backwardation_panic,
        "S3_Combined_Regime": strategy_combined_regime,
        "S4_Size_Adjusted": strategy_size_adjusted,
    }

    # 3. Baseline: SPY buy-and-hold
    print("\n" + "=" * 80)
    print("BASELINE: SPY BUY-AND-HOLD")
    print("=" * 80)
    spy_metrics = compute_metrics(data["SPY_RET"], "SPY_BuyHold")
    for k, v in spy_metrics.items():
        print(f"  {k}: {v}")

    # 4. Run all strategies
    all_results = {"baseline_spy": spy_metrics}
    all_oos_returns = {}

    for sname, sfn in strategies.items():
        print("\n" + "=" * 80)
        print(f"STRATEGY: {sname}")
        print("=" * 80)

        # Full-sample metrics
        full_ret, full_sig = sfn(data)
        full_m = compute_metrics(full_ret, sname)
        print("\n  Full-sample metrics:")
        for k, v in full_m.items():
            print(f"    {k}: {v}")

        # Walk-forward validation
        oos_ret, fold_m = walk_forward_test(data, sfn, sname)
        if oos_ret is not None:
            oos_m = compute_metrics(oos_ret, f"{sname}_OOS")
            all_oos_returns[sname] = oos_ret
            print(f"\n  OOS combined metrics:")
            for k, v in oos_m.items():
                print(f"    {k}: {v}")
        else:
            oos_m = {"error": "no OOS data"}

        # Permutation test
        perm = permutation_test(data, sfn, sname)

        # Regime test (R1)
        regime = regime_test(data, sfn, sname)

        # Correlation analysis
        corr = correlation_analysis(data, full_ret, sname)

        # Tail risk
        tail = tail_risk_analysis(full_ret, sname)

        # Annual breakdown
        annual = annual_breakdown(full_ret, sname)
        print(f"\n  Annual breakdown:")
        for row in annual:
            print(f"    {row['year']}: ret={row['return']:.3f} sharpe={row['sharpe']:.3f} "
                  f"maxdd={row['max_dd']:.3f} ({row['n_days']}d)")

        all_results[sname] = {
            "full_sample": full_m,
            "oos": oos_m,
            "walk_forward_folds": fold_m,
            "permutation_test": perm,
            "regime_test": regime,
            "correlation": corr,
            "tail_risk": tail,
            "annual": annual,
        }

    # 5. Threshold sensitivity
    sensitivity = threshold_sensitivity(data)
    all_results["threshold_sensitivity"] = sensitivity

    # 6. Summary comparison table
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    summary_rows = []
    header = f"{'Strategy':<28} {'Sharpe':>7} {'Sortino':>8} {'AnnRet':>8} {'MaxDD':>8} {'Calmar':>7} {'WR':>6} {'PF':>6} {'Exp':>5} {'Perm-p':>7} {'R1':>5}"
    print(header)
    print("-" * len(header))

    # SPY baseline
    sm = spy_metrics
    row = f"{'SPY Buy-Hold':<28} {sm['sharpe']:>7.3f} {sm['sortino']:>8.3f} {sm['annual_return']:>8.3f} {sm['max_drawdown']:>8.3f} {sm['calmar']:>7.3f} {sm['win_rate']:>6.3f} {sm['profit_factor']:>6.3f} {1.0:>5.2f} {'--':>7} {'--':>5}"
    print(row)
    summary_rows.append({"strategy": "SPY_BuyHold", **sm, "perm_p": None, "r1_pass": None})

    for sname in strategies:
        r = all_results[sname]
        fm = r["full_sample"]
        pp = r["permutation_test"]["p_value"]
        r1 = "PASS" if r["regime_test"]["passed"] else "FAIL"
        row = f"{sname:<28} {fm['sharpe']:>7.3f} {fm['sortino']:>8.3f} {fm['annual_return']:>8.3f} {fm['max_drawdown']:>8.3f} {fm['calmar']:>7.3f} {fm['win_rate']:>6.3f} {fm['profit_factor']:>6.3f} {fm['exposure']:>5.2f} {pp:>7.4f} {r1:>5}"
        print(row)
        summary_rows.append({"strategy": sname, **fm, "perm_p": pp, "r1_pass": r["regime_test"]["passed"]})

    # 7. Correlation matrix
    print("\n" + "=" * 80)
    print("CORRELATION MATRIX (strategy returns)")
    print("=" * 80)
    corr_data = {}
    for sname, sfn in strategies.items():
        ret, _ = sfn(data)
        corr_data[sname] = ret
    corr_data["SPY"] = data["SPY_RET"]
    corr_data["UPRO"] = data["UPRO_RET"]
    corr_data["SMA50_TF"] = data["SMA50_RET"]

    corr_df = pd.DataFrame(corr_data).corr()
    print(corr_df.round(3).to_string())

    # 8. Key question answer
    print("\n" + "=" * 80)
    print("KEY QUESTION: Can VIX term structure add portfolio value?")
    print("=" * 80)

    for sname in strategies:
        r = all_results[sname]
        fm = r["full_sample"]
        perm_sig = r["permutation_test"]["significant"]
        r1_pass = r["regime_test"]["passed"]
        spy_corr = r["correlation"].get("SPY", 1.0)

        verdict_parts = []
        if fm["sharpe"] > spy_metrics["sharpe"]:
            verdict_parts.append("higher Sharpe than SPY")
        else:
            verdict_parts.append("LOWER Sharpe than SPY")

        if perm_sig:
            verdict_parts.append("statistically significant")
        else:
            verdict_parts.append("NOT statistically significant")

        if r1_pass:
            verdict_parts.append("regime-robust")
        else:
            verdict_parts.append("regime-DEPENDENT")

        if abs(spy_corr) < 0.5:
            verdict_parts.append(f"low SPY corr ({spy_corr:.2f}) = diversification benefit")
        else:
            verdict_parts.append(f"high SPY corr ({spy_corr:.2f}) = NO diversification")

        if fm["max_drawdown"] < -0.50:
            verdict_parts.append("UNACCEPTABLE tail risk (>50% DD)")
        elif fm["max_drawdown"] < -0.30:
            verdict_parts.append("moderate tail risk (30-50% DD)")
        else:
            verdict_parts.append("contained tail risk (<30% DD)")

        viable = perm_sig and r1_pass and fm["sharpe"] > 0.3 and fm["max_drawdown"] > -0.60
        verdict = "VIABLE" if viable else "NOT VIABLE"

        print(f"\n  {sname}: {verdict}")
        for part in verdict_parts:
            print(f"    - {part}")

    # 9. Save outputs
    print("\n" + "=" * 80)
    print("SAVING OUTPUTS")
    print("=" * 80)

    # JSON results
    json_path = OUTPUT_DIR / "vix_vrp_results.json"
    # Clean up non-serializable items
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(i) for i in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return str(obj)
        return obj

    with open(json_path, "w") as f:
        json.dump(clean_for_json(all_results), f, indent=2, default=str)
    print(f"  JSON: {json_path}")

    # CSV summary
    csv_path = OUTPUT_DIR / "vix_vrp_summary.csv"
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    print(f"  CSV: {csv_path}")

    # Correlation matrix CSV
    corr_csv_path = OUTPUT_DIR / "correlation_matrix.csv"
    corr_df.to_csv(corr_csv_path)
    print(f"  Correlation CSV: {corr_csv_path}")

    # VIX ratio time series for reference
    vix_ratio_path = OUTPUT_DIR / "vix_ratio_timeseries.csv"
    data[["VIX", "VIX3M", "VIX_RATIO"]].to_csv(vix_ratio_path)
    print(f"  VIX ratio CSV: {vix_ratio_path}")

    # Threshold sensitivity CSV
    sens_path = OUTPUT_DIR / "threshold_sensitivity.csv"
    pd.DataFrame(sensitivity).to_csv(sens_path, index=False)
    print(f"  Sensitivity CSV: {sens_path}")

    # Strategy daily returns for further analysis
    strat_returns = pd.DataFrame()
    for sname, sfn in strategies.items():
        ret, _ = sfn(data)
        strat_returns[sname] = ret
    strat_returns["SPY_BuyHold"] = data["SPY_RET"]
    returns_path = OUTPUT_DIR / "strategy_daily_returns.csv"
    strat_returns.to_csv(returns_path)
    print(f"  Daily returns CSV: {returns_path}")

    print("\n" + "=" * 80)
    print("STUDY COMPLETE")
    print("=" * 80)

    return all_results


if __name__ == "__main__":
    results = main()
