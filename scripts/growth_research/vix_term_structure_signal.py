#!/usr/bin/env python3
"""
VIX Term Structure as a Regime Signal for Gameplan v3
=====================================================
Tests whether VIX futures term structure (contango/backwardation)
improves the Gameplan v3 confluence gate for UPRO switching.

Key hypothesis:
  - VIX in contango (VIXY declining relative to trend) = complacency = bullish
  - VIX in backwardation (VIXY rising sharply) = fear = bearish
  - This captures a DIFFERENT dimension than vol level or price trends

Proxied via:
  - VIXY ETF (short-term VIX futures)
  - VIXY slope vs its own SMA as term structure proxy
  - VIXY/SPY ratio as a normalized fear gauge

Tests:
  1. Standalone term structure signal vs SPY buy-hold
  2. Adding term structure to Gameplan v3 confluence gate
  3. Walk-forward validation (3yr train, 1yr OOS)
  4. Full adversarial suite (permutation, sub-period, outlier, R1)

HC compliance:
  - SLIDING window only (HC #0)
  - Risk-adjusted metrics primary (HC #69)
  - Adversarial validation (HC #709 R2, HC #712 R5)
"""
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Data fetching ────────────────────────────────────────────────────────────

def fetch_data() -> pd.DataFrame:
    """Fetch SPY, UPRO, VIXY data."""
    tickers = ["SPY", "UPRO", "VIXY"]
    print(f"Fetching data for {tickers}...")

    data = {}
    for t in tickers:
        df = yf.download(t, start="2011-10-04", end="2026-07-17", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        data[t] = df["Close"].rename(t)

    combined = pd.concat(data.values(), axis=1).dropna()
    print(f"  Got {len(combined)} trading days ({combined.index[0].strftime('%Y-%m-%d')} to {combined.index[-1].strftime('%Y-%m-%d')})")
    return combined


# ── Signal computation ───────────────────────────────────────────────────────

def compute_vix_term_structure_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute VIX term structure signals from VIXY behavior.

    Returns DataFrame with columns:
      - vixy_slope_5: 5-day slope of VIXY (short-term direction)
      - vixy_slope_20: 20-day slope of VIXY (medium-term direction)
      - vixy_above_sma20: VIXY > 20d SMA (elevated fear)
      - vixy_above_sma50: VIXY > 50d SMA (sustained fear)
      - vixy_contango_proxy: VIXY declining relative to its mean (contango = bullish)
      - vixy_spike: VIXY > 1.5x its 60d mean (panic spike)
      - vixy_collapse: VIXY < 0.8x its 60d mean (post-panic normalization)
      - term_structure_score: composite 0-1 score (higher = more bullish)
    """
    out = pd.DataFrame(index=df.index)
    vixy = df["VIXY"]

    # SMA calculations
    sma5 = vixy.rolling(5).mean()
    sma20 = vixy.rolling(20).mean()
    sma50 = vixy.rolling(50).mean()
    sma60 = vixy.rolling(60).mean()

    # Slopes (rate of change)
    out["vixy_roc_5"] = vixy.pct_change(5) * 100  # 5-day % change
    out["vixy_roc_20"] = vixy.pct_change(20) * 100  # 20-day % change

    # Relative position
    out["vixy_above_sma20"] = (vixy > sma20).astype(int)
    out["vixy_above_sma50"] = (vixy > sma50).astype(int)

    # Contango proxy: VIXY below its 60d mean = futures in contango = bullish
    vixy_rel = vixy / sma60
    out["vixy_rel_60"] = vixy_rel
    out["vixy_contango"] = (vixy_rel < 1.0).astype(int)  # 1 = contango (bullish)

    # Spike / collapse
    out["vixy_spike"] = (vixy_rel > 1.5).astype(int)  # panic
    out["vixy_collapse"] = (vixy_rel < 0.8).astype(int)  # post-panic normalization

    # Composite term structure score (0-1, higher = more bullish)
    # Bullish signals: contango, below SMAs, declining
    score = pd.Series(0.0, index=df.index)
    score += out["vixy_contango"] * 0.25  # In contango
    score += (1 - out["vixy_above_sma20"]) * 0.20  # Below 20d SMA
    score += (1 - out["vixy_above_sma50"]) * 0.15  # Below 50d SMA
    score += (out["vixy_roc_5"] < 0).astype(float) * 0.20  # 5d declining
    score += (out["vixy_roc_20"] < 0).astype(float) * 0.20  # 20d declining
    # Penalty for spikes
    score -= out["vixy_spike"] * 0.30  # Strong bearish signal
    score = score.clip(0, 1)
    out["term_score"] = score

    return out


# ── Gameplan v3 confluence (copied from tested code) ─────────────────────────

def compute_gameplan_v3_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Standard Gameplan v3 confluence gate."""
    spy = df["SPY"]
    out = pd.DataFrame(index=df.index)

    # Short: 5d momentum + 10d RSI > 50
    mom5 = spy.pct_change(5)
    delta = spy.diff()
    gain = delta.clip(lower=0).rolling(10).mean()
    loss = (-delta.clip(upper=0)).rolling(10).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi10 = 100 - (100 / (1 + rs))
    out["short_signal"] = ((mom5 > 0).astype(float) + (rsi10 > 50).astype(float)) / 2

    # Medium: 20/50 MA cross + 21d vol < 15%
    sma20 = spy.rolling(20).mean()
    sma50 = spy.rolling(50).mean()
    vol21 = spy.pct_change().rolling(21).std() * np.sqrt(252) * 100
    out["medium_signal"] = ((sma20 > sma50).astype(float) + (vol21 < 15).astype(float)) / 2

    # Long: 200d MA slope + 63d vol trend declining
    sma200 = spy.rolling(200).mean()
    slope200 = sma200.pct_change(20) * 100  # 20-day slope of 200 SMA
    vol63 = spy.pct_change().rolling(63).std() * np.sqrt(252) * 100
    vol63_slope = vol63.diff(20)  # is vol declining?
    out["long_signal"] = ((slope200 > 0).astype(float) + (vol63_slope < 0).astype(float)) / 2

    out["confluence_score"] = (out["short_signal"] + out["medium_signal"] + out["long_signal"]) * 3 / 1.5
    # Scale to 0-3
    out["confluence_score"] = out["confluence_score"].clip(0, 3)

    # Vol for regime
    out["vol_21d"] = vol21

    return out


# ── Backtest engines ─────────────────────────────────────────────────────────

def backtest_gameplan_v3(df: pd.DataFrame, signals: pd.DataFrame,
                          entry_thresh: float = 2.5, exit_thresh: float = 2.0,
                          vol_high: float = 30, vol_crisis: float = 30) -> pd.DataFrame:
    """Gameplan v3 backtest: confluence gate + vol override."""
    spy_ret = df["SPY"].pct_change()
    upro_ret = df["UPRO"].pct_change()

    position = pd.Series("SPY", index=df.index)
    in_upro = False

    for i in range(1, len(df)):
        score = signals["confluence_score"].iloc[i-1]
        vol = signals["vol_21d"].iloc[i-1]

        if vol > vol_crisis:
            position.iloc[i] = "GLD"
            in_upro = False
        elif vol > 15:
            position.iloc[i] = "SPY"
            in_upro = False
        elif in_upro and score >= exit_thresh:
            position.iloc[i] = "UPRO"
        elif not in_upro and score >= entry_thresh:
            position.iloc[i] = "UPRO"
            in_upro = True
        else:
            position.iloc[i] = "SPY"
            if score < exit_thresh:
                in_upro = False

    # Compute returns based on position
    strat_ret = pd.Series(0.0, index=df.index)
    strat_ret[position == "UPRO"] = upro_ret[position == "UPRO"]
    strat_ret[position == "SPY"] = spy_ret[position == "SPY"]
    # GLD returns (approximate as 0 for simplicity — conservative)

    return pd.DataFrame({
        "returns": strat_ret,
        "position": position,
        "cum_returns": (1 + strat_ret).cumprod()
    })


def backtest_v3_plus_term(df: pd.DataFrame, v3_signals: pd.DataFrame,
                           term_signals: pd.DataFrame,
                           entry_thresh: float = 2.5, exit_thresh: float = 2.0,
                           term_thresh: float = 0.5) -> pd.DataFrame:
    """Gameplan v3 + VIX term structure enhancement."""
    spy_ret = df["SPY"].pct_change()
    upro_ret = df["UPRO"].pct_change()

    position = pd.Series("SPY", index=df.index)
    in_upro = False

    for i in range(1, len(df)):
        score = v3_signals["confluence_score"].iloc[i-1]
        vol = v3_signals["vol_21d"].iloc[i-1]
        term_score = term_signals["term_score"].iloc[i-1]

        if vol > 30:
            position.iloc[i] = "GLD"
            in_upro = False
        elif vol > 15:
            position.iloc[i] = "SPY"
            in_upro = False
        # Enhanced gate: require BOTH confluence AND term structure
        elif in_upro:
            # Stay in UPRO if confluence still above exit AND term not bearish
            if score >= exit_thresh and term_score >= (term_thresh * 0.7):
                position.iloc[i] = "UPRO"
            else:
                position.iloc[i] = "SPY"
                in_upro = False
        elif not in_upro:
            # Enter UPRO only if confluence high AND term structure bullish
            if score >= entry_thresh and term_score >= term_thresh:
                position.iloc[i] = "UPRO"
                in_upro = True
            else:
                position.iloc[i] = "SPY"

    strat_ret = pd.Series(0.0, index=df.index)
    strat_ret[position == "UPRO"] = upro_ret[position == "UPRO"]
    strat_ret[position == "SPY"] = spy_ret[position == "SPY"]

    return pd.DataFrame({
        "returns": strat_ret,
        "position": position,
        "cum_returns": (1 + strat_ret).cumprod()
    })


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(returns: pd.Series) -> dict:
    """Compute risk-adjusted metrics."""
    if len(returns) < 10 or returns.std() == 0:
        return {"sharpe": 0, "sortino": 0, "cagr_pct": 0, "max_dd_pct": 0, "win_rate": 0, "profit_factor": 0}

    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = returns[returns < 0].std() * np.sqrt(252) if (returns < 0).any() else 1e-6
    sortino = ann_ret / downside

    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min() * 100

    n_years = len(returns) / 252
    cagr = (cum.iloc[-1] ** (1 / n_years) - 1) * 100 if n_years > 0 and cum.iloc[-1] > 0 else 0

    wr = (returns > 0).sum() / ((returns != 0).sum() or 1)

    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')

    switches = 0
    # Don't compute switches here — it's slow

    return {
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "cagr_pct": round(cagr, 2),
        "max_dd_pct": round(max_dd, 2),
        "win_rate": round(wr, 4),
        "profit_factor": round(pf, 4),
        "n_days": len(returns),
    }


# ── Walk-forward validation ─────────────────────────────────────────────────

def walk_forward_test(df: pd.DataFrame, train_days: int = 756, test_days: int = 252,
                       step_days: int = 252) -> dict:
    """Walk-forward validation with parameter optimization."""
    print(f"\nWalk-forward: train={train_days}d, test={test_days}d, step={step_days}d")

    v3_signals = compute_gameplan_v3_signals(df)
    term_signals = compute_vix_term_structure_signals(df)

    # Warmup period for indicators
    warmup = 200
    valid_start = warmup

    fold_results = []
    i = valid_start + train_days
    fold_num = 0

    # Parameter grid for term threshold
    term_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    while i + test_days <= len(df):
        fold_num += 1
        train_idx = slice(i - train_days, i)
        test_idx = slice(i, i + test_days)

        # Optimize on train
        best_sharpe = -999
        best_thresh = 0.5

        for tt in term_thresholds:
            bt = backtest_v3_plus_term(df.iloc[train_idx], v3_signals.iloc[train_idx],
                                        term_signals.iloc[train_idx], term_thresh=tt)
            m = compute_metrics(bt["returns"].dropna())
            if m["sharpe"] > best_sharpe:
                best_sharpe = m["sharpe"]
                best_thresh = tt

        # Test OOS with best params
        bt_oos = backtest_v3_plus_term(df.iloc[test_idx], v3_signals.iloc[test_idx],
                                        term_signals.iloc[test_idx], term_thresh=best_thresh)
        oos_metrics = compute_metrics(bt_oos["returns"].dropna())

        # Also run v3 baseline on same OOS window
        bt_v3 = backtest_gameplan_v3(df.iloc[test_idx], v3_signals.iloc[test_idx])
        v3_metrics = compute_metrics(bt_v3["returns"].dropna())

        fold_results.append({
            "fold": fold_num,
            "train_end": df.index[i-1].strftime("%Y-%m-%d"),
            "test_start": df.index[i].strftime("%Y-%m-%d"),
            "test_end": df.index[min(i + test_days - 1, len(df)-1)].strftime("%Y-%m-%d"),
            "best_term_thresh": best_thresh,
            "train_sharpe": round(best_sharpe, 3),
            "oos_sharpe": oos_metrics["sharpe"],
            "oos_sortino": oos_metrics["sortino"],
            "oos_cagr": oos_metrics["cagr_pct"],
            "oos_maxdd": oos_metrics["max_dd_pct"],
            "v3_baseline_sharpe": v3_metrics["sharpe"],
            "v3_baseline_sortino": v3_metrics["sortino"],
            "improvement": round(oos_metrics["sharpe"] - v3_metrics["sharpe"], 4),
        })

        print(f"  Fold {fold_num}: thresh={best_thresh}, OOS Sharpe {oos_metrics['sharpe']:.3f} "
              f"vs v3 {v3_metrics['sharpe']:.3f} (Δ{oos_metrics['sharpe'] - v3_metrics['sharpe']:+.3f})")

        i += step_days

    # Summary
    improvements = [f["improvement"] for f in fold_results]
    oos_sharpes = [f["oos_sharpe"] for f in fold_results]
    v3_sharpes = [f["v3_baseline_sharpe"] for f in fold_results]
    win_count = sum(1 for x in improvements if x > 0)

    summary = {
        "n_folds": len(fold_results),
        "mean_oos_sharpe": round(np.mean(oos_sharpes), 4),
        "mean_v3_sharpe": round(np.mean(v3_sharpes), 4),
        "mean_improvement": round(np.mean(improvements), 4),
        "folds_beating_v3": f"{win_count}/{len(fold_results)}",
        "pct_beating_v3": round(win_count / len(fold_results) * 100, 1),
        "fold_details": fold_results,
    }

    return summary


# ── Adversarial tests ────────────────────────────────────────────────────────

def run_adversarial(df: pd.DataFrame, n_perms: int = 200) -> dict:
    """Full adversarial suite."""
    print("\n=== Adversarial Validation ===")

    v3_signals = compute_gameplan_v3_signals(df)
    term_signals = compute_vix_term_structure_signals(df)

    # Real backtest
    bt_enhanced = backtest_v3_plus_term(df, v3_signals, term_signals)
    real_metrics = compute_metrics(bt_enhanced["returns"].dropna())
    real_sharpe = real_metrics["sharpe"]

    bt_v3 = backtest_gameplan_v3(df, v3_signals)
    v3_metrics = compute_metrics(bt_v3["returns"].dropna())

    print(f"  Real: Enhanced Sharpe {real_sharpe:.4f} vs v3 {v3_metrics['sharpe']:.4f}")

    # 1. Permutation test — shuffle VIX term structure signal
    print(f"  Running permutation test ({n_perms} shuffles)...")
    perm_sharpes = []
    for _ in range(n_perms):
        shuffled_term = term_signals.copy()
        # Shuffle term_score independently to break any real signal
        shuffled_term["term_score"] = np.random.permutation(shuffled_term["term_score"].values)
        bt_perm = backtest_v3_plus_term(df, v3_signals, shuffled_term)
        m = compute_metrics(bt_perm["returns"].dropna())
        perm_sharpes.append(m["sharpe"])

    p_value = np.mean([s >= real_sharpe for s in perm_sharpes])
    print(f"  Permutation p={p_value:.3f} (real={real_sharpe:.3f}, perm mean={np.mean(perm_sharpes):.3f})")

    # 2. Sub-period consistency
    half = len(df) // 2
    bt_first = backtest_v3_plus_term(df.iloc[:half], v3_signals.iloc[:half], term_signals.iloc[:half])
    bt_second = backtest_v3_plus_term(df.iloc[half:], v3_signals.iloc[half:], term_signals.iloc[half:])
    m1 = compute_metrics(bt_first["returns"].dropna())
    m2 = compute_metrics(bt_second["returns"].dropna())

    sub_consistent = (m1["sharpe"] > 0 and m2["sharpe"] > 0) or (abs(m1["sharpe"] - m2["sharpe"]) / max(abs(m1["sharpe"]), abs(m2["sharpe"]), 0.01) < 0.80)
    print(f"  Sub-period: first={m1['sharpe']:.3f}, second={m2['sharpe']:.3f}, consistent={sub_consistent}")

    # 3. Outlier removal (remove 10 best days)
    returns = bt_enhanced["returns"].dropna()
    top10_idx = returns.nlargest(10).index
    trimmed = returns.drop(top10_idx)
    trimmed_metrics = compute_metrics(trimmed)
    outlier_robust = trimmed_metrics["sharpe"] > 0 and trimmed_metrics["sharpe"] > real_sharpe * 0.70
    print(f"  Outlier removal: trimmed Sharpe={trimmed_metrics['sharpe']:.3f} ({trimmed_metrics['sharpe']/real_sharpe*100:.0f}% of real), robust={outlier_robust}")

    # 4. R1 regime check
    spy_ret = df["SPY"].pct_change()
    green_days = spy_ret > 0
    red_days = spy_ret < 0

    strat_ret = bt_enhanced["returns"].dropna()
    green_sharpe = compute_metrics(strat_ret[green_days.reindex(strat_ret.index, fill_value=False)])["sharpe"]
    red_sharpe = compute_metrics(strat_ret[red_days.reindex(strat_ret.index, fill_value=False)])["sharpe"]

    if max(abs(green_sharpe), abs(red_sharpe)) > 0:
        regime_gap = abs(green_sharpe - red_sharpe) / max(abs(green_sharpe), abs(red_sharpe))
    else:
        regime_gap = 0

    r1_pass = regime_gap <= 0.50
    print(f"  R1: green={green_sharpe:.3f}, red={red_sharpe:.3f}, gap={regime_gap:.3f}, pass={r1_pass}")

    return {
        "real_sharpe": real_sharpe,
        "v3_baseline_sharpe": v3_metrics["sharpe"],
        "improvement": round(real_sharpe - v3_metrics["sharpe"], 4),
        "permutation": {
            "p_value": round(p_value, 4),
            "real_sharpe": real_sharpe,
            "perm_mean": round(np.mean(perm_sharpes), 4),
            "perm_std": round(np.std(perm_sharpes), 4),
            "pass": p_value < 0.05,
        },
        "sub_period": {
            "first_half_sharpe": m1["sharpe"],
            "second_half_sharpe": m2["sharpe"],
            "consistent": sub_consistent,
        },
        "outlier_removal": {
            "trimmed_sharpe": trimmed_metrics["sharpe"],
            "retention_pct": round(trimmed_metrics["sharpe"] / real_sharpe * 100, 1) if real_sharpe != 0 else 0,
            "robust": outlier_robust,
        },
        "regime": {
            "green_sharpe": green_sharpe,
            "red_sharpe": red_sharpe,
            "gap": round(regime_gap, 4),
            "r1_pass": r1_pass,
        },
        "all_pass": p_value < 0.05 and sub_consistent and outlier_robust,
    }


# ── Standalone signal analysis ──────────────────────────────────────────────

def analyze_term_structure_standalone(df: pd.DataFrame) -> dict:
    """Test term structure as a standalone signal."""
    print("\n=== Standalone Term Structure Analysis ===")

    term = compute_vix_term_structure_signals(df)
    spy_ret = df["SPY"].pct_change()
    upro_ret = df["UPRO"].pct_change()

    # Simple strategy: UPRO when term_score > 0.5, SPY otherwise
    results = {}
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        signal = term["term_score"].shift(1) >= thresh
        strat_ret = pd.Series(0.0, index=df.index)
        strat_ret[signal] = upro_ret[signal]
        strat_ret[~signal] = spy_ret[~signal]

        metrics = compute_metrics(strat_ret.dropna())
        pct_upro = signal.mean() * 100

        results[f"thresh_{thresh}"] = {
            **metrics,
            "pct_in_upro": round(pct_upro, 1),
        }
        print(f"  Thresh {thresh}: Sharpe={metrics['sharpe']:.3f}, Sortino={metrics['sortino']:.3f}, "
              f"CAGR={metrics['cagr_pct']:.1f}%, MaxDD={metrics['max_dd_pct']:.1f}%, "
              f"UPRO time={pct_upro:.0f}%")

    # SPY buy-hold baseline
    spy_bh = compute_metrics(spy_ret.dropna())
    print(f"  SPY B&H:  Sharpe={spy_bh['sharpe']:.3f}, CAGR={spy_bh['cagr_pct']:.1f}%")

    # UPRO buy-hold
    upro_bh = compute_metrics(upro_ret.dropna())
    print(f"  UPRO B&H: Sharpe={upro_bh['sharpe']:.3f}, CAGR={upro_bh['cagr_pct']:.1f}%")

    return {
        "by_threshold": results,
        "spy_buyhold": spy_bh,
        "upro_buyhold": upro_bh,
    }


# ── Term structure behavior analysis ────────────────────────────────────────

def analyze_term_structure_behavior(df: pd.DataFrame) -> dict:
    """Understand what the term structure signal captures."""
    print("\n=== Term Structure Behavior Analysis ===")

    term = compute_vix_term_structure_signals(df)
    spy_ret = df["SPY"].pct_change()

    # Forward returns by term score quintile
    term_score = term["term_score"].shift(1)  # Lagged for forward returns
    fwd_1d = spy_ret
    fwd_5d = spy_ret.rolling(5).sum()
    fwd_20d = spy_ret.rolling(20).sum()

    try:
        quintiles = pd.qcut(term_score.dropna(), 5, labels=False, duplicates="drop") + 1
    except ValueError:
        quintiles = pd.cut(term_score.dropna(), 5, labels=False, duplicates="drop") + 1

    print("\n  Forward SPY returns by term structure score quintile:")
    print(f"  {'Quintile':>10} {'1d_mean':>10} {'5d_mean':>10} {'20d_mean':>10} {'Count':>8}")

    quintile_data = {}
    for q in sorted(quintiles.dropna().unique()):
        mask = (quintiles == q).reindex(fwd_1d.index, fill_value=False)
        r1d = fwd_1d[mask].mean() * 100
        r5d = fwd_5d[mask].mean() * 100
        r20d = fwd_20d[mask].mean() * 100
        n = mask.sum()
        print(f"  {q:>10} {r1d:>10.3f}% {r5d:>10.3f}% {r20d:>10.3f}% {n:>8}")
        quintile_data[f"Q{q}"] = {"fwd_1d_pct": round(r1d, 4), "fwd_5d_pct": round(r5d, 4),
                                    "fwd_20d_pct": round(r20d, 4), "n": int(n)}

    # VIXY spike analysis — what happens after VIXY spikes?
    spikes = term["vixy_spike"] == 1
    n_spikes = spikes.sum()
    if n_spikes > 0:
        # Forward returns after spikes
        spike_fwd_1d = spy_ret[spikes].mean() * 100
        spike_fwd_5d = fwd_5d[spikes].mean() * 100
        spike_fwd_20d = fwd_20d[spikes].mean() * 100
        print(f"\n  After VIXY spikes (n={n_spikes}): 1d={spike_fwd_1d:.3f}%, 5d={spike_fwd_5d:.3f}%, 20d={spike_fwd_20d:.3f}%")

    # Contango vs backwardation analysis
    contango_days = term["vixy_contango"] == 1
    backwardation_days = term["vixy_contango"] == 0

    c_ret = spy_ret[contango_days].mean() * 252 * 100
    b_ret = spy_ret[backwardation_days].mean() * 252 * 100
    print(f"\n  Contango days: {contango_days.sum()} ({contango_days.mean()*100:.0f}%), ann ret: {c_ret:.1f}%")
    print(f"  Backwardation days: {backwardation_days.sum()} ({backwardation_days.mean()*100:.0f}%), ann ret: {b_ret:.1f}%")

    return {
        "quintile_returns": quintile_data,
        "n_vixy_spikes": int(n_spikes),
        "contango_pct": round(contango_days.mean() * 100, 1),
        "contango_ann_ret": round(c_ret, 2),
        "backwardation_ann_ret": round(b_ret, 2),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("VIX Term Structure Signal for Gameplan v3")
    print("=" * 70)

    df = fetch_data()

    # 1. Analyze term structure behavior
    behavior = analyze_term_structure_behavior(df)

    # 2. Standalone signal test
    standalone = analyze_term_structure_standalone(df)

    # 3. Walk-forward with v3 enhancement
    wf = walk_forward_test(df)

    # 4. Full-period metrics
    v3_signals = compute_gameplan_v3_signals(df)
    term_signals = compute_vix_term_structure_signals(df)

    bt_v3 = backtest_gameplan_v3(df, v3_signals)
    v3_full = compute_metrics(bt_v3["returns"].dropna())

    bt_enhanced = backtest_v3_plus_term(df, v3_signals, term_signals)
    enhanced_full = compute_metrics(bt_enhanced["returns"].dropna())

    print(f"\n=== Full Period Comparison ===")
    print(f"  v3 baseline:  Sharpe={v3_full['sharpe']:.3f}, Sortino={v3_full['sortino']:.3f}, "
          f"CAGR={v3_full['cagr_pct']:.1f}%, MaxDD={v3_full['max_dd_pct']:.1f}%")
    print(f"  v3 + term:    Sharpe={enhanced_full['sharpe']:.3f}, Sortino={enhanced_full['sortino']:.3f}, "
          f"CAGR={enhanced_full['cagr_pct']:.1f}%, MaxDD={enhanced_full['max_dd_pct']:.1f}%")
    print(f"  Δ Sharpe:     {enhanced_full['sharpe'] - v3_full['sharpe']:+.4f}")

    # 5. Adversarial validation
    adversarial = run_adversarial(df)

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "metadata": {
            "strategy": "VIX Term Structure Enhancement for Gameplan v3",
            "data_start": df.index[0].strftime("%Y-%m-%d"),
            "data_end": df.index[-1].strftime("%Y-%m-%d"),
            "n_days": len(df),
        },
        "behavior_analysis": behavior,
        "standalone": standalone,
        "full_period": {
            "v3_baseline": v3_full,
            "v3_plus_term": enhanced_full,
            "improvement": round(enhanced_full["sharpe"] - v3_full["sharpe"], 4),
        },
        "walkforward": wf,
        "adversarial": adversarial,
    }

    output_file = OUTPUT_DIR / "vix_term_structure_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o) if isinstance(o, (np.integer, np.floating)) else str(o))
    print(f"\nResults saved to {output_file}")

    # Summary verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    wf_improves = wf["pct_beating_v3"] > 50
    adv_passes = adversarial["all_pass"]
    meaningful = abs(enhanced_full["sharpe"] - v3_full["sharpe"]) > 0.05

    if wf_improves and adv_passes and meaningful:
        print("✅ VIX term structure ADDS genuine value to Gameplan v3.")
        print(f"   WF: {wf['folds_beating_v3']} folds beat v3")
        print(f"   Adversarial: ALL PASS")
        print(f"   Sharpe improvement: {enhanced_full['sharpe'] - v3_full['sharpe']:+.4f}")
    elif wf_improves or (adversarial["permutation"]["pass"] and meaningful):
        print("⚠️ VIX term structure shows PARTIAL value — needs more validation.")
        print(f"   WF: {wf['folds_beating_v3']} folds beat v3")
        print(f"   Permutation: {'PASS' if adversarial['permutation']['pass'] else 'FAIL'}")
    else:
        print("❌ VIX term structure does NOT improve Gameplan v3.")
        print(f"   WF: {wf['folds_beating_v3']} folds beat v3")
        print(f"   Sharpe change: {enhanced_full['sharpe'] - v3_full['sharpe']:+.4f}")

    print()


if __name__ == "__main__":
    main()
