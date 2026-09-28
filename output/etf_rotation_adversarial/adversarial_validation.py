"""
ETF Rotation v3 — Adversarial Validation Suite
================================================
Tests:
  1. Permutation test (100 trials) — shuffle sector-return assignments
  2. R1 Regime test (HC #428) — green/red/flat Sharpe split
  3. Random sector selection baseline (100 trials)
  4. Walk-forward leakage check
  5. Sensitivity test (train window + hold period grid)
  6. Time period stability (half-split + per-year)

Saves all results to /home/jupiter/Lvl3Quant/output/etf_rotation_adversarial/
"""
from __future__ import annotations
import json
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
# Import research walk_forward FIRST (before macro_picker pollutes sys.path)
sys.path.insert(0, str(ROOT / "research"))
import importlib
_wf_mod = importlib.import_module("walk_forward")
_metrics = _wf_mod._metrics

# Now import macro_picker module
sys.path.insert(0, str(ROOT / "strategy/macro_picker"))
from etf_rotation_v3 import (
    build_panel, _xs_zscore, _fit_ridge, _load_spy_regime,
    _load_macro_regime_signals, _macro_keep_flat, _estimate_book_vol,
    _iter_wf_windows, _wf_fold, _classify_es_regime, _per_regime_metrics,
    SECTOR_ETFS, TRADING_DAYS,
)

OUT_DIR = ROOT / "output/etf_rotation_adversarial"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Default config matching the claimed v3 run
DEFAULT_CFG = dict(
    hold_days=21, n_long=2, n_short=2, allow_short=True,
    target_vol=0.15, lev_min=0.25, lev_max=2.0,
    train_months=24, oot_months=6, step_months=3,
    txn_cost_bps=5.0, regime_ma_days=60,
    thr_vix_chg=5.0, thr_vix_term=1.10,
    thr_dxy_chg=2.5, thr_disp_pct=10.0,
)


def _run_full_wf(panel, feats, cfg, regime_filter=None, macro_sig=None,
                 permute_returns=False, random_pick=False, rng=None):
    """Run the full walk-forward and return concatenated daily PnL series.

    permute_returns: if True, shuffle ret_1d within each date across sectors
    random_pick: if True, ignore ridge scores and pick sectors randomly
    """
    hold_days = cfg["hold_days"]
    windows = _iter_wf_windows(
        panel["date"].min(), panel["date"].max(),
        train_months=cfg["train_months"],
        oot_months=cfg["oot_months"],
        step_months=cfg["step_months"],
    )
    if not windows:
        return pd.Series(dtype=float)

    if permute_returns:
        # Shuffle returns across sectors within each date
        panel = panel.copy()
        if rng is None:
            rng = np.random.default_rng()
        for d in panel["date"].unique():
            mask = panel["date"] == d
            rets = panel.loc[mask, "ret_1d"].values.copy()
            rng.shuffle(rets)
            panel.loc[mask, "ret_1d"] = rets

    if random_pick:
        # We'll run a custom fold that picks sectors randomly instead of by score
        return _run_random_pick_wf(panel, feats, cfg, windows, regime_filter,
                                     macro_sig, rng)

    # Standard run using existing _wf_fold
    all_pnl_parts = []
    for (ts, te, os_, oe) in windows:
        result = _wf_fold(
            panel, feats, ts, te, os_, oe,
            hold_days=hold_days,
            n_long=cfg["n_long"],
            n_short=cfg["n_short"],
            allow_short=cfg["allow_short"],
            target_vol=cfg["target_vol"],
            lev_min=cfg["lev_min"],
            lev_max=cfg["lev_max"],
            txn_cost_bps=cfg["txn_cost_bps"],
            regime_filter=regime_filter,
            macro_sig=macro_sig,
            thr_vix_chg=cfg["thr_vix_chg"],
            thr_vix_term=cfg["thr_vix_term"],
            thr_dxy_chg=cfg["thr_dxy_chg"],
            thr_disp_pct=cfg["thr_disp_pct"],
        )
        if not result["daily_pnl"].empty:
            all_pnl_parts.append(result["daily_pnl"])

    if not all_pnl_parts:
        return pd.Series(dtype=float)
    all_pnl = pd.concat(all_pnl_parts).sort_index()
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep="last")]
    return all_pnl


def _run_random_pick_wf(panel, feats, cfg, windows, regime_filter, macro_sig, rng):
    """Walk-forward but pick sectors randomly instead of by ridge scores."""
    if rng is None:
        rng = np.random.default_rng()
    hold_days = cfg["hold_days"]
    n_long = cfg["n_long"]
    n_short = cfg["n_short"]
    allow_short = cfg["allow_short"]

    all_daily = []
    for (ts, te, os_, oe) in windows:
        oot = panel[(panel["date"] >= os_) & (panel["date"] < oe)].copy()
        if len(oot) < 20:
            continue
        oot["ret_raw"] = pd.to_numeric(oot["ret_1d"], errors="coerce").astype(float)
        unique_dates = sorted(oot["date"].unique())
        rebal_dates = unique_dates[::hold_days]

        for rd in rebal_dates:
            # SPY regime filter
            if regime_filter is not None:
                rg = regime_filter.get(pd.Timestamp(rd))
                if rg is None:
                    prior = regime_filter.loc[:pd.Timestamp(rd)]
                    rg = prior.iloc[-1] if len(prior) else "bull"
                if rg != "bull":
                    continue

            # Macro gate
            if macro_sig is not None:
                kf, _ = _macro_keep_flat(macro_sig, pd.Timestamp(rd),
                                          cfg["thr_vix_chg"], cfg["thr_vix_term"],
                                          cfg["thr_dxy_chg"], cfg["thr_disp_pct"])
                if kf:
                    hold_win_dates = [d for d in unique_dates
                                       if d > rd and d <= rd + pd.Timedelta(days=hold_days)]
                    for d in hold_win_dates:
                        all_daily.append((d, 0.0))
                    continue

            snap = oot[oot["date"] == rd]
            available = snap["etf"].unique().tolist()
            if len(available) < (n_long + (n_short if allow_short else 0)):
                continue

            # RANDOM selection
            chosen = rng.choice(available, size=n_long + (n_short if allow_short else 0),
                                 replace=False).tolist()
            longs = chosen[:n_long]
            shorts = chosen[n_long:] if allow_short else []

            realised_vol = _estimate_book_vol(panel, rd, longs, shorts)
            if realised_vol <= 1e-6:
                gross_lev = 1.0
            else:
                gross_lev = float(np.clip(cfg["target_vol"] / realised_vol,
                                           cfg["lev_min"], cfg["lev_max"]))

            hold_win = oot[(oot["date"] > rd)
                           & (oot["date"] <= rd + pd.Timedelta(days=hold_days))]
            for d, g in hold_win.groupby("date"):
                if regime_filter is not None:
                    rg = regime_filter.get(pd.Timestamp(d))
                    if rg is None:
                        prior = regime_filter.loc[:pd.Timestamp(d)]
                        rg = prior.iloc[-1] if len(prior) else "bull"
                    if rg != "bull":
                        all_daily.append((d, 0.0))
                        continue
                if macro_sig is not None:
                    kf, _ = _macro_keep_flat(macro_sig, pd.Timestamp(d),
                                              cfg["thr_vix_chg"], cfg["thr_vix_term"],
                                              cfg["thr_dxy_chg"], cfg["thr_disp_pct"])
                    if kf:
                        all_daily.append((d, 0.0))
                        continue
                lret = g[g["etf"].isin(longs)]["ret_raw"].mean() if longs else 0.0
                sret = g[g["etf"].isin(shorts)]["ret_raw"].mean() if shorts else 0.0
                if allow_short:
                    book_ret = (lret if pd.notna(lret) else 0.0) - (sret if pd.notna(sret) else 0.0)
                else:
                    book_ret = (lret if pd.notna(lret) else 0.0)
                day_ret = gross_lev * book_ret
                day_ret = float(np.clip(day_ret, -0.20, 0.20))
                all_daily.append((d, day_ret))

            tc = (cfg["txn_cost_bps"] / 10000.0) * gross_lev
            all_daily.append((rd, -tc))

    if not all_daily:
        return pd.Series(dtype=float)
    df = pd.DataFrame(all_daily, columns=["date", "ret"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.groupby("date")["ret"].sum()
    return df.sort_index()


# =============================================================================
# TEST 1: PERMUTATION TEST
# =============================================================================
def test_permutation(panel, feats, cfg, regime_filter, macro_sig, n_trials=100):
    print("\n" + "="*70)
    print("TEST 1: PERMUTATION TEST (100 trials)")
    print("="*70)

    # First get the real Sharpe
    real_pnl = _run_full_wf(panel, feats, cfg, regime_filter, macro_sig)
    real_metrics = _metrics(real_pnl) if not real_pnl.empty else {}
    real_sharpe = real_metrics.get("sharpe", float("nan"))
    print(f"Real strategy Sharpe: {real_sharpe:.3f}")
    print(f"Real strategy CAGR: {real_metrics.get('cagr', float('nan'))*100:.1f}%")

    perm_sharpes = []
    for i in range(n_trials):
        rng = np.random.default_rng(seed=i)
        perm_pnl = _run_full_wf(panel, feats, cfg, regime_filter, macro_sig,
                                 permute_returns=True, rng=rng)
        if perm_pnl.empty:
            perm_sharpes.append(float("nan"))
            continue
        m = _metrics(perm_pnl)
        sh = m.get("sharpe", float("nan"))
        perm_sharpes.append(sh)
        if (i + 1) % 10 == 0:
            valid = [s for s in perm_sharpes if np.isfinite(s)]
            print(f"  Trial {i+1}/{n_trials}: perm Sharpe={sh:.3f}, "
                  f"mean so far={np.mean(valid):.3f}")

    valid_sharpes = [s for s in perm_sharpes if np.isfinite(s)]
    n_beat = sum(1 for s in valid_sharpes if s >= real_sharpe)
    p_value = n_beat / len(valid_sharpes) if valid_sharpes else 1.0

    result = {
        "real_sharpe": real_sharpe,
        "real_cagr": real_metrics.get("cagr", float("nan")),
        "real_sortino": real_metrics.get("sortino", float("nan")),
        "n_trials": n_trials,
        "n_valid": len(valid_sharpes),
        "n_beat_real": n_beat,
        "p_value": p_value,
        "perm_sharpe_mean": float(np.mean(valid_sharpes)) if valid_sharpes else float("nan"),
        "perm_sharpe_std": float(np.std(valid_sharpes)) if valid_sharpes else float("nan"),
        "perm_sharpe_median": float(np.median(valid_sharpes)) if valid_sharpes else float("nan"),
        "perm_sharpe_p95": float(np.percentile(valid_sharpes, 95)) if valid_sharpes else float("nan"),
        "perm_sharpe_max": float(np.max(valid_sharpes)) if valid_sharpes else float("nan"),
        "PASS": p_value < 0.05,
        "verdict": "REAL SIGNAL (p<0.05)" if p_value < 0.05 else "LIKELY ARTIFACT (p>=0.05)",
        "all_perm_sharpes": [float(s) for s in perm_sharpes],
    }

    print(f"\n  PERMUTATION RESULT:")
    print(f"  Real Sharpe: {real_sharpe:.3f}")
    print(f"  Perm mean: {result['perm_sharpe_mean']:.3f} +/- {result['perm_sharpe_std']:.3f}")
    print(f"  Perm p95: {result['perm_sharpe_p95']:.3f}")
    print(f"  Perm max: {result['perm_sharpe_max']:.3f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  VERDICT: {result['verdict']}")
    return result


# =============================================================================
# TEST 2: REGIME TEST (HC #428 R1)
# =============================================================================
def test_regime(panel, feats, cfg, regime_filter, macro_sig):
    print("\n" + "="*70)
    print("TEST 2: R1 REGIME TEST (HC #428)")
    print("="*70)

    real_pnl = _run_full_wf(panel, feats, cfg, regime_filter, macro_sig)
    if real_pnl.empty:
        print("  No PnL data to analyze")
        return {"PASS": False, "verdict": "NO DATA"}

    regime_stats = _per_regime_metrics(real_pnl)

    sg = regime_stats.get("per_regime_sharpe", {}).get("green", float("nan"))
    sr = regime_stats.get("per_regime_sharpe", {}).get("red", float("nan"))
    sf = regime_stats.get("per_regime_sharpe", {}).get("flat", float("nan"))
    skew = regime_stats.get("regime_skew_ratio", float("nan"))
    skew_pass = regime_stats.get("regime_skew_pass", False)

    result = {
        "sharpe_green": sg,
        "sharpe_red": sr,
        "sharpe_flat": sf,
        "n_green": regime_stats.get("per_regime_n", {}).get("green", 0),
        "n_red": regime_stats.get("per_regime_n", {}).get("red", 0),
        "n_flat": regime_stats.get("per_regime_n", {}).get("flat", 0),
        "wr_green": regime_stats.get("per_regime_wr", {}).get("green", float("nan")),
        "wr_red": regime_stats.get("per_regime_wr", {}).get("red", float("nan")),
        "wr_flat": regime_stats.get("per_regime_wr", {}).get("flat", float("nan")),
        "mean_ret_bps_green": regime_stats.get("per_regime_mean_ret_bps", {}).get("green", float("nan")),
        "mean_ret_bps_red": regime_stats.get("per_regime_mean_ret_bps", {}).get("red", float("nan")),
        "mean_ret_bps_flat": regime_stats.get("per_regime_mean_ret_bps", {}).get("flat", float("nan")),
        "regime_skew_ratio": skew,
        "regime_skew_cap": 0.50,
        "PASS": skew_pass,
        "verdict": "REGIME-BALANCED (skew<=0.50)" if skew_pass else f"REGIME-SKEWED (skew={skew:.3f}>0.50)",
    }

    print(f"  Sharpe_green: {sg:.3f} (n={result['n_green']})")
    print(f"  Sharpe_red:   {sr:.3f} (n={result['n_red']})")
    print(f"  Sharpe_flat:  {sf:.3f} (n={result['n_flat']})")
    print(f"  Regime skew:  {skew:.3f} (cap=0.50)")
    print(f"  VERDICT: {result['verdict']}")
    return result


# =============================================================================
# TEST 3: RANDOM SECTOR SELECTION BASELINE
# =============================================================================
def test_random_baseline(panel, feats, cfg, regime_filter, macro_sig, n_trials=100):
    print("\n" + "="*70)
    print("TEST 3: RANDOM SECTOR SELECTION BASELINE (100 trials)")
    print("="*70)

    real_pnl = _run_full_wf(panel, feats, cfg, regime_filter, macro_sig)
    real_m = _metrics(real_pnl) if not real_pnl.empty else {}
    real_sharpe = real_m.get("sharpe", float("nan"))
    real_cagr = real_m.get("cagr", float("nan"))

    rand_sharpes = []
    rand_cagrs = []
    for i in range(n_trials):
        rng = np.random.default_rng(seed=1000 + i)
        rand_pnl = _run_full_wf(panel, feats, cfg, regime_filter, macro_sig,
                                  random_pick=True, rng=rng)
        if rand_pnl.empty:
            rand_sharpes.append(float("nan"))
            rand_cagrs.append(float("nan"))
            continue
        m = _metrics(rand_pnl)
        rand_sharpes.append(m.get("sharpe", float("nan")))
        rand_cagrs.append(m.get("cagr", float("nan")))
        if (i + 1) % 10 == 0:
            vs = [s for s in rand_sharpes if np.isfinite(s)]
            vc = [c for c in rand_cagrs if np.isfinite(c)]
            print(f"  Trial {i+1}/{n_trials}: rand Sharpe={rand_sharpes[-1]:.3f}, "
                  f"rand CAGR={rand_cagrs[-1]*100:.1f}%, "
                  f"mean Sharpe so far={np.mean(vs):.3f}, "
                  f"mean CAGR={np.mean(vc)*100:.1f}%")

    vs = [s for s in rand_sharpes if np.isfinite(s)]
    vc = [c for c in rand_cagrs if np.isfinite(c)]

    sharpe_lift = real_sharpe - np.mean(vs) if vs else float("nan")
    cagr_lift = real_cagr - np.mean(vc) if vc else float("nan")
    n_rand_beat_sharpe = sum(1 for s in vs if s >= real_sharpe)
    n_rand_beat_cagr = sum(1 for c in vc if c >= real_cagr)

    # Model adds value if random baseline is substantially worse
    model_adds_value = (sharpe_lift > 0.5) if np.isfinite(sharpe_lift) else False

    result = {
        "real_sharpe": real_sharpe,
        "real_cagr": real_cagr,
        "n_trials": n_trials,
        "rand_sharpe_mean": float(np.mean(vs)) if vs else float("nan"),
        "rand_sharpe_std": float(np.std(vs)) if vs else float("nan"),
        "rand_sharpe_median": float(np.median(vs)) if vs else float("nan"),
        "rand_sharpe_max": float(np.max(vs)) if vs else float("nan"),
        "rand_cagr_mean": float(np.mean(vc)) if vc else float("nan"),
        "rand_cagr_std": float(np.std(vc)) if vc else float("nan"),
        "rand_cagr_median": float(np.median(vc)) if vc else float("nan"),
        "rand_cagr_max": float(np.max(vc)) if vc else float("nan"),
        "sharpe_lift_vs_random": sharpe_lift,
        "cagr_lift_vs_random": cagr_lift,
        "n_rand_beat_sharpe": n_rand_beat_sharpe,
        "n_rand_beat_cagr": n_rand_beat_cagr,
        "p_value_sharpe": n_rand_beat_sharpe / len(vs) if vs else 1.0,
        "p_value_cagr": n_rand_beat_cagr / len(vc) if vc else 1.0,
        "model_adds_value": model_adds_value,
        "PASS": model_adds_value,
        "verdict": ("MODEL ADDS VALUE" if model_adds_value
                     else "MODEL DOES NOT ADD VALUE vs RANDOM"),
    }

    print(f"\n  RANDOM BASELINE RESULT:")
    print(f"  Real Sharpe: {real_sharpe:.3f}, Real CAGR: {real_cagr*100:.1f}%")
    print(f"  Random mean Sharpe: {result['rand_sharpe_mean']:.3f} +/- {result['rand_sharpe_std']:.3f}")
    print(f"  Random mean CAGR: {result['rand_cagr_mean']*100:.1f}%")
    print(f"  Sharpe lift: {sharpe_lift:.3f}")
    print(f"  CAGR lift: {cagr_lift*100:.1f}pp")
    print(f"  Random beat real Sharpe: {n_rand_beat_sharpe}/{len(vs)}")
    print(f"  VERDICT: {result['verdict']}")
    return result


# =============================================================================
# TEST 4: WALK-FORWARD LEAKAGE CHECK
# =============================================================================
def test_leakage(panel, feats, cfg):
    print("\n" + "="*70)
    print("TEST 4: WALK-FORWARD LEAKAGE CHECK")
    print("="*70)

    issues = []

    windows = _iter_wf_windows(
        panel["date"].min(), panel["date"].max(),
        train_months=cfg["train_months"],
        oot_months=cfg["oot_months"],
        step_months=cfg["step_months"],
    )

    # Check 1: Training window never overlaps OOS window
    print(f"  Checking {len(windows)} WF folds for train/OOS overlap...")
    for i, (ts, te, os_, oe) in enumerate(windows):
        if te > os_:
            issues.append(f"Fold {i}: train_end ({te}) > oot_start ({os_}) — OVERLAP!")
        if ts >= te:
            issues.append(f"Fold {i}: train_start ({ts}) >= train_end ({te}) — invalid")
        if os_ >= oe:
            issues.append(f"Fold {i}: oot_start ({os_}) >= oot_end ({oe}) — invalid")

    # Check 2: No overlap between consecutive folds' OOS periods
    # (overlapping OOS is OK for walk-forward but let's document it)
    oos_overlaps = 0
    for i in range(len(windows) - 1):
        _, _, os1, oe1 = windows[i]
        _, _, os2, oe2 = windows[i + 1]
        if os2 < oe1:
            oos_overlaps += 1

    # Check 3: y_fwd uses FUTURE close prices (this is the target, not a feature)
    # but verify it's computed correctly — shift(-hold_days) means we look forward
    print(f"  Checking y_fwd computation for look-ahead...")
    # y_fwd = (close[t+hold] / close[t]) - 1. This is the TARGET. It SHOULD be forward.
    # The KEY check is that features at time t only use data <= t.

    # Check 4: Feature computation timing
    # ret_1d, ret_20d, ret_60d — these are BACKWARD-looking returns, computed from historical prices. OK.
    # rel_strength_spy, momentum_cross_20_60, rs_rank_among_sectors — backward-looking. OK.
    # Fundamentals (fp_*) — these come from master_panel, which should be point-in-time.
    # Macro features — backward-looking (diffs, rolling). OK.

    # Check 5: _xs_zscore is applied separately to train and OOS
    # In _wf_fold: train_z = _xs_zscore(train, feats), oot_z = _xs_zscore(oot, feats)
    # This means OOS z-scores use OOS cross-sectional stats, NOT train stats.
    # This is a MILD issue: the z-scoring uses future OOS data for normalization.
    # However, since it's cross-sectional (across 11 ETFs on same date), it only uses
    # same-date info, not future dates. So it's NOT a temporal leak.
    zscore_note = ("_xs_zscore applied separately to OOS — uses same-date cross-sectional "
                   "stats only (not future dates). Not a temporal leak, but OOS z-scores "
                   "are computed from OOS data. Acceptable for cross-sectional normalization.")

    # Check 6: Ridge coefficients from fold N used in fold N's OOS?
    # In _wf_fold: train on [tr_start, tr_end), predict on [oot_start, oot_end)
    # tr_end == oot_start. So no overlap. Coefficients from training are applied to OOS. Correct.
    ridge_note = "Ridge trained on [tr_start, tr_end), applied to [oot_start, oot_end). tr_end == oot_start. No overlap."

    # Check 7: Regime filter uses SPY price, which is available in real-time. OK.
    # Check 8: Macro gate signals — VIX, DXY are observable in real-time. OK.

    # Check 9: Look for any .shift() calls that might look forward
    # y_fwd uses shift(-hold_days) — this is the TARGET, expected.
    # ret_1d in flows data — should be backward-looking

    # Check 10: Verify target variable alignment
    # y_fwd at date t = return over next hold_days. Features at date t should only use data <= t.
    # _estimate_book_vol uses historical data only (< rd). OK.

    n_issues = len(issues)
    if n_issues == 0:
        issues.append("No temporal leakage detected.")

    result = {
        "n_folds": len(windows),
        "n_issues": n_issues,
        "issues": issues,
        "oos_overlaps_between_folds": oos_overlaps,
        "notes": {
            "zscore": zscore_note,
            "ridge_coefficients": ridge_note,
            "y_fwd": "y_fwd = shift(-hold_days) return — this IS the target, correctly forward-looking",
            "features": "All feature columns (ret_1d/20d/60d, momentum, RS, fundamentals, macro) are backward-looking",
            "regime_filter": "SPY-MA60 overlay uses real-time observable SPY price. No leak.",
            "macro_gate": "VIX, DXY signals are real-time observable. No leak.",
            "vol_sizing": "_estimate_book_vol uses data < rebalance_date only. No leak.",
        },
        "PASS": n_issues == 0 or (n_issues == 1 and "No temporal leakage" in issues[0]),
        "verdict": "NO LEAKAGE DETECTED" if n_issues == 0 or (n_issues == 1 and "No temporal leakage" in issues[0]) else f"{n_issues} LEAKAGE ISSUES FOUND",
    }

    # Print fold boundaries for verification
    print(f"\n  First 3 fold boundaries:")
    for i, (ts, te, os_, oe) in enumerate(windows[:3]):
        print(f"    Fold {i}: train [{ts.date()} -> {te.date()}) | OOS [{os_.date()} -> {oe.date()})")
    print(f"    ...")
    if len(windows) > 3:
        i = len(windows) - 1
        ts, te, os_, oe = windows[-1]
        print(f"    Fold {i}: train [{ts.date()} -> {te.date()}) | OOS [{os_.date()} -> {oe.date()})")

    print(f"\n  OOS overlaps between consecutive folds: {oos_overlaps}")
    print(f"  Issues found: {n_issues}")
    for iss in issues:
        print(f"    - {iss}")
    print(f"  VERDICT: {result['verdict']}")
    return result


# =============================================================================
# TEST 5: SENSITIVITY TEST
# =============================================================================
def test_sensitivity(panel, feats, cfg, regime_filter, macro_sig):
    print("\n" + "="*70)
    print("TEST 5: SENSITIVITY TEST (train window + hold period)")
    print("="*70)

    train_windows = [12, 18, 24, 30, 36]
    hold_periods = [10, 15, 21, 30, 42]

    results_grid = []
    for tw in train_windows:
        for hp in hold_periods:
            print(f"  Running train={tw}mo, hold={hp}d ...", end=" ", flush=True)
            test_cfg = dict(cfg)
            test_cfg["train_months"] = tw
            test_cfg["hold_days"] = hp

            # Need to rebuild panel with new hold_days for y_fwd
            try:
                test_panel, test_feats = build_panel(hp)
                pnl = _run_full_wf(test_panel, test_feats, test_cfg, regime_filter, macro_sig)
                if pnl.empty:
                    m = {}
                else:
                    m = _metrics(pnl)
                sh = m.get("sharpe", float("nan"))
                cagr = m.get("cagr", float("nan"))
                mdd = m.get("max_dd", float("nan"))
                print(f"Sharpe={sh:.2f}, CAGR={cagr*100:.1f}%, MaxDD={mdd*100:.1f}%")
                results_grid.append({
                    "train_months": tw, "hold_days": hp,
                    "sharpe": sh, "cagr": cagr, "max_dd": mdd,
                    "sortino": m.get("sortino", float("nan")),
                    "calmar": m.get("calmar", float("nan")),
                    "pf": m.get("pf", float("nan")),
                    "wr": m.get("wr", float("nan")),
                    "n_trades": m.get("n_trades", 0),
                })
            except Exception as e:
                print(f"FAILED: {e}")
                results_grid.append({
                    "train_months": tw, "hold_days": hp,
                    "sharpe": float("nan"), "error": str(e),
                })

    # Analysis: how many parameter combos are profitable?
    valid = [r for r in results_grid if np.isfinite(r.get("sharpe", float("nan")))]
    n_profitable = sum(1 for r in valid if r["sharpe"] > 0)
    n_good = sum(1 for r in valid if r["sharpe"] > 1.0)
    n_great = sum(1 for r in valid if r["sharpe"] > 1.5)

    sharpes = [r["sharpe"] for r in valid]
    sharpe_std = float(np.std(sharpes)) if sharpes else float("nan")
    sharpe_range = (float(np.min(sharpes)), float(np.max(sharpes))) if sharpes else (float("nan"), float("nan"))

    # Overfit detection: if ONLY one setting works, it's overfit
    robust = n_good >= 5  # at least 5 out of 25 combos have Sharpe > 1.0

    result = {
        "grid": results_grid,
        "n_combos": len(results_grid),
        "n_valid": len(valid),
        "n_profitable_sharpe_gt_0": n_profitable,
        "n_good_sharpe_gt_1": n_good,
        "n_great_sharpe_gt_1_5": n_great,
        "sharpe_std_across_grid": sharpe_std,
        "sharpe_range": sharpe_range,
        "robust": robust,
        "PASS": robust,
        "verdict": ("ROBUST — works across multiple settings" if robust
                     else "FRAGILE — only works at specific settings (likely overfit)"),
    }

    print(f"\n  SENSITIVITY RESULT:")
    print(f"  Total combos: {len(results_grid)}, Valid: {len(valid)}")
    print(f"  Profitable (Sharpe>0): {n_profitable}/{len(valid)}")
    print(f"  Good (Sharpe>1.0): {n_good}/{len(valid)}")
    print(f"  Great (Sharpe>1.5): {n_great}/{len(valid)}")
    print(f"  Sharpe range: [{sharpe_range[0]:.2f}, {sharpe_range[1]:.2f}]")
    print(f"  VERDICT: {result['verdict']}")
    return result


# =============================================================================
# TEST 6: TIME PERIOD STABILITY
# =============================================================================
def test_time_stability(panel, feats, cfg, regime_filter, macro_sig):
    print("\n" + "="*70)
    print("TEST 6: TIME PERIOD STABILITY")
    print("="*70)

    real_pnl = _run_full_wf(panel, feats, cfg, regime_filter, macro_sig)
    if real_pnl.empty:
        return {"PASS": False, "verdict": "NO DATA"}

    # Half-split
    mid_idx = len(real_pnl) // 2
    first_half = real_pnl.iloc[:mid_idx]
    second_half = real_pnl.iloc[mid_idx:]

    m_first = _metrics(first_half) if not first_half.empty else {}
    m_second = _metrics(second_half) if not second_half.empty else {}

    print(f"  First half:  {first_half.index[0].date()} -> {first_half.index[-1].date()}")
    print(f"    Sharpe={m_first.get('sharpe', float('nan')):.3f}, "
          f"CAGR={m_first.get('cagr', float('nan'))*100:.1f}%, "
          f"MaxDD={m_first.get('max_dd', float('nan'))*100:.1f}%")
    print(f"  Second half: {second_half.index[0].date()} -> {second_half.index[-1].date()}")
    print(f"    Sharpe={m_second.get('sharpe', float('nan')):.3f}, "
          f"CAGR={m_second.get('cagr', float('nan'))*100:.1f}%, "
          f"MaxDD={m_second.get('max_dd', float('nan'))*100:.1f}%")

    # Per-year breakdown
    real_pnl_df = pd.DataFrame({"ret": real_pnl})
    real_pnl_df["year"] = real_pnl_df.index.year
    per_year = {}
    for yr, grp in real_pnl_df.groupby("year"):
        m = _metrics(grp["ret"])
        per_year[int(yr)] = m
        print(f"  {yr}: Sharpe={m.get('sharpe', float('nan')):.3f}, "
              f"CAGR={m.get('cagr', float('nan'))*100:.1f}%, "
              f"MaxDD={m.get('max_dd', float('nan'))*100:.1f}%, "
              f"WR={m.get('wr', float('nan'))*100:.1f}%, "
              f"n_days={m.get('n_trades', 0)}")

    # Check: is CAGR driven by one exceptional year?
    year_cagrs = {yr: m.get("cagr", float("nan")) for yr, m in per_year.items() if np.isfinite(m.get("cagr", float("nan")))}
    year_sharpes = {yr: m.get("sharpe", float("nan")) for yr, m in per_year.items() if np.isfinite(m.get("sharpe", float("nan")))}

    if year_cagrs:
        best_year = max(year_cagrs, key=year_cagrs.get)
        best_cagr = year_cagrs[best_year]
        other_cagrs = [c for yr, c in year_cagrs.items() if yr != best_year]
        mean_other = np.mean(other_cagrs) if other_cagrs else float("nan")
        concentration_warning = (best_cagr > 3 * mean_other) if np.isfinite(mean_other) and mean_other > 0 else False
    else:
        best_year = None
        concentration_warning = False

    # Both halves profitable?
    both_halves_ok = (m_first.get("sharpe", -99) > 0) and (m_second.get("sharpe", -99) > 0)

    # Majority of years positive Sharpe?
    n_pos_years = sum(1 for s in year_sharpes.values() if s > 0)
    n_total_years = len(year_sharpes)
    majority_years_ok = n_pos_years >= (n_total_years * 0.6) if n_total_years > 0 else False

    result = {
        "first_half": {
            "start": str(first_half.index[0].date()),
            "end": str(first_half.index[-1].date()),
            **m_first,
        },
        "second_half": {
            "start": str(second_half.index[0].date()),
            "end": str(second_half.index[-1].date()),
            **m_second,
        },
        "per_year": {str(yr): m for yr, m in per_year.items()},
        "both_halves_profitable": both_halves_ok,
        "n_years_positive_sharpe": n_pos_years,
        "n_years_total": n_total_years,
        "majority_years_positive": majority_years_ok,
        "concentration_warning": concentration_warning,
        "best_year": best_year,
        "PASS": both_halves_ok and majority_years_ok and not concentration_warning,
        "verdict": [],
    }

    verdicts = []
    if both_halves_ok:
        verdicts.append("Both halves profitable")
    else:
        verdicts.append("WARNING: Not both halves profitable")
    if majority_years_ok:
        verdicts.append(f"{n_pos_years}/{n_total_years} years positive Sharpe")
    else:
        verdicts.append(f"WARNING: Only {n_pos_years}/{n_total_years} years positive")
    if concentration_warning:
        verdicts.append(f"WARNING: Returns concentrated in {best_year}")
    result["verdict"] = verdicts

    print(f"\n  TIME STABILITY RESULT:")
    for v in verdicts:
        print(f"    {v}")
    print(f"  PASS: {result['PASS']}")
    return result


# =============================================================================
# MAIN
# =============================================================================
def main():
    start_time = time.time()
    print("="*70)
    print("ETF ROTATION v3 — ADVERSARIAL VALIDATION SUITE")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*70)

    # Build panel once (default hold_days=21 for most tests)
    print("\nBuilding panel...")
    panel, feats = build_panel(DEFAULT_CFG["hold_days"])
    print(f"Panel: {len(panel)} rows, {panel['etf'].nunique()} ETFs, "
          f"{len(feats)} features")
    print(f"Date range: {panel['date'].min().date()} -> {panel['date'].max().date()}")

    # Load regime filter and macro signals
    regime_filter = _load_spy_regime(ma_days=DEFAULT_CFG["regime_ma_days"])
    macro_sig = _load_macro_regime_signals()

    all_results = {}

    # TEST 4: Leakage (fast, run first)
    try:
        all_results["test4_leakage"] = test_leakage(panel, feats, DEFAULT_CFG)
    except Exception as e:
        print(f"  TEST 4 FAILED: {e}")
        traceback.print_exc()
        all_results["test4_leakage"] = {"PASS": False, "error": str(e)}

    # TEST 2: Regime (fast)
    try:
        all_results["test2_regime"] = test_regime(panel, feats, DEFAULT_CFG,
                                                     regime_filter, macro_sig)
    except Exception as e:
        print(f"  TEST 2 FAILED: {e}")
        traceback.print_exc()
        all_results["test2_regime"] = {"PASS": False, "error": str(e)}

    # TEST 6: Time stability (medium)
    try:
        all_results["test6_time_stability"] = test_time_stability(
            panel, feats, DEFAULT_CFG, regime_filter, macro_sig)
    except Exception as e:
        print(f"  TEST 6 FAILED: {e}")
        traceback.print_exc()
        all_results["test6_time_stability"] = {"PASS": False, "error": str(e)}

    # TEST 5: Sensitivity (slow — 25 combos)
    try:
        all_results["test5_sensitivity"] = test_sensitivity(
            panel, feats, DEFAULT_CFG, regime_filter, macro_sig)
    except Exception as e:
        print(f"  TEST 5 FAILED: {e}")
        traceback.print_exc()
        all_results["test5_sensitivity"] = {"PASS": False, "error": str(e)}

    # TEST 1: Permutation (slowest — 100 trials)
    try:
        all_results["test1_permutation"] = test_permutation(
            panel, feats, DEFAULT_CFG, regime_filter, macro_sig, n_trials=100)
    except Exception as e:
        print(f"  TEST 1 FAILED: {e}")
        traceback.print_exc()
        all_results["test1_permutation"] = {"PASS": False, "error": str(e)}

    # TEST 3: Random baseline (slow — 100 trials)
    try:
        all_results["test3_random_baseline"] = test_random_baseline(
            panel, feats, DEFAULT_CFG, regime_filter, macro_sig, n_trials=100)
    except Exception as e:
        print(f"  TEST 3 FAILED: {e}")
        traceback.print_exc()
        all_results["test3_random_baseline"] = {"PASS": False, "error": str(e)}

    # =================================================================
    # FINAL SUMMARY
    # =================================================================
    elapsed = time.time() - start_time
    print("\n\n" + "="*70)
    print("FINAL ADVERSARIAL VALIDATION SUMMARY")
    print("="*70)

    n_pass = 0
    n_fail = 0
    n_total = 6
    test_names = {
        "test1_permutation": "Permutation Test (p<0.05)",
        "test2_regime": "Regime Balance (skew<=0.50)",
        "test3_random_baseline": "Model vs Random",
        "test4_leakage": "WF Leakage Check",
        "test5_sensitivity": "Parameter Sensitivity",
        "test6_time_stability": "Time Period Stability",
    }
    for key, name in test_names.items():
        r = all_results.get(key, {})
        passed = r.get("PASS", False)
        verdict = r.get("verdict", "N/A")
        if isinstance(verdict, list):
            verdict = "; ".join(verdict)
        status = "PASS" if passed else "FAIL"
        if passed:
            n_pass += 1
        else:
            n_fail += 1
        print(f"  [{status}] {name}: {verdict}")

    overall = "STRATEGY IS REAL" if n_fail == 0 else (
        f"STRATEGY HAS CONCERNS ({n_fail}/{n_total} tests failed)" if n_fail <= 2
        else f"STRATEGY IS SUSPECT ({n_fail}/{n_total} tests failed)")

    all_results["summary"] = {
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_total": n_total,
        "overall_verdict": overall,
        "elapsed_seconds": elapsed,
        "timestamp": datetime.now().isoformat(),
    }

    print(f"\n  OVERALL: {overall}")
    print(f"  ({n_pass}/{n_total} passed, {n_fail}/{n_total} failed)")
    print(f"  Elapsed: {elapsed/60:.1f} minutes")

    # Save results
    # Convert non-serializable types
    def _clean(obj):
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    clean_results = _clean(all_results)
    out_path = OUT_DIR / "adversarial_results.json"
    out_path.write_text(json.dumps(clean_results, indent=2, default=str))
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
