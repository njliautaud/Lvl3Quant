#!/usr/bin/env python3
"""
hmm_regime_filter.py — 2-State HMM Regime Filter for BookSpatialCNN

Identifies "active" vs "dead" trading days using realized volatility and
bid-ask spread extracted from NPZ book tensor files.

Walk-forward OOT prediction: For each OOT day, fit HMM on all IS + prior
OOT days, predict state for current day. No look-ahead.

Output: results/hmm_regime_states.json

Usage:
    python alpha_discovery/deep_models/hmm_regime_filter.py
    python alpha_discovery/deep_models/hmm_regime_filter.py --verbose
"""

import json
import sys
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# ── Paths ──
FILE_DIR = Path(__file__).parent.resolve()
ROOT = FILE_DIR.parent.parent
IS_DIR = ROOT / "data/processed/dl_book_cache"
OOT_DIR = ROOT / "data/processed/dl_book_cache_oot"
RESULTS_DIR = FILE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Known IC values from walk-forward folds (for cross-reference)
KNOWN_ICS = {
    # From walkforward_fold_stats.json (first 20 folds)
    "2025-12-01": 0.0991,
    "2025-12-02": 0.0983,
    "2025-12-03": 0.1082,
    "2025-12-04": 0.1062,
    "2025-12-05": 0.1231,
    "2025-12-08": 0.1029,
    "2025-12-09": 0.1047,
    "2025-12-10": 0.1091,
    "2025-12-11": 0.1175,
    "2025-12-12": 0.1167,
    "2025-12-15": 0.1175,
    "2025-12-16": 0.0997,
    "2025-12-17": 0.1108,
    "2025-12-18": 0.0889,
    "2025-12-19": 0.0896,
    "2025-12-22": 0.1013,
    "2025-12-23": 0.0740,
    "2025-12-24": 0.0436,  # Christmas Eve — likely short day
    "2025-12-26": 0.0720,  # Day after Christmas
    "2025-12-29": 0.0730,
    # From memory: folds 21, 22, 23 from Uranus WF run
    # fold 21 = IC 0.161 (approx first few Jan days)
    # fold 22 = IC 0.143
    # fold 23 = IC 0.035 (Jan 2 half-day)
}


def extract_daily_features(npz_path: Path) -> dict | None:
    """
    Extract daily regime features from a book tensor NPZ file.

    Returns:
        dict with keys: date, daily_rvol, daily_spread, n_bars
    """
    try:
        npz = np.load(str(npz_path), allow_pickle=False)
        bt = npz["book_tensors"].astype(np.float32)  # (n_bars, 20, 4)
        mid = npz["mid_prices"].astype(np.float64)   # (n_bars,)
        npz.close()

        n = len(mid)
        if n < 100:
            return None

        date_str = npz_path.name.split("_book_tensors")[0]

        # --- Feature 1: Daily realized volatility ---
        # Std of log-returns of mid_prices, converted to bps
        # Use bar-to-bar log returns (each bar = 100ms)
        log_ret = np.diff(np.log(np.maximum(mid, 1.0)))
        daily_rvol_bps = float(np.std(log_ret)) * 1e4  # bps

        # Also compute in ticks (ES tick = 0.25)
        mid_diff = np.diff(mid) / 0.25  # tick changes
        daily_rvol_ticks = float(np.std(mid_diff))

        # --- Feature 2: Mean bid-ask spread ---
        # book_tensors layout: levels 0-9 = bid side, 10-19 = ask side
        # Feature index 0 = price_relative_to_mid in ticks
        # bid_best = level 0, ask_best = level 10
        # Spread = ask_best_price - bid_best_price
        # From feature 0 (price_relative_to_mid): bid[0] < 0, ask[0] > 0
        # spread = ask_best_offset - bid_best_offset (both in ticks from mid)
        bid_best_offset = bt[:, 0, 0]   # level 0 (best bid), feature 0 = price offset in ticks
        ask_best_offset = bt[:, 10, 0]  # level 10 (best ask), feature 0 = price offset in ticks
        # Spread = ask_offset - bid_offset (should be positive, typically 1 tick for ES)
        spreads = ask_best_offset - bid_best_offset
        # Filter to plausible range (0.5 to 10 ticks)
        valid_spreads = spreads[(spreads >= 0.5) & (spreads <= 10.0)]
        if len(valid_spreads) < 10:
            daily_spread = float(np.median(np.abs(spreads)))
        else:
            daily_spread = float(np.median(valid_spreads))

        return {
            "date": date_str,
            "daily_rvol_bps": daily_rvol_bps,
            "daily_rvol_ticks": daily_rvol_ticks,
            "daily_spread": daily_spread,
            "n_bars": n,
            "mid_range_ticks": float((mid.max() - mid.min()) / 0.25),
        }

    except Exception as e:
        print(f"  ERROR loading {npz_path.name}: {e}")
        return None


def fit_hmm(features_2d: np.ndarray, n_iter: int = 200) -> GaussianHMM:
    """Fit a 2-state Gaussian HMM on the given (n_days, 2) feature matrix."""
    model = GaussianHMM(
        n_components=2,
        covariance_type="full",
        n_iter=n_iter,
        random_state=42,
        tol=1e-4,
    )
    model.fit(features_2d)
    return model


def assign_active_state(model: GaussianHMM, scaler: StandardScaler) -> int:
    """
    Determine which HMM state (0 or 1) corresponds to the "active" (high-IC) regime.

    IMPORTANT: For BookSpatialCNN on ES futures, IC is NEGATIVELY correlated with
    realized volatility (Pearson r=-0.90, Spearman r=-0.97). The model predicts best
    on quiet, low-volatility days — NOT high-vol days. This is consistent with the
    order book being more informative when the market is in a trending/persistent regime
    rather than a noisy high-vol regime where order book signals get swamped.

    Active (tradeable) = LOWER realized volatility state.
    Returns: 0 or 1
    """
    # Inverse transform the means to original scale
    means_original = scaler.inverse_transform(model.means_)
    # Feature 0 = daily_rvol; LOWER rvol = active/tradeable state
    state0_rvol = means_original[0, 0]
    state1_rvol = means_original[1, 0]
    return 0 if state0_rvol < state1_rvol else 1


def main():
    parser = argparse.ArgumentParser(description="HMM Regime Filter for BookSpatialCNN")
    parser.add_argument("--is-dir", type=Path, default=IS_DIR)
    parser.add_argument("--oot-dir", type=Path, default=OOT_DIR)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("HMM Regime Filter — 2-State Gaussian HMM")
    print(f"IS dir : {args.is_dir}")
    print(f"OOT dir: {args.oot_dir}")
    print("=" * 70)

    # ── Step 1: Extract daily features ──
    print("\n[1/4] Extracting daily features from IS data...")
    is_files = sorted(args.is_dir.glob("*_book_tensors.npz"))
    oot_files = sorted(args.oot_dir.glob("*_book_tensors.npz"))

    print(f"  IS files:  {len(is_files)} days")
    print(f"  OOT files: {len(oot_files)} days")

    is_features = []
    for f in is_files:
        feat = extract_daily_features(f)
        if feat is not None:
            is_features.append(feat)
    print(f"  IS features extracted: {len(is_features)} days ({is_files[0].name[:10]} to {is_files[-1].name[:10]})")

    oot_features = []
    for f in oot_files:
        feat = extract_daily_features(f)
        if feat is not None:
            oot_features.append(feat)
    print(f"  OOT features extracted: {len(oot_features)} days ({oot_files[0].name[:10]} to {oot_files[-1].name[:10]})")

    # ── Step 2: Fit HMM on IS data only ──
    print("\n[2/4] Fitting 2-state Gaussian HMM on IS data...")
    is_X_raw = np.array([[f["daily_rvol_bps"], f["daily_spread"]] for f in is_features])

    scaler_is = StandardScaler()
    is_X_scaled = scaler_is.fit_transform(is_X_raw)

    hmm_is = fit_hmm(is_X_scaled, n_iter=200)
    active_state_is = assign_active_state(hmm_is, scaler_is)

    print(f"  IS HMM converged: {hmm_is.monitor_.converged}")
    print(f"  State 0 mean (rvol_bps, spread): {scaler_is.inverse_transform(hmm_is.means_)[0]}")
    print(f"  State 1 mean (rvol_bps, spread): {scaler_is.inverse_transform(hmm_is.means_)[1]}")
    print(f"  Active/tradeable state = {active_state_is} (LOWER rvol — IC is negatively correlated with rvol)")
    print(f"  Transition matrix:\n{hmm_is.transmat_}")

    # IS regime assignments (for reference)
    is_states_raw = hmm_is.predict(is_X_scaled)
    is_states = {f["date"]: int(1 if s == active_state_is else 0) for f, s in zip(is_features, is_states_raw)}
    n_is_active = sum(is_states.values())
    print(f"\n  IS regime: {n_is_active}/{len(is_features)} active days ({n_is_active/len(is_features)*100:.1f}%)")

    # Print IS regime monthly breakdown
    from collections import defaultdict
    monthly_is = defaultdict(lambda: {"active": 0, "total": 0})
    for feat, state in zip(is_features, is_states_raw):
        month = feat["date"][:7]
        monthly_is[month]["total"] += 1
        if state == active_state_is:
            monthly_is[month]["active"] += 1
    print("\n  IS monthly regime:")
    for month in sorted(monthly_is):
        m = monthly_is[month]
        print(f"    {month}: {m['active']}/{m['total']} active ({m['active']/m['total']*100:.0f}%)")

    # ── Step 3: Walk-forward OOT prediction ──
    print("\n[3/4] Walk-forward OOT regime prediction...")
    print("  (Expanding window: IS + prior OOT days, no look-ahead)")

    oot_states = {}  # date -> 0 (dead) or 1 (active)
    oot_probs = {}   # date -> probability of active state

    for i, oot_feat in enumerate(oot_features):
        oot_date = oot_feat["date"]

        # Build training set: IS + OOT[0..i-1] (1-day purge gap)
        train_features = is_features + oot_features[:max(0, i - 1)]
        X_raw = np.array([[f["daily_rvol_bps"], f["daily_spread"]] for f in train_features])

        # Fit scaler and HMM on training data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        try:
            hmm = fit_hmm(X_scaled, n_iter=200)
            active_state = assign_active_state(hmm, scaler)

            # Predict state for current OOT day
            x_curr = np.array([[oot_feat["daily_rvol_bps"], oot_feat["daily_spread"]]])
            x_curr_scaled = scaler.transform(x_curr)

            # Use predict_proba for soft assignment
            # viterbi state
            state_pred = hmm.predict(x_curr_scaled)[0]
            # posterior probability
            log_probs = hmm.predict_proba(x_curr_scaled)
            prob_active = float(log_probs[0, active_state])

            is_active = int(state_pred == active_state)
            oot_states[oot_date] = is_active
            oot_probs[oot_date] = round(prob_active, 4)

            if args.verbose or i < 5:
                print(f"  {oot_date}: state={'ACTIVE' if is_active else 'DEAD  '} (p_active={prob_active:.3f}) "
                      f"rvol={oot_feat['daily_rvol_bps']:.4f}bps spread={oot_feat['daily_spread']:.3f}t "
                      f"train_days={len(train_features)}")

        except Exception as e:
            print(f"  ERROR fold {i} ({oot_date}): {e}")
            oot_states[oot_date] = -1  # unknown
            oot_probs[oot_date] = 0.0

    # ── Step 4: Analysis ──
    print("\n[4/4] Analysis: Regime vs IC cross-reference")
    print("=" * 70)

    n_active = sum(1 for v in oot_states.values() if v == 1)
    n_dead = sum(1 for v in oot_states.values() if v == 0)
    n_unknown = sum(1 for v in oot_states.values() if v == -1)
    print(f"\nOOT Regime Summary:")
    print(f"  Active: {n_active}/{len(oot_states)} ({n_active/len(oot_states)*100:.1f}%)")
    print(f"  Dead:   {n_dead}/{len(oot_states)} ({n_dead/len(oot_states)*100:.1f}%)")
    if n_unknown > 0:
        print(f"  Unknown: {n_unknown}")

    # Cross-reference with known ICs
    print("\nOOT Day-by-Day: Date | Regime | p_active | IC (if known) | rvol_bps | spread")
    print("-" * 80)

    known_ic_active = []
    known_ic_dead = []

    oot_feat_by_date = {f["date"]: f for f in oot_features}

    for date in sorted(oot_states.keys()):
        state = oot_states[date]
        prob = oot_probs.get(date, 0.0)
        regime_str = "ACTIVE" if state == 1 else ("DEAD  " if state == 0 else "UNKNWN")
        ic_str = f"IC={KNOWN_ICS[date]:+.4f}" if date in KNOWN_ICS else "IC=N/A"
        feat = oot_feat_by_date.get(date, {})
        rvol = feat.get("daily_rvol_bps", 0)
        spread = feat.get("daily_spread", 0)

        print(f"  {date} | {regime_str} | p={prob:.3f} | {ic_str} | rvol={rvol:.4f}bps | spread={spread:.3f}t")

        if date in KNOWN_ICS:
            ic = KNOWN_ICS[date]
            if state == 1:
                known_ic_active.append(ic)
            elif state == 0:
                known_ic_dead.append(ic)

    # Summary statistics for known ICs
    print("\n--- IC by Regime (days with known IC) ---")
    if known_ic_active:
        arr_a = np.array(known_ic_active)
        print(f"  ACTIVE days: mean IC={arr_a.mean():.4f}, std={arr_a.std():.4f}, n={len(arr_a)}, "
              f"pct_pos={np.mean(arr_a > 0)*100:.0f}%")
    if known_ic_dead:
        arr_d = np.array(known_ic_dead)
        print(f"  DEAD days:   mean IC={arr_d.mean():.4f}, std={arr_d.std():.4f}, n={len(arr_d)}, "
              f"pct_pos={np.mean(arr_d > 0)*100:.0f}%")

    if known_ic_active and known_ic_dead:
        arr_a = np.array(known_ic_active)
        arr_d = np.array(known_ic_dead)
        improvement = (arr_a.mean() - arr_d.mean()) / arr_d.mean() * 100 if arr_d.mean() != 0 else float("inf")
        gated_pct = n_active / len(oot_states) * 100
        all_ics = known_ic_active + known_ic_dead
        print(f"\n  All-days mean IC: {np.mean(all_ics):.4f}")
        print(f"  Active-only mean IC: {arr_a.mean():.4f} ({improvement:+.1f}% vs all-days)")
        print(f"  Dead-only mean IC: {arr_d.mean():.4f}")
        print(f"  Trading day reduction: {100 - gated_pct:.1f}% fewer days traded")

    # Load walkforward fold stats for more complete IC analysis
    wf_stats_path = RESULTS_DIR / "walkforward_fold_stats.json"
    if wf_stats_path.exists():
        print("\n--- Cross-reference with walkforward_fold_stats.json ---")
        with open(wf_stats_path) as f:
            wf_stats = json.load(f)

        wf_by_date = {s["date"]: s for s in wf_stats}

        active_wf_ics = []
        dead_wf_ics = []

        print(f"\n  {'Date':<12} {'Regime':<8} {'IC':>7} {'rvol_ticks':>11} {'IC_morning':>11} {'IC_afternoon':>13}")
        print(f"  {'-'*70}")
        for date in sorted(oot_states.keys()):
            if date not in wf_by_date:
                continue
            s = wf_by_date[date]
            state = oot_states[date]
            regime_str = "ACTIVE" if state == 1 else "DEAD  "
            ic_m = s.get("ic_morning", float("nan"))
            ic_a = s.get("ic_afternoon", float("nan"))
            ic_m_str = f"{ic_m:+.4f}" if np.isfinite(ic_m) else "   N/A"
            ic_a_str = f"{ic_a:+.4f}" if np.isfinite(ic_a) else "   N/A"
            print(f"  {date} {regime_str} {s['ic']:+.4f} {s['realized_vol_ticks']:>11.4f} {ic_m_str:>11} {ic_a_str:>13}")

            if state == 1:
                active_wf_ics.append(s["ic"])
            elif state == 0:
                dead_wf_ics.append(s["ic"])

        if active_wf_ics:
            print(f"\n  ACTIVE WF ICs: n={len(active_wf_ics)} mean={np.mean(active_wf_ics):.4f} std={np.std(active_wf_ics):.4f}")
        if dead_wf_ics:
            print(f"  DEAD WF ICs:   n={len(dead_wf_ics)} mean={np.mean(dead_wf_ics):.4f} std={np.std(dead_wf_ics):.4f}")
        if active_wf_ics and dead_wf_ics:
            all_wf = active_wf_ics + dead_wf_ics
            print(f"  All WF ICs:    n={len(all_wf)} mean={np.mean(all_wf):.4f}")

    # Feature distribution analysis
    print("\n--- Feature Analysis by Regime ---")
    active_feats = [oot_feat_by_date[d] for d in oot_states if oot_states[d] == 1 and d in oot_feat_by_date]
    dead_feats = [oot_feat_by_date[d] for d in oot_states if oot_states[d] == 0 and d in oot_feat_by_date]

    if active_feats:
        a_rvol = np.array([f["daily_rvol_bps"] for f in active_feats])
        a_spread = np.array([f["daily_spread"] for f in active_feats])
        print(f"  ACTIVE: rvol_bps={a_rvol.mean():.4f}±{a_rvol.std():.4f}, "
              f"spread={a_spread.mean():.3f}±{a_spread.std():.3f}t")
    if dead_feats:
        d_rvol = np.array([f["daily_rvol_bps"] for f in dead_feats])
        d_spread = np.array([f["daily_spread"] for f in dead_feats])
        print(f"  DEAD:   rvol_bps={d_rvol.mean():.4f}±{d_rvol.std():.4f}, "
              f"spread={d_spread.mean():.3f}±{d_spread.std():.3f}t")

    # IS feature distributions for reference
    is_rvols = np.array([f["daily_rvol_bps"] for f in is_features])
    is_spreads = np.array([f["daily_spread"] for f in is_features])
    print(f"\n  IS (training) stats:")
    print(f"    rvol_bps: mean={is_rvols.mean():.4f} std={is_rvols.std():.4f} "
          f"min={is_rvols.min():.4f} max={is_rvols.max():.4f}")
    print(f"    spread:   mean={is_spreads.mean():.3f} std={is_spreads.std():.3f} "
          f"min={is_spreads.min():.3f} max={is_spreads.max():.3f}")
    print(f"    10th pct rvol: {np.percentile(is_rvols, 10):.4f}bps")
    print(f"    50th pct rvol: {np.percentile(is_rvols, 50):.4f}bps")
    print(f"    90th pct rvol: {np.percentile(is_rvols, 90):.4f}bps")

    # ── Save output ──
    output = {
        "metadata": {
            "created": datetime.now().isoformat(),
            "description": "2-state Gaussian HMM regime filter (active=1=low-rvol, dead=0=high-rvol). NOTE: IC negatively correlates with rvol (r=-0.97), so LOW vol = tradeable.",
            "is_days": len(is_features),
            "oot_days": len(oot_features),
            "method": "walk-forward expanding window, 1-day purge gap",
            "features": ["daily_rvol_bps", "daily_spread_ticks"],
            "active_state": "lower_rvol (IC negatively correlated with rvol: r=-0.97)",
        },
        "is_model": {
            "state_means_original": {
                "state_0": scaler_is.inverse_transform(hmm_is.means_)[0].tolist(),
                "state_1": scaler_is.inverse_transform(hmm_is.means_)[1].tolist(),
            },
            "active_state_id": int(active_state_is),
            "transition_matrix": hmm_is.transmat_.tolist(),
            "n_is_active": n_is_active,
            "n_is_dead": len(is_features) - n_is_active,
        },
        "is_states": {f["date"]: int(1 if is_states_raw[i] == active_state_is else 0)
                      for i, f in enumerate(is_features)},
        "oot_states": oot_states,
        "oot_probs_active": oot_probs,
        "oot_features": {f["date"]: {k: v for k, v in f.items() if k != "date"} for f in oot_features},
        "ic_analysis": {
            "active_days": {
                "n": len(known_ic_active),
                "mean_ic": float(np.mean(known_ic_active)) if known_ic_active else None,
                "std_ic": float(np.std(known_ic_active)) if known_ic_active else None,
                "ics": {date: KNOWN_ICS[date] for date in KNOWN_ICS
                        if date in oot_states and oot_states[date] == 1},
            },
            "dead_days": {
                "n": len(known_ic_dead),
                "mean_ic": float(np.mean(known_ic_dead)) if known_ic_dead else None,
                "std_ic": float(np.std(known_ic_dead)) if known_ic_dead else None,
                "ics": {date: KNOWN_ICS[date] for date in KNOWN_ICS
                        if date in oot_states and oot_states[date] == 0},
            },
        },
    }

    out_path = RESULTS_DIR / "hmm_regime_states.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")
    print("=" * 70)
    print("Done.")

    return output


if __name__ == "__main__":
    main()
