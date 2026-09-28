"""
Volume Profile IC Screening — Saturn CPU research.

1. Loads mbo_events_vp/ files (22 features including VP)
2. Computes 10s forward return from price_rel_ticks + time_delta_log
3. Computes Spearman IC for ALL 22 features vs 10s return
4. Computes IC conditioned on VP regime:
   - NEAR_POC: |session_poc_dist| < 2 ticks
   - AT_EXTREME: session_va_pos < 0.1 or > 0.9 (at VAH/VAL)
   - MID_VA: 0.3 < session_va_pos < 0.7
5. Saves results to screen_vp_ic_results.json

Usage: python screen_vp_ic.py [--src mbo_events_vp] [--workers 4] [--max-files 20]
"""

import os, sys, json, logging, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# Column definitions (22 features)
FEATURE_NAMES = [
    "time_delta_log",       # 0
    "event_type_id",        # 1
    "side_id",              # 2
    "price_rel_ticks",      # 3
    "qty_log",              # 4
    "spread_ticks",         # 5
    "cancel_side_asym_50",  # 6
    "rolling_ofi_500",      # 7
    "event_density_20",     # 8
    "price_mom_10",         # 9
    "qty_price_mom_50",     # 10
    "price_sign_mom_200",   # 11
    "event_type_entropy",   # 12
    "fill_add_restore_100", # 13
    "spread_velocity_50",   # 14
    "session_poc_dist",     # 15
    "session_va_pos",       # 16
    "session_above_poc_vol",# 17
    "rolling_poc_dist",     # 18
    "rolling_va_pos",       # 19
    "vol_at_price_ratio",   # 20
    "time_of_day_norm",     # 21
]

N_FEATURES = len(FEATURE_NAMES)
COL_PRICE  = 3
COL_TIME   = 0
COL_SESSION_POC_DIST = 15
COL_SESSION_VA_POS   = 16

# 10s forward return approximation: find next event ~10s later by time_delta_log
# time_delta_log = log(1 + cumulative_seconds) — use price_rel_ticks diff as proxy return
FORWARD_SECONDS = 10.0


def compute_forward_returns(ev: np.ndarray) -> np.ndarray:
    """Approximate 10s forward return using price_rel_ticks."""
    N = len(ev)
    price   = ev[:, COL_PRICE].astype(np.float64)
    raw_time = np.cumsum(np.expm1(np.clip(ev[:, COL_TIME].astype(np.float64), 0, 20)))

    fwd_price = np.full(N, np.nan)
    j = 0
    for i in range(N):
        # advance j until we're >= FORWARD_SECONDS ahead
        target = raw_time[i] + FORWARD_SECONDS
        while j < N and raw_time[j] < target:
            j += 1
        if j < N:
            fwd_price[i] = price[j]

    fwd_ret = fwd_price - price
    return fwd_ret.astype(np.float32)


def screen_file(src_path: Path) -> dict:
    """Compute IC for all features + regime breakdown for one file."""
    data = np.load(src_path, allow_pickle=False)
    ev = data["events"].astype(np.float32)
    N, F = ev.shape

    if F < 13:
        return {"file": src_path.name, "error": f"too few features: {F} (need >=13)"}
    # Adapt: VP features are always the last 7 columns
    VP_START = F - 7
    COL_SESSION_POC_DIST_DYN = VP_START
    COL_SESSION_VA_POS_DYN   = VP_START + 1

    fwd_ret = compute_forward_returns(ev)
    valid   = ~np.isnan(fwd_ret)

    if valid.sum() < 1000:
        return {"file": src_path.name, "error": "too few valid events"}

    fwd_v  = fwd_ret[valid]
    ev_v   = ev[valid]

    # ── Overall IC ─────────────────────────────────────────────────────────────
    ic_all = {}
    for i, name in enumerate(FEATURE_NAMES[:F]):
        feat = ev_v[:, i]
        if np.std(feat) < 1e-8:
            ic_all[name] = 0.0
            continue
        r, _ = spearmanr(feat, fwd_v)
        ic_all[name] = float(r) if not np.isnan(r) else 0.0

    # ── Regime-conditioned IC ──────────────────────────────────────────────────
    poc_dist = ev_v[:, VP_START]
    va_pos   = ev_v[:, VP_START + 1]

    regimes = {
        "near_poc":  np.abs(poc_dist) < 2.0,
        "at_extreme": (va_pos < 0.1) | (va_pos > 0.9),
        "mid_va":    (va_pos >= 0.3) & (va_pos <= 0.7),
        "above_poc": poc_dist > 0,
        "below_poc": poc_dist < 0,
    }

    ic_by_regime = {}
    for regime_name, mask in regimes.items():
        if mask.sum() < 500:
            ic_by_regime[regime_name] = {}
            continue
        ic_r = {}
        for i, name in enumerate(FEATURE_NAMES[:F]):
            feat = ev_v[mask, i]
            if np.std(feat) < 1e-8:
                ic_r[name] = 0.0; continue
            r, _ = spearmanr(feat, fwd_v[mask])
            ic_r[name] = float(r) if not np.isnan(r) else 0.0
        ic_by_regime[regime_name] = ic_r

    return {
        "file":          src_path.name,
        "n_events":      int(N),
        "n_valid":       int(valid.sum()),
        "ic_all":        ic_all,
        "ic_by_regime":  ic_by_regime,
        "regime_counts": {k: int(v.sum()) for k, v in regimes.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src",       default="mbo_events_vp")
    ap.add_argument("--workers",   type=int, default=4)
    ap.add_argument("--max-files", type=int, default=0, help="limit files for quick test")
    ap.add_argument("--out",       default="screen_vp_ic_results.json")
    args = ap.parse_args()

    data_root = Path(os.environ.get("DATA_ROOT", "/home/saturn/Lvl3Quant/data/processed"))
    src_dir   = data_root / args.src
    out_path  = data_root.parent / "results" / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob("*.npz"))
    if args.max_files:
        files = files[:args.max_files]

    log.info(f"Screening {len(files)} files from {src_dir}")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(screen_file, f): f for f in files}
        for fut in as_completed(futures):
            try:
                r = fut.result()
                results.append(r)
                if "error" in r:
                    log.warning(f"{r['file']}: {r['error']}")
                else:
                    # Print top 5 IC features
                    top5 = sorted(r["ic_all"].items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                    log.info(f"{r['file']} | n={r['n_events']:,} | "
                             f"top IC: {', '.join(f'{k}={v:.4f}' for k,v in top5)}")
            except Exception as e:
                log.error(f"FAILED {futures[fut].name}: {e}")

    # ── Aggregate across files ─────────────────────────────────────────────────
    good = [r for r in results if "error" not in r]
    if good:
        agg_ic = {name: np.mean([r["ic_all"].get(name, 0.0) for r in good])
                  for name in FEATURE_NAMES}
        sorted_ic = sorted(agg_ic.items(), key=lambda x: abs(x[1]), reverse=True)

        log.info("\n=== AGGREGATE IC (mean across files) ===")
        for name, ic in sorted_ic:
            log.info(f"  {name:30s}: IC={ic:+.4f}")

        # Regime deltas: near_poc vs at_extreme for key features
        log.info("\n=== REGIME BREAKDOWN (near_poc vs at_extreme) ===")
        for name in FEATURE_NAMES:
            near_poc_ics = [r["ic_by_regime"].get("near_poc", {}).get(name, 0.0)
                            for r in good if "near_poc" in r.get("ic_by_regime", {})]
            extreme_ics  = [r["ic_by_regime"].get("at_extreme", {}).get(name, 0.0)
                            for r in good if "at_extreme" in r.get("ic_by_regime", {})]
            if near_poc_ics and extreme_ics:
                log.info(f"  {name:30s}: near_poc={np.mean(near_poc_ics):+.4f} "
                         f"at_extreme={np.mean(extreme_ics):+.4f}")

    with open(out_path, "w") as fh:
        json.dump({"results": results, "feature_names": FEATURE_NAMES}, fh, indent=2)
    log.info(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
