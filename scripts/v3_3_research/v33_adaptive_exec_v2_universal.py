"""
v33_adaptive_exec_v2_universal.py — HC #375 Track B v2 — Universal entry + alpha-model adaptive exit.

v1 finding (2026-05-15 18:42 ET): K=2 confluence entry filter HURTS the adaptive-exit
signal. Within-day permutation null on K=2 entries gave Sharpe +5.54 (random strides)
> Sharpe +4.40 (K=2 entries). The alpha-model evolving-Δ exit signal works broadly,
not K=2-specifically.

v2 design (honest test):
  1. UNIVERSAL ENTRY: every K-th stride (default K=20 = every 5s) → ~12k candidate
     LONG entries across 5 OOT days.
  2. HELD-OUT TUNING: split each day's strides into TRAIN (first 60%) and TEST (last
     40%). Grid-search (α, θ) on TRAIN, evaluate FIXED policy on TEST.
  3. SAME EXIT LOGIC as v1: walk forward 250ms strides, candidate exit horizons
     {1s, 5s, 10s, 30s}, exit when score < θ.
  4. PERMUTATION NULL on TEST: shuffle stride_idx within day, re-evaluate FIXED policy.
  5. SHARPE COMPARISON: adaptive vs static-30s-from-same-entry. Static = realized
     `target_log_ret_30s[entry] - 0.376`.

Falsification gate: TEST Sharpe > static-30s TEST Sharpe AND p<0.10 under perm null.

CRITICAL HEAD-VALIDITY ASSERT (HC #376): only uses trained heads from
fold_00_predictions.metrics.json (NOT log_ret_60s, NOT mfe_60s, NOT mae_60s).

Output: output/v3_3_full_execution_analysis_20260514/adaptive_exec_v2/
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/jupiter/Lvl3Quant")
PRED_NPZ = ROOT / "output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz"
METRICS_JSON = ROOT / "output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.metrics.json"
OUT_DIR = ROOT / "output/v3_3_full_execution_analysis_20260514/adaptive_exec_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RT_COMMISSION_TICKS = 4.70 / 12.50  # 0.376

STRIDE_PER_SEC = 4
EXIT_HORIZONS_SEC = [1, 5, 10, 30]
EXIT_STRIDES = [h * STRIDE_PER_SEC for h in EXIT_HORIZONS_SEC]
STATIC_BASELINE_SEC = 30

# Heads (HC #376 verified trained)
HEADS = {
    "hit_tp": "pred_fifo_tp4sl3_hit_tp",
    "mfe30": "pred_pred_mfe_30s_ticks",
    "logret30": "pred_log_ret_30s",
    "mae30": "pred_pred_mae_30s_ticks",
}
REQUIRED_TRAINED = ["ic_log_ret_30s", "corr_pred_mfe_30s_ticks", "corr_pred_mae_30s_ticks"]

UNIVERSAL_ENTRY_EVERY = 20  # every 5s
TRAIN_FRAC = 0.60


def validate_heads():
    m = json.load(open(METRICS_JSON))
    for k in REQUIRED_TRAINED:
        if k not in m or m[k] is None or (isinstance(m[k], float) and np.isnan(m[k])):
            raise RuntimeError(f"HC #376 violation: head metric {k} = {m.get(k)} (untrained or missing)")
    print(f"[head-audit] PASS — required heads {REQUIRED_TRAINED} all trained.", flush=True)
    return m


def load_npz():
    npz = np.load(PRED_NPZ, allow_pickle=True)
    n = int(npz["n_samples"])
    # Day boundaries: oot_dates × samples-per-day. NPZ doesn't carry per-stride dates,
    # but `oot_dates` shape gives us 5 days. We approximate equal split.
    oot_dates = list(npz["oot_dates"])
    n_days = len(oot_dates)
    samples_per_day = n // n_days
    day_idx = np.array([min(i // samples_per_day, n_days - 1) for i in range(n)], dtype=np.int32)
    heads = {k: npz[v] for k, v in HEADS.items()}
    realized = {f"r_{h}s": npz[f"target_log_ret_{h}s"] for h in EXIT_HORIZONS_SEC}
    return npz, heads, realized, day_idx, oot_dates


def make_universal_entries(n_strides: int, day_idx: np.ndarray, max_hold_strides: int):
    """Every K-th stride within each day, with enough forward room."""
    entries = []
    # Need entry + 30s*4 strides forward and not too close to next day boundary
    for d in np.unique(day_idx):
        day_mask = (day_idx == d)
        day_strides = np.where(day_mask)[0]
        # last possible entry: day_strides[-1] - max_hold_strides
        last_idx = day_strides[-max_hold_strides - 1] if len(day_strides) > max_hold_strides else None
        if last_idx is None:
            continue
        ent = day_strides[:: UNIVERSAL_ENTRY_EVERY]
        ent = ent[ent <= last_idx]
        entries.extend(ent.tolist())
    return np.array(sorted(entries), dtype=np.int64)


def apply_policy(entries: np.ndarray, heads: dict, realized: dict,
                 a_hit: float, a_mfe: float, a_log: float, a_mae: float, theta: float,
                 y_baseline_static30: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (realized_net_ticks, exit_horizon_sec). Vectorized over entries."""
    N = len(entries)
    exit_sec = np.full(N, STATIC_BASELINE_SEC, dtype=np.int32)
    pnl = y_baseline_static30.copy()  # default = static-30s outcome

    # Pre-stack head arrays for vectorized lookups
    H = {k: heads[k] for k in HEADS}
    R = realized

    # Entry-time references (vectorized)
    ref = {k: H[k][entries] for k in H}

    # For each candidate horizon, in order
    decided = np.zeros(N, dtype=bool)
    for hi, s in enumerate(EXIT_STRIDES):
        target_idx = entries + s
        valid = (target_idx < len(H["hit_tp"])) & (~decided)
        if not valid.any():
            continue
        d_hit = H["hit_tp"][target_idx[valid]] - ref["hit_tp"][valid]
        d_mfe = H["mfe30"][target_idx[valid]] - ref["mfe30"][valid]
        d_log = H["logret30"][target_idx[valid]] - ref["logret30"][valid]
        d_mae = H["mae30"][target_idx[valid]] - ref["mae30"][valid]
        score = a_hit * d_hit + a_mfe * d_mfe + a_log * d_log - a_mae * d_mae

        exit_now = (score < theta)
        h_sec = EXIT_HORIZONS_SEC[hi]
        idx_in_global = np.where(valid)[0][exit_now]
        if h_sec == STATIC_BASELINE_SEC:
            # equivalent to static, no change to pnl
            pass
        else:
            ent_exit = entries[idx_in_global]
            pnl[idx_in_global] = R[f"r_{h_sec}s"][ent_exit] - RT_COMMISSION_TICKS
        exit_sec[idx_in_global] = h_sec
        decided[idx_in_global] = True
    return pnl, exit_sec


def sharpe(p: np.ndarray) -> float:
    if len(p) < 2:
        return 0.0
    sd = p.std(ddof=1)
    return float(p.mean() / sd * np.sqrt(252.0)) if sd > 0 else 0.0


def grid_search(entries: np.ndarray, heads: dict, realized: dict, y_static30: np.ndarray):
    grid = []
    for a_hit in [0.0, 1.0, 2.0]:
        for a_mfe in [0.0, 1.0]:
            for a_log in [0.0, 1.0, 2.0]:
                for a_mae in [0.0, 1.0]:
                    if a_hit + a_mfe + a_log + a_mae == 0:
                        continue
                    for th in [-0.05, -0.10, -0.20, -0.30, -0.50]:
                        grid.append((a_hit, a_mfe, a_log, a_mae, th))
    best = None
    rows = []
    for cfg in grid:
        pnl, exits = apply_policy(entries, heads, realized, *cfg, y_static30)
        sh = sharpe(pnl)
        rows.append((cfg, sh, float(pnl.mean()), float((pnl > 0).mean())))
        if (best is None) or (sh > best[1]):
            best = (cfg, sh, pnl, exits)
    return best, rows


def main():
    t0 = time.time()
    metrics_meta = validate_heads()
    npz, heads, realized, day_idx, oot_dates = load_npz()
    n = len(heads["hit_tp"])
    print(f"[load] strides={n} days={oot_dates}", flush=True)

    entries_all = make_universal_entries(n, day_idx, max_hold_strides=STATIC_BASELINE_SEC * STRIDE_PER_SEC)
    # Filter to only strides where ALL realized targets are valid (non-NaN)
    valid_mask = np.ones(len(entries_all), dtype=bool)
    for h in EXIT_HORIZONS_SEC:
        rvals = realized[f"r_{h}s"][entries_all]
        valid_mask &= ~np.isnan(rvals)
    # Also filter heads non-NaN
    for k in HEADS:
        valid_mask &= ~np.isnan(heads[k][entries_all])
    entries_all = entries_all[valid_mask]
    print(f"[entries] universal (every {UNIVERSAL_ENTRY_EVERY} strides, valid-realized only): {len(entries_all)} LONG entries", flush=True)

    # Static-30s baseline at each universal entry
    y_static30 = realized["r_30s"][entries_all] - RT_COMMISSION_TICKS
    static_sh = sharpe(y_static30)
    print(f"[static-30s @ universal entries] mean={y_static30.mean():+.4f}t  Sharpe={static_sh:+.3f}  WR={(y_static30>0).mean():.3f}  n={len(y_static30)}", flush=True)

    # Split entries TRAIN / TEST within each day
    train_mask = np.zeros(len(entries_all), dtype=bool)
    for d in np.unique(day_idx[entries_all]):
        idx_in_day = np.where(day_idx[entries_all] == d)[0]
        cut = int(len(idx_in_day) * TRAIN_FRAC)
        train_mask[idx_in_day[:cut]] = True
    print(f"[split] train n={train_mask.sum()}  test n={(~train_mask).sum()}", flush=True)

    train_entries = entries_all[train_mask]
    test_entries = entries_all[~train_mask]
    y_train_static = realized["r_30s"][train_entries] - RT_COMMISSION_TICKS
    y_test_static = realized["r_30s"][test_entries] - RT_COMMISSION_TICKS

    # Grid search on TRAIN
    print(f"[train] grid-searching (α, θ) ...", flush=True)
    best, _ = grid_search(train_entries, heads, realized, y_train_static)
    best_cfg, best_train_sh, _, _ = best
    print(f"[train] BEST cfg={best_cfg}  Sharpe_train={best_train_sh:+.3f}", flush=True)

    # Apply FIXED policy to TEST
    pnl_test, exits_test = apply_policy(test_entries, heads, realized, *best_cfg, y_test_static)
    test_sh = sharpe(pnl_test)
    test_wr = float((pnl_test > 0).mean())
    test_mean = float(pnl_test.mean())
    test_pf_num = float(pnl_test[pnl_test > 0].sum())
    test_pf_den = -float(pnl_test[pnl_test < 0].sum())
    test_pf = test_pf_num / test_pf_den if test_pf_den > 0 else float("inf")

    # Day concentration on TEST
    test_days = day_idx[test_entries]
    daily_pnl = {int(d): float(pnl_test[test_days == d].sum()) for d in np.unique(test_days)}
    total_abs = sum(abs(v) for v in daily_pnl.values())
    max_share = max(abs(v) for v in daily_pnl.values()) / total_abs if total_abs > 0 else 0.0

    # Static-30s comparable on TEST
    static_test_sh = sharpe(y_test_static)
    print(f"\n[TEST] adaptive: Sharpe={test_sh:+.3f}  mean={test_mean:+.4f}t  WR={test_wr:.3f}  PF={test_pf:.2f}  day_conc={max_share:.3f}", flush=True)
    print(f"[TEST] static-30s: Sharpe={static_test_sh:+.3f}  mean={y_test_static.mean():+.4f}t  WR={(y_test_static>0).mean():.3f}", flush=True)
    print(f"[TEST] ΔSharpe = {test_sh - static_test_sh:+.3f}", flush=True)

    # Permutation null on TEST: shuffle test_entries within day, re-evaluate FIXED policy
    n_perm = 100
    print(f"\n[null] {n_perm} within-day permutations on TEST with FIXED policy ...", flush=True)
    rng = np.random.default_rng(42)
    null_sh = []
    test_day_arr = day_idx[test_entries].copy()
    # Pool of test-day strides to draw from (within-day shuffle of stride positions)
    day_pools = {}
    for d in np.unique(test_day_arr):
        day_strides = np.where(day_idx == d)[0]
        # restrict to those with forward room AND valid realized + heads
        day_strides = day_strides[day_strides + EXIT_STRIDES[-1] < n]
        # Filter to non-NaN
        valid = np.ones(len(day_strides), dtype=bool)
        for h in EXIT_HORIZONS_SEC:
            valid &= ~np.isnan(realized[f"r_{h}s"][day_strides])
        for k in HEADS:
            valid &= ~np.isnan(heads[k][day_strides])
        day_strides = day_strides[valid]
        day_pools[int(d)] = day_strides
    for pi in range(n_perm):
        shuffled = test_entries.copy()
        for d in np.unique(test_day_arr):
            mask = (test_day_arr == d)
            pool = day_pools[int(d)]
            replacement = rng.choice(pool, size=mask.sum(), replace=False)
            shuffled[mask] = replacement
        y_static_shuf = realized["r_30s"][shuffled] - RT_COMMISSION_TICKS
        pnl_shuf, _ = apply_policy(shuffled, heads, realized, *best_cfg, y_static_shuf)
        null_sh.append(sharpe(pnl_shuf))
    null_arr = np.array(null_sh)
    p_value = float((null_arr >= test_sh).mean())
    print(f"[null] null_mean={null_arr.mean():+.3f}  p95={np.percentile(null_arr,95):+.3f}  max={null_arr.max():+.3f}  p_value={p_value:.3f}  PASS_p10={'YES' if p_value < 0.10 else 'NO'}", flush=True)

    out = dict(
        elapsed_sec=time.time() - t0,
        oot_dates=[str(d) for d in oot_dates],
        n_entries_total=int(len(entries_all)),
        n_train=int(train_mask.sum()),
        n_test=int((~train_mask).sum()),
        universal_entry_every_strides=UNIVERSAL_ENTRY_EVERY,
        train_frac=TRAIN_FRAC,
        best_cfg=dict(alpha_hit=best_cfg[0], alpha_mfe=best_cfg[1], alpha_log=best_cfg[2], alpha_mae=best_cfg[3], theta=best_cfg[4]),
        train_sharpe=best_train_sh,
        test=dict(
            adaptive=dict(sharpe=test_sh, mean_ticks=test_mean, wr=test_wr, pf=test_pf, day_conc=max_share, daily_pnl=daily_pnl),
            static_30s=dict(sharpe=static_test_sh, mean_ticks=float(y_test_static.mean()), wr=float((y_test_static>0).mean())),
            delta_sharpe=test_sh - static_test_sh,
        ),
        null=dict(n_perm=n_perm, null_mean=float(null_arr.mean()), null_p95=float(np.percentile(null_arr,95)), null_max=float(null_arr.max()), p_value=p_value, pass_p10=bool(p_value < 0.10)),
    )
    out_path = OUT_DIR / "adaptive_exec_v2_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[done] elapsed={out['elapsed_sec']:.1f}s  output: {out_path}", flush=True)

    if p_value < 0.10 and test_sh > static_test_sh and max_share <= 0.30:
        print(f"[VERDICT] ✓ Held-out Sharpe lift PASSES under null. Best cfg={best_cfg}.", flush=True)
    else:
        reasons = []
        if p_value >= 0.10: reasons.append(f"p={p_value:.3f}≥0.10")
        if test_sh <= static_test_sh: reasons.append(f"ΔSh={test_sh - static_test_sh:+.2f}≤0")
        if max_share > 0.30: reasons.append(f"day_conc={max_share:.3f}>0.30 (need extended OOT)")
        print(f"[VERDICT] ✗ FAIL: {', '.join(reasons)}", flush=True)


if __name__ == "__main__":
    main()
