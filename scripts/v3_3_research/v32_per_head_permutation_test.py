#!/usr/bin/env python3
"""
HC #345 v3 follow-up — Permutation test on the TOP candidates from
v32_per_head_tick_dashboard.

Pass 8 of all-night research showed: even Top5% RAW (no filter) gets Sharpe 3.51,
because the FIFO+ToD selection structure itself produces apparent edge from
random predictions. Test that explanation against the new per-head candidates.

Method: For each top candidate (head × side × band):
  - Hold the band-selection structure constant (same n_signals, same fill mask)
  - Shuffle the SIGN of the head's prediction 1000x
  - For each shuffle, recompute LONG/SHORT split → re-derive top-band → fifo_tp4 mean
  - p-value = fraction of shuffles that produce ≥ observed mean

Output:
  permutation_results.csv   per-candidate {observed_mean, null_mean, null_p95, p_value}
  permutation_summary.md    surviving candidates (p < 0.05)

NO trainer code modified. Pure analysis. HC #307D.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path("/home/jupiter/Lvl3Quant")
PRED_NPZ = ROOT / "output/v3_2_deep_sim_20260512/fold_00_oot_predictions.npz"
DASH_DIR = ROOT / "output/v3_2_per_head_tick_dashboard_20260514"
OUT_DIR = ROOT / "output/v3_2_per_head_permutation_20260514"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_DIR / "build.log"

N_PERM = 1000
TOP_K_CANDIDATES = int(__import__("os").environ.get("TOP_K_CANDIDATES", "30"))


def log(msg: str):
    ts = datetime.utcnow().isoformat(timespec="seconds")
    line = f"[{ts}Z] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def perm_test(score: np.ndarray, fifo_net: np.ndarray, fifo_mask: np.ndarray,
              side_sign: int, band_pct: float, n_perm: int = N_PERM,
              seed: int = 42) -> dict:
    """Sign-shuffle permutation test.

    Returns {observed_mean, null_mean, null_p95, p_value, observed_n_fills}.
    """
    rng = np.random.default_rng(seed)

    # Observed
    sign = np.sign(score)
    magnitude = np.abs(score)
    NAN_RET = {"observed_mean": float("nan"), "p_value": float("nan"), "observed_n_fills": 0,
               "null_mean": float("nan"), "null_p95": float("nan"), "null_p99": float("nan")}
    side_mask = sign == side_sign
    if side_mask.sum() < 10:
        return NAN_RET
    mag_in_side = magnitude[side_mask]
    k = max(1, int(side_mask.sum() * band_pct))
    if k < 5:
        return NAN_RET
    thr = np.partition(mag_in_side, -k)[-k]
    sel_obs = side_mask & (magnitude >= thr)
    pnl_obs = (fifo_net[sel_obs] if side_sign > 0 else -fifo_net[sel_obs])[fifo_mask[sel_obs]]
    if len(pnl_obs) < 5:
        return NAN_RET
    observed_mean = float(pnl_obs.mean())
    observed_n = int(len(pnl_obs))

    # Null: shuffle the sign of `score`, re-derive top-band, recompute mean.
    null_means = np.empty(n_perm)
    n = len(score)
    for i in range(n_perm):
        # Random sign-flip of the score (preserves magnitude distribution)
        flips = rng.choice([-1, 1], size=n)
        score_p = score * flips
        sign_p = np.sign(score_p)
        magnitude_p = np.abs(score_p)  # same as magnitude
        side_mask_p = sign_p == side_sign
        if side_mask_p.sum() < 10:
            null_means[i] = 0.0
            continue
        mag_in_side_p = magnitude_p[side_mask_p]
        k_p = max(1, int(side_mask_p.sum() * band_pct))
        if k_p < 5:
            null_means[i] = 0.0
            continue
        thr_p = np.partition(mag_in_side_p, -k_p)[-k_p]
        sel_p = side_mask_p & (magnitude_p >= thr_p)
        pnl_p = (fifo_net[sel_p] if side_sign > 0 else -fifo_net[sel_p])[fifo_mask[sel_p]]
        null_means[i] = pnl_p.mean() if len(pnl_p) >= 5 else 0.0

    # One-tailed p (test for positive edge)
    p_value = float((null_means >= observed_mean).mean())
    return {
        "observed_mean": observed_mean,
        "observed_n_fills": observed_n,
        "null_mean": float(null_means.mean()),
        "null_p95": float(np.quantile(null_means, 0.95)),
        "null_p99": float(np.quantile(null_means, 0.99)),
        "p_value": p_value,
    }


def main():
    t0 = time.time()
    log("V32 PER-HEAD PERMUTATION TEST START")
    log(f"Loading dashboard from {DASH_DIR}")
    with open(DASH_DIR / "per_head_master.json") as f:
        cells = json.load(f)
    log(f"  {len(cells)} cells in dashboard")

    # Pick top-K candidates by passive_net_after_comm (with n_fills >= 30)
    cands = [c for c in cells if c["fifo_tp4sl3_n_fills"] >= 30]
    cands.sort(key=lambda c: -c["passive_net_after_comm"])
    cands = cands[:TOP_K_CANDIDATES]
    log(f"  Testing top {len(cands)} candidates by passive_net")

    log(f"Loading {PRED_NPZ}")
    d = np.load(PRED_NPZ)

    fifo_net = d["target_fifo_tp4sl3_net"].astype(np.float32)
    fifo_mask = d["mask_fifo_tp4sl3_net"].astype(bool)

    BAND_PCTS = {"Top0.1%": 0.001, "Top0.5%": 0.005, "Top1%": 0.01,
                 "Top5%": 0.05, "Top10%": 0.10, "Top20%": 0.20}

    results = []
    for i, c in enumerate(cands):
        head = c["head"]
        side = c["side"]
        band = c["band"]
        side_sign = +1 if side == "LONG" else -1
        band_pct = BAND_PCTS[band]

        pkey = f"pred_{head}"
        if pkey not in d.files:
            log(f"  [{i+1}/{len(cands)}] SKIP {head} {side} {band}: no pred")
            continue
        score = d[pkey].astype(np.float32)
        if head.startswith("p_up"):
            score = score - 0.5

        log(f"  [{i+1}/{len(cands)}] {head} {side} {band}: testing... ", )
        t1 = time.time()
        res = perm_test(score, fifo_net, fifo_mask, side_sign, band_pct)
        et = time.time() - t1

        record = {
            "head": head, "side": side, "band": band,
            "n_fills": c["fifo_tp4sl3_n_fills"],
            "observed_mean_t": res["observed_mean"],
            "passive_net_after_comm": res["observed_mean"] - 0.376 if not np.isnan(res["observed_mean"]) else float("nan"),
            "null_mean": res["null_mean"],
            "null_p95": res["null_p95"],
            "null_p99": res["null_p99"],
            "p_value": res["p_value"],
            "n_perm": N_PERM,
            "elapsed_sec": et,
        }
        log(f"      obs={res['observed_mean']:.3f} null_mean={res['null_mean']:.3f} "
            f"null_p95={res['null_p95']:.3f} p={res['p_value']:.3f} ({et:.1f}s)")
        results.append(record)

    # Save
    csv_path = OUT_DIR / "permutation_results.csv"
    with open(csv_path, "w") as f:
        cols = list(results[0].keys()) if results else []
        f.write(",".join(cols) + "\n")
        for r in results:
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    json_path = OUT_DIR / "permutation_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    sm_path = OUT_DIR / "permutation_summary.md"
    survivors = [r for r in results if r["p_value"] < 0.05 and r["observed_mean_t"] > 0.376]
    with open(sm_path, "w") as f:
        f.write("# v3.2 Per-Head Permutation Test — Results\n\n")
        f.write(f"_{datetime.utcnow().isoformat(timespec='seconds')}Z_\n\n")
        f.write(f"Tested top {len(results)} candidates from `per_head_master.json` "
                f"with {N_PERM} sign-shuffles each.\n\n")
        f.write("## SURVIVORS (p < 0.05 AND observed_mean > 0.376 commission)\n\n")
        if not survivors:
            f.write("**ZERO survivors.** No candidate's observed mean is statistically significant "
                    "vs sign-shuffled null. Confirms pass 8 conclusion: edge is from selection "
                    "structure, not from the model's signal.\n\n")
            f.write("→ Per HC #345 final: do NOT deploy v3.2 live. Wait for v3.3.\n\n")
        else:
            f.write(f"**{len(survivors)} survivors.** These ARE statistically distinguishable from random:\n\n")
            f.write("| Head | Side | Band | n_fills | obs_t | passive_net | null_mean | null_p95 | p |\n")
            f.write("|---|---|---|---|---|---|---|---|---|\n")
            for r in sorted(survivors, key=lambda x: x["p_value"]):
                f.write(f"| {r['head']} | {r['side']} | {r['band']} | {r['n_fills']} | "
                        f"{r['observed_mean_t']:.3f} | {r['passive_net_after_comm']:.3f} | "
                        f"{r['null_mean']:.3f} | {r['null_p95']:.3f} | {r['p_value']:.3f} |\n")
        f.write("\n## ALL TESTED (sorted by p-value)\n\n")
        f.write("| Head | Side | Band | n | obs | null_mean | null_p95 | p | verdict |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in sorted(results, key=lambda x: x["p_value"]):
            verdict = "SIGNIF" if (r["p_value"] < 0.05 and r["observed_mean_t"] > 0.376) else "no"
            f.write(f"| {r['head']} | {r['side']} | {r['band']} | {r['n_fills']} | "
                    f"{r['observed_mean_t']:.3f} | {r['null_mean']:.3f} | "
                    f"{r['null_p95']:.3f} | {r['p_value']:.3f} | {verdict} |\n")
        f.write(f"\n---\nTotal compute: {time.time() - t0:.1f}s for {len(results)}x{N_PERM} permutations\n")

    log(f"Wrote {csv_path}")
    log(f"Wrote {json_path}")
    log(f"Wrote {sm_path}")
    log(f"SURVIVORS: {len(survivors)} / {len(results)}")
    log(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
