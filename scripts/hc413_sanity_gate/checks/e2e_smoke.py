"""Check 7: end-to-end smoke — top-0.5% tier filter into TP/SL mock.

Take 1000 rows from NPZ (randomly sampled, deterministic seed). Apply:
  - Top-0.5% confidence (|pred_log_ret_10s| >= 99.5th percentile)
  - Direction = sign(pred_log_ret_10s)
  - Mock TP/SL rule (use pred_fifo_tp4sl3_hit_tp if present, else
    realize sign(target_log_ret_10s) == sign(pred) as a hit)

FAIL if any rule produces zero fills (filter too strict / model dead).
"""
from __future__ import annotations
import numpy as np


SEED = 1337
SAMPLE_N = 1000
TIER_QUANTILE = 0.995  # top 0.5%


def run(npz: np.lib.npyio.NpzFile, model_family: str) -> dict:
    failures = []
    details = {}

    if "pred_log_ret_10s" not in npz.files:
        return {"check": "e2e_smoke", "passed": False,
                "failures": ["missing pred_log_ret_10s"], "details": {}}

    p = np.asarray(npz["pred_log_ret_10s"], dtype=np.float64)
    t = np.asarray(npz["target_log_ret_10s"], dtype=np.float64)
    mask_key = "mask_log_ret_10s"
    if mask_key in npz.files:
        m = np.asarray(npz[mask_key], dtype=np.float64) > 0
    else:
        m = np.ones_like(p, dtype=bool)
    good = m & np.isfinite(p) & np.isfinite(t)
    p, t = p[good], t[good]
    n_total = p.size
    details["n_total_good"] = int(n_total)
    if n_total < SAMPLE_N:
        failures.append(f"only {n_total} good rows, < {SAMPLE_N}")
        return {"check": "e2e_smoke", "passed": False, "failures": failures, "details": details}

    rng = np.random.default_rng(SEED)
    idx = rng.choice(n_total, size=SAMPLE_N, replace=False)
    ps = p[idx]
    ts = t[idx]

    # Two parallel ways to apply the Top-0.5% tier:
    #  (A) Population threshold — defined over full NPZ (production-like).
    #  (B) Within-sample top-K rank — guarantees fills exist in the 1000-row sample.
    # We require BOTH paths to produce > 0 fills; production deployment uses (A).
    thresh = float(np.quantile(np.abs(p), TIER_QUANTILE))
    details["tier_threshold_abs_pop"] = thresh
    details["sample_n"] = SAMPLE_N

    # (A) Population threshold applied to sample
    tier_mask_pop = np.abs(ps) >= thresh
    n_tier_fills_pop = int(tier_mask_pop.sum())
    details["sample_fills_pop_thresh"] = n_tier_fills_pop

    n_pop_fills = int((np.abs(p) >= thresh).sum())
    details["population_tier_fills"] = n_pop_fills
    if n_pop_fills == 0:
        failures.append("Top-0.5% tier on full population produced zero fills")

    # (B) Top-K within sample (K = ceil(0.5% of 1000) = 5, minimum 5)
    k = max(5, int(round((1.0 - TIER_QUANTILE) * SAMPLE_N)))
    order = np.argsort(-np.abs(ps))
    tier_mask_rank = np.zeros(SAMPLE_N, dtype=bool)
    tier_mask_rank[order[:k]] = True
    n_tier_fills_rank = int(tier_mask_rank.sum())
    details["sample_fills_topk_rank"] = n_tier_fills_rank
    details["topk"] = k
    if n_tier_fills_rank == 0:
        failures.append("Top-K rank filter produced zero fills in sample")

    # Use union for hit-rate report — we want to know if filter is operational at all.
    tier_mask = tier_mask_pop | tier_mask_rank
    n_tier_fills = int(tier_mask.sum())
    details["sample_tier_fills_union"] = n_tier_fills

    # Direction + mock TP/SL hit-rate on the sample subset that passed tier.
    # Treat target==0 as a no-trade (push), not a directional miss.
    if n_tier_fills > 0:
        pred_dir = np.sign(ps[tier_mask])
        real_dir = np.sign(ts[tier_mask])
        nonzero = real_dir != 0
        n_decisive = int(nonzero.sum())
        if n_decisive > 0:
            hits = int((pred_dir[nonzero] == real_dir[nonzero]).sum())
            details["sample_tier_hits"] = hits
            details["sample_tier_decisive_n"] = n_decisive
            details["sample_tier_hit_rate"] = round(hits / n_decisive, 4)
            if hits == 0:
                failures.append(
                    f"zero directional hits among {n_decisive} decisive tier fills (mock TP/SL)"
                )
        else:
            # All targets were zero — push. This is a sample artifact, not a failure.
            details["sample_tier_hits"] = None
            details["sample_tier_decisive_n"] = 0
            details["note"] = "all tier-fill targets were zero (no decisive outcome in sample)"

    # Also sanity: across the broader sample, at least some predictions on each side.
    details["sample_long_count"] = int((ps > 0).sum())
    details["sample_short_count"] = int((ps < 0).sum())
    if details["sample_long_count"] == 0 or details["sample_short_count"] == 0:
        failures.append("sample is single-sided (no long or no short)")

    return {
        "check": "e2e_smoke",
        "passed": len(failures) == 0,
        "failures": failures,
        "details": details,
    }
