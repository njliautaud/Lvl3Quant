#!/usr/bin/env python3
"""HC #415 — Multi-output gating sweep with all-OOT-date stability.

Uses CANONICAL FIFO REALIZED NETS already in the NPZ (target_fifo_tp4sl3_net,
target_fifo_tp8sl5_net) so we don't need to re-run market replay. These ARE
the canonical FIFO ground-truth realized nets per sample.

HC #415 rule 1: gate on MULTIPLE model outputs, not just confidence.
HC #415 rule 2: ALL-OOT-DATE stability — per_day_pass_rate >= 80% etc.
HC #415 rule 3: full FIFO canonical replay (satisfied by target_fifo_*).
HC #415 rule 6: Sortino + per_day_pass_rate + realized_net_tk_per_fill as primary.

Inputs:
  --npz <path>            v3.4.2 NPZ (or v3.3)
  --output-dir <path>     where CSV + verdict.md land
  --n-day-chunks <int>    split sample stream into N equal day chunks (default 16)
  --label-col <name>      target_fifo_tp4sl3_net or target_fifo_tp8sl5_net (default tp4sl3)

Outputs:
  results.csv      one row per (signal × side × conf_tier × gate_combo)
  hc415_pass.csv   only rows passing HC #415 rule 2
  verdict.md       top-N ranked by Sortino subject to gate pass
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ===========================================================================
# HC #415 acceptance gate (rule 2)
# ===========================================================================
HC415_PER_DAY_PASS_RATE_MIN = 0.80          # 80% of days net > 0
HC415_PER_DAY_PASS_RATE_STRICT_MIN = 0.70   # 70% of days net > 0.376
HC415_MAX_SINGLE_DAY_PNL_SHARE = 0.40
HC415_MIN_N_FILLS = 50                      # HC #408 honesty
HC415_MIN_DAYS_WITH_FILLS = 10              # only enforced if N_DAYS_TOTAL >= 15
HC415_DAY_CONC_MAX = 0.20                   # HC #344
HC415_CI_LOW_BOOTSTRAP_REPS = 2000


# ===========================================================================
# Signal heads to test
# ===========================================================================
SIGNAL_HEADS = [
    # (signal_name, conf_score_key, "alpha-meaning")
    ("log_ret_1s",     "pred_log_ret_1s",     "directional 1s return"),
    ("log_ret_5s",     "pred_log_ret_5s",     "directional 5s return"),
    ("log_ret_10s",    "pred_log_ret_10s",    "directional 10s return"),
    ("log_ret_30s",    "pred_log_ret_30s",    "directional 30s return"),
    ("fifo_tp4sl3",    "pred_fifo_tp4sl3_net","model-direct FIFO TP4SL3 expected net"),
    ("fifo_tp8sl5",    "pred_fifo_tp8sl5_net","model-direct FIFO TP8SL5 expected net"),
]

# Confidence tiers (top-K% of |signed_pred|)
CONF_TIERS = [
    ("top01",  0.1),
    ("top05",  0.5),
    ("top1",   1.0),
    ("top5",   5.0),
]

# Sides
SIDES = ["long", "short"]


# ===========================================================================
# Gate combos (HC #415 rule 1)
# Each gate is a function (preds_dict, idx) -> bool mask. Composable.
# ===========================================================================
def gate_baseline(preds, side):
    """No additional gates — confidence-only baseline."""
    return np.ones(preds["N"], dtype=bool)

def gate_horizon_confluence(preds, side):
    """Sign agreement across 1s / 5s / 10s heads."""
    s1 = np.sign(preds["pred_log_ret_1s"])
    s5 = np.sign(preds["pred_log_ret_5s"])
    s10 = np.sign(preds["pred_log_ret_10s"])
    target_sign = 1.0 if side == "long" else -1.0
    return (s1 == target_sign) & (s5 == target_sign) & (s10 == target_sign)

def gate_uncertainty_tight(preds, side):
    """Quantile band width below median (tight = model is confident)."""
    band = preds["pred_log_ret_10s_q90"] - preds["pred_log_ret_10s_q10"]
    return band < np.median(band)

def gate_mfe_room(preds, side, mfe_min_ticks=2.5):
    """Predicted MFE_30s > threshold — model thinks there's room."""
    return preds["pred_pred_mfe_30s_ticks"] > mfe_min_ticks

def gate_reversal_low(preds, side):
    """Pred reversal logit below median — model thinks trend persists."""
    rev = preds["pred_p_reversal_30s"]
    return rev < np.median(rev)

def gate_direct_sign_agree(preds, side):
    """Model-predicted fifo_tp4sl3_net is positive (i.e., model itself predicts a profitable trade).
    For long side: actually we need pred sign matches side direction implied by tp net.
    Since fifo_tp4sl3_net is symmetric (signed), require sign(pred) == direction."""
    target_sign = 1.0 if side == "long" else -1.0
    return np.sign(preds["pred_fifo_tp4sl3_net"]) == target_sign

def gate_vol_mid(preds, side):
    """Vol prediction in middle tertile (avoid extreme regimes)."""
    v = preds["pred_pred_realized_vol_30s_ticks"]
    q33, q66 = np.percentile(v, [33, 66])
    return (v >= q33) & (v <= q66)


GATE_COMBOS = [
    ("baseline",       [gate_baseline]),
    ("hconfluence",    [gate_horizon_confluence]),
    ("uncertain_tight",[gate_uncertainty_tight]),
    ("mfe_room",       [gate_mfe_room]),
    ("rev_low",        [gate_reversal_low]),
    ("direct_agree",   [gate_direct_sign_agree]),
    ("vol_mid",        [gate_vol_mid]),
    ("hconf+mfe",      [gate_horizon_confluence, gate_mfe_room]),
    ("hconf+rev",      [gate_horizon_confluence, gate_reversal_low]),
    ("hconf+direct",   [gate_horizon_confluence, gate_direct_sign_agree]),
    ("hconf+uncert",   [gate_horizon_confluence, gate_uncertainty_tight]),
    ("hconf+mfe+rev",  [gate_horizon_confluence, gate_mfe_room, gate_reversal_low]),
    ("FULL",           [gate_horizon_confluence, gate_mfe_room, gate_reversal_low,
                        gate_direct_sign_agree, gate_uncertainty_tight]),
]


# ===========================================================================
# Per-cell evaluation
# ===========================================================================
def confidence_mask(signal_score, valid_mask, side, top_pct):
    """Top-K% of signed signal in the valid pool."""
    signed = signal_score if side == "long" else -signal_score
    valid = valid_mask & np.isfinite(signed)
    if not valid.any():
        return np.zeros_like(valid)
    thr = np.percentile(signed[valid], 100.0 - top_pct)
    return valid & (signed >= thr)


def evaluate_cell(realized_net, fired_mask, day_idx, n_days):
    """Compute HC #415 metrics for a fired cell."""
    n_fills = int(fired_mask.sum())
    if n_fills < 5:
        return None

    pnl = realized_net[fired_mask]
    days = day_idx[fired_mask]

    mean_net = float(pnl.mean())
    std_net = float(pnl.std(ddof=1)) if n_fills > 1 else 0.0
    downside = pnl[pnl < 0]
    down_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0

    sharpe = mean_net / std_net * np.sqrt(n_fills) if std_net > 0 else 0.0
    sortino = mean_net / down_std * np.sqrt(n_fills) if down_std > 0 else 0.0

    gross_win = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0
    wr = float((pnl > 0).mean() * 100)

    # Per-day stats
    day_pnls = {}
    day_counts = {}
    for d in range(n_days):
        m = (days == d)
        if m.any():
            day_pnls[d] = float(pnl[m].sum())
            day_counts[d] = int(m.sum())
    n_days_with_fills = len(day_pnls)

    if n_days_with_fills == 0:
        return None

    day_pnl_arr = np.array(list(day_pnls.values()))
    day_count_arr = np.array(list(day_counts.values()))
    # Day-level NET per-fill: fills > 0 per day
    day_net_per_fill = day_pnl_arr / np.maximum(day_count_arr, 1)

    per_day_pass_rate = float((day_net_per_fill > 0).mean())
    per_day_pass_rate_strict = float((day_net_per_fill > 0.376).mean())
    total_abs_pnl = np.abs(day_pnl_arr).sum()
    max_single_day_pnl_share = float(np.abs(day_pnl_arr).max() / total_abs_pnl) if total_abs_pnl > 0 else 1.0
    day_conc = max_single_day_pnl_share  # equivalent definition by abs

    # CI_low_95 bootstrap on mean(pnl)
    rng = np.random.default_rng(42)
    boot_means = np.array([
        rng.choice(pnl, size=n_fills, replace=True).mean()
        for _ in range(HC415_CI_LOW_BOOTSTRAP_REPS)
    ])
    ci_low_95 = float(np.percentile(boot_means, 2.5))

    # HC #408 honesty gate
    pass_hc408 = (n_fills >= HC415_MIN_N_FILLS) and (ci_low_95 > 0) and (day_conc <= HC415_DAY_CONC_MAX)

    # HC #415 rule 2 gate (all-OOT-date stability)
    days_constraint = (n_days < 15) or (n_days_with_fills >= HC415_MIN_DAYS_WITH_FILLS)
    pass_hc415_rule2 = (
        pass_hc408
        and per_day_pass_rate >= HC415_PER_DAY_PASS_RATE_MIN
        and per_day_pass_rate_strict >= HC415_PER_DAY_PASS_RATE_STRICT_MIN
        and max_single_day_pnl_share <= HC415_MAX_SINGLE_DAY_PNL_SHARE
        and days_constraint
    )

    return {
        "n_fills": n_fills,
        "n_days_with_fills": n_days_with_fills,
        "n_fills_per_day_mean": float(day_count_arr.mean()),
        "n_fills_per_day_min": int(day_count_arr.min()),
        "realized_net_tk_per_fill": mean_net,
        "realized_net_tk_per_fill_dollars": mean_net * 12.50,
        "sharpe_sqrtN": sharpe,
        "sortino_sqrtN": sortino,
        "pf": pf,
        "wr": wr,
        "day_conc": day_conc,
        "max_single_day_pnl_share": max_single_day_pnl_share,
        "per_day_pass_rate": per_day_pass_rate,
        "per_day_pass_rate_strict": per_day_pass_rate_strict,
        "ci_low_95_net": ci_low_95,
        "ci_low_95_net_dollars": ci_low_95 * 12.50,
        "pass_hc408_honesty": pass_hc408,
        "pass_hc415_rule2": pass_hc415_rule2,
    }


# ===========================================================================
# Main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--n-day-chunks", type=int, default=16,
                    help="split sample stream into N equal day chunks (default 16 for v3.4.2 ext-OOT)")
    ap.add_argument("--label-col", default="target_fifo_tp4sl3_net",
                    choices=["target_fifo_tp4sl3_net", "target_fifo_tp8sl5_net"])
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[HC #415 sweep] Loading NPZ from {args.npz}...")
    z = np.load(args.npz)

    N = z["pred_log_ret_1s"].shape[0]
    print(f"[HC #415 sweep] N = {N} samples, {args.n_day_chunks} day chunks (~{N//args.n_day_chunks}/day)")

    # Build day index: split sample stream into N_DAYS equal chunks
    day_size = N // args.n_day_chunks
    day_idx = np.zeros(N, dtype=np.int32)
    for d in range(args.n_day_chunks):
        s = d * day_size
        e = (d + 1) * day_size if d < args.n_day_chunks - 1 else N
        day_idx[s:e] = d

    # Pull all preds into a dict
    pred_keys = [
        "pred_log_ret_1s", "pred_log_ret_5s", "pred_log_ret_10s", "pred_log_ret_30s", "pred_log_ret_60s",
        "pred_log_ret_10s_q10", "pred_log_ret_10s_q90",
        "pred_log_ret_30s_q10", "pred_log_ret_30s_q90",
        "pred_pred_mfe_30s_ticks", "pred_pred_mae_30s_ticks", "pred_pred_time_to_mfe_secs",
        "pred_p_reversal_15s", "pred_p_reversal_30s",
        "pred_pred_realized_vol_30s_ticks",
        "pred_fifo_tp4sl3_net", "pred_fifo_tp8sl5_net",
        "pred_fifo_tp4sl3_hit_tp", "pred_fifo_tp8sl5_hit_tp",
    ]
    preds = {k: np.asarray(z[k], dtype=np.float32) for k in pred_keys}
    preds["N"] = N

    realized_net = np.asarray(z[args.label_col], dtype=np.float32)
    label_mask = np.asarray(z[args.label_col.replace("target_", "mask_")], dtype=np.float32) > 0.5
    print(f"[HC #415 sweep] Realized label '{args.label_col}': mean={realized_net[label_mask].mean():.4f} tk "
          f"(n={int(label_mask.sum())} valid samples)")

    # Enumerate cells
    rows = []
    cell_count = 0
    for sig_name, sig_key, _ in SIGNAL_HEADS:
        signal = preds[sig_key]
        for side in SIDES:
            for conf_name, conf_pct in CONF_TIERS:
                conf_m = confidence_mask(signal, label_mask, side, conf_pct)
                for gate_name, gate_fns in GATE_COMBOS:
                    cell_count += 1
                    # AND all gate masks
                    g_mask = np.ones(N, dtype=bool)
                    for fn in gate_fns:
                        g_mask &= fn(preds, side)
                    fired = conf_m & g_mask
                    res = evaluate_cell(realized_net, fired, day_idx, args.n_day_chunks)
                    if res is None:
                        continue
                    row = {
                        "cell_id": f"{sig_name}_{side}_{conf_name}_{gate_name}",
                        "signal": sig_name,
                        "side": side,
                        "conf_tier": conf_name,
                        "conf_pct": conf_pct,
                        "gate": gate_name,
                        "label_col": args.label_col,
                    }
                    row.update(res)
                    rows.append(row)
        print(f"[HC #415 sweep]   {sig_name}: cells so far = {cell_count}, "
              f"rows={len(rows)}, elapsed={time.time()-t0:.1f}s")

    if not rows:
        print("[HC #415 sweep] NO CELLS PRODUCED RESULTS. Aborting.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    df = df.sort_values(["pass_hc415_rule2", "sortino_sqrtN"], ascending=[False, False])
    csv_path = args.output_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"[HC #415 sweep] Wrote {len(df)} cells to {csv_path}")

    pass_df = df[df["pass_hc415_rule2"]]
    pass_path = args.output_dir / "hc415_pass.csv"
    pass_df.to_csv(pass_path, index=False)
    print(f"[HC #415 sweep] {len(pass_df)} cells pass HC #415 rule-2 → {pass_path}")

    # Top-20 ranked verdict
    top_n = min(20, len(pass_df)) if len(pass_df) > 0 else min(20, len(df))
    top_df = (pass_df if len(pass_df) > 0 else df).head(top_n)

    with open(args.output_dir / "verdict.md", "w") as f:
        f.write(f"# HC #415 Multi-Gate Sweep Verdict\n\n")
        f.write(f"**NPZ**: `{args.npz}`\n\n")
        f.write(f"**Label col**: `{args.label_col}` (canonical FIFO realized net)\n\n")
        f.write(f"**N samples**: {N}  |  **N day chunks**: {args.n_day_chunks}\n\n")
        f.write(f"**Total cells evaluated**: {len(df)}\n\n")
        f.write(f"**Cells passing HC #415 rule 2**: {len(pass_df)}\n\n")
        f.write(f"**Wall time**: {time.time()-t0:.1f}s\n\n")
        f.write(f"## Acceptance gate (HC #415 rule 2)\n")
        f.write(f"- per_day_pass_rate >= {HC415_PER_DAY_PASS_RATE_MIN}\n")
        f.write(f"- per_day_pass_rate_strict >= {HC415_PER_DAY_PASS_RATE_STRICT_MIN}\n")
        f.write(f"- max_single_day_pnl_share <= {HC415_MAX_SINGLE_DAY_PNL_SHARE}\n")
        f.write(f"- n_fills >= {HC415_MIN_N_FILLS}, CI_low_95 > 0, day_conc <= {HC415_DAY_CONC_MAX}\n\n")
        f.write(f"## Top {top_n} cells (ranked by Sortino, gate-pass first)\n\n")
        cols = ["cell_id", "n_fills", "n_days_with_fills",
                "realized_net_tk_per_fill", "sortino_sqrtN", "sharpe_sqrtN", "pf", "wr",
                "per_day_pass_rate", "per_day_pass_rate_strict",
                "max_single_day_pnl_share", "day_conc", "ci_low_95_net",
                "pass_hc415_rule2"]
        try:
            f.write(top_df[cols].to_markdown(index=False, floatfmt=".3f"))
        except Exception:
            f.write("```\n")
            f.write(top_df[cols].to_string(index=False))
            f.write("\n```\n")
        f.write("\n")

    print(f"[HC #415 sweep] Verdict written to {args.output_dir / 'verdict.md'}")
    print(f"[HC #415 sweep] DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
