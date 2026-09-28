"""
HC #402-B — TRIAL 278 REPRODUCIBILITY VERIFIER

Loads a MONDAY_DEPLOYMENT_CANDIDATE.json artifact, re-runs the full canonical
replay using its declared entry/exit/filters/costs, and asserts the resulting
metrics match the embedded `canonical_replay_at_canonical_cost_15day_OOT` block
within tolerance.

This is the Monday-morning pre-open smoke test: if it PASSes, the JSON is the
true source-of-truth for live trading. If it FAILs, something has drifted and
deployment is BLOCKED until reconciled.

Usage:
  python scripts/v3_3_research/verify_trial278_from_json.py
  python scripts/v3_3_research/verify_trial278_from_json.py --json <path>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJ = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(PROJ))

from scripts.v3_3_research.full_market_replay import (
    TradeConfig, full_market_replay,
)
from scripts.v3_3_research.v33_execution_optuna_full_market_replay import (
    apply_post_filters, metrics_from_filtered,
)

DEFAULT_JSON = (
    PROJ / "output" / "hc402_EXTENDED_OOT_20260517_000652"
         / "MONDAY_DEPLOYMENT_CANDIDATE.json"
)
LABELS_DIR = PROJ / "data" / "processed" / "mbo_events_smart_v3_fifo_labels"

# Tolerances for PASS verdict (15-day canonical replay is deterministic, but
# allow small float noise from numpy/cuda summation ordering).
TOL_SHARPE = 0.10
TOL_TK_PER_FILL = 0.05
TOL_DAY_CONC = 0.01
TOL_N_FILLS_PCT = 0.02  # 2% drift in fill count


def _horizon_from_head(head: str) -> str:
    """e.g. 'log_ret_30s' -> '30s'."""
    if head.startswith("log_ret_"):
        return head[len("log_ret_"):]
    return head


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--json", type=Path, default=DEFAULT_JSON)
    args = p.parse_args()

    print(f"[verify] Loading config: {args.json}")
    with open(args.json) as f:
        cfg_doc = json.load(f)

    side = cfg_doc["side"]
    head = cfg_doc["head"]
    entry = cfg_doc["entry"]
    exit_blk = cfg_doc["exit"]
    filt = cfg_doc["filters"]
    costs = cfg_doc["costs"]
    canonical = cfg_doc["canonical_replay_at_canonical_cost_15day_OOT"]

    preds_path = PROJ / canonical["predictions_npz"]
    if not preds_path.exists():
        print(f"[verify] ERROR: predictions NPZ missing: {preds_path}")
        return 2

    cfg = TradeConfig(
        side=side,
        horizon=_horizon_from_head(head),
        confidence_threshold=float(entry["confidence_percentile_threshold"]),
        order_type=entry["order_type"],
        cancel_eval_window=int(exit_blk["cancel_window_evals"]),
        hold_seconds=float(exit_blk["hold_seconds"]),
    )
    print(f"[verify] TradeConfig: side={cfg.side} horizon={cfg.horizon} "
          f"order={cfg.order_type} conf_pctile={cfg.confidence_threshold:.4f} "
          f"hold={cfg.hold_seconds:.3f}s cancel_evals={cfg.cancel_eval_window}")

    ledger = full_market_replay(
        preds_path, LABELS_DIR, cfg,
        spread_ticks_rth=float(entry["spread_ticks_assumption"]),
        rt_commission_ticks=float(costs["commission_ticks_rt"]),
    )

    df_f, _ = apply_post_filters(
        ledger,
        tod_start_hour=int(filt["tod_start_hour_et"]),
        tod_end_hour=int(filt["tod_end_hour_et"]),
        require_min_pred_strength=float(entry["min_pred_strength_abs"]),
    )
    m = metrics_from_filtered(df_f)

    # ----- compare to embedded canonical block -----
    exp_sharpe = canonical["sharpe"]
    exp_tk = canonical["tk_per_fill"]
    exp_conc = canonical["day_conc"]
    exp_fills = canonical["n_fills"]

    d_sharpe = m["sharpe"] - exp_sharpe
    d_tk = m["mean_net"] - exp_tk
    d_conc = m["day_conc"] - exp_conc
    fills_drift_pct = abs(m["n_fills"] - exp_fills) / max(1, exp_fills)

    sharpe_ok = abs(d_sharpe) <= TOL_SHARPE
    tk_ok = abs(d_tk) <= TOL_TK_PER_FILL
    conc_ok = abs(d_conc) <= TOL_DAY_CONC
    fills_ok = fills_drift_pct <= TOL_N_FILLS_PCT
    strict_pass = m["day_conc"] <= 0.20
    relaxed_pass = m["day_conc"] <= 0.70

    print()
    print("=" * 70)
    print("VERIFICATION RESULTS (expected vs replay)")
    print("=" * 70)
    print(f"  Sharpe:     {exp_sharpe:>8.2f}  →  {m['sharpe']:>8.2f}  "
          f"(Δ {d_sharpe:+.3f}, tol {TOL_SHARPE})  "
          f"{'OK' if sharpe_ok else 'FAIL'}")
    print(f"  tk/fill:    {exp_tk:>8.3f}  →  {m['mean_net']:>8.3f}  "
          f"(Δ {d_tk:+.4f}, tol {TOL_TK_PER_FILL})  "
          f"{'OK' if tk_ok else 'FAIL'}")
    print(f"  day_conc:   {exp_conc:>8.3f}  →  {m['day_conc']:>8.3f}  "
          f"(Δ {d_conc:+.4f}, tol {TOL_DAY_CONC})  "
          f"{'OK' if conc_ok else 'FAIL'}")
    print(f"  n_fills:    {exp_fills:>8d}  →  {m['n_fills']:>8d}  "
          f"(drift {fills_drift_pct*100:.2f}%, tol {TOL_N_FILLS_PCT*100:.1f}%)  "
          f"{'OK' if fills_ok else 'FAIL'}")
    print(f"  HC #344 strict (day_conc<=0.20): "
          f"{'PASS' if strict_pass else 'FAIL'}")
    print(f"  HC #344 relaxed (day_conc<=0.70): "
          f"{'PASS' if relaxed_pass else 'FAIL'}")
    print("=" * 70)

    overall = sharpe_ok and tk_ok and conc_ok and fills_ok and strict_pass
    print(f"  OVERALL: {'PASS — config artifact is reproducible' if overall else 'FAIL — JSON has drifted from source replay'}")
    print("=" * 70)
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
