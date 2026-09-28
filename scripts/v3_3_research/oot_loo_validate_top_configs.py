"""
oot_loo_validate_top_configs.py — HC #426 R4 / R6 validation gate.

Take top-K v3.3 Optuna sweep configs and run a per-OOT-day breakdown via
full_market_replay + apply_post_filters (same plumbing as the sweep).

ROBUSTNESS FILTER (all must hold):
  - ≥4/5 OOT days profitable (mean_net > 0 after post-filter)
  - worst-day Sharpe > -0.5
  - ≥4/5 OOT days have ≥1 fill
  - n_fills_total > 30 (HC #344)

OUTPUT:
  output/v33_execution_optuna_20260518/loo_validation_results.json
  output/v33_execution_optuna_20260518/loo_robust_configs.json

MALWARE-GUARD (HC #420): user-owned trading research. Pure analysis.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(PROJ))

from scripts.v3_3_research.full_market_replay import (  # noqa: E402
    full_market_replay,
    TradeConfig,
)
from scripts.v3_3_research.v33_execution_optuna_full_market_replay import (  # noqa: E402
    apply_post_filters,
    metrics_from_filtered,
)


DEFAULT_PREDS = PROJ / "output" / "cnn_mamba_v3_3_uncertainty_weighted" / "fold_00_predictions.npz"
DEFAULT_LABELS = PROJ / "data" / "processed" / "mbo_events_smart_v3_fifo_labels"
DEFAULT_SWEEP_DIR = PROJ / "output" / "v33_execution_optuna_20260518"


def evaluate_config_on_day(params: dict, day: str, preds_path: Path, labels_dir: Path) -> dict:
    """Run replay on a single OOT day and apply post-filters. Returns metrics dict."""
    cfg = TradeConfig(
        side=params["side"],
        horizon=params["head_horizon"],
        confidence_threshold=float(params["conf_thr"]),
        order_type=params["order_type"],
        cancel_eval_window=int(params["cancel_window"]),
        hold_seconds=float(params["hold_seconds"]),
    )
    try:
        ledger = full_market_replay(
            preds_path,
            labels_dir,
            cfg,
            dates=[day],
            spread_ticks_rth=float(params["spread_ticks"]),
            rt_commission_ticks=float(params["commission_ticks"]),
        )
    except Exception as e:
        return {"day": day, "error": str(e)[:200], "n_fills": 0,
                "mean_net": 0.0, "sharpe": 0.0, "pf": 0.0, "wr": 0.0}

    if ledger is None or ledger.n_filled == 0:
        return {"day": day, "n_fills": 0, "n_fills_raw": ledger.n_filled if ledger else 0,
                "mean_net": 0.0, "sharpe": 0.0, "pf": 0.0, "wr": 0.0}

    # Post-filters as in sweep
    try:
        df_f, _ = apply_post_filters(
            ledger,
            tod_start_hour=int(params["tod_start_hour"]),
            tod_end_hour=int(params["tod_end_hour"]),
            require_min_pred_strength=float(params["pred_strength_min"]),
        )
    except Exception as e:
        return {"day": day, "error": f"post_filter:{e}"[:200], "n_fills": 0,
                "mean_net": 0.0, "sharpe": 0.0, "pf": 0.0, "wr": 0.0}

    m = metrics_from_filtered(df_f)
    return {
        "day": day,
        "n_fills": int(m.get("n_fills", 0)),
        "n_fills_raw": int(ledger.n_filled),
        "mean_net": float(m.get("mean_net", 0.0)),
        "sharpe": float(m.get("sharpe", 0.0)),
        "sortino": float(m.get("sortino", 0.0)),
        "pf": float(m.get("pf", 0.0)),
        "wr": float(m.get("wr", 0.0)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default=str(DEFAULT_SWEEP_DIR))
    ap.add_argument("--preds", default=str(DEFAULT_PREDS))
    ap.add_argument("--labels-dir", default=str(DEFAULT_LABELS))
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--min-profitable-days", type=int, default=4)
    ap.add_argument("--min-fills-total", type=int, default=30)
    ap.add_argument("--max-worst-day-sharpe-floor", type=float, default=-0.5)
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    best_path = sweep_dir / "best_configs.json"
    if not best_path.exists():
        print(f"[ERROR] best_configs.json not found at {best_path}", file=sys.stderr)
        return 1

    best = json.loads(best_path.read_text())
    if not isinstance(best, list):
        best = [best]
    print(f"[loo] loaded {len(best)} configs from {best_path}")

    d = np.load(args.preds, allow_pickle=True)
    oot_dates = [str(x) for x in d["oot_dates"]]
    print(f"[loo] OOT dates: {oot_dates}")

    top = best[: args.top_k]
    print(f"[loo] validating top-{len(top)} configs × {len(oot_dates)} days "
          f"= {len(top) * len(oot_dates)} replay runs")

    results = []
    robust = []
    for i, entry in enumerate(top):
        params = entry["params"]
        sweep_metrics = entry.get("metrics", {})

        day_metrics = []
        for day in oot_dates:
            m = evaluate_config_on_day(params, day,
                                       Path(args.preds), Path(args.labels_dir))
            day_metrics.append(m)

        valid = [m for m in day_metrics if "error" not in m]
        n_profitable = sum(1 for m in valid if m["mean_net"] > 0)
        n_fills_total = sum(m.get("n_fills", 0) for m in valid)
        n_days_with_fills = sum(1 for m in valid if m.get("n_fills", 0) > 0)
        sharpes = [m.get("sharpe", 0.0) for m in valid if m.get("n_fills", 0) > 0]
        worst_sharpe = min(sharpes) if sharpes else -999.0
        mean_sharpe = float(np.mean(sharpes)) if sharpes else 0.0

        robust_flag = (
            n_profitable >= args.min_profitable_days
            and n_fills_total >= args.min_fills_total
            and worst_sharpe > args.max_worst_day_sharpe_floor
            and n_days_with_fills >= args.min_profitable_days
        )

        rec = {
            "rank": i,
            "trial": entry.get("trial"),
            "sweep_sharpe": sweep_metrics.get("sharpe"),
            "sweep_pf": sweep_metrics.get("pf"),
            "sweep_n_fills": sweep_metrics.get("n_fills"),
            "sweep_wr": sweep_metrics.get("wr"),
            "params": params,
            "per_day": day_metrics,
            "n_profitable_days": n_profitable,
            "n_days_with_fills": n_days_with_fills,
            "n_fills_total": n_fills_total,
            "worst_day_sharpe": worst_sharpe,
            "mean_day_sharpe": mean_sharpe,
            "loo_robust": robust_flag,
        }
        results.append(rec)
        if robust_flag:
            robust.append(rec)

        sym = "✓" if robust_flag else "✗"
        print(f"  [{i:>3}] {sym} trial={entry.get('trial'):<5} "
              f"sweep_sh={sweep_metrics.get('sharpe', 0):>6.2f} "
              f"prof={n_profitable}/5 dwf={n_days_with_fills}/5 fills={n_fills_total:>4} "
              f"worst={worst_sharpe:>6.2f} mean={mean_sharpe:>6.2f} "
              f"{params['head_horizon']}/{params['side']}/{params['order_type']}")

    out_results = sweep_dir / "loo_validation_results.json"
    out_robust = sweep_dir / "loo_robust_configs.json"
    out_results.write_text(json.dumps(results, indent=2, default=str))
    out_robust.write_text(json.dumps(robust, indent=2, default=str))
    print(f"\n[loo] {len(robust)}/{len(results)} configs passed LOO robustness gate")
    print(f"[loo] → {out_results}")
    print(f"[loo] → {out_robust}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
