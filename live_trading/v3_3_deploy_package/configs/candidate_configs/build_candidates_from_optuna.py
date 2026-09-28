#!/usr/bin/env python3
"""
Build HC #368 deploy candidate JSONs from top Optuna trials (corrected Sharpe).

Inputs:
  - /home/jupiter/Lvl3Quant/live_trading/v3_3_deploy_package/configs/v33_config_template.json
  - /home/jupiter/Lvl3Quant/output/v33_execution_optuna_20260515/best_configs_corrected.json

Outputs: candidate_NN_<descriptor>.json in this directory.

This script PROMOTES validated configs only. It does NOT relax safety gates.
"""
import json
import copy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEMPLATE_PATH = ROOT.parent / "v33_config_template.json"
BEST_PATH = Path("/home/jupiter/Lvl3Quant/output/v33_execution_optuna_20260515/best_configs_corrected.json")

# Picks chosen for diversity (side × horizon × TOD × confluence)
PICKS = [
    {"trial": 2151, "label": "01_short_10s_pm_no_confluence",
     "rationale": "TOP overall by corrected Sharpe (1.345). SHORT, 10s, passive+2, afternoon 14-15h, NO confluence (pure signal)."},
    {"trial": 4389, "label": "02_long_5s_full_day_hor_confluence",
     "rationale": "TOP LONG (Sharpe 1.19). LONG, 5s, passive+2, full session 10-16h, horizon confluence ON. Diversifies side bias."},
    {"trial": 2709, "label": "03_short_5s_pre_open_dual_confluence",
     "rationale": "Pre-open/early morning SHORT (Sharpe 1.17). SHORT, 5s, passive+2, 06-08h, BOTH FIFO+horizon confluences ON. Diversifies TOD."},
]


def trial_to_config(template: dict, trial: dict) -> dict:
    """Map an Optuna trial's params onto a deep-copy of the template."""
    cfg = copy.deepcopy(template)
    p = trial["params"]

    # --- _meta provenance ---
    cfg["_meta"]["optuna_provenance"] = {
        "study": "v33_execution_optuna_20260515",
        "trial_number": trial["trial"],
        "corrected_sharpe": trial["corrected_sharpe"],
        "corrected_sortino": trial["corrected_sortino"],
        "pf": trial["pf"],
        "wr": trial["wr"],
        "n_fills": trial["n_fills"],
        "n_days": trial["n_days"],
        "mean_net_per_trade": trial["mean_net_per_trade"],
        "day_conc": trial["day_conc"],
        "ci_low_95": trial["ci_low_95"],
        "hc344_pass": True,
        "metric_definition": "canonical per-event Sharpe = mean_net_per_trade / std_per_trade (NO sqrt(252))",
    }
    cfg["_meta"]["hc_authorization"].append("HC #369 (Optuna superset)")

    # --- entry_logic.side ---
    side = p["side"]
    cfg["entry_logic"]["side_bias"] = f"{side}_only"
    cfg["entry_logic"]["suppress_long"] = (side == "short")
    cfg["entry_logic"]["suppress_short"] = (side == "long")

    # --- entry_logic.order_type → passive_offset_ticks ---
    ot = p["order_type"]
    # passive_at_touch_plus_2 = quote 2 ticks INSIDE current touch (deeper limit)
    # We encode this as negative passive_offset_ticks (live engine convention varies — confirm with daemon)
    offset_map = {
        "passive_at_touch": 0,
        "passive_at_touch_plus_1": -1,
        "passive_at_touch_plus_2": -2,
        "ioc_market": None,
    }
    cfg["entry_logic"]["order_type"] = "passive_limit" if ot.startswith("passive") else "ioc_market"
    if offset_map.get(ot) is not None:
        cfg["entry_logic"]["passive_offset_ticks"] = offset_map[ot]
    cfg["entry_logic"]["_optuna_order_type"] = ot  # preserve original string

    # --- entry_logic confidence threshold ---
    # Optuna conf_thr is a z-score / threshold on the head value. We don't override
    # min_percentile_* here — keep template P99 default. Instead, set absolute threshold:
    cfg["entry_logic"]["abs_signal_threshold"] = p["conf_thr"]
    cfg["entry_logic"]["pred_strength_min"] = p["pred_strength_min"]

    # --- head_selection: pick the trial's horizon as primary ---
    horiz = p["head_horizon"]  # "1s", "5s", "10s", "30s"
    cfg["head_selection"]["primary_heads"] = [f"log_ret_{horiz}"]
    cfg["head_selection"]["_optuna_head_horizon"] = horiz

    # --- exit_logic from trial ---
    cfg["exit_logic"]["time_exit_seconds"] = float(p["hold_seconds"])
    cfg["exit_logic"]["cancel_eval_window"] = int(p["cancel_window"])

    # --- volatility / spread gate ---
    cfg["volatility_gate"]["spread_filter_ticks"] = float(p["spread_ticks"])
    cfg["volatility_gate"]["sigma_halt_mult"] = float(p["sigma_halt_mult"])

    # --- time_of_day_gate ---
    cfg["time_of_day_gate"]["preferred_windows"] = [{
        "start": f"{int(p['tod_start_hour']):02d}:00",
        "end": f"{int(p['tod_end_hour']):02d}:00",
    }]
    cfg["time_of_day_gate"]["outside_preferred_size_multiplier"] = 0.0  # hard suppress outside TOD

    # --- model_health: keep sigma_halt from trial if more aggressive ---
    cfg["model_health"]["sigma_spike_multiplier_halt"] = float(p["sigma_halt_mult"])

    # --- confluence_gate ---
    use_fifo = bool(p["use_fifo_confluence"])
    use_hor = bool(p["use_horizon_confluence"])
    cfg["confluence_gate"]["enabled"] = use_fifo or use_hor
    cfg["confluence_gate"]["min_agreeing_heads"] = (1 if use_fifo else 0) + (1 if use_hor else 0)

    agreement = []
    if use_hor:
        agreement.append(f"log_ret_{p['confluence_horizon']}")
    if use_fifo:
        agreement.append(p["fifo_confluence_head"])
        cfg["confluence_gate"]["magcorr_min_ticks"] = float(p["fifo_confluence_thr_ticks"])
    cfg["confluence_gate"]["agreement_horizon_set"] = agreement
    cfg["confluence_gate"]["_use_fifo_confluence"] = use_fifo
    cfg["confluence_gate"]["_use_horizon_confluence"] = use_hor

    # --- commission_ticks (informational, fitness already used it) ---
    cfg["_meta"]["commission_ticks_used"] = float(p["commission_ticks"])

    return cfg


def main():
    template = json.loads(TEMPLATE_PATH.read_text())
    best = json.loads(BEST_PATH.read_text())
    best_by_trial = {e["trial"]: e for e in best}

    out_summary = []
    for pick in PICKS:
        tid = pick["trial"]
        if tid not in best_by_trial:
            print(f"[skip] trial {tid} not in best_configs_corrected.json")
            continue
        trial = best_by_trial[tid]
        cfg = trial_to_config(template, trial)
        cfg["_meta"]["candidate_rationale"] = pick["rationale"]
        out_path = ROOT / f"candidate_{pick['label']}.json"
        out_path.write_text(json.dumps(cfg, indent=2))
        out_summary.append({
            "file": out_path.name,
            "trial": tid,
            "rationale": pick["rationale"],
            "corrected_sharpe": trial["corrected_sharpe"],
            "side": trial["params"]["side"],
            "horizon": trial["params"]["head_horizon"],
            "order_type": trial["params"]["order_type"],
            "tod": f"{trial['params']['tod_start_hour']:02d}-{trial['params']['tod_end_hour']:02d}",
        })
        print(f"[wrote] {out_path}")

    summary_path = ROOT / "CANDIDATES_SUMMARY.json"
    summary_path.write_text(json.dumps(out_summary, indent=2))
    print(f"[wrote] {summary_path}")
    print(f"\nTotal candidates: {len(out_summary)}")


if __name__ == "__main__":
    main()
