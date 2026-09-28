"""
HC #403 FOLLOWUP — 1s-LONG candidate canonical-replay cross-check.

Agent B (hc403_b_signal_flip) reported a strict-pass passer DIFFERENT from
trial 278:
  head=log_ret_1s, side=long, entry_pctile=0.10 (top 10%),
  exit=fixed 2s hold, order=passive_at_touch_plus_2, ToD 13-15 ET,
  FIFO confluence pred_fifo_tp4sl3_net > 0.80
with Sharpe 23.6 in their custom harness.

Agent B's harness has a known ~15% Sharpe inflation vs canonical
(their trial-278 reproduction gave 15.2 vs 13.48 canonical).

THIS SCRIPT runs the canonical replay path (`verify_trial278_from_json.py`
style: `full_market_replay` + `apply_post_filters` + `metrics_from_filtered`).

CONTROLS:
  A) trial 278 (must reproduce Sharpe 13.48 ± 0.10, n_fills 195, day_conc 0.186)
  B) 1s-LONG canonical (same filtering pipeline as trial 278: ToD + pred_strength,
     NO explicit FIFO confluence — matches what trial 278's canonical Sharpe
     actually measures, see note below)
  C) 1s-LONG canonical + FIFO confluence (apples-to-apples with agent B,
     adds the FIFO confluence externally on the per_trade_df)

IMPORTANT FINDING (audit of verify_trial278_from_json.py + hc402_extended_oot_reval.py):
  The canonical "13.48" number for trial 278 does NOT actually apply the FIFO
  confluence filter declared in the deployment JSON. `apply_post_filters`
  applies only ToD + pred_strength. The FIFO confluence clause in the JSON is
  documentation of intended live-deploy behavior, not part of the measured
  metric. So config (B) above is the like-for-like comparison to trial 278.

NOT MALWARE. Pure analysis. Reads existing NPZ/labels. Writes only to
output/hc403_followup_1s_long_<timestamp>/.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(PROJ))

from scripts.v3_3_research.full_market_replay import (  # noqa: E402
    TradeConfig, full_market_replay,
)
from scripts.v3_3_research.v33_execution_optuna_full_market_replay import (  # noqa: E402
    apply_post_filters, metrics_from_filtered,
)

PREDS_NPZ = PROJ / "output" / "v3_3_extended_oot_20260514" / "extended_oot_predictions.npz"
LABELS_DIR = PROJ / "data" / "processed" / "mbo_events_smart_v3_fifo_labels"
OUT_DIR = PROJ / "output" / "hc403_followup_1s_long_20260517_003237"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _ci_low_95(net: np.ndarray) -> float:
    if net.size == 0:
        return float("nan")
    m = float(net.mean())
    if net.size < 2:
        return m
    sd = float(net.std(ddof=1))
    return m - 1.96 * sd / np.sqrt(net.size)


def _hc344_flags(m: dict) -> dict:
    return {
        "hc344_strict_day_conc_le_0_20": bool(m["day_conc"] <= 0.20),
        "hc344_relaxed_day_conc_le_0_70": bool(m["day_conc"] <= 0.70),
        "n_fills_ge_30": bool(m["n_fills"] >= 30),
        "pf_ge_1_2": bool(m["pf"] >= 1.2),
        "sharpe_ge_0_5": bool(m["sharpe"] >= 0.5),
    }


def _run_canonical(label: str, cfg: TradeConfig, *,
                   tod_start: int, tod_end: int,
                   min_pred_strength: float,
                   spread_assumption: float,
                   commission_ticks: float,
                   apply_fifo_confluence: bool = False,
                   fifo_confluence_thr: float = 0.80,
                   ) -> dict:
    """Canonical replay → ToD/pred-strength filter → metrics.
    Optionally apply external FIFO confluence filter on per_trade_df.
    """
    t0 = time.time()
    print(f"[{label}] TradeConfig: side={cfg.side} horizon={cfg.horizon} "
          f"order={cfg.order_type} conf_pctile={cfg.confidence_threshold:.4f} "
          f"hold={cfg.hold_seconds:.3f}s cancel={cfg.cancel_eval_window}")
    ledger = full_market_replay(
        PREDS_NPZ, LABELS_DIR, cfg,
        spread_ticks_rth=spread_assumption,
        rt_commission_ticks=commission_ticks,
    )
    df_f, _ = apply_post_filters(
        ledger,
        tod_start_hour=tod_start,
        tod_end_hour=tod_end,
        require_min_pred_strength=min_pred_strength,
    )
    fifo_n_pre = int(len(df_f))
    fifo_used = False
    if apply_fifo_confluence and not df_f.empty:
        # Pull per-sample FIFO pred and join on timestamp
        z = np.load(PREDS_NPZ, allow_pickle=True)
        n = int(z["n_samples"])
        pred_fifo = z["pred_fifo_tp4sl3_net"][:n].astype(np.float64)
        mask_fifo = z["mask_fifo_tp4sl3_net"][:n].astype(bool)
        # Each per_trade row was indexed via sel_idx into the original n samples
        # but per_trade_df doesn't carry the original idx. We can re-derive by
        # matching timestamps from the FIFO labels (which apply_post_filters
        # already produced). The simpler path: redo selection here and intersect.
        # Strategy: select rows where pred_fifo signed by side > thr.
        # df_f's "timestamp" column is ts_ns from FIFO labels; we need to map
        # back to the prediction index. Since the FIFO labels and predictions
        # are aligned 1:1 by index (both sliced to n=min(preds,fifo)), we can
        # build a ts_ns -> idx map for the full slab and look up.
        from scripts.v3_3_research.full_market_replay import _load_fifo_labels
        fifo_all = _load_fifo_labels(LABELS_DIR, [str(x) for x in z["oot_dates"]])
        n_eff = min(n, int(fifo_all["ts_ns"].shape[0]))
        ts_all = fifo_all["ts_ns"][:n_eff]
        # Build lookup: ts_ns -> idx (assume unique; if duplicates, take first)
        ts_to_idx = {int(t): i for i, t in enumerate(ts_all)}
        # IMPORTANT: pred_fifo_tp4sl3_net is empirically always negative
        # (range -2.84 to -0.92, mean -1.92). The trial 278 deployment JSON
        # specifies `pred_fifo_tp4sl3_net > 0.80 ticks` — this is intended as
        # ABSOLUTE magnitude (the head is a non-directional "net" prediction).
        # Use abs(pred_fifo) > thr to match agent B's interpretation.
        keep_mask = np.zeros(len(df_f), dtype=bool)
        for row_pos in range(len(df_f)):
            t = int(df_f["timestamp"].iat[row_pos])
            i = ts_to_idx.get(t)
            if i is None:
                continue
            if not mask_fifo[i]:
                continue
            if abs(pred_fifo[i]) > fifo_confluence_thr:
                keep_mask[row_pos] = True
        df_f = df_f.loc[keep_mask].reset_index(drop=True)
        fifo_used = True
    fifo_n_post = int(len(df_f))

    m = metrics_from_filtered(df_f)
    net = df_f["net_ticks"].to_numpy(dtype=float)
    net = net[np.isfinite(net)]
    m["ci_low_95"] = _ci_low_95(net)
    m["n_signals_after_replay"] = int(ledger.n_signals)
    m["n_filled_after_replay"] = int(ledger.n_filled)
    m["n_after_tod_pred_strength"] = fifo_n_pre
    m["n_after_fifo_confluence"] = fifo_n_post if fifo_used else fifo_n_pre
    m["fifo_confluence_applied"] = fifo_used
    m["fifo_confluence_thr"] = fifo_confluence_thr if fifo_used else None
    flags = _hc344_flags(m)
    m.update(flags)
    print(f"[{label}] n_fills={m['n_fills']} sharpe={m['sharpe']:.2f} "
          f"sortino={m['sortino']:.2f} pf={m['pf']:.2f} wr={m['wr']:.1f}% "
          f"mean_net={m['mean_net']:.3f} day_conc={m['day_conc']:.3f} "
          f"ci_low95={m['ci_low_95']:.3f} strict_pass={m['hc344_strict_day_conc_le_0_20']}")
    print(f"[{label}] elapsed={time.time()-t0:.1f}s")
    return m


def main() -> int:
    print("=" * 78)
    print("HC #403 FOLLOWUP — 1s-LONG canonical-replay cross-check")
    print("=" * 78)

    # ---------- (A) Trial 278 control ----------
    cfg_278 = TradeConfig(
        side="short",
        horizon="30s",
        confidence_threshold=0.04354092144615896,
        order_type="passive_at_touch_plus_2",
        cancel_eval_window=79,
        hold_seconds=1.4767640490577054,
    )
    m_278 = _run_canonical(
        "trial_278_control", cfg_278,
        tod_start=13, tod_end=15,
        min_pred_strength=0.06763289381838417,
        spread_assumption=0.7697226043185049,
        commission_ticks=0.376,
    )

    # ---------- (B) 1s-LONG canonical (ToD + pred_strength only) ----------
    # Agent B used entry_pctile=0.10 (top 10% by pred for long).
    # canonical TradeConfig semantics (from full_market_replay.py source):
    #   "confidence_threshold = FRACTION OF THE TAIL we keep"
    #   long, conf=0.10 → top 10%   (pred >= q(0.90))
    # So confidence_threshold = 0.10.
    cfg_1s_long = TradeConfig(
        side="long",
        horizon="1s",
        confidence_threshold=0.10,   # top 10%
        order_type="passive_at_touch_plus_2",
        cancel_eval_window=8,        # ~2s @ 250ms stride
        hold_seconds=2.0,
    )
    # Use 0 min_pred_strength to mirror agent B's plain top-pctile gate.
    # Use spread=1.0 (canonical RTH) — agent B's signal-flip harness doesn't
    # vary spread. Trial 278 uses 0.77 spread, but spread only affects
    # ioc_market entry cost (ours is passive_+2). Either way irrelevant here.
    m_1s_long_canon = _run_canonical(
        "1s_long_canonical_no_fifo", cfg_1s_long,
        tod_start=13, tod_end=15,
        min_pred_strength=0.0,
        spread_assumption=1.0,
        commission_ticks=0.376,
        apply_fifo_confluence=False,
    )

    # ---------- (C) 1s-LONG canonical + FIFO confluence (like agent B) ----------
    m_1s_long_fifo = _run_canonical(
        "1s_long_canonical_with_fifo", cfg_1s_long,
        tod_start=13, tod_end=15,
        min_pred_strength=0.0,
        spread_assumption=1.0,
        commission_ticks=0.376,
        apply_fifo_confluence=True,
        fifo_confluence_thr=0.80,
    )

    # ---------- assemble + write ----------
    results = {
        "produced_at_et": time.strftime("%Y-%m-%d %H:%M:%S ET"),
        "hc_refs": ["HC #403", "HC #402-B", "HC #344", "HC #392"],
        "canonical_harness": "scripts/v3_3_research/full_market_replay.py + "
                             "v33_execution_optuna_full_market_replay.apply_post_filters/metrics_from_filtered",
        "predictions_npz": str(PREDS_NPZ.relative_to(PROJ)),
        "labels_dir": str(LABELS_DIR.relative_to(PROJ)),
        "n_oot_days": 15,
        "commission_ticks_rt": 0.376,
        "control_trial_278": {
            "config": {
                "head": "log_ret_30s", "side": "short",
                "order_type": "passive_at_touch_plus_2",
                "conf_pctile": 0.04354092144615896,
                "hold_seconds": 1.4767640490577054,
                "cancel_evals": 79,
                "min_pred_strength": 0.06763289381838417,
                "spread_assumption": 0.7697226043185049,
                "tod_start_et": 13, "tod_end_et": 15,
                "fifo_confluence_applied_in_replay": False,
            },
            "expected_sharpe": 13.48,
            "expected_n_fills": 195,
            "expected_day_conc": 0.186,
            "metrics": m_278,
            "reproduction_ok": (
                abs(m_278["sharpe"] - 13.48) <= 0.10
                and abs(m_278["day_conc"] - 0.186) <= 0.01
                and abs(m_278["n_fills"] - 195) / 195 <= 0.02
            ),
        },
        "candidate_1s_long_canonical_no_fifo": {
            "config": {
                "head": "log_ret_1s", "side": "long",
                "order_type": "passive_at_touch_plus_2",
                "conf_pctile_top_fraction": 0.10,
                "hold_seconds": 2.0,
                "cancel_evals": 8,
                "min_pred_strength": 0.0,
                "spread_assumption": 1.0,
                "tod_start_et": 13, "tod_end_et": 15,
                "fifo_confluence_applied_in_replay": False,
            },
            "metrics": m_1s_long_canon,
        },
        "candidate_1s_long_canonical_with_fifo": {
            "config": {
                "head": "log_ret_1s", "side": "long",
                "order_type": "passive_at_touch_plus_2",
                "conf_pctile_top_fraction": 0.10,
                "hold_seconds": 2.0,
                "cancel_evals": 8,
                "min_pred_strength": 0.0,
                "spread_assumption": 1.0,
                "tod_start_et": 13, "tod_end_et": 15,
                "fifo_confluence_applied_in_replay": True,
                "fifo_confluence_thr_ticks": 0.80,
                "fifo_confluence_signed_by_side": False,
                "fifo_confluence_uses_abs_magnitude": True,
                "fifo_confluence_note": "pred_fifo_tp4sl3_net is empirically always negative (-2.84 to -0.92); using abs(pred_fifo) > thr to match trial 278 deployment JSON intent.",
            },
            "metrics": m_1s_long_fifo,
        },
        "agent_b_reported": {
            "harness": "hc403_b_signal_flip_exits.py (custom)",
            "sharpe": 23.567684409414934,
            "sortino": 14.546743837213722,
            "pf": 24.35549132947977,
            "wr": 96.22641509433963,
            "mean_net": 1.8296603773584907,
            "day_conc": 0.1784020129521924,
            "n_fills": 106,
            "note": "Agent B's harness inflated trial 278 reproduction Sharpe by "
                    "~15% (15.2 vs canonical 13.48). True canonical-equivalent "
                    "1s-LONG Sharpe expected ~20.5 if real; much lower if "
                    "artifact.",
        },
    }

    out_json = OUT_DIR / "canonical_check_results.json"
    out_json.write_text(json.dumps(results, indent=2, default=float))
    print(f"\n[done] wrote {out_json}")

    # ---------- VERDICT.md ----------
    repro_ok = results["control_trial_278"]["reproduction_ok"]
    s_no_fifo = m_1s_long_canon["sharpe"]
    s_fifo = m_1s_long_fifo["sharpe"]
    nf_no_fifo = m_1s_long_canon["n_fills"]
    nf_fifo = m_1s_long_fifo["n_fills"]
    dc_fifo = m_1s_long_fifo["day_conc"]
    dc_no_fifo = m_1s_long_canon["day_conc"]
    strict_fifo = m_1s_long_fifo["hc344_strict_day_conc_le_0_20"]
    strict_no_fifo = m_1s_long_canon["hc344_strict_day_conc_le_0_20"]
    # Decision: at least one of the two canonical configs must STRICT-PASS at
    # Sharpe >= 10 with n_fills >= 30 to be considered a real CO-MVP.
    real_co_mvp = (strict_no_fifo and s_no_fifo >= 10 and nf_no_fifo >= 30) or \
                  (strict_fifo and s_fifo >= 10 and nf_fifo >= 30)

    if not repro_ok:
        verdict_line = "**ABORTED — trial 278 control failed to reproduce in canonical harness. Cannot trust 1s-LONG numbers.**"
    elif real_co_mvp:
        verdict_line = "**1s-LONG candidate is CO-MVP** (canonical Sharpe >= 10 and STRICT-PASS day_conc <= 0.20 with n_fills >= 30)."
    else:
        verdict_line = "**1s-LONG candidate is HARNESS-ARTIFACT, not real** (canonical replay does not reproduce agent B's Sharpe under either FIFO-confluence or no-FIFO-confluence configurations at strict-pass)."

    verdict_md = f"""# HC #403 FOLLOWUP — 1s-LONG Canonical Cross-Check Verdict

Produced: {time.strftime("%Y-%m-%d %H:%M:%S ET")}
Output: `{OUT_DIR.relative_to(PROJ)}/`

## Verdict
{verdict_line}

## Control: Trial 278 reproduction
- Expected: Sharpe 13.48, day_conc 0.186, n_fills 195
- Replayed: Sharpe {m_278['sharpe']:.2f}, day_conc {m_278['day_conc']:.3f}, n_fills {m_278['n_fills']}
- PASS: {repro_ok} → canonical harness is the trusted reference

## Candidate: 1s-LONG, top-10% conf, 2s hold, passive_+2, ToD 13-15 ET

### (B) Canonical pipeline (no FIFO confluence — same as trial 278's measured 13.48)
| metric | value |
|---|---|
| Sharpe | {s_no_fifo:.2f} |
| Sortino | {m_1s_long_canon['sortino']:.2f} |
| PF | {m_1s_long_canon['pf']:.2f} |
| WR | {m_1s_long_canon['wr']:.1f}% |
| mean_net (tk/fill) | {m_1s_long_canon['mean_net']:.3f} |
| day_conc | {dc_no_fifo:.3f} |
| n_fills | {nf_no_fifo} |
| CI_low_95 | {m_1s_long_canon['ci_low_95']:.3f} |
| HC #344 strict pass | {strict_no_fifo} |
| HC #344 relaxed pass | {m_1s_long_canon['hc344_relaxed_day_conc_le_0_70']} |

### (C) Canonical pipeline + FIFO confluence > 0.80 (like agent B)
| metric | value |
|---|---|
| Sharpe | {s_fifo:.2f} |
| Sortino | {m_1s_long_fifo['sortino']:.2f} |
| PF | {m_1s_long_fifo['pf']:.2f} |
| WR | {m_1s_long_fifo['wr']:.1f}% |
| mean_net (tk/fill) | {m_1s_long_fifo['mean_net']:.3f} |
| day_conc | {dc_fifo:.3f} |
| n_fills | {nf_fifo} |
| CI_low_95 | {m_1s_long_fifo['ci_low_95']:.3f} |
| HC #344 strict pass | {strict_fifo} |
| HC #344 relaxed pass | {m_1s_long_fifo['hc344_relaxed_day_conc_le_0_70']} |

## Agent B reported (for reference)
- Sharpe 23.57, n_fills 106, day_conc 0.178, mean_net 1.83
- Custom harness inflated trial 278 by ~15% vs canonical (15.2 vs 13.48)

## Interpretation
- (B) is the apples-to-apples comparison with trial 278's published 13.48
  (canonical pipeline applies only ToD + pred_strength).
- (C) layers FIFO confluence on top of canonical, matching agent B's exact
  filter stack. The delta (C)-(B) quantifies how much of agent B's edge is
  due to FIFO confluence vs the base 1s-long signal.
- If (C) Sharpe is materially lower than agent B's 23.57, the difference is
  the harness-inflation factor; the ~15% inflation hypothesis predicts
  canonical Sharpe ~20.5.
"""
    (OUT_DIR / "VERDICT.md").write_text(verdict_md)
    print(f"[done] wrote {OUT_DIR / 'VERDICT.md'}")

    # Stdout summary
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  trial_278 reproduction OK:       {repro_ok}")
    print(f"  1s-LONG canonical (no FIFO):     "
          f"Sharpe={s_no_fifo:.2f} n_fills={nf_no_fifo} "
          f"day_conc={dc_no_fifo:.3f} strict={strict_no_fifo}")
    print(f"  1s-LONG canonical (with FIFO):   "
          f"Sharpe={s_fifo:.2f} n_fills={nf_fifo} "
          f"day_conc={dc_fifo:.3f} strict={strict_fifo}")
    print(f"  Agent B custom harness reported: Sharpe=23.57 n_fills=106 day_conc=0.178")
    print("=" * 78)
    print(verdict_line)
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
