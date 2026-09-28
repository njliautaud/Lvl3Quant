"""
boost_meta_lgbm_gate.py — HC #427 R5 boosting experiment (b).

Train a meta-LGBM classifier that predicts trade-success (mean_net > 0.376t after
canonical commission) from the full multi-head feature vector of the ensemble
NPZ. Use it as a confluence gate on top of the LOO-robust configs.

WHY: each LOO-robust config currently gates on a single conf_thr against one
prediction head. The other 31 heads carry useful confluence info we're not
using. A meta-LGBM tutorialed to predict "this signal will be profitable after
costs" can refine the conf_thr filter with a richer feature vector.

METHOD
  1. Load ensemble NPZ. For each of 32 pred_* heads, compute a feature.
  2. For each LOO-robust v3.4.2 ensemble setup (the strong baseline, 11 configs):
     - replay through full_market_replay → get per-fill ledger with mean_net
     - label each fill: y=1 if net_ticks > 0.376 (passive commission), y=0 else
     - train LGBM classifier on per-fill feature vector (signal-time multi-head
       predictions) → predict P(profitable)
     - re-replay with gate: only trade when P(profitable) >= threshold (sweep
       threshold 0.40, 0.50, 0.60, 0.65, 0.70)
  3. For each (config, threshold) measure improvement in: n_fills, mean_net,
     LOO Sharpe (per-day), worst-day Sharpe, robustness gate pass.
  4. Identify any config × threshold that beats the raw config on worst-day Sh
     while keeping n_fills >= 30. Those become new boosted setups.

OUTPUT
  output/boost_meta_lgbm_v3.4.2_ensemble/
    boosted_configs.json        # any (orig_trial, threshold) that improved
    verdict.md                  # one-page summary
    per_config_results.json     # full sweep results

HC #420 codebase auth. HC #393 autonomy. HC #427 R5 boosting #2.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np

PROJ = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(PROJ))

# Lazy import LGBM (might not be installed; fall back gracefully)
try:
    import lightgbm as lgb
    HAVE_LGBM = True
except ImportError:
    HAVE_LGBM = False

from scripts.v3_3_research.full_market_replay import (  # noqa: E402
    full_market_replay,
    TradeConfig,
)
from scripts.v3_3_research.v33_execution_optuna_full_market_replay import (  # noqa: E402
    apply_post_filters,
    metrics_from_filtered,
)

ENS_NPZ = PROJ / "output" / "cnn_mamba_ensemble_v33_v342" / "fold_00_predictions.npz"
ROBUST_V342 = PROJ / "output" / "cnn_mamba_ensemble_v33_v342" / "loo_robust_configs_on_v342_top.json"
LABELS_DIR = PROJ / "data" / "processed" / "mbo_events_smart_v3_fifo_labels"
OUT_DIR = PROJ / "output" / "boost_meta_lgbm_v3.4.2_ensemble"

CANONICAL_COMMISSION_TICKS = 0.376  # HC #426 R4 / CLAUDE.md COST CONSTANTS
# NOTE: ledger.fills[].net_ticks already includes commission deduction, so
# the "profitable" label is simply net_ticks > 0 (NOT > commission).
PROFITABLE_NET_TICKS_THR = 0.0
ROBUSTNESS_MIN_FILLS = 30  # HC #344
ROBUSTNESS_MIN_PROFITABLE_DAYS = 4
ROBUSTNESS_MIN_DAYS_WITH_FILLS = 4
ROBUSTNESS_WORST_DAY_SH_FLOOR = -0.5

THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.50, 0.60]


def load_ensemble_features() -> tuple[dict, list[str], list[str]]:
    """Return (data dict {key: array}, feature_key_list, oot_dates_list)."""
    z = np.load(ENS_NPZ, allow_pickle=True)
    data = {k: z[k] for k in z.files}
    feature_keys = [k for k in z.files if k.startswith("pred_")]
    oot_dates = [str(x) for x in z["oot_dates"]]
    return data, feature_keys, oot_dates


def collect_per_fill_features(ledger, data: dict, feature_keys: list[str]) -> np.ndarray:
    """
    For each fill in ledger, look up the multi-head feature vector at the
    fill's source-sample index (ledger should record the row index of the
    prediction that triggered the trade).
    """
    # full_market_replay's ledger.fills typically has a 'pred_idx' or 'signal_idx'
    # to map back to the source NPZ row. Fall back gracefully if not present.
    if not hasattr(ledger, "fills") or len(ledger.fills) == 0:
        return np.zeros((0, len(feature_keys)), dtype=np.float32)
    feats = []
    for fill in ledger.fills:
        idx = None
        for attr in ("signal_idx", "pred_idx", "src_idx", "source_idx"):
            if hasattr(fill, attr):
                idx = getattr(fill, attr)
                break
        if idx is None:
            # No mapping — meta-gate not feasible for this ledger schema.
            return None
        row = [float(data[k][idx]) if 0 <= idx < len(data[k]) else 0.0
               for k in feature_keys]
        feats.append(row)
    return np.asarray(feats, dtype=np.float32)


@dataclass
class GateResult:
    threshold: float
    n_fills: int
    mean_net: float
    sharpe: float
    worst_day_sharpe: float
    n_profitable_days: int
    robust: bool


def evaluate_with_gate(
    params: dict,
    oot_dates: list[str],
    data: dict,
    feature_keys: list[str],
    threshold: float,
    model: "lgb.Booster",
) -> dict:
    """Run replay per-day; for each fill, query meta-model; keep only if P(profit) >= threshold."""
    per_day = []
    cfg = TradeConfig(
        side=params["side"],
        horizon=params["head_horizon"],
        confidence_threshold=float(params["conf_thr"]),
        order_type=params["order_type"],
        cancel_eval_window=int(params["cancel_window"]),
        hold_seconds=float(params["hold_seconds"]),
    )
    for day in oot_dates:
        try:
            ledger = full_market_replay(
                ENS_NPZ, LABELS_DIR, cfg, dates=[day],
                spread_ticks_rth=float(params["spread_ticks"]),
                rt_commission_ticks=float(params["commission_ticks"]),
            )
        except Exception as e:
            per_day.append({"day": day, "n_fills": 0, "mean_net": 0.0,
                            "sharpe": 0.0, "error": str(e)[:120]})
            continue
        if ledger is None or ledger.n_filled == 0:
            per_day.append({"day": day, "n_fills": 0, "mean_net": 0.0, "sharpe": 0.0})
            continue
        feats = collect_per_fill_features(ledger, data, feature_keys)
        if feats is None:
            # No idx mapping — can't gate. Use raw ledger.
            try:
                df_f, _ = apply_post_filters(
                    ledger,
                    tod_start_hour=int(params["tod_start_hour"]),
                    tod_end_hour=int(params["tod_end_hour"]),
                    require_min_pred_strength=float(params["pred_strength_min"]),
                )
                m = metrics_from_filtered(df_f)
            except Exception:
                m = {}
            per_day.append({
                "day": day,
                "n_fills": int(m.get("n_fills", 0)),
                "mean_net": float(m.get("mean_net", 0.0)),
                "sharpe": float(m.get("sharpe", 0.0)),
                "gate_active": False,
            })
            continue
        # Gate
        prob = model.predict(feats)
        keep = prob >= threshold
        if keep.sum() == 0:
            per_day.append({"day": day, "n_fills": 0, "mean_net": 0.0, "sharpe": 0.0,
                            "gate_active": True, "gate_kept_pct": 0.0})
            continue
        # Filter the ledger to kept fills, then re-run post-filters
        ledger.fills = [f for f, k in zip(ledger.fills, keep) if k]
        ledger.n_filled = len(ledger.fills)
        try:
            df_f, _ = apply_post_filters(
                ledger,
                tod_start_hour=int(params["tod_start_hour"]),
                tod_end_hour=int(params["tod_end_hour"]),
                require_min_pred_strength=float(params["pred_strength_min"]),
            )
            m = metrics_from_filtered(df_f)
        except Exception as e:
            per_day.append({"day": day, "n_fills": int(keep.sum()),
                            "error": f"postfilter:{e}"[:120]})
            continue
        per_day.append({
            "day": day,
            "n_fills": int(m.get("n_fills", 0)),
            "mean_net": float(m.get("mean_net", 0.0)),
            "sharpe": float(m.get("sharpe", 0.0)),
            "gate_active": True,
            "gate_kept_pct": float(keep.mean()),
        })

    # Aggregate to LOO robustness verdict
    valid = [d for d in per_day if "error" not in d]
    n_profitable = sum(1 for d in valid if d["mean_net"] > 0)
    n_fills_total = sum(d["n_fills"] for d in valid)
    n_days_with_fills = sum(1 for d in valid if d["n_fills"] > 0)
    sharpes = [d["sharpe"] for d in valid if d["n_fills"] > 0]
    worst_sharpe = min(sharpes) if sharpes else -999.0
    mean_sharpe = float(np.mean(sharpes)) if sharpes else 0.0
    robust = (
        n_profitable >= ROBUSTNESS_MIN_PROFITABLE_DAYS
        and n_fills_total >= ROBUSTNESS_MIN_FILLS
        and worst_sharpe > ROBUSTNESS_WORST_DAY_SH_FLOOR
        and n_days_with_fills >= ROBUSTNESS_MIN_DAYS_WITH_FILLS
    )
    return {
        "threshold": threshold,
        "per_day": per_day,
        "n_profitable_days": n_profitable,
        "n_fills_total": n_fills_total,
        "n_days_with_fills": n_days_with_fills,
        "worst_day_sharpe": worst_sharpe,
        "mean_day_sharpe": mean_sharpe,
        "robust": robust,
    }


def train_meta_lgbm(
    train_dates: list[str],
    params: dict,
    data: dict,
    feature_keys: list[str],
) -> "lgb.Booster | None":
    """Train an LGBM on per-fill features from train_dates; label by post-commission net."""
    if not HAVE_LGBM:
        print("[meta] lightgbm not installed — skipping meta-gate training")
        return None
    cfg = TradeConfig(
        side=params["side"], horizon=params["head_horizon"],
        confidence_threshold=float(params["conf_thr"]),
        order_type=params["order_type"],
        cancel_eval_window=int(params["cancel_window"]),
        hold_seconds=float(params["hold_seconds"]),
    )
    Xs, ys = [], []
    for day in train_dates:
        try:
            ledger = full_market_replay(
                ENS_NPZ, LABELS_DIR, cfg, dates=[day],
                spread_ticks_rth=float(params["spread_ticks"]),
                rt_commission_ticks=float(params["commission_ticks"]),
            )
        except Exception:
            continue
        if ledger is None or ledger.n_filled == 0:
            continue
        feats = collect_per_fill_features(ledger, data, feature_keys)
        if feats is None or len(feats) == 0:
            continue
        nets = np.asarray([getattr(f, "net_ticks", 0.0) for f in ledger.fills], dtype=np.float32)
        Xs.append(feats)
        ys.append((nets > PROFITABLE_NET_TICKS_THR).astype(np.int32))
    if not Xs:
        return None
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    if len(np.unique(y)) < 2:
        return None
    train_set = lgb.Dataset(X, label=y, feature_name=feature_keys)
    # Aggressively regularize for the small-n regime (typically 30-200 fills per training set).
    params_lgb = dict(
        objective="binary", metric="binary_logloss",
        num_leaves=4, learning_rate=0.05, min_data_in_leaf=max(5, len(y) // 20),
        feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=3,
        lambda_l1=0.5, lambda_l2=1.0, max_depth=3,
        verbose=-1,
    )
    n_rounds = min(50, max(10, len(y) // 4))
    model = lgb.train(params_lgb, train_set, num_boost_round=n_rounds)
    return model


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not HAVE_LGBM:
        print("FATAL: lightgbm not installed.")
        return 2
    if not ROBUST_V342.exists():
        print(f"FATAL: {ROBUST_V342} missing — run boost_ensemble_v33_v342.py first.")
        return 1

    data, feature_keys, oot_dates = load_ensemble_features()
    print(f"[meta] ensemble loaded, {len(feature_keys)} features, "
          f"OOT dates: {oot_dates}")

    robust_configs = json.loads(ROBUST_V342.read_text())
    print(f"[meta] {len(robust_configs)} LOO-robust v3.4.2-basis ensemble configs to test")

    per_config_results = []
    boosted_configs = []
    for ci, cfg_entry in enumerate(robust_configs):
        params = cfg_entry["params"]
        trial = cfg_entry.get("trial")
        baseline_worst = cfg_entry.get("worst_day_sharpe", 0.0)
        baseline_fills = cfg_entry.get("n_fills_total", 0)
        baseline_mean_sh = cfg_entry.get("mean_day_sharpe", 0.0)

        print(f"\n=== [{ci+1}/{len(robust_configs)}] trial={trial} "
              f"{params['head_horizon']}/{params['side']}/{params['order_type']} ===")
        print(f"  baseline: worst_sh={baseline_worst:.2f} fills={baseline_fills} "
              f"mean_sh={baseline_mean_sh:.2f}")

        # Per-day-LOO meta-training: hold one OOT day out, train on the other 4,
        # gate that one day. Repeat for all 5. This gives a leak-free per-day verdict.
        loo_per_day = []
        for hold_day in oot_dates:
            train_days = [d for d in oot_dates if d != hold_day]
            model = train_meta_lgbm(train_days, params, data, feature_keys)
            if model is None:
                loo_per_day.append({"hold_day": hold_day, "err": "no_train_signal"})
                continue
            # Evaluate ALL thresholds on the held-out day only
            for th in THRESHOLDS:
                r = evaluate_with_gate(params, [hold_day], data, feature_keys, th, model)
                loo_per_day.append({
                    "hold_day": hold_day, "threshold": th,
                    "n_fills": r["n_fills_total"],
                    "mean_net": r["per_day"][0].get("mean_net", 0.0),
                    "sharpe": r["per_day"][0].get("sharpe", 0.0),
                })

        # Aggregate per-threshold across the LOO days
        per_th_summary = {}
        for th in THRESHOLDS:
            sl = [d for d in loo_per_day if d.get("threshold") == th]
            sharpes = [d["sharpe"] for d in sl if d.get("n_fills", 0) > 0]
            fills_total = sum(d.get("n_fills", 0) for d in sl)
            worst = min(sharpes) if sharpes else -999.0
            mean = float(np.mean(sharpes)) if sharpes else 0.0
            n_prof = sum(1 for d in sl if d.get("mean_net", 0) > 0)
            n_dwf = sum(1 for d in sl if d.get("n_fills", 0) > 0)
            robust = (
                n_prof >= ROBUSTNESS_MIN_PROFITABLE_DAYS
                and fills_total >= ROBUSTNESS_MIN_FILLS
                and worst > ROBUSTNESS_WORST_DAY_SH_FLOOR
                and n_dwf >= ROBUSTNESS_MIN_DAYS_WITH_FILLS
            )
            improves = (worst > baseline_worst) and (fills_total >= ROBUSTNESS_MIN_FILLS)
            per_th_summary[th] = {
                "n_fills_total": fills_total, "worst_day_sharpe": worst,
                "mean_day_sharpe": mean, "n_profitable_days": n_prof,
                "n_days_with_fills": n_dwf, "robust": robust,
                "improves_worst_day": improves,
            }
            mark = "✓" if robust else "✗"
            mark_imp = "▲" if improves else " "
            print(f"  th={th:.2f}  fills={fills_total:>3}  worst_sh={worst:>6.2f}  "
                  f"mean_sh={mean:>6.2f}  prof={n_prof}/5  {mark}{mark_imp}")

        # Best threshold = highest worst_day_sh among robust ones
        best_th_robust = sorted(
            [(th, s) for th, s in per_th_summary.items() if s["robust"]],
            key=lambda x: x[1]["worst_day_sharpe"], reverse=True,
        )
        rec = {
            "trial": trial,
            "params": params,
            "baseline": {
                "worst_day_sharpe": baseline_worst,
                "n_fills_total": baseline_fills,
                "mean_day_sharpe": baseline_mean_sh,
            },
            "per_threshold": per_th_summary,
            "best_robust_threshold": (best_th_robust[0][0] if best_th_robust else None),
            "best_robust_metrics": (best_th_robust[0][1] if best_th_robust else None),
        }
        per_config_results.append(rec)

        if best_th_robust:
            th, s = best_th_robust[0]
            if s["worst_day_sharpe"] > baseline_worst:
                rec["boosted_over_baseline"] = True
                boosted_configs.append(rec)
                print(f"  ★ BOOSTED: th={th:.2f} worst_sh {baseline_worst:.2f} → {s['worst_day_sharpe']:.2f}")

    (OUT_DIR / "per_config_results.json").write_text(
        json.dumps(per_config_results, indent=2, default=str))
    (OUT_DIR / "boosted_configs.json").write_text(
        json.dumps(boosted_configs, indent=2, default=str))

    # Verdict
    md = [
        "# Boosting (b) verdict — meta-LGBM trade-success gate on v3.4.2-basis ensemble",
        "",
        f"HC #427 R5 boosting technique #2. Inputs: {len(robust_configs)} LOO-robust configs from"
        f" boosting (a). Per-day LOO meta-training (hold 1 day out, train on 4, gate that day).",
        "",
        f"## Summary",
        f"- Configs tested: {len(robust_configs)}",
        f"- Configs where meta-gate found a robust threshold that BEATS baseline worst-day Sharpe: "
        f"**{len(boosted_configs)}**",
        f"- Thresholds swept: {THRESHOLDS}",
        "",
        "## Boosted configs (worth promoting)",
        "",
    ]
    if boosted_configs:
        md.append("| trial | side/horizon | best_th | baseline worst_Sh | boosted worst_Sh | "
                  "n_fills | mean_Sh |")
        md.append("|---|---|---|---|---|---|---|")
        for r in boosted_configs:
            th = r["best_robust_threshold"]
            s = r["best_robust_metrics"]
            md.append(
                f"| {r['trial']} | {r['params']['head_horizon']}/{r['params']['side']} | "
                f"{th:.2f} | {r['baseline']['worst_day_sharpe']:.2f} | "
                f"{s['worst_day_sharpe']:.2f} | {s['n_fills_total']} | "
                f"{s['mean_day_sharpe']:.2f} |"
            )
    else:
        md.append("_No config saw worst-day Sharpe improvement under any threshold._")
        md.append("")
        md.append("**Interpretation**: The 32-head feature vector at signal time does not"
                  " consistently predict per-fill profitability beyond what the existing"
                  " conf_thr+horizon_confluence+fifo_confluence already captures. Boosting"
                  " (b) does not advance HC #427 R5 with this design; try alternative meta"
                  " architectures (regime conditioning, weighted ensemble).")
    (OUT_DIR / "verdict.md").write_text("\n".join(md))
    print(f"\n[verdict] {OUT_DIR / 'verdict.md'}")
    print(f"[json] per-config: {OUT_DIR / 'per_config_results.json'}")
    print(f"[json] boosted:    {OUT_DIR / 'boosted_configs.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
