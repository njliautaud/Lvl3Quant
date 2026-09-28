#!/usr/bin/env python3
"""
meta_classifier_v1_fifo_replay.py — Canonical FIFO market-replay (HC #74) of the
two surviving candidate cells from meta_classifier_v1_walkforward.

Candidates:
  1) short_10s @ threshold 0.55  (3.46 t/trade label, Sharpe 7.70, 4248 pooled trades)
  2) long_1s   @ threshold 0.55  (0.26 t/trade label, Sharpe 17.4, 26186 pooled trades)

Labels in meta_classifier_v1_walkforward assumed PASSIVE-LIMIT fills with commission
0.376 t baked in. This script tests the fill-realism honestly:
  - Place a passive limit at the joining-side touch (bid for long, ask for short).
  - Track FIFO queue position through the raw MBO event stream.
  - Limit fills only when volume trades through to its position.
  - Cancel unfilled orders after `cancel_after_ns` (= horizon h per HC #428 R2).
  - Filled positions are held until `max_hold_ns` (= 1.5 * h per HC #428 R2),
    then exited at MARKET (+1 t spread crossing). TP/SL are set extremely wide
    so they never fire — exit is by max_hold so all trades exit at the horizon.

Outputs (in /home/jupiter/Lvl3Quant/output/meta_classifier_v1_fifo/):
  - fifo_summary.csv         per-candidate metrics
  - per_trade_diagnostics.csv every signal w/ fill/no-fill + realized P&L
  - per_day_fifo.csv          per-day net ticks
  - REPORT.md                 <=30-line verdict per candidate
  - .regen_complete.json      per HC #485 R5

Run:
  python3 scripts/meta_classifier_v1_fifo_replay.py
"""
from __future__ import annotations

import json
import math
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3_ROOT))

try:
    import lightgbm as lgb
except Exception as e:
    print(f"[fatal] lightgbm import failed: {e}", file=sys.stderr)
    sys.exit(2)

from alpha_discovery.deep_models.fifo_market_replay import (  # noqa: E402
    FIFOReplayEngine,
    COMMISSION_TICKS,
)

# --------- Config -----------------------------------------------------------
DATA_DIR = LVL3_ROOT / "data" / "processed" / "meta_layer_v1"
MBO_EVENT_DIR = LVL3_ROOT / "data" / "processed" / "mbo_events_smart_v3"
MODELS_ROOT = LVL3_ROOT / "output" / "meta_classifier_v1_wf" / "models"
OUT_DIR = LVL3_ROOT / "output" / "meta_classifier_v1_fifo"

N_FOLDS = 4
N_IS_PER_FOLD = 16
N_OOT_PER_FOLD = 4
FOLD_STRIDE = 4
ANNUAL_TRADING_DAYS = 252.0

# Wide TP/SL so neither fires; max_hold determines exit at horizon.
TP_TICKS_WIDE = 1000.0
SL_TICKS_WIDE = 1000.0

# HC #428 R2: cancel <= horizon, hold <= 1.5*horizon
# Market exit at horizon costs +1.0 t spread crossing (post-corrected; engine
# exits at mid for max_hold).
MARKET_EXIT_SPREAD_TICKS = 1.0

# Candidate definitions (label_assumption is the wf-summary pooled net_t/trade)
CANDIDATES = [
    {
        "name": "short_10s_thr55",
        "side": "short",
        "horizon_sec": 10.0,
        "model_tag": "short_10s",
        "threshold": 0.55,
        "label_assumption_net_t": 3.463,
        "label_assumption_sharpe": 7.70,
    },
    {
        "name": "long_1s_thr55",
        "side": "long",
        "horizon_sec": 1.0,
        "model_tag": "long_1s",
        "threshold": 0.55,
        "label_assumption_net_t": 0.264,
        "label_assumption_sharpe": 17.36,
    },
]

# HC #428 deploy gates
GATE_NET = 0.10
GATE_SHARPE = 0.3
GATE_PDAYS_ABS = 11
GATE_PDAYS_TOTAL = 16
GATE_REGIME_IMB = 0.50
GATE_DAYCONC = 0.70

REGIME_THRESH = 0.10  # tick threshold on long_30s_net to classify green/red day


# --------- Helpers ----------------------------------------------------------
def sharpe_per_day(day_means: np.ndarray) -> float:
    d = day_means[np.isfinite(day_means)]
    if d.size < 2:
        return float("nan")
    mu = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    if sd <= 1e-12:
        return float("nan")
    return mu / sd * math.sqrt(ANNUAL_TRADING_DAYS)


def regime_of_day(day_long30s_net: np.ndarray) -> str:
    v = day_long30s_net[np.isfinite(day_long30s_net)]
    if v.size == 0:
        return "flat"
    mu = float(np.mean(v))
    if mu > REGIME_THRESH:
        return "green"
    if mu < -REGIME_THRESH:
        return "red"
    return "flat"


def list_sorted_days() -> List[str]:
    files = sorted(DATA_DIR.glob("*_meta.npz"))
    return [f.stem.replace("_meta", "") for f in files]


def fold_oot_dates(all_days: List[str]) -> Dict[int, List[str]]:
    folds = {}
    for k in range(N_FOLDS):
        is_start = k * FOLD_STRIDE
        oot_start = is_start + N_IS_PER_FOLD
        oot_end = oot_start + N_OOT_PER_FOLD
        folds[k] = all_days[oot_start:oot_end]
    return folds


def load_model(fold_idx: int, model_tag: str) -> lgb.Booster:
    p = MODELS_ROOT / f"fold_{fold_idx}" / f"{model_tag}.txt"
    if not p.exists():
        raise FileNotFoundError(p)
    return lgb.Booster(model_file=str(p))


def load_meta_day(date_str: str) -> Dict:
    p = DATA_DIR / f"{date_str}_meta.npz"
    d = np.load(p, allow_pickle=True)
    return dict(
        X=d["X"].astype(np.float32),
        y=d["y"].astype(np.float32),
        event_idx=d["event_idx"].astype(np.int64),
        label_names=[str(s) for s in d["label_names"]],
    )


def load_mbo_timestamps(date_str: str) -> np.ndarray:
    p = MBO_EVENT_DIR / f"{date_str}_mbo_events.npz"
    d = np.load(p, allow_pickle=True)
    return d["timestamps"].astype(np.int64)


def build_signals(
    proba: np.ndarray,
    event_idx: np.ndarray,
    threshold: float,
    side: str,
    ts_ns_by_event: np.ndarray,
) -> Tuple[List[dict], np.ndarray]:
    """Return (signals list for FIFOReplayEngine, mask of selected rows)."""
    sel = (proba >= threshold) & np.isfinite(proba)
    # Map each selected meta row to its ns timestamp via event_idx
    valid = sel.copy()
    sig_event_idx = event_idx[sel]
    in_range = (sig_event_idx >= 0) & (sig_event_idx < len(ts_ns_by_event))
    # Restrict selection to in-range
    where_sel = np.where(sel)[0]
    drop = where_sel[~in_range]
    valid[drop] = False
    sig_event_idx = event_idx[valid]
    sig_ts = ts_ns_by_event[sig_event_idx]
    direction = side  # 'short' or 'long'
    signals = [
        {
            "ts_ns": int(t),
            "direction": direction,
            "strength": float(proba[i]),
        }
        for t, i in zip(sig_ts, np.where(valid)[0])
    ]
    return signals, valid


def run_day(
    date_str: str,
    fold_idx: int,
    cand: dict,
    booster: lgb.Booster,
) -> Dict:
    """Run FIFO replay for one OOT day under one candidate's fold model."""
    meta = load_meta_day(date_str)
    X = meta["X"]
    event_idx = meta["event_idx"]
    label_names = meta["label_names"]
    label_idx = {n: i for i, n in enumerate(label_names)}

    # Score events (need finite X rows)
    finite_x = np.all(np.isfinite(X), axis=1)
    proba = np.full(X.shape[0], np.nan, dtype=np.float64)
    if finite_x.any():
        proba[finite_x] = booster.predict(X[finite_x])

    # Build signals from selected rows
    ts_ns_by_event = load_mbo_timestamps(date_str)
    signals, valid_mask = build_signals(
        proba=proba,
        event_idx=event_idx,
        threshold=cand["threshold"],
        side=cand["side"],
        ts_ns_by_event=ts_ns_by_event,
    )
    n_signaled = int(valid_mask.sum())

    # Day regime via long_30s_net
    long30s_net_col = label_idx.get("y_long_30s_net")
    if long30s_net_col is not None:
        day_regime = regime_of_day(meta["y"][:, long30s_net_col])
    else:
        day_regime = "flat"

    # Label assumption per signaled row (for comparison)
    side_h_net_col = label_idx[f"y_{cand['side']}_{int(cand['horizon_sec'])}s_net"]
    label_net = meta["y"][:, side_h_net_col][valid_mask]
    label_mean_net = float(np.nanmean(label_net)) if label_net.size else float("nan")

    if n_signaled == 0:
        return dict(
            date=date_str, fold=fold_idx, candidate=cand["name"],
            n_signaled=0, n_filled=0, fill_rate=float("nan"),
            avg_time_to_fill_ms=float("nan"),
            mean_net_t_realized=float("nan"),
            sum_net_t_realized=0.0,
            mean_net_t_label=label_mean_net,
            day_regime=day_regime,
            trades=[],
        )

    cancel_ns = int(round(cand["horizon_sec"] * 1e9))
    max_hold_ns = int(round(1.5 * cand["horizon_sec"] * 1e9))

    try:
        engine = FIFOReplayEngine(
            date=date_str,
            instrument_id=None,
            cancel_after_ns=cancel_ns,
            max_hold_ns=max_hold_ns,
            max_reprices=0,
            reprice_after_ns=int(1e9),
        )
    except Exception as e:
        return dict(
            date=date_str, fold=fold_idx, candidate=cand["name"],
            n_signaled=n_signaled, n_filled=0, fill_rate=0.0,
            avg_time_to_fill_ms=float("nan"),
            mean_net_t_realized=float("nan"),
            sum_net_t_realized=0.0,
            mean_net_t_label=label_mean_net,
            day_regime=day_regime,
            trades=[],
            error=f"engine_init_failed: {e}",
        )

    trades = engine.simulate(
        signals,
        tp_ticks=TP_TICKS_WIDE,
        sl_ticks=SL_TICKS_WIDE,
        order_type="limit",
        order_management="realtime_sl",
    )

    # Post-correction: for max_hold/eod exits, subtract +1.0 t for market spread
    # crossing on exit. The engine already subtracted entry-side commission
    # (0.376 t) — we ADD the spread-crossing cost (this is a CHARGE so reduces PnL).
    trade_records = []
    for t in trades:
        exit_spread_charge = 0.0
        if t.exit_reason in ("max_hold", "eod"):
            exit_spread_charge = MARKET_EXIT_SPREAD_TICKS
        # The engine reports pnl_ticks_net = pnl_ticks - 0.376 (entry commission only).
        # We further subtract spread on market exit.
        realized_net = float(t.pnl_ticks_net) - exit_spread_charge
        trade_records.append(dict(
            signal_ts_ns=int(t.signal_ts_ns),
            fill_ts_ns=int(t.entry_ts_ns),
            exit_ts_ns=int(t.exit_ts_ns),
            direction=t.direction,
            queue_ahead=int(t.queue_ahead),
            queue_wait_ns=int(t.queue_wait_ns),
            exit_reason=t.exit_reason,
            pnl_ticks_gross=float(t.pnl_ticks),
            pnl_ticks_net=realized_net,
        ))

    n_filled = len(trade_records)
    if n_filled > 0:
        nets = np.array([r["pnl_ticks_net"] for r in trade_records])
        wait_ms = np.array([r["queue_wait_ns"] / 1e6 for r in trade_records])
        mean_net = float(np.mean(nets))
        sum_net = float(np.sum(nets))
        avg_wait_ms = float(np.mean(wait_ms))
    else:
        mean_net = float("nan")
        sum_net = 0.0
        avg_wait_ms = float("nan")

    return dict(
        date=date_str, fold=fold_idx, candidate=cand["name"],
        n_signaled=n_signaled, n_filled=n_filled,
        fill_rate=n_filled / max(1, n_signaled),
        avg_time_to_fill_ms=avg_wait_ms,
        mean_net_t_realized=mean_net,
        sum_net_t_realized=sum_net,
        mean_net_t_label=label_mean_net,
        day_regime=day_regime,
        trades=trade_records,
    )


def deploy_verdict(row: dict) -> Tuple[str, List[str]]:
    """Return (verdict, reasons) per HC #428 deploy bar."""
    reasons = []
    if not np.isfinite(row["mean_net_t_realized"]) or row["mean_net_t_realized"] <= GATE_NET:
        reasons.append(f"net_t<={GATE_NET}")
    if not np.isfinite(row["sharpe_pooled"]) or row["sharpe_pooled"] <= GATE_SHARPE:
        reasons.append(f"Sharpe<={GATE_SHARPE}")
    if row["profitable_days"] < GATE_PDAYS_ABS:
        reasons.append(f"pdays<{GATE_PDAYS_ABS}/{GATE_PDAYS_TOTAL}")
    if np.isfinite(row.get("regime_imbalance", float("nan"))) and \
            row["regime_imbalance"] > GATE_REGIME_IMB:
        reasons.append(f"regime_imb>{GATE_REGIME_IMB}")
    if row["day_concentration"] > GATE_DAYCONC:
        reasons.append(f"dayconc>{GATE_DAYCONC}")
    if row["fill_rate_overall"] < 0.05:
        reasons.append(f"fill_rate<5% (economically dead)")
    verdict = "DEPLOY" if not reasons else "REJECT"
    return verdict, reasons


# --------- Main -------------------------------------------------------------
def main():
    t0 = time.time()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[start] {started_at}", flush=True)

    all_days = list_sorted_days()
    folds = fold_oot_dates(all_days)
    print(f"[folds] OOT plan:")
    for k, dates in folds.items():
        print(f"  fold {k}: {', '.join(dates)}", flush=True)

    per_day_rows = []        # one row per (candidate, fold, date)
    per_trade_rows = []      # one row per realized trade
    summary_rows = []        # one row per candidate (pooled)

    for cand in CANDIDATES:
        print(f"\n[candidate] {cand['name']}  side={cand['side']}  "
              f"h={cand['horizon_sec']}s  thr={cand['threshold']}", flush=True)
        # Load all 4 fold models once
        boosters = {k: load_model(k, cand["model_tag"]) for k in range(N_FOLDS)}

        pooled_nets: List[float] = []
        pooled_dates: List[str] = []
        pooled_labels: List[float] = []
        per_day_means: List[Tuple[str, float, float, str]] = []  # (date, day_mean_net, day_sum_net, regime)
        total_signaled = 0
        total_filled = 0

        for fold_idx, oot_dates in folds.items():
            for date_str in oot_dates:
                t_d = time.time()
                res = run_day(date_str, fold_idx, cand, boosters[fold_idx])
                dt = time.time() - t_d
                print(f"  [fold {fold_idx}] {date_str}: "
                      f"sig={res['n_signaled']:>6}  filled={res['n_filled']:>6}  "
                      f"fill%={100*res['fill_rate']:>5.1f}  "
                      f"avg_wait_ms={res['avg_time_to_fill_ms']:>7.1f}  "
                      f"mean_net_t={res['mean_net_t_realized']:>+7.3f}  "
                      f"sum_net_t={res['sum_net_t_realized']:>+8.2f}  "
                      f"({dt:.1f}s)", flush=True)

                per_day_rows.append(dict(
                    candidate=cand["name"], fold=fold_idx, date=date_str,
                    regime=res["day_regime"],
                    n_signaled=res["n_signaled"], n_filled=res["n_filled"],
                    fill_rate=res["fill_rate"],
                    avg_time_to_fill_ms=res["avg_time_to_fill_ms"],
                    day_mean_net_realized=res["mean_net_t_realized"],
                    day_sum_net_realized=res["sum_net_t_realized"],
                    day_mean_net_label_assumed=res["mean_net_t_label"],
                ))

                for tr in res["trades"]:
                    per_trade_rows.append(dict(
                        candidate=cand["name"],
                        fold=fold_idx,
                        date=date_str,
                        **tr,
                    ))
                    pooled_nets.append(tr["pnl_ticks_net"])
                    pooled_dates.append(date_str)

                if res["n_filled"] > 0 and np.isfinite(res["mean_net_t_realized"]):
                    per_day_means.append(
                        (date_str, res["mean_net_t_realized"],
                         res["sum_net_t_realized"], res["day_regime"])
                    )
                # Track label means weighted by signaled count
                if res["n_signaled"] > 0 and np.isfinite(res["mean_net_t_label"]):
                    pooled_labels.extend(
                        [res["mean_net_t_label"]] * res["n_filled"]
                        if res["n_filled"] > 0 else []
                    )

                total_signaled += res["n_signaled"]
                total_filled += res["n_filled"]

        # Pooled metrics
        nets_arr = np.array(pooled_nets, dtype=float)
        n_pooled = int(nets_arr.size)
        mean_realized = float(np.mean(nets_arr)) if n_pooled else float("nan")
        wr = float(np.mean(nets_arr > 0)) if n_pooled else float("nan")
        med_realized = float(np.median(nets_arr)) if n_pooled else float("nan")

        # Per-day arrays
        if per_day_means:
            day_means_arr = np.array([m[1] for m in per_day_means])
            day_sums_arr = np.array([m[2] for m in per_day_means])
            day_regs = np.array([m[3] for m in per_day_means])
        else:
            day_means_arr = np.array([np.nan])
            day_sums_arr = np.array([0.0])
            day_regs = np.array(["flat"])

        sh = sharpe_per_day(day_means_arr)
        sh_g = sharpe_per_day(day_means_arr[day_regs == "green"])
        sh_r = sharpe_per_day(day_means_arr[day_regs == "red"])
        denom = max(
            abs(sh_g) if np.isfinite(sh_g) else 0.0,
            abs(sh_r) if np.isfinite(sh_r) else 0.0,
            1e-9,
        )
        rim = (abs(sh_g - sh_r) / denom) if (
            np.isfinite(sh_g) and np.isfinite(sh_r)
        ) else float("nan")
        abs_tot = np.abs(day_sums_arr)
        dayconc = float(abs_tot.max() / abs_tot.sum()) if abs_tot.sum() > 0 else 1.0
        prof_days = int((day_means_arr > 0).sum())

        # Days with at least one filled trade (vs 16 OOT slots)
        days_with_trades = int(len(per_day_means))
        total_oot_days = sum(len(v) for v in folds.values())

        # Mean label assumption across pooled (informational)
        label_mean_pooled = float(np.mean(pooled_labels)) if pooled_labels else float("nan")

        fill_rate_overall = total_filled / max(1, total_signaled)
        delta_net = mean_realized - cand["label_assumption_net_t"]

        row = dict(
            candidate=cand["name"],
            side=cand["side"],
            horizon_sec=cand["horizon_sec"],
            threshold=cand["threshold"],
            total_signaled=total_signaled,
            total_filled=total_filled,
            fill_rate_overall=fill_rate_overall,
            mean_net_t_realized=mean_realized,
            median_net_t_realized=med_realized,
            win_rate_realized=wr,
            sharpe_pooled=sh,
            sharpe_green=sh_g,
            sharpe_red=sh_r,
            regime_imbalance=rim,
            day_concentration=dayconc,
            profitable_days=prof_days,
            total_days=total_oot_days,
            days_with_any_fill=days_with_trades,
            label_assumption_net_t=cand["label_assumption_net_t"],
            label_assumption_sharpe=cand["label_assumption_sharpe"],
            delta_net_vs_label=delta_net,
            cancel_after_sec=cand["horizon_sec"],
            max_hold_sec=1.5 * cand["horizon_sec"],
            exit_spread_charge_ticks=MARKET_EXIT_SPREAD_TICKS,
        )
        verdict, reasons = deploy_verdict(row)
        row["deploy_verdict"] = verdict
        row["reject_reasons"] = ";".join(reasons) if reasons else ""
        summary_rows.append(row)

        print(f"  [{cand['name']}] POOLED: signaled={total_signaled}  "
              f"filled={total_filled}  fill_rate={100*fill_rate_overall:.2f}%  "
              f"realized_net_t/trade={mean_realized:+.3f} "
              f"(label assumed {cand['label_assumption_net_t']:+.3f}, "
              f"delta={delta_net:+.3f})  Sharpe={sh:.2f}  "
              f"pdays={prof_days}/{total_oot_days}  verdict={verdict}")

    # --- Write outputs ------------------------------------------------------
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(OUT_DIR / "fifo_summary.csv", index=False)

    df_pd = pd.DataFrame(per_day_rows)
    df_pd.to_csv(OUT_DIR / "per_day_fifo.csv", index=False)

    df_pt = pd.DataFrame(per_trade_rows)
    df_pt.to_csv(OUT_DIR / "per_trade_diagnostics.csv", index=False)

    # --- REPORT.md (<=30 lines, lead with verdict) --------------------------
    lines = []
    lines.append("# Meta-Classifier v1 — FIFO Market-Replay Validation (HC #74)")
    lines.append("")
    lines.append("Tests label assumption (passive limit @ 0.376 t commission) against "
                 "realized FIFO fills. Exit = market at horizon (+1 t spread crossing).")
    lines.append("")
    for r in summary_rows:
        lines.append(f"## {r['candidate']} — VERDICT: {r['deploy_verdict']}")
        if r["reject_reasons"]:
            lines.append(f"_reject reasons: {r['reject_reasons']}_")
        lines.append(
            f"- Signaled: {r['total_signaled']:,} | Filled: {r['total_filled']:,} | "
            f"Fill rate: {100*r['fill_rate_overall']:.2f}%"
        )
        lines.append(
            f"- Realized net t/trade: {r['mean_net_t_realized']:+.3f} "
            f"(label assumed {r['label_assumption_net_t']:+.3f}, "
            f"delta {r['delta_net_vs_label']:+.3f})"
        )
        lines.append(
            f"- Sharpe pooled: {r['sharpe_pooled']:.2f} "
            f"(label assumed {r['label_assumption_sharpe']:.2f})  | "
            f"WR realized: {r['win_rate_realized']:.3f}"
        )
        lines.append(
            f"- Profitable days: {r['profitable_days']}/{r['total_days']} "
            f"(gate >= {GATE_PDAYS_ABS}) | days w/ any fill: "
            f"{r['days_with_any_fill']}/{r['total_days']}"
        )
        lines.append(
            f"- Regime imb: {r['regime_imbalance']:.3f} (gate <= {GATE_REGIME_IMB}) | "
            f"day_conc: {r['day_concentration']:.3f} (gate <= {GATE_DAYCONC})"
        )
        lines.append("")
    lines.append("## Notes")
    lines.append(
        "- Cancel window = horizon h; max_hold = 1.5 h; TP/SL set wide so all "
        "fills exit at max_hold (market) per HC #428 R2."
    )
    lines.append(
        "- Net per trade = pnl_ticks - 0.376 (entry commission, by engine) "
        "- 1.0 (market-exit spread, post-correction)."
    )
    lines.append(
        "- Per-fold OOT mapping mirrors meta_classifier_v1_walkforward "
        "(sliding window, HC #0)."
    )
    (OUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n")

    # --- regen_complete.json -----------------------------------------------
    elapsed = time.time() - t0
    finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    regen = dict(
        task="meta_classifier_v1_fifo_replay",
        hc_refs=["HC#74", "HC#428", "HC#420", "HC#485R5", "HC#0", "HC#393"],
        started_at=started_at, finished_at=finished_at,
        elapsed_seconds=round(elapsed, 1),
        candidates=[c["name"] for c in CANDIDATES],
        outputs=dict(
            summary=str(OUT_DIR / "fifo_summary.csv"),
            per_day=str(OUT_DIR / "per_day_fifo.csv"),
            per_trade=str(OUT_DIR / "per_trade_diagnostics.csv"),
            report=str(OUT_DIR / "REPORT.md"),
        ),
        verdicts={r["candidate"]: r["deploy_verdict"] for r in summary_rows},
        delta_net_vs_label={r["candidate"]: r["delta_net_vs_label"] for r in summary_rows},
        fill_rate={r["candidate"]: r["fill_rate_overall"] for r in summary_rows},
    )
    (OUT_DIR / ".regen_complete.json").write_text(json.dumps(regen, indent=2, default=str))

    print(f"\n[done] elapsed={elapsed:.1f}s  output={OUT_DIR}")


if __name__ == "__main__":
    main()
