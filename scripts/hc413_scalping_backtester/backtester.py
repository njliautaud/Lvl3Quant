#!/usr/bin/env python3
"""
HC #413 — TP/SL scalping backtester (Phase 1).

Takes:
  --npz <prediction NPZ>          v3.3 or v3.4.2 multi-head NPZ
  --mfe-config <CSV>              hc411 MFE-at-confidence matrix CSV
  --output-dir <dir>              where results CSV + verdict.md land
  --confidence-tier <name>        one of: top05, top1, top5, top10 (default: all)
  --horizon <h>                   1s/5s/10s/30s (default: all)
  --side <s>                      long/short (default: both)
  --model <name>                  filter MFE config rows by model col ("v3.3"/"v3.4.2"/"all")
  --order-type <t>                passive_at_touch (default) or market
  --cancel-eval-window <int>      default 40 (10 s @ 250 ms stride)
  --labels-dir <dir>              default data/processed/mbo_events_smart_v3_fifo_labels
  --seed <int>                    default 42

Per HC #69, primary metrics reported are risk-adjusted (Sharpe, Sortino,
PF, WR). Raw P&L is secondary. Per HC #344, day_conc is reported and flag
raised when >0.20. Per HC #408, n_fills>=50, CI_low_95>0, day_conc<=0.20
must all hold to pass the "honesty" gate. Per HC #397B, FIFO market replay
only — no midpoint shortcuts.

Per CLAUDE.md COST CONSTANTS: passive=0.376 ticks, market=1.376 ticks
(NEVER 2.0, 1.24, or any other default).

Cell ID = "{model}_{horizon}_{side}_{conf_tier}" e.g. "v3.3_5s_long_top1".

Designed for drop-in swap from v3.3 NPZ to v3.4.2 NPZ on 5/18 — no code
change required, just --npz pointing at the new file.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

LVL3 = Path("/home/jupiter/Lvl3Quant")
if str(LVL3) not in sys.path:
    sys.path.insert(0, str(LVL3))

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fill_sim import (  # noqa: E402
    FillSimConfig, entry_filled_mask, entry_cost_ticks,
    load_fifo_for_dates, ES_TICK_VALUE, DEFAULT_CANCEL_EVAL_WINDOW,
)
from tp_sl_rules import (  # noqa: E402
    thresholds_from_mfe_mae, resolve_exits, HORIZONS_ORDERED,
)
from metrics import summarize_cell  # noqa: E402

DEFAULT_LABELS_DIR = LVL3 / "data/processed/mbo_events_smart_v3_fifo_labels"
DEFAULT_MFE_CONFIG = (
    LVL3 / "output/hc411_regime_agnostic_20260517_215211/mfe_at_confidence_matrix.csv"
)
DEFAULT_NPZ_V33 = LVL3 / "output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz"

CONF_TIERS = ("top05", "top1", "top5", "top10")

OUTPUT_FIELDS = [
    "cell_id", "model", "horizon", "side", "conf_tier",
    "n_fills", "n_tp1_hits", "n_tp2_hits", "n_sl_hits", "n_time_stops",
    "gross_mfe_per_fill", "realized_net_per_fill",
    "realized_net_per_fill_dollars",
    "sharpe_sqrtN", "sortino_sqrtN", "pf", "wr",
    "day_conc", "ci_low_95_net", "ci_low_95_net_dollars",
    "pass_hc344", "pass_hc408_honesty",
    "mfe_source", "mae_source", "tp1", "tp2", "sl",
    "entry_cost_ticks", "order_type",
]


# -----------------------------------------------------------------------------
# Confidence-tier gating on the prediction signal
# -----------------------------------------------------------------------------
def confidence_mask(pred: np.ndarray, mask: np.ndarray, side: str,
                    conf_tier: str) -> np.ndarray:
    """Pick top-K% of |pred| on the chosen direction.

    For long: top-K% of pred (most positive).
    For short: top-K% of (-pred) (most negative).
    """
    pct = {"top05": 0.5, "top1": 1.0, "top5": 5.0, "top10": 10.0}[conf_tier]
    cutoff_q = 100.0 - pct
    signed = pred if side == "long" else -pred
    valid = mask & np.isfinite(signed)
    if not valid.any():
        return np.zeros_like(valid)
    thr = np.percentile(signed[valid], cutoff_q)
    return valid & (signed >= thr)


# -----------------------------------------------------------------------------
# Per-cell backtest
# -----------------------------------------------------------------------------
def backtest_cell(
    *,
    pred_arr: np.ndarray,
    pred_mask: np.ndarray,
    target_arrays: dict,         # horizon -> (N,) ticks
    target_masks: dict,          # horizon -> (N,) bool
    fifo: dict,
    side: str,
    horizon: str,                # entry horizon = the horizon the MFE was tuned for
    conf_tier: str,
    mfe: float,
    mae: float,
    order_type: str,
    cancel_eval_window: int,
    seed: int,
    model_label: str,
) -> dict:
    n_total = pred_arr.shape[0]
    # 1. Signal gate (top-K% confidence on side)
    sig_mask = confidence_mask(pred_arr, pred_mask, side, conf_tier)

    # 2. Restrict to indices present in FIFO labels (entry-level fill data)
    n_fifo = fifo["tp4sl3_long_filled"].shape[0]
    n_use = min(n_total, n_fifo)
    sig_mask = sig_mask[:n_use]
    n_signal = int(sig_mask.sum())

    # 3. Entry-fill simulation (FIFO replay — HC #397B)
    fill_cfg = FillSimConfig(order_type=order_type,
                             cancel_eval_window=cancel_eval_window,
                             side=side)
    entry_mask_all = entry_filled_mask(fifo, fill_cfg, seed=seed)
    entry_mask = sig_mask & entry_mask_all[:n_use]
    fill_idx = np.where(entry_mask)[0]
    n_attempted = n_signal
    n_filled = fill_idx.size
    if n_filled == 0:
        empty = {
            "cell_id": f"{model_label}_{horizon}_{side}_{conf_tier}",
            "model": model_label, "horizon": horizon, "side": side,
            "conf_tier": conf_tier,
            "n_fills": 0, "n_tp1_hits": 0, "n_tp2_hits": 0,
            "n_sl_hits": 0, "n_time_stops": 0,
            "gross_mfe_per_fill": 0.0, "realized_net_per_fill": 0.0,
            "realized_net_per_fill_dollars": 0.0,
            "sharpe_sqrtN": float("nan"), "sortino_sqrtN": float("nan"),
            "pf": float("nan"), "wr": float("nan"),
            "day_conc": float("nan"), "ci_low_95_net": float("nan"),
            "ci_low_95_net_dollars": float("nan"),
            "pass_hc344": False, "pass_hc408_honesty": False,
            "mfe_source": float(mfe), "mae_source": float(mae),
            "tp1": 0.5*max(mfe,0.0), "tp2": max(mfe,0.0),
            "sl": min(abs(mae), 1.5*max(mfe,0.0)),
            "entry_cost_ticks": entry_cost_ticks(order_type),
            "order_type": order_type,
        }
        return empty

    # 4. TP/SL exit resolution using realized target_log_ret_* (ticks)
    side_sign = +1.0 if side == "long" else -1.0
    inpos = {}
    mask_h = {}
    for h in HORIZONS_ORDERED:
        if h not in target_arrays:
            continue
        arr = target_arrays[h][:n_use][fill_idx]
        mk = target_masks[h][:n_use][fill_idx]
        inpos[h] = side_sign * arr
        mask_h[h] = mk
    thr = thresholds_from_mfe_mae(mfe, mae)
    gross, exit_code = resolve_exits(inpos, mask_h, thr)

    # 5. Apply round-trip cost (HC's canonical constants)
    cost = entry_cost_ticks(order_type)
    net = gross - cost

    # 6. Drop rows with no valid horizon (exit_code==0)
    valid = exit_code != 0
    if not valid.all():
        gross = gross[valid]
        net = net[valid]
        exit_code = exit_code[valid]
        fill_idx = fill_idx[valid]

    ts_ns = fifo["ts_ns"][fill_idx].astype(np.int64) if fill_idx.size else np.array([], dtype=np.int64)
    n_fills = net.size
    if n_fills == 0:
        return {
            "cell_id": f"{model_label}_{horizon}_{side}_{conf_tier}",
            "model": model_label, "horizon": horizon, "side": side,
            "conf_tier": conf_tier,
            "n_fills": 0, "n_tp1_hits": 0, "n_tp2_hits": 0,
            "n_sl_hits": 0, "n_time_stops": 0,
            "gross_mfe_per_fill": 0.0, "realized_net_per_fill": 0.0,
            "realized_net_per_fill_dollars": 0.0,
            "sharpe_sqrtN": float("nan"), "sortino_sqrtN": float("nan"),
            "pf": float("nan"), "wr": float("nan"),
            "day_conc": float("nan"), "ci_low_95_net": float("nan"),
            "ci_low_95_net_dollars": float("nan"),
            "pass_hc344": False, "pass_hc408_honesty": False,
            "mfe_source": float(mfe), "mae_source": float(mae),
            "tp1": thr.tp1, "tp2": thr.tp2, "sl": thr.sl,
            "entry_cost_ticks": cost, "order_type": order_type,
        }

    stats = summarize_cell(net, ts_ns)
    out = {
        "cell_id": f"{model_label}_{horizon}_{side}_{conf_tier}",
        "model": model_label, "horizon": horizon, "side": side,
        "conf_tier": conf_tier,
        "n_fills": n_fills,
        "n_tp1_hits": int((exit_code == 1).sum()),
        "n_tp2_hits": int((exit_code == 2).sum()),
        "n_sl_hits": int((exit_code == 3).sum()),
        "n_time_stops": int((exit_code == 4).sum()),
        "gross_mfe_per_fill": float(gross.mean()),
        "realized_net_per_fill": stats["realized_net_per_fill"],
        "realized_net_per_fill_dollars":
            stats["realized_net_per_fill"] * ES_TICK_VALUE,
        "sharpe_sqrtN": stats["sharpe_sqrtN"],
        "sortino_sqrtN": stats["sortino_sqrtN"],
        "pf": stats["pf"],
        "wr": stats["wr"],
        "day_conc": stats["day_conc"],
        "ci_low_95_net": stats["ci_low_95_net"],
        "ci_low_95_net_dollars": stats["ci_low_95_net"] * ES_TICK_VALUE
            if np.isfinite(stats["ci_low_95_net"]) else float("nan"),
        "pass_hc344": stats["pass_hc344"],
        "pass_hc408_honesty": stats["pass_hc408_honesty"],
        "mfe_source": thr.mfe_source, "mae_source": thr.mae_source,
        "tp1": thr.tp1, "tp2": thr.tp2, "sl": thr.sl,
        "entry_cost_ticks": cost, "order_type": order_type,
    }
    return out


# -----------------------------------------------------------------------------
# NPZ loader (multi-head v3.3 / v3.4.2 schema)
# -----------------------------------------------------------------------------
def load_predictions(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    n = int(d["n_samples"])
    oot_dates = [str(x) for x in d["oot_dates"]]
    preds, pred_masks = {}, {}
    targets, target_masks = {}, {}
    for h in HORIZONS_ORDERED:
        pk = f"pred_log_ret_{h}"
        if pk in d.keys():
            preds[h] = d[pk][:n].astype(np.float64)
            pred_masks[h] = (
                d[f"mask_log_ret_{h}"][:n].astype(bool) & np.isfinite(preds[h])
            )
        tk = f"target_log_ret_{h}"
        if tk in d.keys():
            targets[h] = d[tk][:n].astype(np.float64)
            target_masks[h] = (
                d[f"mask_log_ret_{h}"][:n].astype(bool) & np.isfinite(targets[h])
            )
    return {"n": n, "oot_dates": oot_dates, "preds": preds,
            "pred_masks": pred_masks, "targets": targets,
            "target_masks": target_masks}


# -----------------------------------------------------------------------------
# MFE config CSV
# -----------------------------------------------------------------------------
def load_mfe_config(csv_path: Path, model_filter: str | None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if model_filter and model_filter.lower() != "all":
        df = df[df["model"].astype(str) == model_filter].reset_index(drop=True)
    return df


def iter_cells(df: pd.DataFrame, conf_tiers, horizons, sides):
    for _, row in df.iterrows():
        if row["horizon"] not in horizons:
            continue
        if row["side"] not in sides:
            continue
        for tier in conf_tiers:
            mfe_col = f"mfe_{tier}"
            mae_col = f"mae_{tier}"
            n_col = f"n_{tier}"
            if mfe_col not in row.index or mae_col not in row.index:
                continue
            yield {
                "model": str(row["model"]),
                "horizon": str(row["horizon"]),
                "side": str(row["side"]),
                "conf_tier": tier,
                "mfe": float(row[mfe_col]),
                "mae": float(row[mae_col]),
                "n_config": int(row[n_col]) if n_col in row.index else -1,
            }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(DEFAULT_NPZ_V33))
    ap.add_argument("--mfe-config", default=str(DEFAULT_MFE_CONFIG))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--confidence-tier", default="all",
                    choices=("all",) + CONF_TIERS)
    ap.add_argument("--horizon", default="all",
                    choices=("all",) + HORIZONS_ORDERED)
    ap.add_argument("--side", default="both",
                    choices=("both", "long", "short"))
    ap.add_argument("--model", default="all",
                    help='Filter MFE config rows by model col (e.g. "v3.3" or "v3.4.2") or "all"')
    ap.add_argument("--order-type", default="passive_at_touch",
                    choices=("passive_at_touch", "market"))
    ap.add_argument("--cancel-eval-window", type=int,
                    default=DEFAULT_CANCEL_EVAL_WINDOW)
    ap.add_argument("--labels-dir", default=str(DEFAULT_LABELS_DIR))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[main] HC #413 scalping backtester start @ {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[main] args: {vars(args)}")

    # Load predictions
    pd_npz = load_predictions(Path(args.npz))
    print(f"[main] NPZ: n={pd_npz['n']} oot_dates={pd_npz['oot_dates']}")

    # Load FIFO labels for those dates
    fifo = load_fifo_for_dates(Path(args.labels_dir), pd_npz["oot_dates"])
    n_fifo = sum(fifo["_n_per_day"])
    print(f"[main] FIFO labels loaded: n_fifo={n_fifo}")

    # Load MFE config
    mfe_df = load_mfe_config(Path(args.mfe_config), args.model)
    print(f"[main] MFE config rows (post-model-filter): {len(mfe_df)}")

    horizons = HORIZONS_ORDERED if args.horizon == "all" else (args.horizon,)
    sides = ("long", "short") if args.side == "both" else (args.side,)
    tiers = CONF_TIERS if args.confidence_tier == "all" else (args.confidence_tier,)

    rows = []
    for cell in iter_cells(mfe_df, tiers, horizons, sides):
        h = cell["horizon"]
        if h not in pd_npz["preds"]:
            print(f"[skip] no pred for h={h}")
            continue
        pred = pd_npz["preds"][h]
        pmsk = pd_npz["pred_masks"][h]
        row = backtest_cell(
            pred_arr=pred, pred_mask=pmsk,
            target_arrays=pd_npz["targets"],
            target_masks=pd_npz["target_masks"],
            fifo=fifo,
            side=cell["side"], horizon=h, conf_tier=cell["conf_tier"],
            mfe=cell["mfe"], mae=cell["mae"],
            order_type=args.order_type,
            cancel_eval_window=args.cancel_eval_window,
            seed=args.seed,
            model_label=cell["model"],
        )
        rows.append(row)
        flag = "*" if row["pass_hc408_honesty"] else " "
        print(
            f"[cell]{flag} {row['cell_id']:<32} n_fills={row['n_fills']:>6} "
            f"net/fill={row['realized_net_per_fill']:+.3f}t "
            f"Shrp={row['sharpe_sqrtN']:.2f} PF={row['pf']:.2f} "
            f"WR={row['wr']:.1f}% dayC={row['day_conc']:.2f} "
            f"CI95lo={row['ci_low_95_net']:+.3f}t "
            f"TP1={row['tp1']:.2f}/TP2={row['tp2']:.2f}/SL={row['sl']:.2f}"
        )

    out_csv = out_dir / "scalping_backtest_results.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            for k in OUTPUT_FIELDS:
                r.setdefault(k, float("nan"))
            w.writerow(r)
    print(f"[main] wrote: {out_csv}")

    # Verdict markdown
    verdict_md = out_dir / "verdict.md"
    write_verdict(verdict_md, rows, args)
    print(f"[main] wrote: {verdict_md}")
    print(f"[main] DONE in {time.time()-t0:.1f}s")


def write_verdict(path: Path, rows: list[dict], args) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        path.write_text("# HC #413 verdict\n\nNo cells produced.\n")
        return
    pass_df = df[df["pass_hc408_honesty"] & (df["realized_net_per_fill"] > 0)].copy()
    pass_df = pass_df.sort_values("sharpe_sqrtN", ascending=False)
    top = df.sort_values("sharpe_sqrtN", ascending=False).head(10)

    lines = []
    lines.append("# HC #413 — TP/SL Scalping Backtest Verdict\n")
    lines.append(f"NPZ: `{args.npz}`")
    lines.append(f"MFE config: `{args.mfe_config}`")
    lines.append(f"Order type: `{args.order_type}` "
                 f"(entry cost = {entry_cost_ticks(args.order_type):.3f} ticks)")
    lines.append(f"Seed: {args.seed}")
    lines.append("")
    lines.append("## Cells passing HC #408 honesty gate AND realized net > 0\n")
    if pass_df.empty:
        lines.append("**NONE.** No (model × horizon × side × tier) cell cleared "
                     "n_fills>=50 AND CI_low_95>0 AND day_conc<=0.20 AND net>0.\n")
    else:
        lines.append("| cell_id | n_fills | net/fill (t) | Sharpe√N | PF | WR% | day_conc | CI95_lo |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in pass_df.iterrows():
            lines.append(
                f"| {r['cell_id']} | {int(r['n_fills'])} "
                f"| {r['realized_net_per_fill']:+.3f} "
                f"| {r['sharpe_sqrtN']:.2f} | {r['pf']:.2f} | {r['wr']:.1f} "
                f"| {r['day_conc']:.3f} | {r['ci_low_95_net']:+.3f} |"
            )
    lines.append("\n## Top 10 by Sharpe√N (whether passing or not)\n")
    lines.append("| cell_id | pass_408 | n_fills | net/fill (t) | Sharpe√N | PF | WR% | day_conc | CI95_lo |")
    lines.append("|---|:-:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in top.iterrows():
        lines.append(
            f"| {r['cell_id']} | {'Y' if r['pass_hc408_honesty'] else 'N'} | "
            f"{int(r['n_fills'])} | {r['realized_net_per_fill']:+.3f} | "
            f"{r['sharpe_sqrtN']:.2f} | {r['pf']:.2f} | {r['wr']:.1f} | "
            f"{r['day_conc']:.3f} | {r['ci_low_95_net']:+.3f} |"
        )
    lines.append("")
    lines.append("## HC compliance")
    lines.append("- HC #69: risk-adjusted metrics (Sharpe, Sortino, PF, WR) reported as primary.")
    lines.append("- HC #344: day_conc reported; flag when > 0.20.")
    lines.append("- HC #397B: canonical FIFO market replay (no midpoint shortcuts).")
    lines.append("- HC #408: pass_hc408_honesty requires n_fills>=50, CI_low_95>0, day_conc<=0.20.")
    lines.append("- HC #413 rule 3: TP1=0.5·MFE, TP2=1.0·MFE, SL=min(|MAE|,1.5·MFE).")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
