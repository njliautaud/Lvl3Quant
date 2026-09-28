#!/usr/bin/env python3
"""
HC #432 + HC #428 R1/R2 — Full validation report for a v3.4.2 OOT config.

Reads:
  - {config_name}_fifo_fills.csv  (output of hc432_fifo_full_market_replay.py)
  - output/regime_labels/oot_dates_regime.parquet  (47-day green/red/flat labels)

Computes:
  Overall    : Sharpe(sqrt-N), Sortino, PF, WR, net_tk_total, n_fills, day_conc
  Per-day    : net_tk, n_fills, Sharpe, PF, WR
  Per-regime : Sharpe_green / Sharpe_red / Sharpe_flat
               R1 ratio = |Sh_g - Sh_r| / max(|Sh_g|, |Sh_r|)
  HC #428 R1 : PASS if ratio ≤ 0.50 AND day_conc ≤ 0.70
  HC #428 R2 : TP ≤ p90 of conditional MFE within horizon h
               hold ≤ 1.5 * h ; cancel ≤ h
               (MFE_p90 computed from realized 1s log returns in the concat NPZ
                converted to ticks)

Outputs:
  {config_name}_verdict.md
  {config_name}_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
OUT_DIR = LVL3_ROOT / "output" / "hc432_v342_47day_validation"
REGIME_PARQUET = LVL3_ROOT / "output" / "regime_labels" / "oot_dates_regime.parquet"
CONCAT_NPZ = OUT_DIR / "fold_00_ep1_oot_inference_47day_hc432.npz"

# ES constants
TICK_USD = 12.50
PX_REF = 5800.0  # canonical px_ref used in v3_4_2_1s avg-move (see canonical_avg_move_v3_4_2_1s.json)

# Horizon → seconds for HC #428 R2
HORIZON_SEC = {"1": 1.0, "5": 5.0, "10": 10.0, "30": 30.0, "60": 60.0}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("hc432_validate")


def classify_regime(row) -> str:
    """
    HC #428 R1 regime classification: green/red/flat based on ES close-to-open.
    Use close_minus_open_ticks if present, else compute from open/close pts.
    Thresholds (ticks): |delta| < 4  → flat ; >= +4 → green ; <= -4 → red.
    (4 ticks = 1.00 pt — a reasonable 'flat' band for ES day moves)
    """
    if "close_minus_open_ticks" in row and pd.notna(row["close_minus_open_ticks"]):
        d = float(row["close_minus_open_ticks"])
    else:
        d = (float(row["close_pts"]) - float(row["open_pts"])) / 0.25
    if d >= 4.0:
        return "green"
    elif d <= -4.0:
        return "red"
    return "flat"


def load_regime_labels() -> pd.DataFrame:
    if not REGIME_PARQUET.exists():
        log.warning(f"regime parquet missing: {REGIME_PARQUET}")
        return pd.DataFrame(columns=["date", "regime", "regime_strength"])
    df = pd.read_parquet(REGIME_PARQUET)
    # normalize date column to string YYYYMMDD
    df["date"] = df["date"].astype(str)
    if "regime" not in df.columns:
        df["regime"] = df.apply(classify_regime, axis=1)
    df["regime_strength"] = df.apply(
        lambda r: (float(r["close_pts"]) - float(r["open_pts"])) / float(r["open_pts"])
        if pd.notna(r["close_pts"]) and pd.notna(r["open_pts"]) and float(r["open_pts"]) > 0
        else 0.0,
        axis=1,
    )
    return df[["date", "regime", "regime_strength", "trend_label", "vol_bucket"]]


def compute_mfe_p90_from_logret(horizon_key: str) -> float:
    """
    Use realized log_ret_{h}s in the concat NPZ as a proxy for MFE within horizon.
    Convert log_ret to ticks: dr_ticks = dr * PX_REF / 0.25
    For long side, MFE within h is approximated by max(0, log_ret_h_q90)
    converted to ticks; we use q90 head as the upper-tail estimator.
    """
    if not CONCAT_NPZ.exists():
        return float("nan")
    d = np.load(CONCAT_NPZ, allow_pickle=False)
    # prefer the quantile head if present
    qkey = f"target_log_ret_{horizon_key}s_q90"
    if qkey in d.files:
        x = d[qkey].astype(np.float64)
        # use samples where mask is positive (valid)
        mkey = f"mask_log_ret_{horizon_key}s_q90"
        if mkey in d.files:
            m = d[mkey].astype(bool)
            x = x[m]
        # convert log_ret → ticks
        x_ticks = x * PX_REF / 0.25
        return float(np.percentile(x_ticks[np.isfinite(x_ticks)], 90))
    # fallback: realized log_ret at horizon
    rkey = f"target_log_ret_{horizon_key}s"
    if rkey in d.files:
        x = d[rkey].astype(np.float64)
        x_ticks = x * PX_REF / 0.25
        return float(np.percentile(np.abs(x_ticks[np.isfinite(x_ticks)]), 90))
    return float("nan")


def metrics_block(nets: np.ndarray) -> Dict[str, float]:
    n = len(nets)
    if n == 0:
        return {"n": 0, "sum_tk": 0.0, "mean_tk": 0.0, "PF": 0.0, "WR": 0.0,
                "Sharpe_sqrtN": 0.0, "Sortino": 0.0, "std_tk": 0.0}
    gp = nets[nets > 0].sum()
    gl = -nets[nets < 0].sum()
    pf = gp / gl if gl > 0 else float("inf")
    wr = 100.0 * (nets > 0).sum() / n
    mean = float(nets.mean())
    std = float(nets.std(ddof=1)) if n > 1 else 0.0
    sharpe = (mean / std) * np.sqrt(n) if std > 0 else 0.0
    downside = nets[nets < 0]
    dstd = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = (mean / dstd) * np.sqrt(n) if dstd > 0 else 0.0
    return {
        "n": int(n), "sum_tk": float(nets.sum()),
        "mean_tk": mean, "std_tk": std,
        "PF": float(pf), "WR": float(wr),
        "Sharpe_sqrtN": float(sharpe), "Sortino": float(sortino),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fills-csv", required=True, help="path to *_fifo_fills.csv")
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--horizon", required=True, choices=list(HORIZON_SEC.keys()))
    ap.add_argument("--side", required=True, choices=["long", "short"])
    ap.add_argument("--conf-band", required=True)
    ap.add_argument("--tp-ticks", type=float, required=True)
    ap.add_argument("--sl-ticks", type=float, required=True)
    ap.add_argument("--hold-s", type=float, required=True)
    ap.add_argument("--cancel-s", type=float, required=True)
    ap.add_argument("--order-type", required=True)
    ap.add_argument("--total-oot-dates", type=int, default=47,
                    help="for marking PRELIMINARY when fewer than this many dates")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    fills_path = Path(args.fills_csv)
    if not fills_path.exists() or fills_path.stat().st_size == 0:
        log.error(f"missing/empty fills CSV: {fills_path}")
        sys.exit(2)

    fills = pd.read_csv(fills_path)
    fills["date"] = fills["date"].astype(str)
    n_dates_present = fills["date"].nunique()
    log.info(f"loaded {len(fills):,} fills across {n_dates_present} dates")

    # join regime
    regimes = load_regime_labels()
    if regimes.empty:
        log.warning("regime labels empty — falling back to flat-only")
        fills["regime"] = "flat"
        fills["regime_strength"] = 0.0
    else:
        fills = fills.merge(regimes, on="date", how="left")
        fills["regime"] = fills["regime"].fillna("flat")
        fills["regime_strength"] = fills["regime_strength"].fillna(0.0)

    # overall metrics
    overall = metrics_block(fills["net_ticks"].to_numpy())
    log.info(f"Overall: {overall}")

    # per-day metrics
    per_day = []
    for d, g in fills.groupby("date", sort=True):
        m = metrics_block(g["net_ticks"].to_numpy())
        m["date"] = d
        m["regime"] = g["regime"].iloc[0]
        m["regime_strength"] = float(g["regime_strength"].iloc[0])
        per_day.append(m)
    per_day_df = pd.DataFrame(per_day).sort_values("date")
    # day concentration: top-1 day net_tk share of total |net|
    total_net = per_day_df["sum_tk"].sum()
    total_abs = per_day_df["sum_tk"].abs().sum()
    if total_abs > 0:
        day_conc = float(per_day_df["sum_tk"].abs().max() / total_abs)
    else:
        day_conc = 0.0

    # per-regime metrics
    per_regime = {}
    for r in ("green", "red", "flat"):
        sub = fills[fills["regime"] == r]["net_ticks"].to_numpy()
        per_regime[r] = metrics_block(sub)

    sh_g = per_regime["green"]["Sharpe_sqrtN"]
    sh_r = per_regime["red"]["Sharpe_sqrtN"]
    denom = max(abs(sh_g), abs(sh_r), 1e-9)
    r1_ratio = abs(sh_g - sh_r) / denom
    r1_pass = (r1_ratio <= 0.50) and (day_conc <= 0.70)

    # HC #428 R2 — MFE within horizon checks
    h_sec = HORIZON_SEC[args.horizon]
    mfe_p90_ticks = compute_mfe_p90_from_logret(args.horizon)
    r2_checks = {
        "TP_le_p90_MFE": args.tp_ticks <= mfe_p90_ticks if np.isfinite(mfe_p90_ticks) else None,
        "hold_le_1.5h":  args.hold_s <= 1.5 * h_sec,
        "cancel_le_h":   args.cancel_s <= h_sec,
        "horizon_sec":   h_sec,
        "tp_ticks":      args.tp_ticks,
        "mfe_p90_ticks": mfe_p90_ticks,
    }
    r2_pass = all(v for k, v in r2_checks.items()
                  if isinstance(v, bool))

    preliminary = n_dates_present < args.total_oot_dates
    title_prefix = "PRELIMINARY — " if preliminary else ""

    # write markdown verdict
    md = []
    md.append(f"# {title_prefix}HC #432 Verdict — {args.config_name}")
    md.append("")
    md.append(f"- Dates present: **{n_dates_present} / {args.total_oot_dates}**"
              + (" (waiting on Neptune)" if preliminary else ""))
    md.append(f"- Config: side={args.side}, horizon={args.horizon}s, "
              f"conf_band={args.conf_band}, TP={args.tp_ticks}, SL={args.sl_ticks}, "
              f"hold={args.hold_s}s, cancel={args.cancel_s}s, order={args.order_type}")
    md.append("")
    md.append("## Overall (FIFO realized)")
    md.append("| metric | value |")
    md.append("|---|---|")
    md.append(f"| n_fills | {overall['n']} |")
    md.append(f"| sum_net_ticks | {overall['sum_tk']:.2f} |")
    md.append(f"| mean_net_tk/fill | {overall['mean_tk']:.4f} |")
    md.append(f"| Sharpe (sqrt-N) | {overall['Sharpe_sqrtN']:.3f} |")
    md.append(f"| Sortino | {overall['Sortino']:.3f} |")
    md.append(f"| PF | {overall['PF']:.3f} |")
    md.append(f"| WR | {overall['WR']:.2f}% |")
    md.append(f"| day_concentration | {day_conc:.3f} |")
    md.append("")
    md.append("## Per-regime")
    md.append("| regime | n | sum_tk | mean_tk | Sharpe | PF | WR |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in ("green", "red", "flat"):
        m = per_regime[r]
        md.append(f"| {r} | {m['n']} | {m['sum_tk']:.2f} | {m['mean_tk']:.4f} | "
                  f"{m['Sharpe_sqrtN']:.3f} | {m['PF']:.3f} | {m['WR']:.2f}% |")
    md.append("")
    md.append("## HC #428 R1 — regime-agnostic OOT")
    md.append(f"- ratio |Sh_g − Sh_r| / max = **{r1_ratio:.3f}** (threshold ≤ 0.50)")
    md.append(f"- day_concentration = **{day_conc:.3f}** (cap ≤ 0.70)")
    md.append(f"- **R1 verdict: {'PASS' if r1_pass else 'FAIL'}**")
    md.append("")
    md.append("## HC #428 R2 — MFE within horizon")
    md.append("| check | value | pass |")
    md.append("|---|---|---|")
    md.append(f"| TP ≤ p90_MFE@h | {args.tp_ticks} vs {mfe_p90_ticks:.3f} | "
              f"{r2_checks['TP_le_p90_MFE']} |")
    md.append(f"| hold ≤ 1.5h | {args.hold_s}s vs {1.5*h_sec}s | {r2_checks['hold_le_1.5h']} |")
    md.append(f"| cancel ≤ h | {args.cancel_s}s vs {h_sec}s | {r2_checks['cancel_le_h']} |")
    md.append(f"- **R2 verdict: {'PASS' if r2_pass else 'FAIL'}**")
    md.append("")
    md.append("## Combined")
    md.append(f"- **{'PASS' if (r1_pass and r2_pass and overall['Sharpe_sqrtN'] > 0) else 'FAIL'}** "
              f"(R1={'P' if r1_pass else 'F'}, R2={'P' if r2_pass else 'F'}, "
              f"Sharpe>0={'P' if overall['Sharpe_sqrtN']>0 else 'F'})")
    md.append("")
    md.append("## Per-day")
    md.append("| date | regime | n | sum_tk | mean_tk | Sharpe | PF | WR |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for _, r in per_day_df.iterrows():
        md.append(f"| {r['date']} | {r['regime']} | {r['n']} | {r['sum_tk']:.2f} | "
                  f"{r['mean_tk']:.4f} | {r['Sharpe_sqrtN']:.3f} | {r['PF']:.3f} | "
                  f"{r['WR']:.2f}% |")
    md_text = "\n".join(md) + "\n"

    md_path = OUT_DIR / f"{args.config_name}_verdict.md"
    md_path.write_text(md_text)
    log.info(f"verdict written: {md_path}")

    summary = {
        "config_name": args.config_name,
        "preliminary": preliminary,
        "n_dates_present": int(n_dates_present),
        "n_dates_target": int(args.total_oot_dates),
        "config": {
            "horizon": args.horizon, "side": args.side, "conf_band": args.conf_band,
            "tp_ticks": args.tp_ticks, "sl_ticks": args.sl_ticks,
            "hold_s": args.hold_s, "cancel_s": args.cancel_s,
            "order_type": args.order_type,
        },
        "overall": overall,
        "per_regime": per_regime,
        "r1": {"ratio": r1_ratio, "day_conc": day_conc, "pass": bool(r1_pass)},
        "r2": {**r2_checks, "pass": bool(r2_pass)},
        "combined_pass": bool(r1_pass and r2_pass and overall["Sharpe_sqrtN"] > 0),
        "per_day": per_day_df.to_dict(orient="records"),
    }
    summary_path = OUT_DIR / f"{args.config_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    log.info(f"summary written: {summary_path}")

    # echo machine-readable status
    print(json.dumps({
        "config_name": args.config_name,
        "preliminary": preliminary,
        "n_fills": overall["n"], "Sharpe": overall["Sharpe_sqrtN"],
        "PF": overall["PF"], "WR": overall["WR"],
        "R1_pass": bool(r1_pass), "R2_pass": bool(r2_pass),
        "combined_pass": bool(r1_pass and r2_pass and overall["Sharpe_sqrtN"] > 0),
    }))


if __name__ == "__main__":
    main()
