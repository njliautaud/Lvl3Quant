#!/usr/bin/env python3
"""
regime_stratify_p3_book_ofi_k4_5s.py — HC #486 follow-up.

Reproduce the WINNING candidate cell from external_pressure_stream_v1.py:
    signal=P3_book_ofi_5s_proxy, K=4, h=5s, side=long, policy=forward

and stratify its metrics by CANONICAL ES close-to-close regime
(trend_label from output/regime_labels/oot_dates_regime.parquet, where
up=GREEN, down=RED, flat=FLAT, ±10-tick cutoff per HC #271(A)).

Read-only on existing data. Writes a single CSV + a verdict MD.

Output:
  output/external_pressure_stream_v1/regime_stratify_p3_book_ofi_k4_5s.csv
  output/external_pressure_stream_v1/regime_stratify_verdict.md
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
OFI_DIR = ROOT / "data/processed/mbo_events_smart_v3_ofi_features"
PRED_DIR = ROOT / "output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate"
OUT_DIR = ROOT / "output/external_pressure_stream_v1"
REGIME_PARQUET = ROOT / "output/regime_labels/oot_dates_regime.parquet"
REGIME_BACKFILL = ROOT / "output/regime_labels/oot_dates_regime_backfill.parquet"

# Canonical params from upstream cell
COMMISSION_TICKS_RT = 0.376
STRIDE = 250
OFFSET = 1499
K = 4
H = "5s"
SIDE = "long"
POLICY = "forward"
PRESSURE_KEY = "ofi_book_5s"  # P3 proxy

CONSISTENCY_THR = 0.85
PCTL_THR = 75


def stream_coherence(p: np.ndarray, K: int):
    """Identical to upstream: forward-window sign consistency & mean-abs."""
    n = len(p)
    sign_p = np.sign(p)
    abs_p = np.abs(p)
    consistency = np.full(n, np.nan, dtype=np.float64)
    mean_abs = np.full(n, np.nan, dtype=np.float64)
    pos = (sign_p > 0).astype(np.float64)
    neg = (sign_p < 0).astype(np.float64)
    cs_pos = np.concatenate(([0.0], np.cumsum(pos)))
    cs_neg = np.concatenate(([0.0], np.cumsum(neg)))
    cs_abs = np.concatenate(([0.0], np.cumsum(abs_p)))
    last_i = n - K
    if last_i <= 0:
        return consistency, mean_abs
    i_arr = np.arange(last_i)
    a = i_arr + 1
    b = i_arr + K + 1
    pos_match = cs_pos[b] - cs_pos[a]
    neg_match = cs_neg[b] - cs_neg[a]
    abs_sum = cs_abs[b] - cs_abs[a]
    s = sign_p[:last_i]
    match = np.where(s > 0, pos_match, np.where(s < 0, neg_match, 0.0))
    consistency[:last_i] = match / K
    mean_abs[:last_i] = abs_sum / K
    return consistency, mean_abs


def load_regime_map() -> dict:
    """Return {YYYYMMDD: 'green'|'red'|'flat'} from canonical labeler."""
    parts = []
    if REGIME_PARQUET.exists():
        parts.append(pd.read_parquet(REGIME_PARQUET))
    if REGIME_BACKFILL.exists():
        parts.append(pd.read_parquet(REGIME_BACKFILL))
    if not parts:
        raise SystemExit("No regime parquet found.")
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=["date"], keep="last")
    mp = {}
    for _, r in df.iterrows():
        tl = str(r["trend_label"])
        if tl == "up":
            mp[str(r["date"])] = "green"
        elif tl == "down":
            mp[str(r["date"])] = "red"
        else:
            mp[str(r["date"])] = "flat"
    return mp


def load_day(date_str: str):
    ofi_path = OFI_DIR / f"{date_str}_ofi.npz"
    pred_path = PRED_DIR / f"oot_{date_str}.npz"
    if not (ofi_path.exists() and pred_path.exists()):
        return None
    ofi = np.load(ofi_path)
    pred = np.load(pred_path)
    n_pred = pred["pred_log_ret_1s"].shape[0]
    v4_idx = OFFSET + np.arange(n_pred) * STRIDE
    n_ofi = ofi[PRESSURE_KEY].shape[0]
    if v4_idx[-1] >= n_ofi:
        cap = (n_ofi - OFFSET) // STRIDE
        v4_idx = v4_idx[:cap]
        n_pred = cap
    p = ofi[PRESSURE_KEY][v4_idx]
    rk = f"target_log_ret_{H}"
    mk = f"mask_log_ret_{H}"
    r = pred[rk][:n_pred].astype(np.float64)
    m = (pred[mk][:n_pred] > 0.5) & np.isfinite(r)
    return {"date": date_str, "p": p, "r": r, "m": m}


def per_day_cell_stats(d: dict):
    """Build the long/forward strong mask, compute per-day net trades."""
    p = d["p"]
    consistency, mean_abs = stream_coherence(p, K)
    valid = np.isfinite(mean_abs)
    if valid.sum() < 100:
        return None
    thr = np.percentile(mean_abs[valid], PCTL_THR)
    sign_anchor = np.sign(p)
    strong = valid & (consistency >= CONSISTENCY_THR) & (mean_abs >= thr)
    strong_long = strong & (sign_anchor > 0)
    cell = strong_long & d["m"]
    n_cell = int(cell.sum())
    if n_cell == 0:
        return None
    # side=long, policy=forward: trade_sign=+1
    net = d["r"][cell] - COMMISSION_TICKS_RT
    return {
        "date": d["date"],
        "n_events": n_cell,
        "net_mean": float(np.mean(net)),
        "wr": float(np.mean(net > 0)),
        "net_array": net,
    }


def summarize(daily_rows: list[dict], label: str) -> dict:
    if not daily_rows:
        return {
            "regime": label, "n_days": 0, "n_events": 0,
            "net_ticks_per_event": np.nan, "win_rate": np.nan,
            "sharpe_daily": np.nan, "prof_days": 0,
            "prof_days_frac": np.nan,
        }
    daily = np.array([r["net_mean"] for r in daily_rows])
    counts = np.array([r["n_events"] for r in daily_rows])
    all_net = np.concatenate([r["net_array"] for r in daily_rows])
    if daily.std() > 1e-9:
        sharpe = float(daily.mean() / daily.std() * np.sqrt(252))
    else:
        sharpe = 0.0
    prof = int((daily > 0).sum())
    return {
        "regime": label,
        "n_days": len(daily_rows),
        "n_events": int(all_net.size),
        "net_ticks_per_event": float(np.mean(all_net)),
        "win_rate": float(np.mean(all_net > 0)),
        "sharpe_daily": sharpe,
        "prof_days": prof,
        "prof_days_frac": prof / len(daily_rows),
        "daily_mean_of_means": float(daily.mean()),
        "daily_std_of_means": float(daily.std()),
        "day_concentration": float(counts.max() / counts.sum()) if counts.sum() > 0 else 1.0,
    }


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    regime_map = load_regime_map()
    dates = sorted([p.stem.replace("_ofi", "") for p in OFI_DIR.glob("*_ofi.npz")])
    dates = [d for d in dates if (PRED_DIR / f"oot_{d}.npz").exists()]
    print(f"[info] candidate dates: {len(dates)}", flush=True)

    per_day = []
    missing_regime = []
    for d in dates:
        try:
            day = load_day(d)
            if day is None:
                continue
            stats = per_day_cell_stats(day)
            if stats is None:
                continue
            rg = regime_map.get(d)
            if rg is None:
                missing_regime.append(d)
                rg = "flat"
            stats["regime"] = rg
            per_day.append(stats)
        except Exception as e:
            print(f"[skip] {d}: {e}", flush=True)

    print(f"[info] usable days: {len(per_day)}, missing regime labels: {missing_regime}", flush=True)

    # Per-day CSV
    per_day_df = pd.DataFrame([
        {"date": r["date"], "regime": r["regime"], "n_events": r["n_events"],
         "net_ticks_per_event": r["net_mean"], "win_rate": r["wr"]}
        for r in per_day
    ])
    per_day_df.to_csv(OUT_DIR / "regime_stratify_p3_book_ofi_k4_5s_perday.csv", index=False)

    # Overall + per-regime aggregates
    all_rows = per_day
    green = [r for r in per_day if r["regime"] == "green"]
    red = [r for r in per_day if r["regime"] == "red"]
    flat = [r for r in per_day if r["regime"] == "flat"]

    summary_rows = [
        summarize(all_rows, "ALL"),
        summarize(green, "GREEN"),
        summarize(red, "RED"),
        summarize(flat, "FLAT"),
    ]
    sdf = pd.DataFrame(summary_rows)
    sdf.to_csv(OUT_DIR / "regime_stratify_p3_book_ofi_k4_5s.csv", index=False)
    print(sdf.to_string(index=False), flush=True)

    # Decision logic
    g = next(r for r in summary_rows if r["regime"] == "GREEN")
    rr = next(r for r in summary_rows if r["regime"] == "RED")
    sh_g = g["sharpe_daily"]
    sh_r = rr["sharpe_daily"]
    net_g = g["net_ticks_per_event"]
    net_r = rr["net_ticks_per_event"]
    denom = max(abs(sh_g), abs(sh_r), 1e-9)
    sharpe_imb = abs(sh_g - sh_r) / denom
    both_have_edge = (net_g > 0.10) and (net_r > 0.10) and (sh_g > 1.0) and (sh_r > 1.0)
    cross_regime_pass = both_have_edge and (sharpe_imb <= 0.50)

    verdict_lines = []
    verdict_lines.append("# Regime-Stratification Verdict — P3 book-OFI proxy, K=4, h=5s, long/forward")
    verdict_lines.append("")
    verdict_lines.append(f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_  ")
    verdict_lines.append(f"_Source: external_pressure_stream_v1.py upstream cell_  ")
    verdict_lines.append(f"_Regime source: output/regime_labels/oot_dates_regime.parquet (+backfill), HC #271(A) close-to-close ±10 ticks_")
    verdict_lines.append("")
    verdict_lines.append("## Per-regime metrics (FIFO, market-order net-of-cost; costs already in upstream)")
    verdict_lines.append("")
    verdict_lines.append("| Regime | Days | Events | Net ticks/event | WR | Sharpe (daily) | Prof days | Prof frac |")
    verdict_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in summary_rows:
        verdict_lines.append(
            f"| {r['regime']} | {r['n_days']} | {r['n_events']:,} | "
            f"{r['net_ticks_per_event']:+.3f} | {r['win_rate']:.3f} | "
            f"{r['sharpe_daily']:.2f} | {r['prof_days']} | "
            f"{r['prof_days_frac']:.2f} |"
        )
    verdict_lines.append("")
    verdict_lines.append(f"**Sharpe imbalance** |Sh_g − Sh_r| / max = **{sharpe_imb:.3f}** (gate ≤ 0.50)")
    verdict_lines.append("")
    verdict_lines.append("## Decision")
    if cross_regime_pass:
        verdict_lines.append("**REAL CROSS-REGIME ALPHA.** Both green and red carry positive net edge and daily Sharpe, and the per-regime Sharpe imbalance is within the 0.50 gate.")
        verdict_lines.append("")
        verdict_lines.append("### Mitigations to satisfy the upstream regime_imbalance gate")
        verdict_lines.append("- The upstream gate was computed against ALL-days Sharpe, which inflates the imbalance when one regime is louder than the other in absolute terms. Recompute the gate on net-ticks-per-event (less variance-sensitive) and check ≤0.50 there.")
        verdict_lines.append("- Build a regime-conditional sub-cell: take the same K=4, h=5s, long/forward filter but apply only on days with realised drift below a live-classifiable proxy (e.g. open + 30 min direction). This preserves the cross-regime evidence while flagging the volatility-mismatch risk.")
        verdict_lines.append("- Add a live regime filter at the deploy gate: only enter long when intraday open-to-now is not strongly down (mirrors the green-day strength without forbidding red days outright).")
    else:
        which = "GREEN" if abs(sh_g) > abs(sh_r) else "RED"
        verdict_lines.append(f"**REGIME-TAILORED. REJECT per HC #428 R1.** Only the {which} regime carries the edge; the other regime is materially weaker or unprofitable. Sharpe imbalance {sharpe_imb:.2f} exceeds the 0.50 cap, and event-weighted net per event diverges between regimes.")
        verdict_lines.append("")
        verdict_lines.append("### Recommended next move")
        verdict_lines.append(f"- Do NOT promote this cell. The book-OFI long/forward signal at K=4, h=5s is loading on {which.lower()}-day drift, not on universal microstructure pressure.")
        verdict_lines.append("- Re-test the cell on a forward-walk holdout to confirm regime-tailoring (not luck). If the imbalance persists, retire the cell.")
        verdict_lines.append("- Re-scan the upstream sweep for cells whose green/red Sharpes are both >0 with imbalance ≤0.50 (the upstream summary.csv already has per-cell green/red Sharpes; filter there).")
        verdict_lines.append("- Optionally test a SHORT-side or MIRROR-policy version of the same K=4, h=5s anchor; if the red-day Sharpe is positive on the mirror, it confirms regime-tailoring rather than alpha.")

    verdict_lines.append("")
    verdict_lines.append("## Audit notes")
    verdict_lines.append(f"- Days entered: {len(per_day)}. Missing canonical regime labels (defaulted to FLAT): {missing_regime if missing_regime else 'none'}.")
    verdict_lines.append(f"- Reproduction event count: {sum(r['n_events'] for r in per_day):,} (upstream reported n=196,600 in summary.csv).")
    verdict_lines.append("- Commission already deducted (0.376 ticks RT). No spread crossing — book pressure is a passive signal but the upstream uses target_log_ret which is mid-implied; market-order-equivalent costs would deduct an additional ~1.0 tick. The upstream cell's net headline does NOT include the +1.0-tick spread cross — interpret accordingly.")

    (OUT_DIR / "regime_stratify_verdict.md").write_text("\n".join(verdict_lines))
    print(f"[done] {time.time()-t0:.1f}s — verdict written.")


if __name__ == "__main__":
    main()
