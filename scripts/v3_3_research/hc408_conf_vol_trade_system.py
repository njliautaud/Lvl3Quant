"""
HC #408 — Confidence × Vol stratified trade-system build (v3.3 track).

Goal:
  Per user 2026-05-17 ~16:05 ET: build a TRADE-SYSTEM CONFIG, not just analytics,
  by stratifying on confidence tier (Top1 / Top0.5 / Top0.1) × vol tertile
  (low / mid / high vol_30s) × horizon (1s, 5s, 10s, 30s) × side (long, short).
  For each cell, compute gross MFE, realized net (fill-price, HC #405 no
  double-spread), and apply honesty gate (HC #344 day_conc ≤ 0.20, n_fills ≥ 50,
  CI_low_95(net) > 0). Promote passing cells to a deployable JSON config.

Notes:
  - Pure analysis. Read-only on the NPZ. Writes only to its own output dir.
  - CPU-only, ~1 min wall on Jupiter for 673k samples.
  - HC #405: realized net uses target_log_ret_h as the fill-to-fill move in ticks
    minus 0.376 commission. No "+1 tick" added (mid→fill translation only for
    aggressive market-cross, which we report separately as %≥1.376 threshold).
  - HC #344: day_conc = max fraction of fills on any single OOT date.
  - This script does NOT touch existing scripts or training. It is additive.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("/home/jupiter/Lvl3Quant")
NPZ_PATH = PROJ / "output" / "v3_3_extended_oot_20260514" / "extended_oot_predictions.npz"

HORIZONS = ["1s", "5s", "10s", "30s"]
SIDES = ["long", "short"]
CONF_TIERS = [
    ("Top1",   0.01),
    ("Top0.5", 0.005),
    ("Top0.1", 0.001),
]
VOL_BUCKETS = ["vol_low", "vol_mid", "vol_high"]

COMMISSION = 0.376  # HC #392 round-trip commission
SPREAD_CROSS = 1.0  # only for market_cross comparison threshold (mid→fill)
HONESTY_N_FILLS_MIN = 50
HONESTY_DAY_CONC_MAX = 0.20

OUT_DIR = PROJ / "output" / f"hc408_conf_vol_trade_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _now():
    return datetime.now().strftime("%H:%M:%S")


def _log(msg):
    print(f"[{_now()}] {msg}", flush=True)


def wilson_lower(p, n, z=1.96):
    if n == 0:
        return 0.0
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / denom


def mean_ci_lower(arr, z=1.96):
    n = len(arr)
    if n < 2:
        return float("nan")
    se = arr.std(ddof=1) / np.sqrt(n)
    return float(arr.mean() - z * se)


def main():
    t0 = time.time()
    _log(f"loading NPZ: {NPZ_PATH}")
    d = np.load(NPZ_PATH, allow_pickle=False)

    # Sample-level prediction & target arrays
    preds = {h: d[f"pred_log_ret_{h}"] for h in HORIZONS}
    targets = {h: d[f"target_log_ret_{h}"] for h in HORIZONS}
    vol_pred = d["pred_pred_realized_vol_30s_ticks"]
    n_samples = preds["1s"].shape[0]
    _log(f"n_samples={n_samples}")

    # Per-sample date for day_conc — assume samples are time-ordered and chunked
    # by OOT date. Use oot_dates and split evenly (approximation; same approach
    # as canonical replay's day-bin attribution).
    oot_dates = list(d["oot_dates"])
    n_dates = len(oot_dates)
    samples_per_day = n_samples // n_dates
    date_idx = np.minimum(np.arange(n_samples) // samples_per_day, n_dates - 1)
    _log(f"n_dates={n_dates}, samples_per_day≈{samples_per_day}")

    # Vol tertile boundaries on the full population
    vol_finite = np.isfinite(vol_pred)
    vol_lo = np.nanpercentile(vol_pred[vol_finite], 33.3)
    vol_hi = np.nanpercentile(vol_pred[vol_finite], 66.7)
    vol_bucket = np.where(vol_pred <= vol_lo, "vol_low",
                  np.where(vol_pred <= vol_hi, "vol_mid", "vol_high"))
    _log(f"vol tertile bounds: lo={vol_lo:.3f} hi={vol_hi:.3f} ticks")

    rows = []

    for h in HORIZONS:
        pred = preds[h]
        target = targets[h]
        ok = np.isfinite(pred) & np.isfinite(target)
        n_ok = int(ok.sum())
        _log(f"horizon={h} n_ok={n_ok}")
        if n_ok == 0:
            continue

        for side in SIDES:
            sign = 1.0 if side == "long" else -1.0
            signed_pred = sign * pred  # higher = stronger conviction for THIS side

            for tier_name, tier_frac in CONF_TIERS:
                # Pick top tier_frac of the signed_pred distribution AMONG ok
                k = max(1, int(n_ok * tier_frac))
                # argpartition for top-k indices
                ok_idx = np.where(ok)[0]
                signed_ok = signed_pred[ok_idx]
                if k >= len(signed_ok):
                    top_idx = ok_idx
                else:
                    part = np.argpartition(-signed_ok, k - 1)[:k]
                    top_idx = ok_idx[part]
                # Threshold value (informational)
                top_thresh = float(signed_pred[top_idx].min())

                for vb in VOL_BUCKETS:
                    cell_idx = top_idx[vol_bucket[top_idx] == vb]
                    n_cell = len(cell_idx)
                    if n_cell == 0:
                        continue

                    # Realized signed return in ticks (target already in ticks)
                    realized_ticks = sign * target[cell_idx]
                    # Net per fill = realized − commission (HC #405: NO added
                    # spread for the realized fill-price framing)
                    net_per_fill = realized_ticks - COMMISSION

                    # Gross MFE proxy: realized at this horizon as a magnitude
                    # ceiling (the script does NOT compute true intra-horizon
                    # MFE — that requires the per-fill MFE walk; this is the
                    # horizon-endpoint realized, used as the "captured" measure)
                    gross_realized = realized_ticks
                    mean_gross = float(gross_realized.mean())
                    mean_net = float(net_per_fill.mean())
                    median_net = float(np.median(net_per_fill))
                    std_net = float(net_per_fill.std(ddof=1)) if n_cell > 1 else float("nan")
                    sharpe_per_fill = float(mean_net / std_net) if std_net and std_net > 0 else float("nan")
                    ci_low_net = mean_ci_lower(net_per_fill)
                    wr = float((realized_ticks > 0).mean())
                    pct_above_passive = float((gross_realized > COMMISSION).mean())
                    pct_above_market = float((gross_realized > (COMMISSION + SPREAD_CROSS)).mean())
                    wilson_passive = wilson_lower(pct_above_passive, n_cell)

                    # day_conc: largest fraction of fills on any single OOT date
                    dates_cell = date_idx[cell_idx]
                    if n_cell > 0:
                        # bincount over date_idx range
                        counts = np.bincount(dates_cell, minlength=n_dates)
                        day_conc = float(counts.max() / n_cell)
                        n_days_active = int((counts > 0).sum())
                    else:
                        day_conc = 1.0
                        n_days_active = 0

                    n_fills_per_day = n_cell / max(1, n_days_active)

                    # Honesty gate
                    passes_n = n_cell >= HONESTY_N_FILLS_MIN
                    passes_dc = day_conc <= HONESTY_DAY_CONC_MAX
                    passes_ci = ci_low_net > 0
                    promote = bool(passes_n and passes_dc and passes_ci)

                    rows.append(dict(
                        horizon=h,
                        side=side,
                        conf_tier=tier_name,
                        vol_bucket=vb,
                        n_fills=n_cell,
                        n_days_active=n_days_active,
                        fills_per_day=round(n_fills_per_day, 2),
                        signal_thresh_ticks=round(top_thresh, 4),
                        mean_gross_realized_tk=round(mean_gross, 3),
                        mean_net_tk_per_fill=round(mean_net, 3),
                        median_net_tk_per_fill=round(median_net, 3),
                        std_net_tk=round(std_net, 3),
                        sharpe_per_fill=round(sharpe_per_fill, 3),
                        ci_low_95_net=round(ci_low_net, 3),
                        wr=round(wr, 3),
                        pct_above_passive_cost=round(pct_above_passive, 3),
                        pct_above_market_cost=round(pct_above_market, 3),
                        wilson_low_pct_above_passive=round(wilson_passive, 3),
                        day_conc=round(day_conc, 3),
                        pass_n_fills=passes_n,
                        pass_day_conc=passes_dc,
                        pass_ci_low=passes_ci,
                        promote_to_trade_system=promote,
                    ))

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "conf_vol_matrix.csv"
    df.to_csv(csv_path, index=False)
    _log(f"matrix written: {csv_path} ({len(df)} cells)")

    # Promoted cells = trade system config
    promo = df[df["promote_to_trade_system"]].copy()
    promo = promo.sort_values(["sharpe_per_fill", "mean_net_tk_per_fill"], ascending=False)
    promo_path = OUT_DIR / "trade_system_candidates.csv"
    promo.to_csv(promo_path, index=False)
    _log(f"PROMOTED candidates: {len(promo)} cells → {promo_path}")

    # Top-10 trade system config JSON
    cfg = dict(
        generated_at=datetime.now().isoformat(),
        source_npz=str(NPZ_PATH),
        cost_model=dict(commission_per_fill_ticks=COMMISSION,
                        spread_cross_ticks_market_only=SPREAD_CROSS,
                        framing="HC #405 fill-price; no spread on passive/realized"),
        honesty_gate=dict(n_fills_min=HONESTY_N_FILLS_MIN,
                          day_conc_max=HONESTY_DAY_CONC_MAX,
                          ci_low_95_net_gt=0),
        n_cells_total=int(len(df)),
        n_cells_promoted=int(len(promo)),
        top_rules=[
            dict(
                rank=i + 1,
                horizon=r["horizon"],
                side=r["side"],
                conf_tier=r["conf_tier"],
                vol_bucket=r["vol_bucket"],
                signal_threshold_ticks=r["signal_thresh_ticks"],
                expected_net_tk_per_fill=r["mean_net_tk_per_fill"],
                ci_low_95_net=r["ci_low_95_net"],
                sharpe_per_fill=r["sharpe_per_fill"],
                fills_per_day=r["fills_per_day"],
                day_conc=r["day_conc"],
                wr=r["wr"],
                rule_description=(
                    f"IF horizon-{r['horizon']} signed pred ≥ {r['signal_thresh_ticks']:.4f} "
                    f"AND vol_30s in {r['vol_bucket']} → enter {r['side']} (passive_at_touch_+2), "
                    f"hold {r['horizon']}. Expected net {r['mean_net_tk_per_fill']:.3f} tk/fill "
                    f"(CI_low {r['ci_low_95_net']:.3f}). Fills/day≈{r['fills_per_day']}."
                ),
            )
            for i, (_, r) in enumerate(promo.head(10).iterrows())
        ],
    )
    cfg_path = OUT_DIR / "trade_system_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    _log(f"trade-system config: {cfg_path}")

    elapsed = time.time() - t0
    _log(f"DONE in {elapsed:.1f}s. Cells={len(df)} Promoted={len(promo)}")


if __name__ == "__main__":
    main()
