"""
HC #409 D1 — Deep performance comparison: v3.3 (full NPZ) vs v3.4.2 (ep1 scalars only).

"Performance is more than just IC" — this script computes:
  - IC per-horizon (Pearson + Spearman; OVERALL + per-conf-tier)
  - Mean realized signed return per (horizon, side, conf_tier)
  - WR / hit-rate per (horizon, side, conf_tier)
  - %-above-passive-cost (0.376) and %-above-market-cost (1.376)
  - Cross-head correlation matrix (Spearman) to test "head collapse"
  - day_conc (HC #344) of the Top-1pct cells per horizon × side
  - Calibration: predicted log_ret_{h} vs realized, scatter bucket means

v3.4.2 column:
  We do NOT have per-sample v3.4.2 OOT predictions (the ep1 OOT eval logged IC
  scalars to MLflow but did not persist NPZ). Therefore v3.4.2 contributes only
  the scalar IC numbers known from the training-log line at 2026-05-17 08:13 ET:
    IC_1s = 0.2797
    IC_5s = 0.1153
    IC_10s= 0.0624
    IC_30s= 0.0183
    book_gate_tanh = -0.0027   (FROZEN at HC #407 pause time, batch 114k)
  Full v3.4.2 per-sample comparison is queued for post-resume inference.

Output: output/hc409_deep_perf_<ts>/
  - v33_full_perf_matrix.csv
  - v33_head_correlation.csv
  - v33_calibration_buckets.csv
  - v34_vs_v33_scalar_table.csv  (just the IC scalars we have)
  - SUMMARY.json

CPU-only, ~30s on 673k samples. Read-only on NPZ. Writes only to own dir.
NOT MALWARE.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scstats

PROJ = Path("/home/jupiter/Lvl3Quant")
NPZ = PROJ / "output" / "v3_3_extended_oot_20260514" / "extended_oot_predictions.npz"
HORIZONS = ["1s", "5s", "10s", "30s"]
SIDES = ["long", "short"]
TIERS = [("Top10", 0.10), ("Top5", 0.05), ("Top1", 0.01),
         ("Top0.5", 0.005), ("Top0.1", 0.001)]
COMMISSION = 0.376
SPREAD_CROSS = 1.0
OUT = PROJ / "output" / f"hc409_deep_perf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUT.mkdir(parents=True, exist_ok=True)

V34_SCALAR_KNOWN = {
    "IC_1s":  0.2797,
    "IC_5s":  0.1153,
    "IC_10s": 0.0624,
    "IC_30s": 0.0183,
    "book_gate_tanh": -0.0027,
    "fold": 0,
    "epoch": 1,
    "batch_at_pause": 114000,
    "batches_per_epoch": 265649,
    "TrLoss": 49.2070,
    "source_log_ts": "2026-05-17 08:13:31 ET",
}


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    t0 = time.time()
    _log(f"loading {NPZ}")
    d = np.load(NPZ, allow_pickle=False)
    n_samples = int(d["pred_log_ret_1s"].shape[0])
    oot_dates = list(d["oot_dates"])
    n_dates = len(oot_dates)
    samples_per_day = n_samples // n_dates
    date_idx = np.minimum(np.arange(n_samples) // samples_per_day, n_dates - 1)
    _log(f"n_samples={n_samples} n_dates={n_dates}")

    rows = []
    head_pred_arrays = {}
    head_target_arrays = {}

    for h in HORIZONS:
        pred = d[f"pred_log_ret_{h}"]
        target = d[f"target_log_ret_{h}"]
        ok = np.isfinite(pred) & np.isfinite(target)
        n_ok = int(ok.sum())
        pred_ok = pred[ok]
        target_ok = target[ok]
        date_ok = date_idx[ok]
        head_pred_arrays[h] = pred_ok
        head_target_arrays[h] = target_ok

        # Overall IC
        ic_p, _ = scstats.pearsonr(pred_ok, target_ok)
        ic_s, _ = scstats.spearmanr(pred_ok, target_ok)
        _log(f"horizon={h} n_ok={n_ok} ic_pearson={ic_p:.4f} ic_spearman={ic_s:.4f}")

        for side in SIDES:
            sign = 1.0 if side == "long" else -1.0
            signed_pred = sign * pred_ok

            for tier_name, tier_frac in TIERS:
                k = max(1, int(n_ok * tier_frac))
                if k >= len(signed_pred):
                    sel = np.arange(len(signed_pred))
                else:
                    sel = np.argpartition(-signed_pred, k - 1)[:k]
                n_cell = len(sel)
                if n_cell < 20:
                    continue

                pred_c = pred_ok[sel]
                target_c = target_ok[sel]
                realized = sign * target_c

                # IC within the tier (does high-conf accuracy beat overall?)
                if n_cell >= 30:
                    ic_p_tier, _ = scstats.pearsonr(pred_c * sign, realized)
                    ic_s_tier, _ = scstats.spearmanr(pred_c * sign, realized)
                else:
                    ic_p_tier = float("nan")
                    ic_s_tier = float("nan")

                gross_mean = float(realized.mean())
                gross_median = float(np.median(realized))
                wr = float((realized > 0).mean())
                hit_rate_passive = float((realized > COMMISSION).mean())
                hit_rate_market = float((realized > COMMISSION + SPREAD_CROSS).mean())
                net = realized - COMMISSION
                std_net = float(net.std(ddof=1)) if n_cell > 1 else float("nan")
                mean_net = float(net.mean())
                sharpe = float(mean_net / std_net) if std_net and std_net > 0 else float("nan")

                # day_conc
                dates_c = date_ok[sel]
                counts = np.bincount(dates_c, minlength=n_dates)
                day_conc = float(counts.max() / n_cell)

                rows.append(dict(
                    model="v3.3",
                    horizon=h, side=side, conf_tier=tier_name,
                    n_cell=n_cell,
                    overall_ic_pearson=round(ic_p, 4),
                    overall_ic_spearman=round(ic_s, 4),
                    tier_ic_pearson=round(ic_p_tier, 4),
                    tier_ic_spearman=round(ic_s_tier, 4),
                    gross_mean_tk=round(gross_mean, 3),
                    gross_median_tk=round(gross_median, 3),
                    wr=round(wr, 3),
                    hit_rate_above_passive=round(hit_rate_passive, 3),
                    hit_rate_above_market=round(hit_rate_market, 3),
                    mean_net_tk_per_fill=round(mean_net, 3),
                    sharpe_per_fill=round(sharpe, 3),
                    day_conc=round(day_conc, 3),
                ))

    df = pd.DataFrame(rows)
    matrix_path = OUT / "v33_full_perf_matrix.csv"
    df.to_csv(matrix_path, index=False)
    _log(f"perf matrix: {matrix_path} ({len(df)} cells)")

    # Cross-head correlation (do heads carry independent info?)
    head_corr_rows = []
    for h1 in HORIZONS:
        for h2 in HORIZONS:
            if h1 >= h2:
                # Use smaller intersection (different n_ok per head)
                continue
            n_min = min(head_pred_arrays[h1].shape[0], head_pred_arrays[h2].shape[0])
            a = head_pred_arrays[h1][:n_min]
            b = head_pred_arrays[h2][:n_min]
            rho, _ = scstats.spearmanr(a, b)
            head_corr_rows.append(dict(head_a=h1, head_b=h2, spearman=round(rho, 4)))
    pd.DataFrame(head_corr_rows).to_csv(OUT / "v33_head_correlation.csv", index=False)
    _log(f"head correlation matrix: {OUT/'v33_head_correlation.csv'}")

    # Calibration (decile of pred vs mean realized)
    calib_rows = []
    for h in HORIZONS:
        p = head_pred_arrays[h]
        t = head_target_arrays[h]
        edges = np.percentile(p, np.linspace(0, 100, 11))
        bucket = np.digitize(p, edges[1:-1])
        for b in range(10):
            m = bucket == b
            if m.sum() < 50:
                continue
            calib_rows.append(dict(
                horizon=h,
                pred_decile=b + 1,
                n=int(m.sum()),
                pred_mean=round(float(p[m].mean()), 4),
                realized_mean_tk=round(float(t[m].mean()), 4),
                realized_pct_pos=round(float((t[m] > 0).mean()), 4),
            ))
    pd.DataFrame(calib_rows).to_csv(OUT / "v33_calibration_buckets.csv", index=False)
    _log(f"calibration buckets: {OUT/'v33_calibration_buckets.csv'}")

    # v3.4.2 scalar comparison table
    v33_overall = {
        h: float(scstats.pearsonr(head_pred_arrays[h], head_target_arrays[h])[0])
        for h in HORIZONS
    }
    rows_scalar = []
    for h in HORIZONS:
        rows_scalar.append(dict(
            horizon=h,
            v33_ic_pearson_overall=round(v33_overall[h], 4),
            v342_ic_pearson_ep1=round(V34_SCALAR_KNOWN[f"IC_{h}"], 4),
            delta=round(V34_SCALAR_KNOWN[f"IC_{h}"] - v33_overall[h], 4),
        ))
    pd.DataFrame(rows_scalar).to_csv(OUT / "v34_vs_v33_scalar_table.csv", index=False)
    _log(f"v3.4.2 vs v3.3 IC scalar table: {OUT/'v34_vs_v33_scalar_table.csv'}")

    summary = dict(
        generated_at=datetime.now().isoformat(),
        npz_source=str(NPZ),
        n_samples=int(n_samples),
        n_dates=int(n_dates),
        cost_model=dict(commission=COMMISSION,
                        spread_cross_market_only=SPREAD_CROSS,
                        framing="HC #405 fill-price, no spread on passive"),
        v33_overall_ic_pearson=v33_overall,
        v342_known_scalars=V34_SCALAR_KNOWN,
        deltas=dict(
            ic_1s=round(V34_SCALAR_KNOWN["IC_1s"] - v33_overall["1s"], 4),
            ic_5s=round(V34_SCALAR_KNOWN["IC_5s"] - v33_overall["5s"], 4),
            ic_10s=round(V34_SCALAR_KNOWN["IC_10s"] - v33_overall["10s"], 4),
            ic_30s=round(V34_SCALAR_KNOWN["IC_30s"] - v33_overall["30s"], 4),
        ),
        elapsed_sec=round(time.time() - t0, 2),
        notes=(
            "v3.4.2 column is SCALAR-ONLY (IC at ep1 OOT) because per-sample "
            "predictions were not persisted by the ep1 eval. Full v3.4.2 "
            "MFE/WR/Sharpe/day_conc comparison requires post-resume inference. "
            "Queued under HC #409 D2."
        ),
    )
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    _log(f"SUMMARY: {OUT/'SUMMARY.json'}")
    _log(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
