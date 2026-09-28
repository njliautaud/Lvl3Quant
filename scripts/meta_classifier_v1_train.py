#!/usr/bin/env python3
"""
meta_classifier_v1_train.py — FIRST execution of HC #486 R4 meta-layer architecture.

Train + evaluate 8 LightGBM binary classifiers (long/short × {1s,5s,10s,30s})
on the meta_layer_v1 dataset, with walk-forward IS/OOT split, threshold sweep,
and HC #428 deploy-gate evaluation.

CPU-only. Dataset features are causal stream + raw-market features (20-dim).
Labels are pre-computed net ticks at passive limit (commission already baked in).

Author: Claude (Head of Quant), 2026-05-22.
HC refs: HC#486R4, HC#428, HC#0, HC#69, HC#485R5, HC#420, HC#393.
"""
from __future__ import annotations

import os
import sys
import json
import time
import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import lightgbm as lgb
except Exception as e:
    print(f"[fatal] lightgbm import failed: {e}", file=sys.stderr)
    sys.exit(2)

# --------- Config -----------------------------------------------------------
ROOT = Path("/home/jupiter/Lvl3Quant")
DATA_DIR = ROOT / "data/processed/meta_layer_v1"
OUT_DIR = ROOT / "output/meta_classifier_v1"
MODELS_DIR = OUT_DIR / "models"

HORIZONS = ["1s", "5s", "10s", "30s"]
SIDES = ["long", "short"]
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

# Walk-forward split — first 20 IS, last 12 OOT (~62/38) per task spec
N_IS = 20
N_OOT = 12
INNER_VAL_FRAC = 0.20  # last 20% of IS days for early-stopping

# LightGBM defaults
LGB_PARAMS = dict(
    objective="binary",
    metric="binary_logloss",
    num_leaves=63,
    learning_rate=0.05,
    min_data_in_leaf=200,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=5,
    verbose=-1,
    seed=42,
    n_jobs=-1,
)
N_ESTIMATORS = 500
EARLY_STOPPING = 30

# HC #428 deploy gates
GATE_NET = 0.10
GATE_SHARPE = 0.3
GATE_PDAYS_FRAC = 0.65
GATE_REGIME_IMB = 0.50
GATE_DAYCONC = 0.70

# Regime classification proxy (matches stream_stability_v2.py:
# sign of OOT day's mean realized 30s — but here we don't have raw target_log_ret_30s.
# Use the day's y_long_30s_net mean: positive => green (long-favorable),
# negative => red, near-zero => flat. Threshold: ±0.10 ticks.
REGIME_THRESH = 0.10

RTH_SECONDS = 6.5 * 3600.0  # for Sharpe annualization (per-day mean -> annualized)
ANNUAL_TRADING_DAYS = 252.0


# --------- Helpers ----------------------------------------------------------
def sharpe_per_day(day_means: np.ndarray) -> float:
    """Annualized Sharpe from per-day mean net-ticks (252 trading days)."""
    d = day_means[np.isfinite(day_means)]
    if d.size < 2:
        return float("nan")
    mu = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    if sd <= 1e-12:
        return float("nan")
    return mu / sd * math.sqrt(ANNUAL_TRADING_DAYS)


def load_day(npz_path: Path) -> Dict:
    d = np.load(npz_path, allow_pickle=True)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.float32)
    feat_names = [str(s) for s in d["feat_names"]]
    label_names = [str(s) for s in d["label_names"]]
    date = str(d["date"][0]) if d["date"].shape[0] > 0 else npz_path.stem.replace("_meta", "")
    return dict(
        date=date,
        X=X,
        y=y,
        feat_names=feat_names,
        label_names=label_names,
    )


def clean_xy(X: np.ndarray, y_col: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Drop rows with NaN in label or any feature."""
    finite_y = np.isfinite(y_col)
    finite_x = np.all(np.isfinite(X), axis=1)
    m = finite_y & finite_x
    return X[m], y_col[m], m


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


# --------- Main -------------------------------------------------------------
def main():
    t0 = time.time()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[start] {started_at}", flush=True)

    # ----- Load manifest + days -----
    manifest_path = DATA_DIR / "meta_layer_v1_manifest.json"
    if not manifest_path.exists():
        print(f"[fatal] manifest missing: {manifest_path}", file=sys.stderr); sys.exit(2)
    manifest = json.loads(manifest_path.read_text())
    feat_names = list(manifest["feat_names"])
    label_names = list(manifest["label_names"])
    print(f"[manifest] {len(feat_names)} features, {len(label_names)} labels", flush=True)

    files = sorted(DATA_DIR.glob("*_meta.npz"))
    print(f"[files] {len(files)} day-NPZs found", flush=True)
    if len(files) < N_IS + N_OOT:
        print(f"[warn] only {len(files)} days, expected >= {N_IS + N_OOT}", flush=True)

    days = []
    for f in files:
        try:
            d = load_day(f)
        except Exception as e:
            print(f"  corrupt {f.name}: {e}", flush=True); continue
        days.append(d)
    days.sort(key=lambda x: x["date"])
    print(f"[loaded] {len(days)} usable days", flush=True)
    if len(days) < N_IS + 1:
        print("[fatal] not enough days", file=sys.stderr); sys.exit(2)

    # Adjust split if fewer files
    n_total = len(days)
    n_is = min(N_IS, max(1, int(round(n_total * 0.625))))
    n_oot = n_total - n_is
    is_days = days[:n_is]
    oot_days = days[n_is:]
    print(f"[split] IS={n_is} ({is_days[0]['date']}..{is_days[-1]['date']})  "
          f"OOT={n_oot} ({oot_days[0]['date']}..{oot_days[-1]['date']})", flush=True)

    # ----- Build IS / inner-val / OOT matrices per label -----
    # IS-concat features & per-label labels
    inner_val_cut = int(round(n_is * (1 - INNER_VAL_FRAC)))
    inner_val_cut = max(1, min(n_is - 1, inner_val_cut))
    train_days = is_days[:inner_val_cut]
    val_days = is_days[inner_val_cut:]
    print(f"[inner] train={len(train_days)} val={len(val_days)}", flush=True)

    def stack(days_list: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = np.concatenate([d["X"] for d in days_list], axis=0)
        y = np.concatenate([d["y"] for d in days_list], axis=0)
        # day labels per row (date)
        date_per_row = np.concatenate(
            [np.full(d["X"].shape[0], d["date"], dtype=object) for d in days_list]
        )
        return X, y, date_per_row

    X_tr, y_tr_all, _ = stack(train_days)
    X_va, y_va_all, _ = stack(val_days)
    X_oot, y_oot_all, dates_oot = stack(oot_days)
    print(f"[shapes] X_tr={X_tr.shape}  X_va={X_va.shape}  X_oot={X_oot.shape}", flush=True)

    # ----- Train 8 models -----
    label_idx = {n: i for i, n in enumerate(label_names)}
    fi_records: List[Dict] = []
    summary_rows: List[Dict] = []
    per_day_records: List[Dict] = []
    closest_miss = {}  # (side, h) -> best closest-miss row

    # Pre-compute OOT per-day net columns we'll need
    # Index in y for y_<side>_<h>_net
    net_col = {}
    win_col = {}
    for side in SIDES:
        for h in HORIZONS:
            net_col[(side, h)] = label_idx[f"y_{side}_{h}_net"]
            win_col[(side, h)] = label_idx[f"y_{side}_{h}_winner"]

    # Per-day OOT regime — use long_30s_net per day
    day_regime = {}
    for d in oot_days:
        col = d["y"][:, label_idx["y_long_30s_net"]]
        day_regime[d["date"]] = regime_of_day(col)
    n_days_oot = len(oot_days)
    pdays_threshold = int(math.ceil(GATE_PDAYS_FRAC * n_days_oot))
    print(f"[gate] pdays >= {pdays_threshold}/{n_days_oot}", flush=True)

    for side in SIDES:
        for h in HORIZONS:
            t_m = time.time()
            tag = f"{side}_{h}"
            wcol = win_col[(side, h)]
            ncol = net_col[(side, h)]

            # Build clean tr/va sets
            y_tr_w = y_tr_all[:, wcol]
            y_va_w = y_va_all[:, wcol]
            y_oot_w = y_oot_all[:, wcol]
            y_oot_n = y_oot_all[:, ncol]

            X_tr_c, y_tr_c, _ = clean_xy(X_tr, y_tr_w)
            X_va_c, y_va_c, _ = clean_xy(X_va, y_va_w)

            print(f"[fit] {tag}  tr={X_tr_c.shape[0]}  va={X_va_c.shape[0]}  "
                  f"pos_rate_tr={float(np.mean(y_tr_c)):.3f}", flush=True)

            dtr = lgb.Dataset(X_tr_c, label=y_tr_c.astype(np.int32), feature_name=feat_names)
            dva = lgb.Dataset(X_va_c, label=y_va_c.astype(np.int32), feature_name=feat_names, reference=dtr)
            booster = lgb.train(
                LGB_PARAMS,
                dtr,
                num_boost_round=N_ESTIMATORS,
                valid_sets=[dva],
                valid_names=["val"],
                callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                           lgb.log_evaluation(period=0)],
            )
            best_iter = booster.best_iteration or N_ESTIMATORS
            print(f"  best_iter={best_iter}", flush=True)

            # Save model
            mpath = MODELS_DIR / f"{tag}.txt"
            booster.save_model(str(mpath), num_iteration=best_iter)

            # Feature importance (gain)
            gains = booster.feature_importance(importance_type="gain")
            for fn, g in zip(feat_names, gains):
                fi_records.append(dict(model=tag, feature=fn, gain=float(g)))

            # OOT predict (drop rows where y or X non-finite)
            finite_x = np.all(np.isfinite(X_oot), axis=1)
            finite_yw = np.isfinite(y_oot_w)
            finite_yn = np.isfinite(y_oot_n)
            m_oot = finite_x & finite_yw & finite_yn
            X_oot_c = X_oot[m_oot]
            y_oot_w_c = y_oot_w[m_oot]
            y_oot_n_c = y_oot_n[m_oot]
            dates_oot_c = dates_oot[m_oot]

            proba = booster.predict(X_oot_c, num_iteration=best_iter)

            # Threshold sweep
            best_in_pair = None
            for thr in THRESHOLDS:
                sel = proba >= thr
                n_tr = int(sel.sum())
                if n_tr < 30:
                    summary_rows.append(dict(
                        side=side, horizon=h, threshold=thr, n_trades=n_tr,
                        net_ticks_per_trade=np.nan, win_rate=np.nan, sharpe=np.nan,
                        profitable_days=0, total_days=n_days_oot,
                        sharpe_green=np.nan, sharpe_red=np.nan, regime_imbalance=np.nan,
                        day_concentration=np.nan,
                        gate_net=False, gate_sharpe=False, gate_pdays=False,
                        gate_regime=False, gate_dayconc=False,
                        n_gates_passed=0, pass_gates=False, best_iter=int(best_iter),
                    ))
                    continue
                net_sel = y_oot_n_c[sel]
                day_sel = dates_oot_c[sel]
                mean_net = float(np.mean(net_sel))
                wr = float(np.mean(net_sel > 0))

                # Per-day stats
                uniq = sorted(set(d["date"] for d in oot_days))
                day_means = []
                day_sums = []
                day_regs = []
                for ud in uniq:
                    mm = day_sel == ud
                    if mm.sum() == 0:
                        continue
                    day_means.append(float(np.mean(net_sel[mm])))
                    day_sums.append(float(np.sum(net_sel[mm])))
                    day_regs.append(day_regime.get(ud, "flat"))
                day_means_arr = np.array(day_means) if day_means else np.array([np.nan])
                day_sums_arr = np.array(day_sums) if day_sums else np.array([0.0])
                day_regs_arr = np.array(day_regs) if day_regs else np.array(["flat"])

                sh = sharpe_per_day(day_means_arr)
                sh_g = sharpe_per_day(day_means_arr[day_regs_arr == "green"])
                sh_r = sharpe_per_day(day_means_arr[day_regs_arr == "red"])
                denom = max(abs(sh_g) if np.isfinite(sh_g) else 0.0,
                            abs(sh_r) if np.isfinite(sh_r) else 0.0,
                            1e-9)
                if np.isfinite(sh_g) and np.isfinite(sh_r):
                    rim = abs(sh_g - sh_r) / denom
                else:
                    rim = float("nan")
                abs_tot = np.abs(day_sums_arr)
                conc = float(abs_tot.max() / abs_tot.sum()) if abs_tot.sum() > 0 else 1.0
                prof_days = int((day_means_arr > 0).sum())

                gate_net = mean_net > GATE_NET
                gate_sharpe = (sh > GATE_SHARPE) if np.isfinite(sh) else False
                gate_pdays = prof_days >= pdays_threshold
                gate_regime = (rim <= GATE_REGIME_IMB) if np.isfinite(rim) else False
                gate_dayconc = conc <= GATE_DAYCONC
                n_gp = int(gate_net) + int(gate_sharpe) + int(gate_pdays) + int(gate_regime) + int(gate_dayconc)
                pass_all = (n_gp == 5)

                row = dict(
                    side=side, horizon=h, threshold=thr, n_trades=n_tr,
                    net_ticks_per_trade=mean_net, win_rate=wr, sharpe=sh,
                    profitable_days=prof_days, total_days=n_days_oot,
                    sharpe_green=sh_g, sharpe_red=sh_r, regime_imbalance=rim,
                    day_concentration=conc,
                    gate_net=gate_net, gate_sharpe=gate_sharpe, gate_pdays=gate_pdays,
                    gate_regime=gate_regime, gate_dayconc=gate_dayconc,
                    n_gates_passed=n_gp, pass_gates=pass_all, best_iter=int(best_iter),
                )
                summary_rows.append(row)

                # Track closest miss (max n_gates_passed; tie-break by sharpe then net)
                key = (side, h)
                if best_in_pair is None:
                    best_in_pair = row
                else:
                    cur = best_in_pair
                    better = (n_gp > cur["n_gates_passed"]) or \
                             (n_gp == cur["n_gates_passed"] and (sh if np.isfinite(sh) else -9) > (cur["sharpe"] if np.isfinite(cur["sharpe"]) else -9)) or \
                             (n_gp == cur["n_gates_passed"] and (sh if np.isfinite(sh) else -9) == (cur["sharpe"] if np.isfinite(cur["sharpe"]) else -9) and mean_net > cur["net_ticks_per_trade"])
                    if better:
                        best_in_pair = row

                # Capture per-day records for winning OR closest cells (we filter later)
                for ud, dm, ds, dr in zip(uniq, day_means, day_sums, day_regs):
                    per_day_records.append(dict(
                        side=side, horizon=h, threshold=thr,
                        date=ud, regime=dr, day_mean_net=dm, day_sum_net=ds,
                    ))

            if best_in_pair is not None:
                closest_miss[(side, h)] = best_in_pair
            print(f"  done in {time.time()-t_m:.1f}s", flush=True)

    # ----- Build outputs -----
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(OUT_DIR / "summary.csv", index=False)

    # Feature importance (avg gain across 8 models)
    df_fi = pd.DataFrame(fi_records)
    df_fi_avg = df_fi.groupby("feature", as_index=False)["gain"].mean().sort_values("gain", ascending=False)
    df_fi_avg.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    top3_feats = df_fi_avg.head(3)["feature"].tolist()

    # Winners
    winners = df_sum[df_sum["pass_gates"]].copy()
    with open(OUT_DIR / "winning_cells.txt", "w") as f:
        if winners.empty:
            f.write("# No cells passed all 5 HC#428 gates. (Clean REJECT verdict, not a bug.)\n")
        else:
            f.write("# Cells passing all 5 HC#428 gates:\n")
            for _, r in winners.iterrows():
                f.write(f"{r['side']}_{r['horizon']}_thr{r['threshold']:.2f}: "
                        f"n_trades={r['n_trades']}, net={r['net_ticks_per_trade']:.3f}, "
                        f"sharpe={r['sharpe']:.2f}, pdays={r['profitable_days']}/{r['total_days']}, "
                        f"regime_imb={r['regime_imbalance']:.3f}, dayconc={r['day_concentration']:.3f}\n")

    # Closest miss per (side, horizon)
    cm_serializable = {}
    for (side, h), row in closest_miss.items():
        cm_serializable[f"{side}_{h}"] = {k: (None if (isinstance(v, float) and not np.isfinite(v)) else
                                              (float(v) if isinstance(v, (np.floating, float)) else
                                               (int(v) if isinstance(v, (np.integer, bool)) else v)))
                                          for k, v in row.items()}
    (OUT_DIR / "closest_miss.json").write_text(json.dumps(cm_serializable, indent=2, default=str))

    # Best cell overall (max gates passed; tie-break by sharpe then net)
    df_rank = df_sum.copy()
    df_rank["_sh"] = df_rank["sharpe"].fillna(-9)
    df_rank["_nt"] = df_rank["net_ticks_per_trade"].fillna(-9)
    df_rank = df_rank.sort_values(["n_gates_passed", "_sh", "_nt"], ascending=False)
    best = df_rank.iloc[0].to_dict()

    # Per-day stratification — filter to winners + closest cells
    keep_cells = set()
    for _, r in winners.iterrows():
        keep_cells.add((r["side"], r["horizon"], r["threshold"]))
    for (side, h), row in closest_miss.items():
        keep_cells.add((row["side"], row["horizon"], row["threshold"]))
    df_pd = pd.DataFrame(per_day_records)
    if not df_pd.empty:
        df_pd["_key"] = list(zip(df_pd["side"], df_pd["horizon"], df_pd["threshold"]))
        df_pd_keep = df_pd[df_pd["_key"].isin(keep_cells)].drop(columns=["_key"])
        df_pd_keep.to_csv(OUT_DIR / "per_day_stratification.csv", index=False)
    else:
        pd.DataFrame(columns=["side", "horizon", "threshold", "date", "regime",
                              "day_mean_net", "day_sum_net"]).to_csv(
            OUT_DIR / "per_day_stratification.csv", index=False)

    # Verdict
    if not winners.empty:
        verdict = "ACCEPT"
    elif best["n_gates_passed"] >= 4:
        verdict = "CONDITIONAL"
    else:
        verdict = "REJECT"

    # REPORT.md (plain English, <=30 lines, HC #69 risk-adjusted first)
    lines = []
    lines.append(f"# Meta-Classifier v1 Report")
    lines.append("")
    lines.append(f"**Verdict: {verdict}**")
    lines.append("")
    lines.append(f"- IS days: {n_is} ({is_days[0]['date']} → {is_days[-1]['date']})")
    lines.append(f"- OOT days: {n_oot} ({oot_days[0]['date']} → {oot_days[-1]['date']})")
    lines.append(f"- Feature set: 11 prediction-stream + 6 raw-market = 20 features (vs morning's snapshot-only)")
    lines.append("")
    lines.append("## Best cell across all (side × horizon × threshold)")
    lines.append(f"- **{best['side']} @ {best['horizon']}, threshold {best['threshold']:.2f}**")
    lines.append(f"- Trades over OOT: {int(best['n_trades'])}")
    lines.append(f"- Net ticks/trade: {best['net_ticks_per_trade']:.3f}  (gate > {GATE_NET})")
    lines.append(f"- Annualized Sharpe: {best['sharpe']:.2f}  (gate > {GATE_SHARPE})")
    lines.append(f"- Win rate: {best['win_rate']:.3f}")
    lines.append(f"- Profitable days: {int(best['profitable_days'])}/{int(best['total_days'])}  "
                 f"(gate ≥ {pdays_threshold})")
    lines.append(f"- Regime imbalance: {best['regime_imbalance']:.3f}  (gate ≤ {GATE_REGIME_IMB})")
    lines.append(f"- Day concentration: {best['day_concentration']:.3f}  (gate ≤ {GATE_DAYCONC})")
    lines.append(f"- Gates passed: {int(best['n_gates_passed'])}/5")
    lines.append("")
    lines.append("## Top-3 features by average gain across the 8 models")
    for i, fn in enumerate(top3_feats, 1):
        lines.append(f"{i}. {fn}")
    lines.append("")
    lines.append("## Notes")
    lines.append(f"- Cost model: passive-limit (0.376 ticks commission baked into labels).")
    lines.append(f"- Walk-forward: 1 contiguous IS block, 1 contiguous OOT block, no shuffling across days.")
    lines.append(f"- Regime proxy: sign of day's y_long_30s_net mean (green/red/flat, threshold ±{REGIME_THRESH} t).")
    lines.append(f"- Gates: HC #428 (net > {GATE_NET}, Sharpe > {GATE_SHARPE}, pdays ≥ {int(GATE_PDAYS_FRAC*100)}%, "
                 f"regime_imb ≤ {GATE_REGIME_IMB}, dayconc ≤ {GATE_DAYCONC}).")
    lines.append(f"- Inner-val: last {int(INNER_VAL_FRAC*100)}% of IS days for early-stopping.")
    (OUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n")

    # .regen_complete.json (HC #485 R5)
    elapsed = time.time() - t0
    finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    regen = dict(
        task="meta_classifier_v1_train",
        hc_refs=["HC#486R4", "HC#428", "HC#0", "HC#69", "HC#485R5", "HC#420", "HC#393"],
        started_at=started_at,
        finished_at=finished_at,
        elapsed_seconds=round(elapsed, 1),
        verdict=verdict,
        n_days_total=n_total,
        n_is=n_is, n_oot=n_oot,
        n_models=8,
        best_cell=cm_serializable.get(f"{best['side']}_{best['horizon']}", {}),
        top3_features=top3_feats,
        winners_count=int(len(winners)),
        gates=dict(net=GATE_NET, sharpe=GATE_SHARPE, pdays_frac=GATE_PDAYS_FRAC,
                   regime_imb=GATE_REGIME_IMB, dayconc=GATE_DAYCONC),
        outputs=dict(
            summary_csv=str(OUT_DIR / "summary.csv"),
            winning_cells=str(OUT_DIR / "winning_cells.txt"),
            closest_miss=str(OUT_DIR / "closest_miss.json"),
            per_day_stratification=str(OUT_DIR / "per_day_stratification.csv"),
            feature_importance=str(OUT_DIR / "feature_importance.csv"),
            report=str(OUT_DIR / "REPORT.md"),
            models_dir=str(MODELS_DIR),
        ),
    )
    (OUT_DIR / ".regen_complete.json").write_text(json.dumps(regen, indent=2, default=str))

    # Console summary
    print("\n========== SUMMARY ==========", flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    print(f"Best cell: {best['side']} {best['horizon']} thr={best['threshold']:.2f}  "
          f"n={int(best['n_trades'])}  net={best['net_ticks_per_trade']:.3f}  "
          f"Sharpe={best['sharpe']:.2f}  WR={best['win_rate']:.3f}  "
          f"pdays={int(best['profitable_days'])}/{int(best['total_days'])}  "
          f"regime_imb={best['regime_imbalance']:.3f}  dayconc={best['day_concentration']:.3f}  "
          f"gates={int(best['n_gates_passed'])}/5", flush=True)
    print(f"Top-3 features: {top3_feats}", flush=True)
    print(f"Elapsed: {elapsed:.1f}s", flush=True)
    print(f"Outputs: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
