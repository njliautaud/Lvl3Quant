#!/usr/bin/env python3
"""
meta_classifier_v1_walkforward.py — multi-fold walk-forward validation of the
meta-classifier v1 from meta_classifier_v1_train.py.

Validates whether the single-fold +4 t/trade short_10s finding generalizes
across multiple WF windows. Sliding-window protocol per HC #0 (oldest IS day
dropped per fold). CPU-only LightGBM.

Folds (32 day-NPZs total):
  Fold 0: IS days 0-15  -> OOT days 16-19
  Fold 1: IS days 4-19  -> OOT days 20-23
  Fold 2: IS days 8-23  -> OOT days 24-27
  Fold 3: IS days 12-27 -> OOT days 28-31
Each fold: 16 IS, 4 OOT. Total OOT coverage = 16 days.

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
OUT_DIR = ROOT / "output/meta_classifier_v1_wf"
MODELS_ROOT = OUT_DIR / "models"

HORIZONS = ["1s", "5s", "10s", "30s"]
SIDES = ["long", "short"]
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

# Fold protocol: 4 folds, sliding window, IS=16 / OOT=4
N_IS_PER_FOLD = 16
N_OOT_PER_FOLD = 4
FOLD_STRIDE = 4
N_FOLDS = 4
INNER_VAL_FRAC = 0.20  # last 20% of IS days for early-stopping

# LightGBM defaults (match single-fold script)
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

# HC #428 deploy gates — applied to POOLED metrics across all 16 OOT days
GATE_NET = 0.10
GATE_SHARPE = 0.3
# Stricter pdays gate: 11/16 = 0.6875 (>=69%) per task spec
GATE_PDAYS_ABS = 11
GATE_PDAYS_TOTAL = 16
GATE_REGIME_IMB = 0.50
GATE_DAYCONC = 0.70
# NEW gate: per-fold consistency — edge must appear in >= 3/4 folds
# (positive net_ticks AND >= 10 trades on that fold's OOT)
GATE_CONSISTENCY_MIN = 3

REGIME_THRESH = 0.10
ANNUAL_TRADING_DAYS = 252.0
MIN_TRADES_FOR_FOLD_VALID = 10  # for consistency-gate fold counting
MIN_TRADES_POOLED = 30          # below this we skip pooled cell metrics


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


def load_day(npz_path: Path) -> Dict:
    d = np.load(npz_path, allow_pickle=True)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.float32)
    feat_names = [str(s) for s in d["feat_names"]]
    label_names = [str(s) for s in d["label_names"]]
    date = str(d["date"][0]) if d["date"].shape[0] > 0 else npz_path.stem.replace("_meta", "")
    return dict(date=date, X=X, y=y, feat_names=feat_names, label_names=label_names)


def clean_xy_for_train(X: np.ndarray, y_col: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    finite_y = np.isfinite(y_col)
    finite_x = np.all(np.isfinite(X), axis=1)
    m = finite_y & finite_x
    return X[m], y_col[m]


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


def stack(days_list: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.concatenate([d["X"] for d in days_list], axis=0)
    y = np.concatenate([d["y"] for d in days_list], axis=0)
    date_per_row = np.concatenate(
        [np.full(d["X"].shape[0], d["date"], dtype=object) for d in days_list]
    )
    return X, y, date_per_row


# --------- Per-fold training + OOT scoring ----------------------------------
def run_fold(fold_idx: int,
             is_days: List[Dict],
             oot_days: List[Dict],
             feat_names: List[str],
             label_idx: Dict[str, int]):
    """
    Train 8 models on IS, score on OOT, return:
      - per_fold_rows: list of dict (side, horizon, threshold, ...metrics) for this fold
      - per_day_records: list of dict (side, horizon, threshold, date, regime, day_mean_net, day_sum_net)
      - cell_trade_arrays: dict[(side, h, thr)] -> {dates: array, nets: array}
      - fi_records: feature importance rows
    """
    n_is = len(is_days)
    n_oot = len(oot_days)
    inner_val_cut = max(1, min(n_is - 1, int(round(n_is * (1 - INNER_VAL_FRAC)))))
    train_days = is_days[:inner_val_cut]
    val_days = is_days[inner_val_cut:]
    print(f"[fold {fold_idx}] IS={n_is} (inner train={len(train_days)} val={len(val_days)}) OOT={n_oot}",
          flush=True)
    print(f"  IS dates: {is_days[0]['date']}..{is_days[-1]['date']}", flush=True)
    print(f"  OOT dates: {oot_days[0]['date']}..{oot_days[-1]['date']}", flush=True)

    X_tr, y_tr_all, _ = stack(train_days)
    X_va, y_va_all, _ = stack(val_days)
    X_oot, y_oot_all, dates_oot = stack(oot_days)

    net_col = {}
    win_col = {}
    for side in SIDES:
        for h in HORIZONS:
            net_col[(side, h)] = label_idx[f"y_{side}_{h}_net"]
            win_col[(side, h)] = label_idx[f"y_{side}_{h}_winner"]

    # Per-OOT-day regime (using long_30s_net)
    day_regime = {}
    for d in oot_days:
        col = d["y"][:, label_idx["y_long_30s_net"]]
        day_regime[d["date"]] = regime_of_day(col)
    uniq_oot_dates = sorted(set(d["date"] for d in oot_days))

    fold_models_dir = MODELS_ROOT / f"fold_{fold_idx}"
    fold_models_dir.mkdir(parents=True, exist_ok=True)

    per_fold_rows = []
    per_day_records = []
    cell_trade_arrays: Dict[Tuple[str, str, float], Dict[str, np.ndarray]] = {}
    fi_records = []

    for side in SIDES:
        for h in HORIZONS:
            tag = f"{side}_{h}"
            wcol = win_col[(side, h)]
            ncol = net_col[(side, h)]

            y_tr_w = y_tr_all[:, wcol]
            y_va_w = y_va_all[:, wcol]
            y_oot_w = y_oot_all[:, wcol]
            y_oot_n = y_oot_all[:, ncol]

            X_tr_c, y_tr_c = clean_xy_for_train(X_tr, y_tr_w)
            X_va_c, y_va_c = clean_xy_for_train(X_va, y_va_w)

            t_m = time.time()
            dtr = lgb.Dataset(X_tr_c, label=y_tr_c.astype(np.int32), feature_name=feat_names)
            dva = lgb.Dataset(X_va_c, label=y_va_c.astype(np.int32),
                              feature_name=feat_names, reference=dtr)
            booster = lgb.train(
                LGB_PARAMS, dtr,
                num_boost_round=N_ESTIMATORS,
                valid_sets=[dva], valid_names=["val"],
                callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                           lgb.log_evaluation(period=0)],
            )
            best_iter = booster.best_iteration or N_ESTIMATORS
            booster.save_model(str(fold_models_dir / f"{tag}.txt"), num_iteration=best_iter)

            gains = booster.feature_importance(importance_type="gain")
            for fn, g in zip(feat_names, gains):
                fi_records.append(dict(fold=fold_idx, model=tag, feature=fn, gain=float(g)))

            # OOT predict (drop non-finite)
            finite_x = np.all(np.isfinite(X_oot), axis=1)
            finite_yw = np.isfinite(y_oot_w)
            finite_yn = np.isfinite(y_oot_n)
            m_oot = finite_x & finite_yw & finite_yn
            X_oot_c = X_oot[m_oot]
            y_oot_n_c = y_oot_n[m_oot]
            dates_oot_c = dates_oot[m_oot]
            proba = booster.predict(X_oot_c, num_iteration=best_iter)

            for thr in THRESHOLDS:
                sel = proba >= thr
                n_tr = int(sel.sum())
                if n_tr < MIN_TRADES_FOR_FOLD_VALID:
                    # Still record (with NaN metrics) so per-fold consistency counts correctly.
                    per_fold_rows.append(dict(
                        fold=fold_idx, side=side, horizon=h, threshold=thr, n_trades=n_tr,
                        net_ticks_per_trade=float("nan"), win_rate=float("nan"),
                        sharpe=float("nan"), profitable_days=0, total_days=n_oot,
                        sharpe_green=float("nan"), sharpe_red=float("nan"),
                        regime_imbalance=float("nan"), day_concentration=float("nan"),
                        best_iter=int(best_iter), fold_pass_consistency=False,
                    ))
                    cell_trade_arrays.setdefault((side, h, thr),
                                                 {"dates": [], "nets": []})
                    continue

                net_sel = y_oot_n_c[sel]
                day_sel = dates_oot_c[sel]
                mean_net = float(np.mean(net_sel))
                wr = float(np.mean(net_sel > 0))

                day_means, day_sums, day_regs = [], [], []
                for ud in uniq_oot_dates:
                    mm = day_sel == ud
                    if mm.sum() == 0:
                        continue
                    day_means.append(float(np.mean(net_sel[mm])))
                    day_sums.append(float(np.sum(net_sel[mm])))
                    day_regs.append(day_regime.get(ud, "flat"))
                dm_arr = np.array(day_means) if day_means else np.array([np.nan])
                ds_arr = np.array(day_sums) if day_sums else np.array([0.0])
                dr_arr = np.array(day_regs) if day_regs else np.array(["flat"])
                sh = sharpe_per_day(dm_arr)
                sh_g = sharpe_per_day(dm_arr[dr_arr == "green"])
                sh_r = sharpe_per_day(dm_arr[dr_arr == "red"])
                denom = max(abs(sh_g) if np.isfinite(sh_g) else 0.0,
                            abs(sh_r) if np.isfinite(sh_r) else 0.0, 1e-9)
                rim = (abs(sh_g - sh_r) / denom) if (np.isfinite(sh_g) and np.isfinite(sh_r)) else float("nan")
                abs_tot = np.abs(ds_arr)
                conc = float(abs_tot.max() / abs_tot.sum()) if abs_tot.sum() > 0 else 1.0
                prof_days = int((dm_arr > 0).sum())

                fold_pass = (mean_net > 0.0) and (n_tr >= MIN_TRADES_FOR_FOLD_VALID)

                per_fold_rows.append(dict(
                    fold=fold_idx, side=side, horizon=h, threshold=thr, n_trades=n_tr,
                    net_ticks_per_trade=mean_net, win_rate=wr, sharpe=sh,
                    profitable_days=prof_days, total_days=n_oot,
                    sharpe_green=sh_g, sharpe_red=sh_r, regime_imbalance=rim,
                    day_concentration=conc, best_iter=int(best_iter),
                    fold_pass_consistency=bool(fold_pass),
                ))

                for ud, dm, ds, dr in zip(uniq_oot_dates[:len(day_means)],
                                          day_means, day_sums, day_regs):
                    per_day_records.append(dict(
                        fold=fold_idx, side=side, horizon=h, threshold=thr,
                        date=ud, regime=dr, day_mean_net=dm, day_sum_net=ds,
                    ))

                # Save selected trades for pooled aggregation
                key = (side, h, thr)
                prev = cell_trade_arrays.get(key, {"dates": [], "nets": []})
                prev["dates"] = list(prev["dates"]) + list(day_sel)
                prev["nets"] = list(prev["nets"]) + list(net_sel)
                cell_trade_arrays[key] = prev

            print(f"  [{tag}] fit {time.time()-t_m:.1f}s  best_iter={best_iter}", flush=True)

    return per_fold_rows, per_day_records, cell_trade_arrays, fi_records, day_regime


# --------- Main -------------------------------------------------------------
def main():
    t0 = time.time()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[start] {started_at}", flush=True)

    manifest_path = DATA_DIR / "meta_layer_v1_manifest.json"
    if not manifest_path.exists():
        print(f"[fatal] manifest missing: {manifest_path}", file=sys.stderr); sys.exit(2)
    manifest = json.loads(manifest_path.read_text())
    feat_names = list(manifest["feat_names"])
    label_names = list(manifest["label_names"])
    label_idx = {n: i for i, n in enumerate(label_names)}
    print(f"[manifest] {len(feat_names)} features, {len(label_names)} labels", flush=True)

    files = sorted(DATA_DIR.glob("*_meta.npz"))
    days = []
    for f in files:
        try:
            days.append(load_day(f))
        except Exception as e:
            print(f"  corrupt {f.name}: {e}", flush=True)
    days.sort(key=lambda x: x["date"])
    n_total = len(days)
    print(f"[loaded] {n_total} usable days  ({days[0]['date']}..{days[-1]['date']})", flush=True)
    needed = (N_FOLDS - 1) * FOLD_STRIDE + N_IS_PER_FOLD + N_OOT_PER_FOLD
    if n_total < needed:
        print(f"[fatal] need {needed} days, got {n_total}", file=sys.stderr); sys.exit(2)

    # Build fold definitions
    folds = []
    for k in range(N_FOLDS):
        is_start = k * FOLD_STRIDE
        is_end = is_start + N_IS_PER_FOLD
        oot_start = is_end
        oot_end = oot_start + N_OOT_PER_FOLD
        if oot_end > n_total:
            print(f"[warn] fold {k} OOT goes past data; trimming", flush=True)
            oot_end = n_total
        folds.append((k, days[is_start:is_end], days[oot_start:oot_end]))

    all_per_fold_rows: List[Dict] = []
    all_per_day_records: List[Dict] = []
    all_fi_records: List[Dict] = []
    # cell -> {dates: [..], nets: [..], folds_with_trades: set}
    pooled_cells: Dict[Tuple[str, str, float],
                       Dict[str, list]] = {}
    pooled_day_regime: Dict[str, str] = {}

    for fold_idx, is_days_f, oot_days_f in folds:
        pf_rows, pd_recs, cell_arrs, fi_recs, day_reg = run_fold(
            fold_idx, is_days_f, oot_days_f, feat_names, label_idx)
        all_per_fold_rows.extend(pf_rows)
        all_per_day_records.extend(pd_recs)
        all_fi_records.extend(fi_recs)
        pooled_day_regime.update(day_reg)
        for key, dct in cell_arrs.items():
            slot = pooled_cells.setdefault(key, {"dates": [], "nets": [], "folds": set()})
            if len(dct["nets"]) > 0:
                slot["dates"].extend(dct["dates"])
                slot["nets"].extend(dct["nets"])
                slot["folds"].add(fold_idx)

    # --- Per-fold consistency count per cell (positive net + >=10 trades) ---
    df_pf = pd.DataFrame(all_per_fold_rows)
    df_pf.to_csv(OUT_DIR / "wf_per_fold.csv", index=False)

    consistency = {}
    for (side, h, thr), grp in df_pf.groupby(["side", "horizon", "threshold"]):
        n_pass = int(grp["fold_pass_consistency"].sum())
        consistency[(side, h, float(thr))] = n_pass

    # --- Pooled metrics per cell ---
    pooled_rows = []
    for (side, h, thr), slot in pooled_cells.items():
        nets = np.array(slot["nets"], dtype=float)
        dates = np.array(slot["dates"])
        n_tr = int(nets.size)
        if n_tr < MIN_TRADES_POOLED:
            pooled_rows.append(dict(
                side=side, horizon=h, threshold=thr, n_trades_pooled=n_tr,
                net_ticks_per_trade=float("nan"), win_rate=float("nan"),
                sharpe_pooled=float("nan"),
                profitable_days=0, total_days=GATE_PDAYS_TOTAL,
                sharpe_green=float("nan"), sharpe_red=float("nan"),
                regime_imbalance=float("nan"), day_concentration=float("nan"),
                folds_consistent=consistency.get((side, h, float(thr)), 0),
                gate_net=False, gate_sharpe=False, gate_pdays=False,
                gate_regime=False, gate_dayconc=False, gate_consistency=False,
                n_gates_passed=0, pass_gates=False,
            ))
            continue

        mean_net = float(np.mean(nets))
        wr = float(np.mean(nets > 0))
        # Per-day aggregation across ALL OOT days (16)
        uniq_dates = sorted(set(dates.tolist()))
        dm_arr, ds_arr, dr_arr = [], [], []
        for ud in uniq_dates:
            mm = dates == ud
            if mm.sum() == 0:
                continue
            dm_arr.append(float(np.mean(nets[mm])))
            ds_arr.append(float(np.sum(nets[mm])))
            dr_arr.append(pooled_day_regime.get(ud, "flat"))
        dm_arr = np.array(dm_arr)
        ds_arr = np.array(ds_arr)
        dr_arr = np.array(dr_arr)
        sh = sharpe_per_day(dm_arr)
        sh_g = sharpe_per_day(dm_arr[dr_arr == "green"])
        sh_r = sharpe_per_day(dm_arr[dr_arr == "red"])
        denom = max(abs(sh_g) if np.isfinite(sh_g) else 0.0,
                    abs(sh_r) if np.isfinite(sh_r) else 0.0, 1e-9)
        rim = (abs(sh_g - sh_r) / denom) if (np.isfinite(sh_g) and np.isfinite(sh_r)) else float("nan")
        abs_tot = np.abs(ds_arr)
        conc = float(abs_tot.max() / abs_tot.sum()) if abs_tot.sum() > 0 else 1.0
        prof_days = int((dm_arr > 0).sum())
        n_consistent = consistency.get((side, h, float(thr)), 0)

        gate_net = mean_net > GATE_NET
        gate_sharpe = (sh > GATE_SHARPE) if np.isfinite(sh) else False
        gate_pdays = prof_days >= GATE_PDAYS_ABS
        gate_regime = (rim <= GATE_REGIME_IMB) if np.isfinite(rim) else False
        gate_dayconc = conc <= GATE_DAYCONC
        gate_consistency = n_consistent >= GATE_CONSISTENCY_MIN
        n_gp = int(gate_net) + int(gate_sharpe) + int(gate_pdays) + int(gate_regime) + int(gate_dayconc) + int(gate_consistency)
        pass_all = (n_gp == 6)

        pooled_rows.append(dict(
            side=side, horizon=h, threshold=thr, n_trades_pooled=n_tr,
            net_ticks_per_trade=mean_net, win_rate=wr, sharpe_pooled=sh,
            profitable_days=prof_days, total_days=GATE_PDAYS_TOTAL,
            sharpe_green=sh_g, sharpe_red=sh_r,
            regime_imbalance=rim, day_concentration=conc,
            folds_consistent=n_consistent,
            gate_net=gate_net, gate_sharpe=gate_sharpe, gate_pdays=gate_pdays,
            gate_regime=gate_regime, gate_dayconc=gate_dayconc,
            gate_consistency=gate_consistency,
            n_gates_passed=n_gp, pass_gates=pass_all,
        ))

    df_sum = pd.DataFrame(pooled_rows)
    df_sum = df_sum.sort_values(["side", "horizon", "threshold"]).reset_index(drop=True)
    df_sum.to_csv(OUT_DIR / "wf_summary.csv", index=False)

    # --- Feature importance averaged across all folds and models ---
    df_fi = pd.DataFrame(all_fi_records)
    df_fi_avg = (df_fi.groupby("feature", as_index=False)["gain"].mean()
                       .sort_values("gain", ascending=False))
    df_fi_avg.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    top3_feats = df_fi_avg.head(3)["feature"].tolist()

    # --- Winners file ---
    winners = df_sum[df_sum["pass_gates"]].copy()
    with open(OUT_DIR / "winning_cells.txt", "w") as f:
        if winners.empty:
            f.write("# No cells passed all 6 gates (5 HC#428 + per-fold consistency).\n")
            f.write("# This is a clean REJECT or CONDITIONAL verdict, see REPORT.md.\n")
        else:
            f.write("# Cells passing all 6 gates (HC#428 + per-fold consistency):\n")
            for _, r in winners.iterrows():
                f.write(f"{r['side']}_{r['horizon']}_thr{r['threshold']:.2f}: "
                        f"n_pooled={int(r['n_trades_pooled'])}, "
                        f"net={r['net_ticks_per_trade']:.3f}, "
                        f"Sharpe={r['sharpe_pooled']:.2f}, "
                        f"pdays={int(r['profitable_days'])}/{int(r['total_days'])}, "
                        f"regime_imb={r['regime_imbalance']:.3f}, "
                        f"dayconc={r['day_concentration']:.3f}, "
                        f"folds_consistent={int(r['folds_consistent'])}/{N_FOLDS}\n")

    # --- Closest miss per (side, horizon) ---
    def _safe(v):
        if isinstance(v, float) and not np.isfinite(v):
            return None
        if isinstance(v, (np.floating, float)):
            return float(v)
        if isinstance(v, (np.integer, bool, np.bool_)):
            return int(v)
        return v

    closest_miss = {}
    for (side, h), grp in df_sum.groupby(["side", "horizon"]):
        if grp.empty:
            continue
        g2 = grp.copy()
        g2["_sh"] = g2["sharpe_pooled"].fillna(-9)
        g2["_nt"] = g2["net_ticks_per_trade"].fillna(-9)
        g2 = g2.sort_values(["n_gates_passed", "_sh", "_nt"], ascending=False)
        top = g2.iloc[0].to_dict()
        closest_miss[f"{side}_{h}"] = {k: _safe(v) for k, v in top.items() if not k.startswith("_")}
    (OUT_DIR / "closest_miss.json").write_text(json.dumps(closest_miss, indent=2, default=str))

    # --- Regime breakdown per fold (green/red Sharpe) ---
    df_pd = pd.DataFrame(all_per_day_records)
    regime_rows = []
    if not df_pd.empty:
        for (fold_idx, side, h, thr), g in df_pd.groupby(["fold", "side", "horizon", "threshold"]):
            dm_g = g.loc[g["regime"] == "green", "day_mean_net"].to_numpy()
            dm_r = g.loc[g["regime"] == "red", "day_mean_net"].to_numpy()
            dm_f = g.loc[g["regime"] == "flat", "day_mean_net"].to_numpy()
            regime_rows.append(dict(
                fold=fold_idx, side=side, horizon=h, threshold=thr,
                n_green=int(dm_g.size), n_red=int(dm_r.size), n_flat=int(dm_f.size),
                mean_green=float(np.nanmean(dm_g)) if dm_g.size else float("nan"),
                mean_red=float(np.nanmean(dm_r)) if dm_r.size else float("nan"),
                mean_flat=float(np.nanmean(dm_f)) if dm_f.size else float("nan"),
                sharpe_green=sharpe_per_day(dm_g),
                sharpe_red=sharpe_per_day(dm_r),
            ))
    pd.DataFrame(regime_rows).to_csv(OUT_DIR / "regime_breakdown.csv", index=False)

    # --- Best overall pick (max gates -> sharpe -> net) ---
    df_rank = df_sum.copy()
    df_rank["_sh"] = df_rank["sharpe_pooled"].fillna(-9)
    df_rank["_nt"] = df_rank["net_ticks_per_trade"].fillna(-9)
    df_rank = df_rank.sort_values(["n_gates_passed", "_sh", "_nt"], ascending=False)
    best = df_rank.iloc[0].to_dict() if not df_rank.empty else {}

    # --- Short_10s thr=0.50 specifically ---
    short10_row = df_sum[(df_sum["side"] == "short") &
                          (df_sum["horizon"] == "10s") &
                          (np.isclose(df_sum["threshold"], 0.50))]
    short10_pf = df_pf[(df_pf["side"] == "short") &
                        (df_pf["horizon"] == "10s") &
                        (np.isclose(df_pf["threshold"], 0.50))]
    short10_summary = short10_row.iloc[0].to_dict() if len(short10_row) else {}
    short10_fold_consistency = int(short10_pf["fold_pass_consistency"].sum()) if len(short10_pf) else 0

    # --- Verdict ---
    if not winners.empty:
        verdict = "ACCEPT"
    elif (not df_sum.empty) and (
        (df_sum["gate_net"].any() and df_sum["gate_sharpe"].any())
        and df_rank.iloc[0]["n_gates_passed"] >= 4
    ):
        verdict = "CONDITIONAL_ACCEPT"
    else:
        verdict = "REJECT"

    # --- REPORT.md (<=30 lines, plain English) ---
    lines = []
    lines.append(f"# Meta-Classifier v1 — Multi-Fold Walk-Forward Report")
    lines.append("")
    lines.append(f"**Verdict: {verdict}**")
    lines.append("")
    lines.append(f"Protocol: 4 folds, sliding window (HC #0). IS=16 days, OOT=4 days/fold, 16 OOT days total.")
    lines.append(f"Total models trained: {N_FOLDS * len(SIDES) * len(HORIZONS)} (4 folds x 8 models).")
    lines.append("")
    if best:
        lines.append("## Best cell (pooled across all 4 folds)")
        lines.append(f"- **{best['side']} @ {best['horizon']}, threshold {best['threshold']:.2f}**")
        lines.append(f"- Pooled trades over 16 OOT days: {int(best['n_trades_pooled'])}")
        lines.append(f"- Net ticks/trade (pooled): {best['net_ticks_per_trade']:.3f}  (gate > {GATE_NET})")
        lines.append(f"- Annualized Sharpe (pooled day-means): {best['sharpe_pooled']:.2f}  (gate > {GATE_SHARPE})")
        lines.append(f"- Win rate: {best['win_rate']:.3f}")
        lines.append(f"- Profitable days: {int(best['profitable_days'])}/{int(best['total_days'])}  (gate >= {GATE_PDAYS_ABS})")
        lines.append(f"- Regime imbalance (|Sg-Sr|/max): {best['regime_imbalance']:.3f}  (gate <= {GATE_REGIME_IMB})")
        lines.append(f"- Day concentration: {best['day_concentration']:.3f}  (gate <= {GATE_DAYCONC})")
        lines.append(f"- Per-fold consistency: {int(best['folds_consistent'])}/{N_FOLDS}  (gate >= {GATE_CONSISTENCY_MIN})")
        lines.append(f"- Gates passed: {int(best['n_gates_passed'])}/6")
        lines.append("")
    if short10_summary:
        lines.append("## Did short_10s @ thr=0.50 (single-fold leader) survive?")
        lines.append(f"- Pooled trades: {int(short10_summary.get('n_trades_pooled', 0))}")
        lines.append(f"- Net ticks/trade: {short10_summary.get('net_ticks_per_trade', float('nan')):.3f}")
        lines.append(f"- Sharpe: {short10_summary.get('sharpe_pooled', float('nan')):.2f}")
        lines.append(f"- Profitable days: {int(short10_summary.get('profitable_days', 0))}/16")
        lines.append(f"- Per-fold consistency: {short10_fold_consistency}/{N_FOLDS}")
        lines.append(f"- Gates passed: {int(short10_summary.get('n_gates_passed', 0))}/6")
        survived = (short10_summary.get("pass_gates", False) is True)
        lines.append(f"- **SURVIVED MULTI-FOLD: {'YES' if survived else 'NO'}**")
        lines.append("")
    lines.append("## Top-3 features (avg gain across all folds and models)")
    for i, fn in enumerate(top3_feats, 1):
        lines.append(f"{i}. {fn}")
    lines.append("")
    lines.append("## Notes")
    lines.append(f"- Labels: passive-limit net ticks (commission 0.376 baked in).")
    lines.append(f"- Per-fold consistency gate (NEW): >= {GATE_CONSISTENCY_MIN}/{N_FOLDS} folds with positive net AND >= {MIN_TRADES_FOR_FOLD_VALID} trades.")
    lines.append(f"- Pdays gate raised to >= {GATE_PDAYS_ABS}/{GATE_PDAYS_TOTAL} (~69%) per multi-fold spec.")
    (OUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n")

    # --- regen_complete.json ---
    elapsed = time.time() - t0
    finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    regen = dict(
        task="meta_classifier_v1_walkforward",
        hc_refs=["HC#486R4", "HC#428", "HC#0", "HC#69", "HC#485R5", "HC#420", "HC#393"],
        started_at=started_at, finished_at=finished_at,
        elapsed_seconds=round(elapsed, 1),
        verdict=verdict,
        n_days_total=n_total,
        n_folds=N_FOLDS, n_is_per_fold=N_IS_PER_FOLD, n_oot_per_fold=N_OOT_PER_FOLD,
        n_models_total=N_FOLDS * len(SIDES) * len(HORIZONS),
        best_cell={k: _safe(v) for k, v in best.items()} if best else {},
        short10_thr050={k: _safe(v) for k, v in short10_summary.items()} if short10_summary else {},
        short10_fold_consistency=short10_fold_consistency,
        top3_features=top3_feats,
        winners_count=int(len(winners)),
        gates=dict(net=GATE_NET, sharpe=GATE_SHARPE,
                   pdays_abs=GATE_PDAYS_ABS, pdays_total=GATE_PDAYS_TOTAL,
                   regime_imb=GATE_REGIME_IMB, dayconc=GATE_DAYCONC,
                   consistency_min=GATE_CONSISTENCY_MIN),
        outputs=dict(
            wf_summary=str(OUT_DIR / "wf_summary.csv"),
            wf_per_fold=str(OUT_DIR / "wf_per_fold.csv"),
            winning_cells=str(OUT_DIR / "winning_cells.txt"),
            closest_miss=str(OUT_DIR / "closest_miss.json"),
            regime_breakdown=str(OUT_DIR / "regime_breakdown.csv"),
            feature_importance=str(OUT_DIR / "feature_importance.csv"),
            report=str(OUT_DIR / "REPORT.md"),
            models_root=str(MODELS_ROOT),
        ),
    )
    (OUT_DIR / ".regen_complete.json").write_text(json.dumps(regen, indent=2, default=str))

    # --- Console summary ---
    print("\n========== WALK-FORWARD SUMMARY ==========", flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    if best:
        print(f"Best cell: {best['side']} {best['horizon']} thr={best['threshold']:.2f}  "
              f"n_pooled={int(best['n_trades_pooled'])}  "
              f"net={best['net_ticks_per_trade']:.3f}  "
              f"Sharpe={best['sharpe_pooled']:.2f}  "
              f"WR={best['win_rate']:.3f}  "
              f"pdays={int(best['profitable_days'])}/{int(best['total_days'])}  "
              f"regime_imb={best['regime_imbalance']:.3f}  "
              f"dayconc={best['day_concentration']:.3f}  "
              f"folds={int(best['folds_consistent'])}/{N_FOLDS}  "
              f"gates={int(best['n_gates_passed'])}/6", flush=True)
    if short10_summary:
        sv = short10_summary.get('pass_gates', False)
        print(f"short_10s thr=0.50: net={short10_summary.get('net_ticks_per_trade', float('nan')):.3f} "
              f"Sharpe={short10_summary.get('sharpe_pooled', float('nan')):.2f} "
              f"folds={short10_fold_consistency}/{N_FOLDS} "
              f"gates={int(short10_summary.get('n_gates_passed', 0))}/6 "
              f"SURVIVED={'YES' if sv else 'NO'}", flush=True)
    print(f"Top-3 features: {top3_feats}", flush=True)
    print(f"Elapsed: {elapsed:.1f}s", flush=True)
    print(f"Outputs: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
