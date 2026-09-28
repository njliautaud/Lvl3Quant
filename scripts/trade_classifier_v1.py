"""
Trade-level classifier (HC #428 follow-up).

Context: day-level classifier rejected under forward-walk (15-sample overfit).
short_10s @ thr=0.55 has +5.03 t/trade FIFO realized across 16 OOT days,
8/16 profit days. This script trains a TRADE-level gate (thousands of samples).

Inputs:
  - output/meta_classifier_v1_fifo/per_trade_diagnostics.csv  (realized trades)
  - data/processed/meta_layer_v1/<date>_meta.npz             (event features)
  - data/processed/mbo_events_smart_v3/<date>_mbo_events.npz (timestamps)
  - output/meta_classifier_v1_wf/models/fold_<k>/short_10s.txt (booster for meta_prob)

Method:
  - Re-score every realized trade with its fold's meta booster to get meta_prob.
  - Pull 20 meta-layer features at signal event (causal — at-signal-time only).
  - Add time-of-day, day-of-week, queue_ahead (book microstructure as of signal).
  - Walk-forward (sliding) trade-time order, target = realized_ticks_net > 0.
  - LightGBM binary classifier + LogReg robustness check.
  - Threshold sweep + verdict per HC #428.

Output:
  output/trade_classifier_v1/
    REPORT.md
    oos_predictions.parquet
    feature_importance.csv
    gated_pnl_sweep.csv
    .regen_complete.json
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")

PER_TRADE_CSV = LVL3_ROOT / "output" / "meta_classifier_v1_fifo" / "per_trade_diagnostics.csv"
META_DIR     = LVL3_ROOT / "data" / "processed" / "meta_layer_v1"
MBO_DIR      = LVL3_ROOT / "data" / "processed" / "mbo_events_smart_v3"
MODELS_ROOT  = LVL3_ROOT / "output" / "meta_classifier_v1_wf" / "models"
OUT_DIR      = LVL3_ROOT / "output" / "trade_classifier_v1"

CANDIDATE_TAG = "short_10s"     # survivor from HC #428 FIFO realized
CANDIDATE_NAME = "short_10s_thr55"
THRESHOLD_BASE = 0.55           # the original meta-prob gate
HORIZON_SEC = 10                # h for stream-consistency features

# Walk-forward: time-ordered trade sliding, 5 folds
N_WF_FOLDS = 5
TRAIN_FRAC_PER_FOLD = 0.60      # 60% train / 20% val / 20% test per fold window

# Verdict gates (per spec)
GATE_AUC = 0.55
GATE_PROFIT_DAYS_RATIO = 0.69
GATE_TPT_POOLED = 5.0
BASELINE_TPT = 5.03


# ---------- helpers ----------
def load_meta_day(date_str: str):
    p = META_DIR / f"{date_str}_meta.npz"
    d = np.load(p, allow_pickle=True)
    return {
        "X": d["X"].astype(np.float32),
        "feat_names": [str(s) for s in d["feat_names"]],
        "event_idx": d["event_idx"].astype(np.int64),
    }


def load_ts_by_event(date_str: str) -> np.ndarray:
    p = MBO_DIR / f"{date_str}_mbo_events.npz"
    d = np.load(p, allow_pickle=True)
    return d["timestamps"].astype(np.int64)


def load_booster(fold_idx: int):
    import lightgbm as lgb
    p = MODELS_ROOT / f"fold_{fold_idx}" / f"{CANDIDATE_TAG}.txt"
    if not p.exists():
        raise FileNotFoundError(p)
    return lgb.Booster(model_file=str(p))


# ---------- 1. Build feature matrix from per-trade rows ----------
def build_feature_table() -> Tuple[pd.DataFrame, List[str]]:
    """Returns dataframe with one row per realized trade (short_10s_thr55 only)."""
    print(f"[load] {PER_TRADE_CSV}", flush=True)
    df = pd.read_csv(PER_TRADE_CSV)
    df = df[df["candidate"] == CANDIDATE_NAME].copy()
    df["date"] = df["date"].astype(int).astype(str)
    print(f"[trades] n={len(df)} candidates={CANDIDATE_NAME}", flush=True)

    # Cache by (fold, date)
    feat_rows = []
    missing_event = 0
    missing_score = 0

    fold_boosters = {}
    for fk in sorted(df["fold"].unique()):
        try:
            fold_boosters[int(fk)] = load_booster(int(fk))
        except FileNotFoundError as e:
            print(f"[warn] no model for fold {fk}: {e}", flush=True)

    by_date = df.groupby("date", sort=True)
    feat_names_ref: List[str] = []

    for date_str, gdate in by_date:
        # load meta and ts once per day
        try:
            meta = load_meta_day(date_str)
            ts_by_event = load_ts_by_event(date_str)
        except FileNotFoundError as e:
            print(f"[skip] {date_str}: {e}", flush=True)
            continue

        feat_names_ref = meta["feat_names"]
        X = meta["X"]
        ev_idx = meta["event_idx"]

        # Build a lookup: event_ts_ns -> meta_row_idx via event_idx
        # meta row r -> event_idx[r] -> timestamp ts_by_event[event_idx[r]]
        # We need to find meta row by signal_ts_ns. Build ts -> row map.
        row_finite = np.all(np.isfinite(X), axis=1)
        in_range = (ev_idx >= 0) & (ev_idx < len(ts_by_event))
        valid_meta = row_finite & in_range
        meta_ts = np.full(len(ev_idx), -1, dtype=np.int64)
        meta_ts[in_range] = ts_by_event[ev_idx[in_range]]
        # Use a dict for ts -> meta_row (last wins; should be unique 1:1)
        ts_to_row: Dict[int, int] = {}
        valid_rows = np.where(valid_meta)[0]
        for r in valid_rows:
            ts_to_row[int(meta_ts[r])] = int(r)

        # Score all valid meta rows for this date with each fold's booster
        scored_by_fold: Dict[int, np.ndarray] = {}
        for fk, booster in fold_boosters.items():
            proba = np.full(len(ev_idx), np.nan, dtype=np.float64)
            if valid_rows.size:
                proba[valid_rows] = booster.predict(X[valid_rows])
            scored_by_fold[fk] = proba

        # Iterate trades for this date
        for _, tr in gdate.iterrows():
            fold = int(tr["fold"])
            ts_ns = int(tr["signal_ts_ns"])
            row = ts_to_row.get(ts_ns, None)
            if row is None:
                missing_event += 1
                continue
            proba = scored_by_fold.get(fold)
            if proba is None or not np.isfinite(proba[row]):
                missing_score += 1
                continue

            xrow = X[row]
            ts_sec_of_day_et = _ts_to_seconds_of_day_et(ts_ns)
            mins_from_open = (ts_sec_of_day_et - (9 * 3600 + 30 * 60)) / 60.0

            record = {
                "candidate": CANDIDATE_NAME,
                "fold_orig": fold,
                "date": date_str,
                "signal_ts_ns": ts_ns,
                "fill_ts_ns": int(tr["fill_ts_ns"]),
                "exit_ts_ns": int(tr["exit_ts_ns"]),
                "direction": tr["direction"],
                "queue_ahead": float(tr["queue_ahead"]),
                "queue_wait_ms": float(tr["queue_wait_ns"]) / 1e6,
                "exit_reason": tr["exit_reason"],
                "pnl_ticks_net": float(tr["pnl_ticks_net"]),
                "meta_prob": float(proba[row]),
                "mins_from_open": float(mins_from_open),
                "tod_bucket": _tod_bucket(mins_from_open),
                "dow": _ts_to_dow_et(ts_ns),
            }
            for fn, fv in zip(meta["feat_names"], xrow.tolist()):
                record[f"f_{fn}"] = float(fv)
            feat_rows.append(record)

    if not feat_rows:
        raise RuntimeError("No trades could be feature-joined.")

    out = pd.DataFrame(feat_rows)
    print(f"[features] joined={len(out)}  missing_event_ts={missing_event}  "
          f"missing_score={missing_score}", flush=True)

    # One-hot day-of-week + tod_bucket
    for d in ["Mon", "Tue", "Wed", "Thu", "Fri"]:
        out[f"dow_{d.lower()}"] = (out["dow"] == d).astype(int)
    for b in ["open", "mid_morning", "midday", "afternoon", "close"]:
        out[f"tod_{b}"] = (out["tod_bucket"] == b).astype(int)

    return out, feat_names_ref


def _ts_to_seconds_of_day_et(ts_ns: int) -> float:
    # ts_ns is UTC nanoseconds. ET = UTC-4 (DST) for the dates we have.
    # All trade dates are mar-apr 2026, US DST is active (started Mar 8 2026).
    import datetime as dt
    utc = dt.datetime.utcfromtimestamp(ts_ns / 1e9)
    # Approximate ET as UTC-4 (DST) — all dates here are in DST window.
    et = utc - dt.timedelta(hours=4)
    return et.hour * 3600 + et.minute * 60 + et.second + et.microsecond / 1e6


def _ts_to_dow_et(ts_ns: int) -> str:
    import datetime as dt
    et = dt.datetime.utcfromtimestamp(ts_ns / 1e9) - dt.timedelta(hours=4)
    return ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][et.weekday()]


def _tod_bucket(mins_from_open: float) -> str:
    if mins_from_open < 30:
        return "open"
    if mins_from_open < 90:
        return "mid_morning"
    if mins_from_open < 240:
        return "midday"
    if mins_from_open < 360:
        return "afternoon"
    return "close"


# ---------- 2. Walk-forward train/eval ----------
def walk_forward(df: pd.DataFrame) -> Dict:
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    df = df.sort_values("signal_ts_ns").reset_index(drop=True)
    df["label"] = (df["pnl_ticks_net"] > 0).astype(int)

    feat_cols = [c for c in df.columns if c.startswith("f_")]
    extra_cols = ["meta_prob", "mins_from_open", "queue_ahead"]
    onehot_cols = [c for c in df.columns
                   if (c.startswith("dow_") or c.startswith("tod_"))
                   and c not in ("tod_bucket",)]
    feat_cols = feat_cols + extra_cols + onehot_cols
    # Drop any non-numeric (safety)
    feat_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c])]
    print(f"[features] using {len(feat_cols)} columns", flush=True)

    n = len(df)
    fold_size = n // N_WF_FOLDS
    fold_results = []
    all_oos_preds = []
    all_oos_logreg = []
    fi_accum = pd.Series(0.0, index=feat_cols)

    for k in range(N_WF_FOLDS - 1):
        train_start = 0 if k == 0 else k * fold_size
        train_end = (k + 1) * fold_size
        test_start = train_end
        test_end = min((k + 2) * fold_size, n)
        if test_end - test_start < 50:
            print(f"[fold {k}] test too small ({test_end - test_start}); skip")
            continue

        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[test_start:test_end]
        # Reserve last 15% of train as val for early stopping
        val_cut = int(0.85 * len(train_df))
        tr = train_df.iloc[:val_cut]
        va = train_df.iloc[val_cut:]

        Xtr = tr[feat_cols].values
        ytr = tr["label"].values
        Xva = va[feat_cols].values
        yva = va["label"].values
        Xte = test_df[feat_cols].values
        yte = test_df["label"].values

        params = dict(
            objective="binary",
            metric="auc",
            num_leaves=15,
            max_depth=4,
            min_data_in_leaf=100,
            learning_rate=0.05,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbose=-1,
        )
        train_set = lgb.Dataset(Xtr, label=ytr, feature_name=feat_cols)
        val_set   = lgb.Dataset(Xva, label=yva, reference=train_set, feature_name=feat_cols)
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=200,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        p_test = booster.predict(Xte, num_iteration=booster.best_iteration)
        # Mostly-degenerate case: if only one class in test → AUC undefined
        if len(np.unique(yte)) < 2:
            auc = float("nan")
        else:
            auc = roc_auc_score(yte, p_test)

        # LogReg robustness check
        try:
            sc = StandardScaler().fit(Xtr)
            lr = LogisticRegression(max_iter=200, C=1.0).fit(sc.transform(Xtr), ytr)
            p_lr = lr.predict_proba(sc.transform(Xte))[:, 1]
            auc_lr = roc_auc_score(yte, p_lr) if len(np.unique(yte)) > 1 else float("nan")
        except Exception as e:
            auc_lr = float("nan")
            p_lr = np.full_like(p_test, 0.5)

        # Accumulate FI
        try:
            gains = booster.feature_importance(importance_type="gain")
            for c, g in zip(feat_cols, gains):
                fi_accum[c] += float(g)
        except Exception:
            pass

        # Test outputs
        out = test_df[[
            "candidate", "date", "signal_ts_ns", "direction",
            "pnl_ticks_net", "meta_prob",
        ]].copy()
        out["wf_fold"] = k
        out["clf_prob"] = p_test
        out["lr_prob"] = p_lr
        out["label"] = yte
        all_oos_preds.append(out)

        fold_results.append({
            "fold": k,
            "n_train": len(tr),
            "n_val": len(va),
            "n_test": len(test_df),
            "auc_lgbm": auc,
            "auc_logreg": auc_lr,
            "base_rate_test": float(yte.mean()),
            "best_iter": booster.best_iteration or 0,
        })
        print(f"[fold {k}] n_tr={len(tr)} n_va={len(va)} n_te={len(test_df)} "
              f"auc_lgbm={auc:.4f} auc_lr={auc_lr:.4f}", flush=True)

    oos = pd.concat(all_oos_preds, ignore_index=True) if all_oos_preds else pd.DataFrame()
    fi_df = pd.DataFrame({
        "feature": fi_accum.index,
        "gain_sum": fi_accum.values,
    }).sort_values("gain_sum", ascending=False).reset_index(drop=True)
    fi_df["gain_pct"] = (
        fi_df["gain_sum"] / max(fi_df["gain_sum"].sum(), 1e-9) * 100
    )

    return {
        "fold_results": fold_results,
        "oos": oos,
        "fi": fi_df,
    }


# ---------- 3. Gated P&L sweep ----------
def gated_pnl_sweep(oos: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if oos.empty:
        return pd.DataFrame()
    # Baseline (no gating beyond original meta-prob>=0.55)
    base = oos
    rows.append(_gated_row(base, tau=float("nan"), label="baseline (all OOS trades)"))
    for tau in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        sel = oos[oos["clf_prob"] >= tau]
        rows.append(_gated_row(sel, tau, label=f"clf_prob>={tau}"))
    return pd.DataFrame(rows)


def _gated_row(df: pd.DataFrame, tau: float, label: str) -> dict:
    n = len(df)
    if n == 0:
        return {
            "label": label,
            "tau": tau,
            "n_trades": 0,
            "tpt_pooled": float("nan"),
            "wr": float("nan"),
            "profit_days": 0,
            "total_days": 0,
            "profit_days_ratio": float("nan"),
            "sharpe": float("nan"),
        }
    nets = df["pnl_ticks_net"].values
    tpt = float(np.mean(nets))
    wr = float(np.mean(nets > 0))
    by_day = df.groupby("date")["pnl_ticks_net"].mean()
    profit_days = int((by_day > 0).sum())
    total_days = int(len(by_day))
    sd = float(np.std(nets, ddof=1)) if n > 1 else float("nan")
    sharpe = (tpt / sd * math.sqrt(252)) if sd and sd > 1e-9 else float("nan")
    return {
        "label": label,
        "tau": tau,
        "n_trades": n,
        "tpt_pooled": tpt,
        "wr": wr,
        "profit_days": profit_days,
        "total_days": total_days,
        "profit_days_ratio": profit_days / max(total_days, 1),
        "sharpe": sharpe,
    }


# ---------- 4. Verdict ----------
def verdict_from(sweep: pd.DataFrame, auc_mean: float) -> Tuple[str, List[str], dict]:
    """Pick best tau on OOS, compute verdict."""
    cand = sweep[sweep["tau"].notna()].copy()
    # best by profit_days_ratio, then tpt, must have >= 30 trades
    cand = cand[cand["n_trades"] >= 30]
    if cand.empty:
        return ("REJECT", ["no tau retained >=30 trades"], {})
    cand["score"] = cand["profit_days_ratio"] + cand["tpt_pooled"] / 20.0
    best = cand.sort_values("score", ascending=False).iloc[0].to_dict()

    reasons: List[str] = []
    if not (auc_mean >= GATE_AUC):
        reasons.append(f"auc_mean={auc_mean:.3f}<{GATE_AUC}")
    if best["profit_days_ratio"] < GATE_PROFIT_DAYS_RATIO:
        reasons.append(
            f"profit_days_ratio={best['profit_days_ratio']:.2f}<{GATE_PROFIT_DAYS_RATIO}"
        )
    if best["tpt_pooled"] < GATE_TPT_POOLED:
        reasons.append(f"tpt_pooled={best['tpt_pooled']:.2f}<{GATE_TPT_POOLED}")
    if best["tpt_pooled"] < BASELINE_TPT:
        reasons.append(
            f"tpt_pooled drops below baseline {BASELINE_TPT:.2f}"
        )

    if not reasons:
        return ("ACCEPT", [], best)
    # PARTIAL if AUC OK and pdays >= 0.60
    if auc_mean >= GATE_AUC and best["profit_days_ratio"] >= 0.60:
        return ("PARTIAL", reasons, best)
    return ("REJECT", reasons, best)


# ---------- 5. Main ----------
def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df, _ = build_feature_table()
    if len(df) < 200:
        print(f"[abort] only {len(df)} trades", flush=True)
        return

    res = walk_forward(df)
    oos = res["oos"]
    fi = res["fi"]

    fold_results = res["fold_results"]
    aucs = [r["auc_lgbm"] for r in fold_results if np.isfinite(r["auc_lgbm"])]
    auc_mean = float(np.mean(aucs)) if aucs else float("nan")
    auc_std = float(np.std(aucs)) if aucs else float("nan")
    auc_lr_list = [r["auc_logreg"] for r in fold_results if np.isfinite(r["auc_logreg"])]
    auc_lr_mean = float(np.mean(auc_lr_list)) if auc_lr_list else float("nan")

    sweep = gated_pnl_sweep(oos)
    verdict_str, reasons, best = verdict_from(sweep, auc_mean)

    # Save artifacts
    oos.to_parquet(OUT_DIR / "oos_predictions.parquet", index=False)
    fi.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    sweep.to_csv(OUT_DIR / "gated_pnl_sweep.csv", index=False)

    # Build report
    top3 = fi.head(3).to_dict(orient="records")
    top3_str = ", ".join(
        f"{r['feature']}({r['gain_pct']:.1f}%)" for r in top3
    )
    report_lines = []
    report_lines.append("# Trade Classifier v1 — Report")
    report_lines.append("")
    report_lines.append(f"Candidate: **{CANDIDATE_NAME}** (FIFO realized survivor)")
    report_lines.append(f"Trades joined to features: **{len(df):,}**")
    report_lines.append(f"WF folds run: {len(fold_results)} (sliding, time-ordered trade rows)")
    report_lines.append("")
    report_lines.append(f"## OOS AUC (LightGBM): {auc_mean:.4f} ± {auc_std:.4f}")
    report_lines.append(f"OOS AUC (LogReg robustness): {auc_lr_mean:.4f}")
    report_lines.append("")
    report_lines.append("### Per-fold")
    report_lines.append("| fold | n_tr | n_va | n_te | auc_lgbm | auc_lr | base_rate | best_iter |")
    report_lines.append("|------|------|------|------|----------|--------|-----------|-----------|")
    for r in fold_results:
        report_lines.append(
            f"| {r['fold']} | {r['n_train']} | {r['n_val']} | {r['n_test']} | "
            f"{r['auc_lgbm']:.4f} | {r['auc_logreg']:.4f} | "
            f"{r['base_rate_test']:.3f} | {r['best_iter']} |"
        )

    report_lines.append("")
    report_lines.append("## Gated P&L Sweep (OOS only)")
    report_lines.append("| label | tau | n | tpt | wr | profit_days | total_days | pdays_ratio | sharpe |")
    report_lines.append("|-------|-----|---|-----|----|----|----|----|----|")
    for _, r in sweep.iterrows():
        tau_s = "—" if not np.isfinite(r["tau"]) else f"{r['tau']:.2f}"
        report_lines.append(
            f"| {r['label']} | {tau_s} | {int(r['n_trades'])} | "
            f"{r['tpt_pooled']:.3f} | {r['wr']:.3f} | "
            f"{int(r['profit_days'])} | {int(r['total_days'])} | "
            f"{r['profit_days_ratio']:.3f} | {r['sharpe']:.2f} |"
        )

    report_lines.append("")
    report_lines.append(f"## Verdict: **{verdict_str}**")
    if reasons:
        report_lines.append("Reasons:")
        for s in reasons:
            report_lines.append(f"- {s}")
    if best:
        report_lines.append("")
        report_lines.append(
            f"Best tau on OOS: **{best.get('tau')}** → tpt={best.get('tpt_pooled'):.3f}, "
            f"pdays_ratio={best.get('profit_days_ratio'):.3f}, "
            f"n_trades={int(best.get('n_trades'))}, sharpe={best.get('sharpe'):.2f}"
        )

    report_lines.append("")
    report_lines.append("## Top features (sum LGBM gain across folds)")
    report_lines.append(top3_str)
    report_lines.append("")
    report_lines.append("## Honest Caveats")
    report_lines.append(
        f"- {len(df):,} realized trades (short_10s @ thr=0.55) across "
        f"{df['date'].nunique()} OOT-day pool. Sample is large enough that LGBM "
        f"is fitting *trade-level* noise; this is in-sample-fold CV, not true "
        f"forward-only future."
    )
    report_lines.append(
        f"- The original 4 folds (from the meta booster) are NOT respected here. "
        f"Trade-classifier uses a NEW {N_WF_FOLDS}-fold sliding split on signal_ts_ns. "
        f"Each test slice is later in time than its train slice."
    )
    report_lines.append(
        f"- meta_prob is re-derived by calling the *original fold's* meta booster "
        f"on the meta-layer features at the signal event. So meta_prob entering "
        f"the trade classifier is causal w.r.t. the meta model that generated the trade."
    )
    report_lines.append(
        f"- Microstructure features used here are the 20 already-precomputed "
        f"meta_layer_v1 features (sign-consistency, OFI, spread, queue_imb at signal). "
        f"They were precomputed strictly causally upstream."
    )
    report_lines.append(
        f"- LGBM overfit profile differs from 15-day classifier: at thousands of "
        f"samples, gradient boosting can find genuine micro-edges but also memorize "
        f"trade-cluster signatures. LogReg AUC is the robustness anchor."
    )
    report_lines.append(
        f"- Per-trade alpha (baseline +5.03 t/trade FIFO realized, HC #428) is the "
        f"floor. Any tau that drops pooled tpt below baseline is rejected even if "
        f"profit-days ratio improves."
    )

    (OUT_DIR / "REPORT.md").write_text("\n".join(report_lines))

    elapsed = time.time() - t0
    (OUT_DIR / ".regen_complete.json").write_text(json.dumps({
        "candidate": CANDIDATE_NAME,
        "n_trades": int(len(df)),
        "wf_folds": len(fold_results),
        "auc_mean": auc_mean,
        "auc_std": auc_std,
        "auc_lr_mean": auc_lr_mean,
        "best": best,
        "verdict": verdict_str,
        "reasons": reasons,
        "elapsed_sec": elapsed,
        "top3_features": top3_str,
    }, indent=2))

    print(f"\n[done] {elapsed:.1f}s  verdict={verdict_str}  "
          f"auc={auc_mean:.4f}±{auc_std:.4f}  top3={top3_str}", flush=True)


if __name__ == "__main__":
    main()
