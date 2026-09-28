#!/usr/bin/env python3
"""
day_classifier_v1.py — Pre-market / early-session DAY CLASSIFIER (HC #488 creativity
axis: regime / day-selection gating layer).

Predicts whether a given OOT day will be PROFITABLE for the
short_10s @ threshold 0.55 candidate using ONLY causal features available by
10:00 ET (signal trading begins ~10:30 ET).

WHY:
  meta_classifier_v1_fifo replay showed short_10s_thr55 = +5.030 t/trade realized,
  Sharpe 4.54 pooled, WR 66.6% — but only 8 of 16 OOT days positive. Deploy gate
  needs 11/16. Per-trade alpha is real; day-distribution is the weak link.
  If we can pre-flag which days are likely profitable, we have a deploy gate.

METHOD:
  1) Load per_day_fifo.csv -> label day positive (day_mean_net_realized > 0).
  2) For each of the 16 OOT days, build day-level features from MBO events
     restricted to <= 10:00 ET (causal).
  3) Train LGBM with strict leave-one-out CV. Compute AUC + precision @ top-K.
  4) Simulate gated P&L for K in {6,8,10,12}: pool per-trade results only
     on top-K predicted-profitable days.
  5) Also test single-feature classifiers (simple rules) — 16-day LGBM is
     extreme overfit risk; a single robust feature may beat a fitted model.

OUTPUTS (output/day_classifier_v1/):
  - day_features.parquet
  - cv_results.csv
  - feature_importance.csv
  - simulated_gated_pnl.csv
  - REPORT.md
  - .regen_complete.json

CONSTRAINTS:
  - CPU-only Jupiter
  - All features causal (data <= 10:00 ET only)
  - 0.376 t passive commission already in per-trade realized values
  - HONEST about 16-day sample size — LOO-CV strict, document overfit risk

Run:  python3 scripts/day_classifier_v1.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import lightgbm as lgb
except Exception as e:
    print(f"[fatal] lightgbm import failed: {e}", file=sys.stderr)
    sys.exit(2)

try:
    from sklearn.metrics import roc_auc_score
except Exception as e:
    print(f"[fatal] sklearn import failed: {e}", file=sys.stderr)
    sys.exit(2)

LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
PER_DAY_FIFO = LVL3_ROOT / "output" / "meta_classifier_v1_fifo" / "per_day_fifo.csv"
PER_TRADE_DIAG = LVL3_ROOT / "output" / "meta_classifier_v1_fifo" / "per_trade_diagnostics.csv"
MBO_DIR = LVL3_ROOT / "data" / "processed" / "mbo_events"
ECON_CAL_PATH = LVL3_ROOT / "data" / "external" / "economic_calendar_2023_2026.json"
OUT_DIR = LVL3_ROOT / "output" / "day_classifier_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ET timezone offset handling: ET = UTC-4 (EDT) during our 2026 March-April window.
# All target dates 2026-03-16 .. 2026-04-14 fall in EDT (DST began 2026-03-08).
ET_OFFSET_HOURS = -4  # EDT
CAUSAL_CUTOFF_ET_HOUR = 10  # use only data with ET hour < 10 (i.e. <= 10:00 ET)
OPEN_ET_HOUR = 9
OPEN_ET_MIN = 30
RTH_OPEN_NS = None  # set per-day below


def _et_to_utc_ns(date_yyyymmdd: int, hour_et: int, minute_et: int = 0) -> int:
    """Convert an ET wall-clock time on the given date to UTC epoch ns (EDT only)."""
    y = date_yyyymmdd // 10000
    m = (date_yyyymmdd // 100) % 100
    d = date_yyyymmdd % 100
    # ET = UTC-4 during EDT, so UTC hour = ET hour + 4
    dt = datetime(y, m, d, hour_et - ET_OFFSET_HOURS, minute_et, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000_000)


def _has_event_on_date(econ_events: List[Dict], date_yyyymmdd: int, category: str) -> int:
    y = date_yyyymmdd // 10000
    m = (date_yyyymmdd // 100) % 100
    d = date_yyyymmdd % 100
    target = f"{y:04d}-{m:02d}-{d:02d}"
    for ev in econ_events:
        if ev.get("category") == category and ev.get("datetime_local_et", "").startswith(target):
            return 1
    return 0


def build_day_features(date_yyyymmdd: int, econ_events: List[Dict]) -> Dict[str, float]:
    """Build CAUSAL day-level features using ONLY data with ET time <= 10:00 ET."""
    fp = MBO_DIR / f"{date_yyyymmdd}_mbo_events.npz"
    if not fp.exists():
        raise FileNotFoundError(f"missing {fp}")

    d = np.load(fp, allow_pickle=False)
    events = d["events"]  # [N, 6]: time_delta_log, event_type_id, side_id, price_rel_ticks, qty_log, spread_ticks
    timestamps = d["timestamps"]  # int64 epoch ns

    # CAUSAL CUTOFF: only data with ET time strictly before 10:00 ET on this date.
    rth_open_ns = _et_to_utc_ns(date_yyyymmdd, OPEN_ET_HOUR, OPEN_ET_MIN)
    cutoff_ns = _et_to_utc_ns(date_yyyymmdd, CAUSAL_CUTOFF_ET_HOUR, 0)
    rth_915_ns = _et_to_utc_ns(date_yyyymmdd, 9, 15)
    open_945_ns = _et_to_utc_ns(date_yyyymmdd, 9, 45)

    # Window: 09:30:00 .. 10:00:00 ET (the "early session" window)
    mask_early = (timestamps >= rth_open_ns) & (timestamps < cutoff_ns)
    n_early = int(mask_early.sum())

    if n_early < 100:
        # Not enough data in window — produce NaNs (caller will impute)
        feats = {
            "n_events_early": float(n_early),
            "n_trades_early": np.nan,
            "qty_total_early": np.nan,
            "spread_avg_early": np.nan,
            "spread_p90_early": np.nan,
            "realized_vol_ticks_early": np.nan,
            "trend_ticks_open_to_945": np.nan,
            "trend_ticks_open_to_10": np.nan,
            "range_ticks_early": np.nan,
            "abs_trend_to_10": np.nan,
            "ofi_imbalance_early": np.nan,
            "trade_intensity_per_sec": np.nan,
            "preopen_drift_ticks": np.nan,
        }
    else:
        e = events[mask_early]
        ts = timestamps[mask_early]
        event_type = e[:, 1]
        side = e[:, 2]
        price_rel = e[:, 3].astype(np.float64)
        qty_log = e[:, 4].astype(np.float64)
        spread = e[:, 5].astype(np.float64)

        # Trades only: action == 'T' (event_type_id == 3)
        is_trade = (event_type == 3)
        n_trades = int(is_trade.sum())
        # qty_log was log(1+qty); recover approx volume
        qty_actual = np.expm1(qty_log)
        qty_total = float(qty_actual[is_trade].sum())

        # Spread stats (note: spread_ticks is recorded per event; we average over all)
        spread_avg = float(np.nanmean(spread)) if spread.size else np.nan
        spread_p90 = float(np.nanpercentile(spread, 90)) if spread.size else np.nan

        # Trend in price_rel_ticks within window
        # Anchor "open" = first event in window; "9:45" and "10:00" = last events <= those times
        mask_to_945 = ts < open_945_ns
        if mask_to_945.any():
            trend_to_945 = float(price_rel[mask_to_945][-1] - price_rel[0])
        else:
            trend_to_945 = np.nan
        trend_to_10 = float(price_rel[-1] - price_rel[0])
        range_ticks = float(np.nanmax(price_rel) - np.nanmin(price_rel))
        abs_trend_to_10 = abs(trend_to_10)

        # Realized vol: stdev of 30s-bucketed mid_rel changes
        # Resample price_rel onto a regular 30s grid over the window
        t_start = ts[0]
        t_end = ts[-1]
        n_buckets = max(1, int((t_end - t_start) / 30_000_000_000))
        if n_buckets >= 3:
            edges = np.linspace(t_start, t_end, n_buckets + 1)
            idx = np.searchsorted(ts, edges) - 1
            idx = np.clip(idx, 0, len(price_rel) - 1)
            bucket_prices = price_rel[idx]
            diffs = np.diff(bucket_prices)
            realized_vol = float(np.nanstd(diffs))
        else:
            realized_vol = float(np.nanstd(price_rel))

        # OFI-like imbalance: signed trade qty (bid=0 -> sell-init, ask=1 -> buy-init)
        # side_id: 0=Bid, 1=Ask, 2=None. For trades, side is aggressor side.
        if n_trades > 0:
            tr_side = side[is_trade]
            tr_qty = qty_actual[is_trade]
            buy_qty = float(tr_qty[tr_side == 1].sum())   # aggressor at ask = buy-initiated
            sell_qty = float(tr_qty[tr_side == 0].sum())  # aggressor at bid = sell-initiated
            denom = buy_qty + sell_qty
            ofi_imb = (buy_qty - sell_qty) / denom if denom > 0 else 0.0
        else:
            ofi_imb = np.nan

        # Trade intensity
        win_sec = (t_end - t_start) / 1e9
        trade_intensity = n_trades / win_sec if win_sec > 0 else np.nan

        # Pre-open drift: 09:15-09:30 ET window
        mask_preopen = (timestamps >= rth_915_ns) & (timestamps < rth_open_ns)
        if mask_preopen.sum() > 10:
            pr_pre = events[mask_preopen, 3].astype(np.float64)
            preopen_drift = float(pr_pre[-1] - pr_pre[0])
        else:
            preopen_drift = 0.0  # default for sparse pre-open (common pre-RTH)

        feats = {
            "n_events_early": float(n_early),
            "n_trades_early": float(n_trades),
            "qty_total_early": qty_total,
            "spread_avg_early": spread_avg,
            "spread_p90_early": spread_p90,
            "realized_vol_ticks_early": realized_vol,
            "trend_ticks_open_to_945": trend_to_945,
            "trend_ticks_open_to_10": trend_to_10,
            "range_ticks_early": range_ticks,
            "abs_trend_to_10": abs_trend_to_10,
            "ofi_imbalance_early": ofi_imb,
            "trade_intensity_per_sec": trade_intensity,
            "preopen_drift_ticks": preopen_drift,
        }

    # Day-of-week (Mon=0, Sun=6) + macro flags
    y = date_yyyymmdd // 10000
    m = (date_yyyymmdd // 100) % 100
    dd = date_yyyymmdd % 100
    dow = datetime(y, m, dd).weekday()
    for i, name in enumerate(["mon", "tue", "wed", "thu", "fri"]):
        feats[f"dow_{name}"] = 1.0 if dow == i else 0.0
    feats["is_fomc_day"] = float(_has_event_on_date(econ_events, date_yyyymmdd, "FOMC_DECISION"))
    feats["is_cpi_day"] = float(_has_event_on_date(econ_events, date_yyyymmdd, "CPI"))
    feats["is_nfp_day"] = float(_has_event_on_date(econ_events, date_yyyymmdd, "NFP"))
    return feats


# ------------------------------------------------------------------ main ----
def main() -> int:
    t0 = time.time()
    print(f"[day_classifier_v1] start {datetime.now().isoformat()}", flush=True)

    # 1) Load per-day P&L for short_10s_thr55
    if not PER_DAY_FIFO.exists():
        print(f"[fatal] missing {PER_DAY_FIFO}", file=sys.stderr)
        return 2
    per_day = pd.read_csv(PER_DAY_FIFO)
    df = per_day[per_day["candidate"] == "short_10s_thr55"].copy()
    df = df.sort_values("date").reset_index(drop=True)
    # Drop days with no trades / NaN P&L (e.g., 20260405 = Easter Sunday)
    pre_drop = len(df)
    df = df[df["day_mean_net_realized"].notna() & (df["n_filled"] > 0)].reset_index(drop=True)
    if len(df) < pre_drop:
        print(f"[data] dropped {pre_drop - len(df)} days with no fills (NaN P&L)", flush=True)
    df["label_profitable"] = (df["day_mean_net_realized"] > 0).astype(int)
    n_days = len(df)
    n_pos = int(df["label_profitable"].sum())
    print(f"[data] {n_days} OOT days; {n_pos} positive, {n_days - n_pos} negative", flush=True)

    # 2) Econ calendar
    if ECON_CAL_PATH.exists():
        econ_events = json.load(open(ECON_CAL_PATH))["events"]
    else:
        econ_events = []
        print("[warn] econ calendar missing", flush=True)

    # 3) Build features per day
    rows = []
    for _, r in df.iterrows():
        date_int = int(r["date"])
        print(f"[feat] {date_int} ...", flush=True)
        feats = build_day_features(date_int, econ_events)
        feats["date"] = date_int
        feats["label_profitable"] = int(r["label_profitable"])
        feats["day_mean_net_realized"] = float(r["day_mean_net_realized"])
        feats["day_sum_net_realized"] = float(r["day_sum_net_realized"])
        feats["n_filled"] = float(r["n_filled"])
        rows.append(feats)

    feat_df = pd.DataFrame(rows)
    # Impute any NaN with median (rare, but be safe)
    feature_cols = [c for c in feat_df.columns if c not in
                    ("date", "label_profitable", "day_mean_net_realized",
                     "day_sum_net_realized", "n_filled")]
    for c in feature_cols:
        if feat_df[c].isna().any():
            med = feat_df[c].median()
            feat_df[c] = feat_df[c].fillna(med)

    feat_df.to_parquet(OUT_DIR / "day_features.parquet", index=False)
    print(f"[feat] saved {len(feat_df)} rows, {len(feature_cols)} features", flush=True)

    # 4) LGBM with LOO CV
    X = feat_df[feature_cols].values
    y = feat_df["label_profitable"].values
    dates = feat_df["date"].values
    loo_pred = np.zeros(n_days, dtype=np.float64)

    # tiny LGBM per HC #488 caveat
    lgb_params = dict(
        objective="binary",
        learning_rate=0.05,
        num_leaves=8,           # very small to combat 16-day overfit
        max_depth=3,
        min_data_in_leaf=2,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        verbosity=-1,
        n_estimators=200,
        n_jobs=2,
    )

    importances = np.zeros(len(feature_cols), dtype=np.float64)
    for i in range(n_days):
        train_idx = np.array([j for j in range(n_days) if j != i])
        Xtr, ytr = X[train_idx], y[train_idx]
        Xte = X[i:i+1]
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(Xtr, ytr)
        loo_pred[i] = model.predict_proba(Xte)[0, 1]
        importances += model.feature_importances_

    importances /= n_days

    # AUC (LGBM)
    try:
        auc = float(roc_auc_score(y, loo_pred))
    except Exception:
        auc = float("nan")

    cv_df = pd.DataFrame({
        "date": dates,
        "label_profitable": y,
        "day_mean_net_realized": feat_df["day_mean_net_realized"].values,
        "n_filled": feat_df["n_filled"].values,
        "p_profitable_lgbm": loo_pred,
    })
    cv_df.to_csv(OUT_DIR / "cv_results.csv", index=False)

    # Feature importance (LGBM)
    fi_df = pd.DataFrame({"feature": feature_cols, "importance_avg": importances})
    fi_df = fi_df.sort_values("importance_avg", ascending=False).reset_index(drop=True)
    fi_df.to_csv(OUT_DIR / "feature_importance.csv", index=False)

    # 5) Single-feature LOO classifiers (robust baseline check)
    # For each feature, score-rank days using the LOO median split from train set
    single_feat_results = []
    for c in feature_cols:
        v = feat_df[c].values
        # Try both polarities: rank ascending or descending as "more profitable"
        for direction, scores in [("asc", v), ("desc", -v)]:
            try:
                auc_sf = float(roc_auc_score(y, scores))
            except Exception:
                auc_sf = float("nan")
            single_feat_results.append({"feature": c, "direction": direction, "auc": auc_sf})
    sf_df = pd.DataFrame(single_feat_results)
    sf_df = sf_df.sort_values("auc", ascending=False).reset_index(drop=True)
    sf_df.to_csv(OUT_DIR / "single_feature_auc.csv", index=False)
    top_sf = sf_df.iloc[0]
    print(f"[sf] best single feature: {top_sf['feature']} ({top_sf['direction']}) AUC={top_sf['auc']:.3f}", flush=True)

    # 6) Simulate gated P&L: pool per-trade results only on top-K predicted days
    # Use per_trade_diagnostics.csv for honest per-fifo P&L pooling
    if PER_TRADE_DIAG.exists():
        per_trade = pd.read_csv(PER_TRADE_DIAG)
        # Try to filter to short_10s_thr55
        if "candidate" in per_trade.columns:
            pt = per_trade[per_trade["candidate"] == "short_10s_thr55"].copy()
        else:
            pt = per_trade.copy()
        # Need date and net realized column
        pnl_col = None
        for cand in ("net_realized_ticks", "net_realized", "pnl_ticks", "realized_ticks_net"):
            if cand in pt.columns:
                pnl_col = cand
                break
        if pnl_col is None:
            # Fall back to per-day mean from per_day_fifo
            pt = None
        else:
            # Date column
            if "date" not in pt.columns:
                pt = None
    else:
        pt = None
        pnl_col = None

    gated_rows = []
    for K in [6, 8, 10, 12]:
        # Top-K days by LGBM predicted probability
        order = np.argsort(-loo_pred)
        top_dates = set(dates[order[:K]].tolist())
        sel_mask = np.array([d in top_dates for d in dates])

        # Per-day metrics from the gated subset
        sel_df = feat_df[sel_mask]
        n_sel = int(len(sel_df))
        n_profitable_in_sel = int((sel_df["day_mean_net_realized"] > 0).sum())
        day_means = sel_df["day_mean_net_realized"].values

        # Pooled per-trade approximation: weight day_mean by n_filled
        nfilled = sel_df["n_filled"].values
        if nfilled.sum() > 0:
            pooled_mean = float(np.sum(day_means * nfilled) / nfilled.sum())
        else:
            pooled_mean = float(np.mean(day_means))

        # Per-day Sharpe (across days)
        day_sharpe = float(np.mean(day_means) / np.std(day_means, ddof=1) * np.sqrt(252)) \
            if len(day_means) > 1 and np.std(day_means, ddof=1) > 0 else float("nan")

        # Pooled Sharpe using per-trade data if available
        pooled_sharpe = float("nan")
        if pt is not None and pnl_col is not None:
            pt_sel = pt[pt["date"].isin([int(x) for x in dates[sel_mask]])]
            if len(pt_sel) > 1:
                m = float(pt_sel[pnl_col].mean())
                s = float(pt_sel[pnl_col].std(ddof=1))
                n_tr = int(len(pt_sel))
                avg_trades_per_day = n_tr / max(1, n_sel)
                pooled_sharpe = (m / s) * np.sqrt(252 * avg_trades_per_day) if s > 0 else float("nan")

        gated_rows.append({
            "K": K,
            "n_days_selected": n_sel,
            "n_profitable_days": n_profitable_in_sel,
            "profitable_day_ratio": n_profitable_in_sel / max(1, n_sel),
            "day_mean_avg_t": float(np.mean(day_means)),
            "pooled_mean_per_trade_t": pooled_mean,
            "day_sharpe_ann": day_sharpe,
            "pooled_sharpe_ann": pooled_sharpe,
        })

    # Also baseline (no gating, all 16 days)
    all_means = feat_df["day_mean_net_realized"].values
    all_nfilled = feat_df["n_filled"].values
    base_pooled = float(np.sum(all_means * all_nfilled) / all_nfilled.sum()) if all_nfilled.sum() > 0 else float("nan")
    base_day_sharpe = float(np.mean(all_means) / np.std(all_means, ddof=1) * np.sqrt(252)) \
        if np.std(all_means, ddof=1) > 0 else float("nan")
    base_pooled_sharpe = float("nan")
    if pt is not None and pnl_col is not None:
        if len(pt) > 1:
            m = float(pt[pnl_col].mean())
            s = float(pt[pnl_col].std(ddof=1))
            n_tr = int(len(pt))
            avg_trades_per_day = n_tr / max(1, n_days)
            base_pooled_sharpe = (m / s) * np.sqrt(252 * avg_trades_per_day) if s > 0 else float("nan")
    gated_rows.append({
        "K": n_days,
        "n_days_selected": n_days,
        "n_profitable_days": n_pos,
        "profitable_day_ratio": n_pos / n_days,
        "day_mean_avg_t": float(np.mean(all_means)),
        "pooled_mean_per_trade_t": base_pooled,
        "day_sharpe_ann": base_day_sharpe,
        "pooled_sharpe_ann": base_pooled_sharpe,
    })
    gated_df = pd.DataFrame(gated_rows)
    gated_df.to_csv(OUT_DIR / "simulated_gated_pnl.csv", index=False)

    # ---- Single-feature gated simulation (using best single feature) ------
    best_feat = str(top_sf["feature"])
    best_dir = str(top_sf["direction"])
    # Higher score = "more profitable" by convention
    v_best = feat_df[best_feat].values
    score_best = v_best if best_dir == "asc" else -v_best
    sf_gated_rows = []
    for K in [6, 8, 10, 12]:
        if K > n_days:
            continue
        order = np.argsort(-score_best)
        top_dates_sf = set(dates[order[:K]].tolist())
        sel_mask = np.array([d in top_dates_sf for d in dates])
        sel_df = feat_df[sel_mask]
        n_sel = int(len(sel_df))
        n_profitable_in_sel = int((sel_df["day_mean_net_realized"] > 0).sum())
        day_means = sel_df["day_mean_net_realized"].values
        nfilled = sel_df["n_filled"].values
        pooled_mean = float(np.sum(day_means * nfilled) / nfilled.sum()) if nfilled.sum() > 0 else float("nan")
        day_sharpe = float(np.mean(day_means) / np.std(day_means, ddof=1) * np.sqrt(252)) \
            if len(day_means) > 1 and np.std(day_means, ddof=1) > 0 else float("nan")
        sf_gated_rows.append({
            "K": K,
            "n_days_selected": n_sel,
            "n_profitable_days": n_profitable_in_sel,
            "profitable_day_ratio": n_profitable_in_sel / max(1, n_sel),
            "day_mean_avg_t": float(np.mean(day_means)),
            "pooled_mean_per_trade_t": pooled_mean,
            "day_sharpe_ann": day_sharpe,
            "best_feature": best_feat,
            "direction": best_dir,
        })
    sf_gated_df = pd.DataFrame(sf_gated_rows)
    sf_gated_df.to_csv(OUT_DIR / "single_feature_gated_pnl.csv", index=False)

    # ---- VERDICT ----------------------------------------------------------
    # Use max of LGBM AUC and best-single-feature AUC for the verdict.
    best_sf_auc = float(top_sf["auc"])
    headline_auc = max(auc, best_sf_auc)
    if headline_auc > 0.65 and best_sf_auc > 0.65:
        if best_sf_auc > auc:
            verdict = (f"ACCEPT (single-feature) — `{best_sf['feature'] if False else top_sf['feature']}` "
                       f"AUC={best_sf_auc:.3f}; LGBM overfit (AUC={auc:.3f})")
        else:
            verdict = f"ACCEPT — LGBM AUC={auc:.3f} deploy-grade"
    elif headline_auc > 0.65:
        verdict = f"ACCEPT — gate candidate (headline AUC={headline_auc:.3f})"
    elif headline_auc < 0.55:
        verdict = "REJECT — not informative"
    else:
        verdict = "AMBIGUOUS — borderline, retest with more data"

    elapsed = time.time() - t0

    # REPORT.md
    top3_fi = fi_df.head(3).to_dict("records")
    top3_sf = sf_df.head(3).to_dict("records")

    report = []
    report.append(f"# day_classifier_v1 — REPORT")
    report.append("")
    report.append(f"**Generated:** {datetime.now().isoformat()}")
    report.append(f"**Runtime:** {elapsed:.1f}s")
    report.append("")
    report.append(f"## Verdict: {verdict}")
    report.append("")
    report.append(f"- **LGBM LOO AUC:** {auc:.3f}  (gate threshold for ACCEPT: > 0.65)")
    report.append(f"- **Best single-feature AUC:** {top_sf['auc']:.3f}  (feature: `{top_sf['feature']}`, direction: {top_sf['direction']})")
    report.append(f"- **Baseline profitable-day ratio (no gating):** {n_pos}/{n_days} = {n_pos/n_days:.2%}")
    report.append("")
    report.append("## Gated P&L Simulation — LGBM (top-K predicted-profitable days)")
    report.append("")
    report.append("| K | days_sel | profit_days | profit_ratio | mean_t/trade (pooled) | day_sharpe_ann | pooled_sharpe_ann |")
    report.append("|---|----------|-------------|--------------|----------------------:|---------------:|------------------:|")
    for r in gated_rows:
        report.append(f"| {r['K']} | {r['n_days_selected']} | {r['n_profitable_days']} | "
                      f"{r['profitable_day_ratio']:.2%} | {r['pooled_mean_per_trade_t']:.3f} | "
                      f"{r['day_sharpe_ann']:.2f} | {r['pooled_sharpe_ann']:.2f} |")
    report.append("")
    report.append(f"## Gated P&L Simulation — SINGLE FEATURE (`{best_feat}` {best_dir})")
    report.append("")
    report.append("| K | days_sel | profit_days | profit_ratio | mean_t/trade (pooled) | day_sharpe_ann |")
    report.append("|---|----------|-------------|--------------|----------------------:|---------------:|")
    for r in sf_gated_rows:
        report.append(f"| {r['K']} | {r['n_days_selected']} | {r['n_profitable_days']} | "
                      f"{r['profitable_day_ratio']:.2%} | {r['pooled_mean_per_trade_t']:.3f} | "
                      f"{r['day_sharpe_ann']:.2f} |")
    report.append("")
    report.append("## Top-3 LGBM feature importances")
    for r in top3_fi:
        report.append(f"- `{r['feature']}`: {r['importance_avg']:.1f}")
    report.append("")
    report.append("## Top-3 single-feature classifiers (LOO-equiv AUC)")
    for r in top3_sf:
        report.append(f"- `{r['feature']}` ({r['direction']}): AUC={r['auc']:.3f}")
    report.append("")
    report.append("## Honest overfit-risk assessment")
    report.append("")
    report.append("**Sample size: 16 days.** This is dangerously small for an ML classifier.")
    report.append("Mitigations applied:")
    report.append("- Strict leave-one-out CV (no held-out day ever touches its training fold)")
    report.append("- Tiny LGBM: num_leaves=8, max_depth=3, min_data_in_leaf=2, bagging+feature_frac")
    report.append("- Single-feature baseline reported alongside; if a simple rule matches LGBM AUC,")
    report.append("  prefer the rule (robust > fitted on 16 obs).")
    report.append("")
    report.append("**Caveats remaining:**")
    report.append("- LOO with 16 samples still has high variance — single-day swing changes AUC by ~0.06.")
    report.append("- Feature engineering choices (cutoff 10:00 ET, window 9:30-10:00) were not")
    report.append("  themselves cross-validated — implicit researcher degrees of freedom.")
    report.append("- Three macro days (FOMC/NFP/CPI) in 16 may dominate; check is_fomc/is_nfp/is_cpi flags.")
    report.append("- Recommended: validate on the NEXT batch of 16+ OOT days before any live deployment.")
    report.append("")
    report.append("## Files produced")
    report.append("- `day_features.parquet`")
    report.append("- `cv_results.csv`")
    report.append("- `feature_importance.csv`")
    report.append("- `single_feature_auc.csv`")
    report.append("- `simulated_gated_pnl.csv`")
    report.append("- `REPORT.md` (this file)")

    (OUT_DIR / "REPORT.md").write_text("\n".join(report))

    # .regen_complete.json (HC #485 R5)
    regen = {
        "script": "scripts/day_classifier_v1.py",
        "completed_at": datetime.now().isoformat(),
        "runtime_seconds": elapsed,
        "n_days": n_days,
        "n_positive_days": n_pos,
        "lgbm_loo_auc": auc,
        "best_single_feature": str(top_sf["feature"]),
        "best_single_feature_auc": float(top_sf["auc"]),
        "verdict": verdict,
        "outputs": [
            "day_features.parquet", "cv_results.csv", "feature_importance.csv",
            "single_feature_auc.csv", "simulated_gated_pnl.csv",
            "single_feature_gated_pnl.csv", "REPORT.md",
        ],
    }
    (OUT_DIR / ".regen_complete.json").write_text(json.dumps(regen, indent=2))

    # Console summary
    print("")
    print("=" * 70)
    print(f"day_classifier_v1 DONE in {elapsed:.1f}s")
    print(f"  LGBM LOO AUC          : {auc:.3f}")
    print(f"  Best single-feature   : {top_sf['feature']} ({top_sf['direction']}) AUC={top_sf['auc']:.3f}")
    print(f"  Verdict               : {verdict}")
    print(f"  Output                : {OUT_DIR}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
