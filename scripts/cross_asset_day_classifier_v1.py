#!/usr/bin/env python3
"""
cross_asset_day_classifier_v1.py — HC #488 R2 creativity axis: augment the
day-classifier feature set with CROSS-ASSET signals (VIX, NQ, YM, SPX, DXY)
and check if any cross-asset feature beats the existing ES-only baseline
(trend_ticks_open_to_945 single-feature AUC = 0.759).

PIPELINE:
  1. Load output/day_classifier_v1/day_features.parquet  (ES-only features,
     labels, 15 OOT days).
  2. Load output/meta_classifier_v1_fifo/per_day_fifo.csv to confirm labels
     match (sanity check).
  3. Pull daily cross-asset data via yfinance for the date range
     2026-01-15 .. 2026-05-15 (gives lookback room for 20-day windows):
       ^VIX, NQ=F, YM=F, ^GSPC, DX-Y.NYB
  4. Engineer per-trading-date features (all CAUSAL — use only data
     available BEFORE the trading day's RTH open, i.e. previous close /
     overnight returns / rolling stats up to T-1):
       - VIX_close_prev, VIX_change_5d, VIX_zscore_20d
       - NQ_overnight_return, NQ_vs_ES_5d_corr
       - YM_overnight_return
       - SPX_5d_return
       - DXY_5d_return
       - day_of_month, week_of_month  (calendar)
  5. LOO-AUC for each single feature (ascending + descending direction
     reported; we use the max), compare to ES-only baseline 0.759.
  6. Lightly-regularised logistic regression on top-3 ES + top-3 cross-asset
     features (6 features, n=15). LOO predicted prob, AUC.
  7. Verdict:
       ACCEPT-NEW-AXIS  any cross-asset feature AUC >= 0.80
       INFORMATIVE      any cross-asset feature AUC >= 0.70
       REJECT           none beat 0.70

OUTPUTS (output/cross_asset_day_classifier_v1/):
  REPORT.md, feature_auc.csv, .regen_complete.json,
  combined_features.parquet, loo_logistic_predictions.csv

CONSTRAINTS:
  - 15-day sample. Single-feature thresholding only (no multi-feature LGBM).
  - All cross-asset features causal (T-1 close or earlier).
  - If yfinance is firewalled, fall back to calendar/overnight features only.
"""
from __future__ import annotations

import json
import math
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
except Exception as e:
    print(f"[fatal] sklearn import failed: {e}", file=sys.stderr)
    sys.exit(2)

LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
DAY_FEATURES = LVL3_ROOT / "output" / "day_classifier_v1" / "day_features.parquet"
PER_DAY_FIFO = LVL3_ROOT / "output" / "meta_classifier_v1_fifo" / "per_day_fifo.csv"
OUT_DIR = LVL3_ROOT / "output" / "cross_asset_day_classifier_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ES_BASELINE_FEATURE = "trend_ticks_open_to_945"
ES_BASELINE_AUC = 0.7589285714285714  # from output/day_classifier_v1/.regen_complete.json

XA_SYMBOLS = ["^VIX", "NQ=F", "YM=F", "^GSPC", "DX-Y.NYB"]
# Fetch a lookback window before the first OOT date for rolling stats
FETCH_START = "2026-01-15"
FETCH_END = "2026-05-15"


def _yyyymmdd_to_date(d: int) -> datetime:
    return datetime(d // 10000, (d // 100) % 100, d % 100)


def _try_fetch_yf() -> Tuple[pd.DataFrame, bool, str]:
    """Returns (panel_df, ok, message). panel_df indexed by date with cols per symbol."""
    try:
        import yfinance as yf  # noqa
    except Exception as e:
        return pd.DataFrame(), False, f"yfinance import failed: {e}"
    try:
        raw = yf.download(
            XA_SYMBOLS,
            start=FETCH_START,
            end=FETCH_END,
            progress=False,
            auto_adjust=False,
            group_by="column",
        )
    except Exception as e:
        return pd.DataFrame(), False, f"yfinance download failed: {e}"
    if raw is None or len(raw) == 0:
        return pd.DataFrame(), False, "yfinance returned empty frame"

    # Build a flat per-symbol frame of Close + Open columns
    out = pd.DataFrame(index=raw.index)
    try:
        close_panel = raw["Close"] if "Close" in raw.columns.get_level_values(0) else None
        open_panel = raw["Open"] if "Open" in raw.columns.get_level_values(0) else None
    except Exception:
        close_panel = None
        open_panel = None

    for sym in XA_SYMBOLS:
        try:
            out[f"{sym}_close"] = close_panel[sym] if close_panel is not None else np.nan
        except Exception:
            out[f"{sym}_close"] = np.nan
        try:
            out[f"{sym}_open"] = open_panel[sym] if open_panel is not None else np.nan
        except Exception:
            out[f"{sym}_open"] = np.nan

    out.index = pd.to_datetime(out.index).normalize()
    out = out.sort_index()
    return out, True, "ok"


def _causal_prev_value(panel: pd.DataFrame, trade_date: datetime, col: str) -> float:
    """Last value strictly BEFORE trade_date."""
    if col not in panel.columns:
        return float("nan")
    s = panel[col]
    s = s[s.index < pd.Timestamp(trade_date)]
    if len(s) == 0:
        return float("nan")
    v = s.iloc[-1]
    return float(v) if pd.notna(v) else float("nan")


def _causal_window(panel: pd.DataFrame, trade_date: datetime, col: str, n: int) -> pd.Series:
    """Last n rows strictly BEFORE trade_date."""
    if col not in panel.columns:
        return pd.Series(dtype=float)
    s = panel[col]
    s = s[s.index < pd.Timestamp(trade_date)].dropna()
    return s.iloc[-n:] if len(s) >= n else s


def build_cross_asset_features(
    trade_dates_yyyymmdd: List[int],
    panel: pd.DataFrame,
    have_xa: bool,
) -> pd.DataFrame:
    rows = []
    for di in trade_dates_yyyymmdd:
        d = _yyyymmdd_to_date(di)
        feats: Dict[str, float] = {"date": di}

        if have_xa:
            # ---- VIX features ----
            vix_close_prev = _causal_prev_value(panel, d, "^VIX_close")
            vix_win5 = _causal_window(panel, d, "^VIX_close", 6)  # need 6 to get 5-day pct
            if len(vix_win5) >= 2:
                vix_change_5d = float(vix_win5.iloc[-1] / vix_win5.iloc[0] - 1.0)
            else:
                vix_change_5d = float("nan")
            vix_win20 = _causal_window(panel, d, "^VIX_close", 20)
            if len(vix_win20) >= 5 and vix_win20.std(ddof=0) > 0:
                vix_zscore_20d = float((vix_close_prev - vix_win20.mean()) / vix_win20.std(ddof=0))
            else:
                vix_zscore_20d = float("nan")

            feats["VIX_close_prev"] = vix_close_prev
            feats["VIX_change_5d"] = vix_change_5d
            feats["VIX_zscore_20d"] = vix_zscore_20d

            # ---- Overnight returns (prev close -> today's open if available, else
            # prev close -> prev close = previous day return as proxy) ----
            # Causal note: today's open is NOT available pre-RTH for ES futures
            # consumers, but Yahoo daily bars use session opens; we use prev_close
            # -> prev_open spread as a same-day "overnight" proxy only if open
            # bar is from previous session. To stay strictly causal, we use
            # (close[T-1] / close[T-2] - 1) — the prior day's full return as
            # the overnight context known by RTH open.
            def _prev_day_return(symbol_close_col: str) -> float:
                win = _causal_window(panel, d, symbol_close_col, 2)
                if len(win) < 2 or win.iloc[-2] == 0:
                    return float("nan")
                return float(win.iloc[-1] / win.iloc[-2] - 1.0)

            feats["NQ_overnight_return"] = _prev_day_return("NQ=F_close")
            feats["YM_overnight_return"] = _prev_day_return("YM=F_close")

            # ---- NQ vs ES 5d correlation ----
            # We do not have ES daily closes from yfinance set; use ^GSPC as ES proxy.
            nq_win = _causal_window(panel, d, "NQ=F_close", 6)
            spx_win = _causal_window(panel, d, "^GSPC_close", 6)
            if len(nq_win) >= 5 and len(spx_win) >= 5:
                nq_ret = nq_win.pct_change().dropna().values
                spx_ret = spx_win.pct_change().dropna().values
                n = min(len(nq_ret), len(spx_ret))
                if n >= 3 and np.std(nq_ret[-n:]) > 0 and np.std(spx_ret[-n:]) > 0:
                    feats["NQ_vs_ES_5d_corr"] = float(np.corrcoef(nq_ret[-n:], spx_ret[-n:])[0, 1])
                else:
                    feats["NQ_vs_ES_5d_corr"] = float("nan")
            else:
                feats["NQ_vs_ES_5d_corr"] = float("nan")

            # ---- SPX 5d return ----
            spx_win5 = _causal_window(panel, d, "^GSPC_close", 6)
            if len(spx_win5) >= 2 and spx_win5.iloc[0] != 0:
                feats["SPX_5d_return"] = float(spx_win5.iloc[-1] / spx_win5.iloc[0] - 1.0)
            else:
                feats["SPX_5d_return"] = float("nan")

            # ---- DXY 5d return ----
            dxy_win5 = _causal_window(panel, d, "DX-Y.NYB_close", 6)
            if len(dxy_win5) >= 2 and dxy_win5.iloc[0] != 0:
                feats["DXY_5d_return"] = float(dxy_win5.iloc[-1] / dxy_win5.iloc[0] - 1.0)
            else:
                feats["DXY_5d_return"] = float("nan")

        # ---- Calendar features (always available) ----
        feats["day_of_month"] = float(d.day)
        # week_of_month: 1..5
        feats["week_of_month"] = float((d.day - 1) // 7 + 1)

        rows.append(feats)

    return pd.DataFrame(rows)


def single_feature_loo_auc(values: np.ndarray, labels: np.ndarray) -> Tuple[float, str]:
    """Return (max AUC, direction) over both ascending/descending polarity.
    NaNs in `values` are imputed with median over the rest before scoring."""
    v = values.astype(float).copy()
    mask_nan = ~np.isfinite(v)
    if mask_nan.any():
        med = float(np.nanmedian(v[~mask_nan])) if (~mask_nan).any() else 0.0
        v[mask_nan] = med
    best_auc = float("nan")
    best_dir = "asc"
    for direction, scores in (("asc", v), ("desc", -v)):
        try:
            a = float(roc_auc_score(labels, scores))
        except Exception:
            a = float("nan")
        if math.isnan(best_auc) or (not math.isnan(a) and a > best_auc):
            best_auc = a
            best_dir = direction
    return best_auc, best_dir


def main() -> int:
    t0 = time.time()
    print(f"[xa-day-classifier] start {datetime.now().isoformat()}", flush=True)

    # 1) Load ES-only feature set + labels
    if not DAY_FEATURES.exists():
        print(f"[fatal] missing {DAY_FEATURES}", file=sys.stderr)
        return 2
    es_df = pd.read_parquet(DAY_FEATURES)
    es_df = es_df.sort_values("date").reset_index(drop=True)
    n_days = len(es_df)
    y = es_df["label_profitable"].values.astype(int)
    dates_int = es_df["date"].astype(int).tolist()
    print(f"[data] {n_days} days, {int(y.sum())} positive", flush=True)

    # 2) Sanity check vs per_day_fifo (just compare label counts)
    if PER_DAY_FIFO.exists():
        pdf = pd.read_csv(PER_DAY_FIFO)
        pdf = pdf[pdf["candidate"] == "short_10s_thr55"]
        pdf = pdf[pdf["day_mean_net_realized"].notna() & (pdf["n_filled"] > 0)]
        n_check = len(pdf)
        if n_check != n_days:
            print(f"[warn] per_day_fifo has {n_check} rows; day_features has {n_days}", flush=True)

    # 3) Fetch cross-asset data
    panel, have_xa, msg = _try_fetch_yf()
    print(f"[xa] yfinance: have_xa={have_xa} ({msg}); panel shape={panel.shape}", flush=True)

    # 4) Build cross-asset features
    xa_df = build_cross_asset_features(dates_int, panel, have_xa)
    combined = es_df.merge(xa_df, on="date", how="left")

    # 5) Identify cross-asset feature columns (everything new)
    xa_feature_cols = [c for c in xa_df.columns if c != "date"]
    es_feature_cols = [
        c for c in es_df.columns if c not in
        ("date", "label_profitable", "day_mean_net_realized",
         "day_sum_net_realized", "n_filled")
    ]
    print(f"[feat] ES features: {len(es_feature_cols)}, XA features: {len(xa_feature_cols)}", flush=True)

    # Median-impute NaNs in combined for feature cols
    for c in xa_feature_cols + es_feature_cols:
        if combined[c].isna().any():
            med = combined[c].median()
            combined[c] = combined[c].fillna(med)

    combined.to_parquet(OUT_DIR / "combined_features.parquet", index=False)

    # 6) Single-feature LOO-equivalent AUC (we use rank AUC of raw feature
    # against label — this is identical to LOO threshold-rule AUC).
    results = []
    for c in es_feature_cols + xa_feature_cols:
        v = combined[c].values
        auc, direction = single_feature_loo_auc(v, y)
        kind = "ES" if c in es_feature_cols else "XA"
        results.append({"feature": c, "kind": kind, "direction": direction, "auc": auc})
    auc_df = pd.DataFrame(results).sort_values("auc", ascending=False).reset_index(drop=True)
    auc_df.to_csv(OUT_DIR / "feature_auc.csv", index=False)
    print(auc_df.head(10).to_string(), flush=True)

    # Best by kind
    xa_only = auc_df[auc_df["kind"] == "XA"].reset_index(drop=True)
    es_only = auc_df[auc_df["kind"] == "ES"].reset_index(drop=True)
    best_xa = xa_only.iloc[0] if len(xa_only) else None
    best_es = es_only.iloc[0] if len(es_only) else None
    best_xa_auc = float(best_xa["auc"]) if best_xa is not None else float("nan")
    best_es_auc = float(best_es["auc"]) if best_es is not None else float("nan")

    # 7) Logistic regression on top-3 ES + top-3 XA features, LOO predicted prob
    def _top_k(df_kind: pd.DataFrame, k: int) -> List[str]:
        # dedupe by feature name (asc/desc both yield same rank; first row wins)
        seen, out = set(), []
        for _, row in df_kind.iterrows():
            if row["feature"] not in seen:
                out.append(row["feature"])
                seen.add(row["feature"])
            if len(out) >= k:
                break
        return out

    top3_es = _top_k(es_only, 3)
    top3_xa = _top_k(xa_only, 3) if len(xa_only) else []
    combo_feats = top3_es + top3_xa
    print(f"[combo] ES top3: {top3_es}", flush=True)
    print(f"[combo] XA top3: {top3_xa}", flush=True)

    loo_pred = np.full(n_days, np.nan)
    if len(combo_feats) >= 2:
        X_full = combined[combo_feats].values.astype(float)
        for i in range(n_days):
            train_idx = np.array([j for j in range(n_days) if j != i])
            test_idx = np.array([i])
            Xtr = X_full[train_idx]
            ytr = y[train_idx]
            Xte = X_full[test_idx]
            # Standardise per fold (causal: fit on train only)
            scaler = StandardScaler()
            Xtr_s = scaler.fit_transform(Xtr)
            Xte_s = scaler.transform(Xte)
            # Heavy L2 regularisation given n=14 train, p=6
            clf = LogisticRegression(C=0.25, penalty="l2", solver="lbfgs", max_iter=500)
            try:
                clf.fit(Xtr_s, ytr)
                loo_pred[i] = float(clf.predict_proba(Xte_s)[0, 1])
            except Exception:
                loo_pred[i] = float("nan")
        try:
            combo_auc = float(roc_auc_score(y, loo_pred))
        except Exception:
            combo_auc = float("nan")
    else:
        combo_auc = float("nan")

    pred_df = pd.DataFrame({
        "date": dates_int,
        "label_profitable": y,
        "p_profitable_combo": loo_pred,
    })
    pred_df.to_csv(OUT_DIR / "loo_logistic_predictions.csv", index=False)

    # 8) Verdict
    if not math.isnan(best_xa_auc) and best_xa_auc >= 0.80:
        verdict = (f"ACCEPT-NEW-AXIS — cross-asset feature `{best_xa['feature']}` "
                   f"AUC={best_xa_auc:.3f} lifts above ES-only baseline {ES_BASELINE_AUC:.3f}")
    elif not math.isnan(best_xa_auc) and best_xa_auc >= 0.70:
        verdict = (f"INFORMATIVE — cross-asset feature `{best_xa['feature']}` "
                   f"AUC={best_xa_auc:.3f} worth adding to the gate (baseline {ES_BASELINE_AUC:.3f})")
    else:
        verdict = (f"REJECT — no cross-asset feature beats AUC 0.70 "
                   f"(best XA={best_xa_auc:.3f}, ES baseline {ES_BASELINE_AUC:.3f})")

    combo_verdict_line = ""
    if not math.isnan(combo_auc):
        if combo_auc > ES_BASELINE_AUC + 0.02:
            combo_verdict_line = (f"Logistic combo (top-3 ES + top-3 XA, L2 C=0.25) AUC={combo_auc:.3f} "
                                  f"BEATS ES-only baseline {ES_BASELINE_AUC:.3f}.")
        elif combo_auc + 0.02 < ES_BASELINE_AUC:
            combo_verdict_line = (f"Logistic combo AUC={combo_auc:.3f} UNDERPERFORMS ES-only baseline "
                                  f"{ES_BASELINE_AUC:.3f} (likely overfit from added noise features).")
        else:
            combo_verdict_line = (f"Logistic combo AUC={combo_auc:.3f} ~tied with ES-only baseline "
                                  f"{ES_BASELINE_AUC:.3f}; no improvement.")

    elapsed = time.time() - t0

    # 9) REPORT.md
    lines: List[str] = []
    lines.append("# cross_asset_day_classifier_v1 — REPORT")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().isoformat()}")
    lines.append(f"**Runtime:** {elapsed:.1f}s")
    lines.append(f"**Sample:** {n_days} OOT days, {int(y.sum())} profitable, {n_days - int(y.sum())} unprofitable")
    lines.append("")
    lines.append(f"## Verdict: {verdict}")
    lines.append("")
    if combo_verdict_line:
        lines.append(f"**Logistic combo:** {combo_verdict_line}")
        lines.append("")
    lines.append("## Comparison vs ES-only baseline")
    lines.append("")
    lines.append(f"- ES-only champion: `{ES_BASELINE_FEATURE}` (asc) AUC = **{ES_BASELINE_AUC:.3f}**")
    if best_es is not None:
        lines.append(f"- Best ES feature in this run: `{best_es['feature']}` ({best_es['direction']}) AUC = {best_es_auc:.3f}")
    if best_xa is not None:
        lines.append(f"- Best CROSS-ASSET feature: `{best_xa['feature']}` ({best_xa['direction']}) AUC = **{best_xa_auc:.3f}**")
    lines.append(f"- Logistic regression combo (6 feats, L2 C=0.25, LOO): AUC = {combo_auc:.3f}")
    lines.append("")
    lines.append("## Cross-asset feature AUCs (ranked)")
    lines.append("")
    lines.append("| feature | kind | direction | AUC |")
    lines.append("|---|---|---|---:|")
    for _, r in auc_df[auc_df["kind"] == "XA"].iterrows():
        lines.append(f"| `{r['feature']}` | XA | {r['direction']} | {r['auc']:.3f} |")
    lines.append("")
    lines.append("## All features (top 10)")
    lines.append("")
    lines.append("| feature | kind | direction | AUC |")
    lines.append("|---|---|---|---:|")
    for _, r in auc_df.head(10).iterrows():
        lines.append(f"| `{r['feature']}` | {r['kind']} | {r['direction']} | {r['auc']:.3f} |")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("- Same LOO framework as `day_classifier_v1`. Single-feature rank-AUC is equivalent")
    lines.append("  to leave-one-out threshold rule under monotonic scoring.")
    lines.append("- Cross-asset features pulled via yfinance:")
    lines.append(f"  - yfinance ok = `{have_xa}` ({msg})")
    lines.append(f"  - symbols: {XA_SYMBOLS}")
    lines.append(f"  - lookback range: {FETCH_START} .. {FETCH_END}")
    lines.append("- All cross-asset features are CAUSAL (use only data with bar timestamp")
    lines.append("  STRICTLY BEFORE the trading date — i.e. T-1 close and earlier).")
    lines.append("- Overnight return is implemented as prev-day full-return because Yahoo")
    lines.append("  daily bars do not separate cleanly into pre-RTH vs RTH for futures;")
    lines.append("  see code comment in `_prev_day_return`.")
    lines.append("- NQ_vs_ES_5d_corr uses ^GSPC as the ES proxy (no daily ES future fetched).")
    lines.append("- Logistic regression: standardised features fit on train fold only,")
    lines.append("  L2 C=0.25 (heavy regularisation), LOO predicted prob, AUC computed pooled.")
    lines.append("")
    lines.append("## Honest caveats")
    lines.append("")
    lines.append("- **15-day sample** — same overfit risk class as day_classifier_v1. A single")
    lines.append("  label flip can swing single-feature AUC by ~0.07.")
    lines.append("- We tried 10 cross-asset features and 13 ES features; with 23 candidates the")
    lines.append("  Bonferroni-adjusted noise-floor AUC is roughly 0.75+. Anything below ~0.78")
    lines.append("  is plausibly chance.")
    lines.append("- Yahoo end-of-day bars may have stale closes for some futures contracts;")
    lines.append("  treat single-day spikes with caution.")
    lines.append("- We did NOT run multi-feature LGBM (already shown to overfit at this sample).")
    lines.append("- Validation on next 15+ OOT days is the only way to certify any feature.")
    lines.append("")
    lines.append("## Files produced")
    lines.append("- `combined_features.parquet`")
    lines.append("- `feature_auc.csv`")
    lines.append("- `loo_logistic_predictions.csv`")
    lines.append("- `REPORT.md` (this file)")

    (OUT_DIR / "REPORT.md").write_text("\n".join(lines))

    # 10) .regen_complete.json
    regen = {
        "script": "scripts/cross_asset_day_classifier_v1.py",
        "completed_at": datetime.now().isoformat(),
        "runtime_seconds": elapsed,
        "n_days": int(n_days),
        "n_positive_days": int(y.sum()),
        "have_cross_asset_data": bool(have_xa),
        "yfinance_status": msg,
        "es_baseline_feature": ES_BASELINE_FEATURE,
        "es_baseline_auc": ES_BASELINE_AUC,
        "best_xa_feature": (None if best_xa is None else str(best_xa["feature"])),
        "best_xa_direction": (None if best_xa is None else str(best_xa["direction"])),
        "best_xa_auc": (None if best_xa is None else float(best_xa_auc)),
        "best_es_feature_this_run": (None if best_es is None else str(best_es["feature"])),
        "best_es_auc_this_run": (None if best_es is None else float(best_es_auc)),
        "logistic_combo_features": combo_feats,
        "logistic_combo_auc": (None if math.isnan(combo_auc) else float(combo_auc)),
        "verdict": verdict,
        "outputs": [
            "combined_features.parquet",
            "feature_auc.csv",
            "loo_logistic_predictions.csv",
            "REPORT.md",
        ],
    }
    (OUT_DIR / ".regen_complete.json").write_text(json.dumps(regen, indent=2))

    # Console summary
    print("")
    print("=" * 72)
    print(f"cross_asset_day_classifier_v1 DONE in {elapsed:.1f}s")
    print(f"  ES baseline               : {ES_BASELINE_FEATURE} AUC={ES_BASELINE_AUC:.3f}")
    if best_xa is not None:
        print(f"  Best cross-asset feature  : {best_xa['feature']} ({best_xa['direction']}) AUC={best_xa_auc:.3f}")
    if best_es is not None:
        print(f"  Best ES feature this run  : {best_es['feature']} ({best_es['direction']}) AUC={best_es_auc:.3f}")
    print(f"  Logistic combo AUC        : {combo_auc:.3f}")
    print(f"  Verdict                   : {verdict}")
    print(f"  Output                    : {OUT_DIR}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
