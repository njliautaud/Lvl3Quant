"""
K=6 Megacap-Tech Daily-Edge Meta-Classifier (HC #589 A2).

Goal: train an LGBM that predicts whether the K=6 megacap-tech rotation
portfolio will have a positive next-day return conditioned on macro/regime
features.  Then backtest the GATED variant: trade normally when meta predicts
edge, hold cash when meta predicts loss.

This script WRAPS the existing K=6 backtest (megacap_tech_extended_v1.py /
megacap_tech_rotation.py).  It does NOT modify the K=6 harness.  It loads the
already-computed K=6 daily book and overlays a meta-classifier gate built from
OOT predictions only (sliding 24m train / 6m OOT / 3m step, matching HC #0).

Features (all lagged 1 day from prediction day -> NO look-ahead):
  Macro/regime:
    vix_level, vix_chg_5d, vix_chg_10d, vix_chg_20d
    vix_term (VIX/VIX3M)
    yield_curve_10y2y (FRED T10Y2Y) + 30d change
    dxy_level + 20d mom
  SPY structure:
    spy_z_ma20, spy_z_ma60, spy_z_ma200
    spy_rv_20
  K=6 basket:
    k6_rv_20, k6_ret_20d (basket realized vol + 20d return)
  Sector breadth:
    sector_mom_median (median 20d return across 9 sector ETFs)
    sector_mom_dispersion (stdev of 20d returns across sectors)

Targets:
  y_clf : sign(next-day K=6 book daily_ret)  (binary: 1 if >0 else 0)
  y_reg : next-day K=6 book daily_ret        (continuous, for sizing later)

Walk-forward:  sliding 24m train / 6m OOT / 3m step, 2018-2025 (HC #0)
Model: LGBM binary classifier with early stopping on the last 10% of train.
Gate:  on OOT days where predicted P(positive) < 0.45 -> override book.daily_ret = 0
       (the strategy "skips" the day and sits in cash)

Outputs:
  output/macro_picker/k6_meta_classifier_v1/
    feature_matrix.parquet
    oot_predictions.parquet
    gated_book.parquet
    feature_importance.csv
    report.json
  research/findings/k6_meta_classifier_v1.md
"""
from __future__ import annotations
import json
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "strategy" / "macro_picker"))

from walk_forward import _metrics  # type: ignore
from megacap_tech_rotation import (  # type: ignore
    UNIVERSE, BENCH_SPY, TRAIN_MONTHS, OOT_MONTHS, STEP_MONTHS,
    WF_START, WF_END, _load_prices, _load_vix, _spy_regime, _vix_ok,
    _combined_regime, _iter_wf, regime_stratification, deploy_verdict,
)
from megacap_tech_extended_v1 import tail_dd_metric

TRADING_DAYS = 252
PRICE_PATH = ROOT / "wheel_strategy_v1/data/cache/prices_v2.parquet"

# K=6 baseline book (pre-computed by megacap_tech_extended_v1)
K6_BOOK_PATH = ROOT / "output/macro_picker/megacap_tech_extended_v1_20260609_171903/book_K6_mom60.parquet"

# Out dir
OUT_DIR = ROOT / "output/macro_picker/k6_meta_classifier_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# K=6 universe (overrides the 8-name universe in megacap_tech_rotation)
K6_UNIVERSE = ["META", "AVGO", "TSLA", "MSFT", "AAPL", "NVDA"]

SECTORS = ["XLK", "XLF", "XLE", "XLY", "XLP", "XLV", "XLI", "XLB", "XLU"]

# Gate threshold (HC dispatch spec)
GATE_THRESH = 0.45


# ---------------------------------------------------------------------------
# Data loading helpers (extends megacap loaders to fetch extra macro series)
# ---------------------------------------------------------------------------
def _load_yf_close(ticker: str, start: pd.Timestamp, end: pd.Timestamp,
                   cache_name: str | None = None) -> pd.Series:
    """Fetch a single yfinance Close series, return as date-indexed Series."""
    cache_dir = OUT_DIR / "_cache"
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{cache_name or ticker.replace('^','').replace('=','_')}.parquet"
    if cache_file.exists():
        s = pd.read_parquet(cache_file)
        return s.set_index("date")["close"].sort_index()

    import yfinance as yf
    df = yf.Ticker(ticker).history(
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
    )
    df = df.reset_index()[["Date", "Close"]].rename(columns={"Date": "date", "Close": "close"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df.to_parquet(cache_file, index=False)
    return df.set_index("date")["close"].sort_index()


def _load_fred_t10y2y(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Load FRED T10Y2Y series via fredgraph CSV (no API key needed)."""
    cache_dir = OUT_DIR / "_cache"
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / "T10Y2Y.parquet"
    if cache_file.exists():
        s = pd.read_parquet(cache_file)
        return s.set_index("date")["t10y2y"].sort_index()

    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y2Y"
    try:
        df = pd.read_csv(url)
        df.columns = ["date", "t10y2y"]
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["t10y2y"] = pd.to_numeric(df["t10y2y"], errors="coerce")
        df = df.dropna()
        df = df[(df["date"] >= start - pd.Timedelta(days=400)) &
                (df["date"] <= end + pd.Timedelta(days=5))]
        df.to_parquet(cache_file, index=False)
        return df.set_index("date")["t10y2y"].sort_index()
    except Exception as e:
        print(f"[meta] FRED load failed ({e}); using 10Y-2Y proxy from ^TNX-^FVX zero")
        # Fallback: use ^TNX minus ^FVX (rough; 5Y not 2Y but correlated)
        tnx = _load_yf_close("^TNX", start, end, "TNX")
        fvx = _load_yf_close("^FVX", start, end, "FVX")
        idx = tnx.index.union(fvx.index)
        return (tnx.reindex(idx).ffill() - fvx.reindex(idx).ffill()).rename("t10y2y")


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------
def build_feature_matrix(prices: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    """Build the daily feature matrix used by the meta-classifier.

    Every feature is computed from data up to and including day t-1; the target
    is day-t K=6 portfolio return.  Lag is applied at the END.
    """
    fxr_start = WF_START - pd.Timedelta(days=400)
    fxr_end = WF_END + pd.Timedelta(days=5)

    # SPY close (from local cache)
    spy = prices[prices["ticker"] == BENCH_SPY].set_index("date")["close"].sort_index()

    # VIX (already loaded by caller)
    v = vix.copy().sort_index()

    # VIX3M for term structure (yfinance)
    print("[meta] loading VIX3M...")
    try:
        v3m = _load_yf_close("^VIX3M", fxr_start, fxr_end, "VIX3M")
    except Exception as e:
        print(f"[meta] VIX3M load failed ({e}); using flat 1.0 ratio")
        v3m = pd.Series(v.values, index=v.index, name="vix3m")

    # DXY
    print("[meta] loading DXY...")
    try:
        dxy = _load_yf_close("DX-Y.NYB", fxr_start, fxr_end, "DXY")
        if len(dxy) < 100:
            raise ValueError("DXY too short")
    except Exception as e:
        print(f"[meta] DXY load failed ({e}); trying UUP")
        try:
            dxy = _load_yf_close("UUP", fxr_start, fxr_end, "UUP")
        except Exception as e2:
            print(f"[meta] UUP also failed ({e2}); using flat 100")
            dxy = pd.Series(100.0, index=spy.index, name="dxy")

    # T10Y2Y
    print("[meta] loading T10Y2Y from FRED...")
    t10y2y = _load_fred_t10y2y(fxr_start, fxr_end)

    # Sector ETFs (from local prices cache — load directly since _load_prices
    # only returns UNIVERSE+BENCH and excludes sector ETFs)
    sector_close = {}
    for s in SECTORS:
        sub = prices[prices["ticker"] == s].set_index("date")["close"].sort_index()
        if len(sub) > 0:
            sector_close[s] = sub
    if not sector_close:
        # Fallback: load sectors directly from the prices_v2.parquet cache
        _all = pd.read_parquet(PRICE_PATH)
        _all["date"] = pd.to_datetime(_all["date"])
        _all = _all[_all["ticker"].isin(SECTORS)][["ticker", "date", "close"]]
        _all = _all[(_all["date"] >= fxr_start) & (_all["date"] <= fxr_end)]
        for s in SECTORS:
            sub = _all[_all["ticker"] == s].set_index("date")["close"].sort_index()
            if len(sub) > 0:
                sector_close[s] = sub
    print(f"[meta] sectors loaded: {list(sector_close.keys())}")

    # K=6 basket equal-weight return + realized vol
    k6_close = {}
    for t in K6_UNIVERSE:
        sub = prices[prices["ticker"] == t].set_index("date")["close"].sort_index()
        if len(sub) > 0:
            k6_close[t] = sub
    print(f"[meta] K6 names loaded: {list(k6_close.keys())}")

    # Build master daily date index (intersect with SPY trading days)
    all_dates = spy.index
    df = pd.DataFrame(index=all_dates)

    # SPY structure
    df["spy_close"] = spy
    df["spy_ret_1d"] = spy.pct_change()
    df["spy_rv_20"] = df["spy_ret_1d"].rolling(20, min_periods=5).std() * np.sqrt(TRADING_DAYS)
    for m in [20, 60, 200]:
        ma = spy.rolling(m, min_periods=int(m * 0.6)).mean()
        sd = spy.rolling(m, min_periods=int(m * 0.6)).std()
        df[f"spy_z_ma{m}"] = (spy - ma) / sd.replace(0.0, np.nan)

    # VIX
    v_aligned = v.reindex(all_dates).ffill()
    df["vix_level"] = v_aligned
    for h in [5, 10, 20]:
        df[f"vix_chg_{h}d"] = v_aligned.diff(h)

    # VIX term structure
    v3m_aligned = v3m.reindex(all_dates).ffill()
    df["vix_term"] = v_aligned / v3m_aligned.replace(0.0, np.nan)

    # DXY
    dxy_aligned = dxy.reindex(all_dates).ffill()
    df["dxy_level"] = dxy_aligned
    df["dxy_mom_20d"] = dxy_aligned.pct_change(20)

    # Yield curve
    t10_aligned = t10y2y.reindex(all_dates).ffill()
    df["t10y2y"] = t10_aligned
    df["t10y2y_chg_30d"] = t10_aligned.diff(30)

    # K6 basket eq-weight daily ret + realized vol + 20d ret
    k6_rets = pd.DataFrame({t: c.pct_change() for t, c in k6_close.items()})
    k6_rets = k6_rets.reindex(all_dates)
    df["k6_basket_ret_1d"] = k6_rets.mean(axis=1)
    df["k6_basket_rv_20"] = df["k6_basket_ret_1d"].rolling(20, min_periods=5).std() * np.sqrt(TRADING_DAYS)
    k6_20d = pd.DataFrame({t: c.pct_change(20) for t, c in k6_close.items()}).reindex(all_dates)
    df["k6_basket_ret_20d"] = k6_20d.mean(axis=1)

    # Sector breadth: median and dispersion of 20d returns across sectors
    sec_20d = pd.DataFrame({s: c.pct_change(20) for s, c in sector_close.items()}).reindex(all_dates)
    df["sector_mom_median"] = sec_20d.median(axis=1)
    df["sector_mom_dispersion"] = sec_20d.std(axis=1)
    # Pos breadth: fraction of sectors with positive 20d ret
    df["sector_breadth_pos20d"] = (sec_20d > 0).sum(axis=1) / max(len(sector_close), 1)

    # Feature columns (everything except spy_close + spy_ret_1d + k6_basket_ret_1d which are raw/used as target inputs)
    feat_cols = [
        "spy_z_ma20", "spy_z_ma60", "spy_z_ma200", "spy_rv_20",
        "vix_level", "vix_chg_5d", "vix_chg_10d", "vix_chg_20d",
        "vix_term",
        "dxy_level", "dxy_mom_20d",
        "t10y2y", "t10y2y_chg_30d",
        "k6_basket_rv_20", "k6_basket_ret_20d",
        "sector_mom_median", "sector_mom_dispersion", "sector_breadth_pos20d",
    ]

    # LAG every feature by 1 day -> "as-of t-1, predict t"
    for c in feat_cols:
        df[c] = df[c].shift(1)

    df["feature_cols"] = ",".join(feat_cols)  # carry metadata
    return df, feat_cols


# ---------------------------------------------------------------------------
# Walk-forward LGBM training + OOT predictions
# ---------------------------------------------------------------------------
def train_meta_wf(feat_df: pd.DataFrame, feat_cols: list[str],
                  k6_book: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Sliding WF: train LGBM binary classifier on each fold, predict on OOT.

    Returns:
      preds_df : per-OOT-day predicted P(positive_next_day_return)
      meta     : feature importance + per-fold diagnostics
    """
    import lightgbm as lgb

    # Build target series: sign of K=6 daily_ret
    k6 = k6_book.set_index("date")["daily_ret"]
    feat_df = feat_df.copy()
    feat_df["k6_ret_t"] = k6.reindex(feat_df.index)
    feat_df["y_clf"] = (feat_df["k6_ret_t"] > 0).astype(int)
    feat_df["y_reg"] = feat_df["k6_ret_t"]

    # Only train/predict on days where K6 has a return (OOT region 2020+)
    feat_df = feat_df.dropna(subset=feat_cols + ["y_clf"])

    windows = _iter_wf(WF_START, WF_END)
    print(f"[meta] WF windows: {len(windows)}")

    all_preds = []
    fold_diag = []
    importance_acc = pd.Series(0.0, index=feat_cols)

    for fold_i, (tr_s, tr_e, os_, oe) in enumerate(windows):
        train = feat_df[(feat_df.index >= tr_s) & (feat_df.index < tr_e)]
        oot = feat_df[(feat_df.index >= os_) & (feat_df.index < oe)]
        if len(train) < 60 or len(oot) < 5:
            continue

        # split last 10% of train as eval for early stopping
        cut = int(len(train) * 0.9)
        tr = train.iloc[:cut]
        val = train.iloc[cut:]

        Xtr, ytr = tr[feat_cols].values, tr["y_clf"].values
        Xval, yval = val[feat_cols].values, val["y_clf"].values
        Xoot = oot[feat_cols].values

        # Class imbalance: K6 has ~55-60% positive days
        pos_rate = float(ytr.mean()) if len(ytr) else 0.5
        spw = (1.0 - pos_rate) / max(pos_rate, 1e-6)  # scale_pos_weight balances classes

        params = dict(
            objective="binary",
            metric="binary_logloss",
            learning_rate=0.03,
            num_leaves=15,
            max_depth=4,
            min_data_in_leaf=20,
            feature_fraction=0.85,
            bagging_fraction=0.85,
            bagging_freq=5,
            lambda_l2=1.0,
            scale_pos_weight=spw,
            verbose=-1,
        )
        dtr = lgb.Dataset(Xtr, label=ytr, feature_name=feat_cols)
        dval = lgb.Dataset(Xval, label=yval, feature_name=feat_cols, reference=dtr)
        booster = lgb.train(
            params, dtr,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False),
                       lgb.log_evaluation(0)],
        )

        p_oot = booster.predict(Xoot, num_iteration=booster.best_iteration)
        pred_df = pd.DataFrame({
            "date": oot.index,
            "p_pos": p_oot,
            "y_true": oot["y_clf"].values,
            "y_ret": oot["y_reg"].values,
            "fold": fold_i,
            "oot_start": str(os_.date()),
        })
        all_preds.append(pred_df)

        # Feature importance (gain-based)
        imp = booster.feature_importance(importance_type="gain")
        for c, val_ in zip(feat_cols, imp):
            importance_acc[c] += float(val_)

        # Fold diagnostics
        acc = float((pred_df["y_true"] == (pred_df["p_pos"] > 0.5).astype(int)).mean())
        # AUC
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(pred_df["y_true"], pred_df["p_pos"])) if len(set(pred_df["y_true"])) > 1 else float("nan")
        except Exception:
            auc = float("nan")
        fold_diag.append({
            "fold": fold_i, "oot_start": str(os_.date()),
            "n_train": int(len(tr)), "n_val": int(len(val)), "n_oot": int(len(oot)),
            "pos_rate_train": pos_rate, "scale_pos_weight": spw,
            "best_iter": int(booster.best_iteration or 0),
            "acc_oot": acc, "auc_oot": auc,
        })
        print(f"[meta] fold {fold_i} OOT {os_.date()}: n_train={len(tr)} acc={acc:.3f} auc={auc:.3f} best_iter={booster.best_iteration}")

    if not all_preds:
        raise RuntimeError("no fold produced predictions")
    preds_df = pd.concat(all_preds, ignore_index=True)

    # If predictions overlap (folds can overlap due to step=3 < OOT=6), keep the FIRST seen
    # so each day uses the meta-prediction from the fold that first covers it.
    preds_df = preds_df.sort_values(["date", "fold"]).drop_duplicates("date", keep="first").reset_index(drop=True)

    # Normalize importance to fraction
    importance_acc /= max(importance_acc.sum(), 1e-9)
    return preds_df, {
        "fold_diag": fold_diag,
        "feature_importance": importance_acc.sort_values(ascending=False).to_dict(),
    }


# ---------------------------------------------------------------------------
# Gated backtest
# ---------------------------------------------------------------------------
def gate_book(k6_book: pd.DataFrame, preds_df: pd.DataFrame,
              gate_thresh: float = GATE_THRESH) -> pd.DataFrame:
    """Apply meta-classifier gate to K=6 book.

    On days where preds.p_pos < gate_thresh -> set daily_ret to 0 (hold cash).
    On days outside the prediction range -> keep original return.
    """
    book = k6_book.copy().sort_values("date").reset_index(drop=True)
    book["date"] = pd.to_datetime(book["date"])
    p = preds_df.set_index("date")["p_pos"]
    book["p_pos"] = book["date"].map(p)
    book["meta_gate"] = (book["p_pos"] >= gate_thresh).astype(float)
    # When p_pos is NaN (no prediction), default to "trade" (1.0)
    book.loc[book["p_pos"].isna(), "meta_gate"] = 1.0
    book["daily_ret_orig"] = book["daily_ret"]
    book["daily_ret"] = book["daily_ret_orig"] * book["meta_gate"]
    return book


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def summarize(book: pd.DataFrame, spy_ret: pd.Series, spy_close: pd.Series,
              label: str) -> dict:
    s = book.set_index("date")["daily_ret"]
    m = _metrics(s)
    strat = regime_stratification(book, spy_ret)
    gates = deploy_verdict(m, strat, book)
    tail = tail_dd_metric(book, spy_close)
    return {
        "label": label,
        "cagr_pct": (m.get("cagr") or 0) * 100,
        "sharpe": m.get("sharpe"),
        "sortino": m.get("sortino"),
        "calmar": m.get("calmar"),
        "max_dd_pct": (m.get("max_dd") or 0) * 100,
        "pf": m.get("pf"),
        "wr_pct": (m.get("wr") or 0) * 100,
        "regime_gap": gates.get("regime_imbalance"),
        "regime_green_sh": gates.get("regime_green_sharpe"),
        "regime_red_sh": gates.get("regime_red_sharpe"),
        "passes_hc428_r1": bool(gates["PASSES_DEPLOY_GATES"]),
        "day_conc": gates.get("day_concentration"),
        "n_oot_days": int(s.dropna().shape[0]),
        "worst_red_quarter_dd_pct": tail["worst_red_quarter_dd_pct"],
        "passes_tail_dd_25pct": tail["passes_tail_dd_25pct"],
    }


def main():
    print("[meta] === K=6 Meta-Classifier v1 ===")
    t0 = time.time()

    print("[meta] loading prices, VIX...")
    prices = _load_prices()
    vix = _load_vix()

    spy_close = prices[prices["ticker"] == BENCH_SPY].sort_values("date").set_index("date")["close"]
    spy_ret = spy_close.pct_change().dropna()

    print("[meta] building feature matrix...")
    feat_df, feat_cols = build_feature_matrix(prices, vix)
    print(f"[meta] features: {feat_cols} (n={len(feat_cols)})")
    feat_df.to_parquet(OUT_DIR / "feature_matrix.parquet")

    print("[meta] loading K=6 baseline book...")
    k6_book = pd.read_parquet(K6_BOOK_PATH)
    k6_book["date"] = pd.to_datetime(k6_book["date"])
    print(f"[meta] K6 book: {len(k6_book)} days, {k6_book['date'].min()} -> {k6_book['date'].max()}")

    # MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("k6_meta_classifier_v1")
        mlflow_ok = True
    except Exception as e:
        print(f"[meta] MLflow unavailable: {e}")
        mlflow_ok = False

    print("[meta] training meta-classifier (walk-forward)...")
    preds_df, meta = train_meta_wf(feat_df, feat_cols, k6_book)
    preds_df.to_parquet(OUT_DIR / "oot_predictions.parquet", index=False)
    print(f"[meta] OOT predictions: {len(preds_df)} days, P(pos) mean={preds_df['p_pos'].mean():.3f}")

    # Feature importance
    fi = pd.Series(meta["feature_importance"]).reset_index()
    fi.columns = ["feature", "gain_share"]
    fi.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    print("[meta] top features:")
    print(fi.head(10).to_string(index=False))

    print(f"[meta] gating with threshold {GATE_THRESH}...")
    gated = gate_book(k6_book, preds_df, GATE_THRESH)
    gated.to_parquet(OUT_DIR / "gated_book.parquet", index=False)

    # Count active vs skipped days
    n_pred = int(gated["p_pos"].notna().sum())
    n_skip = int((gated["meta_gate"] == 0).sum())
    n_trade = int((gated["meta_gate"] == 1).sum())
    print(f"[meta] days with prediction: {n_pred} | trade: {n_trade} | skip: {n_skip}")

    # Side-by-side summaries
    print("[meta] computing metrics...")
    ungated_summary = summarize(k6_book.rename(columns={"daily_ret": "daily_ret"}),
                                spy_ret, spy_close, "K6_ungated_baseline")
    gated_summary = summarize(gated[["date", "daily_ret"]], spy_ret, spy_close,
                              "K6_meta_gated_v1")

    # Trade count: in the ungated book, count days where daily_ret != 0 (proxy)
    # More accurate: count rebal events, but a daily-active-day count is informative.
    n_active_ungated = int((k6_book["daily_ret"] != 0).sum())
    n_active_gated = int((gated["daily_ret"] != 0).sum())

    report = {
        "strategy": "k6_meta_classifier_v1",
        "test_window": [str(WF_START.date()), str(WF_END.date())],
        "k6_universe": K6_UNIVERSE,
        "n_features": len(feat_cols),
        "feature_cols": feat_cols,
        "gate_threshold": GATE_THRESH,
        "wf_spec": {"train_months": TRAIN_MONTHS, "oot_months": OOT_MONTHS, "step_months": STEP_MONTHS},
        "fold_diagnostics": meta["fold_diag"],
        "feature_importance": meta["feature_importance"],
        "n_predictions": n_pred,
        "n_trade_days_gated": n_trade,
        "n_skip_days_gated": n_skip,
        "n_active_days_ungated": n_active_ungated,
        "n_active_days_gated": n_active_gated,
        "p_pos_mean": float(preds_df["p_pos"].mean()),
        "p_pos_median": float(preds_df["p_pos"].median()),
        "ungated": ungated_summary,
        "gated": gated_summary,
        "wall_seconds": round(time.time() - t0, 1),
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[meta] === RESULTS ===")
    print(f"  ungated:  CAGR={ungated_summary['cagr_pct']:.1f}% Sharpe={ungated_summary['sharpe']:.2f} "
          f"MaxDD={ungated_summary['max_dd_pct']:.1f}% gap={ungated_summary['regime_gap']:.2f}")
    print(f"  gated:    CAGR={gated_summary['cagr_pct']:.1f}% Sharpe={gated_summary['sharpe']:.2f} "
          f"MaxDD={gated_summary['max_dd_pct']:.1f}% gap={gated_summary['regime_gap']:.2f}")
    print(f"  HC #428 R1 PASS (gated): {gated_summary['passes_hc428_r1']}")
    print(f"  active days: ungated={n_active_ungated} gated={n_active_gated} ({100*n_active_gated/max(n_active_ungated,1):.1f}%)")

    if mlflow_ok:
        try:
            import mlflow
            with mlflow.start_run(run_name="k6_meta_classifier_v1"):
                mlflow.log_param("k6_universe", ",".join(K6_UNIVERSE))
                mlflow.log_param("n_features", len(feat_cols))
                mlflow.log_param("gate_threshold", GATE_THRESH)
                mlflow.log_param("wf_train_months", TRAIN_MONTHS)
                mlflow.log_param("wf_oot_months", OOT_MONTHS)
                for label, d in [("ungated", ungated_summary), ("gated", gated_summary)]:
                    for k, v in d.items():
                        if isinstance(v, (int, float, bool)) and v is not None:
                            try:
                                if np.isfinite(float(v)):
                                    mlflow.log_metric(f"{label}_{k}", float(v))
                            except (TypeError, ValueError):
                                pass
                mlflow.log_metric("n_predictions", n_pred)
                mlflow.log_metric("n_trade_days_gated", n_trade)
                mlflow.log_metric("n_skip_days_gated", n_skip)
                mlflow.log_metric("p_pos_mean", float(preds_df["p_pos"].mean()))
                mlflow.log_artifact(str(OUT_DIR / "report.json"))
                mlflow.log_artifact(str(OUT_DIR / "oot_predictions.parquet"))
                mlflow.log_artifact(str(OUT_DIR / "gated_book.parquet"))
                mlflow.log_artifact(str(OUT_DIR / "feature_importance.csv"))
        except Exception as e:
            print(f"[meta] MLflow log failed: {e}")

    return report


if __name__ == "__main__":
    main()
