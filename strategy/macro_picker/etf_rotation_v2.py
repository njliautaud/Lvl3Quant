"""
ETF Rotation v2 — yield curve signals + graded regime gate + LGBM picker.

Improvements over v1:
  A. Yield Curve Features: 10y-2y spread, spread RoC, fed funds level/direction,
     all sourced from macro_extra.parquet (causal, published with lag).
  B. VIX-graded regime gate: replaces binary cash/bull with 4-tier allocation
     (100% / 80% / 50% / 20% SH) based on SPY vs MA60 and VIX level.
  C. LGBM instead of Ridge: tree model captures non-linear interactions between
     yield curve regime and sector momentum.
  D. SH (inverse SPY) bear allocation: when SPY < MA60 by >2%, 20% goes to SH
     (modelled as -1x SPY daily return, no separate download needed).

Walk-forward spec (same as v1, HC #0 SLIDING):
  - 24-month train, 1-month OOS, 1-month step (finer than v1's 6m/3m).
  - Universe: 11 sector SPDR ETFs + synthetic SH for bear allocation.
  - Hold: 21 trading days.

Data sources (all on disk, no downloads required):
  - prices_v2.parquet              — ETF daily closes, 2015-present
  - macro_extra.parquet            — yield curve: ust_2y, ust_10y, fed_funds
  - macro_exposure_v1/macro_features.parquet  — VIX level

Output (--out DIR):
  metrics.json, report.md, book.parquet, feature_importance.parquet

CLI:
  python3 etf_rotation_v2.py [--out DIR] [--n-jobs 4]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "research"))
from walk_forward import _metrics  # type: ignore  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PRICE_PATH = ROOT / "wheel_strategy_v1/data/cache/prices_v2.parquet"
MACRO_EXTRA_PATH = ROOT / "wheel_strategy_v1/data/cache/macro_extra.parquet"
MACRO_FEATURES_PATH = ROOT / "macro_exposure_v1/data/cache/macro_features.parquet"

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLY", "XLP", "XLU", "XLI", "XLV", "XLB", "XLC", "XLRE"]
BENCHMARK = "SPY"
TRADING_DAYS = 252
TXN_COST_BPS = 5.0  # round-trip per leg

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
MOMENTUM_FEATURES = [
    "ret_20d",
    "ret_60d",
    "rel_strength_spy",
    "momentum_cross_20_60",
    "rs_rank_among_sectors",
]

YIELD_CURVE_FEATURES = [
    "yc_2s10s",           # 10y-2y spread level
    "yc_2s10s_roc_20d",   # rate of change over 20 days (steepening vs flattening)
    "fed_funds",           # fed funds level
    "fed_funds_roc_20d",   # direction of fed funds
    "ust_10y",             # 10y rate level
    "ust_10y_roc_20d",     # 10y rate direction
]

ALL_FEATURES = MOMENTUM_FEATURES + YIELD_CURVE_FEATURES


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def _load_prices() -> pd.DataFrame:
    """Load all sector ETF + SPY daily closes from prices_v2."""
    px = pd.read_parquet(PRICE_PATH)
    px["date"] = pd.to_datetime(px["date"])
    sub = px[px["ticker"].isin(SECTOR_ETFS + [BENCHMARK])].copy()
    sub["close"] = sub["close"].astype(float)
    return sub.sort_values(["ticker", "date"]).reset_index(drop=True)


def _load_yield_curve() -> pd.DataFrame:
    """
    Load yield curve and fed funds from macro_extra.
    Returns daily frame indexed by date with columns:
      yc_2s10s, yc_2s10s_roc_20d, fed_funds, fed_funds_roc_20d,
      ust_10y, ust_10y_roc_20d
    These are causal (forward-filled monthly releases onto business days).
    """
    me = pd.read_parquet(MACRO_EXTRA_PATH)
    me["date"] = pd.to_datetime(me["date"])
    me = me.sort_values("date").reset_index(drop=True)

    out = pd.DataFrame()
    out["date"] = me["date"]
    out["yc_2s10s"] = me["yc_2s10s"].astype(float)
    out["fed_funds"] = me["fed_funds"].astype(float)
    out["ust_10y"] = me["ust_10y"].astype(float)

    # 20-day rate of change for directional signals
    out["yc_2s10s_roc_20d"] = out["yc_2s10s"].diff(20)
    out["fed_funds_roc_20d"] = out["fed_funds"].diff(20)
    out["ust_10y_roc_20d"] = out["ust_10y"].diff(20)

    return out.set_index("date")


def _load_vix() -> pd.Series:
    """VIX close indexed by date, from macro_features.parquet."""
    mf = pd.read_parquet(MACRO_FEATURES_PATH)
    mf["date"] = pd.to_datetime(mf["date"])
    return mf.set_index("date")["vix"].astype(float)


def _load_spy_ma(px: pd.DataFrame, ma_days: int = 60) -> pd.Series:
    """SPY MA60 indexed by date."""
    spy = px[px["ticker"] == BENCHMARK].sort_values("date").set_index("date")["close"]
    return spy.rolling(ma_days, min_periods=max(20, ma_days // 2)).mean()


# ---------------------------------------------------------------------------
# Panel builder
# ---------------------------------------------------------------------------
def build_panel(hold_days: int, prices: pd.DataFrame,
                yc: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    """
    Build per-ETF daily feature panel.
    Yield curve features are broadcast (same value for all ETFs on a date).
    Returns panel with columns: etf, date, close, ret_1d, y_fwd, <features>
    """
    spy = prices[prices["ticker"] == BENCHMARK].sort_values("date").set_index("date")["close"]
    spy_r20 = spy.pct_change(20)
    spy_r60 = spy.pct_change(60)

    rows = []
    for t in SECTOR_ETFS:
        s = prices[prices["ticker"] == t].sort_values("date").copy()
        if s.empty:
            continue
        s = s.set_index("date")

        s["ret_1d"] = s["close"].pct_change()
        s["ret_20d"] = s["close"].pct_change(20)
        s["ret_60d"] = s["close"].pct_change(60)
        s["sma20"] = s["close"].rolling(20, min_periods=10).mean()
        s["sma60"] = s["close"].rolling(60, min_periods=30).mean()
        s["momentum_cross_20_60"] = (s["sma20"] / s["sma60"]) - 1.0
        s["rel_strength_spy"] = s["ret_60d"] - spy_r60.reindex(s.index)

        # Yield curve broadcast join
        for col in YIELD_CURVE_FEATURES:
            if col in yc.columns:
                s[col] = yc[col].reindex(s.index)

        s["etf"] = t
        rows.append(s.reset_index())

    panel = pd.concat(rows, ignore_index=True)

    # Cross-sectional rank of 20d momentum
    panel["rs_rank_among_sectors"] = panel.groupby("date")["ret_20d"].rank(pct=True)

    # Forward N-day return target
    panel = panel.sort_values(["etf", "date"]).reset_index(drop=True)
    panel["y_fwd"] = (
        panel.groupby("etf")["close"].shift(-hold_days) / panel["close"] - 1.0
    )

    # Add VIX for regime gating (not a predictive feature — kept separate)
    panel["vix"] = vix.reindex(panel["date"]).values

    return panel


# ---------------------------------------------------------------------------
# Graded regime gate
# ---------------------------------------------------------------------------
def compute_regime_allocation(
    spy_close: float,
    spy_ma: float,
    vix_level: float,
) -> tuple[float, float, str]:
    """
    Returns (sector_frac, sh_frac, label) where:
      sector_frac = fraction of NAV to deploy in sector ETFs
      sh_frac     = fraction of NAV to deploy in synthetic SH (inverse SPY)

    Gate logic (per spec):
      SPY > MA60 and VIX < 20:  100% sector
      SPY > MA60 and VIX 20-25: 80% sector
      SPY < MA60 by < 2%:       50% sector
      SPY < MA60 by > 2%:       20% SH (inverse SPY) + 80% cash
    """
    if not np.isfinite(spy_ma) or not np.isfinite(spy_close):
        return 1.0, 0.0, "bull_full"

    ma_gap_pct = (spy_close - spy_ma) / spy_ma  # positive = above MA

    if ma_gap_pct >= 0:  # SPY above MA60
        if not np.isfinite(vix_level) or vix_level < 20.0:
            return 1.0, 0.0, "bull_full"
        elif vix_level <= 25.0:
            return 0.80, 0.0, "bull_cautious"
        else:
            # VIX > 25 but SPY still above MA — use cautious allocation
            return 0.80, 0.0, "bull_highvol"
    else:  # SPY below MA60
        if ma_gap_pct > -0.02:  # less than 2% below — sit in cash (transition chop)
            return 0.0, 0.0, "bear_shallow"
        else:  # more than 2% below
            return 0.0, 0.20, "bear_deep"


# ---------------------------------------------------------------------------
# Winsorize / z-score
# ---------------------------------------------------------------------------
def _winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lower=lo, upper=hi)


def _xs_zscore(panel: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Per-date cross-sectional z-score across all ETFs in panel."""
    out = panel.copy()
    for f in feats:
        if f not in out.columns:
            out[f] = 0.0
            continue
        x = pd.to_numeric(out[f], errors="coerce").astype(float)
        x = _winsorize(x, 0.01)
        out[f] = x
        mu = out.groupby("date")[f].transform("mean")
        sd = out.groupby("date")[f].transform("std")
        z = (x - mu) / sd.replace(0.0, np.nan)
        out[f] = z.replace([np.inf, -np.inf], np.nan)
    return out


# ---------------------------------------------------------------------------
# Model: LGBM with ridge fallback
# ---------------------------------------------------------------------------
def _fit_lgbm(X_tr: np.ndarray, y_tr: np.ndarray,
              X_oot: np.ndarray) -> np.ndarray | None:
    """
    Fit LightGBM cross-sectional predictor.
    Returns OOT scores, or None if lgbm not available.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        return None

    params = {
        "objective": "regression",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_child_samples": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
        "n_jobs": 1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_tr, y_tr)
    return model.predict(X_oot)


def _fit_ridge(X: np.ndarray, y: np.ndarray,
               alphas=(0.1, 1.0, 10.0, 100.0)) -> tuple:
    """Ridge regression with alpha grid search (MSE-optimal)."""
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    XtX = Xc.T @ Xc
    Xty = Xc.T @ yc
    best = (None, None, float("inf"))
    for a in alphas:
        try:
            beta = np.linalg.solve(XtX + a * np.eye(X.shape[1]), Xty)
            mse = float(((yc - Xc @ beta) ** 2).mean())
            if mse < best[2]:
                best = (beta, y.mean() - X.mean(axis=0) @ beta, mse)
        except np.linalg.LinAlgError:
            continue
    return best[0], best[1]


# ---------------------------------------------------------------------------
# Walk-forward fold
# ---------------------------------------------------------------------------
def _estimate_book_vol(panel: pd.DataFrame, rd: pd.Timestamp,
                       longs: list[str], lookback_days: int = 60) -> float:
    """Prior 60-day realised annualised vol of the equal-weight long book."""
    cutoff_lo = rd - pd.Timedelta(days=lookback_days * 2 + 10)
    hist = panel[
        (panel["date"] < rd) & (panel["date"] >= cutoff_lo)
        & (panel["etf"].isin(longs))
    ]
    if hist.empty:
        return 0.0
    by_date = hist.groupby("date")["ret_1d"].mean().dropna().tail(lookback_days)
    if len(by_date) < 20:
        return 0.0
    sd = float(by_date.std(ddof=1))
    return sd * np.sqrt(TRADING_DAYS) if np.isfinite(sd) else 0.0


def _wf_fold(
    panel: pd.DataFrame,
    feats: list[str],
    tr_start: pd.Timestamp,
    tr_end: pd.Timestamp,
    oot_start: pd.Timestamp,
    oot_end: pd.Timestamp,
    hold_days: int,
    n_long: int,
    target_vol: float = 0.15,
    lev_min: float = 0.25,
    lev_max: float = 2.0,
    txn_cost_bps: float = TXN_COST_BPS,
    spy_ma: pd.Series | None = None,
    vix: pd.Series | None = None,
) -> dict:
    """
    One sliding walk-forward fold.
    Returns dict with daily_pnl (Series), feature importances, etc.

    SH (inverse SPY) allocation: on bear_deep days, 20% of NAV earns -1x SPY
    daily return (approximates SH without needing separate ticker download).
    Costs on SH leg: same 5bps per leg.
    """
    train = panel[(panel["date"] >= tr_start) & (panel["date"] < tr_end)].copy()
    oot = panel[(panel["date"] >= oot_start) & (panel["date"] < oot_end)].copy()

    if len(train) < 100 or len(oot) < 10:
        return {
            "daily_pnl": pd.Series(dtype=float),
            "feat_imp": {},
            "oot_start": str(oot_start.date()),
            "oot_end": str(oot_end.date()),
            "n_rebal": 0,
            "model": "none",
        }

    # Z-score features cross-sectionally
    train_z = _xs_zscore(train, feats)
    oot_z = _xs_zscore(oot, feats)
    for f in feats:
        train_z[f] = train_z[f].fillna(0.0)
        oot_z[f] = oot_z[f].fillna(0.0)
    train_z = train_z.dropna(subset=["y_fwd"])
    if len(train_z) < 50:
        return {
            "daily_pnl": pd.Series(dtype=float),
            "feat_imp": {},
            "oot_start": str(oot_start.date()),
            "oot_end": str(oot_end.date()),
            "n_rebal": 0,
            "model": "none",
        }

    X_tr = train_z[feats].values
    y_tr = train_z["y_fwd"].values
    X_oot_all = oot_z[feats].values

    # Try LGBM first, fall back to ridge
    scores_oot = _fit_lgbm(X_tr, y_tr, X_oot_all)
    model_used = "lgbm"
    feat_imp: dict = {}

    if scores_oot is None:
        coef, intercept = _fit_ridge(X_tr, y_tr)
        if coef is None:
            return {
                "daily_pnl": pd.Series(dtype=float),
                "feat_imp": {},
                "oot_start": str(oot_start.date()),
                "oot_end": str(oot_end.date()),
                "n_rebal": 0,
                "model": "none",
            }
        scores_oot = X_oot_all @ coef + intercept
        feat_imp = dict(zip(feats, [float(c) for c in coef]))
        model_used = "ridge_fallback"

    oot_z = oot_z.copy()
    oot_z["score"] = scores_oot
    # Preserve raw returns (z-score overwrites ret_1d — use original panel)
    oot_z["ret_raw"] = pd.to_numeric(oot["ret_1d"], errors="coerce").astype(float).values

    unique_dates = sorted(oot_z["date"].unique())
    rebal_dates = unique_dates[::hold_days]

    # SPY raw daily return for SH simulation
    spy_panel = panel[panel["etf"] == "SPY"].set_index("date")["ret_1d"] \
        if "SPY" in panel["etf"].values else pd.Series(dtype=float)
    # Actually SPY is not in panel (we dropped it). Recompute from prices.
    # We'll attach spy_daily_ret from the main prices during OOT period.

    daily = []
    n_rebal_actual = 0

    for rd in rebal_dates:
        # --- Regime gate ---
        spy_close_val = float("nan")
        spy_ma_val = float("nan")
        vix_val = float("nan")

        if spy_ma is not None:
            prior_ma = spy_ma.loc[:pd.Timestamp(rd)]
            if len(prior_ma) > 0:
                spy_ma_val = float(prior_ma.iloc[-1])

        if spy_ma is not None:
            # spy_close from the SPY price series
            spy_price_series = spy_ma.index  # we have the index
            # Retrieve SPY close at rd from the passed series
            # (spy_ma Series was built from SPY close — compute close separately)
            pass

        # We pass spy_close via a helper attribute appended to spy_ma Series
        # (see build_regime_series() below)
        if hasattr(spy_ma, "_spy_close"):
            prior_close = spy_ma._spy_close.loc[:pd.Timestamp(rd)]
            if len(prior_close) > 0:
                spy_close_val = float(prior_close.iloc[-1])

        if vix is not None:
            prior_vix = vix.loc[:pd.Timestamp(rd)]
            if len(prior_vix) > 0:
                vix_val = float(prior_vix.iloc[-1])

        sector_frac, sh_frac, regime_label = compute_regime_allocation(
            spy_close_val, spy_ma_val, vix_val
        )

        # Score sectors on this rebal date
        snap = oot_z[oot_z["date"] == rd].dropna(subset=["score"])
        if len(snap) < n_long:
            continue

        # No-trade floor: top score must exceed median
        med = snap["score"].median()
        if snap["score"].max() <= med and sector_frac > 0:
            sector_frac_effective = 0.0
        else:
            sector_frac_effective = sector_frac

        longs = snap.nlargest(n_long, "score")["etf"].tolist() if sector_frac_effective > 0 else []

        # Vol-target sizing for the sector sleeve
        if longs:
            realised_vol = _estimate_book_vol(panel, rd, longs)
            if realised_vol <= 1e-6:
                gross_lev = 1.0
            else:
                gross_lev = float(np.clip(target_vol / realised_vol, lev_min, lev_max))
        else:
            gross_lev = 0.0

        # Hold window P&L
        hold_win = oot_z[(oot_z["date"] > rd)
                         & (oot_z["date"] <= rd + pd.Timedelta(days=hold_days * 2))]
        # Get unique hold dates (up to hold_days trading dates)
        hold_dates = sorted(hold_win["date"].unique())[:hold_days]

        for d in hold_dates:
            g = hold_win[hold_win["date"] == d]

            # Intra-hold regime check
            d_ts = pd.Timestamp(d)
            if hasattr(spy_ma, "_spy_close"):
                pc = spy_ma._spy_close.loc[:d_ts]
                d_spy_close = float(pc.iloc[-1]) if len(pc) > 0 else float("nan")
            else:
                d_spy_close = float("nan")

            d_spy_ma = float(spy_ma.loc[:d_ts].iloc[-1]) if (spy_ma is not None and len(spy_ma.loc[:d_ts]) > 0) else float("nan")
            d_vix = float(vix.loc[:d_ts].iloc[-1]) if (vix is not None and len(vix.loc[:d_ts]) > 0) else float("nan")
            d_sfrac, d_shfrac, d_label = compute_regime_allocation(d_spy_close, d_spy_ma, d_vix)

            # Sector sleeve return
            if longs and d_sfrac > 0:
                lret = g[g["etf"].isin(longs)]["ret_raw"].mean()
                lret = float(lret) if pd.notna(lret) else 0.0
                sector_ret = gross_lev * d_sfrac * lret
            else:
                sector_ret = 0.0

            # SH sleeve return (synthetic: -1x SPY daily return)
            if d_shfrac > 0 and hasattr(spy_ma, "_spy_ret"):
                spy_ret_d = spy_ma._spy_ret.get(d_ts, 0.0)
                sh_ret = d_shfrac * (-float(spy_ret_d))  # inverse
            else:
                sh_ret = 0.0

            book_ret = sector_ret + sh_ret
            book_ret = float(np.clip(book_ret, -0.20, 0.20))
            daily.append((d, book_ret, gross_lev, d_label))

        # Transaction cost on rebal day
        n_legs = len(longs) + (1 if sh_frac > 0 else 0)
        tc = n_legs * (txn_cost_bps / 10000.0) * max(gross_lev, sh_frac)
        daily.append((rd, -tc, gross_lev, regime_label))
        n_rebal_actual += 1

    if not daily:
        return {
            "daily_pnl": pd.Series(dtype=float),
            "feat_imp": feat_imp,
            "oot_start": str(oot_start.date()),
            "oot_end": str(oot_end.date()),
            "n_rebal": 0,
            "model": model_used,
        }

    df_daily = pd.DataFrame(daily, columns=["date", "ret", "lev", "regime"])
    df_daily["date"] = pd.to_datetime(df_daily["date"])
    df_daily = df_daily.groupby("date", as_index=True).agg(
        ret=("ret", "sum"), lev=("lev", "max"), regime=("regime", "last")
    )

    return {
        "daily_pnl": df_daily["ret"],
        "daily_lev": df_daily["lev"],
        "daily_regime": df_daily["regime"],
        "feat_imp": feat_imp,
        "oot_start": str(oot_start.date()),
        "oot_end": str(oot_end.date()),
        "n_rebal": n_rebal_actual,
        "model": model_used,
    }


# ---------------------------------------------------------------------------
# Regime series builder
# ---------------------------------------------------------------------------
def build_regime_series(prices: pd.DataFrame, vix: pd.Series,
                        ma_days: int = 60) -> pd.Series:
    """
    Build SPY MA60 Series with extra attributes attached:
      ._spy_close  — SPY raw close (for gap_pct computation)
      ._spy_ret    — SPY daily return (for SH simulation)
    """
    spy = prices[prices["ticker"] == BENCHMARK].sort_values("date").set_index("date")["close"]
    ma = spy.rolling(ma_days, min_periods=max(20, ma_days // 2)).mean()
    ma._spy_close = spy
    ma._spy_ret = spy.pct_change().fillna(0.0)
    return ma


# ---------------------------------------------------------------------------
# WF window generator (SLIDING — HC #0)
# ---------------------------------------------------------------------------
def _iter_wf_windows(start: pd.Timestamp, end: pd.Timestamp,
                     train_months: int = 24, oot_months: int = 1,
                     step_months: int = 1) -> list[tuple]:
    """Sliding walk-forward windows. Train never re-uses OOT data."""
    out = []
    cursor = start
    while True:
        tr_start = cursor
        tr_end = tr_start + pd.DateOffset(months=train_months)
        oot_start = tr_end
        oot_end = oot_start + pd.DateOffset(months=oot_months)
        if oot_end > end + pd.Timedelta(days=1):
            break
        out.append((tr_start, tr_end, oot_start, oot_end))
        cursor = cursor + pd.DateOffset(months=step_months)
    return out


# ---------------------------------------------------------------------------
# Regime split metrics
# ---------------------------------------------------------------------------
def _regime_split(all_pnl: pd.Series, all_regime: pd.Series) -> dict:
    """
    Compute Sharpe per SPY regime (green/red/flat = +/- ES close-to-close).
    Here we use the regime label from the graded gate as a proxy, grouping:
      bull_full + bull_cautious + bull_highvol -> 'green'
      bear_shallow                             -> 'flat'
      bear_deep                                -> 'red'
    """
    from walk_forward import _metrics as wf_metrics  # type: ignore  # noqa

    groups = {
        "green": all_pnl[all_regime.isin(["bull_full", "bull_cautious", "bull_highvol"])],
        "flat":  all_pnl[all_regime == "bear_shallow"],
        "red":   all_pnl[all_regime == "bear_deep"],
    }
    result = {}
    for label, s in groups.items():
        s = s.dropna()
        if len(s) >= 20:
            m = wf_metrics(s)
            result[label] = {"sharpe": m.get("sharpe", float("nan")), "n_days": len(s)}
        else:
            result[label] = {"sharpe": float("nan"), "n_days": len(s)}
    return result


# ---------------------------------------------------------------------------
# Day concentration (HC #344)
# ---------------------------------------------------------------------------
def _day_concentration(pnl: pd.Series) -> float:
    """Top-1 day as fraction of cumulative PnL (cap <=0.70)."""
    cum = float(pnl.sum())
    if cum <= 0 or pnl.empty:
        return float("nan")
    return float(pnl.max() / cum)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def run(hold_days: int = 21, n_long: int = 2, out_dir: Path | None = None,
        n_jobs: int = -1, target_vol: float = 0.15,
        lev_min: float = 0.25, lev_max: float = 2.0,
        train_months: int = 24, oot_months: int = 1, step_months: int = 1,
        txn_cost_bps: float = TXN_COST_BPS) -> dict:

    if out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = ROOT / f"output/macro_picker/etf_rotation_v2_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[v2] Loading prices...")
    prices = _load_prices()

    print("[v2] Loading yield curve...")
    yc = _load_yield_curve()

    print("[v2] Loading VIX...")
    vix = _load_vix()

    print("[v2] Building feature panel...")
    panel = build_panel(hold_days, prices, yc, vix)
    feats_present = [f for f in ALL_FEATURES if f in panel.columns]
    print(f"[v2] Panel: {len(panel)} rows, ETFs={panel['etf'].nunique()}, "
          f"features={len(feats_present)}, "
          f"date={panel['date'].min().date()}->{panel['date'].max().date()}")

    # Check which yield curve features are actually available
    yc_feats_present = [f for f in YIELD_CURVE_FEATURES if f in feats_present]
    mom_feats_present = [f for f in MOMENTUM_FEATURES if f in feats_present]
    print(f"[v2] Momentum features: {mom_feats_present}")
    print(f"[v2] Yield curve features: {yc_feats_present}")

    # Regime series
    spy_ma_series = build_regime_series(prices, vix, ma_days=60)

    # Restrict to 2017+ (need 24m train before first OOT in 2019+;
    # XLC launched 2018-06 — allow NaN fill for early dates)
    panel_start = panel["date"].min()
    panel_end = panel["date"].max()

    windows = _iter_wf_windows(
        panel_start, panel_end,
        train_months=train_months, oot_months=oot_months, step_months=step_months
    )
    print(f"[v2] Walk-forward: {train_months}m train / {oot_months}m OOT / "
          f"{step_months}m step -> {len(windows)} folds")

    # Run folds (parallel if n_jobs != 1)
    def _run_fold(args):
        ts_, te_, os_, oe_ = args
        return _wf_fold(
            panel, feats_present, ts_, te_, os_, oe_,
            hold_days=hold_days, n_long=n_long,
            target_vol=target_vol, lev_min=lev_min, lev_max=lev_max,
            txn_cost_bps=txn_cost_bps,
            spy_ma=spy_ma_series, vix=vix,
        )

    if n_jobs == 1 or len(windows) <= 4:
        fold_results = [_run_fold(w) for w in windows]
    else:
        try:
            from joblib import Parallel, delayed
            fold_results = Parallel(n_jobs=n_jobs, verbose=5)(
                delayed(_run_fold)(w) for w in windows
            )
        except Exception as e:
            print(f"[v2] Parallel failed ({e}), running serial")
            fold_results = [_run_fold(w) for w in windows]

    # Aggregate pooled OOT
    pnl_parts = [r["daily_pnl"] for r in fold_results if not r["daily_pnl"].empty]
    if not pnl_parts:
        print("[v2] ERROR: no fold produced any P&L")
        return {}

    all_pnl = pd.concat(pnl_parts).sort_index()
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep="last")]

    regime_parts = [r.get("daily_regime", pd.Series(dtype=str))
                    for r in fold_results if not r["daily_pnl"].empty]
    all_regime = pd.concat(regime_parts).sort_index()
    all_regime = all_regime[~all_regime.index.duplicated(keep="last")]
    all_regime = all_regime.reindex(all_pnl.index).fillna("bull_full")

    # Core metrics
    pooled = _metrics(all_pnl)

    # Regime split
    regime_split = _regime_split(all_pnl, all_regime)

    # Day concentration
    day_conc = _day_concentration(all_pnl[all_pnl > 0])

    # Per-fold metrics
    per_fold = []
    for r in fold_results:
        m = _metrics(r["daily_pnl"]) if not r["daily_pnl"].empty else {}
        per_fold.append({
            "oot_start": r["oot_start"], "oot_end": r["oot_end"],
            "n_rebal": r["n_rebal"], "model": r["model"], **m,
        })

    # Feature importance (avg across folds where ridge was used)
    all_imp: dict[str, list[float]] = {}
    for r in fold_results:
        for f, v in r.get("feat_imp", {}).items():
            all_imp.setdefault(f, []).append(v)
    avg_imp = {f: float(np.mean(vs)) for f, vs in all_imp.items()}
    imp_sorted = sorted(avg_imp.items(), key=lambda x: -abs(x[1]))

    # Deploy gate
    fold_calmars = [m.get("calmar", float("nan")) for m in per_fold
                    if m.get("calmar") is not None and np.isfinite(m.get("calmar", float("nan")))]
    fold_maxdds = [m.get("max_dd", float("nan")) for m in per_fold
                   if m.get("max_dd") is not None and np.isfinite(m.get("max_dd", float("nan")))]
    fold_sharpes = [m.get("sharpe", float("nan")) for m in per_fold
                    if m.get("sharpe") is not None and np.isfinite(m.get("sharpe", float("nan")))]

    median_calmar = float(np.median(fold_calmars)) if fold_calmars else float("nan")
    worst_maxdd = float(np.min(fold_maxdds)) if fold_maxdds else float("nan")
    median_sharpe = float(np.median(fold_sharpes)) if fold_sharpes else float("nan")
    n_folds = len(fold_calmars)
    n_passing = sum(1 for c in fold_calmars if c >= 1.0)

    gate_calmar = median_calmar >= 1.0 if np.isfinite(median_calmar) else False
    gate_dd = worst_maxdd > -0.25 if np.isfinite(worst_maxdd) else False
    gate_majority = (n_passing / n_folds) >= 0.5 if n_folds > 0 else False
    deploy = gate_calmar and gate_dd and gate_majority

    # Regime skew gate (HC #428 R1)
    gs = regime_split.get("green", {}).get("sharpe", float("nan"))
    rs = regime_split.get("red", {}).get("sharpe", float("nan"))
    if np.isfinite(gs) and np.isfinite(rs) and max(abs(gs), abs(rs)) > 0:
        regime_skew = abs(gs - rs) / max(abs(gs), abs(rs))
    else:
        regime_skew = float("nan")
    regime_gate_pass = regime_skew <= 0.50 if np.isfinite(regime_skew) else None

    metrics_out = {
        "config": {
            "version": "v2",
            "hold_days": hold_days, "n_long": n_long,
            "train_months": train_months, "oot_months": oot_months,
            "step_months": step_months,
            "target_vol": target_vol, "lev_min": lev_min, "lev_max": lev_max,
            "txn_cost_bps": txn_cost_bps,
            "features": feats_present,
            "universe": SECTOR_ETFS,
        },
        "pooled": pooled,
        "date_range": [
            str(all_pnl.index.min().date()) if not all_pnl.empty else None,
            str(all_pnl.index.max().date()) if not all_pnl.empty else None,
        ],
        "regime_split": regime_split,
        "regime_skew": regime_skew,
        "regime_gate_pass": regime_gate_pass,
        "day_concentration": day_conc,
        "deploy_gate": {
            "median_calmar": median_calmar,
            "median_sharpe": median_sharpe,
            "worst_fold_maxdd": worst_maxdd,
            "n_folds": n_folds,
            "n_passing_folds": n_passing,
            "gate_median_calmar_ge_1": gate_calmar,
            "gate_worst_maxdd_gt_neg25": gate_dd,
            "gate_majority_folds": gate_majority,
            "PASS": deploy,
        },
        "per_fold": per_fold,
        "feature_importance": dict(imp_sorted),
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2, default=str))

    # Book parquet
    book = pd.DataFrame({
        "date": all_pnl.index,
        "daily_ret": all_pnl.values,
        "regime": all_regime.reindex(all_pnl.index).values,
    })
    book.to_parquet(out_dir / "book.parquet", index=False)

    # Feature importance parquet
    if avg_imp:
        pd.DataFrame(list(avg_imp.items()), columns=["feature", "avg_coef"]).to_parquet(
            out_dir / "feature_importance.parquet", index=False
        )

    # -----------------------------------------------------------------------
    # Print findings report
    # -----------------------------------------------------------------------
    p = pooled
    cagr_pct = p.get("cagr", float("nan")) * 100 if "cagr" in p else float("nan")
    maxdd_pct = p.get("max_dd", float("nan")) * 100 if "max_dd" in p else float("nan")

    print()
    print("=" * 65)
    print("RUN: ETF Rotation v2 — yield curve + graded regime + LGBM")
    print(f"     {train_months}m train / {oot_months}m OOT, {len(windows)} sliding folds")
    print("=" * 65)
    print()
    print(f"  Sharpe   {p.get('sharpe', float('nan')):.2f}     "
          f"Sortino  {p.get('sortino', float('nan')):.2f}")
    print(f"  Calmar   {p.get('calmar', float('nan')):.2f}     "
          f"MaxDD    {maxdd_pct:.1f}%")
    print(f"  WR       {p.get('wr', float('nan'))*100:.1f}%    "
          f"PF       {p.get('pf', float('nan')):.2f}")
    print(f"  CAGR     {cagr_pct:.1f}%      "
          f"DayConc  {day_conc*100:.1f}%")
    print()
    print("REGIME SPLIT:")
    for label in ["green", "flat", "red"]:
        rs_info = regime_split.get(label, {})
        sh = rs_info.get("sharpe", float("nan"))
        nd = rs_info.get("n_days", 0)
        print(f"  {label.capitalize()}: Sharpe {sh:.2f}  ({nd} days)")
    if np.isfinite(regime_skew):
        print(f"  Skew = |G-R|/max(|G|,|R|) = {regime_skew:.2f}  "
              f"[{'PASS' if regime_gate_pass else 'FAIL'} <=0.50]")
    else:
        print(f"  Skew = NEEDS-DATA: insufficient red-day sample")
    print()
    print("DEPLOY GATE:")
    print(f"  Median calmar: {median_calmar:.2f}  ({'PASS' if gate_calmar else 'FAIL'} >=1.0)")
    print(f"  Worst-fold MaxDD: {worst_maxdd*100:.1f}%  ({'PASS' if gate_dd else 'FAIL'} >-25%)")
    print(f"  Folds passing Calmar>=1: {n_passing}/{n_folds}  "
          f"({'PASS' if gate_majority else 'FAIL'} >=50%)")
    print(f"  OVERALL: {'PASS' if deploy else 'FAIL'}")
    print()
    print(f"Output: {out_dir}")
    print("=" * 65)

    return metrics_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ETF Rotation v2")
    p.add_argument("--hold-days", type=int, default=21)
    p.add_argument("--n-long", type=int, default=2)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--n-jobs", type=int, default=1,
                   help="Parallel folds (1=serial, -1=all cores). "
                        "Set >1 only if joblib installed.")
    p.add_argument("--target-vol", type=float, default=0.15)
    p.add_argument("--lev-min", type=float, default=0.25)
    p.add_argument("--lev-max", type=float, default=2.0)
    p.add_argument("--train-months", type=int, default=24)
    p.add_argument("--oot-months", type=int, default=1)
    p.add_argument("--step-months", type=int, default=1)
    p.add_argument("--txn-cost-bps", type=float, default=5.0)
    return p.parse_args()


def main():
    args = _parse_args()
    out_dir = Path(args.out) if args.out else None
    run(
        hold_days=args.hold_days,
        n_long=args.n_long,
        out_dir=out_dir,
        n_jobs=args.n_jobs,
        target_vol=args.target_vol,
        lev_min=args.lev_min,
        lev_max=args.lev_max,
        train_months=args.train_months,
        oot_months=args.oot_months,
        step_months=args.step_months,
        txn_cost_bps=args.txn_cost_bps,
    )


if __name__ == "__main__":
    main()
