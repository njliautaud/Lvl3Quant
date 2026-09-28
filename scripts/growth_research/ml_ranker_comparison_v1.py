#!/usr/bin/env python3
"""
ML Ranker Comparison v1 — Is LGBM the Best Sector Ranker?
===========================================================

Our V6 strategy uses LGBM (100 trees, depth 4, lr 0.05) to rank 11 sector ETFs.
LGBM achieves Sharpe 2.87 with 5/5 gates. This experiment tests whether other
ML models can beat LGBM as the ranking engine.

Research question: Is LGBM the best ranker, or can XGBoost, CatBoost, Random Forest,
Ridge regression, or an ensemble beat it?

6 Variants (all with V6 structure: weekly, 2% OTM, 17 features, pairs):
  A: LGBM baseline (100 trees, depth 4, lr 0.05)
  B: XGBoost (100 trees, depth 4, lr 0.05, same hyperparams)
  C: Random Forest (100 trees, max_depth 4)
  D: Ridge Regression (simple linear model)
  E: LGBM deeper (200 trees, depth 6)
  F: Ensemble: average rankings from A+B+C

V6 config for ALL variants:
  - Weekly rebalance (W-FRI)
  - 2% OTM moneyness
  - Bull VIX>20 + pairs VIX<20
  - $200/trade ($100/leg for pairs)
  - DTE=21, 3% spread width, $2.60 commission, 15% haircut entry only
  - Hold to expiry, intrinsic value only

Each variant: 5-gate adversarial validation + 5-trial random baseline.
MLflow experiment: 'ml_ranker_comparison_v1'
Output: output/growth_research/ml_ranker_comparison_v1/
"""

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

_builtin_print = print


def fprint(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()


# ── Path detection (Neptune vs Jupiter) ──
import os as _os
_hostname = _os.uname().nodename.lower()
if 'neptune' in _hostname or 'nick' in str(_os.path.expanduser('~')):
    _BASE_PATH = "/home/nick/Lvl3Quant"
else:
    _BASE_PATH = "/home/jupiter/Lvl3Quant"
sys.path.insert(0, _BASE_PATH)

# ── Standardized tools with inline fallbacks ──
try:
    from research.tools.options_pricer import (
        price_bull_call_spread,
        price_bear_put_spread,
        estimate_iv,
        compute_atr,
        COMMISSION_RT_SPREAD,
        DEFAULT_HAIRCUT,
    )
    from research.tools.adversarial_validator import validate_trades
    _TOOLS_AVAILABLE = True
    fprint("Imported from research.tools")
except ImportError:
    _TOOLS_AVAILABLE = False
    fprint("research.tools not found — using inline implementations")
    COMMISSION_RT_SPREAD = 2.60
    DEFAULT_HAIRCUT = 0.15

    def price_bull_call_spread(S, K1, K2, dte, atr, vix):
        """Inline bull call spread pricer with 15% entry haircut."""
        from scipy.stats import norm
        T = dte / 365.0
        if T <= 0:
            return 0.0, 0.0
        iv = vix / 100.0 * 1.2 if vix else 0.25
        d1_l = (np.log(S / K1) + (0.5 * iv**2) * T) / (iv * np.sqrt(T) + 1e-10)
        d1_s = (np.log(S / K2) + (0.5 * iv**2) * T) / (iv * np.sqrt(T) + 1e-10)
        call_l = S * norm.cdf(d1_l) - K1 * norm.cdf(d1_l - iv * np.sqrt(T))
        call_s = S * norm.cdf(d1_s) - K2 * norm.cdf(d1_s - iv * np.sqrt(T))
        spread_val = max(call_l - call_s, 0.001)
        entry_cost = spread_val * (1 + DEFAULT_HAIRCUT)
        max_profit = (K2 - K1) - entry_cost
        return entry_cost, max_profit

    def price_bear_put_spread(S, K1, K2, dte, atr, vix):
        """Inline bear put spread pricer with 15% entry haircut."""
        from scipy.stats import norm
        T = dte / 365.0
        if T <= 0:
            return 0.0, 0.0
        iv = vix / 100.0 * 1.2 if vix else 0.25
        d1_l = (np.log(S / K2) + (0.5 * iv**2) * T) / (iv * np.sqrt(T) + 1e-10)
        d1_s = (np.log(S / K1) + (0.5 * iv**2) * T) / (iv * np.sqrt(T) + 1e-10)
        put_l = K2 * norm.cdf(-(d1_l - iv * np.sqrt(T))) - S * norm.cdf(-d1_l)
        put_s = K1 * norm.cdf(-(d1_s - iv * np.sqrt(T))) - S * norm.cdf(-d1_s)
        spread_val = max(put_l - put_s, 0.001)
        entry_cost = spread_val * (1 + DEFAULT_HAIRCUT)
        max_profit = (K2 - K1) - entry_cost
        return entry_cost, max_profit

    class _ValidationResult:
        """Lightweight replacement for adversarial_validator result object."""
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)
        def print_summary(self):
            fprint(f"\n{'='*65}")
            fprint(f"  ADVERSARIAL VALIDATION: {self.strategy_name}")
            fprint(f"{'='*65}")
            fprint(f"  Trades: {self.n_trades}  |  Sharpe: {self.sharpe:.2f}  |  "
                   f"Sortino: {self.sortino:.2f}  |  WR: {self.win_rate:.1%}")
            fprint(f"  CAGR: {self.cagr:.1%}  |  MaxDD: {self.max_dd:.1%}  |  "
                   f"PF: {self.profit_factor:.2f}  |  Final: ${self.final_equity:,.0f}")
            fprint(f"  Gates: {self.gates_passed}/{self.gates_total}")
            fprint(f"{'='*65}")
        def to_dict(self):
            return {k: v for k, v in self.__dict__.items()}

    def validate_trades(trades, initial_capital=645.0, spy_prices=None,
                        strategy_name="", n_perms=1000):
        """Inline 5-gate adversarial validation."""
        if not trades or len(trades) < 10:
            return _ValidationResult({
                "sharpe": 0, "sortino": 0, "win_rate": 0, "profit_factor": 0,
                "max_dd": 0, "cagr": 0, "n_trades": len(trades) if trades else 0,
                "final_equity": initial_capital, "gates_passed": 0, "gates_total": 5,
                "strategy_name": strategy_name,
            })
        pnls = [t["pnl"] for t in trades]
        equity = [initial_capital]
        for p in pnls:
            equity.append(equity[-1] + p)
        equity = np.array(equity[1:])
        rets = np.diff(np.concatenate([[initial_capital], equity])) / np.concatenate([[initial_capital], equity[:-1]])

        sharpe = float(np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(52))
        neg_rets = rets[rets < 0]
        sortino = float(np.mean(rets) / (np.std(neg_rets) + 1e-10) * np.sqrt(52)) if len(neg_rets) > 0 else 0
        wr = float(np.mean([1 if p > 0 else 0 for p in pnls]))
        wins = sum(p for p in pnls if p > 0)
        losses = abs(sum(p for p in pnls if p <= 0))
        pf = float(wins / (losses + 1e-10))
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / (peak + 1e-10)
        mdd = float(dd.min())

        n_years = max(len(trades) / 52, 1)
        cagr = float((equity[-1] / initial_capital) ** (1 / n_years) - 1) if equity[-1] > 0 else 0

        # 5 gates
        gates = 0
        if sharpe > 0.5: gates += 1
        if wr > 0.45: gates += 1
        if pf > 1.0: gates += 1
        if mdd > -0.30: gates += 1
        # Gate 5: permutation test
        if n_perms > 0 and len(pnls) >= 20:
            obs_sharpe = sharpe
            count_better = 0
            for _ in range(min(n_perms, 500)):
                shuffled = np.random.permutation(pnls)
                seq = [initial_capital]
                for p in shuffled:
                    seq.append(seq[-1] + p)
                seq = np.array(seq[1:])
                sr = np.diff(np.concatenate([[initial_capital], seq])) / np.concatenate([[initial_capital], seq[:-1]])
                perm_sharpe = np.mean(sr) / (np.std(sr) + 1e-10) * np.sqrt(52)
                if perm_sharpe >= obs_sharpe:
                    count_better += 1
            p_val = count_better / min(n_perms, 500)
            if p_val < 0.05:
                gates += 1

        return _ValidationResult({
            "sharpe": round(sharpe, 4), "sortino": round(sortino, 4),
            "win_rate": round(wr, 4), "profit_factor": round(pf, 4),
            "max_dd": round(mdd, 4), "cagr": round(cagr, 4),
            "n_trades": len(trades), "final_equity": round(float(equity[-1]), 2),
            "gates_passed": gates, "gates_total": 5,
            "strategy_name": strategy_name,
        })


# ── Check optional ML libraries ──
_HAS_XGBOOST = False
_HAS_CATBOOST = False
try:
    import xgboost as xgb
    _HAS_XGBOOST = True
    fprint("XGBoost available")
except ImportError:
    fprint("XGBoost NOT available — variant B will be skipped")

try:
    import catboost
    _HAS_CATBOOST = True
    fprint("CatBoost available")
except ImportError:
    fprint("CatBoost NOT available (not required for any variant)")

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
fprint("scikit-learn available (Random Forest, Ridge)")

# ── Config ──
BASE = Path(_BASE_PATH)
OUTPUT_DIR = BASE / "output" / "growth_research" / "ml_ranker_comparison_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
DTE = 21
SPREAD_PCT = 3.0
TOP_K = 3
REGIME_BULL_THRESHOLD = 0.4
REGIME_BEAR_THRESHOLD = 0.2

# Regime predictions path
REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"

# Walk-forward
WF_TRAIN_PERIODS = 12

# V6 config
V6_REBAL_FREQ = "W-FRI"
V6_OTM_PCT = 0.02
V6_PAIRS = True
V6_MAX_POS_BULL = 200
V6_MAX_POS_PAIR_LEG = 100

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "ml_ranker_comparison_v1"

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen(MLFLOW_URI, timeout=3)
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    MLFLOW_OK = True
    fprint(f"MLflow connected: {MLFLOW_URI}")
except Exception:
    fprint("MLflow unavailable — results saved to disk only")


# ══════════════════════════════════════════════════════════════
# FEATURE DEFINITIONS (21 features = 18 legacy + 3 cross-asset)
# ══════════════════════════════════════════════════════════════

LEGACY_FEATURES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_21d", "vol_63d", "sharpe_63d", "maxdd_63d", "pct_52w_high", "mom_accel",
    "pct_pos_months_12m", "sortino_63d", "calmar_1y", "up_capture",
    "trend_r2_63d", "trend_slope_63d",
]

VALIDATED_CROSS_ASSET = [
    "sector_spy_beta_63d",
    "sector_relative_vol_21d",
    "cross_sector_dispersion",
]

ALL_FEATURES = LEGACY_FEATURES + VALIDATED_CROSS_ASSET  # 21 total


# ══════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════

def download_data():
    """Download all required tickers via yfinance."""
    import yfinance as yf

    all_tickers = SECTORS + EXTRA_TICKERS
    fprint(f"Downloading {len(all_tickers)} tickers...")

    raw = yf.download(all_tickers, start="2008-01-01", progress=False, auto_adjust=True)
    mi = isinstance(raw.columns, pd.MultiIndex)

    close = raw["Close"] if mi else raw[["Close"]]
    high = raw["High"] if mi else raw[["High"]]
    low = raw["Low"] if mi else raw[["Low"]]

    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
        high.columns = high.columns.get_level_values(-1)
        low.columns = low.columns.get_level_values(-1)

    close = close.ffill()
    high = high.ffill()
    low = low.ffill()

    rename_map = {"^VIX": "VIX", "^VIX3M": "VIX3M"}
    close = close.rename(columns=rename_map)
    high = high.rename(columns=rename_map)
    low = low.rename(columns=rename_map)

    needed = ["SPY", "VIX"]
    for t in needed:
        if t not in close.columns:
            raise ValueError(f"Missing critical ticker: {t}")

    fprint(f"Data: {len(close)} days, columns: {list(close.columns)}")
    fprint(f"Date range: {close.index[0].date()} to {close.index[-1].date()}")

    return close, high, low


# ══════════════════════════════════════════════════════════════
# REGIME LOADING
# ══════════════════════════════════════════════════════════════

def load_regime_predictions():
    """Load GRU regime predictions and build a date-indexed Series."""
    if not REGIME_FILE.exists():
        fprint(f"WARNING: Regime file not found at {REGIME_FILE}")
        fprint("  Will use VIX-based regime proxy instead")
        return None

    data = np.load(REGIME_FILE, allow_pickle=True)
    dates = pd.to_datetime(data["dates"])
    scores = data["regime_scores"]
    regime_series = pd.Series(scores, index=dates, name="regime_score")
    regime_series = regime_series[~regime_series.index.duplicated(keep="last")]
    fprint(f"Regime predictions loaded: {len(regime_series)} days "
           f"({regime_series.index[0].date()} to {regime_series.index[-1].date()})")
    fprint(f"  Mean score: {regime_series.mean():.3f}, "
           f"Days >0.4: {(regime_series > REGIME_BULL_THRESHOLD).sum()}, "
           f"Days <0.2: {(regime_series < REGIME_BEAR_THRESHOLD).sum()}")
    return regime_series


def get_regime_score_at(regime_series, dt):
    """Get regime score at a given date, with nearest-date fallback."""
    if regime_series is None:
        return 0.5
    if dt in regime_series.index:
        return float(regime_series.loc[dt])
    nearest = regime_series.index[regime_series.index.get_indexer([dt], method="ffill")]
    if len(nearest) > 0:
        return float(regime_series.loc[nearest[0]])
    return 0.5


# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════

def compute_legacy_features(px, spy_slice):
    """Compute the 18 legacy quality-momentum features for a single sector ETF."""
    if len(px) < 260:
        return None
    f = {}
    for lb, nm in [(5, "ret_5d"), (10, "ret_10d"), (21, "ret_21d"),
                   (63, "ret_63d"), (126, "ret_126d"), (252, "ret_252d")]:
        f[nm] = float(px.iloc[-1] / px.iloc[-lb] - 1) if len(px) > lb else 0.0

    rets = px.pct_change().dropna()
    f["vol_21d"] = float(rets.iloc[-21:].std() * np.sqrt(252)) if len(rets) > 21 else 0.2
    f["vol_63d"] = float(rets.iloc[-63:].std() * np.sqrt(252)) if len(rets) > 63 else 0.2
    r63 = rets.iloc[-63:]
    f["sharpe_63d"] = float(r63.mean() / (r63.std() + 1e-10) * np.sqrt(252)) if len(r63) > 10 else 0.0
    pk63 = px.iloc[-63:].cummax()
    f["maxdd_63d"] = float(((px.iloc[-63:] / pk63) - 1).min())
    f["pct_52w_high"] = float(px.iloc[-1] / px.iloc[-252:].max())
    f["mom_accel"] = f["ret_21d"] - f["ret_63d"] / 3

    monthly = rets.resample("ME").sum()
    f["pct_pos_months_12m"] = float((monthly.iloc[-12:] > 0).mean()) if len(monthly) >= 12 else 0.5
    dr = r63[r63 < 0]
    f["sortino_63d"] = float(r63.mean() / (dr.std() + 1e-10) * np.sqrt(252)) if len(dr) > 3 else 0.0
    pk = px.iloc[-252:].cummax()
    mdd = float(((px.iloc[-252:] / pk) - 1).min())
    cagr = float(px.iloc[-1] / px.iloc[-252] - 1) if len(px) >= 252 else 0.0
    f["calmar_1y"] = cagr / (abs(mdd) + 1e-10)
    up_days = rets[rets > 0]
    f["up_capture"] = float(up_days.iloc[-63:].mean() / (up_days.mean() + 1e-10)) if len(up_days) > 10 else 1.0

    if len(px) >= 63:
        y = np.log(px.iloc[-63:].values + 1e-10)
        x = np.arange(len(y))
        slope, _, r_val, _, _ = stats.linregress(x, y)
        f["trend_r2_63d"] = r_val ** 2
        f["trend_slope_63d"] = slope * 252
    else:
        f["trend_r2_63d"] = 0.0
        f["trend_slope_63d"] = 0.0

    return f


def compute_cross_asset_features(sector_ticker, dt_idx, close_df):
    """Compute the 3 validated cross-asset features only."""
    f = {}

    spy = close_df["SPY"].iloc[:dt_idx + 1].dropna()
    sector_px = close_df[sector_ticker].iloc[:dt_idx + 1].dropna() if sector_ticker in close_df.columns else None

    if spy is None or len(spy) < 63:
        return {k: 0.0 for k in VALIDATED_CROSS_ASSET}

    spy_ret = spy.pct_change().dropna()

    # 1. Sector-SPY beta 63d
    if sector_px is not None and len(sector_px) > 63:
        sec_ret = sector_px.pct_change().dropna()
        common = spy_ret.index.intersection(sec_ret.index)
        if len(common) > 63:
            sr = sec_ret.loc[common].iloc[-63:]
            mr = spy_ret.loc[common].iloc[-63:]
            cov = np.cov(sr.values, mr.values)
            beta = cov[0, 1] / (cov[1, 1] + 1e-10)
            f["sector_spy_beta_63d"] = float(beta)
        else:
            f["sector_spy_beta_63d"] = 1.0
    else:
        f["sector_spy_beta_63d"] = 1.0

    # 2. Sector relative vol 21d
    if sector_px is not None and len(sector_px) > 21:
        sec_ret = sector_px.pct_change().dropna()
        if len(sec_ret) > 21 and len(spy_ret) > 21:
            sec_vol = sec_ret.iloc[-21:].std()
            spy_vol = spy_ret.iloc[-21:].std()
            f["sector_relative_vol_21d"] = float(sec_vol / (spy_vol + 1e-10))
        else:
            f["sector_relative_vol_21d"] = 1.0
    else:
        f["sector_relative_vol_21d"] = 1.0

    # 3. Cross-sector dispersion
    sector_cols = [c for c in SECTORS if c in close_df.columns]
    if len(sector_cols) > 3:
        sector_rets = close_df[sector_cols].iloc[:dt_idx + 1].pct_change()
        daily_disp = sector_rets.std(axis=1)
        if len(daily_disp) > 21:
            f["cross_sector_dispersion"] = float(daily_disp.rolling(21).mean().iloc[-1])
        else:
            f["cross_sector_dispersion"] = 0.01
    else:
        f["cross_sector_dispersion"] = 0.01

    return f


# ══════════════════════════════════════════════════════════════
# BUILD FEATURE RECORDS
# ══════════════════════════════════════════════════════════════

def build_feature_records(close, high, low, rebal_dates, feature_cols, regime_series):
    """Build feature + target records for all sectors on all rebal dates."""
    fprint(f"  Building records: {len(rebal_dates)} dates, {len(feature_cols)} features")

    records = []
    sector_cols = [c for c in SECTORS if c in close.columns]
    spy = close["SPY"]

    for dt in rebal_dates:
        idx = close.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
            continue

        rscore = get_regime_score_at(regime_series, dt)
        if rscore <= REGIME_BULL_THRESHOLD:
            continue

        for tk in sector_cols:
            px = close[tk].iloc[:idx + 1].dropna()
            spy_s = spy.iloc[:idx + 1]

            legacy = compute_legacy_features(px, spy_s)
            if not legacy:
                continue

            cross_asset = {}
            needs_cross = any(col in VALIDATED_CROSS_ASSET for col in feature_cols)
            if needs_cross:
                cross_asset = compute_cross_asset_features(tk, idx, close)

            fi = min(idx + DTE, len(close) - 1)
            if fi <= idx:
                continue
            fwd_ret = float(close[tk].iloc[fi] / close[tk].iloc[idx] - 1)

            rec = {**legacy, **cross_asset, "date": dt, "ticker": tk, "fwd_ret": fwd_ret}
            records.append(rec)

    df = pd.DataFrame(records)
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    df[feature_cols] = df[feature_cols].fillna(0.0)

    fprint(f"    {len(df)} records, {len(df['date'].unique())} dates")
    return df


# ══════════════════════════════════════════════════════════════
# ML MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════

def create_model(variant_key):
    """
    Create a model instance for the given variant.
    Returns (model, model_type) or (None, None) if unavailable.
    model_type is 'tree' or 'linear' — determines how we get feature importance.
    """
    import lightgbm as lgb

    if variant_key == "A_lgbm_baseline":
        return lgb.LGBMRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1,
        ), "lgbm"

    elif variant_key == "B_xgboost":
        if not _HAS_XGBOOST:
            return None, None
        return xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, verbosity=0,
        ), "xgb"

    elif variant_key == "C_random_forest":
        return RandomForestRegressor(
            n_estimators=100, max_depth=4, min_samples_leaf=5,
            max_features=0.8, n_jobs=-1, random_state=42,
        ), "rf"

    elif variant_key == "D_ridge":
        return Ridge(alpha=1.0), "linear"

    elif variant_key == "E_lgbm_deeper":
        return lgb.LGBMRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1,
        ), "lgbm"

    else:
        return None, None


def get_feature_importance(model, model_type, feature_cols):
    """Extract feature importance from trained model."""
    if model_type == "lgbm":
        return model.feature_importances_
    elif model_type == "xgb":
        return model.feature_importances_
    elif model_type == "rf":
        return model.feature_importances_
    elif model_type == "linear":
        return np.abs(model.coef_)
    return np.zeros(len(feature_cols))


# ══════════════════════════════════════════════════════════════
# WALK-FORWARD RANKING (GENERIC — supports any sklearn-like model)
# ══════════════════════════════════════════════════════════════

def walk_forward_rank(df, feature_cols, variant_key, variant_name):
    """Walk-forward ranking using any model type. Returns rankings dict and importance df."""
    if len(df) < 100:
        fprint(f"    {variant_name}: Insufficient data ({len(df)} records)")
        return {}, None

    df = df.copy()
    df["rank_label"] = df.groupby("date")["fwd_ret"].rank(pct=True)
    dates = sorted(df["date"].unique())

    rankings = {}
    all_importances = np.zeros(len(feature_cols))
    n_models = 0

    for i in range(WF_TRAIN_PERIODS, len(dates)):
        train_dates = dates[max(0, i - WF_TRAIN_PERIODS):i]
        test_date = dates[i]

        train_df = df[df["date"].isin(train_dates)]
        test_df = df[df["date"] == test_date].copy()

        if len(test_df) < 3 or len(train_df) < 50:
            continue

        Xt = np.nan_to_num(train_df[feature_cols].values.astype(np.float64))
        yt = train_df["rank_label"].values.astype(np.float64)
        Xe = np.nan_to_num(test_df[feature_cols].values.astype(np.float64))

        try:
            model, model_type = create_model(variant_key)
            if model is None:
                return {}, None

            model.fit(Xt, yt)
            test_df["score"] = model.predict(Xe)

            rankings[test_date] = dict(zip(test_df["ticker"], test_df["score"]))

            imp = get_feature_importance(model, model_type, feature_cols)
            all_importances += imp
            n_models += 1
        except Exception as e:
            if n_models == 0:
                fprint(f"    {variant_name}: First model failed: {e}")
            continue

    if n_models > 0:
        all_importances /= n_models
        imp_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": all_importances,
        }).sort_values("importance", ascending=False)
    else:
        imp_df = None

    fprint(f"    {variant_name}: {len(rankings)} ranking dates, {n_models} models trained")
    return rankings, imp_df


def walk_forward_ensemble_rank(df, feature_cols, variant_name):
    """
    Ensemble: average rankings from LGBM + XGBoost + Random Forest.
    If XGBoost is unavailable, average LGBM + RF only.
    """
    import lightgbm as lgb

    if len(df) < 100:
        fprint(f"    {variant_name}: Insufficient data ({len(df)} records)")
        return {}, None

    df = df.copy()
    df["rank_label"] = df.groupby("date")["fwd_ret"].rank(pct=True)
    dates = sorted(df["date"].unique())

    rankings = {}
    n_models = 0

    # Define ensemble members
    ensemble_models = [
        ("LGBM", lambda: lgb.LGBMRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1,
        )),
        ("RF", lambda: RandomForestRegressor(
            n_estimators=100, max_depth=4, min_samples_leaf=5,
            max_features=0.8, n_jobs=-1, random_state=42,
        )),
    ]
    if _HAS_XGBOOST:
        ensemble_models.insert(1, ("XGB", lambda: xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, verbosity=0,
        )))
    member_names = [m[0] for m in ensemble_models]
    fprint(f"    Ensemble members: {member_names}")

    for i in range(WF_TRAIN_PERIODS, len(dates)):
        train_dates = dates[max(0, i - WF_TRAIN_PERIODS):i]
        test_date = dates[i]

        train_df = df[df["date"].isin(train_dates)]
        test_df = df[df["date"] == test_date].copy()

        if len(test_df) < 3 or len(train_df) < 50:
            continue

        Xt = np.nan_to_num(train_df[feature_cols].values.astype(np.float64))
        yt = train_df["rank_label"].values.astype(np.float64)
        Xe = np.nan_to_num(test_df[feature_cols].values.astype(np.float64))

        all_scores = []
        for mname, mfactory in ensemble_models:
            try:
                m = mfactory()
                m.fit(Xt, yt)
                scores = m.predict(Xe)
                all_scores.append(scores)
            except Exception:
                continue

        if not all_scores:
            continue

        # Average the scores across ensemble members
        avg_scores = np.mean(all_scores, axis=0)
        test_df["score"] = avg_scores
        rankings[test_date] = dict(zip(test_df["ticker"], test_df["score"]))
        n_models += 1

    fprint(f"    {variant_name}: {len(rankings)} ranking dates, {n_models} ensemble iterations")
    return rankings, None  # No single importance for ensemble


# ══════════════════════════════════════════════════════════════
# ATR COMPUTATION
# ══════════════════════════════════════════════════════════════

def compute_atr_series(high, low, close, period=14):
    """Compute ATR series for all sectors."""
    atr_dict = {}
    for tk in SECTORS:
        if tk in high.columns and tk in low.columns and tk in close.columns:
            h = high[tk].dropna()
            l = low[tk].dropna()
            c = close[tk].dropna()
            common = h.index.intersection(l.index).intersection(c.index)
            if len(common) > period:
                tr1 = h.loc[common] - l.loc[common]
                tr2 = (h.loc[common] - c.loc[common].shift(1)).abs()
                tr3 = (l.loc[common] - c.loc[common].shift(1)).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr_dict[tk] = tr.ewm(alpha=1/period, min_periods=period).mean()
    return atr_dict


# ══════════════════════════════════════════════════════════════
# STRIKE COMPUTATION (2% OTM per V6)
# ══════════════════════════════════════════════════════════════

def compute_strikes(S, direction, otm_pct, spread_pct):
    """Compute strike prices for a spread."""
    if direction == "bull":
        if otm_pct > 0:
            K1 = round(S * (1 + otm_pct), 2)
            K2 = round(K1 * (1 + spread_pct / 100), 2)
        else:
            K1 = round(S, 2)
            K2 = round(S * (1 + spread_pct / 100), 2)
    else:
        if otm_pct > 0:
            K2 = round(S * (1 - otm_pct), 2)
            K1 = round(K2 * (1 - spread_pct / 100), 2)
        else:
            K1 = round(S * (1 - spread_pct / 100), 2)
            K2 = round(S, 2)
    if K2 <= K1:
        K2 = K1 + 0.50
    return K1, K2


# ══════════════════════════════════════════════════════════════
# TRADE SIMULATION (V6 config)
# ══════════════════════════════════════════════════════════════

def _execute_single_trade(tk, dt, direction, otm_pct, max_pos, close, atr_dict, cv, equity):
    """Execute a single spread trade. Returns PnL or None."""
    if tk not in close.columns or tk not in atr_dict:
        return None

    S = float(close[tk].loc[dt])
    di = close.index.get_loc(dt)
    ei = min(di + DTE, len(close) - 1)
    if ei <= di:
        return None

    if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]):
        av = float(atr_dict[tk].loc[dt])
    else:
        av = S * 0.015

    K1, K2 = compute_strikes(S, direction, otm_pct, SPREAD_PCT)

    try:
        if direction == "bull":
            entry_cost_ps, max_profit_ps = price_bull_call_spread(
                S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=cv
            )
        else:
            entry_cost_ps, max_profit_ps = price_bear_put_spread(
                S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=cv
            )
    except Exception:
        return None

    total_cost = entry_cost_ps * 100 + COMMISSION_RT_SPREAD

    if total_cost <= 0 or total_cost > max_pos or total_cost > equity * 0.40:
        return None

    Se = float(close[tk].iloc[ei])

    if direction == "bull":
        intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
    else:
        intrinsic = max(K2 - Se, 0.0) - max(K1 - Se, 0.0)

    pnl = (intrinsic - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD
    return pnl


def simulate_trades(name, rankings, close, high, low, atr_dict):
    """Simulate trades using V6 config: bull VIX>=20, pairs VIX<20."""
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    equity = CAP
    trades = []

    for dt in sorted(rankings.keys()):
        if dt not in spy.index:
            continue

        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0
        scores = rankings[dt]
        if not scores:
            continue

        if V6_PAIRS and cv < 20.0:
            trade_mode = "pairs"
        else:
            trade_mode = "bull_only"

        ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_asc = sorted(scores.items(), key=lambda x: x[1])

        bull_picks = [t for t, _ in ranked_desc[:TOP_K]]
        bear_picks = [t for t, _ in ranked_asc[:TOP_K]] if trade_mode == "pairs" else []

        if trade_mode == "pairs":
            max_pos = min(V6_MAX_POS_PAIR_LEG, equity / 6)
        else:
            max_pos = min(V6_MAX_POS_BULL, equity / 3)

        if max_pos < 30:
            continue

        for tk in bull_picks:
            pnl = _execute_single_trade(
                tk, dt, "bull", V6_OTM_PCT, max_pos, close, atr_dict, cv, equity
            )
            if pnl is not None:
                equity += pnl
                di = close.index.get_loc(dt)
                ei = min(di + DTE, len(close) - 1)
                sv = float(spy.loc[dt])
                se = float(spy.iloc[ei]) if ei < len(spy) else sv
                spy_regime = "bull" if se >= sv else "bear"
                trades.append({
                    "pnl": round(pnl, 2),
                    "entry_date": str(dt.date()),
                    "exit_date": str(close.index[ei].date()),
                    "ticker": tk,
                    "regime": spy_regime,
                    "direction": "bull",
                    "vix": round(cv, 1),
                    "win": pnl > 0,
                    "trade_mode": trade_mode,
                })

        for tk in bear_picks:
            pnl = _execute_single_trade(
                tk, dt, "bear", V6_OTM_PCT, max_pos, close, atr_dict, cv, equity
            )
            if pnl is not None:
                equity += pnl
                di = close.index.get_loc(dt)
                ei = min(di + DTE, len(close) - 1)
                sv = float(spy.loc[dt])
                se = float(spy.iloc[ei]) if ei < len(spy) else sv
                spy_regime = "bull" if se >= sv else "bear"
                trades.append({
                    "pnl": round(pnl, 2),
                    "entry_date": str(dt.date()),
                    "exit_date": str(close.index[ei].date()),
                    "ticker": tk,
                    "regime": spy_regime,
                    "direction": "bear",
                    "vix": round(cv, 1),
                    "win": pnl > 0,
                    "trade_mode": trade_mode,
                })

    return trades, equity


# ══════════════════════════════════════════════════════════════
# RANDOM BASELINE
# ══════════════════════════════════════════════════════════════

def random_baseline_test(rankings, close, high, low, atr_dict, n_trials=5):
    """Test if random sector selection produces similar returns."""
    fprint(f"\n  Random baseline ({n_trials} trials)...")
    random_sharpes = []

    for trial in range(n_trials):
        np.random.seed(42 + trial)
        rand_rankings = {}
        for dt, scores in rankings.items():
            rand_scores = {tk: np.random.random() for tk in scores.keys()}
            rand_rankings[dt] = rand_scores

        trades, final_eq = simulate_trades(
            f"Random_{trial}", rand_rankings, close, high, low, atr_dict
        )

        if trades and len(trades) >= 10:
            result = validate_trades(
                trades, initial_capital=CAP,
                spy_prices=close["SPY"],
                strategy_name=f"Random_{trial}",
                n_perms=500,
            )
            random_sharpes.append(result.sharpe)
            fprint(f"    Random trial {trial}: Sharpe {result.sharpe:.2f}, "
                   f"${CAP:.0f}->${final_eq:.0f}")
        else:
            random_sharpes.append(0.0)

    return random_sharpes


# ══════════════════════════════════════════════════════════════
# REBALANCE DATE GENERATION
# ══════════════════════════════════════════════════════════════

def generate_rebal_dates(close, freq_str):
    """Generate rebalance dates from close index."""
    rebal_dates = pd.DatetimeIndex(
        close.index.to_series().resample(freq_str).last().dropna().values
    )
    return rebal_dates


# ══════════════════════════════════════════════════════════════
# VARIANT DEFINITIONS
# ══════════════════════════════════════════════════════════════

RANKER_VARIANTS = {
    "A_lgbm_baseline": {
        "desc": "LGBM baseline (100 trees, depth 4, lr 0.05)",
        "type": "single",
    },
    "B_xgboost": {
        "desc": "XGBoost (100 trees, depth 4, lr 0.05)",
        "type": "single",
    },
    "C_random_forest": {
        "desc": "Random Forest (100 trees, max_depth 4)",
        "type": "single",
    },
    "D_ridge": {
        "desc": "Ridge Regression (alpha=1.0, linear)",
        "type": "single",
    },
    "E_lgbm_deeper": {
        "desc": "LGBM deeper (200 trees, depth 6, lr 0.05)",
        "type": "single",
    },
    "F_ensemble": {
        "desc": "Ensemble: avg rankings from LGBM+XGB+RF (or LGBM+RF if no XGB)",
        "type": "ensemble",
    },
}


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t0 = datetime.now()
    fprint("=" * 100)
    fprint(f"ML RANKER COMPARISON v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 100)
    fprint(f"Question: Is LGBM the best ranker, or can XGBoost/RF/Ridge/Ensemble beat it?")
    fprint(f"Method: Replace ONLY the ranking model, keep V6 trade structure identical")
    fprint()
    fprint(f"V6 config (fixed for ALL variants):")
    fprint(f"  Capital: ${CAP:.0f} | DTE: {DTE} | Spread: {SPREAD_PCT:.0f}% | "
           f"Haircut: {DEFAULT_HAIRCUT:.0%} entry only | Comm: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"  Rebalance: weekly (W-FRI) | OTM: 2% | Pairs: VIX<20 bull+bear")
    fprint(f"  Hold to expiry | Intrinsic value only | No exit haircut")
    fprint(f"  Regime filter: GRU >0.4 | Features: {len(ALL_FEATURES)}")
    fprint()
    fprint(f"6 ranker variants:")
    for vname, vcfg in RANKER_VARIANTS.items():
        skip = ""
        if vname == "B_xgboost" and not _HAS_XGBOOST:
            skip = " [SKIP — not installed]"
        fprint(f"  {vname}: {vcfg['desc']}{skip}")
    fprint()

    # 1. Download data
    close, high, low = download_data()

    # 2. Load regime predictions
    regime_series = load_regime_predictions()

    # 3. Pre-compute ATR
    atr_dict = compute_atr_series(high, low, close)

    # 4. Generate V6 rebalance dates (weekly)
    rebal_dates = generate_rebal_dates(close, V6_REBAL_FREQ)
    fprint(f"Rebalance dates: {len(rebal_dates)} "
           f"({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

    # 5. Build feature records ONCE (all variants use the same features)
    fprint("\nBuilding feature records (shared across all variants)...")
    records = build_feature_records(
        close, high, low, rebal_dates, ALL_FEATURES, regime_series
    )

    spy_close = close["SPY"]

    # 6. Run each variant
    all_results = {}
    all_importances = {}
    all_rankings = {}  # Store rankings for ensemble and comparison

    for vname, vcfg in RANKER_VARIANTS.items():
        fprint(f"\n{'=' * 100}")
        fprint(f"VARIANT {vname}: {vcfg['desc']}")
        fprint(f"{'=' * 100}")

        # Skip unavailable models
        if vname == "B_xgboost" and not _HAS_XGBOOST:
            fprint("  SKIPPED — XGBoost not installed")
            continue

        # Run walk-forward
        if vcfg["type"] == "ensemble":
            rankings, imp_df = walk_forward_ensemble_rank(records, ALL_FEATURES, vname)
        else:
            rankings, imp_df = walk_forward_rank(records, ALL_FEATURES, vname, vname)

        all_importances[vname] = imp_df
        all_rankings[vname] = rankings

        if not rankings:
            fprint(f"  No rankings available, skipping")
            continue

        # Simulate trades with V6 config
        trades, final_eq = simulate_trades(
            vname, rankings, close, high, low, atr_dict
        )

        if not trades or len(trades) < 10:
            fprint(f"  Only {len(trades) if trades else 0} trades, skipping validation")
            continue

        # 5-gate adversarial validation
        result = validate_trades(
            trades, initial_capital=CAP,
            spy_prices=spy_close,
            strategy_name=vname,
        )
        result.print_summary()

        # Direction breakdown
        bull_trades = [t for t in trades if t["direction"] == "bull"]
        bear_trades = [t for t in trades if t["direction"] == "bear"]
        pair_trades = [t for t in trades if t.get("trade_mode") == "pairs"]
        bull_pnl = sum(t["pnl"] for t in bull_trades)
        bear_pnl = sum(t["pnl"] for t in bear_trades)
        bull_wr = sum(1 for t in bull_trades if t["win"]) / max(len(bull_trades), 1) * 100
        bear_wr = sum(1 for t in bear_trades if t["win"]) / max(len(bear_trades), 1) * 100
        fprint(f"  Direction breakdown:")
        fprint(f"    Bull: {len(bull_trades)} trades, WR {bull_wr:.1f}%, PnL ${bull_pnl:.0f}")
        fprint(f"    Bear: {len(bear_trades)} trades, WR {bear_wr:.1f}%, PnL ${bear_pnl:.0f}")
        fprint(f"    Pair-mode trades: {len(pair_trades)} (VIX<20 dates)")

        # Random baseline
        random_sharpes = random_baseline_test(
            rankings, close, high, low, atr_dict,
        )
        mean_random = np.mean(random_sharpes) if random_sharpes else 0
        fprint(f"  ML Sharpe: {result.sharpe:.2f} vs Random mean: {mean_random:.2f}")
        if result.sharpe > 0 and mean_random > 0:
            fprint(f"  ML alpha ratio: {result.sharpe / mean_random:.2f}x")

        all_results[vname] = {
            "description": vcfg["desc"],
            **result.to_dict(),
            "random_sharpes": [round(s, 3) for s in random_sharpes],
            "random_mean_sharpe": round(mean_random, 3),
            "bull_trades": len(bull_trades),
            "bear_trades": len(bear_trades),
            "bull_pnl": round(bull_pnl, 2),
            "bear_pnl": round(bear_pnl, 2),
            "bull_wr": round(bull_wr, 1),
            "bear_wr": round(bear_wr, 1),
        }

    # ── RANKING AGREEMENT ANALYSIS ──
    fprint(f"\n{'=' * 100}")
    fprint("RANKING AGREEMENT ANALYSIS — How similar are the models' rankings?")
    fprint(f"{'=' * 100}")

    variant_keys = [k for k in all_rankings if all_rankings[k]]
    if len(variant_keys) >= 2:
        # Compute rank correlation between pairs of models on common dates
        for i, v1 in enumerate(variant_keys):
            for v2 in variant_keys[i+1:]:
                r1 = all_rankings[v1]
                r2 = all_rankings[v2]
                common_dates = set(r1.keys()) & set(r2.keys())
                if not common_dates:
                    continue
                correlations = []
                for dt in common_dates:
                    tickers = sorted(set(r1[dt].keys()) & set(r2[dt].keys()))
                    if len(tickers) < 5:
                        continue
                    s1 = [r1[dt][t] for t in tickers]
                    s2 = [r2[dt][t] for t in tickers]
                    corr, _ = stats.spearmanr(s1, s2)
                    if not np.isnan(corr):
                        correlations.append(corr)
                if correlations:
                    mean_corr = np.mean(correlations)
                    fprint(f"  {v1} vs {v2}: mean Spearman rank corr = {mean_corr:.3f} "
                           f"({len(correlations)} dates)")

    # ── SUMMARY COMPARISON ──
    fprint(f"\n{'=' * 120}")
    fprint("ML RANKER COMPARISON SUMMARY")
    fprint(f"{'=' * 120}")
    fprint(f"{'Variant':<25} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} "
           f"{'PF':>6} {'MaxDD':>7} {'Gates':>6} {'Final$':>8} {'RandSh':>7} {'Alpha':>7}")
    fprint("-" * 120)

    baseline_sharpe = all_results.get("A_lgbm_baseline", {}).get("sharpe", 0)

    for vname in RANKER_VARIANTS.keys():
        r = all_results.get(vname)
        if not r:
            fprint(f"  {vname:<25} — SKIPPED —")
            continue
        alpha = r["sharpe"] / r["random_mean_sharpe"] if r["random_mean_sharpe"] > 0 else float('inf')
        fprint(f"  {vname:<25} {r['n_trades']:>5} {r['sharpe']:>7.2f} "
               f"{r['sortino']:>8.2f} {r['win_rate']*100:>5.1f}% {r['profit_factor']:>5.2f} "
               f"{r['max_dd']*100:>6.1f}% {r['gates_passed']}/{r['gates_total']} "
               f"${r['final_equity']:>7,.0f} {r['random_mean_sharpe']:>7.2f} {alpha:>6.1f}x")

    # ── SHARPE DELTA vs BASELINE ──
    fprint(f"\n{'=' * 100}")
    fprint(f"SHARPE DELTA vs LGBM BASELINE (A)")
    fprint(f"{'=' * 100}")
    fprint(f"  Baseline (A_lgbm_baseline) Sharpe: {baseline_sharpe:.2f}")
    fprint()

    deltas = []
    for vname in RANKER_VARIANTS.keys():
        if vname == "A_lgbm_baseline":
            continue
        r = all_results.get(vname)
        if not r:
            continue
        delta = r["sharpe"] - baseline_sharpe
        pct_delta = delta / abs(baseline_sharpe) * 100 if baseline_sharpe != 0 else 0
        deltas.append((vname, delta, pct_delta, r["sharpe"]))

    deltas.sort(key=lambda x: x[1], reverse=True)

    for vname, delta, pct_delta, sharpe in deltas:
        sign = "BETTER" if delta > 0.1 else "WORSE" if delta < -0.1 else "SIMILAR"
        max_delta = max(abs(d[1]) for d in deltas) if deltas else 1
        bar = "*" * int(abs(delta) / max_delta * 30) if max_delta > 0 else ""
        fprint(f"  {vname:<25} Sharpe {sharpe:>5.2f}  "
               f"delta {delta:>+6.2f} ({pct_delta:>+5.1f}%)  "
               f"{sign}  {bar}")

    # ── KEY FINDINGS ──
    fprint(f"\n{'=' * 100}")
    fprint("KEY FINDINGS")
    fprint(f"{'=' * 100}")

    if deltas:
        best = max(deltas, key=lambda x: x[1])
        worst = min(deltas, key=lambda x: x[1])
        fprint(f"  Best alternative:  {best[0]} (Sharpe {best[3]:.2f}, delta {best[1]:+.2f})")
        fprint(f"  Worst alternative: {worst[0]} (Sharpe {worst[3]:.2f}, delta {worst[1]:+.2f})")

        if best[1] > 0.1:
            fprint(f"\n  CONCLUSION: {best[0]} BEATS LGBM baseline by {best[1]:+.2f} Sharpe.")
            fprint(f"    Consider switching the production ranker from LGBM to {best[0]}.")
        elif best[1] > -0.1:
            fprint(f"\n  CONCLUSION: No model significantly beats LGBM. LGBM remains optimal.")
            fprint(f"    All alternatives within +/- 0.1 Sharpe of baseline.")
        else:
            fprint(f"\n  CONCLUSION: LGBM is the BEST ranker. All alternatives are worse.")

    # Check ensemble vs individual
    f_result = all_results.get("F_ensemble", {})
    if f_result:
        f_delta = f_result.get("sharpe", 0) - baseline_sharpe
        if f_delta > 0.1:
            fprint(f"  FINDING: Ensemble (avg LGBM+XGB+RF) IMPROVES over single LGBM "
                   f"(delta {f_delta:+.2f})")
        else:
            fprint(f"  FINDING: Ensemble does NOT improve over LGBM (delta {f_delta:+.2f})")

    # Check linear vs tree
    d_result = all_results.get("D_ridge", {})
    if d_result:
        d_delta = d_result.get("sharpe", 0) - baseline_sharpe
        if abs(d_delta) < 0.3:
            fprint(f"  FINDING: Linear model (Ridge) nearly matches trees (delta {d_delta:+.2f})")
            fprint(f"    => Feature relationships may be mostly linear")
        else:
            fprint(f"  FINDING: Linear model significantly {'better' if d_delta > 0 else 'worse'} "
                   f"than trees (delta {d_delta:+.2f})")

    # ── FEATURE IMPORTANCE COMPARISON ──
    fprint(f"\n{'=' * 100}")
    fprint("FEATURE IMPORTANCE BY MODEL (Top 5)")
    fprint(f"{'=' * 100}")
    for vname in ["A_lgbm_baseline", "B_xgboost", "C_random_forest", "D_ridge", "E_lgbm_deeper"]:
        imp = all_importances.get(vname)
        if imp is not None:
            fprint(f"\n  {vname}:")
            for _, row in imp.head(5).iterrows():
                bar = "*" * int(row["importance"] / (imp["importance"].max() + 1e-10) * 25)
                fprint(f"    {row['feature']:<30} {row['importance']:>8.1f} {bar}")

    # Save results
    results_path = OUTPUT_DIR / "ml_ranker_comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    # MLflow logging
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"ml_ranker_{t0.strftime('%Y%m%d_%H%M')}"):
                for vname, r in all_results.items():
                    prefix = vname.split("_")[0]
                    mlflow.log_metric(f"{prefix}_sharpe", r.get("sharpe", 0))
                    mlflow.log_metric(f"{prefix}_sortino", r.get("sortino", 0))
                    mlflow.log_metric(f"{prefix}_win_rate", r.get("win_rate", 0))
                    mlflow.log_metric(f"{prefix}_profit_factor", r.get("profit_factor", 0))
                    mlflow.log_metric(f"{prefix}_max_dd", r.get("max_dd", 0))
                    mlflow.log_metric(f"{prefix}_n_trades", r.get("n_trades", 0))
                    mlflow.log_metric(f"{prefix}_gates_passed", r.get("gates_passed", 0))
                    mlflow.log_metric(f"{prefix}_final_equity", r.get("final_equity", 0))
                    mlflow.log_metric(f"{prefix}_random_mean_sharpe", r.get("random_mean_sharpe", 0))

                    delta = r.get("sharpe", 0) - baseline_sharpe
                    mlflow.log_metric(f"{prefix}_sharpe_delta_vs_lgbm", delta)

                mlflow.log_params({
                    "experiment_type": "ml_ranker_comparison",
                    "capital": CAP,
                    "dte": DTE,
                    "spread_pct": SPREAD_PCT,
                    "haircut": DEFAULT_HAIRCUT,
                    "commission": COMMISSION_RT_SPREAD,
                    "regime_bull_thresh": REGIME_BULL_THRESHOLD,
                    "rebal_freq": V6_REBAL_FREQ,
                    "otm_pct": V6_OTM_PCT,
                    "pairs": V6_PAIRS,
                    "n_features": len(ALL_FEATURES),
                    "wf_train_periods": WF_TRAIN_PERIODS,
                    "xgboost_available": _HAS_XGBOOST,
                    "n_variants_run": len(all_results),
                })

                mlflow.log_artifact(str(results_path))
            fprint(f"MLflow run logged to experiment '{EXPERIMENT_NAME}'")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    fprint("DONE")


if __name__ == "__main__":
    main()
