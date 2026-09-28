#!/usr/bin/env python3
"""
Regime-Adaptive Parameter Experiment v1
========================================

Question: Does adapting K, rebalance frequency, or OTM level to the VIX regime
improve Sharpe vs fixed parameters?

OOS showed different Sharpe by regime: crisis 1.96, bull 3.32, bear 1.99, recent 1.89.
If the strategy adapts parameters to VIX level, can we flatten this variance?

6 Variants — ALL use walk-forward LGBM, GRU regime filter, DTE=21, 3% width:

| Variant              | Description                                                      |
|----------------------|------------------------------------------------------------------|
| A_fixed_v6           | Fixed: K=2, 5d, 2% OTM everywhere (control)                     |
| B_aggressive_highvol | VIX>25: K=3, 3d, 3% OTM. Else: K=2, 5d, 2% OTM                 |
| C_conservative_lowvol| VIX<15: K=1, 7d, 1% OTM. Else: K=2, 5d, 2% OTM                 |
| D_three_tier         | VIX<15: K=1,7d,1%OTM. VIX15-25: K=2,5d,2%OTM. VIX>25: K=3,3d,3%OTM |
| E_adaptive_k         | Only adapt K by VIX tier. Fixed 5d, 2% OTM.                     |
| F_adaptive_rebal     | Only adapt rebal freq by VIX tier. Fixed K=2, 2% OTM.           |

Base: moneyness_crossval_v1.py (production v4 infrastructure).
5-gate adversarial validation + random baseline on all variants.
Results logged to MLflow experiment "regime_adaptive_v1".
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


# ── Standardized tools ──
sys.path.insert(0, "/home/nick/Lvl3Quant")
from research.tools.options_pricer import (
    price_bull_call_spread,
    price_bear_put_spread,
    estimate_iv,
    compute_atr,
    COMMISSION_RT_SPREAD,
    DEFAULT_HAIRCUT,
)
from research.tools.adversarial_validator import validate_trades

# ── Config ──
BASE = Path("/home/nick/Lvl3Quant")
OUTPUT_DIR = BASE / "output" / "growth_research" / "regime_adaptive_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
MAX_POS = 200.0
SPREAD_PCT = 3.0
REGIME_BULL_THRESHOLD = 0.4
REGIME_BEAR_THRESHOLD = 0.2
DTE = 21

# Regime predictions path
REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"

# LGBM walk-forward
WF_TRAIN_PERIODS = 12

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "regime_adaptive_v1"

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen(MLFLOW_URI, timeout=3)
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    MLFLOW_OK = True
    fprint(f"MLflow connected: {MLFLOW_URI}")
except Exception:
    fprint("MLflow unavailable - results saved to disk only")

# ══════════════════════════════════════════════════════════════
# VARIANT GRID — Regime-adaptive parameter rules
# ══════════════════════════════════════════════════════════════
VARIANTS = [
    # A: Fixed v6 baseline (control)
    {"name": "A_fixed_v6", "desc": "Fixed: K=2, 5d, 2% OTM everywhere",
     "rules": {"default": {"top_k": 2, "rebal_days": 5, "moneyness_pct": 2.0}}},

    # B: More aggressive when VIX high (regime score > 0.6 = very elevated)
    {"name": "B_aggressive_highvol", "desc": "VIX>25: K=3, 3d, 3% OTM. Else: K=2, 5d, 2% OTM",
     "rules": {"vix_gt_25": {"top_k": 3, "rebal_days": 3, "moneyness_pct": 3.0},
               "default": {"top_k": 2, "rebal_days": 5, "moneyness_pct": 2.0}}},

    # C: More conservative when VIX low (reduce OTM)
    {"name": "C_conservative_lowvol", "desc": "VIX<15: K=1, 7d, 1% OTM. Else: K=2, 5d, 2% OTM",
     "rules": {"vix_lt_15": {"top_k": 1, "rebal_days": 7, "moneyness_pct": 1.0},
               "default": {"top_k": 2, "rebal_days": 5, "moneyness_pct": 2.0}}},

    # D: 3-tier VIX adaptation
    {"name": "D_three_tier", "desc": "VIX<15: K=1,7d,1%OTM. VIX15-25: K=2,5d,2%OTM. VIX>25: K=3,3d,3%OTM",
     "rules": {"vix_lt_15": {"top_k": 1, "rebal_days": 7, "moneyness_pct": 1.0},
               "vix_15_25": {"top_k": 2, "rebal_days": 5, "moneyness_pct": 2.0},
               "vix_gt_25": {"top_k": 3, "rebal_days": 3, "moneyness_pct": 3.0}}},

    # E: Only adapt K (simplest)
    {"name": "E_adaptive_k", "desc": "VIX<15: K=1. VIX15-25: K=2. VIX>25: K=3. Fixed 5d, 2% OTM",
     "rules": {"vix_lt_15": {"top_k": 1, "rebal_days": 5, "moneyness_pct": 2.0},
               "vix_15_25": {"top_k": 2, "rebal_days": 5, "moneyness_pct": 2.0},
               "vix_gt_25": {"top_k": 3, "rebal_days": 5, "moneyness_pct": 2.0}}},

    # F: Only adapt rebal frequency
    {"name": "F_adaptive_rebal", "desc": "VIX<15: 7d. VIX15-25: 5d. VIX>25: 3d. Fixed K=2, 2% OTM",
     "rules": {"vix_lt_15": {"top_k": 2, "rebal_days": 7, "moneyness_pct": 2.0},
               "vix_15_25": {"top_k": 2, "rebal_days": 5, "moneyness_pct": 2.0},
               "vix_gt_25": {"top_k": 2, "rebal_days": 3, "moneyness_pct": 2.0}}},
]


def resolve_params(rules, vix_level):
    """Given a variant's VIX-conditional rules and current VIX, return the active parameters."""
    if vix_level < 15.0 and "vix_lt_15" in rules:
        return rules["vix_lt_15"]
    elif vix_level >= 15.0 and vix_level <= 25.0 and "vix_15_25" in rules:
        return rules["vix_15_25"]
    elif vix_level > 25.0 and "vix_gt_25" in rules:
        return rules["vix_gt_25"]
    return rules["default"]


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
        return 0.5  # neutral default
    if dt in regime_series.index:
        return float(regime_series.loc[dt])
    nearest = regime_series.index[regime_series.index.get_indexer([dt], method="ffill")]
    if len(nearest) > 0:
        return float(regime_series.loc[nearest[0]])
    return 0.5


# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (IDENTICAL TO PRODUCTION V4)
# ══════════════════════════════════════════════════════════════

# Legacy 18 features (quality-momentum)
LEGACY_FEATURES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_21d", "vol_63d", "sharpe_63d", "maxdd_63d", "pct_52w_high", "mom_accel",
    "pct_pos_months_12m", "sortino_63d", "calmar_1y", "up_capture",
    "trend_r2_63d", "trend_slope_63d",
]

# Only the 3 validated cross-asset features (findings #64-65)
VALIDATED_CROSS_ASSET = [
    "sector_spy_beta_63d",
    "sector_relative_vol_21d",
    "cross_sector_dispersion",
]

V4_FEATURES = LEGACY_FEATURES + VALIDATED_CROSS_ASSET  # 21 total


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

    # 3. Cross-sector dispersion (rolling 21d stdev of sector returns)
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
# WALK-FORWARD LGBM RANKING
# ══════════════════════════════════════════════════════════════

def build_feature_records(close, high, low, rebal_dates, feature_cols, regime_series,
                          dte, regime_mode="bull_bear"):
    """
    Build feature + target records for all sectors on all rebal dates.
    DTE is parameterized for the forward return target.
    """
    import lightgbm as lgb

    fprint(f"  Building records: {len(rebal_dates)} dates, {len(feature_cols)} features, "
           f"mode={regime_mode}, dte={dte}")

    records = []
    sector_cols = [c for c in SECTORS if c in close.columns]
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    for dt in rebal_dates:
        idx = close.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
            continue

        # Get regime score
        rscore = get_regime_score_at(regime_series, dt)

        # Filter by regime mode
        if regime_mode == "bull_only":
            if rscore <= REGIME_BULL_THRESHOLD:
                continue
            direction = "bull"
        elif regime_mode == "bull_bear":
            if rscore > REGIME_BULL_THRESHOLD:
                direction = "bull"
            elif rscore < REGIME_BEAR_THRESHOLD:
                direction = "bear"
            else:
                continue  # gray zone, skip
        else:
            direction = "bull"

        for tk in sector_cols:
            px = close[tk].iloc[:idx + 1].dropna()
            spy_s = spy.iloc[:idx + 1]

            legacy = compute_legacy_features(px, spy_s)
            if not legacy:
                continue

            # Cross-asset features if needed
            cross_asset = {}
            for col in feature_cols:
                if col in VALIDATED_CROSS_ASSET:
                    cross_asset = compute_cross_asset_features(tk, idx, close)
                    break

            # Forward return target (DTE days forward)
            fi = min(idx + dte, len(close) - 1)
            if fi <= idx:
                continue
            fwd_ret = float(close[tk].iloc[fi] / close[tk].iloc[idx] - 1)

            # Store VIX level for regime-adaptive parameter selection
            cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0

            rec = {**legacy, **cross_asset, "date": dt, "ticker": tk,
                   "fwd_ret": fwd_ret, "direction": direction, "vix_level": cv}
            records.append(rec)

    df = pd.DataFrame(records)
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    df[feature_cols] = df[feature_cols].fillna(0.0)

    fprint(f"    {len(df)} records, {len(df['date'].unique())} dates")
    if "direction" in df.columns:
        bull_n = (df["direction"] == "bull").sum()
        bear_n = (df["direction"] == "bear").sum()
        fprint(f"    Bull records: {bull_n}, Bear records: {bear_n}")

    return df


def walk_forward_lgbm_rank(df, feature_cols, variant_name):
    """Walk-forward LGBM ranking: sliding train, predict next period."""
    import lightgbm as lgb

    if len(df) < 100:
        fprint(f"    {variant_name}: Insufficient data ({len(df)} records)")
        return {}, None

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

        Xt = np.nan_to_num(train_df[feature_cols].values.astype(np.float32))
        yt = train_df["rank_label"].values.astype(np.float32)
        Xe = np.nan_to_num(test_df[feature_cols].values.astype(np.float32))

        try:
            m = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=5,
                verbose=-1,
            )
            m.fit(Xt, yt)
            test_df["score"] = m.predict(Xe)

            # Keep direction info and VIX level with the ranking
            direction = test_df["direction"].iloc[0] if "direction" in test_df.columns else "bull"
            vix_level = test_df["vix_level"].iloc[0] if "vix_level" in test_df.columns else 20.0
            rankings[test_date] = {
                "scores": dict(zip(test_df["ticker"], test_df["score"])),
                "direction": direction,
                "vix_level": float(vix_level),
            }

            all_importances += m.feature_importances_
            n_models += 1
        except Exception:
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
# REGIME-ADAPTIVE TRADE SIMULATION
# ══════════════════════════════════════════════════════════════

def simulate_trades_adaptive(name, rankings, close, high, low, regime_series, atr_dict,
                             variant_rules, skip_vix_25_30=True):
    """
    Simulate trades with regime-adaptive parameters.

    On each rebalance date, check VIX level and select parameters (top_k,
    rebal_days, moneyness_pct) from the appropriate tier in variant_rules.

    HONEST RULES (identical to production v4):
      - Hold to expiry (DTE=21)
      - At expiry: intrinsic value only
      - 15% haircut on entry only
      - No exit haircut (automatic exercise)
    """
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    equity = CAP
    trades = []
    regime_usage = {"vix_lt_15": 0, "vix_15_25": 0, "vix_gt_25": 0}

    for dt in sorted(rankings.keys()):
        if dt not in spy.index:
            continue

        ranking_data = rankings[dt]
        cv = ranking_data.get("vix_level", 20.0)

        # VIX 25-30 sit-in-cash filter (finding #49)
        if skip_vix_25_30 and 25.0 <= cv <= 30.0:
            continue

        # ── REGIME-ADAPTIVE: select parameters based on current VIX ──
        params = resolve_params(variant_rules, cv)
        top_k = params["top_k"]
        moneyness_pct = params["moneyness_pct"]
        # rebal_days from rules affects which dates we act on,
        # but since rankings are pre-built on a fixed 5d grid,
        # we skip dates that don't align with the variant's desired frequency.
        # For adaptive rebal: we subsample the ranking dates.
        desired_rebal = params["rebal_days"]

        # Track regime usage
        if cv < 15.0:
            regime_usage["vix_lt_15"] += 1
        elif cv <= 25.0:
            regime_usage["vix_15_25"] += 1
        else:
            regime_usage["vix_gt_25"] += 1

        scores = ranking_data["scores"]
        direction = ranking_data["direction"]

        if not scores:
            continue

        # Pick sectors
        if direction == "bull":
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        else:
            ranked = sorted(scores.items(), key=lambda x: x[1])

        picks = [t for t, _ in ranked[:top_k]]

        # Position sizing: fixed, max $200 per trade or 1/3 of equity
        max_pos = min(MAX_POS, equity / 3)
        if max_pos < 30:
            continue

        n_entered = 0
        for tk in picks:
            if tk not in close.columns or tk not in atr_dict or n_entered >= top_k:
                continue

            S = float(close[tk].loc[dt])
            di = close.index.get_loc(dt)
            ei = min(di + DTE, len(close) - 1)
            if ei <= di:
                continue

            # ATR for pricing
            if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]):
                av = float(atr_dict[tk].loc[dt])
            else:
                av = S * 0.015

            # Price the spread with regime-adaptive moneyness
            K1 = round(S * (1 + moneyness_pct / 100), 2)
            K2 = round(K1 * (1 + SPREAD_PCT / 100), 2)
            if K2 <= K1:
                K2 = K1 + 1.0

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
                continue

            total_cost = entry_cost_ps * 100 + COMMISSION_RT_SPREAD

            if total_cost <= 0 or total_cost > max_pos or total_cost > equity * 0.40:
                continue

            # HOLD TO EXPIRY: compute intrinsic value at expiry
            Se = float(close[tk].iloc[ei])

            if direction == "bull":
                intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
            else:
                intrinsic = max(K2 - Se, 0.0) - max(K1 - Se, 0.0)

            exit_value_ps = intrinsic

            # PnL: exit value - entry cost - commission (no exit haircut at expiry)
            pnl = (exit_value_ps - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD
            equity += pnl
            n_entered += 1

            # Regime classification for trade record
            sv = float(spy.loc[dt]) if dt in spy.index else 0
            se = float(spy.iloc[ei]) if ei < len(spy) else sv
            spy_regime = "bull" if se >= sv else "bear"

            trades.append({
                "pnl": round(pnl, 2),
                "entry_date": str(dt.date()),
                "exit_date": str(close.index[ei].date()),
                "ticker": tk,
                "regime": spy_regime,
                "direction": direction,
                "vix": round(cv, 1),
                "vix_tier": "low" if cv < 15 else ("mid" if cv <= 25 else "high"),
                "top_k_used": top_k,
                "moneyness_used": moneyness_pct,
                "win": pnl > 0,
            })

    return trades, equity, regime_usage


# ══════════════════════════════════════════════════════════════
# REBALANCE-FREQUENCY ADAPTIVE SIMULATION
# ══════════════════════════════════════════════════════════════

def simulate_with_adaptive_rebal(name, rankings, close, high, low, regime_series, atr_dict,
                                 variant_rules, skip_vix_25_30=True):
    """
    For variants that adapt rebalance frequency, we need to subsample
    the ranking dates to match the desired frequency per VIX regime.

    Rankings are built on a 3-day grid (fastest frequency needed).
    Slower frequencies skip intervening dates.
    """
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    sorted_dates = sorted(rankings.keys())
    if not sorted_dates:
        return [], CAP, {}

    # Track last trade date to enforce minimum spacing
    last_trade_date = None
    filtered_dates = []

    for dt in sorted_dates:
        cv = rankings[dt].get("vix_level", 20.0)
        params = resolve_params(variant_rules, cv)
        desired_rebal = params["rebal_days"]

        if last_trade_date is not None:
            days_since = (dt - last_trade_date).days
            if days_since < desired_rebal:
                continue

        filtered_dates.append(dt)
        last_trade_date = dt

    # Build filtered rankings
    filtered_rankings = {dt: rankings[dt] for dt in filtered_dates}

    # Now simulate with the filtered dates using the same adaptive logic
    return simulate_trades_adaptive(
        name, filtered_rankings, close, high, low, regime_series, atr_dict,
        variant_rules, skip_vix_25_30
    )


# ══════════════════════════════════════════════════════════════
# RANDOM BASELINE
# ══════════════════════════════════════════════════════════════

def random_baseline_test(rankings, close, high, low, regime_series, atr_dict,
                         variant_rules, needs_rebal_adapt, n_trials=5):
    """Test if random sector selection also produces similar returns."""
    fprint(f"\n  Random baseline test ({n_trials} trials)...")
    random_sharpes = []

    for trial in range(n_trials):
        np.random.seed(42 + trial)
        rand_rankings = {}
        for dt, data in rankings.items():
            rand_scores = {tk: np.random.random() for tk in data["scores"].keys()}
            rand_rankings[dt] = {
                "scores": rand_scores,
                "direction": data["direction"],
                "vix_level": data.get("vix_level", 20.0),
            }

        if needs_rebal_adapt:
            trades, final_eq, _ = simulate_with_adaptive_rebal(
                f"Random_{trial}", rand_rankings, close, high, low,
                regime_series, atr_dict, variant_rules,
            )
        else:
            trades, final_eq, _ = simulate_trades_adaptive(
                f"Random_{trial}", rand_rankings, close, high, low,
                regime_series, atr_dict, variant_rules,
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
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t0 = datetime.now()
    fprint("=" * 80)
    fprint(f"REGIME-ADAPTIVE PARAMETER EXPERIMENT v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 80)
    fprint(f"Capital: ${CAP:.0f} | Spread: {SPREAD_PCT:.0f}% | Max pos: ${MAX_POS:.0f}")
    fprint(f"Haircut: {DEFAULT_HAIRCUT:.0%} entry only | Comm: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"HOLD TO EXPIRY (DTE={DTE}) | Intrinsic value only at expiry | No exit haircut")
    fprint(f"Regime bull threshold: >{REGIME_BULL_THRESHOLD}, bear: <{REGIME_BEAR_THRESHOLD}")
    fprint(f"VIX 25-30: skip (all variants)")
    fprint()

    fprint("QUESTION: Does adapting K, rebalance frequency, or OTM level to VIX regime improve Sharpe?")
    fprint()

    fprint("VARIANT GRID:")
    fprint(f"{'Name':<25} Description")
    fprint("-" * 80)
    for v in VARIANTS:
        fprint(f"  {v['name']:<25} {v['desc']}")
    fprint()

    # 1. Download data (shared across all variants)
    close, high, low = download_data()

    # 2. Load regime predictions (shared)
    regime_series = load_regime_predictions()

    # 3. Pre-compute ATR (shared)
    atr_dict = compute_atr_series(high, low, close)

    spy_close = close["SPY"]
    vix_series = close["VIX"] if "VIX" in close.columns else None

    # VIX regime distribution in data
    if vix_series is not None:
        vix_clean = vix_series.dropna()
        n_low = (vix_clean < 15).sum()
        n_mid = ((vix_clean >= 15) & (vix_clean <= 25)).sum()
        n_high = (vix_clean > 25).sum()
        fprint(f"\nVIX regime distribution in data:")
        fprint(f"  VIX < 15:  {n_low} days ({n_low/len(vix_clean)*100:.1f}%)")
        fprint(f"  VIX 15-25: {n_mid} days ({n_mid/len(vix_clean)*100:.1f}%)")
        fprint(f"  VIX > 25:  {n_high} days ({n_high/len(vix_clean)*100:.1f}%)")

    # ══════════════════════════════════════════════════════════════
    # BUILD FEATURES — use 3-day rebalance grid (finest granularity needed)
    # Coarser rebalance frequencies subsample from this grid.
    # ══════════════════════════════════════════════════════════════
    feature_cols = V4_FEATURES

    fprint(f"\n{'=' * 80}")
    fprint(f"BUILDING FEATURES: 3d rebalance grid (finest granularity), DTE={DTE}")
    fprint(f"{'=' * 80}")

    rebal_freq = "3B"
    rebal_dates = pd.DatetimeIndex(
        close.index.to_series().resample(rebal_freq).last().dropna().values
    )
    fprint(f"Rebalance dates (3d grid): {len(rebal_dates)} "
           f"({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

    records = build_feature_records(
        close, high, low, rebal_dates, feature_cols,
        regime_series, dte=DTE, regime_mode="bull_bear",
    )

    rankings, imp = walk_forward_lgbm_rank(records, feature_cols, "regime_adaptive_base")

    if not rankings:
        fprint("ERROR: No rankings produced. Cannot proceed.")
        return

    fprint(f"Rankings produced: {len(rankings)} dates")

    # ══════════════════════════════════════════════════════════════
    # SIMULATE ALL VARIANTS
    # ══════════════════════════════════════════════════════════════
    fprint(f"\n{'=' * 80}")
    fprint("SIMULATING ALL 6 REGIME-ADAPTIVE VARIANTS")
    fprint(f"{'=' * 80}")

    all_results = {}

    for v in VARIANTS:
        vname = v["name"]
        rules = v["rules"]
        desc = v["desc"]

        fprint(f"\n{'─' * 70}")
        fprint(f"VARIANT: {vname}")
        fprint(f"  {desc}")
        fprint(f"  Rules: {json.dumps(rules, indent=4)}")

        # Determine if this variant adapts rebalance frequency
        rebal_values = set()
        for tier_params in rules.values():
            rebal_values.add(tier_params["rebal_days"])
        needs_rebal_adapt = len(rebal_values) > 1

        if needs_rebal_adapt:
            fprint(f"  Rebalance-adaptive: YES (frequencies: {sorted(rebal_values)})")
            trades, final_eq, regime_usage = simulate_with_adaptive_rebal(
                vname, rankings, close, high, low, regime_series, atr_dict, rules
            )
        else:
            rebal_days = list(rebal_values)[0]
            # Subsample rankings to match fixed rebalance frequency
            sorted_dates = sorted(rankings.keys())
            if rebal_days > 3:
                filtered = []
                last = None
                for dt in sorted_dates:
                    if last is None or (dt - last).days >= rebal_days:
                        filtered.append(dt)
                        last = dt
                filtered_rankings = {dt: rankings[dt] for dt in filtered}
                fprint(f"  Fixed rebal {rebal_days}d: {len(filtered_rankings)}/{len(rankings)} dates kept")
            else:
                filtered_rankings = rankings

            trades, final_eq, regime_usage = simulate_trades_adaptive(
                vname, filtered_rankings, close, high, low, regime_series, atr_dict, rules
            )

        fprint(f"  Regime usage: {regime_usage}")

        if not trades or len(trades) < 10:
            fprint(f"  Only {len(trades) if trades else 0} trades, skipping validation")
            all_results[vname] = {
                "description": desc,
                "n_trades": len(trades) if trades else 0,
                "error": "insufficient_trades",
            }
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
        bull_pnl = sum(t["pnl"] for t in bull_trades)
        bear_pnl = sum(t["pnl"] for t in bear_trades)
        bull_wr = sum(1 for t in bull_trades if t["win"]) / max(len(bull_trades), 1) * 100
        bear_wr = sum(1 for t in bear_trades if t["win"]) / max(len(bear_trades), 1) * 100
        fprint(f"  Direction breakdown:")
        fprint(f"    Bull: {len(bull_trades)} trades, WR {bull_wr:.1f}%, PnL ${bull_pnl:.0f}")
        fprint(f"    Bear: {len(bear_trades)} trades, WR {bear_wr:.1f}%, PnL ${bear_pnl:.0f}")

        # VIX tier breakdown
        vix_tiers = {"low": [], "mid": [], "high": []}
        for t in trades:
            tier = t.get("vix_tier", "mid")
            vix_tiers[tier].append(t["pnl"])
        fprint(f"  VIX tier breakdown:")
        for tier_name, tier_pnls in vix_tiers.items():
            if tier_pnls:
                tier_wr = sum(1 for p in tier_pnls if p > 0) / len(tier_pnls) * 100
                fprint(f"    {tier_name}: {len(tier_pnls)} trades, WR {tier_wr:.1f}%, "
                       f"PnL ${sum(tier_pnls):.0f}, Avg ${np.mean(tier_pnls):.2f}")

        # CAGR calculation
        first_date = pd.Timestamp(trades[0]["entry_date"])
        last_date = pd.Timestamp(trades[-1]["exit_date"])
        years = max((last_date - first_date).days / 365.25, 0.5)
        cagr = (final_eq / CAP) ** (1 / years) - 1

        # Random baseline
        random_sharpes = random_baseline_test(
            rankings, close, high, low, regime_series, atr_dict,
            rules, needs_rebal_adapt,
        )
        mean_random = np.mean(random_sharpes) if random_sharpes else 0
        fprint(f"  ML Sharpe: {result.sharpe:.2f} vs Random mean: {mean_random:.2f}")
        if result.sharpe > 0 and mean_random > 0:
            fprint(f"  ML alpha ratio: {result.sharpe / mean_random:.2f}x")

        all_results[vname] = {
            "description": desc,
            "rules": rules,
            **result.to_dict(),
            "cagr": round(cagr, 4),
            "years": round(years, 2),
            "bull_trades": len(bull_trades),
            "bear_trades": len(bear_trades),
            "bull_pnl": round(bull_pnl, 2),
            "bear_pnl": round(bear_pnl, 2),
            "bull_wr": round(bull_wr, 3),
            "bear_wr": round(bear_wr, 3),
            "regime_usage": regime_usage,
            "vix_tier_counts": {k: len(v) for k, v in vix_tiers.items()},
            "random_sharpes": [round(s, 3) for s in random_sharpes],
            "random_mean_sharpe": round(mean_random, 3),
        }

    # ══════════════════════════════════════════════════════════════
    # SUMMARY COMPARISON
    # ══════════════════════════════════════════════════════════════
    fprint(f"\n{'=' * 80}")
    fprint("SUMMARY COMPARISON — ALL 6 REGIME-ADAPTIVE VARIANTS")
    fprint(f"{'=' * 80}")
    fprint(f"{'Variant':<25} {'Trd':>5} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>7} "
           f"{'MaxDD':>7} {'WR':>6} {'PF':>6} {'Gates':>6} {'Final$':>8} {'RandSh':>7}")
    fprint("-" * 105)

    baseline_sharpe = None
    for v in VARIANTS:
        vname = v["name"]
        r = all_results.get(vname)
        if not r or "error" in r:
            fprint(f"  {vname:<25} — INSUFFICIENT DATA —")
            continue
        if vname == "A_fixed_v6":
            baseline_sharpe = r["sharpe"]
        fprint(f"  {vname:<25} {r['n_trades']:>5} {r['sharpe']:>7.2f} {r['sortino']:>8.2f} "
               f"{r['cagr']*100:>6.1f}% {r['max_dd']*100:>6.1f}% "
               f"{r['win_rate']*100:>5.1f}% {r['profit_factor']:>5.2f} "
               f"{r['gates_passed']}/{r['gates_total']} "
               f"${r['final_equity']:>7,.0f} {r['random_mean_sharpe']:>7.2f}")

    # ══════════════════════════════════════════════════════════════
    # REGIME-ADAPTIVE ANALYSIS
    # ══════════════════════════════════════════════════════════════
    fprint(f"\n{'=' * 80}")
    fprint("REGIME-ADAPTIVE ANALYSIS")
    fprint(f"{'=' * 80}")

    if baseline_sharpe is not None:
        fprint(f"\n  Fixed v6 baseline Sharpe: {baseline_sharpe:.3f}")
        fprint()
        fprint(f"  {'Variant':<25} {'Sharpe':>8} {'Delta':>8} {'Improvement':>12}")
        fprint(f"  {'-'*60}")
        for v in VARIANTS:
            vn = v["name"]
            r = all_results.get(vn, {})
            if "error" in r:
                continue
            sh = r.get("sharpe", 0)
            delta = sh - baseline_sharpe
            pct = delta / max(abs(baseline_sharpe), 0.001) * 100
            marker = " <<<" if vn != "A_fixed_v6" and sh > baseline_sharpe else ""
            fprint(f"  {vn:<25} {sh:>8.3f} {delta:>+8.3f} {pct:>+10.1f}%{marker}")

    # Key question: does adaptation help?
    fprint(f"\n  KEY FINDINGS:")
    adaptive_variants = [v["name"] for v in VARIANTS if v["name"] != "A_fixed_v6"]
    adaptive_sharpes = [all_results.get(vn, {}).get("sharpe", 0)
                        for vn in adaptive_variants if "error" not in all_results.get(vn, {})]
    if adaptive_sharpes and baseline_sharpe is not None:
        best_adaptive = max(adaptive_sharpes)
        best_name = adaptive_variants[adaptive_sharpes.index(best_adaptive)]
        mean_adaptive = np.mean(adaptive_sharpes)
        fprint(f"    Best adaptive Sharpe: {best_adaptive:.3f} ({best_name})")
        fprint(f"    Mean adaptive Sharpe: {mean_adaptive:.3f}")
        fprint(f"    Fixed baseline Sharpe: {baseline_sharpe:.3f}")
        if best_adaptive > baseline_sharpe:
            fprint(f"    --> Adaptation IMPROVES Sharpe by {best_adaptive - baseline_sharpe:+.3f}")
        else:
            fprint(f"    --> Adaptation does NOT improve over fixed parameters")

    # Determine best overall variant
    valid_results = {k: v for k, v in all_results.items() if "error" not in v}
    if valid_results:
        best_name = max(valid_results.keys(), key=lambda k: valid_results[k].get("sharpe", -999))
        best = valid_results[best_name]

        fprint(f"\n  BEST OVERALL: {best_name}")
        fprint(f"    Sharpe={best.get('sharpe', 0):.3f}, Sortino={best.get('sortino', 0):.3f}, "
               f"CAGR={best.get('cagr', 0)*100:.1f}%, "
               f"Gates={best.get('gates_passed', 0)}/{best.get('gates_total', 0)}")

    # ── RECOMMENDATION ──
    fprint(f"\n{'=' * 80}")
    fprint("RECOMMENDATION")
    fprint(f"{'=' * 80}")

    if valid_results:
        best_gates = best.get("gates_passed", 0)
        best_total = best.get("gates_total", 5)
        if best_name != "A_fixed_v6" and best_gates >= 4:
            fprint(f"  PROMOTE {best_name} to production")
            fprint(f"    Rules: {json.dumps(best.get('rules', {}), indent=4)}")
            if baseline_sharpe:
                fprint(f"    Sharpe: {baseline_sharpe:.3f} -> {best.get('sharpe', 0):.3f} "
                       f"({(best.get('sharpe', 0) - baseline_sharpe)/max(abs(baseline_sharpe), 0.001)*100:+.1f}%)")
        elif best_name == "A_fixed_v6":
            fprint(f"  KEEP fixed v6 parameters. Regime adaptation does not improve risk-adjusted returns.")
        else:
            fprint(f"  CAUTION: Best variant {best_name} only passes {best_gates}/{best_total} gates.")
            fprint(f"  Consider keeping fixed v6 until more evidence accumulated.")

    # Save results
    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    # Save trades for best variant
    if valid_results:
        trades_path = OUTPUT_DIR / f"trades_{best_name}.json"
        # Re-run best to get trades (or save during loop - for simplicity just save results)
        fprint(f"Best variant: {best_name}")

    # MLflow logging
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"regime_adaptive_{t0.strftime('%Y%m%d_%H%M')}"):
                # Log all variant metrics
                for vname, r in all_results.items():
                    if "error" in r:
                        continue
                    mlflow.log_metric(f"{vname}_sharpe", r.get("sharpe", 0))
                    mlflow.log_metric(f"{vname}_sortino", r.get("sortino", 0))
                    mlflow.log_metric(f"{vname}_cagr", r.get("cagr", 0))
                    mlflow.log_metric(f"{vname}_win_rate", r.get("win_rate", 0))
                    mlflow.log_metric(f"{vname}_profit_factor", r.get("profit_factor", 0))
                    mlflow.log_metric(f"{vname}_max_dd", r.get("max_dd", 0))
                    mlflow.log_metric(f"{vname}_n_trades", r.get("n_trades", 0))
                    mlflow.log_metric(f"{vname}_gates_passed", r.get("gates_passed", 0))
                    mlflow.log_metric(f"{vname}_final_equity", r.get("final_equity", 0))
                    mlflow.log_metric(f"{vname}_random_mean_sharpe", r.get("random_mean_sharpe", 0))

                mlflow.log_params({
                    "capital": CAP,
                    "spread_pct": SPREAD_PCT,
                    "max_pos": MAX_POS,
                    "haircut": DEFAULT_HAIRCUT,
                    "dte": DTE,
                    "regime_bull_thresh": REGIME_BULL_THRESHOLD,
                    "regime_bear_thresh": REGIME_BEAR_THRESHOLD,
                    "hold_to_expiry": True,
                    "entry_haircut_only": True,
                    "commission": COMMISSION_RT_SPREAD,
                    "n_features": len(V4_FEATURES),
                    "wf_train_periods": WF_TRAIN_PERIODS,
                    "n_variants": len(VARIANTS),
                    "best_variant": best_name if valid_results else "none",
                    "best_sharpe": best.get("sharpe", 0) if valid_results else 0,
                    "experiment_type": "regime_adaptive_parameters",
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
