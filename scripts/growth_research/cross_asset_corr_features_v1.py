#!/usr/bin/env python3
"""
Cross-Asset Correlation Features for Sector Ranking — v1
=========================================================
Tests whether cross-asset correlation features improve LGBM sector ranking
beyond the 18 legacy quality-momentum features.

Motivated by knowledge base findings:
  #50: TLT-SPY correlation = -0.374 during VIX 25-30 vs normal -0.289
  #58: VIX term structure carries independent info from GRU regime (corr=0.33)
  #60: VIX_ratio (VIX/VIX3M) validated as useful feature
  #59: VIX_slope_change_5d validated

New features (10 cross-asset correlation features):
  1. SPY-TLT rolling 21d correlation (flight-to-safety)
  2. SPY-HYG rolling 21d correlation (credit risk)
  3. SPY-GLD rolling 21d correlation (safe haven demand)
  4. Cross-sector dispersion (rolling 21d stdev of sector returns)
  5. Sector-SPY beta 63d (sector beta to market)
  6. TLT-HYG spread change 5d (credit stress velocity)
  7. GLD momentum 21d (gold fear proxy)
  8. VIX_ratio (VIX/VIX3M — term structure)
  9. VIX_slope_change_5d (VIX momentum)
  10. Sector relative vol (sector 21d vol / SPY 21d vol)

4 variants tested:
  A: Baseline — regime>0.4 + 18 legacy features only
  B: Baseline + 10 cross-asset features (28 total)
  C: Baseline + best 5 cross-asset features (auto-selected by LGBM importance from B)
  D: Baseline + 3 VIX term structure features only (21 total)

All variants: walk-forward LGBM, $645 capital, 3% bull call spreads, DTE=21,
hold to expiry, 15% entry haircut, $2.60 commission.

Full 5-gate adversarial validation.
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
sys.path.insert(0, "/home/jupiter/Lvl3Quant")
from research.tools.options_pricer import (
    price_bull_call_spread,
    estimate_iv,
    compute_atr,
    COMMISSION_RT_SPREAD,
    DEFAULT_HAIRCUT,
)
from research.tools.adversarial_validator import validate_trades

# ── Config ──
BASE = Path("/home/jupiter/Lvl3Quant")
OUTPUT_DIR = BASE / "output" / "growth_research" / "cross_asset_corr_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
DTE = 21
SPREAD_PCT = 3.0
TOP_K = 3
REGIME_THRESHOLD = 0.4

# Regime predictions path (actual location on disk)
REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"

# LGBM walk-forward params (sliding window)
WF_TRAIN_PERIODS = 12  # use last 12 rebal periods for training
WF_REBAL_FREQ = "2W-FRI"

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "cross_asset_corr_v1"

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen(MLFLOW_URI, timeout=3)
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    MLFLOW_OK = True
    fprint(f"MLflow connected: {MLFLOW_URI}")
except Exception:
    fprint("MLflow unavailable — results will be saved to disk only")

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

    # Handle column naming for VIX tickers
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
        high.columns = high.columns.get_level_values(-1)
        low.columns = low.columns.get_level_values(-1)

    close = close.ffill()
    high = high.ffill()
    low = low.ffill()

    # Rename VIX columns for convenience
    rename_map = {"^VIX": "VIX", "^VIX3M": "VIX3M"}
    close = close.rename(columns=rename_map)
    high = high.rename(columns=rename_map)
    low = low.rename(columns=rename_map)

    # Ensure we have SPY and VIX at minimum
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
           f"Days with score>0.4: {(regime_series > REGIME_THRESHOLD).sum()}")
    return regime_series


# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════

# Legacy 18 features (quality-momentum)
LEGACY_FEATURES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_21d", "vol_63d", "sharpe_63d", "maxdd_63d", "pct_52w_high", "mom_accel",
    "pct_pos_months_12m", "sortino_63d", "calmar_1y", "up_capture",
    "trend_r2_63d", "trend_slope_63d",
]

# Cross-asset correlation features (new)
CROSS_ASSET_FEATURES = [
    "spy_tlt_corr_21d",
    "spy_hyg_corr_21d",
    "spy_gld_corr_21d",
    "cross_sector_dispersion_21d",
    "sector_spy_beta_63d",
    "tlt_hyg_spread_chg_5d",
    "gld_momentum_21d",
    "vix_ratio",
    "vix_slope_change_5d",
    "sector_relative_vol_21d",
]

# VIX term structure subset (features 8, 9 + dispersion)
VIX_TERM_FEATURES = [
    "vix_ratio",
    "vix_slope_change_5d",
    "cross_sector_dispersion_21d",
]


def compute_legacy_features(px, spy_slice):
    """Compute the 18 legacy quality-momentum features for a single sector ETF."""
    if len(px) < 260:
        return None
    f = {}
    # Momentum returns
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

    # Quality features
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
    """Compute the 10 new cross-asset correlation features."""
    f = {}

    spy = close_df["SPY"].iloc[:dt_idx + 1].dropna()
    sector_px = close_df[sector_ticker].iloc[:dt_idx + 1].dropna() if sector_ticker in close_df.columns else None

    if spy is None or len(spy) < 63:
        return {k: 0.0 for k in CROSS_ASSET_FEATURES}

    spy_ret = spy.pct_change().dropna()

    # 1. SPY-TLT rolling 21d correlation
    if "TLT" in close_df.columns:
        tlt = close_df["TLT"].iloc[:dt_idx + 1].dropna()
        tlt_ret = tlt.pct_change().dropna()
        common = spy_ret.index.intersection(tlt_ret.index)
        if len(common) > 21:
            corr = spy_ret.loc[common].rolling(21).corr(tlt_ret.loc[common])
            f["spy_tlt_corr_21d"] = float(corr.iloc[-1]) if not pd.isna(corr.iloc[-1]) else -0.3
        else:
            f["spy_tlt_corr_21d"] = -0.3
    else:
        f["spy_tlt_corr_21d"] = -0.3

    # 2. SPY-HYG rolling 21d correlation
    if "HYG" in close_df.columns:
        hyg = close_df["HYG"].iloc[:dt_idx + 1].dropna()
        hyg_ret = hyg.pct_change().dropna()
        common = spy_ret.index.intersection(hyg_ret.index)
        if len(common) > 21:
            corr = spy_ret.loc[common].rolling(21).corr(hyg_ret.loc[common])
            f["spy_hyg_corr_21d"] = float(corr.iloc[-1]) if not pd.isna(corr.iloc[-1]) else 0.6
        else:
            f["spy_hyg_corr_21d"] = 0.6
    else:
        f["spy_hyg_corr_21d"] = 0.6

    # 3. SPY-GLD rolling 21d correlation
    if "GLD" in close_df.columns:
        gld = close_df["GLD"].iloc[:dt_idx + 1].dropna()
        gld_ret = gld.pct_change().dropna()
        common = spy_ret.index.intersection(gld_ret.index)
        if len(common) > 21:
            corr = spy_ret.loc[common].rolling(21).corr(gld_ret.loc[common])
            f["spy_gld_corr_21d"] = float(corr.iloc[-1]) if not pd.isna(corr.iloc[-1]) else 0.0
        else:
            f["spy_gld_corr_21d"] = 0.0
    else:
        f["spy_gld_corr_21d"] = 0.0

    # 4. Cross-sector dispersion (rolling 21d stdev of sector returns)
    sector_cols = [c for c in SECTORS if c in close_df.columns]
    if len(sector_cols) > 3:
        sector_rets = close_df[sector_cols].iloc[:dt_idx + 1].pct_change()
        # Daily cross-sectional stdev, then rolling mean
        daily_disp = sector_rets.std(axis=1)
        if len(daily_disp) > 21:
            f["cross_sector_dispersion_21d"] = float(daily_disp.rolling(21).mean().iloc[-1])
        else:
            f["cross_sector_dispersion_21d"] = 0.01
    else:
        f["cross_sector_dispersion_21d"] = 0.01

    # 5. Sector-SPY beta 63d
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

    # 6. TLT-HYG spread change 5d (credit stress velocity)
    if "TLT" in close_df.columns and "HYG" in close_df.columns:
        tlt = close_df["TLT"].iloc[:dt_idx + 1].dropna()
        hyg = close_df["HYG"].iloc[:dt_idx + 1].dropna()
        common = tlt.index.intersection(hyg.index)
        if len(common) > 10:
            # Normalized spread: TLT/HYG ratio
            spread = (tlt.loc[common] / hyg.loc[common])
            spread_chg = spread.pct_change(5)
            f["tlt_hyg_spread_chg_5d"] = float(spread_chg.iloc[-1]) if not pd.isna(spread_chg.iloc[-1]) else 0.0
        else:
            f["tlt_hyg_spread_chg_5d"] = 0.0
    else:
        f["tlt_hyg_spread_chg_5d"] = 0.0

    # 7. GLD momentum 21d
    if "GLD" in close_df.columns:
        gld = close_df["GLD"].iloc[:dt_idx + 1].dropna()
        if len(gld) > 21:
            f["gld_momentum_21d"] = float(gld.iloc[-1] / gld.iloc[-21] - 1)
        else:
            f["gld_momentum_21d"] = 0.0
    else:
        f["gld_momentum_21d"] = 0.0

    # 8. VIX ratio (VIX/VIX3M)
    if "VIX" in close_df.columns and "VIX3M" in close_df.columns:
        vix_val = close_df["VIX"].iloc[:dt_idx + 1].dropna()
        vix3m_val = close_df["VIX3M"].iloc[:dt_idx + 1].dropna()
        common = vix_val.index.intersection(vix3m_val.index)
        if len(common) > 0:
            ratio = vix_val.loc[common].iloc[-1] / (vix3m_val.loc[common].iloc[-1] + 1e-10)
            f["vix_ratio"] = float(ratio)
        else:
            f["vix_ratio"] = 0.85
    else:
        f["vix_ratio"] = 0.85

    # 9. VIX slope change 5d
    if "VIX" in close_df.columns:
        vix_val = close_df["VIX"].iloc[:dt_idx + 1].dropna()
        if len(vix_val) > 10:
            slope_now = (vix_val.iloc[-1] - vix_val.iloc[-5]) / (vix_val.iloc[-5] + 1e-10)
            slope_prev = (vix_val.iloc[-5] - vix_val.iloc[-10]) / (vix_val.iloc[-10] + 1e-10)
            f["vix_slope_change_5d"] = float(slope_now - slope_prev)
        else:
            f["vix_slope_change_5d"] = 0.0
    else:
        f["vix_slope_change_5d"] = 0.0

    # 10. Sector relative vol (sector 21d vol / SPY 21d vol)
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

    return f


# ══════════════════════════════════════════════════════════════
# WALK-FORWARD LGBM RANKING
# ══════════════════════════════════════════════════════════════

def build_feature_records(close, high, low, rebal_dates, feature_cols, regime_series):
    """Build feature + target records for all sectors on all rebal dates."""
    import lightgbm as lgb

    fprint(f"  Building feature records for {len(rebal_dates)} dates, "
           f"{len(feature_cols)} features...")

    records = []
    sector_cols = [c for c in SECTORS if c in close.columns]
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    for dt in rebal_dates:
        idx = close.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
            continue

        # Regime filter: skip dates where regime score <= threshold
        if regime_series is not None:
            if dt in regime_series.index:
                score = regime_series.loc[dt]
            else:
                # Find nearest date
                nearest = regime_series.index[regime_series.index.get_indexer([dt], method="ffill")]
                if len(nearest) > 0:
                    score = regime_series.loc[nearest[0]]
                else:
                    score = 0.0
            if score <= REGIME_THRESHOLD:
                continue
        elif vix is not None:
            # VIX-based regime proxy: skip when VIX < 15 (too calm, no premium)
            cv = float(vix.iloc[idx]) if idx < len(vix) else 20
            if cv < 15:
                continue

        for tk in sector_cols:
            px = close[tk].iloc[: idx + 1].dropna()
            spy_s = spy.iloc[: idx + 1]

            # Legacy features
            legacy = compute_legacy_features(px, spy_s)
            if not legacy:
                continue

            # Cross-asset features (only if requested)
            cross_asset = {}
            for col in feature_cols:
                if col in CROSS_ASSET_FEATURES:
                    if not cross_asset:
                        cross_asset = compute_cross_asset_features(tk, idx, close)
                    break

            # Forward return target (DTE days forward)
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

    fprint(f"    {len(df)} total records, {len(df['date'].unique())} unique dates")
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
        train_dates = dates[max(0, i - WF_TRAIN_PERIODS): i]
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
            rankings[test_date] = dict(zip(test_df["ticker"], test_df["score"]))

            all_importances += m.feature_importances_
            n_models += 1
        except Exception as e:
            continue

    # Normalize importance
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
# OPTIONS SIMULATION
# ══════════════════════════════════════════════════════════════

def simulate_trades(name, rankings, close, high, low, regime_series):
    """Simulate bull call spread trades from rankings."""
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None
    sector_cols = [c for c in SECTORS if c in close.columns]

    # Pre-compute ATR series for all sectors
    atr_dict = {}
    for tk in sector_cols:
        if tk in high.columns and tk in low.columns:
            h = high[tk].dropna()
            l = low[tk].dropna()
            c = close[tk].dropna()
            common = h.index.intersection(l.index).intersection(c.index)
            if len(common) > 14:
                tr1 = h.loc[common] - l.loc[common]
                tr2 = (h.loc[common] - c.loc[common].shift(1)).abs()
                tr3 = (l.loc[common] - c.loc[common].shift(1)).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr_dict[tk] = tr.ewm(alpha=1/14, min_periods=14).mean()

    equity = CAP
    trades = []

    for dt in sorted(rankings.keys()):
        if dt not in spy.index:
            continue
        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0

        scores = rankings[dt]
        if not scores:
            continue

        # Pick top K sectors
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        picks = [t for t, _ in ranked[:TOP_K]]

        # Position sizing: fixed, max $200 per trade or 1/3 of equity
        max_pos = min(200, equity / 3)
        if max_pos < 30:
            continue

        n_entered = 0
        for tk in picks:
            if tk not in close.columns or tk not in atr_dict or n_entered >= TOP_K:
                continue

            S = float(close[tk].loc[dt])
            di = close.index.get_loc(dt)
            ei = min(di + DTE, len(close) - 1)
            if ei <= di:
                continue

            # ATR for pricing
            av = float(atr_dict[tk].iloc[di]) if di < len(atr_dict[tk]) and not pd.isna(atr_dict[tk].iloc[di]) else S * 0.015

            # Bull call spread: K1=ATM, K2=K1*(1+spread_pct%)
            K1 = round(S, 2)
            K2 = round(S * (1 + SPREAD_PCT / 100), 2)
            if K2 <= K1:
                K2 = K1 + 1.0

            # Price using standardized pricer
            try:
                entry_cost_per_share, max_profit_per_share = price_bull_call_spread(
                    S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=cv
                )
            except Exception:
                continue

            total_cost = entry_cost_per_share * 100 + COMMISSION_RT_SPREAD
            total_max_profit = max_profit_per_share * 100 - COMMISSION_RT_SPREAD

            if total_cost <= 0 or total_cost > max_pos or total_cost > equity * 0.40:
                continue

            # Hold to expiry: compute intrinsic value at expiry
            Se = float(close[tk].iloc[ei])
            intrinsic_long = max(Se - K1, 0.0)
            intrinsic_short = max(Se - K2, 0.0)
            exit_value_per_share = intrinsic_long - intrinsic_short

            pnl = (exit_value_per_share - entry_cost_per_share) * 100 - COMMISSION_RT_SPREAD
            equity += pnl
            n_entered += 1

            # Regime classification for trade record
            sv = float(spy.loc[dt]) if dt in spy.index else 0
            se = float(spy.iloc[ei]) if ei < len(spy) else sv
            regime = "bull" if se >= sv else "bear"

            trades.append({
                "pnl": round(pnl, 2),
                "entry_date": str(dt.date()),
                "exit_date": str(close.index[ei].date()),
                "ticker": tk,
                "regime": regime,
                "vix": round(cv, 1),
                "win": pnl > 0,
            })

    return trades, equity


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t0 = datetime.now()
    fprint("=" * 80)
    fprint(f"Cross-Asset Correlation Features v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 80)
    fprint(f"Capital: ${CAP:.0f} | DTE: {DTE} | Spread: {SPREAD_PCT:.0f}% | "
           f"Haircut: {DEFAULT_HAIRCUT:.0%} | Comm: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"Regime threshold: >{REGIME_THRESHOLD}")
    fprint()

    # 1. Download data
    close, high, low = download_data()

    # 2. Load regime predictions
    regime_series = load_regime_predictions()

    # 3. Build rebalance dates
    rebal_dates = pd.DatetimeIndex(
        close.index.to_series().resample(WF_REBAL_FREQ).last().dropna().values
    )
    fprint(f"Rebalance dates: {len(rebal_dates)} ({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

    # 4. Build feature records for each variant
    fprint("\n" + "=" * 80)
    fprint("VARIANT A: Baseline (18 legacy features, regime>0.4)")
    fprint("=" * 80)
    records_a = build_feature_records(close, high, low, rebal_dates, LEGACY_FEATURES, regime_series)
    rankings_a, imp_a = walk_forward_lgbm_rank(records_a, LEGACY_FEATURES, "Variant_A")

    fprint("\n" + "=" * 80)
    fprint("VARIANT B: Baseline + 10 cross-asset features (28 total)")
    fprint("=" * 80)
    all_28_features = LEGACY_FEATURES + CROSS_ASSET_FEATURES
    records_b = build_feature_records(close, high, low, rebal_dates, all_28_features, regime_series)
    rankings_b, imp_b = walk_forward_lgbm_rank(records_b, all_28_features, "Variant_B")

    # Auto-select best 5 cross-asset features from B's importance
    if imp_b is not None:
        cross_imp = imp_b[imp_b["feature"].isin(CROSS_ASSET_FEATURES)]
        top5_cross = cross_imp.nlargest(5, "importance")["feature"].tolist()
        fprint(f"\n  Top 5 cross-asset features by LGBM importance:")
        for i, row in cross_imp.nlargest(5, "importance").iterrows():
            fprint(f"    {row['feature']}: {row['importance']:.1f}")
    else:
        top5_cross = CROSS_ASSET_FEATURES[:5]

    fprint("\n" + "=" * 80)
    fprint(f"VARIANT C: Baseline + best 5 cross-asset ({len(LEGACY_FEATURES) + 5} total)")
    fprint("=" * 80)
    c_features = LEGACY_FEATURES + top5_cross
    records_c = build_feature_records(close, high, low, rebal_dates, c_features, regime_series)
    rankings_c, imp_c = walk_forward_lgbm_rank(records_c, c_features, "Variant_C")

    fprint("\n" + "=" * 80)
    fprint(f"VARIANT D: Baseline + 3 VIX term structure features ({len(LEGACY_FEATURES) + 3} total)")
    fprint("=" * 80)
    d_features = LEGACY_FEATURES + VIX_TERM_FEATURES
    records_d = build_feature_records(close, high, low, rebal_dates, d_features, regime_series)
    rankings_d, imp_d = walk_forward_lgbm_rank(records_d, d_features, "Variant_D")

    # 5. Simulate trades for all variants
    fprint("\n" + "=" * 80)
    fprint("SIMULATING TRADES")
    fprint("=" * 80)

    spy_close = close["SPY"]
    all_results = {}
    variant_data = [
        ("A_Baseline_18feat", rankings_a, LEGACY_FEATURES, imp_a),
        ("B_Full_28feat", rankings_b, all_28_features, imp_b),
        ("C_Best5_23feat", rankings_c, c_features, imp_c),
        ("D_VIXterm_21feat", rankings_d, d_features, imp_d),
    ]

    for vname, rankings, feat_cols, imp_df in variant_data:
        fprint(f"\n--- {vname} ---")
        if not rankings:
            fprint(f"  No rankings available, skipping")
            continue

        trades, final_eq = simulate_trades(vname, rankings, close, high, low, regime_series)

        if not trades:
            fprint(f"  No trades generated, skipping")
            continue

        fprint(f"  {len(trades)} trades, final equity: ${final_eq:.2f}")

        # 5-gate adversarial validation
        result = validate_trades(
            trades=trades,
            initial_capital=CAP,
            spy_prices=spy_close,
            strategy_name=vname,
            n_perms=2000,
        )
        result.print_summary()

        # Feature importance summary
        if imp_df is not None:
            fprint(f"\n  Feature importance (top 10):")
            for _, row in imp_df.head(10).iterrows():
                marker = " *CROSS*" if row["feature"] in CROSS_ASSET_FEATURES else ""
                fprint(f"    {row['feature']:<30s} {row['importance']:>8.1f}{marker}")

        # Store results
        r = result.to_dict()
        r["features_used"] = feat_cols
        r["n_features"] = len(feat_cols)
        if imp_df is not None:
            r["feature_importance"] = imp_df.to_dict("records")
        all_results[vname] = r

    # 6. Comparative summary
    fprint("\n" + "=" * 80)
    fprint("COMPARATIVE SUMMARY")
    fprint("=" * 80)
    fprint(f"{'Variant':<25s} {'N':>5s} {'Sharpe':>7s} {'Sortino':>8s} {'WR%':>6s} "
           f"{'PF':>6s} {'CAGR%':>7s} {'MaxDD%':>7s} {'Gates':>6s} {'Final$':>8s}")
    fprint("-" * 90)

    for vname in ["A_Baseline_18feat", "B_Full_28feat", "C_Best5_23feat", "D_VIXterm_21feat"]:
        if vname not in all_results:
            fprint(f"  {vname}: NO DATA")
            continue
        r = all_results[vname]
        fprint(f"  {vname:<23s} {r['n_trades']:>5d} {r['sharpe']:>7.2f} {r['sortino']:>8.2f} "
               f"{r['win_rate']*100:>6.1f} {r['profit_factor']:>6.2f} "
               f"{r['cagr']*100:>7.1f} {r['max_dd']*100:>7.1f} "
               f"{r['gates_passed']}/{r['gates_total']}   ${r['final_equity']:>8.0f}")

    # Key question answer
    fprint("\n" + "=" * 80)
    fprint("KEY FINDING: Do cross-asset correlation features add value?")
    fprint("=" * 80)

    if "A_Baseline_18feat" in all_results and "B_Full_28feat" in all_results:
        a_sh = all_results["A_Baseline_18feat"]["sharpe"]
        b_sh = all_results["B_Full_28feat"]["sharpe"]
        diff = b_sh - a_sh
        if diff > 0.1:
            fprint(f"  YES: Full 28-feature model improves Sharpe by {diff:.2f} ({a_sh:.2f} -> {b_sh:.2f})")
        elif diff > 0:
            fprint(f"  MARGINAL: 28-feature model improves Sharpe by only {diff:.2f} ({a_sh:.2f} -> {b_sh:.2f})")
        else:
            fprint(f"  NO: 28-feature model HURTS Sharpe by {abs(diff):.2f} ({a_sh:.2f} -> {b_sh:.2f})")
            fprint(f"  Cross-asset features add noise, similar to flow features finding.")

    if "C_Best5_23feat" in all_results and "A_Baseline_18feat" in all_results:
        c_sh = all_results["C_Best5_23feat"]["sharpe"]
        a_sh = all_results["A_Baseline_18feat"]["sharpe"]
        fprint(f"  Best-5 selection (C): Sharpe {c_sh:.2f} vs baseline {a_sh:.2f} (delta: {c_sh-a_sh:+.2f})")

    if "D_VIXterm_21feat" in all_results and "A_Baseline_18feat" in all_results:
        d_sh = all_results["D_VIXterm_21feat"]["sharpe"]
        a_sh = all_results["A_Baseline_18feat"]["sharpe"]
        fprint(f"  VIX term structure only (D): Sharpe {d_sh:.2f} vs baseline {a_sh:.2f} (delta: {d_sh-a_sh:+.2f})")

    # 7. Log to MLflow
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"cross_asset_corr_v1_{t0.strftime('%Y%m%d_%H%M')}"):
                for vname, r in all_results.items():
                    prefix = vname.split("_")[0]
                    mlflow.log_metric(f"{prefix}_sharpe", r["sharpe"])
                    mlflow.log_metric(f"{prefix}_sortino", r["sortino"])
                    mlflow.log_metric(f"{prefix}_cagr", r["cagr"])
                    mlflow.log_metric(f"{prefix}_max_dd", r["max_dd"])
                    mlflow.log_metric(f"{prefix}_win_rate", r["win_rate"])
                    mlflow.log_metric(f"{prefix}_profit_factor", r["profit_factor"])
                    mlflow.log_metric(f"{prefix}_n_trades", r["n_trades"])
                    mlflow.log_metric(f"{prefix}_final_equity", r["final_equity"])
                    mlflow.log_metric(f"{prefix}_gates_passed", r["gates_passed"])
                    mlflow.log_metric(f"{prefix}_n_features", r["n_features"])

                mlflow.log_param("capital", CAP)
                mlflow.log_param("dte", DTE)
                mlflow.log_param("spread_pct", SPREAD_PCT)
                mlflow.log_param("haircut", DEFAULT_HAIRCUT)
                mlflow.log_param("commission", COMMISSION_RT_SPREAD)
                mlflow.log_param("regime_threshold", REGIME_THRESHOLD)
                mlflow.log_param("wf_train_periods", WF_TRAIN_PERIODS)
                mlflow.log_param("top_k", TOP_K)

                # Log full results as artifact
                results_path = OUTPUT_DIR / "cross_asset_corr_v1_results.json"
                with open(results_path, "w") as f:
                    # Convert non-serializable types
                    clean_results = {}
                    for k, v in all_results.items():
                        clean = {}
                        for kk, vv in v.items():
                            if isinstance(vv, (np.floating, np.integer)):
                                clean[kk] = float(vv)
                            elif isinstance(vv, np.ndarray):
                                clean[kk] = vv.tolist()
                            else:
                                clean[kk] = vv
                        clean_results[k] = clean
                    json.dump(clean_results, f, indent=2, default=str)
                mlflow.log_artifact(str(results_path))
            fprint(f"\nMLflow run logged to experiment '{EXPERIMENT_NAME}'")
        except Exception as e:
            fprint(f"\nMLflow logging failed: {e}")

    # 8. Save results to disk
    results_path = OUTPUT_DIR / "cross_asset_corr_v1_results.json"
    clean_results = {}
    for k, v in all_results.items():
        clean = {}
        for kk, vv in v.items():
            if isinstance(vv, (np.floating, np.integer)):
                clean[kk] = float(vv)
            elif isinstance(vv, np.ndarray):
                clean[kk] = vv.tolist()
            else:
                clean[kk] = vv
        clean_results[k] = clean
    with open(results_path, "w") as f:
        json.dump(clean_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    fprint("Done.")


if __name__ == "__main__":
    main()
