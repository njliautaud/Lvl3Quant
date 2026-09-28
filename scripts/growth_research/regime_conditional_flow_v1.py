#!/usr/bin/env python3
"""
Regime-Conditional Flow Model v1
=================================
HYPOTHESIS: Flow features (CTA positioning, risk-on/off ratios, sector rotation
velocity, volume climax) improve LGBM sector ranking by ~12% in Sharpe during
moderate VIX (20-25), BUT break during crash VIX (30+) with IC going negative.

FIX: Train SEPARATE models per VIX band:
  - Model A: VIX 20-25 (moderate vol) — all 55 flow + 18 legacy features
  - Model B: VIX 25-30 (elevated vol) — all 55 flow + 18 legacy features
  - Model C: VIX 30+ (crisis) — ONLY 18 legacy momentum features (no flow)

Walk-forward: sliding window (252d train, 21d test).
Compares: single-model legacy, single-model legacy+flow, regime-conditional, random.
Full adversarial validation: permutation, regime split, sub-period, yearly.
Logs to MLflow on Jupiter (http://jupiter:5000).

Author: Claude (Head of Quant)
Date: 2026-07-26
"""

import sys
import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════════

if os.path.exists("/home/jupiter"):
    ROOT = Path("/home/jupiter/Lvl3Quant")
elif os.path.exists(r"C:\Users\claude"):
    ROOT = Path(r"C:\Users\claude\Lvl3Quant")
else:
    ROOT = Path("/home/nick/Lvl3Quant")

OUTPUT = ROOT / "output" / "regime_conditional_flow_v1"
OUTPUT.mkdir(parents=True, exist_ok=True)

def fprint(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

# ═══════════════════════════════════════════════════════════════════════════════
# MLFLOW
# ═══════════════════════════════════════════════════════════════════════════════

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen("http://jupiter:5000/", timeout=3)
    import mlflow
    mlflow.set_tracking_uri("http://jupiter:5000")
    MLFLOW_OK = True
    fprint("MLflow connected: http://jupiter:5000")
except Exception:
    fprint("MLflow unavailable — results saved to disk only")

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSE — 11 sector ETFs + macro tickers
# ═══════════════════════════════════════════════════════════════════════════════

SECTOR_ETFS = [
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC",
]
MACRO_TICKERS = [
    "SPY", "QQQ", "IWM", "EFA", "EEM", "GLD", "TLT", "HYG", "DBC", "UUP", "^VIX",
]
ALL_TICKERS = SECTOR_ETFS + [t for t in MACRO_TICKERS if t not in SECTOR_ETFS]

DEFENSIVE_ETFS = {"XLP", "XLU", "XLRE"}
CYCLICAL_ETFS = {"XLK", "XLY", "XLF", "XLI", "XLB", "XLC"}

START_DATE = "2007-01-01"
END_DATE = "2026-07-25"

# Cost
COST_BPS = 20  # 20 bps round-trip
SLIPPAGE_BPS = 5
TOTAL_COST_BPS = COST_BPS + SLIPPAGE_BPS

# Walk-forward
TRAIN_DAYS = 252
TEST_DAYS = 21
TOP_K = 3  # top-K sectors to hold

# Permutation
N_PERMUTATIONS = 500

# VIX bands
VIX_BAND_A = (20, 25)  # moderate
VIX_BAND_B = (25, 30)  # elevated
VIX_BAND_C = (30, 999) # crisis


# ═══════════════════════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════════

def download_data(tickers, cache_path):
    """Download OHLCV data for all tickers, with caching."""
    cache_path = Path(cache_path)

    # Try loading from per-ticker cache
    ticker_cache = cache_path.parent / "ticker_cache"
    ticker_cache.mkdir(parents=True, exist_ok=True)
    cached_data = {}
    missing = []
    for t in tickers:
        tp = ticker_cache / f"{t.replace('^','_')}.parquet"
        if tp.exists():
            try:
                df = pd.read_parquet(tp)
                if len(df) > 200:
                    cached_data[t] = df
                    continue
            except Exception:
                pass
        missing.append(t)

    if not missing and len(cached_data) >= len(SECTOR_ETFS):
        fprint(f"  Cache hit: {len(cached_data)} tickers loaded")
        return cached_data

    import yfinance as yf
    fprint(f"  Downloading {len(missing)} tickers (cached {len(cached_data)})...")
    if missing:
        raw = yf.download(missing, start=START_DATE, end=END_DATE,
                          progress=False, auto_adjust=True, threads=True)
        if raw.empty:
            if cached_data:
                fprint(f"  Download returned empty but have {len(cached_data)} cached")
                return cached_data
            raise RuntimeError("yfinance returned empty data")

        if isinstance(raw.columns, pd.MultiIndex):
            for t in missing:
                try:
                    sub = raw.xs(t, level=1, axis=1).dropna(subset=["Close"])
                    if len(sub) > 200:
                        cached_data[t] = sub
                        tp = ticker_cache / f"{t.replace('^','_')}.parquet"
                        sub.to_parquet(tp)
                except (KeyError, TypeError):
                    pass
        else:
            # Single ticker
            raw.columns = [c if isinstance(c, str) else c[0] for c in raw.columns]
            sub = raw.dropna(subset=["Close"])
            if len(sub) > 200:
                cached_data[missing[0]] = sub
                tp = ticker_cache / f"{missing[0].replace('^','_')}.parquet"
                sub.to_parquet(tp)

    fprint(f"  Total: {len(cached_data)} tickers, "
           f"{min(len(d) for d in cached_data.values())}-{max(len(d) for d in cached_data.values())} rows")
    return cached_data


def get_vix_series(data):
    """Extract VIX series. Try ^VIX first, then compute from SPY."""
    if "^VIX" in data:
        vix_df = data["^VIX"]
        if isinstance(vix_df, pd.DataFrame) and "Close" in vix_df.columns:
            vix = vix_df["Close"].dropna().copy()
        elif isinstance(vix_df, pd.Series):
            vix = vix_df.dropna().copy()
        else:
            vix = vix_df.iloc[:, 0].dropna().copy()
        vix.name = "VIX"
        fprint(f"  VIX series: {len(vix)} days")
        return vix

    # Fallback: compute VIX proxy from SPY realized vol
    if "SPY" in data:
        spy = data["SPY"]["Close"]
        lr = np.log(spy / spy.shift(1))
        vix_proxy = lr.rolling(21).std() * np.sqrt(252) * 100
        vix_proxy.name = "VIX"
        fprint("  WARNING: Using SPY realized vol as VIX proxy")
        return vix_proxy

    raise RuntimeError("Need ^VIX or SPY for regime classification")


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING — 18 LEGACY MOMENTUM FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_legacy_features(close, volume, spy_close=None):
    """18 legacy momentum/quality features per ETF."""
    lr = np.log(close / close.shift(1))

    feat = pd.DataFrame(index=close.index)

    # Momentum (8)
    feat["ret_5d"] = close.pct_change(5)
    feat["ret_10d"] = close.pct_change(10)
    feat["ret_21d"] = close.pct_change(21)
    feat["ret_63d"] = close.pct_change(63)
    feat["ret_126d"] = close.pct_change(126)
    feat["ret_252d"] = close.pct_change(252)
    feat["mom_12_1"] = close.pct_change(252) - close.pct_change(21)
    feat["high_52w_pct"] = close / close.rolling(252).max()

    # Volatility/Quality (6)
    feat["vol_20d"] = lr.rolling(20).std() * np.sqrt(252)
    feat["vol_60d"] = lr.rolling(60).std() * np.sqrt(252)
    feat["vol_ratio"] = lr.rolling(20).std() / (lr.rolling(60).std() + 1e-10)
    feat["sharpe_63d"] = lr.rolling(63).mean() / (lr.rolling(63).std() + 1e-10)
    feat["maxdd_63d"] = (close / close.rolling(63).max() - 1).rolling(63).min()
    feat["skew_63d"] = lr.rolling(63).skew()

    # Trend (2)
    feat["above_sma50"] = (close > close.rolling(50).mean()).astype(float)
    feat["above_sma200"] = (close > close.rolling(200).mean()).astype(float)

    # Relative (2) — vs SPY
    if spy_close is not None:
        feat["rel_ret_21d"] = close.pct_change(21) - spy_close.pct_change(21)
        feat["rel_ret_63d"] = close.pct_change(63) - spy_close.pct_change(63)
    else:
        feat["rel_ret_21d"] = 0.0
        feat["rel_ret_63d"] = 0.0

    return feat

LEGACY_FEATURE_NAMES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "mom_12_1", "high_52w_pct", "vol_20d", "vol_60d", "vol_ratio",
    "sharpe_63d", "maxdd_63d", "skew_63d", "above_sma50", "above_sma200",
    "rel_ret_21d", "rel_ret_63d",
]


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING — 55 FLOW FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_flow_features(close, high, low, volume, spy_close=None, all_sector_closes=None):
    """
    55 flow features per ETF covering:
    - CTA positioning proxies (trend-following signals across timeframes)
    - Risk-on/off ratios
    - Sector rotation velocity
    - Volume climax indicators
    - Money flow (OBV, MFI, A/D, VWAP)
    - Institutional flow proxies
    """
    lr = np.log(close / close.shift(1))
    feat = pd.DataFrame(index=close.index)

    # ─── MONEY FLOW CORE (7 features) ───
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume

    # OBV
    obv = pd.Series(0.0, index=close.index)
    sign = np.sign(close.diff())
    obv = (sign * volume).cumsum()
    feat["obv_slope_10d"] = obv.rolling(10).apply(
        lambda x: np.polyfit(range(len(x)), x / (np.abs(x).mean() + 1e-10), 1)[0], raw=True
    )
    feat["obv_slope_20d"] = obv.rolling(20).apply(
        lambda x: np.polyfit(range(len(x)), x / (np.abs(x).mean() + 1e-10), 1)[0], raw=True
    )

    # MFI (14-period)
    tp_diff = typical_price.diff()
    pos_flow = raw_money_flow.where(tp_diff > 0, 0)
    neg_flow = raw_money_flow.where(tp_diff <= 0, 0)
    mfi = 100 - 100 / (1 + pos_flow.rolling(14).sum() / (neg_flow.rolling(14).sum() + 1e-10))
    feat["mfi_14"] = mfi

    # A/D line
    clv = ((close - low) - (high - close)) / (high - low + 1e-10)
    ad_line = (clv * volume).cumsum()
    feat["ad_slope_10d"] = ad_line.rolling(10).apply(
        lambda x: np.polyfit(range(len(x)), x / (np.abs(x).mean() + 1e-10), 1)[0], raw=True
    )
    feat["ad_slope_20d"] = ad_line.rolling(20).apply(
        lambda x: np.polyfit(range(len(x)), x / (np.abs(x).mean() + 1e-10), 1)[0], raw=True
    )

    # VWAP deviation
    cum_vol_20 = volume.rolling(20).sum()
    cum_pv_20 = (close * volume).rolling(20).sum()
    vwap_20 = cum_pv_20 / (cum_vol_20 + 1e-10)
    feat["vwap_dev_20d"] = (close - vwap_20) / (vwap_20 + 1e-10)

    # Flow score composite
    obv_z = (feat["obv_slope_20d"] - feat["obv_slope_20d"].rolling(60).mean()) / (feat["obv_slope_20d"].rolling(60).std() + 1e-10)
    mfi_z = (mfi - 50) / 50
    ad_z = (feat["ad_slope_20d"] - feat["ad_slope_20d"].rolling(60).mean()) / (feat["ad_slope_20d"].rolling(60).std() + 1e-10)
    feat["flow_composite"] = (obv_z + mfi_z + ad_z) / 3

    # ─── VOLUME CLIMAX (8 features) ───
    vol_ma20 = volume.rolling(20).mean()
    vol_ma50 = volume.rolling(50).mean()
    feat["rel_volume_20d"] = volume / (vol_ma20 + 1e-10)
    feat["rel_volume_50d"] = volume / (vol_ma50 + 1e-10)
    feat["volume_climax_2x"] = (volume > 2 * vol_ma20).astype(float)
    feat["volume_climax_3x"] = (volume > 3 * vol_ma20).astype(float)
    feat["volume_trend_20d"] = vol_ma20.pct_change(20)
    feat["vol_price_corr_20d"] = lr.rolling(20).corr(volume.pct_change())
    feat["up_vol_ratio"] = (volume.where(lr > 0, 0).rolling(20).sum()) / (volume.rolling(20).sum() + 1e-10)
    feat["down_vol_ratio"] = (volume.where(lr < 0, 0).rolling(20).sum()) / (volume.rolling(20).sum() + 1e-10)

    # ─── CTA POSITIONING PROXIES (12 features) ───
    # Trend-following signals across timeframes (what CTAs would trade)
    for window in [10, 20, 50, 100]:
        sma = close.rolling(window).mean()
        feat[f"cta_trend_{window}d"] = (close - sma) / (sma + 1e-10)

    # Dual momentum (absolute + relative)
    feat["cta_abs_mom_63d"] = (close.pct_change(63) > 0).astype(float)
    feat["cta_abs_mom_126d"] = (close.pct_change(126) > 0).astype(float)

    # Breakout signals
    feat["cta_breakout_20d"] = (close >= close.rolling(20).max()).astype(float)
    feat["cta_breakdown_20d"] = (close <= close.rolling(20).min()).astype(float)
    feat["cta_channel_pos"] = (close - close.rolling(20).min()) / (close.rolling(20).max() - close.rolling(20).min() + 1e-10)

    # Trend strength (ADX proxy using directional movement)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
    atr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1).rolling(14).mean()
    feat["cta_plus_di"] = plus_dm.rolling(14).mean() / (atr + 1e-10)
    feat["cta_minus_di"] = minus_dm.rolling(14).mean() / (atr + 1e-10)
    feat["cta_dx"] = abs(feat["cta_plus_di"] - feat["cta_minus_di"]) / (feat["cta_plus_di"] + feat["cta_minus_di"] + 1e-10)

    # ─── RISK-ON / RISK-OFF RATIOS (8 features) ───
    if spy_close is not None:
        spy_lr = np.log(spy_close / spy_close.shift(1))
        feat["risk_beta_21d"] = lr.rolling(21).cov(spy_lr) / (spy_lr.rolling(21).var() + 1e-10)
        feat["risk_beta_63d"] = lr.rolling(63).cov(spy_lr) / (spy_lr.rolling(63).var() + 1e-10)
        feat["risk_corr_spy_21d"] = lr.rolling(21).corr(spy_lr)
        feat["risk_corr_spy_63d"] = lr.rolling(63).corr(spy_lr)
        feat["risk_rel_vol"] = (lr.rolling(21).std()) / (spy_lr.rolling(21).std() + 1e-10)
        feat["risk_capture_up"] = lr.where(spy_lr > 0, np.nan).rolling(63).mean() / (spy_lr.where(spy_lr > 0, np.nan).rolling(63).mean() + 1e-10)
        feat["risk_capture_down"] = lr.where(spy_lr < 0, np.nan).rolling(63).mean() / (spy_lr.where(spy_lr < 0, np.nan).rolling(63).mean() + 1e-10)
        feat["risk_capture_ratio"] = feat["risk_capture_up"] / (feat["risk_capture_down"] + 1e-10)
    else:
        for c in ["risk_beta_21d", "risk_beta_63d", "risk_corr_spy_21d", "risk_corr_spy_63d",
                   "risk_rel_vol", "risk_capture_up", "risk_capture_down", "risk_capture_ratio"]:
            feat[c] = 0.0

    # ─── SECTOR ROTATION VELOCITY (10 features) ───
    if all_sector_closes is not None and len(all_sector_closes) > 1:
        # Cross-sectional dispersion
        all_rets_5d = pd.DataFrame({t: c.pct_change(5) for t, c in all_sector_closes.items()})
        all_rets_21d = pd.DataFrame({t: c.pct_change(21) for t, c in all_sector_closes.items()})

        feat["rot_dispersion_5d"] = all_rets_5d.std(axis=1)
        feat["rot_dispersion_21d"] = all_rets_21d.std(axis=1)

        # Own rank among sectors
        own_ticker = close.name if hasattr(close, "name") and close.name else None
        if own_ticker and own_ticker in all_rets_5d.columns:
            feat["rot_rank_5d"] = all_rets_5d.rank(axis=1, pct=True).get(own_ticker, 0.5)
            feat["rot_rank_21d"] = all_rets_21d.rank(axis=1, pct=True).get(own_ticker, 0.5)
        else:
            feat["rot_rank_5d"] = 0.5
            feat["rot_rank_21d"] = 0.5

        # Rotation velocity = change in rank
        feat["rot_velocity_5d"] = feat["rot_rank_5d"].diff(5)
        feat["rot_velocity_21d"] = feat["rot_rank_21d"].diff(21)

        # Sector momentum persistence
        feat["rot_persistence"] = feat["rot_rank_21d"].rolling(63).apply(
            lambda x: np.corrcoef(range(len(x)), x)[0, 1] if len(x) == 63 else 0, raw=True
        )

        # Herfindahl concentration (are flows concentrated in few sectors?)
        wts = all_rets_21d.clip(lower=0)
        wts_norm = wts.div(wts.sum(axis=1) + 1e-10, axis=0)
        feat["rot_herfindahl"] = (wts_norm ** 2).sum(axis=1)

        # Leadership stability
        leader_5d = all_rets_5d.idxmax(axis=1)
        feat["rot_leader_change"] = (leader_5d != leader_5d.shift(1)).astype(float).rolling(21).mean()

        # Breadth (how many sectors positive)
        feat["rot_breadth"] = (all_rets_5d > 0).sum(axis=1) / all_rets_5d.shape[1]
    else:
        for c in ["rot_dispersion_5d", "rot_dispersion_21d", "rot_rank_5d", "rot_rank_21d",
                   "rot_velocity_5d", "rot_velocity_21d", "rot_persistence", "rot_herfindahl",
                   "rot_leader_change", "rot_breadth"]:
            feat[c] = 0.0

    # ─── INSTITUTIONAL FLOW PROXIES (10 features) ───
    # Smart money indicator: close relative to high-low range
    feat["inst_smart_money"] = (close - low) / (high - low + 1e-10)
    feat["inst_smart_money_5d"] = feat["inst_smart_money"].rolling(5).mean()

    # Buying/selling pressure
    feat["inst_buy_pressure"] = ((close - low) / (high - low + 1e-10) * volume).rolling(10).sum()
    feat["inst_sell_pressure"] = ((high - close) / (high - low + 1e-10) * volume).rolling(10).sum()
    feat["inst_pressure_ratio"] = feat["inst_buy_pressure"] / (feat["inst_sell_pressure"] + 1e-10)

    # Large volume days divergence (institutional activity proxy)
    vol_90pct = volume.rolling(60).quantile(0.9)
    large_vol_days = volume > vol_90pct
    feat["inst_large_vol_ret"] = lr.where(large_vol_days, np.nan).rolling(20, min_periods=1).mean()
    feat["inst_large_vol_direction"] = lr.where(large_vol_days, np.nan).rolling(20, min_periods=1).apply(
        lambda x: (x > 0).mean() if len(x.dropna()) > 0 else 0.5, raw=False
    )

    # Accumulation streak
    feat["inst_accum_streak"] = (feat["flow_composite"] > 0).astype(float).rolling(20).sum()

    # Price-volume trend divergence
    price_trend = close.pct_change(20)
    vol_trend = volume.pct_change(20)
    feat["inst_pv_divergence"] = price_trend * (-vol_trend)

    # Net flow momentum
    net_flow = (pos_flow.rolling(14).sum() - neg_flow.rolling(14).sum()) / (raw_money_flow.rolling(14).sum() + 1e-10)
    feat["inst_net_flow_mom"] = net_flow.diff(5)

    return feat


# All flow feature names (55)
FLOW_FEATURE_NAMES = [
    # Money flow core (7)
    "obv_slope_10d", "obv_slope_20d", "mfi_14", "ad_slope_10d", "ad_slope_20d",
    "vwap_dev_20d", "flow_composite",
    # Volume climax (8)
    "rel_volume_20d", "rel_volume_50d", "volume_climax_2x", "volume_climax_3x",
    "volume_trend_20d", "vol_price_corr_20d", "up_vol_ratio", "down_vol_ratio",
    # CTA positioning (12)
    "cta_trend_10d", "cta_trend_20d", "cta_trend_50d", "cta_trend_100d",
    "cta_abs_mom_63d", "cta_abs_mom_126d", "cta_breakout_20d", "cta_breakdown_20d",
    "cta_channel_pos", "cta_plus_di", "cta_minus_di", "cta_dx",
    # Risk-on/off (8)
    "risk_beta_21d", "risk_beta_63d", "risk_corr_spy_21d", "risk_corr_spy_63d",
    "risk_rel_vol", "risk_capture_up", "risk_capture_down", "risk_capture_ratio",
    # Sector rotation (10)
    "rot_dispersion_5d", "rot_dispersion_21d", "rot_rank_5d", "rot_rank_21d",
    "rot_velocity_5d", "rot_velocity_21d", "rot_persistence", "rot_herfindahl",
    "rot_leader_change", "rot_breadth",
    # Institutional flow (10)
    "inst_smart_money", "inst_smart_money_5d", "inst_buy_pressure", "inst_sell_pressure",
    "inst_pressure_ratio", "inst_large_vol_ret", "inst_large_vol_direction",
    "inst_accum_streak", "inst_pv_divergence", "inst_net_flow_mom",
]

ALL_FEATURE_NAMES = LEGACY_FEATURE_NAMES + FLOW_FEATURE_NAMES


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD FEATURE MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

def build_dataset(data, vix_series):
    """Build full feature matrix for all sector ETFs.
    Returns X (all features), y (fwd 21d return), meta (date/ticker/vix_band)."""
    spy_close = data.get("SPY", {}).get("Close") if isinstance(data.get("SPY"), pd.DataFrame) else None
    if spy_close is None and "SPY" in data:
        spy_close = data["SPY"]["Close"]

    # Prepare sector closes dict for rotation features
    sector_closes = {}
    for t in SECTOR_ETFS:
        if t in data:
            s = data[t]["Close"].copy()
            s.name = t
            sector_closes[t] = s

    # Find common date range
    common_idx = None
    for t in SECTOR_ETFS:
        if t in data:
            idx = data[t].index
            common_idx = idx if common_idx is None else common_idx.intersection(idx)
    if spy_close is not None:
        common_idx = common_idx.intersection(spy_close.index)
    common_idx = common_idx.intersection(vix_series.dropna().index)

    features_list = []
    labels_list = []
    meta_list = []

    for t in SECTOR_ETFS:
        if t not in data:
            continue

        df = data[t]
        close = df["Close"].reindex(common_idx)
        high = df["High"].reindex(common_idx) if "High" in df.columns else close
        low = df["Low"].reindex(common_idx) if "Low" in df.columns else close
        volume = df["Volume"].reindex(common_idx) if "Volume" in df.columns else pd.Series(0, index=common_idx)
        sc = spy_close.reindex(common_idx) if spy_close is not None else None

        close.name = t  # for rotation rank features

        # Legacy features
        legacy = compute_legacy_features(close, volume, spy_close=sc)

        # Flow features
        flow = compute_flow_features(close, high, low, volume, spy_close=sc,
                                     all_sector_closes={k: v.reindex(common_idx) for k, v in sector_closes.items()})

        # Combine
        all_feat = pd.concat([legacy, flow], axis=1)

        # Forward return (target)
        fwd_ret = close.pct_change(21).shift(-21)

        # VIX at signal time
        vix_at_date = vix_series.reindex(common_idx)

        # Replace inf with nan, then fill nan with 0 (LGBM handles this fine)
        all_feat = all_feat.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Need enough warmup for rolling features (252d for ret_252d, etc.)
        valid = all_feat.index[all_feat.index >= all_feat.index[min(300, len(all_feat)-1)]]
        valid = valid.intersection(fwd_ret.dropna().index)

        for d in valid:
            row = all_feat.loc[d].values

            vix_val = vix_at_date.get(d, np.nan) if hasattr(vix_at_date, "get") else (vix_at_date.loc[d] if d in vix_at_date.index else np.nan)
            if pd.isna(vix_val):
                continue

            # Classify VIX band
            if vix_val < VIX_BAND_A[0]:
                vix_band = "low"  # VIX < 20
            elif vix_val < VIX_BAND_A[1]:
                vix_band = "A"  # 20-25
            elif vix_val < VIX_BAND_B[1]:
                vix_band = "B"  # 25-30
            else:
                vix_band = "C"  # 30+

            features_list.append(row)
            labels_list.append(float(fwd_ret.loc[d]))
            meta_list.append({"date": d, "ticker": t, "vix": float(vix_val), "vix_band": vix_band})

    if not features_list:
        raise RuntimeError("No valid samples produced. Check feature computation and date alignment.")

    X = np.array(features_list)
    y = np.array(labels_list)
    meta = pd.DataFrame(meta_list)

    fprint(f"  Dataset: {X.shape[0]} samples x {X.shape[1]} features")
    fprint(f"  Date range: {meta['date'].min().date()} to {meta['date'].max().date()}")
    fprint(f"  VIX bands — low(<20): {(meta['vix_band']=='low').sum()}, "
           f"A(20-25): {(meta['vix_band']=='A').sum()}, "
           f"B(25-30): {(meta['vix_band']=='B').sum()}, "
           f"C(30+): {(meta['vix_band']=='C').sum()}")

    return X, y, meta


# ═══════════════════════════════════════════════════════════════════════════════
# REGIME-CONDITIONAL MODEL TRAINING + WALK-FORWARD
# ═══════════════════════════════════════════════════════════════════════════════

def get_feature_indices(feature_names, target_names):
    """Get column indices for a subset of features."""
    return [i for i, n in enumerate(feature_names) if n in target_names]


def run_walkforward(X, y, meta, data, vix_series, model_type="regime_conditional",
                    top_k=TOP_K, name="regime_conditional"):
    """
    Walk-forward backtest with LGBM.

    model_type:
      - "legacy_only": single LGBM on 18 legacy features
      - "legacy_flow": single LGBM on all 73 features
      - "regime_conditional": 3 models per VIX band (A/B use all, C uses legacy only)
      - "random": random ranking baseline
    """
    import lightgbm as lgb

    dates = sorted(meta["date"].unique())
    n_dates = len(dates)

    legacy_idx = get_feature_indices(ALL_FEATURE_NAMES, LEGACY_FEATURE_NAMES)
    all_idx = list(range(X.shape[1]))

    monthly_returns = []
    monthly_picks_info = []
    all_predictions = []
    feat_imp_accum = None
    n_folds = 0

    fprint(f"\n  [{name}] Walk-forward: {TRAIN_DAYS}d train, {TEST_DAYS}d test, sliding")

    i = TRAIN_DAYS
    while i + TEST_DAYS <= n_dates:
        train_dates = dates[i - TRAIN_DAYS:i]
        test_dates = dates[i:i + TEST_DAYS]

        train_mask = meta["date"].isin(train_dates)
        test_mask = meta["date"].isin(test_dates)

        X_tr_full, y_tr = X[train_mask.values], y[train_mask.values]
        X_te_full = X[test_mask.values]
        meta_te = meta[test_mask].copy()

        if len(X_tr_full) < 50 or len(X_te_full) < 3:
            i += TEST_DAYS
            continue

        # Get VIX at test time
        test_date = test_dates[0]
        vix_val = vix_series.get(test_date, np.nan) if hasattr(vix_series, "get") else vix_series.reindex([test_date]).iloc[0] if test_date in vix_series.index else np.nan
        if np.isnan(vix_val):
            # Fallback: use latest available VIX
            vix_before = vix_series[:test_date]
            vix_val = vix_before.iloc[-1] if len(vix_before) > 0 else 20.0

        if vix_val < VIX_BAND_A[0]:
            current_band = "low"
        elif vix_val < VIX_BAND_A[1]:
            current_band = "A"
        elif vix_val < VIX_BAND_B[1]:
            current_band = "B"
        else:
            current_band = "C"

        # Choose features and training data based on model_type
        lgb_params = dict(
            n_estimators=150, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            reg_lambda=1.0, verbose=-1, n_jobs=-1, random_state=42
        )

        if model_type == "random":
            # Random predictions
            meta_te = meta_te.copy()
            meta_te["pred"] = np.random.randn(len(meta_te))

        elif model_type == "legacy_only":
            X_tr_sel = X_tr_full[:, legacy_idx]
            X_te_sel = X_te_full[:, legacy_idx]
            model = lgb.LGBMRegressor(**lgb_params)
            model.fit(X_tr_sel, y_tr)
            meta_te = meta_te.copy()
            meta_te["pred"] = model.predict(X_te_sel)
            if feat_imp_accum is None:
                feat_imp_accum = model.feature_importances_.astype(float)
            else:
                feat_imp_accum += model.feature_importances_.astype(float)

        elif model_type == "legacy_flow":
            X_tr_sel = X_tr_full[:, all_idx]
            X_te_sel = X_te_full[:, all_idx]
            model = lgb.LGBMRegressor(**lgb_params)
            model.fit(X_tr_sel, y_tr)
            meta_te = meta_te.copy()
            meta_te["pred"] = model.predict(X_te_sel)
            if feat_imp_accum is None:
                feat_imp_accum = model.feature_importances_.astype(float)
            else:
                feat_imp_accum += model.feature_importances_.astype(float)

        elif model_type == "regime_conditional":
            meta_tr = meta[train_mask].copy()

            if current_band == "C":
                # Crisis: legacy features ONLY (flow breaks in VIX 30+)
                use_idx = legacy_idx

                # Train on crisis-like data if available, else all data
                crisis_mask = meta_tr["vix_band"].isin(["B", "C"])
                if crisis_mask.sum() >= 30:
                    X_tr_sel = X_tr_full[crisis_mask.values][:, use_idx]
                    y_tr_sel = y_tr[crisis_mask.values]
                else:
                    # Not enough crisis data — use all training data but legacy features only
                    X_tr_sel = X_tr_full[:, use_idx]
                    y_tr_sel = y_tr

                X_te_sel = X_te_full[:, use_idx]

            elif current_band in ["A", "low"]:
                # Moderate vol: all features (flow works here)
                use_idx = all_idx

                # Prefer training on non-crisis data
                normal_mask = meta_tr["vix_band"].isin(["low", "A"])
                if normal_mask.sum() >= 50:
                    X_tr_sel = X_tr_full[normal_mask.values][:, use_idx]
                    y_tr_sel = y_tr[normal_mask.values]
                else:
                    X_tr_sel = X_tr_full[:, use_idx]
                    y_tr_sel = y_tr

                X_te_sel = X_te_full[:, use_idx]

            else:
                # Band B (25-30): all features but trained on elevated vol data
                use_idx = all_idx

                elev_mask = meta_tr["vix_band"].isin(["A", "B"])
                if elev_mask.sum() >= 40:
                    X_tr_sel = X_tr_full[elev_mask.values][:, use_idx]
                    y_tr_sel = y_tr[elev_mask.values]
                else:
                    X_tr_sel = X_tr_full[:, use_idx]
                    y_tr_sel = y_tr

                X_te_sel = X_te_full[:, use_idx]

            model = lgb.LGBMRegressor(**lgb_params)
            model.fit(X_tr_sel, y_tr_sel)
            meta_te = meta_te.copy()
            meta_te["pred"] = model.predict(X_te_sel)

        # Select top-K sectors
        td = test_dates[0]
        dp = meta_te[meta_te["date"] == td].copy()
        if len(dp) < top_k:
            i += TEST_DAYS
            continue

        top = dp.nlargest(top_k, "pred")
        picks = list(top["ticker"].values)

        # Calculate actual returns (equal weight)
        period_ret = 0
        valid_picks = 0
        for _, row in top.iterrows():
            t = row["ticker"]
            if t in data:
                tc = data[t]["Close"]
                si = tc.index.searchsorted(test_dates[0])
                ei = tc.index.searchsorted(test_dates[-1]) if test_dates[-1] in tc.index else min(si + TEST_DAYS, len(tc) - 1)
                if si < len(tc) and ei < len(tc) and tc.iloc[si] > 0:
                    stock_ret = tc.iloc[ei] / tc.iloc[si] - 1
                    period_ret += stock_ret / top_k
                    valid_picks += 1

        # Deduct costs
        period_ret -= TOTAL_COST_BPS / 10000 * 2

        monthly_returns.append(float(period_ret))
        monthly_picks_info.append({
            "date": str(td.date()),
            "picks": picks,
            "return": float(period_ret),
            "vix": float(vix_val),
            "vix_band": current_band,
        })
        n_folds += 1
        i += TEST_DAYS

    fprint(f"  [{name}] {n_folds} folds completed")

    # SPY benchmark
    spy_returns = []
    if "SPY" in data:
        spy_close_bench = data["SPY"]["Close"]
        i = TRAIN_DAYS
        while i + TEST_DAYS <= n_dates:
            test_dates = dates[i:i + TEST_DAYS]
            si = spy_close_bench.index.searchsorted(test_dates[0])
            ei = spy_close_bench.index.searchsorted(test_dates[-1]) if test_dates[-1] in spy_close_bench.index else min(si + TEST_DAYS, len(spy_close_bench) - 1)
            if si < len(spy_close_bench) and ei < len(spy_close_bench) and spy_close_bench.iloc[si] > 0:
                sr = spy_close_bench.iloc[ei] / spy_close_bench.iloc[si] - 1
                spy_returns.append(float(sr))
            i += TEST_DAYS

    return {
        "monthly_returns": monthly_returns,
        "spy_returns": spy_returns[:len(monthly_returns)],
        "monthly_picks": monthly_picks_info,
        "n_folds": n_folds,
        "feature_importances": feat_imp_accum,
        "name": name,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS + VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(returns, name, spy_returns=None):
    """Compute risk-adjusted metrics."""
    r = np.array(returns)
    if len(r) < 2:
        return {"name": name, "n_months": 0}

    ppy = 12
    sharpe = np.mean(r) / (np.std(r) + 1e-10) * np.sqrt(ppy)
    ds = r[r < 0]
    sortino = np.mean(r) / (np.std(ds) + 1e-10) * np.sqrt(ppy) if len(ds) > 0 else 0

    equity = 100000 * np.cumprod(1 + r)
    years = len(r) / ppy
    cagr = ((equity[-1] / 100000) ** (1 / max(years, 0.01)) - 1) * 100
    peak = np.maximum.accumulate(equity)
    maxdd = float(np.min((equity - peak) / (peak + 1e-10)) * 100)
    wr = len(r[r > 0]) / len(r) * 100
    pf = abs(r[r > 0].sum()) / (abs(r[r < 0].sum()) + 1e-10) if len(ds) > 0 else 999
    calmar = cagr / (abs(maxdd) + 1e-10)

    result = {
        "name": name,
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "cagr_pct": round(float(cagr), 2),
        "maxdd_pct": round(float(maxdd), 2),
        "wr_pct": round(float(wr), 1),
        "pf": round(float(pf), 3),
        "calmar": round(float(calmar), 3),
        "n_months": len(r),
        "final_equity": round(float(equity[-1]), 2),
    }

    if spy_returns is not None and len(spy_returns) > 0:
        sp = np.array(spy_returns[:len(r)])
        spy_sharpe = np.mean(sp) / (np.std(sp) + 1e-10) * np.sqrt(ppy)
        spy_eq = 100000 * np.cumprod(1 + sp)
        spy_cagr = ((spy_eq[-1] / 100000) ** (1 / max(years, 0.01)) - 1) * 100
        result["spy_sharpe"] = round(float(spy_sharpe), 3)
        result["spy_cagr_pct"] = round(float(spy_cagr), 2)
        result["alpha_sharpe"] = round(float(sharpe - spy_sharpe), 3)

    return result


def adversarial_validation(returns, monthly_picks, n_perm=N_PERMUTATIONS):
    """Full adversarial validation: permutation, regime, sub-period, yearly."""
    r = np.array(returns)
    if len(r) < 12:
        return {"error": "Too few months for validation"}

    gates = {}

    # 1. PERMUTATION TEST
    real_sharpe = np.mean(r) / (np.std(r) + 1e-10) * np.sqrt(12)
    count_better = 0
    for _ in range(n_perm):
        signs = np.random.choice([-1, 1], size=len(r))
        perm_sharpe = np.mean(r * signs) / (np.std(r * signs) + 1e-10) * np.sqrt(12)
        if perm_sharpe >= real_sharpe:
            count_better += 1
    perm_p = count_better / n_perm
    gates["permutation"] = {
        "p_value": round(float(perm_p), 4),
        "pass": perm_p < 0.05,
        "real_sharpe": round(float(real_sharpe), 3),
    }

    # 2. REGIME SPLIT (HC #428 R1)
    picks_df = pd.DataFrame(monthly_picks)
    if "vix_band" in picks_df.columns:
        regime_sharpes = {}
        for band in ["low", "A", "B", "C"]:
            mask = picks_df["vix_band"] == band
            if mask.sum() >= 3:
                band_r = r[mask.values[:len(r)]] if len(mask) >= len(r) else r[mask.values]
                regime_sharpes[band] = float(np.mean(band_r) / (np.std(band_r) + 1e-10) * np.sqrt(12))
            else:
                regime_sharpes[band] = None

        # Check regime gap
        valid_sharpes = [v for v in regime_sharpes.values() if v is not None]
        if len(valid_sharpes) >= 2:
            max_gap = 0
            for i_s, s1 in enumerate(valid_sharpes):
                for s2 in valid_sharpes[i_s + 1:]:
                    gap = abs(s1 - s2) / (max(abs(s1), abs(s2)) + 1e-10)
                    max_gap = max(max_gap, gap)
            regime_pass = max_gap <= 0.50
        else:
            max_gap = 0
            regime_pass = True

        gates["regime"] = {
            "sharpes_by_band": regime_sharpes,
            "max_gap": round(float(max_gap), 3),
            "pass": regime_pass,
        }
    else:
        gates["regime"] = {"pass": True, "note": "No VIX band data"}

    # 3. SUB-PERIOD TEST (halves)
    half = len(r) // 2
    first_half = r[:half]
    second_half = r[half:]
    sh1 = np.mean(first_half) / (np.std(first_half) + 1e-10) * np.sqrt(12)
    sh2 = np.mean(second_half) / (np.std(second_half) + 1e-10) * np.sqrt(12)
    sub_gap = abs(sh1 - sh2) / (max(abs(sh1), abs(sh2)) + 1e-10)
    gates["sub_period"] = {
        "first_half_sharpe": round(float(sh1), 3),
        "second_half_sharpe": round(float(sh2), 3),
        "gap": round(float(sub_gap), 3),
        "pass": sub_gap <= 0.60,
    }

    # 4. YEARLY CONSISTENCY
    picks_df = pd.DataFrame(monthly_picks)
    if "date" in picks_df.columns and len(picks_df) >= len(r):
        picks_df["year"] = pd.to_datetime(picks_df["date"]).dt.year
        picks_df["return"] = r[:len(picks_df)]
        yearly = picks_df.groupby("year")["return"].agg(["mean", "std", "count"])
        yearly["sharpe"] = yearly["mean"] / (yearly["std"] + 1e-10) * np.sqrt(12)
        prof_years = (yearly["mean"] > 0).sum()
        total_years = len(yearly)
        gates["yearly"] = {
            "profitable_years": f"{prof_years}/{total_years}",
            "yearly_sharpes": {str(k): round(float(v), 3) for k, v in yearly["sharpe"].items()},
            "pass": prof_years / total_years >= 0.60,
        }
    else:
        gates["yearly"] = {"pass": True}

    # Overall
    gates["all_pass"] = all(g.get("pass", True) for g in gates.values())

    return gates


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    fprint("=" * 72)
    fprint("REGIME-CONDITIONAL FLOW MODEL v1")
    fprint("=" * 72)
    fprint(f"Hypothesis: Regime-conditional model selection fixes crash-regime")
    fprint(f"            fragility while keeping moderate-vol flow improvement")
    fprint(f"Universe: {len(SECTOR_ETFS)} sector ETFs + {len(MACRO_TICKERS)} macro tickers")
    fprint(f"Period: {START_DATE} to {END_DATE}")
    fprint(f"VIX bands: A={VIX_BAND_A}, B={VIX_BAND_B}, C={VIX_BAND_C}")
    fprint(f"Features: {len(LEGACY_FEATURE_NAMES)} legacy + {len(FLOW_FEATURE_NAMES)} flow = {len(ALL_FEATURE_NAMES)} total")
    fprint("=" * 72)

    # ─── Download data ───
    fprint("\n[1] Downloading data...")
    data = download_data(ALL_TICKERS, OUTPUT / "price_cache.parquet")
    vix_series = get_vix_series(data)
    fprint(f"  VIX range: {vix_series.min():.1f} to {vix_series.max():.1f}")

    # ─── Build feature matrix ───
    fprint("\n[2] Building feature matrix ({} legacy + {} flow features)...".format(
        len(LEGACY_FEATURE_NAMES), len(FLOW_FEATURE_NAMES)))
    X, y, meta = build_dataset(data, vix_series)

    # ─── MLflow experiment ───
    mlflow_run = None
    if MLFLOW_OK:
        try:
            mlflow.set_experiment("regime_conditional_flow_v1")
            mlflow_run = mlflow.start_run(run_name=f"rcf_v1_{datetime.now().strftime('%Y%m%d_%H%M')}")
            mlflow.log_params({
                "n_sector_etfs": len(SECTOR_ETFS),
                "n_legacy_features": len(LEGACY_FEATURE_NAMES),
                "n_flow_features": len(FLOW_FEATURE_NAMES),
                "train_days": TRAIN_DAYS,
                "test_days": TEST_DAYS,
                "top_k": TOP_K,
                "cost_bps": TOTAL_COST_BPS,
                "vix_band_A": str(VIX_BAND_A),
                "vix_band_B": str(VIX_BAND_B),
                "vix_band_C": str(VIX_BAND_C),
            })
        except Exception as e:
            fprint(f"  MLflow param logging failed: {e}")

    # ─── Run all model variants ───
    fprint("\n[3] Running walk-forward backtests...")
    variants = {}

    # Variant 1: Legacy only (baseline)
    fprint("\n  --- VARIANT 1: Single model, legacy features only (18 features) ---")
    v1 = run_walkforward(X, y, meta, data, vix_series, model_type="legacy_only", name="legacy_only")
    variants["legacy_only"] = v1

    # Variant 2: Legacy + Flow (single model, all features)
    fprint("\n  --- VARIANT 2: Single model, legacy + flow features (73 features) ---")
    v2 = run_walkforward(X, y, meta, data, vix_series, model_type="legacy_flow", name="legacy_flow")
    variants["legacy_flow"] = v2

    # Variant 3: Regime-conditional (THE FIX)
    fprint("\n  --- VARIANT 3: Regime-conditional (3 models by VIX band) ---")
    v3 = run_walkforward(X, y, meta, data, vix_series, model_type="regime_conditional", name="regime_conditional")
    variants["regime_conditional"] = v3

    # Variant 4: Random baseline
    fprint("\n  --- VARIANT 4: Random ranking baseline ---")
    v4 = run_walkforward(X, y, meta, data, vix_series, model_type="random", name="random")
    variants["random"] = v4

    # ─── Compute metrics ───
    fprint("\n[4] Computing metrics...")
    fprint("=" * 72)

    all_metrics = {}
    for vname, vdata in variants.items():
        m = compute_metrics(vdata["monthly_returns"], vname, spy_returns=vdata.get("spy_returns"))
        all_metrics[vname] = m
        fprint(f"\n  {vname}:")
        fprint(f"    Sharpe: {m.get('sharpe', 'N/A')}")
        fprint(f"    Sortino: {m.get('sortino', 'N/A')}")
        fprint(f"    CAGR: {m.get('cagr_pct', 'N/A')}%")
        fprint(f"    MaxDD: {m.get('maxdd_pct', 'N/A')}%")
        fprint(f"    Win Rate: {m.get('wr_pct', 'N/A')}%")
        fprint(f"    Profit Factor: {m.get('pf', 'N/A')}")
        fprint(f"    Calmar: {m.get('calmar', 'N/A')}")
        fprint(f"    Months: {m.get('n_months', 0)}")
        if "alpha_sharpe" in m:
            fprint(f"    Alpha vs SPY: {m['alpha_sharpe']} Sharpe points")

    # ─── KEY COMPARISON ───
    fprint("\n" + "=" * 72)
    fprint("KEY COMPARISON: Does regime-conditional fix the crash fragility?")
    fprint("=" * 72)

    legacy_sharpe = all_metrics.get("legacy_only", {}).get("sharpe", 0)
    flow_sharpe = all_metrics.get("legacy_flow", {}).get("sharpe", 0)
    regime_sharpe = all_metrics.get("regime_conditional", {}).get("sharpe", 0)
    random_sharpe = all_metrics.get("random", {}).get("sharpe", 0)

    flow_improvement = ((flow_sharpe - legacy_sharpe) / (abs(legacy_sharpe) + 1e-10)) * 100
    regime_improvement = ((regime_sharpe - legacy_sharpe) / (abs(legacy_sharpe) + 1e-10)) * 100

    fprint(f"\n  Legacy only Sharpe:        {legacy_sharpe:.3f}")
    fprint(f"  Legacy+Flow Sharpe:        {flow_sharpe:.3f} ({flow_improvement:+.1f}% vs legacy)")
    fprint(f"  Regime-conditional Sharpe: {regime_sharpe:.3f} ({regime_improvement:+.1f}% vs legacy)")
    fprint(f"  Random baseline Sharpe:    {random_sharpe:.3f}")

    # ─── Adversarial validation ───
    fprint("\n[5] Running adversarial validation (all variants)...")
    fprint("=" * 72)

    all_gates = {}
    for vname, vdata in variants.items():
        if vname == "random":
            continue
        fprint(f"\n  --- {vname} ---")
        gates = adversarial_validation(vdata["monthly_returns"], vdata["monthly_picks"])
        all_gates[vname] = gates

        for gate_name, gate_result in gates.items():
            if gate_name == "all_pass":
                continue
            if isinstance(gate_result, dict):
                status = "PASS" if gate_result.get("pass") else "FAIL"
                fprint(f"    {gate_name}: {status}")
                for k, v in gate_result.items():
                    if k != "pass":
                        fprint(f"      {k}: {v}")

        fprint(f"    ALL GATES: {'PASS' if gates.get('all_pass') else 'FAIL'}")

    # ─── Per-VIX-band IC analysis ───
    fprint("\n[6] Per-VIX-band IC analysis...")
    fprint("=" * 72)

    for vname in ["legacy_only", "legacy_flow", "regime_conditional"]:
        vdata = variants[vname]
        picks = pd.DataFrame(vdata["monthly_picks"])
        if "vix_band" not in picks.columns or "return" not in picks.columns:
            continue
        fprint(f"\n  {vname}:")
        for band in ["low", "A", "B", "C"]:
            band_data = picks[picks["vix_band"] == band]
            if len(band_data) < 3:
                fprint(f"    VIX band {band}: too few samples ({len(band_data)})")
                continue
            band_rets = band_data["return"].values
            band_sharpe = np.mean(band_rets) / (np.std(band_rets) + 1e-10) * np.sqrt(12)
            band_wr = (band_rets > 0).mean() * 100
            fprint(f"    VIX band {band}: Sharpe {band_sharpe:.3f}, WR {band_wr:.1f}%, "
                   f"mean ret {np.mean(band_rets)*100:.2f}%, n={len(band_data)}")

    # ─── Log to MLflow ───
    if MLFLOW_OK and mlflow_run:
        try:
            for vname, m in all_metrics.items():
                for mk, mv in m.items():
                    if isinstance(mv, (int, float)):
                        mlflow.log_metric(f"{vname}_{mk}", mv)

            for vname, g in all_gates.items():
                for gname, gval in g.items():
                    if isinstance(gval, dict) and "pass" in gval:
                        mlflow.log_metric(f"{vname}_{gname}_pass", 1.0 if gval["pass"] else 0.0)
                    elif gname == "all_pass":
                        mlflow.log_metric(f"{vname}_all_gates_pass", 1.0 if gval else 0.0)

            mlflow.end_run()
            fprint("\n  MLflow run logged successfully")
        except Exception as e:
            fprint(f"\n  MLflow logging error: {e}")
            try:
                mlflow.end_run()
            except:
                pass

    # ─── Save report ───
    report = {
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "Regime-conditional model selection fixes crash-regime fragility",
        "config": {
            "sector_etfs": SECTOR_ETFS,
            "macro_tickers": MACRO_TICKERS,
            "n_legacy_features": len(LEGACY_FEATURE_NAMES),
            "n_flow_features": len(FLOW_FEATURE_NAMES),
            "vix_bands": {"A": VIX_BAND_A, "B": VIX_BAND_B, "C": VIX_BAND_C},
            "train_days": TRAIN_DAYS,
            "test_days": TEST_DAYS,
            "top_k": TOP_K,
            "cost_bps": TOTAL_COST_BPS,
            "n_permutations": N_PERMUTATIONS,
        },
        "metrics": all_metrics,
        "adversarial_gates": all_gates,
        "key_comparison": {
            "legacy_sharpe": legacy_sharpe,
            "flow_sharpe": flow_sharpe,
            "regime_conditional_sharpe": regime_sharpe,
            "random_sharpe": random_sharpe,
            "flow_improvement_pct": round(flow_improvement, 2),
            "regime_improvement_pct": round(regime_improvement, 2),
        },
        "conclusion": (
            "REGIME-CONDITIONAL FIXES CRASH FRAGILITY"
            if all_gates.get("regime_conditional", {}).get("all_pass") and regime_sharpe > flow_sharpe
            else "REGIME-CONDITIONAL HELPS BUT NEEDS MORE WORK"
            if regime_sharpe > legacy_sharpe
            else "REGIME-CONDITIONAL DOES NOT HELP"
        ),
    }

    report_path = OUTPUT / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    elapsed = time.time() - t0
    fprint(f"\n{'='*72}")
    fprint(f"DONE in {elapsed/60:.1f} minutes")
    fprint(f"Report: {report_path}")
    fprint(f"{'='*72}")

    return report


if __name__ == "__main__":
    main()
