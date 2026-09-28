#!/usr/bin/env python3
"""
Momentum Crash Detection v1 — Early-Warning Signals for Momentum Reversals
============================================================================

Research question: Can we build an early-warning signal for momentum crashes
that improves V6 sector spread strategy timing?

Background: V6 uses LGBM to rank 11 sector ETFs by momentum/quality features,
then trades bull call spreads on winners and bear put spreads on losers. The
biggest risk is momentum reversals — when yesterday's winners suddenly become
losers. The regime filter (VIX-based) doesn't catch this.

6 Variants (all with V6 structure: weekly, 2% OTM, 17 features):
  A: Baseline V6 (no crash detection)
  B: Skip trades when cross-sector correlation > 0.8 (herding = crash risk)
  C: Skip trades when top-bottom spread narrows > 50% in 5 days (ranking convergence)
  D: Skip trades when sector dispersion drops > 1 std in 5 days (sudden convergence)
  E: Combined: skip if ANY of B/C/D triggers
  F: Inverse: trade MORE aggressively when crash signals are absent

Momentum crash definition:
  Top-3 LGBM sectors underperforming bottom-3 by >2% in next week.

Honest pricing rules (inherited from V6):
  - HOLD TO EXPIRY ONLY (no early exit)
  - At expiry: INTRINSIC VALUE ONLY (no time value)
  - 15% haircut on ENTRY only (automatic exercise at expiry = no exit haircut)
  - DTE=21, $645 starting capital
  - Walk-forward LGBM, weekly rebalance, regime filter via GRU
  - 2% OTM bull call spreads, 3% width
  - Commission: $2.60/spread

Full 5-gate adversarial validation + random baseline comparison (5 trials).
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
    price_bear_put_spread,
    estimate_iv,
    compute_atr,
    COMMISSION_RT_SPREAD,
    DEFAULT_HAIRCUT,
)
from research.tools.adversarial_validator import validate_trades

# ── Config ──
BASE = Path("/home/jupiter/Lvl3Quant")
OUTPUT_DIR = BASE / "output" / "growth_research" / "momentum_crash_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
DTE = 21
SPREAD_PCT = 3.0
OTM_PCT = 0.02  # 2% OTM (V6 standard)
TOP_K = 3
REGIME_BULL_THRESHOLD = 0.4

# Regime predictions path
REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"

# LGBM walk-forward
WF_TRAIN_PERIODS = 12
WF_REBAL_FREQ = "W-FRI"  # Weekly (V6 standard)

# Crash detection thresholds
CORR_THRESHOLD = 0.80        # B: cross-sector correlation threshold
SPREAD_NARROW_PCT = 0.50     # C: top-bottom spread narrows >50% in 5 days
DISP_DROP_SIGMA = 1.0        # D: dispersion drops >1 std in 5 days
CRASH_LOOKBACK = 5           # Days to look back for crash signals
AGGRESSIVE_SIZE_MULT = 1.5   # F: trade 50% bigger when no crash signals

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "momentum_crash_v1"

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
# V6 FEATURE DEFINITIONS (17 features)
# ══════════════════════════════════════════════════════════════

V6_FEATURES = [
    # Momentum features
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    # Quality features (minus vol_21d/vol_63d/maxdd_63d)
    "sharpe_63d", "pct_52w_high", "mom_accel",
    "pct_pos_months_12m", "sortino_63d", "calmar_1y", "up_capture",
    "trend_r2_63d", "trend_slope_63d",
    # Cross-asset features (minus sector_relative_vol_21d)
    "sector_spy_beta_63d",
    "cross_sector_dispersion",
]

assert len(V6_FEATURES) == 17, f"Expected 17 features, got {len(V6_FEATURES)}"


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
           f"Days >0.4: {(regime_series > REGIME_BULL_THRESHOLD).sum()}")
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
    """Compute the quality-momentum features for a single sector ETF."""
    if len(px) < 260:
        return None
    f = {}
    for lb, nm in [(5, "ret_5d"), (10, "ret_10d"), (21, "ret_21d"),
                   (63, "ret_63d"), (126, "ret_126d"), (252, "ret_252d")]:
        f[nm] = float(px.iloc[-1] / px.iloc[-lb] - 1) if len(px) > lb else 0.0

    rets = px.pct_change().dropna()
    r63 = rets.iloc[-63:]
    f["sharpe_63d"] = float(r63.mean() / (r63.std() + 1e-10) * np.sqrt(252)) if len(r63) > 10 else 0.0
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
    """Compute the 2 V6 cross-asset features (beta + dispersion)."""
    f = {}

    spy = close_df["SPY"].iloc[:dt_idx + 1].dropna()
    sector_px = close_df[sector_ticker].iloc[:dt_idx + 1].dropna() if sector_ticker in close_df.columns else None

    if spy is None or len(spy) < 63:
        return {"sector_spy_beta_63d": 1.0, "cross_sector_dispersion": 0.01}

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

    # 2. Cross-sector dispersion
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
# MOMENTUM CRASH DETECTION SIGNALS
# ══════════════════════════════════════════════════════════════

def compute_crash_signals(close, dt):
    """
    Compute momentum crash early-warning signals at a given date.

    Returns dict with:
      - cross_sector_corr: average pairwise correlation of sector 5d returns (21d rolling)
      - spread_narrowing: % change in top-bottom momentum spread over last 5 days
      - disp_drop_zscore: z-score of dispersion drop over last 5 days
      - signal_B: True if cross-sector correlation > threshold (herding)
      - signal_C: True if top-bottom spread narrowed > 50% in 5 days
      - signal_D: True if dispersion dropped > 1 std in 5 days
    """
    sector_cols = [c for c in SECTORS if c in close.columns]
    idx = close.index.get_loc(dt) if dt in close.index else None
    if idx is None or idx < 63:
        return {"signal_B": False, "signal_C": False, "signal_D": False,
                "cross_sector_corr": 0.0, "spread_narrowing": 0.0, "disp_drop_zscore": 0.0}

    # ── Signal B: Cross-sector correlation (herding detection) ──
    # Use 21d rolling returns correlation matrix
    lookback = min(idx + 1, 63)
    sector_rets = close[sector_cols].iloc[idx - lookback + 1:idx + 1].pct_change().dropna()

    if len(sector_rets) >= 21:
        # Rolling 21d correlation matrix, take mean of upper triangle
        recent_rets = sector_rets.iloc[-21:]
        corr_matrix = recent_rets.corr()
        # Upper triangle mean (excluding diagonal)
        mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
        avg_corr = float(corr_matrix.values[mask].mean()) if mask.sum() > 0 else 0.0
    else:
        avg_corr = 0.0

    signal_B = avg_corr > CORR_THRESHOLD

    # ── Signal C: Top-bottom spread narrowing ──
    # Compute 21d momentum for each sector, measure top-bottom spread
    # Check if spread narrowed >50% in last 5 days
    if idx >= CRASH_LOOKBACK + 21:
        spreads = []
        for d in range(CRASH_LOOKBACK + 1):  # today and last 5 days
            di = idx - d
            if di < 21:
                break
            mom_scores = {}
            for tk in sector_cols:
                if di >= 21 and di < len(close):
                    p_now = close[tk].iloc[di]
                    p_ago = close[tk].iloc[di - 21]
                    if p_ago > 0:
                        mom_scores[tk] = float(p_now / p_ago - 1)
            if len(mom_scores) >= 6:
                sorted_scores = sorted(mom_scores.values(), reverse=True)
                top3_avg = np.mean(sorted_scores[:TOP_K])
                bot3_avg = np.mean(sorted_scores[-TOP_K:])
                spreads.append(top3_avg - bot3_avg)
            else:
                spreads.append(None)

        # spreads[0] = today, spreads[-1] = 5 days ago
        if spreads[0] is not None and spreads[-1] is not None and abs(spreads[-1]) > 1e-6:
            spread_change = (spreads[0] - spreads[-1]) / abs(spreads[-1])
            # Negative spread_change means narrowing
            spread_narrowing = -spread_change  # positive = narrowing
        else:
            spread_narrowing = 0.0
    else:
        spread_narrowing = 0.0

    signal_C = spread_narrowing > SPREAD_NARROW_PCT

    # ── Signal D: Dispersion drop (sudden convergence) ──
    # Compute daily cross-sector return dispersion, check for sudden drop
    if idx >= 63:
        sector_daily_rets = close[sector_cols].iloc[max(0, idx - 63):idx + 1].pct_change().dropna()
        daily_disp = sector_daily_rets.std(axis=1)

        if len(daily_disp) >= CRASH_LOOKBACK + 21:
            # Current dispersion (5d avg) vs historical (63d rolling mean/std)
            recent_disp = daily_disp.iloc[-CRASH_LOOKBACK:].mean()
            hist_disp = daily_disp.iloc[:-CRASH_LOOKBACK]
            hist_mean = hist_disp.mean()
            hist_std = hist_disp.std()

            if hist_std > 1e-10:
                # How much did dispersion drop relative to history?
                # Negative z-score = dispersion dropped (sectors converging)
                disp_change = recent_disp - hist_mean
                disp_drop_zscore = -disp_change / hist_std  # positive = drop
            else:
                disp_drop_zscore = 0.0
        else:
            disp_drop_zscore = 0.0
    else:
        disp_drop_zscore = 0.0

    signal_D = disp_drop_zscore > DISP_DROP_SIGMA

    return {
        "signal_B": signal_B,
        "signal_C": signal_C,
        "signal_D": signal_D,
        "cross_sector_corr": round(avg_corr, 4),
        "spread_narrowing": round(spread_narrowing, 4),
        "disp_drop_zscore": round(disp_drop_zscore, 4),
    }


def detect_momentum_crashes(close, rankings):
    """
    Identify actual momentum crashes in the data.

    Definition: Top-3 LGBM-ranked sectors underperform bottom-3 by >2% in next week.

    Returns list of crash dates and crash statistics.
    """
    sector_cols = [c for c in SECTORS if c in close.columns]
    crashes = []
    non_crashes = []

    for dt, scores in sorted(rankings.items()):
        if dt not in close.index:
            continue
        idx = close.index.get_loc(dt)
        fwd_idx = min(idx + 5, len(close) - 1)  # 1 week forward
        if fwd_idx <= idx:
            continue

        # Top 3 and bottom 3 by LGBM score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top3 = [t for t, _ in ranked[:TOP_K]]
        bot3 = [t for t, _ in ranked[-TOP_K:]]

        # Forward returns
        top3_rets = []
        bot3_rets = []
        for tk in top3:
            if tk in close.columns:
                r = float(close[tk].iloc[fwd_idx] / close[tk].iloc[idx] - 1)
                top3_rets.append(r)
        for tk in bot3:
            if tk in close.columns:
                r = float(close[tk].iloc[fwd_idx] / close[tk].iloc[idx] - 1)
                bot3_rets.append(r)

        if top3_rets and bot3_rets:
            top3_avg = np.mean(top3_rets)
            bot3_avg = np.mean(bot3_rets)
            spread = top3_avg - bot3_avg

            if spread < -0.02:  # Top 3 underperform bottom 3 by >2%
                crashes.append({
                    "date": dt,
                    "spread": round(spread, 4),
                    "top3_ret": round(top3_avg, 4),
                    "bot3_ret": round(bot3_avg, 4),
                })
            else:
                non_crashes.append(dt)

    return crashes, non_crashes


# ══════════════════════════════════════════════════════════════
# WALK-FORWARD LGBM RANKING
# ══════════════════════════════════════════════════════════════

def build_feature_records(close, high, low, rebal_dates, feature_cols, regime_series):
    """Build feature + target records for all sectors on all rebal dates (bull regime only)."""
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

            rankings[test_date] = dict(zip(test_df["ticker"], test_df["score"]))

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
# STRIKE COMPUTATION (2% OTM, V6 standard)
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
# TRADE SIMULATION WITH CRASH DETECTION
# ══════════════════════════════════════════════════════════════

def simulate_trades(name, rankings, close, high, low, atr_dict,
                    crash_mode="none", crash_signals_cache=None):
    """
    Simulate V6 bull call spread trades with optional crash detection filters.

    crash_mode:
      'none': Variant A — no crash detection (baseline)
      'B': Skip when cross-sector correlation > 0.8
      'C': Skip when top-bottom spread narrows >50% in 5d
      'D': Skip when dispersion drops >1 std in 5d
      'E': Skip when ANY of B/C/D triggers
      'F': Trade MORE aggressively when NO signals trigger

    HONEST RULES (V6):
      - Hold to expiry, intrinsic value only
      - 15% haircut on entry only
      - 2% OTM, 3% width, weekly rebalance
    """
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    equity = CAP
    trades = []
    skipped_by_crash = 0
    boosted_by_clear = 0

    for dt in sorted(rankings.keys()):
        if dt not in spy.index:
            continue

        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0
        scores = rankings[dt]
        if not scores:
            continue

        # Get crash signals (use cache if available)
        if crash_signals_cache and dt in crash_signals_cache:
            signals = crash_signals_cache[dt]
        else:
            signals = compute_crash_signals(close, dt)

        # Apply crash filter based on mode
        skip_trade = False
        size_multiplier = 1.0

        if crash_mode == "B":
            skip_trade = signals["signal_B"]
        elif crash_mode == "C":
            skip_trade = signals["signal_C"]
        elif crash_mode == "D":
            skip_trade = signals["signal_D"]
        elif crash_mode == "E":
            skip_trade = signals["signal_B"] or signals["signal_C"] or signals["signal_D"]
        elif crash_mode == "F":
            any_signal = signals["signal_B"] or signals["signal_C"] or signals["signal_D"]
            if any_signal:
                skip_trade = True  # Still skip on warnings
            else:
                size_multiplier = AGGRESSIVE_SIZE_MULT  # Trade bigger when clear

        if skip_trade:
            skipped_by_crash += 1
            continue

        # Pick top K sectors for bull call spreads
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        bull_picks = [t for t, _ in ranked[:TOP_K]]

        # Position sizing
        max_pos = min(200 * size_multiplier, equity / 3)
        if max_pos < 30:
            continue

        if size_multiplier > 1.0:
            boosted_by_clear += 1

        for tk in bull_picks:
            if tk not in close.columns or tk not in atr_dict:
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

            # Compute 2% OTM strikes
            K1, K2 = compute_strikes(S, "bull", OTM_PCT, SPREAD_PCT)

            try:
                entry_cost_ps, max_profit_ps = price_bull_call_spread(
                    S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=cv
                )
            except Exception:
                continue

            total_cost = entry_cost_ps * 100 + COMMISSION_RT_SPREAD

            if total_cost <= 0 or total_cost > max_pos or total_cost > equity * 0.40:
                continue

            # HOLD TO EXPIRY: compute intrinsic value
            Se = float(close[tk].iloc[ei])
            intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
            exit_value_ps = intrinsic

            pnl = (exit_value_ps - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD
            equity += pnl

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
                "crash_corr": signals.get("cross_sector_corr", 0),
                "crash_spread_narrow": signals.get("spread_narrowing", 0),
                "crash_disp_zscore": signals.get("disp_drop_zscore", 0),
            })

    return trades, equity, skipped_by_crash, boosted_by_clear


# ══════════════════════════════════════════════════════════════
# RANDOM BASELINE
# ══════════════════════════════════════════════════════════════

def random_baseline_test(rankings, close, high, low, atr_dict,
                         crash_mode, crash_signals_cache, n_trials=5):
    """Test if random sector selection also produces similar returns."""
    fprint(f"\n  Random baseline test ({n_trials} trials)...")
    random_sharpes = []

    for trial in range(n_trials):
        np.random.seed(42 + trial)
        rand_rankings = {}
        for dt, scores in rankings.items():
            rand_scores = {tk: np.random.random() for tk in scores.keys()}
            rand_rankings[dt] = rand_scores

        trades, final_eq, _, _ = simulate_trades(
            f"Random_{trial}", rand_rankings, close, high, low, atr_dict,
            crash_mode=crash_mode, crash_signals_cache=crash_signals_cache,
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
    fprint(f"MOMENTUM CRASH DETECTION v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 80)
    fprint(f"Capital: ${CAP:.0f} | DTE: {DTE} | OTM: {OTM_PCT:.0%} | Spread: {SPREAD_PCT:.0f}%")
    fprint(f"Haircut: {DEFAULT_HAIRCUT:.0%} entry only | Comm: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"HOLD TO EXPIRY | Intrinsic value only | Weekly rebalance")
    fprint(f"Features: {len(V6_FEATURES)} (V6 standard)")
    fprint(f"Crash detection thresholds:")
    fprint(f"  B: Cross-sector correlation > {CORR_THRESHOLD}")
    fprint(f"  C: Top-bottom spread narrows > {SPREAD_NARROW_PCT:.0%} in {CRASH_LOOKBACK}d")
    fprint(f"  D: Dispersion drops > {DISP_DROP_SIGMA:.1f} std in {CRASH_LOOKBACK}d")
    fprint(f"  E: Combined (any of B/C/D)")
    fprint(f"  F: Inverse — {AGGRESSIVE_SIZE_MULT:.0%} size when clear, skip when warning")
    fprint()

    # 1. Download data
    close, high, low = download_data()

    # 2. Load regime predictions
    regime_series = load_regime_predictions()

    # 3. Pre-compute ATR
    atr_dict = compute_atr_series(high, low, close)

    # 4. Build weekly rebalance dates
    rebal_dates = pd.DatetimeIndex(
        close.index.to_series().resample(WF_REBAL_FREQ).last().dropna().values
    )
    fprint(f"Rebalance dates: {len(rebal_dates)} "
           f"({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

    # 5. Build feature records and LGBM rankings (shared across all variants)
    fprint("\n" + "=" * 80)
    fprint("BUILDING LGBM RANKINGS (shared across all 6 variants)")
    fprint("=" * 80)

    records = build_feature_records(close, high, low, rebal_dates, V6_FEATURES, regime_series)
    rankings, imp_df = walk_forward_lgbm_rank(records, V6_FEATURES, "V6_baseline")

    if not rankings:
        fprint("ERROR: No rankings produced. Cannot continue.")
        return

    # 6. Pre-compute crash signals for all ranking dates (cache for efficiency)
    fprint("\n" + "=" * 80)
    fprint("PRE-COMPUTING CRASH SIGNALS")
    fprint("=" * 80)

    crash_signals_cache = {}
    for dt in sorted(rankings.keys()):
        crash_signals_cache[dt] = compute_crash_signals(close, dt)

    # Report signal statistics
    n_total = len(crash_signals_cache)
    n_B = sum(1 for s in crash_signals_cache.values() if s["signal_B"])
    n_C = sum(1 for s in crash_signals_cache.values() if s["signal_C"])
    n_D = sum(1 for s in crash_signals_cache.values() if s["signal_D"])
    n_any = sum(1 for s in crash_signals_cache.values()
                if s["signal_B"] or s["signal_C"] or s["signal_D"])
    n_none = n_total - n_any

    fprint(f"  Total ranking dates: {n_total}")
    fprint(f"  Signal B (high correlation): {n_B} ({n_B/n_total*100:.1f}%)")
    fprint(f"  Signal C (spread narrowing): {n_C} ({n_C/n_total*100:.1f}%)")
    fprint(f"  Signal D (dispersion drop):  {n_D} ({n_D/n_total*100:.1f}%)")
    fprint(f"  Any signal (E):              {n_any} ({n_any/n_total*100:.1f}%)")
    fprint(f"  Clear (F):                   {n_none} ({n_none/n_total*100:.1f}%)")

    # Average signal values
    avg_corr = np.mean([s["cross_sector_corr"] for s in crash_signals_cache.values()])
    avg_narrow = np.mean([s["spread_narrowing"] for s in crash_signals_cache.values()])
    avg_disp_z = np.mean([s["disp_drop_zscore"] for s in crash_signals_cache.values()])
    fprint(f"  Avg cross-sector correlation: {avg_corr:.3f}")
    fprint(f"  Avg spread narrowing: {avg_narrow:.3f}")
    fprint(f"  Avg dispersion z-score: {avg_disp_z:.3f}")

    # 7. Detect actual momentum crashes
    fprint("\n" + "=" * 80)
    fprint("MOMENTUM CRASH ANALYSIS")
    fprint("=" * 80)

    crashes, non_crashes = detect_momentum_crashes(close, rankings)
    fprint(f"  Total momentum crashes (top3 underperform bot3 by >2% in 1wk): {len(crashes)}")
    fprint(f"  Non-crash weeks: {len(non_crashes)}")
    if crashes:
        crash_spreads = [c["spread"] for c in crashes]
        fprint(f"  Worst crash spread: {min(crash_spreads):.4f}")
        fprint(f"  Avg crash spread: {np.mean(crash_spreads):.4f}")

        # Check signal predictive power: how often do signals precede crashes?
        crash_dates = set(c["date"] for c in crashes)
        for sig_name, sig_key in [("B (correlation)", "signal_B"),
                                  ("C (spread narrow)", "signal_C"),
                                  ("D (disp drop)", "signal_D")]:
            # True positives: signal fired AND crash happened
            tp = sum(1 for dt in crash_dates
                     if dt in crash_signals_cache and crash_signals_cache[dt][sig_key])
            # False negatives: no signal AND crash happened
            fn = len(crash_dates) - tp
            # Signal fired total
            sig_total = sum(1 for s in crash_signals_cache.values() if s[sig_key])
            # False positives: signal fired but no crash
            fp = sig_total - tp

            precision = tp / max(sig_total, 1)
            recall = tp / max(len(crash_dates), 1)

            fprint(f"  Signal {sig_name}: TP={tp}, FP={fp}, FN={fn}, "
                   f"Precision={precision:.2f}, Recall={recall:.2f}")

    # 8. Simulate all 6 variants
    fprint("\n" + "=" * 80)
    fprint("SIMULATING ALL 6 VARIANTS")
    fprint("=" * 80)

    spy_close = close["SPY"]
    all_results = {}

    variant_configs = [
        ("A_baseline",       "none", "V6 baseline (no crash detection)"),
        ("B_corr_filter",    "B",    "Skip when cross-sector corr > 0.8"),
        ("C_spread_filter",  "C",    "Skip when top-bot spread narrows > 50%/5d"),
        ("D_disp_filter",    "D",    "Skip when dispersion drops > 1std/5d"),
        ("E_combined",       "E",    "Skip if ANY of B/C/D triggers"),
        ("F_inverse",        "F",    "Trade bigger when clear, skip on warning"),
    ]

    for vname, crash_mode, desc in variant_configs:
        fprint(f"\n--- {vname}: {desc} ---")

        trades, final_eq, n_skipped, n_boosted = simulate_trades(
            vname, rankings, close, high, low, atr_dict,
            crash_mode=crash_mode, crash_signals_cache=crash_signals_cache,
        )

        fprint(f"  Trades skipped by crash filter: {n_skipped}")
        if n_boosted > 0:
            fprint(f"  Dates with boosted sizing: {n_boosted}")

        if not trades or len(trades) < 10:
            fprint(f"  Only {len(trades) if trades else 0} trades, skipping validation")
            all_results[vname] = {"description": desc, "n_trades": len(trades) if trades else 0,
                                  "skipped": n_skipped, "too_few": True}
            continue

        # 5-gate adversarial validation
        result = validate_trades(
            trades, initial_capital=CAP,
            spy_prices=spy_close,
            strategy_name=vname,
        )
        result.print_summary()

        # Random baseline
        random_sharpes = random_baseline_test(
            rankings, close, high, low, atr_dict,
            crash_mode=crash_mode, crash_signals_cache=crash_signals_cache,
        )
        mean_random = np.mean(random_sharpes) if random_sharpes else 0

        fprint(f"  ML Sharpe: {result.sharpe:.2f} vs Random mean: {mean_random:.2f}")
        if result.sharpe > 0 and mean_random > 0:
            fprint(f"  ML alpha ratio: {result.sharpe / mean_random:.2f}x")

        all_results[vname] = {
            "description": desc,
            **result.to_dict(),
            "skipped_by_crash": n_skipped,
            "boosted_by_clear": n_boosted,
            "random_sharpes": [round(s, 3) for s in random_sharpes],
            "random_mean_sharpe": round(mean_random, 3),
        }

    # ── SUMMARY COMPARISON ──
    fprint("\n" + "=" * 80)
    fprint("SUMMARY COMPARISON — ALL 6 VARIANTS")
    fprint("=" * 80)
    fprint(f"{'Variant':<22} {'Trades':>6} {'Skip':>5} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} "
           f"{'PF':>6} {'MaxDD':>7} {'Gates':>6} {'Final$':>8} {'RandSh':>7}")
    fprint("-" * 100)

    for vname, _, _ in variant_configs:
        r = all_results.get(vname)
        if not r or r.get("too_few", False):
            fprint(f"  {vname:<22} — NO DATA —")
            continue
        fprint(f"  {vname:<22} {r['n_trades']:>5} {r.get('skipped_by_crash',0):>5} "
               f"{r['sharpe']:>7.2f} {r['sortino']:>8.2f} "
               f"{r['win_rate']*100:>5.1f}% {r['profit_factor']:>5.2f} "
               f"{r['max_dd']*100:>6.1f}% {r['gates_passed']}/{r['gates_total']} "
               f"${r['final_equity']:>7,.0f} {r['random_mean_sharpe']:>7.2f}")

    # ── KEY FINDINGS ──
    fprint("\n" + "=" * 80)
    fprint("KEY FINDINGS")
    fprint("=" * 80)

    baseline = all_results.get("A_baseline", {})
    baseline_sharpe = baseline.get("sharpe", 0)

    for vname, _, desc in variant_configs[1:]:  # Skip baseline
        r = all_results.get(vname, {})
        if r.get("too_few", False):
            fprint(f"  {vname}: Insufficient trades")
            continue
        v_sharpe = r.get("sharpe", 0)
        delta = v_sharpe - baseline_sharpe
        direction = "BETTER" if delta > 0 else "WORSE" if delta < 0 else "SAME"
        fprint(f"  {vname}: Sharpe {v_sharpe:.2f} ({delta:+.2f} vs baseline) — {direction}")
        fprint(f"    {desc}")
        fprint(f"    Skipped {r.get('skipped_by_crash', 0)} dates, "
               f"Gates {r.get('gates_passed', 0)}/{r.get('gates_total', 0)}")

    # Best variant
    valid_results = {k: v for k, v in all_results.items()
                     if not v.get("too_few", False) and "sharpe" in v}
    if valid_results:
        best_name = max(valid_results, key=lambda k: valid_results[k]["sharpe"])
        best_sharpe = valid_results[best_name]["sharpe"]
        fprint(f"\n  BEST VARIANT: {best_name} (Sharpe {best_sharpe:.2f})")

        if best_name == "A_baseline":
            fprint("  VERDICT: Crash detection did NOT improve V6 timing.")
            fprint("  The existing regime filter is sufficient.")
        else:
            fprint(f"  VERDICT: Crash detection IMPROVES V6 by "
                   f"{best_sharpe - baseline_sharpe:+.2f} Sharpe.")

    # Feature importance
    if imp_df is not None:
        fprint("\n" + "=" * 80)
        fprint("FEATURE IMPORTANCE (V6, 17 features)")
        fprint("=" * 80)
        for _, row in imp_df.iterrows():
            bar = "*" * int(row["importance"] / imp_df["importance"].max() * 30)
            fprint(f"  {row['feature']:<30} {row['importance']:>6.1f} {bar}")

    # Save results
    results_path = OUTPUT_DIR / "momentum_crash_v1_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    # Save crash dates
    crash_path = OUTPUT_DIR / "momentum_crashes.json"
    with open(crash_path, "w") as f:
        json.dump([{**c, "date": str(c["date"])} for c in crashes], f, indent=2)
    fprint(f"Crash dates saved to {crash_path}")

    # MLflow logging
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"mom_crash_v1_{t0.strftime('%Y%m%d_%H%M')}"):
                for vname, r in all_results.items():
                    if r.get("too_few", False):
                        continue
                    prefix = vname.split("_")[0]
                    mlflow.log_metric(f"{prefix}_sharpe", r.get("sharpe", 0))
                    mlflow.log_metric(f"{prefix}_sortino", r.get("sortino", 0))
                    mlflow.log_metric(f"{prefix}_win_rate", r.get("win_rate", 0))
                    mlflow.log_metric(f"{prefix}_profit_factor", r.get("profit_factor", 0))
                    mlflow.log_metric(f"{prefix}_max_dd", r.get("max_dd", 0))
                    mlflow.log_metric(f"{prefix}_n_trades", r.get("n_trades", 0))
                    mlflow.log_metric(f"{prefix}_gates_passed", r.get("gates_passed", 0))
                    mlflow.log_metric(f"{prefix}_final_equity", r.get("final_equity", 0))
                    mlflow.log_metric(f"{prefix}_skipped", r.get("skipped_by_crash", 0))

                mlflow.log_params({
                    "capital": CAP,
                    "dte": DTE,
                    "otm_pct": OTM_PCT,
                    "spread_pct": SPREAD_PCT,
                    "n_features": len(V6_FEATURES),
                    "rebal_freq": WF_REBAL_FREQ,
                    "corr_threshold": CORR_THRESHOLD,
                    "spread_narrow_pct": SPREAD_NARROW_PCT,
                    "disp_drop_sigma": DISP_DROP_SIGMA,
                    "crash_lookback": CRASH_LOOKBACK,
                    "aggressive_mult": AGGRESSIVE_SIZE_MULT,
                    "n_crashes": len(crashes),
                    "n_ranking_dates": len(rankings),
                })

                mlflow.log_artifact(str(results_path))
                mlflow.log_artifact(str(crash_path))
            fprint(f"MLflow run logged to experiment '{EXPERIMENT_NAME}'")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    fprint("DONE")


if __name__ == "__main__":
    main()
