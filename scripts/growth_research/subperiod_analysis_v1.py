#!/usr/bin/env python3
"""
Sub-Period Analysis v1 — V6 Strategy Temporal Stability Test
=============================================================

Tests V6 strategy (weekly, 2% OTM, bull+pairs) across 7 time periods to answer:
  IS THE STRATEGY GETTING BETTER OR WORSE OVER TIME?

Sub-periods:
  A) Full period     — 2008-2026. Baseline.
  B) 2008-2012       — Financial crisis recovery.
  C) 2013-2017       — Bull market, low VIX.
  D) 2018-2022       — Mixed: vol spike 2018, COVID 2020, rate hikes 2022.
  E) 2023-2026       — Most recent (live-trading relevant).
  F) Pre-COVID       — 2008-2019.
  G) Post-COVID      — 2020-2026.

CRITICAL DESIGN:
  - LGBM trains on ALL available data (walk-forward rolling windows).
    Early periods naturally have less training data — this is realistic.
  - ONLY the TRADE SIMULATION is restricted to the sub-period.
  - No 5-gate validation per sub-period (too few trades). Just raw metrics.

V6 config:
  - Weekly rebalance (W-FRI)
  - 2% OTM moneyness
  - Bull + pairs (VIX<20: bull+bear; VIX>=20: bull only)
  - 21 features (18 legacy + 3 cross-asset)
  - DTE=21, $645 capital, 3% spread width, hold to expiry, intrinsic only

For each period reports:
  Sharpe, Sortino, WR, PF, MaxDD, CAGR, trade count,
  VIX breakdown (% trades in VIX>20 vs VIX<20),
  Bull vs Bear PnL split.

Logs to MLflow 'subperiod_analysis_v1'.
Saves to output/growth_research/subperiod_analysis_v1/.
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


# -- Standardized tools --
sys.path.insert(0, "/home/jupiter/Lvl3Quant")
from research.tools.options_pricer import (
    price_bull_call_spread,
    price_bear_put_spread,
    estimate_iv,
    compute_atr,
    COMMISSION_RT_SPREAD,
    DEFAULT_HAIRCUT,
)

# -- Config --
BASE = Path("/home/jupiter/Lvl3Quant")
OUTPUT_DIR = BASE / "output" / "growth_research" / "subperiod_analysis_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
DTE = 21
SPREAD_PCT = 3.0  # 3% width for all spreads
TOP_K = 3
REGIME_BULL_THRESHOLD = 0.4
REGIME_BEAR_THRESHOLD = 0.2

# V6 config: weekly, 2% OTM, bull+pairs
V6_REBAL_FREQ = "W-FRI"
V6_OTM_PCT = 0.02
V6_PAIRS = True
V6_MAX_POS_BULL = 200
V6_MAX_POS_PAIR_LEG = 100

# Regime predictions path
REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"

# LGBM walk-forward
WF_TRAIN_PERIODS = 12

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "subperiod_analysis_v1"

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen(MLFLOW_URI, timeout=3)
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    MLFLOW_OK = True
    fprint(f"MLflow connected: {MLFLOW_URI}")
except Exception:
    fprint("MLflow unavailable -- results saved to disk only")


# Sub-period definitions: (name, start_date, end_date, description)
SUBPERIODS = [
    ("A_full",       "2008-01-01", "2026-12-31", "Full period 2008-2026"),
    ("B_2008_2012",  "2008-01-01", "2012-12-31", "Financial crisis recovery"),
    ("C_2013_2017",  "2013-01-01", "2017-12-31", "Bull market, low VIX"),
    ("D_2018_2022",  "2018-01-01", "2022-12-31", "Mixed: vol spike, COVID, rate hikes"),
    ("E_2023_2026",  "2023-01-01", "2026-12-31", "Most recent (live-trading relevant)"),
    ("F_pre_covid",  "2008-01-01", "2019-12-31", "Pre-COVID"),
    ("G_post_covid", "2020-01-01", "2026-12-31", "Post-COVID"),
]


# ================================================================
# DATA DOWNLOAD
# ================================================================

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


# ================================================================
# REGIME LOADING
# ================================================================

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


# ================================================================
# FEATURE ENGINEERING
# ================================================================

# Legacy 18 features (quality-momentum)
LEGACY_FEATURES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_21d", "vol_63d", "sharpe_63d", "maxdd_63d", "pct_52w_high", "mom_accel",
    "pct_pos_months_12m", "sortino_63d", "calmar_1y", "up_capture",
    "trend_r2_63d", "trend_slope_63d",
]

# 3 validated cross-asset features
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


# ================================================================
# WALK-FORWARD LGBM RANKING
# ================================================================

def build_feature_records(close, high, low, rebal_dates, feature_cols, regime_series):
    """
    Build feature + target records for all sectors on all rebal dates.
    Uses bull_only regime mode (regime>0.4) -- same as V6 production.
    Bear direction is handled at trade time via VIX-based pair logic, NOT regime.
    """
    fprint(f"  Building records: {len(rebal_dates)} dates, {len(feature_cols)} features")

    records = []
    sector_cols = [c for c in SECTORS if c in close.columns]
    spy = close["SPY"]

    for dt in rebal_dates:
        idx = close.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
            continue

        # Regime filter: only trade when GRU says bull (>0.4)
        rscore = get_regime_score_at(regime_series, dt)
        if rscore <= REGIME_BULL_THRESHOLD:
            continue

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


def walk_forward_lgbm_rank(df, feature_cols):
    """Walk-forward LGBM ranking: sliding train, predict next period."""
    import lightgbm as lgb

    if len(df) < 100:
        fprint(f"    Insufficient data ({len(df)} records)")
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

    fprint(f"    {len(rankings)} ranking dates, {n_models} models trained")
    return rankings, imp_df


# ================================================================
# ATR COMPUTATION
# ================================================================

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


# ================================================================
# STRIKE COMPUTATION (2% OTM)
# ================================================================

def compute_strikes(S, direction, otm_pct, spread_pct):
    """
    Compute strike prices for a spread.

    OTM (otm_pct>0, e.g. 0.02 for 2%):
      Bull call: K1=S*(1+otm_pct), K2=K1*(1+spread_pct/100)
      Bear put:  K2=S*(1-otm_pct), K1=K2*(1-spread_pct/100)

    ATM (otm_pct=0):
      Bull call: K1=S, K2=S*(1+spread_pct/100)
      Bear put:  K1=S*(1-spread_pct/100), K2=S

    Returns (K1, K2) where K1 < K2 always.
    """
    if direction == "bull":
        if otm_pct > 0:
            K1 = round(S * (1 + otm_pct), 2)
            K2 = round(K1 * (1 + spread_pct / 100), 2)
        else:
            K1 = round(S, 2)
            K2 = round(S * (1 + spread_pct / 100), 2)
    else:  # bear
        if otm_pct > 0:
            K2 = round(S * (1 - otm_pct), 2)
            K1 = round(K2 * (1 - spread_pct / 100), 2)
        else:
            K1 = round(S * (1 - spread_pct / 100), 2)
            K2 = round(S, 2)

    if K2 <= K1:
        K2 = K1 + 0.50
    return K1, K2


# ================================================================
# TRADE SIMULATION — V6 (weekly, 2% OTM, bull+pairs)
# ================================================================

def _execute_single_trade(tk, dt, direction, otm_pct, max_pos, close, atr_dict, cv, equity):
    """
    Execute a single spread trade. Returns PnL or None if trade could not be entered.

    Uses honest pricing: BS with ATR-based IV, 15% entry haircut,
    hold to expiry, intrinsic value only.
    """
    if tk not in close.columns or tk not in atr_dict:
        return None

    S = float(close[tk].loc[dt])
    di = close.index.get_loc(dt)
    ei = min(di + DTE, len(close) - 1)
    if ei <= di:
        return None

    # ATR for pricing
    if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]):
        av = float(atr_dict[tk].loc[dt])
    else:
        av = S * 0.015

    # Compute strikes
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

    # HOLD TO EXPIRY: compute intrinsic value at expiry
    Se = float(close[tk].iloc[ei])

    if direction == "bull":
        intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
    else:
        intrinsic = max(K2 - Se, 0.0) - max(K1 - Se, 0.0)

    exit_value_ps = intrinsic

    # PnL: exit value - entry cost - commission (no exit haircut at expiry)
    pnl = (exit_value_ps - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD
    return pnl


def simulate_trades_v6(rankings, close, high, low, atr_dict):
    """
    Simulate V6 trades: weekly, 2% OTM, bull+pairs.

    Pairs mode: VIX < 20 -> bull + bear; VIX >= 20 -> bull only.
    """
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

        # Pairs mode: VIX < 20 -> bull + bear; VIX >= 20 -> bull only
        if V6_PAIRS and cv < 20.0:
            trade_mode = "pairs"
        else:
            trade_mode = "bull_only"

        # Pick sectors: top K for bull, bottom K for bear
        ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_asc = sorted(scores.items(), key=lambda x: x[1])

        bull_picks = [t for t, _ in ranked_desc[:TOP_K]]
        bear_picks = [t for t, _ in ranked_asc[:TOP_K]] if trade_mode == "pairs" else []

        # Position sizing
        if trade_mode == "pairs":
            max_pos = min(V6_MAX_POS_PAIR_LEG, equity / 6)
        else:
            max_pos = min(V6_MAX_POS_BULL, equity / 3)

        if max_pos < 30:
            continue

        # Execute bull leg
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

        # Execute bear leg (pairs mode only)
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


# ================================================================
# REBALANCE DATE GENERATION
# ================================================================

def generate_rebal_dates(close, freq_str):
    """Generate rebalance dates from close index based on frequency string."""
    rebal_dates = pd.DatetimeIndex(
        close.index.to_series().resample(freq_str).last().dropna().values
    )
    return rebal_dates


# ================================================================
# METRICS COMPUTATION (no 5-gate, just raw metrics)
# ================================================================

def compute_metrics(trades, label=""):
    """
    Compute raw performance metrics from a list of trade dicts.
    Does NOT run 5-gate validation (too few trades per sub-period).

    Returns dict with: sharpe, sortino, wr, pf, max_dd, cagr, n_trades,
    vix_breakdown, bull_bear_split.
    """
    if not trades or len(trades) < 5:
        return {
            "label": label,
            "n_trades": len(trades) if trades else 0,
            "sharpe": 0.0, "sortino": 0.0, "win_rate": 0.0,
            "profit_factor": 0.0, "max_dd": 0.0, "cagr": 0.0,
            "final_equity": CAP, "total_pnl": 0.0,
            "vix_high_pct": 0.0, "vix_low_pct": 0.0,
            "vix_high_trades": 0, "vix_low_trades": 0,
            "bull_pnl": 0.0, "bear_pnl": 0.0,
            "bull_trades": 0, "bear_trades": 0,
            "bull_wr": 0.0, "bear_wr": 0.0,
            "insufficient": True,
        }

    pnls = [t["pnl"] for t in trades]
    total_pnl = sum(pnls)
    n = len(pnls)

    # Equity curve for Sharpe/Sortino/MaxDD
    equity_curve = [CAP]
    for p in pnls:
        equity_curve.append(equity_curve[-1] + p)
    eq = np.array(equity_curve)
    final_eq = eq[-1]

    # Equity-based returns (pct_change)
    eq_returns = np.diff(eq) / eq[:-1]
    eq_returns = eq_returns[np.isfinite(eq_returns)]

    if len(eq_returns) < 2:
        sharpe = 0.0
        sortino = 0.0
    else:
        # Annualize: assume ~52 trades/year for weekly, scale by actual
        mean_r = np.mean(eq_returns)
        std_r = np.std(eq_returns, ddof=1)
        sharpe = float(mean_r / (std_r + 1e-10) * np.sqrt(52))

        downside = eq_returns[eq_returns < 0]
        if len(downside) > 1:
            sortino = float(mean_r / (np.std(downside, ddof=1) + 1e-10) * np.sqrt(52))
        else:
            sortino = sharpe * 1.5  # all positive, approximate

    # Win rate
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / n

    # Profit factor
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    pf = gross_profit / (gross_loss + 1e-10)

    # Max drawdown
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / (peak + 1e-10)
    max_dd = float(dd.min())

    # CAGR: approximate from first/last trade dates
    first_date = pd.Timestamp(trades[0]["entry_date"])
    last_date = pd.Timestamp(trades[-1]["exit_date"])
    years = max((last_date - first_date).days / 365.25, 0.5)
    if final_eq > 0 and CAP > 0:
        cagr = float((final_eq / CAP) ** (1 / years) - 1)
    else:
        cagr = -1.0

    # VIX breakdown
    vix_high = [t for t in trades if t["vix"] >= 20.0]
    vix_low = [t for t in trades if t["vix"] < 20.0]
    vix_high_pct = len(vix_high) / n * 100
    vix_low_pct = len(vix_low) / n * 100

    # Bull vs bear PnL split
    bull_trades = [t for t in trades if t["direction"] == "bull"]
    bear_trades = [t for t in trades if t["direction"] == "bear"]
    bull_pnl = sum(t["pnl"] for t in bull_trades)
    bear_pnl = sum(t["pnl"] for t in bear_trades)
    bull_wr = sum(1 for t in bull_trades if t["win"]) / max(len(bull_trades), 1)
    bear_wr = sum(1 for t in bear_trades if t["win"]) / max(len(bear_trades), 1)

    return {
        "label": label,
        "n_trades": n,
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "win_rate": round(wr, 4),
        "profit_factor": round(pf, 3),
        "max_dd": round(max_dd, 4),
        "cagr": round(cagr, 4),
        "final_equity": round(final_eq, 2),
        "total_pnl": round(total_pnl, 2),
        "vix_high_pct": round(vix_high_pct, 1),
        "vix_low_pct": round(vix_low_pct, 1),
        "vix_high_trades": len(vix_high),
        "vix_low_trades": len(vix_low),
        "vix_high_pnl": round(sum(t["pnl"] for t in vix_high), 2),
        "vix_low_pnl": round(sum(t["pnl"] for t in vix_low), 2),
        "bull_pnl": round(bull_pnl, 2),
        "bear_pnl": round(bear_pnl, 2),
        "bull_trades": len(bull_trades),
        "bear_trades": len(bear_trades),
        "bull_wr": round(bull_wr, 4),
        "bear_wr": round(bear_wr, 4),
        "insufficient": False,
    }


# ================================================================
# MAIN
# ================================================================

def main():
    t0 = datetime.now()
    fprint("=" * 90)
    fprint(f"SUB-PERIOD ANALYSIS V1 — V6 Temporal Stability Test")
    fprint(f"Started: {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 90)
    fprint(f"V6 Config: Weekly rebalance, 2% OTM, bull+pairs, 21 features")
    fprint(f"Capital: ${CAP:.0f} | DTE: {DTE} | Spread: {SPREAD_PCT:.0f}% | "
           f"Haircut: {DEFAULT_HAIRCUT:.0%} entry only | Comm: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"HOLD TO EXPIRY | Intrinsic value only at expiry | No exit haircut")
    fprint(f"Regime bull threshold: >{REGIME_BULL_THRESHOLD}")
    fprint()
    fprint(f"Testing {len(SUBPERIODS)} sub-periods:")
    for name, start, end, desc in SUBPERIODS:
        fprint(f"  {name}: {start} to {end} -- {desc}")
    fprint()

    # 1. Download data
    close, high, low = download_data()

    # 2. Load regime predictions
    regime_series = load_regime_predictions()

    # 3. Pre-compute ATR
    atr_dict = compute_atr_series(high, low, close)

    # 4. Build rebalance dates for V6 weekly frequency
    rebal_dates = generate_rebal_dates(close, V6_REBAL_FREQ)
    fprint(f"\nRebalance dates (weekly): {len(rebal_dates)} "
           f"({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

    # 5. Build LGBM rankings on ALL data (not restricted to sub-period)
    fprint(f"\n{'=' * 90}")
    fprint("BUILDING LGBM RANKINGS ON ALL DATA (walk-forward)")
    fprint(f"{'=' * 90}")

    records = build_feature_records(
        close, high, low, rebal_dates, V4_FEATURES, regime_series
    )
    rankings, imp_df = walk_forward_lgbm_rank(records, V4_FEATURES)

    if not rankings:
        fprint("ERROR: No rankings produced. Cannot continue.")
        return

    fprint(f"\nTotal ranking dates: {len(rankings)}")
    all_ranking_dates = sorted(rankings.keys())
    fprint(f"  First: {all_ranking_dates[0].date()}, Last: {all_ranking_dates[-1].date()}")

    # 6. Simulate trades on ALL data first (full period)
    fprint(f"\n{'=' * 90}")
    fprint("SIMULATING TRADES ON ALL DATA")
    fprint(f"{'=' * 90}")

    all_trades, full_final_eq = simulate_trades_v6(
        rankings, close, high, low, atr_dict
    )
    fprint(f"Total trades generated: {len(all_trades)}")
    fprint(f"Full-period final equity: ${full_final_eq:,.2f}")

    # 7. Split trades into sub-periods and compute metrics
    fprint(f"\n{'=' * 90}")
    fprint("SUB-PERIOD RESULTS")
    fprint(f"{'=' * 90}")

    all_results = {}

    for sp_name, sp_start, sp_end, sp_desc in SUBPERIODS:
        fprint(f"\n{'~' * 80}")
        fprint(f"  {sp_name}: {sp_desc} ({sp_start} to {sp_end})")
        fprint(f"{'~' * 80}")

        sp_start_dt = pd.Timestamp(sp_start)
        sp_end_dt = pd.Timestamp(sp_end)

        # Filter trades to sub-period by entry_date
        sp_trades = [
            t for t in all_trades
            if sp_start_dt <= pd.Timestamp(t["entry_date"]) <= sp_end_dt
        ]

        fprint(f"  Trades in period: {len(sp_trades)}")

        metrics = compute_metrics(sp_trades, label=sp_name)
        metrics["description"] = sp_desc
        metrics["start_date"] = sp_start
        metrics["end_date"] = sp_end

        if metrics.get("insufficient"):
            fprint(f"  INSUFFICIENT TRADES (<5) -- skipping metrics")
        else:
            fprint(f"  Sharpe: {metrics['sharpe']:.2f}  |  Sortino: {metrics['sortino']:.2f}  |  "
                   f"WR: {metrics['win_rate']*100:.1f}%  |  PF: {metrics['profit_factor']:.2f}")
            fprint(f"  CAGR: {metrics['cagr']*100:.1f}%  |  MaxDD: {metrics['max_dd']*100:.1f}%  |  "
                   f"Final: ${metrics['final_equity']:,.0f}  |  Total PnL: ${metrics['total_pnl']:,.0f}")
            fprint(f"  VIX breakdown: VIX>=20: {metrics['vix_high_pct']:.0f}% ({metrics['vix_high_trades']} trades, "
                   f"PnL ${metrics['vix_high_pnl']:,.0f}) | VIX<20: {metrics['vix_low_pct']:.0f}% "
                   f"({metrics['vix_low_trades']} trades, PnL ${metrics['vix_low_pnl']:,.0f})")
            fprint(f"  Bull: {metrics['bull_trades']} trades, WR {metrics['bull_wr']*100:.1f}%, "
                   f"PnL ${metrics['bull_pnl']:,.0f}  |  Bear: {metrics['bear_trades']} trades, "
                   f"WR {metrics['bear_wr']*100:.1f}%, PnL ${metrics['bear_pnl']:,.0f}")

        all_results[sp_name] = metrics

    # ---- SUMMARY TABLE ----
    fprint(f"\n{'=' * 130}")
    fprint("SUMMARY TABLE -- V6 Sub-Period Performance")
    fprint(f"{'=' * 130}")
    fprint(f"{'Period':<18} {'Desc':<35} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} "
           f"{'WR':>6} {'PF':>6} {'MaxDD':>7} {'CAGR':>7} {'PnL$':>8} "
           f"{'VIX>20%':>7} {'Bull$':>8} {'Bear$':>8}")
    fprint("-" * 130)

    for sp_name, _, _, _ in SUBPERIODS:
        r = all_results.get(sp_name, {})
        if r.get("insufficient"):
            fprint(f"  {sp_name:<18} {r.get('description',''):<35} {r['n_trades']:>5}  "
                   f"--- INSUFFICIENT TRADES ---")
            continue
        fprint(f"  {sp_name:<18} {r.get('description',''):<35} {r['n_trades']:>5} "
               f"{r['sharpe']:>7.2f} {r['sortino']:>8.2f} "
               f"{r['win_rate']*100:>5.1f}% {r['profit_factor']:>5.2f} "
               f"{r['max_dd']*100:>6.1f}% {r['cagr']*100:>6.1f}% "
               f"${r['total_pnl']:>7,.0f} {r['vix_high_pct']:>6.0f}% "
               f"${r['bull_pnl']:>7,.0f} ${r['bear_pnl']:>7,.0f}")

    # ---- KEY QUESTION: Is performance improving or degrading? ----
    fprint(f"\n{'=' * 90}")
    fprint("KEY QUESTION: Is V6 strategy getting BETTER or WORSE over time?")
    fprint(f"{'=' * 90}")

    # Compare chronological sub-periods
    chronological = ["B_2008_2012", "C_2013_2017", "D_2018_2022", "E_2023_2026"]
    fprint("\nChronological Sharpe trend:")
    prev_sharpe = None
    for sp in chronological:
        r = all_results.get(sp, {})
        if r.get("insufficient"):
            fprint(f"  {sp}: INSUFFICIENT DATA")
            continue
        arrow = ""
        if prev_sharpe is not None:
            delta = r["sharpe"] - prev_sharpe
            arrow = f" ({'UP' if delta > 0 else 'DOWN'} {abs(delta):.2f})"
        fprint(f"  {sp}: Sharpe {r['sharpe']:.2f}, WR {r['win_rate']*100:.1f}%, "
               f"PF {r['profit_factor']:.2f}{arrow}")
        prev_sharpe = r["sharpe"]

    # Pre vs post COVID
    fprint("\nPre-COVID vs Post-COVID:")
    pre = all_results.get("F_pre_covid", {})
    post = all_results.get("G_post_covid", {})
    if not pre.get("insufficient") and not post.get("insufficient"):
        fprint(f"  Pre-COVID:  Sharpe {pre['sharpe']:.2f}, WR {pre['win_rate']*100:.1f}%, "
               f"CAGR {pre['cagr']*100:.1f}%, {pre['n_trades']} trades")
        fprint(f"  Post-COVID: Sharpe {post['sharpe']:.2f}, WR {post['win_rate']*100:.1f}%, "
               f"CAGR {post['cagr']*100:.1f}%, {post['n_trades']} trades")
        if post["sharpe"] > pre["sharpe"]:
            fprint(f"  --> IMPROVING: Post-COVID Sharpe is {post['sharpe'] - pre['sharpe']:.2f} higher")
        else:
            fprint(f"  --> DEGRADING: Post-COVID Sharpe is {pre['sharpe'] - post['sharpe']:.2f} lower")

    # Most recent period assessment
    fprint("\nMost recent period (2023-2026) -- live-trading relevance:")
    recent = all_results.get("E_2023_2026", {})
    full = all_results.get("A_full", {})
    if not recent.get("insufficient") and not full.get("insufficient"):
        fprint(f"  Recent Sharpe: {recent['sharpe']:.2f} vs Full-period: {full['sharpe']:.2f}")
        if recent["sharpe"] >= full["sharpe"] * 0.8:
            fprint(f"  --> VIABLE for live trading (recent >= 80% of full-period Sharpe)")
        else:
            fprint(f"  --> CAUTION: Recent performance significantly below full-period")
        fprint(f"  Recent WR: {recent['win_rate']*100:.1f}%, PF: {recent['profit_factor']:.2f}, "
               f"CAGR: {recent['cagr']*100:.1f}%")

    # Feature importance
    if imp_df is not None:
        fprint(f"\n{'=' * 90}")
        fprint("FEATURE IMPORTANCE (Top 10)")
        fprint(f"{'=' * 90}")
        for _, row in imp_df.head(10).iterrows():
            bar = "*" * int(row["importance"] / imp_df["importance"].max() * 30)
            fprint(f"  {row['feature']:<30} {row['importance']:>6.1f} {bar}")

    # Save results
    results_path = OUTPUT_DIR / "subperiod_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    # Save trades for later analysis
    trades_path = OUTPUT_DIR / "all_trades.json"
    with open(trades_path, "w") as f:
        json.dump(all_trades, f, indent=2, default=str)
    fprint(f"All trades saved to {trades_path}")

    # MLflow logging
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"subperiod_v1_{t0.strftime('%Y%m%d_%H%M')}"):
                # Log metrics per sub-period
                for sp_name, r in all_results.items():
                    if r.get("insufficient"):
                        continue
                    prefix = sp_name
                    mlflow.log_metric(f"{prefix}_sharpe", r["sharpe"])
                    mlflow.log_metric(f"{prefix}_sortino", r["sortino"])
                    mlflow.log_metric(f"{prefix}_win_rate", r["win_rate"])
                    mlflow.log_metric(f"{prefix}_profit_factor", r["profit_factor"])
                    mlflow.log_metric(f"{prefix}_max_dd", r["max_dd"])
                    mlflow.log_metric(f"{prefix}_cagr", r["cagr"])
                    mlflow.log_metric(f"{prefix}_n_trades", r["n_trades"])
                    mlflow.log_metric(f"{prefix}_total_pnl", r["total_pnl"])
                    mlflow.log_metric(f"{prefix}_final_equity", r["final_equity"])
                    mlflow.log_metric(f"{prefix}_vix_high_pct", r["vix_high_pct"])
                    mlflow.log_metric(f"{prefix}_bull_pnl", r["bull_pnl"])
                    mlflow.log_metric(f"{prefix}_bear_pnl", r["bear_pnl"])

                mlflow.log_params({
                    "strategy": "V6",
                    "rebal_freq": V6_REBAL_FREQ,
                    "otm_pct": V6_OTM_PCT,
                    "pairs": V6_PAIRS,
                    "capital": CAP,
                    "dte": DTE,
                    "spread_pct": SPREAD_PCT,
                    "haircut": DEFAULT_HAIRCUT,
                    "regime_bull_thresh": REGIME_BULL_THRESHOLD,
                    "n_features": len(V4_FEATURES),
                    "wf_train_periods": WF_TRAIN_PERIODS,
                    "n_subperiods": len(SUBPERIODS),
                    "total_trades": len(all_trades),
                })

                mlflow.log_artifact(str(results_path))
                mlflow.log_artifact(str(trades_path))
            fprint(f"MLflow run logged to experiment '{EXPERIMENT_NAME}'")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    fprint("DONE")


if __name__ == "__main__":
    main()
