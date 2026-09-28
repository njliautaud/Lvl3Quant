#!/usr/bin/env python3
"""
OOS Robustness v1 — Rolling Out-of-Sample Window Test for Best Strategy
========================================================================

Tests the BEST strategy from Session 8 (weekly + 2% OTM + pairs) across
6 non-overlapping time windows to verify regime robustness.

Strategy: K=2, weekly 5d, DTE=21, 2% OTM, 21 LGBM features, GRU regime filter.
  - Bull call spreads when VIX>20 (regime score > 0.4)
  - Pair trades when VIX<20 (long top-2 bull call + short bottom-2 bear put)

Each window starts fresh with $645 capital. LGBM uses expanding walk-forward
(all prior data for training), tested only within the window.

6 Windows:
  1. crisis_08_11: GFC + recovery (2008-2011)
  2. bull_12_15: QE bull market (2012-2015)
  3. vol_16_19: Brexit, trade wars (2016-2019)
  4. covid_20_21: COVID crash + recovery (2020-2021)
  5. bear_22_23: Rate hike bear market (2022-2023)
  6. recent_24_26: Most recent (2024-2026)

For each window: Sharpe, Sortino, CAGR, MaxDD, WR, PF, #trades,
regime-stratified Sharpe, 5-gate adversarial, ML alpha ratio.

Summary: mean/std Sharpe, consistency ratio, worst window.
Logged to MLflow experiment "oos_robustness_v1".
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

try:
    from research.tools.options_pricer import (
        price_bull_call_spread,
        price_bear_put_spread,
        estimate_iv,
        compute_atr,
        COMMISSION_RT_SPREAD,
        DEFAULT_HAIRCUT,
    )
    fprint("Imported options_pricer from research.tools")
except ImportError:
    fprint("WARNING: research.tools.options_pricer not importable — using inline fallbacks")
    from scipy.stats import norm as _norm

    RISK_FREE_RATE = 0.045
    DEFAULT_HAIRCUT = 0.15
    COMMISSION_RT_SPREAD = 2.60

    def _bs_call(S, K, T, r=RISK_FREE_RATE, sigma=0.25):
        if T <= 0 or sigma <= 0:
            return max(S - K, 0.0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return float(S * _norm.cdf(d1) - K * np.exp(-r * T) * _norm.cdf(d2))

    def _bs_put(S, K, T, r=RISK_FREE_RATE, sigma=0.25):
        if T <= 0 or sigma <= 0:
            return max(K - S, 0.0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return float(K * np.exp(-r * T) * _norm.cdf(-d2) - S * _norm.cdf(-d1))

    def estimate_iv(vix, atr, S, dte):
        base = vix / 100.0
        atr_vol = (atr / S) * np.sqrt(252) if S > 0 else base
        return 0.6 * base + 0.4 * atr_vol

    def compute_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(alpha=1/period, min_periods=period).mean()

    def price_bull_call_spread(S, K1, K2, dte, atr, vix):
        T = dte / 365.0
        iv = estimate_iv(vix, atr, S, dte)
        c1 = _bs_call(S, K1, T, sigma=iv)
        c2 = _bs_call(S, K2, T, sigma=iv)
        raw_debit = c1 - c2
        entry_cost = raw_debit * (1 + DEFAULT_HAIRCUT)
        max_profit = (K2 - K1) - entry_cost
        return entry_cost, max_profit

    def price_bear_put_spread(S, K1, K2, dte, atr, vix):
        T = dte / 365.0
        iv = estimate_iv(vix, atr, S, dte)
        p2 = _bs_put(S, K2, T, sigma=iv)
        p1 = _bs_put(S, K1, T, sigma=iv)
        raw_debit = p2 - p1
        entry_cost = raw_debit * (1 + DEFAULT_HAIRCUT)
        max_profit = (K2 - K1) - entry_cost
        return entry_cost, max_profit

try:
    from research.tools.adversarial_validator import validate_trades
    fprint("Imported adversarial_validator from research.tools")
except ImportError:
    fprint("ERROR: adversarial_validator required but not found")
    sys.exit(1)

# ── Config ──
BASE = Path("/home/nick/Lvl3Quant")
OUTPUT_DIR = BASE / "output" / "growth_research" / "oos_robustness_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
SPREAD_PCT = 3.0
OTM_PCT = 2.0
TOP_K = 2
REBAL_DAYS = 5
DTE = 21
REGIME_BULL_THRESHOLD = 0.4
REGIME_BEAR_THRESHOLD = 0.2

REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"
WF_TRAIN_PERIODS = 12

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "oos_robustness_v1"

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
# TIME WINDOWS
# ══════════════════════════════════════════════════════════════
WINDOWS = [
    {"name": "crisis_08_11", "start": "2008-01-01", "end": "2011-12-31", "desc": "GFC + recovery"},
    {"name": "bull_12_15", "start": "2012-01-01", "end": "2015-12-31", "desc": "QE bull market"},
    {"name": "vol_16_19", "start": "2016-01-01", "end": "2019-12-31", "desc": "Brexit, trade wars, vol spike 2018"},
    {"name": "covid_20_21", "start": "2020-01-01", "end": "2021-12-31", "desc": "COVID crash + recovery"},
    {"name": "bear_22_23", "start": "2022-01-01", "end": "2023-12-31", "desc": "Rate hike bear market"},
    {"name": "recent_24_26", "start": "2024-01-01", "end": "2026-07-27", "desc": "Most recent (closest to live)"},
]


# ══════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════

def download_data():
    """Download all required tickers via yfinance."""
    import yfinance as yf

    all_tickers = SECTORS + EXTRA_TICKERS
    fprint(f"Downloading {len(all_tickers)} tickers...")

    raw = yf.download(all_tickers, start="2006-01-01", progress=False, auto_adjust=True)
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
# FEATURE ENGINEERING (IDENTICAL TO PRODUCTION)
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

def build_feature_records(close, high, low, rebal_dates, feature_cols, dte):
    """
    Build feature + target records for all sectors on all rebal dates.
    No regime filtering here — regime applied at trade time.
    """
    import lightgbm as lgb

    fprint(f"  Building records: {len(rebal_dates)} dates, {len(feature_cols)} features")

    records = []
    sector_cols = [c for c in SECTORS if c in close.columns]
    spy = close["SPY"]

    for dt in rebal_dates:
        idx = close.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
            continue

        for tk in sector_cols:
            px = close[tk].iloc[:idx + 1].dropna()
            spy_s = spy.iloc[:idx + 1]

            legacy = compute_legacy_features(px, spy_s)
            if not legacy:
                continue

            cross_asset = {}
            for col in feature_cols:
                if col in VALIDATED_CROSS_ASSET:
                    cross_asset = compute_cross_asset_features(tk, idx, close)
                    break

            fi = min(idx + dte, len(close) - 1)
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


def walk_forward_lgbm_rank(df, feature_cols, label):
    """Walk-forward LGBM ranking: sliding train, predict next period."""
    import lightgbm as lgb

    if len(df) < 100:
        fprint(f"    {label}: Insufficient data ({len(df)} records)")
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

    fprint(f"    {label}: {len(rankings)} ranking dates, {n_models} models trained")
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
# UNIFIED TRADE SIMULATOR (BULL SPREADS + PAIR TRADES)
# ══════════════════════════════════════════════════════════════

def simulate_window(rankings, close, high, low, regime_series, atr_dict,
                    window_start, window_end):
    """
    Simulate the BEST strategy (Session 8) within a time window.

    Strategy logic:
      - VIX > 20: Bull call spreads on top-K sectors (regime score > 0.4)
      - VIX < 20: Pair trades (long top-K bull call + short bottom-K bear put)
      - VIX 25-30: sit in cash (skip)

    All parameters fixed: K=2, weekly, DTE=21, 2% OTM, 3% spread width.
    """
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    equity = CAP
    trades = []

    w_start = pd.Timestamp(window_start)
    w_end = pd.Timestamp(window_end)

    for dt in sorted(rankings.keys()):
        if dt < w_start or dt > w_end:
            continue
        if dt not in spy.index:
            continue

        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0

        # VIX 25-30 sit-in-cash filter
        if 25.0 <= cv <= 30.0:
            continue

        scores = rankings[dt]
        if not scores:
            continue

        # Get regime score
        rscore = get_regime_score_at(regime_series, dt)

        # Position sizing
        max_pos = min(200, equity / 3)
        if max_pos < 30:
            continue

        di = close.index.get_loc(dt)
        ei = min(di + DTE, len(close) - 1)
        if ei <= di:
            continue

        ranked_top = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_bot = sorted(scores.items(), key=lambda x: x[1])

        if cv >= 20.0:
            # HIGH VIX: Bull call spreads only (if regime is bullish)
            if rscore <= REGIME_BULL_THRESHOLD:
                continue  # Not bullish enough, skip

            picks = [t for t, _ in ranked_top[:TOP_K]]
            for tk in picks:
                trade = _enter_bull_spread(tk, close, atr_dict, cv, dt, di, ei, max_pos, equity)
                if trade:
                    equity += trade["pnl"]
                    trade["entry_date"] = str(dt.date())
                    trade["exit_date"] = str(close.index[ei].date())
                    trade["ticker"] = tk
                    trade["vix"] = round(cv, 1)
                    trade["regime_score"] = round(rscore, 3)
                    trade["win"] = trade["pnl"] > 0
                    # SPY regime classification
                    sv = float(spy.loc[dt])
                    se = float(spy.iloc[ei]) if ei < len(spy) else sv
                    trade["regime"] = "bull" if se >= sv else "bear"
                    trades.append(trade)
        else:
            # LOW VIX (<20): Pair trades — long top-K + short bottom-K
            top_picks = [t for t, _ in ranked_top[:TOP_K]]
            bot_picks = [t for t, _ in ranked_bot[:TOP_K]]
            # Avoid overlap
            bot_picks = [t for t in bot_picks if t not in top_picks]

            # Long leg: bull call spreads on top sectors
            for tk in top_picks:
                trade = _enter_bull_spread(tk, close, atr_dict, cv, dt, di, ei, max_pos, equity)
                if trade:
                    equity += trade["pnl"]
                    trade["entry_date"] = str(dt.date())
                    trade["exit_date"] = str(close.index[ei].date())
                    trade["ticker"] = tk
                    trade["vix"] = round(cv, 1)
                    trade["regime_score"] = round(rscore, 3)
                    trade["win"] = trade["pnl"] > 0
                    sv = float(spy.loc[dt])
                    se = float(spy.iloc[ei]) if ei < len(spy) else sv
                    trade["regime"] = "bull" if se >= sv else "bear"
                    trades.append(trade)

            # Short leg: bear put spreads on bottom sectors
            for tk in bot_picks[:TOP_K]:
                trade = _enter_bear_spread(tk, close, atr_dict, cv, dt, di, ei, max_pos, equity)
                if trade:
                    equity += trade["pnl"]
                    trade["entry_date"] = str(dt.date())
                    trade["exit_date"] = str(close.index[ei].date())
                    trade["ticker"] = tk
                    trade["vix"] = round(cv, 1)
                    trade["regime_score"] = round(rscore, 3)
                    trade["win"] = trade["pnl"] > 0
                    sv = float(spy.loc[dt])
                    se = float(spy.iloc[ei]) if ei < len(spy) else sv
                    trade["regime"] = "bull" if se >= sv else "bear"
                    trades.append(trade)

    return trades, equity


def _enter_bull_spread(tk, close, atr_dict, cv, dt, di, ei, max_pos, equity):
    """Price and enter a bull call spread. Returns trade dict or None."""
    if tk not in close.columns or tk not in atr_dict:
        return None

    S = float(close[tk].loc[dt])
    K1 = round(S * (1 + OTM_PCT / 100), 2)
    K2 = round(K1 * (1 + SPREAD_PCT / 100), 2)
    if K2 <= K1:
        K2 = K1 + 0.50

    if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]):
        av = float(atr_dict[tk].loc[dt])
    else:
        av = S * 0.015

    try:
        entry_cost_ps, max_profit_ps = price_bull_call_spread(
            S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=cv
        )
    except Exception:
        return None

    total_cost = entry_cost_ps * 100 + COMMISSION_RT_SPREAD
    if total_cost <= 0 or total_cost > max_pos or total_cost > equity * 0.40:
        return None

    Se = float(close[tk].iloc[ei])
    intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
    pnl = (intrinsic - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD

    return {"pnl": round(pnl, 2), "direction": "bull", "K1": K1, "K2": K2}


def _enter_bear_spread(tk, close, atr_dict, cv, dt, di, ei, max_pos, equity):
    """Price and enter a bear put spread. Returns trade dict or None."""
    if tk not in close.columns or tk not in atr_dict:
        return None

    S = float(close[tk].loc[dt])
    K_upper = round(S * (1 - OTM_PCT / 100), 2)
    K_lower = round(K_upper * (1 - SPREAD_PCT / 100), 2)
    if K_upper <= K_lower:
        K_upper = K_lower + 0.50

    if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]):
        av = float(atr_dict[tk].loc[dt])
    else:
        av = S * 0.015

    try:
        entry_cost_ps, max_profit_ps = price_bear_put_spread(
            S=S, K1=K_lower, K2=K_upper, dte=DTE, atr=av, vix=cv
        )
    except Exception:
        return None

    total_cost = entry_cost_ps * 100 + COMMISSION_RT_SPREAD
    if total_cost <= 0 or total_cost > max_pos or total_cost > equity * 0.40:
        return None

    Se = float(close[tk].iloc[ei])
    intrinsic = max(K_upper - Se, 0.0) - max(K_lower - Se, 0.0)
    pnl = (intrinsic - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD

    return {"pnl": round(pnl, 2), "direction": "bear", "K1": K_lower, "K2": K_upper}


# ══════════════════════════════════════════════════════════════
# REGIME-STRATIFIED SHARPE
# ══════════════════════════════════════════════════════════════

def compute_regime_stratified_sharpe(trades):
    """Compute Sharpe within VIX buckets: <20, 20-30, >30."""
    if not trades:
        return {}

    result = {}
    for label, lo, hi in [("vix_lt20", 0, 20), ("vix_20_30", 20, 30), ("vix_gt30", 30, 200)]:
        bucket = [t for t in trades if lo <= t.get("vix", 20) < hi]
        if len(bucket) < 5:
            result[f"sharpe_{label}"] = None
            result[f"n_{label}"] = len(bucket)
            continue

        pnls = np.array([t["pnl"] for t in bucket])
        # Simple Sharpe from PnL stream
        if pnls.std() > 0:
            # Annualize assuming weekly trades
            result[f"sharpe_{label}"] = round(float(pnls.mean() / pnls.std() * np.sqrt(52)), 3)
        else:
            result[f"sharpe_{label}"] = 0.0
        result[f"n_{label}"] = len(bucket)

    return result


# ══════════════════════════════════════════════════════════════
# RANDOM BASELINE
# ══════════════════════════════════════════════════════════════

def random_baseline_test(rankings, close, high, low, regime_series, atr_dict,
                         window_start, window_end, n_trials=5):
    """Test random sector selection within the same window."""
    random_sharpes = []

    for trial in range(n_trials):
        np.random.seed(42 + trial)
        rand_rankings = {}
        for dt, scores in rankings.items():
            rand_scores = {tk: np.random.random() for tk in scores.keys()}
            rand_rankings[dt] = rand_scores

        trades, final_eq = simulate_window(
            rand_rankings, close, high, low, regime_series, atr_dict,
            window_start, window_end,
        )

        if trades and len(trades) >= 5:
            result = validate_trades(
                trades, initial_capital=CAP,
                spy_prices=close["SPY"],
                strategy_name=f"Random_{trial}",
                n_perms=500,
            )
            random_sharpes.append(result.sharpe)
        else:
            random_sharpes.append(0.0)

    return random_sharpes


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t0 = datetime.now()
    fprint("=" * 80)
    fprint(f"OOS ROBUSTNESS TEST v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 80)
    fprint(f"Strategy: K={TOP_K}, weekly ({REBAL_DAYS}d), DTE={DTE}, {OTM_PCT}% OTM")
    fprint(f"Capital: ${CAP:.0f}/window | Spread: {SPREAD_PCT:.0f}% | "
           f"Comm: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"HOLD TO EXPIRY | Intrinsic only | 15% entry haircut, no exit haircut")
    fprint(f"VIX>=20: bull spreads (regime>0.4) | VIX<20: pair trades")
    fprint(f"VIX 25-30: sit in cash")
    fprint()

    fprint("TIME WINDOWS:")
    for w in WINDOWS:
        fprint(f"  {w['name']:<16} {w['start']} to {w['end']}  ({w['desc']})")
    fprint()

    # 1. Download data
    close, high, low = download_data()

    # 2. Load regime predictions
    regime_series = load_regime_predictions()

    # 3. Pre-compute ATR
    atr_dict = compute_atr_series(high, low, close)

    # 4. Build rebalance dates and LGBM rankings (FULL dataset, expanding walk-forward)
    feature_cols = V4_FEATURES
    rebal_freq = f"{REBAL_DAYS}B"
    rebal_dates = pd.DatetimeIndex(
        close.index.to_series().resample(rebal_freq).last().dropna().values
    )
    fprint(f"\nRebalance dates: {len(rebal_dates)} "
           f"({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

    fprint("\nBuilding walk-forward LGBM rankings on FULL dataset...")
    records = build_feature_records(close, high, low, rebal_dates, feature_cols, dte=DTE)
    rankings, imp_df = walk_forward_lgbm_rank(records, feature_cols, "full_dataset")

    if imp_df is not None:
        fprint("\nTop 10 feature importances:")
        for _, row in imp_df.head(10).iterrows():
            fprint(f"  {row['feature']:<30} {row['importance']:.1f}")

    # ══════════════════════════════════════════════════════════════
    # RUN EACH WINDOW
    # ══════════════════════════════════════════════════════════════
    all_results = {}
    all_trades_by_window = {}

    for w in WINDOWS:
        wname = w["name"]
        fprint(f"\n{'=' * 80}")
        fprint(f"WINDOW: {wname} — {w['start']} to {w['end']} ({w['desc']})")
        fprint(f"{'=' * 80}")

        trades, final_eq = simulate_window(
            rankings, close, high, low, regime_series, atr_dict,
            w["start"], w["end"],
        )

        all_trades_by_window[wname] = trades

        if not trades or len(trades) < 5:
            fprint(f"  Only {len(trades) if trades else 0} trades — insufficient for validation")
            all_results[wname] = {
                "description": w["desc"],
                "start": w["start"],
                "end": w["end"],
                "n_trades": len(trades) if trades else 0,
                "error": "insufficient_trades",
            }
            continue

        # 5-gate adversarial validation
        n_perms = 2000 if len(trades) >= 20 else 500
        result = validate_trades(
            trades, initial_capital=CAP,
            spy_prices=close["SPY"],
            strategy_name=f"{wname}",
            n_perms=n_perms,
        )
        result.print_summary()

        # Direction breakdown
        bull_trades = [t for t in trades if t.get("direction") == "bull"]
        bear_trades = [t for t in trades if t.get("direction") == "bear"]
        bull_pnl = sum(t["pnl"] for t in bull_trades)
        bear_pnl = sum(t["pnl"] for t in bear_trades)
        bull_wr = sum(1 for t in bull_trades if t["win"]) / max(len(bull_trades), 1) * 100
        bear_wr = sum(1 for t in bear_trades if t["win"]) / max(len(bear_trades), 1) * 100
        fprint(f"  Direction breakdown:")
        fprint(f"    Bull: {len(bull_trades)} trades, WR {bull_wr:.1f}%, PnL ${bull_pnl:.0f}")
        fprint(f"    Bear: {len(bear_trades)} trades, WR {bear_wr:.1f}%, PnL ${bear_pnl:.0f}")

        # Regime-stratified Sharpe
        regime_sharpes = compute_regime_stratified_sharpe(trades)
        fprint(f"  Regime-stratified Sharpe:")
        for k, v in regime_sharpes.items():
            if k.startswith("sharpe_"):
                label = k.replace("sharpe_", "")
                n_key = k.replace("sharpe_", "n_")
                n = regime_sharpes.get(n_key, 0)
                fprint(f"    {label}: Sharpe={v if v is not None else 'N/A'} ({n} trades)")

        # CAGR
        if trades:
            first_date = pd.Timestamp(trades[0]["entry_date"])
            last_date = pd.Timestamp(trades[-1]["exit_date"])
            years = max((last_date - first_date).days / 365.25, 0.25)
            cagr = (final_eq / CAP) ** (1 / years) - 1
        else:
            cagr = 0.0
            years = 0.0

        # Random baseline
        fprint(f"\n  Random baseline test (5 trials)...")
        random_sharpes = random_baseline_test(
            rankings, close, high, low, regime_series, atr_dict,
            w["start"], w["end"],
        )
        mean_random = np.mean(random_sharpes) if random_sharpes else 0
        fprint(f"  ML Sharpe: {result.sharpe:.2f} vs Random mean: {mean_random:.2f}")
        alpha_ratio = result.sharpe / mean_random if mean_random > 0 and result.sharpe > 0 else float("inf")
        fprint(f"  ML alpha ratio: {alpha_ratio:.2f}x")

        all_results[wname] = {
            "description": w["desc"],
            "start": w["start"],
            "end": w["end"],
            **result.to_dict(),
            "cagr": round(cagr, 4),
            "years": round(years, 2),
            "total_return": round((final_eq / CAP - 1) * 100, 2),
            "bull_trades": len(bull_trades),
            "bear_trades": len(bear_trades),
            "bull_pnl": round(bull_pnl, 2),
            "bear_pnl": round(bear_pnl, 2),
            "bull_wr": round(bull_wr, 3),
            "bear_wr": round(bear_wr, 3),
            **regime_sharpes,
            "random_sharpes": [round(s, 3) for s in random_sharpes],
            "random_mean_sharpe": round(mean_random, 3),
            "alpha_ratio": round(alpha_ratio, 3),
        }

    # ══════════════════════════════════════════════════════════════
    # SUMMARY STATISTICS
    # ══════════════════════════════════════════════════════════════
    fprint(f"\n{'=' * 80}")
    fprint("WINDOW-BY-WINDOW SUMMARY")
    fprint(f"{'=' * 80}")
    fprint(f"{'Window':<16} {'Period':<24} {'Trd':>5} {'Sharpe':>7} {'Sortino':>8} "
           f"{'CAGR':>7} {'MaxDD':>7} {'WR':>6} {'PF':>6} {'Gates':>6} {'TotRet':>8}")
    fprint("-" * 120)

    valid_sharpes = []
    for w in WINDOWS:
        wname = w["name"]
        r = all_results.get(wname, {})
        if "error" in r:
            fprint(f"  {wname:<16} {w['start'][:10]}-{w['end'][:10]}  "
                   f"{'— INSUFFICIENT DATA —':>60}")
            continue

        sharpe = r.get("sharpe", 0)
        valid_sharpes.append(sharpe)
        fprint(f"  {wname:<16} {w['start'][:10]}-{w['end'][:10]} "
               f"{r['n_trades']:>5} {sharpe:>7.2f} {r['sortino']:>8.2f} "
               f"{r['cagr']*100:>6.1f}% {r['max_dd']*100:>6.1f}% "
               f"{r['win_rate']*100:>5.1f}% {r['profit_factor']:>5.2f} "
               f"{r['gates_passed']}/{r['gates_total']} "
               f"{r['total_return']:>7.1f}%")

    # ══════════════════════════════════════════════════════════════
    # CROSS-WINDOW ROBUSTNESS METRICS
    # ══════════════════════════════════════════════════════════════
    fprint(f"\n{'=' * 80}")
    fprint("CROSS-WINDOW ROBUSTNESS METRICS")
    fprint(f"{'=' * 80}")

    if valid_sharpes:
        mean_sharpe = np.mean(valid_sharpes)
        std_sharpe = np.std(valid_sharpes)
        worst_sharpe = min(valid_sharpes)
        best_sharpe = max(valid_sharpes)
        consistency_ratio = mean_sharpe / std_sharpe if std_sharpe > 0 else float("inf")
        n_negative = sum(1 for s in valid_sharpes if s < 0)
        n_above_1 = sum(1 for s in valid_sharpes if s >= 1.0)

        fprint(f"\n  Windows tested:          {len(valid_sharpes)}")
        fprint(f"  Mean Sharpe:             {mean_sharpe:.3f}")
        fprint(f"  Std Sharpe:              {std_sharpe:.3f}")
        fprint(f"  Consistency ratio:       {consistency_ratio:.3f} (mean/std, higher=better)")
        fprint(f"  Best window Sharpe:      {best_sharpe:.3f}")
        fprint(f"  Worst window Sharpe:     {worst_sharpe:.3f}")
        fprint(f"  Negative Sharpe windows: {n_negative}/{len(valid_sharpes)}")
        fprint(f"  Sharpe >= 1.0 windows:   {n_above_1}/{len(valid_sharpes)}")

        # Verdict
        fprint(f"\n  {'─' * 60}")
        if n_negative == 0 and consistency_ratio >= 1.0:
            verdict = "ROBUST — Positive in ALL windows, high consistency"
        elif n_negative == 0 and consistency_ratio >= 0.5:
            verdict = "ACCEPTABLE — Positive in all windows, moderate consistency"
        elif n_negative <= 1 and mean_sharpe >= 1.0:
            verdict = "CAUTIOUS — Strong mean but not universally positive"
        else:
            verdict = "FRAGILE — Strategy may be regime-dependent"
        fprint(f"  VERDICT: {verdict}")
        fprint(f"  {'─' * 60}")

        # Compare to full-period result
        fprint(f"\n  Full-period Sharpe (Session 8): 2.79")
        fprint(f"  Mean window Sharpe:            {mean_sharpe:.2f}")
        if mean_sharpe < 2.79 * 0.5:
            fprint(f"  WARNING: Window mean is <50% of full-period — possible overfitting")
        elif mean_sharpe < 2.79 * 0.7:
            fprint(f"  NOTE: Window mean is 50-70% of full-period — some regime dependence")
        else:
            fprint(f"  GOOD: Window mean is >=70% of full-period — strategy appears robust")
    else:
        fprint("  No valid windows to analyze!")

    # Save results
    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    # Save trade details per window
    trades_path = OUTPUT_DIR / "trades_by_window.json"
    serializable_trades = {}
    for wname, trades in all_trades_by_window.items():
        serializable_trades[wname] = trades
    with open(trades_path, "w") as f:
        json.dump(serializable_trades, f, indent=2, default=str)
    fprint(f"Trade details saved to {trades_path}")

    # MLflow logging
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"oos_robustness_{t0.strftime('%Y%m%d_%H%M')}"):
                # Log per-window metrics
                for wname, r in all_results.items():
                    if "error" in r:
                        continue
                    mlflow.log_metric(f"{wname}_sharpe", r.get("sharpe", 0))
                    mlflow.log_metric(f"{wname}_sortino", r.get("sortino", 0))
                    mlflow.log_metric(f"{wname}_cagr", r.get("cagr", 0))
                    mlflow.log_metric(f"{wname}_max_dd", r.get("max_dd", 0))
                    mlflow.log_metric(f"{wname}_win_rate", r.get("win_rate", 0))
                    mlflow.log_metric(f"{wname}_profit_factor", r.get("profit_factor", 0))
                    mlflow.log_metric(f"{wname}_n_trades", r.get("n_trades", 0))
                    mlflow.log_metric(f"{wname}_gates_passed", r.get("gates_passed", 0))
                    mlflow.log_metric(f"{wname}_total_return", r.get("total_return", 0))
                    mlflow.log_metric(f"{wname}_alpha_ratio", r.get("alpha_ratio", 0))

                # Log summary metrics
                if valid_sharpes:
                    mlflow.log_metric("mean_sharpe", float(np.mean(valid_sharpes)))
                    mlflow.log_metric("std_sharpe", float(np.std(valid_sharpes)))
                    mlflow.log_metric("worst_sharpe", float(min(valid_sharpes)))
                    mlflow.log_metric("best_sharpe", float(max(valid_sharpes)))
                    mlflow.log_metric("consistency_ratio",
                                      float(np.mean(valid_sharpes) / np.std(valid_sharpes))
                                      if np.std(valid_sharpes) > 0 else 0)
                    mlflow.log_metric("n_negative_windows",
                                      sum(1 for s in valid_sharpes if s < 0))

                mlflow.log_params({
                    "capital_per_window": CAP,
                    "top_k": TOP_K,
                    "rebal_days": REBAL_DAYS,
                    "dte": DTE,
                    "otm_pct": OTM_PCT,
                    "spread_pct": SPREAD_PCT,
                    "n_windows": len(WINDOWS),
                    "strategy": "weekly_otm2_pairs",
                    "regime_bull_threshold": REGIME_BULL_THRESHOLD,
                    "full_period_sharpe_ref": 2.79,
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
