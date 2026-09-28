#!/usr/bin/env python3
"""
Adaptive Rebalance v1 — Does Signal-Triggered Rebalancing Beat Fixed Weekly?
=============================================================================

Tests whether adaptive (event-driven) rebalancing triggers outperform fixed
weekly (5 trading day) rebalancing for sector bull call spreads.

Production v5 baseline: fixed weekly (5d), K=2, DTE=21, GRU regime>0.4 gate,
Sharpe 2.04 with cross-validation.

8 Variants:
  1. v5_baseline        — Fixed weekly (5d), K=2, DTE=21 (control)
  2. vix_crossover      — Rebalance when VIX crosses 20, plus fixed weekly min
  3. momentum_trigger   — Rebalance when top-2 ranking changes (rank flip)
  4. vol_breakout       — Rebalance when 5d SPY vol > 1.5x its 21d avg
  5. drawdown_trigger   — Rebalance if portfolio draws down >5% from peak
  6. fast_3d            — Fixed 3-day rebalance
  7. slow_10d           — Fixed 10-day (old v4 default, sanity check)
  8. combined_trigger   — Rebalance on ANY of: VIX crossover, rank flip, or weekly

All share: LGBM WF ranking (21 features), GRU regime>0.4 gate, DTE=21,
3% spread, ATM, hold-to-expiry, $645 start, max $200/trade,
commission $2.60 RT, 15% entry haircut, no exit haircut.
"""

import json
import sys
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
    COMMISSION_RT_SPREAD,
    DEFAULT_HAIRCUT,
)
from research.tools.adversarial_validator import validate_trades

# -- Config --
BASE = Path("/home/jupiter/Lvl3Quant")
OUTPUT_DIR = BASE / "output" / "growth_research" / "adaptive_rebalance_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
DTE = 21
SPREAD_PCT = 3.0
TOP_K = 2
MAX_POS = 200.0
REGIME_BULL_THRESHOLD = 0.4

REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"

# LGBM walk-forward: use 12 train periods on the eval grid
WF_TRAIN_PERIODS = 12

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "adaptive_rebalance_v1"

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


# 21 features (18 legacy + 3 validated cross-asset)
LEGACY_FEATURES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_21d", "vol_63d", "sharpe_63d", "maxdd_63d", "pct_52w_high", "mom_accel",
    "pct_pos_months_12m", "sortino_63d", "calmar_1y", "up_capture",
    "trend_r2_63d", "trend_slope_63d",
]

CROSS_ASSET_FEATURES = [
    "sector_spy_beta_63d",
    "sector_relative_vol_21d",
    "cross_sector_dispersion",
]

ALL_FEATURES = LEGACY_FEATURES + CROSS_ASSET_FEATURES  # 21 total


# ==============================================================
# DATA DOWNLOAD
# ==============================================================

def download_data():
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
    for t in ["SPY", "VIX"]:
        if t not in close.columns:
            raise ValueError(f"Missing critical ticker: {t}")
    fprint(f"Data: {len(close)} days, {close.index[0].date()} to {close.index[-1].date()}")
    return close, high, low


# ==============================================================
# REGIME
# ==============================================================

def load_regime_predictions():
    if not REGIME_FILE.exists():
        fprint("WARNING: Regime file not found, using VIX-based proxy")
        return None
    data = np.load(REGIME_FILE, allow_pickle=True)
    dates = pd.to_datetime(data["dates"])
    scores = data["regime_scores"]
    regime_series = pd.Series(scores, index=dates, name="regime_score")
    regime_series = regime_series[~regime_series.index.duplicated(keep="last")]
    fprint(f"Regime loaded: {len(regime_series)} days, "
           f"mean={regime_series.mean():.3f}, >0.4: {(regime_series > 0.4).sum()}")
    return regime_series


def get_regime_score_at(regime_series, dt):
    if regime_series is None:
        return 0.5
    if dt in regime_series.index:
        return float(regime_series.loc[dt])
    nearest = regime_series.index[regime_series.index.get_indexer([dt], method="ffill")]
    if len(nearest) > 0:
        return float(regime_series.loc[nearest[0]])
    return 0.5


# ==============================================================
# FEATURE ENGINEERING
# ==============================================================

def compute_legacy_features(px, spy_slice):
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
    f = {}
    spy = close_df["SPY"].iloc[:dt_idx + 1].dropna()
    sector_px = close_df[sector_ticker].iloc[:dt_idx + 1].dropna() if sector_ticker in close_df.columns else None
    if spy is None or len(spy) < 63:
        return {k: 0.0 for k in CROSS_ASSET_FEATURES}
    spy_ret = spy.pct_change().dropna()
    if sector_px is not None and len(sector_px) > 63:
        sec_ret = sector_px.pct_change().dropna()
        common = spy_ret.index.intersection(sec_ret.index)
        if len(common) > 63:
            sr = sec_ret.loc[common].iloc[-63:]
            mr = spy_ret.loc[common].iloc[-63:]
            cov = np.cov(sr.values, mr.values)
            f["sector_spy_beta_63d"] = float(cov[0, 1] / (cov[1, 1] + 1e-10))
        else:
            f["sector_spy_beta_63d"] = 1.0
    else:
        f["sector_spy_beta_63d"] = 1.0
    if sector_px is not None and len(sector_px) > 21:
        sec_ret = sector_px.pct_change().dropna()
        if len(sec_ret) > 21 and len(spy_ret) > 21:
            f["sector_relative_vol_21d"] = float(sec_ret.iloc[-21:].std() / (spy_ret.iloc[-21:].std() + 1e-10))
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


# ==============================================================
# ATR
# ==============================================================

def compute_atr_series(high, low, close, period=14):
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


# ==============================================================
# WALK-FORWARD LGBM: TRAIN ON GRID, PREDICT ON ARBITRARY DATES
# ==============================================================

def build_feature_records(close, high, low, eval_dates, regime_series):
    """Build feature + target records for all sectors on eval dates (regime>0.4 only)."""
    records = []
    sector_cols = [c for c in SECTORS if c in close.columns]
    spy = close["SPY"]
    for dt in eval_dates:
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
    for c in ALL_FEATURES:
        if c not in df.columns:
            df[c] = 0.0
    df[ALL_FEATURES] = df[ALL_FEATURES].fillna(0.0)
    return df


def train_and_rank(df):
    """
    Walk-forward LGBM: train on sliding window, predict on next date.
    Returns: dict[date -> dict[ticker -> score]], feature importance df
    """
    import lightgbm as lgb

    if len(df) < 100:
        fprint(f"    Insufficient data ({len(df)} records)")
        return {}, None

    df["rank_label"] = df.groupby("date")["fwd_ret"].rank(pct=True)
    dates = sorted(df["date"].unique())

    rankings = {}
    all_importances = np.zeros(len(ALL_FEATURES))
    n_models = 0
    trained_models = {}  # date -> model, for reuse on daily scoring

    for i in range(WF_TRAIN_PERIODS, len(dates)):
        train_dates = dates[max(0, i - WF_TRAIN_PERIODS):i]
        test_date = dates[i]

        train_df = df[df["date"].isin(train_dates)]
        test_df = df[df["date"] == test_date].copy()

        if len(test_df) < 3 or len(train_df) < 50:
            continue

        Xt = np.nan_to_num(train_df[ALL_FEATURES].values.astype(np.float32))
        yt = train_df["rank_label"].values.astype(np.float32)
        Xe = np.nan_to_num(test_df[ALL_FEATURES].values.astype(np.float32))

        try:
            m = lgb.LGBMRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
                verbose=-1,
            )
            m.fit(Xt, yt)
            test_df["score"] = m.predict(Xe)
            rankings[test_date] = dict(zip(test_df["ticker"], test_df["score"]))
            trained_models[test_date] = m
            all_importances += m.feature_importances_
            n_models += 1
        except Exception:
            continue

    if n_models > 0:
        all_importances /= n_models
        imp_df = pd.DataFrame({
            "feature": ALL_FEATURES,
            "importance": all_importances,
        }).sort_values("importance", ascending=False)
    else:
        imp_df = None

    fprint(f"    {len(rankings)} ranking dates, {n_models} models trained")
    return rankings, imp_df, trained_models


def score_daily_with_models(close, regime_series, trained_models, grid_dates):
    """
    Use trained models to score sectors on EVERY trading day.
    For each trading day, use the most recently trained model.
    Only score regime-active days (regime > 0.4).
    """
    sector_cols = [c for c in SECTORS if c in close.columns]
    spy = close["SPY"]

    # Sort model dates for bisection
    model_dates = sorted(trained_models.keys())
    if not model_dates:
        return {}

    trading_days = close.index[(close.index >= model_dates[0]) & (close.index <= close.index[-1])]

    daily_rankings = {}
    n_scored = 0

    for dt in trading_days:
        # Check regime
        rscore = get_regime_score_at(regime_series, dt)
        if rscore <= REGIME_BULL_THRESHOLD:
            continue

        # Find most recent model
        # Binary search for latest model_date <= dt
        model_idx = np.searchsorted(model_dates, dt, side='right') - 1
        if model_idx < 0:
            continue
        model = trained_models[model_dates[model_idx]]

        # Build features for this day
        idx = close.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
            continue

        scores = {}
        feature_rows = []
        tickers = []

        for tk in sector_cols:
            px = close[tk].iloc[:idx + 1].dropna()
            spy_s = spy.iloc[:idx + 1]
            legacy = compute_legacy_features(px, spy_s)
            if not legacy:
                continue
            cross_asset = compute_cross_asset_features(tk, idx, close)
            row = {**legacy, **cross_asset}
            feature_rows.append([row.get(f, 0.0) for f in ALL_FEATURES])
            tickers.append(tk)

        if len(feature_rows) < 3:
            continue

        X = np.nan_to_num(np.array(feature_rows, dtype=np.float32))
        preds = model.predict(X)
        daily_rankings[dt] = dict(zip(tickers, preds))
        n_scored += 1

    fprint(f"    Daily scoring: {n_scored} trading days scored")
    return daily_rankings


# ==============================================================
# REBALANCE SCHEDULE GENERATORS (operate on trading day indices)
# ==============================================================

def get_trading_day_idx(ranked_dates):
    """Map dates to sequential trading-day indices."""
    sorted_dates = sorted(ranked_dates)
    return {dt: i for i, dt in enumerate(sorted_dates)}


def fixed_interval_rebal(ranked_dates, interval):
    """Every `interval` trading days."""
    sorted_dates = sorted(ranked_dates)
    rebal = []
    for i, dt in enumerate(sorted_dates):
        if i % interval == 0:
            rebal.append(dt)
    return rebal


def vix_crossover_rebal(ranked_dates, close, min_gap=5):
    """Rebalance when VIX crosses 20, with min_gap trading day cooldown + weekly max."""
    vix = close["VIX"]
    sorted_dates = sorted(ranked_dates)
    rebal = []
    last_rebal = -min_gap
    prev_above = None

    for i, dt in enumerate(sorted_dates):
        if dt not in vix.index:
            continue
        curr_above = float(vix.loc[dt]) >= 20.0
        triggered = False

        # VIX cross trigger
        if prev_above is not None and curr_above != prev_above and (i - last_rebal) >= min_gap:
            triggered = True

        # Fixed weekly fallback (every 5 trading days)
        if i - last_rebal >= 5:
            triggered = True

        if triggered:
            rebal.append(dt)
            last_rebal = i

        prev_above = curr_above

    return rebal


def momentum_trigger_rebal(ranked_dates, rankings, min_gap=3):
    """Rebalance when top-K ranking changes, with min cooldown + weekly max."""
    sorted_dates = sorted(ranked_dates)
    rebal = []
    last_rebal = -5
    prev_top = None

    for i, dt in enumerate(sorted_dates):
        if dt not in rankings:
            continue
        scores = rankings[dt]
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        curr_top = set(t for t, _ in ranked[:TOP_K])

        triggered = False

        # Rank flip trigger
        if prev_top is not None and curr_top != prev_top and (i - last_rebal) >= min_gap:
            triggered = True

        # Fixed weekly fallback
        if i - last_rebal >= 5:
            triggered = True

        if triggered:
            rebal.append(dt)
            last_rebal = i

        prev_top = curr_top

    return rebal


def vol_breakout_rebal(ranked_dates, close, min_gap=3):
    """Rebalance when 5d SPY vol exceeds 1.5x its 21d average, with min cooldown + weekly."""
    spy_ret = close["SPY"].pct_change()
    sorted_dates = sorted(ranked_dates)
    rebal = []
    last_rebal = -5

    for i, dt in enumerate(sorted_dates):
        if dt not in spy_ret.index:
            continue
        triggered = False
        dt_loc = spy_ret.index.get_loc(dt)

        if dt_loc >= 21 and (i - last_rebal) >= min_gap:
            vol_5d = spy_ret.iloc[dt_loc-4:dt_loc+1].std()
            vol_21d = spy_ret.iloc[dt_loc-20:dt_loc+1].std()
            if vol_21d > 0 and vol_5d > 1.5 * vol_21d:
                triggered = True

        # Fixed weekly fallback
        if i - last_rebal >= 5:
            triggered = True

        if triggered:
            rebal.append(dt)
            last_rebal = i

    return rebal


def combined_trigger_rebal(ranked_dates, close, rankings, min_gap=2):
    """Any of VIX cross, rank flip, or weekly -- whichever fires first."""
    vix = close["VIX"]
    sorted_dates = sorted(ranked_dates)
    rebal = []
    last_rebal = -5
    prev_above = None
    prev_top = None

    for i, dt in enumerate(sorted_dates):
        triggered = False

        # VIX crossover
        if dt in vix.index:
            curr_above = float(vix.loc[dt]) >= 20.0
            if prev_above is not None and curr_above != prev_above and (i - last_rebal) >= min_gap:
                triggered = True
            prev_above = curr_above

        # Rank flip
        if dt in rankings:
            scores = rankings[dt]
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            curr_top = set(t for t, _ in ranked[:TOP_K])
            if prev_top is not None and curr_top != prev_top and (i - last_rebal) >= min_gap:
                triggered = True
            prev_top = curr_top

        # Weekly fallback
        if i - last_rebal >= 5:
            triggered = True

        if triggered:
            rebal.append(dt)
            last_rebal = i

    return rebal


# ==============================================================
# TRADE SIMULATION
# ==============================================================

def simulate_trades(name, rankings, close, atr_dict, rebal_dates,
                    is_drawdown_variant=False):
    """
    Simulate bull call spreads from rankings on specified rebalance dates.
    Hold to expiry, intrinsic value only, 15% entry haircut, K=2.
    """
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    equity = CAP
    peak_equity = CAP
    trades = []
    rebal_count = 0
    actual_rebal_dates = []
    rebal_set = set(rebal_dates)

    sorted_ranked = sorted(rankings.keys())

    for dt in sorted_ranked:
        if dt not in spy.index:
            continue

        should_rebal = dt in rebal_set

        # Drawdown trigger override
        if is_drawdown_variant and not should_rebal:
            if peak_equity > 0 and (equity - peak_equity) / peak_equity < -0.05:
                should_rebal = True

        if not should_rebal:
            # Still track equity for drawdown variant via existing positions
            continue

        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0

        scores = rankings[dt]
        if not scores:
            continue

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        picks = [t for t, _ in ranked[:TOP_K]]

        max_pos = min(MAX_POS, equity / 3)
        if max_pos < 30:
            continue

        rebal_count += 1
        actual_rebal_dates.append(dt)
        n_entered = 0

        for tk in picks:
            if tk not in close.columns or tk not in atr_dict or n_entered >= TOP_K:
                continue

            S = float(close[tk].loc[dt])
            di = close.index.get_loc(dt)
            ei = min(di + DTE, len(close) - 1)
            if ei <= di:
                continue

            if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]):
                av = float(atr_dict[tk].loc[dt])
            else:
                av = S * 0.015

            K1 = round(S, 2)
            K2 = round(S * (1 + SPREAD_PCT / 100), 2)
            if K2 <= K1:
                K2 = K1 + 1.0

            try:
                entry_cost_ps, max_profit_ps = price_bull_call_spread(
                    S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=cv
                )
            except Exception:
                continue

            total_cost = entry_cost_ps * 100 + COMMISSION_RT_SPREAD
            if total_cost <= 0 or total_cost > max_pos or total_cost > equity * 0.40:
                continue

            Se = float(close[tk].iloc[ei])
            intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)

            pnl = (intrinsic - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD
            equity += pnl
            peak_equity = max(peak_equity, equity)
            n_entered += 1

            sv = float(spy.loc[dt]) if dt in spy.index else 0
            se = float(spy.iloc[ei]) if ei < len(spy) else sv

            trades.append({
                "pnl": round(pnl, 2),
                "entry_date": str(dt.date()),
                "exit_date": str(close.index[ei].date()),
                "ticker": tk,
                "regime": "bull" if se >= sv else "bear",
                "vix": round(cv, 1),
                "win": pnl > 0,
            })

    if len(actual_rebal_dates) > 1:
        gaps = [(actual_rebal_dates[i] - actual_rebal_dates[i-1]).days
                for i in range(1, len(actual_rebal_dates))]
        avg_gap = np.mean(gaps)
    else:
        avg_gap = 0

    return trades, equity, rebal_count, avg_gap


# ==============================================================
# MAIN
# ==============================================================

def main():
    t0 = datetime.now()
    fprint("=" * 80)
    fprint(f"ADAPTIVE REBALANCE V1 -- {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 80)
    fprint(f"Capital: ${CAP:.0f} | DTE: {DTE} | Spread: {SPREAD_PCT:.0f}% | K={TOP_K}")
    fprint(f"Haircut: {DEFAULT_HAIRCUT:.0%} entry only | Comm: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"HOLD TO EXPIRY | Intrinsic value only | No exit haircut")
    fprint(f"Regime gate: GRU > {REGIME_BULL_THRESHOLD}")
    fprint(f"21 features (18 legacy + 3 cross-asset)")
    fprint()

    # 1. Download data
    close, high, low = download_data()

    # 2. Load regime
    regime_series = load_regime_predictions()

    # 3. ATR
    atr_dict = compute_atr_series(high, low, close)

    # 4. Build LGBM on a 3-day grid for max rebalance frequency resolution
    # Using every-3-day grid means we can test 3d, 5d (~6d), and 10d (~12d) intervals
    trading_days_all = close.index[close.index >= "2010-01-01"]
    grid_3d = []
    for i, dt in enumerate(trading_days_all):
        if i % 3 == 0:
            grid_3d.append(dt)
    grid_3d = pd.DatetimeIndex(grid_3d)
    fprint(f"3-day evaluation grid: {len(grid_3d)} dates")

    fprint("  Building feature records on 3-day grid...")
    records = build_feature_records(close, high, low, grid_3d, regime_series)
    fprint(f"  {len(records)} records, {len(records['date'].unique())} regime-active dates")

    fprint("  Training walk-forward LGBM...")
    rankings_grid, imp_df, trained_models = train_and_rank(records)

    if not rankings_grid:
        fprint("ERROR: No rankings produced. Cannot continue.")
        return

    # 5. Score EVERY trading day using the most recent trained model
    fprint("  Scoring all trading days with trained models...")
    daily_rankings = score_daily_with_models(close, regime_series, trained_models, grid_3d)

    if not daily_rankings or len(daily_rankings) < 50:
        fprint(f"  WARNING: Only {len(daily_rankings)} daily rankings. Using grid rankings.")
        daily_rankings = rankings_grid

    ranked_dates = sorted(daily_rankings.keys())
    fprint(f"Total ranked dates: {len(ranked_dates)} "
           f"({ranked_dates[0].date()} to {ranked_dates[-1].date()})")

    spy_close = close["SPY"]

    # 6. Generate rebalance schedules
    fprint("\n" + "=" * 80)
    fprint("GENERATING REBALANCE SCHEDULES")
    fprint("=" * 80)

    variants = {}

    # V1: Fixed weekly (every 5 trading days)
    variants["v5_baseline"] = {
        "desc": "Fixed weekly (5d), K=2 (control)",
        "rebal_dates": fixed_interval_rebal(ranked_dates, 5),
        "is_dd": False,
    }

    # V2: VIX crossover
    variants["vix_crossover"] = {
        "desc": "VIX crosses 20 + weekly minimum",
        "rebal_dates": vix_crossover_rebal(ranked_dates, close, min_gap=5),
        "is_dd": False,
    }

    # V3: Momentum trigger
    variants["momentum_trigger"] = {
        "desc": "Top-2 rank flip + weekly minimum",
        "rebal_dates": momentum_trigger_rebal(ranked_dates, daily_rankings, min_gap=3),
        "is_dd": False,
    }

    # V4: Vol breakout
    variants["vol_breakout"] = {
        "desc": "SPY 5d vol > 1.5x 21d avg + weekly",
        "rebal_dates": vol_breakout_rebal(ranked_dates, close, min_gap=3),
        "is_dd": False,
    }

    # V5: Drawdown trigger (uses weekly schedule + DD override)
    variants["drawdown_trigger"] = {
        "desc": "Rebal if DD >5% from peak + weekly",
        "rebal_dates": fixed_interval_rebal(ranked_dates, 5),
        "is_dd": True,
    }

    # V6: Fast 3-day
    variants["fast_3d"] = {
        "desc": "Fixed 3-day rebalance",
        "rebal_dates": fixed_interval_rebal(ranked_dates, 3),
        "is_dd": False,
    }

    # V7: Slow 10-day (biweekly)
    variants["slow_10d"] = {
        "desc": "Fixed biweekly (10d), old v4 default",
        "rebal_dates": fixed_interval_rebal(ranked_dates, 10),
        "is_dd": False,
    }

    # V8: Combined trigger
    variants["combined_trigger"] = {
        "desc": "VIX cross OR rank flip OR weekly",
        "rebal_dates": combined_trigger_rebal(ranked_dates, close, daily_rankings, min_gap=2),
        "is_dd": False,
    }

    for vname, vdata in variants.items():
        fprint(f"  {vname:<22} {len(vdata['rebal_dates']):>5} rebal dates | {vdata['desc']}")

    # 7. Simulate all variants
    fprint("\n" + "=" * 80)
    fprint("SIMULATING TRADES")
    fprint("=" * 80)

    all_results = {}

    for vname, vdata in variants.items():
        fprint(f"\n--- {vname}: {vdata['desc']} ---")

        trades, final_eq, rebal_count, avg_gap = simulate_trades(
            vname, daily_rankings, close, atr_dict,
            vdata["rebal_dates"],
            is_drawdown_variant=vdata["is_dd"],
        )

        fprint(f"  Trades: {len(trades)}, Final equity: ${final_eq:.0f}, "
               f"Rebal count: {rebal_count}, Avg gap: {avg_gap:.1f} cal days")

        if not trades or len(trades) < 10:
            fprint(f"  Only {len(trades) if trades else 0} trades, skipping validation")
            all_results[vname] = {
                "description": vdata["desc"],
                "n_trades": len(trades) if trades else 0,
                "sharpe": 0.0, "sortino": 0.0, "win_rate": 0.0,
                "profit_factor": 0.0, "max_dd": -1.0, "cagr": 0.0,
                "final_equity": final_eq,
                "gates_passed": 0, "gates_total": 5, "all_passed": False,
                "rebal_count": rebal_count, "avg_gap_days": round(avg_gap, 1),
                "error": "Too few trades",
            }
            continue

        result = validate_trades(
            trades, initial_capital=CAP,
            spy_prices=spy_close,
            strategy_name=vname,
        )
        result.print_summary()

        all_results[vname] = {
            "description": vdata["desc"],
            **result.to_dict(),
            "rebal_count": rebal_count,
            "avg_gap_days": round(avg_gap, 1),
        }

    # 8. Summary sorted by Sharpe
    fprint("\n" + "=" * 80)
    fprint("SUMMARY COMPARISON (sorted by Sharpe)")
    fprint("=" * 80)
    fprint(f"{'Variant':<22} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} "
           f"{'PF':>6} {'MaxDD':>7} {'CAGR':>7} {'Gates':>6} {'Final$':>8} {'Rebals':>7} {'AvgGap':>7}")
    fprint("-" * 108)

    sorted_variants = sorted(
        all_results.items(),
        key=lambda x: x[1].get("sharpe", 0),
        reverse=True,
    )

    for vname, r in sorted_variants:
        marker = " *" if vname == "v5_baseline" else ""
        fprint(f"  {vname:<22} {r.get('n_trades', 0):>5} {r.get('sharpe', 0):>7.2f} "
               f"{r.get('sortino', 0):>8.2f} "
               f"{r.get('win_rate', 0)*100:>5.1f}% {r.get('profit_factor', 0):>5.2f} "
               f"{r.get('max_dd', 0)*100:>6.1f}% {r.get('cagr', 0)*100:>6.1f}% "
               f"{r.get('gates_passed', 0)}/{r.get('gates_total', 5)} "
               f"${r.get('final_equity', CAP):>7,.0f} {r.get('rebal_count', 0):>6} "
               f"{r.get('avg_gap_days', 0):>6.1f}d{marker}")

    # 9. Key question
    fprint("\n" + "=" * 80)
    fprint("KEY QUESTION: Does signal-triggered rebalancing beat fixed weekly?")
    fprint("=" * 80)

    baseline_sharpe = all_results.get("v5_baseline", {}).get("sharpe", 0)
    best_name = None
    best_sharpe = baseline_sharpe

    for vname, r in all_results.items():
        if vname == "v5_baseline":
            continue
        s = r.get("sharpe", 0)
        if s > best_sharpe:
            best_sharpe = s
            best_name = vname

    if best_name:
        delta = best_sharpe - baseline_sharpe
        pct_improve = (delta / abs(baseline_sharpe) * 100) if baseline_sharpe != 0 else 0
        fprint(f"  YES -- {best_name} beats baseline by {delta:+.2f} Sharpe "
               f"({baseline_sharpe:.2f} -> {best_sharpe:.2f}, +{pct_improve:.0f}%)")
        best_r = all_results[best_name]
        fprint(f"  Best variant: {best_r.get('description', '')}")
        fprint(f"    Trades: {best_r.get('n_trades', 0)}, "
               f"WR: {best_r.get('win_rate', 0)*100:.1f}%, "
               f"PF: {best_r.get('profit_factor', 0):.2f}, "
               f"MaxDD: {best_r.get('max_dd', 0)*100:.1f}%, "
               f"Gates: {best_r.get('gates_passed', 0)}/{best_r.get('gates_total', 5)}")
        fprint(f"    Rebals: {best_r.get('rebal_count', 0)} "
               f"(avg {best_r.get('avg_gap_days', 0):.1f}d gap)")
    else:
        fprint(f"  NO -- No variant beat the baseline Sharpe of {baseline_sharpe:.2f}")
        fprint(f"  Recommendation: Stick with fixed weekly (5d) rebalancing")

    # Commission drag analysis
    fprint("\n  Commission drag comparison:")
    for vname, r in sorted_variants:
        ntrades = r.get("n_trades", 0)
        total_comm = ntrades * COMMISSION_RT_SPREAD
        fprint(f"    {vname:<22} {ntrades:>4} trades x ${COMMISSION_RT_SPREAD:.2f} = "
               f"${total_comm:>7.0f} total commission")

    # Frequency vs Sharpe tradeoff
    fprint("\n  Rebalance frequency vs risk-adjusted return:")
    for vname, r in sorted(all_results.items(), key=lambda x: x[1].get("avg_gap_days", 0)):
        fprint(f"    {vname:<22} gap={r.get('avg_gap_days', 0):>5.1f}d  "
               f"Sharpe={r.get('sharpe', 0):>5.2f}  "
               f"MaxDD={r.get('max_dd', 0)*100:>6.1f}%  "
               f"CAGR={r.get('cagr', 0)*100:>6.1f}%")

    # Feature importance
    if imp_df is not None:
        fprint("\n" + "=" * 80)
        fprint("FEATURE IMPORTANCE (Top 10)")
        fprint("=" * 80)
        for _, row in imp_df.head(10).iterrows():
            bar = "*" * int(row["importance"] / imp_df["importance"].max() * 30)
            fprint(f"  {row['feature']:<30} {row['importance']:>6.1f} {bar}")

    # 10. Save results
    results_path = OUTPUT_DIR / "adaptive_rebalance_v1_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    # 11. MLflow
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"adaptive_rebal_{t0.strftime('%Y%m%d_%H%M')}"):
                for vname, r in all_results.items():
                    prefix = vname[:15]
                    for metric in ["sharpe", "sortino", "win_rate", "profit_factor",
                                   "max_dd", "cagr", "final_equity"]:
                        mlflow.log_metric(f"{prefix}_{metric}", r.get(metric, 0))
                    mlflow.log_metric(f"{prefix}_trades", r.get("n_trades", 0))
                    mlflow.log_metric(f"{prefix}_gates", r.get("gates_passed", 0))
                    mlflow.log_metric(f"{prefix}_rebals", r.get("rebal_count", 0))
                    mlflow.log_metric(f"{prefix}_avg_gap", r.get("avg_gap_days", 0))

                mlflow.log_params({
                    "capital": CAP, "dte": DTE, "spread_pct": SPREAD_PCT,
                    "top_k": TOP_K, "haircut": DEFAULT_HAIRCUT,
                    "commission_rt": COMMISSION_RT_SPREAD,
                    "regime_threshold": REGIME_BULL_THRESHOLD,
                    "hold_to_expiry": True, "n_features": len(ALL_FEATURES),
                    "wf_train_periods": WF_TRAIN_PERIODS, "n_variants": len(variants),
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
