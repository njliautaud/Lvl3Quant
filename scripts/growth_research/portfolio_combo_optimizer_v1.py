#!/usr/bin/env python3
"""
Portfolio Combo Optimizer v1 -- Multi-Strategy Portfolio Diversification Test
============================================================================

Tests whether COMBINING sector rotation strategies improves risk-adjusted returns
through diversification. 5 variants:

  A: V9.1 Only (baseline) -- 3 bull + 3 bear, 17 momentum features, $645
  B: Earnings Only -- 14 earnings features (constructed from price data), $645
  C: 50/50 Split -- $322.50 momentum + $322.50 earnings, independent models
  D: Signal Average -- 0.5*mom_rank + 0.5*earn_rank, single portfolio, $645
  E: Risk Parity Combo -- V9.1 spreads ($450) + TLT/GLD risk parity ($195)

Config: 11 sector ETFs, DTE=28, 3% OTM, adaptive max($3,3%),
$2.60 commission, 15% BS haircut, LGBM 52-week walk-forward.

NO chain data -- pure BS pricing with 15% haircut (Razer compatible).

Output: output/growth_research/portfolio_combo_optimizer_v1/results.json
MLflow experiment: portfolio_combo_optimizer_v1
"""

import json
import os
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

# ==============================================================
# ENVIRONMENT DETECTION (Jupiter / Neptune / Razer)
# ==============================================================

import platform
_IS_WINDOWS = platform.system() == "Windows"

_JUPITER_BASE = Path("/home/jupiter/Lvl3Quant")
_NEPTUNE_BASE = Path("/home/nick/Lvl3Quant")
_RAZER_BASE = Path("C:/Users/claude/Lvl3Quant")

if _IS_WINDOWS and _RAZER_BASE.exists():
    BASE = _RAZER_BASE
    fprint(f"Running on Razer: {BASE}")
elif _NEPTUNE_BASE.exists():
    BASE = _NEPTUNE_BASE
    fprint(f"Running on Neptune: {BASE}")
elif _JUPITER_BASE.exists():
    BASE = _JUPITER_BASE
    fprint(f"Running on Jupiter: {BASE}")
else:
    BASE = Path.cwd()
    fprint(f"Running on unknown host: {BASE}")

sys.path.insert(0, str(BASE))

# Import BS pricer -- NO chain data on Razer
from research.tools.options_pricer import (
    price_bull_call_spread, price_bear_put_spread, COMMISSION_RT_SPREAD,
)
from research.tools.adversarial_validator import validate_trades

OUTPUT_DIR = BASE / "output" / "growth_research" / "portfolio_combo_optimizer_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
TOP_K = 3
DTE = 28
WF_TRAIN_PERIODS = 52

COST_WIDTH_MAX = 0.50
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "portfolio_combo_optimizer_v1"
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_URI

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen(MLFLOW_URI, timeout=3)
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    MLFLOW_OK = True
    fprint(f"MLflow connected: {MLFLOW_URI}")
except Exception:
    fprint("MLflow unavailable -- will skip logging")

# -- Feature sets --
MOMENTUM_FEATURES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "sharpe_63d", "pct_52w_high", "mom_accel",
    "pct_pos_months_12m", "sortino_63d", "calmar_1y", "up_capture",
    "trend_r2_63d", "trend_slope_63d",
    "sector_spy_beta_63d", "cross_sector_dispersion",
]
assert len(MOMENTUM_FEATURES) == 17

EARNINGS_FEATURES = [
    "earnings_pct_reporting_2w",
    "earnings_avg_surprise",
    "earnings_post_drift",
    "earnings_recent_surprise_quality",
    "earnings_beat_rate_1m",
    "earnings_vol_impact",
    # 6 interaction terms (earnings x momentum)
    "earn_x_ret_21d",
    "earn_x_mom_accel",
    "earn_surprise_x_trend_slope",
    "earn_drift_x_sharpe",
    "earn_vol_x_beta",
    "earn_beat_x_up_capture",
    # 2 additional pure earnings
    "earnings_gap_consistency",
    "earnings_calendar_proximity",
]
assert len(EARNINGS_FEATURES) == 14


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
    volume = raw["Volume"] if mi else raw[["Volume"]]
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
        high.columns = high.columns.get_level_values(-1)
        low.columns = low.columns.get_level_values(-1)
        volume.columns = volume.columns.get_level_values(-1)
    close = close.ffill(); high = high.ffill(); low = low.ffill(); volume = volume.ffill()
    rename_map = {"^VIX": "VIX", "^VIX3M": "VIX3M"}
    close = close.rename(columns=rename_map)
    high = high.rename(columns=rename_map)
    low = low.rename(columns=rename_map)
    volume = volume.rename(columns=rename_map)
    fprint(f"Data: {len(close)} days, {close.index[0].date()} to {close.index[-1].date()}")
    return close, high, low, volume


# ==============================================================
# MOMENTUM FEATURES (V9.1 -- 17 features)
# ==============================================================

def compute_momentum_features(px, spy_slice):
    """Compute 17 momentum features for a single sector at a single date."""
    if len(px) < 260:
        return None
    f = {}
    for lb, nm in [(5, "ret_5d"), (10, "ret_10d"), (21, "ret_21d"),
                   (63, "ret_63d"), (126, "ret_126d"), (252, "ret_252d")]:
        f[nm] = float(px.iloc[-1] / px.iloc[-lb] - 1) if len(px) > lb else 0.0
    rets = px.pct_change().dropna()
    f["sharpe_63d"] = float(rets.iloc[-63:].mean() / (rets.iloc[-63:].std() + 1e-10) * np.sqrt(252)) if len(rets) > 63 else 0.0
    f["pct_52w_high"] = float(px.iloc[-1] / px.iloc[-252:].max())
    f["mom_accel"] = f["ret_21d"] - f["ret_63d"] / 3
    monthly = rets.resample("ME").sum()
    f["pct_pos_months_12m"] = float((monthly.iloc[-12:] > 0).mean()) if len(monthly) >= 12 else 0.5
    dr = rets.iloc[-63:][rets.iloc[-63:] < 0]
    f["sortino_63d"] = float(rets.iloc[-63:].mean() / (dr.std() + 1e-10) * np.sqrt(252)) if len(dr) > 3 else 0.0
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
    """Compute sector_spy_beta_63d and cross_sector_dispersion."""
    f = {}
    spy = close_df["SPY"].iloc[:dt_idx + 1].dropna()
    sector_px = close_df[sector_ticker].iloc[:dt_idx + 1].dropna() if sector_ticker in close_df.columns else None
    if spy is None or len(spy) < 63:
        return {"sector_spy_beta_63d": 1.0, "cross_sector_dispersion": 0.01}
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
# EARNINGS FEATURES (14 features -- constructed from price data)
# ==============================================================

def detect_earnings_dates(close_df, volume_df, sector_ticker, up_to_idx):
    """Detect probable earnings dates from price/volume anomalies.

    Earnings proxy: days where |gap| > 3% OR vol_mult > 3x 20d avg.
    This avoids needing actual earnings calendar data.
    """
    if sector_ticker not in close_df.columns:
        return pd.DatetimeIndex([])

    px = close_df[sector_ticker].iloc[:up_to_idx + 1].dropna()
    if len(px) < 60:
        return pd.DatetimeIndex([])

    # Gap detection
    gaps = px.pct_change().abs()

    # Volume spike detection
    vol = None
    if sector_ticker in volume_df.columns:
        vol = volume_df[sector_ticker].iloc[:up_to_idx + 1].dropna()
        vol_ma20 = vol.rolling(20).mean()
        vol_mult = vol / (vol_ma20 + 1)
    else:
        vol_mult = pd.Series(0, index=px.index)

    # Earnings = gap > 3% OR volume multiplier > 3
    earnings_mask = (gaps > 0.03) | (vol_mult > 3.0)
    # Cluster nearby dates (within 3 days) into single events
    earnings_dates = px.index[earnings_mask]
    if len(earnings_dates) == 0:
        return pd.DatetimeIndex([])

    # De-duplicate: keep only first date in each 5-day cluster
    deduped = [earnings_dates[0]]
    for d in earnings_dates[1:]:
        if (d - deduped[-1]).days > 5:
            deduped.append(d)
    return pd.DatetimeIndex(deduped)


def compute_earnings_features(sector_ticker, dt_idx, close_df, volume_df):
    """Compute 14 earnings-based features for a single sector at a date.

    Features are constructed entirely from price data -- no external earnings API.
    """
    f = {}
    px = close_df[sector_ticker].iloc[:dt_idx + 1].dropna()
    if len(px) < 260:
        return None

    current_date = close_df.index[dt_idx]
    earnings_dates = detect_earnings_dates(close_df, volume_df, sector_ticker, dt_idx)

    if len(earnings_dates) == 0:
        # No detected earnings -- return zeros
        for feat in EARNINGS_FEATURES:
            f[feat] = 0.0
        return f

    # 1. earnings_pct_reporting_2w: fraction of detected earnings within last 2 weeks
    recent_2w = earnings_dates[earnings_dates >= (current_date - pd.Timedelta(days=14))]
    # Proxy: among all sectors, what % had earnings in last 2w
    # (computed per-sector, so this is 0 or 1 for single ticker -- aggregate later)
    f["earnings_pct_reporting_2w"] = 1.0 if len(recent_2w) > 0 else 0.0

    # 2. earnings_avg_surprise: average 1-day post-earnings return
    post_returns = []
    for ed in earnings_dates[-8:]:  # Last 8 earnings events
        ed_idx = close_df.index.get_indexer([ed], method="ffill")[0]
        if ed_idx + 1 < len(px):
            post_ret = float(px.iloc[min(ed_idx + 1, len(px) - 1)] / px.iloc[ed_idx] - 1)
            post_returns.append(post_ret)
    f["earnings_avg_surprise"] = float(np.mean(post_returns)) if post_returns else 0.0

    # 3. earnings_post_drift: 5-day post-earnings drift
    drifts = []
    for ed in earnings_dates[-8:]:
        ed_idx = close_df.index.get_indexer([ed], method="ffill")[0]
        if ed_idx + 5 < len(px):
            drift = float(px.iloc[min(ed_idx + 5, len(px) - 1)] / px.iloc[ed_idx] - 1)
            drifts.append(drift)
    f["earnings_post_drift"] = float(np.mean(drifts)) if drifts else 0.0

    # 4. earnings_recent_surprise_quality: quality of most recent batch
    if len(post_returns) >= 2:
        recent = post_returns[-3:]  # Last 3 earnings
        f["earnings_recent_surprise_quality"] = float(np.mean(recent) / (np.std(recent) + 1e-6))
    else:
        f["earnings_recent_surprise_quality"] = 0.0

    # 5. earnings_beat_rate_1m: % of recent earnings with positive surprise
    if post_returns:
        f["earnings_beat_rate_1m"] = float(sum(1 for r in post_returns if r > 0) / len(post_returns))
    else:
        f["earnings_beat_rate_1m"] = 0.5

    # 6. earnings_vol_impact: average vol expansion around earnings
    vol_impacts = []
    rets = px.pct_change().dropna()
    vol_20d = rets.rolling(20).std()
    for ed in earnings_dates[-8:]:
        ed_idx = close_df.index.get_indexer([ed], method="ffill")[0]
        if ed_idx < len(rets) and ed_idx >= 20:
            day_ret = abs(float(rets.iloc[ed_idx]))
            avg_vol = float(vol_20d.iloc[ed_idx - 1]) if ed_idx > 0 else 0.01
            vol_impacts.append(day_ret / (avg_vol + 1e-6))
    f["earnings_vol_impact"] = float(np.mean(vol_impacts)) if vol_impacts else 1.0

    # 7. earnings_gap_consistency: std of post-earnings returns (lower = more predictable)
    f["earnings_gap_consistency"] = float(np.std(post_returns)) if len(post_returns) > 1 else 0.05

    # 8. earnings_calendar_proximity: days until next expected earnings (quarterly cycle)
    if len(earnings_dates) >= 2:
        gaps_between = np.diff([d.toordinal() for d in earnings_dates[-4:]])
        avg_gap = float(np.mean(gaps_between)) if len(gaps_between) > 0 else 90
        last_earn = earnings_dates[-1]
        days_since = (current_date - last_earn).days
        days_until = max(0, avg_gap - days_since)
        f["earnings_calendar_proximity"] = days_until / 90.0  # Normalize to [0, ~1]
    else:
        f["earnings_calendar_proximity"] = 0.5

    return f


def compute_earnings_interaction_features(mom_feats, earn_feats):
    """Compute 6 interaction terms between momentum and earnings features."""
    f = {}
    # Interactions
    f["earn_x_ret_21d"] = earn_feats.get("earnings_avg_surprise", 0) * mom_feats.get("ret_21d", 0)
    f["earn_x_mom_accel"] = earn_feats.get("earnings_post_drift", 0) * mom_feats.get("mom_accel", 0)
    f["earn_surprise_x_trend_slope"] = earn_feats.get("earnings_recent_surprise_quality", 0) * mom_feats.get("trend_slope_63d", 0)
    f["earn_drift_x_sharpe"] = earn_feats.get("earnings_post_drift", 0) * mom_feats.get("sharpe_63d", 0)
    f["earn_vol_x_beta"] = earn_feats.get("earnings_vol_impact", 0) * mom_feats.get("sector_spy_beta_63d", 1)
    f["earn_beat_x_up_capture"] = earn_feats.get("earnings_beat_rate_1m", 0.5) * mom_feats.get("up_capture", 1)
    return f


# ==============================================================
# ATR COMPUTATION
# ==============================================================

def compute_atr_series(high, low, close, period=14):
    atr_dict = {}
    for tk in SECTORS:
        if tk in high.columns and tk in low.columns and tk in close.columns:
            h = high[tk].dropna(); l = low[tk].dropna(); c = close[tk].dropna()
            common = h.index.intersection(l.index).intersection(c.index)
            if len(common) > period:
                tr1 = h.loc[common] - l.loc[common]
                tr2 = (h.loc[common] - c.loc[common].shift(1)).abs()
                tr3 = (l.loc[common] - c.loc[common].shift(1)).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr_dict[tk] = tr.ewm(alpha=1/period, min_periods=period).mean()
    return atr_dict


# ==============================================================
# STRIKES + EXECUTION (BS only -- no chain data)
# ==============================================================

def compute_strikes(S, direction):
    if direction == "bull":
        K1 = round(S * 1.03, 2)
        pct_w = K1 * 0.03
        w = max(3.0, pct_w)
        K2 = round(K1 + w, 2)
    else:
        K2 = round(S * 0.97, 2)
        pct_w = K2 * 0.03
        w = max(3.0, pct_w)
        K1 = round(K2 - w, 2)
    if K2 <= K1:
        K2 = K1 + 0.50
    return K1, K2


def execute_trade(tk, dt, direction, close, atr_dict, vix_val, equity, max_pos):
    """Execute a single spread trade using BS pricing with 15% haircut. No chain data."""
    if tk not in close.columns or tk not in atr_dict:
        return None
    S = float(close[tk].loc[dt])
    di = close.index.get_loc(dt)
    ei = min(di + DTE, len(close) - 1)
    if ei <= di:
        return None

    av = float(atr_dict[tk].loc[dt]) if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]) else S * 0.015
    K1, K2 = compute_strikes(S, direction)

    try:
        if direction == "bull":
            entry_cost_ps, _ = price_bull_call_spread(S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=vix_val)
        else:
            entry_cost_ps, _ = price_bear_put_spread(S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=vix_val)
    except Exception:
        return None

    if entry_cost_ps is None or entry_cost_ps <= 0:
        return None

    spread_width = abs(K2 - K1)
    cwr = entry_cost_ps / max(spread_width, 0.01)
    if cwr > COST_WIDTH_MAX:
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

    return {
        "pnl": round(pnl, 2), "entry_cost_ps": round(entry_cost_ps, 4),
        "total_cost": round(total_cost, 2),
        "K1": K1, "K2": K2, "S_entry": round(S, 2), "S_exit": round(Se, 2),
        "intrinsic": round(intrinsic, 4), "spread_width": round(spread_width, 2),
        "cost_width_ratio": round(cwr, 3),
    }


# ==============================================================
# REBALANCE DATES
# ==============================================================

def get_weekly_fridays(close):
    return pd.DatetimeIndex(
        close.index.to_series().resample("W-FRI").last().dropna().values
    )


# ==============================================================
# LGBM WALK-FORWARD RANKING
# ==============================================================

def build_feature_records(close, high, low, volume, rebal_dates, feature_set="momentum"):
    """Build feature records for all rebalance dates.

    feature_set: "momentum" (17 features), "earnings" (14 features), or "both" (31 features)
    """
    records = []
    sector_cols = [c for c in SECTORS if c in close.columns]
    spy = close["SPY"]

    if feature_set == "momentum":
        feat_names = MOMENTUM_FEATURES
    elif feature_set == "earnings":
        feat_names = EARNINGS_FEATURES
    else:  # "both"
        feat_names = MOMENTUM_FEATURES + EARNINGS_FEATURES

    for dt in rebal_dates:
        idx = close.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
            continue
        for tk in sector_cols:
            px = close[tk].iloc[:idx + 1].dropna()

            if feature_set in ("momentum", "both"):
                mom = compute_momentum_features(px, spy.iloc[:idx + 1])
                if not mom:
                    continue
                cross = compute_cross_asset_features(tk, idx, close)
                mom.update(cross)
            else:
                mom = {}

            if feature_set in ("earnings", "both"):
                earn = compute_earnings_features(tk, idx, close, volume)
                if not earn:
                    continue
                if mom:
                    interactions = compute_earnings_interaction_features(mom, earn)
                    earn.update(interactions)
            else:
                earn = {}

            fi = min(idx + DTE, len(close) - 1)
            if fi <= idx:
                continue
            fwd_ret = float(close[tk].iloc[fi] / close[tk].iloc[idx] - 1)

            rec = {**mom, **earn, "date": dt, "ticker": tk, "fwd_ret": fwd_ret}
            records.append(rec)

    df = pd.DataFrame(records)
    for c in feat_names:
        if c not in df.columns:
            df[c] = 0.0
    df[feat_names] = df[feat_names].fillna(0.0)
    fprint(f"    {len(df)} records, {len(df['date'].unique())} dates [{feature_set}]")
    return df, feat_names


def walk_forward_lgbm_rank(df, feat_names):
    """Walk-forward LGBM ranking with 52-week sliding window."""
    import lightgbm as lgb
    if len(df) < 100:
        return {}
    df["rank_label"] = df.groupby("date")["fwd_ret"].rank(pct=True)
    dates = sorted(df["date"].unique())
    rankings = {}
    for i in range(WF_TRAIN_PERIODS, len(dates)):
        train_dates = dates[max(0, i - WF_TRAIN_PERIODS):i]
        test_date = dates[i]
        train_df = df[df["date"].isin(train_dates)]
        test_df = df[df["date"] == test_date].copy()
        if len(test_df) < 3 or len(train_df) < 50:
            continue
        Xt = np.nan_to_num(train_df[feat_names].values.astype(np.float32))
        yt = train_df["rank_label"].values.astype(np.float32)
        Xe = np.nan_to_num(test_df[feat_names].values.astype(np.float32))
        try:
            m = lgb.LGBMRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1,
            )
            m.fit(Xt, yt)
            preds = m.predict(Xe)
            test_df["score"] = preds
            rankings[test_date] = dict(zip(test_df["ticker"], test_df["score"]))
        except Exception:
            continue
    fprint(f"    {len(rankings)} ranking dates")
    return rankings


# ==============================================================
# SIMULATION ENGINE
# ==============================================================

def simulate_spread_variant(rankings, close, atr_dict, rebal_dates,
                            capital, variant_name=""):
    """Run backtest for a spread-based variant (A, B, or sub-components of C/D)."""
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    equity = capital
    trades = []
    equity_curve = []

    for dt in sorted(rebal_dates):
        if dt not in spy.index:
            continue
        ranking_date = None
        for rd in sorted(rankings.keys()):
            if rd <= dt:
                ranking_date = rd
        if ranking_date is None:
            continue

        scores = rankings[ranking_date]
        if not scores or len(scores) < 6:
            continue

        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0

        ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_asc = sorted(scores.items(), key=lambda x: x[1])
        bull_picks = [t for t, _ in ranked_desc[:TOP_K]]
        bear_picks = [t for t, _ in ranked_asc[:TOP_K]]

        all_picks = bull_picks + bear_picks
        max_pos = capital / len(all_picks)

        for direction, picks in [("bull", bull_picks), ("bear", bear_picks)]:
            for tk in picks:
                result = execute_trade(tk, dt, direction, close, atr_dict, cv, equity, max_pos)
                if result is not None:
                    equity += result["pnl"]
                    di = close.index.get_loc(dt)
                    ei = min(di + DTE, len(close) - 1)
                    sv = float(spy.loc[dt])
                    se = float(spy.iloc[ei]) if ei < len(spy) else sv
                    trades.append({
                        **result, "entry_date": str(dt.date()),
                        "exit_date": str(close.index[ei].date()),
                        "ticker": tk, "regime": "bull" if se >= sv else "bear",
                        "direction": direction, "vix": round(cv, 1),
                        "win": result["pnl"] > 0,
                    })
                    equity_curve.append({"date": str(dt.date()), "equity": round(equity, 2)})

    return trades, equity, equity_curve


def simulate_risk_parity(close, rebal_dates, capital=195.0):
    """Risk parity piece: equal-vol-weighted TLT + GLD as tail hedge.

    Rebalances weekly to maintain equal volatility contribution.
    """
    if "TLT" not in close.columns or "GLD" not in close.columns:
        fprint("  WARNING: TLT or GLD not in data -- skipping risk parity")
        return [], capital, []

    equity = capital
    trades = []
    equity_curve = []

    tlt = close["TLT"]
    gld = close["GLD"]

    for dt in sorted(rebal_dates):
        if dt not in tlt.index or dt not in gld.index:
            continue
        di = close.index.get_loc(dt)
        ei = min(di + 5, len(close) - 1)  # Weekly hold for risk parity
        if ei <= di:
            continue

        # Compute trailing 63-day vol for each
        if di < 63:
            continue
        tlt_px = tlt.iloc[:di + 1]
        gld_px = gld.iloc[:di + 1]
        tlt_vol = float(tlt_px.pct_change().iloc[-63:].std() * np.sqrt(252))
        gld_vol = float(gld_px.pct_change().iloc[-63:].std() * np.sqrt(252))

        if tlt_vol < 0.001 or gld_vol < 0.001:
            continue

        # Inverse vol weighting
        total_inv_vol = 1 / tlt_vol + 1 / gld_vol
        w_tlt = (1 / tlt_vol) / total_inv_vol
        w_gld = (1 / gld_vol) / total_inv_vol

        # Allocate capital
        tlt_alloc = equity * w_tlt
        gld_alloc = equity * w_gld

        # Compute returns over holding period
        tlt_ret = float(tlt.iloc[ei] / tlt.iloc[di] - 1)
        gld_ret = float(gld.iloc[ei] / gld.iloc[di] - 1)

        pnl = tlt_alloc * tlt_ret + gld_alloc * gld_ret
        equity += pnl

        trades.append({
            "pnl": round(pnl, 2),
            "entry_date": str(close.index[di].date()),
            "exit_date": str(close.index[ei].date()),
            "ticker": f"TLT({w_tlt:.0%})/GLD({w_gld:.0%})",
            "direction": "risk_parity",
            "regime": "hedge",
            "vix": 0.0,
            "win": pnl > 0,
            "w_tlt": round(w_tlt, 3),
            "w_gld": round(w_gld, 3),
            "tlt_ret": round(tlt_ret, 4),
            "gld_ret": round(gld_ret, 4),
        })
        equity_curve.append({"date": str(close.index[di].date()), "equity": round(equity, 2)})

    return trades, equity, equity_curve


# ==============================================================
# EQUITY CURVE CORRELATION
# ==============================================================

def compute_equity_curve_series(trades, initial_capital):
    """Convert trades list to a daily equity curve Series."""
    if not trades:
        return pd.Series(dtype=float)
    eq = initial_capital
    points = []
    for t in sorted(trades, key=lambda x: x["entry_date"]):
        eq += t["pnl"]
        points.append({"date": t["entry_date"], "equity": eq})
    if not points:
        return pd.Series(dtype=float)
    df = pd.DataFrame(points)
    df["date"] = pd.to_datetime(df["date"])
    # If multiple trades on same date, take last equity value
    df = df.groupby("date")["equity"].last()
    return df


def compute_correlation_matrix(curves_dict):
    """Compute return correlation between component equity curves."""
    returns_dict = {}
    for name, curve in curves_dict.items():
        if len(curve) > 10:
            returns_dict[name] = curve.pct_change().dropna()
    if len(returns_dict) < 2:
        return None
    combined = pd.DataFrame(returns_dict)
    combined = combined.dropna()
    if len(combined) < 10:
        return None
    return combined.corr()


# ==============================================================
# METRICS COMPUTATION
# ==============================================================

def compute_calmar(trades, initial_capital):
    """Compute Calmar ratio = CAGR / MaxDD."""
    if not trades or len(trades) < 2:
        return 0.0
    eq = initial_capital
    equity_series = [eq]
    dates = []
    for t in sorted(trades, key=lambda x: x["entry_date"]):
        eq += t["pnl"]
        equity_series.append(eq)
        dates.append(t["entry_date"])
    if not dates:
        return 0.0
    equity_arr = np.array(equity_series)
    peak = np.maximum.accumulate(equity_arr)
    dd = (equity_arr - peak) / (peak + 1e-10)
    max_dd = abs(float(np.min(dd)))
    if max_dd < 1e-6:
        return 10.0  # Cap at 10 if no drawdown
    # CAGR
    first_date = pd.Timestamp(dates[0])
    last_date = pd.Timestamp(dates[-1])
    years = max((last_date - first_date).days / 365.25, 0.1)
    cagr = (equity_arr[-1] / initial_capital) ** (1 / years) - 1
    return round(cagr / max_dd, 3)


def monte_carlo_ci(trades, n_bootstrap=1000, seed=42):
    if len(trades) < 10:
        return None
    pnls = np.array([t["pnl"] for t in trades])
    rng = np.random.RandomState(seed)
    sharpes = []
    for _ in range(n_bootstrap):
        sample = rng.choice(pnls, size=len(pnls), replace=True)
        sh = float(np.mean(sample) / (np.std(sample) + 1e-10) * np.sqrt(52))
        sharpes.append(sh)
    sharpes = np.array(sharpes)
    return {
        "mean": round(float(np.mean(sharpes)), 3),
        "ci_95_low": round(float(np.percentile(sharpes, 2.5)), 3),
        "ci_95_high": round(float(np.percentile(sharpes, 97.5)), 3),
    }


# ==============================================================
# MAIN
# ==============================================================

def main():
    t0 = datetime.now()
    fprint("=" * 100)
    fprint(f"PORTFOLIO COMBO OPTIMIZER v1 -- {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 100)
    fprint(f"Capital: ${CAP:.0f} | DTE: {DTE} | Commission: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"Universe: {len(SECTORS)} sector ETFs | Top/Bottom K: {TOP_K}")
    fprint(f"Walk-forward: {WF_TRAIN_PERIODS}-week sliding window")
    fprint(f"Pricing: BS with 15% haircut (no chain data)")
    fprint()

    fprint("Downloading price data...")
    close, high, low, volume = download_data()
    atr_dict = compute_atr_series(high, low, close)
    weekly_dates = get_weekly_fridays(close)
    fprint(f"Weekly rebalance dates: {len(weekly_dates)}")

    spy_close = close["SPY"]

    # ===========================================================
    # BUILD LGBM RANKINGS FOR EACH FEATURE SET
    # ===========================================================

    fprint(f"\n{'=' * 80}")
    fprint("BUILDING MOMENTUM RANKINGS (17 features)")
    fprint(f"{'=' * 80}")
    mom_df, mom_feats = build_feature_records(close, high, low, volume, weekly_dates, "momentum")
    mom_rankings = walk_forward_lgbm_rank(mom_df, mom_feats)

    fprint(f"\n{'=' * 80}")
    fprint("BUILDING EARNINGS RANKINGS (14 features)")
    fprint(f"{'=' * 80}")
    earn_df, earn_feats = build_feature_records(close, high, low, volume, weekly_dates, "earnings")
    earn_rankings = walk_forward_lgbm_rank(earn_df, earn_feats)

    if len(mom_rankings) < 20:
        fprint(f"ERROR: Only {len(mom_rankings)} momentum ranking dates -- insufficient data")
        return

    # ===========================================================
    # BUILD COMBINED RANKINGS FOR VARIANT D
    # ===========================================================

    fprint(f"\n{'=' * 80}")
    fprint("BUILDING SIGNAL-AVERAGE RANKINGS (D)")
    fprint(f"{'=' * 80}")
    combined_rankings = {}
    common_dates = set(mom_rankings.keys()) & set(earn_rankings.keys())
    for dt in common_dates:
        mom_scores = mom_rankings[dt]
        earn_scores = earn_rankings[dt]
        combined = {}
        all_tickers = set(mom_scores.keys()) | set(earn_scores.keys())
        for tk in all_tickers:
            ms = mom_scores.get(tk, 0.5)
            es = earn_scores.get(tk, 0.5)
            combined[tk] = 0.5 * ms + 0.5 * es
        combined_rankings[dt] = combined
    fprint(f"    {len(combined_rankings)} combined ranking dates")

    # ===========================================================
    # SIMULATE ALL 5 VARIANTS
    # ===========================================================

    all_results = {}
    all_trades = {}
    component_curves = {}

    # -- VARIANT A: V9.1 Only (baseline) --
    fprint(f"\n{'~' * 90}")
    fprint("VARIANT A: V9.1 Momentum Only (baseline) -- $645")
    fprint(f"{'~' * 90}")
    a_trades, a_eq, a_curve = simulate_spread_variant(
        mom_rankings, close, atr_dict, weekly_dates, CAP, "A"
    )
    all_trades["A_momentum_only"] = a_trades
    component_curves["A_momentum"] = compute_equity_curve_series(a_trades, CAP)

    # -- VARIANT B: Earnings Only --
    fprint(f"\n{'~' * 90}")
    fprint("VARIANT B: Earnings Only -- $645")
    fprint(f"{'~' * 90}")
    b_trades, b_eq, b_curve = simulate_spread_variant(
        earn_rankings, close, atr_dict, weekly_dates, CAP, "B"
    )
    all_trades["B_earnings_only"] = b_trades
    component_curves["B_earnings"] = compute_equity_curve_series(b_trades, CAP)

    # -- VARIANT C: 50/50 Split --
    fprint(f"\n{'~' * 90}")
    fprint("VARIANT C: 50/50 Split -- $322.50 momentum + $322.50 earnings")
    fprint(f"{'~' * 90}")
    c_mom_trades, c_mom_eq, _ = simulate_spread_variant(
        mom_rankings, close, atr_dict, weekly_dates, 322.50, "C_mom"
    )
    c_earn_trades, c_earn_eq, _ = simulate_spread_variant(
        earn_rankings, close, atr_dict, weekly_dates, 322.50, "C_earn"
    )
    c_trades = c_mom_trades + c_earn_trades
    c_eq = c_mom_eq + c_earn_eq - 322.50  # Avoid double-counting initial capital
    all_trades["C_5050_split"] = c_trades
    component_curves["C_mom_half"] = compute_equity_curve_series(c_mom_trades, 322.50)
    component_curves["C_earn_half"] = compute_equity_curve_series(c_earn_trades, 322.50)

    # -- VARIANT D: Signal Average --
    fprint(f"\n{'~' * 90}")
    fprint("VARIANT D: Signal Average (0.5*mom + 0.5*earn rank) -- $645")
    fprint(f"{'~' * 90}")
    d_trades, d_eq, d_curve = simulate_spread_variant(
        combined_rankings, close, atr_dict, weekly_dates, CAP, "D"
    )
    all_trades["D_signal_average"] = d_trades
    component_curves["D_combined"] = compute_equity_curve_series(d_trades, CAP)

    # -- VARIANT E: Risk Parity Combo --
    fprint(f"\n{'~' * 90}")
    fprint("VARIANT E: Risk Parity Combo -- $450 V9.1 spreads + $195 TLT/GLD")
    fprint(f"{'~' * 90}")
    e_spread_trades, e_spread_eq, _ = simulate_spread_variant(
        mom_rankings, close, atr_dict, weekly_dates, 450.0, "E_spread"
    )
    e_rp_trades, e_rp_eq, _ = simulate_risk_parity(close, weekly_dates, 195.0)
    e_trades = e_spread_trades + e_rp_trades
    e_eq = e_spread_eq + e_rp_eq - 195.0  # Avoid double-counting
    all_trades["E_risk_parity_combo"] = e_trades
    component_curves["E_spreads"] = compute_equity_curve_series(e_spread_trades, 450.0)
    component_curves["E_rp"] = compute_equity_curve_series(e_rp_trades, 195.0)

    # ===========================================================
    # VALIDATE ALL VARIANTS
    # ===========================================================

    VARIANT_NAMES = {
        "A_momentum_only": ("A: V9.1 Momentum Only", CAP),
        "B_earnings_only": ("B: Earnings Only", CAP),
        "C_5050_split": ("C: 50/50 Split", CAP),
        "D_signal_average": ("D: Signal Average", CAP),
        "E_risk_parity_combo": ("E: Risk Parity Combo", CAP),
    }

    for vname, (desc, cap) in VARIANT_NAMES.items():
        trades = all_trades.get(vname, [])
        fprint(f"\n{'-' * 80}")
        fprint(f"VALIDATING {desc} ({len(trades)} trades)")
        fprint(f"{'-' * 80}")

        if len(trades) < 5:
            fprint(f"  Only {len(trades)} trades -- insufficient")
            all_results[vname] = {"error": "too_few_trades", "n_trades": len(trades)}
            continue

        result = validate_trades(trades, initial_capital=cap, spy_prices=spy_close, strategy_name=desc)
        result.print_summary()

        # Side analysis
        for side in ["bull", "bear"]:
            st = [t for t in trades if t.get("direction") == side]
            if st:
                pnls = [t["pnl"] for t in st]
                wr = sum(1 for p in pnls if p > 0) / len(pnls)
                fprint(f"  {side}: {len(st)} trades, WR {wr:.1%}, PnL ${sum(pnls):,.0f}")

        # Risk parity side (for E)
        rp_trades = [t for t in trades if t.get("direction") == "risk_parity"]
        if rp_trades:
            rp_pnls = [t["pnl"] for t in rp_trades]
            fprint(f"  risk_parity: {len(rp_trades)} trades, PnL ${sum(rp_pnls):,.0f}")

        mc = monte_carlo_ci(trades)
        calmar = compute_calmar(trades, cap)

        rd = result.to_dict()
        all_results[vname] = {
            **rd, "calmar": calmar,
            "monte_carlo": mc,
            "description": desc,
        }

    # ===========================================================
    # CORRELATION ANALYSIS
    # ===========================================================

    fprint(f"\n{'=' * 100}")
    fprint("COMPONENT EQUITY CURVE CORRELATION")
    fprint(f"{'=' * 100}")

    corr_matrix = compute_correlation_matrix(component_curves)
    if corr_matrix is not None:
        fprint("\nReturn correlation matrix:")
        fprint(corr_matrix.round(3).to_string())
        # Key diversification metric
        key_pairs = [
            ("A_momentum", "B_earnings"),
            ("C_mom_half", "C_earn_half"),
            ("E_spreads", "E_rp"),
        ]
        fprint("\nKey diversification pairs:")
        for p1, p2 in key_pairs:
            if p1 in corr_matrix.columns and p2 in corr_matrix.columns:
                corr_val = corr_matrix.loc[p1, p2]
                div_benefit = "GOOD" if corr_val < 0.5 else ("MODERATE" if corr_val < 0.7 else "LOW")
                fprint(f"  {p1} vs {p2}: r={corr_val:.3f} ({div_benefit} diversification)")

    # ===========================================================
    # COMPARISON TABLE
    # ===========================================================

    fprint(f"\n{'=' * 140}")
    fprint("COMPARISON -- ALL VARIANTS")
    fprint(f"{'=' * 140}")
    fprint(f"  {'Variant':<30} {'N':>5} {'Sharpe':>7} {'Sort':>7} {'Calmar':>7} "
           f"{'WR':>6} {'PF':>6} {'MDD':>7} {'Gate':>5}")
    fprint(f"  {'-' * 110}")
    for vn in VARIANT_NAMES:
        r = all_results.get(vn)
        if not r or r.get("error"):
            continue
        fprint(f"  {vn:<30} {r['n_trades']:>5} {r['sharpe']:>7.2f} {r['sortino']:>7.2f} "
               f"{r.get('calmar', 0):>7.2f} "
               f"{r['win_rate']*100:>5.1f}% {r['profit_factor']:>5.2f} "
               f"{r['max_dd']*100:>6.1f}% {r['gates_passed']}/{r['gates_total']}")

    # Monte Carlo
    fprint(f"\n{'=' * 100}")
    fprint("MONTE CARLO BOOTSTRAP (1000 resamples)")
    fprint(f"{'=' * 100}")
    fprint(f"  {'Variant':<30} {'Mean':>8} {'95% CI':>20}")
    fprint(f"  {'-' * 65}")
    for vn in VARIANT_NAMES:
        r = all_results.get(vn)
        if not r or r.get("error") or not r.get("monte_carlo"):
            continue
        mc = r["monte_carlo"]
        fprint(f"  {vn:<30} {mc['mean']:>8.2f} [{mc['ci_95_low']:.2f}, {mc['ci_95_high']:.2f}]")

    # 5-Gate Summary
    fprint(f"\n{'=' * 100}")
    fprint("5-GATE VALIDATION SUMMARY")
    fprint(f"{'=' * 100}")
    for vn in VARIANT_NAMES:
        r = all_results.get(vn)
        if not r or r.get("error"):
            fprint(f"  [SKIP] {vn}: {r.get('error', 'no data')}")
            continue
        status = "PASS" if r["all_passed"] else "FAIL"
        fprint(f"  [{status}] {vn}: {r['gates_passed']}/{r['gates_total']} gates")
        if r.get("gates"):
            for g in r["gates"]:
                gs = "PASS" if g["passed"] else "FAIL"
                fprint(f"         [{gs}] {g['name']}: {g['metric_name']}={g['metric_value']:.4f}")

    # ===========================================================
    # VERDICT
    # ===========================================================

    fprint(f"\n{'=' * 100}")
    fprint("VERDICT -- DOES COMBINING STRATEGIES ADD VALUE?")
    fprint(f"{'=' * 100}")

    baseline = all_results.get("A_momentum_only")
    if baseline and not baseline.get("error"):
        fprint(f"\n  Baseline (A -- V9.1 momentum): Sharpe {baseline['sharpe']:.2f}, "
               f"PF {baseline['profit_factor']:.2f}, MDD {baseline['max_dd']*100:.1f}%, "
               f"Calmar {baseline.get('calmar', 0):.2f}")

        for vn in ["B_earnings_only", "C_5050_split", "D_signal_average", "E_risk_parity_combo"]:
            r = all_results.get(vn)
            if not r or r.get("error"):
                continue
            sh_diff = r["sharpe"] - baseline["sharpe"]
            mdd_diff = (r["max_dd"] - baseline["max_dd"]) * 100
            cal_diff = r.get("calmar", 0) - baseline.get("calmar", 0)
            fprint(f"\n  {VARIANT_NAMES[vn][0]}:")
            fprint(f"    Sharpe: {r['sharpe']:.2f} ({sh_diff:+.2f} vs baseline)")
            fprint(f"    MDD:    {r['max_dd']*100:.1f}% ({mdd_diff:+.1f}pp)")
            fprint(f"    Calmar: {r.get('calmar', 0):.2f} ({cal_diff:+.2f})")
            if sh_diff > 0.1 and mdd_diff < 0:
                fprint(f"    -> STRONG IMPROVEMENT: better Sharpe AND lower drawdown")
            elif sh_diff > 0.1:
                fprint(f"    -> IMPROVEMENT: +{sh_diff:.2f} Sharpe")
            elif sh_diff < -0.1:
                fprint(f"    -> DEGRADATION: {sh_diff:.2f} Sharpe")
            else:
                fprint(f"    -> NEUTRAL: within 0.1 Sharpe of baseline")

    # Best overall
    best_vn = None
    best_sh = -999
    for vn in VARIANT_NAMES:
        r = all_results.get(vn)
        if not r or r.get("error"):
            continue
        if r["sharpe"] > best_sh:
            best_sh = r["sharpe"]
            best_vn = vn

    if best_vn:
        r = all_results[best_vn]
        fprint(f"\n  BEST VARIANT: {VARIANT_NAMES[best_vn][0]}")
        fprint(f"    Sharpe {r['sharpe']:.2f}, Sortino {r['sortino']:.2f}, "
               f"PF {r['profit_factor']:.2f}, WR {r['win_rate']*100:.1f}%, "
               f"MDD {r['max_dd']*100:.1f}%, Calmar {r.get('calmar', 0):.2f}")
        fprint(f"    Gates: {r['gates_passed']}/{r['gates_total']}")

    # Diversification verdict
    if corr_matrix is not None and "A_momentum" in corr_matrix.columns and "B_earnings" in corr_matrix.columns:
        corr_ab = corr_matrix.loc["A_momentum", "B_earnings"]
        fprint(f"\n  DIVERSIFICATION VERDICT:")
        fprint(f"    Momentum vs Earnings correlation: {corr_ab:.3f}")
        if corr_ab < 0.3:
            fprint(f"    -> STRONG diversification benefit -- combining adds real value")
        elif corr_ab < 0.6:
            fprint(f"    -> MODERATE diversification -- some benefit to combining")
        else:
            fprint(f"    -> WEAK diversification -- strategies are correlated, limited combo benefit")

    # -- Save --
    results_file = OUTPUT_DIR / "results.json"

    # Add correlation data to results
    save_results = dict(all_results)
    if corr_matrix is not None:
        save_results["_correlation_matrix"] = corr_matrix.round(4).to_dict()

    with open(results_file, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_file}")

    for vn, trades in all_trades.items():
        tf = OUTPUT_DIR / f"trades_{vn}.json"
        with open(tf, "w") as f:
            json.dump(trades, f, indent=1, default=str)

    # MLflow logging
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"combo_opt_{t0.strftime('%Y%m%d_%H%M')}"):
                mlflow.log_param("capital", CAP)
                mlflow.log_param("dte", DTE)
                mlflow.log_param("top_k", TOP_K)
                mlflow.log_param("wf_train_weeks", WF_TRAIN_PERIODS)
                mlflow.log_param("n_sectors", len(SECTORS))
                mlflow.log_param("n_variants", 5)
                mlflow.log_param("momentum_features", len(MOMENTUM_FEATURES))
                mlflow.log_param("earnings_features", len(EARNINGS_FEATURES))
                for vn, r in all_results.items():
                    if vn.startswith("_") or r.get("error"):
                        continue
                    mlflow.log_metric(f"{vn}_sharpe", r["sharpe"])
                    mlflow.log_metric(f"{vn}_sortino", r["sortino"])
                    mlflow.log_metric(f"{vn}_wr", r["win_rate"])
                    mlflow.log_metric(f"{vn}_pf", r["profit_factor"])
                    mlflow.log_metric(f"{vn}_mdd", r["max_dd"])
                    mlflow.log_metric(f"{vn}_gates", r["gates_passed"])
                    mlflow.log_metric(f"{vn}_calmar", r.get("calmar", 0))
                if corr_matrix is not None and "A_momentum" in corr_matrix.columns and "B_earnings" in corr_matrix.columns:
                    mlflow.log_metric("corr_mom_earn", float(corr_matrix.loc["A_momentum", "B_earnings"]))
                mlflow.log_artifact(str(results_file))
            fprint(f"MLflow logged to experiment '{EXPERIMENT_NAME}'")
        except Exception as e:
            fprint(f"MLflow error: {e}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nCompleted in {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
