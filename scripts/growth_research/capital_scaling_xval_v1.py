#!/usr/bin/env python3
"""
Capital Scaling Cross-Validation v1 — Does the Strategy Scale?
================================================================

Based on production_v4_honest_test.py. Tests V6 config at 6 capital levels
to answer: does Sharpe degrade as the account grows from $645 to $100K?

V6 Config (applied identically at all levels):
  - Weekly rebalance (W-FRI)
  - 2% OTM moneyness (buy call at price*1.02, sell at price*1.05)
  - Bull call spreads when VIX>20 (regime>0.4)
  - Pair trades (bull+bear) when VIX<20 (regime<0.4)
  - Top-3 sectors long, bottom-3 short
  - Hold to expiry, intrinsic only
  - 15% entry haircut, $2.60 commission, 3% spread width, DTE=21

Capital Levels:
  A) $645   — $200/trade max ($100/leg for pairs). Current baseline.
  B) $2,000 — $400/trade max ($200/leg).
  C) $5,000 — $1,000/trade max ($500/leg).
  D) $20,000 — $2,000/trade max ($1,000/leg).
  E) $50,000 — $5,000/trade max ($2,500/leg).
  F) $100,000 — $10,000/trade max ($5,000/leg).

Position sizing rule: max_pos = capital * 0.31 (proportional).
Pair leg max = max_pos / 2.
Commission stays $2.60/spread (fixed).
Capital compounds (gains increase position sizes).

KEY HYPOTHESIS: Commission becomes proportionally smaller at higher capital,
which should HELP Sharpe. ATR-based BS pricing and 15% haircut are proportional,
so those don't change. Net effect should be positive at scale.

Production LGBM walk-forward, 21 features, GRU regime. Adversarial validation
for each capital level.
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


# ── Path setup (Neptune or Jupiter) ──
_hostname = os.uname().nodename.lower()
if 'neptune' in _hostname or 'nick' in str(os.path.expanduser('~')):
    BASE = Path('/home/nick/Lvl3Quant')
else:
    BASE = Path('/home/jupiter/Lvl3Quant')

sys.path.insert(0, str(BASE))
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
OUTPUT_DIR = BASE / "output" / "growth_research" / "capital_scaling_xval_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]

# V6 config
DTE = 21
SPREAD_PCT = 3.0           # 3% spread width
OTM_PCT = 2.0              # 2% OTM moneyness
TOP_K = 3
REGIME_BULL_THRESHOLD = 0.4  # VIX>20 proxy: regime>0.4 => bull call spreads
REGIME_BEAR_THRESHOLD = 0.4  # VIX<20 proxy: regime<=0.4 => pair trades
VIX_REGIME_CUTOFF = 20.0     # Direct VIX cutoff for regime switching
POS_SIZE_FRAC = 0.31         # max_pos = capital * 0.31

# Capital levels to test
CAPITAL_LEVELS = [
    {"name": "A_645_baseline",  "capital": 645.0,    "max_pos": 200.0,   "leg_max": 100.0},
    {"name": "B_2000_early",    "capital": 2000.0,   "max_pos": 400.0,   "leg_max": 200.0},
    {"name": "C_5000_growth",   "capital": 5000.0,   "max_pos": 1000.0,  "leg_max": 500.0},
    {"name": "D_20000_mid",     "capital": 20000.0,  "max_pos": 2000.0,  "leg_max": 1000.0},
    {"name": "E_50000_large",   "capital": 50000.0,  "max_pos": 5000.0,  "leg_max": 2500.0},
    {"name": "F_100000_scale",  "capital": 100000.0, "max_pos": 10000.0, "leg_max": 5000.0},
]

# Regime predictions path
REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"

# LGBM walk-forward
WF_TRAIN_PERIODS = 12
WF_REBAL_FREQ = "W-FRI"  # V6: weekly rebalance

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "capital_scaling_xval_v1"

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
           f"Days <=0.4: {(regime_series <= REGIME_BULL_THRESHOLD).sum()}")
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


def get_vix_at(vix_series, dt):
    """Get VIX value at a given date."""
    if vix_series is None:
        return 20.0
    if dt in vix_series.index:
        return float(vix_series.loc[dt])
    idx = vix_series.index.get_indexer([dt], method="ffill")
    if idx[0] >= 0:
        return float(vix_series.iloc[idx[0]])
    return 20.0


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


# ══════════════════════════════════════════════════════════════
# WALK-FORWARD LGBM RANKING
# ══════════════════════════════════════════════════════════════

def build_feature_records(close, high, low, rebal_dates, feature_cols, regime_series):
    """
    Build feature + target records for all sectors on all rebal dates.

    V6 regime logic:
      - regime>0.4 (VIX>20): bull direction (bull call spreads only)
      - regime<=0.4 (VIX<20): pair direction (bull+bear pair trades)
    All dates included — regime determines TRADE TYPE, not inclusion.
    """
    fprint(f"  Building records: {len(rebal_dates)} dates, {len(feature_cols)} features")

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
        cv = get_vix_at(vix, dt) if vix is not None else 20.0

        # V6: regime determines trade type, all dates included
        if rscore > REGIME_BULL_THRESHOLD:
            direction = "bull"
        else:
            direction = "pair"  # bull+bear pair trade

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

            rec = {**legacy, **cross_asset, "date": dt, "ticker": tk,
                   "fwd_ret": fwd_ret, "direction": direction, "vix": cv}
            records.append(rec)

    df = pd.DataFrame(records)
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    df[feature_cols] = df[feature_cols].fillna(0.0)

    fprint(f"    {len(df)} records, {len(df['date'].unique())} dates")
    bull_n = (df["direction"] == "bull").sum()
    pair_n = (df["direction"] == "pair").sum()
    fprint(f"    Bull records: {bull_n}, Pair records: {pair_n}")

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

            direction = test_df["direction"].iloc[0]
            vix_val = test_df["vix"].iloc[0] if "vix" in test_df.columns else 20.0
            rankings[test_date] = {
                "scores": dict(zip(test_df["ticker"], test_df["score"])),
                "direction": direction,
                "vix": vix_val,
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

    fprint(f"    {len(rankings)} ranking dates, {n_models} models trained")
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
# TRADE SIMULATION — V6 CONFIG WITH CAPITAL SCALING
# ══════════════════════════════════════════════════════════════

def simulate_trades_v6(rankings, close, high, low, regime_series, atr_dict,
                       initial_capital, initial_max_pos, initial_leg_max):
    """
    Simulate V6 config trades at a given capital level.

    V6 rules:
      - VIX>20 (regime>0.4): bull call spreads on top-3 sectors
      - VIX<20 (regime<=0.4): pair trades — bull call on top-3 + bear put on bottom-3
      - 2% OTM moneyness: K1 = S*1.02, K2 = S*1.05 (for bull)
      - For bear: K1 = S*0.95, K2 = S*0.98
      - Position sizing compounds with equity
      - Commission fixed at $2.60/spread
    """
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    equity = initial_capital
    trades = []

    for dt in sorted(rankings.keys()):
        if dt not in spy.index:
            continue

        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0

        ranking_data = rankings[dt]
        scores = ranking_data["scores"]
        direction = ranking_data["direction"]

        if not scores:
            continue

        # Compounding position sizing: scale max_pos proportionally with equity growth
        equity_ratio = equity / initial_capital
        max_pos = initial_max_pos * equity_ratio
        leg_max = initial_leg_max * equity_ratio

        # Safety: don't risk more than 40% of equity per trade
        max_pos = min(max_pos, equity * 0.40)
        leg_max = min(leg_max, equity * 0.20)

        if max_pos < 30:
            continue

        # Rank sectors
        ranked_top = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_bot = sorted(scores.items(), key=lambda x: x[1])

        if direction == "bull":
            # Bull only: top-K sectors, bull call spreads
            picks_bull = [t for t, _ in ranked_top[:TOP_K]]
            picks_bear = []
            trade_budget = max_pos
        else:
            # Pair trades: top-K bull + bottom-K bear
            picks_bull = [t for t, _ in ranked_top[:TOP_K]]
            picks_bear = [t for t, _ in ranked_bot[:TOP_K]]
            # Remove overlap
            picks_bear = [t for t in picks_bear if t not in picks_bull]
            trade_budget = leg_max  # Each leg gets leg_max

        # Execute bull call spreads
        n_entered = 0
        for tk in picks_bull:
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

            # V6: 2% OTM moneyness
            K1 = round(S * 1.02, 2)   # Buy call 2% OTM
            K2 = round(S * 1.05, 2)   # Sell call 5% OTM (3% spread width)
            if K2 <= K1:
                K2 = K1 + 1.0

            try:
                entry_cost_ps, max_profit_ps = price_bull_call_spread(
                    S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=cv
                )
            except Exception:
                continue

            total_cost = entry_cost_ps * 100 + COMMISSION_RT_SPREAD

            if total_cost <= 0 or total_cost > trade_budget or total_cost > equity * 0.40:
                continue

            # HOLD TO EXPIRY: intrinsic value
            Se = float(close[tk].iloc[ei])
            intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
            exit_value_ps = intrinsic

            pnl = (exit_value_ps - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD
            equity += pnl
            n_entered += 1

            sv = float(spy.loc[dt]) if dt in spy.index else 0
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
                "equity_at_entry": round(equity - pnl, 2),
                "cost": round(total_cost, 2),
                "commission_pct": round(COMMISSION_RT_SPREAD / total_cost * 100, 2),
            })

        # Execute bear put spreads (pair trades only)
        for tk in picks_bear:
            if tk not in close.columns or tk not in atr_dict:
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

            # V6: 2% OTM for bear puts (mirrored)
            K1 = round(S * 0.95, 2)   # Sell put 5% OTM (lower strike)
            K2 = round(S * 0.98, 2)   # Buy put 2% OTM (higher strike)
            if K2 <= K1:
                K2 = K1 + 1.0

            try:
                entry_cost_ps, max_profit_ps = price_bear_put_spread(
                    S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=cv
                )
            except Exception:
                continue

            total_cost = entry_cost_ps * 100 + COMMISSION_RT_SPREAD

            if total_cost <= 0 or total_cost > trade_budget or total_cost > equity * 0.40:
                continue

            Se = float(close[tk].iloc[ei])
            intrinsic = max(K2 - Se, 0.0) - max(K1 - Se, 0.0)
            exit_value_ps = intrinsic

            pnl = (exit_value_ps - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD
            equity += pnl

            sv = float(spy.loc[dt]) if dt in spy.index else 0
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
                "equity_at_entry": round(equity - pnl, 2),
                "cost": round(total_cost, 2),
                "commission_pct": round(COMMISSION_RT_SPREAD / total_cost * 100, 2),
            })

    return trades, equity


# ══════════════════════════════════════════════════════════════
# RANDOM BASELINE
# ══════════════════════════════════════════════════════════════

def random_baseline_test(rankings, close, high, low, regime_series, atr_dict,
                         initial_capital, initial_max_pos, initial_leg_max,
                         n_trials=5):
    """Test if random sector selection also produces similar returns."""
    fprint(f"  Random baseline ({n_trials} trials)...")
    random_sharpes = []

    for trial in range(n_trials):
        np.random.seed(42 + trial)
        rand_rankings = {}
        for dt, data in rankings.items():
            rand_scores = {tk: np.random.random() for tk in data["scores"].keys()}
            rand_rankings[dt] = {
                "scores": rand_scores,
                "direction": data["direction"],
                "vix": data.get("vix", 20.0),
            }

        trades, final_eq = simulate_trades_v6(
            rand_rankings, close, high, low, regime_series, atr_dict,
            initial_capital=initial_capital,
            initial_max_pos=initial_max_pos,
            initial_leg_max=initial_leg_max,
        )

        if trades and len(trades) >= 10:
            result = validate_trades(
                trades, initial_capital=initial_capital,
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
    fprint(f"CAPITAL SCALING CROSS-VALIDATION v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 80)
    fprint(f"V6 Config: Weekly rebal | 2% OTM | Bull VIX>20 + Pairs VIX<20")
    fprint(f"DTE: {DTE} | Spread: {SPREAD_PCT:.0f}% | Haircut: {DEFAULT_HAIRCUT:.0%} entry only")
    fprint(f"Commission: ${COMMISSION_RT_SPREAD:.2f} (FIXED, not proportional)")
    fprint(f"Position sizing: {POS_SIZE_FRAC:.0%} of capital, compounds with equity")
    fprint(f"HOLD TO EXPIRY | Intrinsic value only | No exit haircut")
    fprint()

    fprint("Capital levels to test:")
    for lvl in CAPITAL_LEVELS:
        fprint(f"  {lvl['name']}: ${lvl['capital']:>9,.0f} | "
               f"max_pos=${lvl['max_pos']:>7,.0f} | leg=${lvl['leg_max']:>6,.0f}")
    fprint()

    # 1. Download data
    close, high, low = download_data()

    # 2. Load regime predictions
    regime_series = load_regime_predictions()

    # 3. Pre-compute ATR
    atr_dict = compute_atr_series(high, low, close)

    # 4. Build rebalance dates (V6: weekly)
    rebal_dates = pd.DatetimeIndex(
        close.index.to_series().resample(WF_REBAL_FREQ).last().dropna().values
    )
    fprint(f"Rebalance dates: {len(rebal_dates)} "
           f"({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

    spy_close = close["SPY"]

    # 5. Build features + LGBM rankings (ONCE — same for all capital levels)
    fprint("\n" + "=" * 80)
    fprint("BUILDING LGBM RANKINGS (shared across all capital levels)")
    fprint("=" * 80)

    records = build_feature_records(
        close, high, low, rebal_dates, V4_FEATURES, regime_series,
    )
    rankings, imp_df = walk_forward_lgbm_rank(records, V4_FEATURES)

    if not rankings:
        fprint("ERROR: No rankings produced. Exiting.")
        return

    fprint(f"\nRankings ready: {len(rankings)} rebalance dates")

    # 6. Simulate at each capital level
    all_results = {}

    for lvl in CAPITAL_LEVELS:
        cap = lvl["capital"]
        name = lvl["name"]
        max_pos = lvl["max_pos"]
        leg_max = lvl["leg_max"]

        fprint("\n" + "=" * 80)
        fprint(f"CAPITAL LEVEL: {name} — ${cap:,.0f} | max_pos=${max_pos:,.0f} | leg=${leg_max:,.0f}")
        fprint("=" * 80)

        trades, final_eq = simulate_trades_v6(
            rankings, close, high, low, regime_series, atr_dict,
            initial_capital=cap,
            initial_max_pos=max_pos,
            initial_leg_max=leg_max,
        )

        if not trades or len(trades) < 10:
            fprint(f"  Only {len(trades) if trades else 0} trades — skipping validation")
            all_results[name] = {"capital": cap, "n_trades": len(trades) if trades else 0,
                                 "error": "insufficient trades"}
            continue

        # Adversarial validation
        result = validate_trades(
            trades, initial_capital=cap,
            spy_prices=spy_close,
            strategy_name=name,
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
        fprint(f"    Bull: {len(bull_trades)} trades, WR {bull_wr:.1f}%, PnL ${bull_pnl:,.0f}")
        fprint(f"    Bear: {len(bear_trades)} trades, WR {bear_wr:.1f}%, PnL ${bear_pnl:,.0f}")

        # Commission analysis
        avg_comm_pct = np.mean([t["commission_pct"] for t in trades])
        avg_cost = np.mean([t["cost"] for t in trades])
        fprint(f"  Commission impact: avg {avg_comm_pct:.1f}% of trade cost "
               f"(avg trade ${avg_cost:,.0f}, fixed comm ${COMMISSION_RT_SPREAD:.2f})")

        # Return profile
        total_return = (final_eq - cap) / cap * 100
        fprint(f"  Total return: ${cap:,.0f} -> ${final_eq:,.0f} ({total_return:+.1f}%)")

        # Random baseline
        random_sharpes = random_baseline_test(
            rankings, close, high, low, regime_series, atr_dict,
            initial_capital=cap, initial_max_pos=max_pos, initial_leg_max=leg_max,
        )
        mean_random = np.mean(random_sharpes) if random_sharpes else 0
        fprint(f"  ML Sharpe: {result.sharpe:.2f} vs Random mean: {mean_random:.2f}")
        if result.sharpe > 0 and mean_random > 0:
            fprint(f"  ML alpha ratio: {result.sharpe / mean_random:.2f}x")

        all_results[name] = {
            "capital": cap,
            "max_pos": max_pos,
            "leg_max": leg_max,
            **result.to_dict(),
            "bull_trades": len(bull_trades),
            "bear_trades": len(bear_trades),
            "bull_pnl": round(bull_pnl, 2),
            "bear_pnl": round(bear_pnl, 2),
            "bull_wr": round(bull_wr, 1),
            "bear_wr": round(bear_wr, 1),
            "avg_commission_pct": round(avg_comm_pct, 2),
            "avg_trade_cost": round(avg_cost, 2),
            "total_return_pct": round(total_return, 2),
            "random_sharpes": [round(s, 3) for s in random_sharpes],
            "random_mean_sharpe": round(mean_random, 3),
        }

    # ── SUMMARY COMPARISON ──
    fprint("\n" + "=" * 80)
    fprint("CAPITAL SCALING SUMMARY")
    fprint("=" * 80)
    fprint(f"{'Level':<22} {'Capital':>10} {'Trades':>7} {'Sharpe':>7} {'Sortino':>8} "
           f"{'WR':>6} {'PF':>6} {'MaxDD':>7} {'Gates':>6} {'Final$':>10} {'CommPct':>8}")
    fprint("-" * 110)

    for lvl in CAPITAL_LEVELS:
        name = lvl["name"]
        r = all_results.get(name)
        if not r or "error" in r:
            fprint(f"  {name:<22} ${r['capital']:>9,.0f} — INSUFFICIENT DATA —")
            continue
        fprint(f"  {name:<22} ${r['capital']:>9,.0f} {r['n_trades']:>6} "
               f"{r['sharpe']:>7.2f} {r['sortino']:>8.2f} "
               f"{r['win_rate']*100:>5.1f}% {r['profit_factor']:>5.2f} "
               f"{r['max_dd']*100:>6.1f}% {r['gates_passed']}/{r['gates_total']} "
               f"${r['final_equity']:>9,.0f} {r['avg_commission_pct']:>7.1f}%")

    # ── KEY QUESTION: Does Sharpe degrade at scale? ──
    fprint("\n" + "=" * 80)
    fprint("KEY QUESTION: Does Sharpe degrade at larger capital?")
    fprint("=" * 80)

    valid_results = [(name, r) for name, r in all_results.items()
                     if "error" not in r and "sharpe" in r]

    if len(valid_results) >= 2:
        capitals = [r["capital"] for _, r in valid_results]
        sharpes = [r["sharpe"] for _, r in valid_results]
        comm_pcts = [r["avg_commission_pct"] for _, r in valid_results]

        # Sharpe trend
        if len(capitals) >= 3:
            slope, intercept, r_val, p_val, _ = stats.linregress(
                np.log10(capitals), sharpes
            )
            fprint(f"  Sharpe vs log10(capital) regression:")
            fprint(f"    Slope: {slope:+.4f} (positive = IMPROVES with scale)")
            fprint(f"    R-squared: {r_val**2:.3f}, p-value: {p_val:.4f}")
            if slope > 0:
                fprint(f"    RESULT: Strategy IMPROVES with scale (+{slope:.3f} Sharpe per 10x capital)")
            elif abs(slope) < 0.05:
                fprint(f"    RESULT: Strategy is SCALE-NEUTRAL (slope near zero)")
            else:
                fprint(f"    RESULT: Strategy DEGRADES with scale ({slope:.3f} Sharpe per 10x capital)")

        # Commission dilution effect
        fprint(f"\n  Commission dilution (key driver):")
        for name, r in valid_results:
            fprint(f"    ${r['capital']:>9,.0f}: commission = {r['avg_commission_pct']:.1f}% of trade cost")

        baseline_sharpe = all_results.get("A_645_baseline", {}).get("sharpe", 0)
        best_name, best_r = max(valid_results, key=lambda x: x[1].get("sharpe", 0))
        fprint(f"\n  Baseline ($645) Sharpe: {baseline_sharpe:.2f}")
        fprint(f"  Best Sharpe: {best_r['sharpe']:.2f} at ${best_r['capital']:,.0f} ({best_name})")

    # Feature importance
    if imp_df is not None:
        fprint("\n" + "=" * 80)
        fprint("FEATURE IMPORTANCE (Top 10)")
        fprint("=" * 80)
        for _, row in imp_df.head(10).iterrows():
            bar = "*" * int(row["importance"] / imp_df["importance"].max() * 30)
            fprint(f"  {row['feature']:<30} {row['importance']:>6.1f} {bar}")

    # Save results
    results_path = OUTPUT_DIR / "capital_scaling_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    # MLflow logging
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"capital_scaling_{t0.strftime('%Y%m%d_%H%M')}"):
                for name, r in all_results.items():
                    if "error" in r:
                        continue
                    prefix = name.split("_")[0]
                    cap_k = int(r["capital"] / 1000) if r["capital"] >= 1000 else r["capital"]
                    tag = f"{prefix}_{cap_k}k" if r["capital"] >= 1000 else f"{prefix}_{int(r['capital'])}"
                    mlflow.log_metric(f"{tag}_sharpe", r.get("sharpe", 0))
                    mlflow.log_metric(f"{tag}_sortino", r.get("sortino", 0))
                    mlflow.log_metric(f"{tag}_win_rate", r.get("win_rate", 0))
                    mlflow.log_metric(f"{tag}_profit_factor", r.get("profit_factor", 0))
                    mlflow.log_metric(f"{tag}_max_dd", r.get("max_dd", 0))
                    mlflow.log_metric(f"{tag}_n_trades", r.get("n_trades", 0))
                    mlflow.log_metric(f"{tag}_gates_passed", r.get("gates_passed", 0))
                    mlflow.log_metric(f"{tag}_final_equity", r.get("final_equity", 0))
                    mlflow.log_metric(f"{tag}_commission_pct", r.get("avg_commission_pct", 0))
                    mlflow.log_metric(f"{tag}_total_return_pct", r.get("total_return_pct", 0))
                    mlflow.log_metric(f"{tag}_random_mean_sharpe", r.get("random_mean_sharpe", 0))

                mlflow.log_params({
                    "config": "V6",
                    "rebal_freq": WF_REBAL_FREQ,
                    "otm_pct": OTM_PCT,
                    "spread_pct": SPREAD_PCT,
                    "dte": DTE,
                    "haircut": DEFAULT_HAIRCUT,
                    "commission": COMMISSION_RT_SPREAD,
                    "pos_size_frac": POS_SIZE_FRAC,
                    "regime_bull_thresh": REGIME_BULL_THRESHOLD,
                    "vix_cutoff": VIX_REGIME_CUTOFF,
                    "hold_to_expiry": True,
                    "entry_haircut_only": True,
                    "n_features": len(V4_FEATURES),
                    "wf_train_periods": WF_TRAIN_PERIODS,
                    "n_capital_levels": len(CAPITAL_LEVELS),
                    "capital_levels": str([l["capital"] for l in CAPITAL_LEVELS]),
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
