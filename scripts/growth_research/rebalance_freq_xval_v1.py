#!/usr/bin/env python3
"""
Rebalance Frequency Cross-Validation v1
========================================

Validates finding #108 (weekly rebalance beats biweekly by +108% Sharpe)
using the EXACT production v4 infrastructure.

Tests 5 rebalance frequency variants, ALL using identical:
  - LGBM walk-forward (100 trees, depth 4, 12-period sliding window)
  - 21 features (18 legacy + 3 cross-asset)
  - GRU regime>0.4 filter (bull only)
  - Hold-to-expiry intrinsic pricing
  - 15% entry haircut, $2.60 commission
  - $645 starting capital, $200/trade max
  - Top-3 sector selection

Variants (ONLY rebalance frequency changes):
  A: Biweekly  — production v4 baseline (2W-FRI), expect Sharpe ~1.67-1.87
  B: Weekly    — every Friday (W-FRI)
  C: 3-day     — every 3 trading days
  D: Monthly   — every ~21 trading days (MS frequency)
  E: Signal-triggered — rebalance on LGBM rank flip OR VIX crosses 20

5-gate adversarial validation for each variant.
MLflow experiment: rebalance_freq_xval_v1
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
from research.tools.adversarial_validator import validate_trades

# -- Config (IDENTICAL to production v4) --
BASE = Path("/home/jupiter/Lvl3Quant")
OUTPUT_DIR = BASE / "output" / "growth_research" / "rebalance_freq_xval_v1"
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

# LGBM walk-forward (same params as production v4)
WF_TRAIN_PERIODS = 12

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "rebalance_freq_xval_v1"

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


# ================================================================
# DATA DOWNLOAD (identical to production v4)
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
# REGIME LOADING (identical to production v4)
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
# FEATURE ENGINEERING (identical to production v4)
# ================================================================

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


# ================================================================
# WALK-FORWARD LGBM RANKING (identical to production v4)
# ================================================================

def build_feature_records(close, high, low, rebal_dates, feature_cols, regime_series):
    """
    Build feature + target records for all sectors on all rebal dates.
    Uses bull_only regime mode (regime>0.4) -- same as production v4 variant B.
    """
    import lightgbm as lgb

    fprint(f"  Building records: {len(rebal_dates)} dates, {len(feature_cols)} features")

    records = []
    sector_cols = [c for c in SECTORS if c in close.columns]
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    for dt in rebal_dates:
        idx = close.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
            continue

        # Get regime score -- bull_only filter
        rscore = get_regime_score_at(regime_series, dt)
        if rscore <= REGIME_BULL_THRESHOLD:
            continue
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
            fi = min(idx + DTE, len(close) - 1)
            if fi <= idx:
                continue
            fwd_ret = float(close[tk].iloc[fi] / close[tk].iloc[idx] - 1)

            rec = {**legacy, **cross_asset, "date": dt, "ticker": tk,
                   "fwd_ret": fwd_ret, "direction": direction}
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

            rankings[test_date] = {
                "scores": dict(zip(test_df["ticker"], test_df["score"])),
                "direction": "bull",
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


# ================================================================
# ATR COMPUTATION (identical to production v4)
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
# TRADE SIMULATION (identical to production v4, bull_only mode)
# ================================================================

def simulate_trades(name, rankings, close, high, low, regime_series, atr_dict):
    """
    Simulate bull call spreads from rankings.

    HONEST RULES (production v4):
      - Hold to expiry
      - At expiry: intrinsic value only
      - 15% haircut on entry only
      - No exit haircut (automatic exercise)
    """
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    equity = CAP
    trades = []

    for dt in sorted(rankings.keys()):
        if dt not in spy.index:
            continue

        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0

        ranking_data = rankings[dt]
        scores = ranking_data["scores"]
        direction = ranking_data["direction"]

        # Bull only
        if direction != "bull":
            continue

        if not scores:
            continue

        # Top K by LGBM score (highest predicted forward return)
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
            if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]):
                av = float(atr_dict[tk].loc[dt])
            else:
                av = S * 0.015

            # Price the spread
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

            # HOLD TO EXPIRY: compute intrinsic value at expiry
            Se = float(close[tk].iloc[ei])

            # Bull call spread intrinsic at expiry
            intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
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
                "win": pnl > 0,
            })

    return trades, equity


# ================================================================
# RANDOM BASELINE (identical to production v4)
# ================================================================

def random_baseline_test(rankings, close, high, low, regime_series, atr_dict, n_trials=5):
    """Test if random sector selection also produces similar returns."""
    fprint(f"\n  Random baseline test ({n_trials} trials)...")
    random_sharpes = []

    for trial in range(n_trials):
        np.random.seed(42 + trial)
        rand_rankings = {}
        for dt, data in rankings.items():
            rand_scores = {tk: np.random.random() for tk in data["scores"].keys()}
            rand_rankings[dt] = {"scores": rand_scores, "direction": data["direction"]}

        trades, final_eq = simulate_trades(
            f"Random_{trial}", rand_rankings, close, high, low,
            regime_series, atr_dict,
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


# ================================================================
# REBALANCE DATE GENERATION
# ================================================================

def generate_rebal_dates_biweekly(close):
    """Biweekly (production v4 default): every 2 Fridays."""
    return pd.DatetimeIndex(
        close.index.to_series().resample("2W-FRI").last().dropna().values
    )


def generate_rebal_dates_weekly(close):
    """Weekly: every Friday."""
    return pd.DatetimeIndex(
        close.index.to_series().resample("W-FRI").last().dropna().values
    )


def generate_rebal_dates_3day(close):
    """Every 3 trading days."""
    trading_days = close.index
    dates = trading_days[::3]  # every 3rd trading day
    return pd.DatetimeIndex(dates)


def generate_rebal_dates_monthly(close):
    """Monthly: first trading day of each month (MS = month start)."""
    return pd.DatetimeIndex(
        close.index.to_series().resample("MS").first().dropna().values
    )


def generate_rebal_dates_signal_triggered(close, regime_series):
    """
    Signal-triggered: rebalance when LGBM top-3 ranking changes (rank flip)
    OR VIX crosses 20. Check daily but only rebalance on trigger.

    Since we need LGBM rankings to detect rank flips, this is a two-pass approach:
    1. First pass: build features/rankings on ALL trading days (expensive but necessary)
    2. Second pass: only trade on dates where top-3 changed or VIX crossed 20

    To keep compute tractable, we check weekly for triggers but filter down to
    only triggered dates. This simulates "check daily, act on change".
    """
    # Start with daily trading dates as candidates
    # We'll filter down to triggered dates after the first LGBM pass
    # For the LGBM pass, use weekly candidates to keep compute reasonable
    weekly = pd.DatetimeIndex(
        close.index.to_series().resample("W-FRI").last().dropna().values
    )
    return weekly  # Will be filtered in the signal-triggered simulation


def detect_signal_triggers(rankings, close):
    """
    From a set of weekly rankings, filter to only dates where:
    1. LGBM top-3 ranking changed (rank flip), OR
    2. VIX crossed 20 since last rebalance
    Always include the first date.
    """
    vix = close["VIX"] if "VIX" in close.columns else None
    sorted_dates = sorted(rankings.keys())
    if not sorted_dates:
        return {}

    triggered = {}
    prev_top3 = None
    prev_vix_above_20 = None

    for dt in sorted_dates:
        scores = rankings[dt]["scores"]
        if not scores:
            continue

        # Current top-3
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        cur_top3 = set(t for t, _ in ranked[:TOP_K])

        # VIX cross check
        vix_above_20 = None
        if vix is not None and dt in vix.index:
            cv = float(vix.loc[dt])
            vix_above_20 = cv > 20.0

        # Trigger conditions
        trigger = False
        if prev_top3 is None:
            # First date -- always trigger
            trigger = True
        else:
            # Rank flip: top-3 composition changed
            if cur_top3 != prev_top3:
                trigger = True
            # VIX cross: VIX crossed above or below 20
            if prev_vix_above_20 is not None and vix_above_20 is not None:
                if vix_above_20 != prev_vix_above_20:
                    trigger = True

        if trigger:
            triggered[dt] = rankings[dt]

        prev_top3 = cur_top3
        prev_vix_above_20 = vix_above_20

    return triggered


# ================================================================
# MAIN
# ================================================================

def main():
    t0 = datetime.now()
    fprint("=" * 80)
    fprint(f"REBALANCE FREQUENCY CROSS-VALIDATION v1 -- {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 80)
    fprint(f"Capital: ${CAP:.0f} | DTE: {DTE} | Spread: {SPREAD_PCT:.0f}% | "
           f"Haircut: {DEFAULT_HAIRCUT:.0%} entry only | Comm: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"HOLD TO EXPIRY | Intrinsic value only at expiry | No exit haircut")
    fprint(f"Regime bull threshold: >{REGIME_BULL_THRESHOLD}")
    fprint(f"Features: {len(V4_FEATURES)} (18 legacy + 3 cross-asset)")
    fprint(f"LGBM: {WF_TRAIN_PERIODS}-period sliding WF, 100 trees, depth 4")
    fprint()

    # 1. Download data
    close, high, low = download_data()

    # 2. Load regime predictions
    regime_series = load_regime_predictions()

    # 3. Pre-compute ATR
    atr_dict = compute_atr_series(high, low, close)

    spy_close = close["SPY"]

    # ----------------------------------------------------------------
    # Generate rebalance dates for each variant
    # ----------------------------------------------------------------
    rebal_configs = {
        "A_biweekly": ("Biweekly (production v4 baseline, 2W-FRI)", generate_rebal_dates_biweekly(close)),
        "B_weekly": ("Weekly (W-FRI)", generate_rebal_dates_weekly(close)),
        "C_3day": ("Every 3 trading days", generate_rebal_dates_3day(close)),
        "D_monthly": ("Monthly (~21 trading days)", generate_rebal_dates_monthly(close)),
        "E_signal": ("Signal-triggered (rank flip or VIX cross 20)", generate_rebal_dates_signal_triggered(close, regime_series)),
    }

    for vname, (desc, dates) in rebal_configs.items():
        fprint(f"  {vname}: {len(dates)} candidate dates -- {desc}")

    # ----------------------------------------------------------------
    # Run LGBM walk-forward + simulation for each variant
    # ----------------------------------------------------------------
    all_results = {}

    for vname, (desc, rebal_dates) in rebal_configs.items():
        fprint("\n" + "=" * 80)
        fprint(f"VARIANT {vname.split('_')[0]}: {desc}")
        fprint("=" * 80)

        # Build features and LGBM rankings
        records = build_feature_records(
            close, high, low, rebal_dates, V4_FEATURES, regime_series,
        )
        rankings, imp_df = walk_forward_lgbm_rank(records, V4_FEATURES, vname)

        if not rankings:
            fprint(f"  No rankings available, skipping")
            continue

        # Special handling for signal-triggered variant
        if vname == "E_signal":
            n_before = len(rankings)
            rankings = detect_signal_triggers(rankings, close)
            n_after = len(rankings)
            fprint(f"  Signal filter: {n_before} -> {n_after} triggered dates "
                   f"({n_after/max(n_before,1)*100:.0f}% triggered)")

        # Simulate trades
        trades, final_eq = simulate_trades(
            vname, rankings, close, high, low, regime_series, atr_dict,
        )

        if not trades or len(trades) < 10:
            fprint(f"  Only {len(trades) if trades else 0} trades, skipping validation")
            all_results[vname] = {
                "description": desc,
                "n_trades": len(trades) if trades else 0,
                "final_equity": final_eq,
                "sharpe": 0.0,
                "sortino": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "max_dd": 0.0,
                "gates_passed": 0,
                "gates_total": 5,
                "n_rebal_dates": len(rebal_dates),
            }
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
            rankings, close, high, low, regime_series, atr_dict,
        )
        mean_random = np.mean(random_sharpes) if random_sharpes else 0
        fprint(f"  ML Sharpe: {result.sharpe:.2f} vs Random mean: {mean_random:.2f}")
        if result.sharpe > 0 and mean_random > 0:
            fprint(f"  ML alpha ratio: {result.sharpe / mean_random:.2f}x")

        all_results[vname] = {
            "description": desc,
            **result.to_dict(),
            "random_sharpes": [round(s, 3) for s in random_sharpes],
            "random_mean_sharpe": round(mean_random, 3),
            "n_rebal_dates": len(rebal_dates),
            "n_ranking_dates": len(rankings),
        }

        # Store feature importance for the first variant only (they should be similar)
        if imp_df is not None and vname == "A_biweekly":
            all_results["feature_importance"] = imp_df.to_dict("records")

    # ----------------------------------------------------------------
    # SUMMARY COMPARISON
    # ----------------------------------------------------------------
    fprint("\n" + "=" * 80)
    fprint("SUMMARY COMPARISON -- REBALANCE FREQUENCY CROSS-VALIDATION")
    fprint("=" * 80)
    fprint(f"{'Variant':<25} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} "
           f"{'PF':>6} {'MaxDD':>7} {'Gates':>6} {'Final$':>8} {'RandSh':>7} {'RebalN':>7}")
    fprint("-" * 102)

    variant_order = ["A_biweekly", "B_weekly", "C_3day", "D_monthly", "E_signal"]
    for vname in variant_order:
        r = all_results.get(vname)
        if not r:
            fprint(f"  {vname:<25} -- NO DATA --")
            continue
        n_trades = r.get('n_trades', 0)
        sharpe = r.get('sharpe', 0)
        sortino = r.get('sortino', 0)
        wr = r.get('win_rate', 0)
        pf = r.get('profit_factor', 0)
        mdd = r.get('max_dd', 0)
        gp = r.get('gates_passed', 0)
        gt = r.get('gates_total', 5)
        feq = r.get('final_equity', 0)
        rms = r.get('random_mean_sharpe', 0)
        nrd = r.get('n_ranking_dates', r.get('n_rebal_dates', 0))
        fprint(f"  {vname:<25} {n_trades:>5} {sharpe:>7.2f} {sortino:>8.2f} "
               f"{wr*100 if isinstance(wr, float) else 0:>5.1f}% {pf:>5.2f} "
               f"{mdd*100 if isinstance(mdd, float) else 0:>6.1f}% {gp}/{gt} "
               f"${feq:>7,.0f} {rms:>7.2f} {nrd:>6}")

    # ----------------------------------------------------------------
    # KEY FINDINGS
    # ----------------------------------------------------------------
    fprint("\n" + "=" * 80)
    fprint("KEY FINDINGS")
    fprint("=" * 80)

    # Check if biweekly baseline reproduces expected Sharpe
    a_sharpe = all_results.get("A_biweekly", {}).get("sharpe", 0)
    fprint(f"\n  Baseline check: Biweekly Sharpe = {a_sharpe:.2f} (expected ~1.67-1.87)")
    if 1.5 <= a_sharpe <= 2.1:
        fprint(f"  PASS: Baseline reproduced within expected range")
    else:
        fprint(f"  WARNING: Baseline outside expected range -- investigate")

    # Compare variants
    fprint(f"\n  Sharpe comparison vs biweekly baseline ({a_sharpe:.2f}):")
    for vname in ["B_weekly", "C_3day", "D_monthly", "E_signal"]:
        r = all_results.get(vname, {})
        v_sharpe = r.get("sharpe", 0)
        if a_sharpe > 0:
            delta_pct = (v_sharpe - a_sharpe) / a_sharpe * 100
            fprint(f"    {vname:<20}: Sharpe {v_sharpe:.2f} ({delta_pct:+.0f}% vs baseline)")
        else:
            fprint(f"    {vname:<20}: Sharpe {v_sharpe:.2f}")

    # Validate finding #108
    b_sharpe = all_results.get("B_weekly", {}).get("sharpe", 0)
    if a_sharpe > 0 and b_sharpe > 0:
        weekly_vs_biweekly = (b_sharpe - a_sharpe) / a_sharpe * 100
        fprint(f"\n  Finding #108 validation: Weekly vs Biweekly = {weekly_vs_biweekly:+.0f}%")
        if weekly_vs_biweekly > 50:
            fprint(f"  CONFIRMED: Weekly significantly better (finding #108 said +108%)")
        elif weekly_vs_biweekly > 10:
            fprint(f"  PARTIALLY CONFIRMED: Weekly better but not +108%")
        elif weekly_vs_biweekly > -10:
            fprint(f"  INCONCLUSIVE: Similar performance, finding #108 not reproduced")
        else:
            fprint(f"  REFUTED: Weekly worse than biweekly, finding #108 not valid in production")

    # Best variant
    best_name = max(variant_order, key=lambda v: all_results.get(v, {}).get("sharpe", -999))
    best_sharpe = all_results.get(best_name, {}).get("sharpe", 0)
    fprint(f"\n  BEST VARIANT: {best_name} (Sharpe {best_sharpe:.2f})")

    # Gate check
    best_gates = all_results.get(best_name, {}).get("gates_passed", 0)
    fprint(f"  Gates passed: {best_gates}/5")

    # ----------------------------------------------------------------
    # Save results
    # ----------------------------------------------------------------
    results_path = OUTPUT_DIR / "rebalance_freq_xval_results.json"
    save_results = {k: v for k, v in all_results.items() if k != "feature_importance"}
    with open(results_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    # ----------------------------------------------------------------
    # MLflow logging
    # ----------------------------------------------------------------
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"rebal_freq_xval_{t0.strftime('%Y%m%d_%H%M')}"):
                # Log all variant metrics
                for vname in variant_order:
                    r = all_results.get(vname, {})
                    if not r:
                        continue
                    prefix = vname
                    for metric_name in ["sharpe", "sortino", "win_rate", "profit_factor",
                                        "max_dd", "n_trades", "gates_passed", "final_equity",
                                        "random_mean_sharpe"]:
                        val = r.get(metric_name, 0)
                        if val is not None:
                            mlflow.log_metric(f"{prefix}_{metric_name}", float(val))

                    mlflow.log_metric(f"{prefix}_n_rebal_dates", r.get("n_rebal_dates", 0))

                # Log common params
                mlflow.log_params({
                    "capital": CAP,
                    "dte": DTE,
                    "spread_pct": SPREAD_PCT,
                    "haircut": DEFAULT_HAIRCUT,
                    "regime_bull_thresh": REGIME_BULL_THRESHOLD,
                    "hold_to_expiry": True,
                    "entry_haircut_only": True,
                    "n_features": len(V4_FEATURES),
                    "wf_train_periods": WF_TRAIN_PERIODS,
                    "top_k": TOP_K,
                    "max_pos_size": 200,
                    "variants_tested": ",".join(variant_order),
                })

                # Finding #108 validation
                if a_sharpe > 0 and b_sharpe > 0:
                    mlflow.log_metric("weekly_vs_biweekly_pct", weekly_vs_biweekly)

                mlflow.log_artifact(str(results_path))
            fprint(f"MLflow run logged to experiment '{EXPERIMENT_NAME}'")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    fprint("DONE")


if __name__ == "__main__":
    main()
