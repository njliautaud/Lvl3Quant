#!/usr/bin/env python3
"""
V5 Candidate Cross-Validation — Parameter Interaction Test
===========================================================

Tests whether DTE=45 and K=1-2/weekly rebalance improvements (found independently
in MLflow exp 161 and exp 163) INTERACT when combined.

8 Variants (ONLY top_k, rebalance_interval, DTE change — everything else identical):

| Variant         | top_k | rebal_days | DTE | Description                          |
|-----------------|-------|-----------|-----|--------------------------------------|
| v4_baseline     | 3     | 10        | 21  | Current production for comparison    |
| dte45_only      | 3     | 10        | 45  | Only change DTE                      |
| k2_weekly_only  | 2     | 5         | 21  | Only change K and rebalance          |
| k1_weekly_only  | 1     | 5         | 21  | Most concentrated + frequent         |
| v5a_k2_w_dte45  | 2     | 5         | 45  | Best combo candidate A               |
| v5b_k1_w_dte45  | 1     | 5         | 45  | Best combo candidate B               |
| v5c_k3_w_dte45  | 3     | 5         | 45  | Keep K=3 but change both timing      |
| v5d_k2_w_dte30  | 2     | 5         | 30  | Intermediate DTE check               |

Base: production_v4_honest_test.py (Variant D = v4c, the best production v4).
All LGBM features, walk-forward windows, BS pricing, haircuts, commissions,
regime filter are IDENTICAL to production v4.

5-gate adversarial validation on ALL variants.
Results logged to MLflow and saved to JSON.
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
OUTPUT_DIR = BASE / "output" / "v5_candidate_crossval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
SPREAD_PCT = 3.0
REGIME_BULL_THRESHOLD = 0.4
REGIME_BEAR_THRESHOLD = 0.2

# Regime predictions path
REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"

# LGBM walk-forward
WF_TRAIN_PERIODS = 12

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "v5_candidate_crossval"

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
# VARIANT GRID (the ONLY things that change)
# ══════════════════════════════════════════════════════════════
VARIANTS = [
    {"name": "v4_baseline",    "top_k": 3, "rebal_days": 10, "dte": 21, "desc": "Current production for comparison"},
    {"name": "dte45_only",     "top_k": 3, "rebal_days": 10, "dte": 45, "desc": "Only change DTE"},
    {"name": "k2_weekly_only", "top_k": 2, "rebal_days": 5,  "dte": 21, "desc": "Only change K and rebalance"},
    {"name": "k1_weekly_only", "top_k": 1, "rebal_days": 5,  "dte": 21, "desc": "Most concentrated + frequent"},
    {"name": "v5a_k2_w_dte45", "top_k": 2, "rebal_days": 5,  "dte": 45, "desc": "Best combo candidate A"},
    {"name": "v5b_k1_w_dte45", "top_k": 1, "rebal_days": 5,  "dte": 45, "desc": "Best combo candidate B"},
    {"name": "v5c_k3_w_dte45", "top_k": 3, "rebal_days": 5,  "dte": 45, "desc": "Keep K=3 but change both timing params"},
    {"name": "v5d_k2_w_dte30", "top_k": 2, "rebal_days": 5,  "dte": 30, "desc": "Intermediate DTE check"},
]


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

            rec = {**legacy, **cross_asset, "date": dt, "ticker": tk,
                   "fwd_ret": fwd_ret, "direction": direction}
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

            # Keep direction info with the ranking
            direction = test_df["direction"].iloc[0] if "direction" in test_df.columns else "bull"
            rankings[test_date] = {
                "scores": dict(zip(test_df["ticker"], test_df["score"])),
                "direction": direction,
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
# TRADE SIMULATION (parameterized by top_k and dte)
# ══════════════════════════════════════════════════════════════

def simulate_trades(name, rankings, close, high, low, regime_series, atr_dict,
                    top_k, dte, bull_only=False, skip_vix_25_30=True):
    """
    Simulate bull call spreads (and bear put spreads) from rankings.

    HONEST RULES (identical to production v4):
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

        # VIX 25-30 sit-in-cash filter (finding #49)
        if skip_vix_25_30 and 25.0 <= cv <= 30.0:
            continue

        ranking_data = rankings[dt]
        scores = ranking_data["scores"]
        direction = ranking_data["direction"]

        # If bull_only mode, skip bear dates
        if bull_only and direction == "bear":
            continue

        if not scores:
            continue

        # Pick sectors
        if direction == "bull":
            # Top K by LGBM score (highest predicted forward return)
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        else:
            # Bear: bottom K by LGBM score (worst predicted forward return)
            ranked = sorted(scores.items(), key=lambda x: x[1])

        picks = [t for t, _ in ranked[:top_k]]

        # Position sizing: fixed, max $200 per trade or 1/3 of equity
        max_pos = min(200, equity / 3)
        if max_pos < 30:
            continue

        n_entered = 0
        for tk in picks:
            if tk not in close.columns or tk not in atr_dict or n_entered >= top_k:
                continue

            S = float(close[tk].loc[dt])
            di = close.index.get_loc(dt)
            ei = min(di + dte, len(close) - 1)
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
                if direction == "bull":
                    entry_cost_ps, max_profit_ps = price_bull_call_spread(
                        S=S, K1=K1, K2=K2, dte=dte, atr=av, vix=cv
                    )
                else:
                    # Bear put spread: buy put at K2 (higher), sell put at K1 (lower)
                    entry_cost_ps, max_profit_ps = price_bear_put_spread(
                        S=S, K1=K1, K2=K2, dte=dte, atr=av, vix=cv
                    )
            except Exception:
                continue

            total_cost = entry_cost_ps * 100 + COMMISSION_RT_SPREAD

            if total_cost <= 0 or total_cost > max_pos or total_cost > equity * 0.40:
                continue

            # HOLD TO EXPIRY: compute intrinsic value at expiry
            Se = float(close[tk].iloc[ei])

            if direction == "bull":
                # Bull call spread intrinsic at expiry
                intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
            else:
                # Bear put spread intrinsic at expiry
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
                "win": pnl > 0,
            })

    return trades, equity


# ══════════════════════════════════════════════════════════════
# RANDOM BASELINE
# ══════════════════════════════════════════════════════════════

def random_baseline_test(rankings, close, high, low, regime_series, atr_dict,
                         top_k, dte, bull_only, skip_vix_25_30, n_trials=5):
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
            regime_series, atr_dict, top_k=top_k, dte=dte,
            bull_only=bull_only, skip_vix_25_30=skip_vix_25_30,
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
    fprint(f"V5 CANDIDATE CROSS-VALIDATION — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 80)
    fprint(f"Capital: ${CAP:.0f} | Spread: {SPREAD_PCT:.0f}% | "
           f"Haircut: {DEFAULT_HAIRCUT:.0%} entry only | Comm: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"HOLD TO EXPIRY | Intrinsic value only at expiry | No exit haircut")
    fprint(f"Regime bull threshold: >{REGIME_BULL_THRESHOLD}, bear: <{REGIME_BEAR_THRESHOLD}")
    fprint(f"VIX 25-30: skip (all variants)")
    fprint()

    fprint("VARIANT GRID:")
    fprint(f"{'Name':<20} {'top_k':>5} {'rebal':>5} {'DTE':>4}  Description")
    fprint("-" * 80)
    for v in VARIANTS:
        fprint(f"  {v['name']:<20} {v['top_k']:>3} {v['rebal_days']:>5}d {v['dte']:>4}  {v['desc']}")
    fprint()

    # 1. Download data (shared across all variants)
    close, high, low = download_data()

    # 2. Load regime predictions (shared)
    regime_series = load_regime_predictions()

    # 3. Pre-compute ATR (shared)
    atr_dict = compute_atr_series(high, low, close)

    spy_close = close["SPY"]

    # ══════════════════════════════════════════════════════════════
    # BUILD FEATURES FOR EACH UNIQUE (rebal_days, dte) COMBINATION
    # ══════════════════════════════════════════════════════════════
    # We cache feature records + rankings by (rebal_days, dte) since
    # multiple variants may share the same rebalance/DTE combo.
    # Only top_k differs at trade simulation time.

    feature_cols = V4_FEATURES  # All variants use the same 21 features

    # Get unique (rebal_days, dte) combos
    combos = set()
    for v in VARIANTS:
        combos.add((v["rebal_days"], v["dte"]))

    fprint(f"\nUnique (rebal_days, dte) combos to build: {len(combos)}")

    rankings_cache = {}
    for rebal_days, dte in sorted(combos):
        fprint(f"\n{'=' * 80}")
        fprint(f"BUILDING FEATURES: rebal={rebal_days}d, DTE={dte}")
        fprint(f"{'=' * 80}")

        # Build rebalance dates for this frequency
        rebal_freq = f"{rebal_days}B"  # Business days
        rebal_dates = pd.DatetimeIndex(
            close.index.to_series().resample(rebal_freq).last().dropna().values
        )
        fprint(f"Rebalance dates: {len(rebal_dates)} "
               f"({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

        # Build feature records with bull+bear regime mode (v4c/D mode)
        records = build_feature_records(
            close, high, low, rebal_dates, feature_cols,
            regime_series, dte=dte, regime_mode="bull_bear",
        )

        # Walk-forward LGBM ranking
        rankings, imp = walk_forward_lgbm_rank(
            records, feature_cols, f"rebal{rebal_days}_dte{dte}"
        )

        rankings_cache[(rebal_days, dte)] = (rankings, imp)

    # ══════════════════════════════════════════════════════════════
    # SIMULATE ALL VARIANTS
    # ══════════════════════════════════════════════════════════════
    fprint(f"\n{'=' * 80}")
    fprint("SIMULATING ALL 8 VARIANTS")
    fprint(f"{'=' * 80}")

    all_results = {}
    all_importance = {}

    for v in VARIANTS:
        vname = v["name"]
        top_k = v["top_k"]
        rebal_days = v["rebal_days"]
        dte = v["dte"]
        desc = v["desc"]

        fprint(f"\n--- {vname}: top_k={top_k}, rebal={rebal_days}d, DTE={dte} ---")
        fprint(f"    {desc}")

        rankings, imp = rankings_cache[(rebal_days, dte)]
        all_importance[vname] = imp

        if not rankings:
            fprint(f"  No rankings available, skipping")
            continue

        # Simulate with this variant's top_k and dte
        trades, final_eq = simulate_trades(
            vname, rankings, close, high, low, regime_series, atr_dict,
            top_k=top_k, dte=dte,
            bull_only=False,  # All variants use bull+bear (v4c mode)
            skip_vix_25_30=True,  # All variants skip VIX 25-30
        )

        if not trades or len(trades) < 10:
            fprint(f"  Only {len(trades) if trades else 0} trades, skipping validation")
            all_results[vname] = {
                "description": desc,
                "top_k": top_k, "rebal_days": rebal_days, "dte": dte,
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

        # CAGR calculation
        if trades:
            first_date = pd.Timestamp(trades[0]["entry_date"])
            last_date = pd.Timestamp(trades[-1]["exit_date"])
            years = max((last_date - first_date).days / 365.25, 0.5)
            cagr = (final_eq / CAP) ** (1 / years) - 1
        else:
            cagr = 0.0
            years = 0.0

        # Random baseline
        random_sharpes = random_baseline_test(
            rankings, close, high, low, regime_series, atr_dict,
            top_k=top_k, dte=dte,
            bull_only=False, skip_vix_25_30=True,
        )
        mean_random = np.mean(random_sharpes) if random_sharpes else 0
        fprint(f"  ML Sharpe: {result.sharpe:.2f} vs Random mean: {mean_random:.2f}")
        if result.sharpe > 0 and mean_random > 0:
            fprint(f"  ML alpha ratio: {result.sharpe / mean_random:.2f}x")

        all_results[vname] = {
            "description": desc,
            "top_k": top_k,
            "rebal_days": rebal_days,
            "dte": dte,
            **result.to_dict(),
            "cagr": round(cagr, 4),
            "years": round(years, 2),
            "bull_trades": len(bull_trades),
            "bear_trades": len(bear_trades),
            "bull_pnl": round(bull_pnl, 2),
            "bear_pnl": round(bear_pnl, 2),
            "bull_wr": round(bull_wr, 3),
            "bear_wr": round(bear_wr, 3),
            "random_sharpes": [round(s, 3) for s in random_sharpes],
            "random_mean_sharpe": round(mean_random, 3),
        }

    # ══════════════════════════════════════════════════════════════
    # SUMMARY COMPARISON
    # ══════════════════════════════════════════════════════════════
    fprint(f"\n{'=' * 80}")
    fprint("SUMMARY COMPARISON — ALL 8 VARIANTS")
    fprint(f"{'=' * 80}")
    fprint(f"{'Variant':<20} {'K':>2} {'Rb':>3} {'DTE':>4} {'Trd':>5} {'Sharpe':>7} "
           f"{'Sortino':>8} {'CAGR':>7} {'MaxDD':>7} {'WR':>6} {'PF':>6} "
           f"{'Gates':>6} {'Final$':>8} {'RandSh':>7}")
    fprint("-" * 115)

    variant_names = [v["name"] for v in VARIANTS]
    for vname in variant_names:
        r = all_results.get(vname)
        if not r or "error" in r:
            fprint(f"  {vname:<20} — INSUFFICIENT DATA —")
            continue
        fprint(f"  {vname:<20} {r['top_k']:>2} {r['rebal_days']:>3} {r['dte']:>4} "
               f"{r['n_trades']:>5} {r['sharpe']:>7.2f} {r['sortino']:>8.2f} "
               f"{r['cagr']*100:>6.1f}% {r['max_dd']*100:>6.1f}% "
               f"{r['win_rate']*100:>5.1f}% {r['profit_factor']:>5.2f} "
               f"{r['gates_passed']}/{r['gates_total']} "
               f"${r['final_equity']:>7,.0f} {r['random_mean_sharpe']:>7.2f}")

    # ── INTERACTION ANALYSIS ──
    fprint(f"\n{'=' * 80}")
    fprint("INTERACTION ANALYSIS: Do DTE and K/rebal improvements compound?")
    fprint(f"{'=' * 80}")

    baseline_sh = all_results.get("v4_baseline", {}).get("sharpe", 0)
    dte45_sh = all_results.get("dte45_only", {}).get("sharpe", 0)
    k2w_sh = all_results.get("k2_weekly_only", {}).get("sharpe", 0)
    k1w_sh = all_results.get("k1_weekly_only", {}).get("sharpe", 0)
    v5a_sh = all_results.get("v5a_k2_w_dte45", {}).get("sharpe", 0)
    v5b_sh = all_results.get("v5b_k1_w_dte45", {}).get("sharpe", 0)

    fprint(f"\n  Baseline (v4):           Sharpe = {baseline_sh:.3f}")
    fprint(f"  DTE45 only:              Sharpe = {dte45_sh:.3f}  (delta = {dte45_sh - baseline_sh:+.3f})")
    fprint(f"  K=2 weekly only:         Sharpe = {k2w_sh:.3f}  (delta = {k2w_sh - baseline_sh:+.3f})")
    fprint(f"  K=1 weekly only:         Sharpe = {k1w_sh:.3f}  (delta = {k1w_sh - baseline_sh:+.3f})")

    # Expected additive improvements
    dte_delta = dte45_sh - baseline_sh
    k2_delta = k2w_sh - baseline_sh
    k1_delta = k1w_sh - baseline_sh

    expected_v5a = baseline_sh + dte_delta + k2_delta
    expected_v5b = baseline_sh + dte_delta + k1_delta

    fprint(f"\n  If ADDITIVE (no interaction):")
    fprint(f"    v5a expected = {baseline_sh:.3f} + {dte_delta:+.3f} + {k2_delta:+.3f} = {expected_v5a:.3f}")
    fprint(f"    v5b expected = {baseline_sh:.3f} + {dte_delta:+.3f} + {k1_delta:+.3f} = {expected_v5b:.3f}")
    fprint(f"\n  ACTUAL:")
    fprint(f"    v5a actual   = {v5a_sh:.3f}  (vs expected {expected_v5a:.3f}, "
           f"interaction = {v5a_sh - expected_v5a:+.3f})")
    fprint(f"    v5b actual   = {v5b_sh:.3f}  (vs expected {expected_v5b:.3f}, "
           f"interaction = {v5b_sh - expected_v5b:+.3f})")

    # Determine best variant
    best_name = max(all_results.keys(),
                    key=lambda k: all_results[k].get("sharpe", -999) if "error" not in all_results[k] else -999)
    best = all_results[best_name]
    fprint(f"\n  BEST VARIANT: {best_name}")
    fprint(f"    Sharpe={best.get('sharpe', 0):.3f}, Sortino={best.get('sortino', 0):.3f}, "
           f"CAGR={best.get('cagr', 0)*100:.1f}%, "
           f"Gates={best.get('gates_passed', 0)}/{best.get('gates_total', 0)}")

    if best_name != "v4_baseline":
        improvement = best.get("sharpe", 0) - baseline_sh
        fprint(f"    Improvement over v4 baseline: {improvement:+.3f} Sharpe "
               f"({improvement/max(abs(baseline_sh), 0.001)*100:+.1f}%)")
    else:
        fprint(f"    No variant beats the current production v4 baseline.")

    # ── RECOMMENDATION ──
    fprint(f"\n{'=' * 80}")
    fprint("RECOMMENDATION")
    fprint(f"{'=' * 80}")

    # Check if best passes all 5 gates
    best_gates = best.get("gates_passed", 0)
    best_total = best.get("gates_total", 5)
    if best_name != "v4_baseline" and best_gates >= 4:
        fprint(f"  PROMOTE {best_name} to production v5")
        fprint(f"    Parameters: top_k={best['top_k']}, rebal_days={best['rebal_days']}, DTE={best['dte']}")
        fprint(f"    Sharpe: {baseline_sh:.3f} -> {best.get('sharpe', 0):.3f} "
               f"({(best.get('sharpe', 0) - baseline_sh)/max(abs(baseline_sh), 0.001)*100:+.1f}%)")
    elif best_name == "v4_baseline":
        fprint(f"  KEEP v4 baseline. No parameter change improves risk-adjusted returns.")
    else:
        fprint(f"  CAUTION: Best variant {best_name} only passes {best_gates}/{best_total} gates.")
        fprint(f"  Consider keeping v4 baseline until more evidence accumulated.")

    # Save results
    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    # MLflow logging
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"v5_crossval_{t0.strftime('%Y%m%d_%H%M')}"):
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
                    "haircut": DEFAULT_HAIRCUT,
                    "regime_bull_thresh": REGIME_BULL_THRESHOLD,
                    "regime_bear_thresh": REGIME_BEAR_THRESHOLD,
                    "hold_to_expiry": True,
                    "entry_haircut_only": True,
                    "n_features": len(V4_FEATURES),
                    "wf_train_periods": WF_TRAIN_PERIODS,
                    "n_variants": len(VARIANTS),
                    "best_variant": best_name,
                    "best_sharpe": best.get("sharpe", 0),
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
