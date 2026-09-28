#!/usr/bin/env python3
"""
Multi-Timeframe Momentum v1 — Entry Quality via Timeframe Agreement
=====================================================================

HYPOTHESIS: Combining multiple momentum timeframes (weekly/monthly/quarterly)
for entry timing improves trade quality beyond single-timeframe (21d) momentum.

Multi-timeframe agreement captures:
  - Fast (5d return) — weekly signal / timing
  - Medium (21d return) — monthly signal / direction
  - Slow (63d return) — quarterly trend / strength

When all three timeframes agree on direction, entry quality should be higher
(stronger conviction). Mixed signals = skip or reduce size.

NEW FEATURES (3 multi-timeframe features):
  1. mtf_agreement: count of timeframes aligned bullish (0-3)
  2. mtf_fast_vs_slow: 5d_ret - 63d_ret (trend acceleration/deceleration)
  3. mtf_reversal_risk: abs(5d_ret) / (abs(63d_ret) + eps) — high = potential reversal

5 VARIANTS:
  A: Baseline v4 (21 legacy + 3 cross-asset features, regime>0.4)
  B: v4 + 3 multi-timeframe features (24 total)
  C: Multi-timeframe agreement filter: only trade when mtf_agreement >= 2
  D: Fast-slow crossover: only trade when 5d > 0 AND 63d > 0
  E: Timeframe-weighted sizing: full size when 3/3, half when 2/3

All: walk-forward LGBM, $645 capital, 3% bull call spreads, DTE=21,
hold to expiry, 15% haircut, $2.60 commission.

Full 5-gate adversarial validation + random baseline.

Author: Claude Opus (autonomous research)
Date: 2026-07-27
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
    estimate_iv,
    compute_atr,
    COMMISSION_RT_SPREAD,
    DEFAULT_HAIRCUT,
)
from research.tools.adversarial_validator import validate_trades

# -- Config --
BASE = Path("/home/jupiter/Lvl3Quant")
OUTPUT_DIR = BASE / "output" / "growth_research" / "multi_timeframe_momentum_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "TLT", "HYG", "GLD"]
CAP = 645.0
DTE = 21
SPREAD_PCT = 3.0
TOP_K = 3
REGIME_THRESHOLD = 0.4

# Regime predictions
REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"

# Walk-forward params (sliding window, biweekly rebalance)
WF_TRAIN_PERIODS = 12
WF_REBAL_FREQ = "2W-FRI"

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "multi_timeframe_momentum_v1"

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

np.random.seed(42)


# ============================================================
# DATA DOWNLOAD
# ============================================================

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

    rename_map = {"^VIX": "VIX"}
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


# ============================================================
# REGIME LOADING
# ============================================================

def load_regime_predictions(close):
    """Load GRU regime predictions or fall back to VIX-based proxy."""
    if REGIME_FILE.exists():
        data = np.load(REGIME_FILE, allow_pickle=True)
        dates = pd.to_datetime(data["dates"])
        scores = data["regime_scores"]
        regime_series = pd.Series(scores, index=dates, name="regime_score")
        regime_series = regime_series[~regime_series.index.duplicated(keep="last")]
        fprint(f"Regime predictions loaded: {len(regime_series)} days")
        return regime_series

    # Fallback: SPY SMA200-based regime proxy
    # Score > 0.4 when SPY > SMA200 (bullish regime = favorable for bull spreads)
    fprint("Regime file not found -- using SPY SMA200 proxy")
    spy = close["SPY"]
    sma200 = spy.rolling(200).mean()
    # Map: SPY > SMA200 -> score 0.7 (bullish), SPY < SMA200 -> score 0.2 (bearish)
    regime_series = pd.Series(
        np.where(spy > sma200, 0.7, 0.2),
        index=spy.index,
        name="regime_score"
    )
    fprint(f"  Regime proxy: {(regime_series > REGIME_THRESHOLD).sum()} bullish days / {len(regime_series)} total")
    return regime_series


# ============================================================
# FEATURE ENGINEERING
# ============================================================

# 18 legacy quality-momentum features
LEGACY_FEATURES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_21d", "vol_63d", "sharpe_63d", "maxdd_63d", "pct_52w_high", "mom_accel",
    "pct_pos_months_12m", "sortino_63d", "calmar_1y", "up_capture",
    "trend_r2_63d", "trend_slope_63d",
]

# 3 cross-asset features from v4
CROSS_ASSET_FEATURES = [
    "spy_tlt_corr_21d",
    "gld_momentum_21d",
    "vix_slope_change_5d",
]

# 3 NEW multi-timeframe features
MTF_FEATURES = [
    "mtf_agreement",      # count of timeframes aligned bullish (0-3)
    "mtf_fast_vs_slow",   # 5d_ret - 63d_ret (trend acceleration)
    "mtf_reversal_risk",  # abs(5d_ret) / abs(63d_ret) (reversal risk)
]

# Combined feature sets
V4_FEATURES = LEGACY_FEATURES + CROSS_ASSET_FEATURES  # 21 features
V4_PLUS_MTF_FEATURES = V4_FEATURES + MTF_FEATURES     # 24 features


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
    """Compute the 3 cross-asset features from v4."""
    f = {}
    spy = close_df["SPY"].iloc[:dt_idx + 1].dropna()
    spy_ret = spy.pct_change().dropna()

    # SPY-TLT rolling 21d correlation
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

    # GLD momentum 21d
    if "GLD" in close_df.columns:
        gld = close_df["GLD"].iloc[:dt_idx + 1].dropna()
        if len(gld) > 21:
            f["gld_momentum_21d"] = float(gld.iloc[-1] / gld.iloc[-21] - 1)
        else:
            f["gld_momentum_21d"] = 0.0
    else:
        f["gld_momentum_21d"] = 0.0

    # VIX slope change 5d
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

    return f


def compute_mtf_features(px):
    """Compute the 3 multi-timeframe momentum features."""
    f = {}
    if len(px) < 64:
        return {"mtf_agreement": 0.0, "mtf_fast_vs_slow": 0.0, "mtf_reversal_risk": 0.0}

    ret_5d = float(px.iloc[-1] / px.iloc[-5] - 1) if len(px) > 5 else 0.0
    ret_21d = float(px.iloc[-1] / px.iloc[-21] - 1) if len(px) > 21 else 0.0
    ret_63d = float(px.iloc[-1] / px.iloc[-63] - 1) if len(px) > 63 else 0.0

    # mtf_agreement: count of timeframes aligned bullish (0-3)
    agreement = 0
    if ret_5d > 0:
        agreement += 1
    if ret_21d > 0:
        agreement += 1
    if ret_63d > 0:
        agreement += 1
    f["mtf_agreement"] = float(agreement)

    # mtf_fast_vs_slow: trend acceleration (positive = accelerating uptrend)
    f["mtf_fast_vs_slow"] = ret_5d - ret_63d

    # mtf_reversal_risk: high fast move relative to slow = potential reversal
    f["mtf_reversal_risk"] = abs(ret_5d) / (abs(ret_63d) + 1e-6)

    return f


# ============================================================
# WALK-FORWARD LGBM RANKING
# ============================================================

def build_feature_records(close, high, low, rebal_dates, feature_cols, regime_series):
    """Build feature + target records for all sectors on all rebal dates."""
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

        # Regime filter
        if regime_series is not None:
            if dt in regime_series.index:
                score = regime_series.loc[dt]
            else:
                nearest = regime_series.index[regime_series.index.get_indexer([dt], method="ffill")]
                score = regime_series.loc[nearest[0]] if len(nearest) > 0 else 0.0
            if score <= REGIME_THRESHOLD:
                continue

        for tk in sector_cols:
            px = close[tk].iloc[:idx + 1].dropna()
            spy_s = spy.iloc[:idx + 1]

            legacy = compute_legacy_features(px, spy_s)
            if not legacy:
                continue

            # Cross-asset features
            cross_asset = {}
            if any(c in CROSS_ASSET_FEATURES for c in feature_cols):
                cross_asset = compute_cross_asset_features(tk, idx, close)

            # Multi-timeframe features
            mtf = {}
            if any(c in MTF_FEATURES for c in feature_cols):
                mtf = compute_mtf_features(px)

            # Forward return target (DTE days forward)
            fi = min(idx + DTE, len(close) - 1)
            if fi <= idx:
                continue
            fwd_ret = float(close[tk].iloc[fi] / close[tk].iloc[idx] - 1)

            rec = {**legacy, **cross_asset, **mtf, "date": dt, "ticker": tk, "fwd_ret": fwd_ret}
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

            # Store per-record data (ticker, score, and mtf features for filtering)
            rec_data = {}
            for _, row in test_df.iterrows():
                rec_data[row["ticker"]] = {
                    "score": row["score"],
                    "mtf_agreement": row.get("mtf_agreement", 0),
                    "ret_5d": row.get("ret_5d", 0),
                    "ret_63d": row.get("ret_63d", 0),
                }
            rankings[test_date] = rec_data

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


# ============================================================
# OPTIONS SIMULATION
# ============================================================

def simulate_trades(name, rankings, close, high, low, regime_series,
                    mtf_filter=None, sizing_mode="full"):
    """
    Simulate bull call spread trades from rankings.

    Args:
        mtf_filter: None, "agreement_2" (mtf>=2), "fast_slow" (5d>0 AND 63d>0)
        sizing_mode: "full" (always 1 contract) or "mtf_weighted" (half when 2/3, full when 3/3)
    """
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None
    sector_cols = [c for c in SECTORS if c in close.columns]

    # Pre-compute ATR series
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

        rec_data = rankings[dt]
        if not rec_data:
            continue

        # Apply MTF filter before ranking
        if mtf_filter == "agreement_2":
            rec_data = {tk: d for tk, d in rec_data.items() if d.get("mtf_agreement", 0) >= 2}
        elif mtf_filter == "fast_slow":
            rec_data = {tk: d for tk, d in rec_data.items()
                        if d.get("ret_5d", 0) > 0 and d.get("ret_63d", 0) > 0}

        if not rec_data:
            continue

        # Rank by LGBM score
        ranked = sorted(rec_data.items(), key=lambda x: x[1]["score"], reverse=True)
        picks = [(t, d) for t, d in ranked[:TOP_K]]

        # Position sizing
        max_pos = min(200, equity / 3)
        if max_pos < 30:
            continue

        n_entered = 0
        for tk, data in picks:
            if tk not in close.columns or tk not in atr_dict or n_entered >= TOP_K:
                continue

            S = float(close[tk].loc[dt])
            di = close.index.get_loc(dt)
            ei = min(di + DTE, len(close) - 1)
            if ei <= di:
                continue

            av = float(atr_dict[tk].iloc[di]) if di < len(atr_dict[tk]) and not pd.isna(atr_dict[tk].iloc[di]) else S * 0.015

            K1 = round(S, 2)
            K2 = round(S * (1 + SPREAD_PCT / 100), 2)
            if K2 <= K1:
                K2 = K1 + 1.0

            try:
                entry_cost_per_share, max_profit_per_share = price_bull_call_spread(
                    S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=cv
                )
            except Exception:
                continue

            # Sizing modifier
            size_mult = 1.0
            if sizing_mode == "mtf_weighted":
                mtf_agr = data.get("mtf_agreement", 0)
                if mtf_agr >= 3:
                    size_mult = 1.0
                elif mtf_agr >= 2:
                    size_mult = 0.5
                else:
                    continue  # skip 0-1 agreement in weighted mode

            total_cost = entry_cost_per_share * 100 + COMMISSION_RT_SPREAD
            adj_cost = total_cost * size_mult

            if adj_cost <= 0 or adj_cost > max_pos or adj_cost > equity * 0.40:
                continue

            Se = float(close[tk].iloc[ei])
            intrinsic_long = max(Se - K1, 0.0)
            intrinsic_short = max(Se - K2, 0.0)
            exit_value_per_share = intrinsic_long - intrinsic_short

            pnl = ((exit_value_per_share - entry_cost_per_share) * 100 - COMMISSION_RT_SPREAD) * size_mult
            equity += pnl
            n_entered += 1

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
                "mtf_agreement": int(data.get("mtf_agreement", 0)),
                "size_mult": size_mult,
            })

    return trades, equity


# ============================================================
# RANDOM BASELINE
# ============================================================

def run_random_baseline(close, high, low, regime_series, n_trials=200):
    """Run random sector selection baseline for comparison."""
    fprint("\n  Running random baseline (200 trials)...")

    sector_cols = [c for c in SECTORS if c in close.columns]
    rebal_dates = pd.DatetimeIndex(
        close.index.to_series().resample(WF_REBAL_FREQ).last().dropna().values
    )

    # Filter by regime
    valid_dates = []
    for dt in rebal_dates:
        if regime_series is not None:
            if dt in regime_series.index:
                score = regime_series.loc[dt]
            else:
                nearest = regime_series.index[regime_series.index.get_indexer([dt], method="ffill")]
                score = regime_series.loc[nearest[0]] if len(nearest) > 0 else 0.0
            if score > REGIME_THRESHOLD:
                valid_dates.append(dt)

    sharpes = []
    for trial in range(n_trials):
        # Random rankings
        random_rankings = {}
        rng = np.random.RandomState(trial + 1000)
        for dt in valid_dates:
            idx = close.index.get_indexer([dt], method="ffill")[0]
            if idx < 260 or idx + DTE >= len(close):
                continue
            rec_data = {}
            for tk in sector_cols:
                rec_data[tk] = {
                    "score": rng.random(),
                    "mtf_agreement": rng.randint(0, 4),
                    "ret_5d": rng.randn() * 0.02,
                    "ret_63d": rng.randn() * 0.05,
                }
            random_rankings[dt] = rec_data

        trades, _ = simulate_trades(f"random_{trial}", random_rankings, close, high, low, regime_series)
        if len(trades) > 20:
            pnls = np.array([t["pnl"] for t in trades])
            eq = CAP + np.cumsum(pnls)
            eq_s = pd.Series(np.concatenate([[CAP], eq]),
                             index=pd.date_range("2010-01-01", periods=len(eq)+1, freq="14D"))
            from research.tools.adversarial_validator import compute_honest_sharpe
            sh, _, _ = compute_honest_sharpe(eq_s)
            sharpes.append(sh)

    if sharpes:
        median_sh = np.median(sharpes)
        p95_sh = np.percentile(sharpes, 95)
        fprint(f"    Random baseline: median Sharpe={median_sh:.2f}, p95={p95_sh:.2f} ({len(sharpes)} valid trials)")
        return {"median_sharpe": float(median_sh), "p95_sharpe": float(p95_sh), "n_trials": len(sharpes)}
    return {"median_sharpe": 0.0, "p95_sharpe": 0.0, "n_trials": 0}


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = datetime.now()
    fprint("=" * 80)
    fprint(f"Multi-Timeframe Momentum v1 -- {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 80)
    fprint(f"Capital: ${CAP:.0f} | DTE: {DTE} | Spread: {SPREAD_PCT:.0f}% | "
           f"Haircut: {DEFAULT_HAIRCUT:.0%} | Comm: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"Regime threshold: >{REGIME_THRESHOLD}")
    fprint(f"Multi-timeframe horizons: 5d (fast) / 21d (medium) / 63d (slow)")
    fprint()

    # 1. Download data
    close, high, low = download_data()

    # 2. Load regime predictions
    regime_series = load_regime_predictions(close)

    # 3. Build rebalance dates
    rebal_dates = pd.DatetimeIndex(
        close.index.to_series().resample(WF_REBAL_FREQ).last().dropna().values
    )
    fprint(f"Rebalance dates: {len(rebal_dates)} ({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

    # ============================================================
    # VARIANT A: Baseline v4 (21 features)
    # ============================================================
    fprint("\n" + "=" * 80)
    fprint("VARIANT A: Baseline v4 (21 features: 18 legacy + 3 cross-asset)")
    fprint("=" * 80)
    records_a = build_feature_records(close, high, low, rebal_dates, V4_FEATURES, regime_series)
    rankings_a, imp_a = walk_forward_lgbm_rank(records_a, V4_FEATURES, "Variant_A")

    # ============================================================
    # VARIANT B: v4 + 3 MTF features (24 total)
    # ============================================================
    fprint("\n" + "=" * 80)
    fprint("VARIANT B: v4 + 3 multi-timeframe features (24 total)")
    fprint("=" * 80)
    records_b = build_feature_records(close, high, low, rebal_dates, V4_PLUS_MTF_FEATURES, regime_series)
    rankings_b, imp_b = walk_forward_lgbm_rank(records_b, V4_PLUS_MTF_FEATURES, "Variant_B")

    # ============================================================
    # Simulate trades for all 5 variants
    # ============================================================
    fprint("\n" + "=" * 80)
    fprint("SIMULATING TRADES (5 variants)")
    fprint("=" * 80)

    spy_close = close["SPY"]
    all_results = {}

    variants = [
        # (name, rankings, feature_cols, imp_df, mtf_filter, sizing_mode)
        ("A_Baseline_v4_21feat", rankings_a, V4_FEATURES, imp_a, None, "full"),
        ("B_v4_plus_MTF_24feat", rankings_b, V4_PLUS_MTF_FEATURES, imp_b, None, "full"),
        ("C_MTF_agreement_ge2", rankings_b, V4_PLUS_MTF_FEATURES, imp_b, "agreement_2", "full"),
        ("D_FastSlow_crossover", rankings_b, V4_PLUS_MTF_FEATURES, imp_b, "fast_slow", "full"),
        ("E_MTF_weighted_sizing", rankings_b, V4_PLUS_MTF_FEATURES, imp_b, None, "mtf_weighted"),
    ]

    for vname, rankings, feat_cols, imp_df, mtf_filter, sizing_mode in variants:
        fprint(f"\n--- {vname} ---")
        if not rankings:
            fprint("  No rankings available, skipping")
            continue

        trades, final_eq = simulate_trades(
            vname, rankings, close, high, low, regime_series,
            mtf_filter=mtf_filter, sizing_mode=sizing_mode
        )

        if not trades:
            fprint("  No trades generated, skipping")
            continue

        fprint(f"  {len(trades)} trades, final equity: ${final_eq:.2f}")

        # MTF agreement distribution
        if any(t.get("mtf_agreement") is not None for t in trades):
            agr_counts = {}
            for t in trades:
                a = t.get("mtf_agreement", -1)
                agr_counts[a] = agr_counts.get(a, 0) + 1
            fprint(f"  MTF agreement distribution: {dict(sorted(agr_counts.items()))}")

            # Win rate by MTF agreement
            for a_level in sorted(agr_counts.keys()):
                subset = [t for t in trades if t.get("mtf_agreement") == a_level]
                if len(subset) >= 5:
                    wr = sum(1 for t in subset if t["pnl"] > 0) / len(subset)
                    avg_pnl = np.mean([t["pnl"] for t in subset])
                    fprint(f"    Agreement={a_level}: {len(subset)} trades, WR={wr:.1%}, avg PnL=${avg_pnl:.2f}")

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
                marker = ""
                if row["feature"] in MTF_FEATURES:
                    marker = " **MTF**"
                elif row["feature"] in CROSS_ASSET_FEATURES:
                    marker = " *CROSS*"
                fprint(f"    {row['feature']:<30s} {row['importance']:>8.1f}{marker}")

        r = result.to_dict()
        r["features_used"] = feat_cols
        r["n_features"] = len(feat_cols)
        r["mtf_filter"] = mtf_filter
        r["sizing_mode"] = sizing_mode
        if imp_df is not None:
            r["feature_importance"] = imp_df.to_dict("records")
        all_results[vname] = r

    # ============================================================
    # RANDOM BASELINE
    # ============================================================
    fprint("\n" + "=" * 80)
    fprint("RANDOM BASELINE")
    fprint("=" * 80)
    random_result = run_random_baseline(close, high, low, regime_series)

    # ============================================================
    # COMPARATIVE SUMMARY
    # ============================================================
    fprint("\n" + "=" * 80)
    fprint("COMPARATIVE SUMMARY")
    fprint("=" * 80)
    fprint(f"{'Variant':<28s} {'N':>5s} {'Sharpe':>7s} {'Sortino':>8s} {'WR%':>6s} "
           f"{'PF':>6s} {'CAGR%':>7s} {'MaxDD%':>7s} {'Gates':>6s} {'Final$':>8s}")
    fprint("-" * 100)

    ordered_names = [
        "A_Baseline_v4_21feat",
        "B_v4_plus_MTF_24feat",
        "C_MTF_agreement_ge2",
        "D_FastSlow_crossover",
        "E_MTF_weighted_sizing",
    ]

    for vname in ordered_names:
        if vname not in all_results:
            fprint(f"  {vname}: NO DATA")
            continue
        r = all_results[vname]
        fprint(f"  {vname:<26s} {r['n_trades']:>5d} {r['sharpe']:>7.2f} {r['sortino']:>8.2f} "
               f"{r['win_rate']*100:>6.1f} {r['profit_factor']:>6.2f} "
               f"{r['cagr']*100:>7.1f} {r['max_dd']*100:>7.1f} "
               f"{r['gates_passed']}/{r['gates_total']}   ${r['final_equity']:>8.0f}")

    fprint(f"\n  Random baseline: median Sharpe={random_result['median_sharpe']:.2f}, "
           f"p95 Sharpe={random_result['p95_sharpe']:.2f}")

    # ============================================================
    # KEY FINDINGS
    # ============================================================
    fprint("\n" + "=" * 80)
    fprint("KEY FINDINGS: Does multi-timeframe momentum improve entry quality?")
    fprint("=" * 80)

    if "A_Baseline_v4_21feat" in all_results and "B_v4_plus_MTF_24feat" in all_results:
        a = all_results["A_Baseline_v4_21feat"]
        b = all_results["B_v4_plus_MTF_24feat"]
        sh_diff = b["sharpe"] - a["sharpe"]
        if sh_diff > 0.1:
            fprint(f"  1. MTF features ADD value: Sharpe {a['sharpe']:.2f} -> {b['sharpe']:.2f} (+{sh_diff:.2f})")
        elif sh_diff > -0.1:
            fprint(f"  1. MTF features NEUTRAL: Sharpe {a['sharpe']:.2f} -> {b['sharpe']:.2f} ({sh_diff:+.2f})")
        else:
            fprint(f"  1. MTF features HURT: Sharpe {a['sharpe']:.2f} -> {b['sharpe']:.2f} ({sh_diff:+.2f})")

    if "C_MTF_agreement_ge2" in all_results and "A_Baseline_v4_21feat" in all_results:
        a = all_results["A_Baseline_v4_21feat"]
        c = all_results["C_MTF_agreement_ge2"]
        fprint(f"  2. Agreement filter (>=2): Sharpe {c['sharpe']:.2f} vs baseline {a['sharpe']:.2f} "
               f"({c['sharpe']-a['sharpe']:+.2f}), "
               f"WR {c['win_rate']*100:.1f}% vs {a['win_rate']*100:.1f}%, "
               f"but {c['n_trades']} vs {a['n_trades']} trades")

    if "D_FastSlow_crossover" in all_results and "A_Baseline_v4_21feat" in all_results:
        a = all_results["A_Baseline_v4_21feat"]
        d = all_results["D_FastSlow_crossover"]
        fprint(f"  3. Fast-slow crossover: Sharpe {d['sharpe']:.2f} vs baseline {a['sharpe']:.2f} "
               f"({d['sharpe']-a['sharpe']:+.2f}), {d['n_trades']} trades")

    if "E_MTF_weighted_sizing" in all_results and "A_Baseline_v4_21feat" in all_results:
        a = all_results["A_Baseline_v4_21feat"]
        e = all_results["E_MTF_weighted_sizing"]
        fprint(f"  4. MTF-weighted sizing: Sharpe {e['sharpe']:.2f} vs baseline {a['sharpe']:.2f} "
               f"({e['sharpe']-a['sharpe']:+.2f})")

    # All vs random
    for vname in ordered_names:
        if vname in all_results:
            sh = all_results[vname]["sharpe"]
            vs_random = "BEATS" if sh > random_result["p95_sharpe"] else "BELOW"
            fprint(f"  {vname}: Sharpe {sh:.2f} {vs_random} random p95 ({random_result['p95_sharpe']:.2f})")

    # ============================================================
    # MLFLOW LOGGING
    # ============================================================
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"mtf_momentum_v1_{t0.strftime('%Y%m%d_%H%M')}"):
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

                mlflow.log_metric("random_median_sharpe", random_result["median_sharpe"])
                mlflow.log_metric("random_p95_sharpe", random_result["p95_sharpe"])

                mlflow.log_param("capital", CAP)
                mlflow.log_param("dte", DTE)
                mlflow.log_param("spread_pct", SPREAD_PCT)
                mlflow.log_param("haircut", DEFAULT_HAIRCUT)
                mlflow.log_param("commission", COMMISSION_RT_SPREAD)
                mlflow.log_param("regime_threshold", REGIME_THRESHOLD)
                mlflow.log_param("wf_train_periods", WF_TRAIN_PERIODS)
                mlflow.log_param("top_k", TOP_K)
                mlflow.log_param("mtf_horizons", "5d_21d_63d")

                results_path = OUTPUT_DIR / "mtf_momentum_v1_results.json"
                clean = {}
                for k, v in all_results.items():
                    c = {}
                    for kk, vv in v.items():
                        if isinstance(vv, (np.floating, np.integer)):
                            c[kk] = float(vv)
                        elif isinstance(vv, np.ndarray):
                            c[kk] = vv.tolist()
                        else:
                            c[kk] = vv
                    clean[k] = c
                clean["random_baseline"] = random_result
                with open(results_path, "w") as f:
                    json.dump(clean, f, indent=2, default=str)
                mlflow.log_artifact(str(results_path))

            fprint(f"\nMLflow run logged to experiment '{EXPERIMENT_NAME}'")
        except Exception as e:
            fprint(f"\nMLflow logging failed: {e}")

    # Save results to disk
    results_path = OUTPUT_DIR / "mtf_momentum_v1_results.json"
    clean = {}
    for k, v in all_results.items():
        c = {}
        for kk, vv in v.items():
            if isinstance(vv, (np.floating, np.integer)):
                c[kk] = float(vv)
            elif isinstance(vv, np.ndarray):
                c[kk] = vv.tolist()
            else:
                c[kk] = vv
        clean[k] = c
    clean["random_baseline"] = random_result
    with open(results_path, "w") as f:
        json.dump(clean, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    fprint("Done.")


if __name__ == "__main__":
    main()
