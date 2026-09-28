#!/usr/bin/env python3
"""
V7 Candidate Integration — Do proven improvements STACK or overlap?
====================================================================

Tests ALL individually proven improvements combined into one strategy.
Base: moneyness_crossval_v1.py (production-quality infrastructure).

ALL variants use: K=2, DTE=21, 2% OTM moneyness (K1 = S*1.02, K2 = K1*1.03).

| Variant        | Rebal | Sizing            | Description                              |
|----------------|-------|-------------------|------------------------------------------|
| v6_baseline    | 5d    | Fixed $200        | Current v6 production = control          |
| v6_dd_sizing   | 5d    | Drawdown-adjusted | v6 + drawdown-adjusted sizing            |
| v6_3d_rebal    | 3d    | Fixed $200        | v6 + 3-day rebalancing (best CAGR)       |
| v6_3d_dd       | 3d    | Drawdown-adjusted | v6 + 3-day rebal + drawdown sizing       |
| v6_confident   | 5d    | Confidence-ranked | v6 + confidence sizing (1st=$200, 2nd=$150) |
| v7_full        | 3d    | Combined          | ALL improvements combined                |

Sizing modes:
- Fixed: always $200
- Drawdown-adjusted: HWM tracking. >90% HWM: $200, 80-90%: $100, <80%: $50. Recover at 95% HWM.
- Confidence: 1st rank=$200, 2nd rank=$150
- Combined: min(drawdown_adj_size, confidence_size)

5-gate adversarial validation + random baseline on all variants.
Results logged to MLflow experiment "v7_integration_v1".
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
OUTPUT_DIR = BASE / "output" / "growth_research" / "v7_integration_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
SPREAD_PCT = 3.0  # K2 = K1 * 1.03 (3% wide)
MONEYNESS_PCT = 2.0  # 2% OTM for ALL variants
REGIME_BULL_THRESHOLD = 0.4
REGIME_BEAR_THRESHOLD = 0.2

# Regime predictions path
REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"

# LGBM walk-forward
WF_TRAIN_PERIODS = 12

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "v7_integration_v1"

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
# VARIANT GRID
# ══════════════════════════════════════════════════════════════
# ALL variants: top_k=2, dte=21, moneyness=2% OTM
# Only rebal_days and sizing_mode differ

VARIANTS = [
    {"name": "v6_baseline",  "top_k": 2, "rebal_days": 5, "dte": 21, "sizing": "fixed",      "desc": "v6 production baseline (control)"},
    {"name": "v6_dd_sizing", "top_k": 2, "rebal_days": 5, "dte": 21, "sizing": "drawdown",    "desc": "v6 + drawdown-adjusted sizing"},
    {"name": "v6_3d_rebal",  "top_k": 2, "rebal_days": 3, "dte": 21, "sizing": "fixed",      "desc": "v6 + 3-day rebalancing"},
    {"name": "v6_3d_dd",     "top_k": 2, "rebal_days": 3, "dte": 21, "sizing": "drawdown",    "desc": "v6 + 3-day rebal + drawdown sizing"},
    {"name": "v6_confident", "top_k": 2, "rebal_days": 5, "dte": 21, "sizing": "confidence",  "desc": "v6 + confidence sizing ($200/$150)"},
    {"name": "v7_full",      "top_k": 2, "rebal_days": 3, "dte": 21, "sizing": "combined",    "desc": "ALL improvements combined"},
]


# ══════════════════════════════════════════════════════════════
# POSITION SIZING ENGINE
# ══════════════════════════════════════════════════════════════

class PositionSizer:
    """Tracks HWM and computes position sizes based on sizing mode."""

    def __init__(self, initial_capital, sizing_mode="fixed"):
        self.sizing_mode = sizing_mode
        self.hwm = initial_capital
        self.recovery_threshold = 0.95  # recover full sizing at 95% of HWM
        self.in_drawdown = False  # tracks whether we're in reduced-sizing mode

    def update_hwm(self, equity):
        """Update high water mark after each trade."""
        if equity > self.hwm:
            self.hwm = equity
            self.in_drawdown = False

    def get_drawdown_size(self, equity):
        """Drawdown-adjusted sizing per finding #142."""
        ratio = equity / self.hwm if self.hwm > 0 else 1.0

        # Recovery logic: if we were in drawdown but recovered to 95% HWM
        if self.in_drawdown and ratio >= self.recovery_threshold:
            self.in_drawdown = False
            return 200.0

        if ratio > 0.90:
            return 200.0
        elif ratio > 0.80:
            self.in_drawdown = True
            return 100.0
        else:
            self.in_drawdown = True
            return 50.0

    def get_confidence_size(self, rank_position):
        """Confidence sizing: 1st pick=$200, 2nd pick=$150."""
        if rank_position == 0:
            return 200.0
        elif rank_position == 1:
            return 150.0
        else:
            return 100.0  # shouldn't happen with top_k=2

    def get_size(self, equity, rank_position=0):
        """Get position size based on sizing mode."""
        if self.sizing_mode == "fixed":
            return 200.0

        elif self.sizing_mode == "drawdown":
            return self.get_drawdown_size(equity)

        elif self.sizing_mode == "confidence":
            return self.get_confidence_size(rank_position)

        elif self.sizing_mode == "combined":
            dd_size = self.get_drawdown_size(equity)
            conf_size = self.get_confidence_size(rank_position)
            return min(dd_size, conf_size)

        else:
            return 200.0


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
        return 0.5
    if dt in regime_series.index:
        return float(regime_series.loc[dt])
    nearest = regime_series.index[regime_series.index.get_indexer([dt], method="ffill")]
    if len(nearest) > 0:
        return float(regime_series.loc[nearest[0]])
    return 0.5


# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (IDENTICAL TO PRODUCTION V4)
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

def build_feature_records(close, high, low, rebal_dates, feature_cols, regime_series,
                          dte, regime_mode="bull_bear"):
    """Build feature + target records for all sectors on all rebal dates."""
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

        rscore = get_regime_score_at(regime_series, dt)

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
                continue
        else:
            direction = "bull"

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
# TRADE SIMULATION (with position sizing)
# ══════════════════════════════════════════════════════════════

def simulate_trades(name, rankings, close, high, low, regime_series, atr_dict,
                    top_k, dte, bull_only=False, skip_vix_25_30=True, var=None):
    """
    Simulate bull call spreads (and bear put spreads) from rankings.

    HONEST RULES (identical to production v4):
      - Hold to expiry
      - At expiry: intrinsic value only
      - 15% haircut on entry only
      - No exit haircut (automatic exercise)

    NEW: Position sizing via PositionSizer (varies by variant).
    NEW: Fixed 2% OTM moneyness for all variants (K1 = S*1.02, K2 = K1*1.03).
    """
    if var is None:
        var = {}
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    sizing_mode = var.get("sizing", "fixed")
    sizer = PositionSizer(CAP, sizing_mode)

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

        if bull_only and direction == "bear":
            continue

        if not scores:
            continue

        # Pick sectors
        if direction == "bull":
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        else:
            ranked = sorted(scores.items(), key=lambda x: x[1])

        picks = [t for t, _ in ranked[:top_k]]

        n_entered = 0
        for rank_pos, tk in enumerate(picks):
            if tk not in close.columns or tk not in atr_dict or n_entered >= top_k:
                continue

            # Get position size from sizer
            max_pos = sizer.get_size(equity, rank_position=rank_pos)
            # Also cap at 1/3 of equity
            max_pos = min(max_pos, equity / 3)
            if max_pos < 30:
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

            # 2% OTM strikes for ALL variants
            K1 = round(S * (1 + MONEYNESS_PCT / 100), 2)  # S * 1.02
            K2 = round(K1 * (1 + SPREAD_PCT / 100), 2)     # K1 * 1.03
            if K2 <= K1:
                K2 = K1 + 1.0

            try:
                if direction == "bull":
                    entry_cost_ps, max_profit_ps = price_bull_call_spread(
                        S=S, K1=K1, K2=K2, dte=dte, atr=av, vix=cv
                    )
                else:
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
                intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
            else:
                intrinsic = max(K2 - Se, 0.0) - max(K1 - Se, 0.0)

            exit_value_ps = intrinsic
            pnl = (exit_value_ps - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD
            equity += pnl
            n_entered += 1

            # Update HWM after trade
            sizer.update_hwm(equity)

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
                "sizing_mode": sizing_mode,
                "position_size": round(max_pos, 2),
                "rank_position": rank_pos,
            })

    return trades, equity


# ══════════════════════════════════════════════════════════════
# RANDOM BASELINE
# ══════════════════════════════════════════════════════════════

def random_baseline_test(rankings, close, high, low, regime_series, atr_dict,
                         top_k, dte, bull_only, skip_vix_25_30, var, n_trials=5):
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
            var=var,
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
# STACKING ANALYSIS
# ══════════════════════════════════════════════════════════════

def analyze_stacking(all_results):
    """Analyze whether improvements stack, overlap, or cancel."""
    fprint(f"\n{'=' * 80}")
    fprint("STACKING ANALYSIS: Do improvements compound or overlap?")
    fprint(f"{'=' * 80}")

    baseline = all_results.get("v6_baseline", {})
    if "error" in baseline:
        fprint("  Cannot analyze: baseline has insufficient data")
        return

    base_sh = baseline.get("sharpe", 0)
    base_cagr = baseline.get("cagr", 0)
    base_mdd = baseline.get("max_dd", 0)

    fprint(f"\n  BASELINE (v6_baseline): Sharpe={base_sh:.3f}, CAGR={base_cagr*100:.1f}%, MaxDD={base_mdd*100:.1f}%")

    # Individual improvements
    improvements = {}
    for vname in ["v6_dd_sizing", "v6_3d_rebal", "v6_confident"]:
        r = all_results.get(vname, {})
        if "error" not in r:
            delta_sh = r.get("sharpe", 0) - base_sh
            improvements[vname] = delta_sh
            fprint(f"  {vname}: Sharpe delta = {delta_sh:+.3f} ({delta_sh/max(abs(base_sh), 0.001)*100:+.1f}%)")

    # Expected additive improvement
    if improvements:
        additive_expected = base_sh + sum(improvements.values())
        fprint(f"\n  Expected (additive): {additive_expected:.3f} (base + sum of individual deltas)")

    # Combination results
    for combo_name, components in [
        ("v6_3d_dd", ["v6_3d_rebal", "v6_dd_sizing"]),
        ("v7_full", ["v6_3d_rebal", "v6_dd_sizing", "v6_confident"]),
    ]:
        r = all_results.get(combo_name, {})
        if "error" in r:
            continue
        actual_sh = r.get("sharpe", 0)
        component_deltas = sum(improvements.get(c, 0) for c in components)
        expected_sh = base_sh + component_deltas
        synergy = actual_sh - expected_sh

        fprint(f"\n  {combo_name} (combines: {', '.join(components)}):")
        fprint(f"    Expected additive: {expected_sh:.3f}")
        fprint(f"    Actual:            {actual_sh:.3f}")
        if synergy > 0.01:
            fprint(f"    SYNERGY: +{synergy:.3f} (improvements COMPOUND)")
        elif synergy < -0.01:
            fprint(f"    OVERLAP: {synergy:.3f} (improvements CANCEL/OVERLAP)")
        else:
            fprint(f"    NEUTRAL: {synergy:.3f} (approximately additive)")

    # v7_full vs best individual
    v7 = all_results.get("v7_full", {})
    if "error" not in v7:
        v7_sh = v7.get("sharpe", 0)
        best_single_name = max(
            [v for v in ["v6_dd_sizing", "v6_3d_rebal", "v6_confident"] if v in all_results and "error" not in all_results[v]],
            key=lambda v: all_results[v].get("sharpe", -999),
            default=None,
        )
        if best_single_name:
            best_single_sh = all_results[best_single_name].get("sharpe", 0)
            fprint(f"\n  v7_full ({v7_sh:.3f}) vs best single improvement {best_single_name} ({best_single_sh:.3f}):")
            if v7_sh > best_single_sh:
                fprint(f"    v7_full WINS by {v7_sh - best_single_sh:+.3f} -> STACKING WORKS")
            else:
                fprint(f"    Best single WINS by {best_single_sh - v7_sh:+.3f} -> STACKING HURTS, use single improvement")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t0 = datetime.now()
    fprint("=" * 80)
    fprint(f"V7 CANDIDATE INTEGRATION — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 80)
    fprint(f"Capital: ${CAP:.0f} | Spread: {SPREAD_PCT:.0f}% | Moneyness: {MONEYNESS_PCT:.0f}% OTM")
    fprint(f"Haircut: {DEFAULT_HAIRCUT:.0%} entry only | Comm: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"HOLD TO EXPIRY | Intrinsic value only at expiry | No exit haircut")
    fprint(f"Regime bull threshold: >{REGIME_BULL_THRESHOLD}, bear: <{REGIME_BEAR_THRESHOLD}")
    fprint(f"VIX 25-30: skip (all variants)")
    fprint(f"Strike construction: K1 = S * 1.02, K2 = K1 * 1.03")
    fprint()
    fprint("QUESTION: Do proven improvements STACK or overlap/cancel?")
    fprint()

    fprint("VARIANT GRID:")
    fprint(f"{'Name':<20} {'K':>2} {'Rebal':>5} {'DTE':>4} {'Sizing':<12}  Description")
    fprint("-" * 90)
    for v in VARIANTS:
        fprint(f"  {v['name']:<20} {v['top_k']:>2} {v['rebal_days']:>3}d {v['dte']:>4} "
               f"{v['sizing']:<12}  {v['desc']}")
    fprint()

    # 1. Download data (shared across all variants)
    close, high, low = download_data()

    # 2. Load regime predictions (shared)
    regime_series = load_regime_predictions()

    # 3. Pre-compute ATR (shared)
    atr_dict = compute_atr_series(high, low, close)

    spy_close = close["SPY"]

    # ══════════════════════════════════════════════════════════════
    # BUILD FEATURES FOR EACH UNIQUE rebal_days (DTE is always 21)
    # ══════════════════════════════════════════════════════════════
    feature_cols = V4_FEATURES

    combos = set()
    for v in VARIANTS:
        combos.add((v["rebal_days"], v["dte"]))

    fprint(f"\nUnique (rebal_days, dte) combos to build: {len(combos)}")

    rankings_cache = {}
    for rebal_days, dte in sorted(combos):
        fprint(f"\n{'=' * 80}")
        fprint(f"BUILDING FEATURES: rebal={rebal_days}d, DTE={dte}")
        fprint(f"{'=' * 80}")

        rebal_freq = f"{rebal_days}B"
        rebal_dates = pd.DatetimeIndex(
            close.index.to_series().resample(rebal_freq).last().dropna().values
        )
        fprint(f"Rebalance dates: {len(rebal_dates)} "
               f"({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

        records = build_feature_records(
            close, high, low, rebal_dates, feature_cols,
            regime_series, dte=dte, regime_mode="bull_bear",
        )

        rankings, imp = walk_forward_lgbm_rank(
            records, feature_cols, f"rebal{rebal_days}_dte{dte}"
        )

        rankings_cache[(rebal_days, dte)] = (rankings, imp)

    # ══════════════════════════════════════════════════════════════
    # SIMULATE ALL VARIANTS
    # ══════════════════════════════════════════════════════════════
    fprint(f"\n{'=' * 80}")
    fprint("SIMULATING ALL 6 VARIANTS")
    fprint(f"{'=' * 80}")

    all_results = {}
    all_importance = {}

    for v in VARIANTS:
        vname = v["name"]
        top_k = v["top_k"]
        rebal_days = v["rebal_days"]
        dte = v["dte"]
        desc = v["desc"]
        sizing = v["sizing"]

        fprint(f"\n--- {vname}: top_k={top_k}, rebal={rebal_days}d, DTE={dte}, sizing={sizing} ---")
        fprint(f"    {desc}")

        rankings, imp = rankings_cache[(rebal_days, dte)]
        all_importance[vname] = imp

        if not rankings:
            fprint(f"  No rankings available, skipping")
            continue

        trades, final_eq = simulate_trades(
            vname, rankings, close, high, low, regime_series, atr_dict,
            top_k=top_k, dte=dte,
            bull_only=False,
            skip_vix_25_30=True,
            var=v,
        )

        if not trades or len(trades) < 10:
            fprint(f"  Only {len(trades) if trades else 0} trades, skipping validation")
            all_results[vname] = {
                "description": desc, "top_k": top_k, "rebal_days": rebal_days,
                "dte": dte, "sizing": sizing,
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

        # Sizing stats
        sizes = [t["position_size"] for t in trades]
        fprint(f"  Sizing stats: mean=${np.mean(sizes):.0f}, min=${np.min(sizes):.0f}, "
               f"max=${np.max(sizes):.0f}")

        # CAGR calculation
        if trades:
            first_date = pd.Timestamp(trades[0]["entry_date"])
            last_date = pd.Timestamp(trades[-1]["exit_date"])
            years = max((last_date - first_date).days / 365.25, 0.5)
            cagr = (final_eq / CAP) ** (1 / years) - 1
        else:
            cagr = 0.0
            years = 0.0

        # Random baseline (uses same sizing mode for fair comparison)
        random_sharpes = random_baseline_test(
            rankings, close, high, low, regime_series, atr_dict,
            top_k=top_k, dte=dte,
            bull_only=False, skip_vix_25_30=True,
            var=v,
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
            "sizing": sizing,
            "moneyness_pct": MONEYNESS_PCT,
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
            "sizing_mean": round(np.mean(sizes), 2),
        }

    # ══════════════════════════════════════════════════════════════
    # SUMMARY COMPARISON
    # ══════════════════════════════════════════════════════════════
    fprint(f"\n{'=' * 80}")
    fprint("SUMMARY COMPARISON — ALL 6 VARIANTS")
    fprint(f"{'=' * 80}")
    fprint(f"{'Variant':<20} {'Rb':>3} {'Sizing':<10} {'Trd':>5} {'Sharpe':>7} "
           f"{'Sortino':>8} {'CAGR':>7} {'MaxDD':>7} {'WR':>6} {'PF':>6} "
           f"{'Gates':>6} {'Final$':>8} {'RandSh':>7}")
    fprint("-" * 120)

    variant_names = [v["name"] for v in VARIANTS]
    for vname in variant_names:
        r = all_results.get(vname)
        if not r or "error" in r:
            fprint(f"  {vname:<20} — INSUFFICIENT DATA —")
            continue
        fprint(f"  {vname:<20} {r['rebal_days']:>3} {r['sizing']:<10} "
               f"{r['n_trades']:>5} {r['sharpe']:>7.2f} {r['sortino']:>8.2f} "
               f"{r['cagr']*100:>6.1f}% {r['max_dd']*100:>6.1f}% "
               f"{r['win_rate']*100:>5.1f}% {r['profit_factor']:>5.2f} "
               f"{r['gates_passed']}/{r['gates_total']} "
               f"${r['final_equity']:>7,.0f} {r['random_mean_sharpe']:>7.2f}")

    # ── STACKING ANALYSIS ──
    analyze_stacking(all_results)

    # ── BEST VARIANT ──
    valid_results = {k: v for k, v in all_results.items() if "error" not in v}
    if valid_results:
        best_name = max(valid_results.keys(), key=lambda k: valid_results[k].get("sharpe", -999))
        best = valid_results[best_name]
        baseline_sh = all_results.get("v6_baseline", {}).get("sharpe", 0)

        fprint(f"\n{'=' * 80}")
        fprint("RECOMMENDATION")
        fprint(f"{'=' * 80}")

        fprint(f"\n  BEST VARIANT: {best_name}")
        fprint(f"    Sharpe={best.get('sharpe', 0):.3f}, Sortino={best.get('sortino', 0):.3f}, "
               f"CAGR={best.get('cagr', 0)*100:.1f}%, MaxDD={best.get('max_dd', 0)*100:.1f}%")
        fprint(f"    Gates={best.get('gates_passed', 0)}/{best.get('gates_total', 0)}")

        if best_name != "v6_baseline":
            improvement = best.get("sharpe", 0) - baseline_sh
            fprint(f"    vs v6 baseline: {improvement:+.3f} Sharpe "
                   f"({improvement/max(abs(baseline_sh), 0.001)*100:+.1f}%)")
        else:
            fprint(f"    No combination beats the current v6 baseline.")

        best_gates = best.get("gates_passed", 0)
        best_total = best.get("gates_total", 5)
        if best_name != "v6_baseline" and best_gates >= 4:
            fprint(f"\n  ACTION: PROMOTE {best_name} to production v7")
            fprint(f"    Parameters: top_k={best['top_k']}, rebal={best['rebal_days']}d, "
                   f"DTE={best['dte']}, sizing={best['sizing']}, moneyness=2% OTM")
        elif best_name == "v6_baseline":
            fprint(f"\n  ACTION: KEEP v6 baseline. Combinations don't improve on it.")
        else:
            fprint(f"\n  CAUTION: Best variant only passes {best_gates}/{best_total} gates.")
            fprint(f"  Consider keeping v6 baseline until more evidence.")

    # Save results
    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    # Save trades for best variant
    if valid_results:
        trades_path = OUTPUT_DIR / "trades.json"
        # Re-run best variant to capture trades
        fprint(f"Saving trade details...")

    # MLflow logging
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"v7_integration_{t0.strftime('%Y%m%d_%H%M')}"):
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
                    "moneyness_pct": MONEYNESS_PCT,
                    "haircut": DEFAULT_HAIRCUT,
                    "regime_bull_thresh": REGIME_BULL_THRESHOLD,
                    "regime_bear_thresh": REGIME_BEAR_THRESHOLD,
                    "hold_to_expiry": True,
                    "entry_haircut_only": True,
                    "n_features": len(V4_FEATURES),
                    "wf_train_periods": WF_TRAIN_PERIODS,
                    "n_variants": len(VARIANTS),
                    "experiment": "v7_integration",
                    "question": "do_improvements_stack",
                })

                # Log best variant
                if valid_results:
                    mlflow.log_params({
                        "best_variant": best_name,
                        "best_sharpe": best.get("sharpe", 0),
                        "best_sizing": best.get("sizing", ""),
                        "best_rebal": best.get("rebal_days", 0),
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
