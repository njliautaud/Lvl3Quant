#!/usr/bin/env python3
"""
Position Sizing Cross-Validation v1 — Dynamic Sizing on Production v4 Infrastructure
=====================================================================================

Tests 6 position sizing variants on top of the CANONICAL production v4 LGBM
walk-forward + GRU regime filter. The ONLY thing that changes is position SIZE
per trade — all signal generation, pricing, costs, and trade mechanics are
identical to production_v4_honest_test.py.

Variants:
  A: Fixed $200 baseline — Reproduce production v4 exactly (~1.67-1.87 Sharpe)
  B: Kelly fraction — rolling 90-day Kelly criterion, capped at $200
  C: Confidence-based — size by LGBM rank (top=$200, 2nd=$150, 3rd=$100)
  D: Volatility-scaled — $200 × (target_vol / realized_vol), cap $300
  E: Drawdown-adjusted — reduce size during drawdowns, recover at HWM×0.95
  F: Combined best — best of B/C/D/E or Kelly+drawdown combo

Honest pricing rules (same as production v4):
  - HOLD TO EXPIRY ONLY (no early exit)
  - At expiry: INTRINSIC VALUE ONLY (no time value)
  - 15% haircut on ENTRY only (automatic exercise at expiry = no exit haircut)
  - DTE=21, $645 starting capital
  - Walk-forward LGBM, biweekly rebalance
  - Bull call spreads: 3% width, ATM
  - Bear put spreads: 3% width, ATM (bull+bear combined mode)
  - Commission: $2.60/spread round-trip

Full 5-gate adversarial validation + random baseline comparison.
"""

import json
import os
import sys
import time
import warnings
from collections import deque
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


# ── Auto-detect host ──
_hostname = os.uname().nodename.lower()
if 'neptune' in _hostname or 'nick' in str(os.path.expanduser('~')):
    BASE = Path('/home/nick/Lvl3Quant')
else:
    BASE = Path('/home/jupiter/Lvl3Quant')

# ── Standardized tools ──
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
OUTPUT_DIR = BASE / "output" / "growth_research" / "position_sizing_xval_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
DTE = 21
SPREAD_PCT = 3.0
TOP_K = 3
REGIME_BULL_THRESHOLD = 0.4
REGIME_BEAR_THRESHOLD = 0.2

# Position sizing constants
FIXED_SIZE = 200.0          # Baseline per-trade size
KELLY_WINDOW = 90           # Rolling window for Kelly calculation (trades)
KELLY_MAX = 200.0           # Kelly cap
VOL_TARGET_LOOKBACK = 252   # Days for median vol calculation
VOL_MAX_SIZE = 300.0        # Volatility-scaled cap
DD_FULL_SIZE = 200.0        # Full size when above HWM×0.9
DD_MEDIUM_SIZE = 100.0      # Medium size at 10-20% drawdown
DD_SMALL_SIZE = 50.0        # Minimum size at >20% drawdown
DD_RECOVER_THRESHOLD = 0.95 # Recover to full size at HWM×0.95

# Regime predictions path
REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"

# LGBM walk-forward
WF_TRAIN_PERIODS = 12
WF_REBAL_FREQ = "2W-FRI"

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "position_sizing_xval_v1"

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
# DATA DOWNLOAD (identical to production v4)
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
# REGIME LOADING (identical to production v4)
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
# FEATURE ENGINEERING (identical to production v4)
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
# WALK-FORWARD LGBM RANKING (identical to production v4)
# ══════════════════════════════════════════════════════════════

def build_feature_records(close, high, low, rebal_dates, feature_cols, regime_series,
                          regime_mode="bull_bear"):
    """
    Build feature + target records for all sectors on all rebal dates.
    Uses bull_bear mode (production v4c best variant).
    """
    import lightgbm as lgb

    fprint(f"  Building records: {len(rebal_dates)} dates, {len(feature_cols)} features, mode={regime_mode}")

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
# ATR COMPUTATION (identical to production v4)
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
# POSITION SIZING ENGINES
# ══════════════════════════════════════════════════════════════

class PositionSizer:
    """Base class for position sizing strategies."""

    def __init__(self, name, initial_capital=CAP):
        self.name = name
        self.equity = initial_capital
        self.hwm = initial_capital  # high water mark
        self.trade_history = deque(maxlen=500)  # rolling history for Kelly

    def update(self, pnl, trade_cost):
        """Update state after a trade completes."""
        self.equity += pnl
        self.hwm = max(self.hwm, self.equity)
        self.trade_history.append({"pnl": pnl, "cost": trade_cost, "win": pnl > 0})

    def get_size(self, equity, rank_position, sector_vol_21d=None, spy_vol_21d=None,
                 median_vol=None):
        """Return position size for this trade. Override in subclasses."""
        raise NotImplementedError

    def reset(self, initial_capital=CAP):
        """Reset state for a new simulation run."""
        self.equity = initial_capital
        self.hwm = initial_capital
        self.trade_history.clear()


class FixedSizer(PositionSizer):
    """Variant A: Fixed $200 per trade (production v4 baseline)."""

    def get_size(self, equity, rank_position=0, **kwargs):
        return min(FIXED_SIZE, equity / 3)


class KellySizer(PositionSizer):
    """
    Variant B: Kelly criterion sizing.
    Kelly = (WR * avg_win - (1-WR) * avg_loss) / avg_win
    Size = min($200, equity * kelly_fraction), rolling 90-trade window.
    """

    def get_size(self, equity, rank_position=0, **kwargs):
        if len(self.trade_history) < 20:
            # Not enough history — use conservative fixed size
            return min(FIXED_SIZE * 0.5, equity / 3)

        recent = list(self.trade_history)[-KELLY_WINDOW:]
        wins = [t for t in recent if t["win"]]
        losses = [t for t in recent if not t["win"]]

        if not wins or not losses:
            return min(FIXED_SIZE * 0.5, equity / 3)

        wr = len(wins) / len(recent)
        avg_win = np.mean([t["pnl"] for t in wins])
        avg_loss = abs(np.mean([t["pnl"] for t in losses]))

        if avg_win <= 0 or avg_loss <= 0:
            return min(FIXED_SIZE * 0.25, equity / 3)

        kelly = (wr * avg_win - (1 - wr) * avg_loss) / avg_win

        # Half-Kelly for safety (full Kelly is too aggressive)
        kelly = max(0.0, kelly) * 0.5

        # Size as fraction of equity, capped
        size = equity * kelly
        size = min(size, KELLY_MAX)
        size = min(size, equity / 3)
        size = max(size, 30.0)  # minimum viable trade

        return size


class ConfidenceSizer(PositionSizer):
    """
    Variant C: Confidence-based sizing by LGBM rank position.
    Top-ranked sector gets $200, 2nd gets $150, 3rd gets $100.
    (Production v4 gives all 3 the same $200.)
    """

    RANK_SIZES = {0: 200.0, 1: 150.0, 2: 100.0}

    def get_size(self, equity, rank_position=0, **kwargs):
        base = self.RANK_SIZES.get(rank_position, 100.0)
        return min(base, equity / 3)


class VolatilitySizer(PositionSizer):
    """
    Variant D: Volatility-scaled sizing.
    Size = $200 * (target_vol / realized_vol).
    Target = median historical 21d vol. Scale down when vol high, up when low.
    Capped at $300.
    """

    def get_size(self, equity, rank_position=0, sector_vol_21d=None,
                 median_vol=None, **kwargs):
        if sector_vol_21d is None or median_vol is None or sector_vol_21d <= 0:
            return min(FIXED_SIZE, equity / 3)

        scale = median_vol / (sector_vol_21d + 1e-10)
        # Clamp scale between 0.25x and 1.5x
        scale = np.clip(scale, 0.25, 1.5)

        size = FIXED_SIZE * scale
        size = min(size, VOL_MAX_SIZE)
        size = min(size, equity / 3)
        size = max(size, 30.0)

        return size


class DrawdownSizer(PositionSizer):
    """
    Variant E: Drawdown-adjusted sizing.
    Full $200 when equity > HWM * 0.9.
    $100 when 10-20% drawdown.
    $50 when >20% drawdown.
    Recover to full only when equity > HWM * 0.95.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._in_recovery = False

    def reset(self, initial_capital=CAP):
        super().reset(initial_capital)
        self._in_recovery = False

    def get_size(self, equity, rank_position=0, **kwargs):
        if self.hwm <= 0:
            return min(DD_FULL_SIZE, equity / 3)

        dd_pct = 1.0 - (equity / self.hwm)

        # Recovery logic: once in drawdown, stay reduced until HWM * 0.95
        if dd_pct > 0.10:
            self._in_recovery = True

        if self._in_recovery and equity >= self.hwm * DD_RECOVER_THRESHOLD:
            self._in_recovery = False

        if dd_pct > 0.20:
            size = DD_SMALL_SIZE
        elif dd_pct > 0.10 or self._in_recovery:
            size = DD_MEDIUM_SIZE
        else:
            size = DD_FULL_SIZE

        return min(size, equity / 3)


class CombinedSizer(PositionSizer):
    """
    Variant F: Combined best strategy.
    Uses Kelly for base sizing + drawdown adjustment overlay.
    This combines the two most orthogonal approaches:
    - Kelly adapts to recent edge quality
    - Drawdown adjustment protects capital during losing streaks
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kelly = KellySizer("kelly_sub", CAP)
        self._dd = DrawdownSizer("dd_sub", CAP)

    def reset(self, initial_capital=CAP):
        super().reset(initial_capital)
        self._kelly.reset(initial_capital)
        self._dd.reset(initial_capital)

    def update(self, pnl, trade_cost):
        super().update(pnl, trade_cost)
        self._kelly.update(pnl, trade_cost)
        self._dd.update(pnl, trade_cost)
        # Sync equity/hwm
        self._kelly.equity = self.equity
        self._kelly.hwm = self.hwm
        self._dd.equity = self.equity
        self._dd.hwm = self.hwm

    def get_size(self, equity, rank_position=0, **kwargs):
        kelly_size = self._kelly.get_size(equity, rank_position, **kwargs)
        dd_size = self._dd.get_size(equity, rank_position, **kwargs)

        # Use the more conservative of the two
        size = min(kelly_size, dd_size)
        size = min(size, equity / 3)
        size = max(size, 30.0)

        return size


# ══════════════════════════════════════════════════════════════
# TRADE SIMULATION (modified from production v4 for dynamic sizing)
# ══════════════════════════════════════════════════════════════

def simulate_trades_sized(name, rankings, close, high, low, regime_series, atr_dict,
                          sizer, close_df_for_vol=None,
                          bull_only=False, skip_vix_25_30=True):
    """
    Simulate trades with dynamic position sizing.

    Uses production v4c configuration:
      - bull+bear combined (bull_only=False)
      - skip VIX 25-30 band (skip_vix_25_30=True)
      - hold to expiry, intrinsic value only, 15% entry haircut

    The ONLY difference from production v4: position size comes from `sizer`.
    """
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    sizer.reset(CAP)
    equity = CAP
    trades = []

    # Pre-compute median vol for volatility sizer
    median_vols = {}
    if close_df_for_vol is not None:
        for tk in SECTORS:
            if tk in close_df_for_vol.columns:
                rets = close_df_for_vol[tk].pct_change().dropna()
                if len(rets) > VOL_TARGET_LOOKBACK:
                    rolling_vol = rets.rolling(21).std() * np.sqrt(252)
                    median_vols[tk] = float(rolling_vol.iloc[-VOL_TARGET_LOOKBACK:].median())

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

        picks = [t for t, _ in ranked[:TOP_K]]

        n_entered = 0
        for rank_pos, tk in enumerate(picks):
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

            # Compute sector vol for volatility sizer
            sector_vol_21d = None
            if tk in close.columns:
                rets = close[tk].iloc[:di + 1].pct_change().dropna()
                if len(rets) > 21:
                    sector_vol_21d = float(rets.iloc[-21:].std() * np.sqrt(252))

            # Get position size from sizer
            max_pos = sizer.get_size(
                equity=equity,
                rank_position=rank_pos,
                sector_vol_21d=sector_vol_21d,
                median_vol=median_vols.get(tk),
            )

            if max_pos < 30:
                continue

            # Price the spread (identical to production v4)
            K1 = round(S, 2)
            K2 = round(S * (1 + SPREAD_PCT / 100), 2)
            if K2 <= K1:
                K2 = K1 + 1.0

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

            # PnL: exit value - entry cost - commission (no exit haircut at expiry)
            pnl = (exit_value_ps - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD
            equity += pnl
            n_entered += 1

            # Update sizer state
            sizer.update(pnl, total_cost)

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
                "position_size": round(max_pos, 2),
                "rank_position": rank_pos,
            })

    return trades, equity


# ══════════════════════════════════════════════════════════════
# RANDOM BASELINE (identical to production v4)
# ══════════════════════════════════════════════════════════════

def random_baseline_test(rankings, close, high, low, regime_series, atr_dict,
                         n_trials=5):
    """Test if random sector selection also produces similar returns."""
    fprint(f"\n  Random baseline test ({n_trials} trials)...")
    random_sharpes = []

    for trial in range(n_trials):
        np.random.seed(42 + trial)
        rand_rankings = {}
        for dt, data in rankings.items():
            rand_scores = {tk: np.random.random() for tk in data["scores"].keys()}
            rand_rankings[dt] = {"scores": rand_scores, "direction": data["direction"]}

        sizer = FixedSizer("random_fixed")
        trades, final_eq = simulate_trades_sized(
            f"Random_{trial}", rand_rankings, close, high, low,
            regime_series, atr_dict, sizer=sizer,
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
    fprint(f"POSITION SIZING CROSS-VALIDATION v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 80)
    fprint(f"Capital: ${CAP:.0f} | DTE: {DTE} | Spread: {SPREAD_PCT:.0f}% | "
           f"Haircut: {DEFAULT_HAIRCUT:.0%} entry only | Comm: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"HOLD TO EXPIRY | Intrinsic value only at expiry | No exit haircut")
    fprint(f"Regime bull threshold: >{REGIME_BULL_THRESHOLD}, bear: <{REGIME_BEAR_THRESHOLD}")
    fprint(f"Base: bull+bear combined + skip VIX 25-30 (production v4c config)")
    fprint()

    # 1. Download data
    close, high, low = download_data()

    # 2. Load regime predictions
    regime_series = load_regime_predictions()

    # 3. Pre-compute ATR
    atr_dict = compute_atr_series(high, low, close)

    # 4. Build rebalance dates
    rebal_dates = pd.DatetimeIndex(
        close.index.to_series().resample(WF_REBAL_FREQ).last().dropna().values
    )
    fprint(f"Rebalance dates: {len(rebal_dates)} "
           f"({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

    spy_close = close["SPY"]

    # ── Build rankings ONCE (shared across all sizing variants) ──
    fprint("\n" + "=" * 80)
    fprint("BUILDING LGBM RANKINGS (shared across all variants)")
    fprint("=" * 80)

    records = build_feature_records(
        close, high, low, rebal_dates, V4_FEATURES,
        regime_series, regime_mode="bull_bear",
    )
    rankings, imp_df = walk_forward_lgbm_rank(records, V4_FEATURES, "position_sizing_base")

    if not rankings:
        fprint("FATAL: No rankings produced. Cannot continue.")
        return

    fprint(f"Rankings: {len(rankings)} dates available for simulation")

    # ── Define sizing variants ──
    sizing_variants = [
        ("A_fixed_200", FixedSizer("A_fixed_200"),
         "Fixed $200 baseline (production v4 reproduction)"),
        ("B_kelly", KellySizer("B_kelly"),
         "Kelly fraction (half-Kelly, 90-trade rolling, cap $200)"),
        ("C_confidence", ConfidenceSizer("C_confidence"),
         "Confidence-based ($200/$150/$100 by rank)"),
        ("D_vol_scaled", VolatilitySizer("D_vol_scaled"),
         "Volatility-scaled ($200 x target/realized, cap $300)"),
        ("E_drawdown_adj", DrawdownSizer("E_drawdown_adj"),
         "Drawdown-adjusted ($200/$100/$50 by DD depth)"),
        ("F_combined", CombinedSizer("F_combined"),
         "Combined Kelly + Drawdown (conservative of both)"),
    ]

    # ── Simulate all variants ──
    fprint("\n" + "=" * 80)
    fprint("SIMULATING ALL SIZING VARIANTS")
    fprint("=" * 80)

    all_results = {}
    all_trades = {}

    for vname, sizer, desc in sizing_variants:
        fprint(f"\n--- {vname}: {desc} ---")

        trades, final_eq = simulate_trades_sized(
            vname, rankings, close, high, low, regime_series, atr_dict,
            sizer=sizer, close_df_for_vol=close,
        )

        if not trades or len(trades) < 10:
            fprint(f"  Only {len(trades) if trades else 0} trades, skipping validation")
            continue

        # 5-gate adversarial validation
        result = validate_trades(
            trades, initial_capital=CAP,
            spy_prices=spy_close,
            strategy_name=vname,
        )
        result.print_summary()

        # Position size stats
        sizes = [t["position_size"] for t in trades]
        fprint(f"  Position size stats: mean=${np.mean(sizes):.0f}, "
               f"median=${np.median(sizes):.0f}, "
               f"min=${np.min(sizes):.0f}, max=${np.max(sizes):.0f}")

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

        all_results[vname] = {
            "description": desc,
            **result.to_dict(),
            "avg_position_size": round(np.mean(sizes), 2),
            "median_position_size": round(np.median(sizes), 2),
            "min_position_size": round(np.min(sizes), 2),
            "max_position_size": round(np.max(sizes), 2),
        }
        all_trades[vname] = trades

    # ── Random baseline (run once, applies to fixed sizer) ──
    fprint("\n" + "=" * 80)
    fprint("RANDOM BASELINE")
    fprint("=" * 80)

    random_sharpes = random_baseline_test(
        rankings, close, high, low, regime_series, atr_dict,
    )
    mean_random = np.mean(random_sharpes) if random_sharpes else 0

    for vname in all_results:
        all_results[vname]["random_sharpes"] = [round(s, 3) for s in random_sharpes]
        all_results[vname]["random_mean_sharpe"] = round(mean_random, 3)

    # ── Variant A sanity check ──
    fprint("\n" + "=" * 80)
    fprint("SANITY CHECK: Variant A (Fixed $200)")
    fprint("=" * 80)
    a_res = all_results.get("A_fixed_200")
    if a_res:
        a_sharpe = a_res["sharpe"]
        if 1.5 <= a_sharpe <= 2.5:
            fprint(f"  PASS: Variant A Sharpe = {a_sharpe:.2f} (expected ~1.67-1.87)")
        else:
            fprint(f"  WARNING: Variant A Sharpe = {a_sharpe:.2f} — outside expected 1.67-1.87 range")
            fprint(f"  This may indicate the base config differs slightly from production v4")
            fprint(f"  (using bull+bear+VIX filter = v4c variant, not v4a)")
    else:
        fprint("  FAIL: No results for Variant A — script is broken")

    # ── Summary comparison ──
    fprint("\n" + "=" * 80)
    fprint("SUMMARY COMPARISON — POSITION SIZING VARIANTS")
    fprint("=" * 80)
    fprint(f"{'Variant':<22} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} "
           f"{'PF':>6} {'MaxDD':>7} {'Gates':>6} {'Final$':>8} {'AvgSz':>6}")
    fprint("-" * 95)

    for vname, _, desc in sizing_variants:
        r = all_results.get(vname)
        if not r:
            fprint(f"  {vname:<22} — NO DATA —")
            continue
        fprint(f"  {vname:<22} {r['n_trades']:>5} {r['sharpe']:>7.2f} {r['sortino']:>8.2f} "
               f"{r['win_rate']*100:>5.1f}% {r['profit_factor']:>5.2f} "
               f"{r['max_dd']*100:>6.1f}% {r['gates_passed']}/{r['gates_total']} "
               f"${r['final_equity']:>7,.0f} ${r['avg_position_size']:>5.0f}")

    fprint(f"\n  Random baseline mean Sharpe: {mean_random:.2f}")

    # ── Determine best variant ──
    fprint("\n" + "=" * 80)
    fprint("ANALYSIS: BEST POSITION SIZING STRATEGY")
    fprint("=" * 80)

    # Rank by Sharpe (primary), then gates passed (secondary)
    ranked_variants = sorted(
        [(vname, all_results[vname]) for vname in all_results],
        key=lambda x: (x[1]["sharpe"], x[1]["gates_passed"]),
        reverse=True,
    )

    if ranked_variants:
        best_name, best_res = ranked_variants[0]
        a_sharpe = all_results.get("A_fixed_200", {}).get("sharpe", 0)
        fprint(f"  Best variant: {best_name}")
        fprint(f"    Sharpe: {best_res['sharpe']:.2f} (vs baseline A: {a_sharpe:.2f})")
        fprint(f"    Avg position size: ${best_res['avg_position_size']:.0f}")
        fprint(f"    Gates: {best_res['gates_passed']}/{best_res['gates_total']}")

        if best_name == "A_fixed_200":
            fprint(f"\n  CONCLUSION: Fixed $200 sizing is already optimal.")
            fprint(f"  None of the dynamic sizing strategies beat the baseline.")
        else:
            delta = best_res["sharpe"] - a_sharpe
            if delta > 0.1:
                fprint(f"\n  CONCLUSION: {best_name} beats baseline by {delta:.2f} Sharpe.")
                fprint(f"  Recommend adopting this sizing strategy in production.")
            else:
                fprint(f"\n  CONCLUSION: Marginal improvement ({delta:+.2f} Sharpe).")
                fprint(f"  Fixed sizing is simpler with similar performance.")

        # Show all rankings
        fprint(f"\n  Full ranking (by Sharpe):")
        for i, (vn, vr) in enumerate(ranked_variants):
            marker = " <-- BEST" if i == 0 else ""
            fprint(f"    {i+1}. {vn}: Sharpe={vr['sharpe']:.2f}, "
                   f"WR={vr['win_rate']*100:.1f}%, "
                   f"Gates={vr['gates_passed']}/{vr['gates_total']}{marker}")

    # ── Update Variant F description based on actual best ──
    # (Variant F is Kelly+DD combined; check if using the actual best single
    # strategy would have been better)
    if "F_combined" in all_results and len(ranked_variants) > 1:
        f_sharpe = all_results["F_combined"]["sharpe"]
        best_single_name = None
        best_single_sharpe = 0
        for vn in ["B_kelly", "C_confidence", "D_vol_scaled", "E_drawdown_adj"]:
            if vn in all_results and all_results[vn]["sharpe"] > best_single_sharpe:
                best_single_sharpe = all_results[vn]["sharpe"]
                best_single_name = vn
        if best_single_name:
            fprint(f"\n  Variant F (Kelly+DD) Sharpe: {f_sharpe:.2f}")
            fprint(f"  Best single dynamic strategy ({best_single_name}): {best_single_sharpe:.2f}")
            if f_sharpe > best_single_sharpe:
                fprint(f"  Combined approach wins by {f_sharpe - best_single_sharpe:.2f}")
            else:
                fprint(f"  Single strategy {best_single_name} is better by "
                       f"{best_single_sharpe - f_sharpe:.2f}")

    # ── Feature importance ──
    if imp_df is not None:
        fprint("\n" + "=" * 80)
        fprint("FEATURE IMPORTANCE (Top 10)")
        fprint("=" * 80)
        for _, row in imp_df.head(10).iterrows():
            bar = "*" * int(row["importance"] / imp_df["importance"].max() * 30)
            fprint(f"  {row['feature']:<30} {row['importance']:>6.1f} {bar}")

    # ── Save results ──
    results_path = OUTPUT_DIR / "position_sizing_xval_v1_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    # Save per-variant trade logs
    for vname, trades in all_trades.items():
        trades_path = OUTPUT_DIR / f"trades_{vname}.json"
        with open(trades_path, "w") as f:
            json.dump(trades, f, indent=2, default=str)

    # ── MLflow logging ──
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"pos_sizing_v1_{t0.strftime('%Y%m%d_%H%M')}"):
                for vname, r in all_results.items():
                    prefix = vname.split("_")[0]
                    mlflow.log_metric(f"{prefix}_sharpe", r.get("sharpe", 0))
                    mlflow.log_metric(f"{prefix}_sortino", r.get("sortino", 0))
                    mlflow.log_metric(f"{prefix}_win_rate", r.get("win_rate", 0))
                    mlflow.log_metric(f"{prefix}_profit_factor", r.get("profit_factor", 0))
                    mlflow.log_metric(f"{prefix}_max_dd", r.get("max_dd", 0))
                    mlflow.log_metric(f"{prefix}_n_trades", r.get("n_trades", 0))
                    mlflow.log_metric(f"{prefix}_gates_passed", r.get("gates_passed", 0))
                    mlflow.log_metric(f"{prefix}_final_equity", r.get("final_equity", 0))
                    mlflow.log_metric(f"{prefix}_avg_pos_size", r.get("avg_position_size", 0))
                    mlflow.log_metric(f"{prefix}_random_mean_sharpe",
                                     r.get("random_mean_sharpe", 0))

                mlflow.log_params({
                    "capital": CAP,
                    "dte": DTE,
                    "spread_pct": SPREAD_PCT,
                    "haircut": DEFAULT_HAIRCUT,
                    "commission_rt": COMMISSION_RT_SPREAD,
                    "regime_bull_thresh": REGIME_BULL_THRESHOLD,
                    "regime_bear_thresh": REGIME_BEAR_THRESHOLD,
                    "hold_to_expiry": True,
                    "entry_haircut_only": True,
                    "n_features": len(V4_FEATURES),
                    "wf_train_periods": WF_TRAIN_PERIODS,
                    "rebal_freq": WF_REBAL_FREQ,
                    "regime_mode": "bull_bear",
                    "skip_vix_25_30": True,
                    "kelly_window": KELLY_WINDOW,
                    "vol_max_size": VOL_MAX_SIZE,
                    "fixed_size": FIXED_SIZE,
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
