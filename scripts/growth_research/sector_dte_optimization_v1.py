#!/usr/bin/env python3
"""
Sector-Specific DTE Optimization v1 — Does DTE Vary by Sector Volatility?
==========================================================================

V6 uses a uniform DTE=21 for all sector ETFs. But different sectors have
different volatility profiles. XLE (energy, high vol) might benefit from
shorter DTE while XLU (utilities, low vol) might benefit from longer DTE.

Research question: Does sector-specific DTE optimization improve V6?

6 Variants (all with V6 structure: weekly, 2% OTM, 17 features, pairs):
  A: DTE=21 baseline (uniform for all sectors)
  B: DTE=14 uniform (shorter duration)
  C: DTE=28 uniform (longer duration)
  D: DTE=35 uniform (monthly+)
  E: Vol-adaptive DTE: high-vol sectors DTE=14, low-vol DTE=28, medium DTE=21
  F: Sector-specific DTE: optimal DTE per sector from in-sample (60/40 split)

For variant F, the first 60% of data is used to find optimal DTE per sector,
then tested on the remaining 40%.

V6 config for ALL variants:
  - Weekly rebalance (W-FRI)
  - 2% OTM moneyness
  - Bull VIX>20 + pairs VIX<20
  - $200/trade ($100/leg for pairs)
  - 3% spread width, $2.60 commission, 15% haircut entry only
  - Hold to expiry, intrinsic value only
  - 21 features (18 legacy + 3 cross-asset)

Each variant: 5-gate adversarial validation + 5-trial random baseline.
MLflow experiment: 'sector_dte_optimization_v1'
Output: output/growth_research/sector_dte_optimization_v1/
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


# ── Standardized tools (with Neptune fallback) ──
import os as _os
_hostname = _os.uname().nodename.lower()
if 'neptune' in _hostname or 'nick' in str(_os.path.expanduser('~')):
    _BASE_PATH = "/home/nick/Lvl3Quant"
else:
    _BASE_PATH = "/home/jupiter/Lvl3Quant"
sys.path.insert(0, _BASE_PATH)

_TOOLS_AVAILABLE = False
try:
    from research.tools.options_pricer import (
        price_bull_call_spread,
        price_bear_put_spread,
        estimate_iv,
        compute_atr,
        COMMISSION_RT_SPREAD,
        DEFAULT_HAIRCUT,
    )
    from research.tools.adversarial_validator import validate_trades
    _TOOLS_AVAILABLE = True
    fprint("Imported from research.tools")
except ImportError:
    fprint("research.tools not found — using inline implementations")
    COMMISSION_RT_SPREAD = 2.60
    DEFAULT_HAIRCUT = 0.15

    def price_bull_call_spread(S, K1, K2, dte, atr, vix=20.0, haircut=0.15, **kw):
        from scipy.stats import norm
        T = dte / 365.0
        if T <= 0:
            return 0.0, 0.0
        iv = max(vix / 100.0 * 1.2, atr / S * np.sqrt(252) * 1.2, 0.10)
        d1_l = (np.log(S / K1) + (0.5 * iv**2) * T) / (iv * np.sqrt(T))
        d1_s = (np.log(S / K2) + (0.5 * iv**2) * T) / (iv * np.sqrt(T))
        call_l = S * norm.cdf(d1_l) - K1 * np.exp(-0.04 * T) * norm.cdf(d1_l - iv * np.sqrt(T))
        call_s = S * norm.cdf(d1_s) - K2 * np.exp(-0.04 * T) * norm.cdf(d1_s - iv * np.sqrt(T))
        spread_val = max(call_l - call_s, 0.001)
        entry_cost = spread_val * (1 + haircut)
        max_profit = (K2 - K1) - entry_cost
        return entry_cost, max(max_profit, 0.001)

    def price_bear_put_spread(S, K1, K2, dte, atr, vix=20.0, haircut=0.15, **kw):
        from scipy.stats import norm
        T = dte / 365.0
        if T <= 0:
            return 0.0, 0.0
        iv = max(vix / 100.0 * 1.2, atr / S * np.sqrt(252) * 1.2, 0.10)
        d1_l = (np.log(S / K1) + (0.5 * iv**2) * T) / (iv * np.sqrt(T))
        d1_s = (np.log(S / K2) + (0.5 * iv**2) * T) / (iv * np.sqrt(T))
        put_l = K1 * np.exp(-0.04 * T) * norm.cdf(-(d1_l - iv * np.sqrt(T))) - S * norm.cdf(-d1_l)
        put_s = K2 * np.exp(-0.04 * T) * norm.cdf(-(d1_s - iv * np.sqrt(T))) - S * norm.cdf(-d1_s)
        spread_val = max(put_s - put_l, 0.001)
        entry_cost = spread_val * (1 + haircut)
        max_profit = (K2 - K1) - entry_cost
        return entry_cost, max(max_profit, 0.001)

    class _ValidationResult:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)
        def print_summary(self):
            fprint(f"\n{'='*65}")
            fprint(f"  ADVERSARIAL VALIDATION: {self.strategy_name}")
            fprint(f"{'='*65}")
            fprint(f"  Trades: {self.n_trades}  |  Sharpe: {self.sharpe:.2f}  |  "
                   f"Sortino: {self.sortino:.2f}  |  WR: {self.win_rate:.1%}")
            fprint(f"  CAGR: {self.cagr:.1%}  |  MaxDD: {self.max_dd:.1%}  |  "
                   f"PF: {self.profit_factor:.2f}  |  Final: ${self.final_equity:,.0f}")
            fprint(f"  Gates: {self.gates_passed}/{self.gates_total}")
            fprint(f"{'='*65}")
        def to_dict(self):
            return {k: v for k, v in self.__dict__.items()}

    def validate_trades(trades, initial_capital=645.0, spy_prices=None,
                        strategy_name="Strategy", n_perms=2000, **kw):
        if not trades or len(trades) < 5:
            return _ValidationResult({
                "strategy_name": strategy_name, "n_trades": len(trades) if trades else 0,
                "sharpe": 0, "sortino": 0, "win_rate": 0, "profit_factor": 0,
                "max_dd": 0, "cagr": 0, "final_equity": initial_capital,
                "gates_passed": 0, "gates_total": 5,
            })
        pnls = [t["pnl"] for t in trades]
        equity = [initial_capital]
        for p in pnls:
            equity.append(equity[-1] + p)
        equity = np.array(equity[1:])
        rets = np.diff(np.concatenate([[initial_capital], equity])) / np.concatenate([[initial_capital], equity[:-1]])
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        sharpe = float(np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(52))
        dr = rets[rets < 0]
        sortino = float(np.mean(rets) / (np.std(dr) + 1e-10) * np.sqrt(52)) if len(dr) > 0 else 0
        wr = len(wins) / len(pnls)
        pf = sum(wins) / (abs(sum(losses)) + 1e-10) if losses else 999
        pk = np.maximum.accumulate(equity)
        dd = (equity / pk) - 1
        mdd = float(dd.min())
        # Parse dates for CAGR
        try:
            d0 = pd.Timestamp(trades[0]["entry_date"])
            d1 = pd.Timestamp(trades[-1]["exit_date"])
            n_years = max((d1 - d0).days / 365.25, 0.5)
        except Exception:
            n_years = max(len(trades) / 52, 0.5)
        cagr = float((equity[-1] / initial_capital) ** (1 / n_years) - 1) if equity[-1] > 0 else 0
        # Simple gate checks
        gates_passed = 0
        if sharpe > 0.5:
            gates_passed += 1
        if wr > 0.45:
            gates_passed += 1
        if pf > 1.0:
            gates_passed += 1
        if mdd > -0.40:
            gates_passed += 1
        if len(pnls) >= 30:
            gates_passed += 1
        return _ValidationResult({
            "strategy_name": strategy_name, "n_trades": len(pnls),
            "sharpe": round(sharpe, 3), "sortino": round(sortino, 3),
            "win_rate": round(wr, 4), "profit_factor": round(pf, 3),
            "max_dd": round(mdd, 4), "cagr": round(cagr, 4),
            "final_equity": round(float(equity[-1]), 2),
            "gates_passed": gates_passed, "gates_total": 5,
        })


# ── Config ──
BASE = Path(_BASE_PATH)
OUTPUT_DIR = BASE / "output" / "growth_research" / "sector_dte_optimization_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
SPREAD_PCT = 3.0
TOP_K = 3
REGIME_BULL_THRESHOLD = 0.4
REGIME_BEAR_THRESHOLD = 0.2

# Regime predictions path
REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"

# LGBM walk-forward
WF_TRAIN_PERIODS = 12

# V6 config
V6_REBAL_FREQ = "W-FRI"
V6_OTM_PCT = 0.02
V6_PAIRS = True
V6_MAX_POS_BULL = 200
V6_MAX_POS_PAIR_LEG = 100

# DTE variants to test
DTE_CANDIDATES = [14, 21, 28, 35]

# Vol classification for sectors (adaptive variant E)
# These are computed from data at runtime, but we need categories
HIGH_VOL_SECTORS = ["XLE", "XLB", "XLF"]   # Energy, Materials, Financials
LOW_VOL_SECTORS = ["XLU", "XLP", "XLRE"]   # Utilities, Staples, Real Estate
MED_VOL_SECTORS = ["XLK", "XLV", "XLY", "XLI", "XLC"]  # Tech, Health, Disc, Industrial, Comm

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "sector_dte_optimization_v1"

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
# FEATURE DEFINITIONS
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

ALL_FEATURES = LEGACY_FEATURES + VALIDATED_CROSS_ASSET  # 21 total


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
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════

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

    # 3. Cross-sector dispersion
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

def build_feature_records(close, high, low, rebal_dates, feature_cols, regime_series, dte):
    """
    Build feature + target records for all sectors on all rebal dates.
    Uses bull_only regime mode (regime>0.4).
    dte parameter controls the forward return horizon.
    """
    fprint(f"  Building records: {len(rebal_dates)} dates, {len(feature_cols)} features, DTE={dte}")

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

            # Cross-asset features
            cross_asset = {}
            needs_cross = any(col in VALIDATED_CROSS_ASSET for col in feature_cols)
            if needs_cross:
                cross_asset = compute_cross_asset_features(tk, idx, close)

            # Forward return target (DTE days forward)
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
# STRIKE COMPUTATION (2% OTM per V6 config)
# ══════════════════════════════════════════════════════════════

def compute_strikes(S, direction, otm_pct, spread_pct):
    """Compute strike prices for a spread."""
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


# ══════════════════════════════════════════════════════════════
# SECTOR VOL CLASSIFICATION (data-driven for variant E)
# ══════════════════════════════════════════════════════════════

def classify_sector_volatility(close):
    """
    Classify sectors into high/medium/low volatility based on historical data.
    Uses full-sample annualized volatility. Returns dict: ticker -> 'high'|'med'|'low'.
    """
    vols = {}
    for tk in SECTORS:
        if tk in close.columns:
            rets = close[tk].pct_change().dropna()
            if len(rets) > 252:
                vols[tk] = float(rets.std() * np.sqrt(252))
    if not vols:
        return {tk: "med" for tk in SECTORS}

    sorted_vols = sorted(vols.items(), key=lambda x: x[1])
    n = len(sorted_vols)
    tercile1 = n // 3
    tercile2 = 2 * n // 3

    classification = {}
    for i, (tk, vol) in enumerate(sorted_vols):
        if i < tercile1:
            classification[tk] = "low"
        elif i < tercile2:
            classification[tk] = "med"
        else:
            classification[tk] = "high"

    fprint(f"  Sector vol classification (data-driven):")
    for tk, vol in sorted_vols:
        fprint(f"    {tk}: {vol:.1%} annualized -> {classification[tk]}")

    return classification


def get_dte_for_sector(tk, dte_map, default_dte=21):
    """Get the DTE to use for a given sector based on the DTE map."""
    if isinstance(dte_map, int):
        return dte_map
    return dte_map.get(tk, default_dte)


# ══════════════════════════════════════════════════════════════
# TRADE SIMULATION (V6 config with variable DTE)
# ══════════════════════════════════════════════════════════════

def _execute_single_trade(tk, dt, direction, otm_pct, max_pos, close, atr_dict, cv, equity, dte):
    """
    Execute a single spread trade with specified DTE.
    Returns (pnl, exit_date_str) or (None, None) if trade could not be entered.
    """
    if tk not in close.columns or tk not in atr_dict:
        return None, None

    S = float(close[tk].loc[dt])
    di = close.index.get_loc(dt)
    ei = min(di + dte, len(close) - 1)
    if ei <= di:
        return None, None

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
                S=S, K1=K1, K2=K2, dte=dte, atr=av, vix=cv
            )
        else:
            entry_cost_ps, max_profit_ps = price_bear_put_spread(
                S=S, K1=K1, K2=K2, dte=dte, atr=av, vix=cv
            )
    except Exception:
        return None, None

    total_cost = entry_cost_ps * 100 + COMMISSION_RT_SPREAD

    if total_cost <= 0 or total_cost > max_pos or total_cost > equity * 0.40:
        return None, None

    # HOLD TO EXPIRY: compute intrinsic value at expiry
    Se = float(close[tk].iloc[ei])

    if direction == "bull":
        intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
    else:
        intrinsic = max(K2 - Se, 0.0) - max(K1 - Se, 0.0)

    exit_value_ps = intrinsic

    # PnL: exit value - entry cost - commission (no exit haircut at expiry)
    pnl = (exit_value_ps - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD
    exit_date = str(close.index[ei].date())
    return pnl, exit_date


def simulate_trades(name, rankings, close, high, low, atr_dict, dte_map):
    """
    Simulate trades using V6 config with variable DTE per sector.

    dte_map: either an int (uniform DTE) or dict {sector: dte} (sector-specific DTE)
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

        # V6 pair logic: VIX < 20 -> bull + bear; VIX >= 20 -> bull only
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
            dte = get_dte_for_sector(tk, dte_map)
            pnl, exit_date = _execute_single_trade(
                tk, dt, "bull", V6_OTM_PCT, max_pos, close, atr_dict, cv, equity, dte
            )
            if pnl is not None:
                equity += pnl
                di = close.index.get_loc(dt)
                ei = min(di + dte, len(close) - 1)
                sv = float(spy.loc[dt])
                se = float(spy.iloc[ei]) if ei < len(spy) else sv
                spy_regime = "bull" if se >= sv else "bear"
                trades.append({
                    "pnl": round(pnl, 2),
                    "entry_date": str(dt.date()),
                    "exit_date": exit_date,
                    "ticker": tk,
                    "regime": spy_regime,
                    "direction": "bull",
                    "vix": round(cv, 1),
                    "win": pnl > 0,
                    "trade_mode": trade_mode,
                    "dte_used": dte,
                })

        # Execute bear leg (pairs mode only)
        for tk in bear_picks:
            dte = get_dte_for_sector(tk, dte_map)
            pnl, exit_date = _execute_single_trade(
                tk, dt, "bear", V6_OTM_PCT, max_pos, close, atr_dict, cv, equity, dte
            )
            if pnl is not None:
                equity += pnl
                di = close.index.get_loc(dt)
                ei = min(di + dte, len(close) - 1)
                sv = float(spy.loc[dt])
                se = float(spy.iloc[ei]) if ei < len(spy) else sv
                spy_regime = "bull" if se >= sv else "bear"
                trades.append({
                    "pnl": round(pnl, 2),
                    "entry_date": str(dt.date()),
                    "exit_date": exit_date,
                    "ticker": tk,
                    "regime": spy_regime,
                    "direction": "bear",
                    "vix": round(cv, 1),
                    "win": pnl > 0,
                    "trade_mode": trade_mode,
                    "dte_used": dte,
                })

    return trades, equity


# ══════════════════════════════════════════════════════════════
# IN-SAMPLE DTE OPTIMIZATION (Variant F)
# ══════════════════════════════════════════════════════════════

def find_optimal_dte_per_sector(close, high, low, atr_dict, regime_series,
                                 rebal_dates_insample):
    """
    For each sector, test all DTE candidates on in-sample data and pick best Sharpe.
    Returns dict: {sector: optimal_dte}.
    """
    fprint(f"\n  Finding optimal DTE per sector using {len(rebal_dates_insample)} in-sample dates...")

    optimal_dte = {}
    sector_results = {}

    for tk in SECTORS:
        if tk not in close.columns:
            optimal_dte[tk] = 21
            continue

        best_sharpe = -999
        best_dte = 21
        dte_sharpes = {}

        for dte_candidate in DTE_CANDIDATES:
            # Build simple rankings (use raw momentum as a simple proxy for in-sample)
            # We don't need LGBM here — just measure how spreads perform at each DTE
            trades = []
            equity = CAP

            for dt in rebal_dates_insample:
                if dt not in close.index:
                    continue
                idx = close.index.get_loc(dt)
                if idx < 260:
                    continue

                rscore = get_regime_score_at(regime_series, dt)
                if rscore <= REGIME_BULL_THRESHOLD:
                    continue

                vix = close["VIX"] if "VIX" in close.columns else None
                cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0

                max_pos = min(V6_MAX_POS_BULL, equity / 3)
                if max_pos < 30:
                    continue

                pnl, exit_date = _execute_single_trade(
                    tk, dt, "bull", V6_OTM_PCT, max_pos, close, atr_dict, cv, equity, dte_candidate
                )
                if pnl is not None:
                    equity += pnl
                    trades.append({"pnl": pnl, "entry_date": str(dt.date()),
                                   "exit_date": exit_date, "ticker": tk, "win": pnl > 0})

            # Compute Sharpe for this sector/DTE combo
            if len(trades) >= 10:
                pnls = [t["pnl"] for t in trades]
                eq = [CAP]
                for p in pnls:
                    eq.append(eq[-1] + p)
                eq = np.array(eq[1:])
                rets = np.diff(np.concatenate([[CAP], eq])) / np.concatenate([[CAP], eq[:-1]])
                sharpe = float(np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(52))
                wr = sum(1 for p in pnls if p > 0) / len(pnls)
            else:
                sharpe = 0
                wr = 0

            dte_sharpes[dte_candidate] = sharpe

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_dte = dte_candidate

        optimal_dte[tk] = best_dte
        sector_results[tk] = dte_sharpes
        fprint(f"    {tk}: best DTE={best_dte} (Sharpe={best_sharpe:.2f}) | "
               f"DTE sharpes: {', '.join(f'{d}={s:.2f}' for d, s in sorted(dte_sharpes.items()))}")

    return optimal_dte, sector_results


# ══════════════════════════════════════════════════════════════
# RANDOM BASELINE
# ══════════════════════════════════════════════════════════════

def random_baseline_test(rankings, close, high, low, atr_dict, dte_map, n_trials=5):
    """Test if random sector selection also produces similar returns."""
    fprint(f"\n  Random baseline ({n_trials} trials)...")
    random_sharpes = []

    for trial in range(n_trials):
        np.random.seed(42 + trial)
        rand_rankings = {}
        for dt, scores in rankings.items():
            rand_scores = {tk: np.random.random() for tk in scores.keys()}
            rand_rankings[dt] = rand_scores

        trades, final_eq = simulate_trades(
            f"Random_{trial}", rand_rankings, close, high, low, atr_dict, dte_map
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
# REBALANCE DATE GENERATION
# ══════════════════════════════════════════════════════════════

def generate_rebal_dates(close, freq_str):
    """Generate rebalance dates from close index based on frequency string."""
    rebal_dates = pd.DatetimeIndex(
        close.index.to_series().resample(freq_str).last().dropna().values
    )
    return rebal_dates


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t0 = datetime.now()
    fprint("=" * 100)
    fprint(f"SECTOR-SPECIFIC DTE OPTIMIZATION v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 100)
    fprint(f"Question: Does sector-specific DTE optimization improve V6?")
    fprint(f"Method: Test uniform DTE=14/21/28/35, vol-adaptive DTE, and sector-optimized DTE")
    fprint()
    fprint(f"V6 config (fixed for ALL variants except DTE):")
    fprint(f"  Capital: ${CAP:.0f} | Spread: {SPREAD_PCT:.0f}% | "
           f"Haircut: {DEFAULT_HAIRCUT:.0%} entry only | Comm: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"  Rebalance: weekly (W-FRI) | OTM: 2% | Pairs: VIX<20 bull+bear")
    fprint(f"  Hold to expiry | Intrinsic value only | No exit haircut")
    fprint(f"  Regime filter: GRU >0.4 | Features: 21 (18 legacy + 3 cross-asset)")
    fprint()
    fprint(f"6 DTE variants:")
    fprint(f"  A: DTE=21 baseline (uniform)")
    fprint(f"  B: DTE=14 uniform (shorter)")
    fprint(f"  C: DTE=28 uniform (longer)")
    fprint(f"  D: DTE=35 uniform (monthly+)")
    fprint(f"  E: Vol-adaptive (high-vol=14, med=21, low-vol=28)")
    fprint(f"  F: Sector-specific optimal (60% in-sample / 40% OOS)")
    fprint()

    # 1. Download data
    close, high, low = download_data()

    # 2. Load regime predictions
    regime_series = load_regime_predictions()

    # 3. Pre-compute ATR
    atr_dict = compute_atr_series(high, low, close)

    # 4. Generate V6 rebalance dates (weekly)
    all_rebal_dates = generate_rebal_dates(close, V6_REBAL_FREQ)
    fprint(f"Total rebalance dates: {len(all_rebal_dates)} "
           f"({all_rebal_dates[0].date()} to {all_rebal_dates[-1].date()})")

    # 5. Classify sector volatility (data-driven)
    vol_classification = classify_sector_volatility(close)

    # 6. For variant F: split into 60% in-sample / 40% OOS
    split_idx = int(len(all_rebal_dates) * 0.60)
    insample_dates = all_rebal_dates[:split_idx]
    oos_dates = all_rebal_dates[split_idx:]
    fprint(f"\nIn-sample dates: {len(insample_dates)} ({insample_dates[0].date()} to {insample_dates[-1].date()})")
    fprint(f"OOS dates: {len(oos_dates)} ({oos_dates[0].date()} to {oos_dates[-1].date()})")

    # 7. Find optimal DTE per sector (in-sample only, for variant F)
    optimal_dte_map, insample_results = find_optimal_dte_per_sector(
        close, high, low, atr_dict, regime_series, insample_dates
    )
    fprint(f"\n  Optimal DTE map: {optimal_dte_map}")

    # 8. Build vol-adaptive DTE map (variant E)
    vol_adaptive_dte = {}
    for tk in SECTORS:
        vc = vol_classification.get(tk, "med")
        if vc == "high":
            vol_adaptive_dte[tk] = 14
        elif vc == "low":
            vol_adaptive_dte[tk] = 28
        else:
            vol_adaptive_dte[tk] = 21
    fprint(f"  Vol-adaptive DTE map: {vol_adaptive_dte}")

    # ── Define 6 DTE variants ──
    # For variants A-E, we use ALL rebalance dates (full sample, like standard V6)
    # For variant F, we use ONLY OOS dates (honest forward test)
    DTE_VARIANTS = {
        "A_dte21_baseline": {
            "desc": "DTE=21 uniform (baseline)",
            "dte_map": 21,
            "rebal_dates": all_rebal_dates,
        },
        "B_dte14_uniform": {
            "desc": "DTE=14 uniform (shorter)",
            "dte_map": 14,
            "rebal_dates": all_rebal_dates,
        },
        "C_dte28_uniform": {
            "desc": "DTE=28 uniform (longer)",
            "dte_map": 28,
            "rebal_dates": all_rebal_dates,
        },
        "D_dte35_uniform": {
            "desc": "DTE=35 uniform (monthly+)",
            "dte_map": 35,
            "rebal_dates": all_rebal_dates,
        },
        "E_vol_adaptive": {
            "desc": f"Vol-adaptive: high={14}, med={21}, low={28}",
            "dte_map": vol_adaptive_dte,
            "rebal_dates": all_rebal_dates,
        },
        "F_sector_optimal": {
            "desc": f"Sector-specific optimal (OOS only)",
            "dte_map": optimal_dte_map,
            "rebal_dates": oos_dates,  # OOS only for honest test
        },
    }

    spy_close = close["SPY"]

    # ── Run each variant ──
    all_results = {}

    for vname, vcfg in DTE_VARIANTS.items():
        dte_map = vcfg["dte_map"]
        rebal_dates = vcfg["rebal_dates"]

        # Determine effective DTE for LGBM target computation
        # For uniform DTE, use that DTE. For sector-specific, use DTE=21 for LGBM
        # (the model predicts forward returns; DTE only affects trade execution)
        if isinstance(dte_map, int):
            lgbm_dte = dte_map
        else:
            lgbm_dte = 21  # Use standard horizon for LGBM ranking

        fprint(f"\n{'=' * 100}")
        fprint(f"VARIANT {vname}: {vcfg['desc']}")
        fprint(f"  LGBM target DTE: {lgbm_dte} | Rebal dates: {len(rebal_dates)}")
        if isinstance(dte_map, dict):
            fprint(f"  Sector DTE map: {dte_map}")
        fprint(f"{'=' * 100}")

        # Build feature records
        records = build_feature_records(
            close, high, low, rebal_dates, ALL_FEATURES, regime_series, lgbm_dte
        )

        # Walk-forward LGBM ranking
        rankings, imp_df = walk_forward_lgbm_rank(records, ALL_FEATURES, vname)

        if not rankings:
            fprint(f"  No rankings available, skipping")
            continue

        # Simulate trades with this variant's DTE map
        trades, final_eq = simulate_trades(
            vname, rankings, close, high, low, atr_dict, dte_map
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

        # Direction breakdown
        bull_trades = [t for t in trades if t["direction"] == "bull"]
        bear_trades = [t for t in trades if t["direction"] == "bear"]
        pair_trades = [t for t in trades if t.get("trade_mode") == "pairs"]
        bull_pnl = sum(t["pnl"] for t in bull_trades)
        bear_pnl = sum(t["pnl"] for t in bear_trades)
        bull_wr = sum(1 for t in bull_trades if t["win"]) / max(len(bull_trades), 1) * 100
        bear_wr = sum(1 for t in bear_trades if t["win"]) / max(len(bear_trades), 1) * 100
        fprint(f"  Direction breakdown:")
        fprint(f"    Bull: {len(bull_trades)} trades, WR {bull_wr:.1f}%, PnL ${bull_pnl:.0f}")
        fprint(f"    Bear: {len(bear_trades)} trades, WR {bear_wr:.1f}%, PnL ${bear_pnl:.0f}")
        fprint(f"    Pair-mode trades: {len(pair_trades)} (VIX<20 dates)")

        # DTE distribution analysis
        dte_counts = {}
        for t in trades:
            d = t.get("dte_used", 21)
            dte_counts[d] = dte_counts.get(d, 0) + 1
        fprint(f"  DTE distribution: {dte_counts}")

        # Per-sector breakdown for sector-specific variants
        if isinstance(dte_map, dict):
            fprint(f"  Per-sector results:")
            for tk in SECTORS:
                tk_trades = [t for t in trades if t["ticker"] == tk]
                if tk_trades:
                    tk_pnl = sum(t["pnl"] for t in tk_trades)
                    tk_wr = sum(1 for t in tk_trades if t["win"]) / len(tk_trades) * 100
                    tk_dte = dte_map.get(tk, 21)
                    fprint(f"    {tk} (DTE={tk_dte}): {len(tk_trades)} trades, "
                           f"WR {tk_wr:.1f}%, PnL ${tk_pnl:.0f}")

        # Random baseline
        random_sharpes = random_baseline_test(
            rankings, close, high, low, atr_dict, dte_map,
        )
        mean_random = np.mean(random_sharpes) if random_sharpes else 0
        fprint(f"  ML Sharpe: {result.sharpe:.2f} vs Random mean: {mean_random:.2f}")
        if result.sharpe > 0 and mean_random > 0:
            fprint(f"  ML alpha ratio: {result.sharpe / mean_random:.2f}x")

        all_results[vname] = {
            "description": vcfg["desc"],
            "dte_map": dte_map if isinstance(dte_map, int) else str(dte_map),
            **result.to_dict(),
            "random_sharpes": [round(s, 3) for s in random_sharpes],
            "random_mean_sharpe": round(mean_random, 3),
            "bull_trades": len(bull_trades),
            "bear_trades": len(bear_trades),
            "bull_pnl": round(bull_pnl, 2),
            "bear_pnl": round(bear_pnl, 2),
            "bull_wr": round(bull_wr, 1),
            "bear_wr": round(bear_wr, 1),
            "dte_distribution": dte_counts,
            "n_rebal_dates": len(rebal_dates),
        }

    # ── SUMMARY COMPARISON ──
    fprint(f"\n{'=' * 120}")
    fprint("SECTOR DTE OPTIMIZATION SUMMARY")
    fprint(f"{'=' * 120}")
    fprint(f"{'Variant':<25} {'DTE':>8} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} "
           f"{'PF':>6} {'MaxDD':>7} {'Gates':>6} {'Final$':>8} {'RandSh':>7}")
    fprint("-" * 120)

    baseline_sharpe = all_results.get("A_dte21_baseline", {}).get("sharpe", 0)

    for vname in DTE_VARIANTS.keys():
        r = all_results.get(vname)
        if not r:
            fprint(f"  {vname:<25} — NO DATA —")
            continue
        dte_str = str(r.get("dte_map", "?"))[:8]
        fprint(f"  {vname:<25} {dte_str:>8} {r['n_trades']:>5} {r['sharpe']:>7.2f} "
               f"{r['sortino']:>8.2f} {r['win_rate']*100:>5.1f}% {r['profit_factor']:>5.2f} "
               f"{r['max_dd']*100:>6.1f}% {r['gates_passed']}/{r['gates_total']} "
               f"${r['final_equity']:>7,.0f} {r['random_mean_sharpe']:>7.2f}")

    # ── SHARPE DELTA ANALYSIS ──
    fprint(f"\n{'=' * 100}")
    fprint("SHARPE DELTA vs BASELINE (A_dte21)")
    fprint(f"{'=' * 100}")
    fprint(f"  Baseline (A_dte21_baseline) Sharpe: {baseline_sharpe:.2f}")
    fprint()

    deltas = []
    for vname in DTE_VARIANTS.keys():
        if vname == "A_dte21_baseline":
            continue
        r = all_results.get(vname)
        if not r:
            continue
        delta = r["sharpe"] - baseline_sharpe
        pct_delta = delta / abs(baseline_sharpe) * 100 if baseline_sharpe != 0 else 0
        deltas.append((vname, delta, pct_delta, r["sharpe"]))

    deltas.sort(key=lambda x: x[1], reverse=True)

    for vname, delta, pct_delta, sharpe in deltas:
        direction = "+" if delta >= 0 else ""
        sign = "BETTER" if delta > 0.1 else "WORSE" if delta < -0.1 else "SIMILAR"
        bar = "*" * int(abs(delta) / max(abs(d[1]) for d in deltas) * 30) if deltas and max(abs(d[1]) for d in deltas) > 0 else ""
        fprint(f"  {vname:<25} Sharpe {sharpe:>5.2f}  "
               f"delta {direction}{delta:>+5.2f} ({direction}{pct_delta:>+5.1f}%)  "
               f"{sign}  {bar}")

    # ── KEY FINDINGS ──
    fprint(f"\n{'=' * 100}")
    fprint("KEY FINDINGS")
    fprint(f"{'=' * 100}")

    if deltas:
        best = max(deltas, key=lambda x: x[1])
        worst = min(deltas, key=lambda x: x[1])
        fprint(f"  Best variant:  {best[0]} (Sharpe {best[3]:.2f}, delta {best[1]:+.2f})")
        fprint(f"  Worst variant: {worst[0]} (Sharpe {worst[3]:.2f}, delta {worst[1]:+.2f})")

    # Check if shorter DTE helps
    b_r = all_results.get("B_dte14_uniform", {})
    if b_r:
        b_delta = b_r.get("sharpe", 0) - baseline_sharpe
        if b_delta > 0.1:
            fprint(f"  FINDING: Shorter DTE=14 IMPROVES over DTE=21 (delta {b_delta:+.2f})")
        elif b_delta < -0.1:
            fprint(f"  FINDING: Shorter DTE=14 HURTS vs DTE=21 (delta {b_delta:+.2f})")
        else:
            fprint(f"  FINDING: DTE=14 similar to DTE=21 (delta {b_delta:+.2f})")

    # Check if longer DTE helps
    c_r = all_results.get("C_dte28_uniform", {})
    d_r = all_results.get("D_dte35_uniform", {})
    if c_r:
        c_delta = c_r.get("sharpe", 0) - baseline_sharpe
        fprint(f"  FINDING: DTE=28 vs DTE=21: delta {c_delta:+.2f}")
    if d_r:
        d_delta = d_r.get("sharpe", 0) - baseline_sharpe
        fprint(f"  FINDING: DTE=35 vs DTE=21: delta {d_delta:+.2f}")

    # Check vol-adaptive
    e_r = all_results.get("E_vol_adaptive", {})
    if e_r:
        e_delta = e_r.get("sharpe", 0) - baseline_sharpe
        if e_delta > 0.1:
            fprint(f"  FINDING: Vol-adaptive DTE IMPROVES over uniform (delta {e_delta:+.2f})")
            fprint(f"    => Different sectors DO benefit from different DTEs")
        else:
            fprint(f"  FINDING: Vol-adaptive DTE does NOT improve over uniform (delta {e_delta:+.2f})")
            fprint(f"    => Uniform DTE=21 is fine for all sectors")

    # Check sector-specific optimal
    f_r = all_results.get("F_sector_optimal", {})
    if f_r:
        f_delta = f_r.get("sharpe", 0) - baseline_sharpe
        fprint(f"  FINDING: Sector-specific optimal DTE (OOS): delta {f_delta:+.2f}")
        if f_delta > 0.2:
            fprint(f"    => IN-SAMPLE optimization DOES generalize OOS")
            fprint(f"    => Recommend: adopt sector-specific DTE for V6")
        elif f_delta < -0.1:
            fprint(f"    => IN-SAMPLE optimization OVERFITS — does NOT generalize")
            fprint(f"    => Recommend: stick with uniform DTE=21")
        else:
            fprint(f"    => Marginal improvement, not enough to justify complexity")

    # In-sample DTE sensitivity analysis
    fprint(f"\n{'=' * 100}")
    fprint("IN-SAMPLE DTE SENSITIVITY BY SECTOR")
    fprint(f"{'=' * 100}")
    if insample_results:
        fprint(f"  {'Sector':<6} {'Vol':>6} {'DTE14':>7} {'DTE21':>7} {'DTE28':>7} {'DTE35':>7} {'Best':>6} {'Range':>7}")
        fprint(f"  {'-'*55}")
        for tk in SECTORS:
            if tk in insample_results:
                sr = insample_results[tk]
                vc = vol_classification.get(tk, "?")
                best_d = optimal_dte_map.get(tk, 21)
                vals = list(sr.values())
                rng = max(vals) - min(vals) if vals else 0
                fprint(f"  {tk:<6} {vc:>6} {sr.get(14,0):>7.2f} {sr.get(21,0):>7.2f} "
                       f"{sr.get(28,0):>7.2f} {sr.get(35,0):>7.2f} {best_d:>5} {rng:>6.2f}")

    # Save results
    results_path = OUTPUT_DIR / "sector_dte_optimization_results.json"
    save_data = {
        "all_results": all_results,
        "optimal_dte_map": optimal_dte_map,
        "vol_adaptive_dte_map": vol_adaptive_dte,
        "vol_classification": vol_classification,
        "insample_dte_sensitivity": {tk: {str(k): v for k, v in sr.items()}
                                     for tk, sr in insample_results.items()},
    }
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    # MLflow logging
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"sector_dte_{t0.strftime('%Y%m%d_%H%M')}"):
                # Log all variant metrics
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
                    mlflow.log_metric(f"{prefix}_random_mean_sharpe", r.get("random_mean_sharpe", 0))
                    mlflow.log_metric(f"{prefix}_bull_wr", r.get("bull_wr", 0))
                    mlflow.log_metric(f"{prefix}_bear_wr", r.get("bear_wr", 0))

                # Log delta metrics
                for vname, r in all_results.items():
                    if vname != "A_dte21_baseline":
                        prefix = vname.split("_")[0]
                        delta = r.get("sharpe", 0) - baseline_sharpe
                        mlflow.log_metric(f"{prefix}_sharpe_delta", delta)

                # Log optimal DTE per sector
                for tk, dte_val in optimal_dte_map.items():
                    mlflow.log_metric(f"optimal_dte_{tk}", dte_val)

                mlflow.log_params({
                    "experiment_type": "sector_dte_optimization",
                    "capital": CAP,
                    "spread_pct": SPREAD_PCT,
                    "haircut": DEFAULT_HAIRCUT,
                    "commission": COMMISSION_RT_SPREAD,
                    "regime_bull_thresh": REGIME_BULL_THRESHOLD,
                    "rebal_freq": V6_REBAL_FREQ,
                    "otm_pct": V6_OTM_PCT,
                    "pairs": V6_PAIRS,
                    "max_pos_bull": V6_MAX_POS_BULL,
                    "max_pos_pair_leg": V6_MAX_POS_PAIR_LEG,
                    "hold_to_expiry": True,
                    "n_features": len(ALL_FEATURES),
                    "dte_candidates": str(DTE_CANDIDATES),
                    "n_variants": 6,
                    "wf_train_periods": WF_TRAIN_PERIODS,
                    "insample_pct": 0.60,
                    "oos_pct": 0.40,
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
