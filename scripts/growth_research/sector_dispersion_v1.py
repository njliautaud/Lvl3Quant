#!/usr/bin/env python3
"""
Sector Dispersion Research v1 — Can Cross-Sectional Dispersion Improve Regime Filtering?
=========================================================================================

Research question: Our V6 strategy's regime filter uses VIX only (VIX<20 = pair trades,
VIX>=20 = bull only). The main weakness is regime imbalance (bull Sharpe 3.91 vs bear 1.02).
Can sector dispersion (cross-sectional return spread among the 11 SPDR sectors) improve
regime filtering?

Hypothesis: When sectors move together (low dispersion), LGBM ranking has less signal
because all sectors are correlated. When sectors diverge (high dispersion), ranking should
be more predictive because there are genuine relative winners/losers.

Sector dispersion = cross-sectional std dev of sector returns over 21d lookback.

6 Variants (all V6 structure: weekly, 2% OTM, 17 features):
  A: Baseline V6 (no dispersion filter) — control
  B: Trade only when dispersion > median (high spread = better rankings)
  C: Scale position size by dispersion (higher dispersion = larger positions)
  D: Use dispersion as additional LGBM feature (22nd feature)
  E: VIX + dispersion dual filter (pairs only when VIX<20 AND dispersion>median)
  F: Adaptive mode: high dispersion = aggressive (more sectors), low = conservative (fewer)

Each variant: 5-gate adversarial validation + 5-trial random baseline.
MLflow experiment: 'sector_dispersion_v1'
Output: output/growth_research/sector_dispersion_v1/
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


# ── Standardized tools with inline fallback (Neptune-safe) ──
import os as _os
_hostname = _os.uname().nodename.lower()
if 'neptune' in _hostname or 'nick' in str(_os.path.expanduser('~')):
    _BASE_PATH = "/home/nick/Lvl3Quant"
else:
    _BASE_PATH = "/home/jupiter/Lvl3Quant"
sys.path.insert(0, _BASE_PATH)

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
    fprint("Imported from research.tools")
except ImportError:
    fprint("research.tools not found — using inline implementations")
    COMMISSION_RT_SPREAD = 2.60
    DEFAULT_HAIRCUT = 0.15

    def compute_atr(prices, window=14):
        high = prices.rolling(window).max()
        low = prices.rolling(window).min()
        return (high - low).mean()

    def estimate_iv(prices, window=21, mult=1.2):
        returns = np.log(prices / prices.shift(1)).dropna()
        hv = returns.rolling(window).std() * np.sqrt(252)
        return hv * mult

    def price_bull_call_spread(S, K1, K2, dte, atr, vix, haircut=0.15, **kw):
        from scipy.stats import norm
        T = dte / 365.0
        iv = max(vix / 100.0 * 1.2, 0.05)
        if T <= 0:
            return 0.0, 0.0
        d1_l = (np.log(S / K1) + (0.5 * iv**2) * T) / (iv * np.sqrt(T))
        d1_s = (np.log(S / K2) + (0.5 * iv**2) * T) / (iv * np.sqrt(T))
        call_l = S * norm.cdf(d1_l) - K1 * norm.cdf(d1_l - iv * np.sqrt(T))
        call_s = S * norm.cdf(d1_s) - K2 * norm.cdf(d1_s - iv * np.sqrt(T))
        spread_val = max(call_l - call_s, 0.001)
        entry = spread_val * (1 + haircut)
        max_profit = (K2 - K1) - entry
        return entry, max_profit

    def price_bear_put_spread(S, K1, K2, dte, atr, vix, haircut=0.15, **kw):
        from scipy.stats import norm
        T = dte / 365.0
        iv = max(vix / 100.0 * 1.2, 0.05)
        if T <= 0:
            return 0.0, 0.0
        d1_l = (np.log(S / K2) + (0.5 * iv**2) * T) / (iv * np.sqrt(T))
        d1_s = (np.log(S / K1) + (0.5 * iv**2) * T) / (iv * np.sqrt(T))
        put_l = K2 * norm.cdf(-(d1_l - iv * np.sqrt(T))) - S * norm.cdf(-d1_l)
        put_s = K1 * norm.cdf(-(d1_s - iv * np.sqrt(T))) - S * norm.cdf(-d1_s)
        spread_val = max(put_l - put_s, 0.001)
        entry = spread_val * (1 + haircut)
        max_profit = (K2 - K1) - entry
        return entry, max_profit

    def validate_trades(trades, initial_capital=645, spy_prices=None,
                        strategy_name="", n_perms=1000, **kw):
        """Inline 5-gate adversarial validation."""
        if not trades or len(trades) < 10:
            return _ValidationResult(0, 0, 0, 1.0, 0, 0, len(trades), initial_capital,
                                     0, 5, "INSUFFICIENT DATA")
        pnls = [t["pnl"] for t in trades]
        equity = [initial_capital]
        for p in pnls:
            equity.append(equity[-1] + p)
        equity = np.array(equity[1:])
        rets = np.diff(np.concatenate([[initial_capital], equity])) / np.concatenate(
            [[initial_capital], equity[:-1]])
        rets = rets[np.isfinite(rets)]
        sharpe = float(np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(52))
        down = rets[rets < 0]
        sortino = float(np.mean(rets) / (np.std(down) + 1e-10) * np.sqrt(52)) if len(down) > 0 else 0
        wr = float(np.mean([1 if p > 0 else 0 for p in pnls]))
        wins = sum(p for p in pnls if p > 0)
        losses = abs(sum(p for p in pnls if p <= 0))
        pf = float(wins / (losses + 1e-10))
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / (peak + 1e-10)
        mdd = float(np.min(dd))

        # 5 gates
        gates = 0
        if sharpe > 0.5: gates += 1    # Gate 1: Sharpe > 0.5
        if wr > 0.45: gates += 1        # Gate 2: WR > 45%
        if pf > 1.0: gates += 1         # Gate 3: PF > 1.0
        if mdd > -0.30: gates += 1      # Gate 4: MaxDD < 30%
        # Gate 5: permutation test
        if n_perms > 0 and len(pnls) >= 20:
            obs_mean = np.mean(pnls)
            perm_means = []
            rng = np.random.RandomState(42)
            for _ in range(min(n_perms, 500)):
                perm = rng.permutation(pnls)
                perm_means.append(np.mean(perm[:len(pnls)]))
            p_val = np.mean([1 if pm >= obs_mean else 0 for pm in perm_means])
            if p_val < 0.05:
                gates += 1

        return _ValidationResult(sharpe, sortino, wr, pf, mdd, 0, len(trades),
                                 float(equity[-1]), gates, 5,
                                 "PASS" if gates >= 4 else "FAIL")

    class _ValidationResult:
        def __init__(self, sharpe, sortino, wr, pf, mdd, cagr, n_trades,
                     final_equity, gates_passed, gates_total, verdict):
            self.sharpe = sharpe
            self.sortino = sortino
            self.win_rate = wr
            self.profit_factor = pf
            self.max_dd = mdd
            self.cagr = cagr
            self.n_trades = n_trades
            self.final_equity = final_equity
            self.gates_passed = gates_passed
            self.gates_total = gates_total
            self.verdict = verdict

        def print_summary(self):
            fprint(f"\n{'='*65}")
            fprint(f"  ADVERSARIAL VALIDATION")
            fprint(f"{'='*65}")
            fprint(f"  Trades: {self.n_trades}  |  Sharpe: {self.sharpe:.2f}  |  "
                   f"Sortino: {self.sortino:.2f}  |  WR: {self.win_rate:.1%}")
            fprint(f"  PF: {self.profit_factor:.2f}  |  MaxDD: {self.max_dd:.1%}  |  "
                   f"Final: ${self.final_equity:,.0f}")
            fprint(f"  Gates: {self.gates_passed}/{self.gates_total}  |  "
                   f"Verdict: {self.verdict}")
            fprint(f"{'='*65}")

        def to_dict(self):
            return {
                "sharpe": round(self.sharpe, 4),
                "sortino": round(self.sortino, 4),
                "win_rate": round(self.win_rate, 4),
                "profit_factor": round(self.profit_factor, 4),
                "max_dd": round(self.max_dd, 4),
                "cagr": round(self.cagr, 4),
                "n_trades": self.n_trades,
                "final_equity": round(self.final_equity, 2),
                "gates_passed": self.gates_passed,
                "gates_total": self.gates_total,
                "verdict": self.verdict,
            }


# ── Config ──
BASE = Path(_BASE_PATH)
OUTPUT_DIR = BASE / "output" / "growth_research" / "sector_dispersion_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
DTE = 21
SPREAD_PCT = 3.0
TOP_K = 3
REGIME_BULL_THRESHOLD = 0.4
REGIME_BEAR_THRESHOLD = 0.2
DISPERSION_LOOKBACK = 21  # days for rolling sector dispersion

# Regime predictions path
REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"

# LGBM walk-forward
WF_TRAIN_PERIODS = 12

# V6 config: weekly rebalance, 2% OTM, bull+pairs
V6_REBAL_FREQ = "W-FRI"
V6_OTM_PCT = 0.02
V6_PAIRS = True
V6_MAX_POS_BULL = 200
V6_MAX_POS_PAIR_LEG = 100

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "sector_dispersion_v1"

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
# FEATURE SET DEFINITIONS
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

ALL_21_FEATURES = LEGACY_FEATURES + VALIDATED_CROSS_ASSET

# Variant D adds dispersion as a separate feature
FEATURES_WITH_DISPERSION = ALL_21_FEATURES + ["sector_dispersion_level"]  # 22 features


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
# SECTOR DISPERSION COMPUTATION
# ══════════════════════════════════════════════════════════════

def compute_sector_dispersion_series(close, lookback=DISPERSION_LOOKBACK):
    """
    Compute rolling sector dispersion: cross-sectional std dev of sector
    returns over lookback window.

    Returns a pd.Series indexed by date with the dispersion value.
    """
    sector_cols = [c for c in SECTORS if c in close.columns]
    if len(sector_cols) < 3:
        raise ValueError(f"Need at least 3 sectors, got {len(sector_cols)}")

    # Daily returns for all sectors
    sector_rets = close[sector_cols].pct_change()

    # Cross-sectional std dev each day (how much do sectors diverge?)
    daily_cross_std = sector_rets.std(axis=1)

    # Smooth with rolling mean over lookback
    dispersion = daily_cross_std.rolling(lookback, min_periods=lookback // 2).mean()

    fprint(f"Sector dispersion computed: {dispersion.dropna().shape[0]} days")
    fprint(f"  Mean: {dispersion.mean():.6f}, Median: {dispersion.median():.6f}")
    fprint(f"  P25: {dispersion.quantile(0.25):.6f}, P75: {dispersion.quantile(0.75):.6f}")

    return dispersion


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
                          dispersion_series=None):
    """
    Build feature + target records for all sectors on all rebal dates.
    Uses bull_only regime mode (regime>0.4).
    If dispersion_series provided and 'sector_dispersion_level' in feature_cols,
    adds the raw dispersion level as a feature.
    """
    fprint(f"  Building records: {len(rebal_dates)} dates, {len(feature_cols)} features")

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

            # Cross-asset features if any are needed
            cross_asset = {}
            needs_cross = any(col in VALIDATED_CROSS_ASSET for col in feature_cols)
            if needs_cross:
                cross_asset = compute_cross_asset_features(tk, idx, close)

            # Add dispersion level as feature for variant D
            if "sector_dispersion_level" in feature_cols and dispersion_series is not None:
                if dt in dispersion_series.index and not pd.isna(dispersion_series.loc[dt]):
                    cross_asset["sector_dispersion_level"] = float(dispersion_series.loc[dt])
                else:
                    # Nearest available
                    valid = dispersion_series.dropna()
                    nearest_idx = valid.index.get_indexer([dt], method="ffill")
                    if nearest_idx[0] >= 0:
                        cross_asset["sector_dispersion_level"] = float(valid.iloc[nearest_idx[0]])
                    else:
                        cross_asset["sector_dispersion_level"] = 0.01

            # Forward return target (DTE days forward)
            fi = min(idx + DTE, len(close) - 1)
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
    """Compute strike prices for a spread (K1 < K2 always)."""
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
# TRADE EXECUTION HELPER
# ══════════════════════════════════════════════════════════════

def _execute_single_trade(tk, dt, direction, otm_pct, max_pos, close, atr_dict, cv, equity):
    """Execute a single spread trade. Returns PnL or None."""
    if tk not in close.columns or tk not in atr_dict:
        return None

    S = float(close[tk].loc[dt])
    di = close.index.get_loc(dt)
    ei = min(di + DTE, len(close) - 1)
    if ei <= di:
        return None

    if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]):
        av = float(atr_dict[tk].loc[dt])
    else:
        av = S * 0.015

    K1, K2 = compute_strikes(S, direction, otm_pct, SPREAD_PCT)

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
    return pnl


# ══════════════════════════════════════════════════════════════
# TRADE SIMULATION (V6 config with dispersion variants)
# ══════════════════════════════════════════════════════════════

def simulate_trades(name, rankings, close, high, low, atr_dict,
                    dispersion_series=None, variant_mode="baseline",
                    dispersion_median=None):
    """
    Simulate trades using V6 config with dispersion-based modifications.

    variant_mode controls behavior:
      'baseline':   Standard V6, no dispersion filter (Variant A)
      'filter':     Only trade when dispersion > median (Variant B)
      'scale':      Scale position size by dispersion percentile (Variant C)
      'baseline_d': Same as baseline but with extra feature (Variant D — filter at model level)
      'dual_filter': VIX < 20 AND dispersion > median for pairs (Variant E)
      'adaptive':   High dispersion = top 4 sectors, low = top 2 (Variant F)
    """
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    equity = CAP
    trades = []

    # Pre-compute dispersion percentiles for scaling (Variant C)
    if dispersion_series is not None and variant_mode == "scale":
        disp_expanding_rank = dispersion_series.expanding().rank(pct=True)
    else:
        disp_expanding_rank = None

    for dt in sorted(rankings.keys()):
        if dt not in spy.index:
            continue

        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0
        scores = rankings[dt]
        if not scores:
            continue

        # ── Get current dispersion ──
        current_disp = None
        if dispersion_series is not None and dt in dispersion_series.index:
            current_disp = float(dispersion_series.loc[dt])
        elif dispersion_series is not None:
            valid = dispersion_series.dropna()
            nearest_idx = valid.index.get_indexer([dt], method="ffill")
            if nearest_idx[0] >= 0:
                current_disp = float(valid.iloc[nearest_idx[0]])

        # ── Variant B: Skip if dispersion below median ──
        if variant_mode == "filter":
            if current_disp is not None and dispersion_median is not None:
                if current_disp <= dispersion_median:
                    continue

        # ── Variant E: Dual filter — pairs only when VIX<20 AND dispersion>median ──
        if variant_mode == "dual_filter":
            if cv < 20.0:
                # For pair trades, also require high dispersion
                if current_disp is not None and dispersion_median is not None:
                    if current_disp <= dispersion_median:
                        # Low dispersion + low VIX: skip pairs, do bull only
                        trade_mode = "bull_only"
                    else:
                        trade_mode = "pairs"
                else:
                    trade_mode = "pairs"
            else:
                trade_mode = "bull_only"
        else:
            # Standard V6 pair logic
            if V6_PAIRS and cv < 20.0:
                trade_mode = "pairs"
            else:
                trade_mode = "bull_only"

        # ── Variant F: Adaptive TOP_K ──
        if variant_mode == "adaptive":
            if current_disp is not None and dispersion_median is not None:
                if current_disp > dispersion_median:
                    top_k = 4  # High dispersion → aggressive (more sectors)
                else:
                    top_k = 2  # Low dispersion → conservative (fewer sectors)
            else:
                top_k = TOP_K
        else:
            top_k = TOP_K

        # Pick sectors
        ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_asc = sorted(scores.items(), key=lambda x: x[1])

        bull_picks = [t for t, _ in ranked_desc[:top_k]]
        bear_picks = [t for t, _ in ranked_asc[:top_k]] if trade_mode == "pairs" else []

        # ── Variant C: Scale position size by dispersion ──
        if variant_mode == "scale" and disp_expanding_rank is not None and dt in disp_expanding_rank.index:
            disp_pctl = float(disp_expanding_rank.loc[dt])
            if pd.isna(disp_pctl):
                disp_pctl = 0.5
            # Scale: 0.5x at P0 → 1.5x at P100
            scale_factor = 0.5 + disp_pctl * 1.0
        else:
            scale_factor = 1.0

        # Position sizing
        if trade_mode == "pairs":
            max_pos = min(V6_MAX_POS_PAIR_LEG * scale_factor, equity / 6)
        else:
            max_pos = min(V6_MAX_POS_BULL * scale_factor, equity / 3)

        if max_pos < 30:
            continue

        # Execute bull leg
        for tk in bull_picks:
            pnl = _execute_single_trade(
                tk, dt, "bull", V6_OTM_PCT, max_pos, close, atr_dict, cv, equity
            )
            if pnl is not None:
                equity += pnl
                di = close.index.get_loc(dt)
                ei = min(di + DTE, len(close) - 1)
                sv = float(spy.loc[dt])
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
                    "trade_mode": trade_mode,
                    "dispersion": round(current_disp, 6) if current_disp else None,
                })

        # Execute bear leg (pairs mode only)
        for tk in bear_picks:
            pnl = _execute_single_trade(
                tk, dt, "bear", V6_OTM_PCT, max_pos, close, atr_dict, cv, equity
            )
            if pnl is not None:
                equity += pnl
                di = close.index.get_loc(dt)
                ei = min(di + DTE, len(close) - 1)
                sv = float(spy.loc[dt])
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
                    "trade_mode": trade_mode,
                    "dispersion": round(current_disp, 6) if current_disp else None,
                })

    return trades, equity


# ══════════════════════════════════════════════════════════════
# RANDOM BASELINE
# ══════════════════════════════════════════════════════════════

def random_baseline_test(rankings, close, high, low, atr_dict,
                         dispersion_series=None, variant_mode="baseline",
                         dispersion_median=None, n_trials=5):
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
            f"Random_{trial}", rand_rankings, close, high, low, atr_dict,
            dispersion_series=dispersion_series,
            variant_mode=variant_mode,
            dispersion_median=dispersion_median,
        )

        if trades and len(trades) >= 10:
            result = validate_trades(
                trades, initial_capital=CAP,
                spy_prices=close["SPY"],
                strategy_name=f"Random_{trial}",
                n_perms=500,
            )
            rs = result.sharpe if hasattr(result, 'sharpe') else result.get("sharpe", 0)
            random_sharpes.append(rs)
            fprint(f"    Random trial {trial}: Sharpe {rs:.2f}, "
                   f"${CAP:.0f}->${final_eq:.0f}")
        else:
            random_sharpes.append(0.0)

    return random_sharpes


# ══════════════════════════════════════════════════════════════
# REBALANCE DATE GENERATION
# ══════════════════════════════════════════════════════════════

def generate_rebal_dates(close, freq_str):
    """Generate rebalance dates from close index based on frequency string."""
    if freq_str == "3B":
        bdays = close.index[close.index.dayofweek < 5]
        rebal_dates = pd.DatetimeIndex([bdays[i] for i in range(0, len(bdays), 3)])
    else:
        rebal_dates = pd.DatetimeIndex(
            close.index.to_series().resample(freq_str).last().dropna().values
        )
    return rebal_dates


# ══════════════════════════════════════════════════════════════
# DISPERSION ANALYSIS HELPER
# ══════════════════════════════════════════════════════════════

def analyze_dispersion_regimes(trades, dispersion_median):
    """Analyze performance in high vs low dispersion periods."""
    if not trades:
        return {}

    high_disp = [t for t in trades if t.get("dispersion") and t["dispersion"] > dispersion_median]
    low_disp = [t for t in trades if t.get("dispersion") and t["dispersion"] <= dispersion_median]

    result = {}
    for label, subset in [("high_dispersion", high_disp), ("low_dispersion", low_disp)]:
        if len(subset) < 5:
            result[label] = {"n": len(subset), "note": "too few trades"}
            continue
        pnls = [t["pnl"] for t in subset]
        wr = sum(1 for p in pnls if p > 0) / len(pnls)
        avg_pnl = np.mean(pnls)
        total_pnl = sum(pnls)
        wins = sum(p for p in pnls if p > 0)
        losses = abs(sum(p for p in pnls if p <= 0))
        pf = wins / (losses + 1e-10)
        result[label] = {
            "n": len(subset),
            "wr": round(wr, 3),
            "avg_pnl": round(avg_pnl, 2),
            "total_pnl": round(total_pnl, 2),
            "pf": round(pf, 2),
        }

    return result


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t0 = datetime.now()
    fprint("=" * 100)
    fprint(f"SECTOR DISPERSION RESEARCH v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 100)
    fprint(f"Question: Can sector dispersion improve V6 regime filtering?")
    fprint(f"Hypothesis: High dispersion = better LGBM rankings (more sector divergence)")
    fprint()
    fprint(f"V6 config (fixed for ALL variants):")
    fprint(f"  Capital: ${CAP:.0f} | DTE: {DTE} | Spread: {SPREAD_PCT:.0f}% | "
           f"Haircut: {DEFAULT_HAIRCUT:.0%} entry only | Comm: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"  Rebalance: weekly (W-FRI) | OTM: 2% | Pairs: VIX<20 bull+bear")
    fprint(f"  Hold to expiry | Intrinsic value only | No exit haircut")
    fprint(f"  Regime filter: GRU >0.4")
    fprint(f"  Dispersion lookback: {DISPERSION_LOOKBACK}d")
    fprint()
    fprint(f"6 Variants:")
    fprint(f"  A: Baseline V6 (no dispersion filter)")
    fprint(f"  B: Trade only when dispersion > median")
    fprint(f"  C: Scale position size by dispersion percentile")
    fprint(f"  D: Dispersion as 22nd LGBM feature")
    fprint(f"  E: VIX + dispersion dual filter (pairs require both VIX<20 AND disp>median)")
    fprint(f"  F: Adaptive TOP_K: high disp = 4 sectors, low = 2 sectors")
    fprint()

    # 1. Download data
    close, high, low = download_data()

    # 2. Load regime predictions
    regime_series = load_regime_predictions()

    # 3. Pre-compute ATR
    atr_dict = compute_atr_series(high, low, close)

    # 4. Compute sector dispersion series
    fprint(f"\n{'=' * 100}")
    fprint("COMPUTING SECTOR DISPERSION")
    fprint(f"{'=' * 100}")
    dispersion_series = compute_sector_dispersion_series(close, DISPERSION_LOOKBACK)
    dispersion_median = float(dispersion_series.median())
    fprint(f"  Dispersion median (filter threshold): {dispersion_median:.6f}")
    fprint(f"  Days above median: {(dispersion_series > dispersion_median).sum()}")
    fprint(f"  Days below median: {(dispersion_series <= dispersion_median).sum()}")

    # Analyze dispersion vs VIX correlation
    vix = close["VIX"] if "VIX" in close.columns else None
    if vix is not None:
        common_idx = dispersion_series.dropna().index.intersection(vix.dropna().index)
        if len(common_idx) > 100:
            corr = dispersion_series.loc[common_idx].corr(vix.loc[common_idx])
            fprint(f"  Dispersion-VIX correlation: {corr:.3f}")
            if abs(corr) > 0.5:
                fprint(f"    WARNING: High correlation — dispersion may be redundant with VIX")
            else:
                fprint(f"    OK: Low correlation — dispersion adds orthogonal information")

    # 5. Generate V6 rebalance dates (weekly)
    rebal_dates = generate_rebal_dates(close, V6_REBAL_FREQ)
    fprint(f"\nRebalance dates: {len(rebal_dates)} "
           f"({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

    spy_close = close["SPY"]

    # ── Build rankings for variants A/B/C/E/F (all use 21 features) ──
    fprint(f"\n{'=' * 100}")
    fprint("BUILDING LGBM RANKINGS (21 features — shared by A/B/C/E/F)")
    fprint(f"{'=' * 100}")
    records_21 = build_feature_records(
        close, high, low, rebal_dates, ALL_21_FEATURES, regime_series
    )
    rankings_21, imp_21 = walk_forward_lgbm_rank(records_21, ALL_21_FEATURES, "shared_21feat")

    # ── Build rankings for variant D (22 features — adds dispersion level) ──
    fprint(f"\n{'=' * 100}")
    fprint("BUILDING LGBM RANKINGS (22 features — variant D with dispersion feature)")
    fprint(f"{'=' * 100}")
    records_22 = build_feature_records(
        close, high, low, rebal_dates, FEATURES_WITH_DISPERSION, regime_series,
        dispersion_series=dispersion_series,
    )
    rankings_22, imp_22 = walk_forward_lgbm_rank(records_22, FEATURES_WITH_DISPERSION, "D_disp_feature")

    # ── SIMULATE ALL 6 VARIANTS ──
    VARIANT_CONFIGS = [
        ("A_baseline",     rankings_21, "baseline",    "V6 baseline, no dispersion filter"),
        ("B_disp_filter",  rankings_21, "filter",      "Trade only when disp > median"),
        ("C_disp_scale",   rankings_21, "scale",       "Scale pos size by disp percentile"),
        ("D_disp_feature", rankings_22, "baseline",    "Dispersion as 22nd LGBM feature"),
        ("E_dual_filter",  rankings_21, "dual_filter", "VIX<20 AND disp>median for pairs"),
        ("F_adaptive_k",   rankings_21, "adaptive",    "High disp=4 sectors, low=2"),
    ]

    all_results = {}

    for vname, rankings, vmode, desc in VARIANT_CONFIGS:
        fprint(f"\n{'=' * 100}")
        fprint(f"VARIANT {vname}: {desc}")
        fprint(f"{'=' * 100}")

        if not rankings:
            fprint(f"  No rankings available, skipping")
            continue

        # Simulate
        trades, final_eq = simulate_trades(
            vname, rankings, close, high, low, atr_dict,
            dispersion_series=dispersion_series,
            variant_mode=vmode,
            dispersion_median=dispersion_median,
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
        if hasattr(result, 'print_summary'):
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

        # Dispersion regime analysis
        disp_analysis = analyze_dispersion_regimes(trades, dispersion_median)
        if disp_analysis:
            fprint(f"  Dispersion regime breakdown:")
            for regime, stats_d in disp_analysis.items():
                if "note" in stats_d:
                    fprint(f"    {regime}: {stats_d['n']} trades ({stats_d['note']})")
                else:
                    fprint(f"    {regime}: {stats_d['n']} trades, WR {stats_d['wr']:.1%}, "
                           f"PF {stats_d['pf']:.2f}, total ${stats_d['total_pnl']:.0f}")

        # Random baseline
        random_sharpes = random_baseline_test(
            rankings, close, high, low, atr_dict,
            dispersion_series=dispersion_series,
            variant_mode=vmode,
            dispersion_median=dispersion_median,
        )
        mean_random = np.mean(random_sharpes) if random_sharpes else 0

        # Extract metrics
        if hasattr(result, 'sharpe'):
            r_sharpe = result.sharpe
            r_sortino = result.sortino
            r_wr = result.win_rate
            r_pf = result.profit_factor
            r_mdd = result.max_dd
            r_gates = result.gates_passed
            r_gtotal = result.gates_total
            r_final = result.final_equity
            r_dict = result.to_dict()
        else:
            r_sharpe = result.get("sharpe", 0)
            r_sortino = result.get("sortino", 0)
            r_wr = result.get("wr", 0)
            r_pf = result.get("pf", 0)
            r_mdd = result.get("max_dd", 0)
            r_gates = result.get("gates_passed", 0)
            r_gtotal = result.get("total_gates", 5)
            r_final = result.get("final_equity", 0)
            r_dict = result

        fprint(f"  ML Sharpe: {r_sharpe:.2f} vs Random mean: {mean_random:.2f}")
        if r_sharpe > 0 and mean_random > 0:
            fprint(f"  ML alpha ratio: {r_sharpe / mean_random:.2f}x")

        all_results[vname] = {
            "description": desc,
            "variant_mode": vmode,
            **r_dict,
            "random_sharpes": [round(s, 3) for s in random_sharpes],
            "random_mean_sharpe": round(mean_random, 3),
            "bull_trades": len(bull_trades),
            "bear_trades": len(bear_trades),
            "bull_pnl": round(bull_pnl, 2),
            "bear_pnl": round(bear_pnl, 2),
            "bull_wr": round(bull_wr, 1),
            "bear_wr": round(bear_wr, 1),
            "dispersion_analysis": disp_analysis,
        }

    # ── SUMMARY COMPARISON ──
    fprint(f"\n{'=' * 120}")
    fprint("SECTOR DISPERSION RESEARCH — SUMMARY COMPARISON")
    fprint(f"{'=' * 120}")
    fprint(f"{'Variant':<25} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} "
           f"{'PF':>6} {'MaxDD':>7} {'Gates':>6} {'Final$':>8} {'RandSh':>7}")
    fprint("-" * 120)

    baseline_sharpe = all_results.get("A_baseline", {}).get("sharpe", 0)

    for vname, _, _, _ in VARIANT_CONFIGS:
        r = all_results.get(vname)
        if not r:
            fprint(f"  {vname:<25} — NO DATA —")
            continue
        _wr = r.get("win_rate", r.get("wr", 0))
        _pf = r.get("profit_factor", r.get("pf", 0))
        _mdd = r.get("max_dd", 0)
        _gp = r.get("gates_passed", 0)
        _gt = r.get("gates_total", r.get("total_gates", 5))
        fprint(f"  {vname:<25} {r.get('n_trades', 0):>5} {r['sharpe']:>7.2f} "
               f"{r['sortino']:>8.2f} {_wr*100:>5.1f}% {_pf:>5.2f} "
               f"{_mdd*100:>6.1f}% {_gp}/{_gt} "
               f"${r.get('final_equity', 0):>7,.0f} {r['random_mean_sharpe']:>7.2f}")

    # ── SHARPE DELTA ANALYSIS ──
    fprint(f"\n{'=' * 100}")
    fprint("SHARPE DELTA vs BASELINE (A_baseline)")
    fprint(f"{'=' * 100}")
    fprint(f"  Baseline (A_baseline) Sharpe: {baseline_sharpe:.2f}")
    fprint()

    deltas = []
    for vname, _, _, desc in VARIANT_CONFIGS:
        if vname == "A_baseline":
            continue
        r = all_results.get(vname)
        if not r:
            continue
        delta = r["sharpe"] - baseline_sharpe
        pct_delta = delta / abs(baseline_sharpe) * 100 if baseline_sharpe != 0 else 0
        deltas.append((vname, delta, pct_delta, r["sharpe"], desc))

    deltas.sort(key=lambda x: x[1], reverse=True)

    max_abs_delta = max(abs(d[1]) for d in deltas) if deltas else 1
    for vname, delta, pct_delta, sharpe, desc in deltas:
        direction = "+" if delta >= 0 else ""
        bar = "*" * int(abs(delta) / max_abs_delta * 30) if max_abs_delta > 0 else ""
        sign = "BETTER" if delta > 0.1 else "WORSE" if delta < -0.1 else "SIMILAR"
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
        fprint(f"  Best variant:  {best[0]} (Sharpe {best[3]:.2f}, delta {best[1]:+.2f}) — {best[4]}")
        fprint(f"  Worst variant: {worst[0]} (Sharpe {worst[3]:.2f}, delta {worst[1]:+.2f}) — {worst[4]}")

    # Dispersion value analysis
    fprint(f"\n  Dispersion regime analysis across variants:")
    for vname, _, _, _ in VARIANT_CONFIGS:
        r = all_results.get(vname)
        if not r or not r.get("dispersion_analysis"):
            continue
        da = r["dispersion_analysis"]
        h = da.get("high_dispersion", {})
        l = da.get("low_dispersion", {})
        if "note" not in h and "note" not in l:
            h_pf = h.get("pf", 0)
            l_pf = l.get("pf", 0)
            fprint(f"  {vname:<25} High-disp PF: {h_pf:.2f} ({h.get('n', 0)} trades)  "
                   f"Low-disp PF: {l_pf:.2f} ({l.get('n', 0)} trades)  "
                   f"{'HIGH WINS' if h_pf > l_pf else 'LOW WINS'}")

    # Feature importance for Variant D
    if imp_22 is not None:
        fprint(f"\n  Feature importance with dispersion (Variant D, top 10):")
        for _, row in imp_22.head(10).iterrows():
            bar = "*" * int(row["importance"] / imp_22["importance"].max() * 25)
            marker = " <== NEW" if row["feature"] == "sector_dispersion_level" else ""
            fprint(f"    {row['feature']:<30} {row['importance']:>6.1f} {bar}{marker}")

    # Research conclusion
    fprint(f"\n{'=' * 100}")
    fprint("RESEARCH CONCLUSION")
    fprint(f"{'=' * 100}")
    if deltas:
        best_delta = max(d[1] for d in deltas)
        if best_delta > 0.3:
            fprint(f"  POSITIVE: Sector dispersion IMPROVES regime filtering.")
            fprint(f"  Best approach: {best[0]} ({best[4]})")
            fprint(f"  Recommendation: Integrate into V7 config.")
        elif best_delta > 0:
            fprint(f"  MARGINAL: Sector dispersion provides small improvement ({best_delta:+.2f} Sharpe).")
            fprint(f"  Best approach: {best[0]} ({best[4]})")
            fprint(f"  Recommendation: Test further with more data before production.")
        else:
            fprint(f"  NEGATIVE: Sector dispersion does NOT improve regime filtering.")
            fprint(f"  All variants underperform baseline.")
            fprint(f"  Recommendation: Keep V6 as-is. Dispersion adds complexity without benefit.")

    # Save results
    results_path = OUTPUT_DIR / "sector_dispersion_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    # Save dispersion series for potential future use
    disp_path = OUTPUT_DIR / "dispersion_series.csv"
    dispersion_series.dropna().to_csv(disp_path)
    fprint(f"Dispersion series saved to {disp_path}")

    # MLflow logging
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"sect_disp_{t0.strftime('%Y%m%d_%H%M')}"):
                for vname, r in all_results.items():
                    prefix = vname.split("_")[0]
                    mlflow.log_metric(f"{prefix}_sharpe", r.get("sharpe", 0))
                    mlflow.log_metric(f"{prefix}_sortino", r.get("sortino", 0))
                    wr = r.get("win_rate", r.get("wr", 0))
                    mlflow.log_metric(f"{prefix}_win_rate", wr)
                    pf = r.get("profit_factor", r.get("pf", 0))
                    mlflow.log_metric(f"{prefix}_profit_factor", pf)
                    mlflow.log_metric(f"{prefix}_max_dd", r.get("max_dd", 0))
                    mlflow.log_metric(f"{prefix}_n_trades", r.get("n_trades", 0))
                    mlflow.log_metric(f"{prefix}_gates_passed", r.get("gates_passed", 0))
                    mlflow.log_metric(f"{prefix}_final_equity", r.get("final_equity", 0))
                    mlflow.log_metric(f"{prefix}_random_mean_sharpe", r.get("random_mean_sharpe", 0))
                    mlflow.log_metric(f"{prefix}_bull_wr", r.get("bull_wr", 0))
                    mlflow.log_metric(f"{prefix}_bear_wr", r.get("bear_wr", 0))

                    # Delta vs baseline
                    if vname != "A_baseline":
                        delta = r.get("sharpe", 0) - baseline_sharpe
                        mlflow.log_metric(f"{prefix}_sharpe_delta", delta)

                mlflow.log_params({
                    "experiment_type": "sector_dispersion",
                    "capital": CAP,
                    "dte": DTE,
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
                    "dispersion_lookback": DISPERSION_LOOKBACK,
                    "dispersion_median": round(dispersion_median, 6),
                    "n_variants": 6,
                    "wf_train_periods": WF_TRAIN_PERIODS,
                    "n_features_base": 21,
                    "n_features_variant_d": 22,
                })

                mlflow.log_artifact(str(results_path))
                mlflow.log_artifact(str(disp_path))
            fprint(f"MLflow run logged to experiment '{EXPERIMENT_NAME}'")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    fprint("DONE")


if __name__ == "__main__":
    main()
