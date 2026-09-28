#!/usr/bin/env python3
"""
DTE Sensitivity Research v1 — Optimal Days-to-Expiration for V6 Sector Rotation
==================================================================================

Research question: Our V6 sector rotation uses DTE=21 (3 weeks to expiration).
Is this optimal? Shorter DTE = more theta decay (cheaper entry, faster time value loss)
but less time for the trade to work. Longer DTE = more expensive but more time.

Hypothesis: There exists an optimal DTE that maximizes risk-adjusted returns (Sharpe).
Too short = insufficient time for directional moves to materialize.
Too long = premium drag and capital efficiency loss.

6 Variants (all V6 structure: weekly rebalance, 3% spread, 2% OTM, 21 features, GRU regime >0.4):
  A: DTE 7  (1 week — very short, maximum theta decay, cheapest)
  B: DTE 14 (2 weeks — short)
  C: DTE 21 (3 weeks — current baseline)
  D: DTE 28 (4 weeks — monthly)
  E: DTE 35 (5 weeks)
  F: DTE 42 (6 weeks — longest, most expensive, most time)

Each variant: 5-gate adversarial validation + 5-trial random baseline.
MLflow experiment: 'dte_sensitivity_v1'
Output: output/growth_research/dte_sensitivity_v1/
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
OUTPUT_DIR = BASE / "output" / "growth_research" / "dte_sensitivity_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
SPREAD_PCT = 3.0        # FIXED at 3% for all variants
TOP_K = 3
REGIME_BULL_THRESHOLD = 0.4

# Regime predictions path
REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"

# LGBM walk-forward
WF_TRAIN_PERIODS = 25

# V6 config: weekly rebalance, 2% OTM, bull+pairs
V6_REBAL_FREQ = "W-FRI"
V6_OTM_PCT = 0.02
V6_PAIRS = True
V6_MAX_POS_BULL = 200
V6_MAX_POS_PAIR_LEG = 100

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "dte_sensitivity_v1"

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
# DTE VARIANT DEFINITIONS
# ══════════════════════════════════════════════════════════════

VARIANT_CONFIGS = {
    "A_dte7":  {"dte": 7,  "desc": "DTE 7 (1 week — very short, max theta decay)"},
    "B_dte14": {"dte": 14, "desc": "DTE 14 (2 weeks — short)"},
    "C_dte21": {"dte": 21, "desc": "DTE 21 (3 weeks — current baseline)"},
    "D_dte28": {"dte": 28, "desc": "DTE 28 (4 weeks — monthly)"},
    "E_dte35": {"dte": 35, "desc": "DTE 35 (5 weeks)"},
    "F_dte42": {"dte": 42, "desc": "DTE 42 (6 weeks — longest, most expensive)"},
}


# ══════════════════════════════════════════════════════════════
# FEATURE SET DEFINITIONS (standard 21 features)
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
           f"Days >0.4: {(regime_series > REGIME_BULL_THRESHOLD).sum()}")
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
    """Compute the 3 validated cross-asset features."""
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
# WALK-FORWARD LGBM RANKING (DTE-AWARE)
# ══════════════════════════════════════════════════════════════

def build_feature_records(close, high, low, rebal_dates, feature_cols, regime_series, dte):
    """Build feature + target records for all sectors on all rebal dates.
    Forward return target uses the variant-specific DTE."""
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

            # Forward return target uses variant-specific DTE
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

def compute_atr_series(high, low, close, period=21):
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
# STRIKE COMPUTATION — FIXED 3% SPREAD WIDTH
# ══════════════════════════════════════════════════════════════

def compute_strikes(S, direction, otm_pct):
    """
    Compute strike prices with fixed 3% spread width.
    Spread width is NOT varying in this study — only DTE varies.
    """
    if direction == "bull":
        K1 = round(S * (1 + otm_pct), 2)
        K2 = round(K1 * (1 + SPREAD_PCT / 100), 2)
    else:  # bear
        K2 = round(S * (1 - otm_pct), 2)
        K1 = round(K2 * (1 - SPREAD_PCT / 100), 2)

    if K2 <= K1:
        K2 = K1 + 0.50
    return K1, K2


# ══════════════════════════════════════════════════════════════
# TRADE EXECUTION HELPER
# ══════════════════════════════════════════════════════════════

def _execute_single_trade(tk, dt, direction, otm_pct, max_pos, close, atr_dict,
                          cv, equity, dte):
    """Execute a single spread trade with variant-specific DTE."""
    if tk not in close.columns or tk not in atr_dict:
        return None

    S = float(close[tk].loc[dt])
    di = close.index.get_loc(dt)
    ei = min(di + dte, len(close) - 1)  # DTE-specific expiry
    if ei <= di:
        return None

    # ATR value for this ticker at this date
    if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]):
        av = float(atr_dict[tk].loc[dt])
    else:
        av = S * 0.015

    K1, K2 = compute_strikes(S, direction, otm_pct)

    spread_width_dollars = K2 - K1

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
        "pnl": round(pnl, 2),
        "entry_cost": round(total_cost, 2),
        "max_profit": round(max_profit_ps * 100, 2),
        "spread_width_dollars": round(spread_width_dollars, 2),
        "K1": K1,
        "K2": K2,
        "S": S,
        "S_exp": round(Se, 2),
        "days_held": ei - di,
    }


# ══════════════════════════════════════════════════════════════
# TRADE SIMULATION
# ══════════════════════════════════════════════════════════════

def simulate_trades(name, rankings, close, high, low, atr_dict, dte):
    """
    Simulate trades using V6 config with a specific DTE.
    All variants use identical LGBM rankings, regime filter, spread width, and position sizing.
    Only the DTE (days to expiration) differs.
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

        # Standard V6 pair logic
        if V6_PAIRS and cv < 20.0:
            trade_mode = "pairs"
        else:
            trade_mode = "bull_only"

        # Pick sectors
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
            result = _execute_single_trade(
                tk, dt, "bull", V6_OTM_PCT, max_pos, close, atr_dict, cv, equity,
                dte
            )
            if result is not None:
                equity += result["pnl"]
                di = close.index.get_loc(dt)
                ei = min(di + dte, len(close) - 1)
                sv = float(spy.loc[dt])
                se = float(spy.iloc[ei]) if ei < len(spy) else sv
                spy_regime = "bull" if se >= sv else "bear"
                trades.append({
                    "pnl": result["pnl"],
                    "entry_date": str(dt.date()),
                    "exit_date": str(close.index[ei].date()),
                    "ticker": tk,
                    "regime": spy_regime,
                    "direction": "bull",
                    "vix": round(cv, 1),
                    "win": result["pnl"] > 0,
                    "trade_mode": trade_mode,
                    "entry_cost": result["entry_cost"],
                    "max_profit": result["max_profit"],
                    "days_held": result["days_held"],
                })

        # Execute bear leg (pairs mode only)
        for tk in bear_picks:
            result = _execute_single_trade(
                tk, dt, "bear", V6_OTM_PCT, max_pos, close, atr_dict, cv, equity,
                dte
            )
            if result is not None:
                equity += result["pnl"]
                di = close.index.get_loc(dt)
                ei = min(di + dte, len(close) - 1)
                sv = float(spy.loc[dt])
                se = float(spy.iloc[ei]) if ei < len(spy) else sv
                spy_regime = "bull" if se >= sv else "bear"
                trades.append({
                    "pnl": result["pnl"],
                    "entry_date": str(dt.date()),
                    "exit_date": str(close.index[ei].date()),
                    "ticker": tk,
                    "regime": spy_regime,
                    "direction": "bear",
                    "vix": round(cv, 1),
                    "win": result["pnl"] > 0,
                    "trade_mode": trade_mode,
                    "entry_cost": result["entry_cost"],
                    "max_profit": result["max_profit"],
                    "days_held": result["days_held"],
                })

    return trades, equity


# ══════════════════════════════════════════════════════════════
# DTE-SPECIFIC ANALYSIS HELPERS
# ══════════════════════════════════════════════════════════════

def analyze_dte_economics(trades, dte):
    """Analyze the cost/profit economics and time-value characteristics."""
    if not trades or len(trades) < 5:
        return {}

    costs = [t["entry_cost"] for t in trades]
    max_profs = [t["max_profit"] for t in trades]
    pnls = [t["pnl"] for t in trades]
    days_held = [t["days_held"] for t in trades]

    winners = [t for t in trades if t["pnl"] > 0]
    losers = [t for t in trades if t["pnl"] <= 0]

    # How often do winners hit max profit?
    near_max = [t for t in winners if t["max_profit"] > 0 and
                t["pnl"] >= t["max_profit"] * 0.80]

    # Capital efficiency: PnL per dollar deployed per day
    cap_efficiency = []
    for t in trades:
        if t["entry_cost"] > 0 and t["days_held"] > 0:
            cap_efficiency.append(t["pnl"] / t["entry_cost"] / t["days_held"])

    # Annualized return on risk (trades per year * avg pnl / avg cost)
    trades_per_year = 52 / max(dte / 7, 1)  # rough estimate

    return {
        "dte": dte,
        "avg_entry_cost": round(np.mean(costs), 2),
        "median_entry_cost": round(np.median(costs), 2),
        "avg_max_profit": round(np.mean(max_profs), 2),
        "cost_profit_ratio": round(np.mean(costs) / (np.mean(max_profs) + 1e-10), 3),
        "avg_days_held": round(np.mean(days_held), 1),
        "n_near_max_profit": len(near_max),
        "pct_near_max_profit": round(len(near_max) / max(len(winners), 1) * 100, 1),
        "avg_win_pnl": round(np.mean([t["pnl"] for t in winners]), 2) if winners else 0,
        "avg_loss_pnl": round(np.mean([t["pnl"] for t in losers]), 2) if losers else 0,
        "win_loss_ratio": round(
            abs(np.mean([t["pnl"] for t in winners])) /
            (abs(np.mean([t["pnl"] for t in losers])) + 1e-10), 2
        ) if winners and losers else 0,
        "avg_cap_efficiency": round(np.mean(cap_efficiency) * 1000, 4) if cap_efficiency else 0,
        "est_trades_per_year": round(trades_per_year, 1),
        "total_pnl": round(sum(pnls), 2),
    }


def analyze_regime_performance(trades):
    """Break down performance by bull/bear market regime."""
    if not trades:
        return {}
    result = {}
    for regime in ["bull", "bear"]:
        subset = [t for t in trades if t["regime"] == regime]
        if len(subset) < 5:
            result[regime] = {"n": len(subset), "note": "too few trades"}
            continue
        pnls = [t["pnl"] for t in subset]
        wr = sum(1 for p in pnls if p > 0) / len(pnls)
        wins = sum(p for p in pnls if p > 0)
        losses = abs(sum(p for p in pnls if p <= 0))
        result[regime] = {
            "n": len(subset),
            "wr": round(wr, 3),
            "avg_pnl": round(np.mean(pnls), 2),
            "total_pnl": round(sum(pnls), 2),
            "pf": round(wins / (losses + 1e-10), 2),
        }
    return result


# ══════════════════════════════════════════════════════════════
# RANDOM BASELINE
# ══════════════════════════════════════════════════════════════

def random_baseline_test(rankings, close, high, low, atr_dict, dte, n_trials=5):
    """Test if random sector selection produces similar returns."""
    fprint(f"\n  Random baseline ({n_trials} trials)...")
    random_sharpes = []

    for trial in range(n_trials):
        np.random.seed(42 + trial)
        rand_rankings = {}
        for dt, scores in rankings.items():
            rand_scores = {tk: np.random.random() for tk in scores.keys()}
            rand_rankings[dt] = rand_scores

        trades, final_eq = simulate_trades(
            f"Random_{trial}", rand_rankings, close, high, low, atr_dict, dte
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
    fprint(f"DTE SENSITIVITY RESEARCH v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 100)
    fprint(f"Question: What is the optimal days-to-expiration for V6 sector rotation options?")
    fprint(f"Hypothesis: Shorter DTE = cheaper entry but less time. Optimal DTE balances")
    fprint(f"  theta decay savings against directional move probability.")
    fprint()
    fprint(f"Fixed V6 config (identical for ALL variants):")
    fprint(f"  Capital: ${CAP:.0f} | Spread: {SPREAD_PCT:.0f}% | OTM: {V6_OTM_PCT:.0%} | "
           f"Haircut: {DEFAULT_HAIRCUT:.0%} entry only | Comm: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"  Rebalance: weekly (W-FRI) | Hold to expiry | Intrinsic value only")
    fprint(f"  Regime filter: GRU >0.4 | Pairs: VIX<20 bull+bear")
    fprint(f"  Walk-forward: {WF_TRAIN_PERIODS} train periods | LGBM ranking: 21 features")
    fprint()
    fprint(f"6 Variants (ONLY DTE differs):")
    for vname, vcfg in VARIANT_CONFIGS.items():
        fprint(f"  {vname}: {vcfg['desc']}")
    fprint()

    # 1. Download data
    close, high, low = download_data()

    # 2. Load regime predictions
    regime_series = load_regime_predictions()

    # 3. Pre-compute ATR (21d)
    atr_dict = compute_atr_series(high, low, close, period=21)

    # 4. Generate V6 rebalance dates (weekly)
    rebal_dates = generate_rebal_dates(close, V6_REBAL_FREQ)
    fprint(f"\nRebalance dates: {len(rebal_dates)} "
           f"({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

    spy_close = close["SPY"]

    # ── SIMULATE ALL 6 VARIANTS ──
    # NOTE: Unlike spread_width study, each DTE variant needs its OWN LGBM rankings
    # because the forward return target horizon changes with DTE. A 7-day forward
    # return produces different sector rankings than a 42-day forward return.
    all_results = {}
    all_imp_dfs = {}

    for vname, vcfg in VARIANT_CONFIGS.items():
        dte = vcfg["dte"]
        fprint(f"\n{'=' * 100}")
        fprint(f"VARIANT {vname}: {vcfg['desc']}")
        fprint(f"{'=' * 100}")

        # Build DTE-specific rankings (forward return target = dte days)
        fprint(f"  Building LGBM rankings with DTE={dte} forward target...")
        records = build_feature_records(
            close, high, low, rebal_dates, ALL_21_FEATURES, regime_series, dte
        )
        rankings, imp_df = walk_forward_lgbm_rank(records, ALL_21_FEATURES, vname)
        all_imp_dfs[vname] = imp_df

        if not rankings:
            fprint(f"  No rankings produced for DTE={dte}. Skipping.")
            all_results[vname] = {"description": vcfg["desc"], "dte": dte,
                                  "n_trades": 0, "verdict": "INSUFFICIENT DATA"}
            continue

        # Simulate
        trades, final_eq = simulate_trades(
            vname, rankings, close, high, low, atr_dict, dte
        )

        if not trades or len(trades) < 10:
            fprint(f"  Only {len(trades) if trades else 0} trades, skipping validation")
            all_results[vname] = {"description": vcfg["desc"], "dte": dte,
                                  "n_trades": len(trades) if trades else 0,
                                  "verdict": "INSUFFICIENT DATA"}
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
        bull_pnl = sum(t["pnl"] for t in bull_trades)
        bear_pnl = sum(t["pnl"] for t in bear_trades)
        bull_wr = sum(1 for t in bull_trades if t["win"]) / max(len(bull_trades), 1) * 100
        bear_wr = sum(1 for t in bear_trades if t["win"]) / max(len(bear_trades), 1) * 100
        fprint(f"  Direction breakdown:")
        fprint(f"    Bull: {len(bull_trades)} trades, WR {bull_wr:.1f}%, PnL ${bull_pnl:.0f}")
        fprint(f"    Bear: {len(bear_trades)} trades, WR {bear_wr:.1f}%, PnL ${bear_pnl:.0f}")

        # DTE economics analysis
        econ = analyze_dte_economics(trades, dte)
        fprint(f"  DTE economics (DTE={dte}):")
        fprint(f"    Avg entry cost: ${econ.get('avg_entry_cost', 0):.2f} | "
               f"Avg max profit: ${econ.get('avg_max_profit', 0):.2f}")
        fprint(f"    Cost/Profit ratio: {econ.get('cost_profit_ratio', 0):.3f}")
        fprint(f"    Avg days held: {econ.get('avg_days_held', 0):.1f}")
        fprint(f"    Near-max-profit trades: {econ.get('pct_near_max_profit', 0):.1f}% of winners")
        fprint(f"    Avg win: ${econ.get('avg_win_pnl', 0):.2f} | "
               f"Avg loss: ${econ.get('avg_loss_pnl', 0):.2f} | "
               f"Win/Loss ratio: {econ.get('win_loss_ratio', 0):.2f}")
        fprint(f"    Capital efficiency (PnL/$deployed/day x1000): "
               f"{econ.get('avg_cap_efficiency', 0):.4f}")

        # Regime breakdown
        regime_analysis = analyze_regime_performance(trades)
        if regime_analysis:
            fprint(f"  Regime breakdown:")
            for regime, stats_r in regime_analysis.items():
                if "note" in stats_r:
                    fprint(f"    {regime}: {stats_r['n']} trades ({stats_r['note']})")
                else:
                    fprint(f"    {regime}: {stats_r['n']} trades, WR {stats_r['wr']:.1%}, "
                           f"PF {stats_r['pf']:.2f}, total ${stats_r['total_pnl']:.0f}")

        # Random baseline
        random_sharpes = random_baseline_test(
            rankings, close, high, low, atr_dict, dte
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
            "description": vcfg["desc"],
            "dte": dte,
            **r_dict,
            "random_sharpes": [round(s, 3) for s in random_sharpes],
            "random_mean_sharpe": round(mean_random, 3),
            "bull_trades": len(bull_trades),
            "bear_trades": len(bear_trades),
            "bull_pnl": round(bull_pnl, 2),
            "bear_pnl": round(bear_pnl, 2),
            "bull_wr": round(bull_wr, 1),
            "bear_wr": round(bear_wr, 1),
            "dte_economics": econ,
            "regime_analysis": regime_analysis,
        }

    # ══════════════════════════════════════════════════════════
    # SUMMARY COMPARISON TABLE
    # ══════════════════════════════════════════════════════════

    fprint(f"\n{'=' * 140}")
    fprint("DTE SENSITIVITY — SUMMARY COMPARISON")
    fprint(f"{'=' * 140}")
    fprint(f"{'Variant':<15} {'DTE':>4} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} "
           f"{'PF':>6} {'MaxDD':>7} {'Gates':>6} {'Final$':>8} {'AvgCost':>8} "
           f"{'CapEff':>8} {'RandSh':>7}")
    fprint("-" * 140)

    baseline_sharpe = all_results.get("C_dte21", {}).get("sharpe", 0)

    for vname in VARIANT_CONFIGS:
        r = all_results.get(vname)
        if not r or r.get("verdict") == "INSUFFICIENT DATA":
            fprint(f"  {vname:<15} {VARIANT_CONFIGS[vname]['dte']:>4} — INSUFFICIENT DATA —")
            continue
        _wr = r.get("win_rate", r.get("wr", 0))
        _pf = r.get("profit_factor", r.get("pf", 0))
        _mdd = r.get("max_dd", 0)
        _gp = r.get("gates_passed", 0)
        _gt = r.get("gates_total", r.get("total_gates", 5))
        _avg_cost = r.get("dte_economics", {}).get("avg_entry_cost", 0)
        _cap_eff = r.get("dte_economics", {}).get("avg_cap_efficiency", 0)
        fprint(f"  {vname:<15} {r['dte']:>4} {r.get('n_trades', 0):>5} {r['sharpe']:>7.2f} "
               f"{r['sortino']:>8.2f} {_wr*100:>5.1f}% {_pf:>5.2f} "
               f"{_mdd*100:>6.1f}% {_gp}/{_gt} "
               f"${r.get('final_equity', 0):>7,.0f} ${_avg_cost:>6.0f} "
               f"{_cap_eff:>7.4f} {r['random_mean_sharpe']:>7.2f}")

    # ── SHARPE DELTA ANALYSIS vs BASELINE (C_dte21) ──
    fprint(f"\n{'=' * 100}")
    fprint("SHARPE DELTA vs BASELINE (C_dte21 — current production DTE)")
    fprint(f"{'=' * 100}")
    fprint(f"  Baseline (C_dte21) Sharpe: {baseline_sharpe:.2f}")
    fprint()

    deltas = []
    for vname in VARIANT_CONFIGS:
        if vname == "C_dte21":
            continue
        r = all_results.get(vname)
        if not r or "sharpe" not in r:
            continue
        delta = r["sharpe"] - baseline_sharpe
        pct_delta = delta / abs(baseline_sharpe) * 100 if baseline_sharpe != 0 else 0
        deltas.append((vname, delta, pct_delta, r["sharpe"], r["description"], r["dte"]))

    deltas.sort(key=lambda x: x[1], reverse=True)

    max_abs_delta = max(abs(d[1]) for d in deltas) if deltas else 1
    for vname, delta, pct_delta, sharpe, desc, dte in deltas:
        direction = "+" if delta >= 0 else ""
        bar = "*" * int(abs(delta) / max_abs_delta * 30) if max_abs_delta > 0 else ""
        sign = "BETTER" if delta > 0.1 else "WORSE" if delta < -0.1 else "SIMILAR"
        fprint(f"  {vname:<15} DTE={dte:>2}  Sharpe {sharpe:>5.2f}  "
               f"delta {direction}{delta:>+5.2f} ({direction}{pct_delta:>+5.1f}%)  "
               f"{sign}  {bar}")

    # ── COST vs PERFORMANCE vs CAPITAL EFFICIENCY ──
    fprint(f"\n{'=' * 110}")
    fprint("COST vs PERFORMANCE vs CAPITAL EFFICIENCY")
    fprint(f"{'=' * 110}")
    fprint(f"{'Variant':<15} {'DTE':>4} {'AvgCost':>8} {'AvgMaxP':>8} {'Cost/P':>7} "
           f"{'NearMax%':>8} {'AvgWin':>8} {'AvgLoss':>8} {'W/L':>6} {'CapEff':>8} {'TotPnL':>9}")
    fprint("-" * 110)

    for vname in VARIANT_CONFIGS:
        r = all_results.get(vname)
        if not r or not r.get("dte_economics"):
            continue
        e = r["dte_economics"]
        fprint(f"  {vname:<15} {r['dte']:>4} ${e.get('avg_entry_cost', 0):>6.0f} "
               f"${e.get('avg_max_profit', 0):>6.0f} "
               f"{e.get('cost_profit_ratio', 0):>6.3f} "
               f"{e.get('pct_near_max_profit', 0):>7.1f}% "
               f"${e.get('avg_win_pnl', 0):>6.0f} "
               f"${e.get('avg_loss_pnl', 0):>6.0f} "
               f"{e.get('win_loss_ratio', 0):>5.2f} "
               f"{e.get('avg_cap_efficiency', 0):>7.4f} "
               f"${e.get('total_pnl', 0):>7.0f}")

    # ── THETA DECAY ANALYSIS ──
    fprint(f"\n{'=' * 100}")
    fprint("THETA DECAY ANALYSIS — Entry Cost Scaling with DTE")
    fprint(f"{'=' * 100}")
    fprint("  Theory: Options cost more with longer DTE (more time value).")
    fprint("  But cost scales ~sqrt(T), not linearly, so longer DTE may be")
    fprint("  more capital-efficient per day of exposure.")
    fprint()

    dte_cost_pairs = []
    for vname in VARIANT_CONFIGS:
        r = all_results.get(vname)
        if r and r.get("dte_economics"):
            dte_val = r["dte"]
            avg_cost = r["dte_economics"].get("avg_entry_cost", 0)
            cost_per_day = avg_cost / max(dte_val, 1)
            cost_per_sqrt_day = avg_cost / max(np.sqrt(dte_val), 1)
            dte_cost_pairs.append((vname, dte_val, avg_cost, cost_per_day, cost_per_sqrt_day))
            fprint(f"  {vname:<15} DTE={dte_val:>2}: "
                   f"Avg cost ${avg_cost:>6.0f} | "
                   f"$/day ${cost_per_day:>5.1f} | "
                   f"$/sqrt(day) ${cost_per_sqrt_day:>5.1f}")

    if len(dte_cost_pairs) >= 3:
        dtes = [x[1] for x in dte_cost_pairs]
        costs = [x[2] for x in dte_cost_pairs]
        slope, intercept, r_val, p_val, _ = stats.linregress(dtes, costs)
        fprint(f"\n  Cost-DTE linear regression:")
        fprint(f"    Slope: ${slope:.2f}/day of DTE | R-squared: {r_val**2:.3f} | p={p_val:.4f}")

        # sqrt fit
        sqrt_dtes = [np.sqrt(d) for d in dtes]
        slope_s, _, r_val_s, p_val_s, _ = stats.linregress(sqrt_dtes, costs)
        fprint(f"    Sqrt fit R-squared: {r_val_s**2:.3f} (theory predicts ~sqrt scaling)")

    # ── KEY FINDINGS ──
    fprint(f"\n{'=' * 100}")
    fprint("KEY FINDINGS")
    fprint(f"{'=' * 100}")

    if deltas:
        best = max(deltas, key=lambda x: x[1])
        worst = min(deltas, key=lambda x: x[1])
        fprint(f"  Best variant:  {best[0]} DTE={best[5]} (Sharpe {best[3]:.2f}, "
               f"delta {best[1]:+.2f}) — {best[4]}")
        fprint(f"  Worst variant: {worst[0]} DTE={worst[5]} (Sharpe {worst[3]:.2f}, "
               f"delta {worst[1]:+.2f}) — {worst[4]}")

    # Trend analysis: does longer DTE = better or worse?
    dte_variants = [(vname, all_results[vname]) for vname in VARIANT_CONFIGS
                    if vname in all_results and "sharpe" in all_results[vname]]
    if len(dte_variants) >= 3:
        dte_vals = [all_results[vn]["dte"] for vn, _ in dte_variants]
        dte_sharpes = [r["sharpe"] for _, r in dte_variants]
        slope, _, r_val, p_val, _ = stats.linregress(dte_vals, dte_sharpes)
        fprint(f"\n  DTE-Sharpe trend (all variants):")
        fprint(f"    Slope: {slope:.4f} Sharpe per day of DTE")
        fprint(f"    R-squared: {r_val**2:.3f}, p-value: {p_val:.4f}")
        if slope > 0.005 and p_val < 0.1:
            fprint(f"    FINDING: Longer DTE IMPROVES Sharpe (statistically suggestive)")
        elif slope < -0.005 and p_val < 0.1:
            fprint(f"    FINDING: Shorter DTE IMPROVES Sharpe (statistically suggestive)")
        else:
            fprint(f"    FINDING: No statistically significant DTE-Sharpe relationship")

        # Check for non-linear (inverted-U) pattern
        if len(dte_vals) >= 4:
            # Quadratic fit
            coeffs = np.polyfit(dte_vals, dte_sharpes, 2)
            if coeffs[0] < -0.0001:
                optimal_dte_cont = -coeffs[1] / (2 * coeffs[0])
                if 5 <= optimal_dte_cont <= 50:
                    fprint(f"    FINDING: Quadratic fit suggests optimal DTE ~ {optimal_dte_cont:.0f} days "
                           f"(inverted-U pattern)")
                    fprint(f"    Quadratic: {coeffs[0]:.6f}*DTE^2 + {coeffs[1]:.4f}*DTE + {coeffs[2]:.2f}")

    # Capital efficiency comparison
    cap_eff_variants = [(vname, all_results[vname]) for vname in VARIANT_CONFIGS
                        if vname in all_results and all_results[vname].get("dte_economics")]
    if cap_eff_variants:
        best_cap_eff = max(cap_eff_variants,
                           key=lambda x: x[1]["dte_economics"].get("avg_cap_efficiency", 0))
        fprint(f"\n  Most capital-efficient variant: {best_cap_eff[0]} "
               f"DTE={best_cap_eff[1]['dte']} "
               f"(CapEff={best_cap_eff[1]['dte_economics']['avg_cap_efficiency']:.4f})")

    # Feature importance comparison across DTEs
    fprint(f"\n  LGBM feature importance stability across DTEs:")
    top3_by_variant = {}
    for vname, imp_df in all_imp_dfs.items():
        if imp_df is not None:
            top3 = list(imp_df.head(3)["feature"])
            top3_by_variant[vname] = top3
            fprint(f"    {vname}: {', '.join(top3)}")

    if len(top3_by_variant) >= 3:
        # Check if same features dominate across DTEs
        all_top3 = [f for feats in top3_by_variant.values() for f in feats]
        from collections import Counter
        feat_counts = Counter(all_top3)
        stable_feats = [f for f, c in feat_counts.items() if c >= len(top3_by_variant) * 0.5]
        if stable_feats:
            fprint(f"    Stable top features across DTEs: {', '.join(stable_feats)}")
        else:
            fprint(f"    WARNING: Feature importance NOT stable across DTEs — "
                   f"different horizons use different signals")

    # Research conclusion
    fprint(f"\n{'=' * 100}")
    fprint("RESEARCH CONCLUSION")
    fprint(f"{'=' * 100}")

    valid_results = {k: v for k, v in all_results.items() if "sharpe" in v}
    if valid_results:
        best_overall = max(valid_results.items(), key=lambda x: x[1]["sharpe"])
        fprint(f"  Optimal DTE: {best_overall[0]} — {best_overall[1]['description']}")
        fprint(f"  Sharpe: {best_overall[1]['sharpe']:.2f} | "
               f"WR: {best_overall[1].get('win_rate', best_overall[1].get('wr', 0)):.1%} | "
               f"PF: {best_overall[1].get('profit_factor', best_overall[1].get('pf', 0)):.2f} | "
               f"Gates: {best_overall[1].get('gates_passed', 0)}/{best_overall[1].get('gates_total', 5)}")

        if best_overall[0] == "C_dte21":
            fprint(f"\n  Current DTE=21 is ALREADY OPTIMAL. No change needed.")
        else:
            delta_vs_baseline = best_overall[1]["sharpe"] - baseline_sharpe
            fprint(f"\n  Improvement vs current DTE=21 baseline: {delta_vs_baseline:+.2f} Sharpe")
            if delta_vs_baseline > 0.3 and best_overall[1].get("gates_passed", 0) >= 4:
                fprint(f"  RECOMMENDATION: Switch to DTE={best_overall[1]['dte']} for V7.")
            elif delta_vs_baseline > 0:
                fprint(f"  RECOMMENDATION: Marginal improvement. Test with more data before switching.")
            else:
                fprint(f"  RECOMMENDATION: Keep current DTE=21.")

    # Save results
    results_path = OUTPUT_DIR / "dte_sensitivity_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    # MLflow logging
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"dte_sensitivity_{t0.strftime('%Y%m%d_%H%M')}"):
                for vname, r in all_results.items():
                    if "sharpe" not in r:
                        continue
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
                    mlflow.log_metric(f"{prefix}_dte", r.get("dte", 0))

                    # DTE economics
                    econ = r.get("dte_economics", {})
                    if econ:
                        mlflow.log_metric(f"{prefix}_avg_entry_cost", econ.get("avg_entry_cost", 0))
                        mlflow.log_metric(f"{prefix}_cost_profit_ratio", econ.get("cost_profit_ratio", 0))
                        mlflow.log_metric(f"{prefix}_win_loss_ratio", econ.get("win_loss_ratio", 0))
                        mlflow.log_metric(f"{prefix}_cap_efficiency", econ.get("avg_cap_efficiency", 0))
                        mlflow.log_metric(f"{prefix}_total_pnl", econ.get("total_pnl", 0))

                    # Delta vs baseline
                    if vname != "C_dte21":
                        delta = r.get("sharpe", 0) - baseline_sharpe
                        mlflow.log_metric(f"{prefix}_sharpe_delta_vs_baseline", delta)

                mlflow.log_params({
                    "experiment_type": "dte_sensitivity",
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
                    "n_variants": 6,
                    "wf_train_periods": WF_TRAIN_PERIODS,
                    "n_features": 21,
                    "baseline_variant": "C_dte21",
                    "dte_range": "7-42",
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
