#!/usr/bin/env python3
"""
Dual Signal Portfolio v1 — V9 (Momentum) + Earnings Sector Rotation Combination
================================================================================

Tests what happens when we combine two complementary sector rotation strategies:
  - V9 (momentum-based, 17 features): Sharpe ~3.0, momentum/quality features
  - Earnings (14 features): Sharpe ~2.85, earnings calendar features
  - Low rank correlation (0.26), low top-3 overlap (1.32/3) => COMPLEMENTARY

4 Variants:
  A: V9 only (baseline) — 17 momentum features, top 3 bull + bottom 3 bear
  B: Earnings only (baseline) — 14 earnings features, top 3 bull + bottom 3 bear
  C: Signal average — Average normalized LGBM rank scores, pick top/bottom 3
  D: Expanded universe — Union of V9 top 3 + Earnings top 3 (up to 6), dedup

Config: DTE=21, 3% OTM, adaptive width max($3, 3%), $645 capital, weekly Friday
        rebalance, $2.60 commission, 15% haircut, hold-to-expiry,
        VIX adaptive (>=20: bulls only, <20: bulls+bears)

Walk-forward: 25-period sliding window LGBM ranking (each model independent)
Validation: 5-gate adversarial for each variant
MLflow experiment: dual_signal_portfolio_v1
"""

import json
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

_builtin_print = print


def fprint(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()


# -- Path auto-detect (Jupiter vs Neptune) --
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
    fprint("research.tools not found -- using inline implementations")
    COMMISSION_RT_SPREAD = 2.60
    DEFAULT_HAIRCUT = 0.15

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
            return _ValidationResult(0, 0, 0, 1.0, 0, 0, len(trades) if trades else 0,
                                     initial_capital, 0, 5, "INSUFFICIENT DATA")
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

        gates = 0
        if sharpe > 0.5: gates += 1
        if wr > 0.45: gates += 1
        if pf > 1.0: gates += 1
        if mdd > -0.30: gates += 1
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


# ================================================================
# CONFIG
# ================================================================

BASE = Path(_BASE_PATH)
OUTPUT_DIR = BASE / "output" / "growth_research" / "dual_signal_portfolio_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]

CAP = 645.0
DTE = 21
OTM_PCT = 0.03
SPREAD_PCT = 3.0  # adaptive width: max($3, 3%)
TOP_K = 3
WF_TRAIN_PERIODS = 25
REBAL_FREQ = "W-FRI"
VIX_THRESHOLD = 20.0
COST_WIDTH_MAX = 0.50

# Regime predictions path
REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"

# Sector constituent tickers (for earnings detection)
SECTOR_CONSTITUENTS = {
    'XLK': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'CRM', 'ADBE', 'CSCO', 'ACN', 'ORCL', 'IBM',
            'INTC', 'AMD', 'QCOM', 'TXN', 'AMAT', 'INTU', 'NOW', 'ADI', 'LRCX', 'SNPS'],
    'XLF': ['BRK-B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'SPGI', 'BLK',
            'C', 'AXP', 'SCHW', 'CB', 'MMC', 'PGR', 'ICE', 'CME', 'AON', 'MET'],
    'XLE': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'PXD', 'OXY',
            'WMB', 'HES', 'DVN', 'HAL', 'FANG', 'BKR', 'TRGP', 'KMI', 'OKE', 'CTRA'],
    'XLV': ['UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'PFE', 'TMO', 'ABT', 'DHR', 'AMGN',
            'BMY', 'ISRG', 'SYK', 'VRTX', 'GILD', 'MDT', 'REGN', 'CI', 'ELV', 'ZTS'],
    'XLI': ['GE', 'CAT', 'HON', 'UNP', 'UPS', 'RTX', 'BA', 'DE', 'LMT', 'ADP',
            'MMM', 'FDX', 'GD', 'NSC', 'NOC', 'WM', 'CSX', 'ITW', 'EMR', 'ETN'],
    'XLY': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TJX', 'BKNG', 'CMG',
            'F', 'GM', 'ORLY', 'AZO', 'ROST', 'DHI', 'LEN', 'MAR', 'HLT', 'YUM'],
    'XLP': ['PG', 'PEP', 'KO', 'COST', 'WMT', 'PM', 'MDLZ', 'MO', 'CL', 'KMB',
            'GIS', 'SJM', 'K', 'HSY', 'STZ', 'KHC', 'TAP', 'CAG', 'CPB', 'HRL'],
    'XLU': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'EXC', 'XEL', 'ED', 'WEC',
            'AWK', 'DTE', 'AEE', 'CMS', 'PPL', 'FE', 'ETR', 'CEG', 'PEG', 'EVRG'],
    'XLB': ['LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NEM', 'NUE', 'VMC', 'MLM', 'DOW',
            'DD', 'PPG', 'IFF', 'CE', 'ALB', 'EMN', 'PKG', 'IP', 'CF', 'MOS'],
    'XLRE': ['PLD', 'AMT', 'CCI', 'EQIX', 'PSA', 'SPG', 'O', 'WELL', 'DLR', 'AVB',
             'EQR', 'ARE', 'VTR', 'MAA', 'UDR', 'KIM', 'REG', 'HST', 'CPT', 'BXP'],
    'XLC': ['META', 'GOOGL', 'GOOG', 'DIS', 'CMCSA', 'NFLX', 'T', 'VZ', 'TMUS', 'CHTR',
            'EA', 'TTWO', 'WBD', 'OMC', 'IPG', 'FOXA', 'FOX', 'PARA', 'LYV', 'MTCH'],
}

# Approximate market cap weights for top constituents
CONSTITUENT_WEIGHTS = {}
for _sector, _tickers in SECTOR_CONSTITUENTS.items():
    weights = {}
    for i, tk in enumerate(_tickers):
        if i < 3:
            weights[tk] = 3.0
        elif i < 7:
            weights[tk] = 2.0
        else:
            weights[tk] = 1.0
    total = sum(weights.values())
    CONSTITUENT_WEIGHTS[_sector] = {tk: w / total for tk, w in weights.items()}

# ================================================================
# FEATURE DEFINITIONS
# ================================================================

# V9 momentum features (17)
V9_FEATURES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "sharpe_63d", "pct_52w_high", "mom_accel",
    "pct_pos_months_12m", "sortino_63d", "calmar_1y", "up_capture",
    "trend_r2_63d", "trend_slope_63d",
    "sector_spy_beta_63d", "cross_sector_dispersion",
]
assert len(V9_FEATURES) == 17

# Earnings features (14) = 8 base + 6 interaction
EARNINGS_BASE_FEATURES = [
    "earnings_pct_reporting_2w",
    "earnings_avg_surprise",
    "earnings_post_drift",
    "earnings_days_to_heavy_week",
    "earnings_recent_surprise_quality",
    "earnings_beat_rate_1m",
    "earnings_vol_impact",
    "sector_earnings_cycle_position",
]

EARNINGS_INTERACTION_FEATURES = [
    "earn_density_x_sector_mom_21d",
    "earn_surprise_x_sector_vol",
    "earn_drift_x_sector_ret_63d",
    "earn_beat_x_sector_sharpe",
    "earn_cycle_x_sector_ret_5d",
    "earn_vol_impact_x_dispersion",
]

EARNINGS_FEATURES = EARNINGS_BASE_FEATURES + EARNINGS_INTERACTION_FEATURES
assert len(EARNINGS_FEATURES) == 14

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "dual_signal_portfolio_v1"

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
# DATA DOWNLOAD
# ================================================================

def download_data():
    """Download sector ETFs + macro data."""
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


def download_constituent_data():
    """Download constituent stock data for earnings detection."""
    import yfinance as yf
    all_constituents = set()
    for tickers in SECTOR_CONSTITUENTS.values():
        all_constituents.update(tickers)
    all_constituents = sorted(list(all_constituents))
    fprint(f"Downloading {len(all_constituents)} constituent stocks for earnings detection...")
    chunk_size = 50
    all_close = {}
    all_volume = {}
    for i in range(0, len(all_constituents), chunk_size):
        chunk = all_constituents[i:i + chunk_size]
        fprint(f"  Chunk {i // chunk_size + 1}: {len(chunk)} tickers...")
        try:
            raw = yf.download(chunk, start="2009-01-01", progress=False, auto_adjust=True)
            if isinstance(raw.columns, pd.MultiIndex):
                c = raw["Close"]
                v = raw["Volume"]
                if isinstance(c.columns, pd.MultiIndex):
                    c.columns = c.columns.get_level_values(-1)
                    v.columns = v.columns.get_level_values(-1)
            else:
                c = raw[["Close"]]
                v = raw[["Volume"]]
            for col in c.columns:
                all_close[col] = c[col].ffill()
                all_volume[col] = v[col].ffill()
        except Exception as e:
            fprint(f"    Warning: chunk download failed: {e}")
            continue
    const_close = pd.DataFrame(all_close)
    const_volume = pd.DataFrame(all_volume)
    fprint(f"  Constituent data: {len(const_close)} days, {len(const_close.columns)} tickers")
    return const_close, const_volume


# ================================================================
# EARNINGS EVENT DETECTION (Volume + Gap Proxy)
# ================================================================

def detect_earnings_events(const_close, const_volume, vol_mult=3.0, gap_thresh=0.03):
    """Detect earnings-like events via volume spike + price gap."""
    fprint(f"Detecting earnings events (vol_mult={vol_mult}, gap_thresh={gap_thresh})...")
    earnings_events = {}
    total_events = 0
    for ticker in const_close.columns:
        c = const_close[ticker].dropna()
        v = const_volume[ticker].dropna() if ticker in const_volume.columns else None
        if v is None or len(c) < 60 or len(v) < 60:
            continue
        common = c.index.intersection(v.index)
        c = c.loc[common]; v = v.loc[common]
        rets = c.pct_change()
        vol_median = v.rolling(20, min_periods=10).median()
        vol_spike = v > (vol_median * vol_mult)
        gap_condition = rets.abs() > gap_thresh
        events = vol_spike & gap_condition
        event_dates = events[events].index
        if len(event_dates) > 0:
            surprise = rets.loc[event_dates]
            earnings_events[ticker] = surprise
            total_events += len(event_dates)
    fprint(f"  Detected {total_events} earnings-like events across {len(earnings_events)} tickers")
    return earnings_events


# ================================================================
# REGIME
# ================================================================

def load_regime_predictions():
    if not REGIME_FILE.exists():
        fprint("WARNING: Regime file not found, using VIX proxy")
        return None
    data = np.load(REGIME_FILE, allow_pickle=True)
    dates = pd.to_datetime(data["dates"])
    scores = data["regime_scores"]
    regime_series = pd.Series(scores, index=dates, name="regime_score")
    regime_series = regime_series[~regime_series.index.duplicated(keep="last")]
    fprint(f"Regime predictions loaded: {len(regime_series)} days")
    return regime_series


def get_regime_score_at(regime_series, dt):
    if regime_series is None:
        return 0.5
    if dt in regime_series.index:
        return float(regime_series.loc[dt])
    nearest = regime_series.index[regime_series.index.get_indexer([dt], method="ffill")]
    if len(nearest) > 0:
        return float(regime_series.loc[nearest[0]])
    return 0.5


# ================================================================
# V9 MOMENTUM FEATURES
# ================================================================

def compute_v9_features(px, spy_slice):
    """Compute 17 V9 momentum/quality features."""
    if len(px) < 260:
        return None
    f = {}
    for lb, nm in [(5, "ret_5d"), (10, "ret_10d"), (21, "ret_21d"),
                   (63, "ret_63d"), (126, "ret_126d"), (252, "ret_252d")]:
        f[nm] = float(px.iloc[-1] / px.iloc[-lb] - 1) if len(px) > lb else 0.0
    rets = px.pct_change().dropna()
    r63 = rets.iloc[-63:]
    f["sharpe_63d"] = float(r63.mean() / (r63.std() + 1e-10) * np.sqrt(252)) if len(r63) > 10 else 0.0
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


# ================================================================
# EARNINGS FEATURES
# ================================================================

def build_original_earnings_features(sector_ticker, dt, idx, close_df,
                                     earnings_events, const_close):
    """Compute 4 original earnings features."""
    f = {}
    constituents = SECTOR_CONSTITUENTS.get(sector_ticker, [])
    if not constituents:
        return {feat: 0.0 for feat in EARNINGS_BASE_FEATURES[:4]}

    current_date = close_df.index[idx]
    lookback_2w = close_df.index[max(0, idx - 10):idx + 1]
    lookback_63d = close_df.index[max(0, idx - 63):idx + 1]

    # 1. Pct reporting in trailing 2 weeks
    n_reported_2w = 0
    for tk in constituents:
        if tk in earnings_events:
            events = earnings_events[tk]
            recent = events.index.intersection(lookback_2w)
            if len(recent) > 0:
                n_reported_2w += 1
    f["earnings_pct_reporting_2w"] = n_reported_2w / len(constituents)

    # 2. Average surprise over trailing 63d
    surprises = []
    for tk in constituents:
        if tk in earnings_events:
            events = earnings_events[tk]
            recent = events.index.intersection(lookback_63d)
            if len(recent) > 0:
                surprises.extend(events.loc[recent].values)
    f["earnings_avg_surprise"] = float(np.mean(surprises)) if surprises else 0.0

    # 3. Post-earnings drift (PEAD) over trailing 63d
    drifts = []
    for tk in constituents:
        if tk in earnings_events and tk in const_close.columns:
            events = earnings_events[tk]
            recent_events = events.index.intersection(lookback_63d)
            for edt in recent_events:
                if edt in const_close.index:
                    edt_idx = const_close.index.get_loc(edt)
                    end_idx = min(edt_idx + 5, len(const_close) - 1)
                    if end_idx > edt_idx and end_idx <= idx:
                        drift = float(const_close[tk].iloc[end_idx] /
                                      const_close[tk].iloc[edt_idx] - 1)
                        drifts.append(drift)
    f["earnings_post_drift"] = float(np.mean(drifts)) if drifts else 0.0

    # 4. Days to heavy earnings week
    all_event_weeks = []
    for tk in constituents:
        if tk in earnings_events:
            for edt in earnings_events[tk].index:
                all_event_weeks.append(edt.isocalendar()[1])
    if all_event_weeks:
        week_counts = pd.Series(all_event_weeks).value_counts()
        peak_weeks = week_counts.head(4).index.tolist()
        current_week = current_date.isocalendar()[1]
        min_dist = 52
        for pw in peak_weeks:
            dist = (pw - current_week) % 52
            min_dist = min(min_dist, dist)
        f["earnings_days_to_heavy_week"] = min_dist / 26.0
    else:
        f["earnings_days_to_heavy_week"] = 0.5

    return f


def build_new_earnings_features(sector_ticker, dt, idx, close_df,
                                earnings_events, const_close):
    """Compute 4 new earnings features."""
    f = {}
    constituents = SECTOR_CONSTITUENTS.get(sector_ticker, [])
    weights = CONSTITUENT_WEIGHTS.get(sector_ticker, {})
    if not constituents:
        return {feat: 0.0 for feat in EARNINGS_BASE_FEATURES[4:]}

    current_date = close_df.index[idx]
    lookback_21d = close_df.index[max(0, idx - 21):idx + 1]
    lookback_63d = close_df.index[max(0, idx - 63):idx + 1]

    # 1. earnings_recent_surprise_quality
    weighted_beats = 0.0
    total_weight = 0.0
    for tk in constituents:
        if tk in earnings_events:
            events = earnings_events[tk]
            recent = events.index.intersection(lookback_63d)
            if len(recent) > 0:
                w = weights.get(tk, 1.0 / len(constituents))
                avg_surprise = float(events.loc[recent].mean())
                quality = avg_surprise if avg_surprise > 0 else avg_surprise * 2
                weighted_beats += quality * w
                total_weight += w
    f["earnings_recent_surprise_quality"] = weighted_beats / (total_weight + 1e-10)

    # 2. earnings_beat_rate_1m
    n_beat = 0
    n_reported = 0
    for tk in constituents:
        if tk in earnings_events:
            events = earnings_events[tk]
            recent = events.index.intersection(lookback_21d)
            if len(recent) > 0:
                n_reported += 1
                if events.loc[recent].mean() > 0:
                    n_beat += 1
    f["earnings_beat_rate_1m"] = n_beat / max(n_reported, 1)

    # 3. earnings_vol_impact
    abs_impacts = []
    for tk in constituents:
        if tk in earnings_events:
            events = earnings_events[tk]
            recent = events.index.intersection(lookback_63d)
            if len(recent) > 0:
                abs_impacts.extend(np.abs(events.loc[recent].values))
    f["earnings_vol_impact"] = float(np.mean(abs_impacts)) if abs_impacts else 0.0

    # 4. sector_earnings_cycle_position
    n_reported_63d = 0
    for tk in constituents:
        if tk in earnings_events:
            events = earnings_events[tk]
            recent = events.index.intersection(lookback_63d)
            if len(recent) > 0:
                n_reported_63d += 1
    f["sector_earnings_cycle_position"] = min(n_reported_63d / len(constituents), 1.0)

    return f


def compute_earnings_momentum_interactions(earn_feats, sector_ticker, idx, close_df):
    """Compute 6 interaction features between earnings and sector momentum."""
    f = {}
    sector_px = close_df[sector_ticker].iloc[:idx + 1].dropna() if sector_ticker in close_df.columns else None
    if sector_px is None or len(sector_px) < 63:
        return {feat: 0.0 for feat in EARNINGS_INTERACTION_FEATURES}

    rets = sector_px.pct_change().dropna()
    ret_5d = float(sector_px.iloc[-1] / sector_px.iloc[-5] - 1) if len(sector_px) > 5 else 0.0
    ret_21d = float(sector_px.iloc[-1] / sector_px.iloc[-21] - 1) if len(sector_px) > 21 else 0.0
    ret_63d = float(sector_px.iloc[-1] / sector_px.iloc[-63] - 1) if len(sector_px) > 63 else 0.0
    vol_21d = float(rets.iloc[-21:].std() * np.sqrt(252)) if len(rets) > 21 else 0.2
    r63 = rets.iloc[-63:]
    sharpe_63d = float(r63.mean() / (r63.std() + 1e-10) * np.sqrt(252)) if len(r63) > 10 else 0.0

    sector_cols = [c for c in SECTORS if c in close_df.columns]
    if len(sector_cols) > 3:
        sector_rets = close_df[sector_cols].iloc[:idx + 1].pct_change()
        daily_disp = sector_rets.std(axis=1)
        dispersion = float(daily_disp.rolling(21).mean().iloc[-1]) if len(daily_disp) > 21 else 0.01
    else:
        dispersion = 0.01

    density = earn_feats.get("earnings_pct_reporting_2w", 0)
    surprise = earn_feats.get("earnings_avg_surprise", 0)
    drift = earn_feats.get("earnings_post_drift", 0)
    beat_rate = earn_feats.get("earnings_beat_rate_1m", 0)
    cycle = earn_feats.get("sector_earnings_cycle_position", 0.5)
    vol_impact = earn_feats.get("earnings_vol_impact", 0)

    f["earn_density_x_sector_mom_21d"] = density * ret_21d
    f["earn_surprise_x_sector_vol"] = surprise * vol_21d
    f["earn_drift_x_sector_ret_63d"] = drift * ret_63d
    f["earn_beat_x_sector_sharpe"] = beat_rate * sharpe_63d
    f["earn_cycle_x_sector_ret_5d"] = cycle * ret_5d
    f["earn_vol_impact_x_dispersion"] = vol_impact * dispersion

    return f


# ================================================================
# ATR
# ================================================================

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


# ================================================================
# FEATURE RECORD BUILDING
# ================================================================

def build_v9_records(close, high, low, rebal_dates, regime_series):
    """Build V9 feature records (17 momentum features)."""
    fprint("  Building V9 (momentum) records...")
    records = []
    sector_cols = [c for c in SECTORS if c in close.columns]
    spy = close["SPY"]

    for dt in rebal_dates:
        idx = close.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
            continue
        rscore = get_regime_score_at(regime_series, dt)
        if rscore <= 0.4:
            continue
        for tk in sector_cols:
            px = close[tk].iloc[:idx + 1].dropna()
            v9 = compute_v9_features(px, spy.iloc[:idx + 1])
            if not v9:
                continue
            cross = compute_cross_asset_features(tk, idx, close)
            fi = min(idx + DTE, len(close) - 1)
            if fi <= idx:
                continue
            fwd_ret = float(close[tk].iloc[fi] / close[tk].iloc[idx] - 1)
            rec = {**v9, **cross, "date": dt, "ticker": tk, "fwd_ret": fwd_ret}
            records.append(rec)

    df = pd.DataFrame(records)
    for c in V9_FEATURES:
        if c not in df.columns:
            df[c] = 0.0
    df[V9_FEATURES] = df[V9_FEATURES].fillna(0.0)
    fprint(f"    {len(df)} records, {len(df['date'].unique())} dates")
    return df


def build_earnings_records(close, high, low, rebal_dates, regime_series,
                           earnings_events, const_close):
    """Build earnings feature records (14 earnings features)."""
    fprint("  Building Earnings records...")
    records = []
    sector_cols = [c for c in SECTORS if c in close.columns]

    for dt in rebal_dates:
        idx = close.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
            continue
        rscore = get_regime_score_at(regime_series, dt)
        if rscore <= 0.4:
            continue
        for tk in sector_cols:
            px = close[tk].iloc[:idx + 1].dropna()
            if len(px) < 260:
                continue

            # Build earnings features
            orig_earn = build_original_earnings_features(
                tk, dt, idx, close, earnings_events, const_close)
            new_earn = build_new_earnings_features(
                tk, dt, idx, close, earnings_events, const_close)
            all_earn = {**orig_earn, **new_earn}
            interactions = compute_earnings_momentum_interactions(all_earn, tk, idx, close)

            fi = min(idx + DTE, len(close) - 1)
            if fi <= idx:
                continue
            fwd_ret = float(close[tk].iloc[fi] / close[tk].iloc[idx] - 1)

            rec = {**all_earn, **interactions, "date": dt, "ticker": tk, "fwd_ret": fwd_ret}
            records.append(rec)

    df = pd.DataFrame(records)
    for c in EARNINGS_FEATURES:
        if c not in df.columns:
            df[c] = 0.0
    df[EARNINGS_FEATURES] = df[EARNINGS_FEATURES].fillna(0.0)
    fprint(f"    {len(df)} records, {len(df['date'].unique())} dates")
    return df


# ================================================================
# WALK-FORWARD LGBM RANKING
# ================================================================

def walk_forward_lgbm_rank(df, feature_cols, model_name):
    """Walk-forward LGBM ranking with sliding window."""
    import lightgbm as lgb

    if len(df) < 100:
        fprint(f"    {model_name}: Insufficient data ({len(df)} records)")
        return {}, None

    df = df.copy()
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
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1,
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

    fprint(f"    {model_name}: {len(rankings)} ranking dates, {n_models} models trained")
    return rankings, imp_df


# ================================================================
# COMBINED RANKINGS (Variants C and D)
# ================================================================

def combine_rankings_average(v9_rankings, earn_rankings):
    """Variant C: Average normalized rank scores, pick top/bottom 3."""
    combined = {}
    common_dates = set(v9_rankings.keys()) & set(earn_rankings.keys())

    for dt in common_dates:
        v9_scores = v9_rankings[dt]
        earn_scores = earn_rankings[dt]
        common_tickers = set(v9_scores.keys()) & set(earn_scores.keys())

        if len(common_tickers) < 3:
            continue

        # Normalize each to [0, 1]
        v9_vals = {tk: v9_scores[tk] for tk in common_tickers}
        earn_vals = {tk: earn_scores[tk] for tk in common_tickers}

        v9_min, v9_max = min(v9_vals.values()), max(v9_vals.values())
        earn_min, earn_max = min(earn_vals.values()), max(earn_vals.values())

        v9_range = v9_max - v9_min if v9_max > v9_min else 1.0
        earn_range = earn_max - earn_min if earn_max > earn_min else 1.0

        avg_scores = {}
        for tk in common_tickers:
            v9_norm = (v9_vals[tk] - v9_min) / v9_range
            earn_norm = (earn_vals[tk] - earn_min) / earn_range
            avg_scores[tk] = (v9_norm + earn_norm) / 2.0

        combined[dt] = avg_scores

    fprint(f"    Signal Average: {len(combined)} combined dates")
    return combined


def combine_rankings_expanded(v9_rankings, earn_rankings):
    """Variant D: Union of V9 top 3 + Earnings top 3, deduplicated."""
    # Returns rankings with a special flag to indicate expanded universe
    combined = {}
    common_dates = set(v9_rankings.keys()) & set(earn_rankings.keys())

    for dt in common_dates:
        v9_scores = v9_rankings[dt]
        earn_scores = earn_rankings[dt]

        v9_ranked = sorted(v9_scores.items(), key=lambda x: x[1], reverse=True)
        earn_ranked = sorted(earn_scores.items(), key=lambda x: x[1], reverse=True)

        # Bull: union of top 3 from each
        v9_top = [t for t, _ in v9_ranked[:TOP_K]]
        earn_top = [t for t, _ in earn_ranked[:TOP_K]]
        bull_union = list(dict.fromkeys(v9_top + earn_top))  # preserves order, deduplicates

        # Bear: union of bottom 3 from each
        v9_bottom = [t for t, _ in v9_ranked[-TOP_K:]]
        earn_bottom = [t for t, _ in earn_ranked[-TOP_K:]]
        bear_union = list(dict.fromkeys(v9_bottom + earn_bottom))

        # Create synthetic scores: rank them by average of the two models' normalized scores
        all_tickers = set(v9_scores.keys()) | set(earn_scores.keys())
        # Normalize
        v9_vals = list(v9_scores.values())
        earn_vals = list(earn_scores.values())
        v9_min, v9_max = (min(v9_vals), max(v9_vals)) if v9_vals else (0, 1)
        earn_min, earn_max = (min(earn_vals), max(earn_vals)) if earn_vals else (0, 1)
        v9_range = v9_max - v9_min if v9_max > v9_min else 1.0
        earn_range = earn_max - earn_min if earn_max > earn_min else 1.0

        avg_scores = {}
        for tk in all_tickers:
            v9_s = (v9_scores.get(tk, v9_min) - v9_min) / v9_range
            earn_s = (earn_scores.get(tk, earn_min) - earn_min) / earn_range
            avg_scores[tk] = (v9_s + earn_s) / 2.0

        combined[dt] = {
            "scores": avg_scores,
            "bull_union": bull_union,
            "bear_union": bear_union,
        }

    fprint(f"    Expanded Universe: {len(combined)} combined dates")
    return combined


# ================================================================
# TRADE EXECUTION
# ================================================================

def compute_strikes(S, direction, otm_pct):
    """Compute strike prices with adaptive width max($3, 3%)."""
    if direction == "bull":
        K1 = round(S * (1 + otm_pct), 2)
        pct_w = K1 * 0.03
        w = max(3.0, pct_w)
        K2 = round(K1 + w, 2)
    else:
        K2 = round(S * (1 - otm_pct), 2)
        pct_w = K2 * 0.03
        w = max(3.0, pct_w)
        K1 = round(K2 - w, 2)
    if K2 <= K1:
        K2 = K1 + 0.50
    return K1, K2


def execute_single_trade(tk, dt, direction, close, atr_dict, cv, equity, max_pos):
    """Execute a single spread trade. Returns trade dict or None."""
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

    K1, K2 = compute_strikes(S, direction, OTM_PCT)

    try:
        if direction == "bull":
            entry_cost_ps, _ = price_bull_call_spread(S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=cv)
        else:
            entry_cost_ps, _ = price_bear_put_spread(S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=cv)
    except Exception:
        return None

    if entry_cost_ps is None or entry_cost_ps <= 0:
        return None

    # Cost/width filter
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

    spy = close["SPY"]
    sv = float(spy.loc[dt])
    se = float(spy.iloc[ei]) if ei < len(spy) else sv

    return {
        "pnl": round(pnl, 2),
        "entry_cost_ps": round(entry_cost_ps, 4),
        "total_cost": round(total_cost, 2),
        "K1": K1, "K2": K2,
        "S_entry": round(S, 2), "S_exit": round(Se, 2),
        "intrinsic": round(intrinsic, 4),
        "spread_width": round(spread_width, 2),
        "cost_width_ratio": round(cwr, 3),
        "entry_date": str(dt.date()),
        "exit_date": str(close.index[ei].date()),
        "ticker": tk,
        "direction": direction,
        "vix": round(cv, 1),
        "win": pnl > 0,
        "regime": "bull" if se >= sv else "bear",
    }


# ================================================================
# SIMULATION — STANDARD VARIANTS (A, B, C)
# ================================================================

def simulate_standard(name, rankings, close, atr_dict, n_bulls=TOP_K, n_bears=TOP_K):
    """Simulate with standard top-K / bottom-K picks."""
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

        # VIX adaptive: >= 20 bulls only, < 20 bulls + bears
        if cv >= VIX_THRESHOLD:
            trade_mode = "bull_only"
        else:
            trade_mode = "pairs"

        ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_asc = sorted(scores.items(), key=lambda x: x[1])
        bull_picks = [t for t, _ in ranked_desc[:n_bulls]]
        bear_picks = [t for t, _ in ranked_asc[:n_bears]] if trade_mode == "pairs" else []

        if trade_mode == "pairs":
            max_pos = min(100, equity / (n_bulls + n_bears))
        else:
            max_pos = min(200, equity / n_bulls)

        if max_pos < 30:
            continue

        for direction, picks in [("bull", bull_picks), ("bear", bear_picks)]:
            for tk in picks:
                result = execute_single_trade(
                    tk, dt, direction, close, atr_dict, cv, equity, max_pos)
                if result is not None:
                    equity += result["pnl"]
                    result["trade_mode"] = trade_mode
                    trades.append(result)

    return trades, equity


# ================================================================
# SIMULATION — EXPANDED UNIVERSE (Variant D)
# ================================================================

def simulate_expanded(name, combined_rankings, close, atr_dict):
    """Simulate with expanded universe: union of V9 + Earnings picks."""
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    equity = CAP
    trades = []

    for dt in sorted(combined_rankings.keys()):
        if dt not in spy.index:
            continue
        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0
        combo = combined_rankings[dt]
        bull_picks = combo["bull_union"]
        bear_picks = combo["bear_union"]

        # VIX adaptive
        if cv >= VIX_THRESHOLD:
            bear_picks = []

        n_bulls = len(bull_picks)
        n_bears = len(bear_picks)
        n_total = n_bulls + n_bears

        if n_total == 0:
            continue

        # Capital split: $645 / n_positions per position
        max_pos = min(200, equity / max(n_total, 1))
        if max_pos < 30:
            continue

        for direction, picks in [("bull", bull_picks), ("bear", bear_picks)]:
            for tk in picks:
                result = execute_single_trade(
                    tk, dt, direction, close, atr_dict, cv, equity, max_pos)
                if result is not None:
                    equity += result["pnl"]
                    result["trade_mode"] = f"expanded_{n_bulls}b_{n_bears}s"
                    trades.append(result)

    return trades, equity


# ================================================================
# ANALYSIS HELPERS
# ================================================================

def compute_metrics(trades, label):
    """Compute comprehensive metrics for a set of trades."""
    if not trades:
        return {"label": label, "n_trades": 0}

    pnls = [t["pnl"] for t in trades]
    equity = [CAP]
    for p in pnls:
        equity.append(equity[-1] + p)
    equity = np.array(equity)

    rets = np.diff(equity) / equity[:-1]
    rets = rets[np.isfinite(rets)]

    wins = sum(1 for p in pnls if p > 0)
    losses_sum = abs(sum(p for p in pnls if p <= 0))
    wins_sum = sum(p for p in pnls if p > 0)

    sharpe = float(np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(52))
    down_rets = rets[rets < 0]
    sortino = float(np.mean(rets) / (np.std(down_rets) + 1e-10) * np.sqrt(52)) if len(down_rets) > 0 else 0
    wr = wins / len(pnls)
    pf = wins_sum / (losses_sum + 1e-10)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / (peak + 1e-10)
    mdd = float(np.min(dd))
    years = len(pnls) / 52.0
    cagr = float((equity[-1] / CAP) ** (1 / max(years, 0.5)) - 1) if equity[-1] > 0 else 0
    calmar = cagr / (abs(mdd) + 1e-10)

    return {
        "label": label,
        "n_trades": len(pnls),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(pf, 3),
        "win_rate": round(wr, 4),
        "max_dd": round(mdd, 4),
        "calmar": round(calmar, 3),
        "total_pnl": round(sum(pnls), 2),
        "final_equity": round(equity[-1], 2),
        "cagr": round(cagr, 4),
        "avg_pnl": round(np.mean(pnls), 2),
        "median_pnl": round(np.median(pnls), 2),
    }


def rank_correlation_analysis(v9_rankings, earn_rankings):
    """Measure rank correlation and overlap between V9 and Earnings rankings."""
    from scipy.stats import spearmanr

    correlations = []
    overlaps_top3 = []
    overlaps_bottom3 = []
    common_dates = sorted(set(v9_rankings.keys()) & set(earn_rankings.keys()))

    for dt in common_dates:
        v9 = v9_rankings[dt]
        earn = earn_rankings[dt]
        common_tk = sorted(set(v9.keys()) & set(earn.keys()))
        if len(common_tk) < 5:
            continue

        v9_vals = [v9[tk] for tk in common_tk]
        earn_vals = [earn[tk] for tk in common_tk]
        corr, _ = spearmanr(v9_vals, earn_vals)
        correlations.append(corr)

        v9_top3 = set(sorted(common_tk, key=lambda x: v9[x], reverse=True)[:3])
        earn_top3 = set(sorted(common_tk, key=lambda x: earn[x], reverse=True)[:3])
        overlaps_top3.append(len(v9_top3 & earn_top3))

        v9_bot3 = set(sorted(common_tk, key=lambda x: v9[x])[:3])
        earn_bot3 = set(sorted(common_tk, key=lambda x: earn[x])[:3])
        overlaps_bottom3.append(len(v9_bot3 & earn_bot3))

    result = {
        "n_common_dates": len(common_dates),
        "mean_rank_corr": round(float(np.mean(correlations)), 4) if correlations else 0,
        "median_rank_corr": round(float(np.median(correlations)), 4) if correlations else 0,
        "mean_top3_overlap": round(float(np.mean(overlaps_top3)), 3) if overlaps_top3 else 0,
        "mean_bottom3_overlap": round(float(np.mean(overlaps_bottom3)), 3) if overlaps_bottom3 else 0,
    }
    return result


def regime_breakdown(trades):
    """Break down metrics by bull/bear regime."""
    bull_trades = [t for t in trades if t.get("regime") == "bull"]
    bear_trades = [t for t in trades if t.get("regime") == "bear"]
    return {
        "bull": compute_metrics(bull_trades, "bull_regime"),
        "bear": compute_metrics(bear_trades, "bear_regime"),
    }


def yearly_breakdown(trades):
    """Break down metrics by year."""
    by_year = defaultdict(list)
    for t in trades:
        yr = t["entry_date"][:4]
        by_year[yr].append(t)
    return {yr: compute_metrics(ts, yr) for yr, ts in sorted(by_year.items())}


# ================================================================
# MAIN
# ================================================================

def main():
    t0 = datetime.now()
    fprint("=" * 100)
    fprint(f"DUAL SIGNAL PORTFOLIO v1 — V9 (Momentum) + Earnings Combination Study")
    fprint(f"Started: {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 100)
    fprint(f"Config: DTE={DTE}, OTM={OTM_PCT:.0%}, adaptive width max($3,3%)")
    fprint(f"Capital: ${CAP:.0f} | Commission: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"V9 features: {len(V9_FEATURES)} | Earnings features: {len(EARNINGS_FEATURES)}")
    fprint(f"Variants: A (V9 only), B (Earnings only), C (Signal Average), D (Expanded Universe)")
    fprint()

    # ── Step 1: Download data ──
    fprint("=" * 80)
    fprint("STEP 1: DATA DOWNLOAD")
    fprint("=" * 80)
    close, high, low, volume = download_data()
    const_close, const_volume = download_constituent_data()

    # ── Step 2: Earnings detection ──
    fprint("\n" + "=" * 80)
    fprint("STEP 2: EARNINGS EVENT DETECTION")
    fprint("=" * 80)
    earnings_events = detect_earnings_events(const_close, const_volume)

    # ── Step 3: Setup ──
    regime_series = load_regime_predictions()
    atr_dict = compute_atr_series(high, low, close)

    rebal_dates = pd.DatetimeIndex(
        close.index.to_series().resample(REBAL_FREQ).last().dropna().values
    )
    fprint(f"\nRebalance dates: {len(rebal_dates)} ({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

    # ── Step 4: Build feature records ──
    fprint("\n" + "=" * 80)
    fprint("STEP 4: BUILDING FEATURE RECORDS")
    fprint("=" * 80)
    v9_records = build_v9_records(close, high, low, rebal_dates, regime_series)
    earn_records = build_earnings_records(close, high, low, rebal_dates, regime_series,
                                         earnings_events, const_close)

    # ── Step 5: Train LGBM models independently ──
    fprint("\n" + "=" * 80)
    fprint("STEP 5: WALK-FORWARD LGBM RANKING (INDEPENDENT MODELS)")
    fprint("=" * 80)
    v9_rankings, v9_imp = walk_forward_lgbm_rank(v9_records, V9_FEATURES, "V9_momentum")
    earn_rankings, earn_imp = walk_forward_lgbm_rank(earn_records, EARNINGS_FEATURES, "Earnings")

    if not v9_rankings or not earn_rankings:
        fprint("ERROR: Insufficient rankings to proceed")
        return

    # ── Step 5b: Rank correlation analysis ──
    fprint("\n" + "-" * 60)
    fprint("RANK CORRELATION ANALYSIS (V9 vs Earnings)")
    fprint("-" * 60)
    corr_analysis = rank_correlation_analysis(v9_rankings, earn_rankings)
    for k, v in corr_analysis.items():
        fprint(f"  {k}: {v}")

    # ── Step 6: Build combined rankings ──
    fprint("\n" + "=" * 80)
    fprint("STEP 6: BUILDING COMBINED RANKINGS")
    fprint("=" * 80)
    avg_rankings = combine_rankings_average(v9_rankings, earn_rankings)
    expanded_rankings = combine_rankings_expanded(v9_rankings, earn_rankings)

    # ── Step 7: Simulate all variants ──
    fprint("\n" + "=" * 80)
    fprint("STEP 7: TRADE SIMULATION")
    fprint("=" * 80)

    results = {}

    fprint("\n  Variant A: V9 Only (baseline)...")
    trades_a, eq_a = simulate_standard("V9_only", v9_rankings, close, atr_dict)
    results["A_V9_only"] = compute_metrics(trades_a, "A: V9 Only")
    fprint(f"    {len(trades_a)} trades, ${CAP:.0f} -> ${eq_a:,.0f}, "
           f"Sharpe {results['A_V9_only']['sharpe']:.2f}")

    fprint("\n  Variant B: Earnings Only...")
    trades_b, eq_b = simulate_standard("Earnings_only", earn_rankings, close, atr_dict)
    results["B_Earnings_only"] = compute_metrics(trades_b, "B: Earnings Only")
    fprint(f"    {len(trades_b)} trades, ${CAP:.0f} -> ${eq_b:,.0f}, "
           f"Sharpe {results['B_Earnings_only']['sharpe']:.2f}")

    fprint("\n  Variant C: Signal Average...")
    trades_c, eq_c = simulate_standard("Signal_average", avg_rankings, close, atr_dict)
    results["C_Signal_average"] = compute_metrics(trades_c, "C: Signal Average")
    fprint(f"    {len(trades_c)} trades, ${CAP:.0f} -> ${eq_c:,.0f}, "
           f"Sharpe {results['C_Signal_average']['sharpe']:.2f}")

    fprint("\n  Variant D: Expanded Universe...")
    trades_d, eq_d = simulate_expanded("Expanded_universe", expanded_rankings, close, atr_dict)
    results["D_Expanded_universe"] = compute_metrics(trades_d, "D: Expanded Universe")
    fprint(f"    {len(trades_d)} trades, ${CAP:.0f} -> ${eq_d:,.0f}, "
           f"Sharpe {results['D_Expanded_universe']['sharpe']:.2f}")

    # ── Step 8: Adversarial validation ──
    fprint("\n" + "=" * 80)
    fprint("STEP 8: 5-GATE ADVERSARIAL VALIDATION")
    fprint("=" * 80)

    all_trades = {"A": trades_a, "B": trades_b, "C": trades_c, "D": trades_d}
    validation_results = {}
    for variant, trades in all_trades.items():
        vr = validate_trades(trades, initial_capital=CAP, strategy_name=variant)
        vr.print_summary()
        validation_results[variant] = vr.to_dict()
        results[f"{variant}_validation"] = vr.to_dict()

    # ── Step 9: Regime breakdown ──
    fprint("\n" + "=" * 80)
    fprint("STEP 9: REGIME BREAKDOWN")
    fprint("=" * 80)
    regime_results = {}
    for variant, trades in all_trades.items():
        rb = regime_breakdown(trades)
        regime_results[variant] = rb
        label = results[f"{variant}_{'V9_only' if variant == 'A' else 'Earnings_only' if variant == 'B' else 'Signal_average' if variant == 'C' else 'Expanded_universe'}"]["label"]
        fprint(f"\n  {label}:")
        for regime, metrics in rb.items():
            fprint(f"    {regime}: Sharpe {metrics.get('sharpe', 0):.2f}, "
                   f"WR {metrics.get('win_rate', 0):.1%}, "
                   f"PF {metrics.get('profit_factor', 0):.2f}, "
                   f"Trades {metrics.get('n_trades', 0)}")

    # ── Step 10: Yearly breakdown ──
    fprint("\n" + "=" * 80)
    fprint("STEP 10: YEARLY BREAKDOWN")
    fprint("=" * 80)
    yearly_results = {}
    for variant, trades in all_trades.items():
        yb = yearly_breakdown(trades)
        yearly_results[variant] = yb
        key = f"{variant}_{'V9_only' if variant == 'A' else 'Earnings_only' if variant == 'B' else 'Signal_average' if variant == 'C' else 'Expanded_universe'}"
        fprint(f"\n  {results[key]['label']}:")
        for yr, metrics in sorted(yb.items()):
            fprint(f"    {yr}: Sharpe {metrics.get('sharpe', 0):+6.2f}  "
                   f"WR {metrics.get('win_rate', 0):.0%}  "
                   f"PnL ${metrics.get('total_pnl', 0):+8.0f}  "
                   f"Trades {metrics.get('n_trades', 0):3d}")

    # ── Step 11: Feature importance ──
    fprint("\n" + "=" * 80)
    fprint("STEP 11: FEATURE IMPORTANCE")
    fprint("=" * 80)
    if v9_imp is not None:
        fprint("\n  V9 (Momentum) Top 10:")
        for _, row in v9_imp.head(10).iterrows():
            fprint(f"    {row['feature']:35s} {row['importance']:8.1f}")
    if earn_imp is not None:
        fprint("\n  Earnings Top 10:")
        for _, row in earn_imp.head(10).iterrows():
            fprint(f"    {row['feature']:35s} {row['importance']:8.1f}")

    # ── Step 12: Summary comparison ──
    fprint("\n" + "=" * 100)
    fprint("FINAL COMPARISON TABLE")
    fprint("=" * 100)
    fprint(f"{'Variant':<25s} {'Trades':>7s} {'Sharpe':>8s} {'Sortino':>8s} "
           f"{'PF':>7s} {'WR':>7s} {'MaxDD':>8s} {'Calmar':>8s} {'Final$':>10s} {'Gates':>6s}")
    fprint("-" * 100)

    variant_keys = ["A_V9_only", "B_Earnings_only", "C_Signal_average", "D_Expanded_universe"]
    for vk in variant_keys:
        m = results[vk]
        vl = vk[0]  # A, B, C, D
        vv = validation_results.get(vl, {})
        gates_str = f"{vv.get('gates_passed', '?')}/{vv.get('gates_total', '?')}"
        fprint(f"{m['label']:<25s} {m['n_trades']:>7d} {m['sharpe']:>8.2f} {m['sortino']:>8.2f} "
               f"{m['profit_factor']:>7.2f} {m['win_rate']:>6.1%} {m['max_dd']:>7.1%} "
               f"{m['calmar']:>8.2f} {m['final_equity']:>10,.0f} {gates_str:>6s}")

    fprint("\n" + "=" * 100)

    # ── Step 13: Save results ──
    output = {
        "timestamp": t0.isoformat(),
        "config": {
            "dte": DTE, "otm_pct": OTM_PCT, "capital": CAP,
            "commission": COMMISSION_RT_SPREAD, "top_k": TOP_K,
            "wf_train_periods": WF_TRAIN_PERIODS, "vix_threshold": VIX_THRESHOLD,
        },
        "rank_correlation": corr_analysis,
        "results": results,
        "validation": validation_results,
        "regime_breakdown": {k: {rk: rv for rk, rv in v.items()} for k, v in regime_results.items()},
        "yearly_breakdown": {k: {yr: m for yr, m in v.items()} for k, v in yearly_results.items()},
        "v9_feature_importance": v9_imp.to_dict('records') if v9_imp is not None else None,
        "earnings_feature_importance": earn_imp.to_dict('records') if earn_imp is not None else None,
    }

    out_path = OUTPUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    fprint(f"\nResults saved to {out_path}")

    # ── Step 14: MLflow logging ──
    if MLFLOW_OK:
        fprint("\nLogging to MLflow...")
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name="dual_signal_portfolio_v1"):
                mlflow.log_params({
                    "dte": DTE, "otm_pct": OTM_PCT, "capital": CAP,
                    "commission": COMMISSION_RT_SPREAD, "top_k": TOP_K,
                    "v9_features": len(V9_FEATURES),
                    "earnings_features": len(EARNINGS_FEATURES),
                    "wf_train_periods": WF_TRAIN_PERIODS,
                    "vix_threshold": VIX_THRESHOLD,
                })
                for vk in variant_keys:
                    m = results[vk]
                    prefix = vk.split("_", 1)[0]
                    for metric_name in ["sharpe", "sortino", "profit_factor", "win_rate",
                                        "max_dd", "calmar", "n_trades", "final_equity"]:
                        if metric_name in m:
                            mlflow.log_metric(f"{prefix}_{metric_name}", m[metric_name])
                # Rank correlation
                mlflow.log_metric("rank_corr_mean", corr_analysis["mean_rank_corr"])
                mlflow.log_metric("top3_overlap_mean", corr_analysis["mean_top3_overlap"])

                mlflow.log_artifact(str(out_path))
            fprint("MLflow logging complete")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nTotal runtime: {elapsed / 60:.1f} minutes")
    fprint("DONE")


if __name__ == "__main__":
    main()
