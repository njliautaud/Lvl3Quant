#!/usr/bin/env python3
"""
Earnings Sector Standalone v1 — Standalone Earnings-Only Sector Rotation Strategy
===================================================================================

CONTEXT: earnings_sector_features_v1.py found that using ONLY 4 earnings features
(earnings_pct_reporting_2w, earnings_avg_surprise, earnings_post_drift,
earnings_days_to_heavy_week) achieves Sharpe 2.62 in sector rotation — +7% better
than the full 21-feature production model. This script explores that finding deeper.

TESTS:
  1. 4-feature LGBM ranking as standalone strategy (reproduce Sharpe 2.62)
  2. Extended earnings features (4 new candidates):
     - earnings_recent_surprise_quality (cap-weighted surprise quality)
     - earnings_beat_rate_1m (% of sector beating in last month)
     - earnings_vol_impact (avg abs return on earnings day)
     - sector_earnings_cycle_position (early/mid/late in reporting season)
  3. Walk-forward cross-validation (V6 config: DTE=21, 2-3% OTM, weekly, $645)
  4. Adversarial validation (5-gate)
  5. Regime analysis (high vs low VIX performance)
  6. Monthly seasonality (earnings season vs non-earnings months)
  7. Correlation with production strategy picks

Variants:
  A: Original 4 earnings features (baseline reproduction)
  B: 4 original + 4 new earnings features (8 total)
  C: Best subset from B (selected by importance)
  D: 8 earnings + interaction features with sector momentum
  E: Production 21-feature model (for correlation comparison)

Walk-forward: 25-period sliding window, weekly W-FRI rebalance
Trade config: V6 (weekly, 2% OTM, bull+pairs, $645 capital, DTE=21)
Validation: 5-gate adversarial + regime + seasonality analysis
MLflow experiment: 'earnings_sector_standalone_v1'
Output: output/growth_research/earnings_sector_standalone_v1/
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


# -- Config --
BASE = Path(_BASE_PATH)
OUTPUT_DIR = BASE / "output" / "growth_research" / "earnings_sector_standalone_v1"
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

# LGBM walk-forward -- 25 periods
WF_TRAIN_PERIODS = 25

# V6 config
V6_REBAL_FREQ = "W-FRI"
V6_OTM_PCT = 0.02
V6_PAIRS = True
V6_MAX_POS_BULL = 200
V6_MAX_POS_PAIR_LEG = 100

# Sector constituent tickers (top holdings for earnings proxy)
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

# Approximate market cap weights for top constituents (for cap-weighted surprise)
# These are rough relative weights within each sector for the top holdings
CONSTITUENT_WEIGHTS = {}
for _sector, _tickers in SECTOR_CONSTITUENTS.items():
    # Top 5 get higher weight, rest equal -- rough approximation
    n = len(_tickers)
    weights = {}
    for i, tk in enumerate(_tickers):
        if i < 3:
            weights[tk] = 3.0  # mega-cap top 3
        elif i < 7:
            weights[tk] = 2.0  # large-cap next 4
        else:
            weights[tk] = 1.0  # rest
    total = sum(weights.values())
    CONSTITUENT_WEIGHTS[_sector] = {tk: w / total for tk, w in weights.items()}

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "earnings_sector_standalone_v1"

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
# FEATURE SET DEFINITIONS
# ================================================================

# Original 4 earnings features (from v1 experiment)
ORIGINAL_EARNINGS_FEATURES = [
    "earnings_pct_reporting_2w",
    "earnings_avg_surprise",
    "earnings_post_drift",
    "earnings_days_to_heavy_week",
]

# New candidate earnings features
NEW_EARNINGS_FEATURES = [
    "earnings_recent_surprise_quality",   # cap-weighted surprise quality
    "earnings_beat_rate_1m",              # % of sector beating estimates in last month
    "earnings_vol_impact",               # avg abs return on earnings day
    "sector_earnings_cycle_position",    # early/mid/late in reporting season
]

ALL_8_EARNINGS = ORIGINAL_EARNINGS_FEATURES + NEW_EARNINGS_FEATURES

# Earnings-momentum interaction features
EARNINGS_MOMENTUM_INTERACTIONS = [
    "earn_density_x_sector_mom_21d",
    "earn_surprise_x_sector_vol",
    "earn_drift_x_sector_ret_63d",
    "earn_beat_x_sector_sharpe",
    "earn_cycle_x_sector_ret_5d",
    "earn_vol_impact_x_dispersion",
]

# Production 21 features (for correlation comparison only)
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


# ================================================================
# DATA DOWNLOAD (reused from v1)
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

    close = close.ffill()
    high = high.ffill()
    low = low.ffill()
    volume = volume.ffill()

    rename_map = {"^VIX": "VIX", "^VIX3M": "VIX3M"}
    close = close.rename(columns=rename_map)
    high = high.rename(columns=rename_map)
    low = low.rename(columns=rename_map)
    volume = volume.rename(columns=rename_map)

    needed = ["SPY", "VIX"]
    for t in needed:
        if t not in close.columns:
            raise ValueError(f"Missing critical ticker: {t}")

    fprint(f"Data: {len(close)} days, columns: {list(close.columns)}")
    fprint(f"Date range: {close.index[0].date()} to {close.index[-1].date()}")

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
    fprint(f"  Date range: {const_close.index[0].date()} to {const_close.index[-1].date()}")

    return const_close, const_volume


# ================================================================
# EARNINGS EVENT DETECTION (Volume + Gap Proxy) -- reused from v1
# ================================================================

def detect_earnings_events(const_close, const_volume, vol_mult=3.0, gap_thresh=0.03):
    """
    Detect earnings-like events for each constituent stock.
    Volume >= vol_mult * rolling 20d median AND abs gap > gap_thresh.
    Returns dict: {ticker: pd.Series of event dates with surprise magnitude}
    """
    fprint(f"Detecting earnings events (vol_mult={vol_mult}, gap_thresh={gap_thresh})...")

    earnings_events = {}
    total_events = 0

    for ticker in const_close.columns:
        c = const_close[ticker].dropna()
        v = const_volume[ticker].dropna() if ticker in const_volume.columns else None

        if v is None or len(c) < 60 or len(v) < 60:
            continue

        common = c.index.intersection(v.index)
        c = c.loc[common]
        v = v.loc[common]

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

    fprint(f"  Detected {total_events} earnings-like events across "
           f"{len(earnings_events)} tickers")

    if earnings_events:
        avg_events = total_events / len(earnings_events)
        n_years = (const_close.index[-1] - const_close.index[0]).days / 365
        fprint(f"  Avg {avg_events / max(n_years, 1):.1f} events/year/ticker "
               f"(expect ~4 for quarterly earnings)")

    return earnings_events


# ================================================================
# ORIGINAL 4 EARNINGS FEATURES (from v1)
# ================================================================

def build_original_earnings_features(sector_ticker, dt, idx, close_df,
                                     earnings_events, const_close):
    """Compute the original 4 earnings features for a sector at a given date."""
    f = {}
    constituents = SECTOR_CONSTITUENTS.get(sector_ticker, [])

    if not constituents:
        return {feat: 0.0 for feat in ORIGINAL_EARNINGS_FEATURES}

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

    # 4. Days to heavy earnings week (seasonal proxy)
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


# ================================================================
# NEW EARNINGS FEATURES (4 additional candidates)
# ================================================================

def build_new_earnings_features(sector_ticker, dt, idx, close_df,
                                earnings_events, const_close):
    """
    Compute 4 NEW earnings features:
      1. earnings_recent_surprise_quality -- cap-weighted surprise quality
      2. earnings_beat_rate_1m -- % of sector constituents beating in last month
      3. earnings_vol_impact -- avg absolute return on earnings day
      4. sector_earnings_cycle_position -- where in reporting season (0-1)
    """
    f = {}
    constituents = SECTOR_CONSTITUENTS.get(sector_ticker, [])
    weights = CONSTITUENT_WEIGHTS.get(sector_ticker, {})

    if not constituents:
        return {feat: 0.0 for feat in NEW_EARNINGS_FEATURES}

    current_date = close_df.index[idx]
    lookback_21d = close_df.index[max(0, idx - 21):idx + 1]
    lookback_63d = close_df.index[max(0, idx - 63):idx + 1]

    # 1. earnings_recent_surprise_quality: cap-weighted positive surprise ratio
    # Quality = weighted average of (surprise > 0) for recent earnings
    weighted_beats = 0.0
    total_weight = 0.0
    for tk in constituents:
        if tk in earnings_events:
            events = earnings_events[tk]
            recent = events.index.intersection(lookback_63d)
            if len(recent) > 0:
                w = weights.get(tk, 1.0 / len(constituents))
                # Quality: positive surprise magnitude, penalize negative
                avg_surprise = float(events.loc[recent].mean())
                quality = avg_surprise if avg_surprise > 0 else avg_surprise * 2  # penalize misses
                weighted_beats += quality * w
                total_weight += w
    f["earnings_recent_surprise_quality"] = weighted_beats / (total_weight + 1e-10)

    # 2. earnings_beat_rate_1m: fraction of constituents with positive surprise in last month
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

    # 3. earnings_vol_impact: avg absolute return on earnings days (trailing 63d)
    abs_impacts = []
    for tk in constituents:
        if tk in earnings_events:
            events = earnings_events[tk]
            recent = events.index.intersection(lookback_63d)
            if len(recent) > 0:
                abs_impacts.extend(np.abs(events.loc[recent].values))
    f["earnings_vol_impact"] = float(np.mean(abs_impacts)) if abs_impacts else 0.0

    # 4. sector_earnings_cycle_position: where in reporting season
    # Compute from trailing 63d: what fraction of the sector's expected quarterly
    # earnings have already reported? 0 = pre-season, 0.5 = mid-season, 1 = post
    n_reported_63d = 0
    for tk in constituents:
        if tk in earnings_events:
            events = earnings_events[tk]
            recent = events.index.intersection(lookback_63d)
            if len(recent) > 0:
                n_reported_63d += 1

    # Expected: ~100% of constituents report each quarter
    # If 63d window captures one quarter, expect all to report
    expected_reporters = len(constituents)
    cycle_pos = min(n_reported_63d / expected_reporters, 1.0)

    # Adjust by recency: are we in early (< 30%), mid (30-70%), or late (> 70%) reporting
    f["sector_earnings_cycle_position"] = cycle_pos

    return f


# ================================================================
# EARNINGS-MOMENTUM INTERACTION FEATURES
# ================================================================

def compute_earnings_momentum_interactions(earn_feats, sector_ticker, idx, close_df):
    """Compute interaction features between earnings and sector momentum."""
    f = {}
    sector_px = close_df[sector_ticker].iloc[:idx + 1].dropna() if sector_ticker in close_df.columns else None

    if sector_px is None or len(sector_px) < 63:
        return {feat: 0.0 for feat in EARNINGS_MOMENTUM_INTERACTIONS}

    rets = sector_px.pct_change().dropna()

    # Sector momentum/vol features for interactions
    ret_5d = float(sector_px.iloc[-1] / sector_px.iloc[-5] - 1) if len(sector_px) > 5 else 0.0
    ret_21d = float(sector_px.iloc[-1] / sector_px.iloc[-21] - 1) if len(sector_px) > 21 else 0.0
    ret_63d = float(sector_px.iloc[-1] / sector_px.iloc[-63] - 1) if len(sector_px) > 63 else 0.0
    vol_21d = float(rets.iloc[-21:].std() * np.sqrt(252)) if len(rets) > 21 else 0.2
    r63 = rets.iloc[-63:]
    sharpe_63d = float(r63.mean() / (r63.std() + 1e-10) * np.sqrt(252)) if len(r63) > 10 else 0.0

    # Cross-sector dispersion
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
# REGIME + LEGACY FEATURES (reused from v1)
# ================================================================

def load_regime_predictions():
    """Load GRU regime predictions."""
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
    return regime_series


def get_regime_score_at(regime_series, dt):
    """Get regime score at a given date."""
    if regime_series is None:
        return 0.5
    if dt in regime_series.index:
        return float(regime_series.loc[dt])
    nearest = regime_series.index[regime_series.index.get_indexer([dt], method="ffill")]
    if len(nearest) > 0:
        return float(regime_series.loc[nearest[0]])
    return 0.5


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


# ================================================================
# WALK-FORWARD LGBM RANKING
# ================================================================

def build_feature_records(close, high, low, rebal_dates, feature_cols, regime_series,
                          earnings_events, const_close, include_legacy=False):
    """Build feature + target records for all sectors on all rebal dates."""
    fprint(f"  Building records: {len(rebal_dates)} dates, {len(feature_cols)} features")

    needs_original_earn = any(f in ORIGINAL_EARNINGS_FEATURES for f in feature_cols)
    needs_new_earn = any(f in NEW_EARNINGS_FEATURES for f in feature_cols)
    needs_interactions = any(f in EARNINGS_MOMENTUM_INTERACTIONS for f in feature_cols)
    needs_legacy = include_legacy or any(f in ALL_21_FEATURES for f in feature_cols)

    records = []
    sector_cols = [c for c in SECTORS if c in close.columns]
    spy = close["SPY"]

    for dt in rebal_dates:
        idx = close.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
            continue

        rscore = get_regime_score_at(regime_series, dt)
        if rscore <= REGIME_BULL_THRESHOLD:
            continue

        for tk in sector_cols:
            px = close[tk].iloc[:idx + 1].dropna()
            spy_s = spy.iloc[:idx + 1]

            # Legacy features (needed for production comparison or interactions)
            legacy = {}
            cross_asset = {}
            if needs_legacy:
                legacy = compute_legacy_features(px, spy_s)
                if not legacy and needs_legacy and not needs_original_earn:
                    continue
                if legacy:
                    cross_asset = compute_cross_asset_features(tk, idx, close)

            # Need at least 260 data points even for earnings-only
            if len(px) < 260:
                continue

            # Original earnings features
            earn_feats = {}
            if needs_original_earn and earnings_events is not None:
                earn_feats = build_original_earnings_features(
                    tk, dt, idx, close, earnings_events, const_close
                )

            # New earnings features
            new_earn_feats = {}
            if needs_new_earn and earnings_events is not None:
                new_earn_feats = build_new_earnings_features(
                    tk, dt, idx, close, earnings_events, const_close
                )

            # Interaction features
            interaction_feats = {}
            if needs_interactions and (earn_feats or new_earn_feats):
                all_earn = {**earn_feats, **new_earn_feats}
                interaction_feats = compute_earnings_momentum_interactions(
                    all_earn, tk, idx, close
                )

            # Forward return target
            fi = min(idx + DTE, len(close) - 1)
            if fi <= idx:
                continue
            fwd_ret = float(close[tk].iloc[fi] / close[tk].iloc[idx] - 1)

            rec = {**legacy, **cross_asset, **earn_feats, **new_earn_feats,
                   **interaction_feats,
                   "date": dt, "ticker": tk, "fwd_ret": fwd_ret}
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


# ================================================================
# TRADE SIMULATION (V6 config, reused from v1)
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


def compute_strikes(S, direction, otm_pct, spread_pct):
    """Compute strike prices for a spread."""
    if direction == "bull":
        if otm_pct > 0:
            K1 = round(S * (1 + otm_pct), 2)
            K2 = round(K1 * (1 + spread_pct / 100), 2)
        else:
            K1 = round(S, 2)
            K2 = round(S * (1 + spread_pct / 100), 2)
    else:
        if otm_pct > 0:
            K2 = round(S * (1 - otm_pct), 2)
            K1 = round(K2 * (1 - spread_pct / 100), 2)
        else:
            K1 = round(S * (1 - spread_pct / 100), 2)
            K2 = round(S, 2)

    if K2 <= K1:
        K2 = K1 + 0.50
    return K1, K2


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


def simulate_trades(name, rankings, close, high, low, atr_dict):
    """Simulate trades using V6 config: bull VIX>=20, pairs VIX<20."""
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

        if V6_PAIRS and cv < 20.0:
            trade_mode = "pairs"
        else:
            trade_mode = "bull_only"

        ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_asc = sorted(scores.items(), key=lambda x: x[1])

        bull_picks = [t for t, _ in ranked_desc[:TOP_K]]
        bear_picks = [t for t, _ in ranked_asc[:TOP_K]] if trade_mode == "pairs" else []

        if trade_mode == "pairs":
            max_pos = min(V6_MAX_POS_PAIR_LEG, equity / 6)
        else:
            max_pos = min(V6_MAX_POS_BULL, equity / 3)

        if max_pos < 30:
            continue

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
                trades.append({
                    "pnl": round(pnl, 2),
                    "entry_date": str(dt.date()),
                    "exit_date": str(close.index[ei].date()),
                    "ticker": tk,
                    "regime": "bull" if se >= sv else "bear",
                    "direction": "bull",
                    "vix": round(cv, 1),
                    "win": pnl > 0,
                    "trade_mode": trade_mode,
                    "month": dt.month,
                })

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
                trades.append({
                    "pnl": round(pnl, 2),
                    "entry_date": str(dt.date()),
                    "exit_date": str(close.index[ei].date()),
                    "ticker": tk,
                    "regime": "bull" if se >= sv else "bear",
                    "direction": "bear",
                    "vix": round(cv, 1),
                    "win": pnl > 0,
                    "trade_mode": trade_mode,
                    "month": dt.month,
                })

    return trades, equity


def generate_rebal_dates(close, freq_str):
    """Generate rebalance dates from close index."""
    return pd.DatetimeIndex(
        close.index.to_series().resample(freq_str).last().dropna().values
    )


# ================================================================
# REGIME ANALYSIS
# ================================================================

def regime_analysis(trades, close):
    """Analyze performance in high vs low VIX regimes."""
    fprint(f"\n{'='*80}")
    fprint("REGIME ANALYSIS: High VIX (>=25) vs Low VIX (<25)")
    fprint(f"{'='*80}")

    if not trades or len(trades) < 20:
        fprint("  Insufficient trades for regime analysis")
        return {}

    high_vix_trades = [t for t in trades if t["vix"] >= 25]
    low_vix_trades = [t for t in trades if t["vix"] < 25]
    mid_vix_trades = [t for t in trades if 15 <= t["vix"] < 25]
    extreme_vix_trades = [t for t in trades if t["vix"] >= 30]

    results = {}
    for label, subset in [("VIX >= 25 (High)", high_vix_trades),
                          ("VIX < 25 (Low)", low_vix_trades),
                          ("15 <= VIX < 25 (Mid)", mid_vix_trades),
                          ("VIX >= 30 (Extreme)", extreme_vix_trades)]:
        if len(subset) < 5:
            fprint(f"  {label}: Only {len(subset)} trades, skipping")
            continue

        pnls = [t["pnl"] for t in subset]
        total_pnl = sum(pnls)
        wr = sum(1 for p in pnls if p > 0) / len(pnls)
        wins = sum(p for p in pnls if p > 0)
        losses = abs(sum(p for p in pnls if p <= 0))
        pf = wins / (losses + 1e-10)
        avg_pnl = np.mean(pnls)
        sharpe_proxy = avg_pnl / (np.std(pnls) + 1e-10) * np.sqrt(52)

        fprint(f"  {label}:")
        fprint(f"    Trades: {len(subset)} | WR: {wr:.1%} | PF: {pf:.2f}")
        fprint(f"    Total PnL: ${total_pnl:.0f} | Avg: ${avg_pnl:.2f}")
        fprint(f"    Sharpe proxy: {sharpe_proxy:.2f}")

        results[label] = {
            "n_trades": len(subset),
            "win_rate": round(wr, 4),
            "profit_factor": round(pf, 4),
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(avg_pnl, 2),
            "sharpe_proxy": round(sharpe_proxy, 4),
        }

    # Bull vs bear market regime
    bull_regime = [t for t in trades if t["regime"] == "bull"]
    bear_regime = [t for t in trades if t["regime"] == "bear"]

    fprint(f"\n  Market regime (SPY direction during hold):")
    for label, subset in [("Bull (SPY up)", bull_regime), ("Bear (SPY down)", bear_regime)]:
        if len(subset) < 5:
            continue
        pnls = [t["pnl"] for t in subset]
        wr = sum(1 for p in pnls if p > 0) / len(pnls)
        total = sum(pnls)
        fprint(f"    {label}: {len(subset)} trades, WR {wr:.1%}, PnL ${total:.0f}")

        results[f"market_{label.split('(')[0].strip().lower()}"] = {
            "n_trades": len(subset),
            "win_rate": round(wr, 4),
            "total_pnl": round(total, 2),
        }

    return results


# ================================================================
# MONTHLY SEASONALITY ANALYSIS
# ================================================================

def seasonality_analysis(trades):
    """Analyze monthly seasonality -- earnings seasons vs non-earnings months."""
    fprint(f"\n{'='*80}")
    fprint("MONTHLY SEASONALITY ANALYSIS")
    fprint(f"{'='*80}")

    if not trades or len(trades) < 30:
        fprint("  Insufficient trades for seasonality analysis")
        return {}

    # Earnings season months: Jan, Apr, Jul, Oct (reporting heavy)
    # Non-earnings: all other months
    EARNINGS_MONTHS = {1, 4, 7, 10}  # quarter-end reporting months
    HEAVY_EARNINGS = {1, 2, 4, 5, 7, 8, 10, 11}  # broader earnings windows

    monthly_stats = defaultdict(list)
    for t in trades:
        m = t.get("month", 1)
        monthly_stats[m].append(t["pnl"])

    fprint(f"  {'Month':<12} {'Trades':>6} {'WR':>6} {'PF':>6} {'TotalPnL':>10} {'AvgPnL':>8} {'Season':>10}")
    fprint(f"  {'-'*64}")

    results = {}
    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}

    for m in range(1, 13):
        pnls = monthly_stats.get(m, [])
        if not pnls:
            continue
        wr = sum(1 for p in pnls if p > 0) / len(pnls)
        wins = sum(p for p in pnls if p > 0)
        losses = abs(sum(p for p in pnls if p <= 0))
        pf = wins / (losses + 1e-10)
        total = sum(pnls)
        avg = np.mean(pnls)
        season = "EARNINGS" if m in EARNINGS_MONTHS else "heavy" if m in HEAVY_EARNINGS else "off"

        fprint(f"  {month_names[m]:<12} {len(pnls):>6} {wr:>5.1%} {pf:>5.2f} "
               f"${total:>9.0f} ${avg:>7.2f} {season:>10}")

        results[month_names[m]] = {
            "n_trades": len(pnls),
            "win_rate": round(wr, 4),
            "profit_factor": round(pf, 4),
            "total_pnl": round(total, 2),
            "avg_pnl": round(avg, 2),
            "is_earnings_month": m in EARNINGS_MONTHS,
        }

    # Aggregate: earnings season vs off-season
    earn_pnls = []
    off_pnls = []
    for t in trades:
        if t.get("month", 1) in HEAVY_EARNINGS:
            earn_pnls.append(t["pnl"])
        else:
            off_pnls.append(t["pnl"])

    fprint(f"\n  Earnings-season months (Jan/Feb/Apr/May/Jul/Aug/Oct/Nov):")
    if earn_pnls:
        wr_e = sum(1 for p in earn_pnls if p > 0) / len(earn_pnls)
        fprint(f"    {len(earn_pnls)} trades, WR {wr_e:.1%}, "
               f"PnL ${sum(earn_pnls):.0f}, Avg ${np.mean(earn_pnls):.2f}")
    fprint(f"  Off-season months (Mar/Jun/Sep/Dec):")
    if off_pnls:
        wr_o = sum(1 for p in off_pnls if p > 0) / len(off_pnls)
        fprint(f"    {len(off_pnls)} trades, WR {wr_o:.1%}, "
               f"PnL ${sum(off_pnls):.0f}, Avg ${np.mean(off_pnls):.2f}")

    results["_earnings_season_aggregate"] = {
        "n_trades": len(earn_pnls),
        "total_pnl": round(sum(earn_pnls), 2) if earn_pnls else 0,
        "win_rate": round(sum(1 for p in earn_pnls if p > 0) / max(len(earn_pnls), 1), 4),
    }
    results["_off_season_aggregate"] = {
        "n_trades": len(off_pnls),
        "total_pnl": round(sum(off_pnls), 2) if off_pnls else 0,
        "win_rate": round(sum(1 for p in off_pnls if p > 0) / max(len(off_pnls), 1), 4),
    }

    return results


# ================================================================
# CORRELATION WITH PRODUCTION STRATEGY
# ================================================================

def correlation_analysis(earnings_rankings, production_rankings):
    """Compare earnings-only picks to production 21-feature picks."""
    fprint(f"\n{'='*80}")
    fprint("CORRELATION ANALYSIS: Earnings vs Production Picks")
    fprint(f"{'='*80}")

    common_dates = set(earnings_rankings.keys()) & set(production_rankings.keys())
    if len(common_dates) < 10:
        fprint(f"  Only {len(common_dates)} overlapping dates, insufficient for analysis")
        return {}

    fprint(f"  Overlapping dates: {len(common_dates)}")

    # Rank correlation per date
    rank_correlations = []
    pick_overlaps = []

    for dt in sorted(common_dates):
        e_scores = earnings_rankings[dt]
        p_scores = production_rankings[dt]
        common_tickers = set(e_scores.keys()) & set(p_scores.keys())

        if len(common_tickers) < 5:
            continue

        tickers = sorted(common_tickers)
        e_vals = [e_scores[t] for t in tickers]
        p_vals = [p_scores[t] for t in tickers]

        corr, p_val = stats.spearmanr(e_vals, p_vals)
        rank_correlations.append(corr)

        # Top-3 overlap
        e_top3 = set(sorted(e_scores, key=e_scores.get, reverse=True)[:TOP_K])
        p_top3 = set(sorted(p_scores, key=p_scores.get, reverse=True)[:TOP_K])
        overlap = len(e_top3 & p_top3)
        pick_overlaps.append(overlap)

    avg_corr = np.mean(rank_correlations)
    avg_overlap = np.mean(pick_overlaps)

    fprint(f"  Avg Spearman rank correlation: {avg_corr:.3f}")
    fprint(f"  Avg top-{TOP_K} pick overlap: {avg_overlap:.1f}/{TOP_K}")

    if avg_corr < 0.3:
        fprint(f"  LOW correlation -- strategies are COMPLEMENTARY (diversification benefit)")
    elif avg_corr < 0.6:
        fprint(f"  MODERATE correlation -- some overlap but distinct signals")
    else:
        fprint(f"  HIGH correlation -- strategies are similar (limited diversification)")

    # Distribution of overlaps
    overlap_counts = defaultdict(int)
    for o in pick_overlaps:
        overlap_counts[o] += 1
    fprint(f"  Top-{TOP_K} overlap distribution:")
    for o in sorted(overlap_counts.keys()):
        pct = overlap_counts[o] / len(pick_overlaps) * 100
        fprint(f"    {o} overlap: {overlap_counts[o]} dates ({pct:.1f}%)")

    return {
        "avg_rank_correlation": round(avg_corr, 4),
        "avg_top3_overlap": round(avg_overlap, 2),
        "n_dates": len(common_dates),
        "overlap_distribution": dict(overlap_counts),
        "correlation_std": round(float(np.std(rank_correlations)), 4),
    }


# ================================================================
# RANDOM BASELINE
# ================================================================

def random_baseline_test(rankings, close, high, low, atr_dict, n_trials=5):
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
            f"Random_{trial}", rand_rankings, close, high, low, atr_dict
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


# ================================================================
# MAIN
# ================================================================

def main():
    t0 = datetime.now()
    fprint("=" * 100)
    fprint(f"EARNINGS SECTOR STANDALONE v1 -- {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 100)
    fprint(f"Context: Earnings-only features achieved Sharpe 2.62, +7% vs 21-feature production model")
    fprint(f"Goal: Validate as standalone strategy + find better earnings features")
    fprint()
    fprint(f"V6 config (fixed):")
    fprint(f"  Capital: ${CAP:.0f} | DTE: {DTE} | Spread: {SPREAD_PCT:.0f}% | "
           f"Haircut: {DEFAULT_HAIRCUT:.0%} entry | Comm: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"  Rebalance: weekly (W-FRI) | OTM: 2% | Pairs: VIX<20 bull+bear")
    fprint(f"  Hold to expiry | Intrinsic value only")
    fprint(f"  Regime filter: GRU >0.4")
    fprint(f"  Walk-forward: {WF_TRAIN_PERIODS} period sliding window")
    fprint()
    fprint(f"Variants:")
    fprint(f"  A: Original 4 earnings features (reproduce Sharpe 2.62)")
    fprint(f"  B: 4 original + 4 new earnings features (8 total)")
    fprint(f"  C: Best subset from B (selected by importance)")
    fprint(f"  D: 8 earnings + 6 momentum interaction features (14 total)")
    fprint(f"  E: Production 21-feature model (correlation comparison)")
    fprint()

    # 1. Download sector data
    close, high, low, volume = download_data()

    # 2. Download constituent data
    fprint(f"\n{'='*100}")
    fprint("DOWNLOADING CONSTITUENT DATA FOR EARNINGS DETECTION")
    fprint(f"{'='*100}")
    const_close, const_volume = download_constituent_data()

    # 3. Detect earnings events
    fprint(f"\n{'='*100}")
    fprint("DETECTING EARNINGS EVENTS")
    fprint(f"{'='*100}")
    earnings_events = detect_earnings_events(const_close, const_volume)

    for sector, constituents in SECTOR_CONSTITUENTS.items():
        n_events = sum(
            len(earnings_events.get(tk, pd.Series(dtype=float)))
            for tk in constituents
        )
        n_tickers_with_events = sum(
            1 for tk in constituents if tk in earnings_events
        )
        fprint(f"  {sector}: {n_events} events from {n_tickers_with_events}/{len(constituents)} tickers")

    # 4. Load regime predictions
    regime_series = load_regime_predictions()

    # 5. Pre-compute ATR
    atr_dict = compute_atr_series(high, low, close)

    # 6. Generate rebalance dates
    rebal_dates = generate_rebal_dates(close, V6_REBAL_FREQ)
    fprint(f"\nRebalance dates: {len(rebal_dates)} "
           f"({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

    spy_close = close["SPY"]

    # -- Define variants --
    VARIANT_FEATURES = {
        "A_original_4_earnings": {
            "desc": "Original 4 earnings features (reproduce Sharpe 2.62)",
            "features": ORIGINAL_EARNINGS_FEATURES[:],
            "include_legacy": False,
        },
        "B_8_earnings": {
            "desc": "4 original + 4 new earnings features (8 total)",
            "features": ALL_8_EARNINGS[:],
            "include_legacy": False,
        },
        "D_earnings_plus_interactions": {
            "desc": "8 earnings + 6 momentum interactions (14 total)",
            "features": ALL_8_EARNINGS + EARNINGS_MOMENTUM_INTERACTIONS,
            "include_legacy": False,
        },
        "E_production_21": {
            "desc": "Production 21 features (correlation comparison)",
            "features": ALL_21_FEATURES[:],
            "include_legacy": True,
        },
    }

    all_results = {}
    all_importances = {}
    all_rankings = {}
    all_trades = {}

    # -- Run A, B, D, E --
    for vname, vcfg in VARIANT_FEATURES.items():
        feature_cols = vcfg["features"]
        fprint(f"\n{'='*100}")
        fprint(f"VARIANT {vname}: {vcfg['desc']}")
        fprint(f"  Features ({len(feature_cols)}): {feature_cols}")
        fprint(f"{'='*100}")

        records = build_feature_records(
            close, high, low, rebal_dates, feature_cols, regime_series,
            earnings_events=earnings_events, const_close=const_close,
            include_legacy=vcfg.get("include_legacy", False),
        )

        rankings, imp_df = walk_forward_lgbm_rank(records, feature_cols, vname)
        all_importances[vname] = imp_df
        all_rankings[vname] = rankings

        if not rankings:
            fprint(f"  No rankings available, skipping")
            continue

        trades, final_eq = simulate_trades(vname, rankings, close, high, low, atr_dict)
        all_trades[vname] = trades

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
        bull_pnl = sum(t["pnl"] for t in bull_trades)
        bear_pnl = sum(t["pnl"] for t in bear_trades)
        bull_wr = sum(1 for t in bull_trades if t["win"]) / max(len(bull_trades), 1) * 100
        bear_wr = sum(1 for t in bear_trades if t["win"]) / max(len(bear_trades), 1) * 100
        fprint(f"  Direction breakdown:")
        fprint(f"    Bull: {len(bull_trades)} trades, WR {bull_wr:.1f}%, PnL ${bull_pnl:.0f}")
        fprint(f"    Bear: {len(bear_trades)} trades, WR {bear_wr:.1f}%, PnL ${bear_pnl:.0f}")

        # Random baseline
        random_sharpes = random_baseline_test(rankings, close, high, low, atr_dict)
        mean_random = np.mean(random_sharpes) if random_sharpes else 0

        r_dict = result.to_dict() if hasattr(result, 'to_dict') else result
        r_sharpe = result.sharpe if hasattr(result, 'sharpe') else r_dict.get("sharpe", 0)

        fprint(f"  ML Sharpe: {r_sharpe:.2f} vs Random mean: {mean_random:.2f}")

        all_results[vname] = {
            "description": vcfg["desc"],
            "n_features": len(feature_cols),
            "feature_list": feature_cols,
            **r_dict,
            "random_sharpes": [round(s, 3) for s in random_sharpes],
            "random_mean_sharpe": round(mean_random, 3),
            "bull_trades": len(bull_trades),
            "bear_trades": len(bear_trades),
            "bull_pnl": round(bull_pnl, 2),
            "bear_pnl": round(bear_pnl, 2),
            "bull_wr": round(bull_wr, 1),
            "bear_wr": round(bear_wr, 1),
        }

    # -- Variant C: Best subset from B's importance --
    fprint(f"\n{'='*100}")
    fprint("VARIANT C: Selecting best earnings features from B's importance")
    fprint(f"{'='*100}")

    imp_b = all_importances.get("B_8_earnings")
    if imp_b is not None:
        fprint(f"  Feature importance from B (all 8 earnings):")
        for _, row in imp_b.iterrows():
            bar = "*" * int(row["importance"] / (imp_b["importance"].max() + 1e-10) * 25)
            is_new = " <== NEW" if row["feature"] in NEW_EARNINGS_FEATURES else ""
            fprint(f"    {row['feature']:<40} {row['importance']:>6.1f} {bar}{is_new}")

        # Select features with above-median importance
        median_imp = imp_b["importance"].median()
        best_features = imp_b[imp_b["importance"] > median_imp]["feature"].tolist()
        # At least keep top 4
        if len(best_features) < 4:
            best_features = imp_b.head(4)["feature"].tolist()

        fprint(f"  Selected {len(best_features)} best features: {best_features}")

        records_c = build_feature_records(
            close, high, low, rebal_dates, best_features, regime_series,
            earnings_events=earnings_events, const_close=const_close,
            include_legacy=False,
        )
        rankings_c, imp_c = walk_forward_lgbm_rank(records_c, best_features, "C_best_subset")
        all_importances["C_best_subset"] = imp_c
        all_rankings["C_best_subset"] = rankings_c

        if rankings_c:
            trades_c, final_eq_c = simulate_trades(
                "C_best_subset", rankings_c, close, high, low, atr_dict
            )
            all_trades["C_best_subset"] = trades_c

            if trades_c and len(trades_c) >= 10:
                result_c = validate_trades(
                    trades_c, initial_capital=CAP,
                    spy_prices=spy_close,
                    strategy_name="C_best_subset",
                )
                if hasattr(result_c, 'print_summary'):
                    result_c.print_summary()

                r_dict_c = result_c.to_dict() if hasattr(result_c, 'to_dict') else result_c

                random_sharpes_c = random_baseline_test(rankings_c, close, high, low, atr_dict)

                bull_c = [t for t in trades_c if t["direction"] == "bull"]
                bear_c = [t for t in trades_c if t["direction"] == "bear"]

                all_results["C_best_subset"] = {
                    "description": f"Best earnings subset: {best_features}",
                    "n_features": len(best_features),
                    "feature_list": best_features,
                    **r_dict_c,
                    "random_sharpes": [round(s, 3) for s in random_sharpes_c],
                    "random_mean_sharpe": round(np.mean(random_sharpes_c), 3),
                    "selected_features": best_features,
                    "bull_trades": len(bull_c),
                    "bear_trades": len(bear_c),
                    "bull_pnl": round(sum(t["pnl"] for t in bull_c), 2),
                    "bear_pnl": round(sum(t["pnl"] for t in bear_c), 2),
                    "bull_wr": round(sum(1 for t in bull_c if t["win"]) / max(len(bull_c), 1) * 100, 1),
                    "bear_wr": round(sum(1 for t in bear_c if t["win"]) / max(len(bear_c), 1) * 100, 1),
                }
    else:
        fprint("  Variant B produced no importance data -- cannot select for C")

    # -- REGIME ANALYSIS (on best earnings variant) --
    best_earnings_variant = None
    best_sharpe = -999
    for vname in ["A_original_4_earnings", "B_8_earnings", "C_best_subset",
                  "D_earnings_plus_interactions"]:
        r = all_results.get(vname)
        if r and r.get("sharpe", 0) > best_sharpe:
            best_sharpe = r["sharpe"]
            best_earnings_variant = vname

    regime_results = {}
    seasonality_results = {}
    if best_earnings_variant and best_earnings_variant in all_trades:
        fprint(f"\n  Running regime + seasonality analysis on best variant: {best_earnings_variant}")
        regime_results = regime_analysis(all_trades[best_earnings_variant], close)
        seasonality_results = seasonality_analysis(all_trades[best_earnings_variant])

    # -- CORRELATION ANALYSIS --
    corr_results = {}
    if "A_original_4_earnings" in all_rankings and "E_production_21" in all_rankings:
        corr_results = correlation_analysis(
            all_rankings["A_original_4_earnings"],
            all_rankings["E_production_21"]
        )

    # -- SUMMARY COMPARISON --
    fprint(f"\n{'='*120}")
    fprint("EARNINGS STANDALONE -- SUMMARY COMPARISON")
    fprint(f"{'='*120}")
    fprint(f"{'Variant':<35} {'#Feat':>5} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} "
           f"{'PF':>6} {'MaxDD':>7} {'Gates':>6} {'Final$':>8} {'RandSh':>7}")
    fprint("-" * 120)

    variant_order = ["A_original_4_earnings", "B_8_earnings", "C_best_subset",
                     "D_earnings_plus_interactions", "E_production_21"]

    for vname in variant_order:
        r = all_results.get(vname)
        if not r:
            fprint(f"  {vname:<35} -- NO DATA --")
            continue
        _wr = r.get("win_rate", r.get("wr", 0))
        _pf = r.get("profit_factor", r.get("pf", 0))
        _mdd = r.get("max_dd", 0)
        _gp = r.get("gates_passed", 0)
        _gt = r.get("gates_total", 5)
        fprint(f"  {vname:<35} {r['n_features']:>4} {r.get('n_trades', 0):>5} {r['sharpe']:>7.2f} "
               f"{r['sortino']:>8.2f} {_wr*100:>5.1f}% {_pf:>5.2f} "
               f"{_mdd*100:>6.1f}% {_gp}/{_gt} "
               f"${r.get('final_equity', 0):>7,.0f} {r['random_mean_sharpe']:>7.2f}")

    # -- SHARPE DELTA vs BASELINE --
    baseline_a_sharpe = all_results.get("A_original_4_earnings", {}).get("sharpe", 0)
    prod_sharpe = all_results.get("E_production_21", {}).get("sharpe", 0)

    fprint(f"\n{'='*100}")
    fprint("SHARPE COMPARISON")
    fprint(f"{'='*100}")
    fprint(f"  Original 4-earnings (A) Sharpe:   {baseline_a_sharpe:.2f}")
    fprint(f"  Production 21-feature (E) Sharpe: {prod_sharpe:.2f}")
    if prod_sharpe > 0:
        delta = baseline_a_sharpe - prod_sharpe
        pct = delta / abs(prod_sharpe) * 100
        fprint(f"  Earnings vs Production delta:     {delta:+.2f} ({pct:+.1f}%)")

    # -- KEY FINDINGS --
    fprint(f"\n{'='*100}")
    fprint("KEY FINDINGS")
    fprint(f"{'='*100}")

    if best_earnings_variant:
        best_r = all_results[best_earnings_variant]
        fprint(f"\n  1. BEST EARNINGS VARIANT: {best_earnings_variant}")
        fprint(f"     Sharpe {best_r['sharpe']:.2f}, {best_r['n_features']} features, "
               f"{best_r.get('n_trades', 0)} trades")

    # Feature importance insights
    for vname in ["A_original_4_earnings", "B_8_earnings"]:
        imp = all_importances.get(vname)
        if imp is not None:
            fprint(f"\n  Feature importance ({vname}, top 5):")
            for _, row in imp.head(5).iterrows():
                bar = "*" * int(row["importance"] / (imp["importance"].max() + 1e-10) * 25)
                is_new = " <== NEW" if row["feature"] in NEW_EARNINGS_FEATURES else ""
                fprint(f"    {row['feature']:<40} {row['importance']:>6.1f} {bar}{is_new}")

    # Correlation insight
    if corr_results:
        fprint(f"\n  2. CORRELATION WITH PRODUCTION:")
        fprint(f"     Rank correlation: {corr_results.get('avg_rank_correlation', 'N/A'):.3f}")
        fprint(f"     Top-3 pick overlap: {corr_results.get('avg_top3_overlap', 'N/A'):.1f}/{TOP_K}")
        if corr_results.get('avg_rank_correlation', 1) < 0.3:
            fprint(f"     HIGHLY COMPLEMENTARY -- could combine for diversification")
        elif corr_results.get('avg_rank_correlation', 1) < 0.6:
            fprint(f"     MODERATELY COMPLEMENTARY -- some diversification benefit")
        else:
            fprint(f"     HIGH OVERLAP -- limited diversification benefit")

    # Regime robustness
    if regime_results:
        fprint(f"\n  3. REGIME ROBUSTNESS:")
        for label, data in regime_results.items():
            if isinstance(data, dict) and "sharpe_proxy" in data:
                fprint(f"     {label}: Sharpe {data['sharpe_proxy']:.2f}, "
                       f"WR {data['win_rate']:.1%}, {data['n_trades']} trades")

    # Seasonality insight
    if seasonality_results:
        earn_agg = seasonality_results.get("_earnings_season_aggregate", {})
        off_agg = seasonality_results.get("_off_season_aggregate", {})
        if earn_agg and off_agg:
            fprint(f"\n  4. SEASONALITY:")
            fprint(f"     Earnings months: {earn_agg.get('n_trades', 0)} trades, "
                   f"WR {earn_agg.get('win_rate', 0):.1%}, "
                   f"PnL ${earn_agg.get('total_pnl', 0):.0f}")
            fprint(f"     Off-season:      {off_agg.get('n_trades', 0)} trades, "
                   f"WR {off_agg.get('win_rate', 0):.1%}, "
                   f"PnL ${off_agg.get('total_pnl', 0):.0f}")

    # -- RESEARCH CONCLUSION --
    fprint(f"\n{'='*100}")
    fprint("RESEARCH CONCLUSION")
    fprint(f"{'='*100}")

    if best_earnings_variant and all_results.get(best_earnings_variant):
        best = all_results[best_earnings_variant]
        best_sharpe_val = best.get("sharpe", 0)

        if best_sharpe_val > 2.0 and best.get("gates_passed", 0) >= 4:
            fprint(f"  STRONG: Earnings standalone strategy is VIABLE")
            fprint(f"  Best variant: {best_earnings_variant} (Sharpe {best_sharpe_val:.2f})")
            if corr_results.get("avg_rank_correlation", 1) < 0.5:
                fprint(f"  COMPLEMENTARY to production -- recommend running BOTH")
            else:
                fprint(f"  SIMILAR to production -- may replace rather than complement")
        elif best_sharpe_val > 1.0:
            fprint(f"  MODERATE: Earnings strategy has signal but weaker standalone")
            fprint(f"  Best variant: {best_earnings_variant} (Sharpe {best_sharpe_val:.2f})")
            fprint(f"  Recommendation: Use as feature input to production model, not standalone")
        else:
            fprint(f"  WEAK: Earnings standalone does not reproduce well")
            fprint(f"  Best variant: {best_earnings_variant} (Sharpe {best_sharpe_val:.2f})")
            fprint(f"  Recommendation: Keep as supplementary features in production model")

        # Check if new features helped
        b_sharpe = all_results.get("B_8_earnings", {}).get("sharpe", 0)
        a_sharpe = all_results.get("A_original_4_earnings", {}).get("sharpe", 0)
        if b_sharpe > a_sharpe + 0.1:
            fprint(f"\n  NEW FEATURES: 4 new earnings features IMPROVED Sharpe "
                   f"({a_sharpe:.2f} -> {b_sharpe:.2f})")
        elif b_sharpe < a_sharpe - 0.1:
            fprint(f"\n  NEW FEATURES: 4 new earnings features HURT Sharpe "
                   f"({a_sharpe:.2f} -> {b_sharpe:.2f})")
            fprint(f"  The original 4 features are sufficient")
        else:
            fprint(f"\n  NEW FEATURES: Marginal impact ({a_sharpe:.2f} -> {b_sharpe:.2f})")

    # Save results
    results_payload = {
        "variant_results": all_results,
        "regime_analysis": regime_results,
        "seasonality_analysis": seasonality_results,
        "correlation_analysis": corr_results,
        "config": {
            "capital": CAP,
            "dte": DTE,
            "spread_pct": SPREAD_PCT,
            "haircut": DEFAULT_HAIRCUT,
            "commission": COMMISSION_RT_SPREAD,
            "regime_bull_thresh": REGIME_BULL_THRESHOLD,
            "rebal_freq": V6_REBAL_FREQ,
            "otm_pct": V6_OTM_PCT,
            "pairs": V6_PAIRS,
            "hold_to_expiry": True,
            "wf_train_periods": WF_TRAIN_PERIODS,
        },
    }

    results_path = OUTPUT_DIR / "earnings_standalone_results.json"
    with open(results_path, "w") as fp:
        json.dump(results_payload, fp, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    # MLflow logging
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"earn_standalone_{t0.strftime('%Y%m%d_%H%M')}"):
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
                    mlflow.log_metric(f"{prefix}_n_features", r.get("n_features", 0))

                # Log correlation metrics
                if corr_results:
                    mlflow.log_metric("corr_rank_spearman", corr_results.get("avg_rank_correlation", 0))
                    mlflow.log_metric("corr_top3_overlap", corr_results.get("avg_top3_overlap", 0))

                # Log regime metrics
                for label, data in regime_results.items():
                    if isinstance(data, dict) and "sharpe_proxy" in data:
                        clean_label = label.replace(" ", "_").replace("(", "").replace(")", "").replace(">=", "gte").replace("<", "lt")[:30]
                        mlflow.log_metric(f"regime_{clean_label}_sharpe", data["sharpe_proxy"])

                mlflow.log_params({
                    "experiment_type": "earnings_sector_standalone",
                    "capital": CAP,
                    "dte": DTE,
                    "spread_pct": SPREAD_PCT,
                    "haircut": DEFAULT_HAIRCUT,
                    "commission": COMMISSION_RT_SPREAD,
                    "regime_bull_thresh": REGIME_BULL_THRESHOLD,
                    "rebal_freq": V6_REBAL_FREQ,
                    "otm_pct": V6_OTM_PCT,
                    "pairs": V6_PAIRS,
                    "hold_to_expiry": True,
                    "wf_train_periods": WF_TRAIN_PERIODS,
                    "n_original_earnings_features": len(ORIGINAL_EARNINGS_FEATURES),
                    "n_new_earnings_features": len(NEW_EARNINGS_FEATURES),
                    "n_interaction_features": len(EARNINGS_MOMENTUM_INTERACTIONS),
                    "earnings_detection": "volume_spike_gap",
                    "best_variant": best_earnings_variant or "none",
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
