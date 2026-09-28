#!/usr/bin/env python3
"""
Earnings DTE=14 Variant v1 — Earnings Strategy With DTE=14 + 2% OTM + Real Pricing
====================================================================================

CONTEXT: Earnings standalone variant D (14 features: 8 earnings + 6 interactions)
achieved Sharpe 2.85 (BS) / 2.49 (real mid) with DTE=21 + 3% OTM.
BUT: Only 10% of trades used real chain data because DTE=21 falls in chain data dead zone.

DTE=14 has near-perfect chain data coverage (14,438 rows per ticker vs 0 for DTE=21).
DTE=14 + 2% OTM is the proven optimal combo (Finding #207).
DTE=14 also improves momentum strategies (V8: Sharpe 3.23 vs V6 DTE=21: 2.79).

Changes from earnings_real_pricing_v1:
  - DTE = 14 (was 21)
  - OTM_PCT = 0.02 (was 0.03 — DTE=14 optimal per Finding #207)
  - DTE_TOLERANCE = 5 (was 10 — DTE=14 has great coverage)
  - MLflow experiment name: 'earnings_dte14_v1'

Config: DTE=14, 2% OTM, adaptive width max($3, 3%), $645 initial, weekly W-FRI rebalance,
commission $2.60 RT, hold to expiry.

Output: output/growth_research/earnings_dte14_v1/
MLflow experiment: 'earnings_dte14_v1'
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
CHAINS_DIR = BASE / "wheel_strategy_v1" / "data" / "cache" / "options_real" / "chains"
OUTPUT_DIR = BASE / "output" / "growth_research" / "earnings_dte14_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
DTE = 14  # DTE=14 has near-perfect chain data coverage (14,438 rows/ticker)
OTM_PCT = 0.02  # 2% OTM — optimal combo with DTE=14 per Finding #207
SPREAD_PCT = 3.0
TOP_K = 3
REGIME_BULL_THRESHOLD = 0.4
COMMISSION = 2.60  # $2.60 RT commission

# Adaptive width: max($3, 3% of underlying)
def adaptive_spread_width(S):
    return max(3.0, S * 0.03)

# Walk-forward
WF_TRAIN_PERIODS = 25
REBAL_FREQ = "W-FRI"

# Chain matching
# Sector ETF chains only have monthly expirations (~14, 28, 49 DTE)
# With DTE target 21, nearest is 14 (dist=7) or 28 (dist=7), so need tolerance >= 8
DTE_TOLERANCE = 5  # Reduced from 10 — DTE=14 has excellent chain coverage
STRIKE_TOLERANCE = 0.05  # wider for sector ETFs (coarser strike grid)
MIN_BID = 0.01  # sector ETF options have tighter markets

# Cost/width filter: reject if entry > 50% of spread width
COST_WIDTH_MAX_RATIO = 0.50

REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "earnings_dte14_v1"

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
# SECTOR CONSTITUENTS (from earnings standalone)
# ================================================================

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
# FEATURE DEFINITIONS (Variant D: 14 features)
# ================================================================

ORIGINAL_EARNINGS_FEATURES = [
    "earnings_pct_reporting_2w",
    "earnings_avg_surprise",
    "earnings_post_drift",
    "earnings_days_to_heavy_week",
]

NEW_EARNINGS_FEATURES = [
    "earnings_recent_surprise_quality",
    "earnings_beat_rate_1m",
    "earnings_vol_impact",
    "sector_earnings_cycle_position",
]

ALL_8_EARNINGS = ORIGINAL_EARNINGS_FEATURES + NEW_EARNINGS_FEATURES

EARNINGS_MOMENTUM_INTERACTIONS = [
    "earn_density_x_sector_mom_21d",
    "earn_surprise_x_sector_vol",
    "earn_drift_x_sector_ret_63d",
    "earn_beat_x_sector_sharpe",
    "earn_cycle_x_sector_ret_5d",
    "earn_vol_impact_x_dispersion",
]

VARIANT_D_FEATURES = ALL_8_EARNINGS + EARNINGS_MOMENTUM_INTERACTIONS
assert len(VARIANT_D_FEATURES) == 14, f"Expected 14 features, got {len(VARIANT_D_FEATURES)}"


# ================================================================
# DATA DOWNLOAD
# ================================================================

def download_data():
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
    fprint(f"Data: {len(close)} days, {close.index[0].date()} to {close.index[-1].date()}")
    return close, high, low, volume


def download_constituent_data():
    import yfinance as yf
    all_constituents = set()
    for tickers in SECTOR_CONSTITUENTS.values():
        all_constituents.update(tickers)
    all_constituents = sorted(list(all_constituents))
    fprint(f"Downloading {len(all_constituents)} constituent stocks...")
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
    const_close = pd.DataFrame(all_close)
    const_volume = pd.DataFrame(all_volume)
    fprint(f"  Constituents: {len(const_close)} days, {len(const_close.columns)} tickers")
    return const_close, const_volume


# ================================================================
# EARNINGS EVENT DETECTION
# ================================================================

def detect_earnings_events(const_close, const_volume, vol_mult=3.0, gap_thresh=0.03):
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
    fprint(f"  Detected {total_events} events across {len(earnings_events)} tickers")
    return earnings_events


# ================================================================
# FEATURE BUILDERS (from earnings standalone)
# ================================================================

def build_original_earnings_features(sector_ticker, dt, idx, close_df,
                                     earnings_events, const_close):
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

    # 3. Post-earnings drift
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
    f = {}
    constituents = SECTOR_CONSTITUENTS.get(sector_ticker, [])
    weights = CONSTITUENT_WEIGHTS.get(sector_ticker, {})
    if not constituents:
        return {feat: 0.0 for feat in NEW_EARNINGS_FEATURES}

    current_date = close_df.index[idx]
    lookback_21d = close_df.index[max(0, idx - 21):idx + 1]
    lookback_63d = close_df.index[max(0, idx - 63):idx + 1]

    # 1. Cap-weighted surprise quality
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

    # 2. Beat rate last month
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

    # 3. Avg absolute return on earnings day
    abs_impacts = []
    for tk in constituents:
        if tk in earnings_events:
            events = earnings_events[tk]
            recent = events.index.intersection(lookback_63d)
            if len(recent) > 0:
                abs_impacts.extend(np.abs(events.loc[recent].values))
    f["earnings_vol_impact"] = float(np.mean(abs_impacts)) if abs_impacts else 0.0

    # 4. Earnings cycle position
    n_reported_63d = 0
    for tk in constituents:
        if tk in earnings_events:
            events = earnings_events[tk]
            recent = events.index.intersection(lookback_63d)
            if len(recent) > 0:
                n_reported_63d += 1
    expected_reporters = len(constituents)
    f["sector_earnings_cycle_position"] = min(n_reported_63d / expected_reporters, 1.0)

    return f


def compute_earnings_momentum_interactions(earn_feats, sector_ticker, idx, close_df):
    f = {}
    sector_px = close_df[sector_ticker].iloc[:idx + 1].dropna() if sector_ticker in close_df.columns else None
    if sector_px is None or len(sector_px) < 63:
        return {feat: 0.0 for feat in EARNINGS_MOMENTUM_INTERACTIONS}

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
    fprint(f"Regime predictions: {len(regime_series)} days")
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
# CHAIN DATA LOADING (from v8_real_pricing)
# ================================================================

def load_all_chains():
    """Load all sector ETF chain data from Dolt parquets."""
    chains = {}
    for tk in SECTORS:
        path = CHAINS_DIR / f"{tk}.parquet"
        if not path.exists():
            fprint(f"  {tk}: chain parquet not found")
            continue
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        df["expiration"] = pd.to_datetime(df["expiration"])
        for c in ["strike", "bid", "ask", "mid", "vol", "delta"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "dte" not in df.columns:
            df["dte"] = (df["expiration"] - df["date"]).dt.days
        chains[tk] = df
        fprint(f"  {tk}: {len(df):,} chain rows, "
               f"{df['date'].min().date()} to {df['date'].max().date()}")
    return chains


def find_chain_spread_price(chain_df, trade_date, ticker, direction, K1, K2, dte_target):
    """
    Look up real spread price from Dolt chain data.
    Returns dict with pricing info or None if no match.
    """
    if chain_df is None:
        return None

    # Filter to trade date
    day_mask = chain_df["date"] == pd.Timestamp(trade_date)
    chain_day = chain_df[day_mask]
    if chain_day.empty:
        nearby = chain_df[(chain_df["date"] >= pd.Timestamp(trade_date) - pd.Timedelta(days=2)) &
                          (chain_df["date"] <= pd.Timestamp(trade_date) + pd.Timedelta(days=2))]
        if nearby.empty:
            return None
        nearest_date = nearby["date"].unique()
        nearest_date = min(nearest_date, key=lambda x: abs((x - pd.Timestamp(trade_date)).days))
        chain_day = chain_df[chain_df["date"] == nearest_date]

    # Find best expiration
    exps = chain_day[["expiration", "dte"]].drop_duplicates()
    exps["dte_dist"] = (exps["dte"] - dte_target).abs()
    valid_exps = exps[exps["dte_dist"] <= DTE_TOLERANCE]
    if valid_exps.empty:
        return None
    best_exp_row = valid_exps.loc[valid_exps["dte_dist"].idxmin()]
    best_exp = best_exp_row["expiration"]
    actual_dte = int(best_exp_row["dte"])

    chain_exp = chain_day[chain_day["expiration"] == best_exp]

    if direction == "bull":
        opt_type = "c"
        near_target = K1  # buy call at K1
        far_target = K2   # sell call at K2
    else:
        opt_type = "p"
        near_target = K2  # buy put at K2
        far_target = K1   # sell put at K1

    # Find near leg (the one we BUY)
    near_opts = chain_exp[chain_exp["type"] == opt_type].copy()
    if near_opts.empty:
        return None
    near_opts["dist"] = (near_opts["strike"] - near_target).abs()
    near_opts = near_opts.sort_values("dist")
    near_leg = near_opts.iloc[0]
    if near_leg["dist"] / max(near_target, 1) > STRIKE_TOLERANCE:
        return None

    # Find far leg (the one we SELL)
    far_opts = chain_exp[chain_exp["type"] == opt_type].copy()
    far_opts["dist"] = (far_opts["strike"] - far_target).abs()
    far_opts = far_opts.sort_values("dist")
    far_leg = far_opts.iloc[0]
    if far_leg["dist"] / max(far_target, 1) > STRIKE_TOLERANCE:
        return None

    # Extract prices
    near_bid = float(near_leg["bid"]) if not pd.isna(near_leg["bid"]) else 0
    near_ask = float(near_leg["ask"]) if not pd.isna(near_leg["ask"]) else 0
    near_mid = float(near_leg["mid"]) if not pd.isna(near_leg["mid"]) else (near_bid + near_ask) / 2
    far_bid = float(far_leg["bid"]) if not pd.isna(far_leg["bid"]) else 0
    far_ask = float(far_leg["ask"]) if not pd.isna(far_leg["ask"]) else 0
    far_mid = float(far_leg["mid"]) if not pd.isna(far_leg["mid"]) else (far_bid + far_ask) / 2

    # Spread cost = what we pay to enter
    # Buy near leg (pay ask), sell far leg (receive bid) for market order
    spread_cost_market = near_ask - far_bid
    spread_cost_mid = near_mid - far_mid

    if spread_cost_market < 0:
        spread_cost_market = abs(spread_cost_market)
    if spread_cost_mid < 0:
        spread_cost_mid = abs(spread_cost_mid)

    fillable = near_bid >= MIN_BID and far_bid >= MIN_BID

    return {
        "found": True,
        "fillable": fillable,
        "spread_cost_market": spread_cost_market,
        "spread_cost_mid": spread_cost_mid,
        "near_bid": near_bid,
        "near_ask": near_ask,
        "near_mid": near_mid,
        "far_bid": far_bid,
        "far_ask": far_ask,
        "far_mid": far_mid,
        "actual_dte": actual_dte,
        "near_strike": float(near_leg["strike"]),
        "far_strike": float(far_leg["strike"]),
    }


# ================================================================
# WALK-FORWARD LGBM RANKING
# ================================================================

def build_feature_records(close, high, low, rebal_dates, regime_series,
                          earnings_events, const_close):
    """Build Variant D feature records."""
    fprint(f"  Building records: {len(rebal_dates)} dates, {len(VARIANT_D_FEATURES)} features")
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
            if len(px) < 260:
                continue

            # Original earnings features
            earn_feats = {}
            if earnings_events is not None:
                earn_feats = build_original_earnings_features(
                    tk, dt, idx, close, earnings_events, const_close
                )

            # New earnings features
            new_earn_feats = {}
            if earnings_events is not None:
                new_earn_feats = build_new_earnings_features(
                    tk, dt, idx, close, earnings_events, const_close
                )

            # Interaction features
            all_earn = {**earn_feats, **new_earn_feats}
            interaction_feats = compute_earnings_momentum_interactions(
                all_earn, tk, idx, close
            )

            # Forward return target
            fi = min(idx + DTE, len(close) - 1)
            if fi <= idx:
                continue
            fwd_ret = float(close[tk].iloc[fi] / close[tk].iloc[idx] - 1)

            rec = {**earn_feats, **new_earn_feats, **interaction_feats,
                   "date": dt, "ticker": tk, "fwd_ret": fwd_ret}
            records.append(rec)

    df = pd.DataFrame(records)
    for c in VARIANT_D_FEATURES:
        if c not in df.columns:
            df[c] = 0.0
    df[VARIANT_D_FEATURES] = df[VARIANT_D_FEATURES].fillna(0.0)
    fprint(f"    {len(df)} records, {len(df['date'].unique())} dates")
    return df


def walk_forward_lgbm_rank(df):
    """Walk-forward LGBM ranking using Variant D features."""
    import lightgbm as lgb
    if len(df) < 100:
        fprint(f"    Insufficient data ({len(df)} records)")
        return {}, None

    df = df.copy()
    df["rank_label"] = df.groupby("date")["fwd_ret"].rank(pct=True)
    dates = sorted(df["date"].unique())
    rankings = {}
    all_importances = np.zeros(len(VARIANT_D_FEATURES))
    n_models = 0

    for i in range(WF_TRAIN_PERIODS, len(dates)):
        train_dates = dates[max(0, i - WF_TRAIN_PERIODS):i]
        test_date = dates[i]
        train_df = df[df["date"].isin(train_dates)]
        test_df = df[df["date"] == test_date].copy()
        if len(test_df) < 3 or len(train_df) < 50:
            continue

        Xt = np.nan_to_num(train_df[VARIANT_D_FEATURES].values.astype(np.float32))
        yt = train_df["rank_label"].values.astype(np.float32)
        Xe = np.nan_to_num(test_df[VARIANT_D_FEATURES].values.astype(np.float32))

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

    imp_df = None
    if n_models > 0:
        all_importances /= n_models
        imp_df = pd.DataFrame({
            "feature": VARIANT_D_FEATURES,
            "importance": all_importances,
        }).sort_values("importance", ascending=False)

    fprint(f"    {len(rankings)} ranking dates, {n_models} models trained")
    return rankings, imp_df


# ================================================================
# ATR COMPUTATION
# ================================================================

def compute_atr_series(high, low, close, period=14):
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
# TRADE EXECUTION WITH MULTI-PRICING
# ================================================================

PRICING_MODES = {
    "A_bs_baseline": "bs",
    "B_real_mid": "real_mid",
    "C_real_market": "real_market",
}


def compute_strikes(S, direction):
    """Compute strikes with adaptive width: max($3, 3% of underlying)."""
    width = adaptive_spread_width(S)
    if direction == "bull":
        K1 = round(S * (1 + OTM_PCT), 2)
        K2 = round(K1 + width, 2)
    else:
        K2 = round(S * (1 - OTM_PCT), 2)
        K1 = round(K2 - width, 2)
    if K2 <= K1:
        K2 = K1 + 0.50
    return K1, K2


def execute_trade(tk, dt, direction, close, atr_dict, vix_val, equity,
                  chains, pricing_mode, max_pos):
    """Execute a single spread trade with specified pricing mode."""
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

    K1, K2 = compute_strikes(S, direction)
    spread_width = K2 - K1

    # Determine entry cost based on pricing mode
    used_real = False
    entry_cost_ps = None
    bs_entry_cost_ps = None

    # Always compute BS price for comparison
    try:
        if direction == "bull":
            bs_entry_cost_ps, _ = price_bull_call_spread(
                S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=vix_val
            )
        else:
            bs_entry_cost_ps, _ = price_bear_put_spread(
                S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=vix_val
            )
    except Exception:
        bs_entry_cost_ps = None

    if pricing_mode != "bs":
        chain_df = chains.get(tk)
        if chain_df is not None:
            chain_result = find_chain_spread_price(
                chain_df, dt, tk, direction, K1, K2, DTE
            )
            if chain_result is not None and chain_result["found"] and chain_result["fillable"]:
                if pricing_mode == "real_mid":
                    entry_cost_ps = chain_result["spread_cost_mid"]
                elif pricing_mode == "real_market":
                    entry_cost_ps = chain_result["spread_cost_market"]
                used_real = True

                # Use actual strikes from chain
                if direction == "bull":
                    K1 = chain_result["near_strike"]
                    K2 = chain_result["far_strike"]
                else:
                    K1 = chain_result["far_strike"]
                    K2 = chain_result["near_strike"]
                spread_width = K2 - K1

    # Fall back to BS if no real price
    if entry_cost_ps is None:
        entry_cost_ps = bs_entry_cost_ps

    if entry_cost_ps is None or entry_cost_ps <= 0:
        return None

    # Cost/width filter: reject if entry > 50% of spread width
    if spread_width > 0 and entry_cost_ps / spread_width > COST_WIDTH_MAX_RATIO:
        return None

    total_cost = entry_cost_ps * 100 + COMMISSION

    if total_cost <= 0 or total_cost > max_pos or total_cost > equity * 0.40:
        return None

    # Hold to expiry: intrinsic value
    Se = float(close[tk].iloc[ei])

    if direction == "bull":
        intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
    else:
        intrinsic = max(K2 - Se, 0.0) - max(K1 - Se, 0.0)

    pnl = (intrinsic - entry_cost_ps) * 100 - COMMISSION

    return {
        "pnl": round(pnl, 2),
        "entry_cost_ps": round(entry_cost_ps, 4),
        "bs_entry_cost_ps": round(bs_entry_cost_ps, 4) if bs_entry_cost_ps else None,
        "total_cost": round(total_cost, 2),
        "used_real_pricing": used_real,
        "K1": K1,
        "K2": K2,
        "spread_width": round(spread_width, 2),
        "cost_width_ratio": round(entry_cost_ps / max(spread_width, 0.01), 4),
        "S_entry": round(S, 2),
        "S_exit": round(Se, 2),
        "intrinsic": round(intrinsic, 4),
    }


# ================================================================
# TRADE SIMULATION
# ================================================================

def simulate_variant(name, pricing_mode, rankings, close, atr_dict, chains):
    """Simulate all trades for one pricing variant."""
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    equity = CAP
    trades = []
    real_count = 0
    bs_count = 0

    for dt in sorted(rankings.keys()):
        if dt not in spy.index:
            continue

        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0
        scores = rankings[dt]
        if not scores:
            continue

        # VIX pairs logic
        if cv < 20.0:
            trade_mode = "pairs"
        else:
            trade_mode = "bull_only"

        ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_asc = sorted(scores.items(), key=lambda x: x[1])

        bull_picks = [t for t, _ in ranked_desc[:TOP_K]]
        bear_picks = [t for t, _ in ranked_asc[:TOP_K]] if trade_mode == "pairs" else []

        if trade_mode == "pairs":
            max_pos = min(100, equity / 6)
        else:
            max_pos = min(200, equity / 3)

        if max_pos < 30:
            continue

        # Execute trades
        for tk in bull_picks:
            result = execute_trade(
                tk, dt, "bull", close, atr_dict, cv, equity,
                chains, pricing_mode, max_pos
            )
            if result is not None:
                equity += result["pnl"]
                if result["used_real_pricing"]:
                    real_count += 1
                else:
                    bs_count += 1

                di = close.index.get_loc(dt)
                ei = min(di + DTE, len(close) - 1)
                sv = float(spy.loc[dt])
                se = float(spy.iloc[ei]) if ei < len(spy) else sv

                trades.append({
                    **result,
                    "entry_date": str(dt.date()),
                    "exit_date": str(close.index[ei].date()),
                    "ticker": tk,
                    "regime": "bull" if se >= sv else "bear",
                    "direction": "bull",
                    "vix": round(cv, 1),
                    "win": result["pnl"] > 0,
                    "trade_mode": trade_mode,
                    "month": dt.month,
                })

        for tk in bear_picks:
            result = execute_trade(
                tk, dt, "bear", close, atr_dict, cv, equity,
                chains, pricing_mode, max_pos
            )
            if result is not None:
                equity += result["pnl"]
                if result["used_real_pricing"]:
                    real_count += 1
                else:
                    bs_count += 1

                di = close.index.get_loc(dt)
                ei = min(di + DTE, len(close) - 1)
                sv = float(spy.loc[dt])
                se = float(spy.iloc[ei]) if ei < len(spy) else sv

                trades.append({
                    **result,
                    "entry_date": str(dt.date()),
                    "exit_date": str(close.index[ei].date()),
                    "ticker": tk,
                    "regime": "bull" if se >= sv else "bear",
                    "direction": "bear",
                    "vix": round(cv, 1),
                    "win": result["pnl"] > 0,
                    "trade_mode": trade_mode,
                    "month": dt.month,
                })

    return trades, equity, real_count, bs_count


# ================================================================
# ANALYSIS FUNCTIONS
# ================================================================

def analyze_real_vs_bs(trades):
    """Compare trades using real pricing vs BS fallback."""
    real_trades = [t for t in trades if t["used_real_pricing"]]
    bs_trades = [t for t in trades if not t["used_real_pricing"]]

    fprint(f"\n  REAL vs BS PRICING BREAKDOWN:")
    fprint(f"  {'Source':<15} {'Trades':>7} {'WR':>7} {'AvgPnL':>9} {'TotalPnL':>10}")
    fprint(f"  {'-'*50}")

    for label, subset in [("Real chain", real_trades), ("BS fallback", bs_trades)]:
        if not subset:
            fprint(f"  {label:<15} {'(none)':>7}")
            continue
        pnls = [t["pnl"] for t in subset]
        wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        avg = np.mean(pnls)
        tot = sum(pnls)
        fprint(f"  {label:<15} {len(subset):>7} {wr:>6.1f}% ${avg:>8.2f} ${tot:>9.0f}")

    if real_trades:
        real_dates = [t["entry_date"] for t in real_trades]
        fprint(f"\n  Real pricing date range: {min(real_dates)} to {max(real_dates)}")
        fprint(f"  Real pricing coverage: {len(real_trades)}/{len(trades)} "
               f"({len(real_trades)/len(trades)*100:.0f}%)")

    # Entry cost comparison for real-priced trades
    if real_trades:
        real_costs = [t["entry_cost_ps"] for t in real_trades]
        bs_costs = [t["bs_entry_cost_ps"] for t in real_trades if t["bs_entry_cost_ps"] is not None]
        fprint(f"\n  Real entry cost (per share): mean=${np.mean(real_costs):.4f}, "
               f"median=${np.median(real_costs):.4f}")
        if bs_costs:
            fprint(f"  BS entry cost (per share):   mean=${np.mean(bs_costs):.4f}, "
                   f"median=${np.median(bs_costs):.4f}")
            ratios = [r / b for r, b in zip(real_costs, bs_costs) if b > 0]
            if ratios:
                fprint(f"  Real/BS cost ratio: mean={np.mean(ratios):.2f}x, "
                       f"median={np.median(ratios):.2f}x")

    # Cost/width stats
    cost_ratios = [t["cost_width_ratio"] for t in trades if "cost_width_ratio" in t]
    if cost_ratios:
        fprint(f"\n  Cost/width ratio: mean={np.mean(cost_ratios):.3f}, "
               f"median={np.median(cost_ratios):.3f}, "
               f"max={max(cost_ratios):.3f} (cap={COST_WIDTH_MAX_RATIO})")


def regime_stratification(trades):
    fprint(f"\n  REGIME STRATIFICATION:")
    fprint(f"  {'Regime':<10} {'Trades':>7} {'WR':>7} {'AvgPnL':>9} {'TotalPnL':>10} {'Sharpe':>8}")
    fprint(f"  {'-'*55}")
    for regime in ["bull", "bear"]:
        rt = [t for t in trades if t["regime"] == regime]
        if not rt:
            continue
        pnls = [t["pnl"] for t in rt]
        wr = sum(1 for p in pnls if p > 0) / len(pnls)
        avg = np.mean(pnls)
        tot = sum(pnls)
        sh = float(np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(52))
        fprint(f"  {regime:<10} {len(rt):>7} {wr:>6.1%} ${avg:>8.2f} ${tot:>9.0f} {sh:>8.2f}")


def yearly_breakdown(trades):
    fprint(f"\n  YEARLY BREAKDOWN:")
    fprint(f"  {'Year':<6} {'Trades':>7} {'WR':>7} {'PnL':>10} {'Sharpe':>8} {'%Real':>7}")
    fprint(f"  {'-'*50}")
    by_year = {}
    for t in trades:
        yr = t["entry_date"][:4]
        by_year.setdefault(yr, []).append(t)
    for yr in sorted(by_year.keys()):
        yt = by_year[yr]
        pnls = [t["pnl"] for t in yt]
        wr = sum(1 for p in pnls if p > 0) / len(pnls)
        tot = sum(pnls)
        sh = float(np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(52))
        pct_real = sum(1 for t in yt if t["used_real_pricing"]) / len(yt) * 100
        fprint(f"  {yr:<6} {len(yt):>7} {wr:>6.1%} ${tot:>9.0f} {sh:>8.2f} {pct_real:>6.0f}%")


# ================================================================
# MAIN
# ================================================================

def main():
    t0 = datetime.now()
    fprint("=" * 100)
    fprint(f"EARNINGS REAL-PRICING VALIDATION v1 -- {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 100)
    fprint(f"Context: Earnings standalone variant D (14 features) achieved Sharpe 2.85 with BS pricing")
    fprint(f"Goal: Test if earnings strategy survives real options chain pricing")
    fprint(f"Prior finding: V8 went from Sharpe 3.24 (BS) to 2.45 (real mid), 0.76x multiplier")
    fprint()
    fprint(f"Config:")
    fprint(f"  Capital: ${CAP:.0f} | DTE: {DTE} | OTM: {OTM_PCT*100:.0f}% | "
           f"Width: adaptive max($3, 3%)")
    fprint(f"  Rebalance: weekly (W-FRI) | Commission: ${COMMISSION:.2f} | Hold to expiry")
    fprint(f"  Features: 14 (8 earnings + 6 interactions)")
    fprint(f"  Walk-forward: {WF_TRAIN_PERIODS} period sliding window")
    fprint(f"  Cost/width filter: reject if entry > {COST_WIDTH_MAX_RATIO*100:.0f}% of spread width")
    fprint(f"  Regime filter: GRU >0.4 | Pairs: VIX<20")
    fprint()
    fprint(f"3 pricing variants:")
    fprint(f"  A: BS + 15% haircut (baseline, should reproduce ~2.85 Sharpe)")
    fprint(f"  B: Real mid-price from Dolt chains (limit order)")
    fprint(f"  C: Real ask/bid from Dolt chains (market order worst case)")
    fprint(f"\nDolt chain data: 2019-2026 for all 11 sector ETFs.")
    fprint(f"Before 2019: all variants use BS (identical results).")
    fprint()

    # 1. Load chain data
    fprint("=" * 80)
    fprint("LOADING DOLT CHAIN DATA")
    fprint("=" * 80)
    chains = load_all_chains()
    fprint(f"  {len(chains)} sectors with chain data\n")

    # 2. Download price data
    fprint("=" * 80)
    fprint("DOWNLOADING PRICE DATA")
    fprint("=" * 80)
    close, high, low, volume = download_data()

    # 3. Download constituent data for earnings detection
    fprint(f"\n{'='*80}")
    fprint("DOWNLOADING CONSTITUENT DATA")
    fprint(f"{'='*80}")
    const_close, const_volume = download_constituent_data()

    # 4. Detect earnings events
    fprint(f"\n{'='*80}")
    fprint("DETECTING EARNINGS EVENTS")
    fprint(f"{'='*80}")
    earnings_events = detect_earnings_events(const_close, const_volume)
    for sector, constituents in SECTOR_CONSTITUENTS.items():
        n_events = sum(
            len(earnings_events.get(tk, pd.Series(dtype=float)))
            for tk in constituents
        )
        fprint(f"  {sector}: {n_events} events")

    # 5. Load regime predictions
    regime_series = load_regime_predictions()

    # 6. ATR
    atr_dict = compute_atr_series(high, low, close)

    # 7. Build LGBM rankings (shared across all variants)
    fprint(f"\n{'='*80}")
    fprint("BUILDING LGBM RANKINGS (Variant D: 14 features)")
    fprint(f"{'='*80}")

    rebal_dates = pd.DatetimeIndex(
        close.index.to_series().resample(REBAL_FREQ).last().dropna().values
    )
    fprint(f"  Rebalance dates: {len(rebal_dates)} "
           f"({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

    records = build_feature_records(close, high, low, rebal_dates, regime_series,
                                    earnings_events, const_close)
    rankings, imp_df = walk_forward_lgbm_rank(records)

    if not rankings:
        fprint("ERROR: No rankings generated. Aborting.")
        return

    if imp_df is not None:
        fprint(f"\n  Feature importances:")
        for _, row in imp_df.iterrows():
            fprint(f"    {row['feature']:<40s} {row['importance']:>8.1f}")

    # 8. Simulate all 3 variants
    fprint(f"\n{'='*100}")
    fprint("SIMULATING ALL PRICING VARIANTS")
    fprint(f"{'='*100}")

    all_results = {}
    all_trades = {}
    spy_close = close["SPY"]

    for vname, pmode in PRICING_MODES.items():
        fprint(f"\n{'~'*90}")
        fprint(f"VARIANT {vname} (pricing: {pmode})")
        fprint(f"{'~'*90}")

        trades, final_eq, real_count, bs_count = simulate_variant(
            vname, pmode, rankings, close, atr_dict, chains
        )

        if not trades or len(trades) < 10:
            fprint(f"  Only {len(trades) if trades else 0} trades, skipping")
            continue

        all_trades[vname] = trades
        fprint(f"\n  Total trades: {len(trades)}")
        fprint(f"  Real-priced: {real_count} ({real_count/(real_count+bs_count)*100:.0f}%)")
        fprint(f"  BS-fallback: {bs_count} ({bs_count/(real_count+bs_count)*100:.0f}%)")
        fprint(f"  Final equity: ${final_eq:,.0f} (from ${CAP:.0f})")

        # 5-gate validation
        fprint(f"\n  5-GATE ADVERSARIAL VALIDATION:")
        result = validate_trades(
            trades, initial_capital=CAP,
            spy_prices=spy_close,
            strategy_name=vname,
        )
        result.print_summary()

        # Detailed analysis
        analyze_real_vs_bs(trades)
        regime_stratification(trades)
        yearly_breakdown(trades)

        rd = result.to_dict()
        all_results[vname] = {
            **rd,
            "pricing_mode": pmode,
            "real_priced_trades": real_count,
            "bs_fallback_trades": bs_count,
            "real_pct": round(real_count / max(real_count + bs_count, 1) * 100, 1),
        }

    # ── COMPARISON TABLE ──
    fprint(f"\n{'='*120}")
    fprint("COMPARISON -- ALL PRICING VARIANTS")
    fprint(f"{'='*120}")
    fprint(f"{'Variant':<20} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} "
           f"{'PF':>6} {'MaxDD':>7} {'Gates':>6} {'Final$':>9} {'%Real':>6}")
    fprint("-" * 90)

    for vname in PRICING_MODES.keys():
        r = all_results.get(vname)
        if not r:
            fprint(f"  {vname:<20} -- NO DATA --")
            continue
        fprint(f"  {vname:<20} {r['n_trades']:>5} {r['sharpe']:>7.2f} {r['sortino']:>8.2f} "
               f"{r['win_rate']*100:>5.1f}% {r['profit_factor']:>5.2f} "
               f"{r['max_dd']*100:>6.1f}% {r['gates_passed']}/{r['gates_total']} "
               f"${r['final_equity']:>8,.0f} {r['real_pct']:>5.0f}%")

    # ── KEY FINDING: BS -> REAL MULTIPLIER ──
    fprint(f"\n{'='*100}")
    fprint("KEY FINDING: BS -> REAL PRICING MULTIPLIER")
    fprint(f"{'='*100}")

    bs_result = all_results.get("A_bs_baseline")
    mid_result = all_results.get("B_real_mid")
    market_result = all_results.get("C_real_market")

    if bs_result and mid_result:
        bs_sh = bs_result["sharpe"]
        mid_sh = mid_result["sharpe"]
        ratio_mid = mid_sh / max(bs_sh, 0.01)
        fprint(f"\n  BS baseline Sharpe:     {bs_sh:.2f}")
        fprint(f"  Real mid-price Sharpe:  {mid_sh:.2f}  ({ratio_mid:.2f}x multiplier)")
        if market_result:
            mkt_sh = market_result["sharpe"]
            ratio_mkt = mkt_sh / max(bs_sh, 0.01)
            fprint(f"  Real market Sharpe:     {mkt_sh:.2f}  ({ratio_mkt:.2f}x multiplier)")

        fprint(f"\n  Prior reference: V8 had 0.76x multiplier (BS -> real mid)")
        fprint(f"  Earnings strategy:     {ratio_mid:.2f}x multiplier")

        fprint(f"\n  VERDICT:")
        if mid_sh >= 2.0:
            fprint(f"    VIABLE -- Sharpe {mid_sh:.2f} >= 2.0 with real limit-order pricing")
        elif mid_sh >= 1.5:
            fprint(f"    MARGINAL -- Sharpe {mid_sh:.2f}, tradeable but needs optimization")
        else:
            fprint(f"    FAILS -- Sharpe {mid_sh:.2f} < 1.5 with real pricing")

        if market_result:
            mkt_sh = market_result["sharpe"]
            if mkt_sh >= 1.5:
                fprint(f"    Even market orders survive (Sharpe {mkt_sh:.2f})")
            else:
                fprint(f"    Market orders do NOT survive (Sharpe {mkt_sh:.2f})")

    # ── CHAIN-ONLY ANALYSIS (2019+ only) ──
    fprint(f"\n{'='*100}")
    fprint("CHAIN-ONLY PERIOD (2019+ where real data exists)")
    fprint(f"{'='*100}")

    for vname in PRICING_MODES.keys():
        trades = all_trades.get(vname, [])
        chain_trades = [t for t in trades if t["entry_date"] >= "2019-01-01"]
        if not chain_trades or len(chain_trades) < 10:
            continue
        pnls = [t["pnl"] for t in chain_trades]
        wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        sh = float(np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(52))
        eq = CAP + sum(pnls)
        real_pct = sum(1 for t in chain_trades if t["used_real_pricing"]) / len(chain_trades) * 100
        fprint(f"  {vname:<20} {len(chain_trades):>5} trades | Sharpe {sh:.2f} | "
               f"WR {wr:.1f}% | ${eq:,.0f} final | {real_pct:.0f}% real-priced")

    # ── BS->Real multiplier for 2019+ only ──
    if "A_bs_baseline" in all_trades and "B_real_mid" in all_trades:
        bs_2019 = [t for t in all_trades["A_bs_baseline"] if t["entry_date"] >= "2019-01-01"]
        mid_2019 = [t for t in all_trades["B_real_mid"] if t["entry_date"] >= "2019-01-01"]
        if len(bs_2019) >= 10 and len(mid_2019) >= 10:
            bs_pnls = [t["pnl"] for t in bs_2019]
            mid_pnls = [t["pnl"] for t in mid_2019]
            bs_sh = float(np.mean(bs_pnls) / (np.std(bs_pnls) + 1e-10) * np.sqrt(52))
            mid_sh = float(np.mean(mid_pnls) / (np.std(mid_pnls) + 1e-10) * np.sqrt(52))
            ratio = mid_sh / max(bs_sh, 0.01)
            fprint(f"\n  2019+ only multiplier: BS Sharpe {bs_sh:.2f} -> "
                   f"Real mid {mid_sh:.2f} ({ratio:.2f}x)")

    # ── Save results ──
    results_data = {
        "experiment": EXPERIMENT_NAME,
        "timestamp": t0.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "capital": CAP,
            "dte": DTE,
            "otm_pct": OTM_PCT,
            "spread_width": "adaptive max($3, 3%)",
            "rebal_freq": REBAL_FREQ,
            "n_features": len(VARIANT_D_FEATURES),
            "features": VARIANT_D_FEATURES,
            "commission": COMMISSION,
            "cost_width_max_ratio": COST_WIDTH_MAX_RATIO,
            "hold_to_expiry": True,
            "lgbm_n_estimators": 100,
            "lgbm_max_depth": 4,
            "lgbm_lr": 0.05,
            "wf_train_periods": WF_TRAIN_PERIODS,
            "chain_data_source": "Dolt options database (2019-2026)",
        },
        "variants": all_results,
    }

    results_path = OUTPUT_DIR / "earnings_real_pricing_v1_results.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    for vname, trades in all_trades.items():
        tpath = OUTPUT_DIR / f"{vname}_trades.json"
        with open(tpath, "w") as f:
            json.dump(trades, f, indent=2, default=str)

    # Feature importances
    if imp_df is not None:
        imp_path = OUTPUT_DIR / "feature_importances.json"
        imp_df.to_json(imp_path, orient="records", indent=2)

    # MLflow
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"earnings_real_{t0.strftime('%Y%m%d_%H%M')}"):
                for vname, r in all_results.items():
                    prefix = vname.split("_")[0]
                    for k in ["sharpe", "sortino", "win_rate", "profit_factor",
                              "max_dd", "n_trades", "gates_passed", "final_equity"]:
                        if k in r:
                            mlflow.log_metric(f"{prefix}_{k}", r[k])
                    mlflow.log_metric(f"{prefix}_real_pct", r.get("real_pct", 0))

                # Log BS->real multiplier
                if bs_result and mid_result:
                    mlflow.log_metric("bs_to_real_mid_multiplier",
                                      mid_result["sharpe"] / max(bs_result["sharpe"], 0.01))
                if bs_result and market_result:
                    mlflow.log_metric("bs_to_real_market_multiplier",
                                      market_result["sharpe"] / max(bs_result["sharpe"], 0.01))

                mlflow.log_params({
                    "capital": CAP,
                    "dte": DTE,
                    "otm_pct": OTM_PCT,
                    "rebal_freq": REBAL_FREQ,
                    "commission": COMMISSION,
                    "n_features": len(VARIANT_D_FEATURES),
                    "cost_width_max_ratio": COST_WIDTH_MAX_RATIO,
                    "n_sectors": len(SECTORS),
                    "chain_coverage": "2019-2026",
                })

                mlflow.log_artifact(str(results_path))
            fprint(f"MLflow run logged to experiment '{EXPERIMENT_NAME}'")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    fprint(f"\n{'='*100}")
    fprint("DONE -- Earnings Real-Pricing Validation v1")
    fprint(f"{'='*100}")


if __name__ == "__main__":
    main()
