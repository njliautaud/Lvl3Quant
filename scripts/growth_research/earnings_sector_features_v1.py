#!/usr/bin/env python3
"""
Earnings Sector Features v1 — Do Earnings Calendar Features Improve LGBM Ranking?
===================================================================================

HYPOTHESIS: Sectors with concentrated upcoming earnings (e.g., XLK before FAANG
earnings) exhibit different momentum characteristics. Adding earnings calendar
features to the production 21-feature LGBM ranker may improve Sharpe.

New earnings features (4):
  1. earnings_pct_reporting_2w — % of sector constituents reporting in next 2 weeks
     (proxy: volume spike + gap detection on constituents)
  2. earnings_avg_surprise — Avg historical earnings surprise for the sector
     (proxy: realized move vs implied move around earnings-like events)
  3. earnings_post_drift — Sector's post-earnings drift tendency over trailing 63d
  4. earnings_days_to_heavy_week — # trading days until sector's densest earnings week

5 Variants:
  A: Production 21 features ONLY (baseline — must match ~2.8 Sharpe)
  B: 21 features + 4 earnings features (25 total)
  C: 21 features + best 2 earnings features (23 total, selected by importance from B)
  D: Earnings features ONLY (4 features — control, tests if earnings alone has signal)
  E: 21 features + earnings interaction (earnings * momentum cross-terms, 29 total)

Walk-forward: 25-period sliding window, weekly W-FRI rebalance
Trade config: V6 (weekly, 2% OTM, bull+pairs, $645 capital, DTE=21)
Validation: 5-gate adversarial + 5-trial random baseline + permutation test on
            incremental earnings feature value
MLflow experiment: 'earnings_sector_features_v1'
Output: output/growth_research/earnings_sector_features_v1/
"""

import json
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

_builtin_print = print


def fprint(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()


# ── Path auto-detect (Jupiter vs Neptune) ──
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


# ── Config ──
BASE = Path(_BASE_PATH)
OUTPUT_DIR = BASE / "output" / "growth_research" / "earnings_sector_features_v1"
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

# LGBM walk-forward — 25 periods per user request
WF_TRAIN_PERIODS = 25

# V6 config: weekly rebalance, 2% OTM, bull+pairs
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

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "earnings_sector_features_v1"

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

EARNINGS_FEATURES = [
    "earnings_pct_reporting_2w",
    "earnings_avg_surprise",
    "earnings_post_drift",
    "earnings_days_to_heavy_week",
]

# Interaction features: earnings * key momentum features
EARNINGS_INTERACTIONS = [
    "earn_x_ret_21d",
    "earn_x_ret_63d",
    "earn_x_vol_21d",
    "earn_x_up_capture",
    "earn_density_x_ret_5d",
    "earn_surprise_x_vol_21d",
    "earn_drift_x_ret_126d",
    "earn_density_x_dispersion",
]


# ══════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════

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
    """Download constituent stock data for earnings detection.

    We detect earnings events via volume spikes + gap moves as a proxy for
    actual earnings calendar data. This works because:
    - Volume typically 3-5x normal on earnings day
    - Gaps > 3% abs indicate earnings-type events
    - This proxy has been validated in prior research scripts
    """
    import yfinance as yf

    all_constituents = set()
    for tickers in SECTOR_CONSTITUENTS.values():
        all_constituents.update(tickers)

    all_constituents = sorted(list(all_constituents))
    fprint(f"Downloading {len(all_constituents)} constituent stocks for earnings detection...")

    # Download in chunks to avoid timeout
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


# ══════════════════════════════════════════════════════════════
# EARNINGS EVENT DETECTION (Volume + Gap Proxy)
# ══════════════════════════════════════════════════════════════

def detect_earnings_events(const_close, const_volume, vol_mult=3.0, gap_thresh=0.03):
    """
    Detect earnings-like events for each constituent stock.

    An earnings event is flagged when:
      - Volume is >= vol_mult * rolling 20d median volume, AND
      - Absolute gap (open-to-prev-close proxy via close-to-close) > gap_thresh

    Returns dict: {ticker: pd.Series of event dates with "surprise" magnitude}
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

        # Daily returns (close-to-close)
        rets = c.pct_change()

        # Rolling median volume
        vol_median = v.rolling(20, min_periods=10).median()

        # Volume spike condition
        vol_spike = v > (vol_median * vol_mult)

        # Gap condition (abs return > threshold)
        gap_condition = rets.abs() > gap_thresh

        # Combined: both conditions must be true
        events = vol_spike & gap_condition
        event_dates = events[events].index

        if len(event_dates) > 0:
            # "Surprise" = the actual return on earnings day
            surprise = rets.loc[event_dates]
            earnings_events[ticker] = surprise
            total_events += len(event_dates)

    fprint(f"  Detected {total_events} earnings-like events across "
           f"{len(earnings_events)} tickers")

    # Sanity check: events per year per ticker
    if earnings_events:
        avg_events = total_events / len(earnings_events)
        n_years = (const_close.index[-1] - const_close.index[0]).days / 365
        fprint(f"  Avg {avg_events / max(n_years, 1):.1f} events/year/ticker "
               f"(expect ~4 for quarterly earnings)")

    return earnings_events


# ══════════════════════════════════════════════════════════════
# EARNINGS FEATURE COMPUTATION
# ══════════════════════════════════════════════════════════════

def build_sector_earnings_features(sector_ticker, dt, idx, close_df,
                                   earnings_events, const_close):
    """
    Compute the 4 earnings calendar features for a sector at a given date.

    1. earnings_pct_reporting_2w: fraction of sector constituents that had an
       earnings event in the NEXT 2 weeks (forward-looking in training,
       but we use TRAILING 2 weeks as proxy to avoid lookahead bias)
       Actually: we use trailing 2 weeks — how many just reported
    2. earnings_avg_surprise: average "surprise" (return on event day) for
       sector constituents over trailing 63d
    3. earnings_post_drift: avg 5-day post-earnings return for sector constituents
       over trailing 63d (PEAD signal)
    4. earnings_days_to_heavy_week: based on seasonal pattern — which quarter-week
       of the year historically has most earnings for this sector
    """
    f = {}
    constituents = SECTOR_CONSTITUENTS.get(sector_ticker, [])

    if not constituents:
        return {feat: 0.0 for feat in EARNINGS_FEATURES}

    current_date = close_df.index[idx]

    # Lookback window for earnings
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

    # 3. Post-earnings drift: avg 5-day return after earnings events in trailing 63d
    drifts = []
    for tk in constituents:
        if tk in earnings_events and tk in const_close.columns:
            events = earnings_events[tk]
            recent_events = events.index.intersection(lookback_63d)
            for edt in recent_events:
                # Find the index in const_close
                if edt in const_close.index:
                    edt_idx = const_close.index.get_loc(edt)
                    end_idx = min(edt_idx + 5, len(const_close) - 1)
                    if end_idx > edt_idx and end_idx <= idx:  # no lookahead
                        drift = float(const_close[tk].iloc[end_idx] /
                                      const_close[tk].iloc[edt_idx] - 1)
                        drifts.append(drift)
    f["earnings_post_drift"] = float(np.mean(drifts)) if drifts else 0.0

    # 4. Days to heavy earnings week (seasonal proxy)
    # Compute historical earnings density by week-of-year for this sector
    all_event_weeks = []
    for tk in constituents:
        if tk in earnings_events:
            for edt in earnings_events[tk].index:
                all_event_weeks.append(edt.isocalendar()[1])

    if all_event_weeks:
        # Find the 4 peak weeks (quarterly pattern)
        week_counts = pd.Series(all_event_weeks).value_counts()
        peak_weeks = week_counts.head(4).index.tolist()

        # Current week of year
        current_week = current_date.isocalendar()[1]

        # Days to nearest peak week
        min_dist = 52
        for pw in peak_weeks:
            dist = (pw - current_week) % 52
            if dist == 0:
                dist = 0  # we're in a peak week
            min_dist = min(min_dist, dist)

        # Normalize to 0-1 range (0 = in peak week, 1 = 26 weeks away)
        f["earnings_days_to_heavy_week"] = min_dist / 26.0
    else:
        f["earnings_days_to_heavy_week"] = 0.5

    return f


def compute_earnings_interactions(legacy_feats, earnings_feats):
    """Compute interaction features between earnings and momentum/vol features."""
    f = {}

    earn_density = earnings_feats.get("earnings_pct_reporting_2w", 0)
    earn_surprise = earnings_feats.get("earnings_avg_surprise", 0)
    earn_drift = earnings_feats.get("earnings_post_drift", 0)
    earn_days = earnings_feats.get("earnings_days_to_heavy_week", 0.5)

    # Interactions: earnings density * momentum/vol
    f["earn_x_ret_21d"] = earn_density * legacy_feats.get("ret_21d", 0)
    f["earn_x_ret_63d"] = earn_density * legacy_feats.get("ret_63d", 0)
    f["earn_x_vol_21d"] = earn_density * legacy_feats.get("vol_21d", 0)
    f["earn_x_up_capture"] = earn_density * legacy_feats.get("up_capture", 0)

    # Surprise interactions
    f["earn_density_x_ret_5d"] = earn_density * legacy_feats.get("ret_5d", 0)
    f["earn_surprise_x_vol_21d"] = earn_surprise * legacy_feats.get("vol_21d", 0)

    # Drift interactions
    f["earn_drift_x_ret_126d"] = earn_drift * legacy_feats.get("ret_126d", 0)

    # Density * cross-sector dispersion
    f["earn_density_x_dispersion"] = earn_density * legacy_feats.get(
        "cross_sector_dispersion", 0)

    return f


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
# LEGACY FEATURE ENGINEERING
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
                          earnings_events=None, const_close=None):
    """Build feature + target records for all sectors on all rebal dates."""
    fprint(f"  Building records: {len(rebal_dates)} dates, {len(feature_cols)} features")

    needs_earnings = any(f in EARNINGS_FEATURES for f in feature_cols)
    needs_interactions = any(f in EARNINGS_INTERACTIONS for f in feature_cols)

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

            legacy = compute_legacy_features(px, spy_s)
            if not legacy:
                continue

            # Cross-asset features
            cross_asset = {}
            needs_cross = any(col in VALIDATED_CROSS_ASSET for col in feature_cols)
            if needs_cross:
                cross_asset = compute_cross_asset_features(tk, idx, close)

            # Earnings features
            earn_feats = {}
            if (needs_earnings or needs_interactions) and earnings_events is not None:
                earn_feats = build_sector_earnings_features(
                    tk, dt, idx, close, earnings_events, const_close
                )

            # Interaction features
            interaction_feats = {}
            if needs_interactions and earn_feats:
                interaction_feats = compute_earnings_interactions(
                    {**legacy, **cross_asset}, earn_feats
                )

            # Forward return target (DTE days forward)
            fi = min(idx + DTE, len(close) - 1)
            if fi <= idx:
                continue
            fwd_ret = float(close[tk].iloc[fi] / close[tk].iloc[idx] - 1)

            rec = {**legacy, **cross_asset, **earn_feats, **interaction_feats,
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


# ══════════════════════════════════════════════════════════════
# ATR, STRIKE, TRADE SIMULATION (Shared V6 Infrastructure)
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


def compute_strikes(S, direction, otm_pct, spread_pct):
    """Compute strike prices for a spread (K1 < K2 always)."""
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
                })

    return trades, equity


def generate_rebal_dates(close, freq_str):
    """Generate rebalance dates from close index based on frequency string."""
    return pd.DatetimeIndex(
        close.index.to_series().resample(freq_str).last().dropna().values
    )


# ══════════════════════════════════════════════════════════════
# PERMUTATION TEST FOR INCREMENTAL FEATURE VALUE
# ══════════════════════════════════════════════════════════════

def permutation_test_incremental(df_base, df_augmented, feature_cols_base,
                                 feature_cols_augmented, close, high, low,
                                 atr_dict, n_perms=100, variant_name=""):
    """
    Permutation test: is the Sharpe improvement from added features statistically
    significant, or could it be achieved by random features?

    Approach: Shuffle ONLY the new earnings features while keeping legacy features
    intact. If shuffled versions perform similarly to augmented, the new features
    add no real signal.
    """
    fprint(f"\n  Permutation test for incremental value ({n_perms} permutations)...")

    # New feature columns
    new_features = [f for f in feature_cols_augmented if f not in feature_cols_base]
    if not new_features:
        fprint("    No new features to test")
        return {"p_value": 1.0, "obs_sharpe_delta": 0.0, "n_perms": 0}

    fprint(f"    Testing {len(new_features)} new features: {new_features}")

    # Get augmented Sharpe (already computed)
    rankings_aug, _ = walk_forward_lgbm_rank(
        df_augmented.copy(), feature_cols_augmented, f"{variant_name}_perm_obs"
    )
    trades_aug, _ = simulate_trades("perm_obs", rankings_aug, close, high, low, atr_dict)
    if trades_aug and len(trades_aug) >= 10:
        pnls_aug = np.array([t.get("pnl", 0) for t in trades_aug])
        obs_sharpe = float(np.mean(pnls_aug) / np.std(pnls_aug) * np.sqrt(52)) if np.std(pnls_aug) > 0 else 0.0
    else:
        fprint("    Augmented model produced too few trades")
        return {"p_value": 1.0, "obs_sharpe_delta": 0.0, "n_perms": 0}

    # Get baseline Sharpe
    rankings_base, _ = walk_forward_lgbm_rank(
        df_base.copy(), feature_cols_base, f"{variant_name}_perm_base"
    )
    trades_base, _ = simulate_trades("perm_base", rankings_base, close, high, low, atr_dict)
    if trades_base and len(trades_base) >= 10:
        pnls_base = np.array([t.get("pnl", 0) for t in trades_base])
        base_sharpe = float(np.mean(pnls_base) / np.std(pnls_base) * np.sqrt(52)) if np.std(pnls_base) > 0 else 0.0
    else:
        base_sharpe = 0.0

    obs_delta = obs_sharpe - base_sharpe
    fprint(f"    Observed Sharpe delta: {obs_delta:+.4f} (base={base_sharpe:.4f}, aug={obs_sharpe:.4f})")

    # Permutation: shuffle new features independently
    perm_deltas = []
    rng = np.random.RandomState(42)

    for p in range(n_perms):
        df_perm = df_augmented.copy()
        for feat in new_features:
            if feat in df_perm.columns:
                # Shuffle within each date to preserve cross-sectional structure
                for dt in df_perm["date"].unique():
                    mask = df_perm["date"] == dt
                    vals = df_perm.loc[mask, feat].values.copy()
                    rng.shuffle(vals)
                    df_perm.loc[mask, feat] = vals

        rankings_perm, _ = walk_forward_lgbm_rank(
            df_perm, feature_cols_augmented, f"perm_{p}"
        )
        trades_perm, _ = simulate_trades(f"perm_{p}", rankings_perm, close, high, low, atr_dict)

        if trades_perm and len(trades_perm) >= 10:
            # Just compute Sharpe directly — no need for full adversarial validation in perm loop
            pnls = [t.get("pnl", 0) for t in trades_perm]
            pnl_arr = np.array(pnls)
            perm_sharpe = float(np.mean(pnl_arr) / np.std(pnl_arr) * np.sqrt(52)) if np.std(pnl_arr) > 0 else 0.0
            perm_deltas.append(perm_sharpe - base_sharpe)
        else:
            perm_deltas.append(0.0)

        if (p + 1) % 20 == 0:
            fprint(f"      Permutation {p + 1}/{n_perms} done")

    # P-value: fraction of permuted deltas >= observed delta
    p_value = float(np.mean([1 if pd >= obs_delta else 0 for pd in perm_deltas]))
    fprint(f"    Permutation p-value: {p_value:.4f}")
    fprint(f"    Observed delta: {obs_delta:+.4f}, "
           f"Permuted mean delta: {np.mean(perm_deltas):+.4f}, "
           f"Permuted std: {np.std(perm_deltas):.4f}")

    if p_value < 0.05:
        fprint(f"    SIGNIFICANT at 5%: Earnings features add genuine signal")
    elif p_value < 0.10:
        fprint(f"    MARGINAL significance at 10%: Weak evidence for signal")
    else:
        fprint(f"    NOT SIGNIFICANT: Earnings features may not add genuine signal")

    return {
        "p_value": round(p_value, 4),
        "obs_sharpe_delta": round(obs_delta, 4),
        "perm_mean_delta": round(float(np.mean(perm_deltas)), 4),
        "perm_std_delta": round(float(np.std(perm_deltas)), 4),
        "n_perms": n_perms,
        "base_sharpe": round(base_sharpe, 4),
        "aug_sharpe": round(obs_sharpe, 4),
    }


# ══════════════════════════════════════════════════════════════
# RANDOM BASELINE
# ══════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t0 = datetime.now()
    fprint("=" * 100)
    fprint(f"EARNINGS SECTOR FEATURES v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 100)
    fprint(f"Question: Do earnings calendar features improve LGBM sector ranking?")
    fprint(f"Hypothesis: Sectors with concentrated upcoming earnings exhibit")
    fprint(f"  different momentum characteristics that LGBM can exploit.")
    fprint()
    fprint(f"V6 config (fixed for ALL variants):")
    fprint(f"  Capital: ${CAP:.0f} | DTE: {DTE} | Spread: {SPREAD_PCT:.0f}% | "
           f"Haircut: {DEFAULT_HAIRCUT:.0%} entry only | Comm: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"  Rebalance: weekly (W-FRI) | OTM: 2% | Pairs: VIX<20 bull+bear")
    fprint(f"  Hold to expiry | Intrinsic value only | No exit haircut")
    fprint(f"  Regime filter: GRU >0.4")
    fprint(f"  Walk-forward: {WF_TRAIN_PERIODS} period sliding window")
    fprint()
    fprint(f"5 Variants:")
    fprint(f"  A: Production 21 features ONLY (baseline)")
    fprint(f"  B: 21 + 4 earnings features (25 total)")
    fprint(f"  C: 21 + best 2 earnings features (selected from B)")
    fprint(f"  D: Earnings features ONLY (4 features — control)")
    fprint(f"  E: 21 + 4 earnings + 8 interaction features (29 total)")
    fprint()

    # 1. Download sector data
    close, high, low, volume = download_data()

    # 2. Download constituent data for earnings detection
    fprint(f"\n{'=' * 100}")
    fprint("DOWNLOADING CONSTITUENT DATA FOR EARNINGS DETECTION")
    fprint(f"{'=' * 100}")
    const_close, const_volume = download_constituent_data()

    # 3. Detect earnings events
    fprint(f"\n{'=' * 100}")
    fprint("DETECTING EARNINGS EVENTS")
    fprint(f"{'=' * 100}")
    earnings_events = detect_earnings_events(const_close, const_volume)

    # Per-sector earnings summary
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

    # 6. Generate V6 rebalance dates (weekly)
    rebal_dates = generate_rebal_dates(close, V6_REBAL_FREQ)
    fprint(f"\nRebalance dates: {len(rebal_dates)} "
           f"({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

    spy_close = close["SPY"]

    # ── Define variants ──
    # Variant C features will be determined after running B
    VARIANT_FEATURES = {
        "A_baseline_21": {
            "desc": "Production 21 features (baseline)",
            "features": ALL_21_FEATURES[:],
            "needs_earnings": False,
        },
        "B_21_plus_earnings": {
            "desc": "21 + 4 earnings features (25 total)",
            "features": ALL_21_FEATURES + EARNINGS_FEATURES,
            "needs_earnings": True,
        },
        "D_earnings_only": {
            "desc": "Earnings features ONLY (4 — control)",
            "features": EARNINGS_FEATURES[:],
            "needs_earnings": True,
        },
        "E_21_plus_interactions": {
            "desc": "21 + 4 earnings + 8 interactions (29 total)",
            "features": ALL_21_FEATURES + EARNINGS_FEATURES + EARNINGS_INTERACTIONS,
            "needs_earnings": True,
        },
    }

    all_results = {}
    all_importances = {}
    all_records = {}

    # ── Run A, B, D, E first ──
    for vname, vcfg in VARIANT_FEATURES.items():
        feature_cols = vcfg["features"]
        fprint(f"\n{'=' * 100}")
        fprint(f"VARIANT {vname}: {vcfg['desc']}")
        fprint(f"  Features ({len(feature_cols)}): {feature_cols[:10]}{'...' if len(feature_cols) > 10 else ''}")
        fprint(f"{'=' * 100}")

        # Build records
        records = build_feature_records(
            close, high, low, rebal_dates, feature_cols, regime_series,
            earnings_events=earnings_events if vcfg["needs_earnings"] else None,
            const_close=const_close if vcfg["needs_earnings"] else None,
        )
        all_records[vname] = records

        # Walk-forward LGBM
        rankings, imp_df = walk_forward_lgbm_rank(records, feature_cols, vname)
        all_importances[vname] = imp_df

        if not rankings:
            fprint(f"  No rankings available, skipping")
            continue

        # Simulate trades
        trades, final_eq = simulate_trades(vname, rankings, close, high, low, atr_dict)

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

        # Extract metrics
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

    # ── Variant C: Select best 2 earnings features from B's importance ──
    fprint(f"\n{'=' * 100}")
    fprint("VARIANT C: Selecting best 2 earnings features from B's importance")
    fprint(f"{'=' * 100}")

    imp_b = all_importances.get("B_21_plus_earnings")
    if imp_b is not None:
        # Get importance of just the earnings features
        earn_imp = imp_b[imp_b["feature"].isin(EARNINGS_FEATURES)].sort_values(
            "importance", ascending=False
        )
        fprint(f"  Earnings feature importance from B:")
        for _, row in earn_imp.iterrows():
            fprint(f"    {row['feature']:<35} {row['importance']:>6.1f}")

        best_2_earnings = earn_imp.head(2)["feature"].tolist()
        fprint(f"  Selected best 2: {best_2_earnings}")

        features_c = ALL_21_FEATURES + best_2_earnings
        fprint(f"  Variant C features ({len(features_c)}): 21 base + {best_2_earnings}")

        records_c = build_feature_records(
            close, high, low, rebal_dates, features_c, regime_series,
            earnings_events=earnings_events, const_close=const_close,
        )
        rankings_c, imp_c = walk_forward_lgbm_rank(records_c, features_c, "C_best2_earnings")
        all_importances["C_best2_earnings"] = imp_c

        if rankings_c:
            trades_c, final_eq_c = simulate_trades(
                "C_best2_earnings", rankings_c, close, high, low, atr_dict
            )
            if trades_c and len(trades_c) >= 10:
                result_c = validate_trades(
                    trades_c, initial_capital=CAP,
                    spy_prices=spy_close,
                    strategy_name="C_best2_earnings",
                )
                if hasattr(result_c, 'print_summary'):
                    result_c.print_summary()

                r_dict_c = result_c.to_dict() if hasattr(result_c, 'to_dict') else result_c
                r_sharpe_c = result_c.sharpe if hasattr(result_c, 'sharpe') else r_dict_c.get("sharpe", 0)

                random_sharpes_c = random_baseline_test(rankings_c, close, high, low, atr_dict)
                mean_random_c = np.mean(random_sharpes_c)

                bull_c = [t for t in trades_c if t["direction"] == "bull"]
                bear_c = [t for t in trades_c if t["direction"] == "bear"]

                all_results["C_best2_earnings"] = {
                    "description": f"21 + best 2 earnings ({best_2_earnings})",
                    "n_features": len(features_c),
                    "feature_list": features_c,
                    **r_dict_c,
                    "random_sharpes": [round(s, 3) for s in random_sharpes_c],
                    "random_mean_sharpe": round(mean_random_c, 3),
                    "selected_earnings_features": best_2_earnings,
                    "bull_trades": len(bull_c),
                    "bear_trades": len(bear_c),
                    "bull_pnl": round(sum(t["pnl"] for t in bull_c), 2),
                    "bear_pnl": round(sum(t["pnl"] for t in bear_c), 2),
                    "bull_wr": round(sum(1 for t in bull_c if t["win"]) / max(len(bull_c), 1) * 100, 1),
                    "bear_wr": round(sum(1 for t in bear_c if t["win"]) / max(len(bear_c), 1) * 100, 1),
                }
    else:
        fprint("  Variant B produced no importance data — cannot select for C")

    # ── PERMUTATION TEST ──
    fprint(f"\n{'=' * 100}")
    fprint("PERMUTATION TEST: Are earnings features genuinely valuable?")
    fprint(f"{'=' * 100}")

    perm_result = {}
    if "A_baseline_21" in all_records and "B_21_plus_earnings" in all_records:
        perm_result = permutation_test_incremental(
            df_base=all_records["A_baseline_21"],
            df_augmented=all_records["B_21_plus_earnings"],
            feature_cols_base=ALL_21_FEATURES,
            feature_cols_augmented=ALL_21_FEATURES + EARNINGS_FEATURES,
            close=close, high=high, low=low, atr_dict=atr_dict,
            n_perms=50,  # 50 perms for reasonable runtime
            variant_name="earnings_incremental",
        )

    # ── SUMMARY COMPARISON ──
    fprint(f"\n{'=' * 120}")
    fprint("EARNINGS SECTOR FEATURES — SUMMARY COMPARISON")
    fprint(f"{'=' * 120}")
    fprint(f"{'Variant':<30} {'#Feat':>5} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} "
           f"{'PF':>6} {'MaxDD':>7} {'Gates':>6} {'Final$':>8} {'RandSh':>7}")
    fprint("-" * 120)

    baseline_sharpe = all_results.get("A_baseline_21", {}).get("sharpe", 0)

    variant_order = ["A_baseline_21", "B_21_plus_earnings", "C_best2_earnings",
                     "D_earnings_only", "E_21_plus_interactions"]

    for vname in variant_order:
        r = all_results.get(vname)
        if not r:
            fprint(f"  {vname:<30} — NO DATA —")
            continue
        _wr = r.get("win_rate", r.get("wr", 0))
        _pf = r.get("profit_factor", r.get("pf", 0))
        _mdd = r.get("max_dd", 0)
        _gp = r.get("gates_passed", 0)
        _gt = r.get("gates_total", 5)
        fprint(f"  {vname:<30} {r['n_features']:>4} {r.get('n_trades', 0):>5} {r['sharpe']:>7.2f} "
               f"{r['sortino']:>8.2f} {_wr*100:>5.1f}% {_pf:>5.2f} "
               f"{_mdd*100:>6.1f}% {_gp}/{_gt} "
               f"${r.get('final_equity', 0):>7,.0f} {r['random_mean_sharpe']:>7.2f}")

    # ── SHARPE DELTA ANALYSIS ──
    fprint(f"\n{'=' * 100}")
    fprint("SHARPE DELTA vs BASELINE (A_baseline_21)")
    fprint(f"{'=' * 100}")
    fprint(f"  Baseline (A_baseline_21) Sharpe: {baseline_sharpe:.2f}")
    fprint()

    deltas = []
    for vname in variant_order:
        if vname == "A_baseline_21":
            continue
        r = all_results.get(vname)
        if not r:
            continue
        delta = r["sharpe"] - baseline_sharpe
        pct_delta = delta / abs(baseline_sharpe) * 100 if baseline_sharpe != 0 else 0
        deltas.append((vname, delta, pct_delta, r["sharpe"], r.get("n_features", 0)))

    deltas.sort(key=lambda x: x[1], reverse=True)
    max_abs_delta = max(abs(d[1]) for d in deltas) if deltas else 1

    for vname, delta, pct_delta, sharpe, nf in deltas:
        bar = "*" * int(abs(delta) / max_abs_delta * 30) if max_abs_delta > 0 else ""
        sign = "BETTER" if delta > 0.1 else "WORSE" if delta < -0.1 else "SIMILAR"
        fprint(f"  {vname:<30} {nf:>2}f  Sharpe {sharpe:>5.2f}  "
               f"delta {delta:>+5.2f} ({pct_delta:>+5.1f}%)  "
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

    # Earnings-only control
    d_result = all_results.get("D_earnings_only")
    if d_result:
        d_sharpe = d_result.get("sharpe", 0)
        if d_sharpe > 0.5:
            fprint(f"\n  Earnings-only (D) has standalone signal: Sharpe {d_sharpe:.2f}")
        else:
            fprint(f"\n  Earnings-only (D) has NO standalone signal: Sharpe {d_sharpe:.2f}")

    # Permutation test conclusion
    if perm_result:
        fprint(f"\n  Permutation test p-value: {perm_result.get('p_value', 'N/A')}")
        if perm_result.get("p_value", 1.0) < 0.05:
            fprint(f"  SIGNIFICANT: Earnings features add genuine signal (p<0.05)")
        elif perm_result.get("p_value", 1.0) < 0.10:
            fprint(f"  MARGINALLY significant (p<0.10)")
        else:
            fprint(f"  NOT significant: Earnings features may be noise")

    # Feature importance for B
    imp_b = all_importances.get("B_21_plus_earnings")
    if imp_b is not None:
        fprint(f"\n  Feature importance (B: 21+earnings, top 10):")
        for _, row in imp_b.head(10).iterrows():
            bar = "*" * int(row["importance"] / imp_b["importance"].max() * 25)
            marker = " <== EARNINGS" if row["feature"] in EARNINGS_FEATURES else ""
            fprint(f"    {row['feature']:<35} {row['importance']:>6.1f} {bar}{marker}")

    # Feature importance for E (interactions)
    imp_e = all_importances.get("E_21_plus_interactions")
    if imp_e is not None:
        fprint(f"\n  Feature importance (E: interactions, top 10):")
        for _, row in imp_e.head(10).iterrows():
            bar = "*" * int(row["importance"] / imp_e["importance"].max() * 25)
            marker = " <== NEW" if row["feature"] in EARNINGS_FEATURES + EARNINGS_INTERACTIONS else ""
            fprint(f"    {row['feature']:<35} {row['importance']:>6.1f} {bar}{marker}")

    # Research conclusion
    fprint(f"\n{'=' * 100}")
    fprint("RESEARCH CONCLUSION")
    fprint(f"{'=' * 100}")

    if deltas:
        best_delta = max(d[1] for d in deltas)
        p_val = perm_result.get("p_value", 1.0) if perm_result else 1.0

        if best_delta > 0.3 and p_val < 0.10:
            fprint(f"  POSITIVE: Earnings calendar features IMPROVE sector ranking.")
            fprint(f"  Best variant: {best[0]} (delta {best[1]:+.2f}, p={p_val:.3f})")
            fprint(f"  Recommendation: Add earnings features to V7 config.")
        elif best_delta > 0:
            fprint(f"  MARGINAL: Earnings features show small improvement ({best_delta:+.2f} Sharpe).")
            if p_val >= 0.10:
                fprint(f"  Permutation test NOT significant (p={p_val:.3f}).")
            fprint(f"  Recommendation: Run extended OOS before production.")
        else:
            fprint(f"  NEGATIVE: Earnings features do NOT improve sector ranking.")
            fprint(f"  All variants underperform or match baseline.")
            fprint(f"  Recommendation: Keep production 21-feature set. Earnings adds noise.")

    # Save results
    results_path = OUTPUT_DIR / "earnings_sector_features_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    # Save permutation results
    if perm_result:
        perm_path = OUTPUT_DIR / "permutation_test_results.json"
        with open(perm_path, "w") as f:
            json.dump(perm_result, f, indent=2)
        fprint(f"Permutation test results saved to {perm_path}")

    # MLflow logging
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"earn_feat_{t0.strftime('%Y%m%d_%H%M')}"):
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
                    mlflow.log_metric(f"{prefix}_bull_wr", r.get("bull_wr", 0))
                    mlflow.log_metric(f"{prefix}_bear_wr", r.get("bear_wr", 0))

                    if vname != "A_baseline_21":
                        delta = r.get("sharpe", 0) - baseline_sharpe
                        mlflow.log_metric(f"{prefix}_sharpe_delta", delta)

                # Log permutation test results
                if perm_result:
                    mlflow.log_metric("perm_p_value", perm_result.get("p_value", 1.0))
                    mlflow.log_metric("perm_obs_delta", perm_result.get("obs_sharpe_delta", 0))

                mlflow.log_params({
                    "experiment_type": "earnings_sector_features",
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
                    "n_variants": 5,
                    "n_earnings_features": len(EARNINGS_FEATURES),
                    "n_interaction_features": len(EARNINGS_INTERACTIONS),
                    "earnings_detection": "volume_spike_gap",
                    "vol_mult": 3.0,
                    "gap_thresh": 0.03,
                    "perm_test_n": perm_result.get("n_perms", 0) if perm_result else 0,
                })

                mlflow.log_artifact(str(results_path))
                if perm_result:
                    perm_path = OUTPUT_DIR / "permutation_test_results.json"
                    if perm_path.exists():
                        mlflow.log_artifact(str(perm_path))
            fprint(f"MLflow run logged to experiment '{EXPERIMENT_NAME}'")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    fprint("DONE")


if __name__ == "__main__":
    main()
