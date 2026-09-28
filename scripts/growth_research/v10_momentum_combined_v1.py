#!/usr/bin/env python3
"""
V10 + Momentum Burst Combined Strategy v1
==========================================

HYPOTHESIS: V10's LGBM sector rankings (Sharpe 6.30 with spreads, needs Level 3)
can improve momentum burst timing (Sharpe 1.28 with single-leg, works Level 2).

APPROACH:
  V10 picks WHICH sectors to trade (monthly LGBM ranking).
  Momentum burst signals decide WHEN to enter (daily scan).
  Single-leg options (calls/puts) for Level 2 compatibility.

SIX VARIANTS:
  A. V10 top-2 + bottom-2 with momentum burst (baseline combined)
  B. V10 top-3 + bottom-3 with momentum burst (wider net)
  C. V10 top-2 only (calls only, bull-only + momentum timing)
  D. V10 bottom-2 only (puts only, bear-only + momentum timing)
  E. V10 all ranks: weight confidence by rank (top-1 full, top-3 half, etc.)
  F. V10 + earnings catalyst: only enter if sector has earnings stock this week

MOMENTUM BURST ENTRY (proven from momentum_v1):
  Entry requires 2+ of:
    1. 5-day momentum > 3%
    2. Relative strength vs SPY > 2% over 10 days
    3. RSI crossed above 60 (calls) or below 40 (puts) within last 3 days
    4. Volume surge > 1.5x 20-day average

EXIT RULES (Variant F from momentum_v1, proven Sharpe 1.28):
  - Take profit: +30%
  - Stop loss: -25%
  - Trailing stop: exit if gives back 50% of peak unrealized gain
  - Time stop: 5 trading days max

PRICING: Inline Black-Scholes, self-contained.
  IV = max(VIX/100, realized_vol_21d * 1.2)
  Commission: $0.65/leg ($1.30 RT)
  Account: $645, max position min($200, 30% equity)
  DTE = 14

LGBM FEATURES (17, same as V10):
  ret_5d, ret_10d, ret_21d, ret_63d, ret_126d, ret_252d, vol_21d, vol_63d,
  sharpe_63d, maxdd_63d, pct_52w_high, mom_accel, pct_pos_months_12m,
  sortino_63d, calmar_1y, trend_r2_63d, trend_slope_63d

Walk-forward: 500-day sliding window for LGBM. Monthly ranking updates.
DATA: yfinance, 2020-01-01 to 2026-07-25. All 11 sector ETFs + SPY + ^VIX.

VALIDATION per variant:
  1. Sharpe, Sortino, WR, PF, MaxDD
  2. Permutation test (100 shuffles)
  3. Regime analysis (green/red SPY days)
  4. Compare to pure momentum burst (is V10 ranking adding value?)

Output: output/growth_research/v10_momentum_combined/
MLflow experiment: v10_momentum_combined
"""

import json
import os
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime
from math import exp, log, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

warnings.filterwarnings("ignore")

_builtin_print = print


def fprint(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()


# ----------------------------------------------------------------
# Environment detection (Jupiter / Neptune)
# ----------------------------------------------------------------
_JUPITER_BASE = Path("/home/jupiter/Lvl3Quant")
_NEPTUNE_BASE = Path("/home/nick/Lvl3Quant")

if _NEPTUNE_BASE.exists():
    BASE = _NEPTUNE_BASE
    fprint(f"Running on Neptune: {BASE}")
else:
    BASE = _JUPITER_BASE
    fprint(f"Running on Jupiter: {BASE}")

OUTPUT_DIR = BASE / "output" / "growth_research" / "v10_momentum_combined"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------
# Constants
# ----------------------------------------------------------------
SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX"]

CAP = 645.0
COMMISSION_PER_LEG = 0.65
COMMISSION_RT = 1.30
DTE = 14
RISK_FREE_RATE = 0.05
MAX_POS_DOLLAR = 200.0
MAX_POS_PCT = 0.30
MAX_POSITIONS = 2

WF_TRAIN_DAYS = 500
REBAL_FREQ = 20  # LGBM ranking update every 20 trading days (~monthly)
FWD_HORIZON = 14

N_PERMUTATIONS = 100

START_DATE = "2020-01-01"
END_DATE = "2026-07-25"

MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "v10_momentum_combined"

FEATURES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_21d", "vol_63d", "sharpe_63d", "maxdd_63d", "pct_52w_high",
    "mom_accel", "pct_pos_months_12m", "sortino_63d", "calmar_1y",
    "trend_r2_63d", "trend_slope_63d",
]
assert len(FEATURES) == 17

# Earnings-heavy tickers per sector (for Variant F catalyst filter)
SECTOR_EARNINGS_TICKERS = {
    "XLK": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL"],
    "XLF": ["JPM", "BAC", "WFC", "GS", "MS"],
    "XLE": ["XOM", "CVX", "COP", "SLB", "EOG"],
    "XLV": ["UNH", "JNJ", "LLY", "PFE", "ABBV"],
    "XLY": ["AMZN", "TSLA", "HD", "MCD", "NKE"],
    "XLP": ["PG", "KO", "PEP", "COST", "WMT"],
    "XLI": ["CAT", "HON", "UNP", "BA", "RTX"],
    "XLB": ["LIN", "APD", "SHW", "ECL", "FCX"],
    "XLU": ["NEE", "DUK", "SO", "D", "AEP"],
    "XLRE": ["PLD", "AMT", "CCI", "EQIX", "SPG"],
    "XLC": ["META", "GOOG", "NFLX", "DIS", "CMCSA"],
}

# MLflow setup
MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen(MLFLOW_URI, timeout=3)
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    MLFLOW_OK = True
    fprint(f"MLflow connected: {MLFLOW_URI}")
except Exception:
    fprint("MLflow unavailable -- will skip logging")


# ================================================================
# BLACK-SCHOLES PRICING (inline, self-contained)
# ================================================================

def bs_d1(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    return (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))


def bs_d2(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
    return bs_d1(S, K, T, r, sigma) - sigma * sqrt(T)


def bs_call_price(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 1e-10:
        return max(S - K * exp(-r * T), 0.0)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * sqrt(T)
    return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)


def bs_put_price(S, K, T, r, sigma):
    if T <= 0:
        return max(K - S, 0.0)
    if sigma <= 1e-10:
        return max(K * exp(-r * T) - S, 0.0)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * sqrt(T)
    return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def option_price(S, K, T, r, sigma, option_type="call"):
    if option_type == "call":
        return bs_call_price(S, K, T, r, sigma)
    else:
        return bs_put_price(S, K, T, r, sigma)


def compute_iv(vix_val, realized_vol_21d):
    """IV = max(VIX/100, realized_vol_21d * 1.2)."""
    vix_iv = vix_val / 100.0 if vix_val is not None and vix_val > 0 else 0.20
    rv_iv = realized_vol_21d * 1.2 if realized_vol_21d is not None and realized_vol_21d > 0 else 0.20
    return max(vix_iv, rv_iv)


# ================================================================
# DATA DOWNLOAD
# ================================================================

def download_data():
    """Download daily close for sectors + SPY + VIX via yfinance with caching."""
    cache_path = BASE / "data" / "v10_momentum_combined_cache.parquet"

    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            if len(df) > 100:
                latest = df.index.max()
                if pd.Timestamp(latest) >= pd.Timestamp("2026-07-20"):
                    fprint(f"Loaded cached data: {len(df)} rows, latest={latest.date()}")
                    return df
        except Exception:
            pass

    import yfinance as yf

    all_tickers = SECTORS + EXTRA_TICKERS
    fprint(f"Downloading {len(all_tickers)} tickers...")
    raw = yf.download(all_tickers, start=START_DATE, end=END_DATE,
                      progress=False, auto_adjust=True)

    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw["Close"] if mi else raw[["Close"]]
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
    close = close.ffill()

    # Also grab volume for momentum signals
    volume = raw["Volume"] if mi else raw[["Volume"]]
    if isinstance(volume.columns, pd.MultiIndex):
        volume.columns = volume.columns.get_level_values(-1)
    volume = volume.fillna(0)

    close = close.rename(columns={"^VIX": "VIX"})
    volume = volume.rename(columns={"^VIX": "VIX"})

    # Save close as cache (primary)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        close.to_parquet(cache_path)
    except Exception:
        pass

    # Also save volume separately
    vol_cache = BASE / "data" / "v10_momentum_combined_volume_cache.parquet"
    try:
        volume.to_parquet(vol_cache)
    except Exception:
        pass

    fprint(f"Data: {len(close)} days, {close.index[0].date()} to {close.index[-1].date()}")
    return close


def load_volume_data():
    """Load volume data (separate from close for caching)."""
    vol_cache = BASE / "data" / "v10_momentum_combined_volume_cache.parquet"
    if vol_cache.exists():
        try:
            return pd.read_parquet(vol_cache)
        except Exception:
            pass

    import yfinance as yf
    all_tickers = SECTORS + EXTRA_TICKERS
    raw = yf.download(all_tickers, start=START_DATE, end=END_DATE,
                      progress=False, auto_adjust=True)
    mi = isinstance(raw.columns, pd.MultiIndex)
    volume = raw["Volume"] if mi else raw[["Volume"]]
    if isinstance(volume.columns, pd.MultiIndex):
        volume.columns = volume.columns.get_level_values(-1)
    volume = volume.fillna(0).rename(columns={"^VIX": "VIX"})
    try:
        vol_cache.parent.mkdir(parents=True, exist_ok=True)
        volume.to_parquet(vol_cache)
    except Exception:
        pass
    return volume


def load_earnings_dates():
    """Load earnings dates for sector catalyst stocks.

    Returns dict: {ticker: set of dates with earnings within that week}.
    We use yfinance earnings_dates where available, falling back to quarterly
    approximation (mid-Jan, mid-Apr, mid-Jul, mid-Oct).
    """
    earnings_cache = BASE / "data" / "v10_momentum_combined_earnings_cache.json"
    if earnings_cache.exists():
        try:
            with open(earnings_cache) as f:
                raw = json.load(f)
            result = {}
            for tk, dates_str in raw.items():
                result[tk] = set(pd.Timestamp(d) for d in dates_str)
            fprint(f"Loaded cached earnings dates for {len(result)} tickers")
            return result
        except Exception:
            pass

    fprint("Building earnings dates (quarterly approximation)...")
    all_earn_tickers = set()
    for sector_tickers in SECTOR_EARNINGS_TICKERS.values():
        all_earn_tickers.update(sector_tickers)

    # Generate approximate quarterly earnings dates 2020-2026
    result = {}
    for tk in all_earn_tickers:
        dates = set()
        for year in range(2020, 2027):
            for month in [1, 4, 7, 10]:
                # Approximate mid-month earnings
                for day in [15, 16, 17, 18, 19, 20, 21, 22]:
                    try:
                        d = pd.Timestamp(year=year, month=month, day=day)
                        if d.weekday() < 5:  # weekday
                            dates.add(d)
                            break
                    except Exception:
                        continue
        result[tk] = dates

    # Try yfinance for real dates (best-effort, fast timeout)
    try:
        import yfinance as yf
        for tk in list(all_earn_tickers)[:10]:  # limit to avoid slowdown
            try:
                stock = yf.Ticker(tk)
                ed = stock.earnings_dates
                if ed is not None and len(ed) > 0:
                    result[tk] = set(pd.Timestamp(d.date()) for d in ed.index)
            except Exception:
                continue
    except Exception:
        pass

    # Cache
    try:
        earnings_cache.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {tk: [str(d.date()) for d in dates] for tk, dates in result.items()}
        with open(earnings_cache, "w") as f:
            json.dump(cache_data, f, indent=2)
    except Exception:
        pass

    fprint(f"  Earnings dates built for {len(result)} tickers")
    return result


def sector_has_earnings_this_week(sector, date, earnings_dates):
    """Check if any earnings-heavy stock in this sector reports within 5 trading days."""
    tickers = SECTOR_EARNINGS_TICKERS.get(sector, [])
    for tk in tickers:
        if tk not in earnings_dates:
            continue
        for earn_date in earnings_dates[tk]:
            delta = (earn_date - date).days
            if 0 <= delta <= 7:  # within next calendar week
                return True
    return False


# ================================================================
# LGBM FEATURE ENGINEERING (17 features, same as V10)
# ================================================================

def compute_lgbm_features(px, spy_px=None):
    """Compute 17 momentum/quality features for a single sector series ending at current bar."""
    if len(px) < 260:
        return None

    f = {}
    for lb, nm in [(5, "ret_5d"), (10, "ret_10d"), (21, "ret_21d"),
                   (63, "ret_63d"), (126, "ret_126d"), (252, "ret_252d")]:
        f[nm] = float(px.iloc[-1] / px.iloc[-lb] - 1) if len(px) > lb else 0.0

    rets = px.pct_change().dropna()

    f["vol_21d"] = float(rets.iloc[-21:].std() * np.sqrt(252)) if len(rets) > 21 else 0.20
    f["vol_63d"] = float(rets.iloc[-63:].std() * np.sqrt(252)) if len(rets) > 63 else 0.20

    f["sharpe_63d"] = float(
        rets.iloc[-63:].mean() / (rets.iloc[-63:].std() + 1e-10) * np.sqrt(252)
    ) if len(rets) > 63 else 0.0

    if len(px) >= 63:
        pk = px.iloc[-63:].cummax()
        f["maxdd_63d"] = float(((px.iloc[-63:] / pk) - 1).min())
    else:
        f["maxdd_63d"] = 0.0

    f["pct_52w_high"] = float(px.iloc[-1] / px.iloc[-252:].max()) if len(px) >= 252 else 1.0
    f["mom_accel"] = f["ret_21d"] - f["ret_63d"] / 3

    monthly = rets.resample("ME").sum()
    f["pct_pos_months_12m"] = float((monthly.iloc[-12:] > 0).mean()) if len(monthly) >= 12 else 0.5

    dr = rets.iloc[-63:][rets.iloc[-63:] < 0]
    f["sortino_63d"] = float(
        rets.iloc[-63:].mean() / (dr.std() + 1e-10) * np.sqrt(252)
    ) if len(dr) > 3 else 0.0

    if len(px) >= 252:
        pk252 = px.iloc[-252:].cummax()
        mdd252 = float(((px.iloc[-252:] / pk252) - 1).min())
        cagr = float(px.iloc[-1] / px.iloc[-252] - 1)
        f["calmar_1y"] = cagr / (abs(mdd252) + 1e-10)
    else:
        f["calmar_1y"] = 0.0

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


# ================================================================
# LGBM WALK-FORWARD RANKING (sliding 500-day window)
# ================================================================

def build_feature_matrix(close, rebal_indices):
    """Build feature matrix for all rebalance dates and all sectors."""
    records = []
    spy = close["SPY"]
    sector_cols = [c for c in SECTORS if c in close.columns]

    for di in rebal_indices:
        dt = close.index[di]
        for tk in sector_cols:
            px = close[tk].iloc[:di + 1].dropna()
            spy_px = spy.iloc[:di + 1].dropna()
            feats = compute_lgbm_features(px, spy_px)
            if feats is None:
                continue
            fi = min(di + FWD_HORIZON, len(close) - 1)
            if fi <= di:
                continue
            fwd_ret = float(close[tk].iloc[fi] / close[tk].iloc[di] - 1)
            rec = {**feats, "date_idx": di, "date": dt, "ticker": tk, "fwd_ret": fwd_ret}
            records.append(rec)

    df = pd.DataFrame(records)
    for c in FEATURES:
        if c not in df.columns:
            df[c] = 0.0
    df[FEATURES] = df[FEATURES].fillna(0.0)
    fprint(f"  Feature matrix: {len(df)} records, {len(df['date_idx'].unique())} rebalance dates")
    return df


def walk_forward_lgbm_rank(df):
    """Walk-forward LGBM ranking with 500-day sliding window.

    Returns: {date_idx: {ticker: predicted_score}} for each OOT rebalance date.
    """
    import lightgbm as lgb

    if len(df) < 50:
        fprint("  WARNING: Too few records for walk-forward")
        return {}

    df["rank_label"] = df.groupby("date_idx")["fwd_ret"].rank(pct=True)
    date_indices = sorted(df["date_idx"].unique())

    train_periods = WF_TRAIN_DAYS // REBAL_FREQ  # 25

    rankings = {}
    for i in range(train_periods, len(date_indices)):
        train_di_list = date_indices[max(0, i - train_periods):i]
        test_di = date_indices[i]

        train_df = df[df["date_idx"].isin(train_di_list)]
        test_df = df[df["date_idx"] == test_di].copy()

        if len(test_df) < 3 or len(train_df) < 30:
            continue

        Xt = np.nan_to_num(train_df[FEATURES].values.astype(np.float32))
        yt = train_df["rank_label"].values.astype(np.float32)
        Xe = np.nan_to_num(test_df[FEATURES].values.astype(np.float32))

        try:
            m = lgb.LGBMRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
                verbose=-1,
            )
            m.fit(Xt, yt)
            preds = m.predict(Xe)
            test_df["score"] = preds
            rankings[test_di] = dict(zip(test_df["ticker"], test_df["score"]))
        except Exception as e:
            fprint(f"  LGBM error at idx {test_di}: {e}")
            continue

    fprint(f"  Walk-forward produced {len(rankings)} ranking dates")
    return rankings


def get_active_ranking(di, rankings):
    """Get the most recent ranking that is still active at date index di.

    Rankings update monthly (~every 20 trading days). Between updates, use the
    most recent ranking.
    """
    ranked_dates = sorted(rankings.keys())
    active = None
    for rd in ranked_dates:
        if rd <= di:
            active = rd
        else:
            break
    if active is not None:
        return rankings[active]
    return None


def classify_sector(ticker, ranking, variant_cfg):
    """Classify a sector as top/bottom/mid based on the current LGBM ranking.

    Returns:
        ("call", size_multiplier) for top-ranked sectors
        ("put", size_multiplier) for bottom-ranked sectors
        None for mid-ranked sectors (don't trade)
    """
    if ranking is None:
        return None

    sorted_sectors = sorted(ranking.items(), key=lambda x: x[1], reverse=True)
    n = len(sorted_sectors)
    ticker_rank = None
    for i, (tk, score) in enumerate(sorted_sectors):
        if tk == ticker:
            ticker_rank = i
            break

    if ticker_rank is None:
        return None

    top_k = variant_cfg["top_k"]
    bottom_k = variant_cfg["bottom_k"]
    weighted = variant_cfg.get("weighted", False)

    if weighted:
        # Variant E: continuous weighting by rank
        if ticker_rank == 0:
            return ("call", 1.0)
        elif ticker_rank == 1:
            return ("call", 0.75)
        elif ticker_rank == 2:
            return ("call", 0.50)
        elif ticker_rank == n - 1:
            return ("put", 1.0)
        elif ticker_rank == n - 2:
            return ("put", 0.75)
        elif ticker_rank == n - 3:
            return ("put", 0.50)
        else:
            return None  # mid-ranked, skip

    # Standard top-k / bottom-k
    if top_k > 0 and ticker_rank < top_k:
        return ("call", 1.0)
    if bottom_k > 0 and ticker_rank >= n - bottom_k:
        return ("put", 1.0)

    return None  # mid-ranked, skip


# ================================================================
# MOMENTUM BURST SIGNALS (daily, proven from momentum_v1)
# ================================================================

def compute_rsi(prices, period=14):
    """RSI over the last `period` daily returns."""
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def check_momentum_burst(close_arr, volume_arr, spy_close_arr, direction):
    """Check if momentum burst entry conditions are met.

    direction: "call" or "put" (determines which signals count).

    Returns (triggered: bool, signal_count: int).
    Triggered if signal_count >= 2.
    """
    if len(close_arr) < 25 or len(spy_close_arr) < 25:
        return False, 0

    signals = 0

    # Signal 1: 5-day momentum > 3%
    mom_5d = (close_arr[-1] / close_arr[-6]) - 1.0
    if direction == "call" and mom_5d > 0.03:
        signals += 1
    elif direction == "put" and mom_5d < -0.03:
        signals += 1

    # Signal 2: Relative strength vs SPY > 2% over 10 days
    if len(close_arr) >= 11 and len(spy_close_arr) >= 11:
        etf_ret_10d = (close_arr[-1] / close_arr[-11]) - 1.0
        spy_ret_10d = (spy_close_arr[-1] / spy_close_arr[-11]) - 1.0
        rel_strength = etf_ret_10d - spy_ret_10d
        if direction == "call" and rel_strength > 0.02:
            signals += 1
        elif direction == "put" and rel_strength < -0.02:
            signals += 1

    # Signal 3: RSI cross within last 3 days
    if len(close_arr) >= 18:
        rsi_today = compute_rsi(close_arr)
        rsi_3d_ago = compute_rsi(close_arr[:-3]) if len(close_arr) > 20 else rsi_today
        if direction == "call" and rsi_today >= 60 and rsi_3d_ago < 60:
            signals += 1
        elif direction == "put" and rsi_today <= 40 and rsi_3d_ago > 40:
            signals += 1

    # Signal 4: Volume surge > 1.5x 20-day average
    if len(volume_arr) >= 21 and volume_arr[-1] > 0:
        avg_vol_20 = np.mean(volume_arr[-21:-1])
        if avg_vol_20 > 0:
            vol_ratio = volume_arr[-1] / avg_vol_20
            if vol_ratio > 1.5:
                # Volume supports momentum direction
                if direction == "call" and mom_5d > 0:
                    signals += 1
                elif direction == "put" and mom_5d < 0:
                    signals += 1

    return signals >= 2, signals


# ================================================================
# POSITION MANAGEMENT
# ================================================================

class Position:
    """Tracks an open single-leg option position."""

    def __init__(self, ticker, option_type, strike, entry_premium, entry_date_idx,
                 entry_spot, dte, iv, total_cost, n_contracts, size_mult=1.0):
        self.ticker = ticker
        self.option_type = option_type  # "call" or "put"
        self.strike = strike
        self.entry_premium = entry_premium  # per-share BS price at entry
        self.entry_date_idx = entry_date_idx
        self.entry_spot = entry_spot
        self.dte = dte
        self.iv = iv
        self.total_cost = total_cost
        self.n_contracts = n_contracts
        self.size_mult = size_mult
        self.days_held = 0
        self.peak_value = entry_premium  # for trailing stop


# ================================================================
# VARIANT DEFINITIONS
# ================================================================

VARIANT_CONFIGS = {
    "A_top2_bot2_momentum": {
        "description": "V10 top-2 + bottom-2 with momentum burst (baseline combined)",
        "top_k": 2,
        "bottom_k": 2,
        "weighted": False,
        "earnings_filter": False,
    },
    "B_top3_bot3_momentum": {
        "description": "V10 top-3 + bottom-3 with momentum burst (wider net)",
        "top_k": 3,
        "bottom_k": 3,
        "weighted": False,
        "earnings_filter": False,
    },
    "C_top2_calls_only": {
        "description": "V10 top-2 only (calls only, bull-only + momentum timing)",
        "top_k": 2,
        "bottom_k": 0,
        "weighted": False,
        "earnings_filter": False,
    },
    "D_bot2_puts_only": {
        "description": "V10 bottom-2 only (puts only, bear-only + momentum timing)",
        "top_k": 0,
        "bottom_k": 2,
        "weighted": False,
        "earnings_filter": False,
    },
    "E_weighted_all_ranks": {
        "description": "V10 all ranks: weight confidence by rank (top-1=full, top-3=half)",
        "top_k": 3,
        "bottom_k": 3,
        "weighted": True,
        "earnings_filter": False,
    },
    "F_earnings_catalyst": {
        "description": "V10 top-2/bot-2 + momentum + earnings catalyst this week",
        "top_k": 2,
        "bottom_k": 2,
        "weighted": False,
        "earnings_filter": True,
    },
}

# Also define a pure momentum baseline (no V10 filtering) for comparison
PURE_MOMENTUM_CONFIG = {
    "description": "Pure momentum burst (no V10 ranking, all sectors eligible)",
    "top_k": 0,
    "bottom_k": 0,
    "weighted": False,
    "earnings_filter": False,
}

# Exit parameters (proven Variant F from momentum_v1)
EXIT_PARAMS = {
    "tp_pct": 0.30,       # +30% take profit
    "sl_pct": 0.25,       # -25% stop loss
    "trailing_giveback": 0.50,  # exit if gives back 50% of peak unrealized
    "time_stop_days": 5,  # max 5 trading days
}


# ================================================================
# BACKTEST ENGINE
# ================================================================

def run_backtest(variant_name, variant_cfg, close, volume, vix_series, rv_series,
                 rankings, earnings_dates, spy_close):
    """Run a single variant backtest.

    Combines V10 LGBM sector rankings with momentum burst entry timing.
    Uses single-leg options (calls/puts) with proven exit rules.
    """
    fprint(f"\n{'=' * 60}")
    fprint(f"VARIANT {variant_name}: {variant_cfg['description']}")
    fprint(f"{'=' * 60}")

    is_pure_momentum = (variant_cfg["top_k"] == 0 and variant_cfg["bottom_k"] == 0)
    earnings_filter = variant_cfg.get("earnings_filter", False)

    equity = CAP
    equity_curve = [{"date": str(close.index[0].date()), "equity": equity}]
    positions = []
    all_trades = []
    daily_returns = []

    sector_cols = [c for c in SECTORS if c in close.columns]

    # Start after enough lookback for features (260 days) + WF warmup
    start_idx = 260 + WF_TRAIN_DAYS // REBAL_FREQ * REBAL_FREQ if not is_pure_momentum else 260

    # For pure momentum, start after basic lookback
    if is_pure_momentum:
        start_idx = 30

    # Ensure we don't start before rankings are available
    if not is_pure_momentum and rankings:
        first_ranked = min(rankings.keys())
        start_idx = max(start_idx, first_ranked)

    for di in range(start_idx, len(close)):
        dt = close.index[di]
        prev_equity = equity

        # --- Mark-to-market and manage exits for open positions ---
        to_close = []
        for pidx, pos in enumerate(positions):
            pos.days_held += 1

            if pos.ticker not in close.columns or di >= len(close):
                continue

            S_now = float(close[pos.ticker].iloc[di])
            remaining_dte = max(pos.dte - pos.days_held, 0)
            T_rem = remaining_dte / 365.0

            # IV update: slight mean-reversion toward current VIX
            vix_now = float(vix_series.iloc[di]) if di < len(vix_series) else 20.0
            rv_now = float(rv_series[pos.ticker].iloc[di]) if (
                pos.ticker in rv_series and di < len(rv_series[pos.ticker])
            ) else 0.20
            iv_now = compute_iv(vix_now, rv_now)

            current_val = option_price(S_now, pos.strike, T_rem, RISK_FREE_RATE,
                                       iv_now, pos.option_type)

            if current_val > pos.peak_value:
                pos.peak_value = current_val

            # --- Exit logic ---
            exit_reason = None
            pct_change = (current_val - pos.entry_premium) / (pos.entry_premium + 1e-10)

            # Take profit
            if pct_change >= EXIT_PARAMS["tp_pct"]:
                exit_reason = "take_profit"

            # Stop loss
            elif pct_change <= -EXIT_PARAMS["sl_pct"]:
                exit_reason = "stop_loss"

            # Time stop
            elif pos.days_held >= EXIT_PARAMS["time_stop_days"]:
                exit_reason = "time_stop"

            # Trailing stop: exit if gives back 50% of peak unrealized gain
            elif pos.peak_value > pos.entry_premium:
                unrealized_peak = pos.peak_value - pos.entry_premium
                giveback = pos.peak_value - current_val
                if giveback > unrealized_peak * EXIT_PARAMS["trailing_giveback"]:
                    exit_reason = "trailing_stop"

            if exit_reason:
                exit_proceeds = current_val * 100.0 * pos.n_contracts
                exit_commission = pos.n_contracts * COMMISSION_PER_LEG if current_val > 0 else 0.0
                pnl = exit_proceeds - pos.total_cost - exit_commission

                equity += pnl
                equity = max(equity, 1.0)

                all_trades.append({
                    "ticker": pos.ticker,
                    "type": pos.option_type,
                    "strike": round(pos.strike, 2),
                    "entry_date": str(close.index[pos.entry_date_idx].date()),
                    "exit_date": str(dt.date()),
                    "entry_idx": pos.entry_date_idx,
                    "exit_idx": di,
                    "entry_premium": round(pos.entry_premium, 4),
                    "exit_premium": round(current_val, 4),
                    "days_held": pos.days_held,
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pct_change * 100, 1),
                    "exit_reason": exit_reason,
                    "size_mult": pos.size_mult,
                    "equity_after": round(equity, 2),
                })
                to_close.append(pidx)

        for idx in sorted(to_close, reverse=True):
            positions.pop(idx)

        # --- Scan for new entries ---
        if len(positions) < MAX_POSITIONS and equity > 50:
            ranking = get_active_ranking(di, rankings) if not is_pure_momentum else None

            candidates = []
            for tk in sector_cols:
                if tk in {p.ticker for p in positions}:
                    continue  # already holding

                # V10 sector classification
                if is_pure_momentum:
                    # Pure momentum: consider both directions for all sectors
                    directions_to_check = ["call", "put"]
                    size_mult = 1.0
                else:
                    classification = classify_sector(tk, ranking, variant_cfg)
                    if classification is None:
                        continue  # mid-ranked, skip
                    direction, size_mult = classification
                    directions_to_check = [direction]

                # Earnings filter (Variant F)
                if earnings_filter and not sector_has_earnings_this_week(tk, dt, earnings_dates):
                    continue

                # Get price/volume arrays for momentum signals
                close_arr = close[tk].iloc[:di + 1].values
                vol_arr = volume[tk].iloc[:di + 1].values if tk in volume.columns else np.zeros(di + 1)
                spy_arr = spy_close.iloc[:di + 1].values

                for direction in directions_to_check:
                    triggered, sig_count = check_momentum_burst(
                        close_arr, vol_arr, spy_arr, direction
                    )
                    if not triggered:
                        continue

                    if is_pure_momentum:
                        size_mult = 1.0

                    candidates.append((tk, direction, sig_count, size_mult))

            # Sort by signal count (strongest momentum first)
            candidates.sort(key=lambda x: x[2], reverse=True)

            held_tickers = {p.ticker for p in positions}
            for tk, direction, sig_count, size_mult in candidates:
                if len(positions) >= MAX_POSITIONS:
                    break
                if tk in held_tickers:
                    continue

                S = float(close[tk].iloc[di])
                vix_val = float(vix_series.iloc[di]) if di < len(vix_series) else 20.0
                rv_val = float(rv_series[tk].iloc[di]) if (
                    tk in rv_series and di < len(rv_series[tk])
                ) else 0.20
                iv = compute_iv(vix_val, rv_val)

                # ATM strike
                K = round(S, 0)
                T = DTE / 365.0
                premium = option_price(S, K, T, RISK_FREE_RATE, iv, direction)

                if premium < 0.10:
                    continue

                contract_cost = premium * 100.0
                max_spend = min(MAX_POS_DOLLAR, MAX_POS_PCT * equity)
                max_spend *= size_mult  # weighted sizing

                if contract_cost > max_spend:
                    continue

                n_contracts = 1
                total_cost = contract_cost + n_contracts * COMMISSION_PER_LEG

                if total_cost > equity:
                    continue

                pos = Position(
                    ticker=tk,
                    option_type=direction,
                    strike=K,
                    entry_premium=premium,
                    entry_date_idx=di,
                    entry_spot=S,
                    dte=DTE,
                    iv=iv,
                    total_cost=total_cost,
                    n_contracts=n_contracts,
                    size_mult=size_mult,
                )
                positions.append(pos)
                held_tickers.add(tk)

        # Daily return
        daily_ret = (equity - prev_equity) / max(prev_equity, 1.0)
        daily_returns.append(daily_ret)
        equity_curve.append({"date": str(dt.date()), "equity": round(equity, 2)})

    # --- Force-close remaining positions ---
    for pos in positions:
        if pos.ticker not in close.columns:
            continue
        S_final = float(close[pos.ticker].iloc[-1])
        remaining_dte = max(pos.dte - pos.days_held, 0)
        T_rem = remaining_dte / 365.0
        vix_final = float(vix_series.iloc[-1]) if len(vix_series) > 0 else 20.0
        rv_final = float(rv_series[pos.ticker].iloc[-1]) if (
            pos.ticker in rv_series and len(rv_series[pos.ticker]) > 0
        ) else 0.20
        iv_final = compute_iv(vix_final, rv_final)
        final_val = option_price(S_final, pos.strike, T_rem, RISK_FREE_RATE,
                                 iv_final, pos.option_type)
        exit_proceeds = final_val * 100.0 * pos.n_contracts
        exit_comm = pos.n_contracts * COMMISSION_PER_LEG if final_val > 0 else 0.0
        pnl = exit_proceeds - pos.total_cost - exit_comm
        equity += pnl
        pct_chg = (final_val - pos.entry_premium) / (pos.entry_premium + 1e-10)
        all_trades.append({
            "ticker": pos.ticker,
            "type": pos.option_type,
            "strike": round(pos.strike, 2),
            "entry_date": str(close.index[pos.entry_date_idx].date()),
            "exit_date": str(close.index[-1].date()),
            "entry_idx": pos.entry_date_idx,
            "exit_idx": len(close) - 1,
            "entry_premium": round(pos.entry_premium, 4),
            "exit_premium": round(final_val, 4),
            "days_held": pos.days_held,
            "pnl": round(pnl, 2),
            "pnl_pct": round(pct_chg * 100, 1),
            "exit_reason": "final_close",
            "size_mult": pos.size_mult,
            "equity_after": round(equity, 2),
        })

    fprint(f"  Total trades: {len(all_trades)}, Final equity: ${equity:.2f}")
    return all_trades, equity_curve, daily_returns


# ================================================================
# METRICS, VALIDATION, REGIME ANALYSIS
# ================================================================

def compute_metrics(trades, equity_curve, daily_returns):
    """Compute Sharpe, Sortino, WR, PF, MaxDD and other validation metrics."""
    if not trades:
        return {
            "n_trades": 0, "sharpe": 0.0, "sortino": 0.0, "win_rate": 0.0,
            "profit_factor": 0.0, "max_dd": 0.0, "total_return": 0.0,
            "avg_pnl": 0.0, "median_pnl": 0.0, "total_pnl": 0.0,
            "avg_hold_days": 0.0, "final_equity": CAP,
        }

    pnls = [t["pnl"] for t in trades]
    pnl_pcts = [t["pnl_pct"] for t in trades]
    n = len(trades)

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / n if n > 0 else 0
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1e-10
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Sharpe from daily returns
    dr = np.array(daily_returns)
    if len(dr) > 5:
        ann = np.sqrt(252)
        mean_ret = np.mean(dr)
        std_ret = np.std(dr)
        sharpe = (mean_ret / (std_ret + 1e-10)) * ann
        down_rets = dr[dr < 0]
        down_std = np.std(down_rets) if len(down_rets) > 1 else std_ret
        sortino = (mean_ret / (down_std + 1e-10)) * ann
    else:
        sharpe = 0.0
        sortino = 0.0

    # Max drawdown from equity curve
    eq_vals = [e["equity"] for e in equity_curve]
    eq_arr = np.array(eq_vals)
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / (peak + 1e-10)
    max_dd = float(dd.min())

    total_pnl = sum(pnls)
    total_return = total_pnl / CAP
    final_eq = equity_curve[-1]["equity"] if equity_curve else CAP

    return {
        "n_trades": n,
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(min(profit_factor, 99.99), 3),
        "max_dd": round(max_dd, 4),
        "total_return": round(total_return, 4),
        "avg_pnl": round(np.mean(pnls), 2),
        "median_pnl": round(np.median(pnls), 2),
        "total_pnl": round(total_pnl, 2),
        "avg_hold_days": round(np.mean([t["days_held"] for t in trades]), 1),
        "final_equity": round(final_eq, 2),
    }


def permutation_test(trades, n_perms=N_PERMUTATIONS):
    """Permutation test: shuffle trade direction, compute Sharpe distribution."""
    if len(trades) < 10:
        return {"p_value": 1.0, "z_score": 0.0}

    pnl_pcts = np.array([t["pnl_pct"] for t in trades])
    actual_sharpe = np.mean(pnl_pcts) / (np.std(pnl_pcts, ddof=1) + 1e-10)

    rng = np.random.RandomState(42)
    null_sharpes = []
    for _ in range(n_perms):
        signs = rng.choice([-1, 1], size=len(pnl_pcts))
        shuffled = pnl_pcts * signs
        s = np.mean(shuffled) / (np.std(shuffled, ddof=1) + 1e-10)
        null_sharpes.append(s)

    null_sharpes = np.array(null_sharpes)
    p_value = float(np.mean(null_sharpes >= actual_sharpe))
    z_score = float((actual_sharpe - np.mean(null_sharpes)) / (np.std(null_sharpes) + 1e-10))

    return {
        "p_value": round(p_value, 4),
        "z_score": round(z_score, 3),
        "actual_sharpe_raw": round(actual_sharpe, 4),
        "null_mean": round(float(np.mean(null_sharpes)), 4),
        "null_std": round(float(np.std(null_sharpes)), 4),
    }


def regime_analysis(trades, close):
    """Stratify performance by green/red SPY regime during holding period."""
    if len(trades) < 5 or "SPY" not in close.columns:
        return {"green": {}, "red": {}, "regime_divergence": 0.0, "regime_agnostic": True}

    spy = close["SPY"]
    spy_daily_ret = spy.pct_change()

    green_trades = []
    red_trades = []

    for t in trades:
        entry_idx = t.get("entry_idx", 0)
        exit_idx = t.get("exit_idx", entry_idx + 1)
        period_rets = spy_daily_ret.iloc[entry_idx:exit_idx + 1]
        spy_period_ret = period_rets.sum() if len(period_rets) > 0 else 0

        if spy_period_ret >= 0:
            green_trades.append(t)
        else:
            red_trades.append(t)

    def summarize(tlist):
        if not tlist:
            return {"n": 0, "sharpe": 0.0, "wr": 0.0, "avg_pnl": 0.0}
        pnl_pcts = [t["pnl_pct"] for t in tlist]
        wins = [p for p in pnl_pcts if p > 0]
        sharpe = (
            np.mean(pnl_pcts) / (np.std(pnl_pcts, ddof=1) + 1e-10)
            if len(pnl_pcts) > 1 else 0.0
        )
        return {
            "n": len(tlist),
            "sharpe": round(sharpe, 3),
            "wr": round(len(wins) / len(tlist), 3),
            "avg_pnl": round(np.mean([t["pnl"] for t in tlist]), 2),
        }

    result = {
        "green": summarize(green_trades),
        "red": summarize(red_trades),
    }

    gs = result["green"].get("sharpe", 0)
    rs = result["red"].get("sharpe", 0)
    max_s = max(abs(gs), abs(rs), 1e-10)
    result["regime_divergence"] = round(abs(gs - rs) / max_s, 3)
    result["regime_agnostic"] = result["regime_divergence"] <= 0.50

    return result


def compare_to_baseline(variant_metrics, baseline_metrics):
    """Compare a variant to the pure momentum baseline."""
    comparison = {}
    for key in ["sharpe", "sortino", "win_rate", "profit_factor", "total_return"]:
        v = variant_metrics.get(key, 0)
        b = baseline_metrics.get(key, 0)
        if b != 0:
            improvement = (v - b) / abs(b) * 100
        else:
            improvement = 0.0
        comparison[f"{key}_improvement_pct"] = round(improvement, 1)
    comparison["v10_adds_value"] = (
        variant_metrics.get("sharpe", 0) > baseline_metrics.get("sharpe", 0)
    )
    return comparison


# ================================================================
# MAIN
# ================================================================

def main():
    t0 = time.time()
    fprint("=" * 70)
    fprint("V10 + MOMENTUM BURST COMBINED STRATEGY v1")
    fprint(f"Started: {datetime.now().isoformat()}")
    fprint("=" * 70)
    fprint(f"Account: ${CAP:.0f} | Universe: {len(SECTORS)} sector ETFs | DTE: {DTE}")
    fprint(f"Period: {START_DATE} to {END_DATE}")
    fprint(f"Variants: {len(VARIANT_CONFIGS)} + 1 pure momentum baseline")
    fprint()

    # ---- Download data ----
    close = download_data()
    volume = load_volume_data()
    earnings_dates = load_earnings_dates()

    # ---- VIX and realized vol ----
    if "VIX" in close.columns:
        vix_series = close["VIX"]
    else:
        fprint("WARNING: VIX not found, using default IV=0.20")
        vix_series = pd.Series(20.0, index=close.index)

    spy_close = close["SPY"] if "SPY" in close.columns else None
    if spy_close is None:
        fprint("ERROR: SPY data missing, cannot proceed")
        return

    rv_series = {}
    for tk in SECTORS:
        if tk in close.columns:
            rets = close[tk].pct_change()
            rv_series[tk] = rets.rolling(21).std() * np.sqrt(252)
            rv_series[tk] = rv_series[tk].fillna(0.20)

    # ---- Build LGBM rankings ----
    min_start_idx = 260
    rebal_indices = list(range(min_start_idx, len(close) - DTE, REBAL_FREQ))
    fprint(f"\nRebalance dates: {len(rebal_indices)} (every {REBAL_FREQ} trading days)")
    fprint(f"  First: {close.index[rebal_indices[0]].date()}, Last: {close.index[rebal_indices[-1]].date()}")

    fprint("\nBuilding feature matrix...")
    feat_df = build_feature_matrix(close, rebal_indices)

    fprint("\nRunning LGBM walk-forward ranking...")
    rankings = walk_forward_lgbm_rank(feat_df)

    if not rankings:
        fprint("ERROR: No rankings produced. Exiting.")
        return

    fprint(f"\nRankings cover {len(rankings)} rebalance dates")

    # ---- Run pure momentum baseline first ----
    fprint("\n" + "=" * 70)
    fprint("RUNNING PURE MOMENTUM BASELINE (no V10 filtering)")
    fprint("=" * 70)

    baseline_trades, baseline_eq, baseline_dr = run_backtest(
        "BASELINE_pure_momentum", PURE_MOMENTUM_CONFIG,
        close, volume, vix_series, rv_series,
        rankings, earnings_dates, spy_close,
    )
    baseline_metrics = compute_metrics(baseline_trades, baseline_eq, baseline_dr)
    baseline_perm = permutation_test(baseline_trades)
    baseline_regime = regime_analysis(baseline_trades, close)

    fprint(f"  Baseline Sharpe: {baseline_metrics['sharpe']}, "
           f"WR: {baseline_metrics['win_rate']:.1%}, "
           f"Trades: {baseline_metrics['n_trades']}")

    # ---- Run all V10+momentum variants ----
    all_results = {}
    all_results["BASELINE_pure_momentum"] = {
        "config": PURE_MOMENTUM_CONFIG,
        "metrics": baseline_metrics,
        "permutation_test": baseline_perm,
        "regime": baseline_regime,
        "comparison": None,
        "trades": baseline_trades,
        "equity_curve": baseline_eq,
    }

    for vname, vcfg in VARIANT_CONFIGS.items():
        trades, eq_curve, daily_rets = run_backtest(
            vname, vcfg, close, volume, vix_series, rv_series,
            rankings, earnings_dates, spy_close,
        )
        metrics = compute_metrics(trades, eq_curve, daily_rets)
        perm = permutation_test(trades)
        regime = regime_analysis(trades, close)
        comparison = compare_to_baseline(metrics, baseline_metrics)

        all_results[vname] = {
            "config": vcfg,
            "metrics": metrics,
            "permutation_test": perm,
            "regime": regime,
            "comparison": comparison,
            "trades": trades,
            "equity_curve": eq_curve,
        }

        m = metrics
        fprint(f"\n  --- {vname} Results ---")
        fprint(f"  Trades: {m['n_trades']}, Sharpe: {m['sharpe']}, Sortino: {m['sortino']}")
        fprint(f"  Win Rate: {m['win_rate']:.1%}, PF: {m['profit_factor']}, MaxDD: {m['max_dd']:.1%}")
        fprint(f"  Total Return: {m['total_return']:.1%}, Final Equity: ${m['final_equity']:.2f}")
        fprint(f"  Avg Hold: {m['avg_hold_days']}d, Avg PnL: ${m['avg_pnl']:.2f}")
        fprint(f"  Permutation p={perm['p_value']}, z={perm['z_score']}")
        fprint(f"  Regime: green_sharpe={regime['green'].get('sharpe', 0)}, "
               f"red_sharpe={regime['red'].get('sharpe', 0)}, "
               f"divergence={regime.get('regime_divergence', 0)}, "
               f"agnostic={regime.get('regime_agnostic', 'N/A')}")
        fprint(f"  V10 adds value: {comparison['v10_adds_value']} "
               f"(Sharpe improvement: {comparison['sharpe_improvement_pct']}%)")

    # ---- Comparison table ----
    fprint("\n" + "=" * 110)
    fprint("VARIANT COMPARISON TABLE")
    fprint("=" * 110)
    header = (
        f"{'Variant':<30} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} "
        f"{'WR':>6} {'PF':>6} {'MaxDD':>7} {'TotRet':>8} "
        f"{'Perm_p':>7} {'Regime':>7} {'V10+':>5}"
    )
    fprint(header)
    fprint("-" * 110)

    for vname, vdata in all_results.items():
        m = vdata["metrics"]
        p = vdata["permutation_test"]
        r = vdata["regime"]
        regime_ok = "PASS" if r.get("regime_agnostic", False) else "FAIL"
        perm_ok = "***" if p["p_value"] < 0.05 else ""
        comp = vdata.get("comparison")
        v10_flag = ""
        if comp is not None:
            v10_flag = "YES" if comp.get("v10_adds_value", False) else "NO"

        fprint(f"{vname:<30} {m['n_trades']:>6} {m['sharpe']:>7.2f} {m['sortino']:>8.2f} "
               f"{m['win_rate']:>5.1%} {m['profit_factor']:>6.2f} {m['max_dd']:>6.1%} "
               f"{m['total_return']:>7.1%} {p['p_value']:>6.3f}{perm_ok} {regime_ok:>7} {v10_flag:>5}")

    fprint("-" * 110)

    # ---- Validation gates ----
    fprint("\n" + "=" * 70)
    fprint("VALIDATION GATE SUMMARY")
    fprint("=" * 70)
    for vname, vdata in all_results.items():
        m = vdata["metrics"]
        p = vdata["permutation_test"]
        r = vdata["regime"]
        comp = vdata.get("comparison")

        gates = [
            ("Sharpe > 0.5", m["sharpe"] > 0.5),
            ("WR > 40%", m["win_rate"] > 0.40),
            ("PF > 1.0", m["profit_factor"] > 1.0),
            ("MaxDD < 50%", m["max_dd"] > -0.50),
            ("Perm p < 0.05", p["p_value"] < 0.05),
            ("Regime agnostic", r.get("regime_agnostic", False)),
        ]
        if comp is not None:
            gates.append(("V10 improves Sharpe", comp.get("v10_adds_value", False)))

        passed = sum(1 for _, v in gates if v)
        total = len(gates)
        status = "PASS" if passed == total else f"PARTIAL ({passed}/{total})"
        fprint(f"\n  {vname}: {status}")
        for gname, gval in gates:
            fprint(f"    {'[x]' if gval else '[ ]'} {gname}")

    # ---- Key question: Does V10 ranking add value? ----
    fprint("\n" + "=" * 70)
    fprint("KEY QUESTION: Does V10 ranking improve momentum burst timing?")
    fprint("=" * 70)
    bl = baseline_metrics
    fprint(f"\n  Pure Momentum Baseline: Sharpe={bl['sharpe']}, Sortino={bl['sortino']}, "
           f"WR={bl['win_rate']:.1%}, PF={bl['profit_factor']}")

    best_variant = None
    best_sharpe = bl["sharpe"]
    for vname, vdata in all_results.items():
        if vname == "BASELINE_pure_momentum":
            continue
        m = vdata["metrics"]
        comp = vdata["comparison"]
        delta = comp["sharpe_improvement_pct"]
        marker = "BETTER" if delta > 0 else "WORSE"
        fprint(f"  {vname}: Sharpe={m['sharpe']} ({delta:+.1f}% vs baseline) [{marker}]")
        if m["sharpe"] > best_sharpe:
            best_sharpe = m["sharpe"]
            best_variant = vname

    if best_variant:
        fprint(f"\n  BEST: {best_variant} (Sharpe {best_sharpe})")
        fprint("  CONCLUSION: V10 ranking adds value to momentum burst timing.")
    else:
        fprint(f"\n  CONCLUSION: V10 ranking does NOT improve momentum burst in any variant.")
        fprint("  Pure momentum burst is sufficient; V10 filtering hurts by reducing opportunity set.")

    # ---- Save results ----
    elapsed = time.time() - t0

    output_data = {
        "metadata": {
            "script": "v10_momentum_combined_v1.py",
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "account_size": CAP,
            "commission_per_leg": COMMISSION_PER_LEG,
            "dte": DTE,
            "wf_train_days": WF_TRAIN_DAYS,
            "rebal_freq": REBAL_FREQ,
            "n_permutations": N_PERMUTATIONS,
            "data_range": f"{close.index[0].date()} to {close.index[-1].date()}",
            "n_sectors": len(SECTORS),
            "exit_params": EXIT_PARAMS,
        },
        "variants": {},
    }

    for vname, vdata in all_results.items():
        output_data["variants"][vname] = {
            "config": vdata["config"],
            "metrics": vdata["metrics"],
            "permutation_test": vdata["permutation_test"],
            "regime": vdata["regime"],
            "comparison": vdata.get("comparison"),
            "n_trades": vdata["metrics"]["n_trades"],
            "trades_sample": vdata["trades"][:10] if vdata["trades"] else [],
        }

    results_path = OUTPUT_DIR / "results_v1.json"
    with open(results_path, "w") as fp:
        json.dump(output_data, fp, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    # Save equity curves
    eq_path = OUTPUT_DIR / "equity_curves_v1.json"
    eq_data = {vname: vdata["equity_curve"] for vname, vdata in all_results.items()}
    with open(eq_path, "w") as fp:
        json.dump(eq_data, fp, indent=2, default=str)
    fprint(f"Equity curves saved to {eq_path}")

    # ---- MLflow logging ----
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            run_name = f"v10_momentum_combined_{datetime.now().strftime('%Y%m%d_%H%M')}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("account_size", CAP)
                mlflow.log_param("commission_per_leg", COMMISSION_PER_LEG)
                mlflow.log_param("dte", DTE)
                mlflow.log_param("wf_train_days", WF_TRAIN_DAYS)
                mlflow.log_param("n_variants", len(VARIANT_CONFIGS) + 1)
                mlflow.log_param("exit_tp_pct", EXIT_PARAMS["tp_pct"])
                mlflow.log_param("exit_sl_pct", EXIT_PARAMS["sl_pct"])
                mlflow.log_param("exit_time_stop_days", EXIT_PARAMS["time_stop_days"])

                for vname, vdata in all_results.items():
                    m = vdata["metrics"]
                    for metric_name, metric_val in m.items():
                        if isinstance(metric_val, (int, float)):
                            safe_name = f"{vname}_{metric_name}"[:250]
                            mlflow.log_metric(safe_name, metric_val)
                    p = vdata["permutation_test"]
                    mlflow.log_metric(f"{vname}_perm_pvalue", p["p_value"])

                    comp = vdata.get("comparison")
                    if comp and "sharpe_improvement_pct" in comp:
                        mlflow.log_metric(f"{vname}_sharpe_vs_baseline",
                                          comp["sharpe_improvement_pct"])

                mlflow.log_artifact(str(results_path))
                mlflow.log_artifact(str(eq_path))
            fprint("MLflow logging complete")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    fprint(f"\nTotal runtime: {elapsed:.1f}s ({elapsed / 60:.1f}min)")
    fprint("DONE.")


if __name__ == "__main__":
    main()
